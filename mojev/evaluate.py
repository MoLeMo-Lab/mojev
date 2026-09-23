"""Evaluate a packed checkpoint: accuracy, calibration per cardinality, controls.

Four things are always reported, whether or not they flatter the model.

*A shuffled-context control.* Each row's menu is paired with its neighbour's
state. A model that still scores well here is reading option priors rather than
the state, and the gap between the two accuracies is the only number that
establishes the state is being read at all. It is printed by default.

*Per-cardinality calibration.* Never a single pooled number. Pooling is what let
an earlier checkpoint ship with a confidence gap running from -0.112 at two
options to +0.015 at sixteen: the average of those looks acceptable.

*The majority baseline.* A field can look accurate while predicting a constant.
The baseline sits next to every accuracy so that case is visible.

*Type safety.* Every returned struct is checked against the schema. The count is
reported rather than asserted, so a regression shows up as a number instead of a
crash.

This reads the model directories ``mojev train full`` writes and scores them
through ``PackedScorer`` and ``packed_collate``, the same pair training and
serving use. The figures in the README results section come from here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .calibrate import calibration_report, field_confidence, reliability_curve
from .data import JsonlDataset
from .full import PackedScorer, move, packed_collate, select_device


def load_packed(checkpoint: str, device: torch.device):
    """Load a trained scorer from a HuggingFace model directory.

    One read of the weights, no pickle, and the schema and token budgets come out
    of ``config.json`` -- so nothing here needs to be told which encoder the model
    was trained on.
    """
    if Path(checkpoint).is_file():
        raise SystemExit(
            f"{checkpoint} is a file. A checkpoint is a model directory: "
            "config.json, safetensors and the tokenizer, as written by "
            "`mojev train full --output`."
        )
    model = PackedScorer.from_pretrained(checkpoint).to(device).eval()
    return model, model.schema, model.config


def _pad_to(tensor: torch.Tensor, width: int, value) -> torch.Tensor:
    """Widen a batch's last dimension so batches with different menus concatenate."""
    if tensor.shape[-1] == width:
        return tensor
    shape = list(tensor.shape)
    shape[-1] = width - tensor.shape[-1]
    return torch.cat([tensor, tensor.new_full(shape, value)], -1)


@torch.no_grad()
def score(model, loader, device, limit: int = 0):
    """Run the model over a loader, returning stacked logits and a stacked batch.

    Padding differs per batch, so tensors are widened to the largest option count
    seen before concatenation: logits pad with ``finfo.min`` (softmax ignores
    them), masks with False, grades with -1.
    """
    chunks, batches, seen = [], [], 0
    for host in loader:
        batch = move(host, device)
        chunks.append(model(batch).cpu())
        batches.append({name: value.cpu() for name, value in batch.items()})
        seen += batch["labels"].shape[0]
        if limit and seen >= limit:
            break
    if not chunks:
        raise ValueError("no rows scored")

    width = max(chunk.shape[-1] for chunk in chunks)
    logits = torch.cat([_pad_to(chunk, width, torch.finfo(chunk.dtype).min)
                        for chunk in chunks])
    merged = {
        "labels": torch.cat([b["labels"] for b in batches]),
        "cardinality": torch.cat([b["cardinality"] for b in batches]),
        "option_mask": torch.cat([_pad_to(b["option_mask"], width, False) for b in batches]),
        "multi_labels": torch.cat([_pad_to(b["multi_labels"], width, 0.0) for b in batches]),
        "preference": torch.cat([_pad_to(b["preference"], width, -1) for b in batches]),
    }
    return logits, merged


def shuffled(collate):
    """Wrap a collator so each row gets its neighbour's state, keeping its own menu.

    The swap happens on the context *string*, before packing. Rolling the packed
    ids instead is tempting and wrong: the sequence holds the state, the questions
    and the candidates, so rolling all of it scores the neighbour's whole row --
    menu and all -- against this row's labels, which measures nothing in
    particular. Masking the roll to the state span is closer but still off, since
    a shorter neighbour leaves that row's own field tokens inside the span.

    Repacking is the control the lift column claims to be: same menu, same label,
    somebody else's state.
    """
    import dataclasses

    def collate_shuffled(examples):
        states = [example.context for example in examples]
        states = states[-1:] + states[:-1]
        return collate([dataclasses.replace(example, context=state)
                        for example, state in zip(examples, states)])

    return collate_shuffled


@torch.no_grad()
def type_safety(logits: torch.Tensor, batch: dict, schema) -> dict:
    """Decode every row and confirm the struct matches the schema.

    The check is structural, not a re-derivation of the argmax: a returned value
    must be one of the field's declared options, and every field must be present.
    """
    checked = violations = 0
    for index, field in enumerate(schema):
        _, prediction = field_confidence(
            logits[:, index], batch["option_mask"][:, index], field.single
        )
        live = batch["option_mask"][:, index]
        if field.single:
            # A chosen index must fall inside the row's live menu.
            chosen_is_live = live.gather(1, prediction.unsqueeze(1)).squeeze(1)
            violations += int((~chosen_is_live).sum())
            checked += prediction.numel()
        else:
            # A multi field may only assert options that exist in the row.
            violations += int((prediction.bool() & ~live).sum())
            checked += int(live.sum())
    return {
        "decisions_checked": checked,
        "schema_violations": violations,
        "violation_rate": violations / max(checked, 1),
    }


def evaluate(model, loader, shuffled_loader, device, schema, limit: int = 0) -> dict:
    """One split: the real report, the shuffled-state null, and the lift.

    Two loaders because the null control swaps the state before packing, so it is
    a property of the collator rather than something to do to a scored batch.
    """
    logits, merged = score(model, loader, device, limit)
    report = calibration_report(logits, merged, schema)
    report["type_safety"] = type_safety(logits, merged, schema)
    report["rows"] = int(logits.shape[0])

    null_logits, null_merged = score(model, shuffled_loader, device, limit)
    null = calibration_report(null_logits, null_merged, schema)
    for field in schema:
        entry = report["fields"][field.name]
        entry["shuffled_accuracy"] = null["fields"][field.name]["accuracy"]
        entry["lift_over_shuffled"] = entry["accuracy"] - entry["shuffled_accuracy"]

    # Reliability for the widest field only; the rest live in by_cardinality.
    first = schema.fields[0]
    if first.single:
        confidence, prediction = field_confidence(
            logits[:, 0], merged["option_mask"][:, 0], True
        )
        report["reliability"] = reliability_curve(
            confidence, prediction.eq(merged["labels"][:, 0])
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="local model directory or Hugging Face repo ID")
    parser.add_argument("data", nargs="+",
                        help="one or more JSONL files; each is reported separately")
    parser.add_argument("--tokenizer",
                        help="override tokenizer with a local path or Hugging Face repo ID")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--context-tokens", type=int,
                        help="inference state-token window; defaults to the checkpoint setting")
    parser.add_argument("--limit", type=int, default=0, help="rows per split; 0 is all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = select_device(args.device)
    model, schema, config = load_packed(args.checkpoint, device)
    context_tokens = config.context_tokens if args.context_tokens is None else args.context_tokens
    if context_tokens < 1:
        parser.error("--context-tokens must be positive")
    collate = packed_collate(
        _tokenizer(args.tokenizer or args.checkpoint),
        schema, context_tokens,
    )

    result = {"checkpoint": args.checkpoint, "device": str(device),
              "context_tokens": context_tokens, "files": {}}
    for path in args.data:
        dataset = JsonlDataset(path, schema)
        loaders = [
            DataLoader(dataset, batch_size=args.batch_size, collate_fn=batcher,
                       num_workers=args.workers)
            for batcher in (collate, shuffled(collate))
        ]
        result["files"][Path(path).name] = evaluate(
            model, loaders[0], loaders[1], device, schema, args.limit
        )

    field = schema.fields[0].name
    print(f"{'split':16s} {'rows':>7s} {'acc':>7s} {'shuf':>7s} {'lift':>8s} "
          f"{'ece':>7s} {'gap':>8s} {'spear':>6s} {'inv':>4s}")
    for name, entry in result["files"].items():
        row = entry["fields"][field]
        mono = row.get("monotonicity") or {}
        print(f"{name:16s} {entry['rows']:7d} {row['accuracy']:7.4f} "
              f"{row['shuffled_accuracy']:7.4f} {row['lift_over_shuffled']:+8.4f} "
              f"{row['ece']:7.4f} {row['gap']:+8.4f} "
              f"{(mono.get('spearman') or 0):6.3f} {(mono.get('inversions') or 0):4d}")

    cardinalities = sorted({
        n for entry in result["files"].values()
        for n in entry["fields"][field]["by_cardinality"]
    })
    if cardinalities:
        print(f"\n{'per cardinality':16s} " + " ".join(f"{n:>7d}" for n in cardinalities))
        for name, entry in result["files"].items():
            cells = []
            for n in cardinalities:
                bucket = entry["fields"][field]["by_cardinality"].get(n)
                cells.append(f"{bucket['accuracy']:7.4f}" if bucket else f"{'-':>7s}")
            print(f"{name:16s} " + " ".join(cells))

    # Every field, with both controls next to the accuracy: the shuffled null
    # (is the state being read?) and the majority baseline (did the field learn
    # anything, or is it predicting a constant?). Either can be passed while the
    # accuracy looks healthy.
    summary = {}
    for name, entry in result["files"].items():
        summary[name] = {
            "rows": entry["rows"],
            "mean_ece": round(entry["mean_ece"], 4),
            "worst_gap": round(entry["worst_gap_any_field"], 4),
            "schema_violations": entry["type_safety"]["schema_violations"],
            "fields": {
                f.name: {
                    key: (None if entry["fields"][f.name].get(source) is None
                          else round(entry["fields"][f.name][source], 4))
                    for key, source in (
                        ("accuracy", "accuracy"),
                        ("shuffled", "shuffled_accuracy"),
                        ("lift_over_shuffled", "lift_over_shuffled"),
                        ("majority", "majority"),
                        ("lift_over_majority", "lift_over_majority"),
                        ("ece", "ece"),
                        ("gap", "gap"),
                    )
                } | {
                    "monotone": (entry["fields"][f.name].get("monotonicity") or {}).get("monotone"),
                }
                for f in schema
            },
        }
    result["summary"] = summary
    print()
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"report": str(args.output)}))


def _tokenizer(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


if __name__ == "__main__":
    main()
