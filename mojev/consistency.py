"""Consistency: "returns similar answers for similar inputs".

TypeSafe lists this beside calibration as a property of the model class, and
contrasts it with LLMs being "inconsistent". It is a separate claim from
calibration and needs its own measurement.

Three perturbations, each leaving the correct answer unchanged, each probing a
different way a scorer can be unstable:

``permute``    shuffle the option order. Candidates are sorted before packing
               (``mojev/serve.py``), so the packed sequence is a function of the
               candidate *set* and the probabilities should be identical up to
               floating point, not merely similar. Anything else indicates the
               sort was bypassed or that candidates leak into each other.

``padding``    add trailing whitespace to the context. No information changes.

``distractor`` swap the distractors for different wrong answers, keeping the
               correct one.

The three are not judged the same way, which matters. ``permute`` and ``padding``
change nothing about the decision, so the model should agree with itself; for
``permute`` the expectation is exact invariance, since sorting the candidates
makes their listed order unobservable to the model.

``distractor`` genuinely changes the problem -- a fresh set of wrong answers can
be easier or harder -- so demanding the same argmax would be measuring the wrong
thing. What should hold is that the correct answer keeps its standing: it is
reported as whether the truth stays selected when it was selected before, not as
raw argmax agreement.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
from pathlib import Path

import torch

from .data import validate
from .evaluate import load_packed
from .full import move, packed_collate, select_device, sort_candidates


def permute_row(row: dict, field: str, rng: random.Random) -> dict:
    menu = list(row["options"][field])
    rng.shuffle(menu)
    out = json.loads(json.dumps(row))
    out["options"][field] = menu
    return out


def pad_row(row: dict, rng: random.Random) -> dict:
    out = json.loads(json.dumps(row))
    out["context"] = out["context"] + "\n" * rng.randint(1, 3)
    return out


def swap_distractors(row: dict, field: str, pool: list[str], rng: random.Random) -> dict:
    """Keep the labelled answer, replace every other option with a fresh one."""
    correct = row["labels"][field]
    menu = [correct]
    seen = {correct.lower()}
    while len(menu) < len(row["options"][field]):
        candidate = pool[rng.randrange(len(pool))]
        if candidate.lower() in seen:
            continue
        seen.add(candidate.lower())
        menu.append(candidate)
    rng.shuffle(menu)
    out = json.loads(json.dumps(row))
    out["options"][field] = menu
    return out


@torch.no_grad()
def probabilities_for(model, collator, rows, schema, device, field_index: int):
    """(rows, N) probabilities for one field, plus the index of each row's label.

    Candidates go through the same sort serving applies, and the probabilities are
    mapped back to each row's listed order. Scoring the listed order directly
    would measure raw packing position instead of what a caller receives, and
    ``permute`` would report a non-invariance the API does not have.
    """
    examples, orders = [], []
    for row in rows:
        example = validate(row, schema)
        menus, row_orders = [], []
        for field_options in example.options:
            order = sort_candidates(field_options)
            row_orders.append(order)
            menus.append(tuple(field_options[index] for index in order))
        examples.append(dataclasses.replace(example, options=tuple(menus)))
        orders.append(row_orders)

    batch = move(collator(examples), device)
    logits = model(batch)[:, field_index]
    sorted_probabilities = logits.softmax(-1).cpu()

    # Undo the sort, so column j is the row's j-th listed candidate again. The
    # label indexes that listed order, so it needs no remapping once this is done.
    probabilities = torch.zeros_like(sorted_probabilities)
    for row_index, row_orders in enumerate(orders):
        for position, original in enumerate(row_orders[field_index]):
            probabilities[row_index, original] = sorted_probabilities[row_index, position]
    return probabilities, batch["labels"][:, field_index].cpu()


def run(model, collator, schema, rows: list[dict], device, field: str,
        seed: int = 0) -> dict:
    field_index = [f.name for f in schema].index(field)
    rng = random.Random(seed)
    pool = sorted({
        option for row in rows for option in row["options"][field]
    })
    base_probs, base_labels = probabilities_for(model, collator, rows, schema, device, field_index)
    base_choice = base_probs.argmax(-1)
    base_correct = base_probs.gather(1, base_labels.unsqueeze(1)).squeeze(1)

    report = {}
    for name, transform in (
        ("permute", lambda row: permute_row(row, field, rng)),
        ("padding", lambda row: pad_row(row, rng)),
        ("distractor", lambda row: swap_distractors(row, field, pool, rng)),
    ):
        changed = [transform(row) for row in rows]
        probs, labels = probabilities_for(model, collator, changed, schema, device, field_index)
        correct = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        entry = {
            "rows": len(rows),
            "mean_abs_change_in_truth": float((correct - base_correct).abs().mean()),
            "max_abs_change_in_truth": float((correct - base_correct).abs().max()),
        }
        if name == "distractor":
            # The problem changed, so the argmax may legitimately move. What is
            # being asked is whether the truth keeps its standing.
            base_hit = base_choice.eq(base_labels)
            hit = probs.argmax(-1).eq(labels)
            entry["truth_selected_before"] = float(base_hit.float().mean())
            entry["truth_selected_after"] = float(hit.float().mean())
            entry["truth_retention"] = (
                float(hit[base_hit].float().mean()) if bool(base_hit.any()) else None
            )
        else:
            # The decision is unchanged, so the model should agree with itself.
            # Compare the chosen *string*, since permutation moves the indices.
            entry["argmax_agreement"] = float(
                torch.tensor([
                    changed[i]["options"][field][int(probs[i].argmax())]
                    == rows[i]["options"][field][int(base_choice[i])]
                    for i in range(len(rows))
                ]).float().mean()
            )
        if name == "permute":
            # Exact invariance is expected here, not approximate stability: the
            # head scores each option against the context independently.
            entry["invariant"] = entry["max_abs_change_in_truth"] < 1e-4
        report[name] = entry
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="local model directory or Hugging Face repo ID")
    parser.add_argument("data")
    parser.add_argument("--field", default="")
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--tokenizer",
                        help="override tokenizer with a local path or Hugging Face repo ID")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--context-tokens", type=int,
                        help="inference state-token window; defaults to the checkpoint setting")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = select_device(args.device)
    model, schema, config = load_packed(args.checkpoint, device)
    context_tokens = config.context_tokens if args.context_tokens is None else args.context_tokens
    if context_tokens < 1:
        parser.error("--context-tokens must be positive")
    from transformers import AutoTokenizer

    collator = packed_collate(
        AutoTokenizer.from_pretrained(args.tokenizer or args.checkpoint),
        schema, context_tokens,
    )
    field = args.field or schema.fields[0].name

    rows = []
    with Path(args.data).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if field in (row.get("options") or {}):
                rows.append(row)
            if len(rows) >= args.rows:
                break
    if not rows:
        raise SystemExit(f"no rows in {args.data} carry a per-row menu for field {field!r}")

    report = {"checkpoint": args.checkpoint, "field": field, "rows": len(rows),
              "context_tokens": context_tokens,
              "perturbations": run(model, collator, schema, rows, device, field, args.seed)}
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
