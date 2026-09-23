"""Full fine-tuning: the encoder itself is trained on the System One objective.

Everything before this trained a head on top of frozen features. TypeSafe's FAQ
rules that reading out: asked whether Jev is a small LLM, the answer is "Jev is
neither small nor an LLM, and is therefore off the intelligence Pareto curve."
A 791k-parameter head on a frozen 0.85B encoder is exactly the small end of that
curve. And their stated reason the whole approach works -- "you get what you
optimise for" -- applies to the encoder too: features trained for next-token
prediction are not features trained for calibrated decisions.

So here the encoder trains. Two consequences drive the design.

**The option cache dies.** Precomputed option vectors are only valid while the
encoder is fixed; once it moves, all 131,693 of them are stale. Re-encoding
options per step would mean 1 + N encoder calls instead of 1 -- at batch 8 with
64 options that is 520 passages per step instead of 8.

The fix is to stop encoding options separately at all. Context and options go
through the encoder in **one packed sequence**, with a block-diagonal attention
mask so each option attends to the context and to itself but not to its rivals.
That keeps the per-option independence the whole design rests on (permutation
invariance, measured at 1.8e-07) while costing one forward pass.

**Memory is not the constraint.** bf16 params + bf16 grads + fp32 AdamW state
for 0.85B is 13.65 GB against 179 GB per card. Sequence length is what binds,
because the packed sequence is context + sum of option lengths.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .calibrate import brier, calibration_report, plackett_luce
from .modeling import PackedScorer, PackedScorerConfig
from .schema import Schema

# The model moved to mojev/modeling.py when checkpoints became HuggingFace
# directories; it is re-exported because the collator, the objective and the
# model are one unit as far as callers are concerned.
__all__ = [
    "PackedScorer", "PackedScorerConfig", "packed_collate", "objective",
    "sort_candidates", "unsort", "move", "select_device",
]


def move(batch: dict, device: torch.device) -> dict:
    return {name: tensor.to(device) for name, tensor in batch.items()}


def select_device(name: str = "auto") -> torch.device:
    """Resolve a device name; cuda when it is there, cpu when nothing else is."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sort_candidates(options: list[str] | tuple[str, ...]) -> list[int]:
    """Indices that sort a menu -- the step that makes scoring permutation-invariant.

    Packing lays candidates out one after another, so the same candidate lands at a
    different absolute position depending on the order the caller listed it in, and
    its score moves with it: reversing three choices shifted probabilities by
    2.5e-01 and flipped the argmax. Scoring the sorted menu and mapping the answer
    back makes the packed sequence a function of the candidate *set*, so the
    invariance is constructive rather than something the encoder has to learn.

    Serving does this on every request; the consistency harness reuses it so that
    what it measures is what callers actually get.
    """
    return sorted(range(len(options)), key=lambda index: options[index])


def unsort(order: list[int], values: list) -> list:
    """Map per-candidate results from sorted order back to the caller's order."""
    out = [None] * len(order)
    for position, original in enumerate(order):
        out[original] = values[position]
    return out


IMAGE_MARKER = "<|image_pad|>"
IMAGE_PLACEHOLDER = f"<|vision_start|>{IMAGE_MARKER}<|vision_end|>"
# A Qwen image placeholder followed by the file to load, up to whitespace. The
# short ``<|image_pad|>path`` form is accepted and normalised to the full marker.
IMAGE_PATH = re.compile(
    rf"(?:<\|vision_start\|>)?{re.escape(IMAGE_MARKER)}(?:<\|vision_end\|>)?\s*(\S+)"
)


def _encode_with_images(processor, texts: list[str], context_tokens: int):
    """Tokenise states that contain image markers, returning ids and pixel tensors.

    Images are passed as ``<|vision_start|><|image_pad|><|vision_end|>path``. The
    path is loaded and removed from the text; the Qwen placeholder stays for the
    processor to expand into visual patch positions. Those positions join the
    state span, so every candidate can attend to the image through the tree mask.
    """
    from PIL import Image

    prepared, images = [], []
    for text in texts:
        paths = IMAGE_PATH.findall(text)
        # One pass over the whole string: strip every path, leave every marker for
        # the processor to expand. Substituting in a loop while keeping the marker
        # would not terminate.
        prepared.append(IMAGE_PATH.sub(lambda _: IMAGE_PLACEHOLDER, text))
        images.extend(Image.open(path).convert("RGB") for path in paths)
    encoded = processor(text=prepared, images=images or None, return_tensors="pt",
                        padding=True, truncation=True, max_length=context_tokens)
    ids = [row[mask.bool()].tolist()
           for row, mask in zip(encoded["input_ids"], encoded["attention_mask"])]
    extra = {name: encoded[name] for name in
             ("pixel_values", "image_grid_thw", "mm_token_type_ids")
             if name in encoded}
    return ids, extra


def packed_collate(tokenizer, schema: Schema, context_tokens: int,
                   field_tokens_max: int = 32, processor=None):
    """Build one packed sequence per row, with span indicators for pooling.

    Layout per row:

        [context] [field 0 prompt] [opt 0] [opt 1] ... [field 1 prompt] [opt 0] ...

    The field prompt is part of the sequence, not a side input. An option attends
    to the context, to its own field's words, and to itself -- so "which question
    am I answering" is something it reads, exactly like everything else it reads.
    """
    pad = tokenizer.pad_token_id
    if pad is None:
        tokenizer.pad_token = tokenizer.eos_token
        pad = tokenizer.pad_token_id
    fields = len(schema)
    width = schema.max_cardinality
    # Field prompts are tokenised once and spliced into every row's sequence.
    field_tokens = tokenizer(list(schema.prompts), truncation=True,
                             max_length=field_tokens_max)["input_ids"]

    def collate(examples):
        # A state carrying image markers is tokenised by the processor, which
        # expands <|image_pad|> into one placeholder per visual patch and returns
        # the pixels alongside. Those placeholders are ordinary sequence
        # positions, so they join the state span and the tree mask needs no
        # change: candidates attend to the picture exactly as they attend to text.
        pixels = None
        if processor is not None and any(
            IMAGE_MARKER in e.context for e in examples
        ):
            contexts, pixels = _encode_with_images(
                processor, [e.context for e in examples], context_tokens
            )
        else:
            contexts = tokenizer([e.context for e in examples], truncation=True,
                                 max_length=context_tokens)["input_ids"]
        menus = [
            e.options if e.options is not None else tuple(f.options for f in schema)
            for e in examples
        ]
        flat = [o for row in menus for menu in row for o in menu]
        encoded = tokenizer(flat, truncation=False)["input_ids"]

        sequences, layouts = [], []
        cursor = 0
        for row, (ctx, menu_row) in enumerate(zip(contexts, menus)):
            tokens = list(ctx)
            spans, field_spans = [], []
            for field_index, menu in enumerate(menu_row):
                # The field's own words go in the sequence, immediately before
                # its options. Nothing about the field is handled specially: an
                # option reads it the same way it reads the context.
                prompt = field_tokens[field_index]
                field_spans.append((field_index, len(tokens), len(tokens) + len(prompt)))
                tokens.extend(prompt)
                for column in range(len(menu)):
                    piece = encoded[cursor]
                    cursor += 1
                    spans.append((field_index, column, len(tokens), len(tokens) + len(piece)))
                    tokens.extend(piece)
            sequences.append(tokens)
            layouts.append((len(ctx), spans, field_spans))

        total = max(len(s) for s in sequences)
        batch = len(examples)
        packed = torch.full((batch, total), pad, dtype=torch.long)
        packed_mask = torch.zeros(batch, total, dtype=torch.bool)
        context_span = torch.zeros(batch, total)
        option_span = torch.zeros(batch, fields, width, total)
        option_mask = torch.zeros(batch, fields, width, dtype=torch.bool)
        # Which positions each option may attend to besides the context: its own
        # field prompt and itself.
        field_span = torch.zeros(batch, fields, total)

        for row, (tokens, (clen, spans, field_spans)) in enumerate(zip(sequences, layouts)):
            packed[row, :len(tokens)] = torch.tensor(tokens)
            packed_mask[row, :len(tokens)] = True
            context_span[row, :clen] = 1.0
            for field_index, start, end in field_spans:
                if end > start:
                    field_span[row, field_index, start:end] = 1.0
            for field_index, column, start, end in spans:
                if end > start:
                    option_span[row, field_index, column, start:end] = 1.0
                    option_mask[row, field_index, column] = True

        labels = torch.zeros(batch, fields, dtype=torch.long)
        preference = torch.full((batch, fields, width), -1, dtype=torch.long)
        for row, item in enumerate(examples):
            for field_index, (field, value) in enumerate(zip(schema, item.labels)):
                if field.single:
                    labels[row, field_index] = value
            if item.preference is not None:
                for field_index, grades in enumerate(item.preference):
                    if grades is not None:
                        preference[row, field_index, :len(grades)] = torch.tensor(grades)

        return {
            "packed_ids": packed, "packed_mask": packed_mask,
            "context_span": context_span, "option_span": option_span,
            "option_mask": option_mask, "labels": labels,
            "preference": preference, "cardinality": option_mask.sum(-1),
            "multi_labels": torch.zeros(batch, fields, width),
            "field_span": field_span,
            **(pixels or {}),
        }

    return collate


def objective(logits, batch, schema, brier_weight: float) -> torch.Tensor:
    """Plackett-Luce over graded options plus a Brier term -- the pair that won.

    Measured on wiki-simpleQA with a frozen encoder: PL alone beat cross-entropy
    on all three fields, and adding Brier at weight 1.0 took answer accuracy from
    0.6218 to 0.6540 (cross-entropy: 0.6153) while halving calibration error.
    """
    total, n = logits.new_zeros(()), 0
    grades = batch["preference"]
    for index, field in enumerate(schema):
        if not field.single:
            continue
        field_logits = logits[:, index]
        mask = batch["option_mask"][:, index]
        field_grades = grades[:, index]
        if bool((field_grades[mask] >= 0).any()):
            loss = plackett_luce(field_logits, field_grades, mask)
        else:
            loss = F.cross_entropy(field_logits, batch["labels"][:, index])
        if brier_weight > 0:
            probabilities = field_logits.softmax(-1)
            targets = F.one_hot(batch["labels"][:, index],
                                field_logits.shape[-1]).to(probabilities.dtype)
            loss = loss + brier_weight * brier(probabilities, targets, mask.to(probabilities.dtype))
        total = total + loss
        n += 1
    return total / max(n, 1)


def setup_distributed() -> tuple[int, int, int]:
    """Join the process group if torchrun launched us. Returns (rank, local, world)."""
    import os

    if "RANK" not in os.environ:
        return 0, 0, 1
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local)
    # An hour, not the default ten minutes: a long validation pass or a slow
    # checkpoint write must not abort a completed training run.
    import datetime

    dist.init_process_group("nccl", rank=rank, world_size=world,
                            timeout=datetime.timedelta(hours=1))
    return rank, local, world


def main() -> None:
    from transformers import AutoProcessor

    from .data import JsonlDataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train")
    parser.add_argument("--validation", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--context-tokens", type=int, default=16_384)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--brier-weight", type=float, default=1.0)
    parser.add_argument("--max-cardinality", type=int, default=16,
                        help="packed sequence length grows with this; 64 options of "
                             "16 tokens adds 1024 positions per row")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rank, local, world = setup_distributed()
    main_process = rank == 0
    torch.manual_seed(args.seed + rank)
    device = torch.device(f"cuda:{local}" if world > 1 else "cuda")
    schema = Schema.from_json(json.loads(Path(args.schema).read_text()))
    processor = AutoProcessor.from_pretrained(args.hf_model)
    # MoJev packs images and state into a 16,384-token branch. Keep
    # each image at 65,536 pixels, which expands to roughly 64 visual tokens.
    processor.image_processor.size = {
        "shortest_edge": 65_536,
        "longest_edge": 65_536,
    }
    tokenizer = processor.tokenizer
    model = PackedScorer.from_encoder(
        args.hf_model, schema, args.rank,
        context_tokens=args.context_tokens,
    ).to(device)
    model.encoder.gradient_checkpointing_enable()

    raw_model = model
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel

        # The encoder is checkpointed, so some parameters are used more than
        # once per step; static_graph lets DDP handle that without the
        # find_unused_parameters overhead.
        model = DistributedDataParallel(model, device_ids=[local], static_graph=True)

    trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    if main_process:
        print(json.dumps({
            "trainable_parameters": trainable,
            "encoder_trained": True,
            "world_size": world,
            "effective_batch": args.batch_size * args.accumulate * world,
            "fields": [f.name for f in schema],
            "brier_weight": args.brier_weight,
        }), flush=True)

    collate = packed_collate(tokenizer, schema, args.context_tokens)
    train_dataset = JsonlDataset(args.train, schema)
    validation_dataset = JsonlDataset(args.validation, schema)
    train_sampler = None
    if world > 1:
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(train_dataset, num_replicas=world,
                                           rank=rank, shuffle=True, drop_last=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=train_sampler is None, sampler=train_sampler,
                              collate_fn=collate, num_workers=args.workers,
                              drop_last=True, pin_memory=True)
    # Validation is sharded across ranks. Running it on rank 0 alone made the
    # other ranks sit in a barrier long enough to trip NCCL's 10-minute
    # collective timeout -- 82,423 validation rows on one card does not finish
    # inside it, so a completed training run died at the end.
    validation_sampler = None
    if world > 1:
        from torch.utils.data.distributed import DistributedSampler

        validation_sampler = DistributedSampler(validation_dataset, num_replicas=world,
                                                rank=rank, shuffle=False, drop_last=False)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size,
                                   sampler=validation_sampler, collate_fn=collate,
                                   num_workers=args.workers)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=0.01, betas=(0.9, 0.95))

    for epoch in range(args.epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)     # otherwise every epoch sees one order
        started, total, steps = time.perf_counter(), 0.0, 0
        optimiser.zero_grad(set_to_none=True)
        for step, host in enumerate(train_loader):
            batch = move(host, device)
            loss = objective(model(batch), batch, schema, args.brier_weight) / args.accumulate
            loss.backward()
            if (step + 1) % args.accumulate == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
            total += float(loss.detach()) * args.accumulate
            steps += 1
            if main_process and steps % 200 == 0:
                rate = steps * args.batch_size * world / (time.perf_counter() - started)
                print(json.dumps({"epoch": epoch + 1, "step": steps,
                                  "loss": round(total / steps, 4),
                                  "rows_per_second": round(rate, 1),
                                  "seconds": round(time.perf_counter() - started)}), flush=True)
        raw_model.eval()
        chunks, batches = [], []
        with torch.no_grad():
            for host in validation_loader:
                batch = move(host, device)
                chunks.append(raw_model(batch).cpu())
                batches.append({k: v.cpu() for k, v in batch.items()})
        # Each rank scores its own shard; only rank 0 reports and writes, so the
        # printed metrics describe 1/world of the split. That is enough to track a
        # run, and it keeps every rank inside the collective timeout.
        if chunks and main_process:
            logits = torch.cat(chunks)
            merged = {k: torch.cat([b[k] for b in batches]) for k in
                      ("labels", "option_mask", "cardinality", "multi_labels", "preference")}
            report = calibration_report(logits, merged, schema)
            first = schema.fields[0].name
            print(json.dumps({
                "epoch": epoch + 1,
                "train_loss": round(total / max(steps, 1), 4),
                "val_accuracy": round(report["fields"][first]["accuracy"], 4),
                "val_ece": round(report["mean_ece"], 4),
                "val_rows_this_rank": int(logits.shape[0]),
                "spearman": report["fields"][first]["monotonicity"]["spearman"],
                "seconds": round(time.perf_counter() - started),
            }), flush=True)
            # A directory, not a pickle: config.json plus safetensors, with the
            # processor alongside so the artefact is the whole model and nothing
            # downstream needs another Hub ID to load it.
            raw_model.save_pretrained(args.output, safe_serialization=True)
            processor.save_pretrained(args.output)
        if world > 1:
            # Every rank waits here, so the others do not start the next epoch
            # while rank 0 is still evaluating and writing the checkpoint.
            import torch.distributed as dist

            dist.barrier()

    if world > 1:
        import torch.distributed as dist

        dist.destroy_process_group()
    if main_process:
        print(json.dumps({"checkpoint": args.output}))


if __name__ == "__main__":
    main()
