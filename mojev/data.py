"""Multi-field JSONL loading and the data builders.

One row per line. ``labels`` holds one entry per schema field:

    {"context": "The customer needs a refund.",
     "labels": {"intent": "refund", "escalate": false,
                "tags": ["billing"], "churn": "low"}}

Options live in the schema, not in the row, so every row shares one option table.
The exception is a ragged field whose options change per row (a click menu):
those rows carry ``options: {"field": [...]}`` and the schema field records the
widest menu it may see.

Batching lives in ``mojev.full.packed_collate``, which is the only collator --
one packed sequence per row, state then questions then candidates, with a tree
mask over it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

from torch.utils.data import Dataset

from .schema import Field, Schema


@dataclass(frozen=True)
class Example:
    context: str
    labels: tuple                      # one entry per field: int, or 0/1 list for multi
    options: tuple[tuple[str, ...], ...] | None = None   # per-field override, in field order
    # Graded option quality, aligned with ``options``. A preference objective
    # needs an ordering over the wrong answers, not just which one is right;
    # without it there is a single bit of supervision and nothing to rank.
    preference: tuple[tuple[int, ...] | None, ...] | None = None


def validate(payload: dict, schema: Schema) -> Example:
    context = payload.get("context")
    if not isinstance(context, str) or not context:
        raise ValueError("each row needs a non-empty string context")
    labels = payload.get("labels")
    if not isinstance(labels, dict):
        raise ValueError("each row needs a labels object")
    overrides = payload.get("options")
    ranks = payload.get("preference") or {}
    if overrides is None:
        return Example(context, tuple(schema.encode(labels)))
    if not isinstance(overrides, dict):
        raise ValueError("options must be an object keyed by field name")
    # A ragged row declares its own menu; the label is then an index into that menu.
    rows, encoded, grades = [], [], []
    for field in schema:
        menu = overrides.get(field.name)
        if menu is None:
            rows.append(field.options)
            encoded.append(field.encode(labels[field.name]))
            grades.append(None)
            continue
        if not isinstance(menu, list) or len(menu) < 2:
            raise ValueError(f"{field.name}: a row menu needs at least two options")
        if len(menu) > field.cardinality:
            raise ValueError(
                f"{field.name}: row menu has {len(menu)} options but the schema declares "
                f"{field.cardinality}; widen the schema field"
            )
        if not field.single:
            raise ValueError(f"{field.name}: per-row menus are only supported for single-choice fields")
        rows.append(tuple(menu))
        encoded.append(Field(field.name, "choice", tuple(menu)).encode(labels[field.name]))
        grade = ranks.get(field.name)
        if grade is not None and len(grade) != len(menu):
            raise ValueError(
                f"{field.name}: preference has {len(grade)} entries for {len(menu)} options"
            )
        grades.append(tuple(grade) if grade is not None else None)
    return Example(
        context, tuple(encoded), tuple(rows),
        tuple(grades) if any(g is not None for g in grades) else None,
    )


class JsonlDataset(Dataset[Example]):
    def __init__(self, path: str | Path, schema: Schema) -> None:
        self.schema = schema
        with Path(path).open(encoding="utf-8") as handle:
            self.examples = [
                validate(json.loads(line), schema) for line in handle if line.strip()
            ]
        if not self.examples:
            raise ValueError(f"no examples in {path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


# ---------------------------------------------------------------- synthetic data

COLOURS = ("amber", "azure", "bronze", "coral", "crimson", "gold", "green", "indigo")
ANIMALS = ("badger", "crane", "dolphin", "falcon", "gecko", "heron", "ibis", "jaguar")
QUEUES = ("refund", "sales", "technical support", "billing")
TAGS = ("urgent", "repeat", "vip")
LEVELS = ("low", "medium", "high")

SYNTHETIC_SCHEMA = Schema((
    Field("badge", "choice", tuple(f"{c} {a}" for c in COLOURS for a in ANIMALS)),
    Field("queue", "choice", QUEUES),
    Field("escalate", "bool"),
    Field("tags", "multi", TAGS),
    Field("urgency", "bucket", LEVELS),
))


def synthetic_example(seed: int, cardinality: int | None = None) -> dict:
    """One row whose every field is decidable from the context alone.

    ``cardinality`` trims the badge menu, which is how the evaluation builds
    matched sets at several option counts.
    """
    rng = random.Random(seed)
    badge = f"{rng.choice(COLOURS)} {rng.choice(ANIMALS)}"
    queue = rng.choice(QUEUES)
    escalate = rng.random() < 0.4
    tags = [tag for tag in TAGS if rng.random() < 0.35]
    urgency = LEVELS[min(2, len(tags))] if escalate else LEVELS[0]
    context = (
        f"Badge: {badge}. Route to {queue}. "
        f"Escalate: {'yes' if escalate else 'no'}. "
        f"Tags: {', '.join(tags) if tags else 'none'}. "
        f"Urgency: {urgency}."
    )
    row = {
        "context": context,
        "labels": {
            "badge": badge, "queue": queue, "escalate": escalate,
            "tags": tags, "urgency": urgency,
        },
    }
    if cardinality is not None:
        menu = {badge}
        while len(menu) < cardinality:
            menu.add(f"{rng.choice(COLOURS)} {rng.choice(ANIMALS)}")
        menu = list(menu)
        rng.shuffle(menu)
        row["options"] = {"badge": menu}
    return row


def write_synthetic(output: Path, sizes: dict[str, int], seed: int,
                    cardinalities: tuple[int, ...] = (2, 4, 8, 16, 32, 64)) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "schema.json").write_text(
        json.dumps(SYNTHETIC_SCHEMA.to_json(), indent=2) + "\n", encoding="utf-8"
    )
    offset = 0
    for split, size in sizes.items():
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for index in range(size):
                # Cycle cardinalities so training sees the full range it is calibrated on.
                card = cardinalities[index % len(cardinalities)]
                row = synthetic_example(seed + offset + index * 104729, card)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        offset += size * 104729


def write_cardinality_sets(output: Path, size: int, seed: int,
                           cardinalities: tuple[int, ...]) -> None:
    """One file per option count, so calibration can be measured per cardinality."""
    output.mkdir(parents=True, exist_ok=True)
    for card in cardinalities:
        with (output / f"card-{card}.jsonl").open("w", encoding="utf-8") as handle:
            for index in range(size):
                row = synthetic_example(seed + card * 7919 + index * 104729, card)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ------------------------------------------------------------- wikispeedia data

def _stable(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def _title(text: str) -> str:
    return unquote(text).replace("_", " ")


def build_wikispeedia(root: Path, output: Path, max_options: int = 512) -> None:
    """Next-click rows from the SNAP archives.

    Splits are bucketed by *target article* hash, so every record for one target
    lands in one split and near-duplicate paths cannot leak across the boundary.
    """
    graph_dir = root / "wikispeedia_paths-and-graph"
    outgoing: dict[str, list[str]] = {}
    for line in (graph_dir / "links.tsv").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            source, target = line.split("\t")
            outgoing.setdefault(source, []).append(target)
    output.mkdir(parents=True, exist_ok=True)
    schema = Schema((Field("click", "choice", tuple(f"slot-{i}" for i in range(max_options))),))
    (output / "schema.json").write_text(json.dumps(schema.to_json(), indent=2) + "\n", encoding="utf-8")

    handles = {
        name: (output / f"{name}.jsonl").open("w", encoding="utf-8")
        for name in ("train", "validation", "test")
    }
    counts = {name: 0 for name in handles}
    widest = 0
    try:
        lines = (graph_dir / "paths_finished.tsv").read_text(encoding="utf-8").splitlines()
        for row, line in enumerate(lines):
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            path = []
            for node in fields[3].split(";"):
                if node == "<":
                    if len(path) > 1:
                        path.pop()
                else:
                    path.append(node)
            if len(path) < 2:
                continue
            step = _stable(f"{fields[0]}:{fields[1]}:{row}") % (len(path) - 1)
            current, click, target = path[step], path[step + 1], path[-1]
            candidates = list(dict.fromkeys(outgoing.get(current, ())))
            if len(candidates) < 2 or click not in candidates:
                continue
            rng = random.Random(_stable(f"{row}:{target}:menu"))
            others = [item for item in candidates if item != click]
            rng.shuffle(others)
            menu = [click] + others[: max_options - 1]
            rng.shuffle(menu)
            widest = max(widest, len(menu))
            article = root / "plaintext_articles" / f"{current}.txt"
            body = " ".join(article.read_text(encoding="utf-8", errors="replace").split())
            titles = [_title(item) for item in menu]
            payload = {
                "context": (
                    f"Target article: {_title(target)}\n"
                    f"Current article: {_title(current)}\n{body[:2048]}"
                ),
                "options": {"click": titles},
                "labels": {"click": titles[menu.index(click)]},
            }
            bucket = _stable(target + ":split") % 10
            split = "test" if bucket == 0 else "validation" if bucket == 1 else "train"
            handles[split].write(json.dumps(payload, ensure_ascii=False) + "\n")
            counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    print(json.dumps({**counts, "widest_menu": widest}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    synthetic = commands.add_parser("synthetic")
    synthetic.add_argument("--output", type=Path, default=Path("data/synthetic"))
    synthetic.add_argument("--train", type=int, default=6000)
    synthetic.add_argument("--validation", type=int, default=1200)
    synthetic.add_argument("--test", type=int, default=1200)
    synthetic.add_argument("--seed", type=int, default=17)
    synthetic.add_argument("--cardinality-size", type=int, default=600)
    cards = commands.add_parser("cardinalities")
    cards.add_argument("--output", type=Path, default=Path("data/synthetic/cards"))
    cards.add_argument("--size", type=int, default=600)
    cards.add_argument("--seed", type=int, default=991)
    wiki = commands.add_parser("wikispeedia")
    wiki.add_argument("--root", type=Path, required=True)
    wiki.add_argument("--output", type=Path, required=True)
    wiki.add_argument("--max-options", type=int, default=512)
    args = parser.parse_args()
    cardinalities = (2, 4, 8, 16, 32, 64)
    if args.command == "synthetic":
        write_synthetic(args.output, {
            "train": args.train, "validation": args.validation, "test": args.test,
        }, args.seed, cardinalities)
        write_cardinality_sets(
            args.output / "cards", args.cardinality_size, args.seed + 991, cardinalities
        )
        print(json.dumps({"output": str(args.output), "cardinalities": list(cardinalities)}))
    elif args.command == "cardinalities":
        write_cardinality_sets(args.output, args.size, args.seed, cardinalities)
        print(json.dumps({"output": str(args.output)}))
    else:
        build_wikispeedia(args.root, args.output, args.max_options)


if __name__ == "__main__":
    main()
