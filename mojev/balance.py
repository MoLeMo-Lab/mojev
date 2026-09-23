"""Cap and rebalance the converted Open-Jev mixture.

Raw conversion gives 1,271,770 rows, but two sources hold 83% of them: painting
geometry (854,272) and mailroom invoices (200,900). Training on that mixture as-is
produces a painting-geometry specialist, while the sources that carry the most
useful supervision are the smallest -- wikispeedia has 1,541 rows and 97.8% of
them come with objective graph-distance grades.

Two caps, applied per source and per split:

* rows whose grading gives three or more levels (ordinal, palette, distance,
  priority) survive to a higher cap, because those are the rows where
  Plackett-Luce has an ordering to learn that cross-entropy cannot see
* rows with two levels (binary, flat) are capped lower -- on those rows PL
  reduces to cross-entropy, so extra copies add volume without adding signal

Sources below the cap are kept whole. Selection is by a stable hash of the row,
so the same subset comes back for a given seed regardless of file order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

GRADED = {"ordinal", "palette", "distance", "priority"}


def levels(grades: list[int]) -> int:
    return len({g for g in grades if g >= 0})


def stable_key(row: dict, seed: int) -> int:
    payload = json.dumps([seed, row["context"], row["options"]], sort_keys=True)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graded-cap", type=int, default=60_000,
                        help="per-source cap for rows with three or more grade levels")
    parser.add_argument("--plain-cap", type=int, default=20_000,
                        help="per-source cap for rows where PL reduces to cross-entropy")
    parser.add_argument("--field", default="answer")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    for split in ("train", "validation", "test", "ood"):
        source_path = args.input / f"{split}.jsonl"
        if not source_path.exists():
            continue
        # Group by (source, graded?) so caps apply per source, then sample by hash.
        buckets: dict[tuple, list] = {}
        with source_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                grades = row.get("preference", {}).get(args.field, [])
                graded = levels(grades) >= 3
                buckets.setdefault((row.get("source", "?"), graded), []).append(row)

        kept = []
        for (source, graded), rows in buckets.items():
            cap = args.graded_cap if graded else args.plain_cap
            if len(rows) <= cap:
                kept.extend(rows)
                continue
            rows.sort(key=lambda r: stable_key(r, args.seed))
            kept.extend(rows[:cap])

        kept.sort(key=lambda r: stable_key(r, args.seed + 1))
        with (args.output / f"{split}.jsonl").open("w", encoding="utf-8") as out:
            graded_kept = 0
            for row in kept:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                if levels(row.get("preference", {}).get(args.field, [])) >= 3:
                    graded_kept += 1
        report[split] = {"rows": len(kept), "graded_rows": graded_kept,
                         "graded_share": round(graded_kept / max(len(kept), 1), 4),
                         "buckets": len(buckets)}

    schema = args.input / "schema.json"
    if schema.exists():
        (args.output / "schema.json").write_text(schema.read_text())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
