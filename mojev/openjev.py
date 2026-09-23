"""Synthesise graded-preference data from the Open-Jev source generators.

Open-Jev ships deterministic generators for fifteen domains (110,324 rows in its
27B mixture: painting geometry, drone and browser snapshots, Snake, security
incidents, ViZDoom, reasoning controls, invoices, customer service, agent traces,
tic-tac-toe, wiki navigation, tile platformer, T-Rex). Their rows carry one-hot
targets: which option is correct, and nothing about the rest.

A Plackett-Luce objective needs more than that -- it needs an order over the
wrong answers too. This module reads Open-Jev rows and adds that order, taking it
from whatever structure the domain actually has rather than inventing tiers:

``palette``   RGB distance. A wrong colour close to the right one ranks above a
              distant one, and the distance is in the option text already.
``ordinal``   Score families list levels in order, so |level - correct| grades
              them directly.
``priority``  Routing and policy families apply ordered rules, so an option the
              rules reach earlier ranks above the fallback.
``binary``    Two options; nothing to grade beyond right and wrong.
``flat``      No recoverable structure: correct versus everything else, which is
              exactly what cross-entropy already sees.

Grades are integers, higher is better, and the correct option always takes the
maximum. Where a domain offers no structure the output degenerates to two levels
and Plackett-Luce reduces to cross-entropy on that row -- so adding a source is
never worse than not grading it.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

RGB = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]")
ORDINAL = re.compile(r"Exactly (\d+)")

# Ordered rule priorities, read off the generators' own rule lists in jev/data.py.
PRIORITIES = {
    "security": 3, "incident": 2, "billing": 1, "general support": 0,
    "ineligible": 2, "eligible": 2, "manual review": 1,
    "entailed": 2, "contradicted": 2, "unknown": 1,
}


def parse_rgb(option: str) -> tuple[int, int, int] | None:
    match = RGB.search(option)
    return tuple(int(g) for g in match.groups()) if match else None


def grade_palette(options: list[str], correct: int) -> list[int] | None:
    """Grade by RGB distance to the correct colour, in four bands."""
    colours = [parse_rgb(option) for option in options]
    if any(colour is None for colour in colours):
        return None
    target = colours[correct]
    distances = [
        sum((a - b) ** 2 for a, b in zip(colour, target)) ** 0.5 for colour in colours
    ]
    furthest = max(distances) or 1.0
    grades = []
    for index, distance in enumerate(distances):
        if index == correct:
            grades.append(3)
        else:
            # Three bands over the remaining range; nearer colour, higher grade.
            share = distance / furthest
            grades.append(2 if share < 0.34 else (1 if share < 0.67 else 0))
    return grades


def grade_ordinal(options: list[str], correct: int, values: list | None) -> list[int] | None:
    """Grade by distance along the ordered scale."""
    levels = values
    if levels is None:
        parsed = [ORDINAL.search(option) for option in options]
        if any(match is None for match in parsed):
            return None
        levels = [int(match.group(1)) for match in parsed]
    if len(levels) != len(options):
        return None
    span = max(1, max(levels) - min(levels))
    grades = []
    for index, level in enumerate(levels):
        if index == correct:
            grades.append(3)
        else:
            gap = abs(level - levels[correct]) / span
            grades.append(2 if gap <= 0.25 else (1 if gap <= 0.6 else 0))
    return grades


def grade_priority(options: list[str], correct: int) -> list[int] | None:
    """Grade by where the domain's ordered rules place each option."""
    ranks = [PRIORITIES.get(option.strip().lower()) for option in options]
    if any(rank is None for rank in ranks):
        return None
    grades = []
    for index, rank in enumerate(ranks):
        if index == correct:
            grades.append(3)
        else:
            # A rule reached earlier is a nearer miss than the fallback.
            grades.append(min(2, max(0, rank)))
    return grades


def grade_distance(options: list[str], metadata: dict) -> list[int] | None:
    """Grade by graph distance to the target, when the source supplies it.

    The wiki-navigation source records ``candidate_distances`` -- how many hops
    each candidate link sits from the target article. That is an objective
    ordering over every option, not a heuristic tier, so it is the strongest
    grading signal available and is tried before anything else. A link two hops
    out genuinely is a better move than one three hops out, even though only the
    nearest ones count as correct.
    """
    distances = metadata.get("candidate_distances")
    if not isinstance(distances, dict):
        return None
    values = [distances.get(option) for option in options]
    if any(value is None for value in values):
        return None
    # Rank ascending distance into grades, best distance highest.
    order = sorted(set(values))
    rank = {value: len(order) - 1 - index for index, value in enumerate(order)}
    return [rank[value] for value in values]


def grade_row(row: dict) -> tuple[list[int], str]:
    """(grades, strategy) for one Open-Jev row."""
    options = row["options"]
    target = row["target"]
    metadata = row.get("metadata", {})
    correct = max(range(len(target)), key=target.__getitem__)
    graph = grade_distance(options, metadata)
    if graph is not None and len(set(graph)) > 1:
        return graph, "distance"
    if len(options) == 2:
        return [3 if i == correct else 0 for i in range(2)], "binary"
    for name, grades in (
        ("palette", grade_palette(options, correct)),
        ("ordinal", grade_ordinal(options, correct, metadata.get("score_values"))),
        ("priority", grade_priority(options, correct)),
    ):
        if grades is not None:
            return grades, name
    return [3 if i == correct else 0 for i in range(len(options))], "flat"


def convert(row: dict, field_name: str = "answer") -> dict:
    """One Open-Jev row -> one row in this repository's schema, with grades."""
    grades, strategy = grade_row(row)
    state = row["state"]
    context = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    correct = max(range(len(row["target"])), key=row["target"].__getitem__)
    return {
        "context": f"{row['question']}\n\nState: {context}",
        "options": {field_name: list(row["options"])},
        "preference": {field_name: grades},
        "labels": {field_name: row["options"][correct]},
        # Kept on the row so downstream rebalancing can group by source and by
        # grading strategy without re-deriving either from the context text.
        "source": row["source"],
        "grading": strategy,
        "meta": {
            "source": row["source"],
            "kind": row["kind"],
            "split": row["split"],
            "group_id": row["group_id"],
            "grading": strategy,
            "cardinality": len(row["options"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+",
                        help="Open-Jev dataset directories (each holding <split>.jsonl)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--field", default="answer")
    parser.add_argument("--max-options", type=int, default=0,
                        help="skip rows wider than this; 0 keeps all")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    handles = {
        name: (args.output / f"{name}.jsonl").open("w", encoding="utf-8")
        for name in ("train", "validation", "test", "ood")
    }
    stats: dict[str, dict] = {}
    widest = 0
    try:
        for directory in args.sources:
            for path in sorted(Path(directory).glob("*.jsonl")):
                split = path.stem
                if split == "calibration":
                    split = "validation"        # this repo has no separate calibration split
                if split not in handles:
                    continue
                for line in path.open(encoding="utf-8"):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if args.max_options and len(row["options"]) > args.max_options:
                        continue
                    out = convert(row, args.field)
                    meta = out.pop("meta")
                    handles[split].write(json.dumps(out, ensure_ascii=False) + "\n")
                    entry = stats.setdefault(meta["source"], {
                        "rows": 0, "grading": {}, "kinds": {}, "max_options": 0,
                    })
                    entry["rows"] += 1
                    entry["grading"][meta["grading"]] = entry["grading"].get(meta["grading"], 0) + 1
                    entry["kinds"][meta["kind"]] = entry["kinds"].get(meta["kind"], 0) + 1
                    entry["max_options"] = max(entry["max_options"], meta["cardinality"])
                    widest = max(widest, meta["cardinality"])
    finally:
        for handle in handles.values():
            handle.close()

    schema = {"fields": [{
        "name": args.field, "kind": "choice",
        "options": [f"slot-{i}" for i in range(max(widest, 2))],
        "description": "the option the state and question license",
    }]}
    (args.output / "schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    total = sum(entry["rows"] for entry in stats.values())
    print(json.dumps({"total_rows": total, "widest_option_set": widest,
                      "by_source": dict(sorted(stats.items()))}, indent=2))


if __name__ == "__main__":
    main()
