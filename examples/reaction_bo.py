"""Bayesian optimisation of reaction conditions, with the scorer as the surrogate.

The task is to find a high-yielding C-N coupling condition with a small
experiment budget. The search space is a complete 4x12x4x3x3 factorial grid
with measured yields for all 1,728 conditions. The optimiser receives yields
only for conditions it selects.

The scorer ranks candidate conditions using experiments recorded as state text:

    Experiments already run, condition -> yield percent:
      CsOAc / CgMe-PPh / DMAc / 0.153M / 105C -> 100%
      KOAc / PPh3 / BuCN / 0.1M / 120C -> 12%
      ...
    Which untested condition gives the highest yield?

The untested conditions form the candidate menu. One forward pass returns a
probability per candidate, used here as the acquisition score.

The model compares condition descriptions and observed yields in context.

**The `context_tokens` setting determines how much experiment history is read.**
The Qwen3.5 backbone supports 262,144 tokens natively and approximately
1 million with YaRN scaling. Training truncates state inputs at 16,384 tokens.
Measured on this task with two inference-window settings:

    encoding             ctx    picks the best   mean yield picked
    compressed codes   16384       13.3%              17.18      (random: 12.5%)
    full condition text  2048      26.7%              28.72      (random pick: 17.83)

Both rows use the same weights and data. This example uses the 16,384-token
training window by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path

import torch

KEYS = ("base", "ligand", "solvent", "concentration", "temperature")


def load_grid(path: Path) -> list[dict]:
    """The factorial grid, with yield and cost as floats."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["y"] = float(row["yield"])
            row["c"] = float(row["cost"])
            rows.append(row)
    if not rows:
        raise SystemExit(f"no rows in {path}")
    return rows


def describe(row: dict) -> str:
    """One condition as the words the model will score."""
    return (f"{row['base']} / {row['ligand']} / {row['solvent']} / "
            f"{row['concentration']}M / {row['temperature']}C")


class Surrogate:
    """The scorer, used as a BO surrogate over text.

    ``context_tokens`` is configurable for experiment histories of different
    lengths. The default is 16,384; ``truncates`` reports coverage at run time.
    """

    def __init__(self, checkpoint: str, context_tokens: int = 16_384,
                 device: str = "auto") -> None:
        from transformers import AutoTokenizer

        from mojev.evaluate import load_packed
        from mojev.full import packed_collate, select_device

        self.device = select_device(device)
        self.model, _, self.config = load_packed(checkpoint, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.context_tokens = context_tokens
        self._packed_collate = packed_collate

    def state(self, observed: list[dict], goal: str) -> str:
        """Observed experiments, best first, as the state to read."""
        lines = ["Reaction yield optimisation. Experiments already run, "
                 "condition -> yield percent:"]
        for row in sorted(observed, key=lambda r: -r["y"]):
            lines.append(f"  {describe(row)} -> {row['y']:.0f}%")
        lines.append(goal)
        return "\n".join(lines)

    @torch.no_grad()
    def acquire(self, observed: list[dict], candidates: list[dict],
                goal: str = "Which untested condition gives the highest yield?",
                question: str = "Highest yield?") -> list[float]:
        """Return per-candidate probabilities as acquisition scores."""
        from mojev.data import Example
        from mojev.full import move, sort_candidates, unsort
        from mojev.schema import Field, Schema

        menu = tuple(describe(row) for row in candidates)
        # Sort before packing and map scores back to the caller's order.
        order = sort_candidates(list(menu))
        sorted_menu = tuple(menu[index] for index in order)
        schema = Schema((Field("pick", "choice", sorted_menu, question),))
        collate = self._packed_collate(self.tokenizer, schema,
                                       self.context_tokens)
        batch = move(collate([Example(context=self.state(observed, goal),
                                      labels=(0,), options=(sorted_menu,))]),
                     self.device)
        probabilities = self.model(batch)[0, 0, :len(menu)].softmax(-1)
        return unsort(order, probabilities.tolist())

    def truncates(self, observed: list[dict], goal: str) -> dict:
        """Report whether the experiment history fits the context budget."""
        tokens = len(self.tokenizer(self.state(observed, goal))["input_ids"])
        return {"state_tokens": tokens, "context_tokens": self.context_tokens,
                "truncated": tokens > self.context_tokens}


def optimise(surrogate: Surrogate, grid: list[dict], rounds: int,
             batch: int, seed: int, initial: int = 8,
             pool_size: int = 0) -> dict:
    """Run BO, and report the best yield found after each round.

    Each round scores every untested condition in one forward pass and selects
    ``batch`` conditions from a high-scoring shortlist. With about 1,720
    candidates, the packed sequence is approximately 36,850 tokens and the
    tree mask is approximately 5.5 GB.
    """
    rng = random.Random(seed)
    observed = rng.sample(grid, initial)
    untested = [row for row in grid if row not in observed]
    trace = [{"round": 0, "experiments": len(observed),
              "best_yield": max(r["y"] for r in observed)}]

    for index in range(1, rounds + 1):
        if not untested:
            break
        searchable = (rng.sample(untested, min(pool_size, len(untested)))
                      if pool_size else untested)
        scores = surrogate.acquire(observed, searchable)
        # Sample the batch from a high-scoring shortlist.
        # Measured over all 1,720 untested conditions: the model's top-10 average
        # 55.71 real yield against 53.33 for its top-3 and 19.43 for the space,
        # The shortlist provides exploration around the top-ranked conditions.
        ranked = [row for row, _ in
                  sorted(zip(searchable, scores), key=lambda pair: -pair[1])]
        shortlist = ranked[:max(batch * 4, 10)]
        chosen = rng.sample(shortlist, min(batch, len(shortlist)))
        observed += chosen
        untested = [row for row in untested if row not in chosen]
        trace.append({
            "round": index,
            "experiments": len(observed),
            "best_yield": max(r["y"] for r in observed),
            "picked": [{"condition": describe(row), "yield": row["y"]}
                       for row in chosen],
        })
    return {"trace": trace, "observed": len(observed),
            "best_yield": trace[-1]["best_yield"]}


def random_baseline(grid: list[dict], rounds: int, batch: int, seed: int,
                    initial: int = 8) -> list[float]:
    """Use the same experiment budget for a random-search baseline."""
    rng = random.Random(seed)
    observed = rng.sample(grid, initial)
    untested = [row for row in grid if row not in observed]
    best = [max(r["y"] for r in observed)]
    for _ in range(rounds):
        chosen = rng.sample(untested, min(batch, len(untested)))
        observed += chosen
        untested = [row for row in untested if row not in chosen]
        best.append(max(r["y"] for r in observed))
    return best


def cost_aware(surrogate: Surrogate, grid: list[dict], seed: int,
               observed_count: int = 24, pool_size: int = 10) -> dict:
    """Ask for cheap *and* good, and see whether the ranking moves.

    Yield and cost are near-independent in this grid (correlation -0.0025), so a
    cost supplies a second objective. Rewording the question changes the
    requested ranking while keeping the model weights fixed.
    """
    rng = random.Random(seed)
    observed = rng.sample(grid, observed_count)
    pool = rng.sample([row for row in grid if row not in observed], pool_size)

    plain = surrogate.acquire(observed, pool)
    thrifty = surrogate.acquire(
        observed, pool,
        goal="Which untested condition gives a high yield at low reagent cost?",
        question="Best yield per unit cost?")
    return {
        "candidates": [
            {"condition": describe(row), "yield": row["y"], "cost": round(row["c"], 4),
             "p_yield": round(a, 4), "p_thrifty": round(b, 4)}
            for row, a, b in sorted(zip(pool, plain, thrifty),
                                    key=lambda t: -t[1])
        ],
        "picked_for_yield": describe(max(zip(pool, plain), key=lambda t: t[1])[0]),
        "picked_for_cost": describe(max(zip(pool, thrifty), key=lambda t: t[1])[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="model directory")
    parser.add_argument("--csv", type=Path, required=True,
                        help="experiments_yield_and_cost.csv")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--pool", type=int, default=0,
                        help="0 searches every untested condition")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--context-tokens", type=int, default=16_384)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    grid = load_grid(args.csv)
    surrogate = Surrogate(args.checkpoint, args.context_tokens, device=args.device)
    ceiling = max(row["y"] for row in grid)

    print(f"grid: {len(grid)} conditions, best yield {ceiling:.1f}%")
    fit = surrogate.truncates(grid[:24], "Which untested condition?")
    print(f"state with 24 observations: {fit['state_tokens']} tokens, "
          f"budget {fit['context_tokens']}"
          + ("  <-- TRUNCATED" if fit["truncated"] else "  ok"))

    print(f"\nBO vs random, {args.repeats} seeds, {args.rounds} rounds x "
          f"{args.batch} experiments from a pool of {args.pool}")
    print(f"{'experiments':>12s} {'BO best':>9s} {'random':>9s} {'gap':>7s}")

    runs, baselines = [], []
    for seed in range(args.repeats):
        runs.append(optimise(surrogate, grid, args.rounds, args.batch, seed,
                             pool_size=args.pool))
        baselines.append(random_baseline(grid, args.rounds, args.batch, seed))

    for step in range(args.rounds + 1):
        bo = statistics.fmean(run["trace"][step]["best_yield"] for run in runs)
        rand = statistics.fmean(base[step] for base in baselines)
        count = runs[0]["trace"][step]["experiments"]
        print(f"{count:12d} {bo:9.2f} {rand:9.2f} {bo - rand:+7.2f}")

    report = {
        "grid_size": len(grid), "best_possible": ceiling,
        "context_tokens": args.context_tokens,
        "bo": [run["trace"] for run in runs],
        "random": baselines,
    }

    print("\ncost-aware: same candidates, question reworded")
    report["cost_aware"] = cost_aware(surrogate, grid, seed=0)
    print(f"{'condition':44s} {'yield':>7s} {'cost':>8s} {'p(yield)':>9s} {'p(cheap)':>9s}")
    for entry in report["cost_aware"]["candidates"]:
        print(f"{entry['condition'][:44]:44s} {entry['yield']:7.2f} "
              f"{entry['cost']:8.4f} {entry['p_yield']:9.4f} {entry['p_thrifty']:9.4f}")
    print(f"\n  asked for yield -> {report['cost_aware']['picked_for_yield']}")
    print(f"  asked for cheap -> {report['cost_aware']['picked_for_cost']}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
