"""Causal inference on a scorer that returns probabilities.

Three analyses use `drone-control-v1` rows. Their numeric sensor readings and
generator rules specify the mechanism:

    closing            = max(0, drone_speed - obstacle_speed)
    stopping_distance  = closing*reaction + closing^2/(2*braking) + margin
    TTC                = ahead_m / closing
    risk               = 2 if ahead_m <= stopping_distance or TTC <= imminent
                         1 if TTC <= caution or the side gaps are too narrow
                         0 otherwise

The generator rules supply a reference causal graph for three analyses:

1. **Intervention** -- do(variable = x). Sweep one variable, hold the rest fixed,
   and record the risk distribution as a dose-response curve.

2. **The causal graph** -- intervene on each variable separately and measure the
   distribution shift. Compare the measured effects with the generator rules.

3. **Sequential updating** -- combine several model outputs with a tempered
   product. Normalised entropy sets the weight of each update.

The scripts measure and print results from the selected checkpoint at run time.

The drone analysis uses the 16,384-token training and inference window.
Step 0 prints token coverage for the selected row.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch


def load_scene(path: Path, source: str = "drone-control-v1") -> dict:
    """The first row from ``source``, parsed into (prefix, state, suffix, row)."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("source") != source:
                continue
            context = row["context"]
            start, end = context.index("{"), context.rindex("}") + 1
            return {
                "prefix": context[:start],
                "state": json.loads(context[start:end]),
                "suffix": context[end:],
                "options": tuple(row["options"]["answer"]),
                "label": row["labels"]["answer"],
            }
    raise SystemExit(f"no {source} rows in {path}")


def render(scene: dict, state: dict) -> str:
    """Put a modified state back into the context the model was trained on."""
    return scene["prefix"] + json.dumps(state, sort_keys=True) + scene["suffix"]


class Scorer:
    """One forward pass in, a distribution plus its entropy out."""

    def __init__(self, checkpoint: str, device: str = "auto",
                 context_tokens: int = 16_384) -> None:
        from transformers import AutoTokenizer

        from mojev.evaluate import load_packed
        from mojev.full import packed_collate, select_device

        self.device = select_device(device)
        self.model, self.schema, self.config = load_packed(checkpoint, self.device)
        if context_tokens < 1:
            raise ValueError("context_tokens must be positive")
        self.context_tokens = context_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self._collate = packed_collate

    @torch.no_grad()
    def __call__(self, context: str, options: tuple[str, ...]) -> dict:
        from mojev.data import Example
        from mojev.full import move, sort_candidates, unsort
        from mojev.schema import Field, Schema

        # Sort before packing and map probabilities back to the caller's order.
        order = sort_candidates(list(options))
        menu = tuple(options[index] for index in order)
        schema = Schema((Field("risk", "choice", menu,
                               "How immediate is the collision risk?"),))
        collate = self._collate(self.tokenizer, schema,
                                self.context_tokens)
        batch = move(collate([Example(context=context, labels=(0,),
                                      options=(menu,))]), self.device)
        sorted_probabilities = self.model(batch)[0, 0, :len(menu)].softmax(-1)
        probabilities = unsort(order, sorted_probabilities.tolist())
        vector = torch.tensor(probabilities).clamp_min(1e-12)
        return {
            "p": dict(zip(options, probabilities)),
            # Normalised, so entropies from menus of different sizes compare.
            "entropy": float(-(vector * vector.log()).sum()) / math.log(len(options)),
        }


# --------------------------------------------------------------- 1. intervention

def intervene(scorer: Scorer, scene: dict, path: list, values: list) -> list[dict]:
    """do(variable = v) for each v, everything else held fixed.

    Set the selected value while holding the remaining state fixed. Under the
    generator rules, the resulting curve measures that variable's effect.
    """
    out = []
    for value in values:
        state = copy.deepcopy(scene["state"])
        target = state
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        result = scorer(render(scene, state), scene["options"])
        out.append({"value": value, **result})
    return out


def dose_response(scorer: Scorer, scene: dict) -> dict:
    """Sweep the level-2 threshold, which the rules make a direct cause.

    Raising the threshold declares more situations imminent, so risk should rise.
    """
    sweep = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    curve = intervene(scorer, scene,
                      ["observation", "control_policy", "imminent_ttc_s"], sweep)
    return {"variable": "control_policy.imminent_ttc_s", "curve": curve}


def truncation_note(scorer: Scorer, scene: dict) -> dict:
    """Measure token coverage under the inference context budget.

    Report token coverage and the visibility of the policy and obstacle keys.
    Their positions follow JSON key order.
    """
    tokenizer = scorer.tokenizer
    baseline = tokenizer(render(scene, scene["state"]))["input_ids"]
    budget = scorer.context_tokens
    visible = tokenizer.decode(baseline[:budget])
    return {
        "state_tokens": len(baseline),
        "context_tokens": budget,
        "fraction_read": round(min(budget, len(baseline)) / len(baseline), 3),
        "obstacles_visible": "obstacles" in visible,
        "control_policy_visible": "control_policy" in visible,
    }


# ------------------------------------------------------------- 2. causal graph

# Generator-rule effects used as a reference for the measured shifts.
MECHANISM = {
    "imminent_ttc_s": "cause: the level-2 threshold -- risk 2 when TTC <= this",
    "caution_ttc_s": "cause: the level-1 threshold -- risk 1 when TTC <= this",
    "braking_deceleration_mps2": "cause: divides closing^2 in stopping_distance",
    "reaction_s": "cause: multiplies closing in stopping_distance",
    "margin_m": "cause: added to stopping_distance, and to the intersection test",
    "failed_scan_count": "target-loss rule; outside the collision-risk rule",
}


def total_variation(before: dict, after: dict) -> float:
    """Half the L1 distance between two distributions over the same options."""
    return 0.5 * sum(abs(after[k] - before[k]) for k in before)


def recover_edges(scorer: Scorer, scene: dict) -> list[dict]:
    """Intervene on each variable and measure how much the distribution moves.

    An edge is claimed when setting a variable to a contrasting value shifts the
    risk distribution. The effect size is total variation, which is bounded in
    [0, 1] and so comparable across variables of different units.
    """
    baseline = scorer(render(scene, scene["state"]), scene["options"])

    # Policy probes follow the generator rules; failed_scan_count controls for
    # the separate target-loss rule.
    probes = [
        ("imminent_ttc_s", [0.1, 30.0]),
        ("caution_ttc_s", [0.2, 60.0]),
        ("braking_deceleration_mps2", [0.5, 20.0]),
        ("reaction_s", [0.01, 3.0]),
        ("margin_m", [0.01, 5.0]),
        ("failed_scan_count", [0, 99]),
    ]
    probes = [(name, ["observation", "control_policy", name], values)
              for name, values in probes]

    edges = []
    for name, path, values in probes:
        results = intervene(scorer, scene, path, values)
        # The effect is the largest move any setting produces, which is what
        # "does this variable matter at all" asks.
        effect = max(total_variation(baseline["p"], r["p"]) for r in results)
        edges.append({
            "variable": name,
            "effect_total_variation": round(effect, 4),
            "entropy_range": [round(min(r["entropy"] for r in results), 4),
                              round(max(r["entropy"] for r in results), 4)],
            "mechanism_says": MECHANISM.get(name, "?"),
            "settings": {str(r["value"]): {k: round(v, 4) for k, v in r["p"].items()}
                         for r in results},
        })
    return sorted(edges, key=lambda e: -e["effect_total_variation"])


# ---------------------------------------------------- 3. tempered aggregation

def tempered_update(prior: dict, scores: dict, weight: float = 1.0) -> dict:
    """Multiply normalised score vectors with a temperature weight.

    ``p^weight`` tempers each model-output vector: weight 1 uses it fully;
    weight 0 leaves the prior unchanged. Weight follows normalised entropy.
    """
    unnormalised = {k: prior[k] * (max(scores[k], 1e-12) ** weight)
                    for k in prior}
    total = sum(unnormalised.values())
    return {k: v / total for k, v in unnormalised.items()}


def sequential_evidence(scorer: Scorer, scene: dict) -> dict:
    """Aggregate model outputs under three settings of the policy threshold."""
    # Vary the visible threshold to produce three distinct model outputs.
    base = scene["state"]["observation"]["control_policy"]["imminent_ttc_s"]
    readings = [round(base * factor, 3) for factor in (0.5, 1.0, 8.0)]
    observations = intervene(
        scorer, scene, ["observation", "control_policy", "imminent_ttc_s"], readings
    )

    options = scene["options"]
    uniform = {k: 1.0 / len(options) for k in options}
    steps = []
    for scheme in ("unweighted", "entropy_weighted"):
        posterior = dict(uniform)
        trace = []
        for observation in observations:
            # Certain observations count fully; ambiguous ones are discounted.
            weight = 1.0 if scheme == "unweighted" else 1.0 - observation["entropy"]
            posterior = tempered_update(posterior, observation["p"], weight)
            trace.append({
                "reading": round(observation["value"], 4),
                "observation_entropy": round(observation["entropy"], 4),
                "weight": round(weight, 4),
                "posterior": {k: round(v, 4) for k, v in posterior.items()},
            })
        steps.append({"scheme": scheme, "trace": trace,
                      "final": {k: round(v, 4) for k, v in posterior.items()}})
    return {"variable": "control_policy.imminent_ttc_s", "schemes": steps}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="model directory")
    parser.add_argument("--data", type=Path, default=Path("data/mix/test.jsonl"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--context-tokens", type=int, default=16_384,
                        help="inference window for the drone probe (default: 16384)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    scorer = Scorer(args.checkpoint, args.device, args.context_tokens)
    scene = load_scene(args.data)
    report = {"label": scene["label"], "options": list(scene["options"])}

    print("=" * 72)
    print("0. INPUT TOKEN COVERAGE")
    print("=" * 72)
    report["truncation"] = truncation_note(scorer, scene)
    note = report["truncation"]
    print(f"  state is {note['state_tokens']} tokens; inference window is "
          f"{note['context_tokens']} tokens -- {note['fraction_read']:.0%} retained")
    print(f"  control_policy visible: {note['control_policy_visible']}")
    print(f"  obstacles visible:      {note['obstacles_visible']}")
    print("\n  Interventions target the control_policy block.")

    print()
    print("=" * 72)
    print("1. INTERVENTION  do(imminent_ttc_s = x), everything else held fixed")
    print("=" * 72)
    report["intervention"] = dose_response(scorer, scene)
    names = scene["options"]
    print(f"{'ttc_s':>9s} " + " ".join(f"{n[:16]:>17s}" for n in names)
          + f" {'entropy':>8s}")
    for point in report["intervention"]["curve"]:
        row = " ".join(f"{point['p'][n]:17.4f}" for n in names)
        print(f"{point['value']:9.2f} {row} {point['entropy']:8.4f}")
    print("\nThe generator's rule: risk is level 2 when TTC <= imminent_ttc_s.")
    print("Raising the threshold calls more situations imminent, so risk should rise.")

    print()
    print("=" * 72)
    print("2. CAUSAL GRAPH  effect of intervening on each variable")
    print("=" * 72)
    report["graph"] = recover_edges(scorer, scene)
    print(f"{'variable':44s} {'effect':>7s}  mechanism")
    for edge in report["graph"]:
        print(f"{edge['variable']:44s} {edge['effect_total_variation']:7.4f}  "
              f"{edge['mechanism_says'][:60]}")
    print("\nEffect is total variation of the risk distribution, in [0, 1].")
    print("Variables the rules call non-causal should sit near 0.")

    print()
    print("=" * 72)
    print("3. TEMPERED AGGREGATION  three threshold settings")
    print("=" * 72)
    report["bayes"] = sequential_evidence(scorer, scene)
    for scheme in report["bayes"]["schemes"]:
        print(f"\n{scheme['scheme']}:")
        for step in scheme["trace"]:
            posterior = " ".join(f"{k[:12]}={v:.4f}"
                                 for k, v in step["posterior"].items())
            print(f"  ttc_s={step['reading']:7.3f} "
                  f"H={step['observation_entropy']:.4f} "
                  f"w={step['weight']:.4f}  ->  {posterior}")
    print("\nWeighting by (1 - entropy) reduces the effect of high-entropy outputs.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
