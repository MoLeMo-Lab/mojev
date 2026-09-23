"""Calibration: the objective, the correction, and the measurement.

jevlike measured calibration and trained on likelihood alone. Its expected
calibration error therefore drifted with the number of options -- measured on a
trained checkpoint, mean confidence minus accuracy went from -0.112 at two
options to +0.015 at sixteen, crossing zero on the way. A router that needs
"escalate below 0.8" cannot use a model whose 0.8 means something different for
a three-item menu than for a fifty-item one.

Three pieces address that:

``brier`` -- a strictly proper scoring rule, differentiable, added to the
training loss. Proper means the score is optimal exactly when the reported
probabilities are the true ones, so minimising it rewards honesty rather than
confidence. Expected calibration error cannot play this role: it bins, so it has
zero gradient almost everywhere.

``CardinalityTemperature`` -- one learned temperature per cardinality bucket,
fitted on validation data after training. Temperature scaling cannot change which
option wins, so accuracy is untouched; grouping by cardinality is what removes
the drift rather than averaging it away.

``calibration_report`` -- the gap reported per cardinality, never pooled. A
single pooled number hides exactly the defect being fixed: overconfidence at one
end and underconfidence at the other average to something that looks fine.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

# Bucket edges: a temperature is shared by cardinalities inside one bucket, so
# rare option counts still get a usable estimate.
BUCKET_EDGES = (2, 3, 5, 9, 17, 33)


def cardinality_bucket(cardinality: torch.Tensor) -> torch.Tensor:
    """Map live option counts to bucket indices (0..len(BUCKET_EDGES))."""
    bucket = torch.zeros_like(cardinality)
    for edge in BUCKET_EDGES[1:]:
        bucket = bucket + (cardinality >= edge).long()
    return bucket


def brier(probabilities: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error between the probability vector and the one-hot target.

    Summed over options, averaged over rows. Masked (absent) options contribute
    nothing. This is the multi-class Brier score.
    """
    squared = (probabilities - targets).pow(2) * mask
    return squared.sum(-1).mean()


def single_field_loss(
    logits: torch.Tensor,       # (B, N) for one field
    labels: torch.Tensor,       # (B,)
    mask: torch.Tensor,         # (B, N)
    brier_weight: float,
    class_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy plus a Brier term over one single-choice field.

    ``class_weight`` counters an imbalanced field collapsing to its majority
    class. Measured case: ``answerable`` is 80/20, and an unweighted byte model
    scored exactly the majority rate (0.8023) with zero lift over a shuffled
    context -- a constant predictor that accuracy alone rates highly.
    """
    loss = F.cross_entropy(logits, labels, weight=class_weight)
    if brier_weight > 0:
        probabilities = logits.softmax(-1)
        targets = F.one_hot(labels, logits.shape[-1]).to(probabilities.dtype)
        loss = loss + brier_weight * brier(probabilities, targets, mask.to(probabilities.dtype))
    return loss


def multi_field_loss(
    logits: torch.Tensor,       # (B, N)
    targets: torch.Tensor,      # (B, N) 0/1
    mask: torch.Tensor,         # (B, N)
    brier_weight: float,
) -> torch.Tensor:
    """Independent per-option binary cross-entropy plus a Brier term."""
    weights = mask.to(logits.dtype)
    element = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = (element * weights).sum(-1).div(weights.sum(-1).clamp_min(1)).mean()
    if brier_weight > 0:
        probabilities = logits.sigmoid()
        loss = loss + brier_weight * brier(probabilities, targets, weights)
    return loss


def class_weights(schema, counts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Inverse-frequency weights per single-choice field, normalised to mean 1.

    Only fields whose options are fixed by the schema get weights. A field with
    per-row menus (a click list, a candidate answer set) has no stable class
    identity across rows, so frequency weighting is meaningless there.
    """
    weights = {}
    for field in schema:
        if not field.single or field.name not in counts:
            continue
        count = counts[field.name].float().clamp_min(1.0)
        weight = count.sum() / (len(count) * count)
        weights[field.name] = weight / weight.mean()
    return weights


def plackett_luce(logits: torch.Tensor, grades: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
    """Plackett-Luce likelihood of the graded option ordering.

    The options carry a quality grade, and the loss asks the model to rank them
    in that order: the top group over everything below it, then the next group
    over what remains, and so on. Each stage is a softmax over the options still
    standing.

    This is a training objective in its own right, not a correction applied to
    something already trained. Cross-entropy is its special case where the only
    grade is "correct versus everything else" -- one bit, no ordering among the
    wrong answers.

    Rows are normalised by their *own* number of stages. Packing puts ragged
    cardinalities in one batch -- a row with two options may have two grade
    levels while its neighbour has four -- and averaging per stage across rows
    would then weight those rows unequally: a row present in few stages would
    contribute less total loss than one present in many, purely because of how
    many options it happened to be given. Accumulating per row and dividing at
    the end makes every row count once.
    """
    floor = torch.finfo(logits.dtype).min
    scores = logits.masked_fill(~mask, floor)
    levels = sorted({int(v) for v in grades[mask].unique().tolist()}, reverse=True)
    per_row = scores.new_zeros(scores.shape[0])
    row_stages = scores.new_zeros(scores.shape[0])
    remaining = mask.clone()
    for level in levels[:-1]:              # the last group has nothing beneath it
        at_level = remaining & grades.eq(level)
        if not bool(at_level.any()):
            continue
        # A stage only exists for a row if something still ranks below the group.
        below = (remaining & ~at_level).any(-1)
        active = at_level.any(-1) & below
        if bool(active.any()):
            logp = scores.masked_fill(~remaining, floor).log_softmax(-1)
            picked = (logp * at_level).sum(-1)
            count = at_level.sum(-1).clamp_min(1)
            per_row = per_row + torch.where(
                active, -(picked / count), torch.zeros_like(per_row)
            )
            row_stages = row_stages + active.to(row_stages.dtype)
        remaining = remaining & ~at_level
    scored = row_stages > 0
    if not bool(scored.any()):
        return scores.new_zeros(())
    return (per_row[scored] / row_stages[scored]).mean()


def schema_loss(logits: torch.Tensor, batch: dict, schema, brier_weight: float,
                weights: dict[str, torch.Tensor] | None = None,
                objective: str = "ce") -> tuple:
    """Total loss over every field, plus a per-field breakdown."""
    total, parts = logits.new_zeros(()), {}
    weights = weights or {}
    grades = batch.get("preference")
    for index, field in enumerate(schema):
        field_logits = logits[:, index]
        field_mask = batch["option_mask"][:, index]
        if field.single:
            weight = weights.get(field.name)
            if weight is not None:
                # Logits are padded to the widest field's cardinality, so the
                # weight vector has to span that width too. Padding slots are
                # masked to finfo.min and can never be the label, so their
                # weight is never used -- one keeps cross_entropy happy.
                width = field_logits.shape[-1]
                weight = weight.to(field_logits.device)
                if weight.numel() < width:
                    weight = torch.cat([
                        weight, weight.new_ones(width - weight.numel())
                    ])
                else:
                    weight = weight[:width]
            field_grades = None if grades is None else grades[:, index]
            use_pl = (
                objective == "pl"
                and field_grades is not None
                and bool((field_grades[field_mask] >= 0).any())
            )
            if use_pl:
                # Plackett-Luce supplies the ordering; Brier is added on top as
                # a proper-scoring-rule regulariser rather than replacing it.
                # They answer different questions -- ranking versus honest
                # magnitudes -- so a field with grades wants both.
                loss = plackett_luce(field_logits, field_grades, field_mask)
                if brier_weight > 0:
                    labels = batch["labels"][:, index]
                    probabilities = field_logits.softmax(-1)
                    targets = F.one_hot(labels, field_logits.shape[-1]).to(probabilities.dtype)
                    loss = loss + brier_weight * brier(
                        probabilities, targets, field_mask.to(probabilities.dtype)
                    )
            else:
                loss = single_field_loss(
                    field_logits, batch["labels"][:, index], field_mask,
                    brier_weight, weight,
                )
        else:
            loss = multi_field_loss(
                field_logits, batch["multi_labels"][:, index], field_mask, brier_weight
            )
        parts[field.name] = float(loss.detach())
        total = total + loss
    return total / len(schema), parts


class CardinalityTemperature(nn.Module):
    """Per-(field, cardinality bucket) temperature applied to logits.

    Fitted after training, on validation data, by minimising negative log
    likelihood. Dividing logits by a positive scalar is monotone, so the argmax
    -- and therefore accuracy -- is unchanged.

    Absent options carry ``finfo.min`` so that softmax ignores them. Dividing
    that by a temperature overflows, so the mask is re-applied after scaling
    rather than scaled along with the live logits.
    """

    def __init__(self, fields: int, buckets: int = len(BUCKET_EDGES)) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(fields, buckets))

    def forward(self, logits: torch.Tensor, cardinality: torch.Tensor,
                option_mask: torch.Tensor | None = None) -> torch.Tensor:
        bucket = cardinality_bucket(cardinality).clamp_max(self.log_temperature.shape[1] - 1)
        temperature = self.log_temperature.exp()[
            torch.arange(logits.shape[1], device=logits.device)[None, :], bucket
        ]
        floor = torch.finfo(logits.dtype).min
        live = option_mask if option_mask is not None else logits > floor / 2
        safe = torch.where(live, logits, torch.zeros_like(logits))
        scaled = safe / temperature.unsqueeze(-1).clamp_min(1e-3)
        return torch.where(live, scaled, torch.full_like(scaled, floor))

    @torch.no_grad()
    def table(self) -> list[list[float]]:
        return self.log_temperature.exp().cpu().tolist()


def fit_temperature(
    logits: torch.Tensor,        # (R, F, N) collected over the validation split
    batch: dict,
    schema,
    steps: int = 300,
    learning_rate: float = 0.05,
) -> CardinalityTemperature:
    """Fit temperatures by NLL on held-out logits. Only the temperatures move."""
    scaler = CardinalityTemperature(len(schema)).to(logits.device)
    optimiser = torch.optim.LBFGS(
        scaler.parameters(), lr=learning_rate, max_iter=steps, line_search_fn="strong_wolfe"
    )
    cardinality = batch["cardinality"]

    def closure():
        optimiser.zero_grad(set_to_none=True)
        scaled = scaler(logits, cardinality, batch["option_mask"])
        total = scaled.new_zeros(())
        for index, field in enumerate(schema):
            if field.single:
                total = total + F.cross_entropy(scaled[:, index], batch["labels"][:, index])
            else:
                weights = batch["option_mask"][:, index].to(scaled.dtype)
                element = F.binary_cross_entropy_with_logits(
                    scaled[:, index], batch["multi_labels"][:, index], reduction="none"
                )
                total = total + (element * weights).sum(-1).div(
                    weights.sum(-1).clamp_min(1)
                ).mean()
        loss = total / len(schema)
        loss.backward()
        return loss

    optimiser.step(closure)
    return scaler


@torch.no_grad()
def field_confidence(logits: torch.Tensor, mask: torch.Tensor, single: bool):
    """(confidence, correct-mask helper) for one field's logits."""
    if single:
        probabilities = logits.softmax(-1)
        confidence, prediction = probabilities.max(-1)
        return confidence, prediction
    probabilities = logits.sigmoid()
    # For a multi field, the decision is per option; confidence is the distance
    # from the 0.5 boundary, averaged over live options.
    confidence = torch.maximum(probabilities, 1 - probabilities)
    return confidence, (probabilities >= 0.5).long()


def expected_calibration_error(
    confidence: torch.Tensor, correct: torch.Tensor, bins: int = 10
) -> float:
    """Binned |confidence - accuracy|, weighted by bin population."""
    if confidence.numel() == 0:
        return 0.0
    error = 0.0
    edges = torch.linspace(0, 1, bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        if selected.any():
            gap = correct[selected].float().mean() - confidence[selected].mean()
            error += float(selected.float().mean() * gap.abs())
    return error


def monotonicity(
    confidence: torch.Tensor, correct: torch.Tensor, bins: int = 10, minimum: int = 20
) -> dict:
    """The criterion TypeSafe actually states: higher confidence means higher accuracy.

    That is a statement about *order*, not about the size of the confidence
    minus accuracy gap. A model can be systematically 3 points overconfident at
    every level and still be perfectly usable for routing -- what breaks a
    threshold rule is confidence that does not rank outcomes.

    Reported as Spearman correlation between the confidence bin and the observed
    accuracy in it, plus the count of adjacent pairs that go the wrong way.
    Sparse bins are dropped: an accuracy estimated from five rows is noise.
    """
    curve = []
    edges = torch.linspace(0, 1, bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        if int(selected.sum()) < minimum:
            continue
        curve.append({
            "rows": int(selected.sum()),
            "confidence": float(confidence[selected].mean()),
            "accuracy": float(correct[selected].float().mean()),
        })
    if len(curve) < 3:
        return {"bins": len(curve), "spearman": None, "inversions": None, "monotone": None}
    accuracies = [point["accuracy"] for point in curve]
    inversions = sum(
        1 for index in range(len(accuracies) - 1)
        if accuracies[index] > accuracies[index + 1] + 1e-9
    )
    # Spearman over bin index against accuracy: ranks are what matter here.
    order = sorted(range(len(accuracies)), key=accuracies.__getitem__)
    ranks = [0.0] * len(accuracies)
    for rank, position in enumerate(order):
        ranks[position] = float(rank)
    count = len(ranks)
    mean_rank = (count - 1) / 2
    numerator = sum((index - mean_rank) * (ranks[index] - mean_rank) for index in range(count))
    denominator = sum((index - mean_rank) ** 2 for index in range(count))
    return {
        "bins": count,
        "spearman": numerator / denominator if denominator else None,
        "inversions": inversions,
        "monotone": inversions == 0,
        "curve": curve,
    }


@torch.no_grad()
def majority_baseline(batch: dict, schema) -> dict:
    """Accuracy of always predicting each field's most common label.

    Reported next to accuracy because a high score can mean the field was not
    learned at all. Measured case: a byte model scored 0.8023 on ``answerable``
    -- exactly the majority-class rate, with zero lift over a shuffled context.
    Accuracy alone called that a strong field.
    """
    out = {}
    for index, field in enumerate(schema):
        if field.single:
            labels = batch["labels"][:, index]
            counts = torch.bincount(labels, minlength=field.cardinality)
            out[field.name] = float(counts.max()) / max(int(counts.sum()), 1)
        else:
            targets = batch["multi_labels"][:, index]
            live = batch["option_mask"][:, index]
            positive = float((targets * live).sum())
            total = float(live.sum())
            rate = positive / max(total, 1.0)
            out[field.name] = max(rate, 1.0 - rate)
    return out


@torch.no_grad()
def calibration_report(
    logits: torch.Tensor, batch: dict, schema, bins: int = 10
) -> dict:
    """Per-field, per-cardinality calibration.

    ``gap`` is mean confidence minus accuracy: negative means the model is less
    confident than it should be, positive means overconfident. The drift this is
    built to catch shows up as gaps of opposite sign at low and high cardinality.

    Each field also carries ``majority`` and ``lift_over_majority`` so that a
    field the model never learned cannot hide behind a high accuracy.
    """
    report: dict = {"fields": {}}
    cardinality = batch["cardinality"]
    majority = majority_baseline(batch, schema)
    for index, field in enumerate(schema):
        confidence, prediction = field_confidence(
            logits[:, index], batch["option_mask"][:, index], field.single
        )
        if field.single:
            correct = prediction.eq(batch["labels"][:, index])
            flat_conf, flat_correct = confidence, correct
        else:
            live = batch["option_mask"][:, index]
            targets = batch["multi_labels"][:, index]
            correct = prediction.eq(targets.long())
            flat_conf = confidence[live]
            flat_correct = correct[live]
        accuracy = float(flat_correct.float().mean())
        entry = {
            "accuracy": accuracy,
            "majority": majority[field.name],
            "lift_over_majority": accuracy - majority[field.name],
            "confidence": float(flat_conf.mean()),
            "gap": float(flat_conf.mean() - flat_correct.float().mean()),
            "ece": expected_calibration_error(flat_conf, flat_correct, bins),
            # The stated criterion. Kept first-class next to the gap because the
            # two can disagree: a uniform offset hurts ECE while leaving the
            # ordering -- the thing a confidence threshold relies on -- intact.
            "monotonicity": monotonicity(flat_conf, flat_correct, bins),
            "by_cardinality": {},
        }
        field_cardinality = cardinality[:, index]
        for value in sorted(set(field_cardinality.tolist())):
            rows = field_cardinality == value
            if not rows.any():
                continue
            if field.single:
                sub_conf, sub_correct = confidence[rows], correct[rows]
            else:
                live = batch["option_mask"][:, index][rows]
                sub_conf = confidence[rows][live]
                sub_correct = correct[rows][live]
            if sub_conf.numel() == 0:
                continue
            entry["by_cardinality"][int(value)] = {
                "rows": int(rows.sum()),
                "accuracy": float(sub_correct.float().mean()),
                "confidence": float(sub_conf.mean()),
                "gap": float(sub_conf.mean() - sub_correct.float().mean()),
                "ece": expected_calibration_error(sub_conf, sub_correct, bins),
            }
        gaps = [row["gap"] for row in entry["by_cardinality"].values()]
        if gaps:
            entry["worst_gap"] = max(gaps, key=abs)
            entry["gap_spread"] = max(gaps) - min(gaps)
            # Monotone drift across cardinality is the jevlike failure mode; a
            # spread with mixed signs is worse than a uniform offset.
            entry["sign_flip"] = min(gaps) < 0 < max(gaps)
        report["fields"][field.name] = entry
    worst = [
        entry.get("worst_gap", 0.0) for entry in report["fields"].values()
    ]
    report["worst_gap_any_field"] = max(worst, key=abs) if worst else 0.0
    report["mean_ece"] = sum(
        entry["ece"] for entry in report["fields"].values()
    ) / len(report["fields"])
    return report


def reliability_curve(
    confidence: torch.Tensor, correct: torch.Tensor, bins: int = 10
) -> list[dict]:
    """Points for a reliability diagram: confidence against observed accuracy."""
    edges = torch.linspace(0, 1, bins + 1)
    curve = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        if not selected.any():
            continue
        curve.append({
            "bin": [float(low), float(high)],
            "rows": int(selected.sum()),
            "confidence": float(confidence[selected].mean()),
            "accuracy": float(correct[selected].float().mean()),
        })
    return curve
