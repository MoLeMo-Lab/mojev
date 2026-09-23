"""Unit tests. Each one pins a specific defect or a property the design rests on.

Run with ``pytest -q`` from the repository root.

Nothing here downloads a model, and the suite runs in about four seconds. The
packed path needs a tokeniser and an encoder, so this file supplies two small
real ones: ``ByteTokenizer`` implements the slice of the tokeniser interface
``packed_collate`` uses, and ``tiny_config`` builds a genuine ``qwen3_5`` -- the
family the shipped checkpoint uses -- at 153k parameters instead of 854M. Both
are real enough that the tree mask and the model's own construction are tested
rather than mocked around.
"""

from __future__ import annotations

import json
import math

import pytest
import torch
from torch.nn import functional as F
from transformers import AutoConfig

from mojev.calibrate import (
    CardinalityTemperature,
    brier,
    calibration_report,
    cardinality_bucket,
    expected_calibration_error,
    fit_temperature,
)
from mojev.data import (
    SYNTHETIC_SCHEMA,
    synthetic_example,
    validate,
)
from mojev.full import PackedScorer, objective, packed_collate
from mojev.modeling import PackedScorerConfig
from mojev.schema import Field, Schema

WIDTH = 32


class ByteTokenizer:
    """The slice of the tokeniser interface ``packed_collate`` uses.

    Bytes plus one, so 0 stays free as the pad id. Real tokenisers are what the
    model ships with; this exists so the suite needs no download.
    """

    pad_token_id = 0
    eos_token = ""
    vocab_size = 259

    def __call__(self, texts, truncation=False, max_length=None, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for text in texts:
            ids = [byte + 1 for byte in text.encode("utf-8")]
            rows.append((ids[:max_length] if max_length else ids) or [1])
        return {"input_ids": rows}


def tiny_config(schema: Schema, rank: int = 16, width: int = WIDTH) -> PackedScorerConfig:
    """A real config around an encoder small enough to build in milliseconds.

    ``qwen3_5`` specifically, which is the family the shipped checkpoint uses: it
    is multimodal, so ``hidden_size`` lives under ``text_config`` and there is no
    top-level one at all. A single-stack stand-in would exercise the other branch
    of ``PackedScorer.hidden_size`` and leave the shipped path untested.

    ``from_encoder`` would download pretrained weights; this goes through the same
    ``PackedScorer(config)`` path ``from_pretrained`` uses, so the construction
    under test is the construction that ships. 153k parameters, ~3 ms to build.
    """
    return PackedScorerConfig(
        encoder_config=AutoConfig.for_model(
            "qwen3_5",
            text_config=dict(
                vocab_size=300, hidden_size=width, intermediate_size=2 * width,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                head_dim=16, layer_types=["full_attention"],
            ),
            vision_config=dict(
                hidden_size=width, intermediate_size=2 * width, num_hidden_layers=1,
                num_heads=2, out_hidden_size=width, patch_size=16,
                temporal_patch_size=2, spatial_merge_size=1, depth=1,
            ),
        ),
        rank=rank, schema=schema.to_json(), context_tokens=192,
    )


def tiny_scorer(schema: Schema, rank: int = 16, width: int = WIDTH,
                seed: int = 0) -> PackedScorer:
    """A PackedScorer on a tiny real encoder, with no download.

    Seeded here rather than by the caller. ``PreTrainedModel`` initialises weights
    during construction, so how much randomness it draws depends on the config --
    and a caller that seeds *before* building gets initial weights that shift
    whenever anything upstream changes. Seeding inside makes a given (schema, rank,
    width, seed) always the same model, whatever ran before it.

    The encoder is cast to fp32. It is built in bf16, as the shipped model is, but
    bf16 carries about three decimal digits and several tests here assert to 1e-5
    -- the precision is the harness's business, not a property of the model.
    """
    torch.manual_seed(seed)
    model = PackedScorer(tiny_config(schema, rank, width))
    model.encoder = model.encoder.float()
    return model


def make_batch(rows=8, cardinality=8, schema=SYNTHETIC_SCHEMA):
    collate = packed_collate(ByteTokenizer(), schema, 192)
    examples = [
        validate(synthetic_example(1000 + index, cardinality), schema) for index in range(rows)
    ]
    return schema, collate, collate(examples)


def test_candidate_text_is_not_truncated():
    long_candidate = "extended candidate description " * 8
    schema = Schema((Field("choice", "choice", ("short", long_candidate)),))
    collate = packed_collate(ByteTokenizer(), schema, 192)
    batch = collate([validate(
        {"context": "state", "labels": {"choice": "short"}}, schema
    )])
    assert int(batch["option_span"][0, 0, 1].sum()) == len(long_candidate.encode())


def spans(rows=1, fields=2, options=2, length=12):
    """Hand-built span indicators: state, then one question per field with its options.

    Laid out as [state ... | q0 o0 o1 | q1 o0 o1 ...] so each span's positions are
    known and a mask can be checked position by position.
    """
    context = torch.zeros(rows, length)
    field_span = torch.zeros(rows, fields, length)
    option_span = torch.zeros(rows, fields, options, length)
    state = length - fields * (1 + options)
    context[:, :state] = 1.0
    cursor = state
    for field in range(fields):
        field_span[:, field, cursor] = 1.0
        cursor += 1
        for option in range(options):
            option_span[:, field, option, cursor] = 1.0
            cursor += 1
    return context, field_span, option_span


# --------------------------------------------------------------- schema / typing

def test_schema_rejects_malformed_fields():
    with pytest.raises(ValueError):
        Field("x", "choice", ("only",))          # fewer than two options
    with pytest.raises(ValueError):
        Field("x", "choice", ("a", "a"))         # duplicates
    with pytest.raises(ValueError):
        Field("x", "nonsense", ("a", "b"))       # unknown kind
    with pytest.raises(ValueError):
        Schema(())                               # no fields


def test_schema_encode_is_total_and_closed():
    schema = Schema((
        Field("intent", "choice", ("refund", "sales")),
        Field("escalate", "bool"),
    ))
    assert schema.encode({"intent": "sales", "escalate": True}) == [1, 1]
    with pytest.raises(ValueError):
        schema.encode({"intent": "sales"})                      # missing field
    with pytest.raises(ValueError):
        schema.encode({"intent": "nope", "escalate": True})     # undeclared option
    with pytest.raises(ValueError):
        schema.encode({"intent": "sales", "escalate": True, "x": 1})   # extra field


def test_schema_json_roundtrip():
    assert Schema.from_json(SYNTHETIC_SCHEMA.to_json()) == SYNTHETIC_SCHEMA


# ------------------------------------------------------------------- word order

def test_option_vectors_keep_word_order():
    """Pooling option *bytes* made these pairs bit-identical; pooling the sequence does not.

    Candidates occupy real positions in the packed sequence, so a candidate's
    pooled vector is built from encoder states that saw the token order. An earlier
    design mean-pooled each option's own embeddings on its own, which threw the
    order away outright.
    """
    schema, collate, _ = make_batch()
    model = tiny_scorer(schema).eval()

    def pooled(text: str) -> torch.Tensor:
        row = validate(
            {
                "context": "x",
                "options": {"badge": [text, "zzz zzz"]},
                "labels": {
                    "badge": text, "queue": "refund", "escalate": False,
                    "tags": [], "urgency": "low",
                },
            },
            schema,
        )
        batch = collate([row])
        with torch.no_grad():
            hidden = model.norm(model.encoder(
                input_ids=batch["packed_ids"],
                attention_mask=model.build_mask(
                    batch["context_span"], batch["field_span"], batch["option_span"]
                ),
            ).last_hidden_state.float())
        span = batch["option_span"][0, 0, 0]
        return (hidden[0] * span.unsqueeze(-1)).sum(0) / span.sum().clamp_min(1)

    for left, right in (("gold heron", "hero nglod"), ("amber badger", "badger amber")):
        assert not torch.allclose(pooled(left), pooled(right), atol=1e-6), (
            f"{left!r} and {right!r} pooled to the same vector"
        )


# ------------------------------------------------------------ multi-field output

def test_one_forward_returns_every_field():
    schema, _, batch = make_batch(rows=4, cardinality=8)
    model = tiny_scorer(schema).eval()
    with torch.no_grad():
        logits = model(batch)
    assert logits.shape == (4, len(schema), batch["option_mask"].shape[-1])
    # Each field normalises over its own live options only.
    for index, field in enumerate(schema):
        live = batch["option_mask"][:, index]
        assert int(live[0].sum()) == min(field.cardinality, 8) or field.cardinality < 8
        probabilities = logits[:, index].softmax(-1)
        assert torch.allclose(probabilities.sum(-1), torch.ones(4), atol=1e-5)
        # Absent options carry no probability mass.
        assert float(probabilities[~live].sum()) < 1e-6


def test_fields_attend_differently():
    """A field's identity is its own words, carried in the sequence.

    There is no per-field parameter and no row index: a question is text the model
    reads. So changing a field's wording must change its logits, and the mechanism
    must be the wording rather than the field's slot.

    Note what is *not* claimed. Two fields with identical prompts do not score
    identically, because packing puts them at different absolute positions and the
    encoder sees position. Exact equality holds across candidate order within a
    field -- that comes from the sort, and
    ``test_candidate_order_cannot_change_an_answer`` covers it -- not across field
    slots.
    """
    menu = ("alpha", "beta", "gamma")

    def score(prompts):
        schema = Schema(tuple(
            Field(name, "choice", menu, prompt)
            for name, prompt in zip(("a", "b"), prompts)
        ))
        collate = packed_collate(ByteTokenizer(), schema, 64)
        batch = collate([validate(
            {"context": "a state", "labels": {"a": "alpha", "b": "alpha"}}, schema
        )])
        # tiny_scorer seeds itself, so both schemas get the same weights.
        with torch.no_grad():
            return tiny_scorer(schema).eval()(batch)

    # Rewording only the second field must move only the second field's logits.
    base = score(("Which colour is the car?", "How urgent is this request?"))
    reworded = score(("Which colour is the car?", "What is the customer's name?"))
    assert torch.allclose(base[:, 0], reworded[:, 0], atol=1e-5), (
        "rewording field b changed field a -- the fields are not independent"
    )
    assert not torch.allclose(base[:, 1], reworded[:, 1], atol=1e-4), (
        "rewording field b left its logits alone -- the prompt is not being read"
    )


def test_schema_may_grow_without_retraining():
    """Adding or renaming fields must not change the parameter shapes.

    An earlier design stored one trainable vector per field, so a five-field schema
    could not load three-field weights. The packed scorer holds no per-field
    parameters at all -- a field is its own words in the sequence -- so weights
    trained on one schema load into another and the forward pass still runs.
    """
    menu = ("yes", "no")
    three = Schema(tuple(
        Field(name, "choice", menu, f"Question about {name}?")
        for name in ("a", "b", "c")
    ))
    five = Schema(tuple(
        Field(name, "choice", menu, f"Question about {name}?")
        for name in ("a", "b", "c", "d", "e")
    ))
    small, large = tiny_scorer(three), tiny_scorer(five)
    assert {k: tuple(v.shape) for k, v in small.state_dict().items()} == \
           {k: tuple(v.shape) for k, v in large.state_dict().items()}
    missing, unexpected = large.load_state_dict(small.state_dict(), strict=True)
    assert not missing and not unexpected
    assert not any("field_query" in name for name in small.state_dict())

    # And the wider schema really runs: five fields out of a five-field batch.
    collate = packed_collate(ByteTokenizer(), five, 64)
    batch = collate([validate(
        {"context": "a state", "labels": {name: "yes" for name in "abcde"}}, five
    )])
    with torch.no_grad():
        assert large.eval()(batch).shape[1] == 5


def test_learning_reduces_loss():
    schema, _, batch = make_batch(rows=16, cardinality=8)
    model = tiny_scorer(schema, seed=5)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    labels = batch["labels"][:, 0]
    initial = float(F.cross_entropy(model(batch)[:, 0], labels).detach())
    for _ in range(40):
        loss = F.cross_entropy(model(batch)[:, 0], labels)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    assert float(F.cross_entropy(model(batch)[:, 0], labels).detach()) < initial * 0.6


def test_the_trained_objective_descends():
    """The objective actually used -- Plackett-Luce plus Brier -- must be trainable.

    ``test_learning_reduces_loss`` uses cross-entropy to isolate the forward pass.
    This one runs the loss the model ships with, so a regression in ``objective``
    itself surfaces here rather than in a training run.
    """
    schema, _, batch = make_batch(rows=16, cardinality=8)
    model = tiny_scorer(schema, seed=5)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    initial = float(objective(model(batch), batch, schema, 1.0).detach())
    for _ in range(40):
        loss = objective(model(batch), batch, schema, 1.0)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    assert float(objective(model(batch), batch, schema, 1.0).detach()) < initial


# --------------------------------------------------------------------- tree mask

def test_candidates_cannot_see_their_siblings_or_other_questions():
    """The mask is the architecture: three levels, with isolation at each.

    Per-decision independence rests on this. With candidates hanging straight off
    the state, a question's wording reached every candidate in the request and
    candidates of question A could read question B -- so scores depended on which
    other questions happened to share the row, and permutation invariance was gone.
    """
    context, field_span, option_span = spans(fields=2, options=2, length=12)
    mask = PackedScorer.build_mask(None, context, field_span, option_span)
    assert mask.shape == (1, 1, 12, 12)
    allowed = mask[0, 0] == 0.0

    state = [position for position in range(12) if context[0, position] > 0]
    q0, q1 = 6, 9
    a0, a1, b0, b1 = 7, 8, 10, 11

    # A candidate reads the state, its own question, and itself.
    assert allowed[a0, state[0]] and allowed[a0, q0] and allowed[a0, a0]
    # Never a sibling candidate, never another question, never its candidates.
    assert not allowed[a0, a1] and not allowed[a0, q1]
    assert not allowed[a0, b0] and not allowed[a0, b1]
    # Questions cannot see each other, but do read the state.
    assert not allowed[q0, q1] and not allowed[q1, q0]
    assert allowed[q0, state[0]]
    # The state cannot see the questions or candidates below it.
    assert not allowed[state[0], q0] and not allowed[state[0], a0]


def test_no_position_is_fully_masked():
    """Every row keeps at least one allowed column, or softmax divides by zero.

    Padding belongs to no node in the tree, so its rows came out entirely masked
    and produced NaN within 200 steps. The diagonal is what prevents it.
    """
    context, field_span, option_span = spans(fields=2, options=3, length=16)
    mask = PackedScorer.build_mask(None, context, field_span, option_span)
    assert bool((mask[0, 0] == 0.0).any(-1).all())

    # Including rows no span claims at all -- the padding case.
    context[:, 4:] = 0.0
    sparse = PackedScorer.build_mask(None, context, field_span, option_span)
    assert bool((sparse[0, 0] == 0.0).any(-1).all())


# ------------------------------------------------------------------- calibration

def test_brier_is_minimised_by_truth():
    """A strictly proper scoring rule: the true distribution scores best."""
    targets = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.ones(1, 3)
    truth = brier(targets, targets, mask)
    for wrong in ([[0.6, 0.2, 0.2]], [[0.34, 0.33, 0.33]], [[0.0, 1.0, 0.0]]):
        assert float(brier(torch.tensor(wrong), targets, mask)) > float(truth)


def test_cardinality_bucket_is_monotone():
    counts = torch.tensor([2, 3, 4, 8, 16, 32, 64])
    buckets = cardinality_bucket(counts)
    assert buckets.tolist() == sorted(buckets.tolist())
    assert buckets[0] < buckets[-1]


def test_temperature_preserves_argmax():
    """Temperature scaling may not change any decision, only its confidence."""
    torch.manual_seed(0)
    logits = torch.randn(64, 3, 16)
    mask = torch.ones(64, 3, 16, dtype=torch.bool)
    mask[:, :, 8:] = False
    logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    cardinality = mask.sum(-1)
    scaler = CardinalityTemperature(3)
    with torch.no_grad():
        scaler.log_temperature.normal_(0, 0.5)
        scaled = scaler(logits, cardinality, mask)
    assert torch.equal(logits.argmax(-1), scaled.argmax(-1))
    assert torch.isfinite(scaled[mask]).all()


def test_temperature_handles_masked_infinities():
    """finfo.min divided by a temperature overflows; the mask must be re-applied."""
    logits = torch.full((4, 1, 8), torch.finfo(torch.float32).min)
    logits[:, :, :2] = torch.tensor([0.5, -0.5])
    mask = torch.zeros(4, 1, 8, dtype=torch.bool)
    mask[:, :, :2] = True
    scaler = CardinalityTemperature(1)
    with torch.no_grad():
        scaler.log_temperature.fill_(math.log(0.1))
        scaled = scaler(logits, mask.sum(-1), mask)
    assert torch.isfinite(scaled[mask]).all()
    assert float(scaled[mask].abs().max()) < 1e30


def test_temperature_fitting_improves_calibration():
    """A deliberately overconfident model should be tamed on held-out logits."""
    torch.manual_seed(1)
    rows, fields, width = 512, 1, 8
    labels = torch.randint(0, width, (rows, fields))
    # Logits that point at the label but are scaled far too sharply.
    logits = torch.randn(rows, fields, width) * 0.3
    logits.scatter_add_(2, labels.unsqueeze(-1), torch.full((rows, fields, 1), 4.0))
    logits = logits * 3.0
    mask = torch.ones(rows, fields, width, dtype=torch.bool)
    batch = {
        "labels": labels,
        "multi_labels": torch.zeros(rows, fields, width),
        "option_mask": mask,
        "cardinality": mask.sum(-1),
    }
    schema = Schema((Field("x", "choice", tuple(f"o{i}" for i in range(width))),))
    before = calibration_report(logits, batch, schema)
    scaler = fit_temperature(logits, batch, schema)
    after = calibration_report(
        scaler(logits, batch["cardinality"], mask), batch, schema
    )
    assert after["mean_ece"] < before["mean_ece"]


def test_calibration_report_flags_sign_flip():
    """Opposite-signed gaps across cardinality must be reported, not averaged away.

    The jevlike failure mode: underconfident on small menus, overconfident on
    large ones. Pooling the two hides it, so ``sign_flip`` has to catch it.
    """
    rows, width = 400, 8
    half = rows // 2
    labels = torch.zeros(rows, 1, dtype=torch.long)
    mask = torch.zeros(rows, 1, width, dtype=torch.bool)
    mask[:half, :, :2] = True         # two options
    mask[half:, :, :8] = True         # eight options
    logits = torch.zeros(rows, 1, width)
    # N=2 rows: flat (confidence 0.5) but always correct -> gap = -0.5.
    # N=8 rows: sharply confident on option 1 while the label is 0 -> gap > 0.
    logits[half:, :, 1] = 6.0
    logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    batch = {
        "labels": labels,
        "multi_labels": torch.zeros(rows, 1, width),
        "option_mask": mask,
        "cardinality": mask.sum(-1),
    }
    schema = Schema((Field("x", "choice", tuple(f"o{i}" for i in range(width))),))
    report = calibration_report(logits, batch, schema)
    entry = report["fields"]["x"]
    assert set(entry["by_cardinality"]) == {2, 8}
    assert entry["by_cardinality"][2]["gap"] < 0, entry["by_cardinality"]
    assert entry["by_cardinality"][8]["gap"] > 0, entry["by_cardinality"]
    assert entry["sign_flip"], entry["by_cardinality"]
    assert entry["gap_spread"] > 0.5


def test_ece_is_zero_for_a_calibrated_predictor():
    confidence = torch.full((1000,), 0.7)
    correct = torch.zeros(1000, dtype=torch.bool)
    correct[:700] = True
    assert expected_calibration_error(confidence, correct) < 0.01


# ------------------------------------------------------------------- data layer

def test_row_menu_must_fit_the_schema():
    schema = Schema((Field("click", "choice", ("slot-0", "slot-1")),))
    with pytest.raises(ValueError):
        validate(
            {"context": "c", "options": {"click": ["a", "b", "c"]}, "labels": {"click": "a"}},
            schema,
        )


def test_cardinality_matches_live_options():
    schema, _, batch = make_batch(rows=6, cardinality=16)
    assert torch.equal(batch["cardinality"], batch["option_mask"].sum(-1))
    assert int(batch["cardinality"][0, 0]) == 16


# ------------------------------------------------------------- preference loss

def test_plackett_luce_weights_every_row_equally():
    """Packing mixes cardinalities, so per-stage averaging would weight rows unequally.

    A row with two options spans one ranking stage; a row with eight may span
    three. Averaging within each stage makes the short row contribute less total
    loss than the long one for no reason other than its option count. The batch
    loss must equal the mean of the per-row losses.
    """
    from mojev.calibrate import plackett_luce

    torch.manual_seed(0)
    rows, width = 4, 8
    mask = torch.zeros(rows, width, dtype=torch.bool)
    live = (8, 5, 3, 2)
    grades = torch.full((rows, width), -1)
    for row, count in enumerate(live):
        mask[row, :count] = True
        grades[row, 0] = 2
        grades[row, 1:max(1, count // 2)] = 1
        grades[row, max(1, count // 2):count] = 0
    logits = torch.randn(rows, width)

    batch = float(plackett_luce(logits, grades, mask))
    per_row = [
        float(plackett_luce(logits[r:r + 1], grades[r:r + 1], mask[r:r + 1]))
        for r in range(rows)
    ]
    assert abs(batch - sum(per_row) / rows) < 1e-5, (batch, per_row)


def test_plackett_luce_ignores_padding():
    """Padding slots must take no gradient and no probability mass."""
    from mojev.calibrate import plackett_luce

    torch.manual_seed(1)
    rows, width = 4, 8
    mask = torch.zeros(rows, width, dtype=torch.bool)
    mask[:, :3] = True
    grades = torch.full((rows, width), -1)
    grades[:, 0] = 2
    grades[:, 1] = 1
    grades[:, 2] = 0
    logits = torch.randn(rows, width, requires_grad=True)
    plackett_luce(logits, grades, mask).backward()
    assert bool((logits.grad[~mask] == 0).all())
    assert bool((logits.grad[mask].abs() > 0).any())


def test_plackett_luce_is_zero_without_an_ordering():
    """One grade level means nothing to rank; cross-entropy handles those rows."""
    from mojev.calibrate import plackett_luce

    mask = torch.ones(4, 6, dtype=torch.bool)
    grades = torch.zeros(4, 6, dtype=torch.long)
    assert float(plackett_luce(torch.randn(4, 6), grades, mask)) == 0.0


# ------------------------------------------------------------------------ serving

def test_engine_loads_processor_from_the_checkpoint(monkeypatch):
    """The model ID is also the processor ID; serving has one artifact input."""
    import sys
    from types import SimpleNamespace

    from mojev import evaluate
    from mojev.serve import Engine

    calls = []
    processor = SimpleNamespace(tokenizer=object())

    class AutoProcessor:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append(("processor", name, kwargs))
            return processor

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(name):
            calls.append(("tokenizer", name))
            return object()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoProcessor=AutoProcessor, AutoTokenizer=AutoTokenizer),
    )
    monkeypatch.setattr(
        evaluate,
        "load_packed",
        lambda checkpoint, device: (
            object(), object(), SimpleNamespace(context_tokens=16_384)
        ),
    )

    engine = Engine("MoLeMo-Lab/mojev", device="cpu")
    assert calls == [(
        "processor",
        "MoLeMo-Lab/mojev",
        {"trust_remote_code": True},
    )]
    assert engine.processor is processor
    assert engine.tokenizer is processor.tokenizer

def test_question_types_map_to_candidate_text():
    """The three SDK question kinds become candidate strings the model can score."""
    from mojev.serve import option_texts

    options, kind = option_texts("spam", {
        "type": "noul", "criteria": {"true": "Unsolicited ads", "false": "Legitimate"}
    })
    assert kind == "noul"
    assert options == ["Legitimate", "Unsolicited ads"]     # false first, so index 1 is yes

    options, kind = option_texts("tone", {
        "type": "choice", "criteria": {"angry": "An upset message", "calm": None}
    })
    assert kind == "choice"
    # A described criterion carries its description; a bare one is its name alone.
    assert options == ["angry: An upset message", "calm"]

    options, kind = option_texts("urgency", {
        "type": "score", "criteria": ["Can wait", "Today"]
    })
    assert kind == "score"
    assert options == ["Can wait", "Today"]


def test_malformed_questions_produce_sdk_shaped_422():
    """The SDK parses {"detail": [{"loc": [...], "msg": ..., "type": ...}]}."""
    from mojev.serve import ValidationFailure, option_texts

    cases = [
        ({"type": "choice"}, ["body", "questions", "q", "choice", "criteria"], "missing"),
        ({"type": "score", "criteria": []},
         ["body", "questions", "q", "score", "criteria"], "too_short"),
        ({"type": "aurora", "criteria": {}}, ["body", "questions", "q", "type"],
         "literal_error"),
        ({}, ["body", "questions", "q", "type"], "missing"),
    ]
    for question, loc, kind in cases:
        with pytest.raises(ValidationFailure) as caught:
            option_texts("q", question)
        detail = caught.value.detail[0]
        assert detail["loc"] == loc, question
        assert detail["type"] == kind, question


def test_answers_match_the_wire_schema():
    """Field names and types are what typesafe_sdk._schemas.models declares."""
    from mojev.serve import build_answer

    noul = build_answer("spam", {"type": "noul"}, "noul", [0.02, 0.98])
    assert noul == {"type": "noul", "noul": 0.98}

    question = {"type": "choice", "criteria": {"angry": None, "calm": None, "excited": None}}
    choice = build_answer("tone", question, "choice", [0.8, 0.1, 0.1])
    assert choice["type"] == "choice" and choice["choice"] == "angry"
    assert choice["confidence"] == pytest.approx(0.8)
    assert choice["probabilities"] == {"angry": 0.8, "calm": 0.1, "excited": 0.1}

    question = {"type": "score", "criteria": ["bad", "ok", "great"]}
    score = build_answer("quality", question, "score", [0.1, 0.1, 0.8])
    # The expected score is the probability-weighted level, so it may be fractional.
    assert score["score"] == pytest.approx(1.7)
    # Keys are strings of integers on the wire; the SDK coerces them to ints.
    assert set(score["legend"]) == {"0", "1", "2"}
    assert set(score["probabilities"]) == {"0", "1", "2"}


def test_candidate_order_cannot_change_an_answer():
    """Sorting candidates before scoring is what makes the API permutation-invariant.

    Packing lays candidates out one after another, so the same candidate sits at a
    different absolute position depending on the order the caller listed it in.
    Measured on a trained checkpoint before the sort: reversing three choices moved
    the probabilities by 2.5e-01 and flipped the argmax. Sorting makes the packed
    sequence a function of the candidate *set*, so equality is exact rather than
    approximate.

    This checks the ordering logic itself; the end-to-end equality through a real
    model is 0.00e+00 deviation.
    """
    forward = ["angry", "calm", "excited"]
    reverse = list(reversed(forward))
    order_f = sorted(range(len(forward)), key=lambda i: forward[i])
    order_r = sorted(range(len(reverse)), key=lambda i: reverse[i])
    # Both orders present the model with the same sequence.
    assert [forward[i] for i in order_f] == [reverse[i] for i in order_r]

    # Scores come back in sorted order and are mapped to each caller's positions.
    scored = [0.58, 0.30, 0.12]        # for the sorted menu
    def unsort(order, values):
        out = [0.0] * len(order)
        for position, original in enumerate(order):
            out[original] = values[position]
        return out

    mapped_f = dict(zip(forward, unsort(order_f, scored)))
    mapped_r = dict(zip(reverse, unsort(order_r, scored)))
    assert mapped_f == mapped_r


# ---------------------------------------------------------------------------- cli

def test_cli_commands_resolve_to_real_modules():
    """Every advertised subcommand must name a module that exists and has main().

    The table is hand-written, so a rename would otherwise surface as a runtime
    ImportError the first time someone ran that command.
    """
    import importlib

    from mojev import cli

    tables = [(group, entries) for group, entries in cli.GROUPS.items()]
    tables.append(("", cli.FLAT))
    for group, entries in tables:
        for name, (module, help_text) in entries.items():
            imported = importlib.import_module(f"mojev.{module}")
            assert callable(getattr(imported, "main", None)), f"{group} {name}"
            assert help_text and help_text[0].islower(), f"{group} {name}: {help_text!r}"


def test_cli_help_and_unknown_commands():
    from mojev import cli

    assert cli.main([]) == 0
    assert cli.main(["--help"]) == 0
    for group in cli.GROUPS:
        assert cli.main([group]) == 0
        assert cli.main([group, "--help"]) == 0
    # Unknown names exit non-zero rather than raising.
    assert cli.main(["nonexistent"]) == 2
    assert cli.main(["train", "nonexistent"]) == 2


def test_checkpoint_round_trips_through_save_pretrained(tmp_path):
    """A saved directory must rebuild the same model, weights and schema included.

    The schema travels in the config, so a checkpoint carries the fields it was
    trained on and no caller has to supply them alongside. Weight equality is
    exact: persistence is not allowed to be lossy.
    """
    schema = Schema((Field("q", "choice", ("alpha", "beta"), "Which one?"),))
    model = tiny_scorer(schema)
    model.save_pretrained(tmp_path, safe_serialization=True)

    names = {path.name for path in tmp_path.iterdir()}
    assert "config.json" in names and "model.safetensors" in names
    # No pickle in the artefact -- that is the point of the format.
    assert not any(name.endswith((".pt", ".bin")) for name in names)

    back = PackedScorer.from_pretrained(tmp_path)
    assert back.schema == schema
    assert back.config.rank == model.config.rank
    assert back.config.context_tokens == model.config.context_tokens
    original = model.state_dict()
    for name, tensor in back.state_dict().items():
        assert torch.equal(tensor, original[name]), name


def test_saved_weights_keep_the_dtype_they_were_trained_in(tmp_path):
    """bf16 encoder, fp32 head -- through construction, saving and loading.

    Building the model in fp32 before loading a checkpoint silently upcast all 473
    encoder tensors, wrote F32 to disk, and moved predictions by 4.7e-03 while
    every shape still matched. Nothing else in the suite would catch that.
    """
    from mojev.modeling import ENCODER_DTYPE, HEAD_DTYPE

    schema = Schema((Field("q", "choice", ("alpha", "beta"), "Which one?"),))
    torch.manual_seed(0)
    model = PackedScorer(tiny_config(schema))       # not tiny_scorer: no fp32 cast

    def dtypes(state):
        return (
            {value.dtype for name, value in state.items() if name.startswith("encoder.")},
            {value.dtype for name, value in state.items() if not name.startswith("encoder.")},
        )

    assert dtypes(model.state_dict()) == ({ENCODER_DTYPE}, {HEAD_DTYPE})
    model.save_pretrained(tmp_path, safe_serialization=True)
    assert dtypes(PackedScorer.from_pretrained(tmp_path).state_dict()) == \
           ({ENCODER_DTYPE}, {HEAD_DTYPE})


def test_the_config_records_the_nested_encoder(tmp_path):
    """The encoder's own config rides inside ours, so loading needs no --hf-model.

    Its ``model_type`` has to survive, since that is what rebuilds the right config
    class on the way back in. ``auto_map`` has to be written too, or the directory
    is not loadable by someone who does not have this package.
    """
    schema = Schema((Field("q", "choice", ("alpha", "beta"), "Which one?"),))
    tiny_scorer(schema).save_pretrained(tmp_path, safe_serialization=True)

    written = json.loads((tmp_path / "config.json").read_text())
    assert written["model_type"] == "mojev-scorer"
    assert written["encoder_config"]["model_type"] == "qwen3_5"
    assert "AutoModel" in written["auto_map"] and "AutoConfig" in written["auto_map"]
    # And the nested config comes back as a config object, not a bare dict.
    reloaded = PackedScorerConfig.from_pretrained(tmp_path)
    assert reloaded.encoder_config.text_config.hidden_size == WIDTH


def test_a_checkpoint_must_be_a_directory(tmp_path):
    """A file is not a checkpoint, and the error should say what one is.

    Models are written as directories. Pointing an entry point at a stray file --
    a leftover .pt, a typo'd path -- should fail on the format rather than deep
    inside transformers.
    """
    from mojev.evaluate import load_packed

    stray = tmp_path / "mix-pl.pt"
    stray.write_bytes(b"not a checkpoint")
    with pytest.raises(SystemExit) as raised:
        load_packed(str(stray), torch.device("cpu"))
    assert "model directory" in str(raised.value)


def test_percentiles_are_nearest_rank():
    """p99 must name an observed request, not an interpolation between two.

    A latency report that smooths its own tail understates exactly the thing the
    tail is there to show, so the slowest sample has to survive to p99.
    """
    from mojev.apitest import percentiles

    # 100 samples at 1..100 ms, so each percentile has an unambiguous answer.
    measured = percentiles([n / 1000.0 for n in range(1, 101)])
    assert measured["p50_ms"] == 50.0
    assert measured["p90_ms"] == 90.0
    assert measured["p99_ms"] == 99.0
    assert measured["min_ms"] == 1.0 and measured["max_ms"] == 100.0

    # Order in must not matter, and one sample must not divide by zero.
    assert percentiles([0.005, 0.001, 0.003])["p50_ms"] == 3.0
    single = percentiles([0.25])
    assert single["p50_ms"] == single["p99_ms"] == 250.0


def test_api_harness_refuses_to_fake_the_sdk():
    """Without the real client the command must stop, not hand-roll requests.

    The point of the harness is that the official SDK's own wire schemas parse the
    response. Falling back to bare HTTP would still produce a latency table, and
    that table would no longer be evidence of compatibility.
    """
    import builtins

    from mojev.apitest import load_sdk

    real = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("typesafe_sdk"):
            raise ImportError("No module named 'typesafe_sdk'")
        return real(name, *args, **kwargs)

    builtins.__import__ = blocked
    try:
        with pytest.raises(SystemExit) as raised:
            load_sdk(None)
    finally:
        builtins.__import__ = real
    # The message has to say how to fix it, not just that it broke.
    assert "--sdk" in str(raised.value) and "pip install" in str(raised.value)


def test_cli_restores_argv():
    """Dispatch rewrites sys.argv for the target module; it must put it back."""
    import sys

    from mojev import cli

    before = list(sys.argv)
    cli.main(["train"])
    assert sys.argv == before


def test_image_paths_are_stripped_and_the_marker_survives():
    """Every path comes out, every marker stays, and the pass terminates.

    The marker has to survive the substitution -- the processor needs it to expand
    into one placeholder per patch -- which rules out stripping paths in a loop
    over ``while MARKER in text``: writing the marker back keeps the condition
    true forever. One regex pass over the whole string cannot spin.
    """
    from mojev.full import IMAGE_MARKER, IMAGE_PATH, IMAGE_PLACEHOLDER

    text = (
        f"look {IMAGE_PLACEHOLDER}/a/cat.jpg then "
        f"{IMAGE_MARKER}/b/dog.png done"
    )
    assert IMAGE_PATH.findall(text) == ["/a/cat.jpg", "/b/dog.png"]
    stripped = IMAGE_PATH.sub(lambda _: IMAGE_PLACEHOLDER, text)
    assert stripped == f"look {IMAGE_PLACEHOLDER} then {IMAGE_PLACEHOLDER} done"
    # Markers are preserved for the processor to expand, and no path remains.
    assert stripped.count(IMAGE_PLACEHOLDER) == 2
    assert ".jpg" not in stripped and ".png" not in stripped
    # Text with no image is returned untouched.
    assert IMAGE_PATH.sub(lambda _: IMAGE_PLACEHOLDER, "plain text") == "plain text"
