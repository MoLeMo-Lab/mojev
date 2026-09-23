"""The scorer as a HuggingFace model: config, weights, and registration.

A checkpoint used to be ``torch.save({"schema", "state_dict", "args"})``, which
cost three things. Loading one ran arbitrary pickle, because the payload carried
the training ``args`` beside the tensors and every reader had to pass
``weights_only=False``. Every load read the pretrained encoder and then threw it
away -- 4.32 s to fetch 473 tensors that ``load_state_dict`` immediately
overwrote, against 0.30 s for the same weights through ``from_pretrained``. And
the layout was private, so four call sites each re-derived it and the encoder's
identity lived in a free-text ``args`` field rather than in a config, which is
why every entry point wanted ``--hf-model`` passed alongside the checkpoint.

So the model is a ``PreTrainedModel``. ``save_pretrained`` writes
``config.json`` plus safetensors, ``from_pretrained`` reads them once, the
encoder config nests inside the model config, and the schema travels with the
weights. Registration below makes the directory loadable as
``AutoModel.from_pretrained(path, trust_remote_code=True)`` by anyone with
transformers and no copy of this package.

**Dtypes are mixed and that is load-bearing.** The encoder is bf16; the head --
``option_proj``, ``context_proj``, ``norm`` -- is fp32, which is how training
produced it and what ``layer_norm`` needs. Both failure modes here were measured:
building the model in fp32 before loading upcast the whole encoder, ``F32`` went
to disk for all 477 tensors, and predictions drifted 4.7e-03; passing
``dtype=torch.bfloat16`` to ``from_pretrained`` casts the head too and
``layer_norm`` then raises. The per-module dtype has to come from ``__init__``,
which is what ``_dtype_for`` below is for.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel

from .schema import Schema

ENCODER_DTYPE = torch.bfloat16
HEAD_DTYPE = torch.float32
DEFAULT_TRAINING_CONTEXT_TOKENS = 16_384


def hub_id(name: str) -> str:
    """Recover ``org/model`` from a local snapshot path, or pass a repo id through.

    ``AutoModel.from_pretrained`` is usually handed a path under
    ``~/.cache/huggingface/hub/models--Org--Name/snapshots/<sha>``. Recording that
    verbatim in a published config exposes the training box's filesystem and is
    useless to anyone else, so the ``models--Org--Name`` segment is turned back
    into ``Org/Name``. Anything that is already a repo id is returned unchanged.
    """
    for part in reversed(str(name).split("/")):
        if part.startswith("models--"):
            return part.removeprefix("models--").replace("--", "/")
    return str(name)


class PackedScorerConfig(PretrainedConfig):
    """Everything needed to rebuild the scorer, including the encoder's own config.

    ``schema`` is the JSON ``Schema.to_json`` already produces, so the schema a
    checkpoint was trained with travels with its weights. ``context_tokens``
    records the 16K state budget used in training. Inference callers can select
    a larger state window without changing the model weights. Candidate text
    has no separate token limit.
    """

    model_type = "mojev-scorer"
    sub_configs = {"encoder_config": AutoConfig}

    def __init__(self, encoder_config=None, rank: int = 512, schema=None,
                 context_tokens: int = DEFAULT_TRAINING_CONTEXT_TOKENS,
                 encoder_name: str | None = None, **kwargs) -> None:
        # from_dict hands back a plain dict; for_model rebuilds the right class
        # using the model_type that to_dict preserved.
        if isinstance(encoder_config, dict):
            encoder_config = AutoConfig.for_model(**encoder_config)
        self.encoder_config = encoder_config
        self.rank = rank
        self.schema = schema
        if context_tokens < 1:
            raise ValueError("context_tokens must be positive")
        self.context_tokens = context_tokens
        kwargs.pop("option_tokens", None)
        # Provenance only: loading never consults it, since the weights are in the
        # checkpoint. Stored as a repo id rather than whatever path the training
        # box happened to use -- a config that ships with someone's local
        # /root/.cache/... in it leaks their directory layout and tells a reader
        # nothing.
        self.encoder_name = hub_id(encoder_name) if encoder_name else None
        super().__init__(**kwargs)


class PackedScorer(PreTrainedModel):
    """Encode context and every option in one sequence, then score each option.

    Layout per row:

        [context tokens] [opt 0 tokens] [opt 1 tokens] ... [opt N-1 tokens]

    The attention mask is block-diagonal over the option spans: an option sees
    the context and itself, never another option. Without that, scores would
    depend on which rivals happened to be present and permutation invariance
    would be gone -- the property that separates this from putting the choices
    in a prompt.
    """

    config_class = PackedScorerConfig
    base_model_prefix = "encoder"
    _supports_sdpa = True

    def __init__(self, config: PackedScorerConfig) -> None:
        super().__init__(config)
        self.encoder = AutoModel.from_config(config.encoder_config, dtype=ENCODER_DTYPE)
        width = self.hidden_size(config.encoder_config)
        self.width = width
        self.rank = config.rank
        self.option_proj = nn.Linear(width, config.rank, bias=False, dtype=HEAD_DTYPE)
        self.context_proj = nn.Linear(width, config.rank, bias=False, dtype=HEAD_DTYPE)
        self.norm = nn.LayerNorm(width, dtype=HEAD_DTYPE)
        # Sets all_tied_weights_keys and the rest of the composite-model state;
        # without it from_pretrained raises AttributeError on the first of them.
        self.post_init()

    @staticmethod
    def hidden_size(encoder_config) -> int:
        """Multimodal configs nest the text stack; single-stack ones do not."""
        return getattr(encoder_config, "text_config", encoder_config).hidden_size

    @property
    def schema(self) -> Schema:
        return Schema.from_json(self.config.schema)

    @classmethod
    def from_encoder(cls, model_name: str, schema: Schema, rank: int = 512,
                     context_tokens: int = DEFAULT_TRAINING_CONTEXT_TOKENS):
        """Build a fresh scorer on a pretrained encoder -- the training entry point.

        This is the one path that should read pretrained weights, because starting
        from them is the point. Loading a trained checkpoint goes through
        ``from_pretrained`` instead and reads the encoder exactly once.
        """
        config = PackedScorerConfig(
            encoder_config=AutoConfig.from_pretrained(model_name),
            rank=rank, schema=schema.to_json(), context_tokens=context_tokens,
            encoder_name=str(model_name),
        )
        model = cls(config)
        model.encoder = AutoModel.from_pretrained(model_name, dtype=ENCODER_DTYPE)
        return model

    def build_mask(self, context_span: torch.Tensor, field_span: torch.Tensor,
                   option_span: torch.Tensor) -> torch.Tensor:
        """(B, 1, L, L) additive mask enforcing a state -> question -> candidate tree.

        Three levels, not two:

            state      attends within itself
            question   attends to the state and to itself
            candidate  attends to the state, to *its own* question, and to itself

        What the two-level version got wrong is the middle row. With candidates
        hanging straight off the context, a question's wording reached every
        candidate in the request, so the eight questions of a customer-service row
        could see each other's text. Worse, question spans sat unmasked in the
        sequence, which let candidates of question A read question B -- scores then
        depend on which other questions happen to be in the same request, and the
        parallel-sampling guarantee (independent decisions, exact permutation
        invariance) no longer holds.

        Sibling isolation is the point: two candidates of the same question cannot
        see each other, and two questions of the same state cannot see each other.
        """
        batch, fields, width, total = option_span.shape
        device = option_span.device
        state = context_span > 0                                 # (B, L)
        question = field_span > 0                                # (B, F, L)
        candidate = option_span > 0                              # (B, F, N, L)

        allow = torch.zeros(batch, total, total, dtype=torch.bool, device=device)
        # state -> state
        allow |= state[:, :, None] & state[:, None, :]
        # question -> state, question -> itself
        q_any = question.any(1)
        allow |= q_any[:, :, None] & state[:, None, :]
        allow |= torch.einsum("bfi,bfj->bij", question.float(), question.float()).bool()
        # candidate -> state
        c_any = candidate.any(1).any(1)
        allow |= c_any[:, :, None] & state[:, None, :]
        # candidate -> its own question (broadcast over that question's candidates)
        own_question = torch.einsum(
            "bfni,bfj->bij", candidate.float(), question.float()
        ).bool()
        allow |= own_question
        # candidate -> itself only, never a sibling
        allow |= torch.einsum("bfni,bfnj->bij", candidate.float(), candidate.float()).bool()
        # Padding positions belong to no node in the tree, so every one of their
        # rows would be entirely masked and softmax would divide by zero -- the
        # NaN this produced showed up within 200 steps. Let each position attend
        # to itself; the result is discarded because nothing pools from padding.
        eye = torch.eye(total, dtype=torch.bool, device=device)
        allow |= eye[None, :, :]
        return torch.where(allow, 0.0, torch.finfo(torch.float32).min).unsqueeze(1)

    def forward(self, batch: dict) -> torch.Tensor:
        # The tree mask, not a plain padding mask: without it every span in the
        # sequence is mutually visible and the per-decision independence this
        # design rests on is lost.
        mask = self.build_mask(
            batch["context_span"], batch["field_span"], batch["option_span"]
        )
        # Padding columns are unreachable, but the diagonal must survive or the
        # padding rows go fully masked again and softmax divides by zero.
        floor = torch.finfo(mask.dtype).min
        total = mask.shape[-1]
        keep = batch["packed_mask"][:, None, None, :] | torch.eye(
            total, dtype=torch.bool, device=mask.device
        )[None, None]
        mask = mask.masked_fill(~keep, floor)
        extra = {}
        if "pixel_values" in batch:
            extra["pixel_values"] = batch["pixel_values"]
            extra["image_grid_thw"] = batch["image_grid_thw"]
            # The processor sizes mm_token_type_ids to the state alone; packing
            # appends questions and candidates, so it is padded with zeros (text)
            # out to the full sequence.
            token_types = batch["mm_token_type_ids"]
            if token_types.shape[1] < total:
                token_types = torch.cat([
                    token_types,
                    token_types.new_zeros(token_types.shape[0], total - token_types.shape[1]),
                ], dim=1)
            extra["mm_token_type_ids"] = token_types[:, :total]
            # M-RoPE derives 3D positions by indexing the attention mask as a 2D
            # (B, L) padding mask. Ours is the 4D additive tree mask, which that
            # code cannot read -- it raised IndexError on attention_mask[b].bool().
            # Compute the positions here from the real padding mask and hand them
            # over, so the encoder skips its own derivation. Position ids are
            # per-token, so the tree mask is irrelevant to them.
            extra["position_ids"] = self.encoder.get_rope_index(
                batch["packed_ids"],
                image_grid_thw=batch["image_grid_thw"],
                attention_mask=batch["packed_mask"].long(),
                mm_token_type_ids=extra["mm_token_type_ids"],
            )[0]
        hidden = self.encoder(
            input_ids=batch["packed_ids"],
            attention_mask=mask,
            **extra,
        ).last_hidden_state.float()
        hidden = self.norm(hidden)

        # Mean-pool the context span and each option span.
        context = (hidden * batch["context_span"].unsqueeze(-1)).sum(1)
        context = context / batch["context_span"].sum(-1, keepdim=True).clamp_min(1)
        query = self.context_proj(context)                     # (B, r)

        spans = batch["option_span"]                           # (B, F, N, L)
        weights = spans.sum(-1, keepdim=True).clamp_min(1)
        pooled = torch.einsum("bfnl,blw->bfnw", spans, hidden) / weights
        keys = self.option_proj(pooled)                        # (B, F, N, r)

        # The field's words are pooled from the sequence like everything else,
        # and added to the context query. No per-field parameters exist.
        field = torch.einsum("bfl,blw->bfw", batch["field_span"], hidden)
        field = field / batch["field_span"].sum(-1, keepdim=True).clamp_min(1)
        query = query[:, None, :] + self.context_proj(field)        # (B, F, r)
        logits = (query[:, :, None, :] * keys).sum(-1) / self.rank ** 0.5
        return logits.masked_fill(~batch["option_mask"], torch.finfo(logits.dtype).min)


# Registering against the Auto classes, and recording auto_map in config.json, is
# what makes a saved directory the whole model: someone with transformers and no
# copy of this package can load it with
# AutoModel.from_pretrained(path, trust_remote_code=True). These have to run at
# module import in a real module -- defining the classes in __main__ silently
# writes auto_map: null and the directory stops being portable.
AutoConfig.register(PackedScorerConfig.model_type, PackedScorerConfig, exist_ok=True)
AutoModel.register(PackedScorerConfig, PackedScorer, exist_ok=True)
PackedScorerConfig.register_for_auto_class()
PackedScorer.register_for_auto_class("AutoModel")
