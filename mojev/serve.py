"""A System One API server, wire-compatible with the official TypeSafe Python SDK.

The contract is taken from the SDK itself, not guessed:

* ``POST /v1/systemone`` and ``GET /v1/models`` (`_core/constants.py`)
* request and response shapes from the generated wire schemas
  (`_schemas/models.py`, itself generated from the published openapi.json)
* ``Authorization: Bearer <key>``, ``x-typesafe-request-id`` echoed on responses
* 422 with a ``{"detail": [{"loc": [...], "msg": ..., "type": ...}]}`` body for
  validation failures, which is what `HTTPValidationError` parses

Three question types map onto this repository's schema kinds directly:

``noul``    a two-option decision; the answer reports P(yes) as a scalar
``choice``  one of the named ``criteria`` keys; the answer reports the winning
            name, its confidence, and the full probability map
``score``   ordered ``criteria`` levels; the answer reports the
            probability-weighted expected level, so it can fall between integers

What makes this serveable at all is that the model takes candidates as text.
Criteria arrive at request time and have never been seen during training, so a
scorer with a fixed class vocabulary could not answer them; this one encodes the
field's words and each candidate's words in the same packed sequence.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from typing import Any

import torch

SYSTEM_ONE_PATH = "/v1/systemone"
MODELS_PATH = "/v1/models"
REQUEST_ID_HEADER = "x-typesafe-request-id"

MODELS = [{
    "name": "mojev-latest",
    "description": "Plackett-Luce trained candidate scorer over packed sequences.",
    "release_date": "2026-09-21",
}]


class ValidationFailure(Exception):
    """Carries a 422 body in the shape the SDK's HTTPValidationError parses."""

    def __init__(self, loc: list, msg: str, kind: str = "value_error",
                 value: Any = None) -> None:
        super().__init__(msg)
        detail: dict[str, Any] = {"loc": loc, "msg": msg, "type": kind}
        if value is not None:
            detail["input"] = value
        self.detail = [detail]


def option_texts(name: str, question: dict) -> tuple[list[str], str]:
    """(candidate strings, kind) for one question, or raise a 422.

    A candidate's text is what the model scores, so a criterion's description is
    included when present -- ``{"angry": "An upset message"}`` scores better as
    "angry: An upset message" than as the bare key.
    """
    kind = question.get("type")
    if not isinstance(kind, str) or not kind:
        raise ValidationFailure(
            ["body", "questions", name, "type"], "Field required", "missing", question
        )

    if kind == "noul":
        criteria = question.get("criteria") or {}
        if not isinstance(criteria, dict):
            raise ValidationFailure(
                ["body", "questions", name, "noul", "criteria"],
                "Input should be an object", "model_type", criteria,
            )
        no = criteria.get("false") or "no"
        yes = criteria.get("true") or "yes"
        return [_render(no), _render(yes)], kind

    if kind == "choice":
        criteria = question.get("criteria")
        if not isinstance(criteria, dict) or not criteria:
            raise ValidationFailure(
                ["body", "questions", name, "choice", "criteria"],
                "Field required", "missing", question,
            )
        return [
            key if value in (None, "") else f"{key}: {_render(value)}"
            for key, value in criteria.items()
        ], kind

    if kind == "score":
        criteria = question.get("criteria")
        if not isinstance(criteria, list) or not criteria:
            raise ValidationFailure(
                ["body", "questions", name, "score", "criteria"],
                "List should have at least 1 item after validation, not 0",
                "too_short", criteria,
            )
        return [_render(level) for level in criteria], kind

    raise ValidationFailure(
        ["body", "questions", name, "type"],
        f"Input should be 'noul', 'choice' or 'score', not {kind!r}",
        "literal_error", kind,
    )


def _render(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def build_answer(name: str, question: dict, kind: str, probabilities: list[float]) -> dict:
    """Assemble one answer object in the exact shape the SDK validates."""
    if kind == "noul":
        # The wire model carries a single scalar: P(yes).
        return {"type": "noul", "noul": float(probabilities[1])}

    if kind == "choice":
        keys = list(question["criteria"].keys())
        mapped = {key: float(p) for key, p in zip(keys, probabilities)}
        best = max(mapped, key=mapped.__getitem__)
        return {"type": "choice", "choice": best,
                "confidence": mapped[best], "probabilities": mapped}

    levels = question["criteria"]
    # Keys are strings on the wire; the SDK coerces them to integer levels.
    mapped = {str(index): float(p) for index, p in enumerate(probabilities)}
    expected = sum(index * p for index, p in enumerate(probabilities))
    return {
        "type": "score",
        "score": float(expected),
        "confidence": float(max(probabilities)),
        "legend": {str(index): _render(level) for index, level in enumerate(levels)},
        "probabilities": mapped,
    }


class Engine:
    """Wraps a trained checkpoint and answers one request at a time.

    Every question in a request is scored in a single forward pass: the questions
    and their candidates are packed into one sequence under the
    state -> question -> candidate tree mask, so N questions cost one pass rather
    than N.
    """

    def __init__(self, checkpoint: str, tokenizer: str | None = None,
                 device: str = "cuda", context_tokens: int | None = None) -> None:
        from transformers import AutoProcessor, AutoTokenizer

        from .evaluate import load_packed
        from .schema import Field, Schema

        self.device = torch.device(device)
        # A local directory or Hub repo carries the config, weights and tokenizer,
        # so serving needs only one model identifier.
        self.model, trained, self.config = load_packed(checkpoint, self.device)
        self.context_tokens = (self.config.context_tokens if context_tokens is None
                               else context_tokens)
        if self.context_tokens < 1:
            raise ValueError("context_tokens must be positive")
        # A released checkpoint carries its tokenizer and image processor as one
        # artifact. Text and images therefore share the exact preprocessing
        # contract used by that release.
        self.processor = AutoProcessor.from_pretrained(
            checkpoint,
            trust_remote_code=True,
        )
        self.tokenizer = (
            AutoTokenizer.from_pretrained(tokenizer)
            if tokenizer is not None else self.processor.tokenizer
        )
        # The checkpoint holds no per-field parameters, so the schema it was
        # trained with does not constrain what can be served.
        self.trained_schema = trained
        self._Field, self._Schema = Field, Schema

    @torch.no_grad()
    def answer(self, state: Any, questions: dict[str, dict]) -> tuple[dict, dict]:
        """(answers keyed by question name, usage)."""
        from .full import packed_collate, sort_candidates, unsort
        from .data import Example

        names, kinds, menus, orders, fields = [], [], [], [], []
        for name, question in questions.items():
            options, kind = option_texts(name, question)
            # Candidates are sorted before scoring and the answer is mapped back;
            # see sort_candidates for why the invariance depends on it.
            order = sort_candidates(options)
            names.append(name)
            kinds.append(kind)
            menus.append(tuple(options[i] for i in order))
            orders.append(order)
            instructions = question.get("instructions")
            fields.append(self._Field(
                name, "choice", tuple(options[i] for i in order),
                _render(instructions) if instructions is not None else "",
            ))

        # A schema per request. The model holds no per-field parameters, so
        # criteria it has never seen are simply text it encodes. The collator
        # pads the ragged option axis to the widest menu and sets option_mask,
        # which keeps padding out of every softmax.
        schema = self._Schema(tuple(fields))
        example = Example(
            context=_render(state),
            labels=tuple(0 for _ in names),
            options=tuple(menus),
        )
        collate = packed_collate(self.tokenizer, schema,
                                 self.context_tokens,
                                 processor=self.processor)
        batch = collate([example])
        batch = {k: v.to(self.device) for k, v in batch.items()}
        logits = self.model(batch)[0]

        answers = {}
        for index, (name, kind, menu, order) in enumerate(zip(names, kinds, menus, orders)):
            row = logits[index, : len(menu)]
            # Undo the sort so probabilities line up with the caller's criteria.
            probabilities = unsort(order, row.softmax(-1).tolist())
            answers[name] = build_answer(name, questions[name], kind, probabilities)
        usage = {
            "input_tokens": int(batch["packed_mask"].sum()),
            "output_tokens": 0,     # the decision is the softmax; nothing is generated
        }
        return answers, usage


def make_app(engine: Engine, api_key: str | None, model_name: str):
    """A WSGI application implementing the two endpoints."""

    def respond(start_response, status: str, payload: dict, request_id: str):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        start_response(status, [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(body))),
            (REQUEST_ID_HEADER, request_id),
        ])
        return [body]

    def app(environ, start_response):
        request_id = str(uuid.uuid4())
        path = environ.get("PATH_INFO", "")
        method = environ.get("REQUEST_METHOD", "GET")

        if api_key is not None:
            supplied = environ.get("HTTP_AUTHORIZATION", "")
            if supplied != f"Bearer {api_key}":
                return respond(start_response, "401 Unauthorized",
                               {"detail": "Invalid API key."}, request_id)

        # The SDK concatenates base_url + path, and base_url may carry a prefix
        # ("https://host/prefix" -> "/prefix/v1/systemone"), so match on suffix.
        if path.endswith(MODELS_PATH) and method == "GET":
            return respond(start_response, "200 OK", {"models": MODELS}, request_id)

        if not path.endswith(SYSTEM_ONE_PATH):
            return respond(start_response, "404 Not Found",
                           {"detail": f"Unknown path {path}."}, request_id)
        if method != "POST":
            return respond(start_response, "405 Method Not Allowed",
                           {"detail": "Use POST."}, request_id)

        try:
            length = int(environ.get("CONTENT_LENGTH") or 0)
            payload = json.loads(environ["wsgi.input"].read(length) or b"{}")
        except (ValueError, KeyError):
            return respond(start_response, "400 Bad Request",
                           {"detail": "Body must be JSON."}, request_id)

        if not isinstance(payload, dict):
            return respond(start_response, "422 Unprocessable Entity",
                           {"detail": [{"loc": ["body"], "msg": "Input should be an object",
                                        "type": "model_type"}]}, request_id)
        if "state" not in payload:
            return respond(start_response, "422 Unprocessable Entity",
                           {"detail": [{"loc": ["body", "state"], "msg": "Field required",
                                        "type": "missing"}]}, request_id)
        if not isinstance(payload.get("model"), str):
            return respond(start_response, "422 Unprocessable Entity",
                           {"detail": [{"loc": ["body", "model"], "msg": "Field required",
                                        "type": "missing"}]}, request_id)
        questions = payload.get("questions")
        if not isinstance(questions, dict) or not questions:
            return respond(start_response, "422 Unprocessable Entity",
                           {"detail": [{"loc": ["body", "questions"],
                                        "msg": "Field required" if questions is None
                                        else "Object should have at least 1 item",
                                        "type": "missing" if questions is None else "too_short"}]},
                           request_id)

        started = time.perf_counter()
        try:
            answers, usage = engine.answer(payload["state"], questions)
        except ValidationFailure as failure:
            return respond(start_response, "422 Unprocessable Entity",
                           {"detail": failure.detail}, request_id)
        except Exception as error:                       # noqa: BLE001 - reported as 500
            return respond(start_response, "500 Internal Server Error",
                           {"detail": f"{type(error).__name__}: {error}"}, request_id)

        environ["mojev.latency_ms"] = (time.perf_counter() - started) * 1000
        return respond(start_response, "200 OK", {
            "model": model_name, "answers": answers, "usage": usage,
        }, request_id)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        help="local model directory or Hugging Face repo ID",
    )
    parser.add_argument("--tokenizer",
                        help="override tokenizer with a local path or Hugging Face repo ID")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--api-key", default=None,
                        help="require this key; omit to accept any request")
    parser.add_argument("--model-name", default="mojev-latest")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--context-tokens", type=int,
                        help="inference state-token window; defaults to the checkpoint setting")
    args = parser.parse_args()

    engine = Engine(args.checkpoint, args.tokenizer, args.device,
                    args.context_tokens)
    app = make_app(engine, args.api_key, args.model_name)
    from wsgiref.simple_server import make_server

    print(json.dumps({"listening": f"http://{args.host}:{args.port}",
                      "endpoints": [SYSTEM_ONE_PATH, MODELS_PATH],
                      "model": args.model_name}), flush=True)
    make_server(args.host, args.port, app).serve_forever()


if __name__ == "__main__":
    main()
