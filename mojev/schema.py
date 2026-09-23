"""Typed decision schemas.

A schema declares, in advance, every field a query returns and every value each
field may take. Type safety is therefore constructive: the model normalises only
over declared options, so a returned struct cannot name an option that does not
exist and cannot omit a declared field. Nothing is parsed or validated at
inference time.

Four field kinds cover the shapes decisions actually take:

- ``choice``  exactly one of N unordered options (softmax over N)
- ``bool``    a two-option choice, kept separate so it calibrates on its own
- ``multi``   any subset of N options (independent sigmoid per option)
- ``bucket``  one of N *ordered* options, scored with cumulative logits so that
              the order is part of the model rather than an accident of labels
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["choice", "bool", "multi", "bucket"]
KINDS: tuple[Kind, ...] = ("choice", "bool", "multi", "bucket")

BOOL_OPTIONS = ("false", "true")


@dataclass(frozen=True)
class Field:
    """One decision within a schema."""

    name: str
    kind: Kind = "choice"
    options: tuple[str, ...] = BOOL_OPTIONS
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("field name must not be empty")
        if self.kind not in KINDS:
            raise ValueError(f"{self.name}: unknown kind {self.kind!r}, expected one of {KINDS}")
        if self.kind == "bool":
            if self.options != BOOL_OPTIONS:
                raise ValueError(f"{self.name}: bool options are fixed to {BOOL_OPTIONS}")
        elif len(self.options) < 2:
            raise ValueError(f"{self.name}: {self.kind} needs at least two options")
        if len(set(self.options)) != len(self.options):
            raise ValueError(f"{self.name}: options must be unique")
        if any(not option for option in self.options):
            raise ValueError(f"{self.name}: options must not be empty strings")

    @property
    def cardinality(self) -> int:
        return len(self.options)

    @property
    def prompt(self) -> str:
        """The text the model reads to know what this field asks.

        This is what makes a schema the model never trained on usable: the field
        is identified by its words, not by a row index into a learned table. An
        earlier version stored one trainable vector per field, which meant adding
        a field to a schema raised a shape error and renaming one made the field
        unrecognisable.
        """
        parts = [self.name.replace("_", " ")]
        if self.description:
            parts.append(self.description)
        parts.append(f"kind: {self.kind}")
        if self.kind != "choice" or len(self.options) <= 8:
            parts.append("options: " + ", ".join(self.options[:8]))
        return " | ".join(parts)

    @property
    def single(self) -> bool:
        """True when exactly one option is correct, so the field owns a softmax."""
        return self.kind in ("choice", "bool", "bucket")

    def index(self, value: str) -> int:
        try:
            return self.options.index(value)
        except ValueError:
            raise ValueError(
                f"{self.name}: {value!r} is not a declared option; expected one of {self.options}"
            ) from None

    def encode(self, value) -> list[int] | int:
        """Label for one row: an option index, or a 0/1 vector for ``multi``."""
        if self.kind == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"{self.name}: expected a bool, got {value!r}")
            return int(value)
        if self.kind == "multi":
            if isinstance(value, str) or not isinstance(value, (list, tuple, set)):
                raise ValueError(f"{self.name}: multi expects a list of options, got {value!r}")
            chosen = {self.index(item) for item in value}
            return [int(index in chosen) for index in range(self.cardinality)]
        if not isinstance(value, str):
            raise ValueError(f"{self.name}: expected an option string, got {value!r}")
        return self.index(value)

    def decode(self, probabilities, threshold: float = 0.5):
        """Probabilities for this field -> the typed value plus its confidence."""
        if len(probabilities) != self.cardinality:
            raise ValueError(
                f"{self.name}: expected {self.cardinality} probabilities, got {len(probabilities)}"
            )
        if self.kind == "multi":
            selected = [option for option, p in zip(self.options, probabilities) if p >= threshold]
            confidence = min((max(p, 1.0 - p) for p in probabilities), default=1.0)
            return selected, float(confidence)
        best = max(range(self.cardinality), key=probabilities.__getitem__)
        value = bool(best) if self.kind == "bool" else self.options[best]
        return value, float(probabilities[best])


@dataclass(frozen=True)
class Schema:
    """The full set of fields one query returns."""

    fields: tuple[Field, ...]

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError("a schema needs at least one field")
        names = [field.name for field in self.fields]
        if len(set(names)) != len(names):
            raise ValueError("field names must be unique")

    def __len__(self) -> int:
        return len(self.fields)

    def __iter__(self):
        return iter(self.fields)

    def __getitem__(self, name: str) -> Field:
        for field in self.fields:
            if field.name == name:
                return field
        raise KeyError(name)

    @property
    def max_cardinality(self) -> int:
        return max(field.cardinality for field in self.fields)

    def encode(self, values: dict) -> list:
        """One row of labels, in field order. Every declared field must be present."""
        missing = [field.name for field in self.fields if field.name not in values]
        if missing:
            raise ValueError(f"missing labels for fields: {missing}")
        extra = set(values) - {field.name for field in self.fields}
        if extra:
            raise ValueError(f"labels for undeclared fields: {sorted(extra)}")
        return [field.encode(values[field.name]) for field in self.fields]

    @property
    def prompts(self) -> tuple[str, ...]:
        return tuple(field.prompt for field in self.fields)

    def to_json(self) -> dict:
        return {
            "fields": [
                {"name": f.name, "kind": f.kind, "options": list(f.options),
                 "description": f.description}
                for f in self.fields
            ]
        }

    @classmethod
    def from_json(cls, payload: dict) -> "Schema":
        return cls(
            tuple(
                Field(item["name"], item.get("kind", "choice"), tuple(item["options"]),
                      item.get("description", ""))
                for item in payload["fields"]
            )
        )
