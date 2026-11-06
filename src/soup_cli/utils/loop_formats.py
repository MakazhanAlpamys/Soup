"""The data formats of the `soup loop` cycle: names, default layout, schemas.

Six append-only JSONL formats connect serving, feedback, training and rollout
(described in ``docs/loop-formats.md``). Their JSON Schemas and one example
file each ship as package data under ``soup_cli/schemas/loop/``. This module
locates and reads them and computes the one value a reader needs to build
itself, ``prompt_hash``: it imports nothing heavy and does not validate
(``jsonschema`` is a test dependency, not a runtime one).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from importlib import resources
from importlib.resources.abc import Traversable
from types import MappingProxyType
from typing import Any

#: The ``schema_version`` every format is currently at.
SCHEMA_VERSION = 1

#: Format name -> default path of its log, relative to the project directory.
#: ``conversation`` is a directory: every ``*.jsonl`` file in it is read.
DEFAULT_LAYOUT: Mapping[str, str] = MappingProxyType(
    {
        "conversation": ".soup-loops/traces",
        "feedback": ".soup-loops/feedback.jsonl",
        "paraphrase": ".soup-loops/paraphrases.jsonl",
        "iteration": ".soup-loops/iterations.jsonl",
        "version": ".soup-loops/versions.jsonl",
        "progress": ".soup-loops/progress.jsonl",
    }
)

#: The format names, in the order the cycle produces them.
FORMATS: tuple[str, ...] = tuple(DEFAULT_LAYOUT)

_PACKAGE = "soup_cli.schemas.loop"


def _check_name(name: str) -> str:
    if name not in DEFAULT_LAYOUT:
        raise ValueError(f"unknown loop format {name!r}; expected one of {', '.join(FORMATS)}")
    return name


def schema_resource(name: str) -> Traversable:
    """The packaged JSON Schema file of format ``name``."""
    return resources.files(_PACKAGE) / f"{_check_name(name)}.schema.json"


def example_resource(name: str) -> Traversable:
    """The packaged example JSONL file of format ``name``."""
    return resources.files(_PACKAGE) / "examples" / f"{_check_name(name)}.jsonl"


def load_schema(name: str) -> dict[str, Any]:
    """Parse and return the JSON Schema of format ``name``."""
    return json.loads(schema_resource(name).read_text(encoding="utf-8"))


def load_examples(name: str) -> list[dict[str, Any]]:
    """Parse and return the example records of format ``name``, one per line."""
    text = example_resource(name).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def prompt_hash(messages: Sequence[Mapping[str, Any]]) -> str:
    """The ``prompt_hash`` of a prompt: the key that ties paraphrases to an answer's prompt.

    ``messages`` is the conversation record's ``messages`` array: what the model
    saw, without the answer. The hash is the lowercase SHA-256 hex digest of that
    array serialised as JSON with sorted keys, no whitespace (``separators=(",",
    ":")``) and non-ASCII characters written as themselves, encoded as UTF-8.
    Nothing is trimmed, case-folded or reordered, so equal messages hash equal and
    a one-character change gives another hash.
    """
    if not isinstance(messages, (list, tuple)):
        raise ValueError(f"messages must be a list of messages, not {type(messages).__name__}")
    if not messages:
        raise ValueError("messages must hold at least one message")
    for position, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(
                f"message {position} must be a mapping with a role and a content, "
                f"not {type(message).__name__}"
            )
    text = json.dumps(
        [dict(message) for message in messages],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
