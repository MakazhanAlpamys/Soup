"""The data formats of the `soup loop` cycle: names, default layout, schemas.

Five append-only JSONL formats connect serving, feedback, training and rollout
(described in ``docs/loop-formats.md``). Their JSON Schemas and one example
file each ship as package data under ``soup_cli/schemas/loop/``. This module
only locates and reads them: it imports nothing heavy and does not validate
(``jsonschema`` is a test dependency, not a runtime one).
"""

from __future__ import annotations

import json
from importlib import resources
from importlib.resources.abc import Traversable
from types import MappingProxyType
from typing import Any, Mapping

#: The ``schema_version`` every format is currently at.
SCHEMA_VERSION = 1

#: Format name -> default path of its log, relative to the project directory.
#: ``conversation`` is a directory: every ``*.jsonl`` file in it is read.
DEFAULT_LAYOUT: Mapping[str, str] = MappingProxyType(
    {
        "conversation": ".soup-loops/conversations",
        "feedback": ".soup-loops/feedback.jsonl",
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
