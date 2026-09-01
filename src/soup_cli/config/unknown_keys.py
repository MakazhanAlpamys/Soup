"""Detect config keys that no model declares (#627).

Pydantic's default is ``extra="ignore"``, and none of the config models
override it, so a key the schema does not know is dropped in silence. The run
then proceeds with the setting the user asked for simply not applied:
``quantizaton: 4bit`` trains in full precision, ``gradient_checkpoint: true``
does no checkpointing, ``data.max_len: 512`` truncates at the default. Each is
one edit away from a real field, which is what makes them likely rather than
exotic.

#623 is the live case: ``training.stream_pin`` reached main two days after
0.73.3 shipped, so a user on the released wheel wrote the documented escape
hatch, ``--dry-run`` reported "Config valid", the key was discarded, and the
resulting OOM was investigated as a layer-streaming defect.

This module is the detection half only. It walks the raw mapping against the
model tree and reports what it cannot place, with a suggestion drawn from the
fields that model actually declares. **What to do about a finding -- raise or
warn -- is the caller's choice**, kept deliberately separate so the severity is
one switch rather than a rewrite.

Pure and dependency-light: stdlib plus the schema. No torch, no I/O, no network,
so it is fully unit-testable on any machine.
"""

from __future__ import annotations

import difflib
import typing
from dataclasses import dataclass

import pydantic

from soup_cli.config.schema import SoupConfig

__all__ = ["UnknownKey", "find_unknown_config_keys", "format_unknown_keys"]

#: difflib cutoff. 0.6 resolves every case reported in #627 on the first
#: suggestion while leaving an unrelated key (``zzzzzzzz``) with none.
_SUGGESTION_CUTOFF = 0.6
_MAX_SUGGESTIONS = 2


@dataclass(frozen=True)
class UnknownKey:
    """One key the schema does not declare."""

    path: str
    key: str
    suggestions: tuple[str, ...]


def _nested_models(model: type[pydantic.BaseModel], field: str) -> list[type]:
    """Every BaseModel a field could hold, unwrapping Optional/Union."""
    annotation = model.model_fields[field].annotation
    candidates = (annotation, *typing.get_args(annotation))
    return [
        c
        for c in candidates
        if isinstance(c, type) and issubclass(c, pydantic.BaseModel)
    ]


def _walk(
    raw: object,
    model: type[pydantic.BaseModel],
    prefix: str,
    found: list[UnknownKey],
) -> None:
    if not isinstance(raw, dict):
        # A non-mapping where a section belongs is a *type* error, which
        # Pydantic reports far better than this walk could. Not our business.
        return

    declared = model.model_fields
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        path = f"{prefix}{key}"
        if key not in declared:
            found.append(
                UnknownKey(
                    path=path,
                    key=key,
                    suggestions=tuple(
                        difflib.get_close_matches(
                            key, list(declared), n=_MAX_SUGGESTIONS, cutoff=_SUGGESTION_CUTOFF
                        )
                    ),
                )
            )
            continue
        for nested in _nested_models(model, key):
            _walk(value, nested, f"{path}.", found)


def find_unknown_config_keys(raw: dict) -> list[UnknownKey]:
    """Return every key in ``raw`` that no config model declares.

    Walks the whole tree -- ``data``, ``training``, ``training.lora`` and the
    rest -- so a guard applied to one model and forgotten on another does not
    look like it works.
    """
    found: list[UnknownKey] = []
    _walk(raw, SoupConfig, "", found)
    return found


def format_unknown_keys(unknown: list[UnknownKey]) -> str:
    """Render findings for an operator, naming the field they likely meant."""
    lines = []
    for item in unknown:
        if item.suggestions:
            hint = " or ".join(f"'{s}'" for s in item.suggestions)
            lines.append(f"unknown config key '{item.path}' - did you mean {hint}?")
        else:
            lines.append(f"unknown config key '{item.path}'")
    return "\n".join(lines)
