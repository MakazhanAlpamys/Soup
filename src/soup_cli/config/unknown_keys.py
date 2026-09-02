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

__all__ = [
    "UNKNOWN_KEY_REJECTION_VERSION",
    "UnknownKey",
    "deadline_notice",
    "find_unknown_config_keys",
    "format_unknown_keys",
]

#: The release that stops warning about unknown keys and starts refusing them.
#:
#: Written out **once**, here. The loader message, the docs line and the
#: deadline test all read this constant rather than repeating the string,
#: because the failure mode of a duplicate is a message that keeps promising a
#: rejection after the rejection has shipped. ``TestTheDeadline`` asserts it
#: against the declared ``soup_cli.__version__`` instead of a literal, so the
#: release that crosses the deadline turns a test red rather than turning the
#: warning into a lie.
#:
#: 0.75 rather than 0.74 because the release carrying *this* warning is v0.74.0
#: itself -- 71 fragments have accumulated since v0.73.3, 18 of them ``added``,
#: which is a minor and not a patch. Naming 0.74 would have given the warning
#: zero releases of notice, which is the outcome the warn-then-forbid decision
#: exists to avoid. One minor of notice: warn in 0.74, refuse in 0.75.
UNKNOWN_KEY_REJECTION_VERSION = "0.75"

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


def deadline_notice() -> str:
    """The one sentence that turns the warning into a deadline.

    Derived from :data:`UNKNOWN_KEY_REJECTION_VERSION` so the version is never
    typed twice.
    """
    return (
        f"Soup v{UNKNOWN_KEY_REJECTION_VERSION} will reject unknown config keys "
        "instead of warning."
    )


def format_unknown_keys(
    unknown: list[UnknownKey], *, include_deadline: bool = True
) -> str:
    """Render findings for an operator, naming the field they likely meant.

    One report per call with every finding listed together -- a config copied
    from a newer Soup trips several keys at once, and a panel per key buries
    the list it exists to present.

    ``include_deadline`` is off for callers that already refuse. ``sweep.py``
    raises today whatever the loader's switch says, so appending a *future*
    rejection there would describe a future that has already arrived for that
    caller.
    """
    lines = []
    for item in unknown:
        if item.suggestions:
            hint = " or ".join(f"'{s}'" for s in item.suggestions)
            lines.append(
                f"unknown config key '{item.path}' - did you mean {hint}? Not applied."
            )
        else:
            lines.append(f"unknown config key '{item.path}' - not applied.")
    if include_deadline and lines:
        lines.append(deadline_notice())
    return "\n".join(lines)
