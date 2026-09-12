"""Detect config keys that no model declares (#627).

Pydantic's default is ``extra="ignore"``, and none of the config models
override it, so a key the schema does not know is dropped in silence. The run
then proceeds with the setting the user asked for simply not applied:
``quantizaton: none`` trains 4-bit quantized, ``gradient_checkpoint: true``
does no checkpointing, ``data.max_len: 512`` truncates at 2048. Each is one
edit away from a real field, which is what makes them likely rather than
exotic.

Each example above is one where the dropped value *differs* from the schema
default, which is the only kind worth printing. ``quantization`` already
defaults to ``"4bit"``, so misspelling that key while asking for the default
is the harmless member of this population -- it reads like a disaster and
changes nothing, which teaches the wrong lesson about what was fixed here.
``TestTheDocumentedExamplesAreOnesThatActuallyBreak`` pins the property
against the live schema, so a later default change cannot quietly make an
example vacuous again.

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

from soup_cli.config.schema import (
    ROOT_LEVEL_MISPLACED_KEYS,
    SoupConfig,
    remap_root_level_misplaced_keys,
)

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

#: Findings are capped, and so is the work: every unknown key costs a difflib
#: pass over the ~240 declared field names, and a config string reaches this
#: walk from the Web UI and the MCP server as well as from a local file. With
#: no cap, 50,000 bogus keys under one section cost ~31 s of CPU per request
#: (measured); the walk now stops once this many findings are collected, which
#: bounds it regardless of input size. A config with more unknown keys than
#: this is not a config with typos, and the report says it was capped.
_MAX_REPORTED_UNKNOWN_KEYS = 100

#: A key longer than this is shown truncated so a megabyte key name is not
#: echoed into a terminal or a JSON error body. (difflib itself is not the
#: cost here: its length pre-filter discards a long key against a short field
#: name at once, so no separate gate is needed for the fuzzy match.)
_MAX_KEY_LEN = 256
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
        # get_origin is not None exactly for parameterized generics, which
        # must be filtered before issubclass: on Python 3.10 a GenericAlias
        # like list[str] passes isinstance(c, type) and then raises
        # TypeError from issubclass (3.11 made isinstance return False).
        # The repo supports 3.10, and any walk of a full model_dump() — what
        # sweep's pre-flight feeds this — visits list[...]/dict[...] fields.
        if isinstance(c, type)
        and typing.get_origin(c) is None
        and issubclass(c, pydantic.BaseModel)
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
        if len(found) >= _MAX_REPORTED_UNKNOWN_KEYS:
            return
        if not isinstance(key, str):
            continue
        shown = key if len(key) <= _MAX_KEY_LEN else key[:_MAX_KEY_LEN] + "..."
        path = f"{prefix}{shown}"
        if key not in declared:
            suggestions = tuple(
                difflib.get_close_matches(
                    key, list(declared), n=_MAX_SUGGESTIONS, cutoff=_SUGGESTION_CUTOFF
                )
            )
            found.append(UnknownKey(path=path, key=shown, suggestions=suggestions))
            continue
        for nested in _nested_models(model, key):
            _walk(value, nested, f"{path}.", found)


def find_unknown_config_keys(raw: dict) -> list[UnknownKey]:
    """Return every key in ``raw`` that no config model declares.

    Walks the whole tree -- ``data``, ``training``, ``training.lora`` and the
    rest -- so a guard applied to one model and forgotten on another does not
    look like it works.

    The walk sees the config the way :class:`SoupConfig` does: the root-level
    ``lora:`` spelling that the schema has accepted and moved under
    ``training`` since v0.40.1 is remapped here first, through the same
    function, so a spelling the validator accepts is never one this detector
    refuses (v0.75.0, #879). A key present at both levels is left for the
    validator, which raises the precise error for it.

    At most :data:`_MAX_REPORTED_UNKNOWN_KEYS` findings are returned; the
    report says so when the cap was hit.
    """
    try:
        normalised = remap_root_level_misplaced_keys(raw)
    except ValueError:
        # The key sits at both levels. The validator raises the precise error
        # for that; walking the raw dict here would report the root key as
        # unknown and refuse first, with the worse message. Walk without it.
        normalised = {k: v for k, v in raw.items() if k not in ROOT_LEVEL_MISPLACED_KEYS}
    found: list[UnknownKey] = []
    _walk(normalised, SoupConfig, "", found)
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

    ``include_deadline`` is off for callers that refuse. ``sweep.py`` raised
    from the start whatever the loader's switch said, and since v0.75 the
    loader refuses too, so appending a *future* rejection would describe a
    future that has already arrived. The same flag picks the per-key suffix:
    a warning caller proceeds with the key ignored ("Not applied."), a refusing
    caller does not proceed at all, and saying "not applied" there would read
    as if the run went ahead without it.
    """
    # ``include_deadline`` does double duty on purpose: the callers that append
    # a deadline are exactly the callers that proceed, so one flag decides both
    # the trailing sentence and the per-key verdict word.
    suffix = "Not applied." if include_deadline else "Refused."
    lines = []
    for item in unknown:
        if item.suggestions:
            # After the question mark the suffix starts a new sentence.
            hint = " or ".join(f"'{s}'" for s in item.suggestions)
            lines.append(f"unknown config key '{item.path}' - did you mean {hint}? {suffix}")
        else:
            # After " - " it continues the sentence, hence lowercase.
            lines.append(f"unknown config key '{item.path}' - {suffix.lower()}")
    if len(unknown) >= _MAX_REPORTED_UNKNOWN_KEYS:
        lines.append(
            f"(report capped at {_MAX_REPORTED_UNKNOWN_KEYS} unknown keys; "
            "fix these and load again)"
        )
    if include_deadline and lines:
        lines.append(deadline_notice())
    return "\n".join(lines)
