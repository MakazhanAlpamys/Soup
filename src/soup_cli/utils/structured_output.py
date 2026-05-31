"""Structured output constraints for inference (v0.30.0).

Supports JSON schema and regex. Backed by ``outlines`` or
``lm-format-enforcer`` if installed; otherwise returns an inert constraint
descriptor so the server can degrade gracefully.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

Mode = Literal["off", "json", "regex"]

_VALID_MODES = {"off", "json", "regex"}
_MAX_REGEX_LEN = 2048
_MAX_SCHEMA_STR_LEN = 65536


def validate_mode(mode: Optional[str]) -> Mode:
    """Normalise and validate the structured-output mode string."""
    if mode is None:
        return "off"
    normalised = mode.strip().lower()
    if normalised not in _VALID_MODES:
        raise ValueError(
            f"Unknown structured-output mode: {mode!r}. "
            f"Valid: {sorted(_VALID_MODES)}"
        )
    return normalised  # type: ignore[return-value]


def validate_regex_pattern(pattern: str) -> str:
    """Validate a regex pattern: length-bounded, must compile.

    We don't try to detect catastrophic-backtracking patterns here — that is
    the library (outlines / lm-format-enforcer) problem at runtime. We just
    cap length to stop attacker-supplied giant strings.
    """
    if not isinstance(pattern, str):
        raise ValueError("regex pattern must be a string")
    if "\x00" in pattern:
        raise ValueError("regex pattern contains null byte")
    if len(pattern) > _MAX_REGEX_LEN:
        raise ValueError(
            f"regex pattern length {len(pattern)} exceeds max {_MAX_REGEX_LEN}"
        )
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"invalid regex: {exc}") from exc
    return pattern


def validate_json_schema(schema: Any) -> dict:
    """Accept a dict-shaped JSON schema. Must be JSON-serialisable and
    declare a top-level ``type`` field. Total serialised size is capped at
    64KB to prevent ReDoS-style schemas."""
    import json

    if not isinstance(schema, dict):
        raise ValueError("JSON schema must be a dict")
    try:
        serialised = json.dumps(schema)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"JSON schema is not serialisable: {exc}") from exc
    if len(serialised) > _MAX_SCHEMA_STR_LEN:
        raise ValueError(
            f"JSON schema size {len(serialised)} exceeds max {_MAX_SCHEMA_STR_LEN}"
        )
    # Minimum viable shape
    if "type" not in schema:
        raise ValueError("JSON schema must declare a 'type' field")
    return schema


def is_outlines_available() -> bool:
    try:
        import outlines  # noqa: F401

        return True
    except ImportError:
        return False


def is_lmfe_available() -> bool:
    try:
        import lmformatenforcer  # noqa: F401

        return True
    except ImportError:
        return False


def build_logits_processors(
    constraint: Optional[dict], tokenizer: Any,
) -> list:
    """Build a list of HF ``LogitsProcessor`` instances for ``constraint``.

    Returns an empty list when:
      - constraint is None / off
      - neither ``outlines`` nor ``lm-format-enforcer`` is installed
      - the chosen library cannot construct a processor for the given kind
        (we degrade to free-form rather than crashing the request)

    The returned list can be passed directly to
    ``model.generate(..., logits_processor=...)``.

    Security: this function never executes user-supplied code. The schema
    and regex are already validated upstream by ``validate_json_schema`` /
    ``validate_regex_pattern``.
    """
    if constraint is None:
        return []
    kind = constraint.get("kind")
    if kind not in ("json_schema", "regex"):
        return []

    # Prefer outlines (broader coverage); fall back to lm-format-enforcer.
    if is_outlines_available():
        try:
            return _build_outlines_processors(constraint, tokenizer)
        except Exception:  # noqa: BLE001 — degrade to free-form rather than 500
            return []
    if is_lmfe_available():
        try:
            return _build_lmfe_processors(constraint, tokenizer)
        except Exception:  # noqa: BLE001
            return []
    return []


def _build_outlines_processors(constraint: dict, tokenizer: Any) -> list:
    """Best-effort outlines integration. Schema-driver may be missing on
    older outlines builds so we try multiple entry points."""
    import outlines  # type: ignore

    kind = constraint["kind"]
    if kind == "json_schema":
        builder = (
            getattr(outlines, "JsonSchema", None)
            or getattr(outlines, "regex", None)
        )
        if builder is None:
            return []
        # outlines >= 0.1: outlines.processors.JSONLogitsProcessor
        proc_factory = getattr(
            __import__("outlines.processors", fromlist=["JSONLogitsProcessor"]),
            "JSONLogitsProcessor", None,
        )
        if proc_factory is None:
            return []
        return [proc_factory(constraint["schema"], tokenizer)]
    if kind == "regex":
        proc_factory = getattr(
            __import__("outlines.processors", fromlist=["RegexLogitsProcessor"]),
            "RegexLogitsProcessor", None,
        )
        if proc_factory is None:
            return []
        return [proc_factory(constraint["pattern"], tokenizer)]
    return []


def _build_lmfe_processors(constraint: dict, tokenizer: Any) -> list:
    """lm-format-enforcer integration."""
    from lmformatenforcer import (  # type: ignore
        JsonSchemaParser,
        RegexParser,
    )
    from lmformatenforcer.integrations.transformers import (  # type: ignore
        build_transformers_prefix_allowed_tokens_fn,
    )
    from transformers import LogitsProcessorList

    kind = constraint["kind"]
    if kind == "json_schema":
        parser = JsonSchemaParser(constraint["schema"])
    elif kind == "regex":
        parser = RegexParser(constraint["pattern"])
    else:
        return []

    fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
    # PrefixConstrainedLogitsProcessor wants num_beams. We use 1 (greedy /
    # standard sampling) for chat completions.
    from transformers import PrefixConstrainedLogitsProcessor

    proc = PrefixConstrainedLogitsProcessor(fn, 1)
    processors = LogitsProcessorList()
    processors.append(proc)
    return list(processors)


def build_constraint(
    mode: Mode,
    json_schema: Optional[dict],
    regex_pattern: Optional[str],
) -> Optional[dict]:
    """Build a constraint descriptor for the server.

    Return shape: a dict with ``kind`` + kind-specific fields, or None if
    constraint is off / unsupported.

    Callers should treat None as "no constraint" (free-form generation).
    """
    if mode == "off":
        return None
    if mode == "json":
        if json_schema is not None:
            schema = validate_json_schema(json_schema)
            return {"kind": "json_schema", "schema": schema}
        return None  # free-form JSON is library-dependent; fall through
    if mode == "regex":
        if not regex_pattern:
            raise ValueError("regex mode requires a non-empty pattern")
        pattern = validate_regex_pattern(regex_pattern)
        return {"kind": "regex", "pattern": pattern}
    # Unreachable — validate_mode would have rejected
    raise ValueError(f"unexpected mode: {mode!r}")
