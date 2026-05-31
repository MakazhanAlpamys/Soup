"""v0.44.0 Part D — `soup serve --reasoning-parser <name>` allowlist.

Closed-allowlist of reasoning parser names compatible with vLLM 0.6+ and
sglang. Schema-only in v0.44.0; live wiring into the inference loop deferred
to v0.44.1.
"""

from __future__ import annotations

import re
from types import MappingProxyType
from typing import Mapping, Optional

# (Closed) parser-name -> short description.
_REASONING_PARSERS: Mapping[str, str] = MappingProxyType(
    {
        "deepseek-r1": "Strip <think>...</think> blocks before final response",
        "qwen3": "Qwen 3 reasoning trace separator",
        "phi4": "Phi-4 reasoning trace separator",
        "openthinker": "OpenThinker chain-of-thought tags",
    }
)


def known_parsers() -> Mapping[str, str]:
    return _REASONING_PARSERS


def validate_parser_name(name: str) -> str:
    """Reject unknown / malformed parser names."""
    if not isinstance(name, str):
        raise TypeError("parser name must be str")
    if not name:
        raise ValueError("parser name must be non-empty")
    if "\x00" in name:
        raise ValueError("parser name contains NUL byte")
    if len(name) > 64:
        raise ValueError("parser name exceeds 64 chars")
    canonical = name.lower()
    if canonical not in _REASONING_PARSERS:
        raise ValueError(
            f"unknown reasoning parser {name!r}; "
            f"expected one of {sorted(_REASONING_PARSERS)}"
        )
    return canonical


def parser_description(name: str) -> Optional[str]:
    """Return the short description for a parser name, or None."""
    if not isinstance(name, str):
        return None
    return _REASONING_PARSERS.get(name.lower())


# v0.53.9 #98 — Per-parser regex matrix for `<think>...</think>` stripping.
# All four parsers strip the same standard `<think>...</think>` block;
# OpenThinker uses `<|begin_of_thought|>...<|end_of_thought|>` per upstream.
_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_OPENTHINKER_RE = re.compile(
    r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>", re.DOTALL
)


def strip_reasoning(text: str, parser: Optional[str]) -> str:
    """Strip reasoning-trace blocks from `text` per `parser`.

    Returns `text` unchanged when `parser` is `None`/empty, when the parser
    is unknown (defensive — caller should validate first), or when `text`
    is not a str. Idempotent: applying twice yields the same result.

    Cap: input length capped at 1 MiB; longer payloads are returned
    unchanged to avoid pathological regex backtracking.
    """
    if not isinstance(text, str):
        return text  # type: ignore[return-value]
    if not parser:
        return text
    if len(text) > 1_048_576:
        return text
    try:
        canonical = validate_parser_name(parser)
    except (TypeError, ValueError):
        return text
    # Fast pre-check: skip the regex entirely when the marker token is
    # absent. Bounds worst-case re.sub time on adversarial inputs with
    # no actual reasoning blocks.
    if canonical == "openthinker":
        if "<|begin_of_thought|>" not in text:
            return text
        out = _OPENTHINKER_RE.sub("", text)
    else:
        if "<think" not in text.lower():
            return text
        out = _THINK_RE.sub("", text)
    # Strip ONLY leading newlines left behind by the removed block. We
    # don't `lstrip()` (would silently corrupt code outputs that begin
    # with intentional spaces / tabs).
    return re.sub(r"^[\r\n]+", "", out)
