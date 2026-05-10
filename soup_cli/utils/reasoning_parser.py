"""v0.44.0 Part D — `soup serve --reasoning-parser <name>` allowlist.

Closed-allowlist of reasoning parser names compatible with vLLM 0.6+ and
sglang. Schema-only in v0.44.0; live wiring into the inference loop deferred
to v0.44.1.
"""

from __future__ import annotations

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
