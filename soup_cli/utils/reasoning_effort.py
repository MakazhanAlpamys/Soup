"""gpt-oss reasoning_effort schema helper + live prompt-prefix injector.

Schema support for ``training.reasoning_effort: Literal["low","medium","high"]``
mirroring the unsloth gpt-oss training recipe. ``apply_reasoning_effort_prefix``
injects the ``<|reasoning_effort|>...<|/reasoning_effort|>`` control tag into
the system message at training-time formatter dispatch (v0.53.2 #137).
"""

from __future__ import annotations

from typing import Mapping

REASONING_EFFORT_LEVELS: frozenset[str] = frozenset({"low", "medium", "high"})

_MAX_REASONING_EFFORT_LEN: int = 16

_REASONING_OPEN: str = "<|reasoning_effort|>"
_REASONING_CLOSE: str = "<|/reasoning_effort|>"


def validate_reasoning_effort(value: object) -> str:
    """Validate a reasoning_effort string and return the canonical form."""
    if isinstance(value, bool):
        raise TypeError(f"reasoning_effort must not be bool, got {value!r}")
    if not isinstance(value, str):
        raise TypeError(
            f"reasoning_effort must be str, got {type(value).__name__}"
        )
    if not value:
        raise ValueError("reasoning_effort must be non-empty")
    if "\x00" in value:
        raise ValueError("reasoning_effort must not contain null bytes")
    if len(value) > _MAX_REASONING_EFFORT_LEN:
        raise ValueError(
            f"reasoning_effort too long (max {_MAX_REASONING_EFFORT_LEN} chars)"
        )
    canonical = value.lower()
    if canonical not in REASONING_EFFORT_LEVELS:
        supported = ", ".join(sorted(REASONING_EFFORT_LEVELS))
        raise ValueError(
            f"reasoning_effort {value!r} not supported. Supported: {supported}"
        )
    return canonical


def apply_reasoning_effort_prefix(
    messages: object,
    level: object,
) -> list[dict]:
    """Inject the gpt-oss reasoning-effort control tag into the system message.

    Returns a NEW list (input is not mutated; mirrors v0.33.0 #47 immutability
    policy). The canonical tag is
    ``<|reasoning_effort|>{level}<|/reasoning_effort|>`` and is prepended to
    the FIRST system message in ``messages``; if no system message is present
    one is inserted at index 0.

    Args:
        messages: list of ``{"role": ..., "content": ...}`` dicts.
        level: ``"low"`` / ``"medium"`` / ``"high"`` (case-insensitive).

    Raises:
        TypeError: ``messages`` not a list, message entry not dict, ``level``
            not a string (bool rejected explicitly).
        ValueError: ``messages`` empty, level outside the allowlist, or level
            contains null bytes.
    """
    canonical_level = validate_reasoning_effort(level)
    if not isinstance(messages, list):
        raise TypeError(
            f"messages must be a list, got {type(messages).__name__}"
        )
    if len(messages) == 0:
        raise ValueError("messages must not be empty")

    out: list[dict] = []
    injected = False
    tag = f"{_REASONING_OPEN}{canonical_level}{_REASONING_CLOSE}"
    for msg in messages:
        if not isinstance(msg, Mapping):
            raise TypeError(
                f"each message must be dict-like, got {type(msg).__name__}"
            )
        if not injected and msg.get("role") == "system":
            copy = dict(msg)
            existing = copy.get("content", "")
            if not isinstance(existing, str):
                existing = str(existing)
            copy["content"] = f"{tag}\n{existing}" if existing else tag
            out.append(copy)
            injected = True
        else:
            out.append(dict(msg))

    if not injected:
        out.insert(0, {"role": "system", "content": tag})
    return out
