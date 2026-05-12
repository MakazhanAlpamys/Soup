"""v0.52.0 Part G — gpt-oss reasoning_effort schema helper.

Schema-only support for ``training.reasoning_effort: Literal["low","medium","high"]``
mirroring the unsloth gpt-oss training recipe. Routes through the prompt
prefix at training time; live formatter wiring lands in v0.52.1.
"""

from __future__ import annotations

REASONING_EFFORT_LEVELS: frozenset[str] = frozenset({"low", "medium", "high"})

_MAX_REASONING_EFFORT_LEN: int = 16


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
