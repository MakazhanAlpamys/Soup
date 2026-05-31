"""v0.45.0 Part B — n-gram-mod speculative decoding (schema-only).

Validates a small configuration surface; the live engine wiring inside
the inference loop lands in v0.45.1.
"""

from __future__ import annotations

from dataclasses import dataclass

_MIN_N = 1
_MAX_N = 8
_MIN_DRAFT = 1
_MAX_DRAFT = 32
_MIN_PROMPT = 0
_MAX_PROMPT = 1_048_576


@dataclass(frozen=True)
class NgramSpecConfig:
    """N-gram speculative-decoding configuration."""

    n: int
    num_draft_tokens: int = 4
    prompt_lookup_max: int = 0  # 0 disables prompt-lookup heuristic


def validate_ngram_n(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an int")
    if n < _MIN_N or n > _MAX_N:
        raise ValueError(f"n must be in [{_MIN_N}, {_MAX_N}]")
    return n


def validate_num_draft_tokens(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("num_draft_tokens must be an int")
    if value < _MIN_DRAFT or value > _MAX_DRAFT:
        raise ValueError(
            f"num_draft_tokens must be in [{_MIN_DRAFT}, {_MAX_DRAFT}]"
        )
    return value


def validate_prompt_lookup_max(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("prompt_lookup_max must be an int")
    if value < _MIN_PROMPT or value > _MAX_PROMPT:
        raise ValueError(
            f"prompt_lookup_max must be in [{_MIN_PROMPT}, {_MAX_PROMPT}]"
        )
    return value


def validate_ngram_config(config: NgramSpecConfig) -> NgramSpecConfig:
    if not isinstance(config, NgramSpecConfig):
        raise TypeError("config must be an NgramSpecConfig")
    validate_ngram_n(config.n)
    validate_num_draft_tokens(config.num_draft_tokens)
    validate_prompt_lookup_max(config.prompt_lookup_max)
    return config


__all__ = [
    "NgramSpecConfig",
    "validate_ngram_config",
    "validate_ngram_n",
    "validate_num_draft_tokens",
    "validate_prompt_lookup_max",
]
