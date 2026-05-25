"""v0.69.0 Part C — `soup data gen magpie` synthetic generator.

The Magpie technique (Xu et al. 2024) feeds an aligned chat-tuned model just
the chat-template prefix (system + user-turn header tokens) and lets the model
generate the user turn itself. The harvested user turns are then passed back to
the same model in a normal completion call to harvest the assistant response.
This is a clever way to produce high-volume SFT data without paying for human
prompts.

This module ships the schema + validators + dry-run planner. Live generation
(invoking v0.20.0 Ollama / Anthropic / vLLM providers + running the quality
filter from v0.47.0 educational + toxicity scorers) is deferred to v0.69.1
under the project-wide stub-then-live cadence (mirrors v0.50.0 / v0.61.0 /
v0.62.0 / v0.68.0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Closed allowlist of providers — reuses v0.20.0 synth-data backends.
_SUPPORTED_MAGPIE_PROVIDERS = ("ollama", "anthropic", "vllm")
SUPPORTED_MAGPIE_PROVIDERS: frozenset = frozenset(_SUPPORTED_MAGPIE_PROVIDERS)

_MAX_BASE_MODEL_LEN = 512
_MAX_TARGET_ROWS = 1_000_000
_MAX_PROVIDER_LEN = 32


@dataclass(frozen=True)
class MagpieConfig:
    """Frozen plan for one ``soup data gen-magpie`` invocation."""

    base_model: str
    provider: str
    target_rows: int
    quality_filter: bool

    def __post_init__(self) -> None:
        # Re-validate so direct construction can't smuggle bad fields past
        # the validators (mirrors v0.67.0 vector_bank policy).
        validate_base_model(self.base_model)
        validate_magpie_provider(self.provider)
        validate_target_rows(self.target_rows)
        if not isinstance(self.quality_filter, bool):
            raise TypeError(
                "MagpieConfig.quality_filter must be bool, "
                f"got {type(self.quality_filter).__name__}"
            )


# -----------------------------------------------------------------------------
# Validators
# -----------------------------------------------------------------------------


def validate_magpie_provider(provider: object) -> str:
    """Return the canonical lower-case provider id."""
    if isinstance(provider, bool) or not isinstance(provider, str):
        raise TypeError(
            f"magpie provider must be str, got {type(provider).__name__}"
        )
    if not provider:
        raise ValueError("magpie provider must be non-empty")
    if "\x00" in provider:
        raise ValueError("magpie provider must not contain null bytes")
    if len(provider) > _MAX_PROVIDER_LEN:
        raise ValueError(
            f"magpie provider must be <= {_MAX_PROVIDER_LEN} chars"
        )
    canonical = provider.strip().lower()
    if canonical not in SUPPORTED_MAGPIE_PROVIDERS:
        raise ValueError(
            f"unknown magpie provider: {provider!r}. "
            f"supported: {sorted(SUPPORTED_MAGPIE_PROVIDERS)}"
        )
    return canonical


def validate_target_rows(target: object) -> int:
    """Validate ``target_rows`` ∈ [1, 1_000_000]; bool-rejected."""
    if isinstance(target, bool):
        raise TypeError("target_rows must be int, not bool")
    if not isinstance(target, int):
        raise TypeError("target_rows must be an integer")
    if target < 1:
        raise ValueError("target_rows must be >= 1")
    if target > _MAX_TARGET_ROWS:
        raise ValueError(f"target_rows must be <= {_MAX_TARGET_ROWS}")
    return target


def validate_base_model(name: object) -> str:
    """Validate the base-model identifier."""
    if isinstance(name, bool) or not isinstance(name, str):
        raise TypeError(
            f"base_model must be str, got {type(name).__name__}"
        )
    if not name:
        raise ValueError("base_model must be non-empty")
    if "\x00" in name:
        raise ValueError("base_model must not contain null bytes")
    if len(name) > _MAX_BASE_MODEL_LEN:
        raise ValueError(
            f"base_model must be <= {_MAX_BASE_MODEL_LEN} chars"
        )
    return name


# -----------------------------------------------------------------------------
# Plan builder
# -----------------------------------------------------------------------------


def build_magpie_config(
    *,
    base: str,
    provider: str,
    target: int,
    quality_filter: bool = True,
) -> MagpieConfig:
    """Convenience factory — every input passes through the validators."""
    if not isinstance(quality_filter, bool):
        raise TypeError(
            f"quality_filter must be bool, got {type(quality_filter).__name__}"
        )
    return MagpieConfig(
        base_model=validate_base_model(base),
        provider=validate_magpie_provider(provider),
        target_rows=validate_target_rows(target),
        quality_filter=quality_filter,
    )


# -----------------------------------------------------------------------------
# Live runner — deferred to v0.69.1
# -----------------------------------------------------------------------------


def run_magpie(config: Any) -> None:
    """Execute the Magpie generation loop. Deferred to v0.69.1.

    The CLI catches ``NotImplementedError`` and exits with code 3 (distinct
    from validation rejection exit 2) — same policy as v0.61.0 / v0.62.0 /
    v0.68.0 deferred-live stubs.
    """
    if not isinstance(config, MagpieConfig):
        raise TypeError("config must be a MagpieConfig")
    raise NotImplementedError(
        "soup data gen-magpie live runner is deferred to v0.69.1 — "
        "only --plan-only is wired today."
    )


__all__ = [
    "MagpieConfig",
    "SUPPORTED_MAGPIE_PROVIDERS",
    "build_magpie_config",
    "run_magpie",
    "validate_base_model",
    "validate_magpie_provider",
    "validate_target_rows",
]
