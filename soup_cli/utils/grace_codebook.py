"""v0.62.0 Part E — GRACE codebook (long-running edit).

Discrete latent-space codebook for thousands of sequential knowledge edits
that survive lifelong deployments without the norm-blowup that haunts
vanilla ROME / MEMIT. Each edit stores a (key, value) pair in a learned
codebook; at inference time the model looks up the closest codebook key
to the current residual stream and applies the stored value.

Schema-only release: ``training.grace_codebook`` opt-in + codebook
size / dim validators + ``GraceCodebookConfig`` dataclass + ``grace``
added to the v0.61.0 ``SUPPORTED_EDIT_METHODS`` allowlist. Live codebook
lookup / write / EditGovernor integration lands in v0.62.1.
"""

from __future__ import annotations

from dataclasses import dataclass

MAX_CODEBOOK_SIZE: int = 100_000
MAX_CODEBOOK_DIM: int = 16_384  # Generous upper bound matches Llama 70B hidden.


@dataclass(frozen=True)
class GraceCodebookConfig:
    """Resolved codebook configuration. Frozen post-construction."""

    size: int
    dim: int


def validate_grace_codebook_size(value: object) -> int:
    """Validate the codebook entry count.

    Bool-rejected (bool is a subclass of int), positive-int only, capped
    at :data:`MAX_CODEBOOK_SIZE` so a misconfigured run cannot allocate
    a multi-GB codebook by accident.
    """
    if isinstance(value, bool):
        raise TypeError(
            f"grace_codebook_size must not be bool, got {value!r}"
        )
    if not isinstance(value, int):
        raise TypeError(
            f"grace_codebook_size must be int, got {type(value).__name__}"
        )
    if value < 1:
        raise ValueError(
            f"grace_codebook_size must be >= 1, got {value}"
        )
    if value > MAX_CODEBOOK_SIZE:
        raise ValueError(
            f"grace_codebook_size must be <= {MAX_CODEBOOK_SIZE}, got {value}"
        )
    return value


def validate_grace_codebook_dim(value: object) -> int:
    """Validate the codebook entry dimension (residual-stream width)."""
    if isinstance(value, bool):
        raise TypeError(
            f"grace_codebook_dim must not be bool, got {value!r}"
        )
    if not isinstance(value, int):
        raise TypeError(
            f"grace_codebook_dim must be int, got {type(value).__name__}"
        )
    if value < 1:
        raise ValueError(
            f"grace_codebook_dim must be >= 1, got {value}"
        )
    if value > MAX_CODEBOOK_DIM:
        raise ValueError(
            f"grace_codebook_dim must be <= {MAX_CODEBOOK_DIM}, got {value}"
        )
    return value


def build_grace_codebook_config(*, size: int, dim: int) -> GraceCodebookConfig:
    """Validate + freeze a :class:`GraceCodebookConfig`."""
    canonical_size = validate_grace_codebook_size(size)
    canonical_dim = validate_grace_codebook_dim(dim)
    return GraceCodebookConfig(size=canonical_size, dim=canonical_dim)


def apply_grace_codebook(config: GraceCodebookConfig) -> None:
    """Apply the codebook at decode time — deferred to v0.62.1.

    Validates the config type first so callers passing a bare dict get a
    crisp ``TypeError`` rather than the deferred-live ``NotImplementedError``.
    Mirrors v0.50.0 ``apply_variant_loss`` / v0.61.0 ``apply_unlearn_loss``
    policy.
    """
    if not isinstance(config, GraceCodebookConfig):
        raise TypeError(
            f"apply_grace_codebook expects GraceCodebookConfig, "
            f"got {type(config).__name__}"
        )
    raise NotImplementedError(
        "apply_grace_codebook is deferred to v0.62.1. Schema accepts "
        f"size={config.size} dim={config.dim} today; live codebook "
        "lookup / write ships next release."
    )
