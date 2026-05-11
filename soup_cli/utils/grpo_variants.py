"""GRPO objective variants — v0.50.0 Part A.

Closed allowlist of GRPO-family RL objectives shipped for parity with
unsloth + axolotl. Schema-load-time validation + metadata only in v0.50.0;
live loss-function wiring lands in v0.50.1 (mirrors v0.27.0 MII /
v0.37.0 multipack / v0.41.0 LLaMA Pro / v0.45.0 plugins / v0.48.0 curriculum
stub-then-live pattern).

Variants:
- gspo         : Group Stabilized Policy Optimization (unsloth)
- dapo         : Decoupled Advantage Policy Optimization (unsloth, axolotl)
- dr_grpo      : Doubly Robust GRPO (unsloth, axolotl)
- bnpo         : Batch Normalized Policy Optimization (unsloth)
- two_sided    : Symmetric clipping w/ delta (unsloth)
- rft          : Reinforced Fine-Tuning (unsloth)
- standard     : Default GRPO (DeepSeek-R1 style — legacy alias)

Security:
- Closed allowlist; arbitrary string at schema level rejected.
- Empty / null-byte / non-string / oversize rejected.
- `_VARIANT_METADATA` wrapped in MappingProxyType (matches v0.36.0
  _REGISTRY / v0.41.0 _OPTIMIZER_PACKAGES policy).
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass

_MAX_VARIANT_NAME_LEN = 32

SUPPORTED_GRPO_VARIANTS: frozenset[str] = frozenset({
    "gspo",
    "dapo",
    "dr_grpo",
    "bnpo",
    "two_sided",
    "rft",
    "standard",
})

# Variants that require an explicit delta (symmetric clipping radius).
_REQUIRES_DELTA: frozenset[str] = frozenset({"two_sided"})

# Variants whose loss kernel live-wiring is deferred to v0.50.1.
_DEFERRED_LIVE: frozenset[str] = frozenset({
    "gspo", "dapo", "dr_grpo", "bnpo", "two_sided", "rft",
})


@dataclass(frozen=True)
class GRPOVariantSpec:
    """Metadata for a GRPO objective variant."""

    name: str
    description: str
    requires_delta: bool
    live_wired: bool


_VARIANT_METADATA = types.MappingProxyType({
    "standard": GRPOVariantSpec(
        name="standard",
        description="Default GRPO (DeepSeek-R1 style)",
        requires_delta=False,
        live_wired=True,
    ),
    "gspo": GRPOVariantSpec(
        name="gspo",
        description="Group Stabilized Policy Optimization",
        requires_delta=False,
        live_wired=False,
    ),
    "dapo": GRPOVariantSpec(
        name="dapo",
        description="Decoupled Advantage Policy Optimization",
        requires_delta=False,
        live_wired=False,
    ),
    "dr_grpo": GRPOVariantSpec(
        name="dr_grpo",
        description="Doubly Robust GRPO",
        requires_delta=False,
        live_wired=False,
    ),
    "bnpo": GRPOVariantSpec(
        name="bnpo",
        description="Batch Normalized Policy Optimization",
        requires_delta=False,
        live_wired=False,
    ),
    "two_sided": GRPOVariantSpec(
        name="two_sided",
        description="Two-sided GRPO with symmetric delta clipping",
        requires_delta=True,
        live_wired=False,
    ),
    "rft": GRPOVariantSpec(
        name="rft",
        description="Reinforced Fine-Tuning",
        requires_delta=False,
        live_wired=False,
    ),
})


def validate_grpo_variant(name: object) -> str:
    """Validate and normalise a GRPO variant name.

    Returns the canonical (lower-cased) name on success. Raises
    ``ValueError`` with an actionable message on any failure.
    """
    if isinstance(name, bool):
        raise ValueError("grpo_variant must be a string, got bool")
    if not isinstance(name, str):
        raise ValueError(
            f"grpo_variant must be a string, got {type(name).__name__}"
        )
    if not name:
        raise ValueError("grpo_variant must be a non-empty string")
    if "\x00" in name:
        raise ValueError("grpo_variant must not contain null bytes")
    if len(name) > _MAX_VARIANT_NAME_LEN:
        raise ValueError(
            f"grpo_variant exceeds {_MAX_VARIANT_NAME_LEN} chars"
        )
    normalised = name.lower()
    if normalised not in SUPPORTED_GRPO_VARIANTS:
        raise ValueError(
            f"grpo_variant={name!r} is not supported. "
            f"Valid: {sorted(SUPPORTED_GRPO_VARIANTS)}"
        )
    return normalised


def get_variant_spec(name: str) -> GRPOVariantSpec:
    """Return the :class:`GRPOVariantSpec` for ``name``.

    Raises ``KeyError`` if ``name`` is not in the allowlist.
    """
    normalised = validate_grpo_variant(name)
    return _VARIANT_METADATA[normalised]


def variant_requires_delta(name: str) -> bool:
    """Return True if the variant requires ``grpo_delta`` to be set."""
    if not isinstance(name, str):
        return False
    return name.lower() in _REQUIRES_DELTA


def variant_is_live_wired(name: str) -> bool:
    """Return True if the variant has a live loss kernel in this release.

    All v0.50.0 additions return False (deferred to v0.50.1); only
    ``standard`` returns True.
    """
    if not isinstance(name, str):
        return False
    return name.lower() not in _DEFERRED_LIVE


def validate_grpo_delta(value: object) -> float:
    """Validate the two-sided clipping delta.

    Must be a finite float in ``(0, 1]``. Bool rejected (matches v0.30.0
    Candidate policy).
    """
    if isinstance(value, bool):
        raise ValueError("grpo_delta must not be bool")
    if not isinstance(value, (int, float)):
        raise ValueError(
            f"grpo_delta must be a number, got {type(value).__name__}"
        )
    fvalue = float(value)
    if not math.isfinite(fvalue):
        raise ValueError("grpo_delta must be finite (no NaN/Inf)")
    if not (0.0 < fvalue <= 1.0):
        raise ValueError(
            f"grpo_delta={fvalue} must be in (0, 1]"
        )
    return fvalue


def apply_variant_loss(name: str) -> None:
    """Live loss kernel for v0.50.0 variants — deferred to v0.50.1.

    Planned v0.50.1 signature:
    ``apply_variant_loss(name, *, model, batch, advantages, beta, delta=None)``.

    Raises ``NotImplementedError`` for any variant in ``_DEFERRED_LIVE``.
    Mirrors v0.27.0 MII / v0.37.0 multipack / v0.41.0 LLaMA Pro /
    v0.45.0 plugins / v0.48.0 curriculum stub-then-live pattern.
    """
    normalised = validate_grpo_variant(name)
    if normalised in _DEFERRED_LIVE:
        raise NotImplementedError(
            f"grpo_variant={normalised!r} live loss kernel deferred to "
            f"v0.50.1. Schema accepts the value but the actual loss math "
            f"is not yet wired. Track via the v0.50.0 known-limitations "
            f"GitHub issue."
        )
    # standard variant — falls through to the existing GRPOTrainerWrapper.


def list_variants() -> tuple[str, ...]:
    """Return a sorted tuple of supported variant names (for CLI help)."""
    return tuple(sorted(SUPPORTED_GRPO_VARIANTS))
