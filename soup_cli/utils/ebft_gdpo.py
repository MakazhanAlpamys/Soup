"""v0.52.0 Part E — Energy-Based FT (EBFT) + Generalized DPO (GDPO) helpers.

Schema-only release: each algorithm has a closed allowlist of variant names
plus pure validators. Live loss kernels land in v0.52.1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

# Closed allowlists.
EBFT_VARIANTS: frozenset[str] = frozenset({"structured", "strided"})
GDPO_VARIANTS: frozenset[str] = frozenset({"standard", "length_normalized", "margin"})

_MAX_VARIANT_LEN: int = 32

_MIN_EBFT_TEMP: float = 1e-4
_MAX_EBFT_TEMP: float = 100.0


@dataclass(frozen=True)
class EBFTSpec:
    """Metadata for an EBFT variant. Frozen — immutable."""

    name: str
    description: str
    live_wired: bool


_EBFT_METADATA: Mapping[str, EBFTSpec] = MappingProxyType({
    "structured": EBFTSpec(
        name="structured",
        description="Structured Energy-Based FT (per-token energies)",
        live_wired=False,
    ),
    "strided": EBFTSpec(
        name="strided",
        description="Strided Energy-Based FT (block-sampled energies)",
        live_wired=False,
    ),
})


@dataclass(frozen=True)
class GDPOSpec:
    """Metadata for a GDPO variant. Frozen — immutable."""

    name: str
    description: str
    live_wired: bool


_GDPO_METADATA: Mapping[str, GDPOSpec] = MappingProxyType({
    "standard": GDPOSpec(
        name="standard",
        description="Standard GDPO (general preference objective)",
        live_wired=False,
    ),
    "length_normalized": GDPOSpec(
        name="length_normalized",
        description="Length-normalized GDPO (SimPO-style normalisation)",
        live_wired=False,
    ),
    "margin": GDPOSpec(
        name="margin",
        description="Margin-augmented GDPO (DPO + margin term)",
        live_wired=False,
    ),
})


def _validate_variant(name: object, allowed: frozenset[str], label: str) -> str:
    """Shared variant-name validator."""
    if isinstance(name, bool):
        raise TypeError(f"{label} must not be bool, got {name!r}")
    if not isinstance(name, str):
        raise TypeError(f"{label} must be str, got {type(name).__name__}")
    if not name:
        raise ValueError(f"{label} must be non-empty")
    if "\x00" in name:
        raise ValueError(f"{label} must not contain null bytes")
    if len(name) > _MAX_VARIANT_LEN:
        raise ValueError(
            f"{label} too long (max {_MAX_VARIANT_LEN} chars)"
        )
    canonical = name.lower()
    if canonical not in allowed:
        supported = ", ".join(sorted(allowed))
        raise ValueError(
            f"{label} {name!r} not supported. Supported: {supported}"
        )
    return canonical


def validate_ebft_variant(name: object) -> str:
    """Validate an EBFT variant and return the canonical form."""
    return _validate_variant(name, EBFT_VARIANTS, "ebft_variant")


def validate_gdpo_variant(name: object) -> str:
    """Validate a GDPO variant and return the canonical form."""
    return _validate_variant(name, GDPO_VARIANTS, "gdpo_variant")


def get_ebft_spec(name: str) -> EBFTSpec:
    """Return the frozen :class:`EBFTSpec` for ``name`` or raise."""
    return _EBFT_METADATA[validate_ebft_variant(name)]


def get_gdpo_spec(name: str) -> GDPOSpec:
    """Return the frozen :class:`GDPOSpec` for ``name`` or raise."""
    return _GDPO_METADATA[validate_gdpo_variant(name)]


def validate_ebft_temperature(value: object) -> float:
    """Validate an EBFT temperature scalar in [1e-4, 100]. Rejects bool/NaN."""
    if isinstance(value, bool):
        raise TypeError(f"ebft_temperature must not be bool, got {value!r}")
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"ebft_temperature must be float, got {type(value).__name__}"
        )
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError(
            f"ebft_temperature must be finite, got {value!r}"
        )
    if fval < _MIN_EBFT_TEMP:
        raise ValueError(
            f"ebft_temperature must be >= {_MIN_EBFT_TEMP}, got {fval}"
        )
    if fval > _MAX_EBFT_TEMP:
        raise ValueError(
            f"ebft_temperature must be <= {_MAX_EBFT_TEMP}, got {fval}"
        )
    return fval


def _check_task_backend(task: object, backend: object) -> None:
    """Shared bool/str guard for cross-compat helpers."""
    for name, value in (("task", task), ("backend", backend)):
        if isinstance(value, bool):
            raise TypeError(f"{name} must not be bool, got {value!r}")
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")


def validate_ebft_compat(*, task: str, backend: str) -> None:
    """Schema-time gate for ``ebft_variant`` — SFT-only, non-MLX."""
    _check_task_backend(task, backend)
    if backend == "mlx":
        raise ValueError(
            "ebft_variant is not supported on backend=mlx in v0.52.0"
        )
    if task != "sft":
        raise ValueError(
            f"ebft_variant requires task='sft'; got task={task!r}"
        )


def validate_gdpo_compat(*, task: str, backend: str) -> None:
    """Schema-time gate for ``gdpo_variant`` — DPO-family-only, non-MLX."""
    _check_task_backend(task, backend)
    if backend == "mlx":
        raise ValueError(
            "gdpo_variant is not supported on backend=mlx in v0.52.0"
        )
    if task not in ("dpo", "preference"):
        raise ValueError(
            f"gdpo_variant requires task in ('dpo', 'preference'); "
            f"got task={task!r}"
        )


def apply_ebft_loss() -> None:
    """Live EBFT loss kernel — deferred to v0.52.1."""
    raise NotImplementedError(
        "EBFT (Energy-Based FT) live loss kernel deferred to v0.52.1. "
        "Schema accepts the variant but no loss is wired yet."
    )


def apply_gdpo_loss() -> None:
    """Live GDPO loss kernel — deferred to v0.52.1."""
    raise NotImplementedError(
        "GDPO (Generalized DPO) live loss kernel deferred to v0.52.1. "
        "Schema accepts the variant but no loss is wired yet."
    )
