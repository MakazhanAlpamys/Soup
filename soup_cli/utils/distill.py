"""v0.52.0 Part C — Knowledge Distillation schema helpers.

Schema-only support for ``task='distill'`` — teacher/student training.
Four divergence options are recognised, mirroring axolotl's distillation
plugin:

* ``kl`` (forward KL — student KL teacher, standard distillation)
* ``forward_kl`` (alias for ``kl``)
* ``reverse_kl`` (teacher KL student)
* ``js`` (Jensen-Shannon, symmetric)

The live distillation trainer lands in v0.52.1; this module exposes pure
validators so the schema gate can fail fast on misconfiguration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

_DIVERGENCE_ALIASES: Mapping[str, str] = MappingProxyType({
    "kl": "forward_kl",
    "forward_kl": "forward_kl",
    "reverse_kl": "reverse_kl",
    "js": "js",
})

# Public, derived from the alias map so adding a new alias updates both the
# accepted-input set and the error message in lockstep.
DIVERGENCES: frozenset[str] = frozenset(_DIVERGENCE_ALIASES)

_MAX_TEACHER_LEN: int = 512
_MAX_DIVERGENCE_LEN: int = 16
_MIN_TEMPERATURE: float = 0.05
_MAX_TEMPERATURE: float = 100.0


@dataclass(frozen=True)
class DivergenceSpec:
    """Metadata for a divergence kernel. Frozen so callers cannot mutate."""

    name: str
    description: str
    symmetric: bool
    live_wired: bool


_DIVERGENCE_METADATA: Mapping[str, DivergenceSpec] = MappingProxyType({
    "forward_kl": DivergenceSpec(
        name="forward_kl",
        description="Forward KL (standard distillation)",
        symmetric=False,
        live_wired=False,
    ),
    "reverse_kl": DivergenceSpec(
        name="reverse_kl",
        description="Reverse KL (mode-seeking)",
        symmetric=False,
        live_wired=False,
    ),
    "js": DivergenceSpec(
        name="js",
        description="Jensen-Shannon (symmetric KL)",
        symmetric=True,
        live_wired=False,
    ),
})


def validate_divergence(name: object) -> str:
    """Validate a divergence name and return the canonical form.

    Accepts ``kl`` as an alias for ``forward_kl``. Mirrors v0.41.0
    ``validate_optimizer_name`` policy.
    """
    if isinstance(name, bool):
        raise TypeError(f"distill_divergence must not be bool, got {name!r}")
    if not isinstance(name, str):
        raise TypeError(
            f"distill_divergence must be str, got {type(name).__name__}"
        )
    if not name:
        raise ValueError("distill_divergence must be non-empty")
    if "\x00" in name:
        raise ValueError("distill_divergence must not contain null bytes")
    if len(name) > _MAX_DIVERGENCE_LEN:
        raise ValueError(
            f"distill_divergence too long (max {_MAX_DIVERGENCE_LEN} chars)"
        )
    canonical = name.lower()
    if canonical not in _DIVERGENCE_ALIASES:
        supported = ", ".join(sorted(DIVERGENCES))
        raise ValueError(
            f"distill_divergence {name!r} not supported. Supported: {supported}"
        )
    return _DIVERGENCE_ALIASES[canonical]


def get_divergence_spec(name: str) -> DivergenceSpec:
    """Return the frozen :class:`DivergenceSpec` for ``name`` or raise."""
    canonical = validate_divergence(name)
    return _DIVERGENCE_METADATA[canonical]


def validate_distill_temperature(value: object) -> float:
    """Validate a distillation temperature scalar.

    Bounds [0.05, 100.0]. Rejects bool, NaN, ±inf.
    """
    if isinstance(value, bool):
        raise TypeError(
            f"distill_temperature must not be bool, got {value!r}"
        )
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"distill_temperature must be float, got {type(value).__name__}"
        )
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError(
            f"distill_temperature must be finite, got {value!r}"
        )
    if fval < _MIN_TEMPERATURE:
        raise ValueError(
            f"distill_temperature must be >= {_MIN_TEMPERATURE}, got {fval}"
        )
    if fval > _MAX_TEMPERATURE:
        raise ValueError(
            f"distill_temperature must be <= {_MAX_TEMPERATURE}, got {fval}"
        )
    return fval


def validate_teacher_model(value: object) -> str:
    """Validate a teacher model string (HF repo id or local path).

    Mirrors the v0.40.5 ``reward_model`` field validator: null-byte
    rejection + 512-char cap.
    """
    if isinstance(value, bool):
        raise TypeError(f"teacher_model must not be bool, got {value!r}")
    if not isinstance(value, str):
        raise TypeError(
            f"teacher_model must be str, got {type(value).__name__}"
        )
    if not value:
        raise ValueError("teacher_model must be non-empty")
    if "\x00" in value:
        raise ValueError("teacher_model must not contain null bytes")
    if len(value) > _MAX_TEACHER_LEN:
        raise ValueError(
            f"teacher_model too long (max {_MAX_TEACHER_LEN} chars)"
        )
    return value


def validate_distill_compat(
    *,
    task: str,
    backend: str,
    teacher_model: object,
) -> None:
    """Schema-time gate for ``task='distill'``.

    Rejects:
    - non-distill task.
    - ``backend == 'mlx'`` (no MLX teacher-load path yet).
    - missing teacher_model — distillation is meaningless without one.
    """
    for name, value in (("task", task), ("backend", backend)):
        if isinstance(value, bool):
            raise TypeError(f"{name} must not be bool, got {value!r}")
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    if task != "distill":
        raise ValueError(
            f"validate_distill_compat called with task={task!r} "
            "(expected 'distill')"
        )
    if backend == "mlx":
        raise ValueError(
            "task='distill' is not supported on backend=mlx in v0.52.0"
        )
    if teacher_model is None:
        raise ValueError(
            "task='distill' requires training.teacher_model to be set"
        )
    # Reuse the standard validator — null-byte / oversize / type check.
    validate_teacher_model(teacher_model)


def build_distill_trainer() -> None:
    """Live distillation trainer factory — deferred to v0.52.1."""
    raise NotImplementedError(
        "Distillation trainer (task='distill') live wiring deferred to "
        "v0.52.1. Schema accepts the value but no trainer wrapper is "
        "registered yet."
    )
