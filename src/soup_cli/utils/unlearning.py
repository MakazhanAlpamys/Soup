"""v0.61.0 Part A — Unlearning method allowlist + compat gate.

Three method backends for the new ``task='unlearn'`` trainer:

* ``npo`` — Negative Preference Optimization (Zhang et al., 2024). Pushes
  the model away from the forget set via a DPO-shaped negative-only loss.
* ``simnpo`` — SimNPO (length-normalised NPO). Removes the reference model
  and uses a SimPO-style normalisation to stabilise long-sequence
  unlearning.
* ``rmu`` — Representation Misdirection Unlearning (Li et al., 2024).
  Adds a noise vector to the residual stream for forget-set inputs while
  preserving retain-set activations.

Schema-only release: validators here are reused by the SoupConfig
cross-validator, while the live trainer wrapper ships in v0.61.1
(mirrors v0.50.0 GRPO Plus / v0.52.0 Modality II / v0.53.0 Quant Menu II
stub-then-live pattern).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

SUPPORTED_UNLEARN_METHODS: frozenset[str] = frozenset({"npo", "simnpo", "rmu"})

_MAX_METHOD_LEN: int = 32


@dataclass(frozen=True)
class UnlearnMethodSpec:
    """Metadata for an unlearning method backend. Frozen so callers cannot mutate."""

    name: str
    description: str
    needs_retain_set: bool
    needs_reference_model: bool
    live_wired: bool


_UNLEARN_METHOD_METADATA: Mapping[str, UnlearnMethodSpec] = MappingProxyType({
    "npo": UnlearnMethodSpec(
        name="npo",
        description=(
            "Negative Preference Optimization — DPO-shaped loss that "
            "pushes the model away from the forget set while a retain "
            "set keeps general capability stable."
        ),
        needs_retain_set=True,
        needs_reference_model=True,
        live_wired=False,
    ),
    "simnpo": UnlearnMethodSpec(
        name="simnpo",
        description=(
            "SimNPO — length-normalised NPO without a reference model. "
            "Faster + more stable on long sequences (Liu et al., 2024)."
        ),
        needs_retain_set=True,
        needs_reference_model=False,
        live_wired=False,
    ),
    "rmu": UnlearnMethodSpec(
        name="rmu",
        description=(
            "Representation Misdirection Unlearning — adds a noise "
            "vector to the residual stream for forget inputs while "
            "preserving retain activations (Li et al., 2024)."
        ),
        needs_retain_set=True,
        needs_reference_model=False,
        live_wired=False,
    ),
})


def validate_unlearn_method(value: object) -> str:
    """Normalise + validate an unlearn-method name.

    Returns the canonical (lowercase) form. Mirrors v0.41.0
    ``validate_optimizer_name`` / v0.51.0 ``validate_hub_name`` policy:
    bool-rejected, null-byte-rejected, oversize-rejected,
    case-insensitive normalisation, unknown rejected with friendly
    actionable message.
    """
    if isinstance(value, bool):
        raise TypeError(
            f"unlearn_method must not be bool, got {value!r}"
        )
    if not isinstance(value, str):
        raise TypeError(
            f"unlearn_method must be str, got {type(value).__name__}"
        )
    if not value:
        raise ValueError("unlearn_method must be non-empty")
    if "\x00" in value:
        raise ValueError("unlearn_method must not contain null bytes")
    if len(value) > _MAX_METHOD_LEN:
        raise ValueError(
            f"unlearn_method must be <= {_MAX_METHOD_LEN} chars"
        )
    canonical = value.lower()
    if canonical not in SUPPORTED_UNLEARN_METHODS:
        supported = ", ".join(sorted(SUPPORTED_UNLEARN_METHODS))
        raise ValueError(
            f"unknown unlearn method {value!r}; supported: {supported}"
        )
    return canonical


def get_unlearn_method_spec(name: str) -> UnlearnMethodSpec:
    """Return the frozen :class:`UnlearnMethodSpec` for ``name`` or raise."""
    canonical = validate_unlearn_method(name)
    return _UNLEARN_METHOD_METADATA[canonical]


def validate_unlearn_alpha(value: object) -> float:
    """Validate an ``unlearn_alpha`` retain-set weight (forget vs retain mixing).

    Bool-rejected, NaN/Inf-rejected, bounded ``[0.0, 10.0]``. Default
    interpretation: 0 = pure forget loss, higher values increasingly
    favour the retain set. The upper bound is a sanity cap — operators
    above 10 should rebalance their dataset instead.
    """
    if isinstance(value, bool):
        raise TypeError(
            f"unlearn_alpha must not be bool, got {value!r}"
        )
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"unlearn_alpha must be a number, got {type(value).__name__}"
        )
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError("unlearn_alpha must be finite (no NaN / Inf)")
    if fval < 0.0:
        raise ValueError(f"unlearn_alpha must be >= 0.0, got {fval}")
    if fval > 10.0:
        raise ValueError(f"unlearn_alpha must be <= 10.0, got {fval}")
    return fval


def validate_unlearn_compat(*, task: str, backend: str) -> None:
    """Schema-time gate for ``task='unlearn'``.

    Rejects:
    - non-string / bool args (defence-in-depth, mirrors v0.52.0
      ``validate_classifier_compat`` policy).
    - non-unlearn task.
    - ``backend == 'mlx'`` (no MLX unlearn path in v0.61.0).

    Multi-method specifics (forget_set required, retain_set conditional)
    are enforced at the SoupConfig cross-validator level so the schema
    can surface a single composite error.
    """
    for name, value in (("task", task), ("backend", backend)):
        if isinstance(value, bool):
            raise TypeError(f"{name} must not be bool, got {value!r}")
        if not isinstance(value, str):
            raise TypeError(
                f"{name} must be str, got {type(value).__name__}"
            )
        if not value:
            raise ValueError(f"{name} must be non-empty")
        if "\x00" in value:
            raise ValueError(f"{name} must not contain null bytes")
    if task != "unlearn":
        raise ValueError(
            f"validate_unlearn_compat called with task={task!r}; "
            f"expected task='unlearn'"
        )
    if backend == "mlx":
        raise ValueError(
            "task='unlearn' is not supported on backend=mlx in v0.61.0 "
            "(deferred to v0.61.1)"
        )


def apply_unlearn_loss(method: str):
    """Return the live per-method unlearn loss kernel (v0.71.9 #193).

    Validates the method name first (unknown → ValueError) then returns the
    callable from :mod:`soup_cli.utils.unlearn_kernels`:

    * ``npo``    → ``npo_loss(policy_logps, ref_logps, *, beta)``
    * ``simnpo`` → ``simnpo_loss(policy_logps, lengths, *, beta, gamma)``
    * ``rmu``    → ``rmu_loss(forget_acts, control_vec, retain_acts,
      retain_frozen, *, alpha)``
    """
    canonical = validate_unlearn_method(method)
    from soup_cli.utils import unlearn_kernels

    return {
        "npo": unlearn_kernels.npo_loss,
        "simnpo": unlearn_kernels.simnpo_loss,
        "rmu": unlearn_kernels.rmu_loss,
    }[canonical]


def build_unlearn_trainer(config: object, **kwargs: object) -> object:
    """Live unlearn trainer factory — deferred stub.

    Lazy-imports the wrapper so the schema-only import path never pulls
    in the heavy ``transformers`` / ``peft`` surface. The wrapper itself
    is a stub that raises ``NotImplementedError`` on ``setup()``.
    """
    from soup_cli.trainer.unlearn import UnlearnTrainerWrapper

    return UnlearnTrainerWrapper(config, **kwargs)  # type: ignore[arg-type]
