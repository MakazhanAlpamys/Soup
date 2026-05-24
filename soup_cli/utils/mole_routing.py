"""MoLE per-token adapter routing (v0.67.0 Part C).

A gating network that routes per-token activations to one of N task
LoRAs. Following the MoLE paper (Mixture of LoRA Experts), inference-
time dispatch uses softmax gating over the per-token hidden state to
select top-K LoRAs and blend them by gating logits.

v0.67.0 ships the schema + cross-validator. The live gating-kernel
training + serving-time dispatch is deferred to v0.67.1 (mirrors
v0.27.0 MII / v0.50.0 GRPO Plus / v0.62.0 steering stub-then-live).

Public surface:

- ``MoleGatingConfig`` frozen dataclass
- ``validate_mole_compat(task, backend, num_task_adapters)``
- ``build_gating_kernel(config)`` deferred-live stub
- New ``task='moe_lora_routing'`` Literal on ``SoupConfig.task``

Design notes:

- ``num_task_adapters`` bounded ``[2, 64]`` — beyond that, per-token
  softmax becomes a bottleneck; operators wanting more should hierarchy
  the gating.
- ``top_k <= num_task_adapters`` so sparse top-K dispatch is sane.
- ``temperature > 0`` to keep softmax non-degenerate; finite-only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Bounds (closed, locked at module load)
# ---------------------------------------------------------------------------

MIN_TASK_ADAPTERS = 2
MAX_TASK_ADAPTERS = 64
MIN_HIDDEN_DIM = 1
MAX_HIDDEN_DIM = 16_384
MIN_TEMPERATURE = 1e-6
MAX_TEMPERATURE = 100.0


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def _check_int(value: object, field: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be int")
    if value < lo:
        raise ValueError(f"{field} {value} below floor {lo}")
    if value > hi:
        raise ValueError(f"{field} {value} above cap {hi}")
    return value


def _check_finite_positive(
    value: object, field: str, lo: float, hi: float
) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must not be bool")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"{field} must be finite")
    if val < lo:
        raise ValueError(f"{field} {val} below floor {lo}")
    if val > hi:
        raise ValueError(f"{field} {val} above cap {hi}")
    return val


def _check_str_field(value: object, field: str) -> str:
    if isinstance(value, bool):
        raise TypeError(f"{field} must not be bool")
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str")
    if not value:
        raise ValueError(f"{field} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain null bytes")
    return value


# ---------------------------------------------------------------------------
# Frozen config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoleGatingConfig:
    """Per-token gating network over N task LoRAs.

    Frozen dataclass — post-construction mutation raises ``FrozenInstanceError``.
    """

    num_task_adapters: int
    hidden_dim: int
    temperature: float
    top_k: int

    def __post_init__(self) -> None:
        _check_int(
            self.num_task_adapters,
            "num_task_adapters",
            MIN_TASK_ADAPTERS,
            MAX_TASK_ADAPTERS,
        )
        _check_int(
            self.hidden_dim, "hidden_dim", MIN_HIDDEN_DIM, MAX_HIDDEN_DIM
        )
        _check_finite_positive(
            self.temperature,
            "temperature",
            MIN_TEMPERATURE,
            MAX_TEMPERATURE,
        )
        # top_k must be positive and not exceed num_task_adapters
        _check_int(self.top_k, "top_k", 1, MAX_TASK_ADAPTERS)
        if self.top_k > self.num_task_adapters:
            raise ValueError(
                f"top_k {self.top_k} > num_task_adapters "
                f"{self.num_task_adapters}"
            )


# ---------------------------------------------------------------------------
# Cross-validator (called from SoupConfig + standalone)
# ---------------------------------------------------------------------------


def validate_mole_compat(
    *,
    task: str,
    backend: str,
    num_task_adapters: int,
) -> None:
    """Schema-time gate.

    - Requires ``task='moe_lora_routing'`` (silent-no-op footgun rejection
      matching v0.52.0 distill / v0.62.0 citation_faithful task-gates).
    - Rejects ``backend='mlx'`` — the gating kernel needs torch dispatch
      that mlx-lm doesn't expose (deferred to a future MLX integration).
    - ``num_task_adapters`` must be in ``[MIN_TASK_ADAPTERS, MAX_TASK_ADAPTERS]``.
    """
    _check_str_field(task, "task")
    _check_str_field(backend, "backend")
    if task != "moe_lora_routing":
        raise ValueError(
            f"validate_mole_compat: task must be 'moe_lora_routing' "
            f"(got {task!r})"
        )
    if backend == "mlx":
        raise ValueError(
            "MoLE routing is not supported on the mlx backend "
            "(live wiring deferred; see v0.67.1)"
        )
    _check_int(
        num_task_adapters,
        "num_task_adapters",
        MIN_TASK_ADAPTERS,
        MAX_TASK_ADAPTERS,
    )


# ---------------------------------------------------------------------------
# Deferred-live stub
# ---------------------------------------------------------------------------


def build_gating_kernel(config: MoleGatingConfig):
    """Build a per-token gating kernel for MoLE dispatch.

    Deferred to v0.67.1: live wiring requires the v0.22.0 multi-adapter
    serving surface plus a torch gating module that emits softmax
    routing logits per token. The schema (this module) ships now so
    operators can wire ``num_task_adapters`` / ``top_k`` / ``temperature``
    into their config; the live kernel comes next.
    """
    if not isinstance(config, MoleGatingConfig):
        raise TypeError("config must be MoleGatingConfig")
    raise NotImplementedError(
        "build_gating_kernel live wiring deferred to v0.67.1"
    )
