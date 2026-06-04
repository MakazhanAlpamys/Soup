"""MoLE per-token adapter routing (v0.67.0 schema / v0.71.12 #222 live).

A gating network that routes per-token activations to a weighted blend of N
task LoRAs. Following the MoLE paper (Mixture of LoRA Experts, Wu et al. 2024),
dispatch uses softmax gating over the per-token hidden state to select top-K
LoRAs and blend them by gating weights.

v0.67.0 shipped the schema + cross-validator with a deferred ``build_gating_kernel``
stub; v0.71.12 #222 lifts the stub to a live ``torch.nn.Module`` and wires the
``MoleRoutingTrainerWrapper`` (``trainer/mole_routing.py``).

Public surface:

- ``MoleGatingConfig`` frozen dataclass
- ``validate_mole_compat(task, backend, num_task_adapters)``
- ``validate_mole_task_adapters(value)``
- ``build_gating_kernel(config)`` -> live ``torch.nn.Module`` (per-token router)
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
from functools import lru_cache

# ---------------------------------------------------------------------------
# Bounds (closed, locked at module load)
# ---------------------------------------------------------------------------

MIN_TASK_ADAPTERS = 2
MAX_TASK_ADAPTERS = 64
MIN_HIDDEN_DIM = 1
MAX_HIDDEN_DIM = 16_384
MIN_TEMPERATURE = 1e-6
MAX_TEMPERATURE = 100.0
_MAX_ADAPTER_PATH_LEN = 4096


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


def validate_mole_task_adapters(value: object) -> list[str]:
    """Validate the per-token MoLE task-adapter path list.

    Requires a list of `[MIN_TASK_ADAPTERS, MAX_TASK_ADAPTERS]` non-empty,
    null-byte-free, `<= _MAX_ADAPTER_PATH_LEN`-char path strings. Returns a
    defensive copy. Duplicate paths are rejected (an adapter routed to twice
    is a config error). Containment / symlink checks happen at trainer load
    time (paths may be HF ids or local dirs — mirrors `cfg.base` policy).
    """
    if not isinstance(value, (list, tuple)):
        raise TypeError("mole_task_adapters must be a list of paths")
    paths = list(value)
    if len(paths) < MIN_TASK_ADAPTERS:
        raise ValueError(
            f"mole_task_adapters needs >= {MIN_TASK_ADAPTERS} adapters, "
            f"got {len(paths)}"
        )
    if len(paths) > MAX_TASK_ADAPTERS:
        raise ValueError(
            f"mole_task_adapters {len(paths)} > cap {MAX_TASK_ADAPTERS}"
        )
    seen: set[str] = set()
    for item in paths:
        if isinstance(item, bool) or not isinstance(item, str):
            raise TypeError("each mole_task_adapter must be a string path")
        if not item:
            raise ValueError("mole_task_adapter path must be non-empty")
        if "\x00" in item:
            raise ValueError("mole_task_adapter path must not contain null bytes")
        if len(item) > _MAX_ADAPTER_PATH_LEN:
            raise ValueError(
                f"mole_task_adapter path exceeds {_MAX_ADAPTER_PATH_LEN} chars"
            )
        if item in seen:
            raise ValueError(f"duplicate mole_task_adapter path {item!r}")
        seen.add(item)
    return paths


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
            "(the gating kernel needs torch dispatch that mlx-lm does not expose)"
        )
    _check_int(
        num_task_adapters,
        "num_task_adapters",
        MIN_TASK_ADAPTERS,
        MAX_TASK_ADAPTERS,
    )


# ---------------------------------------------------------------------------
# Live gating kernel (v0.71.12 #222)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _gating_kernel_cls():
    """Materialise the gating-kernel ``nn.Module`` class (lazy torch import).

    ``lru_cache`` makes it a process-singleton (matches ``make_mole_trainer_class``
    in ``trainer/mole_routing.py``) so the module top stays torch-free for the
    CLI-startup hot path.
    """
    import torch
    from torch import nn

    class MoleGatingKernel(nn.Module):
        """Per-token router over N task LoRAs (MoLE, Wu et al. 2024).

        ``forward(hidden)`` maps a ``[..., hidden_dim]`` hidden state to
        ``[..., num_task_adapters]`` softmax routing weights via a ``nn.Linear``
        gate + temperature-scaled top-k softmax. When ``top_k < num_task_adapters``
        only the top-k adapters per token get non-zero weight (sparse dispatch),
        and the kept logits are renormalised via softmax so the weights sum to 1.
        """

        def __init__(self, config: MoleGatingConfig):
            super().__init__()
            self.num_task_adapters = config.num_task_adapters
            self.hidden_dim = config.hidden_dim
            self.temperature = float(config.temperature)
            self.top_k = config.top_k
            # bias=False — a pure linear router over the residual stream.
            self.gate = nn.Linear(
                config.hidden_dim, config.num_task_adapters, bias=False
            )

        def forward(self, hidden):  # noqa: D401 — torch forward
            logits = self.gate(hidden) / self.temperature
            if self.top_k < self.num_task_adapters:
                topv, topi = torch.topk(logits, self.top_k, dim=-1)
                masked = torch.full_like(logits, float("-inf"))
                masked.scatter_(-1, topi, topv)
                logits = masked
            return torch.softmax(logits, dim=-1)

    return MoleGatingKernel


def build_gating_kernel(config: MoleGatingConfig):
    """Build a live per-token gating kernel for MoLE dispatch (v0.71.12 #222).

    Returns a ``torch.nn.Module`` whose forward maps a ``[..., hidden_dim]``
    hidden state to ``[..., num_task_adapters]`` per-token routing weights. The
    gate is the only trainable parameter in a MoLE run — the base model and
    every task LoRA stay frozen, and the router learns which adapter(s) each
    token should be routed to.

    Lazy torch import (heavy dep stays out of the module top, per project policy).
    """
    if not isinstance(config, MoleGatingConfig):
        raise TypeError("config must be MoleGatingConfig")
    return _gating_kernel_cls()(config)
