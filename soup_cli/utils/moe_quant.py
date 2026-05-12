"""v0.52.0 Part F — MoE expert quantization + router-only training schema.

Two new TrainingConfig fields are introduced this release:

* ``moe_expert_quant: Optional[Literal["nf4", "int8_rowwise"]]`` — per-expert
  weight quantization for fused-MoE Linear blocks. Wraps axolotl's MoE
  expert quant path.
* ``train_router_only: bool`` — freeze every expert + train only the
  gating router (unsloth MoE recipe). Useful for router calibration.

Schema-only this release; live wiring lands in v0.52.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

MOE_EXPERT_QUANT_FORMATS: frozenset[str] = frozenset({"nf4", "int8_rowwise"})

_MAX_QUANT_LEN: int = 32


@dataclass(frozen=True)
class MoEExpertQuantSpec:
    """Metadata for a MoE-expert quant format. Frozen."""

    name: str
    description: str
    bits: int
    live_wired: bool


_MOE_EXPERT_QUANT_METADATA: Mapping[str, MoEExpertQuantSpec] = MappingProxyType({
    "nf4": MoEExpertQuantSpec(
        name="nf4",
        description="NF4 per-expert (BNB 4-bit Normal-Float)",
        bits=4,
        live_wired=False,
    ),
    "int8_rowwise": MoEExpertQuantSpec(
        name="int8_rowwise",
        description="INT8 row-wise per-expert (LLM.int8 row-wise)",
        bits=8,
        live_wired=False,
    ),
})


def validate_moe_expert_quant(name: object) -> str:
    """Validate a MoE expert-quant name. Returns canonical form."""
    if isinstance(name, bool):
        raise TypeError(f"moe_expert_quant must not be bool, got {name!r}")
    if not isinstance(name, str):
        raise TypeError(
            f"moe_expert_quant must be str, got {type(name).__name__}"
        )
    if not name:
        raise ValueError("moe_expert_quant must be non-empty")
    if "\x00" in name:
        raise ValueError("moe_expert_quant must not contain null bytes")
    if len(name) > _MAX_QUANT_LEN:
        raise ValueError(
            f"moe_expert_quant too long (max {_MAX_QUANT_LEN} chars)"
        )
    canonical = name.lower()
    if canonical not in MOE_EXPERT_QUANT_FORMATS:
        supported = ", ".join(sorted(MOE_EXPERT_QUANT_FORMATS))
        raise ValueError(
            f"moe_expert_quant {name!r} not supported. "
            f"Supported: {supported}"
        )
    return canonical


def get_moe_expert_quant_spec(name: str) -> MoEExpertQuantSpec:
    """Return the frozen spec for ``name`` or raise."""
    canonical = validate_moe_expert_quant(name)
    return _MOE_EXPERT_QUANT_METADATA[canonical]


def validate_moe_expert_quant_compat(*, backend: str, moe_lora: bool) -> None:
    """Schema-time gate for ``moe_expert_quant``.

    Rejects:
    - non-string ``backend`` / non-bool ``moe_lora`` (defence-in-depth).
    - ``backend == 'mlx'`` (no MLX MoE expert quant).
    - ``moe_lora == False`` (MoE expert quant only meaningful when training
      LoRA adapters that sit on top of fused-MoE experts — otherwise the
      operator is asking for full-precision MoE training with quantized
      experts, which silently no-ops).
    """
    if isinstance(backend, bool):
        raise TypeError(f"backend must not be bool, got {backend!r}")
    if not isinstance(backend, str) or not backend:
        raise ValueError("backend must be a non-empty string")
    if not isinstance(moe_lora, bool):
        raise TypeError(f"moe_lora must be bool, got {type(moe_lora).__name__}")
    if backend == "mlx":
        raise ValueError(
            "moe_expert_quant is not supported on backend=mlx in v0.52.0"
        )
    if not moe_lora:
        raise ValueError(
            "moe_expert_quant requires moe_lora=true "
            "(per-expert quant is only meaningful with MoE-aware LoRA wiring)"
        )


def validate_train_router_only_compat(*, backend: str, moe_lora: bool) -> None:
    """Schema-time gate for ``train_router_only=True``.

    Requires ``moe_lora=true`` (without it, every expert would still be
    trained and the flag would silently no-op). Defence-in-depth bool /
    str rejection on the args.
    """
    if isinstance(backend, bool):
        raise TypeError(f"backend must not be bool, got {backend!r}")
    if not isinstance(backend, str) or not backend:
        raise ValueError("backend must be a non-empty string")
    if not isinstance(moe_lora, bool):
        raise TypeError(f"moe_lora must be bool, got {type(moe_lora).__name__}")
    if backend == "mlx":
        raise ValueError(
            "train_router_only is not supported on backend=mlx in v0.52.0"
        )
    if not moe_lora:
        raise ValueError(
            "train_router_only requires moe_lora=true "
            "(router-only training freezes the experts, which is only "
            "meaningful with MoE-aware LoRA wiring)"
        )


def apply_moe_expert_quant() -> None:
    """Live MoE expert-quant wiring — deferred to v0.52.1."""
    raise NotImplementedError(
        "MoE expert quantization live wiring deferred to v0.52.1. "
        "Schema accepts the format but no per-expert quant path is wired."
    )
