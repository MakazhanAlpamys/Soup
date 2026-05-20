"""Hardware-fit calculator: static analytical predictor of peak VRAM.

v0.40.3 #64 added a live CUDA OOM probe; that's correct but slow (one
real forward+backward per candidate). v0.64 adds a fast analytical
predictor that takes (params, seq_len, batch_size, optimizer, quant,
peft, gradient_checkpointing) and outputs a five-bucket VRAM breakdown.
Used by ``decide_hardware_fit`` to refuse a training launch when the
predicted peak (with ``VRAM_SAFETY_MARGIN=10%`` headroom) would not fit.

The math is intentionally conservative — operators can supply
``--allow-oom-attempt`` to bypass the gate if they want to try anyway
(opt-out, not opt-in, per the v0.64 Part D plan).

Public surface:
- ``VRAM_SAFETY_MARGIN = 0.10``.
- ``validate_seq_len(v)`` / ``validate_batch_size(v)``.
- ``HardwareFitInput`` frozen dataclass.
- ``VRAMBreakdown`` frozen dataclass + ``total_gb`` property.
- ``HardwareFitReport`` frozen dataclass.
- ``estimate_peak_vram_gb(inp)`` -> ``VRAMBreakdown``.
- ``decide_hardware_fit(inp, *, available_vram_gb)`` -> ``HardwareFitReport``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

VRAM_SAFETY_MARGIN = 0.10  # 10% headroom

_MIN_SEQ_LEN = 64
_MAX_SEQ_LEN = 1_048_576
_MIN_BATCH = 1
_MAX_BATCH = 1024
_MAX_PARAMS_B = 1000.0

# Closed allowlists — schema-shape rejection, no surprises.
_VALID_QUANT = frozenset({
    "none", "4bit", "8bit", "fp8", "gptq", "awq", "aqlm", "eetq", "mxfp4",
})
_VALID_PEFT = frozenset({"full", "lora", "dora", "qlora"})
_VALID_OPTIMIZERS = frozenset({
    "adamw_torch", "adamw_torch_fused", "adafactor", "sgd",
    "adamw_bnb_8bit", "paged_adamw_8bit", "lion_8bit",
    "lomo", "adalomo", "schedule_free_adamw",
})


def validate_seq_len(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("seq_len must be int, not bool")
    if not isinstance(value, int):
        raise TypeError(f"seq_len must be int, got {type(value).__name__}")
    if not (_MIN_SEQ_LEN <= value <= _MAX_SEQ_LEN):
        raise ValueError(
            f"seq_len must be in [{_MIN_SEQ_LEN}, {_MAX_SEQ_LEN}], got {value}"
        )
    return value


def validate_batch_size(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("batch_size must be int, not bool")
    if not isinstance(value, int):
        raise TypeError(f"batch_size must be int, got {type(value).__name__}")
    if not (_MIN_BATCH <= value <= _MAX_BATCH):
        raise ValueError(
            f"batch_size must be in [{_MIN_BATCH}, {_MAX_BATCH}], got {value}"
        )
    return value


@dataclass(frozen=True)
class HardwareFitInput:
    """Inputs needed to predict peak VRAM."""

    params_b: float
    seq_len: int
    batch_size: int
    optimizer: str
    quant: str
    peft: str
    gradient_checkpointing: bool

    def __post_init__(self) -> None:
        if isinstance(self.params_b, bool):
            raise TypeError("params_b must be a number, not bool")
        if not isinstance(self.params_b, (int, float)):
            raise TypeError(
                f"params_b must be a number, got {type(self.params_b).__name__}"
            )
        if not math.isfinite(float(self.params_b)):
            raise ValueError("params_b must be finite")
        if self.params_b <= 0 or self.params_b > _MAX_PARAMS_B:
            raise ValueError(
                f"params_b must be in (0, {_MAX_PARAMS_B}], got {self.params_b}"
            )
        validate_seq_len(self.seq_len)
        validate_batch_size(self.batch_size)
        if not isinstance(self.optimizer, str):
            raise TypeError("optimizer must be str")
        if self.optimizer not in _VALID_OPTIMIZERS:
            raise ValueError(
                f"unknown optimizer {self.optimizer!r}; "
                f"known: {', '.join(sorted(_VALID_OPTIMIZERS))}"
            )
        if not isinstance(self.quant, str):
            raise TypeError("quant must be str")
        if self.quant not in _VALID_QUANT:
            raise ValueError(
                f"unknown quant {self.quant!r}; "
                f"known: {', '.join(sorted(_VALID_QUANT))}"
            )
        if not isinstance(self.peft, str):
            raise TypeError("peft must be str")
        if self.peft not in _VALID_PEFT:
            raise ValueError(
                f"unknown peft {self.peft!r}; "
                f"known: {', '.join(sorted(_VALID_PEFT))}"
            )
        if not isinstance(self.gradient_checkpointing, bool):
            raise TypeError("gradient_checkpointing must be bool")


@dataclass(frozen=True)
class VRAMBreakdown:
    """Per-class peak VRAM in GB."""

    weights_gb: float
    optimizer_gb: float
    gradients_gb: float
    activations_gb: float
    overhead_gb: float

    def __post_init__(self) -> None:
        for fld in (
            "weights_gb", "optimizer_gb", "gradients_gb",
            "activations_gb", "overhead_gb",
        ):
            val = getattr(self, fld)
            if isinstance(val, bool):
                raise TypeError(f"{fld} must be a number, not bool")
            if not isinstance(val, (int, float)):
                raise TypeError(f"{fld} must be a number")
            if not math.isfinite(float(val)):
                raise ValueError(f"{fld} must be finite")
            if val < 0:
                raise ValueError(f"{fld} must not be negative, got {val}")

    @property
    def total_gb(self) -> float:
        return float(
            self.weights_gb
            + self.optimizer_gb
            + self.gradients_gb
            + self.activations_gb
            + self.overhead_gb
        )


@dataclass(frozen=True)
class HardwareFitReport:
    """Outcome of a hardware-fit decision."""

    ok: bool
    peak_vram_gb: float
    required_with_margin_gb: float
    available_vram_gb: float
    breakdown: VRAMBreakdown
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.ok, bool):
            raise TypeError("ok must be bool")
        for fld in ("peak_vram_gb", "required_with_margin_gb", "available_vram_gb"):
            val = getattr(self, fld)
            if isinstance(val, bool):
                raise TypeError(f"{fld} must be a number, not bool")
            if not isinstance(val, (int, float)):
                raise TypeError(f"{fld} must be a number")
            if not math.isfinite(float(val)):
                raise ValueError(f"{fld} must be finite")
            if val < 0:
                raise ValueError(f"{fld} must not be negative")
        if not isinstance(self.breakdown, VRAMBreakdown):
            raise TypeError("breakdown must be VRAMBreakdown")
        if not isinstance(self.reason, str):
            raise TypeError("reason must be str")


# Bytes-per-param multiplier by quant scheme. Approximate; the deep
# truth lives in upstream BNB / GPTQ / AWQ docs.
_BYTES_PER_PARAM_BY_QUANT = {
    "none": 2.0,    # bf16/fp16
    "4bit": 0.5,
    "8bit": 1.0,
    "fp8": 1.0,
    "gptq": 0.55,
    "awq": 0.55,
    "aqlm": 0.5,
    "eetq": 1.0,
    "mxfp4": 0.55,
}

# Optimiser-state bytes per trainable parameter. AdamW = 8 (m + v fp32),
# 8-bit AdamW = 2, Adafactor = 4, SGD = 0 (momentum-less reference).
_OPTIM_BYTES_PER_PARAM = {
    "adamw_torch": 8.0,
    "adamw_torch_fused": 8.0,
    "adafactor": 4.0,
    "sgd": 0.0,
    "adamw_bnb_8bit": 2.0,
    "paged_adamw_8bit": 2.0,
    "lion_8bit": 2.0,
    "lomo": 0.0,
    "adalomo": 2.0,
    "schedule_free_adamw": 8.0,
}


def _trainable_param_fraction(peft: str) -> float:
    """LoRA / DoRA / QLoRA train ~1% of parameters. Full = 100%."""
    if peft == "full":
        return 1.0
    if peft in ("lora", "qlora", "dora"):
        return 0.01
    return 1.0  # defensive default; schema gate already rejects unknown


def _activation_bytes(seq_len: int, batch_size: int, params_b: float) -> float:
    """Per-batch activation memory.

    Very approximate: scales with seq_len × batch_size × hidden_size.
    We treat hidden_size as ~ 64 * sqrt(params_b * 1e9 / 1024) — close
    enough for an order-of-magnitude estimate at the 1B-70B band.

    Result is bounded by ``_MAX_ACTIVATIONS_BYTES`` (~1 EiB) so the
    downstream ``VRAMBreakdown`` validator never sees a +Inf product
    even at the schema max (seq=1M × batch=1024).
    """
    hidden = 64.0 * math.sqrt(max(params_b * 1e9 / 1024.0, 1.0))
    raw = float(seq_len) * float(batch_size) * hidden * 4.0
    # Defensive: clamp absurd products so the math stays finite.
    if not math.isfinite(raw) or raw > _MAX_ACTIVATIONS_BYTES:
        return _MAX_ACTIVATIONS_BYTES
    return raw


_MAX_ACTIVATIONS_BYTES = 1e18  # 1 EB sanity cap; far above any real GPU


def estimate_peak_vram_gb(inp: HardwareFitInput) -> VRAMBreakdown:
    """Static analytical VRAM predictor. Returns a per-class breakdown."""
    if not isinstance(inp, HardwareFitInput):
        raise TypeError(
            f"inp must be HardwareFitInput, got {type(inp).__name__}"
        )

    params = float(inp.params_b) * 1e9  # absolute count
    bytes_per = _BYTES_PER_PARAM_BY_QUANT.get(inp.quant, 2.0)
    weights_b = params * bytes_per

    trainable_frac = _trainable_param_fraction(inp.peft)
    trainable_params = params * trainable_frac
    optim_b = trainable_params * _OPTIM_BYTES_PER_PARAM.get(inp.optimizer, 8.0)
    # Gradients are fp32 of trainable params (4 bytes/param) under
    # mixed-precision; under "none" quant we still keep fp32 grads.
    gradients_b = trainable_params * 4.0

    activations_b = _activation_bytes(inp.seq_len, inp.batch_size, inp.params_b)
    if inp.gradient_checkpointing:
        # Gradient checkpointing roughly halves activation memory.
        activations_b *= 0.5

    # Constant overhead: kernels / autograd graph / NCCL buffers / etc.
    # Scales weakly with params.
    overhead_b = 0.5e9 + 0.05 * weights_b

    return VRAMBreakdown(
        weights_gb=weights_b / 1e9,
        optimizer_gb=optim_b / 1e9,
        gradients_gb=gradients_b / 1e9,
        activations_gb=activations_b / 1e9,
        overhead_gb=overhead_b / 1e9,
    )


def decide_hardware_fit(
    inp: HardwareFitInput,
    *,
    available_vram_gb: float,
) -> HardwareFitReport:
    """Decide whether the planned run fits under ``available_vram_gb``.

    Applies ``VRAM_SAFETY_MARGIN=10%`` headroom on top of the analytical
    estimate. Refuse iff ``predicted * 1.1 > available``.
    """
    if isinstance(available_vram_gb, bool):
        raise TypeError("available_vram_gb must be a number, not bool")
    if not isinstance(available_vram_gb, (int, float)):
        raise TypeError(
            f"available_vram_gb must be a number, got {type(available_vram_gb).__name__}"
        )
    if not math.isfinite(float(available_vram_gb)):
        raise ValueError("available_vram_gb must be finite")
    if available_vram_gb < 0:
        raise ValueError(
            f"available_vram_gb must be >= 0, got {available_vram_gb}"
        )

    breakdown = estimate_peak_vram_gb(inp)
    peak = breakdown.total_gb
    required = peak * (1.0 + VRAM_SAFETY_MARGIN)
    if required <= available_vram_gb:
        return HardwareFitReport(
            ok=True,
            peak_vram_gb=peak,
            required_with_margin_gb=required,
            available_vram_gb=float(available_vram_gb),
            breakdown=breakdown,
            reason=(
                f"fits: peak {peak:.2f} GB + {VRAM_SAFETY_MARGIN:.0%} margin "
                f"<= {available_vram_gb:.2f} GB available"
            ),
        )
    return HardwareFitReport(
        ok=False,
        peak_vram_gb=peak,
        required_with_margin_gb=required,
        available_vram_gb=float(available_vram_gb),
        breakdown=breakdown,
        reason=(
            f"OOM risk: required {required:.2f} GB exceeds "
            f"available {available_vram_gb:.2f} GB. "
            f"Try --batch-size {max(1, inp.batch_size // 2)} or "
            "--quantization 4bit or --gradient-checkpointing auto."
        ),
    )


__all__ = [
    "VRAM_SAFETY_MARGIN",
    "HardwareFitInput",
    "HardwareFitReport",
    "VRAMBreakdown",
    "decide_hardware_fit",
    "estimate_peak_vram_gb",
    "validate_batch_size",
    "validate_seq_len",
]
