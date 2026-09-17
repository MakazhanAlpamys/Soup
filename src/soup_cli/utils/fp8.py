"""FP8 training — 8-bit floating point training via torchao/transformer_engine.

FP8 training on Hopper (H100, H200) and Blackwell (B100, B200) GPUs uses 8-bit
floating point for matmuls, giving ~2x speedup vs bf16 at comparable quality.

This extends the existing int8-QAT infrastructure (``utils/qat.py``). When the
user sets ``quantization_aware: 'fp8'`` in soup.yaml the FP8 recipe is applied;
``quantization_aware: true`` keeps the legacy int8 QAT path.

Requires:
- NVIDIA Ada or newer GPU (SM 8.9+) — RTX 40/50-series, L4, L40S, H100, H200,
  B100, B200. Rowwise recipes on Ada need torch >= 2.7 (#835).
- torchao >= 0.5.0 OR transformer-engine >= 1.0
- CUDA 12.0+
"""

from __future__ import annotations

from typing import Literal, Union

QuantizationAwareLike = Union[bool, Literal["fp8"]]


def is_fp8_available() -> bool:
    """Return True if *any* FP8 training backend is importable.

    Checks torchao's FP8 recipe first, then transformer-engine.
    """
    # torchao path (preferred — we already require torchao for int8 QAT)
    try:
        from torchao.float8 import convert_to_float8_training  # noqa: F401

        return True
    except ImportError:
        pass

    try:
        import transformer_engine  # noqa: F401

        return True
    except ImportError:
        pass

    return False


# #835: torch's own floors, read from its source at the release tags.
# ``_scaled_mm_allowed_device()`` accepts ``major >= 9 or (8, 9)`` for every
# recipe (identical at v2.4.0, v2.6.0, v2.7.0, v2.13.0). Rowwise scaling on
# (8, 9) needs the CUTLASS sm89 kernel, first shipped in v2.7.0; v2.6.0 builds
# only ``cutlass::arch::Sm90`` and raises "Rowwise scaling is not currenlty
# supported on your device".
_FP8_MIN_CAPABILITY = (8, 9)
_ROWWISE_ADA_MIN_TORCH = (2, 7)

_FP8_GPU_REFUSAL = (
    "FP8 training requires an Ada or newer GPU (compute capability >= 8.9): "
    "RTX 40/50-series, L4, L40S, RTX 6000 Ada, Hopper (H100/H200) or "
    "Blackwell (B100/B200)."
)


def _cuda_capability() -> "tuple[int, int] | None":
    """Return the first CUDA device's capability, or None without CUDA."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(0)
        return int(major), int(minor)
    except (ImportError, RuntimeError, AssertionError):
        return None


def _torch_at_least(floor: "tuple[int, int]") -> bool:
    """True when the installed torch's major.minor is at least ``floor``."""
    import re

    import torch

    match = re.match(r"(\d+)\.(\d+)", str(torch.__version__))
    if match is None:
        return False
    return (int(match.group(1)), int(match.group(2))) >= floor


def is_fp8_gpu_supported() -> bool:
    """Return True if the GPU meets torch's FP8 floor (SM 8.9+, Ada or newer).

    This is the recipe-independent part of :func:`fp8_training_supported`.
    """
    capability = _cuda_capability()
    return capability is not None and capability >= _FP8_MIN_CAPABILITY


def fp8_training_supported(recipe: str = "tensorwise") -> "tuple[bool, str]":
    """The one FP8 hardware gate (#835): ``(ok, reason)`` for ``recipe``.

    Both ``quantization_aware: fp8`` (:func:`apply_fp8_training`) and
    ``fp8_attention`` (``advanced_precision.apply_fp8_attention``) ask this.
    """
    if not is_fp8_gpu_supported():
        return False, _FP8_GPU_REFUSAL
    capability = _cuda_capability()
    if (
        recipe != "tensorwise"
        and capability is not None
        and capability < (9, 0)
        and not _torch_at_least(_ROWWISE_ADA_MIN_TORCH)
    ):
        import torch

        return False, (
            f"fp8_recipe '{recipe}' on an Ada GPU (compute capability 8.9) needs "
            f"torch >= 2.7 (installed: {torch.__version__}); upgrade torch or use "
            "fp8_recipe: tensorwise."
        )
    return True, ""


def apply_fp8_training(
    model,
    recipe: str = "tensorwise",
) -> bool:
    """Convert eligible linear layers to FP8 for training.

    Uses torchao's ``convert_to_float8_training`` with a scaling recipe
    selected via :pydata:`Float8LinearConfig.from_recipe_name`.

    Supported recipes (from ``torchao.float8.config.Float8LinearRecipeName``):

    - ``"tensorwise"`` — single scale per tensor, cuBLAS kernel (fastest,
      default, v0.28.0 behavior).
    - ``"rowwise"`` — per-row scale, CUTLASS kernel, e4m3 everywhere,
      power-of-2 scales (more accurate).
    - ``"rowwise_with_gw_hp"`` — rowwise but grad_weight stays in high
      precision (most accurate).

    Args:
        model: PyTorch model to convert (typically after LoRA has been applied).
        recipe: Scaling recipe name. Default ``"tensorwise"``.

    Returns:
        True on success, False if FP8 is unavailable or conversion failed.

    Raises:
        RuntimeError: the GPU cannot run ``recipe`` (#835). Nothing is converted.
    """
    if not is_fp8_available():
        return False

    ok, reason = fp8_training_supported(recipe)
    if not ok:
        raise RuntimeError(reason)

    try:
        from torchao.float8 import convert_to_float8_training
        from torchao.float8.config import Float8LinearConfig

        config = Float8LinearConfig.from_recipe_name(recipe)
        convert_to_float8_training(model, config=config)
        return True
    except (ImportError, RuntimeError, ValueError):
        return False


def validate_fp8_config(
    quantization_aware: QuantizationAwareLike,
    backend: str,
    device: str,
    recipe: str = "tensorwise",
) -> list[str]:
    """Validate FP8 training config.

    Args:
        quantization_aware: TrainingConfig.quantization_aware (False/True/'fp8').
        backend: Training backend (transformers/unsloth/mlx).
        device: Training device (cuda/cpu/mps).
        recipe: TrainingConfig.fp8_recipe; rowwise has its own floor (#835).

    Returns:
        List of error messages. Empty list means valid (or FP8 not requested).
    """
    errors: list[str] = []

    # Only validate when FP8 is explicitly requested
    if quantization_aware != "fp8":
        return errors

    if backend == "unsloth":
        errors.append(
            "FP8 training is not compatible with the unsloth backend. "
            "Unsloth uses its own fused kernels. Use backend: transformers."
        )
        return errors

    if backend == "mlx":
        errors.append(
            "FP8 training is not supported on the mlx backend (Apple Silicon). "
            "Use backend: transformers."
        )
        return errors

    if device != "cuda":
        errors.append(
            "FP8 training requires CUDA. "
            f"Current device: {device}."
        )
        return errors

    ok, reason = fp8_training_supported(recipe)
    if not ok:
        errors.append(reason)

    if not is_fp8_available():
        errors.append(
            "FP8 training dependencies are not installed. "
            "Install with: pip install torchao (>=0.5.0)"
        )

    return errors
