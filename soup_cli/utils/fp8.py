"""FP8 training — 8-bit floating point training via torchao/transformer_engine.

FP8 training on Hopper (H100, H200) and Blackwell (B100, B200) GPUs uses 8-bit
floating point for matmuls, giving ~2x speedup vs bf16 at comparable quality.

This extends the existing int8-QAT infrastructure (``utils/qat.py``). When the
user sets ``quantization_aware: 'fp8'`` in soup.yaml the FP8 recipe is applied;
``quantization_aware: true`` keeps the legacy int8 QAT path.

Requires:
- NVIDIA Hopper+ GPU (SM 9.0+) — H100, H200, B100, B200
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


def is_fp8_gpu_supported() -> bool:
    """Return True if a Hopper+ GPU is detected (FP8 requires SM 9.0+)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        # SM 9.0 = Hopper (H100), SM 10.0 = Blackwell (B100)
        major, _ = torch.cuda.get_device_capability(0)
        return major >= 9
    except (ImportError, RuntimeError, AssertionError):
        return False


def apply_fp8_training(model) -> bool:
    """Convert eligible linear layers to FP8 for training.

    Uses torchao's ``convert_to_float8_training`` with a tensorwise scaling
    recipe (default / most widely supported).

    Args:
        model: PyTorch model to convert (typically after LoRA has been applied).

    Returns:
        True on success, False if FP8 is unavailable or conversion failed.
    """
    if not is_fp8_available():
        return False

    try:
        from torchao.float8 import convert_to_float8_training

        convert_to_float8_training(model)
        return True
    except (ImportError, RuntimeError, ValueError):
        return False


def validate_fp8_config(
    quantization_aware: QuantizationAwareLike,
    backend: str,
    device: str,
) -> list[str]:
    """Validate FP8 training config.

    Args:
        quantization_aware: TrainingConfig.quantization_aware (False/True/'fp8').
        backend: Training backend (transformers/unsloth/mlx).
        device: Training device (cuda/cpu/mps).

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

    if not is_fp8_gpu_supported():
        errors.append(
            "FP8 training requires a Hopper+ GPU (H100/H200/B100/B200, "
            "compute capability >= 9.0)."
        )

    if not is_fp8_available():
        errors.append(
            "FP8 training dependencies are not installed. "
            "Install with: pip install torchao (>=0.5.0)"
        )

    return errors
