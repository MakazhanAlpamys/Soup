"""FP8 training — 8-bit floating point training via torchao/transformer_engine.

FP8 training on Hopper (H100, H200) and Blackwell (B100, B200) GPUs uses 8-bit
floating point for matmuls, giving ~2x speedup vs bf16 at comparable quality.

This extends the existing int8-QAT infrastructure (``utils/qat.py``). When the
user sets ``quantization_aware: 'fp8'`` in soup.yaml the FP8 recipe is applied;
``quantization_aware: true`` keeps the legacy int8 QAT path.

Requires:
- NVIDIA Ada or newer GPU (SM 8.9+) — RTX 40/50-series, L4, L40S, H100, H200,
  B100, B200. Rowwise recipes also need a torch that dispatches their kernel on
  the card (Ada >= 2.7, RTX 50 >= 2.8, 11.x >= 2.10) and never run on Windows
  (#835).
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
#
# Every recipe: ``_scaled_mm_allowed_device()`` accepts ``major >= 9 or (8, 9)``
# (identical at v2.4.0, v2.6.0, v2.7.0, v2.13.0).
#
# Rowwise recipes run a CUTLASS kernel in ``RowwiseScaledMM.cu``, which has two
# separate gates:
#
# * a BUILD gate, ``BUILD_ROWWISE_FP8_KERNEL``, defined only when
#   ``!USE_ROCM && !_WIN32`` (every release through v2.14.0) and, before v2.11.0,
#   ``CUDA_VERSION >= 12000``. Without it, ``f8f8bf16_rowwise`` raises "Rowwise
#   scaling is not currenlty supported on your device" on EVERY device -- the
#   message names the device but the refusal is the build's. So rowwise never
#   runs on Windows, and never on a CUDA 11 build of torch before 2.11.
# * a DEVICE dispatch, first added in v2.7.0: ``sm89 || sm9x || sm10x``, then
#   ``sm12x`` in v2.8.0 and ``sm11x`` in v2.10.0 (still the full list at
#   v2.14.0). Any other major raises "Rowwise scaling is not currently supported
#   on your device" at the first forward.
#: TRANSCRIBED from torch's source, not queried from torch: torch 2.14 exposes no
#: runtime FP8 capability API (no `scaled`/`float8` symbol on `torch._C` or
#: `torch.backends.cuda`), so a table is the only option short of a trial matmul.
#: Checked against torch source through v2.14.0; a major that is not listed below
#: is REFUSED for rowwise until someone updates this table, which is the
#: conservative direction (#1044 review).
_FP8_MIN_CAPABILITY = (8, 9)

#: The first torch whose rowwise dispatch accepts each compute-capability major.
#: A major missing here is accepted by no release this was checked against.
#: 9.x and 10.x are Soup's own torch floor (2.6): v2.6.0 has no device dispatch
#: at all, and v2.7.0 already lists both.
_ROWWISE_MIN_TORCH_BY_MAJOR = {
    8: (2, 7),
    9: (2, 6),
    10: (2, 6),
    11: (2, 10),
    12: (2, 8),
}

#: Before this torch, the rowwise kernel is only built against CUDA >= 12.
_ROWWISE_ANY_CUDA_TORCH = (2, 11)

_FP8_GPU_REFUSAL = (
    "FP8 training requires an Ada or newer GPU (compute capability >= 8.9): "
    "RTX 40/50-series, L4, L40S, RTX 6000 Ada, Hopper (H100/H200) or "
    "Blackwell (B100/B200)."
)


class FP8HardwareUnsupportedError(RuntimeError):
    """An explicitly requested FP8 setting this card, OS or torch build cannot run.

    Raised before anything is converted, and never caught as an optional
    feature: a setting that is accepted and then silently not applied is the
    defect class this exists to stop (#835 review). Every trainer lets it end
    the run at setup.
    """


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


def _version_at_least(version: object, floor: "tuple[int, int]") -> bool:
    """True when ``version``'s major.minor is at least ``floor``; False if unreadable."""
    import re

    match = re.match(r"(\d+)\.(\d+)", str(version))
    if match is None:
        return False
    return (int(match.group(1)), int(match.group(2))) >= floor


def _torch_at_least(floor: "tuple[int, int]") -> bool:
    """True when the installed torch's major.minor is at least ``floor``."""
    import torch

    return _version_at_least(torch.__version__, floor)


def is_fp8_gpu_supported() -> bool:
    """Return True if the GPU meets torch's FP8 floor (SM 8.9+, Ada or newer).

    This is the recipe-independent part of :func:`fp8_training_supported`.
    """
    capability = _cuda_capability()
    return capability is not None and capability >= _FP8_MIN_CAPABILITY


def _rowwise_refusal(recipe: str, capability: "tuple[int, int] | None") -> "str | None":
    """Why ``recipe`` (a rowwise one) cannot run here, or None if it can."""
    import sys

    import torch

    if sys.platform.startswith("win"):
        return (
            f"fp8_recipe '{recipe}' cannot run on Windows: torch never builds its "
            "rowwise FP8 kernel there (BUILD_ROWWISE_FP8_KERNEL requires !_WIN32). "
            "Use fp8_recipe: tensorwise."
        )
    cuda_build = getattr(torch.version, "cuda", None)
    if not _torch_at_least(_ROWWISE_ANY_CUDA_TORCH) and not _version_at_least(
        cuda_build, (12, 0)
    ):
        return (
            f"fp8_recipe '{recipe}' needs a torch built against CUDA >= 12 before "
            f"torch 2.11 (installed: torch {torch.__version__}, CUDA {cuda_build}); "
            "use fp8_recipe: tensorwise."
        )
    if capability is None:
        return None
    floor = _ROWWISE_MIN_TORCH_BY_MAJOR.get(capability[0])
    if floor is None:
        return (
            f"fp8_recipe '{recipe}' is not supported on compute capability "
            f"{capability[0]}.{capability[1]}: no torch release checked (through 2.14) "
            "dispatches its rowwise FP8 kernel there. Use fp8_recipe: tensorwise."
        )
    if not _torch_at_least(floor):
        return (
            f"fp8_recipe '{recipe}' on compute capability {capability[0]}."
            f"{capability[1]} needs torch >= {floor[0]}.{floor[1]} (installed: "
            f"{torch.__version__}); upgrade torch or use fp8_recipe: tensorwise."
        )
    return None


def fp8_training_supported(recipe: str = "tensorwise") -> "tuple[bool, str]":
    """The one FP8 hardware gate (#835): ``(ok, reason)`` for ``recipe``.

    Both ``quantization_aware: fp8`` (:func:`apply_fp8_training`) and
    ``fp8_attention`` (``advanced_precision.apply_fp8_attention``) ask this.
    """
    if not is_fp8_gpu_supported():
        return False, _FP8_GPU_REFUSAL
    if recipe != "tensorwise":
        reason = _rowwise_refusal(recipe, _cuda_capability())
        if reason is not None:
            return False, reason
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
        True on success, False if the torchao/transformer-engine dependency is
        missing or the conversion failed. A card that cannot run ``recipe``
        raises instead, whether or not the dependency is present.

    Raises:
        FP8HardwareUnsupportedError: this card, OS or torch build cannot run
            ``recipe`` (#835). Nothing is converted, and the run must stop.
    """
    # The hardware gate runs FIRST, before the dependency probe (#1044 review).
    # torchao is not a default dependency, so "absent" is the common case: asking
    # availability first meant an Ampere user who wrote quantization_aware: fp8
    # got a yellow line and a bf16 run -- exactly what the ruling removes. What
    # the user asked for cannot run here whether or not torchao is installed.
    ok, reason = fp8_training_supported(recipe)
    if not ok:
        raise FP8HardwareUnsupportedError(reason)

    if not is_fp8_available():
        return False

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
