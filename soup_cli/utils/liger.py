"""Liger Kernel — fused operations for faster, memory-efficient training.

Liger Kernel provides fused CUDA kernels (RMSNorm, SwiGLU, CrossEntropy, RoPE, etc.)
that replace standard HuggingFace operations with optimized fused versions.
This can yield 20-60% memory savings and 20-40% throughput improvement.

Requires: liger-kernel >= 0.3.0
"""

from __future__ import annotations


def is_liger_available() -> bool:
    """Check if liger-kernel is installed."""
    try:
        import liger_kernel  # noqa: F401

        return True
    except ImportError:
        return False


def get_liger_version() -> str | None:
    """Return liger-kernel version string, or None if not installed."""
    try:
        import liger_kernel

        return getattr(liger_kernel, "__version__", "unknown")
    except ImportError:
        return None


def apply_liger_kernel(model_name: str) -> bool:
    """Apply Liger Kernel fused operations for the given model architecture.

    Patches the model class in-place so that all subsequent model instantiations
    use fused kernels (RMSNorm, SwiGLU, CrossEntropy, RoPE, FusedLinearCrossEntropy).

    This must be called BEFORE loading the model.

    Args:
        model_name: HuggingFace model name/path (used to detect architecture).

    Returns:
        True if Liger Kernel was applied, False otherwise.
    """
    if not is_liger_available():
        return False

    model_lower = model_name.lower()

    try:
        from liger_kernel.transformers import (
            AutoLigerKernelForCausalLM,
        )

        # AutoLigerKernelForCausalLM handles architecture detection automatically
        AutoLigerKernelForCausalLM._apply_liger_kernel(model_name)
        return True
    except (ImportError, AttributeError, NotImplementedError):
        # Fallback: try manual patching for known architectures
        return _apply_liger_manual(model_lower)


def _apply_liger_manual(model_lower: str) -> bool:
    """Manually apply Liger Kernel patches for known model architectures."""
    try:
        if "llama" in model_lower or "codellama" in model_lower:
            from liger_kernel.transformers import apply_liger_kernel_to_llama

            apply_liger_kernel_to_llama()
            return True
        elif "mistral" in model_lower or "mixtral" in model_lower:
            from liger_kernel.transformers import apply_liger_kernel_to_mistral

            apply_liger_kernel_to_mistral()
            return True
        elif "gemma" in model_lower:
            from liger_kernel.transformers import apply_liger_kernel_to_gemma2

            apply_liger_kernel_to_gemma2()
            return True
        elif "qwen" in model_lower:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen2

            apply_liger_kernel_to_qwen2()
            return True
        elif "phi" in model_lower:
            from liger_kernel.transformers import apply_liger_kernel_to_phi3

            apply_liger_kernel_to_phi3()
            return True
    except (ImportError, AttributeError):
        pass

    return False


def validate_liger_config(use_liger: bool, backend: str, device: str) -> list[str]:
    """Validate Liger Kernel configuration and return error messages.

    Args:
        use_liger: Whether Liger Kernel is requested.
        backend: Training backend (transformers/unsloth).
        device: Training device (cuda/cpu/mps).

    Returns:
        List of error messages. Empty list means valid.
    """
    errors: list[str] = []

    if not use_liger:
        return errors

    if not is_liger_available():
        errors.append(
            "liger-kernel is not installed. "
            "Install it with: pip install 'soup-cli[liger]'"
        )

    if backend == "unsloth":
        errors.append(
            "Liger Kernel is not compatible with the unsloth backend. "
            "Unsloth has its own fused kernels. Use backend: transformers."
        )

    if device != "cuda":
        errors.append(
            "Liger Kernel requires CUDA. "
            f"Current device: {device}. Use a GPU for Liger Kernel."
        )

    return errors
