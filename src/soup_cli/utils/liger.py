"""Liger Kernel — fused operations for faster, memory-efficient training.

Liger Kernel provides fused CUDA kernels (RMSNorm, SwiGLU, CrossEntropy, RoPE, etc.)
that replace standard HuggingFace operations with optimized fused versions.
Measured on an H100 (sm90) with Llama-3.1-8B + LoRA, batch 4, seq 1024,
5 interleaved repeats: **12.9% memory saving and 5.1% throughput**, verified by
confirming the module classes were actually swapped for Liger's. The saving would
shrink further with gradient checkpointing on.
Numbers: benchmarks/gate-h100-validation.md.

Requires: liger-kernel >= 0.3.0
"""

from __future__ import annotations


def check_liger_available() -> bool:
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
    if not check_liger_available():
        return False

    clean_name = str(model_name or "").strip().replace("\\", "/").rstrip("/")
    last_component = clean_name.rsplit("/", 1)[-1].lower() if clean_name else ""

    # Detect the architecture from the CONFIG, not from the path.
    #
    # This used to call `AutoLigerKernelForCausalLM._apply_liger_kernel(model_name)`
    # and fall back to matching "llama" / "mistral" / "qwen2" as a SUBSTRING of the
    # model name. That private classmethod does not exist in liger-kernel 0.8.1
    # (`AttributeError: type object 'AutoLigerKernelForCausalLM' has no attribute
    # '_apply_liger_kernel'`, verified), so the fallback was the only live path —
    # and a model loaded from a local directory carries no architecture in its
    # name. Every such run printed "no matching architecture found" and trained
    # WITHOUT Liger, on a flag the user had explicitly set.
    #
    # `AutoConfig.model_type` is exactly the string liger's module-level
    # `_apply_liger_kernel(model_type)` expects, and a directory cannot rename it.
    model_type = _detect_model_type(model_name)
    if model_type:
        try:
            from liger_kernel.transformers import _apply_liger_kernel

            _apply_liger_kernel(model_type)
            return True
        except (ImportError, AttributeError, NotImplementedError, KeyError, ValueError):
            # liger does not support this architecture, or moved the entry point
            # again — fall through to the name-based attempt rather than claiming
            # success we cannot back.
            pass

    return _apply_liger_manual(last_component)


def _detect_model_type(model_name: str) -> str:
    """``config.model_type`` for a local path or hub id, or "" when unavailable.

    Deliberately quiet: a missing config is not an error here, it just means the
    caller falls back to the older name-based match and the run still trains.
    """
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
    except Exception:  # noqa: BLE001 — detection is best-effort by design
        return ""
    return str(getattr(config, "model_type", "") or "")


def _apply_liger_manual(last_component: str) -> bool:
    """Manually apply Liger Kernel patches matching the last path component (#456)."""
    try:
        if "llama" in last_component or "codellama" in last_component:
            from liger_kernel.transformers import apply_liger_kernel_to_llama

            apply_liger_kernel_to_llama()
            return True
        elif "mistral" in last_component or "mixtral" in last_component:
            from liger_kernel.transformers import apply_liger_kernel_to_mistral

            apply_liger_kernel_to_mistral()
            return True
        elif "gemma" in last_component:
            from liger_kernel.transformers import apply_liger_kernel_to_gemma2

            apply_liger_kernel_to_gemma2()
            return True
        elif "qwen" in last_component:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen2

            apply_liger_kernel_to_qwen2()
            return True
        elif "phi" in last_component:
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

    if not check_liger_available():
        errors.append(
            "liger-kernel is not installed. "
            "Install it with: pip install \"soup-cli[liger]\""
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
