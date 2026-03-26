"""FlashAttention auto-detection and configuration.

Detects FlashAttention availability (v2/v3/v4) and configures models
to use the best available attention implementation automatically.

FlashAttention provides 2-4x speedup and significant memory savings
for long sequences by avoiding materializing the full attention matrix.
"""

from __future__ import annotations

# Ordered by preference (newest first)
FLASH_ATTN_VERSIONS = ("flash_attention_3", "flash_attention_2")


def detect_flash_attention() -> str | None:
    """Detect the best available FlashAttention implementation.

    Returns:
        The attention implementation string for model_kwargs, or None if unavailable.
        One of: "flash_attention_3", "flash_attention_2", None.
    """
    # FlashAttention requires CUDA
    try:
        import torch

        if not torch.cuda.is_available():
            return None
    except ImportError:
        return None

    # Check FlashAttention 3 (Hopper architecture, H100+)
    try:
        import flash_attn  # noqa: F401

        version = getattr(flash_attn, "__version__", "0.0.0")
        major = int(version.split(".")[0])
        if major >= 3:
            return "flash_attention_3"
    except (ImportError, ValueError, IndexError):
        pass

    # Check FlashAttention 2
    try:
        from transformers.utils import is_flash_attn_2_available

        if is_flash_attn_2_available():
            return "flash_attention_2"
    except ImportError:
        pass

    # Direct import check for flash_attn 2.x
    try:
        import flash_attn  # noqa: F401

        version = getattr(flash_attn, "__version__", "0.0.0")
        major = int(version.split(".")[0])
        if major >= 2:
            return "flash_attention_2"
    except (ImportError, ValueError, IndexError):
        pass

    return None


def get_flash_attn_version() -> str | None:
    """Return the installed flash-attn package version, or None."""
    try:
        import flash_attn

        return getattr(flash_attn, "__version__", "unknown")
    except ImportError:
        return None


def get_attn_implementation(use_flash_attn: bool, device: str) -> str | None:
    """Get the best attention implementation to use.

    Args:
        use_flash_attn: Whether FlashAttention is requested in config.
        device: Training device (cuda/cpu/mps).

    Returns:
        Attention implementation string for from_pretrained(), or None for default.
    """
    if not use_flash_attn:
        return None

    if device != "cuda":
        return None

    return detect_flash_attention()


def validate_flash_attn_config(
    use_flash_attn: bool, backend: str, device: str,
) -> list[str]:
    """Validate FlashAttention configuration and return error messages.

    Args:
        use_flash_attn: Whether FlashAttention is requested.
        backend: Training backend (transformers/unsloth).
        device: Training device (cuda/cpu/mps).

    Returns:
        List of error messages. Empty list means valid.
    """
    errors: list[str] = []

    if not use_flash_attn:
        return errors

    # Unsloth handles FlashAttention internally — no validation needed
    if backend == "unsloth":
        return errors

    if device != "cuda":
        errors.append(
            "FlashAttention requires CUDA. "
            f"Current device: {device}."
        )

    if device == "cuda" and detect_flash_attention() is None:
        errors.append(
            "FlashAttention is not available. "
            "Install it with: pip install flash-attn --no-build-isolation"
        )

    return errors
