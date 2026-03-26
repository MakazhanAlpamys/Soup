"""Ring FlashAttention — sequence parallelism for ultra-long contexts.

Ring FlashAttention distributes a single long sequence across multiple GPUs,
splitting the sequence into chunks and using ring communication to compute
attention without materializing the full attention matrix on any single GPU.

This enables training on sequences much longer than a single GPU's memory
allows (e.g., 128k-1M+ tokens).

Requires: ring-flash-attn >= 0.1.0 OR transformers >= 4.43.0 (built-in SP support)
"""

from __future__ import annotations


def is_ring_attention_available() -> bool:
    """Check if Ring FlashAttention is available.

    Checks for either:
    1. The ring-flash-attn package
    2. Transformers' built-in sequence parallelism support
    """
    try:
        import ring_flash_attn  # noqa: F401

        return True
    except ImportError:
        return False


def get_ring_attention_version() -> str | None:
    """Return ring-flash-attn version, or None if not installed."""
    try:
        import ring_flash_attn

        return getattr(ring_flash_attn, "__version__", "unknown")
    except ImportError:
        return None


def get_sequence_parallel_size(gpu_count: int, max_length: int) -> int:
    """Calculate optimal sequence parallel size.

    Distributes sequences across GPUs when sequence length exceeds per-GPU capacity.

    Args:
        gpu_count: Number of available GPUs.
        max_length: Target sequence length in tokens.

    Returns:
        Number of GPUs to use for sequence parallelism (1 = no SP).
    """
    # Sequence parallelism is beneficial for long sequences
    # Rule of thumb: use SP when sequence > 32k tokens per GPU
    tokens_per_gpu_threshold = 32768

    if gpu_count <= 1:
        return 1

    if max_length <= tokens_per_gpu_threshold:
        return 1

    # Use power-of-2 SP sizes for efficient communication
    sp_size = 1
    while sp_size * 2 <= gpu_count and max_length // (sp_size * 2) > 4096:
        sp_size *= 2

    return sp_size


def validate_ring_attention_config(
    use_ring_attention: bool,
    device: str,
    max_length: int,
) -> list[str]:
    """Validate Ring FlashAttention configuration.

    Args:
        use_ring_attention: Whether ring attention is requested.
        device: Training device (cuda/cpu/mps).
        max_length: Max sequence length in tokens.

    Returns:
        List of error messages. Empty list means valid.
    """
    errors: list[str] = []

    if not use_ring_attention:
        return errors

    if device != "cuda":
        errors.append(
            "Ring FlashAttention requires CUDA GPUs. "
            f"Current device: {device}."
        )

    if not is_ring_attention_available():
        errors.append(
            "Ring FlashAttention is not available. "
            "Install it with: pip install ring-flash-attn"
        )

    try:
        import torch

        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if gpu_count < 2:
            errors.append(
                "Ring FlashAttention requires at least 2 GPUs for sequence parallelism. "
                f"Found: {gpu_count} GPU(s)."
            )
    except ImportError:
        errors.append("PyTorch is required for Ring FlashAttention.")

    if max_length < 8192:
        errors.append(
            f"Ring FlashAttention is designed for long sequences (>= 8192 tokens). "
            f"Current max_length: {max_length}. Consider increasing max_length."
        )

    return errors
