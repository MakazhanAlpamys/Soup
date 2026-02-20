"""GPU detection, memory calculation, and auto batch size."""

import math


def detect_device() -> tuple[str, str]:
    """Detect available device. Returns (device_string, human_name)."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return "cuda", name
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", "Apple Silicon (MPS)"
    except ImportError:
        pass

    return "cpu", "CPU (no GPU detected)"


def get_gpu_info() -> dict:
    """Get GPU memory info."""
    try:
        import torch

        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_mem
            total_gb = total / (1024**3)
            return {
                "memory_total": f"{total_gb:.1f} GB",
                "memory_total_bytes": total,
                "gpu_count": torch.cuda.device_count(),
            }
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS doesn't expose memory easily, estimate from system
            return {
                "memory_total": "shared (Apple Silicon)",
                "memory_total_bytes": 0,
                "gpu_count": 1,
            }
    except ImportError:
        pass

    return {
        "memory_total": "N/A (CPU mode)",
        "memory_total_bytes": 0,
        "gpu_count": 0,
    }


def estimate_batch_size(
    model_params_b: float,
    seq_length: int,
    gpu_memory_bytes: int,
    quantization: str = "4bit",
    lora_r: int = 64,
) -> int:
    """Estimate max batch size that fits in GPU memory.

    Conservative estimate — better to start smaller and gradient accumulate.
    """
    if gpu_memory_bytes == 0:
        return 1  # CPU fallback

    gpu_gb = gpu_memory_bytes / (1024**3)

    # Rough memory per param based on quantization
    bytes_per_param = {"4bit": 0.5, "8bit": 1.0, "none": 2.0}  # FP16
    bpp = bytes_per_param.get(quantization, 2.0)

    # Model memory (static)
    model_mem_gb = model_params_b * bpp

    # LoRA trainable params (usually ~1-3% of total)
    lora_ratio = min(lora_r * 2 / 4096, 0.05)  # rough estimate
    trainable_mem_gb = model_params_b * 2 * lora_ratio  # FP16 for trainable

    # Optimizer states (Adam: 2x params)
    optimizer_mem_gb = trainable_mem_gb * 2

    # Available for activations
    overhead_gb = 1.5  # CUDA overhead, fragmentation
    available_gb = gpu_gb - model_mem_gb - trainable_mem_gb - optimizer_mem_gb - overhead_gb

    if available_gb <= 0:
        return 1

    # Rough activation memory per sample per token
    # ~2 bytes per hidden dim per layer per token for a transformer
    activation_per_sample_gb = (seq_length * model_params_b * 0.001)  # very rough
    activation_per_sample_gb = max(activation_per_sample_gb, 0.5)  # minimum 0.5 GB

    batch_size = max(1, int(available_gb / activation_per_sample_gb))
    # Clamp to power of 2 (common practice)
    batch_size = 2 ** int(math.log2(batch_size)) if batch_size > 1 else 1

    return min(batch_size, 32)  # cap at 32


def model_size_from_name(model_name: str) -> float:
    """Guess model size in billions from model name."""
    name_lower = model_name.lower()

    size_markers = [
        ("70b", 70), ("65b", 65), ("34b", 34), ("33b", 33),
        ("13b", 13), ("8b", 8), ("7b", 7), ("3b", 3),
        ("1.5b", 1.5), ("1b", 1), ("0.5b", 0.5),
    ]

    for marker, size in size_markers:
        if marker in name_lower:
            return size

    return 7.0  # default guess
