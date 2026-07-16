"""GPU detection, memory calculation, and auto batch size."""

import json
import math
import os
import re

# A safetensors file starts with a u64 little-endian header length, then a
# JSON header carrying each tensor's dtype + shape. Reading it costs a few
# KB — the weights themselves are never touched.
_ST_HEADER_LEN_BYTES = 8
_MAX_ST_HEADER_BYTES = 100 * 1024 * 1024  # sanity bound on a crafted file


def _params_from_local_safetensors(path: str) -> float | None:
    """EXACT parameter count (billions) for a local checkpoint, or None.

    A local directory's NAME often carries no size marker at all — a model
    merged to ``./denseA`` or ``./out`` looks nameless to the size guesser,
    which then falls back to the 7B default and makes the hardware-fit gate
    refuse to train a 135M model. The checkpoint knows its own size, so ask
    it instead of guessing.
    """
    try:
        if not os.path.isdir(path):
            return None
        shards = [
            entry.path
            for entry in os.scandir(path)
            if entry.is_file() and entry.name.endswith(".safetensors")
        ]
        if not shards:
            return None
        total = 0
        for shard in shards:
            with open(shard, "rb") as handle:
                raw = handle.read(_ST_HEADER_LEN_BYTES)
                if len(raw) < _ST_HEADER_LEN_BYTES:
                    return None
                header_len = int.from_bytes(raw, "little")
                if header_len <= 0 or header_len > _MAX_ST_HEADER_BYTES:
                    return None
                header = json.loads(handle.read(header_len))
            if not isinstance(header, dict):
                return None
            for name, meta in header.items():
                if name == "__metadata__" or not isinstance(meta, dict):
                    continue
                shape = meta.get("shape")
                if not isinstance(shape, list) or not shape:
                    continue
                numel = 1
                for dim in shape:
                    if not isinstance(dim, int) or dim < 0:
                        return None
                    numel *= dim
                total += numel
        return (total / 1e9) if total else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None  # unreadable/crafted -> fall back to the name guess


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
            total = torch.cuda.get_device_properties(0).total_memory
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
    """Model size in billions: exact for a local checkpoint, else guessed.

    A LOCAL path is measured, not guessed — `soup merge` writes directories
    like ``./denseA`` whose name carries no size marker, so the name-based
    fallback called them 7B and the hardware-fit gate refused to train a
    135M model. That blocked merge -> train-from-merged, the ordinary
    continual-learning flow (found by the v0.71.36 replay smoke; same class
    as the v0.71.32 whisper and v0.71.33 "M suffix" fixes, which were both
    the name guess over-predicting a small model).
    """
    exact = _params_from_local_safetensors(model_name)
    if exact is not None:
        return exact

    name_lower = model_name.lower()

    # Whisper ASR checkpoints carry the size in the name suffix, not an "Nb"
    # token — check these first so a 39M whisper-tiny isn't mistaken for the
    # 7B default (v0.71.32: the default guess blocked ASR training on the
    # hardware-fit gate).
    whisper_markers = [
        ("whisper-large", 1.55), ("whisper-medium", 0.769),
        ("whisper-small", 0.244), ("whisper-base", 0.074),
        ("whisper-tiny", 0.039),
    ]
    for marker, size in whisper_markers:
        if marker in name_lower:
            return size

    # Longer markers first: "1.7b" contains "7b", so a naive scan would call a
    # 1.7B model a 7B one (and over-predict its VRAM by 4x).
    size_markers = [
        ("70b", 70), ("65b", 65), ("34b", 34), ("33b", 33),
        ("13b", 13), ("8b", 8), ("3b", 3),
        ("1.5b", 1.5), ("1.7b", 1.7), ("0.5b", 0.5), ("0.6b", 0.6),
        ("7b", 7), ("1b", 1),
    ]

    for marker, size in size_markers:
        if marker in name_lower:
            return size

    # Sub-billion checkpoints carry their size in MILLIONS (SmolLM2-135M,
    # SmolVLM-256M, ...). Without this they fell through to the 7B default and
    # the hardware-fit gate refused to train them — which blocked `soup draft`
    # for exactly the tiny models drafts are made of (v0.71.33 live smoke;
    # same class as the v0.71.32 whisper fix).
    #
    # Checked AFTER the "b" markers on purpose: `Qwen2.5-7B-Instruct-1M` is a
    # 7B model with a 1M *context*, not a 1M-parameter model.
    million = re.search(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)m(?![a-z0-9])", name_lower)
    if million:
        return float(million.group(1)) / 1000.0

    return 7.0  # default guess


def get_compute_dtype():
    """Return the best compute dtype for the current device.

    Uses bfloat16 on CUDA GPUs that support it, float16 otherwise.
    On CPU, uses float32 to avoid dtype mismatch errors.
    """
    import torch

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32
