"""Training profile estimator — memory, speed, and recommendations."""

import math

# GPU memory lookup table (name → GB VRAM)
GPU_MEMORY: dict[str, int] = {
    "rtx3060": 12,
    "rtx3070": 8,
    "rtx3070ti": 8,
    "rtx3080": 10,
    "rtx3080ti": 12,
    "rtx3090": 24,
    "rtx4060": 8,
    "rtx4060ti": 16,
    "rtx4070": 12,
    "rtx4070ti": 12,
    "rtx4080": 16,
    "rtx4090": 24,
    "rtx5090": 32,
    "a10": 24,
    "a30": 24,
    "a40": 48,
    "a100": 80,
    "a100_40gb": 40,
    "h100": 80,
    "h200": 141,
    "l4": 24,
    "l40": 48,
    "l40s": 48,
    "t4": 16,
    "v100": 32,
}

# Known model architectures: model_size_b → (hidden_size, num_layers, intermediate_size)
_KNOWN_ARCHS: dict[float, tuple[int, int, int]] = {
    0.5: (896, 24, 4864),
    1.0: (2048, 22, 5632),
    1.5: (2048, 28, 5632),
    3.0: (3072, 28, 8192),
    7.0: (4096, 32, 11008),
    8.0: (4096, 32, 14336),
    13.0: (5120, 40, 13824),
    14.0: (5120, 40, 14336),
    32.0: (5120, 64, 27648),
    34.0: (8192, 48, 22016),
    70.0: (8192, 80, 28672),
}


def get_model_arch(model_name: str, model_params_b: float) -> dict:
    """Get model architecture estimates (hidden_size, num_layers).

    Uses known architectures when possible, otherwise estimates from param count.
    """
    # Find closest known architecture
    closest_size = min(_KNOWN_ARCHS.keys(), key=lambda sz: abs(sz - model_params_b))

    # Use closest if within 20% range
    if abs(closest_size - model_params_b) / max(model_params_b, 0.1) <= 0.2:
        hidden, layers, intermediate = _KNOWN_ARCHS[closest_size]
        return {
            "hidden_size": hidden,
            "num_layers": layers,
            "intermediate_size": intermediate,
        }

    # Estimate from param count using scaling laws
    # params ≈ 12 * num_layers * hidden_size^2 (rough transformer scaling)
    hidden_size = int(math.sqrt(model_params_b * 1e9 / (12 * 32)))
    # Round to nearest 128
    hidden_size = max(128, (hidden_size // 128) * 128)
    num_layers = int(model_params_b * 1e9 / (12 * hidden_size * hidden_size))
    num_layers = max(1, num_layers)
    intermediate_size = hidden_size * 3

    return {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "intermediate_size": intermediate_size,
    }


def estimate_model_memory(model_params_b: float, quantization: str) -> float:
    """Estimate model weight memory in GB.

    Args:
        model_params_b: Model size in billions of parameters.
        quantization: '4bit', '8bit', or 'none' (FP16).

    Returns:
        Estimated memory in GB.
    """
    bytes_per_param = {"4bit": 0.5, "8bit": 1.0, "none": 2.0}
    bpp = bytes_per_param.get(quantization, 2.0)
    return model_params_b * bpp


def estimate_trainable_params(
    model_params_b: float, lora_r: int, hidden_size: int
) -> int:
    """Estimate number of trainable LoRA parameters.

    LoRA adds r × hidden_size × 2 params per target module.
    Typical target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    """
    num_target_modules = 7  # typical for LLaMA-family
    arch = get_model_arch("", model_params_b)
    num_layers = arch["num_layers"]

    # Each module: r * hidden_size (A) + hidden_size * r (B) = 2 * r * hidden_size
    params_per_layer = num_target_modules * 2 * lora_r * hidden_size
    return params_per_layer * num_layers


def estimate_lora_memory(
    model_params_b: float, lora_r: int, lora_alpha: int
) -> float:
    """Estimate LoRA adapter memory in GB (FP16 for trainable params)."""
    arch = get_model_arch("", model_params_b)
    trainable = estimate_trainable_params(model_params_b, lora_r, arch["hidden_size"])
    # FP16 = 2 bytes per param
    return trainable * 2 / (1024**3)


def estimate_optimizer_memory(trainable_params: int, optimizer: str) -> float:
    """Estimate optimizer state memory in GB.

    Adam: 2 states per param (mean + variance) = 2 × param_bytes
    SGD: 1 state (momentum) = 1 × param_bytes
    8bit Adam: ~0.5 × normal Adam
    """
    bytes_per_param = 2  # FP16 trainable params

    if "8bit" in optimizer or "bnb" in optimizer:
        # 8-bit optimizer states
        state_multiplier = 1.0  # reduced from 2.0
    elif "sgd" in optimizer.lower():
        state_multiplier = 1.0  # only momentum
    else:
        # AdamW and similar: mean + variance
        state_multiplier = 2.0

    total_bytes = trainable_params * bytes_per_param * state_multiplier
    return total_bytes / (1024**3)


def estimate_activation_memory(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_layers: int,
    gradient_checkpointing: bool = False,
) -> float:
    """Estimate activation memory in GB.

    Activations scale with batch_size × seq_len × hidden_size × num_layers.
    Gradient checkpointing reduces this by ~sqrt(num_layers).
    """
    # ~2 bytes per activation value, ~4 activations per layer per position
    activation_bytes = batch_size * seq_len * hidden_size * 4 * 2

    if gradient_checkpointing:
        # Only store activations at checkpoints (sqrt reduction)
        effective_layers = max(1, int(math.sqrt(num_layers)))
    else:
        effective_layers = num_layers

    total_bytes = activation_bytes * effective_layers
    return total_bytes / (1024**3)


def estimate_total(
    model_name: str,
    model_params_b: float,
    quantization: str,
    lora_r: int,
    lora_alpha: int,
    batch_size: int,
    seq_len: int,
    optimizer: str,
    gradient_checkpointing: bool,
) -> dict:
    """Compute full memory profile.

    Returns dict with breakdown and total.
    """
    arch = get_model_arch(model_name, model_params_b)

    model_mem = estimate_model_memory(model_params_b, quantization)
    lora_mem = estimate_lora_memory(model_params_b, lora_r, lora_alpha)
    trainable = estimate_trainable_params(
        model_params_b, lora_r, arch["hidden_size"]
    )
    opt_mem = estimate_optimizer_memory(trainable, optimizer)
    act_mem = estimate_activation_memory(
        batch_size, seq_len, arch["hidden_size"], arch["num_layers"],
        gradient_checkpointing=gradient_checkpointing,
    )

    overhead = 1.5  # CUDA context + fragmentation
    total = model_mem + lora_mem + opt_mem + act_mem + overhead

    return {
        "model_name": model_name,
        "model_params_b": model_params_b,
        "trainable_params": trainable,
        "quantization": quantization,
        "model_memory_gb": round(model_mem, 2),
        "lora_memory_gb": round(lora_mem, 2),
        "optimizer_memory_gb": round(opt_mem, 2),
        "activation_memory_gb": round(act_mem, 2),
        "overhead_gb": overhead,
        "total_memory_gb": round(total, 2),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "gradient_checkpointing": gradient_checkpointing,
        "hidden_size": arch["hidden_size"],
        "num_layers": arch["num_layers"],
    }


def estimate_speed(
    model_params_b: float, quantization: str, batch_size: int
) -> float:
    """Estimate training tokens/sec (rough lookup-based).

    Based on typical A100 throughput for different model sizes.
    """
    # Base tokens/sec on A100 for different sizes (4bit, batch=4)
    base_speed: dict[float, float] = {
        1.0: 5000,
        3.0: 2500,
        7.0: 1200,
        8.0: 1100,
        13.0: 600,
        14.0: 550,
        32.0: 250,
        34.0: 230,
        70.0: 100,
    }

    # Find closest size
    closest_key = min(
        base_speed.keys(),
        key=lambda size: abs(size - model_params_b),
    )
    speed = base_speed[closest_key]

    # Adjust for quantization
    quant_factor = {"4bit": 1.0, "8bit": 0.8, "none": 0.5}
    speed *= quant_factor.get(quantization, 0.5)

    # Adjust for batch size (relative to base batch=4)
    speed *= min(batch_size / 4, 2.0)  # diminishing returns above 8

    return speed


def estimate_training_time(
    dataset_size: int, epochs: int, samples_per_sec: float
) -> float:
    """Estimate total training time in minutes.

    Returns float('inf') if samples_per_sec is 0.
    """
    if samples_per_sec <= 0:
        return float("inf")

    total_samples = dataset_size * epochs
    total_seconds = total_samples / samples_per_sec
    return total_seconds / 60


def recommend_batch_size(total_memory_gb: float, gpu_memory_gb: float) -> int:
    """Recommend batch size based on available GPU memory.

    Leaves headroom for memory spikes during training.
    """
    available = gpu_memory_gb - total_memory_gb
    if available <= 0:
        return 1

    # Each additional batch sample needs roughly (activation memory / current batch)
    # Simplified: available memory / ~1.5 GB per additional sample for 7B-class models
    extra_samples = int(available / 1.5)
    batch_size = max(1, 1 + extra_samples)

    # Clamp to power of 2
    if batch_size > 1:
        batch_size = 2 ** int(math.log2(batch_size))

    return min(batch_size, 64)


def recommend_gpu(total_memory_gb: float) -> list[str]:
    """Recommend GPUs that can fit the estimated memory.

    Returns list of GPU names that have enough VRAM.
    """
    compatible = []
    for name, vram in sorted(GPU_MEMORY.items(), key=lambda item: item[1]):
        if vram >= total_memory_gb * 1.1:  # 10% headroom
            compatible.append(f"{name.upper()} ({vram} GB)")

    if not compatible:
        compatible.append(
            f"No single GPU fits {total_memory_gb:.1f} GB — use multi-GPU (DeepSpeed/FSDP)"
        )

    return compatible
