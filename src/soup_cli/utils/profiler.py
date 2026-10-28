"""Training profile estimator — memory, speed, and recommendations."""

import math
import re
from typing import Any, Optional

# GPU memory lookup table (name → GB VRAM). Keys are normalized by
# ``normalize_gpu_key``. Every laptop part has its own key, including those with
# the same memory as the desktop card: that is what lets the device name torch
# reports ("... Laptop GPU") resolve at all. A part that ships in two sizes gets a
# ``_<N>gb`` key for the size its name carries ("RTX 3050 6GB Laptop GPU"); where
# one name covers both sizes, the plain key is the larger one.
GPU_MEMORY: dict[str, int] = {
    "rtx3050": 8,
    "rtx3050_6gb": 6,
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
    "rtx5060": 8,
    "rtx5060ti": 16,
    "rtx5060ti_8gb": 8,
    "rtx5070": 12,
    "rtx5070ti": 16,
    "rtx5080": 16,
    "rtx5090": 32,
    # NVIDIA's 30-series laptop specification: "RTX 3050: 4 GB or 6 GB" and
    # "RTX 3080: 8 GB or 16 GB". The driver names the 3050 by size (4GB, 6GB)
    # or not at all (the original 4 GB part); both 3080 sizes share one name.
    "rtx3050laptop": 4,
    "rtx3050laptop_4gb": 4,
    "rtx3050laptop_6gb": 6,
    "rtx3050tilaptop": 4,
    "rtx3060laptop": 6,
    "rtx3070laptop": 8,
    "rtx3070tilaptop": 8,
    "rtx3080laptop": 16,
    "rtx3080laptop_8gb": 8,
    "rtx3080tilaptop": 16,
    "rtx4050laptop": 6,
    "rtx4060laptop": 8,
    "rtx4070laptop": 8,
    "rtx4080laptop": 12,
    "rtx4090laptop": 16,
    "rtx5060laptop": 8,
    "rtx5070laptop": 8,
    "rtx5070tilaptop": 12,
    "rtx5080laptop": 16,
    "rtx5090laptop": 24,
    "a10": 24,
    "a30": 24,
    "a40": 48,
    "a100": 80,
    "a100_40gb": 40,
    "h100": 80,
    "h200": 141,
    # DGX B200: 1,440 GB across eight GPUs.
    "b200": 180,
    "l4": 24,
    "l40": 48,
    "l40s": 48,
    "t4": 16,
    "v100": 32,
}

# What ``soup profile`` and the MCP ``profile`` tool fall back to when there is
# no --gpu and no device is detected. Always reported as assumed, not measured.
ASSUMED_GPU_MEMORY_GB = 24.0

_GPU_NAME_NOISE = re.compile(r"\b(nvidia|geforce|tesla|gpu)\b")
# A memory size written as its own word, which the table spells as a suffix.
_GPU_NAME_SIZE = re.compile(r"[\s-](\d+)\s*gb\b")


def normalize_gpu_key(name: str) -> str:
    """Map a ``--gpu`` value or a device name torch reports to a ``GPU_MEMORY`` key.

    ``"NVIDIA GeForce RTX 5070 Laptop GPU"`` and ``"rtx 5070 laptop"`` both
    become ``"rtx5070laptop"``; ``"RTX-4090"`` becomes ``"rtx4090"``. A size
    word moves to the end as a suffix: ``"NVIDIA GeForce RTX 3050 6GB Laptop
    GPU"`` becomes ``"rtx3050laptop_6gb"``. A size whose suffixed key does not
    exist but which the plain key holds resolves to the plain key: ``"rtx 3080
    laptop 16gb"`` becomes ``"rtx3080laptop"``.
    """
    key = _GPU_NAME_NOISE.sub("", name.lower())
    size = _GPU_NAME_SIZE.search(key)
    if size is None:
        return re.sub(r"[\s-]+", "", key)
    key = re.sub(r"[\s-]+", "", key[: size.start()] + " " + key[size.end() :])
    suffixed = f"{key}_{size.group(1)}gb"
    if suffixed not in GPU_MEMORY and str(GPU_MEMORY.get(key)) == size.group(1):
        return key
    return suffixed


def get_model_arch(model_name: str, model_params_b: float) -> dict:
    """Get model architecture estimates (hidden_size, num_layers).

    A guess from the parameter count alone (``model_shape.guessed_arch``). The VRAM
    estimate reads the real dims from ``config.json`` whenever it can (#2049).
    """
    from soup_cli.utils.model_shape import guessed_arch

    hidden, layers, intermediate = guessed_arch(model_params_b)
    return {
        "hidden_size": hidden,
        "num_layers": layers,
        "intermediate_size": intermediate,
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
    peft: str = "lora",
) -> dict:
    """Compute full memory profile from a parameter count.

    The shared estimate (``utils/vram_estimate.py``) over a shape guessed from
    ``model_params_b``, so the vocabulary and the logits term are unknown here.
    ``soup profile`` reads the model's own shape instead (:func:`profile_from_config`).
    """
    from soup_cli.utils.hardware_fit import (
        _VALID_OPTIMIZERS,
        CONFIG_QUANT_TO_FIT,
        HardwareFitInput,
        estimate_peak_vram_gb,
        validate_batch_size,
        validate_seq_len,
    )

    quant = CONFIG_QUANT_TO_FIT.get(quantization, "none")
    if peft == "full":
        mode = "full"
    else:
        mode = "qlora" if quant == "4bit" else "lora"
    inp = HardwareFitInput(
        params_b=float(model_params_b),
        seq_len=validate_seq_len(min(max(int(seq_len), 64), 1_048_576)),
        batch_size=validate_batch_size(min(max(int(batch_size), 1), 1024)),
        optimizer=optimizer if optimizer in _VALID_OPTIMIZERS else "adamw_torch",
        quant=quant,
        peft=mode,
        gradient_checkpointing=gradient_checkpointing,
        lora_r=int(lora_r),
    )
    result = profile_result(
        estimate_peak_vram_gb(inp),
        model_name=model_name,
        gradient_checkpointing=gradient_checkpointing,
    )
    result["quantization"] = quantization
    result["peft"] = peft
    return result


def checkpointing_tier(value: Any) -> str:
    """``off``, ``full``, ``medium`` or ``selective``: the checkpointing a run is priced with.

    ``value`` is what the estimate was given: ``False`` or a tier name, or a config's
    ``True`` / ``"auto"``, which price as ``full``.
    """
    if not value:
        return "off"
    return value if value in ("medium", "selective") else "full"


def profile_result(
    breakdown: Any, *, model_name: str, gradient_checkpointing: Any = None
) -> dict:
    """``soup profile``'s result fields from a ``hardware_fit.VRAMBreakdown``.

    The old field names keep their meaning (``lora_memory_gb`` is the adapters);
    ``logits_memory_gb``, ``dequant_memory_gb``, the exact byte counts, the shape's
    source and the notes are new with #2049. GB is 1e9 bytes.
    ``gradient_checkpointing`` is reported as the config wrote it (``True``, ``False``,
    ``"auto"`` or a tier), as it always was; ``gradient_checkpointing_tier`` is the
    tier the estimate priced, which ``auto`` and the task resolve.
    """
    est = breakdown.estimate
    shape, run = est.shape, est.run
    return {
        "model_name": model_name,
        "model_params_b": round(shape.params / 1e9, 3),
        "trainable_params": int(
            shape.params if run.peft == "full" else est.terms["adapters"] // 4
        ),
        "quantization": run.quant,
        "peft": run.peft,
        "lora_targets": list(est.lora_targets),
        "model_memory_gb": round(breakdown.weights_gb, 2),
        "lora_memory_gb": round(breakdown.adapters_gb, 2),
        "gradients_memory_gb": round(breakdown.gradients_gb, 2),
        "optimizer_memory_gb": round(breakdown.optimizer_gb, 2),
        "activation_memory_gb": round(breakdown.activations_gb, 2),
        "logits_memory_gb": round(breakdown.logits_gb, 2),
        "dequant_memory_gb": round(breakdown.dequant_gb, 2),
        "overhead_gb": round(breakdown.overhead_gb, 2),
        "total_memory_gb": round(breakdown.total_gb, 2),
        "total_memory_bytes": est.total_bytes,
        "memory_terms_bytes": dict(est.terms),
        "batch_size": run.batch_size,
        "seq_len": run.seq_len,
        "gradient_checkpointing": (
            run.gradient_checkpointing
            if gradient_checkpointing is None
            else gradient_checkpointing
        ),
        "gradient_checkpointing_tier": checkpointing_tier(run.gradient_checkpointing),
        "hidden_size": shape.hidden_size,
        "num_layers": shape.num_layers,
        "vocab_size": shape.vocab_size,
        "shape_source": shape.shape_source,
        "params_source": shape.params_source,
        "notes": list(est.notes),
    }


def profile_from_config(
    cfg: Any, *, allow_network: bool = True, gpu_memory_gb: Optional[float] = None
) -> Optional[dict]:
    """The memory profile of a ``SoupConfig``, from the model's own shape.

    Built from the same input as the train pre-flight
    (``hardware_fit.hardware_fit_input_from_config``), so the two report the same
    peak for the same config (#2049). ``None`` when the run cannot be priced.
    ``gpu_memory_gb`` (GiB) is the card ``gradient_checkpointing: auto`` resolves on.
    """
    from soup_cli.utils.hardware_fit import (
        estimate_peak_vram_gb,
        hardware_fit_input_from_config,
    )

    inp = hardware_fit_input_from_config(
        cfg, allow_network=allow_network, profile_defaults=True, gpu_memory_gb=gpu_memory_gb
    )
    if inp is None:
        return None
    result = profile_result(
        estimate_peak_vram_gb(inp),
        model_name=cfg.base,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
    )
    result["quantization"] = cfg.training.quantization
    return result


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

    Leaves headroom for memory spikes during training. ``total_memory_gb`` is the
    estimate's GB (1e9 bytes); ``gpu_memory_gb`` is a card's GB as the table and the
    driver give it (GiB), so it is converted before the two are compared.
    """
    available = gpu_memory_gb * 1024**3 / 1e9 - total_memory_gb
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
    from soup_cli.utils.vram_estimate import fit_verdict

    compatible = []
    for name, vram in sorted(GPU_MEMORY.items(), key=lambda item: item[1]):
        # In bytes, as ``soup profile``'s fit is: the table's GB are GiB, the
        # estimate's are 1e9 bytes. Only a card that fits with the 10% margin.
        if fit_verdict(total_memory_gb * 1e9, vram * 1024**3) == "fits":
            compatible.append(f"{name.upper()} ({vram} GB)")

    if not compatible:
        compatible.append(
            f"No single GPU fits {total_memory_gb:.1f} GB -- use multi-GPU (DeepSpeed/FSDP)"
        )

    return compatible
