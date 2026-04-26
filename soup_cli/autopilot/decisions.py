"""Autopilot decision engine — pick task, quantization, PEFT, LR, epochs."""

from __future__ import annotations

import re
from typing import Any, Literal

GOAL_TO_TASK: dict[str, str] = {
    "chat": "sft",
    "code": "sft",
    "classification": "sft",
    "tool-calling": "sft",
    "reasoning": "grpo",
    "alignment": "dpo",
    "domain-adapt": "pretrain",
}

MIN_GPU_BUDGET_GB = 1.0
MAX_GPU_BUDGET_GB = 1024.0


def decide_task(goal: str, dataset_profile: Any = None) -> str:
    """Map a high-level goal to a Soup training task.

    ``dataset_profile`` is reserved for future heuristics (e.g. detecting
    preference pairs vs plaintext) but currently unused.
    """
    del dataset_profile  # reserved for future heuristics
    if goal not in GOAL_TO_TASK:
        raise ValueError(
            f"Unknown goal '{goal}'. Options: {', '.join(GOAL_TO_TASK.keys())}"
        )
    return GOAL_TO_TASK[goal]


def decide_quantization(model_params_b: float, vram_gb: float) -> str:
    """Pick a quantization tier from model size vs available VRAM."""
    # Rough: 1 param byte each in 8bit, 0.5 in 4bit, 2 in fp16
    model_gb_fp16 = model_params_b * 2.0
    if vram_gb >= 2.5 * model_gb_fp16:
        return "none"
    if vram_gb >= 1.5 * model_gb_fp16:
        return "8bit"
    if vram_gb >= 0.6 * model_gb_fp16:
        return "4bit"
    raise ValueError(
        f"Model too large for VRAM budget: {model_params_b}B model "
        f"needs at least {0.6 * model_gb_fp16:.1f}GB in 4bit, got {vram_gb:.1f}GB. "
        "Try a smaller model or increase --gpu-budget."
    )


def decide_peft(data_size: int, model_size_b: float, vram_gb: float) -> dict:
    """Pick a LoRA rank and settings based on dataset + model + VRAM."""
    if data_size < 1000:
        rank = 8
    elif data_size < 10_000:
        rank = 16
    elif data_size < 100_000:
        rank = 32
    else:
        rank = 32
    alpha = rank * 2
    use_dora = data_size > 100_000 and vram_gb >= 2.0 * model_size_b
    return {
        "r": rank,
        "alpha": alpha,
        "use_dora": use_dora,
    }


def decide_batch_size(
    model_size_b: float,
    vram_gb: float,
    max_length: int,
    quantization: str,
) -> tuple[int, int]:
    """Return (per-device batch_size, gradient_accumulation_steps)."""
    from soup_cli.utils.gpu import estimate_batch_size

    batch = estimate_batch_size(
        model_params_b=model_size_b,
        seq_length=max_length,
        gpu_memory_bytes=int(vram_gb * 1024**3),
        quantization=quantization,
        lora_r=16,
    )
    batch = max(1, batch)
    target_effective_batch = 32
    grad_accum = max(1, target_effective_batch // batch)
    return batch, grad_accum


def decide_lr(rank: int, quantization: str) -> float:
    """Scale LR by LoRA rank and quantization."""
    base = {8: 3e-4, 16: 2e-4, 32: 1e-4}.get(rank, 2e-4)
    if quantization == "4bit":
        base *= 0.8
    return round(base, 6)


def decide_epochs(num_samples: int) -> int:
    """Pick epochs by dataset size (diminishing returns above 50k)."""
    if num_samples < 500:
        return 5
    if num_samples < 5000:
        return 3
    if num_samples < 50_000:
        return 2
    return 1


def decide_max_length(p95_tokens: int, model_context: int) -> int:
    """Pick max_length from dataset p95 × 1.1 clamped to model context."""
    target = int(p95_tokens * 1.1)
    if target < 256:
        target = 256
    return min(target, model_context)


def decide_performance_flags(
    gpu_name: str,
    compute_capability: float,
    max_length: int = 2048,
    vram_headroom_gb: float = 0.0,
) -> dict:
    """Return perf flags for FlashAttention / Liger / gradient_checkpointing.

    - ``use_flash_attn`` and ``use_liger`` only enabled on Ampere or newer.
    - ``gradient_checkpointing`` enabled when the sequence is long (>8k tokens)
      or VRAM headroom is tight (<4GB), to avoid OOM. Otherwise left off so we
      don't pay the recompute cost when there's plenty of headroom.
    """
    del gpu_name  # reserved for future GPU-specific overrides
    is_ampere_or_newer = compute_capability >= 8.0
    long_sequence = max_length > 8192
    tight_vram = 0.0 < vram_headroom_gb < 4.0
    return {
        "use_flash_attn": bool(is_ampere_or_newer),
        "use_liger": bool(is_ampere_or_newer),
        "gradient_checkpointing": bool(long_sequence or tight_vram),
    }


def decide_warmup(
    num_examples: int, batch_size: int, grad_accum: int, epochs: int,
    ratio: float = 0.03,
) -> int:
    """Wrap ``compute_warmup_steps`` for the autopilot decision flow."""
    from soup_cli.utils.warmup import compute_warmup_steps

    return compute_warmup_steps(
        num_examples=num_examples,
        batch_size=batch_size,
        grad_accum=grad_accum,
        epochs=epochs,
        ratio=ratio,
    )


def decide_mixed_precision(
    model_name: str, compute_capability: float,
) -> Literal["bf16", "fp16", "no"]:
    """Wrap ``pick_mixed_precision`` for the autopilot decision flow."""
    from soup_cli.utils.mixed_precision import pick_mixed_precision

    return pick_mixed_precision(model_name, compute_capability)


_BUDGET_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([gG][bB]?)?\s*$")


def parse_gpu_budget(raw: str) -> float:
    """Parse a GPU budget string like ``24GB`` or ``80`` into GB as float."""
    if not isinstance(raw, str):
        raise ValueError("gpu_budget must be a string")
    match = _BUDGET_RE.match(raw)
    if not match:
        raise ValueError(f"Invalid gpu_budget '{raw}' (expected e.g. '24GB')")
    value = float(match.group(1))
    if value < MIN_GPU_BUDGET_GB or value > MAX_GPU_BUDGET_GB:
        raise ValueError(
            f"gpu_budget {value}GB out of bounds [{MIN_GPU_BUDGET_GB}, {MAX_GPU_BUDGET_GB}]"
        )
    return value
