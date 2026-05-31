"""Autopilot decision engine — pick task, quantization, PEFT, LR, epochs."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Literal, Mapping, Optional

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


# --- v0.53.1 #82 — pre-quantized base detection -----------------------------

# Quant Menu formats Autopilot can recommend back. Mirrors v0.38.0 + v0.40.5
# canonical strings; ``hqq:Nbit`` is the only parameterised entry.
_VALID_PREQUANT_FORMATS: frozenset[str] = frozenset({
    "gptq", "awq", "aqlm", "eetq", "fp8", "mxfp4",
})
# HQQ uses ``hqq:<N>bit`` where N ∈ {1, 2, 3, 4, 8}.
_HQQ_VALID_BITS: frozenset[int] = frozenset({1, 2, 3, 4, 8})

# Word-boundary matchers — substring won't match ``agptqa``.
_PREQUANT_NAME_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?:^|[^a-z0-9])gptq(?:[^a-z0-9]|$)", re.IGNORECASE), "gptq"),
    (re.compile(r"(?:^|[^a-z0-9])awq(?:[^a-z0-9]|$)", re.IGNORECASE), "awq"),
    (re.compile(r"(?:^|[^a-z0-9])aqlm(?:[^a-z0-9]|$)", re.IGNORECASE), "aqlm"),
    (re.compile(r"(?:^|[^a-z0-9])eetq(?:[^a-z0-9]|$)", re.IGNORECASE), "eetq"),
    # FP8 is matched separately so we can distinguish from generic "8"
    (re.compile(r"(?:^|[^a-z0-9])fp8(?:[^a-z0-9]|$)", re.IGNORECASE), "fp8"),
    (re.compile(r"(?:^|[^a-z0-9])mxfp4(?:[^a-z0-9]|$)", re.IGNORECASE), "mxfp4"),
)

_HQQ_NAME_RE = re.compile(
    r"(?:^|[^a-z0-9])hqq(?:[-_]?(?P<bits>[1-9])bit)?(?:[^a-z0-9]|$)",
    re.IGNORECASE,
)

_MAX_BASE_NAME_LEN = 512


def _check_base_name(name: object) -> str:
    if isinstance(name, bool):
        raise TypeError(f"base name must not be bool, got {name!r}")
    if not isinstance(name, str):
        raise TypeError(f"base name must be str, got {type(name).__name__}")
    if not name:
        raise ValueError("base name must be non-empty")
    if "\x00" in name:
        raise ValueError("base name must not contain null bytes")
    if len(name) > _MAX_BASE_NAME_LEN:
        raise ValueError(f"base name too long (max {_MAX_BASE_NAME_LEN} chars)")
    return name


def _detect_from_name(name: str) -> Optional[str]:
    """Return Quant Menu format string if the base name matches a known prefix."""
    hqq_match = _HQQ_NAME_RE.search(name)
    if hqq_match:
        bits = hqq_match.group("bits")
        if bits is None:
            return "hqq:4bit"
        bits_int = int(bits)
        if bits_int in _HQQ_VALID_BITS:
            return f"hqq:{bits_int}bit"
        return "hqq:4bit"  # fall back to safe default

    for pattern, fmt in _PREQUANT_NAME_PATTERNS:
        if pattern.search(name):
            return fmt
    return None


def _detect_from_config(hf_config: Any) -> Optional[str]:
    """Probe an HF-style ``config.json`` dict for a ``quantization_config`` block."""
    if not isinstance(hf_config, Mapping):
        return None
    qc = hf_config.get("quantization_config")
    if not isinstance(qc, Mapping):
        return None
    method_raw = qc.get("quant_method")
    if not isinstance(method_raw, str):
        return None
    method = method_raw.lower()
    if method == "hqq":
        bits = qc.get("bits")
        if isinstance(bits, bool) or not isinstance(bits, int):
            return "hqq:4bit"
        if bits in _HQQ_VALID_BITS:
            return f"hqq:{bits}bit"
        return "hqq:4bit"
    if method in _VALID_PREQUANT_FORMATS:
        return method
    # Map common aliases
    if method in {"bitsandbytes_4bit", "bnb_4bit", "nf4"}:
        return "4bit"
    if method in {"bitsandbytes_8bit", "bnb_8bit"}:
        return "8bit"
    return None


def detect_prequantized_format(
    name: object, hf_config: Any = None,
) -> Optional[str]:
    """Detect a pre-quantized base model's quant format.

    Returns a canonical Quant Menu format string (``gptq`` / ``awq`` /
    ``hqq:4bit`` / ``aqlm`` / ``eetq`` / ``fp8`` / ``mxfp4`` / ``4bit`` /
    ``8bit``) or ``None`` if no hint can be derived.

    Search order:
    1. ``hf_config`` ``quantization_config.quant_method`` (authoritative)
    2. Base-name regex with word-boundary anchoring (heuristic)
    """
    validated = _check_base_name(name)
    config_hit = _detect_from_config(hf_config)
    if config_hit is not None:
        return config_hit
    return _detect_from_name(validated)


def detect_prequantized_format_from_path(model_dir: object) -> Optional[str]:
    """Read ``<model_dir>/config.json`` and delegate to :func:`detect_prequantized_format`.

    Returns ``None`` on missing directory, missing file, or malformed JSON
    (matches v0.36.0 ``model_requires_trust_remote_code`` probe semantics).
    """
    if isinstance(model_dir, bool):
        raise TypeError("model_dir must not be bool")
    if not isinstance(model_dir, str):
        raise TypeError(
            f"model_dir must be str, got {type(model_dir).__name__}"
        )
    if not model_dir:
        return None
    if "\x00" in model_dir:
        return None
    # Soft-probe semantics: callers can pass an HF repo id OR a local path.
    # Only attempt the on-disk probe when ``model_dir`` resolves to a
    # directory under cwd (containment defence). Out-of-cwd local paths
    # silently fall through to name-only detection.
    import stat as _stat

    from soup_cli.utils.paths import is_under_cwd

    if not is_under_cwd(model_dir):
        return None
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        return None
    # Reject symlinks at the config target (TOCTOU defence; mirrors
    # v0.33.0 #22 / v0.43.0 / v0.46.0 / v0.47.0 policy).
    try:
        st = os.lstat(config_path)
    except OSError:
        return None
    if _stat.S_ISLNK(st.st_mode):
        return None
    try:
        with open(config_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return detect_prequantized_format(
        os.path.basename(os.path.normpath(model_dir)) or "model",
        data,
    )


def _validate_prequantized(value: object) -> str:
    if isinstance(value, bool):
        raise TypeError(f"prequantized must not be bool, got {value!r}")
    if not isinstance(value, str):
        raise TypeError(
            f"prequantized must be str, got {type(value).__name__}"
        )
    if not value:
        raise ValueError("prequantized must be non-empty")
    if "\x00" in value:
        raise ValueError("prequantized must not contain null bytes")
    canonical = value.lower()
    if canonical.startswith("hqq:"):
        # Validate hqq:Nbit shape
        tail = canonical.split(":", 1)[1]
        if not tail.endswith("bit"):
            raise ValueError(
                f"prequantized {value!r} invalid HQQ shape; expected hqq:Nbit"
            )
        bits_str = tail[: -len("bit")]
        if not bits_str.isdigit() or int(bits_str) not in _HQQ_VALID_BITS:
            raise ValueError(
                f"prequantized {value!r} HQQ bits invalid; "
                f"expected one of {sorted(_HQQ_VALID_BITS)}"
            )
        return canonical
    if canonical in _VALID_PREQUANT_FORMATS or canonical in {"4bit", "8bit"}:
        return canonical
    raise ValueError(
        f"prequantized {value!r} not a recognised Quant Menu format. "
        f"Supported: {sorted(_VALID_PREQUANT_FORMATS)} | 4bit | 8bit | hqq:Nbit"
    )


def decide_quantization(
    model_params_b: float,
    vram_gb: float,
    prequantized: Optional[str] = None,
) -> str:
    """Pick a quantization tier from model size vs available VRAM.

    v0.53.1 #82 — when ``prequantized`` is set (e.g. ``"gptq"`` / ``"awq"`` /
    ``"hqq:4bit"``), it takes precedence over the VRAM-based heuristic so we
    don't silently stack a fresh BNB 4-bit on top of an already-quantized base.
    """
    if prequantized is not None:
        return _validate_prequantized(prequantized)

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


def decide_peft(
    data_size: int, model_size_b: float, vram_gb: float,
) -> dict[str, Any]:
    """Pick a LoRA rank and settings based on dataset + model + VRAM."""
    if data_size < 1000:
        rank = 8
    elif data_size < 10_000:
        rank = 16
    else:
        # rank capped at 32 for all datasets >= 10k samples
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
