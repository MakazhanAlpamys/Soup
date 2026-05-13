"""Long-context fine-tuning utilities — 128k+ token support.

Configures RoPE (Rotary Position Embedding) scaling to extend model context
windows beyond their pre-training length. Supports multiple scaling strategies:

- linear: Simple linear interpolation (PI) — good baseline
- dynamic: NTK-aware Dynamic scaling — better for large extensions
- yarn: YaRN (Yet another RoPE extensioN) — best quality for 4-8x extension
  (v0.49.0 Part A — math kernel + config-emit; HF Transformers owns the
  actual rotation under the hood)
- longrope: LongRoPE — progressive extension with search-based factors
- llama3: Llama 3.1 frequency-band NTK-aware scaling (v0.49.0 Part D)

Also handles gradient checkpointing configuration for memory efficiency
when training on very long sequences.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

# Supported RoPE scaling methods (v0.49.0 adds "llama3").
ROPE_SCALING_TYPES = ("linear", "dynamic", "yarn", "longrope", "llama3")

# Default context lengths for known model families.
MODEL_DEFAULT_CONTEXT: dict[str, int] = {
    "llama-3": 8192,
    "llama-2": 4096,
    "mistral": 32768,
    "mixtral": 32768,
    "qwen2": 32768,
    "qwen3": 32768,
    "phi-3": 4096,
    "phi-4": 16384,
    "gemma": 8192,
    "gemma-2": 8192,
    "deepseek": 4096,
    "codellama": 16384,
}

# v0.49.0 Part D — Llama 3.1 NTK-aware defaults.
LLAMA3_DEFAULT_SCALE_FACTOR: float = 8.0
LLAMA3_DEFAULT_LOW_FREQ_FACTOR: float = 1.0
LLAMA3_DEFAULT_HIGH_FREQ_FACTOR: float = 4.0
LLAMA3_DEFAULT_OLD_CONTEXT_LEN: int = 8192


def get_model_default_context(model_name: str) -> int:
    """Estimate the default context length for a model based on its name."""
    model_lower = model_name.lower()
    for family, ctx_len in MODEL_DEFAULT_CONTEXT.items():
        if family in model_lower:
            return ctx_len
    # Conservative default for unknown models.
    return 4096


# ---------------------------------------------------------------------------
# YaRN math kernel (v0.49.0 Part A)
# ---------------------------------------------------------------------------


def _finite_positive(value: Any, name: str) -> float:
    """Reject bool/NaN/Inf/<=0 with a typed error (mirrors v0.41.0 Part B)."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a real number, not bool")
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real number") from exc
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite (got {v!r})")
    if v <= 0.0:
        raise ValueError(f"{name} must be > 0 (got {v!r})")
    return v


def yarn_find_correction_dim(
    *,
    num_rotations: float,
    dim: int,
    base: float = 10000.0,
    max_position_embeddings: int,
) -> float:
    """Inverse of the rotation count for a given embedding dimension index.

    Mirrors the upstream YaRN reference implementation
    (https://arxiv.org/abs/2309.00071 §3.4):

        dim_idx = (dim * ln(L / (n * 2π))) / (2 * ln(base))

    Args:
        num_rotations: Rotation count cutoff (``beta_fast`` or ``beta_slow``).
        dim: Embedding dimension per head.
        base: RoPE base (``theta``); typically 10_000 for Llama / Mistral.
        max_position_embeddings: Original model context length.
    """
    num_rotations = _finite_positive(num_rotations, "num_rotations")
    if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
        raise ValueError(f"dim must be a positive int (got {dim!r})")
    base = _finite_positive(base, "base")
    if (
        isinstance(max_position_embeddings, bool)
        or not isinstance(max_position_embeddings, int)
        or max_position_embeddings <= 0
    ):
        raise ValueError(
            f"max_position_embeddings must be a positive int (got {max_position_embeddings!r})"
        )
    return (dim * math.log(max_position_embeddings / (num_rotations * 2.0 * math.pi))) / (
        2.0 * math.log(base)
    )


def yarn_find_correction_range(
    *,
    beta_fast: float,
    beta_slow: float,
    dim: int,
    base: float = 10000.0,
    max_position_embeddings: int,
) -> tuple[int, int]:
    """Return clamped ``(low, high)`` dim indices for YaRN piecewise interpolation."""
    low = math.floor(
        yarn_find_correction_dim(
            num_rotations=beta_fast,
            dim=dim,
            base=base,
            max_position_embeddings=max_position_embeddings,
        )
    )
    high = math.ceil(
        yarn_find_correction_dim(
            num_rotations=beta_slow,
            dim=dim,
            base=base,
            max_position_embeddings=max_position_embeddings,
        )
    )
    half = dim // 2
    low = max(0, min(low, half))
    high = max(0, min(high, half))
    if high <= low:
        # Disambiguate so the ramp-mask denominator is non-zero.
        high = min(low + 1, half)
        if high <= low:
            low = max(0, high - 1)
    return low, high


def yarn_linear_ramp_mask(*, low: int, high: int, dim: int) -> list[float]:
    """Return the YaRN piecewise-linear ramp mask of length ``dim``.

    Each element is clamped to ``[0, 1]``; ``low == high`` is auto-disambiguated
    to avoid a div-by-zero in the upstream formula
    ``(idx - low) / (high - low)``.
    """
    if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
        raise ValueError(f"dim must be a positive int (got {dim!r})")
    if isinstance(low, bool) or isinstance(high, bool):
        raise ValueError("low/high must not be bool")
    if not isinstance(low, int) or not isinstance(high, int):
        raise ValueError("low/high must be ints")
    if low < 0 or high < 0:
        raise ValueError("low/high must be non-negative")
    if high <= low:
        high = low + 1
    denom = float(high - low)
    return [max(0.0, min(1.0, (idx - low) / denom)) for idx in range(dim)]


def yarn_get_mscale(factor: float) -> float:
    """Return the YaRN attention temperature multiplier ``m_scale``.

    Per the YaRN paper §3.5, ``m_scale = 0.1 * ln(s) + 1`` for ``s > 1``;
    factors at or below 1 produce no scaling, so we clamp to ``1.0``.
    """
    if isinstance(factor, bool):
        raise ValueError("factor must be a real number, not bool")
    try:
        f = float(factor)
    except (TypeError, ValueError) as exc:
        raise ValueError("factor must be a real number") from exc
    if not math.isfinite(f):
        # Stay consistent with `_finite_positive` — explicit rejection over a
        # silent identity fallback (python-review LOW finding).
        raise ValueError(f"factor must be finite (got {f!r})")
    if f <= 1.0:
        return 1.0
    return 0.1 * math.log(f) + 1.0


# ---------------------------------------------------------------------------
# Llama 3.1 NTK-aware kernel (v0.49.0 Part D)
# ---------------------------------------------------------------------------


def scale_inv_freq_llama3(
    *,
    inv_freq: float,
    scale_factor: float = LLAMA3_DEFAULT_SCALE_FACTOR,
    low_freq_factor: float = LLAMA3_DEFAULT_LOW_FREQ_FACTOR,
    high_freq_factor: float = LLAMA3_DEFAULT_HIGH_FREQ_FACTOR,
    old_context_len: int = LLAMA3_DEFAULT_OLD_CONTEXT_LEN,
) -> float:
    """Apply Llama 3.1 NTK-aware scaling to a single ``inv_freq`` value.

    Replicates Unsloth ``models/llama.py:1853`` / HF transformers ``modeling_rope_utils``
    ``_compute_llama3_parameters``. The function classifies the frequency by its
    wavelength relative to ``old_context_len``:

      * wavelength < ``high_freq_wavelen``  → unchanged
      * wavelength > ``low_freq_wavelen``   → divided by ``scale_factor``
      * in between                          → smooth linear blend
    """
    for _name, _val in (
        ("inv_freq", inv_freq),
        ("scale_factor", scale_factor),
        ("low_freq_factor", low_freq_factor),
        ("high_freq_factor", high_freq_factor),
    ):
        if isinstance(_val, bool):
            raise ValueError(f"{_name} must be a real number, not bool")
        if not isinstance(_val, (int, float)) or not math.isfinite(float(_val)):
            raise ValueError(f"{_name} must be finite (got {_val!r})")
    if isinstance(old_context_len, bool) or not isinstance(old_context_len, int):
        raise ValueError(f"old_context_len must be an int (got {old_context_len!r})")
    if scale_factor <= 1.0:
        raise ValueError(f"scale_factor must be > 1 (got {scale_factor!r})")
    if high_freq_factor <= low_freq_factor:
        raise ValueError(
            f"high_freq_factor ({high_freq_factor}) must be greater than "
            f"low_freq_factor ({low_freq_factor})"
        )
    if old_context_len <= 0:
        raise ValueError(f"old_context_len must be > 0 (got {old_context_len})")

    inv_freq_f = float(inv_freq)
    if inv_freq_f <= 0.0:
        return inv_freq_f

    wavelen = 2.0 * math.pi / inv_freq_f
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    if wavelen < high_freq_wavelen:
        return inv_freq_f
    if wavelen > low_freq_wavelen:
        return inv_freq_f / scale_factor
    # Smooth blend between the two regions.
    smooth = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    return (1.0 - smooth) * (inv_freq_f / scale_factor) + smooth * inv_freq_f


def detect_llama3_rope_in_config(config: Mapping[str, Any]) -> bool:
    """Return True if the HF-style config dict carries a Llama 3.1 RoPE block.

    Both ``rope_scaling.type`` and the newer ``rope_scaling.rope_type`` keys are
    accepted (transformers >=4.43 uses ``rope_type``).
    """
    if not isinstance(config, Mapping):
        raise TypeError(f"config must be a Mapping (got {type(config).__name__})")
    rope = config.get("rope_scaling")
    if not isinstance(rope, Mapping):
        return False
    # Explicit ``is None`` check (mirrors v0.40.6 review-fix policy) — prevents
    # a falsy-but-set ``type`` from silently falling through to ``rope_type``.
    type_value = rope.get("type") if rope.get("type") is not None else rope.get("rope_type")
    return isinstance(type_value, str) and type_value.lower() == "llama3"


# ---------------------------------------------------------------------------
# Public config emission
# ---------------------------------------------------------------------------


def get_rope_scaling_config(
    scaling_type: str,
    target_length: float,
    original_length: int,
    *,
    yarn_factor: float | None = None,
    yarn_attn_factor: float | None = None,
    yarn_beta_fast: int | None = None,
    yarn_beta_slow: int | None = None,
    llama3_scale_factor: float = LLAMA3_DEFAULT_SCALE_FACTOR,
    llama3_low_freq_factor: float = LLAMA3_DEFAULT_LOW_FREQ_FACTOR,
    llama3_high_freq_factor: float = LLAMA3_DEFAULT_HIGH_FREQ_FACTOR,
    llama3_old_context_len: int | None = None,
) -> dict[str, Any]:
    """Build the ``rope_scaling`` dict consumed by HF model configs.

    The YaRN tunables are emitted only when ``scaling_type='yarn'``; the Llama
    3.1 tunables are emitted only when ``scaling_type='llama3'``.
    """
    if scaling_type not in ROPE_SCALING_TYPES:
        raise ValueError(
            f"Unknown RoPE scaling type: {scaling_type}. "
            f"Options: {', '.join(ROPE_SCALING_TYPES)}"
        )

    # Security review fix — validate numeric inputs at the public boundary.
    # The schema layer already guards Pydantic-loaded values; this protects
    # direct callers from emitting ``{"factor": NaN}`` into HF model configs.
    if isinstance(target_length, bool) or isinstance(original_length, bool):
        raise ValueError("target_length / original_length must not be bool")
    if not isinstance(target_length, (int, float)) or not math.isfinite(float(target_length)):
        raise ValueError(f"target_length must be a finite number (got {target_length!r})")
    if not isinstance(original_length, int) or original_length <= 0:
        raise ValueError(f"original_length must be a positive int (got {original_length!r})")
    if yarn_factor is not None:
        yarn_factor = _finite_positive(yarn_factor, "yarn_factor")

    # If target_length looks like a scaling factor (small number > 1.0 but < 64),
    # treat it as a multiplier; values >= 64 are token counts.
    if target_length < 64 and target_length > 1.0:
        factor = float(target_length)
    else:
        factor = target_length / original_length

    if factor <= 1.0:
        return {}

    factor = float(factor)
    if scaling_type == "linear":
        return {"type": "linear", "factor": factor}
    if scaling_type == "dynamic":
        return {"type": "dynamic", "factor": factor}
    if scaling_type == "yarn":
        cfg: dict[str, Any] = {
            "type": "yarn",
            "factor": yarn_factor if yarn_factor is not None else factor,
            "original_max_position_embeddings": original_length,
        }
        if yarn_attn_factor is not None:
            cfg["attention_factor"] = float(yarn_attn_factor)
        if yarn_beta_fast is not None:
            cfg["beta_fast"] = int(yarn_beta_fast)
        if yarn_beta_slow is not None:
            cfg["beta_slow"] = int(yarn_beta_slow)
        return cfg
    if scaling_type == "longrope":
        return {
            "type": "longrope",
            "factor": factor,
            "original_max_position_embeddings": original_length,
        }
    # scaling_type == "llama3" — defer to the caller-supplied original_length
    # unless an explicit Llama 3.1-specific override is provided.
    return {
        "type": "llama3",
        "factor": factor,
        "original_max_position_embeddings": (
            llama3_old_context_len if llama3_old_context_len is not None else original_length
        ),
        "low_freq_factor": float(llama3_low_freq_factor),
        "high_freq_factor": float(llama3_high_freq_factor),
    }


def apply_long_context_config(
    model_config,
    target_length: int,
    rope_scaling_type: str | None = "dynamic",
    model_name: str = "",
) -> dict | None:
    """Apply long-context configuration to a model config object.

    Args:
        model_config: HF model config object (mutated in place).
        target_length: New ``max_position_embeddings`` to set.
        rope_scaling_type: One of :data:`ROPE_SCALING_TYPES`, or ``None`` to
            auto-detect (v0.53.4 #121). The legacy default ``"dynamic"`` is
            preserved for back-compat — pass ``None`` explicitly to enable
            the new auto-detect path. When auto-detecting, the function
            picks ``"llama3"`` if the model config already carries a
            Llama 3.1 ``rope_scaling`` block, otherwise falls back to
            ``"dynamic"``.
        model_name: Used only for the default-context fallback.
    """
    original_length = getattr(
        model_config,
        "max_position_embeddings",
        get_model_default_context(model_name),
    )
    if target_length <= original_length:
        return None
    if rope_scaling_type is None:
        existing = getattr(model_config, "rope_scaling", None)
        if isinstance(existing, Mapping) and detect_llama3_rope_in_config(
            {"rope_scaling": existing}
        ):
            rope_scaling_type = "llama3"
        else:
            rope_scaling_type = "dynamic"
    rope_config = get_rope_scaling_config(
        scaling_type=rope_scaling_type,
        target_length=target_length,
        original_length=original_length,
    )
    if not rope_config:
        return None
    model_config.rope_scaling = rope_config
    model_config.max_position_embeddings = target_length
    return rope_config


def validate_long_context_config(
    max_length: int,
    rope_scaling_type: str | None,
    use_gradient_checkpointing: bool,
) -> list[str]:
    """Validate long-context configuration."""
    errors: list[str] = []
    if rope_scaling_type and rope_scaling_type not in ROPE_SCALING_TYPES:
        errors.append(
            f"Unknown RoPE scaling type: {rope_scaling_type}. "
            f"Options: {', '.join(ROPE_SCALING_TYPES)}"
        )
    if max_length >= 65536 and not use_gradient_checkpointing:
        errors.append(
            f"Training with max_length={max_length} without gradient checkpointing "
            "will likely cause OOM. Set gradient_checkpointing: true in config."
        )
    return errors
