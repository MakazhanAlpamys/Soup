"""Long-context fine-tuning utilities — 128k+ token support.

Configures RoPE (Rotary Position Embedding) scaling to extend model context
windows beyond their pre-training length. Supports multiple scaling strategies:

- linear: Simple linear interpolation (PI) — good baseline
- dynamic: NTK-aware Dynamic scaling — better for large extensions
- yarn: YaRN (Yet another RoPE extensioN) — best quality for 4-8x extension
  (v0.49.0 Part A — math kernel + config-emit; HF Transformers owns the
  actual rotation under the hood)
- longrope: LongRoPE — progressive extension with search-based factors.
  Refused since #1239: its factors exist only on already-scaled checkpoints.
- llama3: Llama 3.1 frequency-band NTK-aware scaling (v0.49.0 Part D)

Also handles gradient checkpointing configuration for memory efficiency
when training on very long sequences.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

# Supported RoPE scaling methods (v0.49.0 adds "llama3"). get_rope_scaling_config
# can still emit "longrope"; soup.yaml and apply_long_context_config refuse it (#1239).
ROPE_SCALING_TYPES = ("linear", "dynamic", "yarn", "longrope", "llama3")

# Values that remain meaningful when switching from one RoPE algorithm to
# another. Algorithm-specific keys (for example Llama 3 frequency bands) must
# not leak into the replacement block.
_ROPE_TYPE_AGNOSTIC_KEYS = ("rope_theta", "partial_rotary_factor")

# The RoPE type transformers assumes when a checkpoint ships no scaling block.
# Any other native type is a block Soup must not silently replace (#1239).
_UNSCALED_ROPE_TYPE = "default"

# #1239: LongRoPE's per-dimension factors were searched for one extension and
# ship only with checkpoints already scaled by it, which may not be extended
# again, so the type can extend nothing. Config load (schema.py) and
# apply_long_context_config both refuse it with this text.
LONGROPE_REFUSAL = (
    "training.rope_scaling_type='longrope' is refused (#1239): it cannot extend any "
    "checkpoint. Its per-dimension short_factor and long_factor vectors exist only on "
    "checkpoints already scaled with LongRoPE, and extending an already-scaled "
    "checkpoint is refused, because replacing its RoPE block would discard the scaling "
    "it was trained with. To extend a checkpoint without RoPE scaling, use linear, "
    "dynamic, yarn or llama3; on a Llama 3.1-style checkpoint, llama3 composes with its "
    "own block. To fine-tune a LongRoPE checkpoint (Phi-3-mini-128k, for example) at "
    "its native length, leave rope_scaling_type unset."
)

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


def _native_rope_type(rope: Mapping[str, Any]) -> str:
    """Return a checkpoint's RoPE type, read the way transformers reads it."""
    for key in ("rope_type", "type"):
        value = rope.get(key)
        if value is not None:
            return str(value)
    return _UNSCALED_ROPE_TYPE


def compose_llama3_rope_parameters(
    native: Mapping[str, Any],
    *,
    max_position_embeddings: int,
    target_length: float,
) -> dict[str, Any]:
    """Extend a checkpoint's own Llama 3.1 RoPE block instead of replacing it (#1239).

    Keeps the checkpoint's ``original_max_position_embeddings``,
    ``low_freq_factor`` and ``high_freq_factor`` and multiplies its ``factor``
    by ``target_length / max_position_embeddings``. The frequency bands depend
    only on the kept values, so high-frequency pairs are untouched and every
    other pair only slows down. At ``target_length == max_position_embeddings``
    the block is the checkpoint's own. A missing
    ``original_max_position_embeddings`` means ``max_position_embeddings``, as
    in transformers.
    """
    native_type = _native_rope_type(native)
    if native_type.lower() != "llama3":
        raise ValueError(
            "compose_llama3_rope_parameters needs a llama3 RoPE block "
            f"(got rope_type={native_type!r})"
        )
    if (
        isinstance(max_position_embeddings, bool)
        or not isinstance(max_position_embeddings, int)
        or max_position_embeddings <= 0
    ):
        raise ValueError(
            f"max_position_embeddings must be a positive int (got {max_position_embeddings!r})"
        )
    target = _finite_positive(target_length, "target_length")
    if target < max_position_embeddings:
        raise ValueError(
            f"target_length ({target_length}) is below max_position_embeddings "
            f"({max_position_embeddings}): a smaller llama3 factor would make the "
            "checkpoint's low-frequency pairs rotate faster than it was trained with (#1239)"
        )
    factor = _finite_positive(native.get("factor"), "the checkpoint's llama3 factor")
    low = _finite_positive(native.get("low_freq_factor"), "the checkpoint's llama3 low_freq_factor")
    # No high > low check: the bands are the checkpoint's own, and some ship
    # high == low (Llama 4 Scout), which transformers only warns about.
    high = _finite_positive(
        native.get("high_freq_factor"), "the checkpoint's llama3 high_freq_factor"
    )
    original = native.get("original_max_position_embeddings", max_position_embeddings)
    if isinstance(original, float) and original.is_integer():
        original = int(original)  # transformers accepts 8192.0 there, with a warning
    if isinstance(original, bool) or not isinstance(original, int) or original <= 0:
        raise ValueError(
            "the checkpoint's llama3 original_max_position_embeddings must be a positive "
            f"int (got {original!r})"
        )
    return {
        "rope_type": "llama3",
        "factor": factor * (target / max_position_embeddings),
        "original_max_position_embeddings": original,
        "low_freq_factor": low,
        "high_freq_factor": high,
    }


def _requested_label(requested: str | None, resolved: str) -> str:
    """How a refusal names the requested type, auto-detection included."""
    return repr(resolved) if requested is not None else f"None (auto-detected {resolved!r})"


def _already_scaled_error(
    native: Mapping[str, Any],
    *,
    native_type: str,
    requested: str | None,
    resolved: str,
    original_length: int,
    target_length: Any,
) -> ValueError:
    """Refusal for extending a checkpoint whose RoPE block is already scaled (#1239)."""
    shown = _requested_label(requested, resolved)
    if native_type.lower() == "llama3":
        way_out = (
            "Use rope_scaling_type='llama3', which composes with the checkpoint's own "
            f"llama3 block, or keep data.max_length at or below {original_length}."
        )
    else:
        way_out = (
            f"Keep data.max_length at or below {original_length}, or start from a "
            "checkpoint without RoPE scaling."
        )
    return ValueError(
        f"training.rope_scaling_type={shown} cannot extend this checkpoint from "
        f"{original_length} to {target_length} tokens: its RoPE block is already scaled "
        f"(rope_type={native_type!r}, factor={native.get('factor')!r}) and {resolved!r} "
        "has no defined composition with it, so replacing the block would discard the "
        f"scaling the checkpoint was trained with (#1239). {way_out}"
    )


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
    """Build the Transformers 5 ``rope_parameters`` mapping.

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
        return {"rope_type": "linear", "factor": factor}
    if scaling_type == "dynamic":
        return {"rope_type": "dynamic", "factor": factor}
    if scaling_type == "yarn":
        cfg: dict[str, Any] = {
            "rope_type": "yarn",
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
            "rope_type": "longrope",
            "factor": factor,
            "original_max_position_embeddings": original_length,
        }
    # scaling_type == "llama3" — defer to the caller-supplied original_length
    # unless an explicit Llama 3.1-specific override is provided.
    return {
        "rope_type": "llama3",
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
    *,
    yarn_factor: float | None = None,
    yarn_attn_factor: float | None = None,
    yarn_beta_fast: int | None = None,
    yarn_beta_slow: int | None = None,
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
            Llama 3.1 RoPE block, otherwise falls back to
            ``"dynamic"``.
        model_name: Used only for the default-context fallback.

    The mapping is assigned to ``config.rope_parameters`` before the model is
    constructed.  Transformers 5 derives ``inv_freq`` in the rotary embedding
    constructor, so mutating the config after ``from_pretrained`` is a no-op.
    Existing model-native values such as ``rope_theta`` are retained.

    A checkpoint whose RoPE block is already scaled (any ``rope_type`` other
    than ``default``) is never replaced (#1239). ``llama3`` on a native
    ``llama3`` block composes through :func:`compose_llama3_rope_parameters`;
    every other combination raises ``ValueError`` before ``model_config`` is
    touched, naming the checkpoint's ``rope_type`` and ``factor``. ``longrope``
    is refused on a checkpoint without scaling too (:data:`LONGROPE_REFUSAL`),
    so it extends nothing. A ``target_length`` at or below
    ``max_position_embeddings`` stays a no-op.
    """
    original_length = getattr(
        model_config,
        "max_position_embeddings",
        get_model_default_context(model_name),
    )
    if target_length <= original_length:
        return None
    existing = getattr(model_config, "rope_parameters", None)
    if not isinstance(existing, Mapping):
        existing = getattr(model_config, "rope_scaling", None)
    existing = dict(existing) if isinstance(existing, Mapping) else {}
    nested_sections = sorted(
        str(name) for name, value in existing.items() if isinstance(value, Mapping)
    )
    if nested_sections:
        raise ValueError(
            "nested rope_parameters are not supported safely; model-specific "
            f"sections found: {', '.join(nested_sections)}. Soup refuses to "
            "change max_position_embeddings without scaling every RoPE section"
        )
    # #1239: a scaled native block was already extended to max_position_embeddings,
    # so rebuilding a block over that length discards the scaling it was trained with.
    # One reader, transformers' key order, serves both the auto-detect and the check.
    native_type = _native_rope_type(existing)
    requested_type = rope_scaling_type
    if rope_scaling_type is None:
        rope_scaling_type = "llama3" if native_type.lower() == "llama3" else "dynamic"
    if native_type.lower() == _UNSCALED_ROPE_TYPE:
        if rope_scaling_type == "longrope":
            raise ValueError(LONGROPE_REFUSAL)
        rope_config = get_rope_scaling_config(
            scaling_type=rope_scaling_type,
            target_length=target_length,
            original_length=original_length,
            yarn_factor=yarn_factor,
            yarn_attn_factor=yarn_attn_factor,
            yarn_beta_fast=yarn_beta_fast,
            yarn_beta_slow=yarn_beta_slow,
        )
    elif native_type.lower() == "llama3" and rope_scaling_type == "llama3":
        try:
            rope_config = compose_llama3_rope_parameters(
                existing,
                max_position_embeddings=original_length,
                target_length=target_length,
            )
        except ValueError as exc:
            raise ValueError(
                f"training.rope_scaling_type={_requested_label(requested_type, 'llama3')} "
                f"cannot compose with this checkpoint's llama3 RoPE block (#1239): {exc}. "
                "Fix the checkpoint's rope_parameters, or keep data.max_length at or below "
                f"{original_length}."
            ) from exc
    else:
        raise _already_scaled_error(
            existing,
            native_type=native_type,
            requested=requested_type,
            resolved=rope_scaling_type,
            original_length=original_length,
            target_length=target_length,
        )
    if not rope_config:
        return None
    merged = {
        name: existing[name]
        for name in _ROPE_TYPE_AGNOSTIC_KEYS
        if name in existing and existing[name] is not None
    }
    merged.update(rope_config)
    model_config.rope_parameters = merged
    model_config.max_position_embeddings = target_length
    return merged


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
