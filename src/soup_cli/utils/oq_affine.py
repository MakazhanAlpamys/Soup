"""Read oMLX/oQ affine weights without making MLX a Soup dependency.

oQ checkpoints store each quantised matrix as a packed ``uint32`` weight plus
``bfloat16`` scales and biases.  MLX packs the low-order bits as one contiguous
little-endian bit stream along the final dimension.  The frozen base can
therefore be expanded one streamed layer at a time with PyTorch, while very
large embedding tables can decode only the requested rows.

No top-level torch import: this module is reachable from the light CLI path.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Mapping

_CONFIG_LIMIT = 16 * 1024 * 1024
_SUPPORTED_BITS = frozenset({2, 3, 4, 5, 6, 8})
_SUPPORTED_GROUP_SIZES = frozenset({16, 32, 64, 96, 128, 256})
_TARGET_CHUNK_ELEMENTS = 4 * 1024 * 1024


@dataclass(frozen=True)
class AffineQuantSpec:
    bits: int
    group_size: int
    mode: str = "affine"

    def __post_init__(self) -> None:
        if self.bits not in _SUPPORTED_BITS:
            raise ValueError(
                f"unsupported oQ affine bit width {self.bits}; "
                f"supported: {sorted(_SUPPORTED_BITS)}"
            )
        if self.group_size not in _SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"unsupported oQ affine group size {self.group_size}; "
                f"supported: {sorted(_SUPPORTED_GROUP_SIZES)}"
            )
        if self.mode != "affine":
            raise ValueError(
                f"unsupported oQ quantization mode {self.mode!r}; only 'affine' is supported"
            )


@dataclass(frozen=True)
class AffineQuantConfig:
    default: AffineQuantSpec
    overrides: Mapping[str, AffineQuantSpec]

    def for_module(self, module_name: str) -> AffineQuantSpec:
        return self.overrides.get(module_name, self.default)


def _parse_spec(payload: Mapping[str, Any], *, label: str) -> AffineQuantSpec:
    try:
        bits = int(payload["bits"])
        group_size = int(payload["group_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid oQ quantization metadata for {label!r}") from exc
    return AffineQuantSpec(
        bits=bits,
        group_size=group_size,
        mode=str(payload.get("mode", "affine")),
    )


def load_affine_quant_config(weights_dir: str) -> AffineQuantConfig:
    """Load and validate the oQ ``quantization_config`` from ``config.json``."""
    root = os.path.realpath(os.path.expanduser(weights_dir))
    path = os.path.join(root, "config.json")
    if not os.path.isfile(path) or os.path.islink(path):
        raise ValueError("oQ checkpoint needs a regular config.json beside its weights")
    size = os.path.getsize(path)
    if size <= 0 or size > _CONFIG_LIMIT:
        raise ValueError(
            f"oQ config.json size must be in [1, {_CONFIG_LIMIT}] bytes; got {size}"
        )
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("oQ config.json is not readable canonical JSON") from exc
    quant = payload.get("quantization_config") if isinstance(payload, dict) else None
    if not isinstance(quant, dict):
        raise ValueError("oQ checkpoint is missing config.json quantization_config")
    default = _parse_spec(quant, label="default")
    overrides = {}
    for name, value in quant.items():
        if isinstance(value, dict):
            overrides[str(name)] = _parse_spec(value, label=str(name))
    return AffineQuantConfig(default=default, overrides=overrides)


def logical_width(packed_width: int, bits: int) -> int:
    """Return the unpacked final dimension, rejecting partial bit streams."""
    total_bits = int(packed_width) * 32
    if packed_width <= 0 or bits not in _SUPPORTED_BITS or total_bits % bits:
        raise ValueError(
            f"invalid oQ packed width {packed_width} for {bits}-bit affine weights"
        )
    return total_bits // bits


def dequantize_affine(
    weight: Any,
    scales: Any,
    biases: Any,
    *,
    spec: AffineQuantSpec,
    dtype: str,
):
    """Expand an MLX affine matrix, bounded to one output tensor plus a chunk.

    Leading dimensions are treated as independent rows.  This supports both
    ordinary 2-D linears and oQ's packed 3-D Switch-MLP expert tensors.
    """
    import torch

    if weight.dtype != torch.uint32:
        raise ValueError(f"oQ packed weight must be uint32; got {weight.dtype}")
    if weight.ndim < 2:
        raise ValueError(f"oQ packed weight must have at least 2 dimensions; got {weight.shape}")
    width = logical_width(int(weight.shape[-1]), spec.bits)
    if width % spec.group_size:
        raise ValueError(
            f"oQ logical width {width} is not divisible by group size {spec.group_size}"
        )
    stats_shape = (*tuple(weight.shape[:-1]), width // spec.group_size)
    if tuple(scales.shape) != stats_shape or tuple(biases.shape) != stats_shape:
        raise ValueError(
            "oQ affine scales/biases do not match the packed weight: "
            f"weight={tuple(weight.shape)}, scales={tuple(scales.shape)}, "
            f"biases={tuple(biases.shape)}, expected_stats={stats_shape}"
        )
    if not scales.dtype.is_floating_point or not biases.dtype.is_floating_point:
        raise ValueError("oQ affine scales and biases must be floating-point tensors")

    target_dtype = getattr(torch, dtype)
    rows = math.prod(int(dim) for dim in weight.shape[:-1])
    packed = weight.reshape(rows, int(weight.shape[-1])).contiguous()
    scales_2d = scales.reshape(rows, -1).to(target_dtype)
    biases_2d = biases.reshape(rows, -1).to(target_dtype)
    output = torch.empty((rows, width), dtype=target_dtype, device="cpu")

    positions = torch.arange(width, dtype=torch.int64)
    bit_offsets = positions * spec.bits
    word_indices = torch.div(bit_offsets, 32, rounding_mode="floor")
    shifts = bit_offsets.remainder(32)
    crosses = shifts + spec.bits > 32
    mask = (1 << spec.bits) - 1
    group_indices = torch.div(positions, spec.group_size, rounding_mode="floor")
    chunk_rows = max(1, _TARGET_CHUNK_ELEMENTS // width)

    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        current = packed[start:end].to(torch.int64)
        # One zero sentinel keeps the cross-word gather in bounds for the final
        # logical value without changing any valid packed bits.
        padded = torch.nn.functional.pad(current, (0, 1))
        low = torch.bitwise_right_shift(padded[:, word_indices], shifts)
        high = torch.bitwise_left_shift(padded[:, word_indices + 1], 32 - shifts)
        quantized = torch.where(crosses, torch.bitwise_or(low, high), low)
        quantized = torch.bitwise_and(quantized, mask).to(target_dtype)
        chunk_scales = scales_2d[start:end][:, group_indices]
        chunk_biases = biases_2d[start:end][:, group_indices]
        output[start:end] = quantized.mul(chunk_scales).add_(chunk_biases)

    return output.reshape(*tuple(weight.shape[:-1]), width).contiguous()
