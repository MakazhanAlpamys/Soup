"""DeepSeek-V3 FP8 block-quantised checkpoints, dequantised when layer streaming shards them.

DeepSeek-V3 is published in this format, and Kimi K2 uses it too:

- each converted linear weight is ``float8_e4m3fn`` ``[rows, cols]``;
- beside it sits ``<name>.weight_scale_inv``, float32, one entry per ``bm x bn`` block:
  shape ``[ceil(rows / bm), ceil(cols / bn)]``; the last block row and column are cropped;
- ``config.json`` carries ``quantization_config`` with ``quant_method: "fp8"`` and a
  two-int ``weight_block_size: [bm, bn]`` (DeepSeek also writes ``fmt: "e4m3"`` and
  ``activation_scheme: "dynamic"``; transformers adds ``scale_fmt: "float"``);
- embeddings, the head, the norms and the router stay bf16.

The dequantised weight is ``w_fp8.float() * scale_inv`` with each scale expanded over its
block, the rule of DeepSeek's own ``weight_dequant``. The sharder casts that fp32 result to
the stream dtype and then stores it by the ordinary rules (NF4 under ``quantization:
4bit``); the scales are consumed, never stored. The disk pre-flight sizes the cache from
the same plan (:func:`plan_fp8_block`), so the two cannot drift.

Every other FP8 spelling is refused, naming what is supported: per-tensor scales, e5m2,
compressed-tensors, ue8m0 scales, static activation scales, a missing or misshapen scale,
an FP8 tensor without a scale and a scale without its tensor.

No top-level torch: ``layer_shard`` imports this module on the light CLI path.
"""

import enum
import json
import math
import os
from dataclasses import dataclass
from typing import Any, FrozenSet, Mapping, Optional, Sequence, Tuple

#: The scale beside an FP8 weight ``<key>`` is ``<key>_scale_inv``
#: (``...q_proj.weight`` -> ``...q_proj.weight_scale_inv``).
FP8_SCALE_SUFFIX = "_scale_inv"
#: The safetensors spellings of the one accepted weight dtype and the one scale dtype.
FP8_WEIGHT_DTYPE = "F8_E4M3"
FP8_SCALE_DTYPE = "F32"
#: Architectures admitted for an FP8 block source: each passed the streamed-from-FP8
#: vs resident parity tests in ``tests/test_fp8_block_streaming.py``.
FP8_BLOCK_ARCHS = ("deepseek_v3",)

#: A real ``config.json`` is a few KB; past this it is not read.
_CONFIG_LIMIT = 16 * 1024 * 1024
#: Longest rendering of an untrusted value (a config entry, a shape) in a message.
_BRIEF_LIMIT = 80
#: fp32 elements one dequantisation band works on (16 MiB of product, about as much of
#: expanded scales). The output is preallocated at the stream dtype, so a tensor costs its
#: output plus one band, whatever its size.
_DEQUANT_BAND_ELEMENTS = 1 << 22

SUPPORTED_FP8_FORMAT = (
    "Layer streaming dequantises one FP8 format, DeepSeek-V3's block FP8: float8_e4m3fn "
    "(F8_E4M3) weights, each with a float32 '<name>_scale_inv' of shape "
    "[ceil(rows/bm), ceil(cols/bn)], and a config.json quantization_config with "
    "quant_method 'fp8' and weight_block_size [bm, bn] (fmt 'e4m3', activation_scheme "
    "'dynamic' and scale_fmt 'float' where present), on model_type "
    f"{', '.join(FP8_BLOCK_ARCHS)}. Dequantise this checkpoint to bf16 first."
)


class UnsupportedFP8Error(ValueError):
    """An FP8 checkpoint layer streaming does not dequantise; the message names the format."""


def _brief(value: Any) -> str:
    """``repr`` of an untrusted value, truncated: it can reach a terminal."""
    text = repr(value)
    return text if len(text) <= _BRIEF_LIMIT else text[: _BRIEF_LIMIT - 3] + "..."


def _refuse(problem: str) -> UnsupportedFP8Error:
    return UnsupportedFP8Error(f"{problem}. {SUPPORTED_FP8_FORMAT}")


def as_quantization_mapping(quantization_config: Any) -> Optional[Mapping[str, Any]]:
    """A config's ``quantization_config`` as a mapping, or ``None`` when there is none.

    ``AutoConfig`` keeps it as the ``config.json`` dict; a transformers quantization
    config object is read through its own ``to_dict``.
    """
    if quantization_config is None:
        return None
    if isinstance(quantization_config, Mapping):
        return quantization_config
    to_dict = getattr(quantization_config, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return payload
    return None


def _setting(value: Any) -> Any:
    """A config value as its plain JSON form (transformers stores some as enums)."""
    return value.value if isinstance(value, enum.Enum) else value


def _quant_method(quant: Mapping[str, Any]) -> str:
    method = _setting(quant.get("quant_method"))
    return method if isinstance(method, str) else ""


def is_fp8_quantization_config(quantization_config: Any) -> bool:
    """Whether a ``quantization_config`` declares ``quant_method: "fp8"`` (any spelling)."""
    quant = as_quantization_mapping(quantization_config)
    return quant is not None and _quant_method(quant) == "fp8"


def fp8_block_size(quantization_config: Any, *, arch: str) -> Optional[Tuple[int, int]]:
    """``(bm, bn)`` for a supported FP8 block config, ``None`` for a config without FP8.

    Reads the config alone, so a caller can refuse before any weight is resolved. A
    ``quant_method: "fp8"`` config in any other spelling, or on an architecture not in
    :data:`FP8_BLOCK_ARCHS`, raises :class:`UnsupportedFP8Error`. Another
    ``quant_method`` returns ``None`` here; :func:`plan_fp8_block` refuses it when the
    checkpoint holds FP8 tensors.
    """
    quant = as_quantization_mapping(quantization_config)
    if quant is None or _quant_method(quant) != "fp8":
        return None
    block = _setting(quant.get("weight_block_size"))
    if block is None:
        raise _refuse(
            "config.json quantization_config declares per-tensor FP8 scales "
            "(quant_method 'fp8' without weight_block_size)"
        )
    if (
        not isinstance(block, (list, tuple))
        or len(block) != 2
        or any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 1 for dim in block)
    ):
        raise _refuse(
            f"config.json weight_block_size must be two positive ints; got {_brief(block)}"
        )
    fmt = _setting(quant.get("fmt"))
    if fmt is not None and fmt != "e4m3":
        raise _refuse(
            f"config.json quantization_config declares fmt {_brief(fmt)}; only e4m3 "
            f"(float8_e4m3fn) weights are this format, e5m2 is not"
        )
    scale_fmt = _setting(quant.get("scale_fmt"))
    if scale_fmt is not None and scale_fmt != "float":
        raise _refuse(
            f"config.json quantization_config declares scale_fmt {_brief(scale_fmt)}; "
            f"only float32 scales are this format (ue8m0 power-of-two scales, the "
            f"DeepSeek-V4 spelling, are not)"
        )
    activation = _setting(quant.get("activation_scheme"))
    if activation is not None and activation != "dynamic":
        raise _refuse(
            f"config.json quantization_config declares activation_scheme "
            f"{_brief(activation)}; only 'dynamic' is this format (static activation "
            f"scales are not)"
        )
    if arch not in FP8_BLOCK_ARCHS:
        raise UnsupportedFP8Error(
            f"an FP8 block-quantised base is gated on resident parity per architecture; "
            f"admitted: {', '.join(FP8_BLOCK_ARCHS)}; this base streams as {_brief(arch)}. "
            f"Dequantise it to bf16 first."
        )
    return int(block[0]), int(block[1])


@dataclass(frozen=True)
class FP8BlockPlan:
    """Which tensors of a checkpoint are FP8 block weights, and which are their scales.

    Keys are the sharder's canonical stream keys. ``weights`` are dequantised at shard
    time; ``scales`` are consumed by them and never stored, nor charged on disk.
    """

    block: Tuple[int, int]
    weights: FrozenSet[str]
    scales: FrozenSet[str]

    @staticmethod
    def scale_key(weight_key: str) -> str:
        return weight_key + FP8_SCALE_SUFFIX


def fp8_scale_shape(shape: Sequence[int], block: Tuple[int, int]) -> Tuple[int, int]:
    """The ``weight_scale_inv`` grid a ``[rows, cols]`` weight needs: one entry per block."""
    rows, cols = (int(dim) for dim in shape)
    block_rows, block_cols = block
    return -(-rows // block_rows), -(-cols // block_cols)


def plan_fp8_block(
    quantization_config: Any,
    tensors: Mapping[str, Tuple[str, Sequence[int]]],
    *,
    arch: str,
) -> Optional[FP8BlockPlan]:
    """Classify a checkpoint's FP8 tensors, or refuse it before anything is read.

    ``tensors`` maps every key to ``(safetensors dtype, shape)``, from the headers alone.
    Returns ``None`` when the checkpoint holds no FP8 tensor and no block scale. The
    sharder and the disk pre-flight both call this, so they agree on which keys are
    dequantised and which are consumed.
    """
    block = fp8_block_size(quantization_config, arch=arch)
    scales = {key for key in tensors if key.endswith(FP8_SCALE_SUFFIX)}
    weights = {
        key
        for key, (dtype, _shape) in tensors.items()
        if str(dtype).upper().startswith("F8") and key not in scales
    }
    if not weights and not scales:
        return None
    if block is None:
        first = min(weights or scales)
        quant = as_quantization_mapping(quantization_config)
        method = _quant_method(quant) if quant is not None else ""
        if method == "compressed-tensors":
            raise _refuse(
                f"the checkpoint holds FP8 tensors (e.g. {_brief(first)}) and config.json "
                f"declares quant_method 'compressed-tensors', whose scales are not "
                f"'weight_scale_inv' block scales"
            )
        if method:
            raise _refuse(
                f"the checkpoint holds FP8 tensors or block scales (e.g. {_brief(first)}), "
                f"but config.json declares quant_method {_brief(method)}"
            )
        raise _refuse(
            f"the checkpoint holds FP8 tensors or block scales (e.g. {_brief(first)}), but "
            f"config.json declares no FP8 quantization_config (it is missing, unreadable, a "
            f"symlink, or has no quant_method 'fp8')"
        )
    for key in sorted(weights):
        dtype, shape = tensors[key]
        if dtype != FP8_WEIGHT_DTYPE:
            raise _refuse(f"tensor {_brief(key)} is {_brief(dtype)}, not F8_E4M3")
        if len(shape) != 2:
            raise _refuse(
                f"FP8 tensor {_brief(key)} has shape {_brief(list(shape))}; block FP8 "
                f"covers 2-D weights only"
            )
        if FP8BlockPlan.scale_key(key) not in tensors:
            raise _refuse(
                f"FP8 tensor {_brief(key)} has no block scale "
                f"{_brief(FP8BlockPlan.scale_key(key))}"
            )
    for scale_key in sorted(scales):
        weight_key = scale_key[: -len(FP8_SCALE_SUFFIX)]
        if weight_key not in tensors:
            raise _refuse(
                f"block scale {_brief(scale_key)} has no tensor {_brief(weight_key)}"
            )
        if weight_key not in weights:
            raise _refuse(
                f"block scale {_brief(scale_key)} sits beside {_brief(weight_key)}, which "
                f"is {_brief(tensors[weight_key][0])}, not an FP8 tensor"
            )
        dtype, shape = tensors[scale_key]
        if dtype != FP8_SCALE_DTYPE:
            raise _refuse(f"block scale {_brief(scale_key)} is {_brief(dtype)}, not F32")
        expected = fp8_scale_shape(tensors[weight_key][1], block)
        if tuple(int(dim) for dim in shape) != expected:
            per_tensor = math.prod(int(dim) for dim in shape) == 1 and expected != (1, 1)
            raise _refuse(
                f"block scale {_brief(scale_key)} has shape {_brief(list(shape))}; "
                f"{_brief(weight_key)} of shape {list(tensors[weight_key][1])} with "
                f"weight_block_size {list(block)} needs {list(expected)}"
                + (" (a per-tensor scale is not block FP8)" if per_tensor else "")
            )
    return FP8BlockPlan(block=block, weights=frozenset(weights), scales=frozenset(scales))


def read_checkpoint_quantization_config(weights_dir: str) -> Optional[Mapping[str, Any]]:
    """``quantization_config`` from ``config.json`` beside the weights, or ``None``.

    ``None`` when there is no regular, readable, size-bounded ``config.json`` or it has no
    such mapping: :func:`plan_fp8_block` then refuses an FP8 checkpoint by saying so.
    """
    path = os.path.join(os.path.realpath(os.path.expanduser(weights_dir)), "config.json")
    if not os.path.isfile(path) or os.path.islink(path):
        return None
    try:
        if not 0 < os.path.getsize(path) <= _CONFIG_LIMIT:
            return None
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeDecodeError, ValueError, RecursionError):
        return None
    quant = payload.get("quantization_config") if isinstance(payload, dict) else None
    return quant if isinstance(quant, dict) else None


def dequantize_fp8_block(
    weight: Any,
    scale_inv: Any,
    block: Tuple[int, int],
    *,
    name: str = "weight",
    dtype: Any = None,
) -> Any:
    """``weight.float() * scale_inv`` expanded per ``block``, cropped at a ragged edge,
    cast to ``dtype`` (float32 when ``None``).

    Bit-exact against a per-block loop: each element is one fp32 multiply of the same two
    operands, then the cast. The work is bounded:

    - a block past the matrix edge costs nothing: rows index their scale row, and the
      column block is expanded as ``min(block, cols)``, so a hostile ``weight_block_size``
      allocates no more than the matrix;
    - the product is formed one band of rows at a time (:data:`_DEQUANT_BAND_ELEMENTS`) and
      cast into a preallocated output, so the peak is the output plus one band.

    Refuses a weight or scale that is not the format, a non-finite or negative scale (a
    zero scale, an all-zero block's, is kept), and a result that is not finite after the
    cast: an e4m3fn NaN byte, or a weight past ``dtype``'s range.
    """
    import torch

    out_dtype = torch.float32 if dtype is None else dtype
    if weight.dtype != torch.float8_e4m3fn or weight.ndim != 2:
        raise _refuse(
            f"tensor {_brief(name)} is {weight.dtype} of shape {list(weight.shape)}, not a "
            f"2-D float8_e4m3fn weight"
        )
    expected = fp8_scale_shape(weight.shape, block)
    if scale_inv.dtype != torch.float32 or tuple(scale_inv.shape) != expected:
        raise _refuse(
            f"block scale of {_brief(name)} is {scale_inv.dtype} {list(scale_inv.shape)}; "
            f"a float32 {list(expected)} is needed"
        )
    if not bool(torch.isfinite(scale_inv).all()):
        raise UnsupportedFP8Error(
            f"block scale of {_brief(name)} holds a non-finite value; it would put inf or "
            f"NaN into every element of its block. The checkpoint is damaged."
        )
    if bool((scale_inv < 0).any()):
        raise UnsupportedFP8Error(
            f"block scale of {_brief(name)} holds a negative value; no FP8 block producer "
            f"writes one. The checkpoint is damaged."
        )
    rows, cols = (int(dim) for dim in weight.shape)
    # Rows pick their scale row by index, so the row block is never expanded; the column
    # block is, and a block past the matrix edge is that edge.
    block_rows, block_cols = block[0], min(block[1], cols)
    out = torch.empty((rows, cols), dtype=out_dtype)
    band = max(1, _DEQUANT_BAND_ELEMENTS // max(cols, 1))
    for start in range(0, rows, band):
        stop = min(start + band, rows)
        row_scales = scale_inv.index_select(0, torch.arange(start, stop) // block_rows)
        expanded = row_scales.repeat_interleave(block_cols, dim=1)[:, :cols]
        product = weight[start:stop].to(torch.float32)
        product.mul_(expanded)
        target = out[start:stop]
        target.copy_(product)
        if not bool(torch.isfinite(target).all()):
            raise _non_finite(name, product, out_dtype)
    return out


def _non_finite(name: str, product: Any, out_dtype: Any) -> UnsupportedFP8Error:
    """Why a dequantised band is not finite: a NaN byte, or a value past the stream dtype."""
    import torch

    if not bool(torch.isfinite(product).all()):
        return UnsupportedFP8Error(
            f"tensor {_brief(name)} holds NaN or infinity after dequantisation (the e4m3fn "
            f"bytes 0x7F and 0xFF are NaN); training on it would make every loss NaN. The "
            f"checkpoint is damaged."
        )
    dtype_name = str(out_dtype).replace("torch.", "")
    peak = float(product.abs().max())
    return UnsupportedFP8Error(
        f"tensor {_brief(name)} dequantises to {peak:g}, which overflows {dtype_name} (max "
        f"{torch.finfo(out_dtype).max:g}). Stream it in bfloat16: on a card that runs "
        f"bfloat16 (Ampere or newer) Soup picks it."
    )
