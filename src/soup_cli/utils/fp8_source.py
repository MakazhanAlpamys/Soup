"""Read block-scaled FP8 source checkpoints (DeepSeek-V3 / Kimi K2 layout).

DeepSeek-V3 and Kimi K2 publish their linear weights as ``float8_e4m3fn`` plus,
for each weight, an fp32 ``<name>.weight_scale_inv`` holding one scale per
``weight_block_size`` block (128 x 128). The weight is ``q * scale_inv`` per
block; the suffix names the quantiser's point of view, not the operation here.
``config.json`` declares the format::

    "quantization_config": {"quant_method": "fp8", "fmt": "e4m3",
                            "weight_block_size": [128, 128], ...}

The sharder dequantises such a weight to its target dtype once, at shard time,
and from then on it is an ordinary dense tensor (and, with ``quant="nf4"``, goes
through the unchanged NF4 path). Dequantisation runs in block-aligned row chunks
so the fp32 intermediate is bounded by one chunk, not by the tensor.

No top-level torch import: this module is reachable from the light CLI path.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

#: Companion suffix appended to a weight key: ``x.weight`` -> ``x.weight_scale_inv``.
SCALE_SUFFIX = "_scale_inv"
#: The only FP8 storage type this path decodes.
E4M3_SAFETENSORS_DTYPE = "F8_E4M3"
_CONFIG_LIMIT = 16 * 1024 * 1024
_MAX_BLOCK_EDGE = 4096
#: fp32 bytes one dequantisation chunk may hold (before rounding to whole blocks).
_TARGET_CHUNK_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class Fp8SourceConfig:
    """``weight_block_size`` as (rows, cols)."""

    block: Tuple[int, int]


def is_float8_dtype(safetensors_dtype: Any) -> bool:
    """True for any safetensors float8 storage type (``F8_E4M3``, ``F8_E5M2``, ...)."""
    return str(safetensors_dtype).upper().startswith("F8")


def scale_key_for(weight_key: str) -> str:
    return weight_key + SCALE_SUFFIX


def weight_key_for(scale_key: str) -> str:
    return scale_key[: -len(SCALE_SUFFIX)]


def is_scale_key(key: str) -> bool:
    return key.endswith(".weight" + SCALE_SUFFIX)


def load_fp8_source_config(weights_dir: str) -> Optional[Fp8SourceConfig]:
    """The FP8 block config from ``config.json``, or None when it declares none.

    Absent ``config.json``, absent ``quantization_config`` or a ``quant_method``
    other than ``fp8`` all mean "not an FP8 checkpoint" here; the sharder still
    refuses float8 tensors it finds without this config. An ``fp8`` config the
    sharder cannot honour is refused by name.
    """
    root = os.path.realpath(os.path.expanduser(weights_dir))
    path = os.path.join(root, "config.json")
    if not os.path.isfile(path) or os.path.islink(path):
        return None
    size = os.path.getsize(path)
    if size <= 0 or size > _CONFIG_LIMIT:
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    quant = payload.get("quantization_config") if isinstance(payload, dict) else None
    if not isinstance(quant, dict) or quant.get("quant_method") != "fp8":
        return None
    fmt = quant.get("fmt", "e4m3")
    if fmt != "e4m3":
        raise ValueError(
            f"FP8 checkpoint declares fmt={fmt!r}; only block-scaled e4m3 "
            f"(DeepSeek-V3 / Kimi K2 layout) is supported for layer streaming"
        )
    block = quant.get("weight_block_size")
    if (
        not isinstance(block, (list, tuple))
        or len(block) != 2
        or not all(isinstance(edge, int) and not isinstance(edge, bool) for edge in block)
        or not all(0 < edge <= _MAX_BLOCK_EDGE for edge in block)
    ):
        raise ValueError(
            f"FP8 checkpoint needs quantization_config.weight_block_size as two "
            f"positive integers <= {_MAX_BLOCK_EDGE} (DeepSeek-V3: [128, 128]); "
            f"got {block!r}"
        )
    return Fp8SourceConfig(block=(int(block[0]), int(block[1])))


#: Where the reason for the FP8 double-quant rule is recorded.
DOUBLE_QUANT_RECORD = "benchmarks/gate-b3-fp8-source.md (G3)"


def is_block_fp8_model_config(model_config: Any) -> bool:
    """True when a model config declares a block-scaled FP8 source checkpoint.

    Reads ``quantization_config.quant_method`` from either a plain dict (how most
    configs carry it) or a config object, so the trainer can decide before any
    weights are located. Tolerates a missing or malformed field: that is "not
    FP8" here, and the sharder still refuses float8 tensors it cannot decode.
    """
    quant = getattr(model_config, "quantization_config", None)
    if quant is None:
        return False
    if isinstance(quant, dict):
        method = quant.get("quant_method")
    else:
        method = getattr(quant, "quant_method", None)
    method = getattr(method, "value", method)
    return isinstance(method, str) and method.lower() == "fp8"


def resolve_double_quant(requested: Optional[bool], *, fp8_source: bool, quant: str) -> bool:
    """The NF4 double-quant setting a run actually uses.

    ``requested`` is the tri-state ``training.bnb_4bit_use_double_quant``: None
    when unset. For block-scaled FP8 sources the supported configuration is
    double-quant OFF: with it on, the bitsandbytes codec fails B3's preregistered
    per-block reconstruction gate (``0.17 x absmax``) on real DeepSeek-V3 weights
    while it passes with it off. So unset resolves to off there, and an explicit
    ``True`` is refused by name rather than silently overridden. Every other
    source keeps the shipped default (on). Irrelevant unless ``quant`` is NF4.
    """
    if not fp8_source or quant != "nf4":
        return requested is not False
    if requested is True:
        raise ValueError(
            "bnb_4bit_use_double_quant=true is not supported for a block-scaled FP8 "
            "source checkpoint: with double-quant on, the NF4 codec fails the "
            "per-block reconstruction gate (0.17 x absmax) on real DeepSeek-V3 "
            f"weights, and passes with it off ({DOUBLE_QUANT_RECORD}). Leave it unset "
            "or set it to false; this costs about 0.37 bit per parameter of extra "
            "NF4 statistics."
        )
    return False


def stream_double_quant(tcfg: Any, model_config: Any, *, quant: str) -> bool:
    """The double-quant setting layer streaming uses for this run.

    Any source other than block-scaled FP8 reads ``tcfg.double_quant_on``
    exactly as before; an FP8 source goes through ``resolve_double_quant``
    with the setting as the user wrote it (unset, true or false).
    """
    if is_block_fp8_model_config(model_config):
        return resolve_double_quant(tcfg.double_quant_requested, fp8_source=True, quant=quant)
    return tcfg.double_quant_on


def chunk_rows_for(cols: int, *, block_rows: int, target_bytes: int = _TARGET_CHUNK_BYTES) -> int:
    """Rows per dequantisation chunk: whole block rows, at least one block row."""
    fitting = target_bytes // max(int(cols) * 4, 1)
    return max(block_rows, (fitting // block_rows) * block_rows)


def validate_scale_grid(
    scale: Any, weight_shape: Tuple[int, ...], block: Tuple[int, int], key: str
) -> None:
    """Refuse a scale grid that cannot belong to this weight, or that holds a
    value no correct quantiser writes (non-finite, zero or negative)."""
    import torch

    if len(weight_shape) != 2:
        raise ValueError(
            f"FP8 weight {key!r} has shape {tuple(weight_shape)}; block scaling "
            f"is defined for 2-D weights only"
        )
    expected = (math.ceil(weight_shape[0] / block[0]), math.ceil(weight_shape[1] / block[1]))
    if tuple(scale.shape) != expected:
        raise ValueError(
            f"FP8 weight {key!r} of shape {tuple(weight_shape)} needs a scale grid of "
            f"{expected} for {block[0]}x{block[1]} blocks; got {tuple(scale.shape)}"
        )
    if not scale.is_floating_point():
        raise ValueError(f"FP8 scale for {key!r} must be floating point; got {scale.dtype}")
    as_fp32 = scale.to(torch.float32)
    if not torch.isfinite(as_fp32).all():
        raise ValueError(f"FP8 scale for {key!r} holds a non-finite value")
    if not (as_fp32 > 0).all():
        raise ValueError(
            f"FP8 scale for {key!r} holds a zero or negative value; a zero scale "
            f"would erase its block whatever the weights hold"
        )


def _dequantize_rows(out_rows: Any, q_rows: Any, scale_rows: Any, block_cols: int) -> None:
    """``out_rows = q_rows * scale`` for one chunk, via one fp32 buffer.

    ``scale_rows`` is (rows, column_blocks), already expanded along rows. The
    product is taken in fp32 and rounded once to ``out_rows``' dtype.
    """
    import torch

    values = q_rows.to(torch.float32)
    for column_block in range(scale_rows.shape[1]):
        start = column_block * block_cols
        values[:, start : start + block_cols].mul_(scale_rows[:, column_block : column_block + 1])
    out_rows.copy_(values)


def dequantize_fp8_blockwise(
    weight_slice: Any,
    scale: Any,
    *,
    block: Tuple[int, int],
    dtype: str,
    key: str,
    chunk_rows: Optional[int] = None,
) -> Any:
    """Dequantise one block-scaled e4m3 weight into an owned ``dtype`` tensor.

    ``weight_slice`` is a safetensors slice (``handle.get_slice(key)``), read one
    chunk of rows at a time so the source is never materialised whole.
    """
    import torch

    stored = weight_slice.get_dtype()
    if str(stored).upper() != E4M3_SAFETENSORS_DTYPE:
        raise ValueError(
            f"FP8 weight {key!r} is stored as {stored}; only e4m3 "
            f"({E4M3_SAFETENSORS_DTYPE}) is supported"
        )
    shape = tuple(int(dim) for dim in weight_slice.get_shape())
    validate_scale_grid(scale, shape, block, key)
    rows, cols = shape
    step = chunk_rows if chunk_rows is not None else chunk_rows_for(cols, block_rows=block[0])
    if step <= 0 or step % block[0]:
        raise ValueError(f"chunk_rows must be a positive multiple of {block[0]}; got {step}")
    scale32 = scale.to(torch.float32)
    out = torch.empty(shape, dtype=getattr(torch, dtype))
    for start in range(0, rows, step):
        stop = min(start + step, rows)
        q_rows = weight_slice[start:stop]
        scale_rows = scale32[start // block[0] : math.ceil(stop / block[0])].repeat_interleave(
            block[0], dim=0
        )[: stop - start]
        _dequantize_rows(out[start:stop], q_rows, scale_rows, block[1])
        del q_rows, scale_rows
        if not torch.isfinite(out[start:stop]).all():
            raise ValueError(
                f"FP8 weight {key!r} decodes to a non-finite value in rows "
                f"{start}..{stop - 1} (an e4m3 NaN byte, or a product beyond "
                f"{dtype}'s range)"
            )
    return out
