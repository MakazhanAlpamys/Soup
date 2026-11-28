"""Why an FP8 block base shards NF4 without double-quant.

A block-scaled FP8 weight carries one scale per ``bm x bn`` block, and the scales of a
real checkpoint differ by orders of magnitude from block to block. NF4 stores one absmax
per 64 values; double quantisation then quantises those absmax values again, in groups
of 256, relative to the group's largest. When the group's absmax values span a wide range
the small ones lose their precision and the reconstruction error of their blocks leaves
the per-block bound.

The inputs are built from integer arithmetic (no random draws) and run through the
bitsandbytes codec that ``layer_shard._quantize_nf4`` calls, on CPU.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

if not hasattr(torch, "float8_e4m3fn"):  # pragma: no cover - old torch builds
    pytest.skip("this torch has no float8_e4m3fn", allow_module_level=True)

from bitsandbytes.functional import dequantize_4bit, quantize_4bit  # noqa: E402

from soup_cli.utils.fp8_block import dequantize_fp8_block  # noqa: E402
from soup_cli.utils.layer_shard import NF4_BLOCKSIZE, QUANT_NF4  # noqa: E402

#: Per-block reconstruction bound: the largest error in an NF4 block, as a fraction of the
#: block's absmax. Plain NF4's own worst case is half its widest code gap (0.138), and the
#: bound leaves a margin over that.
BOUND = 0.17
ROWS = COLS = 512
BLOCK = (128, 128)


def _fp8_block_weight(*, stride, modulus, offset):
    """Decode an FP8 block weight whose block scales are ``2 ** ((i * stride) % modulus -
    offset)`` for block index ``i``: integer codes in [-20, 20], exact power-of-two scales."""
    grid = (torch.arange(ROWS).unsqueeze(1) * 7 + torch.arange(COLS).unsqueeze(0) * 3) % 41
    weight = (grid - 20).to(torch.float32).to(torch.float8_e4m3fn)
    grid_rows, grid_cols = math.ceil(ROWS / BLOCK[0]), math.ceil(COLS / BLOCK[1])
    index = torch.arange(grid_rows).unsqueeze(1) * grid_cols + torch.arange(grid_cols)
    scale = torch.exp2(((index * stride) % modulus - offset).to(torch.float32))
    return dequantize_fp8_block(weight, scale, BLOCK, name="w")


def _worst_block_error(weight, *, double_quant):
    """The largest ``max|error| / absmax`` over the NF4 blocks of ``weight`` (blocks with
    an all-zero absmax have no error to bound)."""
    packed, state = quantize_4bit(
        weight,
        blocksize=NF4_BLOCKSIZE,
        compress_statistics=double_quant,
        quant_type=QUANT_NF4,
    )
    assert bool(state.nested) is double_quant
    restored = dequantize_4bit(packed, state).float()
    blocks = weight.reshape(-1, NF4_BLOCKSIZE)
    error = (restored.reshape(-1, NF4_BLOCKSIZE) - blocks).abs().amax(dim=1)
    peak = blocks.abs().amax(dim=1)
    live = peak > 0
    return float((error[live] / peak[live]).max())


def test_wide_block_scales_break_the_bound_only_with_double_quant():
    weight = _fp8_block_weight(stride=3, modulus=13, offset=6)  # scales 2**-6 .. 2**6
    plain = _worst_block_error(weight, double_quant=False)
    nested = _worst_block_error(weight, double_quant=True)
    assert plain <= BOUND, plain
    assert nested > BOUND, nested


def test_equal_block_scales_stay_within_the_bound_either_way():
    """The control: the effect needs scales that differ, so it is the scale range, not
    NF4 on FP8 data in general, that the refusal is about."""
    weight = _fp8_block_weight(stride=1, modulus=1, offset=0)  # every scale 2**0
    assert _worst_block_error(weight, double_quant=False) <= BOUND
    assert _worst_block_error(weight, double_quant=True) <= BOUND
