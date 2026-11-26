"""Parity of ``dequantize_fp8_block`` against an independent block-scaled FP8 decoder.

The oracle below is the chunked ``repeat_interleave`` decoder written for the
DeepSeek-V3 layout: the weight is ``q.float() * scale_inv`` with each scale expanded over
its ``bm x bn`` block and the last block row and column cropped. Main's decoder expands
scales by row index and a clamped column block instead, so agreement here is bitwise, not
approximate. Weights and scales come from integer arithmetic (no random draws), so every
case is reproducible and the multiplies are exact fp32 operations on both sides.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

if not hasattr(torch, "float8_e4m3fn"):  # pragma: no cover - old torch builds
    pytest.skip("this torch has no float8_e4m3fn", allow_module_level=True)

from soup_cli.utils.fp8_block import UnsupportedFP8Error, dequantize_fp8_block  # noqa: E402


def _oracle(weight_fp8, scale_inv, block, *, chunk_rows):
    """Reference decoder: block-aligned row chunks, scale rows expanded with
    ``repeat_interleave`` then cropped, one fp32 multiply per element."""
    rows, cols = weight_fp8.shape
    block_rows, block_cols = block
    assert chunk_rows % block_rows == 0
    out = torch.empty((rows, cols), dtype=torch.float32)
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        scale_rows = scale_inv[start // block_rows : math.ceil(stop / block_rows)]
        scale_rows = scale_rows.repeat_interleave(block_rows, dim=0)[: stop - start]
        values = weight_fp8[start:stop].to(torch.float32)
        for column_block in range(scale_rows.shape[1]):
            lo = column_block * block_cols
            values[:, lo : lo + block_cols].mul_(
                scale_rows[:, column_block : column_block + 1]
            )
        out[start:stop] = values
    return out


def _fp8_weight(rows, cols):
    """Every e4m3fn code point reachable from integers: ``(7*r + 3*c) % 41 - 20`` is an
    integer in [-20, 20], exactly representable in e4m3fn."""
    grid = (torch.arange(rows).unsqueeze(1) * 7 + torch.arange(cols).unsqueeze(0) * 3) % 41
    return (grid - 20).to(torch.float32).to(torch.float8_e4m3fn)


def _scales(rows, cols, block):
    """A float32 ``[ceil(rows/bm), ceil(cols/bn)]`` grid of distinct powers-of-two
    multiples (``(1 + (i + 2j) % 5) / 8``), all exact in fp32."""
    grid_rows, grid_cols = math.ceil(rows / block[0]), math.ceil(cols / block[1])
    index = torch.arange(grid_rows).unsqueeze(1) + 2 * torch.arange(grid_cols).unsqueeze(0)
    return ((index % 5) + 1).to(torch.float32) / 8.0


@pytest.mark.parametrize(
    ("shape", "block"),
    [
        pytest.param((8, 8), (4, 4), id="full-blocks"),
        pytest.param((12, 16), (4, 8), id="full-rectangular-blocks"),
        pytest.param((10, 11), (4, 4), id="ragged-last-block-both-dimensions"),
        pytest.param((9, 16), (4, 8), id="ragged-last-block-rows-only"),
        pytest.param((8, 13), (4, 8), id="ragged-last-block-cols-only"),
        pytest.param((3, 5), (4, 8), id="matrix-smaller-than-one-block"),
        pytest.param((5, 7), (1, 1), id="one-by-one-blocks"),
    ],
)
@pytest.mark.parametrize("chunk_blocks", [1, 2, 100])
def test_block_dequantisation_equals_the_reference_bitwise(shape, block, chunk_blocks):
    rows, cols = shape
    weight = _fp8_weight(rows, cols)
    scale = _scales(rows, cols, block)
    expected = _oracle(weight, scale, block, chunk_rows=chunk_blocks * block[0])
    actual = dequantize_fp8_block(weight, scale, block, name="w")
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)


def test_a_ragged_edge_uses_its_own_scale_not_a_neighbours():
    """The cropped last block row and column read the last scale row/column: a decoder
    that padded or wrapped the grid would differ in exactly these cells."""
    weight = torch.ones((5, 5)).to(torch.float8_e4m3fn)
    scale = torch.tensor([[1.0, 2.0], [4.0, 8.0]])
    expected = torch.tensor(
        [
            [1.0, 1.0, 1.0, 2.0, 2.0],
            [1.0, 1.0, 1.0, 2.0, 2.0],
            [1.0, 1.0, 1.0, 2.0, 2.0],
            [4.0, 4.0, 4.0, 8.0, 8.0],
            [4.0, 4.0, 4.0, 8.0, 8.0],
        ]
    )
    actual = dequantize_fp8_block(weight, scale, (3, 3), name="w")
    assert torch.equal(actual, expected)
    assert torch.equal(actual, _oracle(weight, scale, (3, 3), chunk_rows=3))


def test_the_stream_dtype_cast_is_the_oracles_cast():
    weight = _fp8_weight(10, 11)
    scale = _scales(10, 11, (4, 4))
    expected = _oracle(weight, scale, (4, 4), chunk_rows=4).to(torch.bfloat16)
    actual = dequantize_fp8_block(weight, scale, (4, 4), name="w", dtype=torch.bfloat16)
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual, expected)


def test_a_one_dimensional_scale_is_refused_not_broadcast():
    """The reference decoder also reads only a 2-D grid; a flat scale is not the format."""
    weight = _fp8_weight(8, 8)
    flat = torch.ones(4)
    with pytest.raises(UnsupportedFP8Error, match=r"a float32 \[2, 2\] is needed"):
        dequantize_fp8_block(weight, flat, (4, 4), name="w")
