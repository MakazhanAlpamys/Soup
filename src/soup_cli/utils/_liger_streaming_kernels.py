"""A width-4096 Triton RMS forward with ATen's vector accumulation order."""

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def _kernel() -> Any:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def forward(
        hidden_ptr, output_ptr, inverse_ptr,
        epsilon: tl.constexpr, lanes: tl.constexpr, approximate: tl.constexpr,
        warp_folds: tl.constexpr,
    ):
        row = tl.program_id(0)
        lane = tl.arange(0, lanes)
        sum_zero = tl.full((lanes,), 0, tl.float32)
        sum_one = tl.full((lanes,), 0, tl.float32)
        sum_two = tl.full((lanes,), 0, tl.float32)
        sum_three = tl.full((lanes,), 0, tl.float32)
        # ATen reads four adjacent values per lane into independent sequential
        # accumulators, then folds them in this order before the warp reduction.
        for base in range(0, 4096, 4 * lanes):
            offset = row * 4096 + base + 4 * lane
            zero = tl.load(hidden_ptr + offset)
            one = tl.load(hidden_ptr + offset + 1)
            two = tl.load(hidden_ptr + offset + 2)
            three = tl.load(hidden_ptr + offset + 3)
            sum_zero = sum_zero + zero * zero
            sum_one = sum_one + one * one
            sum_two = sum_two + two * two
            sum_three = sum_three + three * three
        combined = ((sum_zero + sum_one) + sum_two) + sum_three
        # ATen folds shared-memory warp groups from the largest offset down.
        # Triton's generic cross-warp reduction can associate these differently.
        for fold in tl.static_range(warp_folds):
            offset = lanes >> (fold + 1)
            other = tl.gather(combined, (lane + offset) % lanes, 0)
            combined = combined + other
        total = tl.sum(tl.where(lane < 32, combined, 0), 0)
        mean = total / 4096
        if approximate:
            inverse = tl.rsqrt(mean + epsilon)
        else:
            inverse = libdevice.rsqrt(mean + epsilon)
        columns = tl.arange(0, 4096)
        values = tl.load(hidden_ptr + row * 4096 + columns)
        tl.store(output_ptr + row * 4096 + columns, values * inverse)
        tl.store(inverse_ptr + row, inverse)

    return forward


def normalized_forward(
    hidden: Any, epsilon: float, *, approximate: bool = False,
) -> tuple[Any, Any, Any]:
    """Return normalized values, contiguous FP32 input, and inverse RMS."""
    import torch

    if hidden.dtype != torch.float32 or hidden.shape[-1] != 4096 or not hidden.numel():
        raise ValueError("RMS compatibility kernel requires nonempty FP32 rows of width 4096")
    flat = hidden.contiguous().reshape(-1, 4096)
    rows = flat.shape[0]
    lanes = 512 // min(16, 1 << (rows.bit_length() - 1))
    output = torch.empty_like(flat)
    inverse = torch.empty(rows, device=flat.device, dtype=torch.float32)
    _kernel()[(rows,)](
        flat, output, inverse, epsilon, lanes, approximate, max(0, lanes.bit_length() - 6),
        num_warps=4, enable_fp_fusion=False,
    )
    return output.reshape(hidden.shape), flat, inverse
