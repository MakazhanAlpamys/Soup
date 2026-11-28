# SPDX-License-Identifier: Apache-2.0
# Written for Soup (new layout guard for #842), not ported from the challenge.
"""Shape and quant-state layout required by the fused NF4 kernel (#3455).

Kept separate from the Triton module so the refusal boundary can be tested on
a CPU-only host, where Triton is not installed.
"""

from __future__ import annotations

import torch

BLOCK_K = 64


def validate_supported(packed, quant_state) -> None:
    """Refuse a shape or quant state the kernel cannot decode (#3455).

    The kernel assumes an aligned NF4 layout: both dimensions are multiples of
    BLOCK_K (64, the NF4 blocksize), the state is nested double quant with
    state2 blocksize 256, the first-level absmax holds one code per 64 weights,
    the second-level absmax is indexed in groups of 256 blocks, and the offset
    is the single scalar bitsandbytes stores. An unsupported layout is refused
    before any launch rather than reading out of bounds in the kernel.
    """
    shape = quant_state.shape
    if len(shape) != 2:
        raise ValueError(f"fused NF4 expects a 2-D shape, got {shape}")
    n, k = int(shape[0]), int(shape[1])
    if n % BLOCK_K or k % BLOCK_K:
        raise ValueError(f"fused NF4 needs dimensions divisible by {BLOCK_K}: {n}x{k}")
    if int(quant_state.blocksize) != BLOCK_K:
        raise ValueError(f"fused NF4 needs blocksize {BLOCK_K}, got {quant_state.blocksize}")
    if quant_state.quant_type != "nf4":
        raise ValueError(f"fused NF4 needs quant_type nf4, got {quant_state.quant_type}")
    if not quant_state.nested or int(quant_state.state2.blocksize) != 256:
        raise ValueError("fused NF4 needs nested double quant with state2 blocksize 256")
    if packed.numel() != n * k // 2:
        raise ValueError(f"packed weight has {packed.numel()} bytes, expected {n * k // 2}")
    if quant_state.absmax.numel() != n * k // BLOCK_K:
        raise ValueError("absmax must hold one code per 64 weights")
    if quant_state.absmax.dtype != torch.uint8:
        raise ValueError("absmax codes must be uint8")
    n_groups = quant_state.absmax.numel() // 256
    if quant_state.state2.absmax.numel() != n_groups:
        raise ValueError("state2 absmax must hold one value per 256-block group")
    if quant_state.offset.numel() != 1:
        raise ValueError("bitsandbytes stores one scalar offset per weight")
