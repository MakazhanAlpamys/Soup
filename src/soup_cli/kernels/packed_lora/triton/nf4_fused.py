# SPDX-License-Identifier: Apache-2.0
# Written for Soup (new kernel for #842), not ported from the Kernel Challenge.
# Modified for Soup: none of the challenge's code is used here; the NF4 decode
# follows the block layout documented in this package's nf4.py.
# ruff: noqa: N803  (Triton constexpr arguments are upper case by convention)
"""Fused NF4 dequantise-and-multiply Triton kernel for #842.

The kernel decodes bitsandbytes' NF4 weights (blocksize 64, double quant with
state2 blocksize 256) *inside* the GEMM, so no dense weight is materialised and
nothing is captured on the autograd context except the packed bytes and their
quantisation tensors through ``save_for_backward``.

Reduction order (named for the record): K is traversed in sequential blocks of
``BLOCK_K = 64`` weights, one accumulation per block into an fp32 accumulator,
and the output is rounded once to the compute dtype at the end. No atomics and
no split-K. The 64-weight NF4 block aligns with BLOCK_K, so each K block has
exactly one first-level scale.

A kernel that is not exact against bitsandbytes' own dispatch is published as
measured and is not adopted by ``install_dequant_forward``; this module only
provides the operation and its harness cells.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

BLOCK_M = 32
BLOCK_N = 64
BLOCK_K = 64  # one NF4 block: sequential K traversal, no atomics or split-K


@triton.jit
def _decode_tile(packed_ptr, qscale_ptr, scale2_ptr, offset_ptr, code2_ptr, nf4_ptr,
                 row, k0, K_HALF: tl.constexpr, GROUP: tl.constexpr,
                 BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Decode the K block starting at ``k0`` into a [BLOCK_N, BLOCK_K] tile."""
    # Two codes per byte; the even K element is the high nibble. Keep the
    # block start as a scalar in the arithmetic: indexing a scalar with
    # ``[None, :]`` broadcasts incorrectly for later K blocks on Triton 3.4.
    bk = tl.arange(0, BLOCK_K // 2)
    byte_in_block = k0 // 2 + bk[None, :]
    at = row[:, None] * K_HALF + byte_in_block
    byte = tl.load(packed_ptr + at).to(tl.int32)
    # One first-level scale per 64 weights = per 32 bytes.
    block = row[:, None] * (K_HALF // 32) + byte_in_block // 32
    q = tl.load(qscale_ptr + block).to(tl.int32)
    group = block // GROUP
    # bitsandbytes stores one scalar offset for the whole weight, not one per
    # state2 group; indexing it by group reads out of bounds past group 0.
    scale = libdevice.add_rn(
        libdevice.mul_rn(tl.load(code2_ptr + q), tl.load(scale2_ptr + group)),
        tl.load(offset_ptr),
    )
    high = tl.load(nf4_ptr + (byte >> 4)) * scale
    low = tl.load(nf4_ptr + (byte & 15)) * scale
    return tl.join(high, low).reshape(BLOCK_N, BLOCK_K)


@triton.jit
def _nf4_forward_kernel(X, Packed, QScale, Scale2, Offset, Code2, NF4, Out,
                        M, N, K, K_HALF: tl.constexpr, GROUP: tl.constexpr,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(X + rm[:, None] * K + rk[None, :],
                    mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        w = _decode_tile(Packed, QScale, Scale2, Offset, Code2, NF4,
                         rn, k0, K_HALF, GROUP, BLOCK_N, BLOCK_K)
        acc += tl.dot(x, tl.trans(w).to(x.dtype))
    tl.store(Out + rm[:, None] * N + rn[None, :], acc.to(Out.dtype.element_ty),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _nf4_grad_x_kernel(GY, Packed, QScale, Scale2, Offset, Code2, NF4, Out,
                       M, N, K, K_HALF: tl.constexpr, GROUP: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m, pid_k = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk0 = pid_k * BLOCK_K
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        rn = n0 + tl.arange(0, BLOCK_N)
        gy = tl.load(GY + rm[:, None] * N + rn[None, :],
                     mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)
        w = _decode_tile(Packed, QScale, Scale2, Offset, Code2, NF4,
                         rn, rk0, K_HALF, GROUP, BLOCK_N, BLOCK_K)
        acc += tl.dot(gy, w.to(gy.dtype))
    tl.store(Out + rm[:, None] * K + (rk0 + tl.arange(0, BLOCK_K))[None, :],
             acc.to(Out.dtype.element_ty),
             mask=(rm[:, None] < M) & ((rk0 + tl.arange(0, BLOCK_K))[None, :] < K))


def _state_tensors(quant_state):
    absmax = quant_state.absmax
    if quant_state.nested:
        state2_absmax = quant_state.state2.absmax
        state2_code = quant_state.state2.code
        offset = quant_state.offset
    else:
        empty = absmax.new_empty((0,))
        state2_absmax = state2_code = offset = empty
    return absmax, state2_absmax, state2_code, offset


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


def fused_nf4_linear(x, packed, quant_state) -> torch.Tensor:
    """Out = x @ dequant(packed)^T without materialising the dense weight."""
    validate_supported(packed, quant_state)
    n, k = quant_state.shape
    m = x.shape[0]
    absmax, state2_absmax, state2_code, offset = _state_tensors(quant_state)
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))
    _nf4_forward_kernel[grid](
        x, packed, absmax, state2_absmax, offset, state2_code, quant_state.code, out,
        m, n, k, k // 2, 256, BLOCK_M, BLOCK_N, BLOCK_K,
    )
    return out


def fused_nf4_grad_x(grad_out, packed, quant_state) -> torch.Tensor:
    """grad_x = grad_out @ dequant(packed), same reduction order as forward."""
    validate_supported(packed, quant_state)
    n, k = quant_state.shape
    m = grad_out.shape[0]
    absmax, state2_absmax, state2_code, offset = _state_tensors(quant_state)
    out = torch.empty((m, k), device=grad_out.device, dtype=grad_out.dtype)
    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(k, BLOCK_K))
    _nf4_grad_x_kernel[grid](
        grad_out, packed, absmax, state2_absmax, offset, state2_code, quant_state.code, out,
        m, n, k, k // 2, 256, BLOCK_M, BLOCK_N, BLOCK_K,
    )
    return out


class FusedNF4Linear(torch.autograd.Function):
    """Checkpoint-visible fused NF4 forward/backward for the #842 kernel."""

    @staticmethod
    def forward(ctx, x, packed, quant_state):
        ctx.save_for_backward(x, packed, quant_state.absmax)
        ctx.quant_state = quant_state
        return fused_nf4_linear(x, packed, quant_state)

    @staticmethod
    def backward(ctx, grad_out):
        x, packed, absmax = ctx.saved_tensors
        del x, absmax
        return fused_nf4_grad_x(grad_out, packed, ctx.quant_state), None, None


def fused_nf4_apply(x, packed, quant_state):
    return FusedNF4Linear.apply(x, packed, quant_state)
