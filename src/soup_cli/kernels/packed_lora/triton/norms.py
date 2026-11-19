# SPDX-License-Identifier: Apache-2.0
#
# Origin: Soup Kernel Challenge (Decentrathon 6.0), the submission of team Luck
# (BAITC-Hacks/hack-e79529fc-luck@b3f1225f); the int8 gate/up packs follow ideas seen in the
# solutions of teams Null Pointers, 21, DarkDaur and OrbiTeam.
# Modified for Soup: the LoRA-dropout outputs and masks of the original are not ported
# (dropout is 0).
# ruff: noqa: N803, N806  (Triton's constexpr arguments are upper case by convention)
"""RMSNorm over the fp32 residual stream, Qwen2's formula: w * (x * rsqrt(mean(x^2) + eps)).

The base model computes the norm in fp32 and keeps the fp32 result, which the 4-bit layers then
cast to the compute dtype on their way in. These kernels write the compute-dtype copy straight
away, so the fp32 output never exists. Backward recomputes x_hat from the saved input and rstd.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kz_rmsnorm_fwd_kernel(x_ptr, w_ptr, rstd_ptr, y_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    inside = cols < n_cols
    at = row * n_cols + cols
    x = tl.load(x_ptr + at, mask=inside, other=0.0)
    rstd = tl.math.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    y = tl.load(w_ptr + cols, mask=inside, other=0.0) * (x * rstd)
    tl.store(rstd_ptr + row, rstd)
    dt = y_ptr.dtype.element_ty  # fp16 or bf16: the compute dtype of the 4-bit layers
    tl.store(y_ptr + at, y.to(dt), mask=inside)


@triton.jit
def kz_rmsnorm_bwd_kernel(dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    inside = cols < n_cols
    at = row * n_cols + cols
    dy = tl.load(dy_ptr + at, mask=inside, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + at, mask=inside, other=0.0)
    rstd = tl.load(rstd_ptr + row)
    x_hat = x * rstd
    g = dy * tl.load(w_ptr + cols, mask=inside, other=0.0)
    mean_gx = tl.sum(g * x_hat, axis=0) / n_cols
    tl.store(dx_ptr + at, (g - x_hat * mean_gx) * rstd, mask=inside)


def _block(n_cols):
    return triton.next_power_of_2(n_cols)


def rmsnorm_fwd(x, weight, eps, dtype=torch.float16):
    """x: fp32 [T, H] contiguous. Returns (y, rstd): y in ``dtype`` (fp16 or bf16), rstd fp32."""
    rows, n_cols = x.shape
    rstd = torch.empty(rows, device=x.device, dtype=torch.float32)
    y = torch.empty(rows, n_cols, device=x.device, dtype=dtype)
    kz_rmsnorm_fwd_kernel[(rows,)](x, weight, rstd, y, n_cols, eps, BLOCK=_block(n_cols),
                                   num_warps=8)
    return y, rstd


def rmsnorm_bwd(dy, x, weight, rstd):
    """dy: [T, H] fp32 or fp16 / bf16 -> dx, fp32 like ``x``."""
    rows, n_cols = x.shape
    dx = torch.empty_like(x)
    kz_rmsnorm_bwd_kernel[(rows,)](dy, x, weight, rstd, dx, n_cols, BLOCK=_block(n_cols),
                                   num_warps=8)
    return dx
