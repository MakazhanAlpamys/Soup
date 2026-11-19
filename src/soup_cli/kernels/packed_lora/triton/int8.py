# SPDX-License-Identifier: Apache-2.0
#
# Origin: Soup Kernel Challenge (Decentrathon 6.0), the submission of team Luck
# (BAITC-Hacks/hack-e79529fc-luck@b3f1225f); the int8 gate/up packs follow ideas seen in the
# solutions of teams Null Pointers, 21, DarkDaur and OrbiTeam.
# Modified for Soup: the rounding is an argument, not a module switch.
# ruff: noqa: N803, N806  (Triton's constexpr arguments are upper case by convention)
"""int8 tensor cores for the backward's input-gradient GEMMs of the frozen MLP weights.

A T4's tensor cores multiply int8 at twice the fp16 rate, and three of the largest GEMMs of a
step are the MLP's input gradients in backward: g_act = g @ W_down, then d_x = d_gate @ W_gate
+ d_up @ W_up. Those run here as torch._int_mm (int32 accumulation) on:
- the frozen weight, decoded from NF4 straight to int8 and transposed (nf4.kz_nf4_int8t_kernel),
  with one scale per output column (127 / that column's max |w|, computed once per run);
- the gradient, quantized per token (row) to int8 with one scale per row.
The forward, the loss, the LoRA gradients' own GEMMs, attention, norms and the head are not
touched; only these three products carry int8 rounding (~1% of their largest values), within
the checks' tolerances. Kernels below: per-row quantization, the SwiGLU backward fused with
the int32 -> fp32 scaling of g_act and the int8 quantization of d_gate / d_up, and the scaled
sum of the two d_x products.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit(do_not_specialize=["seed"])
def kz_quant_rows_kernel(x_ptr, q_ptr, s_ptr, n_cols, seed, BLOCK: tl.constexpr,
                         SR: tl.constexpr = False):
    # one row: scale = max |x| / 127, q = rint(x / scale)
    row = tl.program_id(0).to(tl.int64)
    top = tl.full((), 0.0, tl.float32)
    for c0 in range(0, n_cols, BLOCK):
        c = c0 + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + row * n_cols + c, mask=c < n_cols, other=0.0).to(tl.float32)
        top = tl.maximum(top, tl.max(tl.abs(x), axis=0))
    scale = tl.maximum(top, 1e-30) / 127.0
    tl.store(s_ptr + row, scale)
    for c0 in range(0, n_cols, BLOCK):
        c = c0 + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + row * n_cols + c, mask=c < n_cols, other=0.0).to(tl.float32)
        if SR:
            u = tl.rand(seed, (row * n_cols + c).to(tl.int32))
            q = tl.minimum(tl.maximum(tl.floor(x / scale + u), -127.0), 127.0)
        else:
            q = tl.minimum(tl.maximum(libdevice.rint(x / scale), -127.0), 127.0)
        tl.store(q_ptr + row * n_cols + c, q.to(tl.int8), mask=c < n_cols)


@triton.jit
def kz_swiglu_bwd_int8_kernel(acc_ptr, srow_ptr, scol_ptr, gate_ptr, up_ptr, dg_ptr, du_ptr,
                              sdg_ptr, sdu_ptr, n_cols, BLOCK: tl.constexpr):
    # one row: g_act = acc * srow * scol (the int8 GEMM's int32 result, scaled back), then the
    # SwiGLU backward (the starter kit's formulas, fp32), d_gate and d_up quantized to int8 with
    # one scale each per row; two passes (max, then quantize), the second recomputes
    row = tl.program_id(0).to(tl.int64)
    srow = tl.load(srow_ptr + row)
    top_g = tl.full((), 0.0, tl.float32)
    top_u = tl.full((), 0.0, tl.float32)
    for c0 in range(0, n_cols, BLOCK):
        c = c0 + tl.arange(0, BLOCK)
        ok = c < n_cols
        at = row * n_cols + c
        grad = tl.load(acc_ptr + at, mask=ok, other=0).to(tl.float32) * srow * \
            tl.load(scol_ptr + c, mask=ok, other=0.0)
        gate = tl.load(gate_ptr + at, mask=ok, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + at, mask=ok, other=0.0).to(tl.float32)
        sig = tl.sigmoid(gate)
        top_u = tl.maximum(top_u, tl.max(tl.abs(grad * gate * sig), axis=0))
        top_g = tl.maximum(top_g, tl.max(tl.abs(grad * up * sig * (1.0 + gate * (1.0 - sig))),
                                         axis=0))
    s_g = tl.maximum(top_g, 1e-30) / 127.0
    s_u = tl.maximum(top_u, 1e-30) / 127.0
    tl.store(sdg_ptr + row, s_g)
    tl.store(sdu_ptr + row, s_u)
    for c0 in range(0, n_cols, BLOCK):
        c = c0 + tl.arange(0, BLOCK)
        ok = c < n_cols
        at = row * n_cols + c
        grad = tl.load(acc_ptr + at, mask=ok, other=0).to(tl.float32) * srow * \
            tl.load(scol_ptr + c, mask=ok, other=0.0)
        gate = tl.load(gate_ptr + at, mask=ok, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + at, mask=ok, other=0.0).to(tl.float32)
        sig = tl.sigmoid(gate)
        du = grad * gate * sig
        dg = grad * up * sig * (1.0 + gate * (1.0 - sig))
        tl.store(dg_ptr + at, tl.minimum(tl.maximum(libdevice.rint(dg / s_g), -127.0), 127.0)
                 .to(tl.int8), mask=ok)
        tl.store(du_ptr + at, tl.minimum(tl.maximum(libdevice.rint(du / s_u), -127.0), 127.0)
                 .to(tl.int8), mask=ok)


@triton.jit
def kz_dequant_sum_kernel(a_ptr, sa_row_ptr, sa_col_ptr, b_ptr, sb_row_ptr, sb_col_ptr, out_ptr,
                          n_cols, BLOCK: tl.constexpr):
    # out = a * sa_row * sa_col + b * sb_row * sb_col (two int32 GEMM results, fp32 out)
    row = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ok = c < n_cols
    at = row * n_cols + c
    a = tl.load(a_ptr + at, mask=ok, other=0).to(tl.float32) * tl.load(sa_row_ptr + row) * \
        tl.load(sa_col_ptr + c, mask=ok, other=0.0)
    b = tl.load(b_ptr + at, mask=ok, other=0).to(tl.float32) * tl.load(sb_row_ptr + row) * \
        tl.load(sb_col_ptr + c, mask=ok, other=0.0)
    tl.store(out_ptr + at, a + b, mask=ok)


def quant_rows(x, rounding=None):
    """x [rows, n] -> (int8 values, fp32 row scales). ``rounding``: a ``policy.Rounding``;
    ``None`` rounds to nearest, otherwise the gradient quantiser rounds stochastically
    where the run's rounding says so."""
    rows, n = x.shape
    q = torch.empty(rows, n, device=x.device, dtype=torch.int8)
    s = torch.empty(rows, device=x.device, dtype=torch.float32)
    sr = rounding is not None and rounding.grad
    kz_quant_rows_kernel[(rows,)](x, q, s, n, rounding.grad_seed() if sr else 0, BLOCK=2048,
                                  SR=sr, num_warps=8)
    return q, s


def swiglu_bwd_int8(acc, srow, scol, gate, up):
    rows, n = acc.shape
    dg = torch.empty(rows, n, device=acc.device, dtype=torch.int8)
    du = torch.empty_like(dg)
    sdg = torch.empty(rows, device=acc.device, dtype=torch.float32)
    sdu = torch.empty_like(sdg)
    kz_swiglu_bwd_int8_kernel[(rows,)](acc, srow, scol, gate, up, dg, du, sdg, sdu, n,
                                       BLOCK=2048, num_warps=8)
    return dg, sdg, du, sdu


def dequant_sum(a, sa_row, sa_col, b, sb_row, sb_col):
    rows, n = a.shape
    out = torch.empty(rows, n, device=a.device, dtype=torch.float32)
    kz_dequant_sum_kernel[(rows, triton.cdiv(n, 1024))](a, sa_row, sa_col, b, sb_row, sb_col, out,
                                                        n, BLOCK=1024, num_warps=4)
    return out
