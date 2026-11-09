# SPDX-License-Identifier: Apache-2.0
#
# Origin: Soup Kernel Challenge (Decentrathon 6.0), the submission of team Luck
# (BAITC-Hacks/hack-e79529fc-luck@b3f1225f); the int8 gate/up packs follow ideas seen in the
# solutions of teams Null Pointers, 21, DarkDaur and OrbiTeam.
# Modified for Soup: the rounding is an argument; pack sizes come from layout.py so the
# planner can use them without Triton; the layers always pack gate and up together.
# ruff: noqa: N803, N806  (Triton's constexpr arguments are upper case by convention)
"""gate and up go to host memory as int8 (one fp32 scale per 64 values) instead of fp16.

With activations offloaded (offload.py), the MLP's gate and up projections are the largest tensors
that cross PCIe: 2 x 8960 fp16 values per token per layer, out in the forward and back in the
backward. A Kaggle T4 step is largely bound by that link (profile: ~240 ms of host-to-device and
~90 ms of device-to-host copies against ~345 ms of compute per step). Their only reader is the
backward's SwiGLU, which feeds the int8 tensor-core GEMMs of int8.py anyway, so they are stored in
the same precision those GEMMs use. A stored tensor is ONE flat int8 buffer, "pack": the int8 values
of the whole [rows, cols] tensor, then one fp32 scale per group of 64 consecutive values of a row
(scale = max |x| / 127 of the group, q = round(x / scale)); a pack is 1.88x smaller than the fp16
tensor and moves with one copy each way.
- kz_swiglu_fwd_q8_kernel computes silu(gate) * up exactly as the elementwise SwiGLU kernel does
  (same fp32 formula, same rounding to the compute dtype: the forward and the loss are unchanged)
  and, in the same pass, packs gate and up, after up's GEMM. (With PACK_GATE off it packs up
  only, for a gate packed by kz_quant_q8_kernel right after its own GEMM; the layers here do
  not use that option.)
- kz_swiglu_bwd_q8_kernel is int8.kz_swiglu_bwd_int8_kernel reading the two packs.
Only backward's d_gate / d_up see the rounding (~0.4% of a group's largest value), on top of the
int8 GEMMs they already go through. Idea seen in the solutions of Null Pointers, 21, DarkDaur
and OrbiTeam; these kernels are written here.

The kernels are written for the T4, where a kernel that moves ~90 MB is bound by memory (~250 GB/s)
but also by the instructions per value (the fp32 and integer pipes are 16 lanes wide per
sub-partition, conversions and sigmoid's exp / reciprocal 4 lanes). So:
- a group of 64 consecutive values of a row is a row of a [rows * groups, 64] matrix (n_cols is a
  multiple of 64), and a program takes whole groups: no mask except at the end, one scale per
  group, loaded once for the group, not once per value;
- rounding to int8 uses the fp32 adder, not division, rint, clamp and a float -> int convert:
  x * inv + 1.5 * 2**23 (inv = 127 / max |group|, one division per group) rounds x * inv to the
  nearest integer (ties to even, as rint does) and leaves it in the low mantissa bits, so the low
  byte of the sum's bits is the int8 (two's complement); |x * inv| is 127 up to rounding error, so
  no clamp is needed;
- the backward holds the whole row in registers (three tiles of up to 64 groups) and quantizes
  d_gate / d_up once the row's maxima are known, instead of a second pass that recomputes them.

Triton compiles a kernel, and builds its C launcher, at its first launch (~0.5-1 s each). No
argument's divisibility by 16 depends on the batch size (the group count is n_values // GROUP
inside the kernels), so a kernel compiles once per shape class, not once per batch size.
"""

import torch
import triton
import triton.language as tl

from ..layout import Q8_GROUP, q8_pack_bytes, q8_usable_width

GROUP = Q8_GROUP
pack_bytes = q8_pack_bytes  # the planner shares these two (layout.py)
usable_width = q8_usable_width
BLOCK_G = 16  # groups per program of the forward kernels (1024 values, the swiglu kernel's BLOCK)
FWD_WARPS = 4
MAX_GROUPS = 192  # backward, one-pass kernel: a row is three tiles of up to 64 groups (12288
#                   columns); wider rows (Qwen2.5-7B's 18944) and rows with the LoRA terms of
#                   swiglu_bwd_q8 use the two-pass kernel instead (any width)
LOOP_TILE_G = 32  # groups per tile of the two-pass kernel (2048 values)
BWD_WARPS = 8


@triton.jit
def _int8(x, inv):
    # round-half-even(x * inv) as int8: the fp32 add rounds to an integer in the low mantissa bits
    # 1.5 * 2**23 (valid for |x * inv| < 2**22); the low byte of those bits is the two's complement
    return (x * inv + 12582912.0).to(tl.int32, bitcast=True).to(tl.int8)


@triton.jit
def _scales(pack_ptr, n_values):
    # the fp32 scales sit behind the n_values int8 values (n_values is a multiple of 4)
    return (pack_ptr + n_values).to(tl.pointer_type(tl.float32))


@triton.jit
def kz_quant_q8_kernel(x_ptr, pack_ptr, n_values, GROUP: tl.constexpr, BLOCK_G: tl.constexpr):
    # BLOCK_G groups of GROUP consecutive values: pack = int8 values, then one scale per group.
    # (The group count is n_values // GROUP, not an argument: an argument that is a multiple of 16
    # only for some batch sizes would make Triton compile a second variant mid-run.)
    g = tl.program_id(0).to(tl.int64) * BLOCK_G + tl.arange(0, BLOCK_G)
    g_ok = g < n_values // GROUP
    at = g[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
    x = tl.load(x_ptr + at, mask=g_ok[:, None], other=0.0).to(tl.float32)
    top = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-30)
    tl.store(pack_ptr + at, _int8(x, (127.0 / top)[:, None]), mask=g_ok[:, None])
    tl.store(_scales(pack_ptr, n_values) + g, top / 127.0, mask=g_ok)


@triton.jit
def kz_swiglu_fwd_q8_kernel(gate_ptr, up_ptr, act_ptr, gpack_ptr, upack_ptr, n_values,
                            GROUP: tl.constexpr, BLOCK_G: tl.constexpr, PACK_GATE: tl.constexpr):
    # act = gate * sigmoid(gate) * up in fp32 -> fp16 (the example kernel's formula), and up packed;
    # gate too if PACK_GATE (else the caller packed it already, quant_q8)
    g = tl.program_id(0).to(tl.int64) * BLOCK_G + tl.arange(0, BLOCK_G)
    g_ok = g < n_values // GROUP
    at = g[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
    gate = tl.load(gate_ptr + at, mask=g_ok[:, None], other=0.0).to(tl.float32)
    up = tl.load(up_ptr + at, mask=g_ok[:, None], other=0.0).to(tl.float32)
    act = gate * tl.sigmoid(gate) * up
    tl.store(act_ptr + at, act.to(act_ptr.dtype.element_ty), mask=g_ok[:, None])
    top = tl.maximum(tl.max(tl.abs(up), axis=1), 1e-30)
    tl.store(upack_ptr + at, _int8(up, (127.0 / top)[:, None]), mask=g_ok[:, None])
    tl.store(_scales(upack_ptr, n_values) + g, top / 127.0, mask=g_ok)
    if PACK_GATE:
        top = tl.maximum(tl.max(tl.abs(gate), axis=1), 1e-30)
        tl.store(gpack_ptr + at, _int8(gate, (127.0 / top)[:, None]), mask=g_ok[:, None])
        tl.store(_scales(gpack_ptr, n_values) + g, top / 127.0, mask=g_ok)


@triton.jit
def _swiglu_bwd_tile(g0, G: tl.constexpr, row, srow, acc_ptr, scol_ptr, qg_ptr, qu_ptr, sg_ptr,
                     su_ptr, n_cols, n_groups, GROUP: tl.constexpr, extra_ptr=None,
                     HAS_EXTRA: tl.constexpr = False):
    # d_gate, d_up (fp32) of the G groups from g0 of a row: int8.kz_swiglu_bwd_int8_kernel formulas;
    # with HAS_EXTRA the gradient of the activation also gets extra (the down projection's LoRA
    # term g_mid @ A, in the compute dtype) added
    gi = g0 + tl.arange(0, G)
    g_ok = gi < n_groups
    c = gi[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
    ok = g_ok[:, None]
    at = row * n_cols + c
    grad = tl.load(acc_ptr + at, mask=ok, other=0).to(tl.float32) * \
        (srow * tl.load(scol_ptr + c, mask=ok, other=0.0))
    if HAS_EXTRA:
        grad += tl.load(extra_ptr + at, mask=ok, other=0.0).to(tl.float32)
    gate = tl.load(qg_ptr + at, mask=ok, other=0).to(tl.float32) * \
        tl.load(sg_ptr + row * n_groups + gi, mask=g_ok, other=0.0)[:, None]
    up = tl.load(qu_ptr + at, mask=ok, other=0).to(tl.float32) * \
        tl.load(su_ptr + row * n_groups + gi, mask=g_ok, other=0.0)[:, None]
    sig = tl.sigmoid(gate)
    return grad * up * sig * (1.0 + gate * (1.0 - sig)), grad * gate * sig


@triton.jit
def _int8_sr(x, inv, seed, at):
    # stochastic rounding: floor(x * inv + u), u uniform in [0, 1) (|x * inv| <= 127)
    u = tl.rand(seed, at.to(tl.int32))
    return tl.minimum(tl.maximum(tl.floor(x * inv + u), -127.0), 127.0).to(tl.int8)


@triton.jit
def _store_tile(g0, G: tl.constexpr, row, dg, du, inv_g, inv_u, dg_ptr, du_ptr, n_cols, n_groups,
                GROUP: tl.constexpr, seed, SR: tl.constexpr):
    gi = g0 + tl.arange(0, G)
    at = row * n_cols + gi[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
    ok = (gi < n_groups)[:, None]
    if SR:
        tl.store(dg_ptr + at, _int8_sr(dg, inv_g, seed, at), mask=ok)
        tl.store(du_ptr + at, _int8_sr(du, inv_u, seed + 7919, at), mask=ok)
    else:
        tl.store(dg_ptr + at, _int8(dg, inv_g), mask=ok)
        tl.store(du_ptr + at, _int8(du, inv_u), mask=ok)


@triton.jit
def _tile_top(x):
    return tl.max(tl.max(tl.abs(x), axis=1), axis=0)


@triton.jit(do_not_specialize=["seed"])
def kz_swiglu_bwd_q8_kernel(acc_ptr, srow_ptr, scol_ptr, gpack_ptr, upack_ptr, dg_ptr, du_ptr,
                            sdg_ptr, sdu_ptr, n_cols, n_values, n_groups, seed, GROUP: tl.constexpr,
                            GA: tl.constexpr, GB: tl.constexpr, GC: tl.constexpr,
                            SR: tl.constexpr = False):
    # int8.kz_swiglu_bwd_int8_kernel with gate / up read from their packs. One row: g_act = acc *
    # srow * scol (the int8 GEMM's int32 result, scaled back), the SwiGLU backward, d_gate and d_up
    # quantized to int8 with one scale each per row. The row is three tiles of GA, GB, GC groups
    # whose d_gate / d_up stay in registers until the row's maxima are known (one pass over the
    # inputs; the int8 kernel's second pass recomputes them)
    row = tl.program_id(0).to(tl.int64)
    srow = tl.load(srow_ptr + row)
    sg_ptr = _scales(gpack_ptr, n_values)
    su_ptr = _scales(upack_ptr, n_values)
    dgA, duA = _swiglu_bwd_tile(0, GA, row, srow, acc_ptr, scol_ptr, gpack_ptr, upack_ptr, sg_ptr,
                                su_ptr, n_cols, n_groups, GROUP)
    dgB, duB = _swiglu_bwd_tile(GA, GB, row, srow, acc_ptr, scol_ptr, gpack_ptr, upack_ptr, sg_ptr,
                                su_ptr, n_cols, n_groups, GROUP)
    dgC, duC = _swiglu_bwd_tile(GA + GB, GC, row, srow, acc_ptr, scol_ptr, gpack_ptr, upack_ptr,
                                sg_ptr, su_ptr, n_cols, n_groups, GROUP)
    top_g = tl.maximum(
        tl.maximum(tl.maximum(_tile_top(dgA), _tile_top(dgB)), _tile_top(dgC)), 1e-30)
    top_u = tl.maximum(
        tl.maximum(tl.maximum(_tile_top(duA), _tile_top(duB)), _tile_top(duC)), 1e-30)
    tl.store(sdg_ptr + row, top_g / 127.0)
    tl.store(sdu_ptr + row, top_u / 127.0)
    inv_g = 127.0 / top_g
    inv_u = 127.0 / top_u
    _store_tile(0, GA, row, dgA, duA, inv_g, inv_u, dg_ptr, du_ptr, n_cols, n_groups, GROUP, seed,
                SR)
    _store_tile(GA, GB, row, dgB, duB, inv_g, inv_u, dg_ptr, du_ptr, n_cols, n_groups, GROUP, seed,
                SR)
    _store_tile(GA + GB, GC, row, dgC, duC, inv_g, inv_u, dg_ptr, du_ptr, n_cols, n_groups, GROUP,
                seed, SR)


@triton.jit(do_not_specialize=["seed"])
def kz_swiglu_bwd_q8_loop_kernel(acc_ptr, srow_ptr, scol_ptr, gpack_ptr, upack_ptr, extra_ptr,
                                 dg_ptr, du_ptr, sdg_ptr, sdu_ptr, dgf_ptr, duf_ptr, n_cols,
                                 n_values, n_groups, scale_g, scale_u, seed, GROUP: tl.constexpr,
                                 TILE_G: tl.constexpr, HAS_EXTRA: tl.constexpr,
                                 WRITE_FP: tl.constexpr, PRESCALE: tl.constexpr,
                                 SR: tl.constexpr = False):
    # kz_swiglu_bwd_q8_kernel for rows of any width: two passes over TILE_G groups at a time (the
    # row's maxima first, then the same formulas again, quantizing), instead of holding the
    # whole row in registers. Same values. HAS_EXTRA: the LoRA term of the activation's gradient
    # is added (see _swiglu_bwd_tile); WRITE_FP: d_gate and d_up are also written in the compute
    # dtype (the LoRA gradients of gate_proj and up_proj read them: no int8 rounding in those)
    row = tl.program_id(0).to(tl.int64)
    srow = tl.load(srow_ptr + row)
    sg_ptr = _scales(gpack_ptr, n_values)
    su_ptr = _scales(upack_ptr, n_values)
    top_g = tl.full((), 1e-30, tl.float32)
    top_u = tl.full((), 1e-30, tl.float32)
    for g0 in range(0, n_groups, TILE_G):
        dg, du = _swiglu_bwd_tile(g0, TILE_G, row, srow, acc_ptr, scol_ptr, gpack_ptr, upack_ptr,
                                  sg_ptr, su_ptr, n_cols, n_groups, GROUP, extra_ptr, HAS_EXTRA)
        top_g = tl.maximum(top_g, _tile_top(dg))
        top_u = tl.maximum(top_u, _tile_top(du))
    tl.store(sdg_ptr + row, top_g / 127.0)
    tl.store(sdu_ptr + row, top_u / 127.0)
    inv_g = 127.0 / top_g
    inv_u = 127.0 / top_u
    for g0 in range(0, n_groups, TILE_G):
        dg, du = _swiglu_bwd_tile(g0, TILE_G, row, srow, acc_ptr, scol_ptr, gpack_ptr, upack_ptr,
                                  sg_ptr, su_ptr, n_cols, n_groups, GROUP, extra_ptr, HAS_EXTRA)
        _store_tile(g0, TILE_G, row, dg, du, inv_g, inv_u, dg_ptr, du_ptr, n_cols, n_groups, GROUP,
                    seed, SR)
        if WRITE_FP:
            gi = g0 + tl.arange(0, TILE_G)
            at = row * n_cols + gi[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
            ok = (gi < n_groups)[:, None]
            dg16 = dg.to(dgf_ptr.dtype.element_ty)
            du16 = du.to(duf_ptr.dtype.element_ty)
            if PRESCALE:
                # torch's `g * scaling` of the compute-dtype d_gate / d_up (product in fp32, one
                # more rounding), done here instead of in an elementwise pass of its own
                dg16 = (dg16.to(tl.float32) * scale_g).to(dgf_ptr.dtype.element_ty)
                du16 = (du16.to(tl.float32) * scale_u).to(duf_ptr.dtype.element_ty)
            tl.store(dgf_ptr + at, dg16, mask=ok)
            tl.store(duf_ptr + at, du16, mask=ok)


# --- the forward's LoRA epilogues, SwiGLU and int8 packing in one pass ----------------------
# kernel idea: soupk/mlp.py swiglu_lora_fwd_kernel (tiles of BT rows x BI columns, the LoRA terms
# as tl.dot against the small [rows, r] intermediates); the output format is this file's packs
FUSED_BT = 32
FUSED_BI = 128
FUSED_WARPS = 4
FUSED_STAGES = 2


@triton.jit(do_not_specialize=["T", "n_values"])
def kz_swiglu_lora_fwd_q8_kernel(gate_ptr, up_ptr, hg_ptr, hu_ptr, bg_ptr, bu_ptr, ad_ptr, act_ptr,
                                 gpack_ptr, upack_ptr, hd_ptr, scale_g, scale_u, T, INTER, n_values,
                                 R: tl.constexpr, BT: tl.constexpr, BI: tl.constexpr,
                                 GROUP: tl.constexpr, HAS_G: tl.constexpr, HAS_U: tl.constexpr,
                                 HAS_D: tl.constexpr):
    # gate, up: the base GEMMs' outputs [T, I] (compute dtype). Per tile: gate = round(gate +
    # scale_g * hg @ bg^T) (what `gate.addmm_(mid, B^T, alpha=scaling)` does: fp32 product and
    # sum, one rounding), up likewise, act = gate * sigmoid(gate) * up (fp32, rounded), the two
    # packs of (rounded) gate and up (one fp32 scale per GROUP values of a row, behind the int8
    # values: kz_quant_q8_kernel's format), and hd += act @ ad^T (the down projection's LoRA
    # intermediate, summed over the columns of the row block)
    dt = act_ptr.dtype.element_ty
    rows = tl.program_id(0) * BT + tl.arange(0, BT)
    rm = rows < T
    rr = tl.arange(0, R)
    if HAS_G:
        hg = tl.load(hg_ptr + rows[:, None] * R + rr[None, :], mask=rm[:, None], other=0.0)
    if HAS_U:
        hu = tl.load(hu_ptr + rows[:, None] * R + rr[None, :], mask=rm[:, None], other=0.0)
    if HAS_D:
        hd = tl.zeros([BT, R], tl.float32)
    row_at = rows[:, None].to(tl.int64) * INTER
    NGB: tl.constexpr = BI // GROUP
    for i0 in range(0, INTER, BI):
        cols = i0 + tl.arange(0, BI)
        cm = cols < INTER
        m = rm[:, None] & cm[None, :]
        at = row_at + cols[None, :]
        g = tl.load(gate_ptr + at, mask=m, other=0.0).to(tl.float32)
        u = tl.load(up_ptr + at, mask=m, other=0.0).to(tl.float32)
        if HAS_G:
            bgT = tl.load(bg_ptr + cols[None, :] * R + rr[:, None], mask=cm[None, :], other=0.0)
            g = (g + scale_g * tl.dot(hg, bgT)).to(dt).to(tl.float32)
        if HAS_U:
            buT = tl.load(bu_ptr + cols[None, :] * R + rr[:, None], mask=cm[None, :], other=0.0)
            u = (u + scale_u * tl.dot(hu, buT)).to(dt).to(tl.float32)
        act = (g * tl.sigmoid(g) * u).to(dt)
        tl.store(act_ptr + at, act, mask=m)
        if HAS_D:
            adT = tl.load(ad_ptr + rr[None, :] * INTER + cols[:, None], mask=cm[:, None], other=0.0)
            hd = tl.dot(act, adT, hd)
        # packs: groups of GROUP consecutive columns of a row
        g3 = tl.reshape(g, (BT, NGB, GROUP))
        u3 = tl.reshape(u, (BT, NGB, GROUP))
        top_g = tl.maximum(tl.max(tl.abs(g3), axis=2), 1e-30)
        top_u = tl.maximum(tl.max(tl.abs(u3), axis=2), 1e-30)
        qg = tl.reshape(_int8(g3, (127.0 / top_g)[:, :, None]), (BT, BI))
        qu = tl.reshape(_int8(u3, (127.0 / top_u)[:, :, None]), (BT, BI))
        tl.store(gpack_ptr + at, qg, mask=m)
        tl.store(upack_ptr + at, qu, mask=m)
        gid = (rows[:, None].to(tl.int64) * (INTER // GROUP) + i0 // GROUP
               + tl.arange(0, NGB)[None, :])
        gm = rm[:, None] & ((i0 + tl.arange(0, NGB) * GROUP) < INTER)[None, :]
        tl.store(_scales(gpack_ptr, n_values) + gid, top_g / 127.0, mask=gm)
        tl.store(_scales(upack_ptr, n_values) + gid, top_u / 127.0, mask=gm)
    if HAS_D:
        tl.store(hd_ptr + rows[:, None] * R + rr[None, :], hd.to(dt), mask=rm[:, None])


def fused_fwd_usable(inter, rank):
    """Whether swiglu_lora_fwd_q8 handles this shape: whole groups, and `tl.dot` wants the LoRA
    rank to be a power of two of at least 16."""
    return inter % GROUP == 0 and rank >= 16 and rank & (rank - 1) == 0


def swiglu_lora_fwd_q8(gate, up, hg=None, bg=None, scale_g=1.0, hu=None, bu=None, scale_u=1.0,
                       a_down=None):
    """(act, gate pack, up pack, hd) from the base GEMMs' outputs gate, up [T, I] (compute dtype):
    the LoRA terms of gate (hg [T, r] @ bg^T [r, I] * scale_g; bg is peft's lora_B weight [I, r])
    and up are added in the kernel, act = silu(gate) * up, both packed (q8 format), and, with
    a_down (peft's lora_A weight of the down projection, [r, I]), hd = act @ a_down^T [T, r]
    (None without it). A missing adapter is skipped. Same values as the three torch ops it
    replaces (`addmm_` per adapter, swiglu_fwd_q8, `act @ a_down.t()`) up to the order of the
    fp32 sums."""
    rows, n = gate.shape
    rank = next(t.shape[-1] if t is bg or t is bu else t.shape[0]
                for t in (bg, bu, a_down) if t is not None)
    act = torch.empty_like(gate)
    gate_pack, up_pack = _empty_pack(gate), _empty_pack(up)
    hd = None
    if a_down is not None:
        hd = torch.empty(rows, rank, device=gate.device, dtype=gate.dtype)
    dummy = gate
    kz_swiglu_lora_fwd_q8_kernel[(triton.cdiv(rows, FUSED_BT),)](
        gate, up, dummy if hg is None else hg, dummy if hu is None else hu,
        dummy if bg is None else bg, dummy if bu is None else bu,
        dummy if a_down is None else a_down, act, gate_pack, up_pack, dummy if hd is None else hd,
        float(scale_g), float(scale_u), rows, n, rows * n, R=rank, BT=FUSED_BT, BI=FUSED_BI,
        GROUP=GROUP, HAS_G=bg is not None, HAS_U=bu is not None, HAS_D=a_down is not None,
        num_warps=FUSED_WARPS, num_stages=FUSED_STAGES)
    return act, gate_pack, up_pack, hd


@triton.jit
def kz_act_from_q8_kernel(gpack_ptr, upack_ptr, act_ptr, n_values, GROUP: tl.constexpr,
                          BLOCK_G: tl.constexpr):
    # act = silu(gate) * up from the two packs (what the backward's int8 SwiGLU sees), in the
    # compute dtype: the down projection's LoRA gradient (dA) reads it
    g = tl.program_id(0).to(tl.int64) * BLOCK_G + tl.arange(0, BLOCK_G)
    g_ok = g < n_values // GROUP
    at = g[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
    ok = g_ok[:, None]
    gate = tl.load(gpack_ptr + at, mask=ok, other=0).to(tl.float32) * \
        tl.load(_scales(gpack_ptr, n_values) + g, mask=g_ok, other=0.0)[:, None]
    up = tl.load(upack_ptr + at, mask=ok, other=0).to(tl.float32) * \
        tl.load(_scales(upack_ptr, n_values) + g, mask=g_ok, other=0.0)[:, None]
    tl.store(act_ptr + at, (gate * tl.sigmoid(gate) * up).to(act_ptr.dtype.element_ty), mask=ok)


@triton.jit
def kz_dequant_q8_kernel(pack_ptr, out_ptr, n_values, GROUP: tl.constexpr, BLOCK_G: tl.constexpr):
    # the pack's values times their group scales, in the output's dtype (INT8_BACKWARD = "stash":
    # the backward reads the int8 stash but runs the compute-dtype GEMMs)
    g = tl.program_id(0).to(tl.int64) * BLOCK_G + tl.arange(0, BLOCK_G)
    g_ok = g < n_values // GROUP
    at = g[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
    ok = g_ok[:, None]
    x = tl.load(pack_ptr + at, mask=ok, other=0).to(tl.float32) * \
        tl.load(_scales(pack_ptr, n_values) + g, mask=g_ok, other=0.0)[:, None]
    tl.store(out_ptr + at, x.to(out_ptr.dtype.element_ty), mask=ok)


def dequant_q8(pack, rows, n, dtype):
    """[rows, n] in `dtype` from a pack made by quant_q8 / swiglu_fwd_q8."""
    out = torch.empty(rows, n, device=pack.device, dtype=dtype)
    kz_dequant_q8_kernel[(triton.cdiv(rows * n // GROUP, BLOCK_G),)](
        pack, out, rows * n, GROUP=GROUP, BLOCK_G=BLOCK_G, num_warps=FWD_WARPS)
    return out


def _empty_pack(x):
    rows, n = x.shape
    return torch.empty(rows * pack_bytes(n), device=x.device, dtype=torch.int8)


def quant_q8(x):
    """fp16 [rows, cols] -> its pack (int8 values, then fp32 scales per group of 64 of a row)."""
    pack = _empty_pack(x)
    kz_quant_q8_kernel[(triton.cdiv(x.numel() // GROUP, BLOCK_G),)](
        x, pack, x.numel(), GROUP=GROUP, BLOCK_G=BLOCK_G, num_warps=FWD_WARPS)
    return pack


def swiglu_fwd_q8(gate, up, gate_pack=None):
    """(act fp16, pack of gate, pack of up) from fp16 gate / up. With gate_pack (quant_q8's) given,
    only up is packed here; without it both are, in the same pass."""
    up_pack = _empty_pack(up)
    pack_gate = gate_pack is None
    if pack_gate:
        gate_pack = _empty_pack(gate)
    act = torch.empty_like(gate)
    kz_swiglu_fwd_q8_kernel[(triton.cdiv(up.numel() // GROUP, BLOCK_G),)](
        gate, up, act, gate_pack, up_pack, up.numel(), GROUP=GROUP, BLOCK_G=BLOCK_G,
        PACK_GATE=pack_gate, num_warps=FWD_WARPS)
    return act, gate_pack, up_pack


def _tiles(n_groups):
    """Sizes (powers of two, at most 64 groups) of the backward's three tiles: 64 + 64 + 16 for the
    task's 140 groups; a tile that starts past the row is fully masked."""
    sizes, left = [], n_groups
    for _ in range(3):
        size = min(64, 1 << max(left - 1, 0).bit_length())
        sizes.append(size)
        left = max(left - size, 0)
    return sizes


def swiglu_bwd_q8(acc, srow, scol, gpack, upack, extra=None, fp_dtype=None, fp_scales=None,
                  rounding=None):
    """int8.swiglu_bwd_int8 for gate / up packed by quant_q8 / swiglu_fwd_q8. extra: [rows, n] in
    the compute dtype, added to the activation's gradient (the down projection's LoRA term).
    fp_dtype: also return d_gate and d_up in that dtype (rounded from the fp32 values the int8
    ones are made from). fp_scales = (scale_gate, scale_up): the fp outputs come back already
    multiplied by the LoRA scalings (gs = g * scaling as torch rounds it). Returns (dg8, sdg, du8,
    sdu) or (dg8, sdg, du8, sdu, dgate, dup). A row
    of up to MAX_GROUPS groups without extra / fp_dtype goes through the one-pass kernel, any
    other through the two-pass one. rounding: a ``policy.Rounding``; the int8 gradients round
    stochastically where it says so (None: to nearest)."""
    sr = rounding is not None and rounding.grad
    seed = rounding.grad_seed() if sr else 0
    rows, n = acc.shape
    dg = torch.empty(rows, n, device=acc.device, dtype=torch.int8)
    du = torch.empty_like(dg)
    sdg = torch.empty(rows, device=acc.device, dtype=torch.float32)
    sdu = torch.empty_like(sdg)
    if extra is None and fp_dtype is None and n // GROUP <= MAX_GROUPS:
        ga, gb, gc = _tiles(n // GROUP)
        kz_swiglu_bwd_q8_kernel[(rows,)](acc, srow, scol, gpack, upack, dg, du, sdg, sdu, n,
                                         rows * n, n // GROUP, seed, GROUP=GROUP, GA=ga, GB=gb,
                                         GC=gc, SR=sr, num_warps=BWD_WARPS)
        return dg, sdg, du, sdu
    dgf = duf = None
    if fp_dtype is not None:
        dgf = torch.empty(rows, n, device=acc.device, dtype=fp_dtype)
        duf = torch.empty_like(dgf)
    kz_swiglu_bwd_q8_loop_kernel[(rows,)](
        acc, srow, scol, gpack, upack, acc if extra is None else extra, dg, du, sdg, sdu,
        dg if dgf is None else dgf, dg if duf is None else duf, n, rows * n, n // GROUP,
        1.0 if fp_scales is None else fp_scales[0], 1.0 if fp_scales is None else fp_scales[1],
        seed, GROUP=GROUP, TILE_G=LOOP_TILE_G, HAS_EXTRA=extra is not None,
        WRITE_FP=dgf is not None, PRESCALE=fp_scales is not None and dgf is not None, SR=sr,
        num_warps=BWD_WARPS)
    if fp_dtype is None:
        return dg, sdg, du, sdu
    return dg, sdg, du, sdu, dgf, duf


def act_from_q8(gpack, upack, rows, n, dtype):
    """silu(gate) * up, [rows, n] in `dtype`, from the two packs (kz_act_from_q8_kernel)."""
    act = torch.empty(rows, n, device=gpack.device, dtype=dtype)
    kz_act_from_q8_kernel[(triton.cdiv(rows * n // GROUP, BLOCK_G),)](
        gpack, upack, act, rows * n, GROUP=GROUP, BLOCK_G=BLOCK_G, num_warps=FWD_WARPS)
    return act
