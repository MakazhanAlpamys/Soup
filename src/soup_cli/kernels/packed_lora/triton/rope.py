# SPDX-License-Identifier: Apache-2.0
#
# Origin: Soup Kernel Challenge (Decentrathon 6.0), the submission of team Luck
# (BAITC-Hacks/hack-e79529fc-luck@b3f1225f); the int8 gate/up packs follow ideas seen in the
# solutions of teams Null Pointers, 21, DarkDaur and OrbiTeam.
# Modified for Soup: the LoRA-dropout masks and dropout inputs of the original are not ported
# (dropout is 0).
# ruff: noqa: N803, N806  (Triton's constexpr arguments are upper case by convention)
"""The step between the fused q/k/v matmul and attention, forward and backward, one launch each.

Base Soup does this in many small ops: each 4-bit layer's fp16 (bf16) output is cast to fp32, the
LoRA delta (fp16, already times scaling) is added in fp32, rotary embedding runs in fp32 with fp32
cos/sin, k and v are repeated for GQA in fp32, and SDPA casts everything back to fp16.

Here one program per token reads the packed [q | k | v] row of the matmul and the LoRA-B outputs
of whichever of q, k, v carry a LoRA (times scaling, rounded to the compute dtype as the base's
multiply rounds them, here rather than in an op of its own) and writes q, k and v in the compute
dtype (fp16 or bf16), k and v already repeated to the query heads: exactly the tensors the base's
SDPA sees, in the [tokens, heads, dim] layout PyTorch's memory-efficient attention kernel reads
when it runs over all rows at once (see layers.py). Backward reads that kernel's per-head dk and dv
and sums each kv head's copies in fp32, as the base's fp32 repeat_kv backward does; it also writes
the q, k and v parts of its output times scaling (the gradients of the LoRA-B outputs) and (stashed
layers, see layers.py) the token's LoRA input again: the norm output itself, which would
otherwise take a norm launch of its own. Positions restart at 0
for every row of the batch, so `pos` indexes cos/sin. The head counts, the head size and which
projections carry a LoRA are compile-time arguments (any GQA ratio, kv heads a power of two).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kz_qkv_rope_fwd_kernel(qkv_ptr, lq_ptr, lk_ptr, lv_ptr, cos_ptr, sin_ptr, pos_ptr,
                           q_ptr, k_ptr, v_ptr, scaling,
                           qw_ptr, kw_ptr, xq_ptr, xk_ptr, rq_ptr, rk_ptr, eps,
                           QH: tl.constexpr, QH_POW2: tl.constexpr, KH: tl.constexpr,
                           D: tl.constexpr, HAS_LQ: tl.constexpr, HAS_LK: tl.constexpr,
                           HAS_LV: tl.constexpr, NORM: tl.constexpr):
    dt = q_ptr.dtype.element_ty  # fp16 or bf16
    t = tl.program_id(0).to(tl.int64)
    p = tl.load(pos_ptr + t).to(tl.int64)
    half = tl.arange(0, D // 2)[None, :]
    c1 = tl.load(cos_ptr + p * D + half)
    c2 = tl.load(cos_ptr + p * D + D // 2 + half)
    s1 = tl.load(sin_ptr + p * D + half)
    s2 = tl.load(sin_ptr + p * D + D // 2 + half)
    width = (QH + 2 * KH) * D
    src = qkv_ptr + t * width
    hq = tl.arange(0, QH_POW2)[:, None]
    q_ok = hq < QH
    lo = hq * D + half
    hi = lo + D // 2
    row_out = t * (QH * D)

    # q: base 4-bit output (compute dtype, bias inside the matmul) + LoRA delta, added in fp32.
    # The delta is the LoRA-B matmul's output times scaling, rounded to the compute dtype as
    # torch's multiply by a Python float rounds it (product in fp32, one rounding)
    q1 = tl.load(src + lo, mask=q_ok, other=0.0).to(tl.float32)
    q2 = tl.load(src + hi, mask=q_ok, other=0.0).to(tl.float32)
    if HAS_LQ:
        lq1 = tl.load(lq_ptr + row_out + lo, mask=q_ok, other=0.0).to(tl.float32)
        q1 += (lq1 * scaling).to(dt).to(tl.float32)
        lq2 = tl.load(lq_ptr + row_out + hi, mask=q_ok, other=0.0).to(tl.float32)
        q2 += (lq2 * scaling).to(dt).to(tl.float32)
    if NORM:
        # Qwen3: RMSNorm of every query head (weight [D], fp32 like the base run's q_proj output)
        # between the projection (+ LoRA) and the rotary embedding; the normalised rows x_hat
        # (compute dtype) and the heads' rstd (fp32) are kept for the backward
        heads = tl.arange(0, QH_POW2)
        rs = tl.math.rsqrt((tl.sum(q1 * q1, axis=1) + tl.sum(q2 * q2, axis=1)) / D + eps)
        xh1 = q1 * rs[:, None]
        xh2 = q2 * rs[:, None]
        tl.store(xq_ptr + row_out + lo, xh1.to(dt), mask=q_ok)
        tl.store(xq_ptr + row_out + hi, xh2.to(dt), mask=q_ok)
        tl.store(rq_ptr + t * QH + heads, rs, mask=heads < QH)
        q1 = tl.load(qw_ptr + half) * xh1
        q2 = tl.load(qw_ptr + D // 2 + half) * xh2
    tl.store(q_ptr + row_out + lo, (q1 * c1 - q2 * s1).to(dt), mask=q_ok)
    tl.store(q_ptr + row_out + hi, (q2 * c2 + q1 * s2).to(dt), mask=q_ok)

    # k and v: query head h reads kv head h // (QH // KH)
    kv_lo = (hq // (QH // KH)) * D + half
    kv_hi = kv_lo + D // 2
    k1 = tl.load(src + QH * D + kv_lo, mask=q_ok, other=0.0).to(tl.float32)
    k2 = tl.load(src + QH * D + kv_hi, mask=q_ok, other=0.0).to(tl.float32)
    if HAS_LK:
        lk1 = tl.load(lk_ptr + t * (KH * D) + kv_lo, mask=q_ok, other=0.0).to(tl.float32)
        k1 += (lk1 * scaling).to(dt).to(tl.float32)
        lk2 = tl.load(lk_ptr + t * (KH * D) + kv_hi, mask=q_ok, other=0.0).to(tl.float32)
        k2 += (lk2 * scaling).to(dt).to(tl.float32)
    if NORM:
        # the same per-head norm for k; every query head of a group computes its kv head's norm
        # again (identical values), the first one of the group keeps the stash
        rs = tl.math.rsqrt((tl.sum(k1 * k1, axis=1) + tl.sum(k2 * k2, axis=1)) / D + eps)
        xh1 = k1 * rs[:, None]
        xh2 = k2 * rs[:, None]
        first = q_ok & ((hq % (QH // KH)) == 0)
        kv_at = t * (KH * D) + kv_lo
        tl.store(xk_ptr + kv_at, xh1.to(dt), mask=first)
        tl.store(xk_ptr + kv_at + D // 2, xh2.to(dt), mask=first)
        kheads = tl.arange(0, QH_POW2)
        tl.store(rk_ptr + t * KH + kheads // (QH // KH), rs,
                 mask=(kheads < QH) & ((kheads % (QH // KH)) == 0))
        k1 = tl.load(kw_ptr + half) * xh1
        k2 = tl.load(kw_ptr + D // 2 + half) * xh2
    tl.store(k_ptr + row_out + lo, (k1 * c1 - k2 * s1).to(dt), mask=q_ok)
    tl.store(k_ptr + row_out + hi, (k2 * c2 + k1 * s2).to(dt), mask=q_ok)

    v1 = tl.load(src + (QH + KH) * D + kv_lo, mask=q_ok, other=0.0).to(tl.float32)
    v2 = tl.load(src + (QH + KH) * D + kv_hi, mask=q_ok, other=0.0).to(tl.float32)
    if HAS_LV:
        lv1 = tl.load(lv_ptr + t * (KH * D) + kv_lo, mask=q_ok, other=0.0).to(tl.float32)
        v1 += (lv1 * scaling).to(dt).to(tl.float32)
        lv2 = tl.load(lv_ptr + t * (KH * D) + kv_hi, mask=q_ok, other=0.0).to(tl.float32)
        v2 += (lv2 * scaling).to(dt).to(tl.float32)
    tl.store(v_ptr + row_out + lo, v1.to(dt), mask=q_ok)
    tl.store(v_ptr + row_out + hi, v2.to(dt), mask=q_ok)


@triton.jit
def kz_qkv_rope_bwd_kernel(dq_ptr, dk_ptr, dv_ptr, cos_ptr, sin_ptr, pos_ptr, g_ptr,
                           glq_ptr, glk_ptr, glv_ptr, scaling,
                           x_ptr, w_ptr, rstd_ptr, y_ptr, n_cols,
                           qw_ptr, kw_ptr, xq_ptr, xk_ptr, rq_ptr, rk_ptr,
                           QH: tl.constexpr, QH_POW2: tl.constexpr, KH: tl.constexpr,
                           G_POW2: tl.constexpr, D: tl.constexpr, LORA_Q: tl.constexpr,
                           LORA_K: tl.constexpr, LORA_V: tl.constexpr,
                           NORM_INPUT: tl.constexpr, BLOCK: tl.constexpr,
                           NORM: tl.constexpr):
    dt = g_ptr.dtype.element_ty  # fp16 or bf16
    t = tl.program_id(0).to(tl.int64)
    if NORM_INPUT:
        # the token's LoRA input again, as norms.kz_rmsnorm_fwd_kernel wrote it in the forward
        # (the same operations on the same values, with the rstd it stored there):
        # w * (x * rstd), written to y_ptr
        cols = tl.arange(0, BLOCK)
        inside = cols < n_cols
        at = t * n_cols + cols
        x = tl.load(x_ptr + at, mask=inside, other=0.0)
        y = tl.load(w_ptr + cols, mask=inside, other=0.0) * (x * tl.load(rstd_ptr + t))
        tl.store(y_ptr + at, y.to(dt), mask=inside)
    p = tl.load(pos_ptr + t).to(tl.int64)
    half = tl.arange(0, D // 2)
    c1 = tl.load(cos_ptr + p * D + half)
    c2 = tl.load(cos_ptr + p * D + D // 2 + half)
    s1 = tl.load(sin_ptr + p * D + half)
    s2 = tl.load(sin_ptr + p * D + D // 2 + half)
    width = (QH + 2 * KH) * D
    dst = g_ptr + t * width
    row_in = t * (QH * D)

    # rotation transposed: d_lo = g_lo*cos + g_hi*sin, d_hi = g_hi*cos - g_lo*sin
    hq = tl.arange(0, QH_POW2)[:, None]
    q_ok = hq < QH
    lo = hq * D + half[None, :]
    hi = lo + D // 2
    g1 = tl.load(dq_ptr + row_in + lo, mask=q_ok, other=0.0).to(tl.float32)
    g2 = tl.load(dq_ptr + row_in + hi, mask=q_ok, other=0.0).to(tl.float32)
    d1 = g1 * c1[None, :] + g2 * s2[None, :]
    d2 = g2 * c2[None, :] - g1 * s1[None, :]
    if NORM:
        # backward of the per-head RMSNorm (frozen weight w):
        # dx = rstd * (w dy - x_hat * mean(w dy x_hat))
        heads = tl.arange(0, QH_POW2)
        xh1 = tl.load(xq_ptr + row_in + lo, mask=q_ok, other=0.0).to(tl.float32)
        xh2 = tl.load(xq_ptr + row_in + hi, mask=q_ok, other=0.0).to(tl.float32)
        rs = tl.load(rq_ptr + t * QH + heads, mask=heads < QH, other=0.0)
        wd1 = tl.load(qw_ptr + half)[None, :] * d1
        wd2 = tl.load(qw_ptr + D // 2 + half)[None, :] * d2
        mean = (tl.sum(wd1 * xh1, axis=1) + tl.sum(wd2 * xh2, axis=1)) / D
        d1 = rs[:, None] * (wd1 - xh1 * mean[:, None])
        d2 = rs[:, None] * (wd2 - xh2 * mean[:, None])
    o1 = d1.to(dt)
    o2 = d2.to(dt)
    tl.store(dst + lo, o1, mask=q_ok)
    tl.store(dst + hi, o2, mask=q_ok)
    if LORA_Q:
        # gradient of the q LoRA delta (before scaling): torch's multiply by scaling in the
        # compute dtype
        tl.store(glq_ptr + row_in + lo, (o1.to(tl.float32) * scaling).to(dt), mask=q_ok)
        tl.store(glq_ptr + row_in + hi, (o2.to(tl.float32) * scaling).to(dt), mask=q_ok)

    # [kv head, group member, D/2]: each kv head's per-query-head gradients, summed in fp32
    G: tl.constexpr = QH // KH
    kh = tl.arange(0, KH)[:, None, None]
    gm = tl.arange(0, G_POW2)[None, :, None]
    in_group = gm < G
    rep_lo = (kh * G + gm) * D + half[None, None, :]
    rep_hi = rep_lo + D // 2
    a1 = tl.sum(tl.load(dk_ptr + row_in + rep_lo, mask=in_group, other=0.0).to(tl.float32), 1)
    a2 = tl.sum(tl.load(dk_ptr + row_in + rep_hi, mask=in_group, other=0.0).to(tl.float32), 1)
    out_lo = tl.arange(0, KH)[:, None] * D + half[None, :]
    out_hi = out_lo + D // 2
    kd1 = a1 * c1[None, :] + a2 * s2[None, :]
    kd2 = a2 * c2[None, :] - a1 * s1[None, :]
    if NORM:
        kheads = tl.arange(0, KH)
        kxh1 = tl.load(xk_ptr + t * (KH * D) + out_lo).to(tl.float32)
        kxh2 = tl.load(xk_ptr + t * (KH * D) + out_hi).to(tl.float32)
        krs = tl.load(rk_ptr + t * KH + kheads)
        kwd1 = tl.load(kw_ptr + half)[None, :] * kd1
        kwd2 = tl.load(kw_ptr + D // 2 + half)[None, :] * kd2
        kmean = (tl.sum(kwd1 * kxh1, axis=1) + tl.sum(kwd2 * kxh2, axis=1)) / D
        kd1 = krs[:, None] * (kwd1 - kxh1 * kmean[:, None])
        kd2 = krs[:, None] * (kwd2 - kxh2 * kmean[:, None])
    ko1 = kd1.to(dt)
    ko2 = kd2.to(dt)
    tl.store(dst + QH * D + out_lo, ko1)
    tl.store(dst + QH * D + out_hi, ko2)
    row_v = t * (KH * D)
    if LORA_K:
        tl.store(glk_ptr + row_v + out_lo, (ko1.to(tl.float32) * scaling).to(dt))
        tl.store(glk_ptr + row_v + out_hi, (ko2.to(tl.float32) * scaling).to(dt))
    b1 = tl.sum(tl.load(dv_ptr + row_in + rep_lo, mask=in_group, other=0.0).to(tl.float32), 1)
    b2 = tl.sum(tl.load(dv_ptr + row_in + rep_hi, mask=in_group, other=0.0).to(tl.float32), 1)
    gv1 = b1.to(dt)
    gv2 = b2.to(dt)
    tl.store(dst + (QH + KH) * D + out_lo, gv1)
    tl.store(dst + (QH + KH) * D + out_hi, gv2)
    if LORA_V:
        tl.store(glv_ptr + row_v + out_lo, (gv1.to(tl.float32) * scaling).to(dt))
        tl.store(glv_ptr + row_v + out_hi, (gv2.to(tl.float32) * scaling).to(dt))


def qkv_rope_fwd(qkv, lq, lk, lv, cos, sin, pos, q_heads, kv_heads, head_dim, scaling=1.0,
                 qk_norm=None):
    """qkv: [T, (QH+2KH)*D] (fp16 or bf16); lq [T, QH*D], lk and lv [T, KH*D] (each or None
    where that projection has no LoRA): the LoRA-B matmul outputs, added times scaling;
    cos/sin fp32 [P, D]; pos int32 [T]. Returns q, k, v [T, QH, D] in qkv's dtype, k and v
    repeated per query head. qk_norm = (q_norm weight, k_norm weight, eps) (fp32 [head_dim], Qwen3):
    q and k get their per-head RMSNorm before the rotary embedding, and the return value gets a
    4th element, the stash the backward needs: (x_hat of q [T, QH*D], x_hat of k [T, KH*D] (compute
    dtype), rstd of q [T, QH], rstd of k [T, KH] (fp32))."""
    rows = qkv.shape[0]
    q = torch.empty(rows, q_heads, head_dim, device=qkv.device, dtype=qkv.dtype)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    if qk_norm is None:
        qw = kw = xq = xk = rq = rk = q
        eps = 0.0
    else:
        qw, kw, eps = qk_norm
        xq = torch.empty(rows, q_heads * head_dim, device=qkv.device, dtype=qkv.dtype)
        xk = torch.empty(rows, kv_heads * head_dim, device=qkv.device, dtype=qkv.dtype)
        rq = torch.empty(rows, q_heads, device=qkv.device, dtype=torch.float32)
        rk = torch.empty(rows, kv_heads, device=qkv.device, dtype=torch.float32)
    kz_qkv_rope_fwd_kernel[(rows,)](
        qkv, qkv if lq is None else lq, qkv if lk is None else lk, qkv if lv is None else lv,
        cos, sin, pos, q, k, v, scaling, qw, kw, xq, xk, rq, rk, eps, QH=q_heads,
        QH_POW2=triton.next_power_of_2(q_heads), KH=kv_heads, D=head_dim, HAS_LQ=lq is not None,
        HAS_LK=lk is not None, HAS_LV=lv is not None, NORM=qk_norm is not None, num_warps=4)
    if qk_norm is None:
        return q, k, v
    return q, k, v, (xq, xk, rq, rk)


def qkv_rope_bwd(dq, dk, dv, cos, sin, pos, q_heads, kv_heads, head_dim, scaling=None,
                 norm=None, lora=(True, False, True), qk_norm=None, qk_saved=None):
    """dq, dk, dv [T, QH, D] (dk, dv per query head) -> gradient of the packed [q | k | v] matmul
    output, [T, (QH + 2 KH) * D] in dq's dtype. With scaling, also the gradients of the LoRA-B
    matmul outputs of the projections flagged in `lora` = (q, k, v) ([T, QH*D], [T, KH*D],
    [T, KH*D]; None for the others): (g, glq, glk, glv). With norm = (x, weight, rstd) (the
    input_layernorm's fp32 input and weight and the rstd rmsnorm_fwd returned), also the LoRA
    input rmsnorm_fwd wrote, bit for bit: (g, glq, glk, glv, y).
    qk_norm = (q_norm weight, k_norm weight) and qk_saved = the 4 tensors qkv_rope_fwd returned
    (Qwen3): the gradients g, glq, glk are then those of the pre-norm projection outputs."""
    rows = dq.shape[0]
    dev = dq.device
    dt = dq.dtype
    g = torch.empty(rows, (q_heads + 2 * kv_heads) * head_dim, device=dev, dtype=dt)
    lora_q, lora_k, lora_v = (bool(f) and scaling is not None for f in lora)
    glq = torch.empty(rows, q_heads * head_dim, device=dev, dtype=dt) if lora_q else None
    glk = torch.empty(rows, kv_heads * head_dim, device=dev, dtype=dt) if lora_k else None
    glv = torch.empty(rows, kv_heads * head_dim, device=dev, dtype=dt) if lora_v else None
    y = None
    if norm is None:
        x = w = rstd = y_ = g
        n_cols = 8
    else:
        x, w, rstd = norm
        n_cols = x.shape[1]
        y = y_ = torch.empty(rows, n_cols, device=dev, dtype=dt)
    if qk_norm is None:
        qw = kw = xq = xk = rq = rk = g
    else:
        qw, kw = qk_norm
        xq, xk, rq, rk = qk_saved
    kz_qkv_rope_bwd_kernel[(rows,)](dq, dk, dv, cos, sin, pos, g,
                                    g if glq is None else glq, g if glk is None else glk,
                                    g if glv is None else glv,
                                    1.0 if scaling is None else scaling,
                                    x, w, rstd, y_, n_cols,
                                    qw, kw, xq, xk, rq, rk,
                                    QH=q_heads, QH_POW2=triton.next_power_of_2(q_heads),
                                    KH=kv_heads, G_POW2=triton.next_power_of_2(q_heads // kv_heads),
                                    D=head_dim, LORA_Q=lora_q, LORA_K=lora_k, LORA_V=lora_v,
                                    NORM_INPUT=norm is not None,
                                    BLOCK=triton.next_power_of_2(n_cols),
                                    NORM=qk_norm is not None, num_warps=4)
    if norm is not None:
        return g, glq, glk, glv, y
    return g, glq, glk, glv
