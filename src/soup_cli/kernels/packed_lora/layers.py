# SPDX-License-Identifier: Apache-2.0
#
# Origin: Soup Kernel Challenge (Decentrathon 6.0), the submission of team Luck
# (BAITC-Hacks/hack-e79529fc-luck@b3f1225f); the int8 gate/up packs follow ideas seen in the
# solutions of teams Null Pointers, 21, DarkDaur and OrbiTeam.
# Modified for Soup: LoRA dropout (masks, packed masks, the dropout inputs) is not ported (it is
# 0); the int8 switch and the rounding of the quantisers come from the step (``Step.int8``,
# ``Step.rounding``) instead of module switches and environment variables; the bench
# instrumentation and the gate-first packing option are removed.
"""Autograd functions of a Qwen2 decoder layer over packed (unpadded) tokens, and of the head.

Each one follows the base run's numerics (fp32 residual stream and norms, fp16 / bf16 matmuls with
4-bit weights dequantized to the compute dtype, LoRA through compute-dtype matmuls, fp32 rotary),
with two deliberate differences: the backward's three MLP input-gradient GEMMs may run in int8
(``triton/int8.py``, lossy, frozen weights only) and the head sums the vocabulary slab by slab
(``triton/head.py``, fp32 summation order). Each decides itself what to keep for backward: the
base keeps fp32 copies of tensors that only ever held fp16 values, the dequantized lm_head weight,
and the full logits several times over.

Frozen 4-bit weights are decoded again in backward, as bitsandbytes' own MatMul4Bit does (to the
compute dtype, or straight to int8 for the MLP weights, ``nf4.kz_nf4_int8t_kernel``).

LoRA: any subset of the seven projections (q, k, v, o, gate, up, down) may carry an adapter, each
with its own rank. Where a LoRA output joins the base output is the only place its forward touches
the layer: q, k, v inside the rotary kernel (``triton/rope.py``), o / gate / up / down in the
epilogue of the base matmul (``out.addmm_(mid, B.t(), alpha=scaling)``: the product in fp32, one
rounding); the backward of an adapter is :func:`_lora_bwd`. LoRA dropout must be 0. The gradients
of the LoRA parameters of gate / up / down are computed from the compute-dtype d_gate / d_up
(written next to their int8 copies) and, for down, from silu(gate) * up rebuilt from the int8
packs of gate and up (``triton/q8.py``: the one place where a LoRA gradient sees int8 rounding,
~0.4 % of a group's largest value, averaged over the tokens of the batch).
"""

import torch
import triton

from .triton.head import head_forward, head_forward_streamed
from .triton.int8 import dequant_sum, quant_rows
from .triton.nf4 import launch_int8t
from .triton.norms import rmsnorm_bwd, rmsnorm_fwd
from .triton.q8 import (
    act_from_q8,
    dequant_q8,
    fused_fwd_usable,
    swiglu_bwd_q8,
    swiglu_fwd_q8,
    swiglu_lora_fwd_q8,
)
from .triton.q8 import usable_width as q8_usable
from .triton.rope import qkv_rope_bwd, qkv_rope_fwd
from .triton.swiglu import BLOCK as SWIGLU_BLOCK
from .triton.swiglu import kz_swiglu_bwd_kernel, kz_swiglu_fwd_kernel

# num_splits_key of the attention backward: None lets PyTorch's heuristic pick; 1 never splits
# (bit-reproducible); n > 1 splits each row's keys over n thread blocks, whose dq parts are summed
# with atomics (not bit-reproducible, as PyTorch's deterministic mode says)
ATTN_BWD_SPLITS = None

# the seven projections, in the order their adapters' parameters are passed to DecoderLayer
PROJECTIONS = ("q", "k", "v", "o", "gate", "up", "down")


class Adapter:
    """One LoRA adapter of a projection: the fp32 parameters, the compute-dtype copies the matmuls
    use (refreshed once per step, ``model.forward``), and the scaling."""

    def __init__(self, a32, b32, scaling):
        self.a32, self.b32, self.scaling = a32, b32, scaling
        self.a16 = self.b16 = None


def _int8_supported(layer, tokens):
    return tokens > 16 and q8_usable(layer.pack.shapes[4][0])


def uses_int8(layer, tokens, switch):
    """Whether this layer's backward takes the int8 path (gate / up stashed as q8 packs) for a
    batch of ``tokens`` tokens: ``torch._int_mm`` wants more than 16 rows, the width must pack,
    and the run's int8 switch (``policy.Int8Switch``) must allow it."""
    return _int8_supported(layer, tokens) and switch.enabled()


uses_int8.supported = _int8_supported


def _lora_bwd(ad, g, z, mid):
    """Adapter backward. g: gradient of the projection's output [T, N] (compute dtype), z: its
    input [T, K], mid = z @ A^T [T, r]. Returns (dA [r, K], dB [N, r], g_mid [T, r]); the input
    gradient the adapter adds is g_mid @ A. gs = g * scaling is torch's multiply in the compute
    dtype (the one the base run's autograd does for `lora_B(...) * scaling`)."""
    return _lora_bwd_scaled(ad, g * ad.scaling, z, mid)


def _lora_bwd_scaled(ad, gs, z, mid):
    """_lora_bwd for a gradient that already is gs = g * scaling (the rotary kernel does that)."""
    g_mid = gs @ ad.b16
    return g_mid.t() @ z, gs.t() @ mid, g_mid


def _project_qkv(r, layer, step, w=None):
    """input_layernorm -> q/k/v (one fused matmul with the biases) -> LoRA on whichever of q, k, v
    has one -> rotary. Deterministic. w: the decoded q/k/v weight, if the caller has it. Returns
    (q, k, v, mids, rstd, qk_norm_stash); mids[name] = LoRA input @ A^T."""
    dt = layer.dt
    y, rstd = rmsnorm_fwd(r, layer.ln1, layer.eps, dtype=dt)
    if w is None:
        w = layer.pack.matrix(0, 3)
    if layer.qkv_bias is not None:
        qkv = torch.addmm(layer.qkv_bias, y, w.t())
    else:
        qkv = y @ w.t()
    del w
    mids, deltas = {}, {}
    for name in ("q", "k", "v"):
        ad = layer.lora.get(name)
        if ad is not None:
            mids[name] = y @ ad.a16.t()
            deltas[name] = mids[name] @ ad.b16.t()
    del y
    # the rotary kernel multiplies the LoRA-B outputs by scaling (as torch's multiply)
    out = qkv_rope_fwd(qkv, deltas.get("q"), deltas.get("k"), deltas.get("v"), step.cos,
                       step.sin, step.pos, layer.q_heads, layer.kv_heads, layer.head_dim,
                       layer.scaling_qkv, layer.qk_norm)
    q, k, v = out[:3]
    qkn = out[3] if layer.qk_norm is not None else None  # Qwen3: (x_hat q, x_hat k, rstd q, rstd k)
    return q, k, v, mids, rstd, qkn


def _attention(q, k, v, step):
    """Causal attention of every row at once: PyTorch's memory-efficient kernel (the one the
    base run's SDPA picks on a T4, once per layer over the padded batch with an explicit mask)
    through its varlen interface, row boundaries in step.cu, so no query sees another row and
    no pad is computed. q, k, v [T, heads, head_dim] (k, v repeated per query head by the rotary
    kernel). The op is private to torch: ``support.attention_signature_problems`` checks its
    signature at start-up."""
    out, lse, seed, offset, _, _ = torch.ops.aten._efficient_attention_forward(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), None, step.cu, step.cu, step.max_len,
        step.max_len, 0.0, 1, True, scale=step.attn_scale)
    return out[0], lse, seed, offset


def _column_scales(pack):
    """Per weight 4, 5, 6 (gate, up, down): (127 / max |column|, max |column| / 127), from the
    decoded weight, once per run."""
    scales = {}
    for i in (4, 5, 6):
        (w,) = pack.dequant(i, i + 1)
        top = w.abs().amax(0).float().clamp_min(1e-30)
        scales[i] = (127.0 / top, top / 127.0)
        del w
    return scales


def _act_grad_int8(layer, g_next, rounding=None):
    """g_next @ W_down through an int8 GEMM: (int32 [T, intermediate], g_next's row scales),
    for swiglu_bwd_q8 with the MLP's forward gate and up projections."""
    pack = layer.pack
    scales = getattr(layer, "int8_scales", None)
    if scales is None:
        scales = layer.int8_scales = _column_scales(pack)
    g8, sg = quant_rows(g_next.contiguous(), rounding)
    n_down, k_down = pack.shapes[6]  # W_down [hidden, intermediate]
    w8 = launch_int8t(pack, 6, scales[6][0],
                      torch.empty(k_down, n_down, device=g_next.device, dtype=torch.int8),
                      rounding)
    return torch._int_mm(g8, w8.t()), sg  # g_act, int32 [T, intermediate]


def _mlp_input_grad_int8(layer, dg8, sdg, du8, sdu, rounding=None):
    """d loss / d (post-attention norm output), fp32, through int8 GEMMs, from swiglu_bwd_q8's
    int8 gate and up gradients."""
    pack, scales = layer.pack, layer.int8_scales
    n_gate, k_gate = pack.shapes[4]  # W_gate / W_up [intermediate, hidden]
    w8 = torch.empty(k_gate, n_gate, device=dg8.device, dtype=torch.int8)
    a_gate = torch._int_mm(dg8, launch_int8t(pack, 4, scales[4][0], w8, rounding).t())
    a_up = torch._int_mm(du8, launch_int8t(pack, 5, scales[5][0], w8, rounding).t())
    del w8
    return dequant_sum(a_gate, sdg, scales[4][1], a_up, sdu, scales[5][1])


def _fused_forward(layer, lora):
    """Whether the forward may use ``swiglu_lora_fwd_q8`` for this layer: the shape, and one rank
    for the MLP adapters that exist (at least one of them)."""
    ranks = {lora[n].b16.shape[1] for n in ("gate", "up") if n in lora}
    ranks |= {lora["down"].a16.shape[0]} if "down" in lora else set()
    return len(ranks) == 1 and fused_fwd_usable(layer.pack.shapes[4][0], next(iter(ranks)))


def prefetch_saved(stash, layer, index, device):
    """Start copying layer ``index``'s offloaded tensors back to the GPU, gate / up first (a no-op
    for a layer that stayed on the GPU)."""
    for name in ("gate_up", "x", "rest"):
        group = layer.groups.get(name)
        if group is not None:
            stash.prefetch(index, device, *group)


class DecoderLayer(torch.autograd.Function):
    """One whole decoder layer, fp32 residual stream in and out: input_layernorm -> q/k/v ->
    LoRA -> rotary -> causal attention -> o_proj -> residual add -> post_attention_layernorm ->
    gate, up -> silu(gate) * up -> down -> residual add.

    Attention runs over all rows at once (no padding, no cross-row attention) through PyTorch's
    memory-efficient attention kernel. The rotary kernel writes k and v already repeated for GQA
    in the compute dtype, and its backward sums the repeated gradients in fp32, as the base run's
    fp32 repeat_kv does. The activation's backward is fused with the int8 quantization
    (``q8.kz_swiglu_bwd_q8_kernel``, reading gate and up from their int8 packs; the plain
    elementwise kernel where the layer does not stash int8).

    One autograd node per layer rather than one for attention and one for the MLP: fewer Python
    calls a step, and the q/k/v and o weights decode in one launch. The base keeps three fp32
    [tokens, intermediate] tensors per layer, the largest share of its activation memory. Here the
    fp32 residual, q, k, v, the attention output, (x, with a gate / up LoRA), gate and up go to
    the layer's stash (``offload.py``: pinned host memory, or kept on the GPU where it has room)
    and come back in two or three copies; backward recomputes only the mid-layer residual (one
    matmul). All of it is deterministic except the stochastic rounding of the int8 gradients.
    Kept on the GPU either way: two rstd vectors and the small LoRA intermediates."""

    @staticmethod
    def forward(ctx, r, layer, step, stash, *params):
        # the compute-dtype copies of the LoRA weights were refreshed at the start of the step
        # (model.forward); they keep these values until the optimizer step, after this backward
        lora = layer.lora
        # q/k/v and o decoded in one launch (o is small next to the MLP weights decoded below)
        w_qkv, w_o = layer.pack.split(0, 3, 4)
        q, k, v, mids, rstd1, qkn = _project_qkv(r, layer, step, w=w_qkv)
        del w_qkv
        out, lse, seed, offset = _attention(q, k, v, step)
        o = out.view(out.shape[0], -1)
        # k and v once per kv head (the other query heads hold copies): 1/group of the bytes
        group = step.q_heads // step.kv_heads
        stash.save(layer.index, r, q, k[:, ::group], v[:, ::group], o,
                   *((qkn[0], qkn[1]) if qkn is not None else ()))
        del q, k, v
        o_out = o @ w_o.t()
        ad = lora.get("o")
        if ad is not None:
            mids["o"] = o @ ad.a16.t()
            o_out.addmm_(mids["o"], ad.b16.t(), alpha=ad.scaling)
        h = r + o_out
        del w_o, o_out
        x, rstd2 = rmsnorm_fwd(h, layer.ln2, layer.eps, dtype=layer.dt)
        if layer.save_x:  # the input of the gate / up LoRA: back in the backward with gate and up
            stash.save(layer.index, x)
        # One MLP weight at a time: the decoded weights are the forward's largest transients
        # (each one freed before the next is decoded, so the next can take its block). Gate and up
        # are copied out as soon as each exists, and the blocks of copies that have normally
        # finished by then are given back (offload.Stash.release): the previous layer's before the
        # gate decode, this layer's r, q, k, v, o once up exists.
        # With the int8 backward, gate and up are stashed as int8 packs (q8.py; the same backward
        # decision as DecoderLayer.backward's), packed by the SwiGLU kernel.
        q8 = uses_int8(layer, x.shape[0], step.int8)
        fused = q8 and _fused_forward(layer, lora)
        stash.release(keep=2 if layer.save_x else 1)  # the previous layer's gate and up
        (w,) = layer.pack.dequant(4, 5)
        gate = x @ w.t()
        del w
        ad = lora.get("gate")
        if ad is not None:
            mids["gate"] = x @ ad.a16.t()
            if not fused:
                gate.addmm_(mids["gate"], ad.b16.t(), alpha=ad.scaling)
        if not q8:
            stash.save(layer.index, gate)
        (w,) = layer.pack.dequant(5, 6)
        up = x @ w.t()
        del w
        ad = lora.get("up")
        if ad is not None:
            mids["up"] = x @ ad.a16.t()
            if not fused:
                up.addmm_(mids["up"], ad.b16.t(), alpha=ad.scaling)
        del x
        if fused:
            ad_g, ad_u, ad_d = lora.get("gate"), lora.get("up"), lora.get("down")
            act, packed_gate, packed_up, hd = swiglu_lora_fwd_q8(
                gate, up, mids.get("gate"), ad_g.b16 if ad_g else None,
                ad_g.scaling if ad_g else 1.0, mids.get("up"), ad_u.b16 if ad_u else None,
                ad_u.scaling if ad_u else 1.0, ad_d.a16 if ad_d else None)
            if hd is not None:
                mids["down"] = hd
            stash.save(layer.index, packed_gate, packed_up)
            del packed_gate, packed_up
            stash.release(keep=1)  # this layer's r, q, k, v, o (and x)
        elif q8:
            act, packed_gate, packed_up = swiglu_fwd_q8(gate, up)
            stash.save(layer.index, packed_gate, packed_up)
            del packed_gate, packed_up
            stash.release(keep=1)  # this layer's r, q, k, v, o (and x)
        else:
            stash.save(layer.index, up)
            stash.release(keep=2)  # this layer's r, q, k, v, o (and x)
            act = torch.empty_like(gate)
            n = gate.numel()
            kz_swiglu_fwd_kernel[(triton.cdiv(n, SWIGLU_BLOCK),)](gate, up, act, n,
                                                                  BLOCK=SWIGLU_BLOCK)
        del gate, up
        (w_down,) = layer.pack.dequant(6, 7)
        down = act @ w_down.t()
        ad = lora.get("down")
        if ad is not None:
            if "down" not in mids:  # (the fused forward made it already)
                mids["down"] = act @ ad.a16.t()
            down.addmm_(mids["down"], ad.b16.t(), alpha=ad.scaling)
        h += down
        del act, down, w_down
        ctx.layer, ctx.step, ctx.stash, ctx.q8 = layer, step, stash, q8
        ctx.mid_names = tuple(mids)
        ctx.save_for_backward(rstd1, rstd2, *((qkn[2], qkn[3]) if qkn is not None else ()),
                              *mids.values(), lse, seed, offset)
        return h

    @staticmethod
    def backward(ctx, g_next):
        layer, step, stash, q8_stash = ctx.layer, ctx.step, ctx.stash, ctx.q8
        rounding = step.rounding
        # the int8 GEMMs and the int8 SwiGLU backward, or (int8_backward: stash) the compute-dtype
        # path over gate and up dequantized from their stashed packs
        int8 = q8_stash and step.int8.gemm_enabled()
        lora, dt = layer.lora, layer.dt
        rstd1, rstd2, *saved = ctx.saved_tensors
        rq = rk = None
        if layer.qk_norm is not None:
            rq, rk, *saved = saved
        mids = dict(zip(ctx.mid_names, saved[:len(ctx.mid_names)]))
        lse, seed, offset = saved[len(ctx.mid_names):]
        index, device = layer.index, g_next.device
        groups = layer.groups
        grads = {}  # adapter name -> (dA, dB)

        def load(name):
            return stash.load(index, device, *groups[name])

        # Offloaded, with the int8 MLP backward: gate and up come back from the host right before
        # the SwiGLU backward (their only reader), the rest after the MLP's GEMMs, and the next
        # backward's copies start once this layer's gate and up are freed. So the next layer's
        # tensors are not on the GPU next to the int8 temporaries (the backward's peak), and
        # each copy still has one layer's backward of time to arrive.
        if not int8:
            gate, up = load("gate_up")
            if q8_stash:
                rows, n = g_next.shape[0], layer.pack.shapes[4][0]
                gate, up = dequant_q8(gate, rows, n, dt), dequant_q8(up, rows, n, dt)
            x = load("x")[0] if layer.save_x else None
            r, q, k, v, o, *qk_x = load("rest")
            prefetch_saved(stash, layer, index - 1, device)  # the next backward's
            out = o.view(q.shape)

        # MLP part. The down projection's LoRA term of the activation's gradient first
        g_down = lora.get("down")
        extra = g_mid_down = None
        if g_down is not None:
            gs = g_next.to(dt) * g_down.scaling
            grads["down"] = [None, gs.t() @ mids["down"]]
            g_mid_down = gs @ g_down.b16
            del gs
            extra = g_mid_down @ g_down.a16  # [T, intermediate]
        fp_dtype = dt if ("gate" in lora or "up" in lora) else None
        d_gate_fp = d_up_fp = None
        if int8:
            acc, sg = _act_grad_int8(layer, g_next, rounding)
            gate, up = load("gate_up")  # int8 packs
            x = load("x")[0] if layer.save_x else None
            # the compute-dtype d_gate / d_up come back already times their LoRA scalings
            fp_scales = (lora["gate"].scaling if "gate" in lora else 1.0,
                         lora["up"].scaling if "up" in lora else 1.0) if fp_dtype else None
            res = swiglu_bwd_q8(acc, sg, layer.int8_scales[6][1], gate, up, extra, fp_dtype,
                                fp_scales, rounding)
            dg8, sdg, du8, sdu, *fp = res
            if fp:
                d_gate_fp, d_up_fp = fp
            if g_down is not None:
                act = act_from_q8(gate, up, acc.shape[0], acc.shape[1], dt)
                grads["down"][0] = g_mid_down.t() @ act
                del act
            del acc, gate, up, extra
            prefetch_saved(stash, layer, index - 1, device)  # the next backward's
            gx = _mlp_input_grad_int8(layer, dg8, sdg, du8, sdu, rounding)
            del dg8, du8
        else:
            # gate and up's gradients written over gate and up (they are not needed after the
            # SwiGLU backward kernel)
            (w_down,) = layer.pack.dequant(6, 7)
            g_act = g_next.to(dt) @ w_down
            del w_down
            if extra is not None:
                g_act += extra
                act = torch.empty_like(gate)
                n = gate.numel()
                kz_swiglu_fwd_kernel[(triton.cdiv(n, SWIGLU_BLOCK),)](gate, up, act, n,
                                                                      BLOCK=SWIGLU_BLOCK)
                grads["down"][0] = g_mid_down.t() @ act
                del act, extra
            n = gate.numel()
            kz_swiglu_bwd_kernel[(triton.cdiv(n, SWIGLU_BLOCK),)](g_act, gate, up, gate, up,
                                                                  n, BLOCK=SWIGLU_BLOCK)
            del g_act
            d_gate_fp, d_up_fp = gate, up
            (w_gate,) = layer.pack.dequant(4, 5)
            gx = torch.mm(gate, w_gate, out_dtype=torch.float32)
            del w_gate
            (w_up,) = layer.pack.dequant(5, 6)
            gx += torch.mm(up, w_up, out_dtype=torch.float32)
            del w_up
        for name, d_fp in (("gate", d_gate_fp), ("up", d_up_fp)):
            ad = lora.get(name)
            if ad is not None:
                backward = _lora_bwd_scaled if int8 else _lora_bwd  # (int8: d_fp is g * scaling)
                d_a, d_b, g_mid = backward(ad, d_fp, x, mids[name])
                grads[name] = (d_a, d_b)
                torch.addmm(gx, g_mid, ad.a16, out_dtype=torch.float32, out=gx)
        del d_gate_fp, d_up_fp, x
        if int8:
            r, q, k, v, o, *qk_x = load("rest")
            out = o.view(q.shape)
        (w_o,) = layer.pack.dequant(3, 4)
        h_out = o @ w_o.t()
        ad = lora.get("o")
        if ad is not None:
            h_out.addmm_(mids["o"], ad.b16.t(), alpha=ad.scaling)
        h = r + h_out
        del h_out
        g_h = rmsnorm_bwd(gx, h, layer.ln2, rstd2)
        del gx, h
        g_h += g_next
        g_h16 = g_h.to(dt)
        g_o = g_h16 @ w_o
        del w_o
        if ad is not None:
            d_a, d_b, g_mid = _lora_bwd(ad, g_h16, o, mids["o"])
            grads["o"] = (d_a, d_b)
            g_o.addmm_(g_mid, ad.a16)
        del g_h16

        # attention part
        group = step.q_heads // step.kv_heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        gq, gk, gv, _ = torch.ops.aten._efficient_attention_backward(
            g_o.view(out.shape).unsqueeze(0), q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
            None, out.unsqueeze(0), step.cu, step.cu, step.max_len, step.max_len, lse, 0.0, seed,
            offset, 1, False, scale=step.attn_scale, num_splits_key=ATTN_BWD_SPLITS)
        del q, k, v, g_o
        # g: gradient of the q/k/v matmul output; glq, glk, glv: its parts times scaling, the
        # gradients of the LoRA-B outputs; the same kernel also writes the LoRA input again (the
        # norm output y: bit-identical to the forward's)
        has_qkv = tuple(name in lora for name in "qkv")
        norm = (r, layer.ln1, rstd1) if any(has_qkv) else None
        res = qkv_rope_bwd(gq[0], gk[0], gv[0], step.cos, step.sin, step.pos, layer.q_heads,
                           layer.kv_heads, layer.head_dim, layer.scaling_qkv, norm=norm,
                           lora=has_qkv,
                           qk_norm=layer.qk_norm[:2] if layer.qk_norm is not None else None,
                           qk_saved=((qk_x[0], qk_x[1], rq, rk)
                                     if layer.qk_norm is not None else None))
        del gq, gk, gv
        g, glq, glk, glv = res[:4]
        z = res[4] if norm is not None else None
        g_mids = {}
        for name, gl in (("q", glq), ("k", glk), ("v", glv)):
            if gl is not None:
                d_a, d_b, g_mids[name] = _lora_bwd_scaled(lora[name], gl, z, mids[name])
                grads[name] = (d_a, d_b)
        del glq, glk, glv, z, res
        dr = None
        if ctx.needs_input_grad[0]:
            w = layer.pack.matrix(0, 3)
            gy = torch.mm(g, w, out_dtype=torch.float32)
            del w, g
            for name, g_mid in g_mids.items():
                torch.addmm(gy, g_mid, lora[name].a16, out_dtype=torch.float32, out=gy)
            dr = rmsnorm_bwd(gy, r, layer.ln1, rstd1)
            # the residual's two gradients, summed as autograd sums them (one fp32 add)
            dr += g_h
        # fp16 / bf16 gradients: autograd casts them to the fp32 LoRA weights' dtype (the same
        # cast as .float() here)
        out_grads = []
        for name in layer.names:
            out_grads.extend(grads[name])
        return (dr, None, None, None, *out_grads)


class NormHead(torch.autograd.Function):
    """Final norm + lm_head + loss. The head's gradient is computed in the forward (see
    ``triton/head.py``), so backward is one multiply and the norm backward."""

    @staticmethod
    def forward(ctx, r, norm_weight, eps, head_weight, targets, n_items, n_labelled, stats,
                on_backward=None, target_rows=None, dtype=torch.float16):
        x, rstd = rmsnorm_fwd(r, norm_weight, eps, dtype=dtype)
        if target_rows is None:
            loss, d_x, entropy, token_loss, hit = head_forward(x, head_weight, targets, n_items,
                                                               n_labelled)
        else:  # the weight in pinned host memory (hostembed.py), walked in slabs
            loss, d_x, entropy, token_loss, hit = head_forward_streamed(
                x, head_weight, targets, n_items, n_labelled, target_rows)
        stats.update(entropy=entropy, token_loss=token_loss, hit=hit)
        ctx.norm_weight, ctx.on_backward = norm_weight, on_backward
        ctx.save_for_backward(r, rstd, d_x)
        return loss

    @staticmethod
    def backward(ctx, gloss):
        if ctx.on_backward is not None:
            ctx.on_backward()
        r, rstd, d_x = ctx.saved_tensors
        d_x.mul_(gloss)
        return (rmsnorm_bwd(d_x, r, ctx.norm_weight, rstd), None, None, None, None, None, None,
                None, None, None, None)
