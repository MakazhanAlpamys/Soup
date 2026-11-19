# SPDX-License-Identifier: Apache-2.0
#
# Origin: Soup Kernel Challenge (Decentrathon 6.0), the submission of team Luck
# (BAITC-Hacks/hack-e79529fc-luck@b3f1225f); the int8 gate/up packs follow ideas seen in the
# solutions of teams Null Pointers, 21, DarkDaur and OrbiTeam.
# Modified for Soup: packs live on the GPU only (the host-streaming mode of the original is
# not ported) and the int8 decode takes the run's rounding as an argument.
# ruff: noqa: N803, N806  (Triton's constexpr arguments are upper case by convention)
"""NF4 + double-quant dequantization of bitsandbytes 4-bit weights to fp16 or bf16 (the compute
dtype of the 4-bit layers: fp16 on a T4, bf16 on Ampere and newer; the output dtype is the
`dtype` of the pack's `dequant` calls, rounded from the same fp32 product).

bitsandbytes does it in three launches per weight: dequantize the 8-bit block scales to fp32, add
the offset (a separate torch op), then decode the 4-bit codes. Every training matmul in this run
dequantizes its full weight this way, forward and again in backward: 389 dequantizations (1,167
launches) a step.

Here a layer's seven 4-bit weights (q, k, v, o, gate, up, down) are moved once into contiguous
buffers (the model's own parameters become views of them, so nothing is duplicated), and one
launch decodes any run of consecutive weights: q..o, gate, up and down in forward; o, down,
gate, up and q+k+v in backward, one weight at a time in the order of use so that only one MLP
weight is decoded at a time (peak memory): 9 launches a layer instead of 14 (251 a step, layer
0 needs no q/k/v input gradient). With the int8 backward the backward's down, gate and up are
decoded straight to int8 instead (kz_nf4_int8t_kernel below). Each block's scale is computed
with the same two roundings as bitsandbytes (code2[q] * absmax2, then + offset, no fused
multiply-add), so the fp16 weights are bit-identical to bitsandbytes'.

Layout (bitsandbytes 0.50): weight [N, K] row-major, two codes per byte, the even element in the
high nibble; one scale per 64 weights, stored as an 8-bit code into a 256-entry table times an
fp32 absmax shared by 256 blocks, plus the weight's offset.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

BYTES_PER_PROGRAM = 1024
WARPS = 4
_TL_DTYPE = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}


@triton.jit
def kz_nf4_dequant_kernel(packed_ptr, qscale_ptr, scale2_ptr, offset_ptr, code2_ptr, nf4_ptr,
                          out_ptr, first_byte, n_bytes, BYTES: tl.constexpr, GROUP: tl.constexpr):
    # BYTES consecutive bytes (2 * BYTES weights) from byte first_byte on; each byte looks up the
    # scale of its 64-weight block (cached loads: 32 bytes share one)
    local = tl.program_id(0).to(tl.int64) * BYTES + tl.arange(0, BYTES)
    ok = local < n_bytes
    at = first_byte + local
    byte = tl.load(packed_ptr + at, mask=ok, other=0).to(tl.int32)
    block = at // 32
    q = tl.load(qscale_ptr + block, mask=ok, other=0).to(tl.int32)
    group = block // GROUP
    scale = libdevice.add_rn(
        libdevice.mul_rn(tl.load(code2_ptr + q, mask=ok, other=0.0),
                         tl.load(scale2_ptr + group, mask=ok, other=0.0)),
        tl.load(offset_ptr + group, mask=ok, other=0.0))
    high = tl.load(nf4_ptr + (byte >> 4)) * scale
    low = tl.load(nf4_ptr + (byte & 15)) * scale
    weights = tl.join(high, low).reshape(2 * BYTES).to(out_ptr.dtype.element_ty)
    dst = 2 * tl.program_id(0).to(tl.int64) * BYTES + tl.arange(0, 2 * BYTES)
    tl.store(out_ptr + dst, weights, mask=dst < 2 * n_bytes)


def _same_tables(states):
    """All weights use the same NF4 and second-level code tables: one comparison (and one host
    sync) for the whole layer instead of one per weight and table."""
    codes = torch.stack([s.code for s in states])
    codes2 = torch.stack([s.state2.code for s in states])
    return bool(((codes == codes[0]).all() & (codes2 == codes2[0]).all()).item())


def _supported(states):
    first = states[0]
    for i, s in enumerate(states):
        ok = (s.quant_type == "nf4" and s.nested and s.dtype in (torch.float16, torch.bfloat16)
              and s.dtype == first.dtype
              and s.blocksize == 64 and s.state2.blocksize == 256 and len(s.shape) == 2
              and s.shape.numel() % 64 == 0 and s.absmax.dtype == torch.uint8
              and s.code.shape == first.code.shape
              and s.state2.code.shape == first.state2.code.shape)
        # second-level groups of 256 blocks must not straddle two weights
        if not ok or (i + 1 < len(states) and s.absmax.numel() % 256):
            return False
    return _same_tables(states)


class Pack:
    """Consecutive 4-bit weights of one layer in shared contiguous buffers on the GPU. The
    model's own parameters become views of the buffers, so nothing is duplicated."""

    def __init__(self, weights):
        states = [w.quant_state for w in weights]
        self.dtype = states[0].dtype  # the decoded weights' dtype: the 4-bit layers' compute dtype
        dev = weights[0].device
        n_bytes = [s.shape.numel() // 2 for s in states]
        n_blocks = [s.absmax.numel() for s in states]
        n_groups = [s.state2.absmax.numel() for s in states]
        self.codes_bytes, self.scales_bytes = sum(n_bytes), sum(n_blocks)
        self.scales_at = -(-self.codes_bytes // 256) * 256
        self.slot_bytes = self.scales_at + self.scales_bytes
        self.packed = torch.empty(sum(n_bytes), device=dev, dtype=torch.uint8)
        self.qscale = torch.empty(sum(n_blocks), device=dev, dtype=torch.uint8)
        packed, qscale = self.packed, self.qscale
        self.scale2 = torch.empty(sum(n_groups), device=dev, dtype=torch.float32)
        self.offset = torch.empty(sum(n_groups), device=dev, dtype=torch.float32)
        self.code2, self.nf4 = states[0].state2.code, states[0].code
        self.byte_start, self.shapes, self.block_start = [], [], []
        self.weights = list(weights)
        at_byte = at_block = at_group = 0
        for w, s, nb, nk, ng in zip(weights, states, n_bytes, n_blocks, n_groups):
            # move the weight's storage into the shared buffers; the parameter keeps its object
            part = packed[at_byte:at_byte + nb]
            part.copy_(w.data.reshape(-1))
            scales = qscale[at_block:at_block + nk]
            scales.copy_(s.absmax)
            w.data, s.absmax = part.view(w.data.shape), scales
            self.scale2[at_group:at_group + ng].copy_(s.state2.absmax)
            s.state2.absmax = self.scale2[at_group:at_group + ng]
            self.offset[at_group:at_group + ng] = s.offset
            self.byte_start.append(at_byte)
            self.block_start.append(at_block)
            self.shapes.append(tuple(s.shape))
            at_byte, at_block, at_group = at_byte + nb, at_block + nk, at_group + ng
        self.byte_start.append(at_byte)
        self.block_start.append(at_block)

    def slot_views(self, buffer):
        """(codes, block scales) of this layer inside a slot-sized uint8 buffer."""
        return buffer[:self.codes_bytes], buffer[self.scales_at:self.slot_bytes]

    @staticmethod
    def supported(weights):
        states = [getattr(w, "quant_state", None) for w in weights]
        return (all(s is not None for s in states) and all(w.dtype == torch.uint8 for w in weights)
                and all(w.data.is_contiguous() and s.absmax.is_contiguous()
                        for w, s in zip(weights, states))
                and _supported(states))

    def _decode(self, first, stop):
        start, end = self.byte_start[first], self.byte_start[stop]
        out = torch.empty(2 * (end - start), device=self.scale2.device, dtype=self.dtype)
        launch(self, start, end - start, out)
        return out

    def dequant(self, first, stop):
        """Views [N_i, K_i] (compute dtype) of weights first..stop-1, decoded in one launch."""
        out = self._decode(first, stop)
        views, at = [], 0
        for n, k in self.shapes[first:stop]:
            views.append(out[at:at + n * k].view(n, k))
            at += n * k
        return views

    def split(self, first, mid, stop):
        """One launch: weights first..mid-1 (same in_features) as one [sum N_i, K] matrix, then
        views [N_i, K_i] of weights mid..stop-1."""
        out = self._decode(first, stop)
        at = 2 * (self.byte_start[mid] - self.byte_start[first])
        views = [out[:at].view(-1, self.shapes[first][1])]
        for n, k in self.shapes[mid:stop]:
            views.append(out[at:at + n * k].view(n, k))
            at += n * k
        return views

    def matrix(self, first, stop):
        """Weights first..stop-1 (same in_features) decoded as one [sum N_i, K] matrix."""
        return self._decode(first, stop).view(-1, self.shapes[first][1])


def launch(pack, first_byte, n_bytes, out):
    kz_nf4_dequant_kernel[(triton.cdiv(n_bytes, BYTES_PER_PROGRAM),)](
        pack.packed, pack.qscale, pack.scale2, pack.offset, pack.code2, pack.nf4, out,
        first_byte, n_bytes, BYTES=BYTES_PER_PROGRAM, GROUP=256, num_warps=WARPS)


# --- int8 for the backward's input-gradient GEMMs (layers.py, INT8_BACKWARD) -----------------
# d_x = g @ W contracts over W's rows, and int8 tensor-core GEMMs want both operands contiguous
# along the contracted dimension, so W is decoded straight to int8 *transposed* ([K_in, N_out]),
# each column of W (row of the output) scaled by 127 / its max |w| (computed once per run).
@triton.jit(do_not_specialize=["seed"])
def kz_nf4_int8t_kernel(packed_ptr, qscale_ptr, scale2_ptr, offset_ptr, code2_ptr, nf4_ptr,
                        inv_ptr, out_ptr, first_byte, n_rows, n_cols, seed,
                        BR: tl.constexpr, BC: tl.constexpr, GROUP: tl.constexpr,
                        ROUND: tl.constexpr, SR: tl.constexpr = False):
    # one [BR, BC] tile of W (rows of n_cols weights, two per byte, high nibble first), written
    # as the [BC, BR] tile of W^T, rounded to int8 with the column's 127 / max |w|
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    hb = tl.program_id(1) * (BC // 2) + tl.arange(0, BC // 2)  # byte column within a row
    ok = (r[:, None] < n_rows) & (hb[None, :] < n_cols // 2)
    at = first_byte + r[:, None].to(tl.int64) * (n_cols // 2) + hb[None, :]
    byte = tl.load(packed_ptr + at, mask=ok, other=0).to(tl.int32)
    block = at // 32
    q = tl.load(qscale_ptr + block, mask=ok, other=0).to(tl.int32)
    group = block // GROUP
    scale = libdevice.add_rn(
        libdevice.mul_rn(tl.load(code2_ptr + q, mask=ok, other=0.0),
                         tl.load(scale2_ptr + group, mask=ok, other=0.0)),
        tl.load(offset_ptr + group, mask=ok, other=0.0))
    # round to the compute dtype first (ROUND: tl.float16 or tl.bfloat16): the same weight values
    # the fp16 / bf16 path multiplies with
    high = (tl.load(nf4_ptr + (byte >> 4)) * scale).to(ROUND).to(tl.float32)
    low = (tl.load(nf4_ptr + (byte & 15)) * scale).to(ROUND).to(tl.float32)
    w = tl.join(high, low).reshape(BR, BC)  # columns 2j, 2j+1 of this tile
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    inv = tl.load(inv_ptr + c, mask=c < n_cols, other=0.0)
    if SR:  # stochastic rounding of the weights (int8_rounding: stochastic): a fresh dither
        u = tl.rand(seed, (r[:, None] * n_cols + c[None, :]).to(tl.int32))
        w8 = tl.floor(w * inv[None, :] + u)
    else:
        w8 = libdevice.rint(w * inv[None, :])
    w8 = tl.minimum(tl.maximum(w8, -127.0), 127.0).to(tl.int8)
    dst = c[None, :].to(tl.int64) * n_rows + r[:, None]
    tl.store(out_ptr + tl.trans(dst), tl.trans(w8),
             mask=tl.trans((r[:, None] < n_rows) & (c[None, :] < n_cols)))


def launch_int8t(pack, index, inv_colscale, out, rounding=None, br=64, bc=64, dtype=None):
    """Weight `index` of the pack as int8, transposed: out int8 [K_in, N_out] (contiguous); the
    weights are rounded to `dtype` (the compute dtype) before they are scaled to int8.
    ``rounding``: a ``policy.Rounding``; the weights are dithered only where it says so."""
    dtype = dtype or getattr(pack, "dtype", torch.float16)
    n_rows, n_cols = pack.shapes[index]
    first_byte = pack.byte_start[index]
    grid = (triton.cdiv(n_rows, br), triton.cdiv(n_cols, bc))
    sr = rounding is not None and rounding.weights
    kz_nf4_int8t_kernel[grid](pack.packed, pack.qscale, pack.scale2, pack.offset, pack.code2,
                              pack.nf4, inv_colscale, out, first_byte, n_rows, n_cols,
                              rounding.weight_seed() if sr else 0,
                              BR=br, BC=bc, GROUP=256, ROUND=_TL_DTYPE[dtype], SR=sr, num_warps=4)
    return out
