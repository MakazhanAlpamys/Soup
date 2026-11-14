# SPDX-License-Identifier: Apache-2.0
#
# Origin: Soup Kernel Challenge (Decentrathon 6.0), the submission of team Luck
# (BAITC-Hacks/hack-e79529fc-luck@b3f1225f); the int8 gate/up packs follow ideas seen in the
# solutions of teams Null Pointers, 21, DarkDaur and OrbiTeam.
# Modified for Soup: no functional change (the host-parked lm_head is passed in by the caller).
# ruff: noqa: N803, N806  (Triton's constexpr arguments are upper case by convention)
"""lm_head + cross-entropy + TRL's metrics, chunked, without ever holding the [tokens, 151936]
logits.

Base Soup materializes the fp16 logits, an fp32 copy for the loss, the saved fp32 log_softmax,
accelerate's fp32 copy of the outputs and TRL's `.contiguous()` copy for accuracy: several GB
around the loss. Here a chunk of tokens at a time goes through:

1. cuBLAS: logits of the chunk in the compute dtype (fp16 or bf16: the same rounding the base loss
   and metrics see);
2. `kz_vocab_rows_kernel`, one program per token, one pass over the vocabulary with an online
   max: log-sum-exp, entropy (lse - sum p*z), first-index argmax and the target's logit; for
   tokens with a label, a second pass overwrites the chunk in place with (softmax - onehot),
   times a constant that keeps small probabilities out of fp16 subnormals;
3. cuBLAS again for the labelled tokens: d hidden = grad logits @ W, accumulated in fp32.

So the gradient of the head is ready at the end of the forward and the logits are never
recomputed. Ideas: the chunked/fused linear cross-entropy of Liger-Kernel
(github.com/linkedin/Liger-Kernel) and Apple's Cut Cross-Entropy (arXiv:2411.09009); the code
here is our own.
"""

import torch
import triton
import triton.language as tl

GRAD_SCALE = 1024.0
BLOCK_V = 2048
CHUNK = 256  # tokens per logits chunk: [CHUNK, 151936] fp16 logits at a time


@triton.jit(do_not_specialize=["first_row"])
def kz_vocab_rows_kernel(logits_ptr, row_stride, first_row, target_ptr, entropy_ptr, loss_ptr,
                         hit_ptr, n_vocab, grad_mul, WITH_GRAD: tl.constexpr,
                         BLOCK: tl.constexpr):
    # one row of a logits chunk; rows of the per-token arrays are offset by first_row (passing
    # the offset instead of sliced pointers keeps one compiled specialization)
    row = tl.program_id(0)
    base = logits_ptr + row.to(tl.int64) * row_stride
    row += first_row
    target = tl.load(target_ptr + row)
    best = tl.full((), float("-inf"), tl.float32)
    best_at = tl.full((), 0, tl.int32)
    sum_e = tl.full((), 0.0, tl.float32)
    sum_ez = tl.full((), 0.0, tl.float32)
    for start in range(0, n_vocab, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        inside = cols < n_vocab
        z = tl.load(base + cols, mask=inside, other=float("-inf")).to(tl.float32)
        top = tl.max(z, axis=0)
        # first index of the maximum, as torch.argmax: lowest column inside the block, and a
        # later block only wins with a strictly larger value
        top_at = tl.min(tl.where(z == top, cols, n_vocab), axis=0)
        best_at = tl.where(top > best, top_at, best_at)
        new_best = tl.maximum(best, top)
        rescale = tl.exp(best - new_best)
        e = tl.exp(z - new_best)
        sum_e = sum_e * rescale + tl.sum(e, axis=0)
        sum_ez = sum_ez * rescale + tl.sum(tl.where(inside, e * z, 0.0), axis=0)
        best = new_best
    lse = best + tl.log(sum_e)
    tl.store(entropy_ptr + row, lse - sum_ez / sum_e)
    labelled = target >= 0
    z_target = tl.load(base + tl.maximum(target, 0)).to(tl.float32)
    tl.store(loss_ptr + row, tl.where(labelled, lse - z_target, 0.0))
    tl.store(hit_ptr + row, (labelled & (best_at == target)).to(tl.int32))
    if WITH_GRAD:
        for start in range(0, n_vocab, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            inside = cols < n_vocab
            z = tl.load(base + cols, mask=inside, other=0.0).to(tl.float32)
            p = tl.exp(z - lse)
            g = tl.where(cols == target, p - 1.0, p) * grad_mul
            tl.store(base + cols, g.to(logits_ptr.dtype.element_ty), mask=inside)


def head_forward(x16, weight16, targets, n_items, n_labelled=None, chunk=CHUNK):
    """x16: fp16 or bf16 [T, H] (final norm output as the head sees it); weight16: [V, H], same
    dtype;
    targets: int64 [T], the next token's label or -100; n_items: the loss denominator;
    n_labelled: the number of targets that are not -100, if the caller knows it (saves a sync).

    Returns (loss fp32 scalar, d_x fp32 [T, H] = d loss / d x16, entropy fp32 [T],
    token_loss fp32 [T], hit int32 [T])."""
    rows, hidden = x16.shape
    n_vocab = weight16.shape[0]
    dev = x16.device
    entropy = torch.empty(rows, device=dev, dtype=torch.float32)
    token_loss = torch.empty(rows, device=dev, dtype=torch.float32)
    hit = torch.empty(rows, device=dev, dtype=torch.int32)
    d_x = torch.zeros(rows, hidden, device=dev, dtype=torch.float32)

    # labelled tokens first: only they need the gradient pass and the second matmul
    labelled = targets >= 0
    order = torch.argsort((~labelled).to(torch.int8), stable=True)
    if n_labelled is None:
        n_labelled = int(labelled.sum())
    x_sorted = x16.index_select(0, order)
    t_sorted = targets.index_select(0, order)
    ent_sorted, loss_sorted, hit_sorted = (torch.empty_like(entropy), torch.empty_like(token_loss),
                                           torch.empty_like(hit))
    start = 0
    while start < rows:
        with_grad = start < n_labelled
        # never mix labelled and unlabelled rows in one chunk
        stop = min(start + chunk, n_labelled if with_grad else rows)
        logits = torch.mm(x_sorted[start:stop], weight16.t())
        kz_vocab_rows_kernel[(stop - start,)](
            logits, logits.stride(0), start, t_sorted, ent_sorted, loss_sorted, hit_sorted,
            n_vocab, GRAD_SCALE, WITH_GRAD=with_grad, BLOCK=BLOCK_V, num_warps=8)
        if with_grad:
            d_x[order[start:stop]] = torch.mm(logits, weight16, out_dtype=torch.float32)
        del logits
        start = stop
    entropy[order] = ent_sorted
    token_loss[order] = loss_sorted
    hit[order] = hit_sorted
    d_x.mul_(1.0 / GRAD_SCALE / n_items)
    loss = token_loss.sum() / n_items
    return loss, d_x, entropy, token_loss, hit


# --- the tied weight in pinned host memory (hostembed.py) -------------------------------------
# Without the weight on the GPU the head walks the vocabulary in slabs of SLAB rows, each copied
# from pinned host memory on a side stream while the previous slab is computed (two buffers):
#   cuBLAS logits [T, slab] for all tokens -> kz_head_slab_kernel: online max / first-index
#   argmax / sum e / sum e*z across slabs and the target's logit; for labelled tokens it
#   overwrites the slab with e = exp(z - m) * GRAD_SCALE (m = running max) and rescales the
#   token's fp32 gradient accumulator by exp(m_old - m) when the max moved -> cuBLAS
#   acc += e @ W_slab (fp32 accumulate).
# A few torch ops then form lse, entropy, loss, hit and
#   d_x = (acc / (sum e * GRAD_SCALE) - W[target]) / n_items in token order.
# One pass over the weight per step (~445 MiB host -> GPU); the online softmax-weighted
# accumulation follows FlashAttention (Dao et al., 2022); the code is ours. Per token, lse,
# argmax and the sums match head_forward's up to fp32 summation order (slab boundaries).
SLAB = 8192
_copy_streams = {}


@triton.jit(do_not_specialize=["n_cols", "col0", "n_grad", "first"])
def kz_head_slab_kernel(z_ptr, n_cols, col0, ts_ptr, m_ptr, s_ptr, sz_ptr, best_ptr, zt_ptr,
                        acc_ptr, n_grad, hidden, grad_scale, first,
                        BLOCK: tl.constexpr, BLOCK_H: tl.constexpr):
    row = tl.program_id(0)
    base = z_ptr + row.to(tl.int64) * n_cols
    t = tl.load(ts_ptr + row)
    carry = first == 0  # the first slab starts the running state
    m_old = tl.load(m_ptr + row, mask=carry, other=float("-inf"))
    s = tl.load(s_ptr + row, mask=carry, other=0.0)
    sz = tl.load(sz_ptr + row, mask=carry, other=0.0)
    best = tl.load(best_ptr + row, mask=carry, other=0)
    zt = tl.load(zt_ptr + row, mask=carry, other=0.0)
    local = t - col0
    in_slab = (local >= 0) & (local < n_cols)
    z_t = tl.load(base + tl.minimum(tl.maximum(local, 0), n_cols - 1)).to(tl.float32)
    zt = tl.where(in_slab, z_t, zt)
    # one online pass over the slab: max, first-index argmax, sum e, sum e*z (relative to top)
    top = tl.full((), float("-inf"), tl.float32)
    top_at = tl.full((), 0, tl.int32)
    se = tl.full((), 0.0, tl.float32)
    sez = tl.full((), 0.0, tl.float32)
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        inside = cols < n_cols
        z = tl.load(base + cols, mask=inside, other=float("-inf")).to(tl.float32)
        bm = tl.max(z, axis=0)
        at = tl.min(tl.where(z == bm, cols, n_cols), axis=0)
        top_at = tl.where(bm > top, at, top_at)
        new_top = tl.maximum(top, bm)
        rs = tl.exp(top - new_top)
        e = tl.exp(z - new_top)
        se = se * rs + tl.sum(e, axis=0)
        sez = sez * rs + tl.sum(tl.where(inside, e * z, 0.0), axis=0)
        top = new_top
    m_new = tl.maximum(m_old, top)
    best = tl.where(top > m_old, col0 + top_at, best)
    r_old = tl.exp(m_old - m_new)
    r_top = tl.exp(top - m_new)
    tl.store(m_ptr + row, m_new)
    tl.store(s_ptr + row, s * r_old + se * r_top)
    tl.store(sz_ptr + row, sz * r_old + sez * r_top)
    tl.store(best_ptr + row, best)
    tl.store(zt_ptr + row, zt)
    if row < n_grad:
        for start in range(0, n_cols, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            inside = cols < n_cols
            z = tl.load(base + cols, mask=inside, other=float("-inf")).to(tl.float32)
            tl.store(base + cols, (tl.exp(z - m_new) * grad_scale).to(z_ptr.dtype.element_ty),
                     mask=inside)
        if carry & (r_old < 1.0):  # the max moved: rescale this token's accumulator
            arow = acc_ptr + row.to(tl.int64) * hidden
            for h0 in range(0, hidden, BLOCK_H):
                hc = h0 + tl.arange(0, BLOCK_H)
                a = tl.load(arow + hc, mask=hc < hidden)
                tl.store(arow + hc, a * r_old, mask=hc < hidden)


def head_forward_streamed(x16, host_weight, targets, n_items, n_labelled, target_rows,
                          slab=SLAB, block=BLOCK_V):
    """head_forward with the weight in pinned host memory: host_weight [V, H] in x16's dtype (CPU,
    pinned); target_rows [n_labelled, H] on the GPU: the weight's rows of the labelled tokens'
    targets, in token order. Same outputs as head_forward up to fp32 summation order (slabs)."""
    rows, hidden = x16.shape
    n_vocab = host_weight.shape[0]
    dev = x16.device
    main = torch.cuda.current_stream(dev)
    copy = _copy_streams.get(dev)
    if copy is None:
        copy = _copy_streams[dev] = torch.cuda.Stream(dev)
    labelled = targets >= 0
    order = torch.argsort((~labelled).to(torch.int8), stable=True)
    n_labelled = int(n_labelled)
    xs = x16.index_select(0, order)
    ts = targets.index_select(0, order)
    padded = (rows + 3) // 4 * 4  # keeps every state row 16-byte aligned (one specialization)
    st = torch.empty(5, padded, device=dev, dtype=torch.float32)
    m, s, sz, zt, best = st[0], st[1], st[2], st[3], st[4].view(torch.int32)
    acc = torch.empty(n_labelled, hidden, device=dev, dtype=torch.float32)
    wbuf = [torch.empty(slab * hidden, device=dev, dtype=x16.dtype) for _ in range(2)]
    zbuf = torch.empty(rows * slab, device=dev, dtype=x16.dtype)
    ready = [torch.cuda.Event() for _ in range(2)]
    free = [torch.cuda.Event() for _ in range(2)]
    n_slabs = triton.cdiv(n_vocab, slab)

    def issue(i):  # slab i: host -> buffer i % 2, once the compute on its previous slab is queued
        b, v0 = i % 2, i * slab
        vs = min(slab, n_vocab - v0)
        if i >= 2:
            copy.wait_event(free[b])
        with torch.cuda.stream(copy):
            wbuf[b][:vs * hidden].view(vs, hidden).copy_(host_weight[v0:v0 + vs], non_blocking=True)
            ready[b].record(copy)

    # small [T, slab] outputs make cuBLAS pick split-K with fp16 partial sums; ask for fp32
    # reductions, as the full-vocabulary GEMM gets (restored right after; no other thread runs
    # CUDA work at this point)
    matmul = torch.backends.cuda.matmul
    reduced = matmul.allow_fp16_reduced_precision_reduction
    reduced_bf16 = matmul.allow_bf16_reduced_precision_reduction
    matmul.allow_fp16_reduced_precision_reduction = False
    matmul.allow_bf16_reduced_precision_reduction = False
    try:
        # the slab buffers come from the main stream's pool: their previous users may still be
        # queued on it, so the copy stream starts after everything queued so far
        copy.wait_stream(main)
        for i in range(min(2, n_slabs)):
            issue(i)
        for i in range(n_slabs):
            b, v0 = i % 2, i * slab
            vs = min(slab, n_vocab - v0)
            main.wait_event(ready[b])
            w = wbuf[b][:vs * hidden].view(vs, hidden)
            z = zbuf[:rows * vs].view(rows, vs)
            torch.mm(xs, w.t(), out=z)
            kz_head_slab_kernel[(rows,)](z, vs, v0, ts, m, s, sz, best, zt, acc, n_labelled,
                                         hidden, GRAD_SCALE, int(i == 0), BLOCK=block,
                                         BLOCK_H=512, num_warps=8)
            if n_labelled:
                if i == 0:
                    torch.mm(z[:n_labelled], w, out_dtype=torch.float32, out=acc)
                else:
                    torch.addmm(acc, z[:n_labelled], w, out_dtype=torch.float32, out=acc)
            free[b].record(main)
            if i + 2 < n_slabs:
                issue(i + 2)
    finally:
        matmul.allow_fp16_reduced_precision_reduction = reduced
        matmul.allow_bf16_reduced_precision_reduction = reduced_bf16
    wbuf.clear()  # main has waited for every copy, so the blocks are free on this stream
    del zbuf
    m, s, sz, zt, best = m[:rows], s[:rows], sz[:rows], zt[:rows], best[:rows]
    lse = m + s.log()
    labelled_sorted = ts >= 0
    entropy = torch.empty(rows, device=dev, dtype=torch.float32)
    token_loss = torch.empty(rows, device=dev, dtype=torch.float32)
    hit = torch.empty(rows, device=dev, dtype=torch.int32)
    entropy[order] = lse - sz / s
    token_loss[order] = torch.where(labelled_sorted, lse - zt, 0.0)
    hit[order] = (labelled_sorted & (best == ts)).to(torch.int32)
    # d loss / d x = (softmax - onehot) @ W / n_items = (acc / sum e / GRAD_SCALE - W[t]) / n
    d_x = torch.zeros(rows, hidden, device=dev, dtype=torch.float32)
    if n_labelled:
        lab = slice(0, n_labelled)
        acc.mul_((1.0 / GRAD_SCALE / s[lab])[:, None]).sub_(target_rows.float())
        d_x[order[lab]] = acc.div_(n_items)
    loss = token_loss.sum() / n_items
    return loss, d_x, entropy, token_loss, hit
