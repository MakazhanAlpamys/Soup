"""Correctness of the Triton kernels of ``training.packed_lora`` against plain PyTorch (and
bitsandbytes), forward and backward. GPU tests: ``pytest -m gpu --no-cov`` on a Linux CUDA machine
(WSL2 included); skipped everywhere else. The kernels were written for the Soup Kernel Challenge
(see the origin note of each module); the tests are the challenge team's, adapted.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

pytestmark = pytest.mark.gpu

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}


def _rotate_half(x):
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)



def test_qkv_rope_matches_pytorch():
    from soup_cli.kernels.packed_lora.triton.rope import qkv_rope_bwd, qkv_rope_fwd

    torch.manual_seed(0)
    rows, qh, kh, d = 300, 12, 2, 128
    qkv = torch.randn(rows, (qh + 2 * kh) * d, device="cuda").half()
    lq = torch.randn(rows, qh * d, device="cuda").half()
    lv = torch.randn(rows, kh * d, device="cuda").half()
    pos = torch.randint(0, 512, (rows,), device="cuda", dtype=torch.int32)
    inv_freq = 1.0 / (1e6 ** (torch.arange(0, d, 2, device="cuda").float() / d))
    freqs = torch.arange(512, device="cuda").float()[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos(), emb.sin()

    q32 = (qkv[:, : qh * d].float() + lq.float()).view(rows, qh, d).requires_grad_()
    k32 = qkv[:, qh * d:(qh + kh) * d].float().view(rows, kh, d).requires_grad_()
    c, s = cos[pos.long()][:, None, :], sin[pos.long()][:, None, :]
    q_ref = q32 * c + _rotate_half(q32) * s
    k_ref = k32 * c + _rotate_half(k32) * s
    v_ref = qkv[:, (qh + kh) * d:].float() + lv.float()

    q, k, v = qkv_rope_fwd(qkv, lq, None, lv, cos, sin, pos, qh, kh, d)
    # fp16 outputs, k and v repeated per query head (head h reads kv head h // 6); the kernel
    # may fuse a multiply-add where PyTorch rounds twice (1 fp16 ulp)
    group = qh // kh
    k_rep = k_ref.repeat_interleave(group, dim=1)
    v_rep = v_ref.view(rows, kh, d).repeat_interleave(group, dim=1)
    torch.testing.assert_close(q.float(), q_ref.half().float(), rtol=1e-3, atol=2e-3)
    torch.testing.assert_close(k.float(), k_rep.half().float(), rtol=1e-3, atol=2e-3)
    assert torch.equal(v, v_rep.half())

    # backward: per-query-head fp16 dk/dv (what the attention kernel returns), summed per kv
    # head in fp32
    gq = torch.randn(rows, qh, d, device="cuda").half()
    gk = torch.randn(rows, qh, d, device="cuda").half()
    gv = torch.randn(rows, qh, d, device="cuda").half()
    (q_ref * gq.float()).sum().backward()
    (k_ref.repeat_interleave(group, dim=1) * gk.float()).sum().backward()
    g, *_ = qkv_rope_bwd(gq, gk, gv, cos, sin, pos, qh, kh, d)
    torch.testing.assert_close(g[:, : qh * d].float(), q32.grad.reshape(rows, -1).half().float(),
                               rtol=1e-3, atol=2e-3)
    torch.testing.assert_close(g[:, qh * d:(qh + kh) * d].float(),
                               k32.grad.reshape(rows, -1).half().float(), rtol=1e-3, atol=2e-3)
    gv_sum = gv.float().view(rows, kh, group, d).sum(2).reshape(rows, -1)
    torch.testing.assert_close(g[:, (qh + kh) * d:].float(), gv_sum.half().float(), rtol=1e-3,
                               atol=2e-3)


@pytest.mark.parametrize("scaling", [2.0, 1.0 / 3.0])
def test_qkv_rope_lora_scaling_matches_torch_multiply(scaling):
    """The LoRA scaling the rotary kernels apply is bit-identical to torch's fp16 multiply by a
    Python float, forward (LoRA-B outputs) and backward (their gradients)."""
    from soup_cli.kernels.packed_lora.triton.rope import qkv_rope_bwd, qkv_rope_fwd

    torch.manual_seed(0)
    rows, qh, kh, d = 300, 12, 2, 128
    qkv = torch.randn(rows, (qh + 2 * kh) * d, device="cuda").half()
    lq = torch.randn(rows, qh * d, device="cuda").half()
    lv = torch.randn(rows, kh * d, device="cuda").half()
    pos = torch.randint(0, 512, (rows,), device="cuda", dtype=torch.int32)
    cos, sin = torch.rand(512, d, device="cuda"), torch.rand(512, d, device="cuda")

    lk = torch.randn(rows, kh * d, device="cuda").half()
    got = qkv_rope_fwd(qkv, lq, lk, lv, cos, sin, pos, qh, kh, d, scaling)
    want = qkv_rope_fwd(qkv, lq * scaling, lk * scaling, lv * scaling, cos, sin, pos, qh, kh, d)
    for a, b in zip(got, want):
        assert torch.equal(a, b)

    gq, gk, gv = (torch.randn(rows, qh, d, device="cuda").half() for _ in range(3))
    g, glq, glk, glv = qkv_rope_bwd(gq, gk, gv, cos, sin, pos, qh, kh, d, scaling,
                                    lora=(True, True, True))
    assert torch.equal(g, qkv_rope_bwd(gq, gk, gv, cos, sin, pos, qh, kh, d)[0])
    assert torch.equal(glq, g[:, : qh * d] * scaling)
    assert torch.equal(glk, g[:, qh * d:(qh + kh) * d] * scaling)
    assert torch.equal(glv, g[:, (qh + kh) * d:] * scaling)


@pytest.mark.parametrize("rows,vocab", [(700, 151936), (33, 5000)])
def test_head_matches_pytorch(rows, vocab):
    from soup_cli.kernels.packed_lora.triton.head import head_forward

    torch.manual_seed(0)
    hidden = 1536
    x = torch.randn(rows, hidden, device="cuda").half().requires_grad_()
    weight = (torch.randn(vocab, hidden, device="cuda") * 0.03).half()
    targets = torch.randint(0, vocab, (rows,), device="cuda")
    targets[torch.rand(rows, device="cuda") < 0.4] = -100
    n_items = (targets >= 0).sum()

    logits = (x @ weight.t()).float()
    token_ce = torch.nn.functional.cross_entropy(logits, targets, reduction="none",
                                                 ignore_index=-100)
    (token_ce.sum() / n_items).backward()
    with torch.no_grad():
        logp = torch.log_softmax(logits, -1)
        entropy = -(logp.exp() * logp).sum(-1)
        hit = (logits.argmax(-1) == targets) & (targets >= 0)

    loss, d_x, ent, token_loss, hits = head_forward(x.detach(), weight, targets, n_items)
    torch.testing.assert_close(loss, token_ce.sum() / n_items, rtol=1e-4, atol=1e-4)
    # the logits are rounded to fp16 (an ulp of 4e-3 near 5): which cuBLAS algorithm the tiny
    # shapes get changes the last bit, so the bar is half an ulp, not 1e-4
    torch.testing.assert_close(ent, entropy, rtol=1e-3, atol=3e-3)
    torch.testing.assert_close(token_loss, token_ce, rtol=1e-3, atol=3e-3)
    assert torch.equal(hits.bool(), hit)
    assert torch.all(token_loss[targets < 0] == 0)
    # the reference rounds the gradient to fp16; ours accumulates in fp32
    torch.testing.assert_close(d_x, x.grad.float(), rtol=2e-2, atol=2e-6)


def test_nf4_pack_dequant_is_bit_exact():
    import bitsandbytes as bnb
    import bitsandbytes.functional as bnb_functional

    from soup_cli.kernels.packed_lora.triton.nf4 import Pack

    torch.manual_seed(0)
    # one Qwen2.5-1.5B layer: q, k, v, o, gate, up, down
    shapes = [(1536, 1536), (256, 1536), (256, 1536), (1536, 1536), (8960, 1536), (8960, 1536),
              (1536, 8960)]
    weights = [bnb.nn.Params4bit((torch.randn(shape) * 0.02).half(), requires_grad=False,
                                 compress_statistics=True, quant_type="nf4").cuda()
               for shape in shapes]
    reference = [bnb_functional.dequantize_4bit(w.data, w.quant_state) for w in weights]
    assert Pack.supported(weights)
    pack = Pack(weights)
    for first, stop in [(0, 3), (3, 6), (6, 7), (3, 7), (0, 7)]:
        for got, want in zip(pack.dequant(first, stop), reference[first:stop]):
            assert torch.equal(got, want)
    assert torch.equal(pack.matrix(0, 3), torch.cat(reference[:3]))
    # the parameters now live in the pack; bitsandbytes still decodes them the same way
    for w, want in zip(weights, reference):
        assert torch.equal(bnb_functional.dequantize_4bit(w.data, w.quant_state), want)


def test_nf4_pack_split_matches_dequant():
    """split(0, 3, 4) decodes q/k/v (one matrix) and o in one launch, bit for bit as before."""
    import bitsandbytes as bnb
    import bitsandbytes.functional as bnb_functional

    from soup_cli.kernels.packed_lora.triton.nf4 import Pack

    torch.manual_seed(0)
    shapes = [(1536, 1536), (256, 1536), (256, 1536), (1536, 1536), (8960, 1536)]
    weights = [bnb.nn.Params4bit((torch.randn(shape) * 0.02).half(), requires_grad=False,
                                 compress_statistics=True, quant_type="nf4").cuda()
               for shape in shapes]
    reference = [bnb_functional.dequantize_4bit(w.data, w.quant_state) for w in weights]
    pack = Pack(weights)
    qkv, o = pack.split(0, 3, 4)
    assert torch.equal(qkv, torch.cat(reference[:3])) and torch.equal(o, reference[3])
    qkv, o, gate = pack.split(0, 3, 5)
    assert torch.equal(qkv, pack.matrix(0, 3)) and torch.equal(gate, reference[4])


@pytest.mark.parametrize("rows,vocab", [(700, 151936), (33, 5000)])
def test_streamed_head_matches_pytorch(rows, vocab):
    """The head with its weight in pinned host memory, walked in slabs (hostembed.py)."""
    from soup_cli.kernels.packed_lora.triton.head import head_forward_streamed

    torch.manual_seed(0)
    hidden = 1536
    x = torch.randn(rows, hidden, device="cuda").half().requires_grad_()
    weight = (torch.randn(vocab, hidden, device="cuda") * 0.03).half()
    targets = torch.randint(0, vocab, (rows,), device="cuda")
    targets[torch.rand(rows, device="cuda") < 0.4] = -100
    n_items = (targets >= 0).sum()

    logits = (x @ weight.t()).float()
    token_ce = torch.nn.functional.cross_entropy(logits, targets, reduction="none",
                                                 ignore_index=-100)
    (token_ce.sum() / n_items).backward()
    with torch.no_grad():
        logp = torch.log_softmax(logits, -1)
        entropy = -(logp.exp() * logp).sum(-1)
        hit = (logits.argmax(-1) == targets) & (targets >= 0)

    host = weight.cpu().pin_memory()
    labelled = targets >= 0
    target_rows = weight[targets[labelled]]
    loss, d_x, ent, token_loss, hits = head_forward_streamed(
        x.detach(), host, targets, n_items, int(labelled.sum()), target_rows, slab=4096)
    torch.testing.assert_close(loss, token_ce.sum() / n_items, rtol=1e-4, atol=1e-4)
    # the logits are rounded to fp16 (an ulp of 4e-3 near 5): which cuBLAS algorithm the tiny
    # shapes get changes the last bit, so the bar is half an ulp, not 1e-4
    torch.testing.assert_close(ent, entropy, rtol=1e-3, atol=3e-3)
    torch.testing.assert_close(token_loss, token_ce, rtol=1e-3, atol=3e-3)
    assert torch.equal(hits.bool(), hit)
    assert torch.all(token_loss[targets < 0] == 0)
    torch.testing.assert_close(d_x, x.grad.float(), rtol=2e-2, atol=2e-6)


def test_int8_mlp_input_grad_close_to_fp32():
    """The int8 backward of the MLP input gradient (int8.py, the path of
    layers._mlp_input_grad_int8 with torch-made int8 weights): d_x = SwiGLU-bwd(g @ W_down)
    through W_gate / W_up, against fp32 PyTorch. Lossy by design (int8 rounding of the gradient
    and of the weights): close, not equal, and the same on every run."""
    from soup_cli.kernels.packed_lora.triton import int8

    torch.manual_seed(0)
    rows, hidden, inter = 300, 1536, 8960
    w_down = (torch.randn(hidden, inter, device="cuda") * 0.02).half()  # [out, in] as Linear
    w_gate = (torch.randn(inter, hidden, device="cuda") * 0.02).half()
    w_up = (torch.randn(inter, hidden, device="cuda") * 0.02).half()
    g = torch.randn(rows, hidden, device="cuda") * 1e-3
    gate = torch.randn(rows, inter, device="cuda").half()
    up = torch.randn(rows, inter, device="cuda").half()

    def int8_t(w):  # as nf4.kz_nf4_int8t_kernel: W^T in int8, one scale per column of W
        top = w.abs().amax(0).float().clamp_min(1e-30)
        q = torch.round(w.float() * (127.0 / top)[None, :]).clamp(-127, 127).to(torch.int8)
        return q.t().contiguous(), top / 127.0

    def int8_path():
        g8, sg = int8.quant_rows(g)
        w8, s_down = int8_t(w_down)
        acc = torch._int_mm(g8, w8.t())
        dg8, sdg, du8, sdu = int8.swiglu_bwd_int8(acc, sg, s_down, gate, up)
        wg8, s_gate = int8_t(w_gate)
        wu8, s_up = int8_t(w_up)
        return int8.dequant_sum(torch._int_mm(dg8, wg8.t()), sdg, s_gate,
                                torch._int_mm(du8, wu8.t()), sdu, s_up)

    got = int8_path()
    g_act = g @ w_down.float()
    gt, ut = gate.float(), up.float()
    sig = torch.sigmoid(gt)
    want = (g_act * ut * sig * (1 + gt * (1 - sig))) @ w_gate.float() + \
        (g_act * gt * sig) @ w_up.float()
    cos = torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim=0)
    rel = (got - want).norm() / want.norm()
    assert cos > 0.999 and rel < 0.05, (cos.item(), rel.item())
    assert torch.equal(got, int8_path())


@pytest.mark.parametrize("rows,n", [(1, 64), (17, 4160), (300, 8960), (5, 12288)])
def test_q8_gate_up_packs(rows, n):
    """gate / up stored as int8 + one fp32 scale per 64 values (q8.py): the forward's activation is
    the example kernel's, bit for bit; the stored values are within half a step of their group's
    scale; the backward reading the packs gives d_gate / d_up close to the fp16-input int8 kernel's
    (lossy by design, the same on every run)."""
    import triton

    from soup_cli.kernels.packed_lora.triton import int8, q8
    from soup_cli.kernels.packed_lora.triton.swiglu import BLOCK, kz_swiglu_fwd_kernel

    torch.manual_seed(0)
    gate = (torch.randn(rows, n, device="cuda") * 2).half()
    up = (torch.randn(rows, n, device="cuda") * 1.5).half()
    gate[:, 7] = 40.0  # an outlier column
    if n > 64:
        gate[0, :64] = 0.0  # an all-zero group
    want = torch.empty_like(gate)
    kz_swiglu_fwd_kernel[(triton.cdiv(gate.numel(), BLOCK),)](gate, up, want, gate.numel(),
                                                                   BLOCK=BLOCK)
    gate_pack = q8.quant_q8(gate)
    act, gate_pack_2, up_pack = q8.swiglu_fwd_q8(gate, up)  # packs both in one pass
    assert torch.equal(act, want) and torch.equal(gate_pack_2, gate_pack)
    act, gate_pack_3, up_pack_2 = q8.swiglu_fwd_q8(gate, up, gate_pack)  # gate's pack made before
    assert torch.equal(act, want) and gate_pack_3 is gate_pack and torch.equal(up_pack_2, up_pack)
    for pack, x in ((gate_pack, gate), (up_pack, up)):
        assert pack.numel() == rows * n + 4 * rows * (n // q8.GROUP)
        q = pack[:rows * n].view(rows, n)
        step = pack[rows * n:].view(torch.float32).view(rows, -1).repeat_interleave(q8.GROUP, dim=1)
        assert ((q.float() * step - x.float()).abs() <= step * 0.5001).all()
        assert q.abs().max() <= 127
    acc = torch.randint(-2000, 2000, (rows, n), device="cuda", dtype=torch.int32)
    srow = torch.rand(rows, device="cuda") * 1e-3
    scol = torch.rand(n, device="cuda") * 1e-2
    a = int8.swiglu_bwd_int8(acc, srow, scol, gate, up)
    b = q8.swiglu_bwd_q8(acc, srow, scol, gate_pack, up_pack)
    for i in (0, 2):  # d_gate, d_up: int8 values times their row's scale
        got, ref = b[i].float() * b[i + 1][:, None], a[i].float() * a[i + 1][:, None]
        assert (got - ref).norm() / ref.norm() < 0.05
        assert torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0) > 0.999
    assert all(torch.equal(x, y) for x, y in zip(b, q8.swiglu_bwd_q8(acc, srow, scol, gate_pack,
                                                                     up_pack)))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nf4_dequant_bit_exact_in_both_dtypes(dtype):
    """The decoder's output equals bitsandbytes' own dequantization in fp16 and in bf16."""
    import bitsandbytes as bnb
    import bitsandbytes.functional as bnb_functional

    from soup_cli.kernels.packed_lora.triton import nf4

    torch.manual_seed(0)
    weights = []
    for shape in ((256, 512), (128, 512), (512, 256)):
        w = (torch.randn(*shape, device="cuda") * 0.05).to(dtype)
        packed, state = bnb_functional.quantize_4bit(w, quant_type="nf4", compress_statistics=True)
        weights.append(bnb.nn.Params4bit(packed, requires_grad=False, quant_state=state,
                                         bnb_quantized=True))
    if not nf4.Pack.supported(weights):
        pytest.skip("layout the decoder does not handle")
    want = [bnb_functional.dequantize_4bit(w.data, w.quant_state).view(w.quant_state.shape)
            for w in weights]
    pack = nf4.Pack(weights)
    assert pack.dtype == dtype
    for got, ref in zip(pack.dequant(0, 3), want):
        assert got.dtype == dtype and torch.equal(got, ref)


@pytest.mark.parametrize("n", [8960, 18944, 12288, 64 * 300])
def test_q8_loop_kernel_equals_one_pass_kernel(n):
    """The two-pass SwiGLU backward (any width, optional LoRA term and compute-dtype outputs) gives
    the one-pass kernel's values where both apply, and rebuilds act from the packs."""
    from soup_cli.kernels.packed_lora.triton import q8

    torch.manual_seed(0)
    rows = 37
    gate = torch.randn(rows, n, device="cuda").half()
    up = torch.randn(rows, n, device="cuda").half()
    gp, up_p = q8.quant_q8(gate), q8.quant_q8(up)
    acc = torch.randint(-5000, 5000, (rows, n), device="cuda", dtype=torch.int32)
    srow = torch.rand(rows, device="cuda") * 1e-3
    scol = torch.rand(n, device="cuda") * 1e-3
    if n // q8.GROUP <= q8.MAX_GROUPS:
        want = q8.swiglu_bwd_q8(acc, srow, scol, gp, up_p)
        got = q8.swiglu_bwd_q8(acc, srow, scol, gp, up_p, fp_dtype=torch.float16)
        for a, b in zip(got[:4], want):
            assert torch.equal(a, b)
    extra = (torch.randn(rows, n, device="cuda") * 1e-3).half()
    *_, dgf, duf = q8.swiglu_bwd_q8(acc, srow, scol, gp, up_p, extra, torch.bfloat16)
    g = acc.float() * srow[:, None] * scol[None, :] + extra.float()
    gate_q = (gp[: rows * n].view(rows, n // 64, 64).float()
              * gp[rows * n:].view(torch.float32).view(rows, -1, 1)).view(rows, n)
    up_q = (up_p[: rows * n].view(rows, n // 64, 64).float()
            * up_p[rows * n:].view(torch.float32).view(rows, -1, 1)).view(rows, n)
    sig = torch.sigmoid(gate_q)
    want_dg = g * up_q * sig * (1 + gate_q * (1 - sig))
    want_du = g * gate_q * sig
    torch.testing.assert_close(dgf.float(), want_dg.bfloat16().float(), rtol=1e-2, atol=1e-6)
    torch.testing.assert_close(duf.float(), want_du.bfloat16().float(), rtol=1e-2, atol=1e-6)
    act = q8.act_from_q8(gp, up_p, rows, n, torch.float16)
    torch.testing.assert_close(act.float(), (gate_q * torch.sigmoid(gate_q) * up_q).half().float(),
                               rtol=1e-3, atol=1e-5)


def _reference(qkv, lq, lk, lv, cos, sin, pos, qh, kh, d, scaling, qw, kw, eps, dt):
    """The base run's order: projection (+ LoRA delta rounded to the compute dtype, summed in fp32)
    -> RMSNorm per head in fp32 (Qwen3RMSNorm) -> rotary in fp32 -> k, v repeated per query head."""
    t = qkv.shape[0]
    q = qkv[:, :qh * d].float().view(t, qh, d)
    k = qkv[:, qh * d:(qh + kh) * d].float().view(t, kh, d)
    v = qkv[:, (qh + kh) * d:].float().view(t, kh, d)
    if lq is not None:
        q = q + (lq.float().view(t, qh, d) * scaling).to(dt).float()
    if lk is not None:
        k = k + (lk.float().view(t, kh, d) * scaling).to(dt).float()
    if lv is not None:
        v = v + (lv.float().view(t, kh, d) * scaling).to(dt).float()

    def norm(x, w):
        rs = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return w * (x * rs), x * rs, rs[..., 0]

    qn, xq, rq = norm(q, qw)
    kn, xk, rk = norm(k, kw)

    def rope(x):
        c, s = cos[pos.long()][:, None, :], sin[pos.long()][:, None, :]
        x1, x2 = x[..., :d // 2], x[..., d // 2:]
        return torch.cat([x1 * c[..., :d // 2] - x2 * s[..., :d // 2],
                          x2 * c[..., d // 2:] + x1 * s[..., d // 2:]], -1)

    g = qh // kh
    return (rope(qn), rope(kn).repeat_interleave(g, 1), v.repeat_interleave(g, 1), xq, xk, rq, rk)


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("qh,kh,d", [(4, 2, 64), (6, 2, 32), (8, 8, 64), (16, 4, 128)])
def test_qk_norm_rope_forward_and_backward_match_pytorch(dtype, qh, kh, d):
    from soup_cli.kernels.packed_lora.triton import rope

    torch.manual_seed(0)
    dev, rows, scaling, eps = "cuda", 37, 2.0, 1e-6
    width = (qh + 2 * kh) * d
    qkv = (torch.randn(rows, width, device=dev) * 1.5).to(dtype)
    lq = torch.randn(rows, qh * d, device=dev).to(dtype)
    lk = torch.randn(rows, kh * d, device=dev).to(dtype)
    cos = torch.randn(64, d, device=dev)
    sin = torch.randn(64, d, device=dev)
    pos = torch.randint(0, 64, (rows,), device=dev, dtype=torch.int32)
    qw = 1.0 + 0.3 * torch.randn(d, device=dev)
    kw = 1.0 + 0.3 * torch.randn(d, device=dev)

    q, k, v, (xq, xk, rq, rk) = rope.qkv_rope_fwd(qkv, lq, lk, None, cos, sin, pos, qh, kh, d,
                                                  scaling, (qw, kw, eps))
    rq_, rk_, rv_, rxq, rxk, rrq, rrk = _reference(qkv, lq, lk, None, cos, sin, pos, qh, kh, d,
                                                   scaling, qw, kw, eps, dtype)
    assert _rel(q, rq_) < 6e-3 and _rel(k, rk_) < 6e-3 and _rel(v, rv_) < 4e-3
    assert _rel(xq, rxq.reshape(rows, -1)) < 6e-3 and _rel(xk, rxk.reshape(rows, -1)) < 6e-3
    assert _rel(rq, rrq) < 1e-5 and _rel(rk, rrk) < 1e-5

    # backward: gradients of a random linear function of (q, k, v) wrt the pre-norm projection
    dq = torch.randn(rows, qh, d, device=dev).to(dtype)
    dk = torch.randn(rows, qh, d, device=dev).to(dtype)
    dv = torch.randn(rows, qh, d, device=dev).to(dtype)
    x = qkv.float().clone().requires_grad_(True)
    out = _reference(x.to(dtype), lq, lk, None, cos, sin, pos, qh, kh, d, scaling, qw, kw, eps,
                     dtype)
    # (x.to(dtype) is a straight-through rounding: the gradient reaches x unchanged)
    loss = (out[0] * dq.float()).sum() + (out[1] * dk.float()).sum() + (out[2] * dv.float()).sum()
    loss.backward()
    g, glq, glk, _ = rope.qkv_rope_bwd(dq, dk, dv, cos, sin, pos, qh, kh, d, scaling,
                                       lora=(True, True, False),
                                       qk_norm=(qw, kw), qk_saved=(xq, xk, rq, rk))
    assert _rel(g, x.grad) < 1.2e-2
    assert _rel(glq, (g[:, :qh * d].float() * scaling)) < 1e-2
    assert _rel(glk, (g[:, qh * d:(qh + kh) * d].float() * scaling)) < 1e-2


def test_layer_bytes_counts_the_norm_stash():
    from soup_cli.kernels.packed_lora import layout as offload

    plain = offload.layer_bytes(1000, 2048, 6144, 2048, 1024, True, True, False)
    qwen3 = offload.layer_bytes(1000, 2048, 6144, 2048, 1024, True, True, True)
    assert qwen3 - plain == 1000 * 2 * (2048 + 1024)


# --- elementwise kernels and norms (the original suite went through autograd wrappers) --------


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 8960), (3, 77, 1001)])
def test_swiglu_kernels_match_pytorch(dtype, shape):
    import triton

    from soup_cli.kernels.packed_lora.triton.swiglu import (
        BLOCK,
        kz_swiglu_bwd_kernel,
        kz_swiglu_fwd_kernel,
    )

    torch.manual_seed(0)
    gate = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)
    ref = torch.nn.functional.silu(gate.float()) * up.float()
    ref.backward(grad.float())
    ref_dgate, ref_dup = gate.grad.float(), up.grad.float()
    n = gate.numel()
    out = torch.empty_like(gate)
    dgate, dup = torch.empty_like(gate), torch.empty_like(up)
    kz_swiglu_fwd_kernel[(triton.cdiv(n, BLOCK),)](gate.detach(), up.detach(), out, n, BLOCK=BLOCK)
    kz_swiglu_bwd_kernel[(triton.cdiv(n, BLOCK),)](grad, gate.detach(), up.detach(), dgate, dup, n,
                                                   BLOCK=BLOCK)
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref.detach(), rtol=tol, atol=tol)
    torch.testing.assert_close(dgate.float(), ref_dgate, rtol=tol, atol=tol)
    torch.testing.assert_close(dup.float(), ref_dup, rtol=tol, atol=tol)


def _qwen_rmsnorm(x, weight, eps):
    return weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 37, 700])
def test_rmsnorm_matches_pytorch(dtype, rows):
    from soup_cli.kernels.packed_lora.triton.norms import rmsnorm_bwd, rmsnorm_fwd

    torch.manual_seed(0)
    hidden, eps = 1536, 1e-6
    x = torch.randn(rows, hidden, device="cuda")
    weight = torch.rand(hidden, device="cuda") + 0.5
    y, rstd = rmsnorm_fwd(x, weight, eps, dtype=dtype)
    assert y.dtype == dtype and rstd.dtype == torch.float32
    torch.testing.assert_close(y, _qwen_rmsnorm(x, weight, eps).to(dtype), rtol=1e-2, atol=1e-2)
    xr = x.clone().requires_grad_(True)
    dy = torch.randn(rows, hidden, device="cuda")
    (_qwen_rmsnorm(xr, weight, eps) * dy).sum().backward()
    dx = rmsnorm_bwd(dy, x, weight, rstd)
    torch.testing.assert_close(dx, xr.grad, rtol=1e-4, atol=1e-5)
    # a gradient in the compute dtype is read as it is
    dx16 = rmsnorm_bwd(dy.to(dtype), x, weight, rstd)
    torch.testing.assert_close(dx16, rmsnorm_bwd(dy.to(dtype).float(), x, weight, rstd))


@pytest.mark.parametrize("rows", [1, 300])
def test_qkv_rope_bwd_rebuilds_the_norm_output(rows):
    """With norm=..., the rotary backward also writes the LoRA input, bit-identical to what
    rmsnorm_fwd wrote in the forward, and its other outputs do not change."""
    from soup_cli.kernels.packed_lora.triton.norms import rmsnorm_fwd
    from soup_cli.kernels.packed_lora.triton.rope import qkv_rope_bwd

    torch.manual_seed(0)
    qh, kh, d, hidden, eps = 12, 2, 128, 1536, 1e-6
    x = torch.randn(rows, hidden, device="cuda")
    weight = torch.rand(hidden, device="cuda") + 0.5
    y, rstd = rmsnorm_fwd(x, weight, eps, dtype=torch.float16)
    pos = torch.randint(0, 512, (rows,), device="cuda", dtype=torch.int32)
    cos, sin = torch.rand(512, d, device="cuda"), torch.rand(512, d, device="cuda")
    gq, gk, gv = (torch.randn(rows, qh, d, device="cuda").half() for _ in range(3))
    want = qkv_rope_bwd(gq, gk, gv, cos, sin, pos, qh, kh, d, 2.0)
    got = qkv_rope_bwd(gq, gk, gv, cos, sin, pos, qh, kh, d, 2.0, norm=(x, weight, rstd))
    want = tuple(t for t in want if t is not None)
    got_rest = tuple(t for t in got[:-1] if t is not None)
    assert len(got_rest) == len(want)
    for a, b in zip(got_rest, want):
        assert torch.equal(a, b)
    assert torch.equal(got[-1], y)


# --- the fused MLP forward and the pre-scaled backward outputs (q8.py) ------------------------


def _unpack_q8(pack, rows, n):
    values = pack[: rows * n].view(rows, n // 64, 64).float()
    scales = pack[rows * n:].view(torch.float32).view(rows, n // 64, 1)
    return (values * scales).view(rows, n)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows,inter,rank,which", [(203, 704, 16, "gud"), (64, 1664, 16, "gud"),
                                                   (130, 1152, 32, "gud"), (96, 640, 16, "gu"),
                                                   (96, 640, 16, "d")])
def test_fused_forward_matches_the_torch_sequence(dtype, rows, inter, rank, which):
    from soup_cli.kernels.packed_lora.triton import q8

    torch.manual_seed(0)
    dev = "cuda"
    gate = (torch.randn(rows, inter, device=dev) * 2).to(dtype)
    up = (torch.randn(rows, inter, device=dev) * 2).to(dtype)
    hg = torch.randn(rows, rank, device=dev).to(dtype)
    hu = torch.randn(rows, rank, device=dev).to(dtype)
    bg = (torch.randn(inter, rank, device=dev) * 0.1).to(dtype)
    bu = (torch.randn(inter, rank, device=dev) * 0.1).to(dtype)
    a_down = (torch.randn(rank, inter, device=dev) * 0.1).to(dtype)
    sg, su = 2.0, 1.5
    use_g, use_u, use_d = "g" in which, "u" in which, "d" in which
    gate_ref, up_ref = gate.clone(), up.clone()  # the three torch ops of the unfused forward
    if use_g:
        gate_ref.addmm_(hg, bg.t(), alpha=sg)
    if use_u:
        up_ref.addmm_(hu, bu.t(), alpha=su)
    act_ref, gp_ref, up_pack_ref = q8.swiglu_fwd_q8(gate_ref, up_ref)
    hd_ref = act_ref @ a_down.t() if use_d else None
    act, gp, upk, hd = q8.swiglu_lora_fwd_q8(
        gate, up, hg if use_g else None, bg if use_g else None, sg, hu if use_u else None,
        bu if use_u else None, su, a_down if use_d else None)
    tol = dict(rtol=2e-2, atol=2e-2) if dtype == torch.bfloat16 else dict(rtol=4e-3, atol=4e-3)
    torch.testing.assert_close(act.float(), act_ref.float(), **tol)
    for got, ref in ((gp, gp_ref), (upk, up_pack_ref)):  # same scales up to input rounding
        a, b = _unpack_q8(got, rows, inter), _unpack_q8(ref, rows, inter)
        step = 2.5 * b.abs().view(rows, inter // 64, 64).amax(2, keepdim=True).clamp_min(1e-6) / 127
        assert ((a - b).abs().view(rows, inter // 64, 64) <= step).all()
    if use_d:
        torch.testing.assert_close(hd.float(), hd_ref.float(), rtol=3e-2,
                                   atol=3e-1 if dtype == torch.bfloat16 else 3e-2)
    else:
        assert hd is None


def test_prescaled_backward_outputs_equal_the_separate_multiply():
    from soup_cli.kernels.packed_lora.triton import q8

    torch.manual_seed(1)
    rows, n = 70, 1280
    dev, dt = "cuda", torch.bfloat16
    acc = torch.randint(-3000, 3000, (rows, n), device=dev, dtype=torch.int32)
    srow = torch.rand(rows, device=dev) * 1e-3
    scol = torch.rand(n, device=dev) * 1e-2
    gate_pack = q8.quant_q8((torch.randn(rows, n, device=dev) * 2).to(dt))
    up_pack = q8.quant_q8((torch.randn(rows, n, device=dev) * 2).to(dt))
    extra = (torch.randn(rows, n, device=dev) * 1e-3).to(dt)
    plain = q8.swiglu_bwd_q8(acc, srow, scol, gate_pack, up_pack, extra, dt)
    scaled = q8.swiglu_bwd_q8(acc, srow, scol, gate_pack, up_pack, extra, dt, (2.0, 0.5))
    for a, b in zip(plain[:4], scaled[:4]):
        assert torch.equal(a, b)  # the int8 gradients do not change
    assert torch.equal(scaled[4], plain[4] * 2.0)
    assert torch.equal(scaled[5], plain[5] * 0.5)


# --- rounding of the int8 gradient quantisers -------------------------------------------------


def test_stochastic_rounding_is_unbiased_where_nearest_is_not():
    from soup_cli.kernels.packed_lora.policy import Rounding
    from soup_cli.kernels.packed_lora.triton import int8

    torch.manual_seed(0)
    rows, n = 8, 4096
    # a small coherent signal (0.01) next to one large entry (10) per row: the step is 10 / 127 =
    # 0.079, so round-to-nearest zeroes the whole signal, stochastic rounding keeps it in
    # expectation
    x = 0.01 + 0.002 * torch.randn(rows, n, device="cuda")
    x[:, 0] = 10.0
    q, s = int8.quant_rows(x)  # no rounding given: to nearest
    nearest = q.float() * s[:, None]
    assert float((nearest[:, 1:] - x[:, 1:]).mean()) < -0.009  # everything lost
    rounding = Rounding("grad")
    acc = torch.zeros_like(x)
    reps = 300
    for _ in range(reps):
        q, s = int8.quant_rows(x, rounding)
        acc += q.float() * s[:, None]
    mean = acc / reps
    assert float((mean[:, 1:] - x[:, 1:]).mean().abs()) < 3e-4  # the expectation is the input
    q1, _ = int8.quant_rows(x, rounding)
    q2, _ = int8.quant_rows(x, rounding)
    assert not torch.equal(q1, q2)  # and it is random: two launches differ
    q3, _ = int8.quant_rows(x, Rounding("nearest"))
    q4, _ = int8.quant_rows(x, Rounding("nearest"))
    assert torch.equal(q3, q4)  # round-to-nearest is deterministic


def test_two_runs_with_the_same_rounding_state_draw_the_same_numbers():
    from soup_cli.kernels.packed_lora.policy import Rounding
    from soup_cli.kernels.packed_lora.triton import int8

    x = torch.randn(4, 512, device="cuda")
    first, second = Rounding("grad"), Rounding("grad")
    for _ in range(3):
        qa, _ = int8.quant_rows(x, first)
        qb, _ = int8.quant_rows(x, second)
        assert torch.equal(qa, qb)


def test_the_weight_decode_is_dithered_only_for_stochastic_weights():
    import bitsandbytes as bnb

    from soup_cli.kernels.packed_lora.policy import Rounding
    from soup_cli.kernels.packed_lora.triton.nf4 import Pack, launch_int8t

    torch.manual_seed(0)
    shapes = [(256, 256), (128, 256), (128, 256), (256, 256), (512, 256), (512, 256), (256, 512)]
    weights = [bnb.nn.Params4bit((torch.randn(shape) * 0.02).bfloat16(), requires_grad=False,
                                 compress_statistics=True, quant_type="nf4").cuda()
               for shape in shapes]
    pack = Pack(weights)
    (w,) = pack.dequant(4, 5)
    inv = 127.0 / w.abs().amax(0).float().clamp_min(1e-30)

    def decode(rounding):
        out = torch.empty(256, 512, device="cuda", dtype=torch.int8)
        return launch_int8t(pack, 4, inv, out, rounding).clone()

    assert torch.equal(decode(None), decode(Rounding("grad")))  # grad: weights round to nearest
    dither = Rounding("stochastic")
    first, second = decode(dither), decode(dither)  # a new seed for every decode
    assert not torch.equal(first, second) and not torch.equal(first, decode(None))
    assert (first.int() - decode(None).int()).abs().max() <= 1
