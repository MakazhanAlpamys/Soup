# SPDX-License-Identifier: Apache-2.0
#
# Origin: Soup Kernel Challenge (Decentrathon 6.0), the submission of team Luck
# (BAITC-Hacks/hack-e79529fc-luck@b3f1225f); the int8 gate/up packs follow ideas seen in the
# solutions of teams Null Pointers, 21, DarkDaur and OrbiTeam.
# Modified for Soup: the verdict is recorded on the run's ``policy.Int8Switch`` (not on module
# globals), and the bar comes from the run's ``policy.Rounding``.
"""The gradient check behind ``int8_backward: auto``.

The backward's three MLP input-gradient GEMMs (g @ W_down, d_gate @ W_gate, d_up @ W_up; 44 % of a
layer's GEMM FLOPs) can run in int8 (``layers._act_grad_int8``, ``layers._mlp_input_grad_int8``,
``triton/int8.py``): on sm_80+ ``torch._int_mm`` is ~3x the bf16 GEMM rate. The int8 path is lossy
(a per-row scale for the gradient, a per-column scale for the decoded weight), so "auto" measures
it once, on the model's own first layer, at the model's own shapes, before the first step:

  * the real decoded weights of layer 0 (gate, up, down), ``ROWS`` synthetic tokens: a hidden state
    and an output gradient that are Gaussian with log-normal row scales (sigma 0.5) and 0.05 %
    outliers of 6 sigma (real activations are not available before step 1; this is a sanity gate
    on the int8 path, the end-to-end effect is measured on real runs);
  * the compute-dtype path: g_act = g @ W_down (compute dtype), the SwiGLU backward in fp32 rounded
    to the compute dtype, d_x = d_gate @ W_gate + d_up @ W_up with fp32 accumulation (the code of
    the non-int8 branch of ``layers.DecoderLayer.backward``);
  * the int8 path: the exact functions the training backward calls (stash packs of gate and up
    included), with the run's rounding;
  * COSINE between the two d_x: at least ``rounding.cosine_min`` (0.999 with round-to-nearest, 0.998
    with stochastic rounding, which is unbiased but noisier per step) keeps the int8 path on, else
    it is switched off for the run and the layers stash gate / up in the compute dtype.

The bar is a sanity gate, not a proof for every data distribution: with 0.2 % spikes of 10 sigma
the 7B-shape random layer measured 0.99883 and went off. Measured: round-to-nearest 0.99949
(Qwen2.5-1.5B) and 0.99932 (7B), both bf16 on an RTX 5070; stochastic rounding 0.99909 (1.5B,
bf16) and, on the Kaggle T4 in fp16, 0.99881 against the 0.998 bar, a small margin: do not
raise the bar without data.

It draws from its own generator (no global random state is touched), holds the three decoded
weights of one layer (0.4 GB at 7B) for a moment and takes ~0.2 s, once per run.
"""

import torch

ROWS = 256  # tokens of the check: a few tens of MB of tensors next to the decoded weights
ROW_SIGMA = 0.5  # log-normal spread of the rows' scales (tokens differ in norm by e^+-0.5 sigma)
SPIKES = 0.0005  # fraction of entries that are 6-sigma outliers


def _heavy(rows, cols, scale, gen, dev, spikes=None):
    spikes = SPIKES if spikes is None else spikes
    base = torch.randn(rows, cols, device=dev, generator=gen)
    row_scale = torch.exp(ROW_SIGMA * torch.randn(rows, 1, device=dev, generator=gen))
    out = ((torch.rand(rows, cols, device=dev, generator=gen) < spikes).float()
           * torch.randn(rows, cols, device=dev, generator=gen) * 6.0)
    return (base + out) * row_scale * scale


def _cosine(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


@torch.no_grad()
def measure(layer, dtype, rounding=None, rows=ROWS, seed=1234,
            spikes=None):  # pragma: no cover - needs Triton and CUDA (gpu tests)
    """(cosine, relative error) of the int8 d_x against the compute-dtype d_x, for this layer
    (``model.Layer``: ``.pack``, ``.int8_scales``)."""
    from .layers import _act_grad_int8, _mlp_input_grad_int8
    from .triton.q8 import quant_q8, swiglu_bwd_q8

    pack = layer.pack
    dev = pack.scale2.device
    hidden = pack.shapes[4][1]
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    x = _heavy(rows, hidden, 1.0, gen, dev, spikes).to(dtype)
    g = _heavy(rows, hidden, 1e-2, gen, dev, spikes).to(dtype)
    (w_gate,) = pack.dequant(4, 5)
    gate = x @ w_gate.t()
    (w_up,) = pack.dequant(5, 6)
    up = x @ w_up.t()
    (w_down,) = pack.dequant(6, 7)

    def swiglu_bwd(g_act, gate_, up_):
        gate32, up32, grad = gate_.float(), up_.float(), g_act.float()
        sig = torch.sigmoid(gate32)
        return (grad * up32 * sig * (1.0 + gate32 * (1.0 - sig))), grad * gate32 * sig

    # compute-dtype path
    d_gate, d_up = swiglu_bwd(g @ w_down, gate, up)
    ref = torch.mm(d_gate.to(dtype), w_gate, out_dtype=torch.float32)
    ref += torch.mm(d_up.to(dtype), w_up, out_dtype=torch.float32)
    # int8 path, as the training backward runs it
    acc, sg = _act_grad_int8(layer, g, rounding)
    gate_pack, up_pack = quant_q8(gate), quant_q8(up)
    dg8, sdg, du8, sdu = swiglu_bwd_q8(acc, sg, layer.int8_scales[6][1], gate_pack, up_pack,
                                       rounding=rounding)
    out = _mlp_input_grad_int8(layer, dg8, sdg, du8, sdu, rounding)
    cosine = _cosine(out, ref)
    rel = float((out - ref).norm() / ref.norm().clamp_min(1e-30))
    del acc, gate_pack, up_pack, dg8, du8, out, ref, gate, up, w_gate, w_up, w_down
    return cosine, rel


def decide(packed, switch, rounding, log=None):
    """Run the check on the first layer of ``packed`` (``model.prepare``'s result) when the run's
    switch is "auto" and nothing has been decided yet; record the verdict on ``switch``. Returns
    whether int8 is on."""
    from .layers import uses_int8

    if not switch.needs_check:
        return switch.enabled()
    layer = packed.layers[0]
    tokens = 4096
    if not uses_int8.supported(layer, tokens):
        switch.decide(False)
        return False
    try:
        cosine, rel = measure(layer, packed.dtype, rounding)
    except Exception as exc:  # noqa: BLE001 - no check, no int8: the safe side
        switch.decide(False)
        if log:
            log(f"int8_backward=auto -> off (the gradient check failed: {exc!r})")
        return False
    threshold = rounding.cosine_min
    passed = switch.decide(cosine >= threshold, cosine, rel)
    if log:
        log(f"int8_backward=auto -> {'on' if passed else 'off'} (MLP d_x, int8 vs "
            f"{str(packed.dtype)[6:]} path on layer 0: cosine {cosine:.6f}, rel err {rel:.4f}; "
            f"threshold {threshold}{', stochastic rounding' if rounding.grad else ''})")
    return passed
