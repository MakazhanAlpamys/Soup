"""Fast-LoRA keeps PEFT's dtype boundary at a 4-bit projection (#792).

Each kernel runs on CPU against a plain-autograd reference of the path peft's
``lora.bnb.Linear4bit`` takes: bitsandbytes' ``Linear4bit.forward`` casts the
input to its compute dtype (fp16 here), multiplies, and returns the result in
the input dtype; the LoRA branch reads the original input in the adapter's
dtype. The reference dequantises with the same ``dequantize_4bit`` call the
kernels make, so both base terms are the same operations on the same tensors.

The input is scaled into fp16's subnormal range (spacing 2**-24). There a
wholesale ``x.to(float16)`` keeps only a few significant bits, so a kernel that
rounds the LoRA input to the compute dtype, or runs the base matmul outside it,
misses the reference by much more than ``_REL``.

The backward is pinned to the same boundary: bitsandbytes' ``MatMul4Bit``
receives the gradient after ``Linear4bit``'s ``.to(inp_dtype)``, so its base
``dX`` GEMM runs in the compute dtype; the LoRA contribution and the saved
activations stay in the input dtype.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

_SCALE = 2.0**-22
# Normalised max error. It absorbs one bf16 rounding in the LoRA add and, for
# the MLP's gate/up grads, the dtype of the forward base GEMM.
_REL = 2e-2
# ``x.grad`` is compared more tightly for fp32 inputs: the base dX GEMM is
# then the only fp16 rounding in the reference, so a kernel that keeps the
# gradient in the input dtype instead of casting it is visible. A bf16
# ``x.grad`` is rounded to bf16 itself, which hides that boundary; its limit
# only checks that the input-dtype results stay in the same ballpark.
_XGRAD_REL = 1e-4
_BF16_XGRAD_REL = 5e-2


def _deps():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    bnb = pytest.importorskip("bitsandbytes")
    return torch, bnb


def _nf4(torch, bnb, in_features, out_features):
    layer = bnb.nn.Linear4bit(
        in_features,
        out_features,
        bias=False,
        compute_dtype=torch.float16,
        quant_type="nf4",
    )
    with torch.no_grad():
        layer.weight.mul_(4)  # keep the base term comparable to the LoRA term
    return layer


def _adapt(torch, bnb, model, targets):
    from peft import LoraConfig, inject_adapter_in_model

    model = model.to("cpu")  # Params4bit quantizes on the move
    inject_adapter_in_model(
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            bias="none",
            target_modules=targets,
        ),
        model,
    )
    with torch.no_grad():
        for _name, param in model.named_parameters():
            if "lora_" in _name:
                # PEFT leaves adapters in their own dtype; keep them fp32 so
                # the LoRA branch has more precision than the fp16 base and a
                # wholesale input cast is observable.
                param.data = param.data.float()
                if "lora_B" in _name:
                    torch.nn.init.normal_(param, std=1.0)
    quantized = [
        module for module in model.modules() if isinstance(module, bnb.nn.Linear4bit)
    ]
    assert len(quantized) == len(targets)
    assert all(module.weight.quant_state is not None for module in quantized)
    return model


def _reference(torch, bnb, layer, x):
    """PEFT's ``lora.bnb.Linear4bit`` forward in plain autograd ops."""
    functional = torch.nn.functional
    base = layer.get_base_layer()
    dense = bnb.functional.dequantize_4bit(base.weight, base.weight.quant_state)
    out = functional.linear(
        x.to(base.compute_dtype), dense.to(base.compute_dtype)
    ).to(x.dtype)
    adapter = layer.active_adapters[0]
    lora_a = layer.lora_A[adapter].weight
    lora_b = layer.lora_B[adapter].weight
    term = functional.linear(
        functional.linear(x.to(lora_a.dtype), lora_a), lora_b
    )
    return out + (term * float(layer.scaling[adapter])).to(out.dtype)


def _rel(got, want):
    want = want.detach().double()
    scale = want.abs().max().item()
    assert scale > 0, "all-zero reference: the comparison would prove nothing"
    return (got.detach().double() - want).abs().max().item() / scale


def _probe_loss(outputs, probes):
    return sum((out.double() * probe).sum() for out, probe in zip(outputs, probes))


def _assert_adapter_grads_match(model, reference):
    got = {
        name: param.grad
        for name, param in model.named_parameters()
        if "lora_" in name
    }
    want = {
        name: param.grad
        for name, param in reference.named_parameters()
        if "lora_" in name
    }
    assert got and got.keys() == want.keys()
    for name, grad in got.items():
        assert grad is not None and want[name] is not None, name
        assert _rel(grad, want[name]) <= _REL, name


def _assert_input_grad_matches(x, reference):
    import torch

    assert x.grad is not None and reference.grad is not None
    limit = _XGRAD_REL if x.grad.dtype == torch.float32 else _BF16_XGRAD_REL
    assert _rel(x.grad, reference.grad) <= limit


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_single_projection_keeps_pefts_dtype_boundary(
    dtype_name, aten_half_matmuls
):
    torch, bnb = _deps()
    import torch.nn as nn

    from soup_cli.utils.fast_lora import patch_fast_lora_single_projection

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.o_proj = _nf4(torch, bnb, 16, 16)

    torch.manual_seed(0)
    model = _adapt(torch, bnb, Block(), ["o_proj"])
    reference = copy.deepcopy(model)
    dtype = getattr(torch, dtype_name)
    x0 = (torch.randn(3, 5, 16) * _SCALE).to(dtype)
    probe = torch.randn(3, 5, 16, dtype=torch.float64)

    x_ref = x0.clone().requires_grad_(True)
    want = _reference(torch, bnb, reference.o_proj, x_ref)
    _probe_loss([want], [probe]).backward()

    assert patch_fast_lora_single_projection(model) == 1
    x = x0.clone().requires_grad_(True)
    got = model.o_proj(x)
    assert type(got.grad_fn).__name__ == "_FastLoraSingleProjectionBackward"
    assert got.dtype == dtype
    _probe_loss([got], [probe]).backward()

    assert _rel(got, want) <= _REL
    _assert_adapter_grads_match(model, reference)
    _assert_input_grad_matches(x, x_ref)


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_swiglu_mlp_keeps_pefts_dtype_boundary(dtype_name, aten_half_matmuls):
    torch, bnb = _deps()
    import torch.nn as nn

    from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = _nf4(torch, bnb, 16, 32)
            self.up_proj = _nf4(torch, bnb, 16, 32)
            self.down_proj = _nf4(torch, bnb, 32, 16)
            self.act_fn = nn.SiLU()

        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_act="silu")
            self.mlp = MLP()

        def forward(self, x):
            return self.mlp(x)

    torch.manual_seed(0)
    model = _adapt(
        torch, bnb, Model(), ["gate_proj", "up_proj", "down_proj"]
    )
    reference = copy.deepcopy(model)
    dtype = getattr(torch, dtype_name)
    x0 = (torch.randn(3, 5, 16) * _SCALE).to(dtype)
    probe = torch.randn(3, 5, 16, dtype=torch.float64)

    x_ref = x0.clone().requires_grad_(True)
    mlp = reference.mlp
    gate = _reference(torch, bnb, mlp.gate_proj, x_ref)
    up = _reference(torch, bnb, mlp.up_proj, x_ref)
    want = _reference(torch, bnb, mlp.down_proj, torch.nn.functional.silu(gate) * up)
    _probe_loss([want], [probe]).backward()

    assert patch_fast_lora_mlp(model) == 1
    x = x0.clone().requires_grad_(True)
    got = model(x)
    assert type(got.grad_fn).__name__ == "_FastLoraSwiGLUBackward"
    assert got.dtype == dtype
    _probe_loss([got], [probe]).backward()

    assert _rel(got, want) <= _REL
    _assert_adapter_grads_match(model, reference)
    _assert_input_grad_matches(x, x_ref)


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_shared_x_qkv_keeps_pefts_dtype_boundary(dtype_name, aten_half_matmuls):
    torch, bnb = _deps()
    import torch.nn as nn

    from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = _nf4(torch, bnb, 16, 16)
            self.k_proj = _nf4(torch, bnb, 16, 8)
            self.v_proj = _nf4(torch, bnb, 16, 8)

        def forward(self, x):
            return self.q_proj(x), self.k_proj(x), self.v_proj(x)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attention()

        def forward(self, x):
            return self.attn(x)

    torch.manual_seed(0)
    model = _adapt(torch, bnb, Model(), ["q_proj", "k_proj", "v_proj"])
    reference = copy.deepcopy(model)
    dtype = getattr(torch, dtype_name)
    x0 = (torch.randn(3, 5, 16) * _SCALE).to(dtype)
    probes = [
        torch.randn(3, 5, width, dtype=torch.float64) for width in (16, 8, 8)
    ]

    x_ref = x0.clone().requires_grad_(True)
    attn = reference.attn
    want = [
        _reference(torch, bnb, getattr(attn, name), x_ref)
        for name in ("q_proj", "k_proj", "v_proj")
    ]
    _probe_loss(want, probes).backward()

    assert patch_fast_lora_qkv(model) == 1
    x = x0.clone().requires_grad_(True)
    got = model(x)
    assert all(part.dtype == dtype for part in got)
    _probe_loss(got, probes).backward()
    hits, grad_fns = getattr(model.attn, "_soup_fast_lora_qkv_last_cache_hits")
    assert hits == 2
    assert grad_fns == ("_FastLoraQKVBackward",) * 3

    for part, wanted in zip(got, want):
        assert _rel(part, wanted) <= _REL
    _assert_adapter_grads_match(model, reference)
    _assert_input_grad_matches(x, x_ref)
