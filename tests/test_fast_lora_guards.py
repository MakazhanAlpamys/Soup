"""Fast-LoRA delegation guards, unused-output gradients and forward ownership."""

import copy
from types import SimpleNamespace

import pytest


@pytest.fixture
def torch_peft():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    rng = torch.get_rng_state()
    torch.manual_seed(1165)
    yield torch
    torch.set_rng_state(rng)


def _inject(torch, module, targets):
    from peft import LoraConfig, inject_adapter_in_model

    inject_adapter_in_model(
        LoraConfig(r=3, lora_alpha=7, target_modules=targets, init_lora_weights=False), module
    )
    for name, parameter in module.named_parameters():
        if "lora_" in name:
            parameter.data = parameter.data.float()
    return module


class _Attention:
    @staticmethod
    def build(torch):
        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(16, 16)
                self.k_proj = torch.nn.Linear(16, 8)
                self.v_proj = torch.nn.Linear(16, 8)

            def forward(self, x):
                return self.q_proj(x), self.k_proj(x), self.v_proj(x)

        return _inject(torch, Attention(), ["q_proj", "k_proj", "v_proj"])


class _Mlp:
    @staticmethod
    def build(torch):
        class Mlp(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = torch.nn.Linear(16, 32)
                self.up_proj = torch.nn.Linear(16, 32)
                self.down_proj = torch.nn.Linear(32, 16)
                self.act_fn = torch.nn.SiLU()

            def forward(self, x):
                return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        module = _inject(torch, Mlp(), ["gate_proj", "up_proj", "down_proj"])
        module.config = SimpleNamespace(hidden_act="silu")
        return module


def _grad_fn_names(output):
    nodes, seen, names = [output.grad_fn], set(), []
    while nodes:
        node = nodes.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        names.append(type(node).__name__)
        nodes.extend(child for child, _index in node.next_functions)
    return names


def test_qkv_unused_outputs_leave_adapter_grads_none(torch_peft):
    torch = torch_peft
    from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv

    reference = _Attention.build(torch)
    fast = copy.deepcopy(reference)
    assert patch_fast_lora_qkv(fast) == 1
    value = torch.randn(2, 5, 16)
    results = []
    for model in (reference, fast):
        x = value.clone().requires_grad_()
        q_out, _k_out, v_out = model(x)
        (q_out.sum() + 2 * v_out.sum()).backward()
        results.append((x.grad, {n: p.grad for n, p in model.named_parameters() if "lora_" in n}))
    assert any(name.startswith("_FastLora") for name in _grad_fn_names(q_out))
    k_names = [n for n in results[1][1] if "k_proj" in n]
    assert k_names
    for name in k_names:
        assert results[0][1][name] is None, name
        assert results[1][1][name] is None, name
    for name, gradient in results[0][1].items():
        if gradient is None:
            continue
        torch.testing.assert_close(gradient, results[1][1][name], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(results[0][0], results[1][0], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("path", ["qkv", "mlp"])
def test_mixed_adapter_dtypes_delegate_to_peft(torch_peft, path):
    torch = torch_peft
    from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp
    from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv

    model = (_Attention if path == "qkv" else _Mlp).build(torch)
    # Both halves of one projection go to float64 so PEFT's own forward runs
    # cleanly; its siblings stay float32, which is the mix the guard refuses.
    first = next(n for n, _p in model.named_parameters() if "lora_A" in n)
    prefix = first.split("lora_A")[0]
    for name, parameter in model.named_parameters():
        if name.startswith(prefix) and "lora_" in name:
            parameter.data = parameter.data.double()
    patched = (patch_fast_lora_qkv if path == "qkv" else patch_fast_lora_mlp)(model)
    assert patched == 1
    x = torch.randn(2, 5, 16, requires_grad=True)
    output = model(x)
    output = output[0] if isinstance(output, tuple) else output
    assert not any(name.startswith("_FastLora") for name in _grad_fn_names(output))
    output.sum().backward()
    assert x.grad is not None


@pytest.mark.parametrize("path", ["qkv", "mlp"])
def test_uniform_adapter_dtypes_still_use_the_fused_kernel(torch_peft, path):
    torch = torch_peft
    from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp
    from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv

    model = (_Attention if path == "qkv" else _Mlp).build(torch)
    assert (patch_fast_lora_qkv if path == "qkv" else patch_fast_lora_mlp)(model) == 1
    output = model(torch.randn(2, 5, 16, requires_grad=True))
    output = output[0] if isinstance(output, tuple) else output
    assert any(name.startswith("_FastLora") for name in _grad_fn_names(output))


def test_nf4_autocast_fallback_decision(torch_peft):
    torch = torch_peft
    from soup_cli.utils.fast_lora import _nf4_autocast_requires_fallback

    def state(qmeta, compute_dtype=None):
        return SimpleNamespace(qmeta=qmeta, compute_dtype=compute_dtype)

    quantized = {"_count": 0}
    half = torch.randn(2, 4, dtype=torch.bfloat16)
    full = torch.randn(2, 4)
    dense_states = (state(None),)
    # Outside autocast nothing is modelled, so nothing falls back.
    assert not _nf4_autocast_requires_fallback(full, (state(quantized),))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert not _nf4_autocast_requires_fallback(full, dense_states)
        assert not _nf4_autocast_requires_fallback(half, (state(quantized, torch.bfloat16),))
        assert not _nf4_autocast_requires_fallback(half, (state(quantized, None),))
        # fp32 input over an NF4 base adds cast boundaries the backward lacks.
        assert _nf4_autocast_requires_fallback(full, (state(quantized),))
        # A base compute dtype that differs from the activation dtype does too.
        assert _nf4_autocast_requires_fallback(half, (state(quantized, torch.float32),))


def test_scaled_lora_add_peft_order_rounds_the_scaled_update(torch_peft):
    torch = torch_peft
    from soup_cli.utils.fast_lora import _peft_ordered_lora_add, _scaled_lora_add

    out = torch.randn(64, 32).to(torch.bfloat16)
    hidden = torch.randn(64, 4).to(torch.bfloat16)
    lora_b = torch.randn(32, 4).to(torch.bfloat16)
    scaling = 7 / 3
    folded = _scaled_lora_add(out, hidden, lora_b, scaling)
    peft_order = _peft_ordered_lora_add(out, hidden, lora_b, scaling)
    expected = out + torch.matmul(hidden, lora_b.t()) * scaling
    assert torch.equal(peft_order, expected)
    assert peft_order.dtype == folded.dtype == torch.bfloat16
    # The two orders differ in the last bit for some elements, which is the
    # whole reason the autocast path asks for PEFT's.
    assert not torch.equal(peft_order, folded)


def test_mlp_unpatch_leaves_a_forward_installed_over_the_kernel(torch_peft):
    torch = torch_peft
    from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp, unpatch_fast_lora_mlp

    model = _Mlp.build(torch)
    assert patch_fast_lora_mlp(model) == 1
    sentinel = lambda *args, **kwargs: "replacement"  # noqa: E731
    model.forward = sentinel
    assert unpatch_fast_lora_mlp(model) == 1
    assert model.forward is sentinel


def test_mlp_unpatch_restores_a_class_level_forward(torch_peft):
    torch = torch_peft
    from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp, unpatch_fast_lora_mlp

    model = _Mlp.build(torch)
    assert patch_fast_lora_mlp(model) == 1
    assert unpatch_fast_lora_mlp(model) == 1
    assert "forward" not in vars(model)
