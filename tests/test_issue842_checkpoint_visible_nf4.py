"""Checkpoint-visible NF4 fused-forward groundwork (#842, folding #841)."""

from __future__ import annotations

import os
import sys

import pytest

from soup_cli.utils.layer_stream_runtime import checkpoint_visible_nf4_linear

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
bnb_functional = pytest.importorskip("bitsandbytes.functional")


@pytest.mark.parametrize("nested", [False, True])
def test_fused_forward_and_input_gradient_match_dequant_linear(nested):
    torch.manual_seed(17)
    weight = torch.randn(16, 16)
    packed, state = bnb_functional.quantize_4bit(
        weight, quant_type="nf4", compress_statistics=nested
    )
    x = torch.randn(3, 16, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)

    actual = checkpoint_visible_nf4_linear(x, packed, state)
    dense = bnb_functional.dequantize_4bit(packed, state).to(reference_x.dtype)
    expected = torch.nn.functional.linear(reference_x, dense)

    assert torch.equal(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    assert torch.equal(x.grad, reference_x.grad)


def _non_nested_state(absmax, template):
    return bnb_functional.QuantState(
        absmax=absmax,
        shape=template.shape,
        dtype=template.dtype,
        blocksize=template.blocksize,
        quant_type=template.quant_type,
    )


def test_save_for_backward_makes_pool_recycling_recompute_and_preserve_gradients():
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(23)
    weight0 = torch.randn(16, 16)
    weight1 = torch.randn(16, 16)
    packed0, state0 = bnb_functional.quantize_4bit(weight0, quant_type="nf4")
    packed1, state1 = bnb_functional.quantize_4bit(weight1, quant_type="nf4")

    shared_packed = packed0.clone()
    shared_absmax = state0.absmax.clone()
    shared_state = _non_nested_state(shared_absmax, state0)
    calls = {"n": 0}

    def streamed_body(value):
        calls["n"] += 1

        with torch.no_grad():
            shared_packed.copy_(packed0)
            shared_absmax.copy_(state0.absmax)
        return checkpoint_visible_nf4_linear(value, shared_packed, shared_state)

    x = torch.randn(3, 16, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)
    output = checkpoint(streamed_body, x, use_reentrant=False)

    # Recycle the pool slot to another layer before backward.
    with torch.no_grad():
        shared_packed.copy_(packed1)
        shared_absmax.copy_(state1.absmax)

    output.square().sum().backward()
    dense0 = bnb_functional.dequantize_4bit(packed0, state0).to(reference_x.dtype)
    reference = torch.nn.functional.linear(reference_x, dense0)
    reference.square().sum().backward()

    assert calls["n"] == 2, "checkpoint did not recompute the custom NF4 op"
    assert torch.equal(x.grad, reference_x.grad)


def test_plain_ctx_control_is_detectably_wrong_after_pool_recycling():
    """Negative control: reproduces the #331 lifetime bug on a tiny CPU fixture."""
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(29)
    weight0 = torch.randn(16, 16)
    weight1 = torch.randn(16, 16)

    packed0, state0 = bnb_functional.quantize_4bit(weight0, quant_type="nf4")
    packed1, state1 = bnb_functional.quantize_4bit(weight1, quant_type="nf4")
    shared_packed = packed0.clone()
    shared_absmax = state0.absmax.clone()
    shared_state = _non_nested_state(shared_absmax, state0)

    class PlainCtx4Bit(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, packed):
            ctx.packed = packed
            ctx.state = shared_state
            return torch.ops.bitsandbytes.gemm_4bit.default(
                value,
                packed,
                shared_state.shape,
                shared_state.absmax,
                shared_state.blocksize,
                shared_state.quant_type,
            )

        @staticmethod
        def backward(ctx, grad_output):
            dense = bnb_functional.dequantize_4bit(ctx.packed, ctx.state).to(
                grad_output.dtype
            )
            return torch.matmul(grad_output, dense), None

    calls = {"n": 0}

    def broken_body(value):
        calls["n"] += 1
        with torch.no_grad():
            shared_packed.copy_(packed0)
            shared_absmax.copy_(state0.absmax)
        return PlainCtx4Bit.apply(value, shared_packed)

    x = torch.randn(3, 16, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)
    output = checkpoint(broken_body, x, use_reentrant=False)
    with torch.no_grad():
        shared_packed.copy_(packed1)
        shared_absmax.copy_(state1.absmax)
    output.square().sum().backward()

    dense0 = bnb_functional.dequantize_4bit(packed0, state0).to(reference_x.dtype)
    reference = torch.nn.functional.linear(reference_x, dense0)
    reference.square().sum().backward()

    assert calls["n"] == 1, "negative control unexpectedly became checkpoint-visible"
    assert not torch.equal(x.grad, reference_x.grad)
    assert (x.grad - reference_x.grad).abs().max().item() > 1e-3


def test_production_capability_gate_keeps_cpu_on_the_old_path():
    from soup_cli.utils.layer_stream_runtime import _can_use_checkpoint_visible_nf4_gemm

    packed, state = bnb_functional.quantize_4bit(torch.randn(16, 16), quant_type="nf4")
    assert packed.device.type == "cpu"
    assert _can_use_checkpoint_visible_nf4_gemm(torch.randn(2, 16), state) is False


def test_bitsandbytes_private_dispatch_contract_is_visible_to_cpu_ci():
    try:
        from bitsandbytes.backends.cuda import ops as bnb_cuda_ops
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"bitsandbytes CUDA ops cannot import with this torch build: {exc}")

    assert hasattr(bnb_cuda_ops, "_gemm_4bit_use_custom_fn")
    assert hasattr(bnb_cuda_ops, "_gemm_4bit_custom_max_m")


@pytest.mark.parametrize(
    ("m", "k", "dispatch", "expected"),
    [
        (3, 64, True, True),
        (3, 64, False, False),
        (97, 64, True, False),
        (3, 65, True, False),
    ],
    ids=["fused", "bnb-dequant", "above-m-cap", "misaligned-k"],
)
def test_cuda_route_follows_bitsandbytes_shape_decision_on_cpu(
    monkeypatch, m, k, dispatch, expected
):
    from types import SimpleNamespace

    import bitsandbytes.backends.cuda as bnb_cuda_package

    from soup_cli.utils.layer_stream_runtime import _can_use_checkpoint_visible_nf4_gemm

    calls = []

    def use_custom_fn(device_index, dtype, rows, out_features, in_features):
        calls.append((device_index, dtype, rows, out_features, in_features))
        return dispatch

    fake_ops = SimpleNamespace(
        _gemm_4bit_custom_max_m=96,
        _gemm_4bit_use_custom_fn=use_custom_fn,
    )
    monkeypatch.setitem(sys.modules, "bitsandbytes.backends.cuda.ops", fake_ops)
    monkeypatch.setattr(bnb_cuda_package, "ops", fake_ops, raising=False)
    x = SimpleNamespace(
        device=SimpleNamespace(type="cuda", index=0),
        dtype=torch.bfloat16,
        shape=(m, k),
        numel=lambda: m * k,
    )
    state = SimpleNamespace(
        nested=False,
        shape=(128, k),
        blocksize=64,
    )

    assert _can_use_checkpoint_visible_nf4_gemm(x, state) is expected
    if m <= 96 and k % 64 == 0:
        assert calls == [(0, torch.bfloat16, m, 128, k)]
    else:
        assert calls == []


def test_backward_refuses_a_recycled_stream_slot():
    torch.manual_seed(31)
    packed, state = bnb_functional.quantize_4bit(
        torch.randn(16, 16), quant_type="nf4"
    )
    owner = {"current": 0}

    def assert_owner():
        if owner["current"] != 0:
            raise RuntimeError("recycled weight slot")

    packed._soup_stream_owner_check = assert_owner
    x = torch.randn(3, 16, requires_grad=True)
    output = checkpoint_visible_nf4_linear(x, packed, state)
    owner["current"] = 1
    with pytest.raises(RuntimeError, match="recycled weight slot"):
        output.square().sum().backward()


def test_streamed_layer_caches_substitution_views_for_one_pool_mapping(
    tmp_path, monkeypatch
):
    from test_v07202 import _nf4_stream

    import soup_cli.utils.layer_stream_runtime as runtime_module

    model, runtime, _, _, _ = _nf4_stream(tmp_path)
    layer = runtime_module.decoder_owner(model).layers[0]
    buffers = runtime.pool.buffers[runtime.pool.slot_for(layer.idx)]

    # Start from a known-empty cache: model construction must not be what makes
    # this test pass.
    layer._cached_substitution_buffers = None
    layer._cached_substitution_weights = None

    real_rebuild = runtime_module.rebuild_params4bit
    calls = {"n": 0}

    def counting_rebuild(*args, **kwargs):
        calls["n"] += 1
        return real_rebuild(*args, **kwargs)

    monkeypatch.setattr(runtime_module, "rebuild_params4bit", counting_rebuild)

    quantized = sum(
        ckpt in layer.quant_specs for ckpt in layer.name_map.values()
    )
    assert quantized > 0

    first = layer._substituted_weights(buffers)
    second = layer._substituted_weights(buffers)
    assert second is first
    assert calls["n"] == quantized

    # A different mapping object represents a different pool view and must not
    # inherit the cache, even when it currently points at the same tensors.
    other_mapping = dict(buffers)
    third = layer._substituted_weights(other_mapping)
    assert third is not first
    assert calls["n"] == 2 * quantized


@pytest.mark.gpu
def test_missing_bnb_private_dispatch_symbols_fail_toward_the_safe_function(
    monkeypatch,
):
    from bitsandbytes.backends.cuda import ops as bnb_cuda_ops

    from soup_cli.utils.layer_stream_runtime import (
        _can_use_checkpoint_visible_nf4_gemm,
    )

    weight = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    _packed, state = bnb_functional.quantize_4bit(weight, quant_type="nf4")
    x = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16)
    monkeypatch.delattr(bnb_cuda_ops, "_gemm_4bit_use_custom_fn")
    assert _can_use_checkpoint_visible_nf4_gemm(x, state) is True


@pytest.mark.gpu
def test_fused_cuda_recycling_matches_private_buffer_and_plain_ctx_fails():
    from torch.utils.checkpoint import checkpoint

    from soup_cli.utils.layer_stream_runtime import (
        _can_use_checkpoint_visible_nf4_gemm,
    )

    torch.manual_seed(37)
    weight0 = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    weight1 = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    packed0, state0 = bnb_functional.quantize_4bit(weight0, quant_type="nf4")
    packed1, state1 = bnb_functional.quantize_4bit(weight1, quant_type="nf4")
    shared_packed = packed0.clone()
    shared_absmax = state0.absmax.clone()
    shared_state = _non_nested_state(shared_absmax, state0)
    x = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    assert _can_use_checkpoint_visible_nf4_gemm(x, shared_state) is True

    calls = {"visible": 0, "plain": 0}

    def visible_body(value):
        calls["visible"] += 1
        with torch.no_grad():
            shared_packed.copy_(packed0)
            shared_absmax.copy_(state0.absmax)
        return checkpoint_visible_nf4_linear(value, shared_packed, shared_state)

    output = checkpoint(visible_body, x, use_reentrant=False)
    with torch.no_grad():
        shared_packed.copy_(packed1)
        shared_absmax.copy_(state1.absmax)
    output.float().square().sum().backward()

    private_x = x.detach().clone().requires_grad_(True)
    private = checkpoint_visible_nf4_linear(private_x, packed0, state0)
    private.float().square().sum().backward()
    assert calls["visible"] == 2
    assert torch.equal(x.grad, private_x.grad)

    class PlainCtx4Bit(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, packed):
            ctx.packed = packed
            ctx.state = shared_state
            return torch.ops.bitsandbytes.gemm_4bit.default(
                value,
                packed,
                shared_state.shape,
                shared_state.absmax,
                shared_state.blocksize,
                shared_state.quant_type,
            )

        @staticmethod
        def backward(ctx, grad_output):
            dense = bnb_functional.dequantize_4bit(ctx.packed, ctx.state).to(
                grad_output.dtype
            )
            return torch.matmul(grad_output, dense), None

    def plain_body(value):
        calls["plain"] += 1
        with torch.no_grad():
            shared_packed.copy_(packed0)
            shared_absmax.copy_(state0.absmax)
        return PlainCtx4Bit.apply(value, shared_packed)

    plain_x = x.detach().clone().requires_grad_(True)
    plain = checkpoint(plain_body, plain_x, use_reentrant=False)
    with torch.no_grad():
        shared_packed.copy_(packed1)
        shared_absmax.copy_(state1.absmax)
    plain.float().square().sum().backward()
    assert calls["plain"] == 1
    assert not torch.equal(plain_x.grad, private_x.grad)


@pytest.mark.gpu
@pytest.mark.parametrize("seq_len", [8, 128], ids=["fused-m", "training-m"])
def test_streamed_matches_unpatched_resident_logits_and_all_lora_grads(
    tmp_path, monkeypatch, seq_len
):
    import bitsandbytes as bnb
    from test_v07202 import (
        _nf4_stream,
        _randomise_lora_b,
        _resident_nf4,
        _sync_adapters,
    )

    import soup_cli.utils.layer_stream_runtime as runtime_module

    streamed, _runtime, weights, _index, _shards = _nf4_stream(
        tmp_path, device="cuda", dtype="bfloat16"
    )
    resident = _resident_nf4(weights, dtype="bfloat16", device=0)
    _randomise_lora_b(resident)
    assert _sync_adapters(streamed, resident) > 0

    calls = {"function": 0, "streamed_matmul": 0, "resident_matmul": 0}
    phase = {"name": "streamed"}
    real_function = runtime_module.checkpoint_visible_nf4_linear
    real_matmul = bnb.matmul_4bit
    real_gate = runtime_module._can_use_checkpoint_visible_nf4_gemm
    gate_decisions = []

    def counting_function(*args, **kwargs):
        calls["function"] += 1
        return real_function(*args, **kwargs)

    def counting_matmul(*args, **kwargs):
        calls[phase["name"] + "_matmul"] += 1
        return real_matmul(*args, **kwargs)

    def recording_gate(*args, **kwargs):
        decision = real_gate(*args, **kwargs)
        gate_decisions.append(decision)
        return decision

    monkeypatch.setattr(
        runtime_module, "checkpoint_visible_nf4_linear", counting_function
    )
    monkeypatch.setattr(
        runtime_module, "_can_use_checkpoint_visible_nf4_gemm", recording_gate
    )
    monkeypatch.setattr(bnb, "matmul_4bit", counting_matmul)

    generator = torch.Generator(device="cuda").manual_seed(41)
    input_ids = torch.randint(
        0, 64, (1, seq_len), device="cuda", generator=generator
    )
    streamed_logits = streamed(input_ids=input_ids, use_cache=False).logits
    streamed_logits.float().square().mean().backward()
    phase["name"] = "resident"
    resident_logits = resident(input_ids=input_ids, use_cache=False).logits
    resident_logits.float().square().mean().backward()

    assert calls["streamed_matmul"] == 0
    assert calls["resident_matmul"] > 0
    assert gate_decisions
    assert (calls["function"] > 0) is any(gate_decisions)
    assert torch.equal(streamed_logits, resident_logits)

    def adapter_grads(model):
        return {
            name.replace(".inner.", "."): parameter.grad
            for name, parameter in model.named_parameters()
            if "lora_" in name and parameter.grad is not None
        }

    streamed_grads = adapter_grads(streamed)
    resident_grads = adapter_grads(resident)
    assert streamed_grads and streamed_grads.keys() == resident_grads.keys()
    for name, grad in streamed_grads.items():
        assert torch.equal(grad, resident_grads[name]), name
