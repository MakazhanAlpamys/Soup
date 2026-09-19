"""#839: the fast-LoRA single-projection kernel and its own tests (tracker #792).

``utils/fast_lora.py`` replaces peft's generic autograd for one projection at a
time: ``Y = X @ W^T + b + s * (X @ A^T) @ B^T``, with a hand-written backward
that saves only what the backward reads. Nothing here claims a speedup; the
tracker measures before any number is published, and the tests below assert
correctness only.

Asserted on CPU (no CUDA needed):

- ``torch.autograd.gradcheck`` in float64, with and without a base bias.
- fp32 forward, input-grad and adapter-grad parity against the unpatched peft
  ``lora.Linear`` at ``assert_close`` defaults. Measured on the writing box,
  all four comparisons came out bit-exact; that readout lives in the PR rather
  than in an assertion, so a platform where the op order rounds differently
  still passes while broken math does not.
- Delegation to peft's own forward: disabled adapters, merged adapters, a DoRA
  variant, non-zero dropout, and the ``adapter_names`` kwarg.
- The patched forward reads the base weight at call time (the streaming
  contract); patching is counted, idempotent and reversible.
- A 4-layer tiny Llama with 2 stream buffers: the patched streamed model's loss
  and all 16 adapter grads match the resident twin (#331 shape).
- Saved-for-backward bytes are not worse than peft's graph at the unit shapes
  (measured equal; the assertion is ``<=``).

Marked ``gpu`` (the protocol for a CUDA run; skipped in CI):

- NF4 forward/backward parity against unpatched ``lora.bnb.Linear4bit``. The
  tolerance is loose on purpose: the fused-vs-dequant divergence #968 pins is
  part of whatever delta is measured there, and the honest magnitude needs a
  card.
- A micro-benchmark at the Llama-3.1-8B ``o_proj`` shape. Report, do not
  assert.

No CUDA device was available while writing this, so the gpu-marked tests have
not executed yet; the PR says so. Everything CPU above ran green on macOS with
torch 2.14.0 / peft 0.21.0 / Python 3.12 (the versions CI installs).
"""

from __future__ import annotations

import copy

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _requires_train_extra():
    for mod in ("torch", "peft"):
        pytest.importorskip(mod, reason=f"{mod} is only in the [train] extra")


def _spy(real_forward, calls, *args, **kwargs):
    """A stand-in forward that records that it was called, then behaves normally."""
    calls.append("delegated")
    return real_forward(*args, **kwargs)


def _randomise_lora_b(module):
    """Give ``lora_B`` a non-zero weight so the LoRA term is actually exercised.

    peft initialises ``lora_B`` to zeros, which makes ``h @ B^T`` vanish: every
    comparison then holds trivially, ``dA`` is compared zero against zero, and a
    kernel that drops the dropout, ``disable_adapters``, ``merged`` or DoRA guard
    still passes. Randomising B is what makes those tests mean anything.
    """
    import torch

    with torch.no_grad():
        for _name, param in module.named_parameters():
            if "lora_B" in _name:
                torch.nn.init.normal_(param, std=0.5)
    return module


def _make_adapted_linear(in_f=8, out_f=6, r=2, alpha=4, bias=True, dropout=0.0, dora=False):
    """A minimal peft-adapted module: one ``nn.Linear`` named ``linear``."""
    import torch.nn as nn
    from peft import LoraConfig, inject_adapter_in_model

    class _Wrapper(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.linear = layer

    model = _Wrapper(nn.Linear(in_f, out_f, bias=bias))
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=["linear"],
        use_dora=dora,
    )
    inject_adapter_in_model(config, model)
    _randomise_lora_b(model)
    return model, model.linear


def _saved_bytes(module, x):
    """Bytes held for backward during one forward+backward, deduped by storage."""
    import torch

    seen: dict[int, int] = {}

    def pack(tensor):
        storage = tensor.untyped_storage()
        seen[storage.data_ptr()] = storage.nbytes()
        return tensor

    def unpack(tensor):
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        out = module(x)
        out.sum().backward()
    return sum(seen.values())


class TestGradcheckFloat64:
    """#839: gradcheck in float64 on CPU, with and without a base bias."""

    @pytest.mark.parametrize("use_bias", [False, True])
    def test_gradcheck_wrt_x_a_b(self, use_bias):
        _requires_train_extra()
        import torch

        from soup_cli.utils.fast_lora import _single_projection_function

        fn = _single_projection_function()
        torch.manual_seed(0)
        n, in_f, out_f, r = 5, 8, 6, 2
        w = torch.randn(out_f, in_f, dtype=torch.float64)
        b = torch.randn(out_f, dtype=torch.float64) if use_bias else None
        x = torch.randn(n, in_f, dtype=torch.float64, requires_grad=True)
        a = torch.randn(r, in_f, dtype=torch.float64, requires_grad=True)
        bb = torch.randn(out_f, r, dtype=torch.float64, requires_grad=True)

        def f(x_, a_, b_):
            return fn.apply(x_, w, b, a_, b_, 0.5, None)

        assert torch.autograd.gradcheck(f, (x, a, bb))

    def test_double_backward_is_refused_rather_than_silently_wrong(self):
        """#792 marks ``backward`` once-differentiable, and the refusal must be real.

        Not ``gradgradcheck``: that numerically differentiates and compares
        Jacobians, and it reports a mismatch for this Function with or without
        the decorator (measured on torch 2.14 both ways), so it cannot tell the
        two apart. What a caller can actually observe is that a second
        differentiation of the same graph raises instead of returning a number,
        which is the behaviour ``@once_differentiable`` provides.
        """
        _requires_train_extra()
        import torch

        from soup_cli.utils.fast_lora import _single_projection_function

        fn = _single_projection_function()
        torch.manual_seed(0)
        n, in_f, out_f, r = 4, 6, 5, 2
        w = torch.randn(out_f, in_f, dtype=torch.float64)
        x = torch.randn(n, in_f, dtype=torch.float64, requires_grad=True)
        a = torch.randn(r, in_f, dtype=torch.float64, requires_grad=True)
        bb = torch.randn(out_f, r, dtype=torch.float64, requires_grad=True)

        out = fn.apply(x, w, None, a, bb, 0.5, None)
        first = torch.autograd.grad(out.sum(), (x, a, bb), create_graph=True)

        with pytest.raises(RuntimeError) as caught:
            torch.autograd.grad(first[0].sum(), x)
        # The refusal, rather than a silently wrong second derivative. torch
        # reports it as the backward outputs carrying no graph, which is what
        # ``once_differentiable`` does by running backward under no_grad.
        assert "does not require grad" in str(caught.value) or "once_differentiable" in str(
            caught.value
        ), f"unexpected refusal: {caught.value}"

    def test_x_that_needs_no_grad_still_gets_correct_parameter_grads(self):
        """``needs_input_grad[0]`` false is the layer-0 case, and the measured 2x.

        The dX GEMM is skipped there; dA and dB must still be right, because the
        adapter is what is being trained.
        """
        _requires_train_extra()
        import torch

        from soup_cli.utils.fast_lora import patch_fast_lora_single_projection

        torch.manual_seed(0)
        model, layer = _make_adapted_linear()
        x = torch.randn(5, 8)  # no requires_grad: this is the skip path

        before = layer(x)
        before.sum().backward()
        expected = {
            name: p.grad.detach().clone()
            for name, p in layer.named_parameters()
            if p.grad is not None
        }
        assert expected, "fixture produced no gradients at all"
        for p in layer.parameters():
            p.grad = None

        assert patch_fast_lora_single_projection(model) == 1
        after = layer(x)
        torch.testing.assert_close(after, before)
        after.sum().backward()

        for name, p in layer.named_parameters():
            if p.grad is None or name not in expected:
                continue
            torch.testing.assert_close(p.grad, expected[name])


class TestPeftParity:
    """#839: forward and backward parity against the unpatched peft forward."""

    @pytest.mark.parametrize("bias", [True, False])
    def test_fp32_forward_and_backward_match_unpatched_peft(self, bias):
        _requires_train_extra()
        import torch

        from soup_cli.utils.fast_lora import patch_fast_lora_single_projection

        torch.manual_seed(0)
        model, layer = _make_adapted_linear(bias=bias)
        x = torch.randn(5, 8, requires_grad=True)

        ref = layer(x)
        ref.sum().backward()
        ref_x = x.grad.detach().clone()
        ref_p = {
            name: p.grad.detach().clone()
            for name, p in layer.named_parameters()
            if p.grad is not None
        }

        x.grad = None
        for p in layer.parameters():
            p.grad = None

        assert patch_fast_lora_single_projection(model) == 1
        out = layer(x)
        out.sum().backward()

        torch.testing.assert_close(out, ref)
        torch.testing.assert_close(x.grad, ref_x)
        for name, p in layer.named_parameters():
            if name in ref_p:
                torch.testing.assert_close(p.grad, ref_p[name], msg=name)

    def test_n_dimensional_input_matches_unpatched_peft(self):
        """transformers calls projections with ``[B, S, H]``; the kernel has to
        keep every leading dimension."""
        _requires_train_extra()
        import torch

        from soup_cli.utils.fast_lora import patch_fast_lora_single_projection

        torch.manual_seed(0)
        model, layer = _make_adapted_linear()
        x = torch.randn(2, 3, 8, requires_grad=True)

        ref = layer(x)
        ref.sum().backward()
        ref_x = x.grad.detach().clone()
        ref_p = {
            name: p.grad.detach().clone()
            for name, p in layer.named_parameters()
            if p.grad is not None
        }

        x.grad = None
        for p in layer.parameters():
            p.grad = None

        assert patch_fast_lora_single_projection(model) == 1
        out = layer(x)
        out.sum().backward()

        torch.testing.assert_close(out, ref)
        torch.testing.assert_close(x.grad, ref_x)
        for name, p in layer.named_parameters():
            if name in ref_p:
                torch.testing.assert_close(p.grad, ref_p[name], msg=name)


class TestDelegation:
    """#839: peft's own forward must run whenever it owns the call."""

    def test_disabled_adapters_delegate(self):
        _requires_train_extra()
        import torch

        from soup_cli.utils.fast_lora import patch_fast_lora_single_projection

        torch.manual_seed(0)
        model, layer = _make_adapted_linear()
        x = torch.randn(5, 8)

        layer.enable_adapters(False)
        expected = layer(x)
        layer.enable_adapters(True)

        assert patch_fast_lora_single_projection(model) == 1
        layer.enable_adapters(False)
        got = layer(x)
        layer.enable_adapters(True)

        # With adapters disabled peft returns the base projection; a kernel
        # call would still add the LoRA term, so equality means delegation.
        torch.testing.assert_close(got, expected)
        torch.testing.assert_close(got, layer.base_layer(x))

    def test_merged_adapters_delegate(self):
        _requires_train_extra()
        import torch

        from soup_cli.utils.fast_lora import patch_fast_lora_single_projection

        torch.manual_seed(0)
        model, layer = _make_adapted_linear()
        x = torch.randn(5, 8)

        layer.merged_adapters = ["default"]
        expected = layer(x)
        layer.merged_adapters = []

        assert patch_fast_lora_single_projection(model) == 1
        layer.merged_adapters = ["default"]
        got = layer(x)
        layer.merged_adapters = []

        torch.testing.assert_close(got, expected)

    def test_fused_variant_delegates(self):
        """DoRA and friends produce tuple results the hand-written backward
        does not model, so they must run through peft."""
        _requires_train_extra()
        import torch

        from soup_cli.utils.fast_lora import patch_fast_lora_single_projection

        torch.manual_seed(0)
        model, layer = _make_adapted_linear(dora=True)
        x = torch.randn(5, 8)

        expected = layer(x)

        assert patch_fast_lora_single_projection(model) == 1
        got = layer(x)
        torch.testing.assert_close(got, expected)

    def test_dropout_delegates(self):
        """Non-zero dropout is refused by the tracker's validators; a direct
        caller gets peft's own (stochastic) path, not silently unregularised
        math. Seeded, both paths must draw the same masks."""
        _requires_train_extra()
        import torch

        from soup_cli.utils.fast_lora import patch_fast_lora_single_projection

        torch.manual_seed(0)
        model, layer = _make_adapted_linear(dropout=0.5)
        twin = copy.deepcopy(model)

        assert patch_fast_lora_single_projection(model) == 1
        layer.train()
        twin.linear.train()
        x = torch.randn(5, 8)

        torch.manual_seed(7)
        got = layer(x)
        torch.manual_seed(7)
        expected = twin.linear(x)

        torch.testing.assert_close(got, expected)

    def test_adapter_names_kwarg_goes_to_peft(self):
        """The mixed-batch forward is peft's; the patched and unpatched calls
        must produce the same outcome, value or exception."""
        _requires_train_extra()
        import torch

        from soup_cli.utils.fast_lora import patch_fast_lora_single_projection

        torch.manual_seed(0)
        model, layer = _make_adapted_linear()
        x = torch.randn(5, 8)

        def run():
            try:
                return ("value", layer(x, adapter_names=["default"]))
            except Exception as exc:  # noqa: BLE001
                return ("raise", type(exc), str(exc))

        before = run()
        assert patch_fast_lora_single_projection(model) == 1
        after = run()

        assert before[0] == after[0]
        if before[0] == "raise":
            assert before[1:] == after[1:]
        else:
            torch.testing.assert_close(before[1], after[1])

    def test_fan_in_fan_out_base_delegates(self):
        """A ``Conv1D``-shaped base is peft's, not the kernel's.

        ``fan_in_fan_out`` stores the weight transposed, so the kernel's
        ``x @ W^T`` and ``dY @ W`` are computed against the wrong layout: on a
        square projection that is silently wrong and on a non-square one it
        raises. #839 scopes Conv1D out, so the patched forward must return
        peft's own result.
        """
        _requires_train_extra()
        import pytest as _pytest
        import torch
        import torch.nn as nn
        from peft import LoraConfig, inject_adapter_in_model

        from soup_cli.utils.fast_lora import patch_fast_lora_single_projection

        conv1d_cls = _pytest.importorskip(
            "transformers.pytorch_utils", reason="the Conv1D fixture is transformers'"
        ).Conv1D

        class _Wrapper(nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.proj = layer

        def _build(in_f, out_f):
            torch.manual_seed(0)
            model = _Wrapper(conv1d_cls(in_f, out_f))
            inject_adapter_in_model(
                LoraConfig(
                    r=2, lora_alpha=4, lora_dropout=0.0, bias="none", target_modules=["proj"]
                ),
                model,
            )
            layer = model.proj
            assert getattr(layer, "fan_in_fan_out", False) is True
            _randomise_lora_b(model)
            return model, layer

        # Both shapes, because they fail differently. A ``Conv1D`` stores
        # ``[in, out]``, so a "non-square" one is ``Conv1D(out_f, in_f)``: on the
        # square case the kernel's wrong layout happens to agree, and only the
        # non-square case can tell "delegated" from "computed the other way
        # round" - which is the one that raises rather than drifts.
        for in_f, out_f in ((8, 8), (8, 6)):
            model, layer = _build(out_f, in_f)
            torch.manual_seed(5)
            x = torch.randn(3, in_f, requires_grad=True)
            torch.manual_seed(9)
            expected = layer(x)
            expected.sum().backward()
            expected_grads = {
                name: p.grad.detach().clone()
                for name, p in layer.named_parameters()
                if p.grad is not None
            }
            expected_x_grad = x.grad.detach().clone()

            for p in layer.parameters():
                p.grad = None
            x.grad = None

            assert patch_fast_lora_single_projection(model) == 1
            torch.manual_seed(9)
            got = layer(x)
            torch.testing.assert_close(got, expected)
            got.sum().backward()

            # Backward too, where ``dY @ W`` is the term whose layout a
            # transposed weight would get wrong.
            torch.testing.assert_close(x.grad, expected_x_grad)
            for name, p in layer.named_parameters():
                if p.grad is None or name not in expected_grads:
                    continue
                torch.testing.assert_close(p.grad, expected_grads[name])

    def test_delegation_is_what_happens_not_merely_what_agrees(self):
        """Assert the guard ran, not just that the output happened to match.

        Output alone cannot prove the guard ran: a square ``Conv1D`` gives the
        right answer whether the kernel handled it or peft did. So the original
        forward is captured and spied on, and the test asserts the delegate was
        the thing that ran.
        """
        _requires_train_extra()
        import functools

        import pytest as _pytest
        import torch
        import torch.nn as nn
        from peft import LoraConfig, inject_adapter_in_model

        from soup_cli.utils.fast_lora import patch_fast_lora_single_projection

        conv1d_cls = _pytest.importorskip(
            "transformers.pytorch_utils", reason="the Conv1D fixture is transformers'"
        ).Conv1D

        class _Wrapper(nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.proj = layer

        # peft wraps a ``Conv1D`` base in its own ``lora.Linear``, which this
        # patch does match, and sets ``fan_in_fan_out`` for exactly that case.
        # ``torch.nn.Conv1d`` is wrapped in ``lora.Conv1d``, a type the patch
        # deliberately does not match, so it is not a case the guard sees.
        cases = (("fan_in_fan_out", lambda: conv1d_cls(8, 8), True),)
        for label, make_layer, expected_fifo in cases:
            torch.manual_seed(0)
            model = _Wrapper(make_layer())
            inject_adapter_in_model(
                LoraConfig(
                    r=2, lora_alpha=4, lora_dropout=0.0, bias="none", target_modules=["proj"]
                ),
                model,
            )
            layer = model.proj
            assert getattr(layer, "fan_in_fan_out", None) is expected_fifo, label
            _randomise_lora_b(model)

            # Capture-and-spy: the patch takes ``child.forward`` at patch time,
            # so the spy becomes the delegate the guard falls back to.
            real_forward = layer.forward
            calls = []
            layer.forward = functools.partial(_spy, real_forward, calls)

            assert patch_fast_lora_single_projection(model) == 1
            torch.manual_seed(5)
            x = torch.randn(3, 8)
            layer(x)

            assert calls == ["delegated"], (
                f"{label}: the patched forward should have delegated to peft's own "
                f"path, but the kernel ran instead (delegations seen: {calls})"
            )

    def test_lora_bias_delegates(self):
        """``lora_bias=True`` puts a bias on ``lora_B`` the kernel does not carry."""
        _requires_train_extra()
        import torch
        import torch.nn as nn
        from peft import LoraConfig, inject_adapter_in_model

        from soup_cli.utils.fast_lora import patch_fast_lora_single_projection

        class _Wrapper(nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.linear = layer

        torch.manual_seed(0)
        model = _Wrapper(nn.Linear(8, 6))
        inject_adapter_in_model(
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                bias="none",
                lora_bias=True,
                target_modules=["linear"],
            ),
            model,
        )
        layer = model.linear
        assert layer.lora_B["default"].bias is not None, "fixture needs lora_bias=True to hold"
        _randomise_lora_b(model)
        with torch.no_grad():
            layer.lora_B["default"].bias.copy_(torch.randn(6) * 0.1)

        torch.manual_seed(5)
        x = torch.randn(4, 8)
        torch.manual_seed(9)
        expected = layer(x)

        assert patch_fast_lora_single_projection(model) == 1
        torch.manual_seed(9)
        got = layer(x)
        # The bias would change the output if the kernel ran instead of peft.
        torch.testing.assert_close(got, expected)


class TestPatchPlumbing:
    """#839: instance-level patching, counting, reversibility, call-time reads."""

    def test_counts_only_adapted_linears_and_is_idempotent(self):
        _requires_train_extra()
        import torch.nn as nn

        from soup_cli.utils.fast_lora import (
            patch_fast_lora_single_projection,
            unpatch_fast_lora_single_projection,
        )

        model, _layer = _make_adapted_linear()
        model.extra = nn.Linear(8, 6)

        assert patch_fast_lora_single_projection(model) == 1
        assert patch_fast_lora_single_projection(model) == 0
        assert unpatch_fast_lora_single_projection(model) == 1
        assert unpatch_fast_lora_single_projection(model) == 0
        assert patch_fast_lora_single_projection(model) == 1

    def test_unpatch_restores_peft_forward(self):
        _requires_train_extra()
        import torch

        from soup_cli.utils.fast_lora import (
            patch_fast_lora_single_projection,
            unpatch_fast_lora_single_projection,
        )

        torch.manual_seed(0)
        model, layer = _make_adapted_linear()
        x = torch.randn(5, 8)

        expected = layer(x)
        assert patch_fast_lora_single_projection(model) == 1
        assert unpatch_fast_lora_single_projection(model) == 1
        assert not hasattr(layer, "_soup_fast_lora_single_projection")
        torch.testing.assert_close(layer(x), expected)

    def test_patched_forward_reads_the_base_weight_at_call_time(self):
        """The streaming contract: ``functional_call`` substitutes weights only
        for the duration of a call, so a reference captured at patch time is
        worthless. Mutating the base weight after patching must change the
        output."""
        _requires_train_extra()
        import torch

        from soup_cli.utils.fast_lora import patch_fast_lora_single_projection

        torch.manual_seed(0)
        model, layer = _make_adapted_linear()
        x = torch.randn(5, 8)

        assert patch_fast_lora_single_projection(model) == 1
        first = layer(x)

        with torch.no_grad():
            layer.base_layer.weight.mul_(2.0)
        second = layer(x)

        assert not torch.equal(first, second)

        with torch.no_grad():
            w = layer.base_layer.weight.detach().clone()
            b = layer.base_layer.bias
            a = layer.lora_A["default"].weight.detach()
            bb = layer.lora_B["default"].weight.detach()
            s = layer.scaling["default"]
        expected = x @ w.T + b + s * ((x @ a.T) @ bb.T)
        torch.testing.assert_close(second, expected)


class TestSavedBytes:
    """#839: bytes held for backward, reported; no figure asserted in advance."""

    def test_not_worse_than_pefts_graph(self):
        _requires_train_extra()
        import torch

        from soup_cli.utils.fast_lora import patch_fast_lora_single_projection

        torch.manual_seed(0)
        model, layer = _make_adapted_linear(in_f=8, out_f=6, r=2, alpha=4)

        plain = _saved_bytes(layer, torch.randn(64, 8, requires_grad=True))

        for p in layer.parameters():
            p.grad = None
        assert patch_fast_lora_single_projection(model) == 1

        fast = _saved_bytes(layer, torch.randn(64, 8, requires_grad=True))

        assert 0 < fast <= plain
        print(f"#839 saved bytes: peft={plain} fast={fast}")


class TestStreamedModel:
    """#331 shape: more decoder layers than stream buffers, CPU."""

    def test_fast_lora_matches_resident_on_a_streamed_tiny_llama(self, tmp_path):
        _requires_train_extra()
        for mod in ("transformers", "safetensors"):
            pytest.importorskip(mod, reason=f"{mod} is only in the [train] extra")
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from safetensors.torch import save_file
        from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

        from soup_cli.utils.fast_lora import patch_fast_lora_single_projection
        from soup_cli.utils.layer_shard import shard_checkpoint
        from soup_cli.utils.layer_stream_runtime import build_streamed_model

        def lora_config():
            return LoraConfig(
                r=4,
                lora_alpha=8,
                lora_dropout=0.0,
                bias="none",
                target_modules=["q_proj", "v_proj"],
                task_type=TaskType.CAUSAL_LM,
            )

        torch.manual_seed(7)
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            tie_word_embeddings=False,
            max_position_embeddings=128,
        )
        weights = tmp_path / "model"
        weights.mkdir(parents=True, exist_ok=True)
        save_file(
            {k: v.contiguous() for k, v in LlamaForCausalLM(config).state_dict().items()},
            str(weights / "model.safetensors"),
        )
        config.save_pretrained(str(weights))

        shards = str(tmp_path / "shards")
        index = shard_checkpoint(str(weights), shards, dtype="float32", arch="llama")
        model, _runtime = build_streamed_model(
            model_id=str(weights),
            shard_dir=shards,
            index=index,
            lora_config=lora_config(),
            device="cpu",
            dtype="float32",
            buffers=2,
            pin=False,
            seed=3,
        )

        # q_proj + v_proj on each of the 4 layers, all single projections.
        assert patch_fast_lora_single_projection(model) == 8

        # peft leaves lora_B at zeros, so the resident twin would be compared
        # LoRA-term-against-LoRA-term with the term equal to zero. Randomise
        # before the copy so both twins carry a live adapter.
        _randomise_lora_b(model)

        resident = get_peft_model(
            AutoModelForCausalLM.from_pretrained(str(weights), dtype=torch.float32),
            lora_config(),
        )
        src = dict(model.named_parameters())
        dst = dict(resident.named_parameters())
        copied = 0
        for name, param in src.items():
            if "lora_" not in name or param.is_meta:
                continue
            assert name in dst, f"resident model has no parameter {name!r}"
            with torch.no_grad():
                dst[name].copy_(param)
            copied += 1
        assert copied == 16
        assert patch_fast_lora_single_projection(resident) == 8

        torch.manual_seed(11)
        input_ids = torch.randint(0, 64, (1, 8))
        labels = torch.randint(0, 64, (1, 8))

        streamed_loss = model(input_ids=input_ids, labels=labels).loss
        streamed_loss.backward()
        resident_loss = resident(input_ids=input_ids, labels=labels).loss
        resident_loss.backward()

        torch.testing.assert_close(streamed_loss, resident_loss, rtol=1e-5, atol=1e-5)

        streamed_grads = {
            n: p.grad for n, p in model.named_parameters() if "lora_" in n and p.grad is not None
        }
        resident_grads = {
            n: p.grad for n, p in resident.named_parameters() if "lora_" in n and p.grad is not None
        }
        assert len(streamed_grads) == 16
        assert len(resident_grads) == 16
        matched = 0
        for name, grad in sorted(streamed_grads.items()):
            target = next((g for n, g in resident_grads.items() if n.endswith(name)), None)
            assert target is not None, f"no resident grad for {name!r}"
            # Every layer is checked, not just the first buffer's worth.
            torch.testing.assert_close(grad, target, rtol=1e-4, atol=1e-5, msg=name)
            matched += 1
        assert matched == 16


@pytest.mark.gpu
class TestNf4Parity:
    """NF4: parity against unpatched ``lora.bnb.Linear4bit``. Needs a card."""

    def test_nf4_forward_and_backward_match_unpatched_peft(self):
        _requires_train_extra()
        pytest.importorskip("bitsandbytes", reason="4-bit tests need bitsandbytes")
        import bitsandbytes as bnb
        import torch
        import torch.nn as nn
        from peft import LoraConfig, inject_adapter_in_model

        from soup_cli.utils.fast_lora import patch_fast_lora_single_projection

        class _Wrapper(nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.linear = layer

        torch.manual_seed(0)
        base = bnb.nn.Linear4bit(32, 32, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4")
        model = _Wrapper(base).to("cuda")
        inject_adapter_in_model(
            LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none", target_modules=["linear"]),
            model,
        )
        # As elsewhere: zeros in lora_B would make the parity comparison vacuous.
        _randomise_lora_b(model)
        layer = model.linear

        x = torch.randn(5, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        ref = layer(x)
        ref.sum().backward()
        ref_x = x.grad.detach().clone()
        ref_p = {
            name: p.grad.detach().clone()
            for name, p in layer.named_parameters()
            if p.grad is not None
        }

        x.grad = None
        for p in layer.parameters():
            p.grad = None

        assert patch_fast_lora_single_projection(model) == 1
        out = layer(x)
        out.sum().backward()

        # Loose on purpose: the fused-vs-dequant divergence pinned by #968 is
        # part of this delta, and its magnitude on this shape is unmeasured
        # without a card. The PR asks for the readout from the first CUDA run.
        atol = 1e-2
        rtol = 1e-2
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
        torch.testing.assert_close(x.grad, ref_x, rtol=rtol, atol=atol)
        for name, p in layer.named_parameters():
            if name in ref_p:
                torch.testing.assert_close(p.grad, ref_p[name], rtol=rtol, atol=atol, msg=name)


@pytest.mark.gpu
class TestMicroBenchmark:
    """Llama-3.1-8B ``o_proj`` shape: report, do not assert (tracker #792)."""

    def _time_fwd_bwd(self, layer, x, iters=5):
        import torch

        def step():
            out = layer(x)
            out.sum().backward()

        step()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            step()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    def test_report_only(self):
        _requires_train_extra()
        pytest.importorskip("bitsandbytes", reason="NF4 timing needs bitsandbytes")
        import bitsandbytes as bnb
        import torch
        import torch.nn as nn
        from peft import LoraConfig, inject_adapter_in_model

        from soup_cli.utils.fast_lora import patch_fast_lora_single_projection

        class _Wrapper(nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.linear = layer

        config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0, bias="none", target_modules=["linear"]
        )
        x = torch.randn(1, 4096, 4096, device="cuda", dtype=torch.bfloat16)

        dense = _Wrapper(nn.Linear(4096, 4096, bias=False, dtype=torch.bfloat16)).to("cuda")
        inject_adapter_in_model(config, dense)
        plain_ms = self._time_fwd_bwd(dense.linear, x)
        assert patch_fast_lora_single_projection(dense) == 1
        fast_ms = self._time_fwd_bwd(dense.linear, x)
        assert torch.isfinite(dense.linear(x)).all()
        print(f"#839 o_proj bf16 dense: peft={plain_ms:.2f}ms fast={fast_ms:.2f}ms")

        nf4 = _Wrapper(
            bnb.nn.Linear4bit(
                4096, 4096, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4"
            )
        ).to("cuda")
        inject_adapter_in_model(config, nf4)
        nf4_plain_ms = self._time_fwd_bwd(nf4.linear, x)
        assert patch_fast_lora_single_projection(nf4) == 1
        nf4_fast_ms = self._time_fwd_bwd(nf4.linear, x)
        assert torch.isfinite(nf4.linear(x)).all()
        print(f"#839 o_proj bf16 nf4: peft={nf4_plain_ms:.2f}ms fast={nf4_fast_ms:.2f}ms")
