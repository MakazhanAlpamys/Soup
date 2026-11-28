"""Native DeepSeek-V3 routing replay on CPU."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from soup_cli.utils.host_memory_budget import HostMemoryBudget
from soup_cli.utils.routing_oracle import DeepseekRoutingOracle
from tests.test_l2l_mbr import _run
from tests.test_l2l_mbr import window as _mbr_window

window = _mbr_window  # the shared streamed-window fixture, re-exported under its own name


@pytest.fixture
def router(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tests.test_expert_stream import _tiny_deepseek_v3

    _, model = _tiny_deepseek_v3(tmp_path)
    model.requires_grad_(False)
    idx = model.config.first_k_dense_replace
    return model, idx, model.model.layers[idx].mlp.gate


@pytest.mark.parametrize(
    ("dtype", "normalized", "trainable"),
    [("bfloat16", True, False), ("float32", False, True), ("float32", True, True)],
)
def test_replay_matches_native_route_and_first_order_vjp(router, dtype, normalized, trainable):
    import torch

    model, idx, gate = router
    model.to(getattr(torch, dtype))
    for layer in model.model.layers[idx:]:
        layer.mlp.gate.norm_topk_prob = normalized
        layer.mlp.gate.routed_scaling_factor = 2.5
    hidden = torch.randn(1, 5, gate.hidden_dim, dtype=gate.weight.dtype).requires_grad_()
    gate.weight.requires_grad_(trainable)
    inputs = (hidden, gate.weight) if trainable else (hidden,)
    logits, weights, ids = gate(hidden)
    upstream = torch.randn_like(weights)
    expected = torch.autograd.grad(weights, inputs, upstream)
    gate.weight.requires_grad_(False)

    with DeepseekRoutingOracle(model, pin=False) as oracle:
        oracle.begin_step([hidden.shape])
        with torch.no_grad(), oracle.visit(idx, 0, "forward"):
            forward = gate(hidden)
        with torch.no_grad(), oracle.visit(idx, 0, "peek"):
            peek = gate(hidden)
        gate.weight.requires_grad_(trainable)
        try:
            with oracle.visit(idx, 0, "recompute"):
                replay = gate(hidden)
            actual = torch.autograd.grad(replay[1], inputs, upstream)
        finally:
            gate.weight.requires_grad_(False)
        assert oracle.report()["step_counts"] == {
            "native_forward": 1, "oracle_peek": 1, "oracle_recompute": 1, "oracle_vjp": 1,
        }

    assert all(torch.equal(a, b) for a, b in zip(forward, (logits, weights, ids), strict=True))
    for result in (peek, replay):
        assert result[0] is None
        assert torch.equal(result[1], weights) and torch.equal(result[2], ids)
    for got, want in zip(actual, expected, strict=True):
        assert torch.count_nonzero(want)
        tolerance = 1e-6 if dtype == "float32" and normalized else 0
        torch.testing.assert_close(got, want, atol=tolerance, rtol=tolerance)


def test_frozen_router_does_not_save_recompute_hidden(router):
    import weakref

    import torch

    model, idx, gate = router
    hidden = torch.randn(1, 5, gate.hidden_dim, requires_grad=True)
    with DeepseekRoutingOracle(model, pin=False) as oracle:
        oracle.begin_step([hidden.shape])
        with torch.no_grad(), oracle.visit(idx, 0, "forward"):
            gate(hidden)
        recompute = hidden * 1.0
        ref = weakref.ref(recompute)
        with oracle.visit(idx, 0, "recompute"):
            weights = gate(recompute)[1]
        del recompute
        assert ref() is None
        (gradient,) = torch.autograd.grad(weights, hidden, torch.randn_like(weights))
        assert torch.count_nonzero(gradient)


@pytest.mark.parametrize("window", [{"dtype": "bfloat16"}], indirect=True)
@pytest.mark.parametrize(
    ("count", "mode", "group_size"),
    [(1, "regular", 1), (2, "regular", 2), (4, "regular", 3),
     (4, "split", 3), (4, "ordered", 3), (1, "decoupled", 1),
     (2, "decoupled", 2), (4, "decoupled", 3)],
)
def test_bf16_mbr_replay_preserves_losses_and_lora_gradients(window, count, mode, group_size):
    import torch

    model, stepper, batches, calls = window
    short_window = (model, stepper, batches[:count], calls[:count])
    options = {
        "route_preserving_mbr": True, "mbr_group_size": group_size, "collect_phase_stats": True,
        "mbr_split_backward": mode == "split", "mbr_ordered_backward": mode == "ordered",
        "mbr_decoupled_subwindow": mode == "decoupled",
    }
    expected_losses, expected_grads = _run(short_window, **options)
    with DeepseekRoutingOracle(model, pin=False) as oracle:
        losses, grads = _run(short_window, routing_oracle=oracle, **options)
        assert oracle.report()["steps_completed"] == 1
        stats = stepper.routing_oracle_stats
        assert stats["complete"]
        assert stats["step_counts"] == {
            "native_forward": 2 * count,
            "oracle_peek": 2 * count if mode == "decoupled" else 0,
            "oracle_recompute": 2 * count, "oracle_vjp": 2 * count,
        }
    assert len(losses) == len(expected_losses) == count
    assert all(torch.equal(a, b) for a, b in zip(losses, expected_losses, strict=True))
    assert grads.keys() == expected_grads.keys()
    for name, want in expected_grads.items():
        assert torch.equal(grads[name], want), name
    for layer in stepper.route_preserving_mbr_stats["layers"].values():
        assert layer["route_id_mismatch_slots"] == 0
        assert layer["gate_weight_max_abs"] == 0.0


def test_router_contract_refuses_trainable_weights_and_logit_consumers(router):
    model, _idx, gate = router
    gate.weight.requires_grad_(True)
    with pytest.raises(RuntimeError) as caught:
        DeepseekRoutingOracle(model)
    assert str(caught.value) == "DeepSeek routing oracle: router weights must be frozen"
    gate.weight.requires_grad_(False)
    model.config.output_router_logits = True
    with pytest.raises(RuntimeError) as caught:
        DeepseekRoutingOracle(model)
    assert str(caught.value) == (
        "DeepSeek routing oracle: full router-logit consumers are unsupported"
    )
    assert "forward" not in gate.__dict__


def test_oracle_aborts_when_decoupled_expert_inputs_change_between_replays(window):
    import torch

    model, stepper, _batches, _calls = window
    norm = stepper.layers[2].inner.post_attention_layernorm

    def change_recompute_input(_module, _args, output):
        return output + 1 if torch.is_grad_enabled() else output

    options = {
        "route_preserving_mbr": True, "mbr_decoupled_subwindow": True, "mbr_group_size": 2,
    }
    with DeepseekRoutingOracle(model, pin=False) as oracle:
        handle = norm.register_forward_hook(change_recompute_input)
        try:
            with pytest.raises(RuntimeError) as caught:
                _run(window, routing_oracle=oracle, **options)
            assert str(caught.value) == (
                "routing_oracle: decoupled MBR expert inputs changed between replays "
                "at layer 2, micro-batch 0"
            )
            assert stepper.routing_oracle_stats is None
            assert not oracle.report()["complete"]
        finally:
            handle.remove()

        _run(window, routing_oracle=oracle, **options)
        assert oracle.report()["complete"]
        assert oracle.report()["steps_completed"] == 1


@pytest.mark.parametrize("autocast", [False, True])
def test_oracle_refuses_incompatible_step_before_activations(window, autocast):
    import torch

    from soup_cli.utils.l2l_step import ReferencePass

    model, stepper, batches, calls = window
    with DeepseekRoutingOracle(model, pin=False) as oracle:
        if autocast:
            options = {"route_preserving_mbr": True}
            context = torch.autocast("cpu", dtype=torch.bfloat16)
            expected = "routing_oracle requires native router precision with autocast disabled"
        else:
            options = {}
            context = nullcontext()
            expected = "routing_oracle requires route_preserving_mbr=True"
        with context, pytest.raises(ValueError) as caught:
            stepper.run(batches, None, calls=calls, routing_oracle=oracle, **options)
        assert str(caught.value) == expected
        with pytest.raises(ValueError) as caught:
            stepper.run(
                batches, None, calls=calls, routing_oracle=oracle,
                route_preserving_mbr=True, reference=ReferencePass(head=lambda *_: None),
            )
        assert str(caught.value) == "routing_oracle does not support reference passes"
        assert oracle.host_bytes()["owned_host_storage_bytes"] == 0


def test_oracle_step_refuses_fp8_before_activations(window):
    from soup_cli.utils.fp8_activations import FP8ActivationStore

    model, stepper, batches, calls = window
    store = FP8ActivationStore.__new__(FP8ActivationStore)
    with DeepseekRoutingOracle(model, pin=False) as oracle:
        with pytest.raises(ValueError) as caught:
            stepper.run(
                batches, store, calls=calls, routing_oracle=oracle, route_preserving_mbr=True,
            )
        assert str(caught.value) == "routing_oracle requires uncompressed L2L activations"
        assert oracle.host_bytes()["owned_host_storage_bytes"] == 0


def test_abort_clears_pending_replay_and_allows_a_new_window(router):
    import torch

    model, idx, gate = router
    budget = HostMemoryBudget(800)
    hidden = torch.randn(1, 5, gate.hidden_dim, requires_grad=True)
    with DeepseekRoutingOracle(model, host_budget=budget, pin=False) as oracle:
        oracle.begin_step([hidden.shape])
        with torch.no_grad(), oracle.visit(idx, 0, "forward"):
            gate(hidden)
        with oracle.visit(idx, 0, "recompute"):
            old_weights = gate(hidden)[1]
        oracle.abort_step()
        assert not oracle.report()["complete"]
        with pytest.raises(RuntimeError) as caught:
            torch.autograd.grad(old_weights, hidden, torch.randn_like(old_weights))
        assert str(caught.value) == "DeepSeek routing oracle: replay is closed"
        # Shape capture between windows still uses the native gate.
        assert gate(hidden)[0] is not None
        assert budget.used_bytes == 160
        with pytest.raises(MemoryError) as caught:
            oracle.begin_step([(1, 30, gate.hidden_dim)])
        assert str(caught.value) == (
            f"DeepSeek routing oracle buffers {id(oracle)}: needs 960 bytes; 800 available"
        )
        assert budget.used_bytes == 0
        oracle.begin_step([(1, 8, gate.hidden_dim)])
        assert budget.used_bytes == 256
    assert budget.used_bytes == 0
    assert "forward" not in gate.__dict__


def test_trainer_reuses_oracle_on_larger_window_and_restores_gate(window, monkeypatch):
    import torch

    from soup_cli.trainer.l2l_trainer import L2LState, make_l2l_trainer_class
    from soup_cli.utils import l2l_spill

    model, stepper, batches, _calls = window
    gate = stepper.layers[1].inner.mlp.gate
    original = gate.forward
    state = L2LState(stepper.runtime, torch.float32, route_preserving_mbr=True, routing_oracle=True)
    trainer = make_l2l_trainer_class(object)()
    trainer._soup_l2l = state
    trainer.args = SimpleNamespace(neftune_noise_alpha=None, device="cpu")
    trainer.accelerator = SimpleNamespace(autocast=lambda: torch.autocast("cpu"))
    budget_observations = []

    def budget():
        budget_observations.append(
            state._routing_oracle.host_bytes()["owned_host_storage_bytes"]
        )
        return l2l_spill.RamBudget(None, 10**7, None, {}, None)

    monkeypatch.setattr(l2l_spill, "ram_budget", budget)
    monkeypatch.setattr(l2l_spill.RamBudget, "describe", lambda _self: "fixture")
    try:
        for count in (1, 2):
            state.window = [{"input_ids": ids, "labels": ids} for ids in batches[:count]]
            state.position = 0
            with torch.autocast("cpu"):
                loss = trainer.training_step(
                    model, state.window[0], sum(ids.numel() - 1 for ids in batches[:count]),
                )
                assert torch.is_autocast_enabled("cpu")
            assert loss.isfinite()
            assert state.stepper.routing_oracle_stats["steps_completed"] == count
            charged = (96, 288)[count - 1]
            assert state.spill_plan["budget_bytes"] == 10**7 - charged
        assert state.log_fields()["l2l_routing_oracle"] is True
        assert budget_observations == [0, 0]
    finally:
        state.close()
    assert gate.forward == original
