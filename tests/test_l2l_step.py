"""On CPU: the shipped L2L step equals gradient accumulation bit for bit.

The seam is ``soup_cli.utils.l2l_step.L2LStep.run`` plus the shipped
``ActivationStore``. The gradient-accumulation reference and the comparison
helpers come from the benchmark harness (``benchmarks/harness/l2l_schedule.py``), which lives
outside the package. The main runtime keeps no NF4 weight-restore counter, so there are
no restore-count tests.
"""

from __future__ import annotations

import pytest

from tests.test_l2l_probe_harness import SEQ, _load, _micro_batches, streamed  # noqa: F401


@pytest.fixture()
def harness():
    return _load("l2l_schedule")


def _store(k, n_layers=3, hidden=64):
    import torch

    from soup_cli.utils.l2l_activations import ActivationStore

    return ActivationStore(
        n_layers=n_layers,
        k=k,
        chunk_shape=(1, SEQ, hidden),
        dtype=torch.float32,
        device="cpu",
        pin=False,
    )


@pytest.mark.parametrize("k", [1, 2, 4])
def test_shipped_l2l_step_equals_gradient_accumulation(harness, streamed, k):  # noqa: F811
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    model, runtime = streamed
    harness.make_non_vacuous(model)
    mbs = _micro_batches(k)

    harness.zero_grads(model)
    ga_losses = harness.ga_step(model, mbs)
    ga = harness.collect_grads(model)

    step = L2LStep(model, runtime, capture_layer_call(model, mbs[0]))
    harness.zero_grads(model)
    l2l_losses = step.run(mbs, _store(k))
    l2l = harness.collect_grads(model)

    result = harness.compare(ga, l2l)
    assert result["tensors"] == 3 * 4 * 2  # 3 layers x q/k/v/o x A/B
    assert result["all_zero"] == []
    assert result["equal"], result
    assert harness.losses_equal(ga_losses, l2l_losses)


@pytest.mark.parametrize("k", [1, 2])
def test_l2l_scales_gradients_and_returns_unscaled_losses(harness, streamed, k):  # noqa: F811
    import torch

    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    model, runtime = streamed
    harness.make_non_vacuous(model)
    micro_batches = _micro_batches(k)
    harness.zero_grads(model)
    expected_losses = harness.ga_step(model, micro_batches)
    expected_grads = harness.collect_grads(model)

    class Scaler:
        calls = 0

        def scale(self, loss):
            self.calls += 1
            return loss * 1024.0

    scaler = Scaler()
    step = L2LStep(model, runtime, capture_layer_call(model, micro_batches[0]))
    harness.zero_grads(model)
    losses = step.run(micro_batches, _store(k), scaler=scaler)
    grads = harness.collect_grads(model)

    assert scaler.calls == k
    assert harness.losses_equal(expected_losses, losses)
    assert grads.keys() == expected_grads.keys()
    for name, grad in expected_grads.items():
        assert torch.equal(grads[name], grad * 1024.0), name


# --------------------------------------------------------------------------
# layer loads, read from the runtime counters
# --------------------------------------------------------------------------
def _warm_walk(model, ids):
    """Loads depend on what the previous walk left in the slots; start every
    measurement after the same forward-only walk."""
    from soup_cli.utils.l2l_step import capture_layer_call

    capture_layer_call(model, ids)


def _l2l_counters(harness, model, runtime, k, cold=None):
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    mbs = _micro_batches(k)
    step = L2LStep(model, runtime, capture_layer_call(model, mbs[0]))
    harness.zero_grads(model)
    _warm_walk(model, mbs[0])
    if cold:
        cold(model)
    step.run(mbs, _store(k))
    return step.counters


def _ga_counters(harness, model, runtime, k, cold=None):
    from soup_cli.utils.l2l_step import count_loads

    mbs = _micro_batches(k)
    harness.zero_grads(model)
    _warm_walk(model, mbs[0])
    if cold:
        cold(model)
    with count_loads(runtime) as counters:
        harness.ga_step(model, mbs)
    return counters


def test_count_loads_reads_the_runtime_counters_without_patching(streamed):  # noqa: F811
    from soup_cli.utils.l2l_step import count_loads

    model, runtime = streamed
    load_async = runtime.pool.load_async
    with count_loads(runtime) as counters:
        _warm_walk(model, _micro_batches(1)[0])
    assert runtime.pool.load_async == load_async
    assert counters.layer_loads > 0


def test_l2l_counters_exist_after_a_step_and_are_zero_before(harness, streamed):  # noqa: F811
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    model, runtime = streamed
    mbs = _micro_batches(1)
    step = L2LStep(model, runtime, capture_layer_call(model, mbs[0]))
    assert step.counters is None
    step.run(mbs, _store(1))
    assert step.counters.layer_loads > 0


def test_k1_l2l_loads_what_gradient_accumulation_loads(harness, streamed):  # noqa: F811
    model, runtime = streamed
    harness.make_non_vacuous(model)
    ga = _ga_counters(harness, model, runtime, 1)
    l2l = _l2l_counters(harness, model, runtime, 1)
    assert l2l.layer_loads == ga.layer_loads
    assert l2l.layer_loads > 0


def test_k4_does_not_load_four_times_what_k1_loads(harness, streamed):  # noqa: F811
    model, runtime = streamed
    harness.make_non_vacuous(model)
    k1 = _l2l_counters(harness, model, runtime, 1)
    k4 = _l2l_counters(harness, model, runtime, 4)
    assert k4.layer_loads == k1.layer_loads
    ga4 = _ga_counters(harness, model, runtime, 4)
    assert ga4.layer_loads > k4.layer_loads  # accumulation reloads per micro-batch


# --------------------------------------------------------------------------
# SFT labels (-100 on padding / masked tokens) reach the loss
# --------------------------------------------------------------------------
def test_l2l_step_with_masked_labels_equals_gradient_accumulation(harness, streamed):  # noqa: F811
    import torch

    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    model, runtime = streamed
    harness.make_non_vacuous(model)
    mbs = _micro_batches(2)
    labels = [ids.clone() for ids in mbs]
    labels[0][:, :3] = -100
    labels[1][:, -2:] = -100
    n_items = sum(int((lab[:, 1:] != -100).sum()) for lab in labels)

    harness.zero_grads(model)
    for ids, lab in zip(mbs, labels):
        model(input_ids=ids, labels=lab, num_items_in_batch=n_items).loss.backward()
    ga = harness.collect_grads(model)

    step = L2LStep(model, runtime, capture_layer_call(model, mbs[0]))
    harness.zero_grads(model)
    step.run(mbs, _store(2), n_items=n_items, labels=labels)
    l2l = harness.collect_grads(model)

    assert harness.compare(ga, l2l)["equal"]
    unmasked = harness.collect_grads(model)
    assert any(torch.count_nonzero(g) for g in unmasked.values())


# --------------------------------------------------------------------------
# rows of different lengths run at their own length, no padding
# --------------------------------------------------------------------------
MIXED_LENGTHS = [5, 8, 3, 6]


def _mixed_rows(k, vocab=64, seed=23):
    import torch

    gen = torch.Generator().manual_seed(seed)
    mbs = [torch.randint(0, vocab, (1, n), generator=gen) for n in MIXED_LENGTHS[:k]]
    labels = [ids.clone() for ids in mbs]
    labels[0][:, :2] = -100
    labels[-1][:, -1:] = -100
    return mbs, labels


def _mixed_store(mbs, hidden=64):
    import torch

    from soup_cli.utils.l2l_activations import ActivationStore

    shapes = [(*ids.shape, hidden) for ids in mbs]
    return ActivationStore(
        n_layers=3,
        k=len(mbs),
        chunk_shape=tuple(max(dims) for dims in zip(*shapes)),
        mb_shapes=shapes,
        dtype=torch.float32,
        device="cpu",
        pin=False,
    )


@pytest.mark.parametrize("k", [1, 2, 4])
def test_l2l_step_with_rows_of_different_lengths_equals_gradient_accumulation(
    harness, streamed, k  # noqa: F811
):
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    model, runtime = streamed
    harness.make_non_vacuous(model)
    mbs, labels = _mixed_rows(k)
    n_items = sum(int((lab[:, 1:] != -100).sum()) for lab in labels)

    harness.zero_grads(model)
    ga_losses = []
    for ids, lab in zip(mbs, labels):
        out = model(input_ids=ids, labels=lab, num_items_in_batch=n_items)
        out.loss.backward()
        ga_losses.append(out.loss.detach())
    ga = harness.collect_grads(model)

    calls = [capture_layer_call(model, ids) for ids in mbs]
    step = L2LStep(model, runtime, calls[0])
    harness.zero_grads(model)
    store = _mixed_store(mbs)
    l2l_losses = step.run(mbs, store, n_items=n_items, labels=labels, calls=calls)
    l2l = harness.collect_grads(model)

    result = harness.compare(ga, l2l)
    assert result["all_zero"] == []
    assert result["equal"], result
    assert harness.losses_equal(ga_losses, l2l_losses)
    # the store holds each micro-batch at its own size, not the longest one's
    assert [store._resident[(0, mb)].numel() for mb in range(k)] == [
        ids.numel() * 64 for ids in mbs
    ]


# --------------------------------------------------------------------------
# k = 1 against plain streamed SFT: the loss and every LoRA gradient, torch.equal
# --------------------------------------------------------------------------
def test_k1_loss_and_lora_gradients_equal_plain_streamed_sft(harness, streamed):  # noqa: F811
    import torch

    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    model, runtime = streamed
    harness.make_non_vacuous(model)
    mbs = _micro_batches(1)
    n_items = int(mbs[0].shape[0]) * (int(mbs[0].shape[1]) - 1)

    harness.zero_grads(model)
    plain = model(input_ids=mbs[0], labels=mbs[0].clone(), num_items_in_batch=n_items).loss
    plain.backward()
    reference = harness.collect_grads(model)
    assert len(reference) == 3 * 4 * 2
    assert all(torch.count_nonzero(grad) for grad in reference.values())

    step = L2LStep(model, runtime, capture_layer_call(model, mbs[0]))
    harness.zero_grads(model)
    losses = step.run(mbs, _store(1), n_items=n_items)
    l2l = harness.collect_grads(model)

    assert len(losses) == 1 and torch.equal(losses[0].detach().cpu(), plain.detach().cpu())
    assert set(l2l) == set(reference)
    for name, grad in reference.items():
        assert torch.equal(grad, l2l[name]), name


# --------------------------------------------------------------------------
# a MoE model is refused with one message, before anything reaches expert_stream
# --------------------------------------------------------------------------
MOE_REFUSAL = "L2L on a MoE model needs per-expert streaming parity (not yet in this version)"


class _FakeCausal:
    def __init__(self, model_type):
        self.config = type("Config", (), {"model_type": model_type})()
        self.lm_head = object()
        self.loss_function = object()


def test_the_refusal_text_is_the_one_users_grep_for():
    from soup_cli.utils import l2l_step

    assert l2l_step.MOE_L2L_REFUSAL == MOE_REFUSAL


def test_model_parts_refuses_a_moe_model_with_the_refusal_message(monkeypatch):
    from soup_cli.utils import l2l_step, layer_stream_runtime

    monkeypatch.setattr(layer_stream_runtime, "decoder_owner", lambda model: object())
    with pytest.raises(ValueError) as caught:
        l2l_step.model_parts(_FakeCausal("deepseek_v3"))
    assert str(caught.value) == MOE_REFUSAL


def test_model_parts_still_names_an_unknown_architecture(monkeypatch):
    from soup_cli.utils import l2l_step, layer_stream_runtime

    monkeypatch.setattr(layer_stream_runtime, "decoder_owner", lambda model: object())
    with pytest.raises(ValueError, match="re-implements the top-level forward"):
        l2l_step.model_parts(_FakeCausal("gpt2"))


@pytest.mark.parametrize("layout", ["fused", "per_expert"])
@pytest.mark.parametrize("fp16", [False, True])
def test_the_trainer_refuses_a_moe_model_before_it_checks_the_layout(layout, fp16):
    from soup_cli.trainer.l2l_trainer import refuse_unsupported_l2l

    config = type("Config", (), {"model_type": "deepseek_v3"})()
    with pytest.raises(ValueError) as caught:
        refuse_unsupported_l2l(config, n_gpu=1, expert_layout=layout, fp16=fp16)
    assert str(caught.value) == MOE_REFUSAL


def test_the_trainer_refuses_fp16_even_if_per_expert_moe_is_enabled(monkeypatch):
    from soup_cli.trainer.l2l_trainer import refuse_unsupported_l2l
    from soup_cli.utils import l2l_step

    monkeypatch.setattr(l2l_step, "MOE_L2L_ENABLED", True)
    config = type("Config", (), {"model_type": "deepseek_v3"})()
    with pytest.raises(ValueError) as caught:
        refuse_unsupported_l2l(config, n_gpu=1, expert_layout="per_expert", fp16=True)
    assert str(caught.value) == (
        "training.l2l does not support fp16 for model_type 'deepseek_v3': the per-expert "
        "path is not validated under the fp16 loss scale. Use bf16 or fp32."
    )


@pytest.mark.parametrize("arch", ["llama", "mistral", "qwen3_5", "qwen3_5_text"])
@pytest.mark.parametrize("fp16", [False, True])
def test_the_trainer_admits_dense_models(arch, fp16):
    from soup_cli.trainer.l2l_trainer import refuse_unsupported_l2l

    config = type("Config", (), {"model_type": arch})()
    refuse_unsupported_l2l(config, n_gpu=1, fp16=fp16)


# --------------------------------------------------------------------------
# weight restores: "not counted" on a runtime without the counter, never zero
# --------------------------------------------------------------------------
class _StatsRuntime:
    def __init__(self, **stats):
        self._stats = stats

    def stats(self):
        return dict(self._stats)


_BASE_STATS = {"layer_loads": 4, "large_loads": 1, "expert_fetches": 0, "expert_bytes_read": 0}


def test_a_runtime_without_a_restore_counter_reports_none_not_zero():
    from soup_cli.utils.l2l_step import count_loads

    runtime = _StatsRuntime(**_BASE_STATS)
    with count_loads(runtime) as counters:
        runtime._stats["layer_loads"] += 3
    assert counters.layer_loads == 3
    assert counters.restores is None


def test_a_runtime_with_a_restore_counter_reports_its_delta():
    from soup_cli.utils.l2l_step import count_loads

    runtime = _StatsRuntime(layer_restores=2, **_BASE_STATS)
    with count_loads(runtime) as counters:
        runtime._stats["layer_restores"] += 5
    assert counters.restores == 5


def test_adding_counters_keeps_not_counted_apart_from_zero():
    from soup_cli.utils.l2l_step import LoadCounters

    total = LoadCounters()
    total.add(LoadCounters(layer_loads=2))
    assert total.layer_loads == 2 and total.restores is None
    total.add(LoadCounters(restores=3))
    total.add(LoadCounters(restores=None))
    assert total.restores == 3


def test_the_trainer_log_omits_weight_restores_when_they_are_not_counted():
    from soup_cli.trainer.l2l_trainer import L2LState
    from soup_cli.utils.l2l_step import LoadCounters

    state = L2LState(runtime=_StatsRuntime(**_BASE_STATS), dtype=None)
    assert "l2l_weight_restores" not in state.log_fields()
    state.totals = LoadCounters(restores=0)
    assert state.log_fields()["l2l_weight_restores"] == 0
