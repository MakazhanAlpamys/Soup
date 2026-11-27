"""DPO on L2L, the reference pass inside the loaded layer (CPU).

TRL 0.29.1 gives a ``PeftModel`` without ``ref_model`` a frozen ``ref`` adapter, a copy of the
policy adapter at trainer build, and computes the reference log-probabilities per micro-batch
with ``use_adapter(model, "ref")`` under ``no_grad``, after the policy pass. The reference of
these tests is that trainer under gradient accumulation, built by the real
``DPOTrainerWrapper.setup()``. The L2L run is the same wrapper with ``l2l: true``. Both start
from the same adapters and the same data order; the LoRA gradient of every optimizer step and
the logged loss must be equal with ``torch.equal``.

Seams:

* ``L2LStep.run(..., reference=ReferencePass(...))``: the second, gradient-free pass through
  the loaded layer (step level, ``test_reference_pass_*``).
* ``dpo_window_loss(args=...)``: how a window of TRL collator batches is read.
* The trainer built by ``DPOTrainerWrapper`` with and without ``training.l2l``.
* The schema refusals and the short-training example.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_l2l_probe_harness import _load
from tests.test_v07204 import _stream_cfg, _tiny_llama_dir, _write_tiny_tokenizer

ROOT = Path(__file__).resolve().parent.parent

#: Pairs of one length: no padding inside a micro-batch.
UNIFORM_ROWS = [
    {"prompt": "hi", "chosen": f" good answer {i}", "rejected": f" bad answer {i}"}
    for i in range(10)
]
#: Pairs of different lengths: chosen longer than rejected, and the other way round.
UNEVEN_ROWS = [
    {"prompt": "hi", "chosen": " good answer the cat sat on mat", "rejected": " bad"},
    {"prompt": "the cat", "chosen": " good", "rejected": " bad answer the cat sat on the mat"},
    {"prompt": "hello world yo", "chosen": " good answer", "rejected": " bad answer the soup"},
    {"prompt": "hi", "chosen": " good answer the soup", "rejected": " bad"},
    {"prompt": "the cat sat", "chosen": " good", "rejected": " bad answer"},
    {"prompt": "yo", "chosen": " good answer the cat", "rejected": " bad answer the cat sat on"},
    {"prompt": "hi", "chosen": " good", "rejected": " bad answer the"},
    {"prompt": "hello", "chosen": " good answer the cat sat", "rejected": " bad"},
    {"prompt": "soup", "chosen": " good answer", "rejected": " bad the cat sat on mat"},
    {"prompt": "the", "chosen": " good the", "rejected": " bad answer the cat"},
]
LORA = {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]}


def _wrapper(tmp_path, monkeypatch, name, *, l2l, k, rows, batch_size=2, **training):
    """A streamed DPO wrapper through the real ``setup()``, on CPU in float32."""
    from soup_cli.trainer.dpo import DPOTrainerWrapper

    root = tmp_path / name
    root.mkdir()
    weights, _model, _config = _tiny_llama_dir(root, n_layers=3, tie=False)
    _write_tiny_tokenizer(weights)
    monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(root / "cache"))
    # TrainingArguments would pick MPS or CUDA by itself; the model here is on the CPU.
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    # TRL's DPO trainer, unlike ORPO's, does not move a batch to the model's device, and
    # ``args.device`` still reads MPS on a Mac whose model sits on the CPU.
    import torch
    from transformers import TrainingArguments

    original_device = TrainingArguments.device.fget

    def cpu_device(self):
        original_device(self)  # still sets up the distributed state
        return torch.device("cpu")

    monkeypatch.setattr(TrainingArguments, "device", property(cpu_device))
    monkeypatch.chdir(root)
    options = {
        "batch_size": batch_size,
        "gradient_accumulation_steps": k,
        "seed": 1,
        "lr": 0.01,
        "optimizer": "sgd",
        "scheduler": "constant",
        "warmup_ratio": 0.0,
        "dpo_beta": 0.3,
        "lora": LORA,
    }
    if l2l:
        options["l2l"] = True
    options.update(training)
    cfg = _stream_cfg(weights, root / "out", task="dpo", **options)
    wrapper = DPOTrainerWrapper(cfg, device="cpu")
    wrapper.setup({"train": list(rows)})
    return wrapper


def _train(wrapper):
    """Train through the trainer; return the LoRA gradient of every optimizer step."""
    from transformers import TrainerCallback

    grads = []

    class Recorder(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
            grads.append(
                {
                    name: p.grad.detach().clone()
                    for name, p in model.named_parameters()
                    if "lora_" in name and p.grad is not None
                }
            )

    wrapper.trainer.add_callback(Recorder())
    wrapper.trainer.train()
    logs = [row for row in wrapper.trainer.state.log_history if "loss" in row]
    return grads, logs


def _assert_equal_runs(ga_result, l2l_result, windows):
    import math

    import torch

    ga_grads, ga_logs = ga_result
    l2l_grads, l2l_logs = l2l_result
    assert len(ga_grads) == len(l2l_grads) == windows
    for gradient_ga, gradient_l2l in zip(ga_grads, l2l_grads):
        assert gradient_ga.keys() == gradient_l2l.keys()
        assert any(bool(g.abs().sum() > 0) for g in gradient_l2l.values())
        for name in gradient_ga:
            assert torch.isfinite(gradient_l2l[name]).all(), name
            assert torch.equal(gradient_ga[name], gradient_l2l[name]), name
    assert all(math.isfinite(row["loss"]) for row in l2l_logs)
    assert [row["loss"] for row in l2l_logs] == [row["loss"] for row in ga_logs]


def _ref_params(model):
    return {
        name: param for name, param in model.named_parameters() if ".ref." in name
    }


def _set_ref_to_policy(model):
    """The reference as TRL builds it for a trained-from-scratch adapter: a copy of the policy."""
    import torch

    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, param in _ref_params(model).items():
            param.copy_(params[name.replace(".ref.", ".default.", 1)])


def _copy_ref(source, target):
    import torch

    target_params = _ref_params(target)
    with torch.no_grad():
        for name, param in _ref_params(source).items():
            target_params[name].copy_(param)


def _pair(tmp_path, monkeypatch, *, k, rows, ref_is_policy=True, tag="", **options):
    """The accumulation run and the L2L run from the same start; returns both results."""
    harness = _load("l2l_schedule")
    ga = _wrapper(tmp_path, monkeypatch, f"ga{tag}", l2l=False, k=k, rows=rows, **options)
    l2l = _wrapper(tmp_path, monkeypatch, f"l2l{tag}", l2l=True, k=k, rows=rows, **options)
    try:
        harness.make_non_vacuous(ga.model)
        start = harness.snapshot_adapters(ga.model)
        harness.restore_adapters(l2l.model, start)
        if ref_is_policy:
            _set_ref_to_policy(ga.model)
        _copy_ref(ga.model, l2l.model)
        ref_before = {n: p.detach().clone() for n, p in _ref_params(l2l.model).items()}
        ga_result = _train(ga)
        l2l_result = _train(l2l)
        return ga, l2l, ga_result, l2l_result, ref_before
    finally:
        for wrapper in (ga, l2l):
            wrapper._close_stream_runtime()


# 10 rows at batch 2 are 5 micro-batches: k=2 gives windows of 2, 2, 1; k=4 gives 4, 1
@pytest.mark.parametrize("ref_is_policy", [True, False])
@pytest.mark.parametrize("k,windows", [(1, 5), (2, 3), (4, 2)])
def test_dpo_l2l_equals_accumulation_for_pairs_of_one_length(
    tmp_path, monkeypatch, k, windows, ref_is_policy
):
    _ga, l2l, ga_result, l2l_result, _ref = _pair(
        tmp_path, monkeypatch, k=k, rows=UNIFORM_ROWS, ref_is_policy=ref_is_policy
    )
    _assert_equal_runs(ga_result, l2l_result, windows)
    assert l2l.trainer._soup_l2l.steps == windows


@pytest.mark.parametrize("k,windows", [(1, 5), (2, 3), (4, 2)])
def test_dpo_l2l_equals_accumulation_for_pairs_of_different_lengths(
    tmp_path, monkeypatch, k, windows
):
    _ga, l2l, ga_result, l2l_result, _ref = _pair(tmp_path, monkeypatch, k=k, rows=UNEVEN_ROWS)
    _assert_equal_runs(ga_result, l2l_result, windows)
    # the micro-batches really differ in shape, so the store and the calls are per shape
    assert len(l2l.trainer._soup_l2l.calls_by_shape) > 1


@pytest.mark.parametrize("k,windows", [(2, 3), (4, 2)])
def test_dpo_l2l_equals_accumulation_with_the_config_default_lora_dropout(
    tmp_path, monkeypatch, k, windows
):
    """lora.dropout 0.05 reaches the adapters, and TRL zeroes it before the first forward."""
    lora = {**LORA, "dropout": 0.05}
    _ga, l2l, ga_result, l2l_result, _ref = _pair(
        tmp_path, monkeypatch, k=k, rows=UNEVEN_ROWS, lora=lora
    )
    _assert_equal_runs(ga_result, l2l_result, windows)
    import torch

    dropouts = [m for m in l2l.model.modules() if isinstance(m, torch.nn.Dropout)]
    assert dropouts and all(m.p == 0 for m in dropouts)


def test_dpo_l2l_equals_accumulation_for_a_batch_of_one_pair(tmp_path, monkeypatch):
    _ga, _l2l, ga_result, l2l_result, _ref = _pair(
        tmp_path, monkeypatch, k=2, rows=UNEVEN_ROWS[:5], batch_size=1
    )
    _assert_equal_runs(ga_result, l2l_result, 3)


def test_only_the_policy_adapter_gets_gradients_and_the_reference_does_not_move(
    tmp_path, monkeypatch
):
    import torch

    _ga, l2l, _ga_result, (grads, _logs), ref_before = _pair(
        tmp_path, monkeypatch, k=2, rows=UNIFORM_ROWS, ref_is_policy=False
    )
    assert grads
    for step in grads:
        assert not [name for name in step if ".ref." in name]
        assert [name for name in step if ".default." in name]
    ref_after = _ref_params(l2l.model)
    assert ref_after.keys() == ref_before.keys() and ref_after
    for name, param in ref_after.items():
        assert not param.requires_grad, name
        assert param.grad is None, name
        assert torch.equal(param.detach(), ref_before[name]), name
    # the policy adapter did train, and is back as the active one
    assert l2l.model.active_adapter == "default"
    assert any(p.requires_grad for n, p in l2l.model.named_parameters() if ".default." in n)


def test_a_reference_that_differs_from_the_policy_changes_the_loss(tmp_path, monkeypatch):
    """The reference pass is not decorative: what the ``ref`` adapter holds reaches the loss."""
    same = _pair(tmp_path, monkeypatch, k=2, rows=UNIFORM_ROWS, ref_is_policy=True, tag="1")
    distinct = _pair(tmp_path, monkeypatch, k=2, rows=UNIFORM_ROWS, ref_is_policy=False, tag="2")
    logs_same, logs_distinct = same[3][1], distinct[3][1]
    assert logs_same[0]["loss"] != logs_distinct[0]["loss"]


def test_the_layer_load_counters_are_in_the_training_log(tmp_path, monkeypatch):
    _ga, _l2l, _ga_result, (_grads, logs), _ref = _pair(
        tmp_path, monkeypatch, k=2, rows=UNIFORM_ROWS
    )[:5]
    assert [row["l2l_steps"] for row in logs] == [1, 2, 3]
    for key in ("l2l_layer_loads", "l2l_large_loads"):
        assert logs[0][key] > 0
        assert all(later[key] > earlier[key] for earlier, later in zip(logs, logs[1:]))
    assert all("l2l_weight_restores" not in row for row in logs)


# --------------------------------------------------------------------------
# the reference adapter on a streamed model
# --------------------------------------------------------------------------
def test_the_reference_adapter_has_real_storage_equal_to_the_policy_adapter_at_start(
    tmp_path, monkeypatch
):
    import torch

    wrapper = _wrapper(tmp_path, monkeypatch, "start", l2l=True, k=2, rows=UNIFORM_ROWS)
    try:
        ref = _ref_params(wrapper.model)
        params = dict(wrapper.model.named_parameters())
        assert ref
        for name, param in ref.items():
            assert not param.is_meta, name
            policy = params[name.replace(".ref.", ".default.", 1)]
            assert not policy.is_meta, name
            assert param.shape == policy.shape
            assert torch.equal(param, policy), name
            assert param.data_ptr() != policy.data_ptr(), name
    finally:
        wrapper._close_stream_runtime()


def test_the_built_trainer_has_the_settings_the_l2l_loss_assumes(tmp_path, monkeypatch):
    """The L2L loss is TRL's sigmoid DPO with the ``ref`` adapter; pin what it relies on."""
    wrapper = _wrapper(tmp_path, monkeypatch, "defaults", l2l=True, k=2, rows=UNIFORM_ROWS)
    try:
        trainer = wrapper.trainer
        assert trainer.loss_types == ["sigmoid"]
        assert trainer.ref_model is None
        assert not trainer.precompute_ref_logps
        assert trainer.ld_alpha is None
        assert trainer.f_divergence_type == "reverse_kl"
        assert "ref" in wrapper.model.peft_config
        assert trainer._soup_l2l.loss.divide_by == "micro_batches"
    finally:
        wrapper._close_stream_runtime()


def test_the_trainer_is_the_l2l_subclass_only_with_l2l_true(tmp_path, monkeypatch):
    from trl import DPOTrainer

    plain = _wrapper(tmp_path, monkeypatch, "plain", l2l=False, k=2, rows=UNIFORM_ROWS)
    streamed = _wrapper(tmp_path, monkeypatch, "streamed", l2l=True, k=2, rows=UNIFORM_ROWS)
    try:
        assert type(plain.trainer) is DPOTrainer
        assert not hasattr(plain.trainer, "soup_l2l")
        assert type(streamed.trainer) is not DPOTrainer
        assert isinstance(streamed.trainer, DPOTrainer)
        assert streamed.trainer.soup_l2l is True
    finally:
        plain._close_stream_runtime()
        streamed._close_stream_runtime()


@pytest.mark.parametrize(
    "attribute,value,match",
    [
        ("loss_types", ["ipo"], "loss_type"),
        ("loss_weights", [2.0], "loss_weights"),
        ("ref_model", object(), "ref_model"),
        ("precompute_ref_logps", True, "precompute_ref_log_probs"),
        ("ld_alpha", 0.5, "ld_alpha"),
        ("f_divergence_type", "forward_kl", "f_divergence_type"),
        ("label_smoothing", 0.1, "label_smoothing"),
        ("use_weighting", True, "use_weighting"),
    ],
)
def test_a_trainer_that_departs_from_plain_sigmoid_dpo_is_refused(attribute, value, match):
    from soup_cli.trainer.l2l_dpo import refuse_unsupported_dpo

    trainer = SimpleNamespace(
        loss_types=["sigmoid"],
        loss_weights=[1.0],
        ref_model=None,
        precompute_ref_logps=False,
        ld_alpha=None,
        f_divergence_type="reverse_kl",
        label_smoothing=0.0,
        use_weighting=False,
        padding_free=False,
    )
    refuse_unsupported_dpo(trainer)
    setattr(trainer, attribute, value)
    with pytest.raises(ValueError, match=match):
        refuse_unsupported_dpo(trainer)


@pytest.mark.parametrize(
    "attribute",
    ["loss_types", "loss_weights", "ref_model", "precompute_ref_logps", "ld_alpha",
     "f_divergence_type", "label_smoothing", "use_weighting"],
)
def test_a_trainer_that_lacks_a_checked_attribute_is_refused(attribute):
    from soup_cli.trainer.l2l_dpo import refuse_unsupported_dpo

    values = dict(
        loss_types=["sigmoid"], loss_weights=[1.0], ref_model=None, precompute_ref_logps=False,
        ld_alpha=None, f_divergence_type="reverse_kl", label_smoothing=0.0, use_weighting=False,
    )
    del values[attribute]
    with pytest.raises(ValueError, match=f"this trainer has no attribute '{attribute}'"):
        refuse_unsupported_dpo(SimpleNamespace(**values))


# --------------------------------------------------------------------------
# the second pass inside the L2L step (step level)
# --------------------------------------------------------------------------
SEQ = 6
ROWS = 4  # two pairs, stacked chosen over rejected


def _ids(k):
    import torch

    generator = torch.Generator().manual_seed(11)
    return [torch.randint(4, 60, (ROWS, SEQ), generator=generator) for _ in range(k)]


class _CountingStore:
    """The shipped store, counting the activations written to it."""

    def __init__(self, k, n_layers=3, hidden=64):
        import torch

        from soup_cli.utils.l2l_activations import ActivationStore

        self.inner = ActivationStore(
            n_layers=n_layers,
            k=k,
            chunk_shape=(ROWS, SEQ, hidden),
            dtype=torch.float32,
            device="cpu",
            pin=False,
        )
        self.k = k
        self.puts = 0

    def put(self, layer, mb, tensor):
        self.puts += 1
        self.inner.put(layer, mb, tensor)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _logits_head(causal, owner, hidden, mb):
    return causal.lm_head(owner.norm(hidden))


@pytest.fixture()
def streamed_dpo(tmp_path, monkeypatch):
    wrapper = _wrapper(tmp_path, monkeypatch, "step", l2l=True, k=2, rows=UNIFORM_ROWS)
    harness = _load("l2l_schedule")
    harness.make_non_vacuous(wrapper.model)  # the policy differs from the reference
    try:
        yield wrapper, harness
    finally:
        wrapper._close_stream_runtime()


def _step(wrapper, ids):
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    return L2LStep(wrapper.model, wrapper._stream_runtime, capture_layer_call(wrapper.model, ids))


def _grads(harness, model):
    return harness.collect_grads(model)


def test_reference_pass_returns_the_logits_of_the_ref_adapter(streamed_dpo):
    import torch
    from trl.trainer.utils import use_adapter

    from soup_cli.utils.l2l_step import ReferencePass

    wrapper, _harness = streamed_dpo
    ids = _ids(2)
    reference = ReferencePass(adapter="ref", head=_logits_head)
    _step(wrapper, ids[0]).run(ids, _CountingStore(2), reference=reference)

    with torch.no_grad():
        policy = [wrapper.model(input_ids=x).logits for x in ids]
        with use_adapter(wrapper.model, adapter_name="ref"):
            expected = [wrapper.model(input_ids=x).logits for x in ids]
    assert sorted(reference.outputs) == [0, 1]
    for mb in range(2):
        assert not reference.outputs[mb].requires_grad
        assert torch.equal(reference.outputs[mb], expected[mb])
        assert not torch.equal(reference.outputs[mb], policy[mb])  # the adapters differ


def test_reference_pass_leaves_the_policy_losses_and_gradients_unchanged(streamed_dpo):
    import torch

    from soup_cli.utils.l2l_step import ReferencePass

    wrapper, harness = streamed_dpo
    ids = _ids(2)
    step = _step(wrapper, ids[0])
    harness.zero_grads(wrapper.model)
    plain_losses = step.run(ids, _CountingStore(2))
    plain = _grads(harness, wrapper.model)

    harness.zero_grads(wrapper.model)
    losses = step.run(
        ids, _CountingStore(2), reference=ReferencePass(adapter="ref", head=_logits_head)
    )
    with_reference = _grads(harness, wrapper.model)

    assert all(torch.equal(a, b) for a, b in zip(plain_losses, losses))
    assert plain.keys() == with_reference.keys()
    assert all(torch.equal(plain[name], with_reference[name]) for name in plain)


def test_reference_pass_writes_nothing_to_the_activation_store(streamed_dpo):
    from soup_cli.utils.l2l_step import ReferencePass

    wrapper, _harness = streamed_dpo
    ids = _ids(2)
    step = _step(wrapper, ids[0])
    plain = _CountingStore(2)
    step.run(ids, plain)
    with_reference = _CountingStore(2)
    step.run(
        ids, with_reference, reference=ReferencePass(adapter="ref", head=_logits_head)
    )
    assert plain.puts == with_reference.puts == 3 * 2  # layers x micro-batches


def test_reference_pass_gives_the_adapters_back_as_it_found_them(streamed_dpo):
    from soup_cli.utils.l2l_step import ReferencePass

    wrapper, harness = streamed_dpo
    ids = _ids(2)
    before = {n: p.requires_grad for n, p in wrapper.model.named_parameters()}
    harness.zero_grads(wrapper.model)
    _step(wrapper, ids[0]).run(
        ids, _CountingStore(2), reference=ReferencePass(adapter="ref", head=_logits_head)
    )
    assert wrapper.model.active_adapter == "default"
    assert {n: p.requires_grad for n, p in wrapper.model.named_parameters()} == before
    assert all(p.grad is None for n, p in _ref_params(wrapper.model).items())


def test_reference_pass_is_refused_when_the_layers_draw_dropout(streamed_dpo):
    import dataclasses

    from soup_cli.utils.l2l_step import ReferencePass

    wrapper, _harness = streamed_dpo
    ids = _ids(2)
    step = _step(wrapper, ids[0])
    step.call = dataclasses.replace(step.call, draws=((object(),), ()))
    with pytest.raises(ValueError, match="dropout"):
        step.run(
            ids, _CountingStore(2), reference=ReferencePass(adapter="ref", head=_logits_head)
        )


def _warm_walk(model, ids):
    """Loads depend on what the previous walk left in the slots; start every measurement
    after the same forward-only walk."""
    from soup_cli.utils.l2l_step import capture_layer_call

    capture_layer_call(model, ids)


@pytest.mark.parametrize("k", [1, 2, 4])
def test_a_dpo_step_loads_each_layer_as_often_as_an_sft_step(streamed_dpo, k):
    """The reference pass adds no layer load and no full read of the weights.

    Both numbers come from the runtime's own counters (``LoadCounters``): the SFT step and
    the DPO step walk the same model with the same k micro-batches.
    """
    from soup_cli.utils.l2l_step import ReferencePass

    wrapper, harness = streamed_dpo
    ids = _ids(k)
    step = _step(wrapper, ids[0])

    harness.zero_grads(wrapper.model)
    _warm_walk(wrapper.model, ids[0])
    step.run(ids, _CountingStore(k))
    sft = step.counters

    harness.zero_grads(wrapper.model)
    _warm_walk(wrapper.model, ids[0])
    step.run(
        ids, _CountingStore(k), reference=ReferencePass(adapter="ref", head=_logits_head)
    )
    dpo = step.counters

    assert sft.layer_loads > 0
    assert dpo.layer_loads == sft.layer_loads
    assert dpo.large_loads <= sft.large_loads
    assert dpo.restores is None
    assert sft.restores is None


def test_a_dpo_trainer_window_loads_as_often_as_an_sft_window(streamed_dpo):
    """The same count through the trainer's own path (``run_window``)."""
    import torch

    from soup_cli.trainer.l2l_trainer import SFT_WINDOW_LOSS, run_window
    from soup_cli.utils.l2l_step import label_tokens

    wrapper, harness = streamed_dpo
    state = wrapper.trainer._soup_l2l
    batches = [_collated([[1, 5, 6, 7]], [[1, 9]]) for _ in range(2)]
    device = wrapper._stream_runtime.device
    dpo_ids, dpo_extras = state.loss.read_window(
        batches, device, neftune=False, has_dropout=False
    )

    harness.zero_grads(wrapper.model)
    _warm_walk(wrapper.model, dpo_ids[0])
    dpo_losses, dpo = run_window(state, wrapper.model, dpo_ids, dpo_extras, None, device)

    dpo_state_loss = state.loss
    state.loss = SFT_WINDOW_LOSS
    try:
        harness.zero_grads(wrapper.model)
        # A cache miss primes layer 0 and the embedding before the step counters start.
        # Give both windows that same capture after the warm walk.
        state.calls_by_shape.clear()
        _warm_walk(wrapper.model, dpo_ids[0])
        _sft_losses, sft = run_window(
            state, wrapper.model, dpo_ids, dpo_ids, label_tokens(dpo_ids), device
        )
    finally:
        state.loss = dpo_state_loss
    assert torch.isfinite(torch.stack(dpo_losses)).all()
    assert dpo.layer_loads == sft.layer_loads > 0
    assert dpo.large_loads <= sft.large_loads
    assert dpo.restores is None
    assert sft.restores is None


# --------------------------------------------------------------------------
# reading a micro-batch: TRL's collator, truncation by TRL's own rule
# --------------------------------------------------------------------------
def _collated(chosen, rejected, prompt=(1, 2)):
    """What TRL's ``DataCollatorForPreference`` hands the trainer, built by that class."""
    from trl.trainer.dpo_trainer import DataCollatorForPreference

    examples = [
        {"prompt_ids": list(prompt), "chosen_ids": c, "rejected_ids": r}
        for c, r in zip(chosen, rejected)
    ]
    return DataCollatorForPreference(pad_token_id=3)(examples)


def _args(max_length=None, truncation_mode="keep_start", beta=0.3):
    return SimpleNamespace(beta=beta, max_length=max_length, truncation_mode=truncation_mode)


def test_the_window_reader_hands_the_collator_batch_to_the_layers_as_it_is():
    import torch

    from soup_cli.trainer.l2l_dpo import dpo_window_loss

    strategy = dpo_window_loss(args=_args())
    batch = _collated(chosen=[[5, 6, 7], [8]], rejected=[[9], [4, 5]])
    ids_list, _extras = strategy.read_window([batch], "cpu", neftune=False, has_dropout=False)
    assert len(ids_list) == 1
    assert torch.equal(ids_list[0], batch["input_ids"])
    assert tuple(ids_list[0].shape) == (4, 5)


def test_the_window_reader_gives_each_micro_batch_its_own_width():
    from soup_cli.trainer.l2l_dpo import dpo_window_loss

    strategy = dpo_window_loss(args=_args())
    short = _collated([[5]], [[9]])
    long = _collated([[5, 6, 7, 8]], [[9]])
    ids_list, _ = strategy.read_window([short, long], "cpu", neftune=False, has_dropout=False)
    assert [tuple(ids.shape) for ids in ids_list] == [(2, 3), (2, 6)]


@pytest.mark.parametrize("mode", ["keep_start", "keep_end"])
def test_the_window_reader_truncates_as_trl_does(mode):
    from trl.trainer.dpo_trainer import DPOTrainer

    from soup_cli.trainer.l2l_dpo import dpo_window_loss

    batch = _collated(chosen=[[5, 6, 7, 8, 9]], rejected=[[4, 5]])
    expected_ids, _mask, _completion = DPOTrainer._truncate_inputs(
        SimpleNamespace(args=_args(max_length=4, truncation_mode=mode)),
        batch["input_ids"],
        batch["attention_mask"],
        batch["completion_mask"],
    )
    strategy = dpo_window_loss(args=_args(max_length=4, truncation_mode=mode))
    ids_list, _ = strategy.read_window([batch], "cpu", neftune=False, has_dropout=False)
    assert tuple(ids_list[0].shape) == (2, 4)
    assert (ids_list[0] == expected_ids).all()


def test_the_window_reader_refuses_left_padding():
    from soup_cli.trainer.l2l_dpo import dpo_window_loss

    strategy = dpo_window_loss(args=_args())
    batch = _collated(chosen=[[5, 6, 7], [8]], rejected=[[9], [4, 5]])
    batch["attention_mask"][1] = batch["attention_mask"][1].flip(0)
    with pytest.raises(ValueError, match="right-padded"):
        strategy.read_window([batch], "cpu", neftune=False, has_dropout=False)


def test_the_window_reader_keeps_the_mixed_shape_refusals_of_sft():
    from soup_cli.trainer.l2l_dpo import dpo_window_loss

    strategy = dpo_window_loss(args=_args())
    window = [_collated([[5]], [[9]]), _collated([[5, 6, 7]], [[9]])]
    with pytest.raises(ValueError, match="neftune_alpha"):
        strategy.read_window(window, "cpu", neftune=True, has_dropout=False)
    with pytest.raises(ValueError, match="dropout"):
        strategy.read_window(window, "cpu", neftune=False, has_dropout=True)


def test_dpo_divides_the_window_by_k_and_needs_no_item_count():
    from soup_cli.trainer.l2l_dpo import dpo_window_loss

    strategy = dpo_window_loss(args=_args())
    assert strategy.divide_by == "micro_batches"
    assert strategy.divisor(3, None) == 3


# --------------------------------------------------------------------------
# refusals in the schema
# --------------------------------------------------------------------------
def _l2l_yaml(task="dpo", lora_dropout=0.0, **training):
    import yaml

    return yaml.safe_dump(
        {
            "base": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "task": task,
            "data": {"train": "data.jsonl", "format": "dpo"},
            "training": {
                "batch_size": 1,
                "quantization": "none",
                "stream_layers": True,
                "l2l": True,
                "gradient_accumulation_steps": 2,
                "lora": {"r": 4, "alpha": 8, "dropout": lora_dropout},
                **training,
            },
        }
    )


def test_the_schema_accepts_dpo_with_l2l():
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.config.schema import L2L_SUPPORTED_TASKS

    assert "dpo" in L2L_SUPPORTED_TASKS
    cfg = load_config_from_string(_l2l_yaml())
    assert cfg.task == "dpo" and cfg.training.l2l is True


def test_the_schema_still_refuses_ipo_with_l2l():
    from soup_cli.config.loader import load_config_from_string

    with pytest.raises(ValueError, match="got task='ipo'"):
        load_config_from_string(_l2l_yaml(task="ipo"))


def test_the_schema_refuses_dpo_with_l2l_when_streaming_is_off():
    from soup_cli.config.loader import load_config_from_string

    text = _l2l_yaml().replace("stream_layers: true", "stream_layers: false")
    with pytest.raises(ValueError, match="requires training.stream_layers"):
        load_config_from_string(text)


_BETA_FIXED = (
    "training.l2l with task='dpo' keeps beta fixed: remove "
    "training.dpo_beta_schedule and training.dpo_beta_end."
)
_REF_FIXED = (
    "training.l2l with task='dpo' keeps the reference fixed: remove "
    "training.dpo_ref_regen_epochs, which would overwrite the 'ref' adapter between steps."
)
_NEFTUNE = (
    "training.l2l with task='dpo' does not support training.neftune_alpha: the reference "
    "forward draws NEFTune noise between the policy forwards, an order the layer-major walk "
    "does not reproduce."
)


@pytest.mark.parametrize(
    "extra,message",
    [
        ({"dpo_beta_schedule": "linear", "dpo_beta_end": 0.05}, _BETA_FIXED),
        ({"dpo_beta_schedule": "linear"}, _BETA_FIXED),
        ({"dpo_beta_end": 0.05}, _BETA_FIXED),
        ({"dpo_ref_regen_epochs": 1}, _REF_FIXED),
        ({"neftune_alpha": 5.0}, _NEFTUNE),
    ],
)
def test_the_schema_refuses_what_changes_the_reference_between_steps(extra, message):
    from soup_cli.config.loader import load_config_from_string

    with pytest.raises(ValueError) as caught:
        load_config_from_string(_l2l_yaml(**extra))
    assert message in str(caught.value)


def test_the_schema_accepts_the_default_lora_dropout_with_dpo_and_l2l():
    """TRL's DPOTrainer zeroes every nn.Dropout, so the config default is harmless."""
    from soup_cli.config.loader import load_config_from_string

    cfg = load_config_from_string(_l2l_yaml(lora_dropout=0.05))
    assert cfg.training.lora.dropout == 0.05 and cfg.training.l2l is True


def test_dpo_without_l2l_keeps_its_variants_and_dropout():
    from soup_cli.config.loader import load_config_from_string

    text = _l2l_yaml(
        lora_dropout=0.05, dpo_beta_schedule="linear", dpo_beta_end=0.05, neftune_alpha=5.0
    ).replace("l2l: true", "l2l: false")
    cfg = load_config_from_string(text)
    assert cfg.training.dpo_beta_schedule == "linear"


@pytest.mark.parametrize("task", ["kto", "orpo", "simpo"])
def test_the_tasks_that_are_not_ready_stay_refused_with_l2l(task):
    from soup_cli.config.loader import load_config_from_string

    pattern = rf"training\.l2l supports task in \['sft', 'dpo'\]; got task='{task}'"
    with pytest.raises(ValueError, match=pattern):
        load_config_from_string(_l2l_yaml(task=task, batch_size=2 if task == "kto" else 1))


# --------------------------------------------------------------------------
# launch guards, through the DPO wrapper: each refuses before any model load
# --------------------------------------------------------------------------
class _ReachedError(Exception):
    """The streamed model load, which these tests replace."""


def _launch_wrapper(monkeypatch, **training):
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.dpo import DPOTrainerWrapper

    wrapper = DPOTrainerWrapper.__new__(DPOTrainerWrapper)
    wrapper.config = load_config_from_string(_l2l_yaml(**training))
    wrapper.device = "cpu"

    def reached(self, cfg, tcfg):
        raise _ReachedError

    monkeypatch.setattr(DPOTrainerWrapper, "_setup_streaming_transformers", reached)
    return wrapper


_DISTRIBUTED = (
    "training.l2l cannot run on a distributed layer stream: the L2L step runs its "
    "own backward, so the adapter-gradient average between ranks would be skipped. "
    "Run one process, or set training.l2l to false."
)


def test_a_clean_l2l_config_resolves_the_recipe_before_the_model_load(monkeypatch):
    wrapper = _launch_wrapper(monkeypatch, gradient_accumulation_steps=4)
    with pytest.raises(_ReachedError):
        wrapper.setup({})
    assert (wrapper._l2l_recipe.requested_k, wrapper._l2l_recipe.effective_k) == (4, 4)


def test_without_l2l_the_dpo_wrapper_resolves_no_recipe(monkeypatch):
    wrapper = _launch_wrapper(monkeypatch)
    wrapper.config.training.l2l = False
    with pytest.raises(_ReachedError):
        wrapper.setup({})
    assert wrapper._l2l_recipe is None


def test_fp8_is_refused_by_the_dpo_wrapper_before_the_model_load(monkeypatch):
    wrapper = _launch_wrapper(monkeypatch, l2l_activation_codec="fp8_e4m3")
    with pytest.raises(ValueError) as caught:
        wrapper.setup({})
    assert str(caught.value) == (
        "training.l2l_activation_codec='fp8_e4m3' is not available yet: the launch "
        "supports 'none'. Set it to 'none'."
    )


def test_auto_k_is_refused_by_the_dpo_wrapper_before_the_model_load(monkeypatch):
    wrapper = _launch_wrapper(monkeypatch, l2l_auto_k=True)
    with pytest.raises(ValueError, match="l2l_auto_k is not available yet for this model"):
        wrapper.setup({})


def test_a_spill_dir_outside_the_roots_is_refused_by_the_dpo_wrapper_and_never_swept(
    monkeypatch, tmp_path
):
    from soup_cli.utils import l2l_spill

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    monkeypatch.setattr(l2l_spill, "_allowed_roots", lambda: (str(allowed),))
    swept = []
    monkeypatch.setattr(l2l_spill, "sweep_stale", lambda *args, **kwargs: swept.append(args))
    wrapper = _launch_wrapper(monkeypatch, l2l_spill_dir=str(outside))
    with pytest.raises(ValueError, match=r"training\.l2l_spill_dir is refused: .*outside \$HOME"):
        wrapper.setup({})
    assert not outside.exists()
    assert swept == []


def test_l2l_is_refused_by_the_dpo_wrapper_in_a_multi_process_launch(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    wrapper = _launch_wrapper(monkeypatch)
    with pytest.raises(ValueError) as caught:
        wrapper.setup({})
    assert str(caught.value) == _DISTRIBUTED


def test_l2l_is_refused_by_the_dpo_wrapper_when_the_stream_is_distributed(monkeypatch):
    wrapper = _launch_wrapper(monkeypatch)
    wrapper._distributed_stream = True
    with pytest.raises(ValueError) as caught:
        wrapper.setup({})
    assert str(caught.value) == _DISTRIBUTED


# --------------------------------------------------------------------------
# the recipe: written by wrapper.train(), checked on resume
# --------------------------------------------------------------------------
def _train_wrapper(monkeypatch, tmp_path):
    from soup_cli.utils.l2l_recipe import resolve_l2l_recipe

    wrapper = _launch_wrapper(monkeypatch, gradient_accumulation_steps=2)
    monkeypatch.chdir(tmp_path)
    wrapper._output_dir = str(tmp_path / "out")
    (tmp_path / "out").mkdir()
    wrapper._batch_size = 1
    wrapper.trainer = SimpleNamespace()
    wrapper._l2l_recipe = resolve_l2l_recipe(wrapper.config.training)

    def stop(**kwargs):
        raise _ReachedError

    # the first thing wrapper.train() does after the recipe, so the test stops there
    monkeypatch.setattr("soup_cli.utils.dpo_variants.build_dpo_variant_callbacks", stop)
    return wrapper


def test_wrapper_train_writes_the_l2l_recipe(monkeypatch, tmp_path):
    import json

    wrapper = _train_wrapper(monkeypatch, tmp_path)
    with pytest.raises(_ReachedError):
        wrapper.train()
    saved = json.loads((tmp_path / "out" / "l2l_recipe.json").read_text(encoding="utf-8"))
    assert saved["effective_k"] == 2 and saved["activation_codec"] == "none"


def test_wrapper_train_refuses_a_resume_whose_saved_recipe_differs(monkeypatch, tmp_path):
    import dataclasses

    from soup_cli.utils.l2l_recipe import write_recipe

    wrapper = _train_wrapper(monkeypatch, tmp_path)
    old = dataclasses.replace(wrapper._l2l_recipe, requested_k=4, effective_k=4)
    path = write_recipe(wrapper._output_dir, old)
    before = Path(path).read_text(encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        wrapper.train(resume_from_checkpoint=str(tmp_path / "out" / "checkpoint-1"))
    assert str(caught.value) == (
        "cannot resume: the checkpoint was trained with effective_k=4, but this run has "
        "effective_k=2. Restore the old value in the config."
    )
    assert Path(path).read_text(encoding="utf-8") == before


def _other_run_checkpoint(tmp_path, recipe=None):
    """A checkpoint in run A (``tmp_path/runA``); its recipe is written there when given."""
    from soup_cli.utils.l2l_recipe import write_recipe

    run = tmp_path / "runA"
    (run / "checkpoint-1").mkdir(parents=True)
    if recipe is not None:
        write_recipe(run, recipe)
    return str(run / "checkpoint-1")


def test_wrapper_train_checks_the_recipe_of_the_source_checkpoint_not_the_output(
    monkeypatch, tmp_path
):
    import dataclasses

    from soup_cli.utils.l2l_recipe import write_recipe

    wrapper = _train_wrapper(monkeypatch, tmp_path)
    # the destination run B holds a MATCHING recipe; the source run A holds another k
    write_recipe(wrapper._output_dir, wrapper._l2l_recipe)
    old = dataclasses.replace(wrapper._l2l_recipe, requested_k=4, effective_k=4)
    checkpoint = _other_run_checkpoint(tmp_path, old)
    with pytest.raises(ValueError) as caught:
        wrapper.train(resume_from_checkpoint=checkpoint)
    assert str(caught.value) == (
        "cannot resume: the checkpoint was trained with effective_k=4, but this run has "
        "effective_k=2. Restore the old value in the config."
    )


def test_wrapper_train_accepts_a_source_checkpoint_with_a_matching_recipe(
    monkeypatch, tmp_path
):
    import json

    wrapper = _train_wrapper(monkeypatch, tmp_path)
    checkpoint = _other_run_checkpoint(tmp_path, wrapper._l2l_recipe)
    with pytest.raises(_ReachedError):
        wrapper.train(resume_from_checkpoint=checkpoint)
    saved = json.loads((tmp_path / "out" / "l2l_recipe.json").read_text(encoding="utf-8"))
    assert saved["effective_k"] == 2


def test_wrapper_train_refuses_resuming_an_l2l_checkpoint_with_l2l_off(monkeypatch, tmp_path):
    wrapper = _train_wrapper(monkeypatch, tmp_path)
    checkpoint = _other_run_checkpoint(tmp_path, wrapper._l2l_recipe)
    wrapper._l2l_recipe = None
    with pytest.raises(ValueError) as caught:
        wrapper.train(resume_from_checkpoint=checkpoint)
    assert str(caught.value) == (
        "cannot resume: the checkpoint was trained with training.l2l (effective_k=2, "
        "activation_codec='none'), but this run has training.l2l off. Set training.l2l to "
        "true with the same settings."
    )
    assert not (tmp_path / "out" / "l2l_recipe.json").exists()


def test_wrapper_train_accepts_a_checkpoint_without_a_recipe_with_l2l_on_or_off(
    monkeypatch, tmp_path
):
    wrapper = _train_wrapper(monkeypatch, tmp_path)
    checkpoint = _other_run_checkpoint(tmp_path)
    with pytest.raises(_ReachedError):
        wrapper.train(resume_from_checkpoint=checkpoint)
    wrapper._l2l_recipe = None
    with pytest.raises(_ReachedError):
        wrapper.train(resume_from_checkpoint=checkpoint)


# --------------------------------------------------------------------------
# the logits reach the DPO log-probabilities in the dtype the ordinary forward returns
# --------------------------------------------------------------------------
def _head_parts():
    import torch

    torch.manual_seed(0)
    lm_head = torch.nn.Linear(8, 16, bias=False)
    causal = SimpleNamespace(lm_head=lm_head)
    owner = SimpleNamespace(norm=torch.nn.Identity())
    ids = torch.randint(0, 16, (4, 6))
    mask = torch.tensor([[0, 0, 1, 1, 1]] * 4)
    labels = (ids[..., 1:].contiguous(), mask)
    return causal, owner, labels


def _dpo_loss_under_autocast(causal, owner, labels, hidden, ref_hidden, *, ordinary):
    import torch

    from soup_cli.trainer.l2l_dpo import _completion_logps_of, dpo_loss

    with torch.autocast("cpu", dtype=torch.bfloat16):
        if not ordinary:
            policy = _completion_logps_of(causal, owner, hidden, labels)
            ref = _completion_logps_of(causal, owner, ref_hidden, labels).chunk(2, dim=0)
        else:
            from trl.trainer.utils import selective_log_softmax

            def logps(source):
                logits = causal.lm_head(source).float()  # what the ordinary forward returns
                per_token = selective_log_softmax(logits[..., :-1, :].contiguous(), labels[0])
                per_token[labels[1] == 0] = 0.0
                return per_token.sum(dim=1)

            policy = logps(hidden)
            ref = logps(ref_hidden).chunk(2, dim=0)
    return dpo_loss(policy, ref, 0.3)


def test_dpo_loss_and_gradients_under_bf16_autocast_equal_the_float32_logits_computation():
    import torch

    causal, owner, labels = _head_parts()
    base = torch.randn(4, 6, 8)
    ref_hidden = torch.randn(4, 6, 8)
    results = []
    for ordinary in (False, True):
        causal.lm_head.weight.grad = None
        hidden = base.clone().requires_grad_(True)
        loss = _dpo_loss_under_autocast(
            causal, owner, labels, hidden, ref_hidden, ordinary=ordinary
        )
        loss.backward()
        results.append((loss.detach(), hidden.grad.clone(), causal.lm_head.weight.grad.clone()))
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])
    assert torch.equal(results[0][2], results[1][2])


def test_the_logits_reach_selective_log_softmax_as_float32_under_autocast(monkeypatch):
    import torch
    import trl.trainer.utils as trl_utils

    from soup_cli.trainer.l2l_dpo import _completion_logps_of

    seen = []
    original = trl_utils.selective_log_softmax

    def recording(logits, index):
        seen.append(logits.dtype)
        return original(logits, index)

    monkeypatch.setattr(trl_utils, "selective_log_softmax", recording)
    causal, owner, labels = _head_parts()
    hidden = torch.randn(4, 6, 8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert causal.lm_head(hidden).dtype == torch.bfloat16  # the raw logits are half precision
        _completion_logps_of(causal, owner, hidden, labels)
    assert seen == [torch.float32]


def test_the_logits_dtype_is_left_alone_without_autocast(monkeypatch):
    import torch
    import trl.trainer.utils as trl_utils

    from soup_cli.trainer.l2l_dpo import _completion_logps_of

    seen = []
    original = trl_utils.selective_log_softmax

    def recording(logits, index):
        seen.append(logits.dtype)
        return original(logits, index)

    monkeypatch.setattr(trl_utils, "selective_log_softmax", recording)
    causal, owner, labels = _head_parts()
    causal.lm_head = causal.lm_head.to(torch.bfloat16)
    _completion_logps_of(causal, owner, torch.randn(4, 6, 8, dtype=torch.bfloat16), labels)
    assert seen == [torch.bfloat16]  # the ordinary forward returns the model dtype here too


def test_a_legacy_checkpoint_resumed_with_l2l_keeps_its_provenance_in_the_same_output(
    monkeypatch, tmp_path
):
    import json

    wrapper = _train_wrapper(monkeypatch, tmp_path)
    out = tmp_path / "out"
    (out / "checkpoint-3").mkdir()
    legacy = str(out / "checkpoint-3")
    # legacy checkpoint-3 (no recipe anywhere) resumed with l2l on: the recipe records the start
    with pytest.raises(_ReachedError):
        wrapper.train(resume_from_checkpoint=legacy)
    saved = json.loads((out / "l2l_recipe.json").read_text(encoding="utf-8"))
    assert saved["start_step"] == 3
    (out / "checkpoint-5").mkdir()
    # the same older checkpoint again with l2l off is still a non-L2L checkpoint
    wrapper._l2l_recipe = None
    with pytest.raises(_ReachedError):
        wrapper.train(resume_from_checkpoint=legacy)
    # a checkpoint the L2L run wrote afterwards is L2L, and refused with l2l off
    with pytest.raises(ValueError) as caught:
        wrapper.train(resume_from_checkpoint=str(out / "checkpoint-5"))
    assert str(caught.value) == (
        "cannot resume: the checkpoint was trained with training.l2l (effective_k=2, "
        "activation_codec='none'), but this run has training.l2l off. Set training.l2l to "
        "true with the same settings."
    )


@pytest.mark.parametrize("task", ["dpo", "sft"])
def test_same_output_checkpoint_saves_preserve_the_training_mode(monkeypatch, tmp_path, task):
    from dataclasses import replace

    from soup_cli.utils.l2l_recipe import load_recipe, resolve_l2l_recipe

    pytest.importorskip("transformers")
    if task == "dpo":
        wrapper = _launch_wrapper(monkeypatch, gradient_accumulation_steps=2)
        monkeypatch.setattr("soup_cli.trainer.dpo.reload_root_adapter_on_resume", lambda _: None)
    else:
        from tests.test_l2l_sft_dispatch import _wrapper as sft_wrapper

        wrapper = sft_wrapper(monkeypatch, l2l=True)
        wrapper._quest_metadata = None
        wrapper._stream_loss_chunk_hook = None

    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    legacy = out / "checkpoint-3"
    legacy.mkdir(parents=True)
    wrapper._output_dir = str(out)
    original_recipe = resolve_l2l_recipe(wrapper.config.training)
    callbacks = []
    args = SimpleNamespace(output_dir=str(out), should_save=True)
    save_step = 5

    def train(*, resume_from_checkpoint):
        assert Path(resume_from_checkpoint).is_dir()
        if save_step is not None:
            (out / f"checkpoint-{save_step}").mkdir()
            state = SimpleNamespace(global_step=save_step)
            for callback in callbacks:
                callback.on_save(args, state, None)
        raise _ReachedError

    # Simulate the Trainer's save event after the checkpoint directory has been written.
    monkeypatch.setattr(f"soup_cli.trainer.{task}.manual_save_context", lambda _: nullcontext())
    wrapper.trainer = SimpleNamespace(
        args=args,
        model=None,
        train=train,
        add_callback=callbacks.append,
        remove_callback=callbacks.remove,
    )

    wrapper._l2l_recipe = original_recipe
    with pytest.raises(_ReachedError):
        wrapper.train(resume_from_checkpoint=str(legacy))
    assert load_recipe(out / "checkpoint-5") == replace(original_recipe, start_step=3)
    assert not callbacks

    wrapper._l2l_recipe = None
    save_step = 4
    with pytest.raises(_ReachedError):
        wrapper.train(resume_from_checkpoint=str(legacy))
    assert not callbacks

    ordinary = out / "checkpoint-4"
    save_step = None
    with pytest.raises(_ReachedError):
        wrapper.train(resume_from_checkpoint=str(ordinary))
    with pytest.raises(ValueError) as caught:
        wrapper.train(resume_from_checkpoint=str(out / "checkpoint-5"))
    assert str(caught.value) == (
        "cannot resume: the checkpoint was trained with training.l2l (effective_k=2, "
        "activation_codec='none'), but this run has training.l2l off. Set training.l2l to "
        "true with the same settings."
    )

    wrapper._l2l_recipe = replace(original_recipe, requested_k=4, effective_k=4)
    save_step = 6
    with pytest.raises(_ReachedError):
        wrapper.train(resume_from_checkpoint=str(ordinary))
    assert load_recipe(out).effective_k == 4
    with pytest.raises(ValueError) as caught:
        wrapper.train(resume_from_checkpoint=str(out / "checkpoint-5"))
    assert str(caught.value) == (
        "cannot resume: the checkpoint was trained with effective_k=2, but this run has "
        "effective_k=4. Restore the old value in the config."
    )
    assert load_recipe(out / "checkpoint-6").effective_k == 4


@pytest.mark.parametrize("old_recipe_exists", [False, True])
def test_publishing_a_recipe_preserves_retained_checkpoints(
    monkeypatch, tmp_path, old_recipe_exists
):
    from dataclasses import replace

    from soup_cli.utils.l2l_recipe import load_recipe, write_recipe

    wrapper = _train_wrapper(monkeypatch, tmp_path)
    out = tmp_path / "out"
    legacy = out / "checkpoint-3"
    legacy.mkdir()
    retained = out / "checkpoint-5"
    retained.mkdir()
    if old_recipe_exists:
        previous = replace(wrapper._l2l_recipe, requested_k=4, effective_k=4, start_step=3)
        write_recipe(out, previous)

    with pytest.raises(_ReachedError):
        wrapper.train(resume_from_checkpoint=str(legacy))
    assert load_recipe(out).effective_k == 2
    if old_recipe_exists:
        with pytest.raises(ValueError) as caught:
            wrapper.train(resume_from_checkpoint=str(retained))
        assert str(caught.value) == (
            "cannot resume: the checkpoint was trained with effective_k=4, but this run has "
            "effective_k=2. Restore the old value in the config."
        )
    else:
        wrapper._l2l_recipe = None
        with pytest.raises(_ReachedError):
            wrapper.train(resume_from_checkpoint=str(retained))
