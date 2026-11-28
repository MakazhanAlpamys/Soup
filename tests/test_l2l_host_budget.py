"""Admission, allocation and shutdown for the public shared host envelope."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from soup_cli.utils.l2l_host_budget import plan_l2l_host_budget
from soup_cli.utils.partial_ram_source import PartialRamBudget


def _plan(weights=None, **changes):
    options = dict(
        n_layers=3, batch_size=1, max_length=65, hidden_size=33,
        microbatches=2, dtype="float32", pin=False,
    )
    options.update(changes)
    return plan_l2l_host_budget(weights or PartialRamBudget(10_000_000, 1234, 45_000), **options)


@pytest.mark.parametrize("pin,expected", [(False, 3 * 2 * 128 * 33 * 4), (True, 3 * 2 * 32768)])
def test_maximum_window_is_reserved_before_cache_including_pinned_rounding(pin, expected):
    plan = _plan(pin=pin)
    assert plan.max_shape == (1, 128, 33)
    assert plan.weights == PartialRamBudget(10_000_000, 1234 + expected, 45_000)
    assert plan.activations.limit_bytes == expected
    assert plan.activations.used_bytes == 0


@pytest.mark.parametrize(
    "changes",
    [
        dict(shapes=[(1, 128, 33)] * 3),
        dict(shapes=[(1, 32, 33), (2, 64, 33)]),
        dict(shapes=[(1, 32, 33), (1, 129, 33)]),
        dict(shapes=[(1, 64, 32)]),
        dict(n_layers=4),
        dict(dtype="bfloat16"),
        dict(pin=True),
    ],
)
def test_windows_cannot_borrow_cache_partition(changes):
    options = dict(n_layers=3, shapes=[(1, 128, 33)], dtype="float32", pin=False)
    options.update(changes)
    with pytest.raises(ValueError) as excinfo:
        _plan().validate_microbatches(**options)
    assert str(excinfo.value) == (
        "training.l2l activation window exceeds or differs from the window "
        "reserved by training.stream_partial_ram"
    )


def test_window_dimensions_reject_boolean_values():
    with pytest.raises(ValueError) as excinfo:
        _plan().validate_microbatches(
            n_layers=3, shapes=[(True, 128, 33)], dtype="float32", pin=False
        )
    assert str(excinfo.value) == "training.l2l activation dimensions must be positive integers"


def test_insufficient_fixed_budget_refuses_before_weight_allocation():
    with pytest.raises(ValueError) as excinfo:
        _plan(PartialRamBudget(100, 50, 20), max_length=64, hidden_size=32)
    assert str(excinfo.value) == (
        "training.stream_partial_ram.limit_bytes=100 cannot cover "
        "50 fixed bytes and 49152 bytes for training.l2l"
    )


@pytest.mark.parametrize(
    "options, message",
    [
        (
            {"batch_size": "auto"},
            "training.stream_partial_ram with training.l2l requires a fixed "
            "positive integer training.batch_size",
        ),
        (
            {"l2l_spill_dir": "spill"},
            "training.l2l_spill_dir cannot be set with training.stream_partial_ram: "
            "the shared budget uses RAM-only activations",
        ),
    ],
)
def test_schema_names_options_that_cannot_use_the_shared_budget(options, message):
    from pydantic import ValidationError

    from soup_cli.config.schema import SoupConfig
    from tests.test_stream_partial_ram import _partial_payload

    with pytest.raises(ValidationError) as excinfo:
        SoupConfig.model_validate(_partial_payload(l2l=True, **options))
    assert [str(error["ctx"]["error"]) for error in excinfo.value.errors()] == [message]


@pytest.mark.parametrize(
    "key,guidance",
    [
        (
            "training.stream_partial_ram",
            "With training.l2l, Soup reserves the maximum RAM activation window in addition "
            "to reserved_bytes for other host consumers. Otherwise reserved_bytes also "
            "covers activations.",
        ),
        (
            "training.stream_partial_ram.reserved_bytes",
            "With training.l2l enabled, Soup reserves the RAM activation window separately. "
            "Otherwise, include activation memory in reserved_bytes.",
        ),
    ],
)
def test_config_reference_explains_activation_reservations(key: str, guidance: str) -> None:
    from scripts import gen_reference_docs as gen

    reference = Path(__file__).resolve().parents[1] / gen.CONFIG_PAGE
    prefix = f"| `{key}` |"
    for page in (gen.render_config_reference(), reference.read_text(encoding="utf-8")):
        row = next(line for line in page.splitlines() if line.startswith(prefix))
        assert guidance in row, row


def test_store_growth_releases_old_lease_and_oversize_retains_current_lease(monkeypatch):
    import torch

    from soup_cli.trainer.l2l_trainer import L2LState, _store_for
    from soup_cli.utils.budgeted_activations import BudgetedActivationStore

    plan = _plan()
    state = L2LState(runtime=None, dtype=torch.float32, host_budget_plan=plan)
    stepper = SimpleNamespace(
        causal=SimpleNamespace(config=SimpleNamespace(hidden_size=33)), layers=[None] * 3
    )
    first = None
    allocate = BudgetedActivationStore.__init__

    def confirm_old_lease_released(store, **kwargs):
        assert plan.activations.used_bytes == 0
        if first is not None:
            assert not first._resident
        allocate(store, **kwargs)

    monkeypatch.setattr(BudgetedActivationStore, "__init__", confirm_old_lease_released)
    try:
        first = _store_for(state, stepper, [torch.empty((1, 32))] * 2, "cpu")
        assert plan.activations.used_bytes == 3 * 2 * 32 * 33 * 4
        second = _store_for(state, stepper, [torch.empty((1, 128))] * 2, "cpu")
        assert first is not second
        assert plan.activations.used_bytes == plan.activations.limit_bytes
        with pytest.raises(ValueError) as excinfo:
            _store_for(state, stepper, [torch.empty((1, 129))] * 2, "cpu")
        assert str(excinfo.value) == (
            "training.l2l activation window exceeds or differs from the window "
            "reserved by training.stream_partial_ram"
        )
        assert state.store is second
        assert plan.activations.used_bytes == plan.activations.limit_bytes
    finally:
        state.close()
    assert plan.activations.used_bytes == 0


def test_failed_activation_drain_prevents_weight_cleanup():
    from soup_cli.trainer.stream_setup import StreamingSetupMixin

    order = []

    def fail():
        order.append("activation")
        raise RuntimeError("drain failed")

    wrapper = SimpleNamespace(
        trainer=SimpleNamespace(_soup_l2l=SimpleNamespace(host_budget_plan=object(), close=fail)),
        _stream_runtime=SimpleNamespace(close=lambda: order.append("weights")),
    )
    with pytest.raises(RuntimeError) as excinfo:
        StreamingSetupMixin._close_stream_runtime(wrapper)
    assert str(excinfo.value) == "drain failed"
    assert order == ["activation"]
    wrapper.trainer._soup_l2l.close = lambda: order.append("activation retry")
    StreamingSetupMixin._close_stream_runtime(wrapper)
    assert order == ["activation", "activation retry", "weights"]


def test_public_setup_and_optimizer_use_reserved_l2l_store(tmp_path, monkeypatch):
    import torch
    from transformers import TrainerCallback

    from soup_cli.config.schema import SoupConfig
    from soup_cli.utils.budgeted_activations import BudgetedActivationStore
    from tests.test_v07200 import TestStreamingEndToEndSetup

    wrapper, dataset = TestStreamingEndToEndSetup()._wrapper(tmp_path, monkeypatch)
    config = wrapper.config.model_dump()
    config["training"].update(
        l2l=True,
        gradient_accumulation_steps=2,
        stream_source="disk",
        stream_disk_kind="nvme",
        stream_pin=False,
        stream_partial_ram=dict(limit_bytes=10_000_000, reserved_bytes=100_000, cache_bytes=45_000),
    )
    wrapper.config = SoupConfig.model_validate(config)
    stores = []

    class ObserveStore(TrainerCallback):
        def on_step_end(self, args, trainer_state, control, **kwargs):
            stores.append(type(state.store))
            assert 0 < plan.activations.used_bytes <= plan.activations.limit_bytes
            assert state.spill_plan == {"n_layers": 2, "spill_layers": 0, "codec": "none"}

    try:
        wrapper.setup(dataset)
        state = wrapper.trainer._soup_l2l
        plan = wrapper._l2l_host_budget_plan
        assert state.host_budget_plan is plan
        source = wrapper._stream_runtime.source
        assert source.budget == plan.weights
        assert source.resident_indices
        assert source.accounted_peak_bytes <= plan.weights.limit_bytes
        assert source.host_budget.reserved_bytes >= 100_000 + plan.activations.limit_bytes
        before = {
            name: param.detach().clone()
            for name, param in wrapper.model.named_parameters()
            if "lora_" in name and param.requires_grad
        }
        wrapper.trainer.add_callback(ObserveStore())
        wrapper.trainer.args.max_steps = 2
        wrapper.trainer.train()
        assert wrapper.trainer.state.global_step == 2
        assert stores == [BudgetedActivationStore, BudgetedActivationStore]
        assert state.store is None
        assert plan.activations.used_bytes == 0
        assert any(
            not torch.equal(before[name], param)
            for name, param in wrapper.model.named_parameters()
            if name in before
        )
    finally:
        wrapper._close_stream_runtime()
    assert source.host_budget.used_bytes == source.host_budget.reserved_bytes


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)])
def test_hf_l2l_updates_match_disk_only(tmp_path, monkeypatch, device):
    import torch
    from transformers import Trainer, TrainingArguments

    from benchmarks.harness import l2l_schedule
    from soup_cli.trainer.l2l_trainer import attach_l2l_state, make_l2l_trainer_class
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model
    from tests.test_l2l_probe_harness import hermetic_ram
    from tests.test_l2l_trainer_labels import _MaskedRows
    from tests.test_v07200 import _tiny_llama_dir, _tiny_lora

    cuda = device.startswith("cuda")
    if cuda and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    hermetic_ram(monkeypatch, tmp_path)
    weights, _, _ = _tiny_llama_dir(tmp_path, n_layers=4, tie=False)
    shards = str(tmp_path / "shards")
    index = shard_checkpoint(weights, shards, dtype="float32", arch="llama")
    lengths = [32, 32, 64, 64, 32, 32]
    results = []
    for cached in (False, True):
        plan = (
            _plan(
                PartialRamBudget(10_000_000, 500, 65_536 if cuda else 45_000),
                n_layers=4, max_length=64, hidden_size=32, pin=cuda,
            )
            if cached else None
        )
        model, runtime = build_streamed_model(
            model_id=weights,
            shard_dir=shards,
            index=index,
            lora_config=_tiny_lora(),
            device=device,
            dtype="float32",
            pin=cuda,
            seed=3,
            tier="disk",
            read_ahead=1,
            partial_ram_budget=plan.weights if plan else None,
        )
        state = None
        try:
            l2l_schedule.make_non_vacuous(model)
            before = l2l_schedule.snapshot_adapters(model)
            args = TrainingArguments(
                output_dir=str(tmp_path / str(cached)),
                per_device_train_batch_size=1,
                gradient_accumulation_steps=2,
                optim="sgd",
                lr_scheduler_type="constant",
                learning_rate=0.01,
                num_train_epochs=1,
                save_strategy="no",
                report_to=[],
                use_cpu=not cuda,
                seed=1,
                data_seed=1,
                max_grad_norm=0,
                disable_tqdm=True,
                logging_steps=1,
            )
            trainer = make_l2l_trainer_class(Trainer)(
                model=model, args=args, train_dataset=_MaskedRows(lengths, [[]] * len(lengths))
            )
            state = attach_l2l_state(trainer, runtime, dtype=torch.float32, host_budget_plan=plan)
            trainer.train()
            assert trainer.state.global_step == 3
            after = l2l_schedule.snapshot_adapters(model)
            assert any(not torch.equal(before[name], after[name]) for name in before)
            losses = [row["loss"] for row in trainer.state.log_history if "loss" in row]
            results.append((after, losses))
            if plan is not None:
                assert 0 < len(runtime.source.resident_indices) < 4
                assert runtime.source.accounted_peak_bytes <= plan.weights.limit_bytes
                assert state.store is None
                assert plan.activations.used_bytes == 0
        finally:
            if state is not None:
                state.close()
            runtime.close()
    assert results[0][1] == results[1][1]
    assert l2l_schedule.compare(results[0][0], results[1][0])["equal"]
