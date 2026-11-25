"""Portable contracts for the native SFT loss watchdog (#1989)."""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from soup_cli.config.loader import load_config_from_string


def test_public_config_accepts_mlx_sft_watchdog():
    cfg = load_config_from_string("""
base: ./local-model
backend: mlx
task: sft
data:
  train: ./train.jsonl
training:
  quantization: none
  loss_watchdog: true
""")
    assert cfg.training.loss_watchdog is True


@pytest.mark.parametrize("values, patience, expected", [
    ([3, 2, 3.1], 1, [False, False, True]),
    ([4, 3, 4, 4], 2, [False, False, False, True]),
    ([4, None, 4], 2, [False, False, True]),
    ([4, 2, 4, 2, 4, 4, 4], 3, [False] * 6 + [True]),
])
def test_report_threshold_patience_and_reset(values, patience, expected):
    from soup_cli.trainer.mlx_sft import _LossWatchdog

    watchdog = _LossWatchdog(3.0, patience)
    assert [watchdog.observe(value) for value in values] == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_loss_is_an_error(value):
    from soup_cli.trainer.mlx_sft import _LossWatchdog

    with pytest.raises(FloatingPointError, match="non-finite.*loss"):
        _LossWatchdog(3.0, 1).observe(value)


@pytest.mark.parametrize("field, value", [
    ("loss_watchdog_threshold", 0), ("loss_watchdog_threshold", -1),
    ("loss_watchdog_threshold", float("nan")),
    ("loss_watchdog_threshold", float("inf")),
    ("loss_watchdog_patience", 0), ("loss_watchdog_patience", 1001),
])
def test_invalid_watchdog_bounds(field, value):
    from soup_cli.config.schema import TrainingConfig

    with pytest.raises(ValueError, match=field):
        TrainingConfig(**{field: value})


@pytest.mark.parametrize("task", ["dpo", "orpo"])
def test_other_native_tasks_still_refuse_watchdog(task):
    from soup_cli.config.schema import SoupConfig

    with pytest.raises(ValueError, match="loss_watchdog.*MLX text SFT"):
        SoupConfig(base="./model", backend="mlx", task=task,
                   data={"train": "./data.jsonl"}, training={"loss_watchdog": True})


def test_nontext_sft_still_refuses_watchdog():
    from soup_cli.config.schema import SoupConfig

    with pytest.raises(ValueError, match="loss_watchdog.*MLX text SFT"):
        SoupConfig(base="./model", backend="mlx", task="sft", modality="vision",
                   data={"train": "./data.jsonl"}, training={"loss_watchdog": True})


@pytest.fixture
def loop(monkeypatch, tmp_path):
    """Real Soup control flow, simulated vendor loop/optimizer and disk boundary.

    This tests orchestration on every platform; the native CI file separately
    compares real tensors, optimizer states and reloadable safetensors.
    """
    from soup_cli.trainer import mlx_optim
    from tests.test_issue634_mlx_resume import (
        _FakeMlxModel,
        _install_fake_mlx,
        _mlx_wrapper,
    )

    _install_fake_mlx(monkeypatch)
    wrapper = _mlx_wrapper(tmp_path, epochs=1, loss_watchdog=True, loss_watchdog_threshold=3,
                           loss_watchdog_patience=2, batch_size=1, logging_steps=1,
                           gradient_accumulation_steps=2, rewind_log=False)
    wrapper.config.data.train_on_responses_only = False
    model = _FakeMlxModel()
    model.weight = 10
    model.trainable_parameters = lambda: {"adapter": model.weight}
    wrapper.model, wrapper.tokenizer = model, object()
    wrapper._dataset = {"train": [{"text": "x"}] * 16, "val": []}
    optimizer = SimpleNamespace(completed=0)
    optimizer.step = SimpleNamespace(item=lambda: optimizer.completed)
    monkeypatch.setattr(mlx_optim, "build_optimizer", lambda plan: optimizer)
    utils = ModuleType("mlx.utils")
    utils.tree_flatten = lambda weights: list(weights.items())
    monkeypatch.setitem(sys.modules, "mlx.utils", utils)
    core = sys.modules["mlx.core"]
    core.distributed = SimpleNamespace(init=lambda: SimpleNamespace(rank=lambda: 0))
    core.save_safetensors = lambda path, weights: Path(path).write_text(json.dumps(weights))
    state = SimpleNamespace(wrapper=wrapper, optimizer=optimizer, core=core,
                            losses=[4] * 16, failure=None, reports=[], stopped=False)
    state.display = SimpleNamespace(start=lambda total: None,
                                    update=lambda **kwargs: state.reports.append(kwargs),
                                    stop=lambda: setattr(state, "stopped", True))

    def train(**kwargs):
        args = kwargs["args"]
        state.args = args
        reports = iter(state.losses)
        for iteration in range(1, args.iters + 1):
            if state.failure is not None:
                raise state.failure
            if iteration % args.grad_accumulation_steps == 0:
                optimizer.completed += 1
                model.weight -= 1
            if iteration % args.steps_per_report == 0 or iteration == args.iters:
                payload = {"iteration": iteration, "train_loss": next(reports)}
                kwargs["training_callback"].on_train_loss_report(payload)

    sys.modules["mlx_lm.tuner.trainer"].train = train
    return state


@pytest.mark.parametrize("accum, logging, patience", [(1, 1, 1), (2, 2, 2), (3, 1, 2)])
def test_public_configuration_controls_exact_stop(loop, accum, logging, patience):
    wrapper = loop.wrapper
    wrapper.config.training.gradient_accumulation_steps = accum
    wrapper.config.training.logging_steps = logging
    wrapper.config.training.loss_watchdog_patience = patience
    result = wrapper.train(display=loop.display)
    updates = logging * patience
    assert loop.optimizer.completed == updates
    assert wrapper.model.weight == 10 - updates
    assert result["total_steps"] == updates * accum
    assert result["completed_updates"] == updates
    assert result["stop_reason"] == "loss_watchdog"
    assert loop.reports[-1]["step"] == updates * accum
    assert loop.stopped
    output = Path(wrapper.config.output)
    assert json.loads((output / "adapters.safetensors").read_text()) == {"adapter": 10 - updates}
    assert json.loads((output / "adapter_config.json").read_text())["training_outcome"] == "stopped"


def test_missing_reports_neither_reset_nor_advance_patience(loop):
    loop.losses = [4, None, 4]
    result = loop.wrapper.train()
    assert result["completed_updates"] == 3
    assert loop.wrapper.model.weight == 7


def test_enabled_watchdog_can_complete_budget(loop):
    loop.losses = [1] * 16
    result = loop.wrapper.train()
    assert result["stop_reason"] == "budget_completed"
    assert result["training_outcome"] == "completed"
    assert result["completed_updates"] == 8
    assert loop.wrapper.model.weight == 2


def test_disabled_mode_keeps_original_report_cadence(loop):
    loop.wrapper.config.training.loss_watchdog = False
    result = loop.wrapper.train(display=loop.display)
    assert loop.args.steps_per_report == 1
    assert len(loop.reports) == 16
    assert loop.optimizer.completed == 8
    assert "stop_reason" not in result


@pytest.mark.parametrize("failure", [ValueError("vendor failure"), KeyboardInterrupt()])
def test_unrelated_failure_is_not_a_watchdog_stop(loop, failure):
    loop.failure = failure
    with pytest.raises(type(failure)) as caught:
        loop.wrapper.train(display=loop.display)
    assert caught.value is failure
    assert loop.stopped
    assert loop.optimizer.completed == 0
    assert not (Path(loop.wrapper.config.output) / "adapter_config.json").exists()


def test_nonfinite_report_fails_and_cleans_up(loop):
    loop.losses = [float("nan")]
    with pytest.raises(FloatingPointError, match="non-finite training loss"):
        loop.wrapper.train(display=loop.display)
    assert loop.optimizer.completed == 1
    assert loop.stopped
    assert not (Path(loop.wrapper.config.output) / "adapter_config.json").exists()


def test_failed_final_save_propagates_and_releases_recorder_and_display(loop, monkeypatch):
    from soup_cli.trainer import mlx_sft

    finished = []
    recorder = SimpleNamespace(wrap=lambda rows: rows, loss=object(),
                               finish=lambda: finished.append(True))
    monkeypatch.setattr(mlx_sft, "_start_rewind", lambda **kwargs: recorder)
    loop.wrapper.config.training.rewind_log = True

    def fail_save(*args):
        raise OSError("disk full")

    monkeypatch.setattr(loop.core, "save_safetensors", fail_save)
    with pytest.raises(OSError, match="disk full"):
        loop.wrapper.train(display=loop.display)
    assert finished == [True]
    assert loop.stopped
    assert loop.optimizer.completed == 2
    assert not (Path(loop.wrapper.config.output) / "adapter_config.json").exists()


def test_failing_recorder_finish_still_releases_the_display(loop, monkeypatch):
    from soup_cli.trainer import mlx_sft

    def failing_finish():
        raise OSError("recorder finish failed")

    recorder = SimpleNamespace(wrap=lambda rows: rows, loss=object(), finish=failing_finish)
    monkeypatch.setattr(mlx_sft, "_start_rewind", lambda **kwargs: recorder)
    loop.wrapper.config.training.rewind_log = True
    with pytest.raises(OSError, match="recorder finish failed"):
        loop.wrapper.train(display=loop.display)
    assert loop.stopped


@pytest.mark.parametrize("iteration, updates, diagnostic", [
    (1, 0, "not at an optimizer boundary"),
    (2, 0, "precedes its optimizer update"),
])
def test_changed_vendor_callback_order_is_an_error(loop, iteration, updates, diagnostic):
    loop.wrapper.config.training.loss_watchdog_patience = 1
    loop.optimizer.completed = updates

    def misordered_train(**kwargs):
        kwargs["training_callback"].on_train_loss_report(
            {"iteration": iteration, "train_loss": 4}
        )

    sys.modules["mlx_lm.tuner.trainer"].train = misordered_train
    with pytest.raises(RuntimeError, match=diagnostic):
        loop.wrapper.train(display=loop.display)
    assert loop.stopped
    assert not (Path(loop.wrapper.config.output) / "adapter_config.json").exists()
