"""The GRPO monitor reports existing TRL measurements without changing training."""

from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from rich.console import Console

from soup_cli.config.schema import SoupConfig
from soup_cli.monitoring.callback import SoupTrainerCallback
from soup_cli.monitoring.display import TrainingDisplay


def _callback(task="grpo"):
    config = SoupConfig(base="test-model", task=task, data={"train": "data.jsonl"})
    display = TrainingDisplay(config)
    display.total_steps = 100
    return SoupTrainerCallback(display=display), display


def _log(callback, step, value=.6, *, extra=None, rank_zero=True):
    state = SimpleNamespace(global_step=step, max_steps=100, epoch=0.,
                            is_world_process_zero=rank_zero)
    logs = {"loss": 1., "frac_reward_zero_std": value} if extra is None else extra
    control = SimpleNamespace(should_training_stop=False)
    before = dict(logs)
    callback.on_log(SimpleNamespace(), state, control, logs=logs)
    assert logs == before
    assert not control.should_training_stop


def _render(display):
    out = StringIO()
    Console(file=out, width=140, color_system=None).print(display._render())
    return out.getvalue()


def test_five_distinct_high_measurements_warn_once(caplog):
    callback, display = _callback()
    for step in range(1, 5):
        _log(callback, step)
    assert "reward diversity" not in caplog.text
    _log(callback, 5)
    for step in range(6, 12):
        _log(callback, step)
    assert caplog.text.count("GRPO reward diversity is low") == 1
    assert "60.0%" in _render(display)


@pytest.mark.parametrize("value", [0., .1, .5])
def test_healthy_and_boundary_measurements_do_not_warn(caplog, value):
    callback, display = _callback()
    for step in range(1, 8):
        _log(callback, step, value)
    assert "reward diversity" not in caplog.text
    assert f"Zero-spread groups: {value:.1%}" in _render(display)


def test_healthy_measurement_resets_the_streak(caplog):
    callback, _ = _callback()
    for step in range(1, 5):
        _log(callback, step)
    _log(callback, 5, .1)
    for step in range(6, 10):
        _log(callback, step)
    assert "reward diversity" not in caplog.text
    _log(callback, 10)
    assert "GRPO reward diversity is low" in caplog.text


def test_duplicate_log_events_cannot_trigger_warning(caplog):
    callback, _ = _callback()
    for _ in range(8):
        _log(callback, 1)
    assert "reward diversity" not in caplog.text
    for step in range(2, 6):
        _log(callback, step)
    assert caplog.text.count("GRPO reward diversity is low") == 1


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), -.1, 1.1, "0.6", None])
def test_invalid_values_neither_count_nor_create_a_measurement(caplog, value):
    callback, display = _callback()
    for step in range(1, 7):
        _log(callback, step, value)
    assert "reward diversity" not in caplog.text
    assert "Zero-spread groups" not in _render(display)


def test_summary_and_eval_logs_keep_the_last_measured_row_without_counting(caplog):
    callback, display = _callback()
    _log(callback, 1)
    for step in range(2, 9):
        _log(callback, step, extra={"eval_frac_reward_zero_std": .9, "eval_reward": 1.})
    assert "reward diversity" not in caplog.text
    assert "Zero-spread groups: 60.0%" in _render(display)


def test_non_grpo_run_ignores_the_metric(caplog):
    callback, display = _callback("sft")
    for step in range(1, 8):
        _log(callback, step)
    assert "reward diversity" not in caplog.text
    assert "Zero-spread groups" not in _render(display)


def test_only_world_rank_zero_warns(caplog):
    callback, _ = _callback()
    for step in range(1, 8):
        _log(callback, step, rank_zero=False)
    assert "reward diversity" not in caplog.text


def test_a_new_training_run_resets_the_warning_latch(caplog, monkeypatch):
    callback, display = _callback()
    monkeypatch.setattr(display, "start", Mock())
    state = SimpleNamespace(global_step=0, max_steps=100)
    for _ in range(2):
        callback.on_train_begin(SimpleNamespace(), state, SimpleNamespace())
        for step in range(1, 6):
            _log(callback, step)
    assert caplog.text.count("GRPO reward diversity is low") == 2


def test_invalid_reading_breaks_the_warning_streak(caplog):
    callback, _ = _callback()
    for step in range(1, 5):
        _log(callback, step)
    _log(callback, 5, float("nan"))
    _log(callback, 6)
    assert "reward diversity" not in caplog.text


def test_display_accepts_existing_config_stand_ins_without_task():
    display = TrainingDisplay(SimpleNamespace(training=SimpleNamespace(epochs=1),
                                              base="test", experiment_name=""))
    display.update(step=1, epoch=0, loss=1., lr=1e-5)
    assert "Loss:" in _render(display)
    assert "Zero-spread groups" not in _render(display)
