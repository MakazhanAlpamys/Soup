"""Tests for SoupTrainerCallback."""

from unittest.mock import MagicMock, patch

from soup_cli.monitoring.callback import SoupTrainerCallback


def _make_state(global_step=10, max_steps=100, epoch=1.0):
    """Create a mock TrainerState."""
    state = MagicMock()
    state.global_step = global_step
    state.max_steps = max_steps
    state.epoch = epoch
    return state


def _make_args():
    """Create a mock TrainingArguments."""
    return MagicMock()


def test_on_train_begin_starts_display():
    """on_train_begin should call display.start with total_steps."""
    display = MagicMock()
    callback = SoupTrainerCallback(display=display)
    state = _make_state(max_steps=500)

    callback.on_train_begin(_make_args(), state, MagicMock())

    display.start.assert_called_once_with(total_steps=500)


def test_on_train_end_stops_display():
    """on_train_end should call display.stop."""
    display = MagicMock()
    callback = SoupTrainerCallback(display=display)

    callback.on_train_end(_make_args(), _make_state(), MagicMock())

    display.stop.assert_called_once()


def test_on_log_updates_display():
    """on_log should call display.update with metrics from logs."""
    display = MagicMock()
    callback = SoupTrainerCallback(display=display)

    logs = {
        "loss": 1.234,
        "learning_rate": 2e-5,
        "grad_norm": 0.5,
        "train_steps_per_second": 3.0,
    }
    state = _make_state(global_step=42, epoch=1.5)

    with patch("soup_cli.monitoring.callback.torch", create=True):
        callback.on_log(_make_args(), state, MagicMock(), logs=logs)

    display.update.assert_called_once()
    call_kwargs = display.update.call_args
    assert call_kwargs[1]["step"] == 42 or call_kwargs[0][0] == 42


def test_on_log_none_logs():
    """on_log with logs=None should do nothing."""
    display = MagicMock()
    callback = SoupTrainerCallback(display=display)

    callback.on_log(_make_args(), _make_state(), MagicMock(), logs=None)

    display.update.assert_not_called()


def test_summary_log_preserves_last_training_metrics():
    """HF's final runtime summary must not reset the live panel to zero."""
    display = MagicMock()
    callback = SoupTrainerCallback(display=display)
    state = _make_state(global_step=128, max_steps=128, epoch=1.0)

    callback.on_log(
        _make_args(),
        state,
        MagicMock(),
        logs={
            "loss": 1.3589,
            "learning_rate": 1.2e-8,
            "grad_norm": 6.55,
        },
    )
    callback.on_log(
        _make_args(),
        state,
        MagicMock(),
        logs={"train_runtime": 1016.0, "train_steps_per_second": 0.126},
    )

    final_update = display.update.call_args_list[-1].kwargs
    assert final_update["loss"] == 1.3589
    assert final_update["lr"] == 1.2e-8
    assert final_update["grad_norm"] == 6.55


def test_real_zero_metrics_replace_previous_values():
    """Preservation must not hide a genuine logged zero."""
    display = MagicMock()
    callback = SoupTrainerCallback(display=display)
    state = _make_state()

    callback.on_log(
        _make_args(),
        state,
        MagicMock(),
        logs={"loss": 1.0, "learning_rate": 1e-4, "grad_norm": 2.0},
    )
    callback.on_log(
        _make_args(),
        state,
        MagicMock(),
        logs={"loss": 0.0, "learning_rate": 0.0, "grad_norm": 0.0},
    )

    final_update = display.update.call_args_list[-1].kwargs
    assert final_update["loss"] == 0.0
    assert final_update["lr"] == 0.0
    assert final_update["grad_norm"] == 0.0


def test_on_log_with_tracker():
    """on_log should forward metrics to tracker if provided."""
    display = MagicMock()
    tracker = MagicMock()
    callback = SoupTrainerCallback(display=display, tracker=tracker, run_id="run_123")

    logs = {"loss": 0.5, "learning_rate": 1e-5}
    state = _make_state(global_step=10, epoch=1.0)

    callback.on_log(_make_args(), state, MagicMock(), logs=logs)

    tracker.log_metrics.assert_called_once()
    call_kwargs = tracker.log_metrics.call_args[1]
    assert call_kwargs["run_id"] == "run_123"
    assert call_kwargs["step"] == 10
    assert call_kwargs["loss"] == 0.5


def test_on_log_without_tracker():
    """on_log without tracker should not crash."""
    display = MagicMock()
    callback = SoupTrainerCallback(display=display, tracker=None, run_id="")

    logs = {"loss": 0.5}
    callback.on_log(_make_args(), _make_state(), MagicMock(), logs=logs)

    display.update.assert_called_once()
