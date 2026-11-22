"""Tests for TrainingDisplay."""

from io import StringIO

import pytest
from rich.console import Console

from soup_cli.config.schema import SoupConfig
from soup_cli.monitoring.display import TrainingDisplay


def _render_to_str(panel) -> str:
    """Render a Rich Panel to a plain string for assertion."""
    buf = StringIO()
    console = Console(file=buf, width=120, force_terminal=True)
    console.print(panel)
    return buf.getvalue()


def _make_config():
    """Create a minimal SoupConfig for display testing."""
    return SoupConfig(
        base="test-model",
        data={"train": "./data.jsonl"},
        training={"epochs": 3},
    )


def test_display_init():
    """Display should initialize with default values."""
    display = TrainingDisplay(_make_config(), device_name="cuda")
    assert display.current_step == 0
    assert display.total_steps == 0
    assert display.loss is None  # #1225: no loss measured yet, shown as "-"
    assert display.device_name == "cuda"


def test_display_update():
    """Update should store new metric values."""
    display = TrainingDisplay(_make_config())
    display.total_steps = 100

    display.update(step=50, epoch=1.5, loss=0.876, lr=1e-5, speed=3.2, gpu_mem="12/24 GB")

    assert display.current_step == 50
    assert display.current_epoch == 1.5
    assert display.loss == 0.876
    assert display.lr == 1e-5
    assert display.speed == 3.2
    assert display.gpu_mem == "12/24 GB"


def test_display_render_panel():
    """_render should produce a Panel with correct content."""
    display = TrainingDisplay(_make_config(), device_name="cuda:0")
    display.total_steps = 100
    display.update(step=62, epoch=2.0, loss=0.847, lr=1.4e-5, speed=3.2, gpu_mem="18/24 GB")

    panel = display._render()
    rendered = _render_to_str(panel)
    assert "62/100" in rendered
    assert "0.847" in rendered
    # The panel label must identify the figure as a peak, not just "GPU:" (#650).
    assert "GPU peak:" in rendered


def test_display_render_zero_steps():
    """_render with total_steps=0 should not crash (division by zero)."""
    display = TrainingDisplay(_make_config())
    display.total_steps = 0
    panel = display._render()
    assert panel is not None


def test_display_experiment_name():
    """Display should use experiment_name in panel title if set."""
    config = SoupConfig(
        base="test-model",
        data={"train": "./data.jsonl"},
        experiment_name="my-experiment",
    )
    display = TrainingDisplay(config)
    display.total_steps = 10
    panel = display._render()
    rendered = _render_to_str(panel)
    assert "my-experiment" in rendered


def test_display_start_stop():
    """Start and stop should not crash (we don't test actual terminal rendering)."""
    display = TrainingDisplay(_make_config())
    display.start(total_steps=100)
    assert display.total_steps == 100
    assert display._live is not None
    display.stop()


def test_display_update_without_live():
    """Update without calling start should not crash."""
    display = TrainingDisplay(_make_config())
    display.update(step=1, epoch=0.1, loss=2.0, lr=1e-4)
    assert display.current_step == 1


def _freeze_local_clock(monkeypatch, *, hour=12, minute=0, second=0):
    from datetime import datetime

    import soup_cli.monitoring.display as module

    class FixedDateTime(datetime):
        @classmethod
        def now(cls):
            return cls(2026, 12, 31, hour, minute, second)

    monkeypatch.setattr(module, "datetime", FixedDateTime, raising=False)


def test_display_estimates_remaining_optimizer_steps_and_local_finish(monkeypatch):
    _freeze_local_clock(monkeypatch)
    display = TrainingDisplay(_make_config())
    display.total_steps = 1000
    display.update(step=250, epoch=0.75, loss=1.0, lr=1e-5, speed=2.0)
    rendered = _render_to_str(display._render())
    assert "Remaining: 6m 15s" in rendered
    assert "Estimated finish (local): 2026-12-31 12:06:15" in rendered


@pytest.mark.parametrize("initial_step, next_step, expected", [
    (0, 250, "6m 15s"),
    (500, 750, "2m 5s"),
])
def test_display_estimates_eta_when_callback_logs_omit_speed(
    monkeypatch, initial_step, next_step, expected
):
    from types import SimpleNamespace

    import soup_cli.monitoring.display as module

    _freeze_local_clock(monkeypatch)
    clock = iter([0.0, 125.0])
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    display = TrainingDisplay(_make_config())
    display.total_steps = 1000
    # The callback forwards 0.0 for an ordinary Trainer log without throughput.
    logs = {"loss": 1.0, "grad_norm": 0.5, "learning_rate": 1e-5, "epoch": 0.75}
    speed = logs.get("train_steps_per_second", 0.0)
    display.update(step=initial_step, epoch=0.5, loss=1.0, lr=1e-5, speed=speed)
    assert "Remaining:" not in _render_to_str(display._render())
    display.update(step=next_step, epoch=0.75, loss=1.0, lr=1e-5, speed=speed)
    assert f"Remaining: {expected}" in _render_to_str(display._render())


def test_display_refreshes_eta_with_speed_and_hides_a_stale_estimate(monkeypatch):
    _freeze_local_clock(monkeypatch)
    display = TrainingDisplay(_make_config())
    display.total_steps = 1000
    display.update(step=250, epoch=0.75, loss=1.0, lr=1e-5, speed=2.0)
    assert "Remaining: 6m 15s" in _render_to_str(display._render())

    display.update(step=250, epoch=0.75, loss=1.0, lr=1e-5, speed=1.0)
    rendered = _render_to_str(display._render())
    assert "Remaining: 12m 30s" in rendered
    assert "Estimated finish (local): 2026-12-31 12:12:30" in rendered
    assert "6m 15s" not in rendered

    display.update(step=250, epoch=0.75, loss=1.0, lr=1e-5)
    rendered = _render_to_str(display._render())
    assert "Remaining:" not in rendered
    assert "Estimated finish" not in rendered


def test_display_finish_estimate_rolls_into_the_next_local_date(monkeypatch):
    _freeze_local_clock(monkeypatch, hour=23, minute=59, second=50)
    display = TrainingDisplay(_make_config())
    display.total_steps = 10
    display.update(step=0, epoch=0.0, loss=None, lr=0, speed=0.5)
    rendered = _render_to_str(display._render())
    assert "Remaining: 20s" in rendered
    assert "Estimated finish (local): 2027-01-01 00:00:10" in rendered


@pytest.mark.parametrize("step", [1000, 1001])
def test_display_finished_run_clamps_remaining_to_zero(monkeypatch, step):
    _freeze_local_clock(monkeypatch)
    display = TrainingDisplay(_make_config())
    display.total_steps = 1000
    display.update(step=step, epoch=3.0, loss=0.5, lr=0, speed=2.0)
    rendered = _render_to_str(display._render())
    assert "Remaining: 0s" in rendered
    assert "Estimated finish (local): 2026-12-31 12:00:00" in rendered


def test_display_remaining_duration_includes_hours(monkeypatch):
    _freeze_local_clock(monkeypatch)
    display = TrainingDisplay(_make_config())
    display.total_steps = 10000
    display.update(step=2554, epoch=0.5, loss=1.0, lr=0, speed=2.0)
    rendered = _render_to_str(display._render())
    assert "Remaining: 1h 2m 3s" in rendered
    assert "Estimated finish (local): 2026-12-31 13:02:03" in rendered


def test_display_rounds_fractional_remaining_seconds_up(monkeypatch):
    _freeze_local_clock(monkeypatch)
    display = TrainingDisplay(_make_config())
    display.total_steps = 10
    display.update(step=9, epoch=0.5, loss=1.0, lr=0, speed=2.0)
    rendered = _render_to_str(display._render())
    assert "Remaining: 1s" in rendered
    assert "Estimated finish (local): 2026-12-31 12:00:01" in rendered


@pytest.mark.parametrize("speed", [None, 0, -1, float("nan"), float("inf"),
                                  float("-inf"), True, "2", 5e-324])
def test_display_hides_eta_for_unknown_or_unusable_speed(monkeypatch, speed):
    _freeze_local_clock(monkeypatch)
    display = TrainingDisplay(_make_config())
    display.total_steps = 1000
    display.update(step=250, epoch=0.75, loss=1.0, lr=0, speed=speed)
    rendered = _render_to_str(display._render())
    assert "Remaining:" not in rendered
    assert "Estimated finish" not in rendered
    assert "Step:" in rendered


@pytest.mark.parametrize("total", [None, 0, -1, float("nan"), float("inf"),
                                  float("-inf"), True, "1000", 10**1000, 1e300])
def test_display_hides_eta_for_unknown_or_unrepresentable_total(monkeypatch, total):
    _freeze_local_clock(monkeypatch)
    display = TrainingDisplay(_make_config())
    display.total_steps = total
    display.update(step=0, epoch=0.0, loss=None, lr=0, speed=2.0)
    rendered = _render_to_str(display._render())
    assert "Remaining:" not in rendered
    assert "Estimated finish" not in rendered


@pytest.mark.parametrize("step", [-1, None, float("nan"), float("inf"), True, "1"])
def test_display_hides_eta_for_invalid_progress(monkeypatch, step):
    _freeze_local_clock(monkeypatch)
    display = TrainingDisplay(_make_config())
    display.total_steps = 1000
    display.update(step=step, epoch=0.0, loss=None, lr=0, speed=2.0)
    rendered = _render_to_str(display._render())
    assert "Remaining:" not in rendered
    assert "Estimated finish" not in rendered


def test_display_measures_again_when_the_step_counter_restarts(monkeypatch):
    from types import SimpleNamespace

    import soup_cli.monitoring.display as module

    _freeze_local_clock(monkeypatch)
    clock = iter([0.0, 100.0, 110.0, 135.0])
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    display = TrainingDisplay(_make_config())
    display.total_steps = 1000
    for step in (500, 600, 0, 50):
        display.update(step=step, epoch=0.0, loss=1.0, lr=0, speed=0.0)
    assert "Remaining: 7m 55s" in _render_to_str(display._render())


def test_display_start_discards_the_baseline_of_an_earlier_run(monkeypatch):
    from types import SimpleNamespace

    import soup_cli.monitoring.display as module

    _freeze_local_clock(monkeypatch)
    clock = iter([0.0, 100.0, 110.0])
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    display = TrainingDisplay(_make_config())
    display.start(total_steps=1000)
    try:
        display.update(step=500, epoch=0.0, loss=1.0, lr=0, speed=0.0)
        display.update(step=600, epoch=0.0, loss=1.0, lr=0, speed=0.0)
    finally:
        display.stop()
    display.start(total_steps=1000)
    try:
        display.update(step=700, epoch=0.0, loss=1.0, lr=0, speed=0.0)
    finally:
        display.stop()
    assert "Remaining:" not in _render_to_str(display._render())


def test_display_progress_never_passes_100_percent():
    display = TrainingDisplay(_make_config())
    display.total_steps = 1000
    display.update(step=1100, epoch=3.0, loss=0.5, lr=0, speed=2.0)
    rendered = _render_to_str(display._render())
    assert "100%" in rendered and "110%" not in rendered
