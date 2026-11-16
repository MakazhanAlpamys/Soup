"""#2791: a step row records the speed the panel shows, not a fabricated 0.0.

Hugging Face sets ``train_steps_per_second`` only in the final summary log, so
reading it on every step stored 0.0 on every step row. A step row now records
the run average since the first logged step, the same number the live panel
shows, and None on the first row, where there is no interval yet.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import soup_cli.monitoring.callback as callback_module
import soup_cli.monitoring.display as display_module
from soup_cli.config.schema import SoupConfig
from soup_cli.experiment.tracker import ExperimentTracker
from soup_cli.monitoring.callback import SoupTrainerCallback
from soup_cli.monitoring.display import TrainingDisplay
from tests.ui_helpers import ui_client

_TRAIN_LOG = {"loss": 1.0, "learning_rate": 1e-5, "grad_norm": 0.5}


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    fake_time = SimpleNamespace(monotonic=fake.monotonic)
    # raising=False: before the fix the callback had no clock, and the
    # regression test must fail on the recorded value, not on this patch.
    monkeypatch.setattr(callback_module, "time", fake_time, raising=False)
    monkeypatch.setattr(display_module, "time", fake_time)
    return fake


@pytest.fixture
def tracker(tmp_path):
    tracker = ExperimentTracker(db_path=tmp_path / "runs.db")
    yield tracker
    tracker.close()


def _start_run(tracker) -> str:
    return tracker.start_run(
        config_dict={"base": "test", "task": "sft"},
        device="cpu",
        device_name="CPU",
        gpu_info={},
    )


def _log(callback, step: int, logs: dict) -> None:
    state = MagicMock()
    state.global_step = step
    state.max_steps = 100
    state.epoch = step / 100
    callback.on_log(MagicMock(), state, MagicMock(), logs=logs)


def _recorded_speeds(tracker, run_id: str) -> list:
    return [row["speed"] for row in tracker.get_metrics(run_id)]


def test_step_rows_record_the_measured_rate_not_zero(clock, tracker):
    run_id = _start_run(tracker)
    callback = SoupTrainerCallback(display=MagicMock(), tracker=tracker, run_id=run_id)
    for step, now in ((10, 100.0), (20, 105.0), (30, 120.0)):
        clock.now = now
        _log(callback, step, dict(_TRAIN_LOG))

    speeds = _recorded_speeds(tracker, run_id)
    assert speeds[0] is None  # no interval yet on the first row
    # (step - first_step) / (now - first_time): 10/5 and 20/20
    assert speeds[1:] == pytest.approx([2.0, 1.0])
    assert 0.0 not in speeds


def test_panel_shows_the_speed_the_record_holds(clock, tracker):
    run_id = _start_run(tracker)
    display = TrainingDisplay(SoupConfig(base="test-model", data={"train": "./data.jsonl"}))
    callback = SoupTrainerCallback(display=display, tracker=tracker, run_id=run_id)
    shown = []
    for step, now in ((10, 100.0), (20, 104.0), (35, 110.0)):
        clock.now = now
        _log(callback, step, dict(_TRAIN_LOG))
        shown.append(display.speed)

    assert _recorded_speeds(tracker, run_id) == shown
    assert shown[0] is None
    assert shown[-1] == pytest.approx(25 / 10.0)


def test_a_new_training_run_measures_from_its_own_first_step(clock, tracker):
    run_id = _start_run(tracker)
    callback = SoupTrainerCallback(display=MagicMock(), tracker=tracker, run_id=run_id)
    for step, now in ((10, 100.0), (20, 105.0)):
        clock.now = now
        _log(callback, step, dict(_TRAIN_LOG))
    callback.on_train_begin(MagicMock(), MagicMock(max_steps=100), MagicMock())
    clock.now = 110.0
    _log(callback, 30, dict(_TRAIN_LOG))

    # Without the reset in on_train_begin this row would be (30 - 10) / 10 = 2.0,
    # measured from the previous run's first step.
    assert _recorded_speeds(tracker, run_id)[-1] is None


def test_final_summary_row_records_hugging_face_throughput(clock, tracker):
    run_id = _start_run(tracker)
    callback = SoupTrainerCallback(display=MagicMock(), tracker=tracker, run_id=run_id)
    clock.now = 100.0
    _log(callback, 10, dict(_TRAIN_LOG))
    clock.now = 110.0
    _log(callback, 20, dict(_TRAIN_LOG))
    clock.now = 111.0
    _log(callback, 20, {"train_runtime": 11.0, "train_steps_per_second": 1.818})

    assert _recorded_speeds(tracker, run_id)[-1] == 1.818


def test_mlx_grpo_validation_row_records_no_speed(tracker):
    from soup_cli.trainer.mlx_grpo import log_val_reward

    run_id = _start_run(tracker)
    log_val_reward(tracker, run_id, step=4, epoch=1.0, lr=0.01, val_reward=0.5)

    row = tracker.get_metrics(run_id)[-1]
    assert row["val_reward"] == 0.5
    # The held-out row measured no step rate, as it measured no loss (#2830).
    # A stored 0.0 would count as a measured zero in get_metric_series.
    assert row["speed"] is None
    assert row["loss"] is None
    assert tracker.get_metric_series(run_id, "speed") == []


def test_tracker_stores_none_speed_and_keeps_its_zero_default(tracker):
    run_id = _start_run(tracker)
    tracker.log_metrics(run_id, step=1, loss=1.0, speed=None)
    # PPO and MLX GRPO omit speed; their rows are unchanged by #2791.
    tracker.log_metrics(run_id, step=2, loss=1.0)

    assert _recorded_speeds(tracker, run_id) == [None, 0.0]


def test_old_zero_rows_and_new_rows_read_back(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")

    from soup_cli.ui.app import create_app, get_auth_token

    db_path = tmp_path / "runs.db"
    monkeypatch.setenv("SOUP_DB_PATH", str(db_path))
    tracker = ExperimentTracker(db_path=db_path)
    # A run recorded before #2791 keeps its 0.0 step speeds: nothing rewrites it.
    old_run = _start_run(tracker)
    tracker.log_metrics(old_run, step=1, loss=1.0, speed=0.0)
    tracker.log_metrics(old_run, step=2, loss=1.0, speed=0.0)
    new_run = _start_run(tracker)
    tracker.log_metrics(new_run, step=1, loss=1.0, speed=None)
    tracker.log_metrics(new_run, step=2, loss=1.0, speed=2.5)

    assert tracker.get_metric_series(old_run, "speed") == [0.0, 0.0]
    assert tracker.get_metric_series(new_run, "speed") == [2.5]
    tracker.close()

    client = ui_client(create_app())
    headers = {"Authorization": f"Bearer {get_auth_token()}"}
    old = client.get(f"/api/runs/{old_run}/metrics", headers=headers)
    new = client.get(f"/api/runs/{new_run}/metrics", headers=headers)
    assert old.status_code == 200 and new.status_code == 200
    assert [m["speed"] for m in old.json()["metrics"]] == [0.0, 0.0]
    assert [m["speed"] for m in new.json()["metrics"]] == [None, 2.5]


def test_web_speed_chart_keeps_null_as_a_gap():
    source = (
        Path(__file__).parents[1] / "src" / "soup_cli" / "ui" / "static" / "app.js"
    ).read_text(encoding="utf-8")
    assert "const speeds = metrics.map(m => m.speed ?? null);" in source


def test_web_speed_chart_axis_names_steps_not_tokens():
    # Every writer of `speed` stores steps (iterations) per second: the
    # transformers callback, MLX SFT's iterations_per_second and MLX DPO.
    source = (
        Path(__file__).parents[1] / "src" / "soup_cli" / "ui" / "static" / "app.js"
    ).read_text(encoding="utf-8")
    assert "chartOpts('Steps/sec')" in source
    assert "Tokens/sec" not in source


# StepRate is imported inside each test so the regression tests above can run,
# and fail on the recorded value, against code that does not have it yet.


def test_step_rate_first_step_has_no_rate():
    from soup_cli.monitoring.step_rate import StepRate

    assert StepRate().update(10, 100.0) is None


def test_step_rate_is_the_run_average_since_the_first_step():
    from soup_cli.monitoring.step_rate import StepRate

    rate = StepRate()
    rate.update(10, 100.0)
    assert rate.update(20, 105.0) == pytest.approx(2.0)
    assert rate.update(30, 120.0) == pytest.approx(1.0)


def test_step_rate_has_no_rate_until_the_step_or_clock_moves():
    from soup_cli.monitoring.step_rate import StepRate

    rate = StepRate()
    rate.update(10, 100.0)
    assert rate.update(10, 105.0) is None
    assert rate.update(20, 100.0) is None


def test_step_rate_measures_again_when_the_step_goes_backwards():
    from soup_cli.monitoring.step_rate import StepRate

    rate = StepRate()
    rate.update(500, 0.0)
    assert rate.update(600, 100.0) == pytest.approx(1.0)
    assert rate.update(0, 110.0) is None
    assert rate.update(50, 135.0) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "step", [float("nan"), float("inf"), float("-inf"), -1, True, None, "20"]
)
def test_step_rate_ignores_a_step_that_is_not_a_count(step):
    from soup_cli.monitoring.step_rate import StepRate

    rate = StepRate()
    rate.update(10, 0.0)
    assert rate.update(step, 5.0) is None
    # The baseline is kept: the next real step still measures from step 10.
    assert rate.update(20, 10.0) == pytest.approx(1.0)


def test_step_rate_reset_forgets_the_first_step():
    from soup_cli.monitoring.step_rate import StepRate

    rate = StepRate()
    rate.update(10, 0.0)
    rate.reset()
    assert rate.update(20, 10.0) is None
