"""Unmeasured gradient norms must stay absent in every monitoring sink (#1147)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from soup_cli.experiment.tracker import ExperimentTracker
from soup_cli.monitoring.callback import SoupTrainerCallback
from soup_cli.monitoring.display import TrainingDisplay
from soup_cli.utils.why import diagnose


def test_callback_separates_display_from_recorded_norms(monkeypatch) -> None:
    from soup_cli.utils import train_event_buffer
    from soup_cli.utils.sse_train_stream import to_payload

    display = MagicMock()
    tracker = MagicMock()
    events = []
    monkeypatch.setattr(train_event_buffer, "push_train_event", events.append)
    callback = SoupTrainerCallback(display=display, tracker=tracker, run_id="run")

    for step, norm in enumerate((None, 0.7, None, 0.0), start=1):
        state = SimpleNamespace(global_step=step, epoch=1.0)
        logs = {"loss": 2.0 - step * 0.1, "learning_rate": 1e-4}
        if norm is not None:
            logs["grad_norm"] = norm
        callback.on_log(args=None, state=state, control=None, logs=logs)

    assert [c.kwargs["grad_norm"] for c in display.update.call_args_list] == [
        None,
        0.7,
        0.7,
        0.0,
    ]
    assert [c.kwargs["grad_norm"] for c in tracker.log_metrics.call_args_list] == [
        None,
        0.7,
        None,
        0.0,
    ]
    assert [event.grad_norm for event in events] == [None, 0.7, None, 0.0]
    assert ["grad_norm" in to_payload(event) for event in events] == [
        False,
        True,
        False,
        True,
    ]
    assert to_payload(events[-1])["grad_norm"] == 0.0


def test_tracker_stores_null_for_omission_and_keeps_real_zero(tmp_path) -> None:
    tracker = ExperimentTracker(db_path=tmp_path / "runs.db")
    run_id = tracker.start_run(
        config_dict={},
        device="cpu",
        device_name="CPU",
        gpu_info={},
    )
    tracker.log_metrics(run_id, step=1, loss=1.0)
    tracker.log_metrics(run_id, step=2, loss=0.9, grad_norm=0.7)
    tracker.log_metrics(run_id, step=3, loss=0.8)
    tracker.log_metrics(run_id, step=4, loss=0.7, grad_norm=0.0)

    rows = tracker.get_metrics(run_id)
    assert [row["grad_norm"] for row in rows] == [None, 0.7, None, 0.0]
    assert tracker.get_metric_series(run_id, "grad_norm") == [0.7, 0.0]
    tracker.close()


def test_terminal_panel_hides_missing_norm_but_shows_measured_zero() -> None:
    config = MagicMock()
    config.training.epochs = 1
    config.experiment_name = "test"
    display = TrainingDisplay(config)
    display.total_steps = 10

    display.update(step=1, epoch=1.0, loss=1.0, lr=1e-4, grad_norm=None)
    assert "Grad:" not in display._render().renderable

    display.update(step=2, epoch=1.0, loss=0.9, lr=1e-4, grad_norm=0.0)
    assert "Grad:  0.0000" in display._render().renderable


def test_runs_and_why_read_null_norm_without_crashing(tmp_path, monkeypatch) -> None:
    from soup_cli.cli import app

    db_path = tmp_path / "runs.db"
    monkeypatch.setenv("SOUP_DB_PATH", str(db_path))
    tracker = ExperimentTracker(db_path=db_path)
    run_id = tracker.start_run(
        config_dict={"base": "test", "task": "sft"},
        device="cpu",
        device_name="CPU",
        gpu_info={},
    )
    for step in range(1, 12):
        tracker.log_metrics(run_id, step=step, loss=2.0 - step * 0.1, grad_norm=None)

    rows = tracker.get_metrics(run_id)
    assert all(row["grad_norm"] is None for row in rows)
    assert not any(f.category == "grad_norm_high" for f in diagnose(rows))
    result = CliRunner().invoke(app, ["runs", "show", run_id])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    why_result = CliRunner().invoke(app, ["why", run_id])
    assert why_result.exit_code == 0, (why_result.output, repr(why_result.exception))
    tracker.close()


def test_web_chart_keeps_null_as_a_gap_instead_of_fabricating_zero() -> None:
    source = (
        Path(__file__).parents[1] / "src" / "soup_cli" / "ui" / "static" / "app.js"
    ).read_text(encoding="utf-8")
    assert "const gradNorms = metrics.map(m => m.grad_norm ?? null);" in source


def test_web_metrics_api_serializes_null_norm(tmp_path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from soup_cli.ui.app import create_app, get_auth_token

    db_path = tmp_path / "runs.db"
    monkeypatch.setenv("SOUP_DB_PATH", str(db_path))
    tracker = ExperimentTracker(db_path=db_path)
    run_id = tracker.start_run(
        config_dict={"base": "test", "task": "sft"},
        device="cpu",
        device_name="CPU",
        gpu_info={},
    )
    tracker.log_metrics(run_id, step=1, loss=1.0, grad_norm=None)
    tracker.close()

    client = TestClient(create_app())
    response = client.get(
        f"/api/runs/{run_id}/metrics",
        headers={"Authorization": f"Bearer {get_auth_token()}"},
    )
    assert response.status_code == 200
    assert response.json()["metrics"][0]["grad_norm"] is None
