"""Tests for soup runs CLI commands."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.experiment.tracker import ExperimentTracker

runner = CliRunner()


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path: Path, monkeypatch):
    """Redirect experiment DB to a temp directory for all tests."""
    db_path = str(tmp_path / "test_experiments.db")
    monkeypatch.setenv("SOUP_DB_PATH", db_path)


@pytest.fixture
def tracker(tmp_path: Path) -> ExperimentTracker:
    """Create tracker using the same temp DB."""
    import os

    db_path = Path(os.environ["SOUP_DB_PATH"])
    return ExperimentTracker(db_path=db_path)


def test_runs_empty():
    """soup runs with no runs should show a message, not crash."""
    result = runner.invoke(app, ["runs"])
    assert result.exit_code == 0
    assert "No runs found" in result.output


def test_runs_list_with_data(tracker):
    """soup runs should show a table when runs exist."""
    run_id = tracker.start_run(
        config_dict={"base": "test-model", "task": "sft"},
        device="cpu",
        device_name="CPU",
        gpu_info={"memory_total": "N/A"},
        experiment_name="test-exp",
    )
    result = runner.invoke(app, ["runs"])
    assert result.exit_code == 0
    assert "Training Runs" in result.output
    # Run ID column has no_wrap so it should always be visible
    assert run_id in result.output


def test_runs_show_not_found():
    """soup runs show nonexistent should fail."""
    result = runner.invoke(app, ["runs", "show", "nonexistent_run_id"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_runs_show_with_data(tracker):
    """soup runs show should display run details."""
    run_id = tracker.start_run(
        config_dict={"base": "test-model", "task": "sft"},
        device="cpu",
        device_name="CPU",
        gpu_info={"memory_total": "16 GB"},
    )
    tracker.finish_run(
        run_id=run_id,
        initial_loss=2.5,
        final_loss=0.8,
        total_steps=100,
        duration_secs=600.0,
        output_dir="/tmp/output",
    )
    result = runner.invoke(app, ["runs", "show", run_id, "--no-plot"])
    assert result.exit_code == 0
    assert run_id in result.output
    assert "test-model" in result.output


def test_runs_compare_not_found():
    """soup runs compare with bad IDs should fail."""
    result = runner.invoke(app, ["runs", "compare", "run_1", "run_2"])
    assert result.exit_code == 1


def test_runs_compare_with_data(tracker):
    """soup runs compare should show side-by-side table."""
    id1 = tracker.start_run(
        config_dict={"base": "model-a", "task": "sft", "training": {"epochs": 3, "lr": 2e-5}},
        device="cpu", device_name="CPU", gpu_info={},
        experiment_name="exp-a",
    )
    id2 = tracker.start_run(
        config_dict={"base": "model-b", "task": "dpo", "training": {"epochs": 5, "lr": 1e-5}},
        device="cuda", device_name="RTX 4090", gpu_info={},
        experiment_name="exp-b",
    )
    result = runner.invoke(app, ["runs", "compare", id1, id2])
    assert result.exit_code == 0
    assert "model-a" in result.output
    assert "model-b" in result.output


def test_runs_delete_not_found():
    """soup runs delete nonexistent should fail."""
    result = runner.invoke(app, ["runs", "delete", "nonexistent", "--force"])
    assert result.exit_code == 1


def test_runs_delete_with_data(tracker):
    """soup runs delete should remove the run."""
    run_id = tracker.start_run(
        config_dict={}, device="cpu", device_name="CPU", gpu_info={},
    )
    result = runner.invoke(app, ["runs", "delete", run_id, "--force"])
    assert result.exit_code == 0
    assert "Deleted" in result.output
    # Verify it's gone
    assert tracker.get_run(run_id) is None
