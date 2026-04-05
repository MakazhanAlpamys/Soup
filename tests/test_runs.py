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


def test_runs_clean_not_found():
    """soup runs clean with bad IDs should fail."""
    result = runner.invoke(app, ["runs", "clean", "nonexistent"])
    assert result.exit_code == 1


def test_runs_clean_with_data(tracker, tmp_path):
    """soup runs clean should reclaim space natively."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    # create some dummy checkpoints
    ckpt1 = out_dir / "checkpoint-100"
    ckpt1.mkdir()
    (ckpt1 / "optimizer.pt").write_text("dummy")
    (ckpt1 / "adapter_model.bin").write_text("model")

    ckpt2 = out_dir / "checkpoint-200"
    ckpt2.mkdir()
    (ckpt2 / "optimizer.pt").write_text("dummy")
    (ckpt2 / "adapter_model.bin").write_text("model")

    run_id = tracker.start_run(
        config_dict={}, device="cpu", device_name="CPU", gpu_info={},
    )
    tracker.finish_run(
        run_id=run_id,
        initial_loss=2.0,
        final_loss=1.0, # let's say step 100 has lower loss, but we mock tracker
        total_steps=200,
        duration_secs=100.0,
        output_dir=str(out_dir),
    )
    # mock metrics: step 200 is best
    tracker.log_metrics(run_id, step=100, loss=2.0)
    tracker.log_metrics(run_id, step=200, loss=1.0)

    result = runner.invoke(app, ["runs", "clean", run_id, "--force"])
    assert result.exit_code == 0
    assert "Successfully reclaimed" in result.output

    # verify optimizer in ckpt1 is gone, but in ckpt2 it's potentially kept or not
    # based on keep-weights
    # if it's the best (200), ckpt2 optimizer should exist. ckpt1 should not exist
    assert not (ckpt1 / "optimizer.pt").exists()
    assert (ckpt2 / "optimizer.pt").exists()

    # test --all
    result = runner.invoke(app, ["runs", "clean", "--all", "--force"])
    assert "No disposable checkpoint files found" in result.output

