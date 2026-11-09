"""Tests for experiment tracker (SQLite-backed)."""

from pathlib import Path

import pytest

from soup_cli.experiment.tracker import (
    ExperimentTracker,
    generate_run_id,
    run_lookup_failure,
)


@pytest.fixture
def tracker(tmp_path: Path) -> ExperimentTracker:
    """Create a tracker with a temporary database."""
    db_path = tmp_path / "test_experiments.db"
    return ExperimentTracker(db_path=db_path)


def test_generate_run_id():
    """Run IDs should be unique and match expected format."""
    rid = generate_run_id()
    assert rid.startswith("run_")
    # run_ (4) + YYYYMMDD (8) + _ (1) + HHMMSS (6) + _ (1) + xxxxxxxx (8) = 28
    assert len(rid) == 28
    # Uniqueness (8 hex chars = 4 billion possibilities, no collisions in 100)
    ids = {generate_run_id() for _ in range(100)}
    assert len(ids) == 100


def test_start_and_list_run(tracker):
    """Starting a run should make it appear in list_runs()."""
    run_id = tracker.start_run(
        config_dict={"base": "test-model", "task": "sft"},
        device="cpu",
        device_name="CPU",
        gpu_info={"memory_total": "N/A"},
    )
    runs = tracker.list_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id
    assert runs[0]["status"] == "running"
    assert runs[0]["base_model"] == "test-model"
    assert runs[0]["task"] == "sft"


def test_start_run_with_experiment_name(tracker):
    """Experiment name should be stored."""
    run_id = tracker.start_run(
        config_dict={"base": "test-model"},
        device="cpu",
        device_name="CPU",
        gpu_info={},
        experiment_name="my-experiment",
    )
    run = tracker.get_run(run_id)
    assert run["experiment_name"] == "my-experiment"


def test_finish_run(tracker):
    """Finishing a run should update its status and metrics."""
    run_id = tracker.start_run(
        config_dict={"base": "test-model"},
        device="cpu",
        device_name="CPU",
        gpu_info={},
    )
    tracker.finish_run(
        run_id=run_id,
        initial_loss=2.5,
        final_loss=0.8,
        total_steps=100,
        duration_secs=3600.0,
        output_dir="/tmp/output",
    )
    run = tracker.get_run(run_id)
    assert run["status"] == "completed"
    assert run["initial_loss"] == 2.5
    assert run["final_loss"] == 0.8
    assert run["total_steps"] == 100
    assert run["duration_secs"] == 3600.0
    assert run["output_dir"] == "/tmp/output"


def test_fail_run(tracker):
    """Failing a run should update its status."""
    run_id = tracker.start_run(
        config_dict={},
        device="cpu",
        device_name="CPU",
        gpu_info={},
    )
    tracker.fail_run(run_id)
    run = tracker.get_run(run_id)
    assert run["status"] == "failed"


def test_log_and_get_metrics(tracker):
    """Metrics should be logged and retrievable in order."""
    run_id = tracker.start_run(
        config_dict={},
        device="cpu",
        device_name="CPU",
        gpu_info={},
    )
    tracker.log_metrics(run_id, step=1, loss=2.5, lr=1e-5)
    tracker.log_metrics(run_id, step=2, loss=2.3, lr=9e-6)
    tracker.log_metrics(run_id, step=3, loss=2.1, lr=8e-6)

    metrics = tracker.get_metrics(run_id)
    assert len(metrics) == 3
    assert metrics[0]["step"] == 1
    assert metrics[0]["loss"] == 2.5
    assert metrics[1]["step"] == 2
    assert metrics[2]["loss"] == 2.1


def test_get_run_not_found(tracker):
    """get_run should return None for non-existent run."""
    assert tracker.get_run("nonexistent") is None


def test_get_run_prefix_match(tracker):
    """get_run should support prefix matching."""
    run_id = tracker.start_run(
        config_dict={},
        device="cpu",
        device_name="CPU",
        gpu_info={},
    )
    # Use first 12 chars as prefix
    prefix = run_id[:12]
    run = tracker.get_run(prefix)
    assert run is not None
    assert run["run_id"] == run_id


def _start(tracker, run_id: str) -> str:
    return tracker.start_run(
        config_dict={}, device="cpu", device_name="CPU", gpu_info={}, run_id=run_id
    )


def test_count_run_id_matches(tracker):
    """count_run_id_matches uses the same exact-then-prefix lookup as get_run."""
    _start(tracker, "run_dup_1")
    _start(tracker, "run_dup_2")
    _start(tracker, "run_dup")
    _start(tracker, "run_solo_1")

    # An exact id is one match even when it is also a prefix of other ids.
    assert tracker.count_run_id_matches("run_dup") == 1
    assert tracker.count_run_id_matches("run_dup_1") == 1
    assert tracker.count_run_id_matches("run_solo") == 1
    assert tracker.count_run_id_matches("run_dup_") == 2
    assert tracker.count_run_id_matches("run_") == 4
    assert tracker.count_run_id_matches("gone") == 0
    # LIKE wildcards are matched literally, as in get_run.
    assert tracker.count_run_id_matches("run%") == 0
    assert tracker.count_run_id_matches("run_dup__") == 0


def test_get_run_still_returns_none_for_an_ambiguous_prefix(tracker):
    _start(tracker, "run_dup_1")
    _start(tracker, "run_dup_2")

    assert tracker.get_run("run_dup_") is None
    assert tracker.get_run("run_dup_2")["run_id"] == "run_dup_2"


def test_run_lookup_failure_names_an_ambiguous_prefix(tracker):
    _start(tracker, "run_dup_1")
    _start(tracker, "run_dup_2")

    assert run_lookup_failure(tracker, "run_dup_") == (
        "Run id prefix 'run_dup_' matches 2 runs; give more characters"
    )


def test_run_lookup_failure_keeps_run_not_found_for_no_match(tracker):
    _start(tracker, "run_dup_1")

    assert run_lookup_failure(tracker, "gone") == "Run not found: gone"


@pytest.mark.parametrize(
    ("markup", "shown"),
    [(True, "x\\[red]a"), (False, "x[red]a")],
)
def test_run_lookup_failure_escapes_the_id_only_for_markup(tracker, markup, shown):
    _start(tracker, "x[red]a1")
    _start(tracker, "x[red]a2")

    assert run_lookup_failure(tracker, "x[red]a", markup=markup) == (
        f"Run id prefix '{shown}' matches 2 runs; give more characters"
    )
    assert run_lookup_failure(tracker, "x[red]b\x9b\u202e", markup=markup) == (
        f"Run not found: {shown[:-1]}b"
    )


def test_run_lookup_failure_can_leave_the_id_out_of_the_no_match_text(tracker):
    _start(tracker, "run_dup_1")
    _start(tracker, "run_dup_2")

    assert run_lookup_failure(tracker, "run_gone", name_missing_id=False) == "Run not found"
    assert run_lookup_failure(tracker, "run_dup", name_missing_id=False) == (
        "Run id prefix 'run_dup' matches 2 runs; give more characters"
    )


def test_list_runs_ordering(tracker):
    """Runs should be listed newest first."""
    id1 = tracker.start_run(
        config_dict={"base": "model-a"},
        device="cpu", device_name="CPU", gpu_info={},
    )
    id2 = tracker.start_run(
        config_dict={"base": "model-b"},
        device="cpu", device_name="CPU", gpu_info={},
    )
    runs = tracker.list_runs()
    assert len(runs) == 2
    assert runs[0]["run_id"] == id2  # newest first
    assert runs[1]["run_id"] == id1


def test_list_runs_limit(tracker):
    """list_runs should respect limit."""
    for idx in range(5):
        tracker.start_run(
            config_dict={"base": f"model-{idx}"},
            device="cpu", device_name="CPU", gpu_info={},
        )
    runs = tracker.list_runs(limit=3)
    assert len(runs) == 3


def test_save_and_get_eval_results(tracker):
    """Eval results should be saved and retrievable."""
    tracker.save_eval_result(
        model_path="/tmp/model",
        benchmark="mmlu",
        score=0.65,
        details={"acc": 0.65, "samples": 100},
    )
    results = tracker.get_eval_results()
    assert len(results) == 1
    assert results[0]["benchmark"] == "mmlu"
    assert results[0]["score"] == 0.65
    assert results[0]["run_id"] is None


def test_eval_results_linked_to_run(tracker):
    """Eval results should be filterable by run_id."""
    run_id = tracker.start_run(
        config_dict={}, device="cpu", device_name="CPU", gpu_info={},
    )
    tracker.save_eval_result(
        model_path="/tmp/model",
        benchmark="mmlu",
        score=0.65,
        details={},
        run_id=run_id,
    )
    tracker.save_eval_result(
        model_path="/tmp/model2",
        benchmark="gsm8k",
        score=0.40,
        details={},
    )
    # Filter by run_id
    linked = tracker.get_eval_results(run_id=run_id)
    assert len(linked) == 1
    assert linked[0]["benchmark"] == "mmlu"

    # All results
    all_results = tracker.get_eval_results()
    assert len(all_results) == 2


def test_delete_run(tracker):
    """Deleting a run should remove it and its metrics."""
    run_id = tracker.start_run(
        config_dict={}, device="cpu", device_name="CPU", gpu_info={},
    )
    tracker.log_metrics(run_id, step=1, loss=2.0)
    tracker.log_metrics(run_id, step=2, loss=1.5)

    assert tracker.delete_run(run_id) is True
    assert tracker.get_run(run_id) is None
    assert tracker.get_metrics(run_id) == []


def test_delete_nonexistent_run(tracker):
    """Deleting a non-existent run should return False."""
    assert tracker.delete_run("nonexistent") is False


def test_config_json_stored(tracker):
    """Full config should be stored as JSON and recoverable."""
    import json

    config = {
        "base": "meta-llama/Llama-3.1-8B",
        "task": "sft",
        "training": {"epochs": 3, "lr": 2e-5},
    }
    run_id = tracker.start_run(
        config_dict=config,
        device="cuda",
        device_name="NVIDIA RTX 4090",
        gpu_info={"memory_total": "24.0 GB"},
    )
    run = tracker.get_run(run_id)
    recovered = json.loads(run["config_json"])
    assert recovered["base"] == "meta-llama/Llama-3.1-8B"
    assert recovered["training"]["epochs"] == 3
