"""Regression tests for issue #812 — connect eval results to ``eval against``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.conftest import strip_ansi


def _save_score(run_id: str, benchmark: str, score: float) -> None:
    from soup_cli.experiment.tracker import ExperimentTracker

    ExperimentTracker().save_eval_result(
        "model-under-test",
        benchmark,
        score,
        {"total": 1, "correct": int(score == 1.0)},
        run_id=run_id,
    )


def _against(*args: str):
    from soup_cli.commands.eval import app

    return CliRunner().invoke(app, ["against", *args])


def test_custom_eval_results_produce_a_verdict_and_report_degenerate_ci():
    _save_score("run-base", "custom", 0.80)
    _save_score("run-candidate", "custom", 0.90)

    result = _against(
        "run-base",
        "--candidate",
        "run-candidate",
        "--metric",
        "custom",
        "--json-only",
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    output = strip_ansi(result.output)
    assert "Empty series" not in output
    payload = json.loads(output)
    assert payload["metric"] == "custom"
    assert payload["regressed"] is False
    assert payload["sample_count"] == 1
    assert payload["ci_degenerate"] is True
    assert payload["ci_lower"] is None
    assert payload["ci_upper"] is None


def test_default_task_accuracy_falls_back_to_custom_results():
    """The installed hook does not pass ``--metric``, so its default must work."""
    _save_score("run-base", "custom", 0.80)
    _save_score("run-candidate", "custom", 0.90)

    result = _against("run-base", "--candidate", "run-candidate", "--json-only")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    payload = json.loads(strip_ansi(result.output))
    assert payload["metric"] == "task_accuracy"
    assert payload["source_metric"] == "custom"


def test_explicit_benchmark_metric_reads_lm_eval_task_result():
    _save_score("run-base", "mmlu", 0.61)
    _save_score("run-candidate", "mmlu", 0.64)

    result = _against(
        "run-base",
        "--candidate",
        "run-candidate",
        "--metric",
        "benchmark:mmlu",
        "--json-only",
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    payload = json.loads(strip_ansi(result.output))
    assert payload["source_metric"] == "mmlu"
    assert payload["regressed"] is False


@pytest.mark.parametrize("metric", ["aider_polyglot", "judge:openai/judge-model"])
def test_named_soup_eval_results_are_directly_comparable(metric):
    _save_score("run-base", metric, 0.50)
    _save_score("run-candidate", metric, 0.75)

    result = _against(
        "run-base",
        "--candidate",
        "run-candidate",
        "--metric",
        metric,
        "--json-only",
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    payload = json.loads(strip_ansi(result.output))
    assert payload["source_metric"] == metric


def test_unknown_metric_is_usage_error_before_tracker_initialization(monkeypatch):
    import soup_cli.experiment.tracker as tracker_module

    touched_database = False

    def fail_if_initialized(self, *args, **kwargs):
        nonlocal touched_database
        touched_database = True
        raise AssertionError("metric validation must happen before database access")

    monkeypatch.setattr(tracker_module.ExperimentTracker, "__init__", fail_if_initialized)

    result = _against(
        "run-base",
        "--candidate",
        "run-candidate",
        "--metric",
        "accuracy",
    )

    output = strip_ansi(result.output)
    assert result.exit_code == 3, (output, repr(result.exception))
    assert touched_database is False
    assert "unknown metric 'accuracy'" in output
    assert "allowed:" in output
    assert "custom" in output
    assert "benchmark:<name>" in output


def test_empty_series_has_honest_unavailable_message():
    result = _against(
        "run-base",
        "--candidate",
        "run-candidate",
        "--metric",
        "custom",
    )

    output = strip_ansi(result.output)
    assert result.exit_code == 3, (output, repr(result.exception))
    assert "comparison unavailable" in output.lower()
    assert "no eval results" in output.lower()
    assert "run-base" in output
    assert "run-candidate" in output
    assert "regression" not in output.lower()


@pytest.mark.parametrize(
    "metric",
    ["loss", "val_loss", "lr", "grad_norm", "speed", "gpu_mem", "epoch"],
)
def test_benchmark_namespace_rejects_reserved_training_metrics(metric, monkeypatch):
    import soup_cli.experiment.tracker as tracker_module

    def fail_if_initialized(self, *args, **kwargs):
        raise AssertionError("reserved metric must be rejected before database access")

    monkeypatch.setattr(tracker_module.ExperimentTracker, "__init__", fail_if_initialized)
    result = _against(
        "run-base",
        "--candidate",
        "run-candidate",
        "--metric",
        f"benchmark:{metric}",
    )

    output = strip_ansi(result.output)
    assert result.exit_code == 3, (output, repr(result.exception))
    assert "reserved training metric" in output


def test_oversized_unknown_metric_is_bounded_before_database_access(monkeypatch):
    import soup_cli.experiment.tracker as tracker_module

    def fail_if_initialized(self, *args, **kwargs):
        raise AssertionError("oversized metric must be rejected before database access")

    monkeypatch.setattr(tracker_module.ExperimentTracker, "__init__", fail_if_initialized)
    metric = "x" * 10_000
    result = _against(
        "run-base",
        "--candidate",
        "run-candidate",
        "--metric",
        metric,
    )

    output = strip_ansi(result.output)
    assert result.exit_code == 3, (output, repr(result.exception))
    assert "exceeds 256 characters" in output
    assert len(output) < 1_000


@pytest.mark.skipif(os.name == "nt", reason="generated hook is a bash script")
def test_rendered_hook_does_not_call_empty_series_a_regression(tmp_path, monkeypatch):
    from soup_cli.utils.eval_design import EvalDesign
    from soup_cli.utils.eval_gate_hook import render_pre_push_hook
    from soup_cli.utils.eval_lock_coverage import lock_suite

    monkeypatch.chdir(tmp_path)
    lock_suite(
        EvalDesign(goal="gate", row_count=0, dimensions=()),
        "evals/locked.json",
    )
    hook = tmp_path / "pre-push"
    hook.write_text(
        render_pre_push_hook(
            baseline_run_id="run-base",
            suite_path="evals/locked.json",
        ),
        encoding="utf-8",
    )
    soup_bin = Path(sys.executable).parent / "soup"
    assert soup_bin.is_file(), f"console script missing beside interpreter: {soup_bin}"
    env = os.environ.copy()
    env["PATH"] = f"{soup_bin.parent}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env["SOUP_CANDIDATE_RUN_ID"] = "run-candidate"

    completed = subprocess.run(
        ["bash", str(hook)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    output = strip_ansi(completed.stdout + completed.stderr)
    assert completed.returncode == 3
    assert "comparison unavailable" in output.lower()
    assert "run-base" in output
    assert "run-candidate" in output
    assert "regression" not in output.lower()
