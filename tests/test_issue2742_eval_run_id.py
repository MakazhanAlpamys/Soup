"""#2742: the eval write paths link a result to the run the user named.

``soup eval benchmark --run-id`` and ``soup eval custom --run-id`` handed the
option straight to ``save_eval_result``, which stores the string it is given.
The tracker resolves a unique prefix to a full run id, so a prefix was stored
under the prefix and an id that names no run was stored as its own run; both
commands exited 0.

Both commands now resolve the run through the tracker's run lookup before the
model is loaded, refuse a missing or ambiguous run with the wording the other
run-lookup commands use, and save the row under the run's full id.  Rows that
exist today keep the id they were stored with.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import types
from pathlib import Path

import pytest
from rich.console import Console

from soup_cli.cli import app
from soup_cli.experiment.tracker import ExperimentTracker
from tests.conftest import make_cli_runner, strip_ansi

runner = make_cli_runner()

FULL_RUN_ID = "run_reference_v1"
PREFIX = "run_reference"
MISSING = "run_gone"
AMBIGUOUS = "run_dup_"


@pytest.fixture(autouse=True)
def db(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "experiments.db"
    monkeypatch.setenv("SOUP_DB_PATH", str(path))
    monkeypatch.setenv("COLUMNS", "500")
    monkeypatch.setattr("soup_cli.commands.eval.console", Console(width=500))
    monkeypatch.chdir(tmp_path)
    tracker = ExperimentTracker(db_path=path)
    try:
        for run_id in (FULL_RUN_ID, "run_dup_1", "run_dup_2"):
            tracker.start_run(
                config_dict={},
                device="cpu",
                device_name="CPU",
                gpu_info={},
                run_id=run_id,
            )
    finally:
        tracker.close()
    return path


@pytest.fixture
def model_dir(tmp_path) -> Path:
    path = tmp_path / "model"
    path.mkdir()
    return path


@pytest.fixture
def tasks(tmp_path) -> Path:
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps({"prompt": "q", "expected": "a"}) + "\n")
    return path


@pytest.fixture
def custom_generator(monkeypatch) -> list[str]:
    """Replace the model generator of ``soup eval custom``; record its calls.

    The factory call is recorded as well as the prompts, so an empty list
    means the model was never loaded, not only that no prompt was generated.
    """
    calls: list[str] = []

    def generator(model_path, trust_remote_code=False):
        calls.append(f"factory:{model_path}")

        def generate(prompt: str) -> str:
            calls.append(prompt)
            return "an answer"

        return generate

    monkeypatch.setattr("soup_cli.eval.custom._create_default_generator", generator)
    return calls


@pytest.fixture
def lm_eval(monkeypatch) -> list[str]:
    """Stand in for lm-evaluation-harness; record the runs it was asked for."""
    calls: list[str] = []

    monkeypatch.setitem(sys.modules, "lm_eval", types.ModuleType("lm_eval"))

    def run(model_arg, tasks, num_fewshot, batch_size, device):
        calls.append(model_arg)
        return {"results": {"mmlu": {"acc,none": 0.5}}}

    monkeypatch.setattr("soup_cli.commands.eval._run_lm_eval", run)
    return calls


def _custom(model_dir: Path, tasks: Path, run_id: str | None):
    argv = ["eval", "custom", "--tasks", str(tasks), "--model", str(model_dir)]
    if run_id is not None:
        argv += ["--run-id", run_id]
    return runner.invoke(app, argv)


def _benchmark(model_dir: Path, run_id: str | None):
    argv = ["eval", "benchmark", "--model", str(model_dir), "--device", "cpu"]
    if run_id is not None:
        argv += ["--run-id", run_id]
    return runner.invoke(app, argv)


def _saved_runs(db: Path) -> list[str]:
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute("SELECT run_id FROM eval_results ORDER BY created_at, rowid")
        return [row[0] for row in rows]
    finally:
        conn.close()


def _lines(result) -> list[str]:
    return [line.strip() for line in strip_ansi(result.stdout).splitlines() if line.strip()]


@pytest.fixture(params=["custom", "benchmark"])
def command(request):
    return request.param


def test_a_full_run_id_saves_the_row_under_that_run(
    db, command, model_dir, tasks, custom_generator, lm_eval
):
    result = (
        _custom(model_dir, tasks, FULL_RUN_ID)
        if command == "custom"
        else _benchmark(model_dir, FULL_RUN_ID)
    )

    assert result.exit_code == 0, result.output
    assert _saved_runs(db) == [FULL_RUN_ID]


def test_a_unique_prefix_saves_the_row_under_the_full_run_id(
    db, command, model_dir, tasks, custom_generator, lm_eval
):
    result = (
        _custom(model_dir, tasks, PREFIX)
        if command == "custom"
        else _benchmark(model_dir, PREFIX)
    )

    assert result.exit_code == 0, result.output
    assert _saved_runs(db) == [FULL_RUN_ID]


def test_a_missing_run_id_is_refused_before_the_model_is_loaded(
    db, command, model_dir, tasks, custom_generator, lm_eval
):
    result = (
        _custom(model_dir, tasks, MISSING)
        if command == "custom"
        else _benchmark(model_dir, MISSING)
    )

    assert result.exit_code == 1, result.output
    assert f"Run not found: {MISSING}" in _lines(result)
    assert custom_generator == []
    assert lm_eval == []
    assert _saved_runs(db) == []


def test_an_ambiguous_prefix_is_refused_before_the_model_is_loaded(
    db, command, model_dir, tasks, custom_generator, lm_eval
):
    result = (
        _custom(model_dir, tasks, AMBIGUOUS)
        if command == "custom"
        else _benchmark(model_dir, AMBIGUOUS)
    )

    assert result.exit_code == 1, result.output
    assert f"Run id prefix '{AMBIGUOUS}' matches 2 runs; give more characters" in _lines(result)
    assert custom_generator == []
    assert lm_eval == []
    assert _saved_runs(db) == []


def test_without_a_run_id_the_row_is_stored_without_a_run(
    db, command, model_dir, tasks, custom_generator, lm_eval
):
    result = (
        _custom(model_dir, tasks, None)
        if command == "custom"
        else _benchmark(model_dir, None)
    )

    assert result.exit_code == 0, result.output
    # Unchanged: the row is saved as before, with no run attached to it.
    assert _saved_runs(db) == [None]
