from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
from rich.console import Console

from soup_cli.cli import app
from soup_cli.eval.judge import JudgeEvaluator
from soup_cli.experiment.tracker import ExperimentTracker
from tests.conftest import make_cli_runner, strip_ansi

runner = make_cli_runner()

FULL_ID = "run_reference_v1"
SECOND_ID = "run_twin_a"
THIRD_ID = "run_twin_b"
REPLY = json.dumps({"scores": {"helpfulness": 5, "accuracy": 5, "safety": 5}})


@pytest.fixture
def db(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "experiments.db"
    monkeypatch.setenv("SOUP_DB_PATH", str(path))
    monkeypatch.setattr("soup_cli.commands.eval.console", Console(width=500))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "responses.jsonl").write_text(
        json.dumps({"prompt": "question", "response": "answer"}) + "\n", encoding="utf-8"
    )
    tracker = ExperimentTracker(db_path=path)
    try:
        for run_id in (FULL_ID, SECOND_ID, THIRD_ID):
            tracker.start_run(
                config_dict={}, device="cpu", device_name="CPU", gpu_info={}, run_id=run_id
            )
    finally:
        tracker.close()
    return path


def _judge(run_id: str):
    with mock.patch.object(JudgeEvaluator, "_call_llm", return_value=REPLY) as call:
        result = runner.invoke(
            app, ["eval", "judge", "--target", "responses.jsonl", "--run-id", run_id]
        )
    return result, call


def _saved(db: Path) -> list[tuple[str, str]]:
    tracker = ExperimentTracker(db_path=db)
    try:
        return [(row["run_id"], row["benchmark"]) for row in tracker.get_eval_results()]
    finally:
        tracker.close()


def test_full_id_saves_under_that_run(db):
    result, call = _judge(FULL_ID)

    assert result.exit_code == 0, result.output
    assert call.call_count == 1
    assert _saved(db) == [(FULL_ID, "judge:gpt-4o-mini")]


def test_unique_prefix_saves_under_the_resolved_full_id(db):
    result, _ = _judge("run_reference")

    assert result.exit_code == 0, result.output
    assert _saved(db) == [(FULL_ID, "judge:gpt-4o-mini")]


def test_missing_run_is_refused_before_the_judge_is_called(db):
    result, call = _judge("run_missing")

    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert "Run not found: run_missing" in strip_ansi(result.stdout)
    assert call.call_count == 0
    assert _saved(db) == []


def test_ambiguous_prefix_is_refused_before_the_judge_is_called(db):
    result, call = _judge("run_twin_")

    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert "matches 2 runs; give more characters" in strip_ansi(result.stdout)
    assert call.call_count == 0
    assert _saved(db) == []


def test_without_run_id_nothing_is_saved_and_no_tracker_is_needed(db):
    with mock.patch.object(JudgeEvaluator, "_call_llm", return_value=REPLY):
        result = runner.invoke(app, ["eval", "judge", "--target", "responses.jsonl"])

    assert result.exit_code == 0, result.output
    assert _saved(db) == []


@pytest.mark.parametrize("run_id", [FULL_ID, "run_missing", "run_twin_"])
def test_the_tracker_is_closed_exactly_once(db, run_id):
    closes = []
    real_close = ExperimentTracker.close

    def counting_close(self):
        closes.append(self.db_path)
        real_close(self)

    with mock.patch.object(ExperimentTracker, "close", counting_close):
        _judge(run_id)

    assert closes == [db]
