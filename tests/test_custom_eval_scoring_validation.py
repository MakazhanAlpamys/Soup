"""Custom-eval scoring diagnostics and refusal before evaluation side effects."""

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.eval import custom
from soup_cli.experiment import tracker
from tests.conftest import strip_ansi

INVALID_SCORING = [
    pytest.param([], id="empty-array"),
    pytest.param(["exact"], id="array"),
    pytest.param({}, id="empty-object"),
    pytest.param({"mode": "exact"}, id="object"),
    pytest.param(0, id="zero"),
    pytest.param(1, id="integer"),
    pytest.param(1.5, id="float"),
    pytest.param(True, id="true"),
    pytest.param(False, id="false"),
    pytest.param(None, id="null"),
    pytest.param("bogus", id="unknown-string"),
]


def _write_tasks(path: Path, scoring: Any) -> None:
    first = {"prompt": "first", "expected": "a", "scoring": "exact"}
    third = {"prompt": "second", "expected": "b", "scoring": scoring}
    path.write_text(json.dumps(first) + "\n\n" + json.dumps(third) + "\n", encoding="utf-8")


@pytest.mark.parametrize("scoring", INVALID_SCORING)
def test_invalid_scoring_names_field_and_physical_line(tmp_path: Path, scoring: Any) -> None:
    path = tmp_path / "tasks.jsonl"
    _write_tasks(path, scoring)

    with pytest.raises(ValueError) as caught:
        custom.load_eval_tasks(path)

    message = str(caught.value)
    assert message.startswith(f"Line 3: invalid scoring '{scoring}', must be one of: ")
    assert message.endswith(
        "answer, contains, exact, regex, semantic, tool_call_args_subset, "
        "tool_call_match, tool_call_name_match"
    )


def test_valid_labels_defaults_and_blank_lines_are_preserved(tmp_path: Path) -> None:
    labels = [
        "exact", "contains", "answer", "regex", "semantic",
        "tool_call_match", "tool_call_name_match", "tool_call_args_subset",
    ]
    rows = [{"prompt": "default"}] + [
        {"prompt": label, "expected": "42", "scoring": label, "source": "fixture"}
        for label in labels
    ]
    path = tmp_path / "tasks.jsonl"
    path.write_text("\n\n".join(json.dumps(row) for row in rows) + "\n\n", encoding="utf-8")

    tasks = custom.load_eval_tasks(path)

    assert len(tasks) == 9
    assert tasks[0] == custom.EvalTask(prompt="default")
    assert [task.scoring for task in tasks[1:]] == labels
    assert [task.prompt for task in tasks[1:]] == labels
    assert all(task.expected == "42" for task in tasks[1:])
    assert all(task.category == "default" for task in tasks[1:])
    assert all(task.metadata == {"source": "fixture"} for task in tasks[1:])


@pytest.mark.parametrize("scoring", INVALID_SCORING)
@pytest.mark.parametrize("force_color", [False, True], ids=["plain", "color"])
def test_cli_invalid_scoring_stops_before_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scoring: Any, force_color: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    if force_color:
        monkeypatch.setenv("FORCE_COLOR", "1")
    else:
        monkeypatch.delenv("FORCE_COLOR", raising=False)
    path = tmp_path / "tasks.jsonl"
    _write_tasks(path, scoring)
    model = tmp_path / "model"
    model.mkdir()
    output = tmp_path / "result.json"
    database = tmp_path / "custom-eval.db"
    monkeypatch.setenv("SOUP_DB_PATH", str(database))
    calls = {"factory": 0, "generate": 0, "tracker": 0}
    real_tracker = tracker.ExperimentTracker

    def generate(prompt: str) -> str:
        calls["generate"] += 1
        return "a" if prompt == "first" else "b"

    def factory(*args: Any, **kwargs: Any) -> Any:
        calls["factory"] += 1
        return generate

    def create_tracker(*args: Any, **kwargs: Any) -> tracker.ExperimentTracker:
        calls["tracker"] += 1
        return real_tracker(*args, **kwargs)

    monkeypatch.setattr(custom, "_create_default_generator", factory)
    monkeypatch.setattr(tracker, "ExperimentTracker", create_tracker)

    result = CliRunner().invoke(app, [
        "eval", "custom", "--tasks", str(path), "--model", str(model), "--output", str(output),
    ])

    assert result.exit_code == 1
    text = " ".join(strip_ansi(result.output).split())
    assert "Line 3: invalid scoring" in text
    assert "must be one of:" in text
    assert calls == {"factory": 0, "generate": 0, "tracker": 0}
    assert not database.exists()
    assert not output.exists()


def test_cli_valid_tasks_evaluate_and_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        '{"prompt": "first", "expected": "a"}\n\n'
        '{"prompt": "second", "expected": "b", "scoring": "contains"}\n',
        encoding="utf-8",
    )
    model = tmp_path / "model"
    model.mkdir()
    output = tmp_path / "result.json"
    database = tmp_path / "custom-eval.db"
    monkeypatch.setenv("SOUP_DB_PATH", str(database))
    calls: list[str] = []

    def generate(prompt: str) -> str:
        calls.append(prompt)
        return "a" if prompt == "first" else "answer b"

    monkeypatch.setattr(custom, "_create_default_generator", lambda *args, **kwargs: generate)

    result = CliRunner().invoke(app, [
        "eval", "custom", "--tasks", str(path), "--model", str(model), "--output", str(output),
    ])

    assert result.exit_code == 0, result.output
    assert calls == ["first", "second"]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert (payload["total"], payload["correct"], payload["accuracy"]) == (2, 2, 1.0)
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT benchmark, score, details_json FROM eval_results"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][:2] == ("custom", 1.0)
    assert json.loads(rows[0][2])["total"] == 2
