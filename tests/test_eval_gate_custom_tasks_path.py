"""A custom tasks file outside the working directory must be refused (#3189)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from soup_cli.eval import gate


def _write_tasks_file(path: Path) -> None:
    path.write_text(
        json.dumps({"prompt": "2+2=", "expected": "4"}) + "\n",
        encoding="utf-8",
    )


def _write_suite(tmp_path: Path, tasks_path: str) -> Path:
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        yaml.safe_dump({
            "suite": "identity",
            "tasks": [{
                "type": "custom", "name": "math", "tasks": tasks_path,
                "scorer": "exact", "threshold": 0.5,
            }],
        }),
        encoding="utf-8",
    )
    return suite


def _expected_error(tasks_path: str) -> str:
    return (
        f"task 'math' tasks file '{tasks_path}' "
        "is outside cwd - refusing to load"
    )


def test_run_gate_refuses_tasks_file_outside_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "outside_tasks_load.jsonl"
    _write_tasks_file(outside)
    try:
        suite = gate.load_suite(str(_write_suite(tmp_path, str(outside))))
        calls: list[str] = []
        result = gate.run_gate(
            suite, generate_fn=lambda prompt: calls.append(prompt) or "4",
        )
    finally:
        outside.unlink(missing_ok=True)
    assert calls == []
    assert result.passed is False
    assert result.task_results[0].score is None
    assert result.task_results[0].passed is False
    assert result.task_results[0].error == _expected_error(str(outside))


def test_run_gate_refuses_tasks_file_assigned_after_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    inside = tmp_path / "tasks.jsonl"
    _write_tasks_file(inside)
    outside = tmp_path.parent / "outside_tasks_after_load.jsonl"
    _write_tasks_file(outside)
    try:
        suite = gate.load_suite(str(_write_suite(tmp_path, str(inside))))
        suite.tasks[0].tasks = str(outside)  # type: ignore[assignment]
        calls: list[str] = []
        result = gate.run_gate(
            suite, generate_fn=lambda prompt: calls.append(prompt) or "4",
        )
    finally:
        outside.unlink(missing_ok=True)
    assert calls == []
    assert result.passed is False
    assert result.task_results[0].score is None
    assert result.task_results[0].passed is False
    assert result.task_results[0].error == _expected_error(str(outside))


def test_run_gate_keeps_tasks_file_inside_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    inside = tmp_path / "tasks.jsonl"
    _write_tasks_file(inside)
    suite = gate.load_suite(str(_write_suite(tmp_path, str(inside))))
    calls: list[str] = []
    result = gate.run_gate(
        suite, generate_fn=lambda prompt: calls.append(prompt) or "4",
    )
    assert calls == ["2+2="]
    assert result.task_results[0].score == 1.0
    assert result.task_results[0].error is None
    assert result.passed is True
