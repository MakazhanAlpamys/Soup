"""Thresholds assigned after load must be re-checked before generation."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from soup_cli.eval import gate

BAD_THRESHOLDS = [True, float("nan"), float("inf"), float("-inf"), 9999, -5, "5", None]
GOOD_THRESHOLDS = [(0.0, True), (1000.0, False)]


def _suite(tmp_path: Path) -> Path:
    path = tmp_path / "tasks0.jsonl"
    path.write_text(
        json.dumps({"prompt": "0", "expected": "gold"}) + "\n", encoding="utf-8",
    )
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        yaml.safe_dump({
            "suite": "identity",
            "tasks": [{
                "type": "custom", "name": "first", "tasks": str(path),
                "scorer": "exact", "threshold": 0.0,
            }],
        }),
        encoding="utf-8",
    )
    return suite


def _threshold_error(value: object) -> str:
    return (
        f"eval-gate suite 'identity' task 'first' threshold {value!r} "
        "must be a finite number within 0.0-1000.0"
    )


@pytest.mark.parametrize("value", BAD_THRESHOLDS)
def test_direct_run_refuses_threshold_mutation_after_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: object,
) -> None:
    monkeypatch.chdir(tmp_path)
    suite = gate.load_suite(str(_suite(tmp_path)))
    suite.tasks[0].threshold = value  # type: ignore[assignment]
    calls: list[str] = []
    with pytest.raises(ValueError) as caught:
        gate.run_gate(suite, generate_fn=lambda prompt: calls.append(prompt) or "gold")
    assert str(caught.value) == _threshold_error(value)
    assert calls == []


@pytest.mark.parametrize(("value", "passed"), GOOD_THRESHOLDS)
def test_direct_run_keeps_boundary_thresholds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: float, passed: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    suite = gate.load_suite(str(_suite(tmp_path)))
    suite.tasks[0].threshold = value
    calls: list[str] = []
    result = gate.run_gate(suite, generate_fn=lambda prompt: calls.append(prompt) or "gold")
    assert calls == ["0"]
    assert [row.threshold for row in result.task_results] == [value]
    assert [row.score for row in result.task_results] == [1.0]
    assert result.passed is passed


def test_direct_run_keeps_decimal_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    suite = gate.load_suite(str(_suite(tmp_path)))
    suite.tasks[0].threshold = Decimal("0.5")  # type: ignore[assignment]
    result = gate.run_gate(suite, generate_fn=lambda prompt: "gold")
    assert result.passed
    assert [row.score for row in result.task_results] == [1.0]


def test_direct_run_refuses_huge_integer_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    suite = gate.load_suite(str(_suite(tmp_path)))
    suite.tasks[0].threshold = 10**10000  # type: ignore[assignment]
    calls: list[str] = []
    with pytest.raises(ValueError) as caught:
        gate.run_gate(suite, generate_fn=lambda prompt: calls.append(prompt) or "gold")
    assert "task 'first'" in str(caught.value)
    assert "must be a finite number within 0.0-1000.0" in str(caught.value)
    assert calls == []


def test_direct_run_refuses_scorer_mutation_after_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    suite = gate.load_suite(str(_suite(tmp_path)))
    suite.tasks[0].scorer = "bogus"  # type: ignore[assignment]
    calls: list[str] = []
    with pytest.raises(ValueError) as caught:
        gate.run_gate(suite, generate_fn=lambda prompt: calls.append(prompt) or "gold")
    assert str(caught.value) == (
        "eval-gate suite 'identity' task 'first' scorer 'bogus' must be one of "
        "'exact', 'contains', 'answer', 'regex', 'semantic'"
    )
    assert calls == []
