"""A boolean suite threshold must not silently become a numeric cutoff."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.eval.gate import GateTask, load_suite, run_gate
from tests.conftest import strip_ansi


def write_suite(root: Path, threshold: str) -> Path:
    (root / "tasks.jsonl").write_text(
        json.dumps({"prompt": "q", "expected": "A"}) + "\n", encoding="utf-8",
    )
    suite = root / "suite.yaml"
    suite.write_text(
        "suite: synthetic\ntasks:\n"
        "  - type: custom\n    name: one\n    tasks: tasks.jsonl\n"
        f"    threshold: {threshold}\n",
        encoding="utf-8",
    )
    return suite


@pytest.mark.parametrize("value", [False, True])
def test_direct_task_refuses_boolean_threshold(value: bool) -> None:
    with pytest.raises(ValidationError) as exc:
        GateTask(type="custom", name="one", threshold=value)
    assert exc.value.errors()[0]["loc"] == ("threshold",)
    assert "must be a number, not a boolean" in str(exc.value)


BOOLEAN_SPELLINGS = [
    spelling
    for word in ("yes", "no", "on", "off", "true", "false")
    for spelling in (word, word.title(), word.upper())
]


@pytest.mark.parametrize("spelling", BOOLEAN_SPELLINGS)
def test_yaml_boolean_threshold_is_refused(
    spelling: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    suite = write_suite(tmp_path, spelling)
    with pytest.raises(ValidationError) as exc:
        load_suite(str(suite))
    assert exc.value.errors()[0]["loc"] == ("tasks", 0, "threshold")
    assert "must be a number, not a boolean" in str(exc.value)


@pytest.mark.parametrize("spelling", ["false", "true"])
def test_cli_refuses_boolean_before_model_or_baseline_write(
    spelling: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    suite = write_suite(tmp_path, spelling)
    baseline = tmp_path / "baseline.json"
    baseline.write_bytes(b"prior baseline bytes\n")
    calls = {"factory": 0, "generate": 0}

    def factory(_model: str) -> Callable[[str], str]:
        calls["factory"] += 1

        def generate(_prompt: str) -> str:
            calls["generate"] += 1
            return "wrong"

        return generate

    monkeypatch.setattr("soup_cli.eval.quant_check.make_model_generator", factory)
    result = CliRunner().invoke(app, [
        "eval", "gate", "--suite", str(suite), "--model", "synthetic-model",
        "--write-baseline", str(baseline),
    ])
    assert result.exit_code == 3, result.output
    output = " ".join(strip_ansi(result.output).split())
    assert "Cannot load suite" in output
    assert "tasks.0.threshold" in output
    assert "must be a number, not a boolean" in output
    assert calls == {"factory": 0, "generate": 0}
    assert baseline.read_bytes() == b"prior baseline bytes\n"


@pytest.mark.parametrize("value", [0, 0.0, 0.5, 1000, "0", "0.5", "1000"])
def test_numeric_thresholds_keep_existing_coercion(value: object) -> None:
    task = GateTask(type="custom", name="one", threshold=value)
    assert task.threshold == float(value)


@pytest.mark.parametrize("value", [-0.1, 1000.1, float("nan"), float("inf"),
                                  float("-inf"), "false", None])
def test_existing_invalid_thresholds_stay_invalid(value: object) -> None:
    with pytest.raises(ValidationError):
        GateTask(type="custom", name="one", threshold=value)


@pytest.mark.parametrize(("threshold", "passed"), [("0", True), ("0.5", False)])
def test_explicit_numeric_cutoff_keeps_measured_zero_behavior(
    threshold: str, passed: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    suite = load_suite(str(write_suite(tmp_path, threshold)))
    result = run_gate(suite, generate_fn=lambda _prompt: "wrong")
    assert result.passed is passed
    assert result.task_results[0].score == 0.0
