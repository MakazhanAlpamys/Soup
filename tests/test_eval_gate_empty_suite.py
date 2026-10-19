"""An eval gate must measure a task before it can report a verdict."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest
import yaml
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import eval as eval_command
from soup_cli.eval import gate, quant_check

from .conftest import strip_ansi

EMPTY_SUITES = [
    {"suite": "empty", "tasks": []},
    {"suite": "empty"},
    {"suite": "empty", "task": [{"name": "misspelled"}]},
]
ERROR = "eval-gate suite 'empty' field 'tasks' must contain at least one task"


def _write_suite(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "gate.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _nonempty_suite(tmp_path: Path, threshold: float) -> dict[str, object]:
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps({"prompt": "control", "expected": "gold"}) + "\n", encoding="utf-8",
    )
    return {
        "suite": "control",
        "tasks": [{
            "type": "custom", "name": "control", "tasks": str(tasks),
            "scorer": "exact", "threshold": threshold,
        }],
    }


@pytest.mark.parametrize("payload", EMPTY_SUITES, ids=["explicit", "missing", "misspelled"])
def test_load_suite_refuses_zero_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: dict[str, object],
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _write_suite(tmp_path, payload)
    with pytest.raises(ValueError) as caught:
        gate.load_suite(str(path))
    assert str(caught.value) == ERROR


@pytest.mark.parametrize(
    "content",
    [b"", b"# TODO: write the gate suite\n", b"  \n\n   \n"],
    ids=["zero-bytes", "comment-only", "whitespace-only"],
)
def test_load_suite_empty_file_is_a_value_error_naming_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: bytes,
) -> None:
    """The contract, not the implementation: an empty suite file is refused with a
    ValueError whose message names the missing "suite" field (pydantic's "Field required"
    report today; a "must be a mapping" refusal under #1878), never with a TypeError from
    `**None`, and never with a made-up suite name or a message that hides what is wrong."""
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "gate.yaml"
    path.write_bytes(content)
    with pytest.raises(ValueError) as caught:
        gate.load_suite(str(path))
    message = str(caught.value)
    assert "suite" in message, repr(caught.value)
    assert "Field required" in message or "must be a mapping" in message, repr(caught.value)


@pytest.mark.parametrize("payload", EMPTY_SUITES, ids=["explicit", "missing", "misspelled"])
@pytest.mark.parametrize("with_model", [False, True], ids=["smoke", "model-factory"])
def test_cli_refuses_before_model_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object], with_model: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _write_suite(tmp_path, payload)
    factory_calls: list[str] = []

    def factory(model: str) -> Callable[[str], str]:
        factory_calls.append(model)
        return lambda prompt: ""

    monkeypatch.setattr(quant_check, "make_model_generator", factory)
    monkeypatch.setattr(eval_command, "console", Console(width=200))
    args = ["eval", "gate", "--suite", str(path)]
    if with_model:
        args.extend(["--model", "synthetic-model"])
    result = CliRunner().invoke(app, args)
    output = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 3
    assert f"Cannot load suite: {ERROR}" in output
    assert "GATE PASSED" not in output
    assert "GATE FAILED" not in output
    assert factory_calls == []


@pytest.mark.parametrize("payload", EMPTY_SUITES, ids=["explicit", "missing", "misspelled"])
def test_direct_run_gate_refuses_zero_tasks(payload: dict[str, object]) -> None:
    suite = gate.EvalSuite.model_validate(payload)
    calls: list[str] = []
    with pytest.raises(ValueError) as caught:
        gate.run_gate(suite, generate_fn=lambda prompt: calls.append(prompt) or "")
    assert str(caught.value) == ERROR
    assert calls == []


def test_suite_emptied_after_loading_cannot_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    payload = _nonempty_suite(tmp_path, 0.0)
    payload["suite"] = "empty"
    suite = gate.load_suite(str(_write_suite(tmp_path, payload)))
    suite.tasks.clear()
    with pytest.raises(ValueError) as caught:
        gate.run_gate(suite, generate_fn=lambda prompt: "")
    assert str(caught.value) == ERROR


@pytest.mark.parametrize("threshold, baseline, passed, regressed", [
    (0.0, {}, True, False),
    (1.0, {}, False, False),
    (0.0, {"control": 1.0}, False, True),
])
def test_nonempty_api_preserves_score_threshold_and_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    threshold: float, baseline: dict[str, float], passed: bool, regressed: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    suite = gate.load_suite(str(_write_suite(tmp_path, _nonempty_suite(tmp_path, threshold))))
    calls: list[str] = []
    result = gate.run_gate(
        suite, generate_fn=lambda prompt: calls.append(prompt) or "", baseline=baseline,
    )
    assert result.passed is passed
    assert result.regression is regressed
    assert len(result.task_results) == 1
    assert result.task_results[0].score == 0.0
    assert result.task_results[0].baseline == baseline.get("control")
    assert calls == ["control"]


@pytest.mark.parametrize("with_model", [False, True], ids=["smoke", "model-factory"])
@pytest.mark.parametrize("threshold, baseline, exit_code", [
    (0.0, {}, 0), (1.0, {}, 2), (0.0, {"control": 1.0}, 2),
])
def test_nonempty_cli_preserves_smoke_and_model_factory_verdicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    threshold: float, baseline: dict[str, float], exit_code: int, with_model: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _write_suite(tmp_path, _nonempty_suite(tmp_path, threshold))
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(
        quant_check, "make_model_generator",
        lambda model: lambda prompt: calls.append(prompt) or "",
    )
    monkeypatch.setattr(eval_command, "console", Console(width=200))
    args = ["eval", "gate", "--suite", str(path), "--baseline", str(baseline_path)]
    if with_model:
        args.extend(["--model", "synthetic-model"])
    result = CliRunner().invoke(app, args)
    output = strip_ansi(result.output)
    assert result.exit_code == exit_code
    assert ("GATE PASSED" in output) is (exit_code == 0)
    assert ("GATE FAILED" in output) is (exit_code == 2)
    assert calls == (["control"] if with_model else [])


def test_empty_model_construction_is_not_a_gate_verdict() -> None:
    assert gate.EvalSuite(suite="fixture", tasks=[]).tasks == []


def test_nonempty_suite_still_ignores_unknown_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    payload = _nonempty_suite(tmp_path, 0.0)
    payload["future_key"] = True
    assert len(gate.load_suite(str(_write_suite(tmp_path, payload))).tasks) == 1
