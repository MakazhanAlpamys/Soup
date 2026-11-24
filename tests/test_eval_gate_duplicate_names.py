"""Exact task names must not collapse distinct gate baseline evidence."""

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
from tests.conftest import strip_ansi

DUPLICATES = ["shared", "Task", "éval"]
DISTINCT = [("first", "second"), ("shared", "Shared"), ("shared", " shared")]


def _suite(tmp_path: Path, names: tuple[str, ...]) -> Path:
    tasks = []
    for index, name in enumerate(names):
        path = tmp_path / f"tasks{index}.jsonl"
        path.write_text(
            json.dumps({"prompt": str(index), "expected": "gold"}) + "\n", encoding="utf-8",
        )
        tasks.append({
            "type": "custom", "name": name, "tasks": str(path),
            "scorer": "exact", "threshold": 0.0,
        })
    suite = tmp_path / "suite.yaml"
    suite.write_text(yaml.safe_dump({"suite": "identity", "tasks": tasks}), encoding="utf-8")
    return suite


def _error(name: str) -> str:
    return f"eval-gate suite 'identity' repeats task name {name!r}"


@pytest.mark.parametrize("name", DUPLICATES)
def test_loader_names_exact_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _suite(tmp_path, (name, name))
    with pytest.raises(ValueError) as caught:
        gate.load_suite(str(path))
    assert str(caught.value) == _error(name)


@pytest.mark.parametrize("name", DUPLICATES)
def test_direct_run_refuses_constructed_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _suite(tmp_path, (name, name))
    suite = gate.EvalSuite.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    calls = []
    with pytest.raises(ValueError) as caught:
        gate.run_gate(suite, generate_fn=lambda prompt: calls.append(prompt) or "gold")
    assert str(caught.value) == _error(name)
    assert calls == []


@pytest.mark.parametrize("names", [("shared", "other", "shared"), ("a", "b", "c", "a")])
@pytest.mark.parametrize("entrypoint", ["load_suite", "run_gate"])
def test_non_adjacent_duplicate_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    names: tuple[str, ...], entrypoint: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _suite(tmp_path, names)
    calls: list[str] = []
    with pytest.raises(ValueError) as caught:
        if entrypoint == "load_suite":
            gate.load_suite(str(path))
        else:
            suite = gate.EvalSuite.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
            gate.run_gate(suite, generate_fn=lambda prompt: calls.append(prompt) or "gold")
    assert str(caught.value) == _error(names[0])
    assert calls == []


@pytest.mark.parametrize("name", DUPLICATES)
def test_direct_run_refuses_duplicate_mutation_after_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    suite = gate.load_suite(str(_suite(tmp_path, (name, "other"))))
    suite.tasks[1].name = name
    with pytest.raises(ValueError) as caught:
        gate.run_gate(suite, generate_fn=lambda prompt: "gold")
    assert str(caught.value) == _error(name)


@pytest.mark.parametrize("name", DUPLICATES)
def test_cli_duplicate_refusal_never_mints_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _suite(tmp_path, (name, name))
    factory_calls = []

    def factory(model: str) -> Callable[[str], str]:
        factory_calls.append(model)
        return lambda prompt: "gold"

    monkeypatch.setattr(quant_check, "make_model_generator", factory)
    monkeypatch.setattr(eval_command, "console", Console(width=200))
    baseline = tmp_path / "baseline.json"
    result = CliRunner().invoke(app, [
        "eval", "gate", "--suite", str(path), "--model", "synthetic-base",
        "--write-baseline", str(baseline),
    ])
    output = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 3
    assert f"Cannot load suite: {_error(name)}" in output
    assert "GATE PASSED" not in output
    assert factory_calls == []
    assert not baseline.exists()


@pytest.mark.parametrize(
    "bad_name", ["", "draft\x00task", 7, ("draft", "task")],
    ids=["empty", "nul", "integer", "tuple"],
)
def test_run_gate_refuses_invalid_name_assigned_after_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_name: object,
) -> None:
    monkeypatch.chdir(tmp_path)
    suite = gate.load_suite(str(_suite(tmp_path, ("math", "other"))))
    setattr(suite.tasks[1], "name", bad_name)
    calls: list[str] = []

    with pytest.raises(ValueError, match=r"tasks\[1\]\.name must be a non-empty, null-free string"):
        gate.run_gate(suite, generate_fn=lambda prompt: calls.append(prompt) or "gold")
    assert calls == []


def test_run_gate_scores_valid_names_after_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    names = ("math", " another task ")
    suite = gate.load_suite(str(_suite(tmp_path, names)))
    calls: list[str] = []

    result = gate.run_gate(suite, generate_fn=lambda prompt: calls.append(prompt) or "gold")

    assert result.passed
    assert calls == ["0", "1"]
    assert [(row.name, row.score) for row in result.task_results] == [(name, 1.0) for name in names]


@pytest.mark.parametrize("names", DISTINCT)
def test_exact_distinct_names_preserve_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, names: tuple[str, str],
) -> None:
    monkeypatch.chdir(tmp_path)
    suite = gate.load_suite(str(_suite(tmp_path, names)))
    result = gate.run_gate(suite, generate_fn=lambda prompt: "gold")
    assert result.passed
    assert [row.name for row in result.task_results] == list(names)
    assert [row.score for row in result.task_results] == [1.0, 1.0]


@pytest.mark.parametrize("names", DISTINCT)
def test_distinct_baseline_roundtrip_retains_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, names: tuple[str, str],
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _suite(tmp_path, names)

    def factory(model: str) -> Callable[[str], str]:
        if model == "synthetic-base":
            return lambda prompt: "gold" if prompt == "0" else "wrong"
        return lambda prompt: "wrong"

    monkeypatch.setattr(quant_check, "make_model_generator", factory)
    monkeypatch.setattr(eval_command, "console", Console(width=200))
    runner = CliRunner()
    baseline = tmp_path / "baseline.json"
    base = runner.invoke(app, [
        "eval", "gate", "--suite", str(path), "--model", "synthetic-base",
        "--write-baseline", str(baseline),
    ])
    assert base.exit_code == 0
    saved = json.loads(baseline.read_text(encoding="utf-8"))
    assert saved["scores"] == {names[0]: 1.0, names[1]: 0.0}
    candidate = runner.invoke(app, [
        "eval", "gate", "--suite", str(path), "--model", "synthetic-candidate",
        "--baseline", str(baseline),
    ])
    assert candidate.exit_code == 2
    assert "GATE FAILED" in strip_ansi(candidate.output)


def test_cli_gate_displays_distinct_task_names_literally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    marked_name = "[red]math[/red]"
    suite = _suite(tmp_path, ("math", marked_name))

    def factory(_model: str) -> Callable[[str], str]:
        return lambda prompt: "gold" if prompt == "0" else "wrong"

    monkeypatch.setattr(quant_check, "make_model_generator", factory)
    monkeypatch.setattr(eval_command, "console", Console(width=200))
    baseline = tmp_path / "baseline.json"
    result = CliRunner().invoke(app, [
        "eval", "gate", "--suite", str(suite), "--model", "synthetic",
        "--write-baseline", str(baseline),
    ])

    assert result.exit_code == 0, (result.output, result.exception)
    lines = [" ".join(strip_ansi(line).split()) for line in result.output.splitlines()]
    rows = [line for line in lines if "│" in line and "math" in line]
    assert [line.split("│")[1].strip() for line in rows] == ["math", marked_name]
    assert json.loads(baseline.read_text(encoding="utf-8"))["scores"] == {
        "math": 1.0,
        marked_name: 0.0,
    }


@pytest.mark.parametrize("name", ["[/x]", ":warning:"], ids=["closing-tag", "emoji-shortcode"])
def test_cli_gate_displays_special_task_names_literally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    suite = _suite(tmp_path, (name,))
    monkeypatch.setattr(quant_check, "make_model_generator", lambda _model: lambda _prompt: "gold")
    monkeypatch.setattr(eval_command, "console", Console(width=200))
    baseline = tmp_path / "baseline.json"
    result = CliRunner().invoke(app, [
        "eval", "gate", "--suite", str(suite), "--model", "synthetic",
        "--write-baseline", str(baseline),
    ])

    assert result.exit_code == 0, (result.output, result.exception)
    lines = [" ".join(strip_ansi(line).split()) for line in result.output.splitlines()]
    rows = [line for line in lines if "│" in line and name in line]
    assert [line.split("│")[1].strip() for line in rows] == [name]
    assert json.loads(baseline.read_text(encoding="utf-8"))["scores"] == {name: 1.0}


@pytest.mark.parametrize(
    "message", ["[bold]dataset unavailable[/bold]", "dataset unavailable :x:"],
    ids=["markup", "emoji-shortcode"],
)
def test_cli_gate_displays_task_error_text_literally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, message: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    suite = _suite(tmp_path, ("broken",))

    def generate(_prompt: str) -> str:
        raise RuntimeError(message)

    monkeypatch.setattr(quant_check, "make_model_generator", lambda _model: generate)
    monkeypatch.setattr(eval_command, "console", Console(width=200))
    result = CliRunner().invoke(
        app, ["eval", "gate", "--suite", str(suite), "--model", "synthetic"],
    )

    assert result.exit_code == 2, (result.output, result.exception)
    lines = [" ".join(strip_ansi(line).split()) for line in result.output.splitlines()]
    assert any("GATE FAILED" in line for line in lines)
    assert any(message in line for line in lines)


@pytest.mark.parametrize(
    ("baseline", "error", "message"),
    [
        ([], TypeError, "baseline must be a mapping"),
        ({"shared": "0"}, TypeError, "must be a number"),
        ({"shared": False}, TypeError, "must be a number"),
        ({"shared": float("nan")}, ValueError, "must be finite"),
        ({"shared": float("inf")}, ValueError, "must be finite"),
        ({"shared": float("-inf")}, ValueError, "must be finite"),
    ],
)
def test_baseline_validation_precedes_duplicate_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    baseline: object, error: type[Exception], message: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _suite(tmp_path, ("shared", "shared"))
    suite = gate.EvalSuite.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    with pytest.raises(error, match=message):
        gate.run_gate(suite, generate_fn=lambda prompt: "gold", baseline=baseline)
