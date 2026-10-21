"""Regression tests for eval-gate suite YAML root and syntax validation (#1878)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pydantic
import pytest
import yaml
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import eval as eval_command
from soup_cli.eval import gate, quant_check

from .conftest import strip_ansi

INVALID_SUITES = [
    (
        "- suite: bad-root\n  tasks: []\n",
        "must be a mapping with 'suite' and 'tasks', got a list",
        "sequence-root",
    ),
    (
        "true\n",
        "must be a mapping with 'suite' and 'tasks', got a bool",
        "scalar-root",
    ),
    (
        "[]\n",
        "must be a mapping with 'suite' and 'tasks', got a list",
        "empty-list-root",
    ),
    (
        "false\n",
        "must be a mapping with 'suite' and 'tasks', got a bool",
        "bool-false-root",
    ),
    (
        "0\n",
        "must be a mapping with 'suite' and 'tasks', got a int",
        "int-zero-root",
    ),
    (
        '""\n',
        "must be a mapping with 'suite' and 'tasks', got a str",
        "empty-str-root",
    ),
    (
        "suite: [\n",
        "is not valid YAML:",
        "malformed-yaml",
    ),
    (
        "\t\n",
        "is not valid YAML:",
        "tab-only",
    ),
    (
        "suite: foo\n  tasks:\n - name: bar\n",
        "is not valid YAML:",
        "mis-indented",
    ),
]


def _write_suite_text(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "gate.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def _nonempty_suite(tmp_path: Path) -> dict[str, object]:
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps({"prompt": "control", "expected": "gold"}) + "\n",
        encoding="utf-8",
    )
    return {
        "suite": "control",
        "tasks": [
            {
                "type": "custom",
                "name": "control",
                "tasks": str(tasks),
                "scorer": "exact",
                "threshold": 0.0,
            }
        ],
    }


@pytest.mark.parametrize(
    "content, expected_fragment, _id",
    INVALID_SUITES,
    ids=[s[2] for s in INVALID_SUITES],
)
def test_load_suite_invalid_yaml_raises_value_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    content: str, expected_fragment: str, _id: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _write_suite_text(tmp_path, content)
    with pytest.raises(ValueError) as caught:
        gate.load_suite(str(path))
    assert expected_fragment in str(caught.value)


@pytest.mark.parametrize(
    "content, expected_fragment, _id",
    INVALID_SUITES,
    ids=[s[2] for s in INVALID_SUITES],
)
@pytest.mark.parametrize("with_model", [False, True], ids=["smoke", "model-factory"])
def test_cli_eval_gate_refuses_invalid_suite_with_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    content: str, expected_fragment: str, _id: str, with_model: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _write_suite_text(tmp_path, content)
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
    assert result.exit_code == 3, (result.output, repr(result.exception))
    assert "Cannot load suite:" in output
    assert expected_fragment in output
    assert "GATE PASSED" not in output
    assert "GATE FAILED" not in output
    assert factory_calls == []


def test_cli_control_missing_suite_name_exits_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _write_suite_text(tmp_path, "tasks: []\n")
    monkeypatch.setattr(eval_command, "console", Console(width=200))
    result = CliRunner().invoke(app, ["eval", "gate", "--suite", str(path)])
    output = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 3, (result.output, repr(result.exception))
    assert "Cannot load suite:" in output


def test_load_suite_empty_file_raises_validation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _write_suite_text(tmp_path, "")
    with pytest.raises(pydantic.ValidationError) as caught:
        gate.load_suite(str(path))
    message = str(caught.value)
    assert "1 validation error for EvalSuite" in message
    assert "suite" in message
    assert "Field required" in message


def test_cli_eval_gate_empty_file_exits_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _write_suite_text(tmp_path, "")
    monkeypatch.setattr(eval_command, "console", Console(width=200))
    result = CliRunner().invoke(app, ["eval", "gate", "--suite", str(path)])
    output = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 3, (result.output, repr(result.exception))
    assert "Cannot load suite: 1 validation error for EvalSuite" in output
    assert "Field required" in output


def test_cli_control_valid_suite_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    suite_dict = _nonempty_suite(tmp_path)
    path = _write_suite_text(tmp_path, yaml.safe_dump(suite_dict))
    monkeypatch.setattr(eval_command, "console", Console(width=200))
    result = CliRunner().invoke(app, ["eval", "gate", "--suite", str(path)])
    output = strip_ansi(result.output)
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "GATE PASSED" in output
