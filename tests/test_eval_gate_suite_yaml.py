"""Regression tests for eval-gate suite YAML root and syntax validation (#1878)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
from unittest.mock import MagicMock

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
        "is not valid YAML: while parsing a flow node",
        "malformed-yaml",
    ),
    (
        "\t\n",
        "is not valid YAML: while scanning for the next token",
        "tab-only",
    ),
    (
        "suite: foo\n  tasks:\n - name: bar\n",
        "is not valid YAML: mapping values are not allowed here",
        "mis-indented",
    ),
    (
        "---\na: 1\n---\nb: 2\n",
        "is not valid YAML: expected a single document in the stream",
        "two-documents",
    ),
    (
        "a: !foo x\n",
        "is not valid YAML: could not determine a constructor",
        "unknown-tag",
    ),
    (
        "a: \x00\n",
        "is not valid YAML: unacceptable character",
        "nul-byte",
    ),
]

# Suffix is the text after ``eval-gate suite '<path>'``.
SCHEMA_FAILURES = [
    (
        "",
        " is empty: it needs 'suite' (a name) and 'tasks' (a list of tasks)",
        "empty-file",
    ),
    (
        "{}\n",
        ": 'suite' is required",
        "empty-mapping",
    ),
    (
        "tasks: []\n",
        ": 'suite' is required",
        "missing-suite",
    ),
    (
        "suite: demo\ntasks:\n  - type: custom\n    name: 1\n    threshold: 0.0\n",
        ": tasks[0].name: Input should be a valid string",
        "bad-task-field",
    ),
    (
        "suite: 1\ntasks:\n  - type: custom\n    name: 2\n    threshold: nope\n",
        ": suite: Input should be a valid string; "
        "tasks[0].name: Input should be a valid string; "
        "tasks[0].threshold: Input should be a valid number, unable to parse string as a number",
        "several-fields",
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
    assert f"eval-gate suite '{path}'" in str(caught.value)
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
    assert "gate.yaml'" in output
    assert expected_fragment in output
    assert "GATE PASSED" not in output
    assert "GATE FAILED" not in output
    assert factory_calls == []


@pytest.mark.parametrize(
    "content", ["suite: [\n", "- suite: bad-root\n  tasks: []\n"],
    ids=["malformed-yaml", "list-root"],
)
def test_training_callback_does_not_silently_disable_a_malformed_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str,
) -> None:
    """Malformed YAML and root shapes still abort setup for a configured gate."""
    from soup_cli.config.schema import EvalGateConfig
    from soup_cli.monitoring.callback import SoupTrainerCallback

    monkeypatch.chdir(tmp_path)
    path = _write_suite_text(tmp_path, content)
    callback = SoupTrainerCallback(
        MagicMock(),
        eval_gate_config=EvalGateConfig(enabled=True, suite=str(path), on_regression="stop"),
    )
    state = SimpleNamespace(max_steps=1, global_step=0, epoch=0)
    callback._build_gate_generate_fn = MagicMock()

    with pytest.raises(gate.MalformedSuiteError) as caught:
        callback.on_train_begin(
            SimpleNamespace(), state, SimpleNamespace(), model=object(), tokenizer=object(),
        )

    assert f"eval-gate suite '{path}'" in str(caught.value)
    assert callback._gate_suite is None
    callback._build_gate_generate_fn.assert_not_called()


def test_training_callback_empty_gate_still_warns_and_stays_inert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """Empty files retain the pre-existing warn-and-inert validation policy."""
    from soup_cli.config.schema import EvalGateConfig
    from soup_cli.monitoring.callback import SoupTrainerCallback

    monkeypatch.chdir(tmp_path)
    path = _write_suite_text(tmp_path, "")
    callback = SoupTrainerCallback(
        MagicMock(),
        eval_gate_config=EvalGateConfig(enabled=True, suite=str(path), on_regression="stop"),
    )
    state = SimpleNamespace(max_steps=1, global_step=0, epoch=0)
    callback._build_gate_generate_fn = MagicMock()

    callback.on_train_begin(
        SimpleNamespace(), state, SimpleNamespace(), model=object(), tokenizer=object(),
    )

    assert "eval gate: could not load suite" in caplog.text
    assert f"eval-gate suite '{path}' is empty" in caplog.text
    assert "errors.pydantic.dev" not in caplog.text
    assert callback._gate_suite is None
    callback._build_gate_generate_fn.assert_not_called()


def _schema_message(path: Path, suffix: str) -> str:
    return f"eval-gate suite '{path}'{suffix}"


@pytest.mark.parametrize(
    "content, suffix, _id",
    SCHEMA_FAILURES,
    ids=[item[2] for item in SCHEMA_FAILURES],
)
def test_load_suite_schema_error_names_file_and_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    content: str, suffix: str, _id: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _write_suite_text(tmp_path, content)
    with pytest.raises(ValueError) as caught:
        gate.load_suite(str(path))
    message = str(caught.value)
    # A subclass would abort training setup; this stays a plain ValueError.
    assert type(caught.value) is ValueError
    assert message == _schema_message(path, suffix)
    assert "errors.pydantic.dev" not in message


@pytest.mark.parametrize(
    "content, suffix, _id",
    SCHEMA_FAILURES,
    ids=[item[2] for item in SCHEMA_FAILURES],
)
def test_cli_eval_gate_schema_error_exits_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    content: str, suffix: str, _id: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _write_suite_text(tmp_path, content)
    # Height is required: under TERM=dumb Rich ignores width alone and folds
    # the path, which whitespace-collapse then splits.
    monkeypatch.setattr(eval_command, "console", Console(width=1000, height=40))
    result = CliRunner().invoke(app, ["eval", "gate", "--suite", str(path)])
    output = " ".join(strip_ansi(result.output).split())
    message = _schema_message(path, suffix)
    assert result.exit_code == 3, (result.output, repr(result.exception))
    assert f"Cannot load suite: {message}" in output
    assert "errors.pydantic.dev" not in output
    assert "validation error for" not in output
    assert "GATE PASSED" not in output
    assert "GATE FAILED" not in output


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
