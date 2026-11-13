"""#1915: eval gate: a suite mapping with a non-string key exits 1 instead of usage error 3.

``EvalSuite(**data)`` raises ``TypeError: keywords must be strings`` for such a mapping, which the
CLI does not catch. ``load_suite`` must refuse it with a ``MalformedSuiteError`` (a ``ValueError``)
naming the key, so a configured training gate aborts instead of going inert.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.eval import quant_check
from soup_cli.eval.gate import EvalSuite, MalformedSuiteError, load_suite
from tests.conftest import strip_ansi

runner = CliRunner()

_BAD_KEYS = [
    ("1", "1"),
    ("true", "True"),
    ("~", "None"),
    ("2024-01-01", "datetime.date(2024, 1, 1)"),
]


def _write_tasks_file(tmp_path: Path) -> Path:
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text(
        json.dumps({"prompt": "2+2=", "expected": "4", "scorer": "exact"}) + "\n",
        encoding="utf-8",
    )
    return tasks_file


def _suite_yaml(tasks_file: Path, extra_line: str = "") -> str:
    return (
        "suite: control-gate\n"
        f"{extra_line}"
        "tasks:\n"
        "  - type: custom\n"
        "    name: math\n"
        f"    tasks: {tasks_file.name}\n"
        "    scorer: exact\n"
        "    threshold: 0.0\n"
    )


class TestIssue1915LoaderRejectsNonStringKey:
    @pytest.mark.parametrize(("yaml_key", "shown"), _BAD_KEYS)
    def test_load_suite_rejects_non_string_key(self, tmp_path, monkeypatch, yaml_key, shown):
        monkeypatch.chdir(tmp_path)
        tasks_file = _write_tasks_file(tmp_path)
        (tmp_path / "bad.yaml").write_text(
            _suite_yaml(tasks_file, f"{yaml_key}: x\n"), encoding="utf-8"
        )
        with pytest.raises(MalformedSuiteError, match="non-string key") as excinfo:
            load_suite("bad.yaml")
        message = str(excinfo.value)
        assert "bad.yaml" in message
        assert f"key {shown};" in message

    def test_control_valid_suite_loads(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        tasks_file = _write_tasks_file(tmp_path)
        (tmp_path / "ok.yaml").write_text(_suite_yaml(tasks_file), encoding="utf-8")
        loaded = load_suite("ok.yaml")
        assert isinstance(loaded, EvalSuite)
        assert len(loaded.tasks) == 1

    def test_control_nested_non_string_value_keys_unaffected(self, tmp_path, monkeypatch):
        # Only the top-level mapping keys are checked; nested task fields keep their own handling.
        monkeypatch.chdir(tmp_path)
        tasks_file = _write_tasks_file(tmp_path)
        (tmp_path / "nested.yaml").write_text(
            _suite_yaml(tasks_file) + "    options:\n      1: kept\n", encoding="utf-8"
        )
        assert load_suite("nested.yaml").tasks[0].name == "math"


class TestIssue1915TrainingCallbackAborts:
    def test_training_callback_does_not_silently_disable_a_gate_with_a_non_string_key(
        self, tmp_path, monkeypatch
    ):
        """A plain ValueError would only be logged and leave the gate inert."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from soup_cli.config.schema import EvalGateConfig
        from soup_cli.monitoring.callback import SoupTrainerCallback

        monkeypatch.chdir(tmp_path)
        tasks_file = _write_tasks_file(tmp_path)
        path = tmp_path / "bad.yaml"
        path.write_text(_suite_yaml(tasks_file, "1: x\n"), encoding="utf-8")
        callback = SoupTrainerCallback(
            MagicMock(),
            eval_gate_config=EvalGateConfig(
                enabled=True, suite=str(path), on_regression="stop"
            ),
        )
        state = SimpleNamespace(max_steps=1, global_step=0, epoch=0)
        callback._build_gate_generate_fn = MagicMock()

        with pytest.raises(MalformedSuiteError, match="non-string key") as caught:
            callback.on_train_begin(
                SimpleNamespace(), state, SimpleNamespace(),
                model=object(), tokenizer=object(),
            )

        assert f"eval-gate suite '{path}'" in str(caught.value)
        assert callback._gate_suite is None
        callback._build_gate_generate_fn.assert_not_called()


class TestIssue1915EvalGateCLIUsageError:
    @pytest.mark.parametrize(("yaml_key", "shown"), _BAD_KEYS)
    def test_cli_exits_3_without_model(self, tmp_path, monkeypatch, yaml_key, shown):
        monkeypatch.chdir(tmp_path)
        tasks_file = _write_tasks_file(tmp_path)
        (tmp_path / "bad.yaml").write_text(
            _suite_yaml(tasks_file, f"{yaml_key}: x\n"), encoding="utf-8"
        )
        result = runner.invoke(app, ["eval", "gate", "--suite", "bad.yaml"])
        assert result.exit_code == 3, (result.output, repr(result.exception))
        cleaned = " ".join(strip_ansi(result.output).split())
        assert "Cannot load suite:" in cleaned
        assert f"non-string key {shown};" in cleaned

    def test_cli_exits_3_builds_no_model_and_writes_no_baseline(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        tasks_file = _write_tasks_file(tmp_path)
        (tmp_path / "bad.yaml").write_text(
            _suite_yaml(tasks_file, "1: x\n"), encoding="utf-8"
        )
        factory_called = False

        def mock_generator(model: str) -> Callable[[str], str]:
            nonlocal factory_called
            factory_called = True
            return lambda prompt: ""

        monkeypatch.setattr(quant_check, "make_model_generator", mock_generator)
        result = runner.invoke(
            app,
            [
                "eval", "gate", "--suite", "bad.yaml", "--model", "synthetic-model",
                "--write-baseline", "baseline.json",
            ],
        )
        assert result.exit_code == 3, (result.output, repr(result.exception))
        assert "Cannot load suite:" in " ".join(strip_ansi(result.output).split())
        assert not factory_called
        assert not (tmp_path / "baseline.json").exists()

    def test_control_valid_suite_cli_exits_0(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        tasks_file = _write_tasks_file(tmp_path)
        (tmp_path / "ok.yaml").write_text(_suite_yaml(tasks_file), encoding="utf-8")
        result = runner.invoke(app, ["eval", "gate", "--suite", "ok.yaml"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
