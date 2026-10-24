"""#1878: eval gate: malformed or non-object suite YAML exits 1 instead of usage error 3.

Invalid eval-gate suite YAML must consistently reach the usage-error boundary (EXIT_USAGE_ERROR = 3)
rather than raising an unhandled TypeError or ParserError with exit code 1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.eval import quant_check
from soup_cli.eval.gate import EvalSuite, load_suite
from tests.conftest import strip_ansi

runner = CliRunner()


def _write_tasks_file(tmp_path: Path) -> Path:
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text(
        json.dumps({"prompt": "2+2=", "expected": "4", "scorer": "exact"}) + "\n",
        encoding="utf-8",
    )
    return tasks_file


def _valid_suite_yaml(tasks_file: Path) -> str:
    return (
        "suite: control-gate\n"
        "tasks:\n"
        "  - type: custom\n"
        "    name: math\n"
        f"    tasks: {tasks_file.name}\n"
        "    scorer: exact\n"
        "    threshold: 0.0\n"
    )


class TestIssue1878LoaderNormalization:
    """load_suite must normalize yaml.YAMLError and non-mapping roots to ValueError."""

    def test_load_suite_rejects_list_root(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        suite = tmp_path / "list_root.yaml"
        suite.write_text("- suite: bad-root\n  tasks: []\n", encoding="utf-8")
        with pytest.raises(
            ValueError,
            match=r"must be a mapping with 'suite' and 'tasks', got a list",
        ):
            load_suite("list_root.yaml")

    def test_load_suite_rejects_scalar_root(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        suite = tmp_path / "scalar_root.yaml"
        suite.write_text("true\n", encoding="utf-8")
        with pytest.raises(
            ValueError,
            match=r"must be a mapping with 'suite' and 'tasks', got a bool",
        ):
            load_suite("scalar_root.yaml")

    def test_load_suite_rejects_malformed_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        suite = tmp_path / "malformed.yaml"
        suite.write_text("suite: [\n", encoding="utf-8")
        with pytest.raises(ValueError, match="is not valid YAML:"):
            load_suite("malformed.yaml")

    def test_load_suite_rejects_tab_only_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        suite = tmp_path / "tab_only.yaml"
        suite.write_text("\t\n", encoding="utf-8")
        with pytest.raises(ValueError, match="is not valid YAML:"):
            load_suite("tab_only.yaml")

    def test_load_suite_rejects_misindented_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        suite = tmp_path / "misindented.yaml"
        suite.write_text("suite: control\n tasks:\n- bad: indentation\n", encoding="utf-8")
        with pytest.raises(ValueError, match="is not valid YAML:"):
            load_suite("misindented.yaml")

    def test_control_invalid_mapping_raises_value_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        suite = tmp_path / "invalid_mapping.yaml"
        suite.write_text("tasks: []\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_suite("invalid_mapping.yaml")

    def test_control_valid_suite_loads_successfully(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        tasks_file = _write_tasks_file(tmp_path)
        suite = tmp_path / "valid.yaml"
        suite.write_text(_valid_suite_yaml(tasks_file), encoding="utf-8")
        loaded = load_suite("valid.yaml")
        assert isinstance(loaded, EvalSuite)
        assert loaded.suite == "control-gate"
        assert len(loaded.tasks) == 1


class TestIssue1878EvalGateCLIUsageError:
    """soup eval gate CLI must exit 3 on malformed or non-object YAML with or without --model."""

    @pytest.mark.parametrize(
        ("bad_content", "expected_fragment"),
        [
            (
                "- suite: bad-root\n  tasks: []\n",
                "must be a mapping with 'suite' and 'tasks', got a list",
            ),
            (
                "true\n",
                "must be a mapping with 'suite' and 'tasks', got a bool",
            ),
            (
                "suite: [\n",
                "is not valid YAML:",
            ),
            (
                "\t\n",
                "is not valid YAML:",
            ),
            (
                "suite: control\n tasks:\n- bad: indentation\n",
                "is not valid YAML:",
            ),
        ],
    )
    def test_eval_gate_cli_exits_3_without_model(
        self, tmp_path, monkeypatch, bad_content, expected_fragment
    ):
        monkeypatch.chdir(tmp_path)
        suite = tmp_path / "bad.yaml"
        suite.write_text(bad_content, encoding="utf-8")

        result = runner.invoke(app, ["eval", "gate", "--suite", "bad.yaml"])
        assert result.exit_code == 3, (result.output, repr(result.exception))
        cleaned = " ".join(strip_ansi(result.output).split())
        assert "Cannot load suite:" in cleaned
        assert expected_fragment in cleaned

    @pytest.mark.parametrize(
        ("bad_content", "expected_fragment"),
        [
            (
                "- suite: bad-root\n  tasks: []\n",
                "must be a mapping with 'suite' and 'tasks', got a list",
            ),
            (
                "true\n",
                "must be a mapping with 'suite' and 'tasks', got a bool",
            ),
            (
                "suite: [\n",
                "is not valid YAML:",
            ),
            (
                "\t\n",
                "is not valid YAML:",
            ),
            (
                "suite: control\n tasks:\n- bad: indentation\n",
                "is not valid YAML:",
            ),
        ],
    )
    def test_eval_gate_cli_exits_3_with_model(
        self, tmp_path, monkeypatch, bad_content, expected_fragment
    ):
        monkeypatch.chdir(tmp_path)
        suite = tmp_path / "bad.yaml"
        suite.write_text(bad_content, encoding="utf-8")

        factory_called = False

        def mock_generator(model: str) -> Callable[[str], str]:
            nonlocal factory_called
            factory_called = True
            return lambda prompt: ""

        monkeypatch.setattr(quant_check, "make_model_generator", mock_generator)

        result = runner.invoke(
            app, ["eval", "gate", "--suite", "bad.yaml", "--model", "synthetic-model"]
        )
        assert result.exit_code == 3, (result.output, repr(result.exception))
        cleaned = " ".join(strip_ansi(result.output).split())
        assert "Cannot load suite:" in cleaned
        assert expected_fragment in cleaned
        assert not factory_called

    def test_control_invalid_mapping_cli_exits_3(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        suite = tmp_path / "no_name.yaml"
        suite.write_text("tasks: []\n", encoding="utf-8")

        result = runner.invoke(app, ["eval", "gate", "--suite", "no_name.yaml"])
        assert result.exit_code == 3, (result.output, repr(result.exception))
        cleaned = " ".join(strip_ansi(result.output).split())
        assert "Cannot load suite:" in cleaned

    def test_control_valid_suite_cli_exits_0(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        tasks_file = _write_tasks_file(tmp_path)
        suite = tmp_path / "valid.yaml"
        suite.write_text(_valid_suite_yaml(tasks_file), encoding="utf-8")

        result = runner.invoke(app, ["eval", "gate", "--suite", "valid.yaml"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
