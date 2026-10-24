"""#1915: eval gate: a suite mapping with a non-string key exits 1 instead of usage error 3.

``EvalSuite(**data)`` raises ``TypeError: keywords must be strings`` for such a mapping, which the
CLI does not catch. ``load_suite`` must refuse it with a ``ValueError`` naming the key.
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

_BAD_KEYS = [
    ("1", "1"),
    ("true", "True"),
    ("~", "None"),
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
        with pytest.raises(ValueError, match="non-string key") as excinfo:
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
        try:
            load_suite("nested.yaml")
        except ValueError as exc:  # a nested rejection is fine, but not our new message
            assert "non-string key" not in str(exc)
        except Exception as exc:  # noqa: BLE001
            assert "keywords must be strings" not in str(exc)


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
        baseline = tmp_path / "baseline.json"
        result = runner.invoke(
            app,
            [
                "eval", "gate", "--suite", "bad.yaml", "--model", "synthetic-model",
                "--baseline", str(baseline),
            ],
        )
        assert result.exit_code == 3, (result.output, repr(result.exception))
        assert "Cannot load suite:" in " ".join(strip_ansi(result.output).split())
        assert not factory_called
        assert not baseline.exists()

    def test_control_valid_suite_cli_exits_0(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        tasks_file = _write_tasks_file(tmp_path)
        (tmp_path / "ok.yaml").write_text(_suite_yaml(tasks_file), encoding="utf-8")
        result = runner.invoke(app, ["eval", "gate", "--suite", "ok.yaml"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
