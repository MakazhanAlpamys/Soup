"""Regression tests for issue #1830: soup expect error messages and docstring."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import expect
from tests.conftest import strip_ansi

runner = CliRunner()


def _flat(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def test_expect_module_docstring_exit_code() -> None:
    """Module docstring must document exit code 2 (not 3) on suite failure."""
    flat_doc = _flat(expect.__doc__ or "")
    assert "Exit code 3 on suite failure" not in flat_doc
    assert "Exit code 2 on suite failure" in flat_doc


def test_expect_missing_data_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing data file prints descriptive error with file path and exits 3."""
    monkeypatch.chdir(tmp_path)
    suite = tmp_path / "suite.yaml"
    suite.write_text("expectations:\n  - name: expect_no_pii\n")

    result = runner.invoke(app, ["expect", "missing_data.jsonl", str(suite)])
    assert result.exit_code == 3
    out = strip_ansi(result.output)
    assert "data file not found: 'missing_data.jsonl'" in out


def test_expect_missing_suite_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing suite file prints descriptive error with file path and exits 3."""
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data.jsonl"
    data.write_text('{"text": "sample"}\n')

    result = runner.invoke(app, ["expect", str(data), "missing_suite.yaml"])
    assert result.exit_code == 3
    out = strip_ansi(result.output)
    assert "suite file not found: 'missing_suite.yaml'" in out


def test_expect_data_is_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Directory supplied as data file prints not-a-regular-file error and exits 3."""
    monkeypatch.chdir(tmp_path)
    suite = tmp_path / "suite.yaml"
    suite.write_text("expectations:\n  - name: expect_no_pii\n")
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()

    result = runner.invoke(app, ["expect", "data_dir", str(suite)])
    assert result.exit_code == 3
    out = strip_ansi(result.output)
    assert "data file is not a regular file: 'data_dir'" in out


def test_expect_suite_is_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Directory supplied as suite file prints not-a-regular-file error and exits 3."""
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data.jsonl"
    data.write_text('{"text": "sample"}\n')
    suite_dir = tmp_path / "suite_dir"
    suite_dir.mkdir()

    result = runner.invoke(app, ["expect", str(data), "suite_dir"])
    assert result.exit_code == 3
    out = _flat(result.output)
    real_dir = os.path.realpath("suite_dir")
    assert f"suite file is not a regular file: '{real_dir}'" in out
