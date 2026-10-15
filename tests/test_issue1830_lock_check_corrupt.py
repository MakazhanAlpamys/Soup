"""Regression tests for issue #1830: soup lock check and show on corrupt lock files."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi

runner = CliRunner()


def test_lock_check_corrupt_file_names_file_and_explains_json_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """soup lock check on corrupt JSON prints the file name, 'not valid JSON', and exits 3."""
    monkeypatch.chdir(tmp_path)
    corrupt_lock = tmp_path / "soup.lock"
    corrupt_lock.write_text("garbage: [")

    result = runner.invoke(
        app,
        [
            "lock",
            "check",
            "soup.lock",
            "--base-sha",
            "0" * 64,
            "--dataset-sha",
            "0" * 64,
            "--env-hash",
            "0" * 64,
            "--base-model",
            "meta-llama/Llama-3-8B",
        ],
    )
    assert result.exit_code == 3
    out = strip_ansi(result.output)
    assert "soup.lock" in out
    assert "not valid JSON" in out


def test_lock_show_corrupt_file_names_file_and_explains_json_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """soup lock show on corrupt JSON prints the file name, 'not valid JSON', and exits 3."""
    monkeypatch.chdir(tmp_path)
    corrupt_lock = tmp_path / "custom.lock"
    corrupt_lock.write_text("garbage: [")

    result = runner.invoke(app, ["lock", "show", "custom.lock"])
    assert result.exit_code == 3
    out = strip_ansi(result.output)
    assert "custom.lock" in out
    assert "not valid JSON" in out
