"""``prune-layer-cache --model`` refuses a value that is neither a Hub id nor a path (#3088)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi


def _text(output: str | None) -> str:
    return " ".join(strip_ansi(output).split())


@pytest.mark.parametrize("extra", [[], ["--yes"]])
def test_a_value_that_is_neither_a_hub_id_nor_a_path_is_refused(tmp_path, extra):
    result = CliRunner().invoke(
        app,
        ["prune-layer-cache", "--model", "bad name", "--cache-dir", str(tmp_path), *extra],
    )

    assert result.exit_code == 1, (result.output, repr(result.exception))
    text = _text(result.output)
    assert "Invalid --model:" in text
    assert "Nothing to prune." not in text


@pytest.mark.parametrize("extra", [[], ["--yes"]])
def test_a_missing_path_is_refused(tmp_path, extra):
    result = CliRunner().invoke(
        app,
        ["prune-layer-cache", "--model", "./missing", "--cache-dir", str(tmp_path), *extra],
    )

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "Model path not found" in _text(result.output)


def test_a_hub_id_reaches_the_listing(tmp_path):
    result = CliRunner().invoke(
        app, ["prune-layer-cache", "--model", "org/model", "--cache-dir", str(tmp_path)]
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "Nothing to prune." in _text(result.output)


def test_an_existing_directory_reaches_the_listing(tmp_path):
    local = tmp_path / "base"
    local.mkdir()

    result = CliRunner().invoke(
        app,
        ["prune-layer-cache", "--model", str(local), "--cache-dir", str(tmp_path / "cache")],
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "Nothing to prune." in _text(result.output)
