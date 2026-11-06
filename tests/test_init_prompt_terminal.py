"""Overwrite prompts strip controls while preserving literal paths and real file targets."""

import hashlib

import pytest
from rich.console import Console

from soup_cli.cli import app
from soup_cli.templates import load_template
from tests.conftest import make_cli_runner

runner = make_cli_runner()

ORIGINAL = b"# preserve these original bytes\r\nbase: before\n"
CONTROL_NAME = "[red]config\x9b\u202e.yaml"
VISIBLE_NAME = "[red]config.yaml"


@pytest.fixture(autouse=True)
def _prompt_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("soup_cli.commands.init.console", Console(width=500, height=100))


def _assert_safe_prompt(result, visible):
    # Inspect the captured stream before any ANSI cleaning can hide controls.
    assert "\x9b" not in result.stdout and "\u202e" not in result.stdout
    assert result.stdout.startswith(f"{visible} already exists. Overwrite?")
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("name", "visible"),
    [
        pytest.param("config.yaml", "config.yaml", id="plain"),
        pytest.param(VISIBLE_NAME, VISIBLE_NAME, id="brackets"),
        pytest.param(CONTROL_NAME, VISIBLE_NAME, id="controls-and-tag"),
    ],
)
def test_declining_overwrite_keeps_original_bytes(tmp_path, name, visible):
    output = tmp_path / name
    output.write_bytes(ORIGINAL)
    before_hash = hashlib.sha256(output.read_bytes()).hexdigest()

    result = runner.invoke(
        app, ["init", "--template", "chat", "--output", name], input="n\n"
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert result.exception is None
    assert output.read_bytes() == ORIGINAL
    assert hashlib.sha256(output.read_bytes()).hexdigest() == before_hash
    _assert_safe_prompt(result, visible)
    assert "Using template:" not in result.stdout
    assert "Config saved to" not in result.stdout


def test_confirmed_overwrite_writes_typed_path_and_preserves_visible_sibling(tmp_path):
    output = tmp_path / CONTROL_NAME
    sibling = tmp_path / VISIBLE_NAME
    output.write_bytes(ORIGINAL)
    sibling_bytes = b"# keep the sanitized-name sibling\r\nbase: untouched\n"
    sibling.write_bytes(sibling_bytes)
    sibling_hash = hashlib.sha256(sibling.read_bytes()).hexdigest()

    result = runner.invoke(
        app, ["init", "--template", "chat", "--output", CONTROL_NAME], input="y\n"
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert result.exception is None
    assert output.read_text(encoding="utf-8") == load_template("chat")
    assert sibling.read_bytes() == sibling_bytes
    assert hashlib.sha256(sibling.read_bytes()).hexdigest() == sibling_hash
    _assert_safe_prompt(result, VISIBLE_NAME)
