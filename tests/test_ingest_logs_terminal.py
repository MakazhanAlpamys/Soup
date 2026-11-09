"""Offline ingest diagnostics display the original log path safely."""

import os
from pathlib import Path

import pytest
from rich.console import Console

from soup_cli.cli import app
from soup_cli.commands import ingest as ingest_cmd
from tests.conftest import make_cli_runner, strip_ansi


@pytest.mark.parametrize(
    ("suffix", "displayed"),
    [
        pytest.param("plain", "plain", id="plain-control"),
        pytest.param("[bold]x", "[bold]x", id="markup-control"),
        pytest.param("[red]x\x9b\u202e", "[red]x", id="terminal-controls"),
        pytest.param(r"\[unclosed", r"\[unclosed", id="backslash-before-bracket"),
        pytest.param(
            ":warning:",
            ":warning:",
            id="emoji-alias",
            marks=pytest.mark.skipif(os.name == "nt", reason="Colon is not a Windows filename"),
        ),
    ],
)
@pytest.mark.parametrize("failure", ["outside", "missing", "directory", "invalid-utf8"])
def test_ingest_offline_log_path_diagnostics(tmp_path, monkeypatch, suffix, displayed, failure):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setattr(ingest_cmd, "console", Console(width=220, height=40))
    log_arg = f"log-{suffix}.jsonl"
    display_path = f"log-{displayed}.jsonl"
    output = tmp_path / "output.jsonl"
    output.write_bytes(b"previous output\n")
    if failure == "outside":
        log_arg = f"../{log_arg}"
        expected = f"--logs '../{display_path}' is outside cwd — refusing"
    elif failure == "missing":
        expected = f"--logs not found: {display_path}"
    elif failure == "directory":
        Path(log_arg).mkdir(parents=True)
        expected = f"--logs '{display_path}' is a directory, not an export file"
    else:
        log_path = Path(log_arg)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_bytes(b"\xff\n")
        expected = f"--logs '{display_path}' is not valid UTF-8 (byte 0)"

    result = make_cli_runner().invoke(
        app,
        ["ingest", "--source", "langfuse", "--logs", log_arg, "--output", output.name],
    )

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert isinstance(result.exception, SystemExit) and result.exception.code == 1
    assert output.read_bytes() == b"previous output\n"
    if failure == "invalid-utf8":
        assert Path(log_arg).read_bytes() == b"\xff\n"
    assert "\x9b" not in result.stdout and "\u202e" not in result.stdout
    assert strip_ansi(result.stdout).splitlines()[-1] == expected
