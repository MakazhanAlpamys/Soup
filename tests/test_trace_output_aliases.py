import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi


@pytest.mark.parametrize("command", ["ingest", "prune-prompt"])
@pytest.mark.parametrize(
    "alias_kind",
    [
        pytest.param("hardlink", marks=pytest.mark.requires_hardlink),
        "windows-case",
        "dotdot",
    ],
)
def test_trace_command_rejects_input_alias_without_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    alias_kind: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    source = Path("traces.jsonl")
    if command == "ingest":
        original = b'{"id":"t1","input":"hi","output":"hello"}\n'
    else:
        original = (
            b'{"prompt":"SYSTEM alpha","output":"one"}\n'
            b'{"prompt":"SYSTEM beta","output":"two"}\n'
        )
    source.write_bytes(original)

    if alias_kind == "hardlink":
        linked = Path("linked-traces.jsonl")
        linked.hardlink_to(source)
        alias = str(linked)
    elif alias_kind == "windows-case":
        if os.name != "nt":
            pytest.skip("Windows case-alias test")
        alias = "TRACES.JSONL"
        if not os.path.exists(alias) or not os.path.samefile(source, alias):
            pytest.skip("Working directory is case-sensitive")
    else:
        Path("unused").mkdir()
        alias = str(Path("unused") / ".." / source.name)

    before_files = {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
        if path.is_file()
    }
    before_entries = {path.name for path in tmp_path.iterdir()}

    if command == "ingest":
        arguments = [
            "ingest",
            "--source",
            "langfuse",
            "--logs",
            str(source),
            "--output",
            alias,
        ]
        input_option = "--logs"
    else:
        arguments = [
            "prune-prompt",
            "--input",
            str(source),
            "--output",
            alias,
        ]
        input_option = "--input"

    result = CliRunner().invoke(app, arguments)
    message = " ".join(strip_ansi(result.output).split())

    after_files = {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
        if path.is_file()
    }
    assert after_files == before_files
    assert {path.name for path in tmp_path.iterdir()} == before_entries
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert input_option in message
    assert "--output" in message
    assert "same file" in message
