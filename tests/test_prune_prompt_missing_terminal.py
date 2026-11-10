"""Missing prune inputs are diagnosed without interpreting the typed path."""

from pathlib import Path

import pytest
from rich.console import Console

from soup_cli.cli import app
from soup_cli.commands import prune_prompt
from tests.conftest import make_cli_runner, strip_ansi


@pytest.mark.parametrize(
    ("name", "visible"),
    [
        ("missing.jsonl", "missing.jsonl"),
        ("missing[red]x.jsonl", "missing[red]x.jsonl"),
        (r"missing\[unclosed.jsonl", r"missing\[unclosed.jsonl"),
        ("missing:warning:.jsonl", "missing:warning:.jsonl"),
        ("missing\x9b\u202e.jsonl", "missing.jsonl"),
    ],
)
def test_missing_prune_input_is_literal(name, visible, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(prune_prompt, "console", Console(width=240, height=80))
    # If controls were removed from the lookup, the last case would incorrectly succeed.
    Path("missing.jsonl").write_text('{"prompt":"A"}\n{"prompt":"B"}\n', encoding="utf-8")
    if name == "missing.jsonl":
        Path(name).unlink()
    destination = Path("result.jsonl")
    destination.write_bytes(b"preserve me\n")

    result = make_cli_runner().invoke(
        app, ["prune-prompt", "--input", name, "--output", str(destination)]
    )

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert isinstance(result.exception, SystemExit)
    assert "\x9b" not in result.stdout and "\u202e" not in result.stdout
    assert strip_ansi(result.stdout) == f"Input not found: {visible}\n"
    assert destination.read_bytes() == b"preserve me\n"
