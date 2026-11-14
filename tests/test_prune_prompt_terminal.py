"""The prune report displays a literal preview of the original prompt prefix."""

import json

import pytest
from rich.console import Console

from soup_cli.cli import app
from soup_cli.commands import prune_prompt
from tests.conftest import make_cli_runner, strip_ansi


@pytest.mark.parametrize(
    ("prefix", "displayed"),
    [
        pytest.param("plain prefix", "plain prefix", id="plain-control"),
        pytest.param("sys [red]x[/]", "sys [red]x[/]", id="markup-control"),
        pytest.param("sys :warning:", "sys :warning:", id="emoji-alias"),
        pytest.param("sys [bold]x\x9b\u202e", "sys [bold]x", id="terminal-controls"),
        pytest.param(r"sys \[unclosed", r"sys \[unclosed", id="backslash-before-bracket"),
        pytest.param("T" * 198 + "\x9b\u202e" + "hidden", "T" * 198 + "...", id="truncation"),
    ],
)
def test_prune_prefix_preview_is_literal(tmp_path, monkeypatch, prefix, displayed):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setattr(
        prune_prompt,
        "console",
        Console(
            width=260,
            height=40,
            color_system="standard",
            force_terminal=True,
            legacy_windows=False,
            no_color=False,
        ),
    )
    source = tmp_path / "traces.jsonl"
    destination = tmp_path / "pruned.jsonl"
    remainder = [
        {"prompt": "A\x9b\u202e", "output": "keep :warning:", "metadata": {"source": "one"}},
        {"prompt": "B", "output": "keep [bold]x\x9b", "metadata": {"source": "two"}},
    ]
    rows = [dict(row, prompt=prefix + row["prompt"]) for row in remainder]
    original = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
    source.write_bytes(original)

    result = make_cli_runner().invoke(
        app,
        [
            "prune-prompt",
            "--input",
            source.name,
            "--output",
            destination.name,
            "--min-frequency",
            "1",
        ],
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert source.read_bytes() == original
    assert [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()] == (
        remainder
    )
    stdout = result.stdout
    assert "\x9b" not in stdout and "\u202e" not in stdout
    assert "\x1b[32mStripped prefix" in stdout
    assert "\x1b[2mPrefix:\x1b[0m" in stdout
    rendered = strip_ansi(stdout)
    assert f"Stripped prefix ({len(prefix)} chars) from 2/2 rows." in rendered
    prefix_line = next(line for line in rendered.splitlines() if "Prefix: " in line)
    assert prefix_line.split("Prefix: ", 1)[1].rstrip(" │") == displayed
