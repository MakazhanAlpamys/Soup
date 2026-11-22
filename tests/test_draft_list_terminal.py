"""Registry text stays literal when Rich renders the draft list table."""

import json

import pytest
from rich.console import Console

from soup_cli.cli import app
from soup_cli.commands import draft as draft_cmd
from tests.conftest import make_cli_runner, strip_ansi


@pytest.mark.parametrize(
    ("suffix", "displayed"),
    [
        pytest.param("plain", "plain", id="plain-control"),
        pytest.param("[bold]x[/]", "[bold]x[/]", id="markup-control"),
        pytest.param("[red]x\x9b\u202e", "[red]x", id="terminal-controls-control"),
        pytest.param(":warning:", ":warning:", id="emoji-alias"),
        pytest.param(r"\[unclosed", r"\[unclosed", id="backslash-before-bracket"),
    ],
)
def test_draft_list_displays_stored_values_literally(tmp_path, monkeypatch, suffix, displayed):
    registry = tmp_path / "drafts.json"
    original = json.dumps(
        {
            "drafts": [
                {"target": "older", "draft": "old-path", "created": "2026-10-01"},
                {
                    "target": f"target-{suffix}",
                    "draft": f"draft-{suffix}",
                    "created": f"created-{suffix}",
                    "acceptance_rate": 0.625,
                },
            ]
        }
    ).encode("utf-8")
    registry.write_bytes(original)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setenv("SOUP_DRAFT_REGISTRY_PATH", str(registry))
    monkeypatch.setattr(draft_cmd, "console", Console(width=220, height=40, legacy_windows=False))

    result = make_cli_runner().invoke(app, ["draft", "list"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert registry.read_bytes() == original
    assert "\x9b" not in result.stdout and "\u202e" not in result.stdout
    rendered = strip_ansi(result.stdout)
    rows = [
        [cell.strip() for cell in line.split("│")[1:-1]]
        for line in rendered.splitlines()
        if "target-" in line or "old-path" in line
    ]
    assert rows == [
        [f"target-{displayed}", f"draft-{displayed}", "62.5%", f"created-{displayed}"],
        ["older", "old-path", "-", "2026-10-01"],
    ]
