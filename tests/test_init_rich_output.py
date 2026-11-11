"""Rich init output keeps emoji aliases literal while stripping terminal controls."""

from pathlib import Path

import pytest
from rich.console import Console

from soup_cli.cli import app
from soup_cli.commands import init as init_cmd
from soup_cli.templates import load_template
from tests._windows_filenames import skip_colon_on_windows
from tests.conftest import make_cli_runner, strip_ansi

runner = make_cli_runner()

_CONTROLS = "\x9b\u202e"
_VISIBLE = "starter[bold]:rocket:"
_FILENAME = f"{_VISIBLE}{_CONTROLS}.yaml"


@pytest.fixture(autouse=True)
def _isolated_console(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(init_cmd, "console", Console(width=200, height=100))


def _output(result, expected_exit_code: int) -> str:
    assert result.exit_code == expected_exit_code, (result.output, repr(result.exception))
    if expected_exit_code:
        assert isinstance(result.exception, SystemExit)
    else:
        assert result.exception is None
    assert result.stderr == ""
    assert "\x9b" not in result.stdout and "\u202e" not in result.stdout
    return strip_ansi(result.stdout)


def test_unknown_template_keeps_its_emoji_alias_literal():
    result = runner.invoke(app, ["init", "--template", f"{_VISIBLE}{_CONTROLS}"])

    text = _output(result, 1)
    assert f"Unknown template: {_VISIBLE}" in text.splitlines()
    assert "Available:" in text
    assert not Path("soup.yaml").exists()


@skip_colon_on_windows
@pytest.mark.requires_symlink
def test_link_refusal_keeps_its_emoji_alias_literal():
    victim = Path("victim.txt")
    victim.write_bytes(b"must not be overwritten\n")
    link = Path(_FILENAME)
    link.symlink_to(victim)

    result = runner.invoke(app, ["init", "--template", "chat", "--output", _FILENAME, "--force"])

    assert _output(result, 1).splitlines() == [
        f"{_VISIBLE}.yaml is a symbolic link, junction or other reparse "
        "point; soup init does not write through it.",
        "Remove it or choose a different --output.",
    ]
    assert link.is_symlink()
    assert victim.read_bytes() == b"must not be overwritten\n"


@skip_colon_on_windows
def test_saved_path_panel_keeps_its_emoji_alias_literal():
    result = runner.invoke(app, ["init", "--template", "chat", "--output", _FILENAME])

    text = _output(result, 0)
    assert f"Config saved to {_VISIBLE}.yaml" in text
    assert f"Next step: soup train --config {_VISIBLE}.yaml" in text
    assert Path(_FILENAME).read_text(encoding="utf-8") == load_template("chat")
    assert not Path(f"{_VISIBLE}.yaml").exists()
