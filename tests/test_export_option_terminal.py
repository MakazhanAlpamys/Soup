"""Unsupported export option values are data in diagnostics."""

import pytest
from rich.console import Console

from soup_cli.cli import app
from soup_cli.commands import export
from tests.conftest import make_cli_runner, strip_ansi


@pytest.mark.parametrize(
    ("value", "visible"),
    [
        ("bad", "bad"),
        ("bad[red]x", "bad[red]x"),
        ("bad[/]", "bad[/]"),
        (r"bad\[unclosed", r"bad\[unclosed"),
        ("bad:warning:", "bad:warning:"),
        ("bad\x9b\u202e", "bad"),
    ],
)
@pytest.mark.parametrize("option", ["--format", "--deploy"])
def test_unsupported_export_option_is_literal(value, visible, option, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(export, "console", Console(width=240, height=80))
    (tmp_path / "model").mkdir()

    def unexpected(*args, **kwargs):
        pytest.fail("unsupported option must refuse before converter work")

    monkeypatch.setattr(export, "_check_gguf_capabilities", unexpected)

    result = make_cli_runner().invoke(app, ["export", "--model", "model", option, value])

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert isinstance(result.exception, SystemExit)
    assert "\x9b" not in result.stdout and "\u202e" not in result.stdout
    text = strip_ansi(result.stdout)
    if option == "--format":
        assert text.splitlines()[0] == f"Unsupported format: {visible}"
        assert "Supported:" in text
    else:
        assert text.splitlines()[0] == f"Unsupported deploy target: {visible}"
        assert text.splitlines()[1] == "Supported: ollama"
    assert not list(tmp_path.glob("*.gguf"))
