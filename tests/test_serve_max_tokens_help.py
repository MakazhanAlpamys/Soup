"""The serve help states the same token range the command enforces."""

from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi


def test_max_tokens_help_states_the_inclusive_range(monkeypatch) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    result = CliRunner().invoke(app, ["serve", "--help"], terminal_width=160)
    assert result.exit_code == 0, repr(result.exception)
    output = " ".join(strip_ansi(result.output).replace("│", " ").split())
    assert "Default max tokens for generation (1 to 16384)" in output
