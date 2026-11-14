"""refuse_output_over_input names the refused path as typed, not as Rich markup."""

import io
from pathlib import Path

import pytest
import typer
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands._common import refuse_output_over_input
from soup_cli.utils.paths import quote_path
from tests._windows_filenames import skip_colon_on_windows
from tests.conftest import strip_ansi

runner = CliRunner()

_CONTROLS = "\x9b\u202e"


# One spelling per way the old code went wrong: a tag, a closing tag with
# nothing to close (a MarkupError), an emoji code, and a tag with control
# characters, so the strip and the escape are pinned separately.
_NAMES = [
    "x[red]n.jsonl",
    "[/]n.jsonl",
    pytest.param("n:warning:.jsonl", id="emoji-code"),
    f"x[bold]n{_CONTROLS}.jsonl",
]


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def _expected(name: str) -> str:
    # quote_path already shows C0/C1 controls as \xNN text, so only the bidi
    # override is left for the strip to remove.
    quoted = quote_path(name).replace("\u202e", "")
    return f"--output {quoted} is the same file as the input {quoted}; choose a different --output"


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


@pytest.mark.parametrize("name", _NAMES)
def test_the_refused_path_is_named_as_typed(name):
    buffer = io.StringIO()
    console = Console(file=buffer, width=500, height=100)

    with pytest.raises(typer.Exit) as raised:
        refuse_output_over_input(
            [("the input", name)], [("--output", name)], console=console
        )

    assert raised.value.exit_code == 1
    text = _plain(buffer.getvalue())
    assert _expected(name) in text
    assert _not_shown(text)


@skip_colon_on_windows
def test_a_command_names_the_refused_path_as_typed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLUMNS", "500")
    monkeypatch.setattr("soup_cli.commands.data_topics.console", Console(width=500))
    name = "n:warning:.jsonl"
    Path(name).write_text('{"text": "a"}\n', encoding="utf-8")

    result = runner.invoke(app, ["data", "topics", name, "--output", name])

    assert result.exit_code == 1
    assert _expected(name) in _plain(result.output)
