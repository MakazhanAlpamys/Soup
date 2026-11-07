"""soup audit-log prints the text of the log file and a typed path as typed, not as markup."""

import json
import re
import sys
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi

runner = CliRunner()

_CONTROLS = "\x9b\u202e"

# A colon is not a legal file-name character on Windows, and "n:" there is a drive.
_NO_COLON_ON_WINDOWS = pytest.mark.skipif(
    sys.platform == "win32", reason="':' is not valid in a Windows name"
)

# One spelling per way the old code went wrong: a tag, a closing tag with
# nothing to close (a MarkupError), an emoji code, and a tag with control
# characters, so the strip and the escape are pinned separately.
_VALUES = [
    "x[red]n",
    "[/]n",
    pytest.param("n:warning:", id="emoji-code"),
    f"x[bold]n{_CONTROLS}",
]
# A file name cannot hold a slash, so no closing tag with nothing to close.
_NAMES = [
    "x[red]n.jsonl",
    pytest.param("n:warning:.jsonl", marks=_NO_COLON_ON_WINDOWS, id="emoji-code"),
    f"x[bold]n{_CONTROLS}.jsonl",
]
_FIELDS = ["timestamp", "command", "exit_code", "operator_id", "host_id"]
_PLAIN = {
    "timestamp": "t0",
    "command": "cmd",
    "exit_code": "0",
    "operator_id": "op",
    "host_id": "host",
}


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setattr("soup_cli.commands.audit_log.console", Console(width=500, height=100))
    monkeypatch.setenv("COLUMNS", "500")
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _shown(value: str) -> str:
    return "".join(ch for ch in value if ch not in _CONTROLS)


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


@pytest.mark.parametrize("field", _FIELDS)
@pytest.mark.parametrize("value", _VALUES)
def test_tail_shows_a_field_of_the_log_as_typed(field, value):
    record = dict(_PLAIN, **{field: value})
    Path("audit.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    result = runner.invoke(app, ["audit-log", "tail", "--path", "audit.jsonl"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    cells = [_shown(str(record[name])) for name in _FIELDS]
    # Cells are separated by a border: a box character, or "|" on a terminal without them.
    assert re.search(r" \W+ ".join(re.escape(cell) for cell in cells), text)
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_rotate_names_the_rotated_log_as_typed(name):
    Path(name).write_bytes(b"x" * (1024 * 1024 + 1))

    result = runner.invoke(app, ["audit-log", "rotate", "--path", name, "--cap-mb", "1"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Rotated {_shown(name)} -> {_shown(name)}.1" in text
    assert _not_shown(text)
