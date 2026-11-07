"""soup expect prints typed paths, judge URLs and loader errors as typed, not as Rich markup."""

import re
import sys
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.eval.judge import JudgeUnavailableError
from soup_cli.utils.exit_codes import EXIT_RUNTIME_ERROR, EXIT_USAGE_ERROR
from soup_cli.utils.paths import quote_path
from tests.conftest import strip_ansi

runner = CliRunner()

_CONTROLS = "\x9b\u202e"
_DASH = chr(0x2014)
_SUITE = "expectations:\n  - name: expect_no_pii\n"
_ROW = '{"text": "hello world"}\n'

# A colon is not a legal file-name character on Windows, and "n:" there is a drive.
_NO_COLON_ON_WINDOWS = pytest.mark.skipif(
    sys.platform == "win32", reason="':' is not valid in a Windows name"
)

# One spelling per way the old code went wrong: a tag, an emoji code, and a tag
# with control characters, so the strip and the escape are pinned separately.
_NAMES = [
    "x[red]n.jsonl",
    pytest.param("n:warning:.jsonl", marks=_NO_COLON_ON_WINDOWS, id="emoji-code"),
    f"x[bold]n{_CONTROLS}.jsonl",
]
# Values that are not file names can also carry a closing tag with nothing to close.
_VALUES = [
    "x[red]n",
    "[/]n",
    pytest.param("n:warning:", id="emoji-code"),
    f"x[bold]n{_CONTROLS}",
]
# PyYAML refuses C1 controls in a document but accepts U+202E, so a YAML error
# can quote a line that carries it.
_YAML_VALUES = [
    "x[red]n",
    "[/]n",
    pytest.param("n:warning:", id="emoji-code"),
    "x[bold]n\u202e",
]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setattr("soup_cli.commands.expect.console", Console(width=500, height=100))
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


def _outside(name: str) -> str:
    # quote_path shows C0/C1 controls as \xNN text but leaves the bidi override.
    return quote_path(f"../{name}").replace("\u202e", "")


def _suite(name: str = "suite.yaml", text: str = _SUITE) -> str:
    Path(name).write_text(text, encoding="utf-8")
    return name


def _data(name: str = "data.jsonl", text: str = _ROW) -> str:
    Path(name).write_text(text, encoding="utf-8")
    return name


def _expect(data: str, suite: str, *args: str):
    return runner.invoke(app, ["expect", data, suite, *args])


@pytest.mark.parametrize("name", _NAMES)
def test_the_table_title_names_the_data_path_as_typed(name):
    result = _expect(_data(name), _suite())

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"soup expect {_DASH} {_shown(name)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_a_refused_data_file_is_named_as_typed(name):
    result = _expect(_data(name, "not json\n"), _suite())

    assert result.exit_code == EXIT_USAGE_ERROR
    text = _plain(result)
    assert (
        f"refusing {_shown(name)}: 1 line(s) could not be checked: line 1: malformed JSON"
    ) in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_an_empty_data_file_is_named_as_typed(name):
    result = _expect(_data(name, "\n"), _suite())

    assert result.exit_code == EXIT_USAGE_ERROR
    text = _plain(result)
    assert f"refusing {_shown(name)}: no checkable rows were found" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_a_missing_data_file_is_named_as_typed(name):
    result = _expect(name, _suite())

    assert result.exit_code == EXIT_USAGE_ERROR
    text = _plain(result)
    assert f"data file not found: {name!r}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_a_data_file_outside_cwd_is_named_as_typed(name):
    Path("..", name).write_text(_ROW, encoding="utf-8")

    result = _expect(f"../{name}", _suite())

    assert result.exit_code == EXIT_USAGE_ERROR
    text = _plain(result)
    assert f"data path {_outside(name)} must stay under cwd" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_a_missing_suite_file_is_named_as_typed(name):
    result = _expect(_data(), name)

    assert result.exit_code == EXIT_USAGE_ERROR
    text = _plain(result)
    assert f"suite file not found: {name!r}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_a_suite_file_outside_cwd_is_named_as_typed(name):
    Path("..", name).write_text(_SUITE, encoding="utf-8")

    result = _expect(_data(), f"../{name}")

    assert result.exit_code == EXIT_USAGE_ERROR
    text = _plain(result)
    assert f"suite path {_outside(name)} must stay under cwd" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _YAML_VALUES)
def test_a_yaml_error_shows_the_quoted_line_as_typed(value):
    suite = _suite(text=f'expectations: "{value}\n')

    result = _expect(_data(), suite)

    assert result.exit_code == EXIT_USAGE_ERROR
    text = _plain(result)
    assert "invalid YAML: while scanning a quoted scalar" in text
    assert f'expectations: "{_shown(value)}' in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_a_refused_judge_url_is_shown_as_typed(value):
    result = _expect(_data(), _suite(), "--judge", value)

    assert result.exit_code == EXIT_USAGE_ERROR
    text = _plain(result)
    assert f"judge_model '{_shown(value)}' uses unsupported scheme" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_a_refused_suite_run_is_shown_as_typed(monkeypatch, value):
    def refuse(rows, spec, judge_fn=None):
        raise ValueError(f"suite said {value}")

    monkeypatch.setattr("soup_cli.utils.expectations.run_suite", refuse)

    result = _expect(_data(), _suite())

    assert result.exit_code == EXIT_USAGE_ERROR
    text = _plain(result)
    assert f"suite said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_an_unreachable_judge_is_shown_as_typed(monkeypatch, value):
    def unreachable(rows, spec, judge_fn=None):
        raise JudgeUnavailableError(f"judge said {value}", url="http://judge.example")

    monkeypatch.setattr("soup_cli.utils.expectations.run_suite", unreachable)

    result = _expect(_data(), _suite())

    assert result.exit_code == EXIT_RUNTIME_ERROR
    text = _plain(result)
    assert f"judge unavailable at http://judge.example: judge said {_shown(value)}" in text
    assert _not_shown(text)


def test_the_table_title_keeps_its_italic_style(monkeypatch):
    monkeypatch.setattr(
        "soup_cli.commands.expect.console",
        Console(width=500, force_terminal=True, color_system="standard", legacy_windows=False),
    )

    result = _expect(_data(), _suite())

    assert result.exit_code == 0, (result.output, repr(result.exception))
    # Italic is SGR 3.
    assert re.search(r"\x1b\[3m\s*soup expect", result.output)
