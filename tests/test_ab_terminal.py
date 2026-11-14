"""soup ab prints a typed metric and input path as typed, not as Rich markup."""


import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from tests._windows_filenames import skip_colon_on_windows
from tests.conftest import strip_ansi

runner = CliRunner()

_CONTROLS = "\x9b\u202e"


# One spelling per way the old code went wrong: a tag, an emoji code, and a tag
# with control characters, so the strip and the escape are pinned separately.
_NAMES = [
    "x[red]n.jsonl",
    pytest.param("n:warning:.jsonl", marks=skip_colon_on_windows, id="emoji-code"),
    f"x[bold]n{_CONTROLS}.jsonl",
]
# A metric is not a file name, so it can also carry a closing tag with nothing to close.
_METRICS = [
    "x[red]n",
    "[/]n",
    pytest.param("n:warning:", id="emoji-code"),
    f"x[bold]n{_CONTROLS}",
]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setattr("soup_cli.commands.ab.console", Console(width=500, height=100))
    monkeypatch.setenv("COLUMNS", "500")
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _shown(value: str) -> str:
    return value.replace(_CONTROLS, "")


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


def _ab(input_path: str, metric: str = "latency"):
    return runner.invoke(app, ["ab", "--input", input_path, "--metric", metric])


@pytest.mark.parametrize("metric", _METRICS)
def test_an_unknown_metric_is_shown_as_typed(metric):
    result = _ab("rows.jsonl", metric)

    assert result.exit_code == 2
    text = _plain(result)
    assert f"unknown metric {metric!r}; supported:" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_a_missing_input_is_named_as_typed(name):
    result = _ab(name)

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Input not found: {_shown(name)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_an_input_outside_cwd_is_shown_as_typed(name):
    typed = f"../{name}"

    result = _ab(typed)

    assert result.exit_code == 1
    text = _plain(result)
    assert f"input_path {typed!r} is outside cwd" in text
    assert _not_shown(text)
