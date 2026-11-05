"""soup apple-adapter prints typed paths, a refused direction and conversion errors as typed."""

import re
import sys
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.apple_adapter import ConversionReport
from soup_cli.utils.paths import quote_path
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
# A directory name cannot hold a slash, so no closing tag with nothing to close.
_NAMES = [
    "x[red]n",
    pytest.param("n:warning:", marks=_NO_COLON_ON_WINDOWS, id="emoji-code"),
    f"x[bold]n{_CONTROLS}",
]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setattr("soup_cli.commands.apple_adapter.console", Console(width=500, height=100))
    monkeypatch.setenv("COLUMNS", "500")
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    (workdir / "src").mkdir()


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _shown(value: str) -> str:
    return "".join(ch for ch in value if ch not in _CONTROLS)


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


def _convert(*args: str, source: str = "src", output: str = "out"):
    return runner.invoke(app, ["apple-adapter", source, "-o", output, *args])


def _outside(name: str) -> str:
    # quote_path shows C0/C1 controls as \xNN text but leaves the bidi override.
    return quote_path(f"../{name}").replace("\u202e", "")


@pytest.mark.parametrize("value", _VALUES)
def test_a_refused_direction_is_shown_as_typed(value):
    result = _convert("--direction", value)

    assert result.exit_code == 2
    text = _plain(result)
    assert f"unknown direction {value!r}; supported:" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_a_refused_plan_is_shown_as_typed(monkeypatch, value):
    def refuse(**kwargs):
        raise ValueError(f"plan said {value}")

    monkeypatch.setattr("soup_cli.commands.apple_adapter.build_apple_adapter_plan", refuse)

    result = _convert()

    assert result.exit_code == 2
    text = _plain(result)
    assert f"plan said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_the_plan_names_the_source_and_the_output_as_typed(name):
    Path(name).mkdir()

    result = _convert("--plan-only", source=name, output=f"{name}-out")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    # Two lines of a panel are separated by its border: a box character, or "|".
    assert re.search(
        r"Source: " + re.escape(_shown(name)) + r" (?:\W )+Output: " + re.escape(_shown(name))
        + r"-out",
        text,
    )
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_an_output_outside_cwd_is_shown_as_typed(name):
    result = _convert(output=f"../{name}")

    assert result.exit_code == 2
    text = _plain(result)
    assert f"output_dir {_outside(name)} must stay under cwd" in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [ValueError, FileNotFoundError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_conversion_error_is_shown_as_typed(monkeypatch, error, value):
    def refuse(plan):
        raise error(f"convert said {value}")

    monkeypatch.setattr("soup_cli.commands.apple_adapter.convert_apple_adapter", refuse)

    result = _convert()

    assert result.exit_code == 2
    text = _plain(result)
    assert f"convert said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_the_converted_panel_names_the_output_as_typed(monkeypatch, name):
    report = ConversionReport(
        direction="hf-to-mlx", output_dir=name, converted_keys=2, skipped_keys=(), signed=False
    )
    monkeypatch.setattr(
        "soup_cli.commands.apple_adapter.convert_apple_adapter", lambda plan: report
    )

    result = _convert(output=name)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(r"Converted LoRA keys: 2 (?:\W )+Output: " + re.escape(_shown(name)), text)
    assert _not_shown(text)
