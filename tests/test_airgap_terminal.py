"""soup airgap-bundle prints typed paths and build errors as typed, not as Rich markup."""

import re
import sys

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.paths import quote_path
from tests.conftest import strip_ansi

runner = CliRunner()

_CONTROLS = "\x9b\u202e"

# A colon is not a legal file-name character on Windows, and "n:" there is a drive.
_NO_COLON_ON_WINDOWS = pytest.mark.skipif(
    sys.platform == "win32", reason="':' is not valid in a Windows name"
)

# One spelling per way the old code went wrong: a tag, an emoji code, and a tag
# with control characters, so the strip and the escape are pinned separately.
_NAMES = [
    "x[red]n",
    pytest.param("n:warning:", marks=_NO_COLON_ON_WINDOWS, id="emoji-code"),
    f"x[bold]n{_CONTROLS}",
]
# Text that is not a path can also carry a closing tag with nothing to close.
_VALUES = [
    "x[red]n",
    "[/]n",
    pytest.param("n:warning:", id="emoji-code"),
    f"x[bold]n{_CONTROLS}",
]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setattr("soup_cli.commands.airgap.console", Console(width=500, height=100))
    monkeypatch.setenv("COLUMNS", "500")
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    (workdir / "model").mkdir()
    (workdir / "model" / "weights.bin").write_bytes(b"x")


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _shown(value: str) -> str:
    return value.replace(_CONTROLS, "")


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


def _outside(name: str) -> str:
    # quote_path shows C0/C1 controls as \xNN text but leaves the bidi override.
    return quote_path(f"../{name}").replace("\u202e", "")


def _bundle(*args: str, output: str = "out.tar", model: str = "model"):
    return runner.invoke(app, ["airgap-bundle", "-o", output, "--model", model, *args])


@pytest.mark.parametrize("name", _NAMES)
def test_the_output_is_named_as_typed(name):
    result = _bundle(output=f"{name}.tar")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(r"Output: " + re.escape(_shown(name)) + r"\.tar", text)
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_the_model_directory_is_named_as_typed(name):
    result = _bundle(model=name)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(r"Model: " + re.escape(_shown(name)) + r" ", text)
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_a_missing_receipt_is_named_as_typed(name):
    result = _bundle("--repro-receipt", f"{name}.json")

    assert result.exit_code == 2
    text = _plain(result)
    # The rest of the line is the operating system's own text.
    assert f"--repro-receipt {_shown(name)}.json: " in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_a_receipt_outside_cwd_is_shown_as_typed(name):
    result = _bundle("--repro-receipt", f"../{name}.json")

    assert result.exit_code == 2
    text = _plain(result)
    assert (
        f"--repro-receipt ../{_shown(name)}.json: repro-receipt {_outside(name + '.json')} "
        "must stay under cwd"
    ) in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_an_output_outside_cwd_is_shown_as_typed(name):
    result = _bundle(output=f"../{name}.tar")

    assert result.exit_code == 2
    text = _plain(result)
    assert f"output {_outside(name + '.tar')} must stay under cwd" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_a_missing_input_error_is_shown_as_typed(monkeypatch, value):
    def refuse(plan):
        raise FileNotFoundError(f"bundle said {value}")

    monkeypatch.setattr("soup_cli.utils.airgap_bundle.build_airgap_bundle", refuse)

    result = _bundle()

    assert result.exit_code == 1
    text = _plain(result)
    assert f"bundle said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_a_refused_build_is_shown_as_typed(monkeypatch, value):
    def refuse(plan):
        raise ValueError(f"bundle said {value}")

    monkeypatch.setattr("soup_cli.utils.airgap_bundle.build_airgap_bundle", refuse)

    result = _bundle()

    assert result.exit_code == 2
    text = _plain(result)
    assert f"bundle said {_shown(value)}" in text
    assert _not_shown(text)
