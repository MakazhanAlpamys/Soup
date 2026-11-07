"""soup card prints a typed reference and output path as typed, not as Rich markup."""

import re
import sys

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.registry.store import RegistryStore
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
    "x[red]n.md",
    pytest.param("n:warning:.md", marks=_NO_COLON_ON_WINDOWS, id="emoji-code"),
    f"x[bold]n{_CONTROLS}.md",
]
# A reference is not a file name, so it can also carry a closing tag with nothing to close.
_REFS = [
    "x[red]n",
    "[/]n",
    pytest.param("n:warning:", id="emoji-code"),
    f"x[bold]n{_CONTROLS}",
]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setattr("soup_cli.commands.card.console", Console(width=500, height=100))
    monkeypatch.setenv("COLUMNS", "500")
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _shown(value: str) -> str:
    return value.replace(_CONTROLS, "")


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


def _entry() -> str:
    with RegistryStore() as store:
        store.push(
            name="demo", tag="v1", base_model="org/base", task="sft", run_id=None, config={}
        )
    return "demo:v1"


def _card(*args: str):
    return runner.invoke(app, ["card", *args])


@pytest.mark.parametrize("ref", _REFS)
def test_a_missing_reference_is_named_as_typed(ref):
    result = _card(ref)

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Registry entry not found: {_shown(ref)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_the_written_card_is_named_as_typed(name):
    result = _card(_entry(), "-o", name)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    # The panel names the resolved path: the working directory, then the name as typed.
    assert re.search(r"Model card written to \S*" + re.escape(_shown(name)), text)
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_an_output_outside_cwd_is_shown_as_typed(name):
    result = _card(_entry(), "-o", f"../{name}")

    assert result.exit_code == 1
    text = _plain(result)
    # quote_path shows C0/C1 controls as \xNN text but leaves the bidi override.
    quoted = quote_path(f"../{name}").replace("\u202e", "")
    assert f"Cannot write card: card output {quoted} must stay under cwd" in text
    assert _not_shown(text)
