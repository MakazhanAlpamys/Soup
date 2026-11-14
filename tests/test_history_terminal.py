"""soup history prints a typed name and a branch file name as typed, not as Rich markup."""

import re
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.registry.store import RegistryStore
from tests._windows_filenames import skip_colon_on_windows
from tests.conftest import strip_ansi

runner = CliRunner()

_CONTROLS = "\x9b\u202e"


# One spelling per way the old code went wrong: a tag, a closing tag with
# nothing to close (a MarkupError), an emoji code, and a tag with control
# characters, so the strip and the escape are pinned separately.
_NAMES = [
    "x[red]n",
    "[/]n",
    pytest.param("n:warning:", id="emoji-code"),
    f"x[bold]n{_CONTROLS}",
]
# A file name cannot hold a slash, so no closing tag with nothing to close.
_FILES = [
    "x[red]n.json",
    pytest.param("n:warning:.json", marks=skip_colon_on_windows, id="emoji-code"),
    f"x[bold]n{_CONTROLS}.json",
]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setattr("soup_cli.commands.history.console", Console(width=500, height=100))
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


@pytest.mark.parametrize("value", _NAMES)
def test_a_refused_name_is_shown_as_typed(value):
    result = runner.invoke(app, ["history", value])

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Invalid name: registry name '{_shown(value)}' is invalid" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _FILES)
def test_a_branch_file_is_named_as_typed(name):
    Path(name).write_text("{}\n", encoding="utf-8")
    with RegistryStore() as store:
        entry_id = store.push(
            name="demo", tag="v1", base_model="org/base", task="sft", run_id=None, config={}
        )
        store.add_artifact(entry_id=entry_id, kind="branch_ref", path=name)

    result = runner.invoke(app, ["history", "demo"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(r"branch " + re.escape(_shown(name)), text)
    assert _not_shown(text)
