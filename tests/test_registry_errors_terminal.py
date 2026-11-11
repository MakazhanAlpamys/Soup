"""soup registry prints typed refs, names, tags, paths and queries as typed, not as Rich markup."""

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
_VALUES = [
    "x[red]n",
    "[/]n",
    pytest.param("n:warning:", id="emoji-code"),
    f"x[bold]n{_CONTROLS}",
]
# A file name cannot hold a slash, so no closing tag with nothing to close.
_NAMES = [
    "x[red]n.jsonl",
    pytest.param("n:warning:.jsonl", marks=skip_colon_on_windows, id="emoji-code"),
    f"x[bold]n{_CONTROLS}.jsonl",
]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setattr("soup_cli.commands.registry.console", Console(width=500, height=100))
    monkeypatch.setenv("COLUMNS", "500")
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "experiments.db"))
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _shown(value: str) -> str:
    return value.replace(_CONTROLS, "")


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


def _registry(*args: str):
    return runner.invoke(app, ["registry", *args])


def _push(*args: str, run_id: str = "run_x", name: str = "demo", tag: str = "v1"):
    return _registry("push", "--run-id", run_id, "--name", name, "--tag", tag, *args)


@pytest.mark.parametrize("ref", _VALUES)
def test_show_names_a_missing_entry_as_typed(ref):
    result = _registry("show", ref)

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Registry entry not found: {_shown(ref)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_push_shows_a_refused_name_as_typed(value):
    result = _push(name=value)

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Invalid input: registry name '{_shown(value)}' is invalid" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_push_shows_a_refused_tag_as_typed(value):
    result = _push(tag=value)

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Invalid input: registry tag '{_shown(value)}' is invalid" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_push_names_a_missing_run_as_typed(value):
    result = _push(run_id=value)

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Run not found: {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_push_names_a_missing_data_file_as_typed(name):
    result = _push("--data", name)

    assert result.exit_code == 1
    text = _plain(result)
    assert f"--data file not found: {_shown(name)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_push_names_a_data_directory_as_typed(name):
    Path(name).mkdir()

    result = _push("--data", name)

    assert result.exit_code == 1
    text = _plain(result)
    assert f"--data is not a file: {_shown(name)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_push_names_a_data_file_outside_cwd_as_typed(name):
    Path("..", name).write_text("{}\n", encoding="utf-8")

    result = _push("--data", f"../{name}")

    assert result.exit_code == 1
    text = _plain(result)
    assert (
        f"--data must be under the current working directory: ../{_shown(name)}"
    ) in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_promote_shows_a_refused_tag_as_typed(value):
    result = _registry("promote", "demo:v1", "--tag", value)

    assert result.exit_code == 1
    text = _plain(result)
    assert f"registry tag '{_shown(value)}' is invalid" in text
    assert _not_shown(text)


@pytest.mark.parametrize("query", _VALUES)
def test_search_names_a_query_nothing_matches_as_typed(query):
    result = _registry("search", query)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"No entries match '{_shown(query)}'." in text
    assert _not_shown(text)


@pytest.mark.parametrize("query", _VALUES)
def test_search_names_the_query_in_its_title_as_typed(query):
    with RegistryStore() as store:
        store.push(
            name="demo", tag="v1", base_model="org/base", task="sft",
            run_id=None, config={}, notes=f"about {query}",
        )

    result = _registry("search", query)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(r"Search: '" + re.escape(_shown(query)) + r"'", text)
    assert "demo" in text
    assert _not_shown(text)


def test_search_keeps_the_italic_of_its_title(monkeypatch):
    monkeypatch.setattr(
        "soup_cli.commands.registry.console",
        Console(width=500, force_terminal=True, color_system="standard", legacy_windows=False),
    )
    with RegistryStore() as store:
        store.push(
            name="demo", tag="v1", base_model="org/base", task="sft",
            run_id=None, config={}, notes="about plain",
        )

    result = _registry("search", "plain")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    # Italic is SGR 3.
    assert re.search(r"\x1b\[3m\s*Search: 'plain'", result.output)
