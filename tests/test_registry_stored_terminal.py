"""soup registry prints stored text (a base model, notes, config values, paths) as typed."""

import re
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.experiment.tracker import ExperimentTracker
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
    "x[red]n.json",
    pytest.param("n:warning:.json", marks=skip_colon_on_windows, id="emoji-code"),
    f"x[bold]n{_CONTROLS}.json",
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
    return "".join(ch for ch in value if ch not in _CONTROLS)


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


def _push(name="demo", tag="v1", **fields) -> str:
    values = {"base_model": "org/base", "task": "sft", "run_id": None, "config": {}}
    values.update(fields)
    with RegistryStore() as store:
        return store.push(name=name, tag=tag, **values)


def _registry(*args: str):
    return runner.invoke(app, ["registry", *args])


def _cell(label: str, value: str) -> str:
    # A row of a table: the label, a border (a box character, or "|"), then the value.
    return re.escape(label) + r" \W+ " + re.escape(_shown(value))


@pytest.mark.parametrize("keyword", ["base_model", "task"])
@pytest.mark.parametrize("value", _VALUES)
def test_list_shows_the_base_model_and_the_task_as_typed(keyword, value):
    _push(**{keyword: value})

    result = _registry("list")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert _shown(value) in text
    assert _not_shown(text)


@pytest.mark.parametrize("keyword, label", [
    pytest.param("base_model", "base_model", id="base-model"),
    pytest.param("task", "task", id="task"),
    pytest.param("run_id", "run_id", id="run-id"),
    pytest.param("notes", "notes", id="notes"),
])
@pytest.mark.parametrize("value", _VALUES)
def test_show_prints_the_free_text_fields_as_typed(keyword, label, value):
    _push(**{keyword: value})

    result = _registry("show", "demo:v1")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(_cell(label, value), text)
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_show_names_an_artifact_path_as_typed(name):
    entry_id = _push()
    Path(name).write_text("{}\n", encoding="utf-8")
    with RegistryStore() as store:
        store.add_artifact(entry_id=entry_id, kind="config", path=name)

    result = _registry("show", "demo:v1")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(r"config \W+ \S*" + re.escape(_shown(name)), text)
    assert _not_shown(text)


@pytest.mark.parametrize("keyword", ["base_model", "task"])
@pytest.mark.parametrize("value", _VALUES)
def test_search_shows_the_base_model_and_the_task_as_typed(keyword, value):
    _push(**{keyword: value})

    result = _registry("search", "demo")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert _shown(value) in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_diff_shows_a_changed_config_value_as_typed(value):
    left = _push(tag="v1", config={"key": value})
    right = _push(tag="v2", config={"key": "plain"})

    result = _registry("diff", left, right)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(r"key \W+ changed \W+ " + re.escape(_shown(value)) + r" \W+ plain", text)
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_diff_shows_the_changed_config_value_of_the_right_entry_as_typed(value):
    left = _push(tag="v1", config={"key": "plain"})
    right = _push(tag="v2", config={"key": value})

    result = _registry("diff", left, right)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(r"key \W+ changed \W+ plain \W+ " + re.escape(_shown(value)), text)
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_diff_shows_a_config_key_as_typed(value):
    left = _push(tag="v1", config={value: 1})
    right = _push(tag="v2", config={value: 2})

    result = _registry("diff", left, right)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(re.escape(_shown(value)) + r" \W+ changed", text)
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_diff_names_a_benchmark_as_typed(value):
    left = _push(tag="v1", run_id="run_left")
    right = _push(tag="v2", run_id="run_right")
    tracker = ExperimentTracker()
    try:
        for run_id, score in (("run_left", 0.5), ("run_right", 0.75)):
            tracker.save_eval_result(
                model_path="model", benchmark=value, score=score, details={}, run_id=run_id
            )
    finally:
        tracker.close()

    result = _registry("diff", left, right)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(re.escape(_shown(value)) + r" \W+ 0\.500 \W+ 0\.750", text)
    assert _not_shown(text)
