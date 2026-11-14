"""soup data mix prints dataset paths, recipes and errors as typed, not as Rich markup."""

import json
import re
from pathlib import Path

import pytest
import yaml
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from tests._windows_filenames import skip_colon_on_windows
from tests.conftest import strip_ansi

runner = CliRunner()

_CONTROLS = "\x9b\u202e"


# One spelling per way the old code went wrong: a tag, an emoji code, and a tag
# with control characters, so the strip and the escape are pinned separately.
_FILE_NAMES = [
    "x[red]n.jsonl",
    pytest.param("n:warning:.jsonl", marks=skip_colon_on_windows, id="emoji-code"),
    f"x[bold]n{_CONTROLS}.jsonl",
]
# Values that are not file names can also carry a closing tag with nothing to close.
_VALUES = [
    "x[red]n",
    "[/]n",
    pytest.param("n:warning:", id="emoji-code"),
    f"x[bold]n{_CONTROLS}",
]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setattr("soup_cli.commands.data_mix.console", Console(width=500, height=100))
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


def _dataset(name: str) -> str:
    Path(name).write_text('{"text": "a"}\n', encoding="utf-8")
    return name


def _recipe(name: str, train) -> str:
    block = {"interleave": {"strategy": "probs", "probs": [0.5, 0.5]}, "train": train}
    Path(name).write_text(json.dumps({"data": block}), encoding="utf-8")
    return name


def _mix(*args: str):
    return runner.invoke(app, ["data", "mix", *args])


def _as_yaml_name(name: str) -> str:
    return name.replace(".jsonl", ".yaml")


@pytest.mark.parametrize("name", _FILE_NAMES)
def test_optimize_echo_names_the_dataset_as_typed(name):
    _dataset(name)
    _dataset("b.jsonl")

    result = _mix("--optimize", "--datasets", f"{name},b.jsonl", "--output", "out.yaml")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(r"# \d\.\d{6} \S*" + re.escape(_shown(name)), text)
    assert json.dumps(name)[1:-1] + '"' in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _FILE_NAMES)
def test_optimize_names_the_written_recipe_as_typed(name):
    _dataset("a.jsonl")
    _dataset("b.jsonl")
    out = _as_yaml_name(name)

    result = _mix("--optimize", "--datasets", "a.jsonl,b.jsonl", "--output", out)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(r"wrote recipe: \S*" + re.escape(_shown(out)) + r" \(best_loss=", text)
    assert _not_shown(text)


@pytest.mark.parametrize("name", _FILE_NAMES)
def test_apply_names_the_recipe_as_typed(name):
    recipe = _recipe(_as_yaml_name(name), ["a.jsonl", "b.jsonl"])

    result = _mix("--apply", recipe)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Loaded recipe: {_shown(recipe)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_apply_lists_the_dataset_as_typed(value):
    recipe = _recipe("r.yaml", ["a.jsonl", value])

    result = _mix("--apply", recipe)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    output = strip_ansi(result.output)
    assert yaml.safe_load(output[output.index("data:\n"):])["data"]["train"] == ["a.jsonl", value]
    assert _not_shown(result.output)


@pytest.mark.parametrize("value", _VALUES)
def test_apply_shows_a_single_train_string_as_typed(value):
    recipe = _recipe("r.yaml", value)

    result = _mix("--apply", recipe)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    output = strip_ansi(result.output)
    assert yaml.safe_load(output[output.index("data:\n"):])["data"]["train"] == value
    assert _not_shown(result.output)


@pytest.mark.parametrize("name", _FILE_NAMES)
def test_apply_failure_names_the_recipe_as_typed(name):
    result = _mix("--apply", _as_yaml_name(name))

    assert result.exit_code == 2
    text = _plain(result)
    assert f"apply failed: recipe not found: {_as_yaml_name(name)!r}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _FILE_NAMES)
def test_plan_failure_names_the_dataset_as_typed(name):
    _dataset(name)

    result = _mix("--optimize", "--datasets", f"{name},{name}", "--output", "out.yaml")

    assert result.exit_code == 2
    text = _plain(result)
    assert f"plan validation failed: duplicate dataset path: {name!r}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _FILE_NAMES)
def test_write_failure_names_the_output_as_typed(name):
    _dataset("a.jsonl")
    _dataset("b.jsonl")
    out = _as_yaml_name(name)
    Path(out).write_text("old\n", encoding="utf-8")

    result = _mix("--optimize", "--datasets", "a.jsonl,b.jsonl", "--output", out)

    assert result.exit_code == 2
    text = _plain(result)
    assert f"write failed: output_path already exists (use overwrite=True): {out!r}" in text
    assert _not_shown(text)


def test_apply_failure_shows_the_quoted_recipe_line_as_typed():
    # A value YAML cannot build is reported with the line it sits on. PyYAML
    # refuses C0 and C1 controls in a document but accepts U+202E, so that
    # line can carry one that repr() never sees.
    bidi = _CONTROLS[-1]
    line = f'data: {{when: 2026-02-30, note: "x[bold]n{bidi}"}}'
    Path("r.yaml").write_text(line + "\n", encoding="utf-8")

    result = _mix("--apply", "r.yaml")

    assert result.exit_code == 2
    text = _plain(result)
    assert "apply failed: cannot build this value: day is out of range for month" in text
    assert line.replace(bidi, "") in text
    assert _not_shown(text)
