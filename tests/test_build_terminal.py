"""soup build shows manifest text, paths and errors as typed."""

import json
import re
import sys
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

import soup_cli.utils.build_dag as build_dag
from soup_cli.cli import app
from soup_cli.utils.paths import quote_path
from tests._windows_filenames import skip_colon_on_windows
from tests.conftest import strip_ansi

runner = CliRunner()

_CONTROLS = "\x9b\u202e"

# A backslash separates directories on Windows, so a name that holds one cannot be a file there.
_NO_BACKSLASH_ON_WINDOWS = pytest.mark.skipif(
    sys.platform == "win32", reason="a backslash separates directories on Windows"
)

# One spelling per way the old code went wrong: a tag, a closing tag with
# nothing to close (a MarkupError), an emoji code, and a tag with control
# characters, so the strip and the escape are pinned separately. A backslash before
# an unmatched bracket and a trailing pair of backslashes are what markup rewrites.
_VALUES = [
    "x[red]n",
    "[/]n",
    pytest.param("n:warning:", id="emoji-code"),
    pytest.param("x\\[n", id="backslash-bracket"),
    pytest.param("n\\\\", id="trailing-backslashes"),
    f"x[bold]n{_CONTROLS}",
]
# A file name cannot hold a slash, so no closing tag with nothing to close.
_NAMES = [
    "x[red]n",
    pytest.param("n:warning:", marks=skip_colon_on_windows, id="emoji-code"),
    pytest.param("x\\[n", marks=_NO_BACKSLASH_ON_WINDOWS, id="backslash-bracket"),
    pytest.param("n\\\\", marks=_NO_BACKSLASH_ON_WINDOWS, id="trailing-backslashes"),
    f"x[bold]n{_CONTROLS}",
]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setattr("soup_cli.commands.build.console", Console(width=500, height=100))
    monkeypatch.setenv("COLUMNS", "500")
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    (workdir / "data.jsonl").write_text('{"text": "A"}\n{"text": "b"}\n', encoding="utf-8")
    _write_plan("plan.yaml")


def _write_plan(path: str, transform: str = "identity", source: str = "data.jsonl") -> None:
    # A JSON string is a valid YAML double-quoted scalar and holds only ASCII.
    Path(path).write_text(
        "models:\n"
        "  - name: raw\n"
        "    kind: table\n"
        f"    transform: {json.dumps(transform)}\n"
        f"    source: {json.dumps(source)}\n",
        encoding="utf-8",
    )


def _build(*args: str, config: str = "plan.yaml"):
    return runner.invoke(app, ["build", config, *args])


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _shown(value: str) -> str:
    return "".join(ch for ch in value if ch not in _CONTROLS)


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


def _raising(error):
    def fail(*args, **kwargs):
        raise error

    return fail


@pytest.mark.parametrize("name", _NAMES)
def test_a_missing_manifest_is_shown_as_typed(name):
    result = _build(config=f"{name}.yaml")

    assert result.exit_code == 2
    text = _plain(result)
    assert text == f"{_shown(name)}.yaml"
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_a_manifest_outside_cwd_is_shown_as_typed(name):
    result = _build(config=f"../{name}.yaml")

    assert result.exit_code == 2
    text = _plain(result)
    quoted = quote_path(f"../{name}.yaml").replace("\u202e", "")
    assert text == f"build path {quoted} must stay under cwd"
    assert _not_shown(text)


@pytest.mark.parametrize(
    "value",
    ["x[red]n", "[/]n", "n:warning:", "x\\[n", "n\\\\", "x[bold]n\u202e"],
    ids=["tag", "closing-tag", "emoji-code", "backslash-bracket", "trailing-backslashes", "bidi"],
)
def test_a_yaml_error_quoting_the_manifest_is_shown_as_typed(value):
    # An unterminated quote makes the YAML error quote the line that holds the value.
    Path("plan.yaml").write_text(f'models: "{value}\n', encoding="utf-8")

    result = _build()

    assert result.exit_code == 2
    text = _plain(result)
    assert "invalid YAML:" in text
    assert f'models: "{_shown(value)}' in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [FileNotFoundError, TypeError, ValueError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_manifest_failure_with_any_text_is_shown_as_typed(monkeypatch, error, value):
    monkeypatch.setattr(build_dag, "load_build_yaml", _raising(error(f"load said {value}")))

    result = _build()

    assert result.exit_code == 2
    text = _plain(result)
    assert text == f"load said {_shown(value)}"
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_the_plan_panel_shows_a_transform_as_typed(value):
    _write_plan("plan.yaml", transform=value)

    result = _build("--dry-run")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"raw [table] transform={_shown(value)} refs=[(no refs)]" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_the_plan_panel_title_shows_the_manifest_path_as_typed(name):
    _write_plan(f"{name}.yaml")

    result = _build("--dry-run", config=f"{name}.yaml")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"soup build \u2014 {_shown(name)}.yaml" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_a_refused_transform_is_shown_as_typed(value):
    _write_plan("plan.yaml", transform=value)

    result = _build()

    assert result.exit_code == 2
    text = _plain(result)
    assert repr(value) in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_a_missing_source_is_shown_as_typed(name):
    _write_plan("plan.yaml", source=f"{name}.jsonl")

    result = _build()

    assert result.exit_code == 2
    text = _plain(result)
    assert f"build source not found: {f'{name}.jsonl'!r}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [FileNotFoundError, TypeError, ValueError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_build_failure_with_any_text_is_shown_as_typed(monkeypatch, error, value):
    monkeypatch.setattr(build_dag, "run_build", _raising(error(f"run said {value}")))

    result = _build()

    assert result.exit_code == 2
    text = _plain(result)
    assert text.endswith(f"run said {_shown(value)}")
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_the_materialised_title_shows_the_output_directory_as_typed(name):
    result = _build("--output-dir", f"out-{name}")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    # A table title wraps at the width of the table, so compare without the spaces.
    assert "Materialised\u2014" in "".join(text.split())
    assert f"/out-{_shown(name)}" in "".join(text.split())
    assert _not_shown(text)


# --- styles ----------------------------------------------------------------


def _coloured(monkeypatch):
    monkeypatch.setattr(
        "soup_cli.commands.build.console",
        Console(
            force_terminal=True, no_color=False, width=500, height=100, color_system="standard"
        ),
    )


def test_an_error_is_red(monkeypatch):
    _coloured(monkeypatch)

    result = _build(config="missing.yaml")

    assert result.exit_code == 2
    assert "\x1b[31mmissing.yaml\x1b[0m" in result.output


def test_a_build_error_is_red(monkeypatch):
    _coloured(monkeypatch)
    _write_plan("plan.yaml", source="missing.jsonl")

    result = _build()

    assert result.exit_code == 2
    assert "\x1b[31mbuild source not found: 'missing.jsonl'\x1b[0m" in result.output


def test_the_materialised_title_keeps_its_italic(monkeypatch):
    _coloured(monkeypatch)

    result = _build()

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert re.search(r"\x1b\[3m\s*Materialised \u2014", result.output)


def test_every_line_of_the_plan_has_its_own_row():
    _write_plan("plan.yaml")
    result = _build("--dry-run")

    lines = strip_ansi(result.output).splitlines()
    assert [line for line in lines if re.match(r"\W*Build plan \(topological order\):", line)]
    assert [line for line in lines if re.match(r"\W*raw \[table\] transform=identity", line)]
