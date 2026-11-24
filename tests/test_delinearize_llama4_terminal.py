"""soup delinearize-llama4 shows model ids, directories and errors as typed."""

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from typer.testing import CliRunner

import soup_cli.commands.delinearize_llama4 as command
from soup_cli.cli import app
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
    monkeypatch.setattr(command, "console", Console(width=500, height=100))
    monkeypatch.setenv("COLUMNS", "500")
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    _make_source("ckpt")


def _make_source(name: str) -> None:
    directory = Path(name)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model-00001.safetensors").write_bytes(b"not a real checkpoint")


def _run(source: str = "ckpt", *args: str, target: str = "out"):
    return runner.invoke(app, ["delinearize-llama4", source, "--target", target, *args])


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


@pytest.mark.parametrize("value", _VALUES)
def test_a_model_id_that_is_not_llama4_is_shown_as_typed(value):
    result = _run("ckpt", "--model-id", value, "--plan-only")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert (
        f"model id {_shown(value)} doesn't match the Llama 4 naming pattern - "
        "proceed only if you're sure."
    ) in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_a_missing_source_directory_is_shown_as_typed(name):
    result = _run(name)

    assert result.exit_code == 2
    text = _plain(result)
    assert text == f"source_dir not found: {_shown(name)}"
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_a_source_directory_outside_cwd_is_shown_as_typed(name):
    result = _run(f"../{name}")

    assert result.exit_code == 2
    text = _plain(result)
    assert text == f"source_dir is outside cwd: {_shown(name)}"
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_a_target_directory_outside_cwd_is_shown_as_typed(name):
    result = _run("ckpt", target=f"../{name}")

    assert result.exit_code == 2
    text = _plain(result)
    assert text == f"target_dir is outside cwd: {_shown(name)}"
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_the_plan_names_the_source_and_target_directories_as_typed(name):
    _make_source(f"src-{name}")

    result = _run(f"src-{name}", "--plan-only", target=f"out-{name}")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    shown = re.escape(_shown(name))
    source = r"Weight files: 1 (?:\W )+Source dir: \S*src-" + shown
    assert re.search(source + r" (?:\W )+Target dir:", text)
    assert re.search(r"Target dir: \S*out-" + shown + r" \W", text)
    assert _not_shown(text)


@pytest.mark.parametrize("error", [TypeError, ValueError, FileNotFoundError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_failed_delinearization_is_shown_as_typed(monkeypatch, error, value):
    monkeypatch.setattr(command, "run_delinearize", _raising(error(f"run said {value}")))

    result = _run()

    assert result.exit_code == 2
    text = _plain(result)
    assert text.endswith(f"run said {_shown(value)}")
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_a_missing_dependency_is_shown_as_typed(monkeypatch, value):
    monkeypatch.setattr(command, "run_delinearize", _raising(ImportError(f"dependency {value}")))

    result = _run()

    assert result.exit_code == 1
    text = _plain(result)
    assert text.endswith(f"dependency {_shown(value)}")
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_the_written_target_is_shown_as_typed(monkeypatch, value):
    done = SimpleNamespace(
        files_written=["a", "b"], reshaped_keys=4, already_3d_keys=1, passthrough_keys=2,
        sidecars_copied=3, target_dir=f"written-{value}",
    )
    monkeypatch.setattr(command, "run_delinearize", lambda plan, num_experts=None: done)

    result = _run()

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Target dir: written-{_shown(value)} " in text
    assert "Files written: 2" in text
    assert _not_shown(text)


# --- styles and layout -----------------------------------------------------


def _coloured(monkeypatch):
    monkeypatch.setattr(
        command,
        "console",
        Console(
            force_terminal=True, no_color=False, width=500, height=100, color_system="standard"
        ),
    )


def test_the_model_id_warning_is_yellow(monkeypatch):
    _coloured(monkeypatch)

    result = _run("ckpt", "--model-id", "other", "--plan-only")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert (
        "\x1b[33mmodel id other doesn't match the Llama 4 naming pattern - "
        "proceed only if you're sure.\x1b[0m"
    ) in result.output


def test_an_error_is_red(monkeypatch):
    _coloured(monkeypatch)

    result = _run("missing")

    assert result.exit_code == 2
    assert "\x1b[31msource_dir not found: missing\x1b[0m" in result.output


def test_the_plan_keeps_one_detail_per_line():
    result = _run("ckpt", "--plan-only")

    lines = strip_ansi(result.output).splitlines()
    for label in ("Weight files:", "Source dir:", "Target dir:"):
        assert [line for line in lines if re.match(r"\W*" + label, line)], label


@pytest.mark.parametrize("error", [ValueError, ImportError])
def test_a_failed_run_is_red(monkeypatch, error):
    _coloured(monkeypatch)
    monkeypatch.setattr(command, "run_delinearize", _raising(error("run said no")))

    result = _run()

    assert result.exit_code == (1 if error is ImportError else 2)
    assert "\x1b[31mrun said no\x1b[0m" in result.output
