"""soup iterative-dpo shows plan paths, adapter paths and errors as typed."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from typer.testing import CliRunner

import soup_cli.utils.iterative_dpo as iterative_dpo_utils
from soup_cli.commands import iterative_dpo
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


# The plan turns every backslash of the output directory into a slash, so the spellings that
# hold one are not shown "as typed" there; the other names are.
_DIRECTORY_NAMES = [_NAMES[0], _NAMES[1], _NAMES[4]]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setattr(iterative_dpo, "console", Console(width=500, height=100))
    monkeypatch.setenv("COLUMNS", "500")
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    (workdir / "prompts.jsonl").write_text('{"prompt": "hello"}\n', encoding="utf-8")


def _run(*extra: str, prompts: str = "prompts.jsonl", output: str = "out"):
    return runner.invoke(
        iterative_dpo.app,
        [
            "--base-model", "base", "--reward-model", "rm", "--prompts", prompts,
            "--output-dir", output, "--rounds", "2", *extra,
        ],
    )


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


@pytest.mark.parametrize("error", [ValueError, TypeError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_refused_plan_is_shown_as_typed(monkeypatch, error, value):
    monkeypatch.setattr(
        iterative_dpo_utils, "build_iterative_dpo_plan", _raising(error(f"plan said {value}"))
    )

    result = _run("--plan-only")

    assert result.exit_code == 2
    text = _plain(result)
    assert f"Error: plan said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _DIRECTORY_NAMES)
def test_the_plan_table_shows_the_pairs_and_adapter_paths_as_typed(name):
    result = _run("--plan-only", output=f"out-{name}")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    # Two rounds, each with a pairs path and an adapter path under the typed output directory.
    assert text.count(f"out-{_shown(name)}/round-") == 4
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_a_prompts_file_without_prompts_is_shown_as_typed(name):
    Path(f"{name}.jsonl").write_text("", encoding="utf-8")

    result = _run(prompts=f"{name}.jsonl")

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Iterative DPO failed: prompts file {f'{name}.jsonl'!r} has no usable prompts" in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [RuntimeError, ValueError, OSError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_failed_prompt_check_is_shown_as_typed(monkeypatch, error, value):
    check = _raising(error(f"check said {value}"))
    monkeypatch.setattr(iterative_dpo_utils, "validate_iterative_dpo_prompts", check)

    result = _run()

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Iterative DPO failed: check said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [RuntimeError, ValueError, OSError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_failed_run_is_shown_as_typed(monkeypatch, error, value):
    monkeypatch.setattr(iterative_dpo_utils, "validate_iterative_dpo_prompts", lambda plan: None)
    failing = _raising(error(f"run said {value}"))
    monkeypatch.setattr(iterative_dpo_utils, "run_iterative_dpo", failing)

    result = _run()

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Iterative DPO failed: run said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_the_final_adapter_is_shown_as_typed(monkeypatch, value):
    monkeypatch.setattr(iterative_dpo_utils, "validate_iterative_dpo_prompts", lambda plan: None)
    done = SimpleNamespace(
        rounds_completed=2, final_adapter=f"adapter-{value}", per_round_pairs=(10, 20)
    )
    monkeypatch.setattr(iterative_dpo_utils, "run_iterative_dpo", lambda plan: done)

    result = _run()

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    adapter = f"adapter-{_shown(value)}"
    assert f"Done \u2014 2 round(s); final adapter: {adapter}; pairs/round: [10, 20]" in text
    assert _not_shown(text)


# --- styles ----------------------------------------------------------------


def _coloured(monkeypatch):
    monkeypatch.setattr(
        iterative_dpo,
        "console",
        Console(
            force_terminal=True, no_color=False, width=500, height=100, color_system="standard"
        ),
    )


def test_the_error_label_is_red(monkeypatch):
    _coloured(monkeypatch)
    monkeypatch.setattr(
        iterative_dpo_utils, "build_iterative_dpo_plan", _raising(ValueError("plan said no"))
    )

    result = _run("--plan-only")

    assert result.exit_code == 2
    assert "\x1b[31mError:\x1b[0m plan said no" in result.output


def test_the_failure_panel_label_is_red(monkeypatch):
    _coloured(monkeypatch)
    monkeypatch.setattr(iterative_dpo_utils, "validate_iterative_dpo_prompts", lambda plan: None)
    failing = _raising(RuntimeError("run said no"))
    monkeypatch.setattr(iterative_dpo_utils, "run_iterative_dpo", failing)

    result = _run()

    assert result.exit_code == 1
    assert "\x1b[31mIterative DPO failed:\x1b[0m run said no" in result.output


def test_the_done_label_is_green(monkeypatch):
    _coloured(monkeypatch)
    monkeypatch.setattr(iterative_dpo_utils, "validate_iterative_dpo_prompts", lambda plan: None)
    done = SimpleNamespace(rounds_completed=1, final_adapter="adapter", per_round_pairs=(5,))
    monkeypatch.setattr(iterative_dpo_utils, "run_iterative_dpo", lambda plan: done)

    result = _run()

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "\x1b[32mDone\x1b[0m \u2014 1 round(s); final adapter: adapter;" in result.output


def test_the_prompt_check_failure_label_is_red(monkeypatch):
    _coloured(monkeypatch)
    failing = _raising(RuntimeError("check said no"))
    monkeypatch.setattr(iterative_dpo_utils, "validate_iterative_dpo_prompts", failing)

    result = _run()

    assert result.exit_code == 1
    assert "\x1b[31mIterative DPO failed:\x1b[0m check said no" in result.output
