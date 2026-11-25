"""soup compile prints typed paths, a refused optimizer and compile errors as typed."""

import re
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.paths import quote_path
from soup_cli.utils.prompt_compile import CompileResult
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
    "x[red]n",
    pytest.param("n:warning:", marks=skip_colon_on_windows, id="emoji-code"),
    f"x[bold]n{_CONTROLS}",
]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setattr("soup_cli.commands.compile_cmd.console", Console(width=500, height=100))
    monkeypatch.setenv("COLUMNS", "500")
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    (workdir / "program.py").write_text("x = 1\n", encoding="utf-8")
    (workdir / "suite.jsonl").write_text("{}\n", encoding="utf-8")


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _shown(value: str) -> str:
    return "".join(ch for ch in value if ch not in _CONTROLS)


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


def _compile(*args: str, program: str = "program.py", suite: str = "suite.jsonl"):
    return runner.invoke(app, ["compile", program, "--eval", suite, *args])


def _result() -> CompileResult:
    return CompileResult(program_text="y = 2\n", score=0.5, iterations=3, converged=True)


def _stub_run(monkeypatch, outcome):
    def run(plan):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr("soup_cli.commands.compile_cmd.run_compile", run)


@pytest.mark.parametrize("value", _VALUES)
def test_a_refused_optimizer_is_shown_as_typed(value):
    result = _compile("--optimizer", value)

    assert result.exit_code == 2
    text = _plain(result)
    assert f"unknown optimizer {value!r}; supported:" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_a_program_outside_cwd_is_shown_as_typed(name):
    result = _compile(program=f"../{name}.py")

    assert result.exit_code == 2
    text = _plain(result)
    quoted = quote_path(f"../{name}.py").replace("\u202e", "")
    assert f"program_path {quoted} must stay under cwd" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_the_plan_names_the_program_the_suite_and_the_output_as_typed(name):
    Path(f"{name}.py").write_text("x = 1\n", encoding="utf-8")
    Path(f"{name}.jsonl").write_text("{}\n", encoding="utf-8")

    result = _compile(
        "--plan-only", "-o", f"{name}-out.py", program=f"{name}.py", suite=f"{name}.jsonl"
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    shown = re.escape(_shown(name))
    assert re.search(r"Program: \S*" + shown + r"\.py (?:\W )+Eval suite: \S*" + shown, text)
    assert re.search(r"\.jsonl (?:\W )+Optimizer: .*Output: " + shown + r"-out\.py", text)
    assert _not_shown(text)


@pytest.mark.parametrize("error", [ValueError, TypeError, RuntimeError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_compile_error_is_shown_as_typed(monkeypatch, error, value):
    _stub_run(monkeypatch, error(f"compile said {value}"))

    result = _compile()

    assert result.exit_code == 2
    text = _plain(result)
    assert f"compile said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_a_missing_dependency_is_shown_as_typed(monkeypatch, value):
    _stub_run(monkeypatch, ImportError(f"dependency said {value}"))

    result = _compile()

    assert result.exit_code == 2
    text = _plain(result)
    assert f"dependency said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_an_output_outside_cwd_is_shown_as_typed(name):
    result = _compile("-o", f"../{name}.py")

    assert result.exit_code == 2
    text = _plain(result)
    quoted = quote_path(f"../{name}.py").replace("\u202e", "")
    assert f"output_path {quoted} must stay under cwd" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_a_refused_write_is_shown_as_typed(monkeypatch, value):
    _stub_run(monkeypatch, _result())

    def refuse(text, path, field=""):
        raise ValueError(f"write said {value}")

    monkeypatch.setattr("soup_cli.commands.compile_cmd.atomic_write_text", refuse)

    result = _compile()

    assert result.exit_code == 2
    text = _plain(result)
    assert f"write said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_the_done_panel_names_the_output_as_typed(monkeypatch, name):
    _stub_run(monkeypatch, _result())

    result = _compile("-o", f"{name}.py")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(r"Output: " + re.escape(_shown(name)) + r"\.py (?:\W )+Score: 0\.500", text)
    assert _not_shown(text)
