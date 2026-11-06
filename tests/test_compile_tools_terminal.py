"""soup compile-tools prints typed paths, model names and errors as typed."""

import json
import re
import sys
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

import soup_cli.utils.compile_tools as compile_tools_util
from soup_cli.cli import app
from soup_cli.utils.paths import quote_path
from tests.conftest import strip_ansi

runner = CliRunner()

_CONTROLS = "\x9b\u202e"

# A colon is not a legal file-name character on Windows, and "n:" there is a drive.
_NO_COLON_ON_WINDOWS = pytest.mark.skipif(
    sys.platform == "win32", reason="':' is not valid in a Windows name"
)

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
    pytest.param("n:warning:", marks=_NO_COLON_ON_WINDOWS, id="emoji-code"),
    f"x[bold]n{_CONTROLS}",
]

_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "t", "version": "1"},
    "paths": {
        "/widgets": {
            "get": {"operationId": "listWidgets", "description": "List the widgets"},
            "post": {"operationId": "createWidget", "description": "Create a widget"},
        }
    },
}


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setattr(
        "soup_cli.commands.compile_tools.console", Console(width=500, height=100)
    )
    monkeypatch.setenv("COLUMNS", "500")
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    (workdir / "spec.json").write_text(json.dumps(_SPEC), encoding="utf-8")
    (workdir / "suite.jsonl").write_text('{"q": "1"}\n', encoding="utf-8")


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _shown(value: str) -> str:
    return "".join(ch for ch in value if ch not in _CONTROLS)


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


def _compile(*args: str, spec: str = "spec.json", suite: str = "suite.jsonl"):
    return runner.invoke(app, ["compile-tools", spec, "--eval", suite, *args])


def _stub_run(monkeypatch, outcome):
    def run(plan):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr("soup_cli.commands.compile_tools.run_tool_compile", run)


def _fake_optimizer(monkeypatch):
    monkeypatch.setattr(
        compile_tools_util, "_TOOL_OPTIMIZER_OVERRIDE", lambda desc, examples, opt: "D"
    )


@pytest.mark.parametrize("value", _VALUES)
def test_a_refused_optimizer_is_shown_as_typed(value):
    result = _compile("--optimizer", value)

    assert result.exit_code == 2
    text = _plain(result)
    assert f"unknown optimizer {value!r}; supported:" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_a_spec_outside_cwd_is_shown_as_typed(name):
    result = _compile(spec=f"../{name}.json")

    assert result.exit_code == 2
    text = _plain(result)
    quoted = quote_path(f"../{name}.json").replace("\u202e", "")
    assert f"spec_path {quoted} must stay under cwd" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_an_eval_suite_outside_cwd_is_shown_as_typed(name):
    result = _compile(suite=f"../{name}.jsonl")

    assert result.exit_code == 2
    text = _plain(result)
    quoted = quote_path(f"../{name}.jsonl").replace("\u202e", "")
    assert f"eval_suite_path {quoted} must stay under cwd" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_the_plan_names_the_spec_the_suite_and_the_output_as_typed(name):
    Path(f"{name}.json").write_text(json.dumps(_SPEC), encoding="utf-8")
    Path(f"{name}.jsonl").write_text("{}\n", encoding="utf-8")

    result = _compile(
        "--plan-only", "-o", f"{name}-out.json", spec=f"{name}.json", suite=f"{name}.jsonl"
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    shown = _shown(name)
    assert f"Spec: {shown}.json" in text
    assert f"Eval: {shown}.jsonl" in text
    assert f"Output: {shown}-out.json" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_the_plan_names_the_models_and_the_engine_as_typed(value):
    result = _compile(
        "--plan-only", "--task-lm", value, "--reflection-lm", value, "--engine", value
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    shown = _shown(value)
    assert f"Task LM: {shown} " in text
    assert f"Reflect LM: {shown} " in text
    assert f"Engine: {shown} " in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [ValueError, TypeError, FileNotFoundError, RuntimeError])
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
def test_an_output_outside_cwd_is_shown_as_typed(monkeypatch, name):
    _fake_optimizer(monkeypatch)

    result = _compile("-o", f"../{name}.json")

    assert result.exit_code == 2
    text = _plain(result)
    quoted = quote_path(f"../{name}.json").replace("\u202e", "")
    assert f"output_path {quoted} must stay under cwd" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_the_done_panel_names_the_output_as_typed(monkeypatch, name):
    _fake_optimizer(monkeypatch)

    result = _compile("-o", f"{name}-out.json")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(r"Tools: 2 (?:\W )+Output: " + re.escape(_shown(name)) + r"-out\.json", text)
    assert _not_shown(text)
    written = json.loads(Path(f"{name}-out.json").read_text(encoding="utf-8"))
    assert [tool["tool"] for tool in written["tools"]] == ["listWidgets", "createWidget"]
