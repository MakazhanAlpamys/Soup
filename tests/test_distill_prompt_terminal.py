"""soup distill-prompt prints typed paths, model ids and errors as typed."""

import json
import re
import sys
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

import soup_cli.utils.prompt_distill as distill_mod
from soup_cli.cli import app
from soup_cli.utils.paths import quote_path
from tests.conftest import strip_ansi

runner = CliRunner()

_CONTROLS = "\x9b\u202e"

# A colon is not a legal file-name character on Windows, and "n:" there is a drive.
_NO_COLON_ON_WINDOWS = pytest.mark.skipif(
    sys.platform == "win32", reason="':' is not valid in a Windows name"
)

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
    pytest.param("n:warning:", marks=_NO_COLON_ON_WINDOWS, id="emoji-code"),
    pytest.param("x\\[n", marks=_NO_BACKSLASH_ON_WINDOWS, id="backslash-bracket"),
    pytest.param("n\\\\", marks=_NO_BACKSLASH_ON_WINDOWS, id="trailing-backslashes"),
    f"x[bold]n{_CONTROLS}",
]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setattr(
        "soup_cli.commands.distill_prompt.console", Console(width=500, height=100)
    )
    monkeypatch.setenv("COLUMNS", "500")
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    rows = [{"prompt": "first question"}, {"prompt": "second question"}]
    (workdir / "traces.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _shown(value: str) -> str:
    return "".join(ch for ch in value if ch not in _CONTROLS)


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


def _distill(*args: str, traces: str = "traces.jsonl", teacher: str = "big"):
    return runner.invoke(
        app,
        [
            "distill-prompt", "--traces", traces, "--teacher", teacher,
            "--student", "small", *args,
        ],
    )


def _stub_provider(monkeypatch, behaviour):
    """Replace the provider builder; ``behaviour`` is a reply dict or an exception."""

    def build(*args, **kwargs):
        if isinstance(behaviour, Exception):
            raise behaviour

        def reply(prompt):
            return behaviour

        return reply

    monkeypatch.setattr(distill_mod, "_build_provider_fn", build)


def _stub_failing_calls(monkeypatch, message):
    def build(*args, **kwargs):
        def reply(prompt):
            raise RuntimeError(message)

        return reply

    monkeypatch.setattr(distill_mod, "_build_provider_fn", build)


@pytest.mark.parametrize("value", _VALUES)
def test_a_refused_strategy_is_shown_as_typed(value):
    result = _distill("--strategy", value)

    assert result.exit_code == 2
    text = _plain(result)
    assert f"unknown strategy {value!r}; supported:" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_traces_outside_cwd_are_shown_as_typed(name):
    result = _distill(traces=f"../{name}.jsonl")

    assert result.exit_code == 2
    text = _plain(result)
    quoted = quote_path(f"../{name}.jsonl").replace("\u202e", "")
    assert f"traces_path {quoted} must stay under cwd" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_the_plan_names_the_traces_and_the_output_as_typed(name):
    Path(f"{name}.jsonl").write_text('{"prompt": "q"}\n', encoding="utf-8")

    result = _distill("--plan-only", "-o", f"{name}-out.jsonl", traces=f"{name}.jsonl")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    shown = re.escape(_shown(name))
    assert re.search(r"Traces: \S*" + shown + r"\.jsonl (?:\W )+Teacher:", text)
    assert re.search(r"Strategy: sft (?:\W )+Provider: .*Output: " + shown + r"-out\.jsonl", text)
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_the_plan_names_the_models_and_the_provider_as_typed(value):
    result = _distill("--plan-only", "--provider", value, teacher=value)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    shown = re.escape(_shown(value))
    assert re.search(r"Teacher: " + shown + r" (?:\W )+Student: small", text)
    assert re.search(r"Provider: " + shown + r" (?:\W )+Output:", text)
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_the_student_is_shown_as_typed(value):
    result = runner.invoke(
        app,
        [
            "distill-prompt", "--traces", "traces.jsonl", "--teacher", "big",
            "--student", value, "--plan-only",
        ],
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(r"Student: " + re.escape(_shown(value)) + r" (?:\W )+Strategy:", text)
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_a_refused_provider_is_shown_as_typed(value):
    result = _distill("--provider", value)

    assert result.exit_code == 2
    text = _plain(result)
    assert f"unknown judge provider: {value!r}." in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [ValueError, TypeError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_provider_setup_error_is_shown_as_typed(monkeypatch, error, value):
    _stub_provider(monkeypatch, error(f"setup said {value}"))

    result = _distill()

    assert result.exit_code == 2
    text = _plain(result)
    assert f"setup said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_a_missing_dependency_is_shown_as_typed(monkeypatch, value):
    _stub_provider(monkeypatch, ImportError(f"dependency said {value}"))

    result = _distill()

    assert result.exit_code == 2
    text = _plain(result)
    assert f"dependency said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_failed_provider_calls_are_shown_as_typed(monkeypatch, value):
    _stub_failing_calls(monkeypatch, f"call said {value}")

    result = _distill("--provider", value)

    assert result.exit_code == 1
    text = _plain(result)
    shown = _shown(value)
    assert "No usable rows produced:" in text
    assert f"--provider {shown} ({shown} default endpoint)" in text
    assert f"first error: call said {shown}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_an_output_outside_cwd_is_shown_as_typed(monkeypatch, name):
    _stub_provider(monkeypatch, {"text": "an answer"})

    result = _distill("-o", f"../{name}.jsonl")

    assert result.exit_code == 2
    text = _plain(result)
    quoted = quote_path(f"../{name}.jsonl").replace("\u202e", "")
    assert f"output_path {quoted} must stay under cwd" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_the_done_panel_names_the_output_as_typed(monkeypatch, name):
    _stub_provider(monkeypatch, {"text": "an answer"})

    result = _distill("-o", f"{name}-out.jsonl")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(r"Rows: 2 (?:\W )+Output: " + re.escape(_shown(name)) + r"-out\.jsonl", text)
    assert _not_shown(text)
    rows = Path(f"{name}-out.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2


def _coloured(monkeypatch):
    monkeypatch.setattr(
        "soup_cli.commands.distill_prompt.console",
        Console(force_terminal=True, width=500, height=100, color_system="standard"),
    )


def test_the_plan_keeps_its_values_bold(monkeypatch):
    _coloured(monkeypatch)

    result = _distill("--plan-only")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "Traces:    \x1b[1mtraces.jsonl\x1b[0m" in result.output
    assert "Strategy:  \x1b[1msft\x1b[0m" in result.output
    assert "Provider:  \x1b[1mollama\x1b[0m" in result.output


def test_every_detail_of_the_plan_has_its_own_line():
    result = _distill("--plan-only")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    lines = strip_ansi(result.output).splitlines()
    for label in ("Traces:", "Teacher:", "Student:", "Strategy:", "Provider:", "Output:"):
        assert [line for line in lines if re.match(r"\W*" + label, line)], label


def test_the_done_panel_keeps_its_values_bold(monkeypatch):
    _coloured(monkeypatch)
    _stub_provider(monkeypatch, {"text": "an answer"})

    result = _distill()

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "Rows:   \x1b[1m2\x1b[0m" in result.output
    assert "Output: \x1b[1mdistilled.jsonl\x1b[0m" in result.output


def test_a_done_panel_with_failures_and_skipped_rows_is_yellow(monkeypatch):
    _coloured(monkeypatch)
    Path("traces.jsonl").write_text(
        '{"prompt": "first"}\n{"prompt": "second"}\n{"nothing": 1}\n', encoding="utf-8"
    )
    calls = []

    def build(*args, **kwargs):
        def reply(prompt):
            calls.append(prompt)
            if len(calls) == 1:
                raise RuntimeError("one call failed")
            return {"text": "an answer"}

        return reply

    monkeypatch.setattr(distill_mod, "_build_provider_fn", build)

    result = _distill()

    assert result.exit_code == 0, (result.output, repr(result.exception))
    plain = strip_ansi(result.output)
    assert "done with provider failures" in plain
    lines = plain.splitlines()
    assert [line for line in lines if re.match(r"\W*Provider calls: 1 of 2 failed\W*$", line)]
    assert [line for line in lines if re.match(r"\W*Rows: ", line)]
    assert "Provider calls: \x1b[1;33m1 of 2 failed\x1b[0m" in result.output
    skipped = "\x1b[33m(1 of 3 trace rows skipped: no prompt found)"
    assert f"Rows:   \x1b[1m1\x1b[0m {skipped}" in result.output


@pytest.mark.parametrize("error", [ValueError, TypeError])
def test_errors_are_red_and_a_missing_dependency_yellow(monkeypatch, error):
    _coloured(monkeypatch)
    _stub_provider(monkeypatch, error("setup said no"))
    failed = _distill()
    _stub_provider(monkeypatch, ImportError("dependency said no"))
    missing = _distill()

    assert failed.exit_code == 2
    assert missing.exit_code == 2
    assert "\x1b[31msetup said no" in failed.output
    assert "\x1b[33mdependency said no" in missing.output


def test_failed_provider_calls_start_with_a_red_label(monkeypatch):
    _coloured(monkeypatch)
    _stub_failing_calls(monkeypatch, "call said no")

    result = _distill()

    assert result.exit_code == 1
    assert "\x1b[31mNo usable rows produced:\x1b[0m" in result.output


def test_a_refused_plan_is_red(monkeypatch):
    _coloured(monkeypatch)

    result = _distill("--strategy", "nope")

    assert result.exit_code == 2
    assert "\x1b[31munknown strategy" in result.output
