"""soup ra-dit prints typed config paths, output directories and errors as typed."""

import json
import re
import sys
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

import soup_cli.utils.ra_dit_run as ra_dit_run
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
    monkeypatch.setattr("soup_cli.commands.ra_dit.console", Console(width=500, height=100))
    monkeypatch.setenv("COLUMNS", "500")
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _shown(value: str) -> str:
    return "".join(ch for ch in value if ch not in _CONTROLS)


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


def _write_config(path: str, output: str) -> None:
    # A JSON string is a valid YAML double-quoted scalar, and writes every
    # control character as a \u escape, so the file holds only ASCII.
    Path(path).write_text(f"output: {json.dumps(output)}\ntraining:\n  x: 1\n", encoding="utf-8")


def _configs(retriever_output: str = "out/r", generator_output: str = "out/g") -> None:
    _write_config("retriever.yaml", retriever_output)
    _write_config("generator.yaml", generator_output)


def _ra_dit(*args: str, retriever: str = "retriever.yaml", generator: str = "generator.yaml"):
    return runner.invoke(app, ["ra-dit", "-r", retriever, "-g", generator, *args])


def _stub_train(monkeypatch, error=None):
    def train(config_path, *, timeout_seconds):
        if error is not None:
            raise error

    monkeypatch.setattr(ra_dit_run, "_run_train_subprocess", train)


@pytest.mark.parametrize("name", _NAMES)
def test_a_retriever_config_outside_cwd_is_shown_as_typed(name):
    _configs()

    result = _ra_dit(retriever=f"../{name}.yaml")

    assert result.exit_code == 2
    text = _plain(result)
    quoted = quote_path(f"../{name}.yaml").replace("\u202e", "")
    assert f"Invalid ra-dit config: --retriever-config {quoted} must stay under cwd" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_a_missing_generator_config_is_shown_as_typed(name):
    _configs()

    result = _ra_dit(generator=f"{name}.yaml")

    assert result.exit_code == 2
    text = _plain(result)
    assert f"--generator-config not found: {f'{name}.yaml'!r}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_overlapping_outputs_are_shown_as_typed(value):
    _configs(f"out/same-{value}", f"out/same-{value}")

    result = _ra_dit("--plan-only")

    assert result.exit_code == 2
    text = _plain(result)
    quoted = quote_path(f"out/same-{value}").replace("\u202e", "")
    assert f"Invalid ra-dit config: --generator-config output {quoted} is the same file as" in text
    assert f"--retriever-config output {quoted}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_the_plan_names_the_config_paths_as_typed(name):
    _write_config(f"{name}-r.yaml", "out/r")
    _write_config(f"{name}-g.yaml", "out/g")

    result = _ra_dit("--plan-only", retriever=f"{name}-r.yaml", generator=f"{name}-g.yaml")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    shown = _shown(name)
    assert f"Stage 1 (retriever): {shown}-r.yaml" in text
    assert f"Stage 2 (generator): {shown}-g.yaml" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_the_plan_names_the_output_directories_as_typed(value):
    _configs(f"out/r-{value}", f"out/g-{value}")

    result = _ra_dit("--plan-only")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    shown = _shown(value)
    assert re.search(r"Retriever output: out/r-" + re.escape(shown) + r" (?:\W )+Stage 2", text)
    assert re.search(r"Generator output: out/g-" + re.escape(shown) + r" (?:\W )+Retriever", text)
    assert f"Retriever link: auto-link to stage-1 output: out/r-{shown} " in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_the_plan_names_a_manual_retriever_as_typed(value):
    _configs()

    result = _ra_dit("--plan-only", "--retriever-model", value)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Retriever link: manual override: {_shown(value)} " in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [ValueError, TypeError, FileNotFoundError, RuntimeError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_failed_run_is_shown_as_typed(monkeypatch, error, value):
    _configs()
    _stub_train(monkeypatch, error(f"stage said {value}"))

    result = _ra_dit()

    assert result.exit_code == 2
    text = _plain(result)
    assert f"RA-DIT run failed: stage said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_the_done_panel_names_the_outputs_as_typed(monkeypatch, value):
    _configs(f"out/r-{value}", f"out/g-{value}")
    _stub_train(monkeypatch)

    result = _ra_dit()

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    shown = re.escape(_shown(value))
    assert re.search(r"Retriever output: out/r-" + shown + r" (?:\W )+Generator output:", text)
    assert re.search(r"Generator output: out/g-" + shown + r" (?:\W )+Retriever model", text)
    assert re.search(r"Retriever model used: out/r-" + shown + r" \(auto-linked\)", text)
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_the_done_panel_names_a_manual_retriever_as_typed(monkeypatch, value):
    _configs()
    _stub_train(monkeypatch)

    result = _ra_dit("--retriever-model", value)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Retriever model used: {_shown(value)} (manual override)" in text
    assert _not_shown(text)


def _coloured(monkeypatch):
    monkeypatch.setattr(
        "soup_cli.commands.ra_dit.console",
        Console(force_terminal=True, width=500, height=100, color_system="standard"),
    )


def test_the_plan_keeps_its_labels_bold(monkeypatch):
    _coloured(monkeypatch)
    _configs()

    result = _ra_dit("--plan-only")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "\x1b[1mStage 1 (retriever):\x1b[0m retriever.yaml" in result.output
    assert "\x1b[1mRetriever output:\x1b[0m out/r" in result.output
    assert "\x1b[1mRetriever link:\x1b[0m auto-link to stage-1 output: out/r" in result.output
    assert "\x1b[1mPer-stage timeout:\x1b[0m 21600s" in result.output


def test_every_detail_of_the_plan_has_its_own_line():
    _configs()

    result = _ra_dit("--plan-only")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    lines = strip_ansi(result.output).splitlines()
    labels = (
        "Stage 1 (retriever):", "Retriever output:", "Stage 2 (generator):",
        "Generator output:", "Retriever link:", "Per-stage timeout:",
    )
    for label in labels:
        assert [line for line in lines if re.match(r"\W*" + re.escape(label), line)], label


def test_the_done_panel_keeps_its_labels_bold_and_has_one_detail_per_line(monkeypatch):
    _coloured(monkeypatch)
    _configs()
    _stub_train(monkeypatch)

    result = _ra_dit()

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "\x1b[1mRetriever output:\x1b[0m out/r" in result.output
    assert "\x1b[1mGenerator output:\x1b[0m out/g" in result.output
    assert "\x1b[1mRetriever model used:\x1b[0m out/r (auto-linked)" in result.output
    lines = strip_ansi(result.output).splitlines()
    for label in ("Retriever output:", "Generator output:", "Retriever model used:"):
        assert [line for line in lines if re.match(r"\W*" + label, line)], label


def test_errors_start_with_a_red_label(monkeypatch):
    _coloured(monkeypatch)
    _configs()
    refused = _ra_dit(retriever="../missing.yaml")
    _stub_train(monkeypatch, RuntimeError("stage said no"))
    failed = _ra_dit()

    assert refused.exit_code == 2
    assert failed.exit_code == 2
    assert "\x1b[31mInvalid ra-dit config:\x1b[0m" in refused.output
    assert "\x1b[31mRA-DIT run failed:\x1b[0m" in failed.output


def test_a_plan_only_config_error_starts_with_a_red_label(monkeypatch):
    _coloured(monkeypatch)
    _configs("out/same", "out/same")

    result = _ra_dit("--plan-only")

    assert result.exit_code == 2
    assert "\x1b[31mInvalid ra-dit config:\x1b[0m" in result.output
