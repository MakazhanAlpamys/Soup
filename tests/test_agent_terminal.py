"""soup agent prints spec text, paths and errors as typed."""

import json
import re
import sys
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

import soup_cli.utils.agent_forge as agent_forge
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
    monkeypatch.setattr("soup_cli.commands.agent.console", Console(width=500, height=100))
    monkeypatch.setenv("COLUMNS", "500")
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    _write_spec("spec.json")


def _write_spec(path: str, paths: dict = None, version: str = "3.0.0") -> None:
    if paths is None:
        paths = {"/items": {"get": {"operationId": "listItems", "parameters": []}}}
    spec = {"openapi": version, "info": {"title": "t", "version": "1"}, "paths": paths}
    Path(path).write_text(json.dumps(spec), encoding="utf-8")


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _shown(value: str) -> str:
    return "".join(ch for ch in value if ch not in _CONTROLS)


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


def _synth(*args: str, spec: str = "spec.json"):
    return runner.invoke(app, ["agent", "synth", "--spec", spec, *args])


def _train(*args: str, spec: str = "spec.json", base: str = "tiny-model"):
    return runner.invoke(app, ["agent", "train", "--spec", spec, "--base", base, *args])


def _eval(*args: str, spec: str = "spec.json", predictions: str = "preds.jsonl"):
    return runner.invoke(
        app, ["agent", "eval", "--spec", spec, "--predictions", predictions, *args]
    )


def _raising(error):
    def fail(*args, **kwargs):
        raise error

    return fail


# --- synth -----------------------------------------------------------------


@pytest.mark.parametrize("name", _NAMES)
def test_a_missing_spec_is_shown_as_typed(name):
    result = _synth(spec=f"{name}.json")

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Spec load failed: {_shown(name)}.json" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_a_spec_outside_cwd_is_shown_as_typed(name):
    result = _synth(spec=f"../{name}.json")

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Spec load failed: spec_path must stay under cwd: {_shown(name)}.json" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_a_spec_that_fails_to_parse_is_shown_as_typed(monkeypatch, value):
    monkeypatch.setattr(agent_forge, "load_spec_file", _raising(RuntimeError(f"yaml said {value}")))

    result = _synth()

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Spec parse failed: yaml said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_a_refused_kind_is_shown_as_typed(value):
    result = _synth("--kind", value)

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Parse failed: unknown spec kind: {value!r}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_the_warnings_of_a_spec_without_endpoints_are_shown_as_typed(value):
    _write_spec("spec.json", paths={}, version=value)

    result = _synth()

    assert result.exit_code == 1
    text = _plain(result)
    assert "No endpoints discovered." in text
    assert f"- unrecognised openapi version: {value!r}; parser is OpenAPI 3.x" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_the_warnings_of_a_spec_whose_endpoints_are_all_skipped_are_shown_as_typed(value):
    _write_spec("spec.json", paths={f"/bad{value}\nz": {"get": {"operationId": "getBad"}}})

    result = _synth()

    assert result.exit_code == 1
    text = _plain(result)
    assert "No endpoints discovered." in text
    assert f"- skip GET /bad{_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_the_table_shows_a_path_of_the_spec_as_typed(value):
    _write_spec("spec.json", paths={f"/p{value}": {"get": {"operationId": "getP"}}})

    result = _synth()

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"getP \u2502 get \u2502 /p{_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_a_skipped_endpoint_warning_is_shown_as_typed(value):
    _write_spec(
        "spec.json",
        paths={
            "/ok": {"get": {"operationId": "listOk"}},
            f"/bad{value}\nz": {"get": {"operationId": "getBad"}},
        },
    )

    result = _synth()

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"warning: skip GET /bad{_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_the_synth_panel_names_the_output_as_typed(name):
    result = _synth("-o", f"{name}.jsonl")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(r"Output: \S*" + re.escape(_shown(name)) + r"\.jsonl", text)
    assert _not_shown(text)
    assert Path(f"{name}.jsonl").is_file()


@pytest.mark.parametrize("name", _NAMES)
def test_a_refused_synth_output_is_shown_as_typed(name):
    result = _synth("-o", f"../{name}.jsonl")

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Write failed: output_path must stay under cwd: {_shown(name)}.jsonl" in text
    assert _not_shown(text)


# --- train -----------------------------------------------------------------


@pytest.mark.parametrize("name", _NAMES)
def test_train_shows_a_missing_spec_as_typed(name):
    result = _train(spec=f"{name}.json")

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Spec error: {_shown(name)}.json" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_train_shows_a_refused_dataset_as_typed(name):
    result = _train("--dataset-out", f"../{name}.jsonl")

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Dataset write failed: output_path must stay under cwd: {_shown(name)}.jsonl" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_the_train_plan_shows_the_base_and_the_output_as_typed(value):
    result = _train("--output-dir", f"o-{value}", base=f"b-{value}")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    shown = _shown(value)
    escaped = re.escape(shown)
    assert re.search(r"Base: b-" + escaped + r" (?:\W )+Output: o-" + escaped, text)
    assert re.search(r"base: b-" + escaped + r" (?:\W )+task: sft", text)
    assert re.search(r"output: o-" + escaped + r" ", text)
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_the_train_plan_shows_the_spec_and_the_dataset_as_typed(name):
    _write_spec(f"{name}.json")

    result = _train("--dataset-out", f"{name}.jsonl", spec=f"{name}.json")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    shown = re.escape(_shown(name))
    assert re.search(r"Spec: " + shown + r"\.json (?:\W )+Endpoints: 1", text)
    assert re.search(r"Dataset: \S*" + shown + r"\.jsonl (?:\W )+Base:", text)
    assert re.search(r"train: \S*" + shown + r"\.jsonl (?:\W )+format:", text)
    assert _not_shown(text)


# --- eval ------------------------------------------------------------------


@pytest.mark.parametrize("name", _NAMES)
def test_eval_shows_a_missing_spec_as_typed(name):
    Path("preds.jsonl").write_text('{"tool": "listItems", "arguments": {}}\n', encoding="utf-8")

    result = _eval(spec=f"{name}.json")

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Spec error: {_shown(name)}.json" in text
    assert _not_shown(text)


_NO_DIR_OPEN_ERROR_ON_WINDOWS = pytest.mark.skipif(
    sys.platform == "win32", reason="opening a directory raises a different error on Windows"
)


@_NO_DIR_OPEN_ERROR_ON_WINDOWS
@pytest.mark.parametrize("sandbox", [[], ["--sandbox"]], ids=["heuristic", "sandbox"])
@pytest.mark.parametrize("name", _NAMES)
def test_eval_shows_an_unreadable_predictions_path_as_typed(name, sandbox):
    Path(name).mkdir()

    result = _eval(*sandbox, predictions=name)

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Predictions read failed: [Errno 21] Is a directory: {name!r}" in text
    assert _not_shown(text)


# --- styles and layout -----------------------------------------------------


def _coloured(monkeypatch):
    monkeypatch.setattr(
        "soup_cli.commands.agent.console",
        Console(
            force_terminal=True, no_color=False, width=500, height=100, color_system="standard"
        ),
    )


def test_errors_start_with_a_red_label(monkeypatch):
    _coloured(monkeypatch)
    Path("preds.jsonl").write_text('{"tool": "listItems", "arguments": {}}\n', encoding="utf-8")

    synth = _synth(spec="missing.json")
    train = _train(spec="missing.json")
    evaluated = _eval(spec="missing.json")

    for result in (synth, train, evaluated):
        assert result.exit_code == 1
    assert "\x1b[31mSpec load failed:\x1b[0m missing.json" in synth.output
    assert "\x1b[31mSpec error:\x1b[0m missing.json" in train.output
    assert "\x1b[31mSpec error:\x1b[0m missing.json" in evaluated.output


def test_a_warning_has_a_yellow_label_and_a_dim_line(monkeypatch):
    _coloured(monkeypatch)
    _write_spec("spec.json", paths={"/ok": {"get": {"operationId": "listOk"}}}, version="2.0")
    warned = _synth()
    _write_spec("spec.json", paths={}, version="2.0")
    empty = _synth()

    assert warned.exit_code == 0, (warned.output, repr(warned.exception))
    assert empty.exit_code == 1
    assert "\x1b[33mwarning:\x1b[0m unrecognised openapi version" in warned.output
    assert "\x1b[2m  - unrecognised openapi version" in empty.output


def test_the_synth_panel_keeps_its_values_bold(monkeypatch):
    _coloured(monkeypatch)

    result = _synth()

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "Spec kind:  \x1b[1mopenapi\x1b[0m" in result.output
    assert "Endpoints:  \x1b[1m1\x1b[0m (skipped duplicates: 0)" in result.output
    assert "Rows:       \x1b[1m" in result.output
    assert "Output:     \x1b[1m" in result.output


def test_the_train_panel_keeps_its_labels_bold(monkeypatch):
    _coloured(monkeypatch)

    result = _train()

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "Spec:      \x1b[1mspec.json\x1b[0m" in result.output
    assert "Endpoints: \x1b[1m1\x1b[0m" in result.output
    assert "Base:      \x1b[1mtiny-model\x1b[0m" in result.output
    assert "Output:    \x1b[1m./agent_train_output\x1b[0m" in result.output
    assert re.search(r"Dataset:   \x1b\[1m\S*agent_dataset\.jsonl\x1b\[0m", result.output)
    assert "\x1b[1mPlanned recipe (agent_train.yaml):\x1b[0m" in result.output
    assert "\x1b[1mRun:\x1b[0m soup train --config agent_train.yaml --yes" in result.output


def test_every_detail_of_the_panels_has_its_own_line():
    synth = _synth()
    train = _train()

    assert synth.exit_code == 0 and train.exit_code == 0
    synth_lines = strip_ansi(synth.output).splitlines()
    for label in ("Spec kind:", "Endpoints:", "Rows:", "Output:"):
        assert [line for line in synth_lines if re.match(r"\W*" + label, line)], label
    train_lines = strip_ansi(train.output).splitlines()
    labels = ("Spec:", "Endpoints:", "Dataset:", "Base:", "Output:", "Planned recipe", "Run:")
    for label in labels:
        assert [line for line in train_lines if re.match(r"\W*" + label, line)], label
    at = next(i for i, line in enumerate(train_lines) if re.match(r"\W*Planned recipe", line))
    assert re.fullmatch(r"\W*", train_lines[at - 1])
    assert re.match(r"\W*base: ", train_lines[at + 1])
    run_at = next(i for i, line in enumerate(train_lines) if re.match(r"\W*Run:", line))
    assert re.fullmatch(r"\W*", train_lines[run_at - 1])
