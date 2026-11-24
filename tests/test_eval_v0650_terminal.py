"""soup eval behavior / capability / checklist / irt-subset print typed text as typed."""

import json
import re
import sys
from pathlib import Path

import pytest
import typer
import yaml
from rich.console import Console
from typer.testing import CliRunner

import soup_cli.utils.behavior_battery as behavior_battery
import soup_cli.utils.capability_suite as capability_suite
import soup_cli.utils.checklist_dsl as checklist_dsl
import soup_cli.utils.irt as irt
from soup_cli.commands._eval_v0650 import register
from soup_cli.utils.exit_codes import EXIT_RUNTIME_ERROR, EXIT_USAGE_ERROR
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
    monkeypatch.setenv("COLUMNS", "500")
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setattr("soup_cli.commands._eval_v0650.warn_stopped_inputs", lambda *a, **k: None)
    (workdir / "ev.json").write_text(
        json.dumps({
            "pre_responses": ["safe"] * 5, "post_responses": ["safe"] * 5, "oracle": ["safe"] * 5,
        }),
        encoding="utf-8",
    )


def _app(console: Console) -> typer.Typer:
    app = typer.Typer()
    register(app, console)
    return app


def _plain_console() -> Console:
    return Console(width=500, height=100)


def _coloured_console() -> Console:
    return Console(
        force_terminal=True, no_color=False, width=500, height=100, color_system="standard"
    )


def _run(*args: str, console: Console = None):
    return runner.invoke(_app(console or _plain_console()), list(args))


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


# --- behavior --------------------------------------------------------------


@pytest.mark.parametrize("value", _VALUES)
def test_a_refused_battery_is_shown_as_typed(value):
    result = _run("behavior", "r1", "--battery", value, "--evidence", "ev.json")

    assert result.exit_code == EXIT_USAGE_ERROR
    text = _plain(result)
    assert f"Invalid battery: unknown battery {value.lower()!r}; valid:" in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [ValueError, TypeError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_battery_refusal_with_any_text_is_shown_as_typed(monkeypatch, error, value):
    monkeypatch.setattr(
        behavior_battery, "validate_battery_name", _raising(error(f"battery said {value}"))
    )

    result = _run("behavior", "r1", "--evidence", "ev.json")

    assert result.exit_code == EXIT_USAGE_ERROR
    text = _plain(result)
    assert f"Invalid battery: battery said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [RuntimeError, ValueError, TypeError, OSError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_failed_live_diff_is_shown_as_typed(monkeypatch, error, value):
    live = _raising(error(f"live said {value}"))
    monkeypatch.setattr(behavior_battery, "run_behavior_live", live)

    result = _run("behavior", "r1", "--base-model", "tiny-model")

    assert result.exit_code == EXIT_RUNTIME_ERROR
    text = _plain(result)
    assert f"Live behaviour diff failed: live said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_an_output_outside_cwd_is_shown_as_typed(name):
    result = _run("behavior", "r1", "--evidence", "ev.json", "--output", f"../{name}.json")

    assert result.exit_code == EXIT_USAGE_ERROR
    text = _plain(result)
    quoted = quote_path(f"../{name}.json").replace("\u202e", "")
    assert f"Invalid --output: --output {quoted} must stay under cwd" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_the_written_output_is_shown_as_typed(name):
    result = _run("behavior", "r1", "--evidence", "ev.json", "--output", f"{name}.json")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Wrote {_shown(name)}.json" in text
    assert _not_shown(text)
    assert json.loads(Path(f"{name}.json").read_text(encoding="utf-8"))["run_id"] == "r1"


# --- capability ------------------------------------------------------------


@pytest.mark.parametrize("value", _VALUES)
def test_a_refused_suite_is_shown_as_typed(value):
    result = _run("capability", "r1", "--suite", value)

    assert result.exit_code == 2
    text = _plain(result)
    assert f"Invalid --suite: unknown suite {value.lower()!r}; valid:" in text
    assert "(valid: code, fast, full, math)" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_a_suite_refusal_with_any_text_is_shown_as_typed(monkeypatch, value):
    monkeypatch.setattr(
        capability_suite, "validate_suite_name", _raising(ValueError(f"suite said {value}"))
    )

    result = _run("capability", "r1")

    assert result.exit_code == 2
    text = _plain(result)
    assert f"Invalid --suite: suite said {_shown(value)} (valid: code, fast, full, math)" in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [RuntimeError, ValueError, TypeError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_failed_capability_run_is_shown_as_typed(monkeypatch, error, value):
    monkeypatch.setattr(
        capability_suite, "run_capability_suite", _raising(error(f"lm-eval said {value}"))
    )

    result = _run("capability", "r1", "--live", "--model", "tiny-model")

    assert result.exit_code == 2
    text = _plain(result)
    assert f"Capability run failed: lm-eval said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_the_live_table_shows_benchmarks_metrics_and_errors_as_typed(monkeypatch, value):
    payload = {
        "results": [
            {"benchmark": f"bench-{value}", "metric": f"metric-{value}", "score": 0.5},
            {"benchmark": f"broken-{value}", "error": f"error-{value}"},
        ]
    }
    monkeypatch.setattr(capability_suite, "run_capability_suite", lambda **kwargs: payload)

    result = _run("capability", "r1", "--live", "--model", "tiny-model")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    shown = _shown(value)
    assert f"\u2502 bench-{shown} \u2502 metric-{shown} \u2502 0.5000 \u2502" in text
    assert f"\u2502 broken-{shown} \u2502 error \u2502 error-{shown} \u2502" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_the_manifest_output_is_shown_as_typed(name):
    result = _run("capability", "r1", "--output", f"{name}.json")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Wrote {_shown(name)}.json" in text
    assert _not_shown(text)


# --- checklist -------------------------------------------------------------


def _write_spec(path: str, test_name: str = "t1") -> None:
    spec = {"tests": [{"name": test_name, "kind": "mft", "prompts": ["p"], "expected": ["a"]}]}
    Path(path).write_text(yaml.safe_dump(spec), encoding="utf-8")


@pytest.mark.parametrize("name", _NAMES)
def test_a_missing_checklist_spec_is_shown_as_typed(name):
    result = _run("checklist", f"{name}.yaml", "--evidence", "ev.json")

    assert result.exit_code == EXIT_USAGE_ERROR
    text = _plain(result)
    assert "Failed to load spec: " in text
    assert repr(f"{name}.yaml") in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_a_checklist_spec_outside_cwd_is_shown_as_typed(name):
    result = _run("checklist", f"../{name}.yaml", "--evidence", "ev.json")

    assert result.exit_code == EXIT_USAGE_ERROR
    text = _plain(result)
    quoted = quote_path(f"../{name}.yaml").replace("\u202e", "")
    assert f"Failed to load spec: spec_path {quoted} must stay under cwd" in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [TypeError, ValueError, OSError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_checklist_spec_failure_with_any_text_is_shown_as_typed(monkeypatch, error, value):
    monkeypatch.setattr(
        checklist_dsl, "load_checklist_spec", _raising(error(f"spec said {value}"))
    )

    result = _run("checklist", "spec.yaml", "--evidence", "ev.json")

    assert result.exit_code == EXIT_USAGE_ERROR
    text = _plain(result)
    assert f"Failed to load spec: spec said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_the_checklist_table_shows_a_test_name_as_typed(value):
    _write_spec("spec.yaml", test_name=f"t-{value}")
    Path("checks.json").write_text(json.dumps({f"t-{value}": ["a"]}), encoding="utf-8")

    result = _run("checklist", "spec.yaml", "--evidence", "checks.json")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"\u2502 t-{_shown(value)} \u2502 mft \u2502 1 \u2502 1 \u2502 OK \u2502" in text
    assert _not_shown(text)


# --- irt-subset ------------------------------------------------------------


def _write_responses(path: str, items: int = 12) -> None:
    rows = [{"item_id": f"item-{n}", "correct": n % 3 != 0} for n in range(items)]
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.mark.parametrize("value", _VALUES)
def test_a_refused_irt_size_is_shown_as_typed(value):
    _write_responses("r.jsonl")

    result = _run("irt-subset", "r.jsonl", "--size", value)

    assert result.exit_code == 2
    text = _plain(result)
    assert f"Invalid --size: {_shown(value)} (valid: full, small, tiny)" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_a_refused_irt_model_is_shown_as_typed(value):
    _write_responses("r.jsonl")

    result = _run("irt-subset", "r.jsonl", "--model", value)

    assert result.exit_code == 2
    text = _plain(result)
    assert f"Invalid --model: {_shown(value)} (valid: 1pl, 2pl, 3pl)" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_a_missing_responses_file_is_shown_as_typed(name):
    result = _run("irt-subset", f"{name}.jsonl")

    assert result.exit_code == 2
    text = _plain(result)
    assert "Failed to load responses: " in text
    assert repr(f"{name}.jsonl") in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_a_responses_file_outside_cwd_is_shown_as_typed(name):
    result = _run("irt-subset", f"../{name}.jsonl")

    assert result.exit_code == 2
    text = _plain(result)
    quoted = quote_path(f"../{name}.jsonl").replace("\u202e", "")
    assert f"Failed to load responses: responses_path {quoted} must stay under cwd" in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [TypeError, ValueError, OSError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_responses_failure_with_any_text_is_shown_as_typed(monkeypatch, error, value):
    monkeypatch.setattr(irt, "load_response_rows", _raising(error(f"rows said {value}")))

    result = _run("irt-subset", "r.jsonl")

    assert result.exit_code == 2
    text = _plain(result)
    assert f"Failed to load responses: rows said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_the_irt_output_is_shown_as_typed(name):
    _write_responses("r.jsonl")

    result = _run("irt-subset", "r.jsonl", "--output", f"{name}.json")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Wrote {_shown(name)}.json" in text
    assert _not_shown(text)


# --- styles and layout -----------------------------------------------------


def test_errors_start_with_a_red_label():
    console = _coloured_console()

    battery = _run("behavior", "r1", "--battery", "nope", "--evidence", "ev.json", console=console)
    size = _run("irt-subset", "r.jsonl", "--size", "nope", console=console)

    assert battery.exit_code == EXIT_USAGE_ERROR
    assert size.exit_code == 2
    assert "\x1b[31mInvalid battery:\x1b[0m unknown battery 'nope'" in battery.output
    assert "\x1b[31mInvalid --size: nope (valid: full, small, tiny)\x1b[0m" in size.output


def test_the_suite_refusal_has_a_red_label_and_an_unstyled_tail():
    result = _run("capability", "r1", "--suite", "nope", console=_coloured_console())

    assert result.exit_code == 2
    assert "\x1b[31mInvalid --suite:\x1b[0m unknown suite 'nope'" in result.output
    assert "(valid: code, fast, full, math)" in result.output


def test_the_written_path_is_green():
    result = _run(
        "behavior", "r1", "--evidence", "ev.json", "--output", "out.json",
        console=_coloured_console(),
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "\x1b[32mWrote out.json\x1b[0m" in result.output


def test_a_benchmark_that_failed_has_a_red_error_cell(monkeypatch):
    payload = {"results": [{"benchmark": "bad", "error": "boom"}]}
    monkeypatch.setattr(capability_suite, "run_capability_suite", lambda **kwargs: payload)

    result = _run(
        "capability", "r1", "--live", "--model", "tiny-model", console=_coloured_console()
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert re.search(r"\x1b\[31merror\s*\x1b\[0m", result.output)


def test_a_failed_live_diff_starts_with_a_red_label(monkeypatch):
    monkeypatch.setattr(behavior_battery, "run_behavior_live", _raising(RuntimeError("down")))

    result = _run("behavior", "r1", "--base-model", "tiny-model", console=_coloured_console())

    assert result.exit_code == EXIT_RUNTIME_ERROR
    assert "\x1b[31mLive behaviour diff failed:\x1b[0m down" in result.output


def test_other_error_labels_are_red(monkeypatch):
    console = _coloured_console()
    monkeypatch.setattr(
        capability_suite, "run_capability_suite", _raising(RuntimeError("lm-eval down"))
    )
    monkeypatch.setattr(checklist_dsl, "load_checklist_spec", _raising(ValueError("no spec")))
    monkeypatch.setattr(irt, "load_response_rows", _raising(ValueError("no rows")))

    capability = _run("capability", "r1", "--live", "--model", "m", console=console)
    checklist = _run("checklist", "spec.yaml", "--evidence", "ev.json", console=console)
    responses = _run("irt-subset", "r.jsonl", console=console)

    assert "\x1b[31mCapability run failed:\x1b[0m lm-eval down" in capability.output
    assert "\x1b[31mFailed to load spec:\x1b[0m no spec" in checklist.output
    assert "\x1b[31mFailed to load responses:\x1b[0m no rows" in responses.output


def test_a_refused_irt_model_is_red_throughout():
    result = _run("irt-subset", "r.jsonl", "--model", "nope", console=_coloured_console())

    assert result.exit_code == 2
    assert "\x1b[31mInvalid --model: nope (valid: 1pl, 2pl, 3pl)\x1b[0m" in result.output
