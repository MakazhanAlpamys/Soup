"""soup eval design, discover, lock, coverage, against and gate-install show typed text."""

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from rich.console import Console
from typer.testing import CliRunner

import soup_cli.experiment.tracker as tracker_module
import soup_cli.registry.attach as registry_attach
import soup_cli.utils.advise as advise
import soup_cli.utils.canary_discovery as canary_discovery
import soup_cli.utils.eval_design as eval_design
import soup_cli.utils.eval_gate_hook as eval_gate_hook
import soup_cli.utils.eval_lock_coverage as eval_lock_coverage
from soup_cli.commands._eval_v0550 import register
from soup_cli.utils.exit_codes import (
    EXIT_GATE_FAILED,
    EXIT_OK,
    EXIT_USAGE_ERROR,
)
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


def _run(*args: str, console: Console = None, input_text: str = None):
    return runner.invoke(_app(console or _plain_console()), list(args), input=input_text)


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setenv("COLUMNS", "500")
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    rows = [{"prompt": f"write sql query {n}", "response": f"SELECT {n} FROM t"} for n in range(30)]
    (workdir / "data.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    made = _run("design", "data.jsonl", "--goal", "better at SQL", "-n", "3", "-o", "design.json")
    assert made.exit_code == 0, made.output


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


class _Tracker:
    """Stands in for the experiment database: ``series`` maps (run id, metric) to a list."""

    series: dict = {}
    error: Exception = None

    def get_metric_series(self, run_id, metric):
        if self.error is not None:
            raise self.error
        return list(self.series.get((run_id, metric), []))


@pytest.fixture
def tracker(monkeypatch):
    class Tracker(_Tracker):
        series = {}
        error = None

    monkeypatch.setattr(tracker_module, "ExperimentTracker", Tracker)
    return Tracker


# --- design ----------------------------------------------------------------


@pytest.mark.parametrize("name", _NAMES)
def test_a_missing_dataset_is_shown_as_typed(name):
    result = _run("design", f"{name}.jsonl", "--goal", "g")

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Cannot read dataset: dataset not found: {_shown(name)}.jsonl" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_a_dataset_outside_cwd_is_shown_as_typed(name):
    Path(f"../{name}.jsonl").write_text("{}\n", encoding="utf-8")

    result = _run("design", f"../{name}.jsonl", "--goal", "g")

    assert result.exit_code == 1
    text = _plain(result)
    expected = f"Cannot read dataset: dataset path '../{_shown(name)}.jsonl' must stay under cwd"
    assert expected in text
    assert _not_shown(text)


@pytest.mark.parametrize("command", ["design", "discover"])
@pytest.mark.parametrize("error", [FileNotFoundError, ValueError, TypeError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_dataset_failure_with_any_text_is_shown_as_typed(monkeypatch, command, error, value):
    monkeypatch.setattr(advise, "load_advise_dataset", _raising(error(f"dataset said {value}")))
    extra = ["--goal", "g"] if command == "design" else []

    result = _run(command, "data.jsonl", *extra)

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Cannot read dataset: dataset said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_the_goal_in_the_rubric_is_shown_as_typed(value):
    result = _run("design", "data.jsonl", "--goal", value, "-n", "3", "-o", "again.json")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Goal: {_shown(value)}." in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_the_design_table_shows_dimension_scorer_and_rubric_as_typed(monkeypatch, value):
    dimension = SimpleNamespace(
        name=f"dim-{value}", scorer_type=f"scorer-{value}", rubric=f"rubric-{value}"
    )
    design = SimpleNamespace(dimensions=[dimension])
    monkeypatch.setattr(eval_design, "design_evals_from_data", lambda *a, **k: design)
    monkeypatch.setattr(eval_design, "write_eval_design", lambda design, output: "out.json")

    result = _run("design", "data.jsonl", "--goal", "g")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    shown = _shown(value)
    assert f"\u2502 dim-{shown} \u2502 scorer-{shown} \u2502 rubric-{shown} \u2502" in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [TypeError, ValueError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_failed_design_is_shown_as_typed(monkeypatch, error, value):
    monkeypatch.setattr(
        eval_design, "design_evals_from_data", _raising(error(f"design said {value}"))
    )

    result = _run("design", "data.jsonl", "--goal", "g")

    assert result.exit_code == 2
    text = _plain(result)
    assert f"Cannot build design: design said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [ValueError, OSError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_failed_design_write_is_shown_as_typed(monkeypatch, error, value):
    monkeypatch.setattr(eval_design, "write_eval_design", _raising(error(f"write said {value}")))

    result = _run("design", "data.jsonl", "--goal", "g")

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Cannot write design: write said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_the_written_design_path_is_shown_as_typed(name):
    result = _run("design", "data.jsonl", "--goal", "g", "-n", "3", "-o", f"{name}.json")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Wrote 3 dimensions to {_shown(name)}.json" in text
    assert _not_shown(text)


# --- discover --------------------------------------------------------------


@pytest.mark.parametrize("error", [TypeError, ValueError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_failed_discovery_is_shown_as_typed(monkeypatch, error, value):
    monkeypatch.setattr(
        canary_discovery, "discover_canaries", _raising(error(f"discover said {value}"))
    )

    result = _run("discover", "data.jsonl")

    assert result.exit_code == 2
    text = _plain(result)
    assert f"Cannot discover canaries: discover said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [ValueError, OSError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_failed_canary_write_is_shown_as_typed(monkeypatch, error, value):
    monkeypatch.setattr(
        canary_discovery, "write_canary_set", _raising(error(f"write said {value}"))
    )

    result = _run("discover", "data.jsonl")

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Cannot write canary set: write said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_the_written_canary_path_is_shown_as_typed(name):
    result = _run("discover", "data.jsonl", "-o", f"{name}.json")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(
        r"Wrote \d+ held-out \+ \d+ adjacent \+ \d+ memorization probes to "
        + re.escape(_shown(name))
        + r"\.json",
        text,
    )
    assert _not_shown(text)


# --- lock and coverage -----------------------------------------------------


@pytest.mark.parametrize("command", [["lock"], ["coverage", "--task", "reasoning"]])
@pytest.mark.parametrize("name", _NAMES)
def test_a_missing_design_is_shown_as_typed(command, name):
    result = _run(command[0], f"{name}.json", *command[1:])

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Cannot load design: eval design file not found: {_shown(name)}.json" in text
    assert _not_shown(text)


@pytest.mark.parametrize("command", [["lock"], ["coverage", "--task", "reasoning"]])
@pytest.mark.parametrize("error", [FileNotFoundError, ValueError, TypeError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_design_failure_with_any_text_is_shown_as_typed(monkeypatch, command, error, value):
    monkeypatch.setattr(eval_design, "load_eval_design", _raising(error(f"design said {value}")))

    result = _run(command[0], "design.json", *command[1:])

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Cannot load design: design said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [ValueError, OSError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_failed_lock_is_shown_as_typed(monkeypatch, error, value):
    monkeypatch.setattr(eval_lock_coverage, "lock_suite", _raising(error(f"lock said {value}")))

    result = _run("lock", "design.json")

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Cannot lock suite: lock said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_the_locked_suite_path_is_shown_as_typed(name):
    result = _run("lock", "design.json", "-o", f"{name}.json")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    pattern = r"Locked 3 dimensions sha256=[0-9a-f]{16}\u2026 \u2192 "
    assert re.search(pattern + re.escape(_shown(name)) + r"\.json", text)
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_an_attached_registry_entry_is_shown_as_typed(monkeypatch, value):
    monkeypatch.setattr(registry_attach, "attach_artifact", lambda *a, **k: None)

    result = _run("lock", "design.json", "--attach-to-registry", value)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Attached to registry entry {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [ValueError, FileNotFoundError, ImportError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_skipped_registry_attach_is_shown_as_typed(monkeypatch, error, value):
    monkeypatch.setattr(registry_attach, "attach_artifact", _raising(error(f"attach said {value}")))

    result = _run("lock", "design.json", "--attach-to-registry", "entry")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Registry attach skipped: attach said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_a_refused_task_category_is_shown_as_typed(value):
    result = _run("coverage", "design.json", "--task", value)

    assert result.exit_code == 2
    text = _plain(result)
    assert f"Cannot compute coverage: unknown task_category {value!r}; allowed:" in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [TypeError, ValueError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_coverage_failure_with_any_text_is_shown_as_typed(monkeypatch, error, value):
    monkeypatch.setattr(
        eval_lock_coverage, "compute_coverage", _raising(error(f"coverage said {value}"))
    )

    result = _run("coverage", "design.json", "--task", "reasoning")

    assert result.exit_code == 2
    text = _plain(result)
    assert f"Cannot compute coverage: coverage said {_shown(value)}" in text
    assert _not_shown(text)


# --- against ---------------------------------------------------------------


@pytest.mark.parametrize("value", _VALUES)
def test_a_refused_metric_is_shown_as_typed(tracker, value):
    result = _run("against", "base", "--candidate", "cand", "--metric", value)

    assert result.exit_code == EXIT_USAGE_ERROR
    text = _plain(result)
    assert f"Invalid --metric: unknown metric {value!r}; allowed:" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_a_suite_outside_cwd_is_shown_as_typed(tracker, name):
    result = _run("against", "base", "--candidate", "cand", "--suite", f"../{name}.json")

    assert result.exit_code == EXIT_USAGE_ERROR
    text = _plain(result)
    assert f"--suite is outside cwd: ../{_shown(name)}.json" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_a_missing_locked_suite_is_shown_as_typed(tracker, name):
    result = _run("against", "base", "--candidate", "cand", "--suite", f"{name}.json")

    assert result.exit_code == EXIT_USAGE_ERROR
    text = _plain(result)
    expected = "Locked eval suite invalid \u2014 gate blocked: eval design file not found: "
    assert f"{expected}{_shown(name)}.json" in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [FileNotFoundError, ValueError, OSError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_locked_suite_failure_with_any_text_is_shown_as_typed(tracker, monkeypatch, error, value):
    monkeypatch.setattr(
        eval_lock_coverage, "load_locked_suite", _raising(error(f"suite said {value}"))
    )

    result = _run("against", "base", "--candidate", "cand", "--suite", "design.json")

    assert result.exit_code == EXIT_USAGE_ERROR
    text = _plain(result)
    assert f"Locked eval suite invalid \u2014 gate blocked: suite said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [FileNotFoundError("gone"), ValueError("bad")])
@pytest.mark.parametrize("value", _VALUES)
def test_a_failed_series_fetch_is_shown_as_typed(tracker, error, value):
    tracker.error = type(error)(f"{error} {value}")

    result = _run("against", "base", "--candidate", "cand")

    assert result.exit_code == EXIT_USAGE_ERROR
    text = _plain(result)
    assert f"Cannot fetch metric series: {error} {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_an_unavailable_comparison_names_the_metric_and_the_runs_as_typed(tracker, value):
    result = _run(
        "against", f"base-{value}", "--candidate", f"cand-{value}", "--metric", f"judge:{value}"
    )

    assert result.exit_code == EXIT_USAGE_ERROR
    text = _plain(result)
    shown = _shown(value)
    assert (
        f"Comparison unavailable \u2014 no eval results or tracked metrics for judge:{shown}: "
        f"baseline base-{shown}=0, candidate cand-{shown}=0."
    ) in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [TypeError, ValueError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_failed_regression_check_is_shown_as_typed(tracker, monkeypatch, error, value):
    tracker.series = {("base", "task_accuracy"): [0.5] * 6, ("cand", "task_accuracy"): [0.5] * 6}
    monkeypatch.setattr(
        eval_gate_hook, "decide_regression", _raising(error(f"decide said {value}"))
    )

    result = _run("against", "base", "--candidate", "cand")

    assert result.exit_code == EXIT_USAGE_ERROR
    text = _plain(result)
    assert f"Regression check failed: decide said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_the_verdict_line_names_the_metric_as_typed(tracker, value):
    metric = f"judge:{value}"
    tracker.series = {("base", metric): [0.5] * 8, ("cand", metric): [0.5] * 8}

    result = _run("against", "base", "--candidate", "cand", "--metric", metric)

    assert result.exit_code == EXIT_OK, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(r"OK judge:" + re.escape(_shown(value)) + r" delta_mean=\+0\.0000 ci=\[", text)
    assert _not_shown(text)


def test_a_regression_is_reported_with_its_tag(tracker):
    tracker.series = {
        ("base", "task_accuracy"): [0.9] * 8,
        ("cand", "task_accuracy"): [0.1] * 8,
    }

    result = _run("against", "base", "--candidate", "cand")

    assert result.exit_code == EXIT_GATE_FAILED
    assert re.search(r"REGRESSED task_accuracy delta_mean=-0\.8000 ci=\[", _plain(result))


def test_one_aggregate_result_has_no_interval(tracker):
    tracker.series = {("base", "task_accuracy"): [0.5], ("cand", "task_accuracy"): [0.5]}

    result = _run("against", "base", "--candidate", "cand")

    assert result.exit_code == EXIT_OK, (result.output, repr(result.exception))
    assert (
        "OK task_accuracy delta_mean=+0.0000 confidence interval unavailable "
        "(one aggregate result)"
    ) in _plain(result)


# --- gate-install ----------------------------------------------------------


def test_the_installed_hook_names_the_baseline_and_the_default_suite():
    result = _run("gate-install", "--baseline", "base_1", "--hook-path", "hook.sh")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "Baseline: base_1 \u2022 suite: evals/locked.json" in _plain(result)


@pytest.mark.parametrize("name", _NAMES)
def test_the_installed_hook_names_the_hook_and_the_suite_as_typed(name):
    result = _run(
        "gate-install", "--baseline", "base", "--suite", f"{name}.json",
        "--hook-path", f"{name}-hook",
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    shown = _shown(name)
    assert re.search(r"Installed pre-push gate \u2192 \S*" + re.escape(shown) + r"-hook", text)
    assert f"Baseline: base \u2022 suite: {shown}.json" in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [TypeError, ValueError, OSError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_failed_hook_install_is_shown_as_typed(monkeypatch, error, value):
    monkeypatch.setattr(
        eval_gate_hook, "write_pre_push_hook", _raising(error(f"hook said {value}"))
    )

    result = _run("gate-install", "--baseline", "base")

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Cannot install hook: hook said {_shown(value)}" in text
    assert _not_shown(text)


# --- styles ----------------------------------------------------------------


def test_errors_start_with_a_red_label_and_a_skipped_attach_with_a_yellow_one(monkeypatch):
    console = _coloured_console()
    missing = _run("design", "nope.jsonl", "--goal", "g", console=console)
    monkeypatch.setattr(registry_attach, "attach_artifact", _raising(ValueError("attach said no")))
    skipped = _run("lock", "design.json", "--attach-to-registry", "entry", console=console)

    assert missing.exit_code == 1
    assert skipped.exit_code == 0, (skipped.output, repr(skipped.exception))
    assert "\x1b[31mCannot read dataset:\x1b[0m dataset not found: nope.jsonl" in missing.output
    assert "\x1b[33mRegistry attach skipped:\x1b[0m attach said no" in skipped.output


def test_the_design_and_canary_lines_keep_their_colours():
    console = _coloured_console()

    args = ["design", "data.jsonl", "--goal", "g", "-n", "3", "-o", "out.json"]
    design = _run(*args, console=console)
    canary = _run("discover", "data.jsonl", "-o", "canary.json", console=console)

    assert design.exit_code == 0, (design.output, repr(design.exception))
    assert canary.exit_code == 0, (canary.output, repr(canary.exception))
    assert "\x1b[32mWrote 3 dimensions\x1b[0m to \x1b[36mout.json\x1b[0m" in design.output
    assert re.search(
        r"\x1b\[32mWrote \d+ held-out \+ \d+ adjacent \+ \d+ memorization probes\x1b\[0m"
        r" to \x1b\[36mcanary\.json\x1b\[0m",
        canary.output,
    )


def test_the_lock_lines_keep_their_colours(monkeypatch):
    monkeypatch.setattr(registry_attach, "attach_artifact", lambda *a, **k: None)

    result = _run(
        "lock", "design.json", "-o", "locked.json", "--attach-to-registry", "entry",
        console=_coloured_console(),
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert re.search(
        r"\x1b\[32mLocked 3 dimensions\x1b\[0m sha256=\x1b\[2m[0-9a-f]{16}\u2026\x1b\[0m"
        r" \u2192 \x1b\[36mlocked\.json\x1b\[0m",
        result.output,
    )
    assert "\x1b[32mAttached to registry entry entry\x1b[0m" in result.output


def test_the_verdict_tag_is_green_or_red(tracker):
    console = _coloured_console()
    tracker.series = {("base", "task_accuracy"): [0.5] * 8, ("cand", "task_accuracy"): [0.5] * 8}
    passed = _run("against", "base", "--candidate", "cand", console=console)
    tracker.series = {("base", "task_accuracy"): [0.9] * 8, ("cand", "task_accuracy"): [0.1] * 8}
    failed = _run("against", "base", "--candidate", "cand", console=console)

    assert "\x1b[32mOK\x1b[0m task_accuracy delta_mean=" in passed.output
    assert "\x1b[31mREGRESSED\x1b[0m task_accuracy delta_mean=" in failed.output


def test_the_gate_errors_are_red_and_the_unavailable_comparison_is_red_throughout(tracker):
    console = _coloured_console()

    metric = _run("against", "base", "--candidate", "cand", "--metric", "nope", console=console)
    outside = _run(
        "against", "base", "--candidate", "cand", "--suite", "../s.json", console=console
    )
    empty = _run("against", "base", "--candidate", "cand", console=console)

    assert "\x1b[31mInvalid --metric:\x1b[0m unknown metric 'nope'" in metric.output
    assert "\x1b[31m--suite is outside cwd:\x1b[0m ../s.json" in outside.output
    assert re.search(
        r"\x1b\[31mComparison unavailable \u2014 no eval results or tracked metrics "
        r"for task_accuracy: baseline base=0, candidate cand=0\.\x1b\[0m",
        empty.output,
    )


def test_the_installed_hook_lines_keep_their_colours():
    result = _run(
        "gate-install", "--baseline", "base_1", "--hook-path", "hook.sh",
        console=_coloured_console(),
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert re.search(
        r"\x1b\[32mInstalled pre-push gate\x1b\[0m \u2192 \x1b\[36m\S*hook\.sh\x1b\[0m",
        result.output,
    )
    assert "\x1b[2mBaseline: base_1 \u2022 suite: evals/locked.json\x1b[0m" in result.output
