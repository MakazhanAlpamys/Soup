"""``soup eval design / discover / lock / coverage / against / gate-install``: text as typed.

#2489. The dataset, output, design, suite and hook paths, the goal, the
registry entry and run ids, the metric, and the errors raised about them keep
``:name:`` aliases, markup and a backslash before a bracket as written, and lose
their control and bidi characters. A closing tag in ``--output`` no longer crashes ``design``,
``discover`` or ``lock``. Exit codes, written files and registry calls are
unchanged. No test here reaches the user's registry, tracker database or the
network. Each test registers the commands on a fresh Typer app with its own
console, as eval.py does with the shared one.
"""

import hashlib
import os
import re
import sys

import pytest
import typer
from rich.console import Console

from soup_cli.commands import _eval_v0550 as eval_v0550
from soup_cli.experiment import tracker as tracker_module
from soup_cli.registry import attach as registry_attach
from soup_cli.utils import canary_discovery, eval_design, eval_gate_hook, eval_lock_coverage
from soup_cli.utils.eval_lock_coverage import CoverageReport
from soup_cli.utils.exit_codes import EXIT_OK, EXIT_USAGE_ERROR
from soup_cli.utils.paths import quote_path
from tests._windows_filenames import skip_colon_on_windows
from tests.conftest import make_cli_runner, strip_ansi

VALUES = ["plain", "x[red]n", "[/]n", "n:rocket:", "x\\[n", "x\x9b\u202en", "x\x1b\x07n"]
#: What the display drops: C0 controls, a C1 CSI and a bidi override.
DROPPED = ("\x1b", "\x07", "\x9b", "\u202e")
#: A closing tag with nothing to close: raised MarkupError in the write refusals.
CLOSING_TAG = "a[/x]b"
DASH = "—"
ELLIPSIS = "…"

_NOT_A_WINDOWS_NAME = "{} cannot be part of a file name on Windows"
#: ``VALUES`` that a test passes as part of a file name the command looks up.
FILE_VALUES = [
    "plain",
    "x[red]n",
    "[/]n",
    pytest.param("n:rocket:", marks=skip_colon_on_windows),
    pytest.param(
        "x\\[n",
        marks=pytest.mark.skipif(
            sys.platform == "win32", reason=_NOT_A_WINDOWS_NAME.format("'\\'")
        ),
    ),
    "x\x9b\u202en",
    pytest.param(
        "x\x1b\x07n",
        marks=pytest.mark.skipif(
            sys.platform == "win32", reason=_NOT_A_WINDOWS_NAME.format("A C0 control")
        ),
    ),
]


def _shown(value: str) -> str:
    for char in DROPPED:
        value = value.replace(char, "")
    return value


def _assert_line(result, exit_code: int, line: str) -> None:
    """``line`` is a whole line of the output, and no dropped character is anywhere in it."""
    assert result.exit_code == exit_code, (result.output, repr(result.exception))
    assert result.exception is None or isinstance(result.exception, SystemExit), repr(
        result.exception
    )
    output = strip_ansi(result.output)
    lines = [" ".join(row.split()) for row in output.splitlines()]
    assert " ".join(line.split()) in lines, output
    assert [char for char in DROPPED if char in output] == []


def _invoke(console: Console, args):
    """Run a v0.55.0 eval command on a fresh app whose commands print to ``console``.

    ``register`` is handed its console, as eval.py does; the console writes to the
    runner's stdout, so the text is in ``result.output``.
    """
    eval_app = typer.Typer()

    @eval_app.callback()
    def _root() -> None:
        """Keep each command a named subcommand, as under ``soup eval``."""

    eval_v0550.register(eval_app, console)
    return make_cli_runner().invoke(eval_app, list(args))


def _eval(*args: str):
    return _invoke(Console(width=1000, height=100), args)


def _eval_coloured(*args: str):
    return _invoke(
        Console(
            width=1000, height=100, force_terminal=True, color_system="standard", no_color=False
        ),
        args,
    )


def _raising(error: Exception):
    def fail(*_args, **_kwargs):
        raise error

    return fail


def _empty_dataset() -> None:
    with open("data.jsonl", "w", encoding="utf-8"):
        pass


def _design_file(path: str = "design.json") -> None:
    eval_design.write_eval_design(eval_design.design_evals_from_data([], goal="g"), path)


class _Tracker:
    """Stands in for ``ExperimentTracker``: per-run series, or an error on every read."""

    def __init__(self, series=None, error=None):
        self.series = series or {}
        self.error = error

    def get_metric_series(self, run_id, _metric):
        if self.error is not None:
            raise self.error
        return self.series.get(run_id, [])


def _tracker(monkeypatch, **kwargs) -> None:
    fake = _Tracker(**kwargs)
    monkeypatch.setattr(tracker_module, "ExperimentTracker", lambda: fake)


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# design / discover
# ---------------------------------------------------------------------------

_DATA_COMMANDS = {"design": ("--goal", "g"), "discover": ()}


@pytest.mark.parametrize("command", sorted(_DATA_COMMANDS))
@pytest.mark.parametrize("value", FILE_VALUES)
def test_a_missing_dataset_is_shown_as_typed(workdir, command, value):
    result = _eval(command, f"d{value}.jsonl", *_DATA_COMMANDS[command])

    _assert_line(result, 1, f"Cannot read dataset: dataset not found: d{_shown(value)}.jsonl")


_WRITE_REFUSALS = {"design": "Cannot write design:", "discover": "Cannot write canary set:"}


@pytest.mark.parametrize("command", sorted(_WRITE_REFUSALS))
@pytest.mark.parametrize("value", [*VALUES, CLOSING_TAG])
def test_an_output_outside_cwd_is_shown_as_typed(workdir, command, value):
    _empty_dataset()

    result = _eval(command, "data.jsonl", *_DATA_COMMANDS[command], "--output", f"../{value}")

    reason = f"output_path {quote_path(f'../{value}')} must stay under cwd"
    _assert_line(result, 1, f"{_WRITE_REFUSALS[command]} {_shown(reason)}")


@pytest.mark.parametrize("value", VALUES)
def test_a_design_builder_error_is_shown_as_typed(workdir, monkeypatch, value):
    _empty_dataset()
    monkeypatch.setattr(eval_design, "design_evals_from_data", _raising(ValueError(value)))

    result = _eval("design", "data.jsonl", "--goal", "g")

    _assert_line(result, 2, f"Cannot build design: {_shown(value)}")


@pytest.mark.parametrize("value", VALUES)
def test_a_canary_discovery_error_is_shown_as_typed(workdir, monkeypatch, value):
    _empty_dataset()
    monkeypatch.setattr(canary_discovery, "discover_canaries", _raising(ValueError(value)))

    result = _eval("discover", "data.jsonl")

    _assert_line(result, 2, f"Cannot discover canaries: {_shown(value)}")


@pytest.mark.parametrize("value", VALUES)
def test_the_goal_in_a_rubric_is_shown_as_typed(workdir, value):
    _empty_dataset()

    result = _eval("design", "data.jsonl", "--goal", value)

    (dimension,) = eval_design.design_evals_from_data([], goal=value).dimensions
    _assert_line(
        result,
        0,
        f"│ {dimension.name} │ {dimension.scorer_type} │ {_shown(dimension.rubric)} │",
    )


@pytest.mark.parametrize("value", FILE_VALUES)
def test_the_design_receipt_shows_the_output_as_typed(workdir, value):
    _empty_dataset()
    output = f"evals/o{value}.json"

    result = _eval("design", "data.jsonl", "--goal", "g", "--output", output)

    _assert_line(result, 0, f"Wrote 1 dimensions to {_shown(output)}")
    assert os.path.isfile(output)


@pytest.mark.parametrize("value", FILE_VALUES)
def test_the_canary_receipt_shows_the_output_as_typed(workdir, value):
    _empty_dataset()
    output = f"evals/c{value}.json"

    result = _eval("discover", "data.jsonl", "--output", output)

    canary = canary_discovery.discover_canaries([], num_clusters=5, per_cluster=3, seed=0)
    _assert_line(
        result,
        0,
        f"Wrote {len(canary.held_out)} held-out + {len(canary.adjacent_skills)} adjacent + "
        f"{len(canary.memorization_probes)} memorization probes to {_shown(output)}",
    )
    assert os.path.isfile(output)


# ---------------------------------------------------------------------------
# lock / coverage
# ---------------------------------------------------------------------------

_DESIGN_COMMANDS = {"lock": (), "coverage": ("--task", "reasoning")}


@pytest.mark.parametrize("command", sorted(_DESIGN_COMMANDS))
@pytest.mark.parametrize("value", FILE_VALUES)
def test_a_missing_design_is_shown_as_typed(workdir, command, value):
    path = f"d{value}.json"

    result = _eval(command, path, *_DESIGN_COMMANDS[command])

    name = _shown(os.path.basename(path))
    _assert_line(result, 1, f"Cannot load design: eval design file not found: {name}")


@pytest.mark.parametrize("command", sorted(_DESIGN_COMMANDS))
@pytest.mark.parametrize("value", VALUES)
def test_a_design_outside_cwd_names_the_reason(workdir, command, value):
    result = _eval(command, f"../d{value}.json", *_DESIGN_COMMANDS[command])

    _assert_line(result, 1, "Cannot load design: path must stay under cwd")


@pytest.mark.parametrize("value", [*VALUES, CLOSING_TAG])
def test_a_lock_output_outside_cwd_is_shown_as_typed(workdir, value):
    _design_file()

    result = _eval("lock", "design.json", "--output", f"../{value}")

    reason = f"output_path {quote_path(f'../{value}')} must stay under cwd"
    _assert_line(result, 1, f"Cannot lock suite: {_shown(reason)}")


@pytest.mark.parametrize("value", FILE_VALUES)
def test_the_lock_receipt_shows_the_output_as_typed(workdir, value):
    _design_file()
    output = f"evals/l{value}.json"

    result = _eval("lock", "design.json", "--output", output)

    with open(output, "rb") as handle:
        checksum = hashlib.sha256(handle.read()).hexdigest()
    _assert_line(
        result, 0, f"Locked 1 dimensions sha256={checksum[:16]}{ELLIPSIS} → {_shown(output)}"
    )


@pytest.mark.parametrize("value", VALUES)
def test_the_attach_receipt_shows_the_entry_as_typed(workdir, monkeypatch, value):
    _design_file()
    calls = []
    monkeypatch.setattr(
        registry_attach, "attach_artifact", lambda *args, **kwargs: calls.append((args, kwargs))
    )

    result = _eval("lock", "design.json", "--attach-to-registry", value)

    _assert_line(result, 0, f"Attached to registry entry {_shown(value)}")
    assert calls == [((value,), {"kind": "eval_suite", "path": "evals/locked.json"})]


@pytest.mark.parametrize("value", VALUES)
def test_a_skipped_attach_is_shown_as_typed(workdir, monkeypatch, value):
    _design_file()
    monkeypatch.setattr(registry_attach, "attach_artifact", _raising(ValueError(value)))

    result = _eval("lock", "design.json", "--attach-to-registry", "entry")

    _assert_line(result, 0, f"Registry attach skipped: {_shown(value)}")


@pytest.mark.parametrize("value", [*VALUES, CLOSING_TAG])
def test_an_unknown_task_is_shown_as_typed(workdir, value):
    _design_file()

    result = _eval("coverage", "design.json", "--task", value)

    design = eval_design.load_eval_design("design.json")
    with pytest.raises(ValueError) as refused:
        eval_lock_coverage.compute_coverage(design, task_category=value)
    _assert_line(result, 2, f"Cannot compute coverage: {_shown(str(refused.value))}")


@pytest.mark.parametrize("value", VALUES)
def test_the_coverage_title_and_advice_are_shown_as_typed(workdir, monkeypatch, value):
    _design_file()
    report = CoverageReport(
        task_category=value,
        scorer_mix={"judge": 1},
        missing_scorers=(),
        recommendations=(f"add {value}",),
    )
    monkeypatch.setattr(eval_lock_coverage, "compute_coverage", lambda *_a, **_k: report)

    result = _eval("coverage", "design.json", "--task", "reasoning")

    _assert_line(result, 0, f"Coverage {DASH} {_shown(value)}")
    _assert_line(result, 0, f"• add {_shown(value)}")


# ---------------------------------------------------------------------------
# against
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [*VALUES, CLOSING_TAG])
def test_an_unknown_metric_is_shown_as_typed(workdir, value):
    result = _eval("against", "base", "--candidate", "cand", "--metric", value)

    with pytest.raises(ValueError) as refused:
        eval_gate_hook.resolve_gate_metric(value)
    _assert_line(result, EXIT_USAGE_ERROR, f"Invalid --metric: {_shown(str(refused.value))}")


@pytest.mark.parametrize("value", [*VALUES, CLOSING_TAG])
def test_a_suite_outside_cwd_is_shown_as_typed(workdir, value):
    result = _eval("against", "base", "--candidate", "cand", "--suite", f"../{value}")

    _assert_line(result, EXIT_USAGE_ERROR, f"--suite is outside cwd: ../{_shown(value)}")


@pytest.mark.parametrize("value", FILE_VALUES)
def test_a_missing_suite_is_shown_as_typed(workdir, value):
    path = f"s{value}.json"

    result = _eval("against", "base", "--candidate", "cand", "--suite", path)

    _assert_line(
        result,
        EXIT_USAGE_ERROR,
        f"Locked eval suite invalid {DASH} gate blocked: eval design file not found: "
        f"{_shown(os.path.basename(path))}",
    )


@pytest.mark.parametrize("value", VALUES)
def test_a_metric_series_error_is_shown_as_typed(workdir, monkeypatch, value):
    _tracker(monkeypatch, error=ValueError(value))

    result = _eval("against", "base", "--candidate", "cand")

    _assert_line(result, EXIT_USAGE_ERROR, f"Cannot fetch metric series: {_shown(value)}")


@pytest.mark.parametrize("value", [*VALUES, CLOSING_TAG])
def test_an_unavailable_comparison_shows_the_run_ids_as_typed(workdir, monkeypatch, value):
    _tracker(monkeypatch, series={f"c{value}": [0.5]})

    result = _eval("against", value, "--candidate", f"c{value}")

    _assert_line(
        result,
        EXIT_USAGE_ERROR,
        f"Comparison unavailable {DASH} no eval results or tracked metrics for task_accuracy: "
        f"baseline {_shown(value)}=0, candidate c{_shown(value)}=1.",
    )


@pytest.mark.parametrize("value", VALUES)
def test_a_regression_check_error_is_shown_as_typed(workdir, monkeypatch, value):
    _tracker(monkeypatch, series={"base": [0.5], "cand": [0.5]})
    monkeypatch.setattr(eval_gate_hook, "decide_regression", _raising(ValueError(value)))

    result = _eval("against", "base", "--candidate", "cand")

    _assert_line(result, EXIT_USAGE_ERROR, f"Regression check failed: {_shown(value)}")


#: ``judge:<model>`` refuses a C0 control in the model name.
@pytest.mark.parametrize("value", [v for v in VALUES if "\x1b" not in v])
def test_the_verdict_shows_the_metric_as_typed(workdir, monkeypatch, value):
    _tracker(monkeypatch, series={"base": [0.5, 0.6], "cand": [0.5, 0.6]})

    result = _eval("against", "base", "--candidate", "cand", "--metric", f"judge:{value}")

    _assert_line(
        result,
        EXIT_OK,
        f"OK judge:{_shown(value)} delta_mean=+0.0000 ci=[+0.0000, +0.0000]",
    )


# ---------------------------------------------------------------------------
# gate-install
# ---------------------------------------------------------------------------

#: Suite paths the hook accepts: its shell quoting refuses a C0 control, and on
#: Windows the cwd check does not resolve a name with ``:`` under cwd.
_SUITE_VALUES = [
    pytest.param(v, marks=skip_colon_on_windows) if ":" in v else v
    for v in VALUES
    if "\x1b" not in v
]


@pytest.mark.parametrize("value", VALUES)
def test_a_hook_install_error_is_shown_as_typed(workdir, monkeypatch, value):
    monkeypatch.setattr(eval_gate_hook, "write_pre_push_hook", _raising(ValueError(value)))

    result = _eval("gate-install", "--baseline", "base")

    _assert_line(result, 1, f"Cannot install hook: {_shown(value)}")


@pytest.mark.parametrize("value", [v for v in [*VALUES, CLOSING_TAG] if v != "plain"])
def test_a_refused_baseline_names_the_rule(workdir, value):
    result = _eval("gate-install", "--baseline", value)

    _assert_line(result, 1, "Cannot install hook: run_id must be alphanumeric + '_-' (1-128 chars)")
    assert not os.path.exists(".git/hooks/pre-push")


@pytest.mark.parametrize("value", _SUITE_VALUES)
def test_the_gate_receipt_shows_the_suite_as_typed(workdir, value):
    suite = f"evals/{value}.json"

    result = _eval("gate-install", "--baseline", "base", "--suite", suite)

    _assert_line(result, 0, "Installed pre-push gate → .git/hooks/pre-push")
    _assert_line(result, 0, f"Baseline: base • suite: {_shown(suite)}")


@pytest.mark.parametrize("value", FILE_VALUES)
def test_the_gate_receipt_shows_the_hook_path_as_typed(workdir, value):
    hook = f"hooks/{value}"

    result = _eval("gate-install", "--baseline", "base", "--hook-path", hook)

    _assert_line(result, 0, f"Installed pre-push gate → {_shown(hook)}")
    assert os.path.isfile(hook)


@pytest.mark.parametrize("value", VALUES)
def test_the_gate_receipt_shows_the_baseline_as_typed(workdir, monkeypatch, value):
    calls = []

    def write(**kwargs):
        calls.append(kwargs)
        return kwargs["hook_path"]

    monkeypatch.setattr(eval_gate_hook, "write_pre_push_hook", write)

    result = _eval(
        "gate-install", "--baseline", value, "--suite", f"s{value}", "--hook-path", f"h{value}"
    )

    _assert_line(result, 0, f"Installed pre-push gate → h{_shown(value)}")
    _assert_line(result, 0, f"Baseline: {_shown(value)} • suite: s{_shown(value)}")
    assert calls == [
        {
            "baseline_run_id": value,
            "suite_path": f"s{value}",
            "hook_path": f"h{value}",
            "overwrite": False,
        }
    ]


# ---------------------------------------------------------------------------
# Styles kept on a colour terminal
# ---------------------------------------------------------------------------


# Every test here runs in ``workdir`` and prints through ``_eval_coloured``.


def test_a_refusal_label_stays_red(workdir):
    _empty_dataset()

    result = _eval_coloured("design", "data.jsonl", "--goal", "g", "--output", "../out.json")

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert (
        "\x1b[31mCannot write design:\x1b[0m output_path '../out.json' must stay under cwd"
        in result.output
    )


def test_a_skipped_attach_label_stays_yellow(workdir, monkeypatch):
    _design_file()
    monkeypatch.setattr(registry_attach, "attach_artifact", _raising(ValueError("no entry")))

    result = _eval_coloured("lock", "design.json", "--attach-to-registry", "entry")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "\x1b[33mRegistry attach skipped:\x1b[0m no entry" in result.output


def test_the_design_receipt_stays_green_and_cyan(workdir):
    _empty_dataset()

    result = _eval_coloured("design", "data.jsonl", "--goal", "g")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "\x1b[32mWrote 1 dimensions\x1b[0m to \x1b[36mevals/design.json\x1b[0m" in result.output


def test_the_lock_checksum_stays_dim(workdir):
    _design_file()

    result = _eval_coloured("lock", "design.json")

    with open("evals/locked.json", "rb") as handle:
        checksum = hashlib.sha256(handle.read()).hexdigest()
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert (
        f"\x1b[32mLocked 1 dimensions\x1b[0m sha256=\x1b[2m{checksum[:16]}{ELLIPSIS}\x1b[0m"
        f" → \x1b[36mevals/locked.json\x1b[0m" in result.output
    )


def test_the_coverage_title_stays_italic_and_the_bullet_dim(workdir):
    _design_file()

    result = _eval_coloured("coverage", "design.json", "--task", "reasoning")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    # Rich centres the title inside its own style, so the padding is italic too.
    assert re.search(f"\x1b\\[3m +Coverage {DASH} reasoning +\x1b\\[0m\n", result.output)
    assert "\x1b[2m•\x1b[0m task category 'reasoning' benefits from" in result.output


def test_an_unavailable_comparison_stays_red(workdir, monkeypatch):
    _tracker(monkeypatch, series={"cand": [0.5]})

    result = _eval_coloured("against", "base", "--candidate", "cand")

    assert result.exit_code == EXIT_USAGE_ERROR, (result.output, repr(result.exception))
    assert (
        f"\x1b[31mComparison unavailable {DASH} no eval results or tracked metrics for "
        "task_accuracy: baseline base=0, candidate cand=1.\x1b[0m" in result.output
    )


def test_the_gate_receipt_stays_green_and_cyan_with_a_dim_line(workdir):
    result = _eval_coloured("gate-install", "--baseline", "base")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert (
        "\x1b[32mInstalled pre-push gate\x1b[0m → \x1b[36m.git/hooks/pre-push\x1b[0m"
        in result.output
    )
    assert "\x1b[2mBaseline: base • suite: evals/locked.json\x1b[0m" in result.output
