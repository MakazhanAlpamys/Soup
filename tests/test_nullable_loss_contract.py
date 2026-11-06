"""A run that measured no loss reports NULL, not an invented zero.

Soup Zero ran a patched 0.73.3 wheel (``0.73.3.post1.dev20261027+zero.loss1``,
SoupZero #230) for one contract: initial/final loss of a completed run are
nullable, and a callback event that measured nothing writes no metric row. This
file holds the part of that contract that fits what ``main`` already promises:

* ``runs.initial_loss`` / ``runs.final_loss`` are NULL for a run whose trainer
  summary says ``loss_summary_kind == "unavailable"`` (the trainers report 0.0
  there; the 0.0 is a placeholder, not a measurement);
* ``ExperimentTracker.finish_run`` takes ``None``; ``log_metrics`` leaves an
  omitted ``loss`` / ``lr`` NULL rather than 0.0;
* the completion panel and ``soup runs`` do not crash on, or hide, those values;
* HF's final runtime-summary ``on_log`` event (``train_runtime``, ``train_loss``
  mean, throughput) writes no tracker row and no SSE event;
* every metric row has a positive, strictly increasing integer ``id``, the row
  identity Soup Zero's ``row_id`` / ``after_row_id`` metric API is built on.

What is NOT here, because ``main`` pins the opposite: an evaluation row carries
the last measured training loss/lr (``tests/test_issue23_mlx_val_loss_producer.py``),
and a run-level ``train_loss`` mean is stored as the loss of a run too short to
log a step (``loss_summary_kind == "mean"``, #908).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands.train import _format_training_complete_loss
from soup_cli.experiment.tracker import ExperimentTracker
from soup_cli.monitoring.callback import SoupTrainerCallback
from soup_cli.trainer.loss_summary import stored_run_losses

SRC = Path(__file__).resolve().parent.parent / "src" / "soup_cli"
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
runner = CliRunner()


def _plain(text: str) -> str:
    """Strip colour escapes and collapse whitespace (Rich colours differ per OS)."""
    return " ".join(_ANSI_RE.sub("", text).split())


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "experiments.db"))


@pytest.fixture
def tracker(tmp_path):
    instance = ExperimentTracker(db_path=tmp_path / "experiments.db")
    yield instance
    instance.close()


def _start(tracker, run_id="run_null_contract"):
    return tracker.start_run(
        {"base": "test-model", "task": "sft"}, device="cpu", device_name="CPU",
        gpu_info={}, experiment_name="null-contract", run_id=run_id,
    )


# ---------------------------------------------------------------------------
# stored_run_losses: what the runs row keeps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        # A run that observed no loss: the 0.0 the trainer reports is a placeholder.
        ({"initial_loss": 0.0, "final_loss": 0.0, "loss_summary_kind": "unavailable"},
         (None, None)),
        # Real per-step values, a real two-point delta.
        ({"initial_loss": 2.5, "final_loss": 0.8, "loss_summary_kind": "delta"}, (2.5, 0.8)),
        # One real step.
        ({"initial_loss": 1.25, "final_loss": 1.25, "loss_summary_kind": "single"},
         (1.25, 1.25)),
        # Run-level mean: stored as before (#908). The mean is NOT dropped here.
        ({"initial_loss": 3.828, "final_loss": 3.828, "loss_summary_kind": "mean"},
         (3.828, 3.828)),
        # A trainer that reports no kind keeps its numbers.
        ({"initial_loss": 2.0, "final_loss": 1.0}, (2.0, 1.0)),
        # A genuine measured zero stays zero; absence and zero are different facts.
        ({"initial_loss": 0.0, "final_loss": 0.0, "loss_summary_kind": "delta"}, (0.0, 0.0)),
        ({"initial_loss": 0.0, "final_loss": 0.0}, (0.0, 0.0)),
        # Divergence is a result, not an absence: an infinite tail is kept.
        ({"initial_loss": 2.0, "final_loss": float("inf"), "loss_summary_kind": "delta"},
         (2.0, float("inf"))),
        # Ints become floats (SQLite REAL).
        ({"initial_loss": 3, "final_loss": 1, "loss_summary_kind": "delta"}, (3.0, 1.0)),
    ],
)
def test_stored_run_losses_keeps_real_values_and_nulls_the_placeholder(result, expected):
    assert stored_run_losses(result) == expected


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"loss_summary_kind": "delta"},
        {"initial_loss": None, "final_loss": None},
        {"initial_loss": None, "final_loss": None, "loss_summary_kind": "delta"},
        # bool is an int subclass and a string is not a measurement.
        {"initial_loss": True, "final_loss": False},
        {"initial_loss": "0", "final_loss": "1.5"},
        {"initial_loss": [], "final_loss": {}},
    ],
)
def test_stored_run_losses_never_fails_or_invents_for_a_result_without_numbers(result):
    assert stored_run_losses(result) == (None, None)


def test_stored_run_losses_judges_each_end_on_its_own():
    assert stored_run_losses({"initial_loss": None, "final_loss": 0.5}) == (None, 0.5)


def test_stored_run_losses_keeps_a_numpy_scalar_a_trainer_left_in_its_result():
    """The completion panel formatted these directly before; they must not turn
    into "unavailable" now that the panel asks stored_run_losses."""
    np = pytest.importorskip("numpy")
    result = {"initial_loss": np.float32(2.0), "final_loss": np.float64(1.0)}
    assert stored_run_losses(result) == (2.0, 1.0)


# ---------------------------------------------------------------------------
# tracker.finish_run / log_metrics
# ---------------------------------------------------------------------------


def test_finish_run_with_no_loss_stores_null_and_completes(tracker):
    run_id = _start(tracker)
    tracker.finish_run(run_id, None, None, 0, 1.0, "./output")
    run = tracker.get_run(run_id)
    assert run["status"] == "completed"
    assert run["initial_loss"] is None
    assert run["final_loss"] is None


def test_finish_run_keeps_a_genuine_zero_distinct_from_null(tracker):
    run_id = _start(tracker)
    tracker.finish_run(run_id, 0.0, 0.0, 5, 1.0, "./output")
    run = tracker.get_run(run_id)
    assert run["initial_loss"] == 0.0 and run["final_loss"] == 0.0


def test_an_omitted_loss_and_lr_are_null_not_zero(tracker):
    run_id = _start(tracker)
    tracker.log_metrics(run_id, step=3)
    row = tracker.get_metrics(run_id)[0]
    assert row["loss"] is None
    assert row["lr"] is None
    assert row["grad_norm"] is None


def test_a_logged_zero_is_still_a_zero(tracker):
    run_id = _start(tracker)
    tracker.log_metrics(run_id, step=3, loss=0.0, lr=0.0)
    row = tracker.get_metrics(run_id)[0]
    assert row["loss"] == 0.0 and row["lr"] == 0.0


def test_metric_rows_have_increasing_positive_integer_ids_even_at_one_step(tracker):
    """The row identity Soup Zero's ``row_id`` / ``after_row_id`` filters are built on.

    Two rows at the same step (a late real measurement) and a row with NULL loss
    must all be kept, in recorded order, each with its own id.
    """
    run_id = _start(tracker)
    tracker.log_metrics(run_id, step=5, loss=2.0, lr=1e-4)
    tracker.log_metrics(run_id, step=5, loss=None, lr=0.0)
    tracker.log_metrics(run_id, step=5, loss=0.0, lr=0.0)
    # get_metrics orders by step; arrival order is the order of the ids.
    rows = sorted(tracker.get_metrics(run_id), key=lambda row: row["id"])
    ids = [row["id"] for row in rows]
    assert len(rows) == 3
    assert all(isinstance(value, int) and not isinstance(value, bool) and value > 0
               for value in ids)
    assert len(set(ids)) == 3
    assert [row["loss"] for row in rows] == [2.0, None, 0.0]


# ---------------------------------------------------------------------------
# the call sites persist through stored_run_losses
# ---------------------------------------------------------------------------


def _finish_run_calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == "finish_run"
    ]


@pytest.mark.parametrize("relative", ["commands/train.py", "commands/sweep.py"])
def test_finish_run_callers_do_not_index_the_result_for_the_loss(relative):
    """``result["initial_loss"]`` raised KeyError on a short result and stored the
    trainer's 0.0 placeholder; both callers go through ``stored_run_losses``."""
    calls = _finish_run_calls(SRC / relative)
    assert calls, f"{relative} no longer calls finish_run: update this guard"
    for call in calls:
        for keyword in call.keywords:
            if keyword.arg not in {"initial_loss", "final_loss"}:
                continue
            offenders = [
                node for node in ast.walk(keyword.value)
                if isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name) and node.value.id == "result"
            ]
            assert not offenders, (
                f"{relative}:{call.lineno} passes {keyword.arg}=result[...] to finish_run; "
                "use stored_run_losses(result)"
            )
    source = (SRC / relative).read_text(encoding="utf-8")
    assert "stored_run_losses" in source, f"{relative} must use stored_run_losses"


def _sweep_config():
    from soup_cli.config.schema import SoupConfig

    return SoupConfig(
        base="some-model", task="sft", data={"train": "./data.jsonl", "format": "alpaca"},
    )


@pytest.mark.parametrize(
    ("trainer_result", "expected"),
    [
        ({"initial_loss": 0.0, "final_loss": 0.0, "loss_summary_kind": "unavailable"},
         (None, None)),
        ({"initial_loss": 2.0, "final_loss": 0.5, "loss_summary_kind": "delta"}, (2.0, 0.5)),
    ],
)
def test_sweep_persists_the_stored_losses(trainer_result, expected):
    from soup_cli.commands.sweep import _run_single

    result = {
        **trainer_result, "total_steps": 3, "duration_secs": 2.0, "output_dir": "./output",
        "duration": "0m",
    }
    gpu_info = {"memory_total": "0 MB", "memory_total_bytes": 0}
    with mock_patch(
        "soup_cli.data.loader.load_dataset", return_value={"train": [{"instruction": "a"}]},
    ), mock_patch(
        "soup_cli.utils.gpu.detect_device", return_value=("cpu", "CPU"),
    ), mock_patch(
        "soup_cli.utils.gpu.get_gpu_info", return_value=gpu_info,
    ), mock_patch(
        "soup_cli.experiment.tracker.ExperimentTracker",
    ) as tracker_cls, mock_patch(
        "soup_cli.monitoring.display.TrainingDisplay",
    ), mock_patch(
        "soup_cli.trainer.dispatch.build_trainer",
        return_value=MagicMock(train=MagicMock(return_value=result)),
    ):
        mock_tracker = MagicMock()
        mock_tracker.start_run.return_value = "run-sweep-1"
        tracker_cls.return_value = mock_tracker
        outcome = _run_single(_sweep_config(), {}, "sweep_run_1", None)

    kwargs = mock_tracker.finish_run.call_args.kwargs
    assert (kwargs["initial_loss"], kwargs["final_loss"]) == expected
    # The sweep ranks on the returned result: an arm with no loss must not carry the
    # trainer's 0.0 placeholder into the ranking.
    assert (outcome["initial_loss"], outcome["final_loss"]) == expected


# ---------------------------------------------------------------------------
# what a reader does with NULL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "result",
    [
        {"initial_loss": None, "final_loss": None},
        {"initial_loss": None, "final_loss": None, "loss_summary_kind": "delta"},
        {},
        {"initial_loss": 0.0, "final_loss": 0.0, "loss_summary_kind": "unavailable"},
    ],
)
def test_completion_panel_says_unavailable_instead_of_raising(result):
    assert _format_training_complete_loss(result) == "Loss: [bold]unavailable[/]"


def test_completion_panel_still_formats_real_values():
    assert _format_training_complete_loss(
        {"initial_loss": 2.5, "final_loss": 0.8, "loss_summary_kind": "delta"}
    ) == "Loss: [bold]2.5000 -> 0.8000[/]"
    assert _format_training_complete_loss(
        {"initial_loss": 0.0, "final_loss": 0.0, "loss_summary_kind": "single"}
    ) == "Loss: [bold]0.0000[/]"


def _sweep_summary(monkeypatch, rows) -> str:
    from io import StringIO

    from rich.console import Console

    import soup_cli.commands.sweep as sweep

    output = StringIO()
    monkeypatch.setattr(sweep, "console", Console(file=output, width=200, force_terminal=False))
    sweep._display_summary(rows, {})
    return _plain(output.getvalue())


def _arm(name, loss, status="completed"):
    return {"name": name, "params": {}, "run_id": f"id_{name}", "final_loss": loss,
            "duration": "1m", "status": status}


def test_a_sweep_arm_with_no_loss_is_never_the_best_run(monkeypatch):
    text = _sweep_summary(monkeypatch, [_arm("nothing", None), _arm("measured", 0.5)])
    assert "Best run: measured (loss: 0.5000)" in text
    nothing = next(line for line in text.splitlines() if "nothing" in line)
    assert "*" not in nothing


def test_a_sweep_with_only_no_loss_arms_has_no_best_run_and_no_star(monkeypatch):
    text = _sweep_summary(monkeypatch, [_arm("nothing", None)])
    assert "Best run:" not in text
    assert "*" not in text


def test_a_sweep_arm_that_measured_zero_can_still_win(monkeypatch):
    text = _sweep_summary(monkeypatch, [_arm("zero", 0.0), _arm("measured", 0.5)])
    assert "Best run: zero (loss: 0.0000)" in text


def test_runs_list_shows_a_genuine_zero_loss_and_nothing_for_null(tracker, monkeypatch):
    from io import StringIO

    from rich.console import Console

    import soup_cli.commands.runs as runs_module

    zero = _start(tracker, "run_zero_loss")
    tracker.finish_run(zero, 0.0, 0.0, 4, 10.0, "./a")
    unavailable = _start(tracker, "run_no_loss")
    tracker.finish_run(unavailable, None, None, 0, 10.0, "./b")

    # The module's Console froze its width at import; a wide one keeps each run on one line.
    output = StringIO()
    monkeypatch.setattr(runs_module, "console", Console(file=output, width=240))
    result = runner.invoke(app, ["runs"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    listing = _plain(output.getvalue())
    rows = {
        name: next(line for line in listing.splitlines() if name in line)
        for name in ("run_zero_loss", "run_no_loss")
    }
    assert "0.000 -> 0.000" in rows["run_zero_loss"]
    assert "->" not in rows["run_no_loss"]
    assert "None" not in rows["run_no_loss"]


# ---------------------------------------------------------------------------
# the callback: an event that measured nothing writes nothing
# ---------------------------------------------------------------------------


def _event(callback, step, **logs):
    callback.on_log(
        SimpleNamespace(), SimpleNamespace(global_step=step, epoch=1.0), SimpleNamespace(),
        logs=logs,
    )


@pytest.fixture
def session(tracker):
    from soup_cli.utils.train_event_buffer import reset_global_buffer

    reset_global_buffer()
    run_id = _start(tracker)
    display = MagicMock()
    callback = SoupTrainerCallback(display=display, tracker=tracker, run_id=run_id)
    yield callback, tracker, run_id, display
    reset_global_buffer()


def test_the_final_runtime_summary_writes_no_metric_row(session):
    callback, tracker, run_id, display = session
    _event(callback, 22, loss=3.6936, learning_rate=1e-5, grad_norm=2.5)
    _event(callback, 22, train_runtime=3.38, train_loss=3.828, train_steps_per_second=6.5)
    rows = tracker.get_metrics(run_id)
    assert [(row["step"], row["loss"], row["lr"]) for row in rows] == [(22, 3.6936, 1e-5)]
    # The panel still keeps the last real values (#541): only the sinks are quiet.
    assert display.update.call_args.kwargs["loss"] == 3.6936
    assert display.update.call_args.kwargs["lr"] == 1e-5


def test_a_summary_before_any_measurement_writes_no_row_and_the_panel_stays_empty(session):
    callback, tracker, run_id, display = session
    _event(callback, 22, train_runtime=3.38, train_loss=3.828)
    assert tracker.get_metrics(run_id) == []
    assert display.update.call_args.kwargs["loss"] is None


def test_the_final_runtime_summary_pushes_no_sse_event(session):
    from soup_cli.utils.train_event_buffer import get_global_buffer

    callback, _, _, _ = session
    _event(callback, 22, loss=3.6936, learning_rate=1e-5)
    _event(callback, 22, train_runtime=3.38, train_loss=3.828)
    events = get_global_buffer().snapshot()
    assert len(events) == 1
    assert events[0].loss == 3.6936


@pytest.mark.parametrize("logs", [{}, {"train_runtime": 2.0}, {"loss": None}, {"tool_calls": 2}])
def test_other_events_with_no_measurement_write_nothing(session, logs):
    from soup_cli.utils.train_event_buffer import get_global_buffer

    callback, tracker, run_id, _ = session
    _event(callback, 9, **logs)
    assert tracker.get_metrics(run_id) == []
    assert get_global_buffer().snapshot() == []


def test_a_real_zero_is_a_measurement_and_is_recorded(session):
    callback, tracker, run_id, _ = session
    _event(callback, 4, loss=0.0, learning_rate=0.0, grad_norm=0.0)
    rows = tracker.get_metrics(run_id)
    assert len(rows) == 1
    assert (rows[0]["loss"], rows[0]["lr"], rows[0]["grad_norm"]) == (0.0, 0.0, 0.0)


def test_an_evaluation_row_is_still_written_with_its_validation_loss(session):
    """Pinned contract (#23): an evaluation creates a row that carries the last
    measured training values next to its own ``val_loss``. This stays."""
    callback, tracker, run_id, _ = session
    _event(callback, 10, loss=3.0, learning_rate=1e-4)
    _event(callback, 10, eval_loss=2.5)
    rows = sorted(tracker.get_metrics(run_id), key=lambda row: row["id"])
    assert len(rows) == 2
    assert rows[1]["val_loss"] == 2.5
    assert rows[1]["loss"] == 3.0


def test_a_learning_rate_only_event_is_still_recorded(session):
    callback, tracker, run_id, _ = session
    _event(callback, 5, learning_rate=2e-5)
    rows = tracker.get_metrics(run_id)
    assert len(rows) == 1
    assert rows[0]["lr"] == 2e-5
