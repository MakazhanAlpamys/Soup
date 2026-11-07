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
  identity Soup Zero's ``row_id`` / ``after_row_id`` metric API is built on;
* two nullable, add-only ``kind`` columns say what a stored number is, so a reader
  need not guess: ``runs.loss_summary_kind`` (``per_step`` / ``mean`` /
  ``unavailable``) and ``metrics.kind`` (``train`` / ``eval`` / ``summary``). Their
  value sets live in ``soup_cli.experiment.row_kinds``; old rows stay NULL.

What is NOT changed, because ``main`` pins it: an evaluation row carries the last
measured training loss/lr (``tests/test_issue23_mlx_val_loss_producer.py``; it is
now marked ``kind = "eval"``), and a run-level ``train_loss`` mean is stored as the
loss of a run too short to log a step (#908; ``loss_summary_kind = "mean"`` now says
so).
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
from soup_cli.experiment import row_kinds
from soup_cli.experiment.tracker import ExperimentTracker
from soup_cli.monitoring.callback import SoupTrainerCallback
from soup_cli.trainer.loss_summary import stored_loss_summary_kind, stored_run_losses

SRC = Path(__file__).resolve().parent.parent / "src" / "soup_cli"
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
runner = CliRunner()


def _plain_lines(text: str) -> str:
    """Strip colour escapes (they differ per OS) and collapse whitespace within each line.

    The line breaks stay: the repo's flattening helpers join everything into one line,
    and a ``next(line for line in text.splitlines() if name in line)`` over that is the
    whole table, not the table row.
    """
    return "\n".join(" ".join(line.split()) for line in _ANSI_RE.sub("", text).splitlines())


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
    for call in calls:
        assert "loss_summary_kind" in {keyword.arg for keyword in call.keywords}, (
            f"{relative}:{call.lineno} must pass loss_summary_kind to finish_run"
        )


def _sweep_config():
    from soup_cli.config.schema import SoupConfig

    return SoupConfig(
        base="some-model", task="sft", data={"train": "./data.jsonl", "format": "alpaca"},
    )


@pytest.mark.parametrize(
    ("trainer_result", "expected", "kind"),
    [
        ({"initial_loss": 0.0, "final_loss": 0.0, "loss_summary_kind": "unavailable"},
         (None, None), "unavailable"),
        ({"initial_loss": 2.0, "final_loss": 0.5, "loss_summary_kind": "delta"},
         (2.0, 0.5), "per_step"),
        ({"initial_loss": 1.5, "final_loss": 1.5, "loss_summary_kind": "single"},
         (1.5, 1.5), "per_step"),
        # #908 stays: the mean is still the stored loss; the column says it is a mean.
        ({"initial_loss": 3.8, "final_loss": 3.8, "loss_summary_kind": "mean"},
         (3.8, 3.8), "mean"),
    ],
)
def test_sweep_persists_the_stored_losses(trainer_result, expected, kind):
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
    assert kwargs["loss_summary_kind"] == kind
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
    return _plain_lines(output.getvalue())


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
    listing = _plain_lines(output.getvalue())
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


# ---------------------------------------------------------------------------
# the two nullable ``kind`` columns: names, validation, storage, migration
# ---------------------------------------------------------------------------


def test_the_kind_value_sets_are_exactly_these_names():
    """Soup Zero's pin tests compare against these strings; renaming one is a break."""
    assert row_kinds.LOSS_SUMMARY_KINDS == ("per_step", "mean", "unavailable")
    assert (row_kinds.LOSS_SUMMARY_PER_STEP, row_kinds.LOSS_SUMMARY_MEAN,
            row_kinds.LOSS_SUMMARY_UNAVAILABLE) == ("per_step", "mean", "unavailable")
    assert row_kinds.METRIC_ROW_KINDS == ("train", "eval", "summary")
    assert (row_kinds.ROW_KIND_TRAIN, row_kinds.ROW_KIND_EVAL,
            row_kinds.ROW_KIND_SUMMARY) == ("train", "eval", "summary")


def test_the_module_docstring_lists_every_value_exactly():
    doc = row_kinds.__doc__ or ""
    for value in (*row_kinds.LOSS_SUMMARY_KINDS, *row_kinds.METRIC_ROW_KINDS):
        assert f"``{value}``" in doc, f"{value} is not listed in row_kinds' docstring"
    assert "runs.loss_summary_kind" in doc and "metrics.kind" in doc


def test_the_docs_list_every_value_exactly():
    docs = (SRC.parent.parent / "docs" / "backends-and-ops.md").read_text(encoding="utf-8")
    for value in (*row_kinds.LOSS_SUMMARY_KINDS, *row_kinds.METRIC_ROW_KINDS):
        assert f"`{value}`" in docs, f"{value} is not documented in docs/backends-and-ops.md"
    assert "runs.loss_summary_kind" in docs and "metrics.kind" in docs


@pytest.mark.parametrize("value", [None, *row_kinds.LOSS_SUMMARY_KINDS])
def test_check_loss_summary_kind_accepts_the_closed_set_and_none(value):
    assert row_kinds.check_loss_summary_kind(value) == value


@pytest.mark.parametrize("value", ["delta", "single", "PER_STEP", "", " mean", True, 1, ["mean"]])
def test_check_loss_summary_kind_refuses_everything_else_by_name(value):
    with pytest.raises(
        ValueError, match=r"loss_summary_kind must be one of per_step, mean, unavailable"
    ):
        row_kinds.check_loss_summary_kind(value)


@pytest.mark.parametrize("value", [None, *row_kinds.METRIC_ROW_KINDS])
def test_check_metric_row_kind_accepts_the_closed_set_and_none(value):
    assert row_kinds.check_metric_row_kind(value) == value


@pytest.mark.parametrize("value", ["val", "training", "EVAL", "", True, 0, ("eval",)])
def test_check_metric_row_kind_refuses_everything_else_by_name(value):
    with pytest.raises(ValueError, match=r"kind must be one of train, eval, summary"):
        row_kinds.check_metric_row_kind(value)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, "train"),
        ({"val_loss": None, "val_reward": None}, "train"),
        ({"val_loss": 2.5}, "eval"),
        ({"val_loss": 0.0}, "eval"),   # a measured zero is a measurement
        ({"val_reward": 15.25}, "eval"),
        ({"val_loss": 2.5, "val_reward": 1.0}, "eval"),
    ],
)
def test_metric_row_kind_says_eval_when_a_validation_number_is_present(kwargs, expected):
    assert row_kinds.metric_row_kind(**kwargs) == expected


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"initial_loss": 2.0, "final_loss": 1.0, "loss_summary_kind": "delta"}, "per_step"),
        ({"initial_loss": 1.0, "final_loss": 1.0, "loss_summary_kind": "single"}, "per_step"),
        ({"initial_loss": 3.8, "final_loss": 3.8, "loss_summary_kind": "mean"}, "mean"),
        ({"initial_loss": 0.0, "final_loss": 0.0, "loss_summary_kind": "unavailable"},
         "unavailable"),
        # No kind from the trainer: absent numbers are "unavailable"; present ones are not
        # claimed to be anything (NULL).
        ({"initial_loss": None, "final_loss": None}, "unavailable"),
        ({}, "unavailable"),
        ({"initial_loss": 2.0, "final_loss": 1.0}, None),
        ({"initial_loss": 2.0, "final_loss": 1.0, "loss_summary_kind": "bogus"}, None),
        # A kind that claims numbers the result does not have is not believed.
        ({"initial_loss": None, "final_loss": None, "loss_summary_kind": "delta"},
         "unavailable"),
    ],
)
def test_stored_loss_summary_kind_maps_the_trainers_kind_to_the_column(result, expected):
    assert stored_loss_summary_kind(result) == expected


@pytest.mark.parametrize("kind", row_kinds.LOSS_SUMMARY_KINDS)
def test_finish_run_writes_each_loss_summary_kind(tracker, kind):
    run_id = _start(tracker)
    tracker.finish_run(run_id, 2.0, 1.0, 3, 1.0, "./out", loss_summary_kind=kind)
    assert tracker.get_run(run_id)["loss_summary_kind"] == kind


def test_finish_run_without_a_kind_leaves_the_column_null(tracker):
    run_id = _start(tracker)
    tracker.finish_run(run_id, 2.0, 1.0, 3, 1.0, "./out")
    run = tracker.get_run(run_id)
    assert "loss_summary_kind" in run and run["loss_summary_kind"] is None


def test_finish_run_refuses_an_unknown_kind_and_leaves_the_run_unfinished(tracker):
    run_id = _start(tracker)
    with pytest.raises(ValueError, match="loss_summary_kind must be one of"):
        tracker.finish_run(run_id, 2.0, 1.0, 3, 1.0, "./out", loss_summary_kind="delta")
    assert tracker.get_run(run_id)["status"] != "completed"


@pytest.mark.parametrize("kind", row_kinds.METRIC_ROW_KINDS)
def test_log_metrics_writes_each_row_kind_and_get_metrics_returns_it(tracker, kind):
    run_id = _start(tracker)
    tracker.log_metrics(run_id, step=4, loss=1.0, lr=1e-4, kind=kind)
    rows = tracker.get_metrics(run_id)
    assert [row["kind"] for row in rows] == [kind]


def test_log_metrics_without_a_kind_is_null_and_unknown_is_refused(tracker):
    run_id = _start(tracker)
    tracker.log_metrics(run_id, step=1, loss=1.0)
    assert tracker.get_metrics(run_id)[0]["kind"] is None
    with pytest.raises(ValueError, match="kind must be one of train, eval, summary"):
        tracker.log_metrics(run_id, step=2, loss=1.0, kind="val")
    assert len(tracker.get_metrics(run_id)) == 1   # the refused row was not written


def _legacy_db(path: Path) -> None:
    """A database from before both columns: the LIVE schema with them stripped."""
    import sqlite3

    from soup_cli.experiment import tracker as tracker_module

    live = re.search(
        r'_SCHEMA_SQL = """(.*?)"""',
        Path(tracker_module.__file__).read_text(encoding="utf-8"), re.S,
    ).group(1)
    legacy = live.replace("    loss_summary_kind TEXT,\n", "").replace("    kind      TEXT,\n", "")
    assert "loss_summary_kind" not in legacy and "    kind      TEXT" not in legacy, (
        "the fixture must genuinely predate both columns"
    )
    conn = sqlite3.connect(str(path))
    conn.executescript(legacy)
    conn.execute(
        "INSERT INTO runs (run_id, created_at, status, config_json, initial_loss, final_loss,"
        " total_steps) VALUES ('old-run', '2026-01-01T00:00:00', 'completed', '{}', 3.0, 1.0, 3)"
    )
    for step in (1, 2, 3):
        conn.execute(
            "INSERT INTO metrics (run_id, step, epoch, loss, lr, timestamp)"
            " VALUES ('old-run', ?, ?, ?, 1e-4, '2026-01-01T00:00:00')",
            (step, step / 3, 4.0 - step),
        )
    conn.commit()
    conn.close()


def _columns(path: Path, table: str) -> set[str]:
    import sqlite3

    conn = sqlite3.connect(str(path))
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    finally:
        conn.close()


def test_an_existing_database_gains_both_columns_and_old_rows_read_null(tmp_path):
    db = tmp_path / "legacy.db"
    _legacy_db(db)
    assert "kind" not in _columns(db, "metrics")
    assert "loss_summary_kind" not in _columns(db, "runs")

    old = ExperimentTracker(db_path=db)
    try:
        assert "kind" in _columns(db, "metrics")
        assert "loss_summary_kind" in _columns(db, "runs")
        # nothing is backfilled: NULL says "written before the column existed"
        run = old.get_run("old-run")
        assert run["loss_summary_kind"] is None
        assert (run["initial_loss"], run["final_loss"]) == (3.0, 1.0)
        rows = old.get_metrics("old-run")
        assert len(rows) == 3 and all("kind" in row for row in rows)
        assert [row["kind"] for row in rows] == [None, None, None]
        assert [row["loss"] for row in rows] == [3.0, 2.0, 1.0]   # existing data survives
        # new rows on the migrated file carry kinds; the second open is a no-op
        old.log_metrics("old-run", step=4, loss=0.5, lr=1e-4, kind="train")
        old.finish_run("old-run", 3.0, 0.5, 4, 1.0, "./out", loss_summary_kind="per_step")
    finally:
        old.close()
    again = ExperimentTracker(db_path=db)
    try:
        again.init_db()
        ordered = sorted(again.get_metrics("old-run"), key=lambda row: row["id"])
        assert [row["kind"] for row in ordered] == [None, None, None, "train"]
        assert again.get_run("old-run")["loss_summary_kind"] == "per_step"
    finally:
        again.close()


def test_a_fresh_database_has_both_columns(tmp_path):
    fresh = tmp_path / "fresh.db"
    ExperimentTracker(db_path=fresh).close()
    assert "kind" in _columns(fresh, "metrics")
    assert "loss_summary_kind" in _columns(fresh, "runs")


# ---------------------------------------------------------------------------
# who writes which kind
# ---------------------------------------------------------------------------


def test_the_callback_marks_training_rows_train_and_evaluation_rows_eval(session):
    callback, tracker, run_id, _ = session
    _event(callback, 10, loss=3.0, learning_rate=1e-4)
    _event(callback, 10, eval_loss=2.5)
    _event(callback, 11, learning_rate=2e-5)
    rows = sorted(tracker.get_metrics(run_id), key=lambda row: row["id"])
    assert [row["kind"] for row in rows] == ["train", "eval", "train"]
    # #23 is untouched: the eval row still carries the last measured training values.
    assert (rows[1]["val_loss"], rows[1]["loss"], rows[1]["lr"]) == (2.5, 3.0, 1e-4)


def test_the_callback_writes_no_summary_row_so_summary_is_never_written_by_it(session):
    callback, tracker, run_id, _ = session
    _event(callback, 10, loss=3.0, learning_rate=1e-4)
    _event(callback, 10, train_runtime=3.38, train_loss=3.828)
    assert [row["kind"] for row in tracker.get_metrics(run_id)] == ["train"]


def test_the_live_event_carries_the_row_kind(session):
    from soup_cli.utils.sse_train_stream import to_payload
    from soup_cli.utils.train_event_buffer import get_global_buffer

    callback, _, _, _ = session
    _event(callback, 10, loss=3.0, learning_rate=1e-4)
    _event(callback, 10, eval_loss=2.5)
    events = get_global_buffer().snapshot()
    assert [event.kind for event in events] == ["train", "eval"]
    assert to_payload(events[1])["kind"] == "eval"


def test_a_train_event_without_a_kind_omits_it_and_a_bad_kind_is_refused():
    from soup_cli.utils.sse_train_stream import TrainEvent, to_payload

    assert "kind" not in to_payload(TrainEvent(type="metric", step=1, loss=1.0))
    assert to_payload(TrainEvent(type="metric", step=1, kind="summary"))["kind"] == "summary"
    with pytest.raises(ValueError, match="kind must be one of train, eval, summary"):
        TrainEvent(type="metric", step=1, kind="val")


def test_every_in_repo_metric_row_writer_states_its_kind():
    """A new ``log_metrics`` caller must say what its row is; NULL is for rows that
    predate the column, not for code written after it."""
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "log_metrics"
                and "kind" not in {keyword.arg for keyword in node.keywords}
            ):
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert not offenders, f"log_metrics called without kind=: {offenders}"


def test_the_http_metric_api_returns_the_row_kind_and_the_run_returns_its_summary_kind(
    tmp_path, monkeypatch,
):
    pytest.importorskip("fastapi")
    from soup_cli.ui.app import create_app, get_auth_token
    from tests.ui_helpers import ui_client

    db = tmp_path / "http.db"
    monkeypatch.setenv("SOUP_DB_PATH", str(db))
    store = ExperimentTracker(db_path=db)
    run_id = _start(store, "run_http_kind")
    store.log_metrics(run_id, step=1, loss=2.0, lr=1e-4, kind="train")
    store.log_metrics(run_id, step=1, loss=2.0, lr=1e-4, val_loss=1.5, kind="eval")
    store.finish_run(run_id, 2.0, 2.0, 1, 1.0, "./out", loss_summary_kind="per_step")
    store.close()

    client = ui_client(create_app())
    headers = {"Authorization": f"Bearer {get_auth_token()}"}
    metrics = client.get(f"/api/runs/{run_id}/metrics", headers=headers).json()["metrics"]
    assert sorted(row["kind"] for row in metrics) == ["eval", "train"]
    assert all(isinstance(row["id"], int) and row["id"] > 0 for row in metrics)
    run = client.get(f"/api/runs/{run_id}", headers=headers).json()
    assert run["loss_summary_kind"] == "per_step"
