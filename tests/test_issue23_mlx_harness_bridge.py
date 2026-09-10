"""Run the benchmark entry point through the real MLX wrapper and Rich/SQLite bridge.

Only MLX/model loading and host measurements are faked. These tests run without
Apple Silicon; they do not establish real Metal performance or callback cadence.
"""

from __future__ import annotations

import importlib.util
import io
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from rich.file_proxy import FileProxy

from tests.test_issue634_mlx_resume import _FakeMlxModel, _install_fake_mlx


@pytest.fixture
def harness_run(monkeypatch, tmp_path):
    import soup_cli.experiment.tracker as tracker_module
    import soup_cli.monitoring.display as display_module
    import soup_cli.utils.mlx as mlx_utils
    import soup_cli.utils.train_event_buffer as event_buffer

    # Exercise detection independently of the developer/CI terminal overrides.
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("TTY_COMPATIBLE", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")

    path = Path(__file__).parents[1] / "benchmarks" / "harness" / "mlx_sft_smoke.py"
    spec = importlib.util.spec_from_file_location("mlx_sft_smoke", path)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)

    _install_fake_mlx(monkeypatch)
    core = sys.modules["mlx.core"]
    core.reset_peak_memory = lambda: None
    core.get_peak_memory = lambda: 512 * 1024**2
    monkeypatch.setattr(mlx_utils, "load_mlx_model", lambda *a, **k: (_FakeMlxModel(), object()))
    artifacts = tmp_path / "benchmark"
    artifacts.mkdir()
    monkeypatch.setattr(harness.tempfile, "mkdtemp", lambda **k: str(artifacts))
    monkeypatch.setattr(harness, "host_mem", lambda: ("50%", "0 MB"))
    clock = iter([0.0, 1.0, 10.0, 30.0, 40.0, 41.0])
    monkeypatch.setattr(harness, "time", SimpleNamespace(time=lambda: next(clock)))
    monkeypatch.setattr(sys, "argv", [str(path), harness.DEFAULT_MODEL, "10", "1"])

    state = SimpleNamespace(
        harness=harness, artifacts=artifacts, displays=[], trackers=[], reloads=[],
        events=event_buffer.TrainEventBuffer(), fail=None, emit_reports=True, rich_redirects=[],
    )
    # Isolate the real SSE buffer, rather than pretending a recorder display
    # or tracker=None suppresses it: the merged bridge gates only on display.
    monkeypatch.setattr(event_buffer, "_GLOBAL_BUFFER", state.events)
    original_display = display_module.TrainingDisplay
    original_tracker = tracker_module.ExperimentTracker
    state.display_class = original_display
    state.tracker_class = original_tracker
    monkeypatch.setattr(display_module, "console", Console(file=io.StringIO(), width=160))

    def display_factory(*args, **kwargs):
        display = original_display(*args, **kwargs)
        state.displays.append(display)
        return display

    def tracker_factory(*args, **kwargs):
        tracker = original_tracker(*args, **kwargs)
        state.trackers.append(tracker)
        return tracker

    monkeypatch.setattr(display_module, "TrainingDisplay", display_factory)
    monkeypatch.setattr(tracker_module, "ExperimentTracker", tracker_factory)

    def train(**kwargs):
        for step, loss, speed, tokens in [(5, 3.639, 0.432, 222), (10, 1.976, 5.481, 454)]:
            state.rich_redirects.append(isinstance(sys.stdout, FileProxy))
            # Match mlx-lm's stdout independently of its callback. Moving train
            # outside the tee loses this counter despite a working bridge.
            sys.stdout.write(
                f"Iter {step}: Train loss {loss}, Learning Rate 1e-4, "
                f"It/sec {speed}, Tokens/sec 254.313, Trained Tokens {tokens}, Peak mem 0.497 GB\n"
            )
            if state.emit_reports:
                kwargs["training_callback"].on_train_loss_report({
                    "iteration": step, "train_loss": loss, "learning_rate": 1e-4,
                    "iterations_per_second": speed, "tokens_per_second": 254.313,
                    "trained_tokens": tokens, "peak_memory": 0.497,
                })
            if state.fail:
                raise state.fail
        Path(kwargs["args"].adapter_file).write_bytes(b"fake adapter")

    sys.modules["mlx_lm.tuner.trainer"].train = train
    sys.modules["mlx_lm"].load = lambda *a, **k: state.reloads.append((a, k))
    return state


@pytest.mark.parametrize(
    ("terminal_width", "force_terminal"),
    [(None, False), *((width, True) for width in range(20, 201)), (80, None)],
)
def test_harness_drives_rich_and_persists_metrics_without_losing_throughput(
    harness_run, capsys, monkeypatch, terminal_width, force_terminal,
):
    state = harness_run
    if terminal_width is not None:
        import soup_cli.monitoring.display as display_module

        if force_terminal is None:
            monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        monkeypatch.setattr(
            display_module, "console", Console(force_terminal=force_terminal, width=terminal_width),
        )
    assert state.harness.main() == 0
    assert state.rich_redirects == [terminal_width is not None] * 2

    display, = state.displays
    assert (display.current_step, display.total_steps) == (10, 10)
    assert display.loss == pytest.approx(1.976)
    assert display.speed == pytest.approx(5.481)  # it/s, not the 254.313 tok/s
    assert display._live is None

    tracker, = state.trackers
    assert tracker.db_path == state.artifacts / "experiments.db"
    assert tracker._conn is None
    with sqlite3.connect(tracker.db_path) as conn:
        run_id, status, total_steps = conn.execute(
            "SELECT run_id, status, total_steps FROM runs"
        ).fetchone()
        metrics = conn.execute(
            "SELECT step, loss, speed FROM metrics WHERE run_id = ? ORDER BY step", (run_id,)
        ).fetchall()
    assert (status, total_steps) == ("completed", 10)
    assert metrics == [(5, 3.639, 0.432), (10, 1.976, 5.481)]
    assert [event.step for event in state.events.snapshot()] == [5, 10]
    output = capsys.readouterr().out
    assert "454 trained tokens / 20.0s, whole-run average" in output
    assert "22.7 tok/s" in output
    assert state.reloads == [((state.harness.DEFAULT_MODEL,), {
        "adapter_path": str(state.artifacts / "out"),
    })]


@pytest.mark.parametrize("failure", [RuntimeError("training failed"), KeyboardInterrupt()])
def test_harness_closes_display_and_records_failed_run(harness_run, failure):
    state = harness_run
    state.fail = failure
    with pytest.raises(type(failure)):
        state.harness.main()

    display, = state.displays
    tracker, = state.trackers
    assert display._live is None
    assert tracker._conn is None
    with sqlite3.connect(tracker.db_path) as conn:
        assert conn.execute("SELECT status FROM runs").fetchone() == ("failed",)
    assert state.reloads == []


def test_harness_refuses_success_when_the_bridge_receives_no_reports(harness_run):
    state = harness_run
    state.emit_reports = False
    with pytest.raises(RuntimeError, match="bridge"):
        state.harness.main()
    tracker, = state.trackers
    assert tracker._conn is None
    with sqlite3.connect(tracker.db_path) as conn:
        assert conn.execute("SELECT status FROM runs").fetchone() == ("failed",)


@pytest.mark.parametrize("sink", ["display", "tracker"])
def test_harness_refuses_success_when_one_bridge_sink_is_disconnected(
    harness_run, monkeypatch, sink,
):
    if sink == "display":
        monkeypatch.setattr(harness_run.display_class, "update", lambda *a, **k: None)
    else:
        monkeypatch.setattr(harness_run.tracker_class, "log_metrics", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="bridge"):
        harness_run.harness.main()


@pytest.mark.parametrize("is_terminal", [True, False])
def test_tee_preserves_rich_terminal_detection(harness_run, is_terminal):
    class Output(io.StringIO):
        def isatty(self):
            return is_terminal

    output = Output()
    buffer = io.StringIO()
    tee = harness_run.harness._Tee(output, buffer)
    assert Console(file=tee, force_terminal=None).is_terminal is is_terminal


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("no cumulative counter\nTokens/sec 254.313\n", None),
        ("Trained Tokens 0\n", 0),
        ("Trained Tokens 222\nTrained Tokens 454\n", 454),
        # The final report, not the earlier matching count, must win.
        ("Trained Tokens 222\nTrained\nTokens\t  454\n", 454),
        ("Trained Tokens 222\nTrained\x1b[0m Tokens \x1b[32m454\x1b[0m\n", 454),
        ("Trained Tokens 999\nTrained\x1b[2K\nTokens\r\n\x1b[32m1000\x1b[0m\n", 1000),
        ("Trained Tokens 222\nTrained \x1b[0m \nTokens \x1b[32m 454\n", 454),
    ],
)
def test_token_parser_recovers_the_last_report(harness_run, stdout, expected):
    assert harness_run.harness._final_trained_tokens(stdout) == expected


def test_summary_prints_a_copyable_path_before_tracker_close(harness_run, monkeypatch, capsys):
    state = harness_run
    monkeypatch.setenv("COLUMNS", "20")
    original_print = Console.print
    summary_connections = []

    def record_print(console, *args, **kwargs):
        if args and isinstance(args[0], str) and args[0].startswith("bridge        :"):
            summary_connections.append(state.trackers[0]._conn is not None)
        return original_print(console, *args, **kwargs)

    monkeypatch.setattr(Console, "print", record_print)
    assert state.harness.main() == 0
    summary = f"bridge        : 2 metric reports saved to {state.artifacts / 'experiments.db'}"
    assert summary in capsys.readouterr().out.splitlines()
    assert summary_connections == [True]
    assert state.trackers[0]._conn is None
