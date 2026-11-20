"""A baseline refresh must not drop a task this run could not score (#2951).

``--write-baseline`` wrote whatever map it had, so a refresh in which one task
errored replaced a complete baseline with a partial one. The next run then had
no reference for the missing task: a candidate that dropped it from 1.0 to 0.0
passed the gate, and the training callback stopped warning on it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi

SUITE = "suite.yaml"
FIRST = "first.jsonl"
SECOND = "second.jsonl"
BASELINE = "baseline.json"


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _suite(*names: str, threshold: float = 0.5) -> str:
    tasks = "".join(
        f"  - {{name: {name}, type: custom, tasks: {name}.jsonl, "
        f"scorer: exact, threshold: {threshold}}}\n"
        for name in names
    )
    return f"suite: partial\ntasks:\n{tasks}"


@pytest.fixture
def gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Two custom tasks, a generator that can regress the second one."""
    monkeypatch.chdir(tmp_path)
    for name in ("first", "second"):
        Path(f"{name}.jsonl").write_text(
            json.dumps({"prompt": name, "expected": "gold"}) + "\n", encoding="utf-8"
        )
    Path(SUITE).write_text(_suite("first", "second"), encoding="utf-8")

    def factory(model: str):
        def generate(prompt: str) -> str:
            if model == "candidate" and prompt == "second":
                return "wrong"
            return "gold"

        return generate

    monkeypatch.setattr("soup_cli.eval.quant_check.make_model_generator", factory)

    def run(model: str, *flags: str):
        return CliRunner().invoke(
            app, ["eval", "gate", "--suite", SUITE, "--model", model, *flags]
        )

    run.hide = lambda name: Path(f"{name}.jsonl").rename(f"{name}.jsonl.hidden")
    run.show = lambda name: Path(f"{name}.jsonl.hidden").rename(f"{name}.jsonl")
    return run


def test_refresh_after_a_task_error_keeps_the_complete_baseline(gate):
    assert gate("gold", "--write-baseline", BASELINE).exit_code == 0
    before = Path(BASELINE).read_bytes()
    assert json.loads(before)["scores"] == {"first": 1.0, "second": 1.0}

    gate.hide("second")
    refresh = gate("gold", "--write-baseline", BASELINE)

    assert refresh.exit_code == 2, (refresh.output, repr(refresh.exception))
    assert Path(BASELINE).read_bytes() == before
    text = _plain(refresh)
    assert "Cannot write --write-baseline: second not scored this run" in text
    assert "Wrote stamped baseline" not in text

    gate.show("second")
    regressed = gate("candidate", "--baseline", BASELINE)
    assert regressed.exit_code == 2, (regressed.output, repr(regressed.exception))
    assert "GATE FAILED (regression vs baseline)" in _plain(regressed)


def test_scorer_exception_keeps_the_complete_baseline(gate, monkeypatch):
    assert gate("gold", "--write-baseline", BASELINE).exit_code == 0
    before = Path(BASELINE).read_bytes()

    import soup_cli.eval.gate as gate_mod

    original = gate_mod._run_custom_task

    def explode(task, generate_fn):
        if task.name == "second":
            raise RuntimeError("scorer blew up")
        return original(task, generate_fn)

    monkeypatch.setattr(gate_mod, "_run_custom_task", explode)

    refresh = gate("gold", "--write-baseline", BASELINE)

    assert refresh.exit_code == 2, (refresh.output, repr(refresh.exception))
    assert Path(BASELINE).read_bytes() == before
    assert "Wrote stamped baseline" not in _plain(refresh)


def test_registry_coverage_blocks_a_fresh_partial_write(gate, tmp_path, monkeypatch):
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    from soup_cli.experiment.tracker import ExperimentTracker
    from soup_cli.registry.store import RegistryStore

    tracker = ExperimentTracker()
    run_id = tracker.start_run({"base": "m", "task": "sft"}, "cpu", "cpu", {})
    tracker.save_eval_result("m", "first", 1.0, {}, run_id=run_id)
    tracker.save_eval_result("m", "second", 1.0, {}, run_id=run_id)
    tracker.close()
    with RegistryStore() as store:
        store.push(name="base", tag="v1", base_model="m", task="sft",
                   run_id=run_id, config={})

    gate.hide("second")
    result = gate("gold", "--baseline", "registry://base:v1",
                  "--write-baseline", "new.json")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert not Path("new.json").exists()
    text = _plain(result)
    assert "second not scored this run" in text
    assert "not writing 'new.json'" in text


def test_fresh_mixed_write_still_records_the_measured_subset(gate):
    gate.hide("second")

    result = gate("gold", "--write-baseline", "fresh.json")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert json.loads(Path("fresh.json").read_text())["scores"] == {"first": 1.0}
    assert "Wrote stamped baseline" in _plain(result)


def test_empty_map_still_uses_the_writers_refusal(gate):
    """The empty-map guard routes an all-unscored refresh to the writer's refusal.

    The empty-result test in test_issue404_baseline_scorer_stamp.py uses a fresh
    output path with no prior coverage, so it does not exercise this guard.
    """
    assert gate("gold", "--write-baseline", BASELINE).exit_code == 0
    before = Path(BASELINE).read_bytes()
    gate.hide("first")
    gate.hide("second")

    result = gate("gold", "--write-baseline", BASELINE)

    assert result.exit_code == 1, (result.output, repr(result.exception))
    text = _plain(result)
    assert "refusing to write an empty baseline" in text
    assert "not scored this run" not in text
    assert Path(BASELINE).read_bytes() == before


def test_fully_measured_run_still_writes(gate):
    result = gate("gold", "--write-baseline", BASELINE)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert json.loads(Path(BASELINE).read_text())["scores"] == {
        "first": 1.0,
        "second": 1.0,
    }


def test_measured_threshold_failure_still_writes(gate):
    result = gate("candidate", "--write-baseline", BASELINE)

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert json.loads(Path(BASELINE).read_text())["scores"] == {
        "first": 1.0,
        "second": 0.0,
    }


def test_unreadable_output_file_is_replaced(gate):
    """An unreadable file covers nothing, so replacing it loses nothing."""
    Path(BASELINE).write_text("{not json", encoding="utf-8")

    result = gate("gold", "--write-baseline", BASELINE)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert json.loads(Path(BASELINE).read_text())["scores"] == {
        "first": 1.0,
        "second": 1.0,
    }


def test_task_removed_from_the_suite_does_not_block_later_writes(gate):
    assert gate("gold", "--write-baseline", BASELINE).exit_code == 0
    Path(SUITE).write_text(_suite("first"), encoding="utf-8")

    result = gate("gold", "--write-baseline", BASELINE)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert json.loads(Path(BASELINE).read_text())["scores"] == {"first": 1.0}


def test_training_gate_still_stops_after_a_mixed_refresh(gate):
    """The consumer that makes the severity visible: on main the refresh above
    truncates the file, the callback loads it, and training no longer stops on
    the regression it exists to catch."""
    from soup_cli.config.schema import EvalGateConfig
    from soup_cli.monitoring.callback import SoupTrainerCallback

    class _Display:
        def start(self, total_steps=0):
            self.total_steps = total_steps

    class _Args:
        pass

    class _State:
        def __init__(self):
            self.epoch = 0
            self.max_steps = 1
            self.global_step = 0

    class _Control:
        def __init__(self):
            self.should_training_stop = False

    # Threshold 0.0 for both: a measured 0.0 still passes the absolute check, so
    # the only signal left is the regression comparison against the baseline.
    Path(SUITE).write_text(_suite("first", "second", threshold=0.0), encoding="utf-8")
    assert gate("gold", "--write-baseline", BASELINE).exit_code == 0
    gate.hide("second")
    assert gate("gold", "--write-baseline", BASELINE).exit_code == 2
    gate.show("second")

    config = EvalGateConfig(
        enabled=True, suite=SUITE, baseline=BASELINE,
        on_regression="stop", regression_threshold=0.05,
    )
    callback = SoupTrainerCallback(_Display(), eval_gate_config=config)
    callback._gate_generate_fn = lambda prompt: "wrong" if prompt == "second" else "gold"
    args, state, control = _Args(), _State(), _Control()
    callback.on_train_begin(args, state, control)
    state.epoch = 1
    callback.on_epoch_end(args, state, control)

    assert control.should_training_stop is True
