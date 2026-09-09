"""#764 — a failure during setup left the tracker row at status='running' forever.

`soup train` registers a run in the experiment tracker before it loads the
model, and the only `except` that marked a run failed covered the training
call alone. Everything in between -- tokenizer load, model load,
quantization, LoRA attach, dataset mapping, trainer construction -- left the
row unreconcilable: on a real database, 220 of 432 runs were stuck at
'running', the oldest since 2026-03-04.

Two independent parts, both covered here:
1. The failure boundary now wraps from right after the run is registered
   through the end of the command, so any setup exception reaches
   ``fail_run`` with the exception type and message recorded.
2. ``soup train`` now records its own pid via ``mark_running`` the way the
   MCP execution path already does, so ``_reconcile_orphaned_run`` (#401)
   can rescue a SIGKILL'd or Ctrl+C'd run on the next `soup runs` the same
   way it already does for MCP-spawned runs.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")


def _write_config(tmp_path):
    (tmp_path / "data.jsonl").write_text(
        '{"instruction": "hi", "output": "hello"}\n', encoding="utf-8",
    )
    (tmp_path / "soup.yaml").write_text(
        "base: sshleifer/tiny-gpt2\n"
        "task: sft\n"
        "data: {train: data.jsonl, format: alpaca}\n"
        "training: {epochs: 1, lr: 1e-4, batch_size: 1}\n",
        encoding="utf-8",
    )


def _invoke_train(tmp_path, monkeypatch, db_path):
    from typer.testing import CliRunner

    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_DB_PATH", str(db_path))
    _write_config(tmp_path)

    return CliRunner().invoke(app, ["train", "--config", "soup.yaml", "--yes"])


def _the_run(db_path):
    from soup_cli.experiment.tracker import ExperimentTracker

    tracker = ExperimentTracker(db_path=db_path)
    runs = tracker.list_runs(limit=5)
    assert runs, "no run was registered in the tracker at all"
    return runs[0]


class TestSetupFailureReachesFailRun:
    """A setup-phase exception must land the run at 'failed', not 'running'."""

    def test_a_setup_failure_marks_the_run_failed_with_a_reason(self, tmp_path, monkeypatch):
        from soup_cli.trainer.sft import SFTTrainerWrapper

        def _raise_setup(self, dataset):
            raise RuntimeError("simulated tokenizer/model load failure")

        monkeypatch.setattr(SFTTrainerWrapper, "setup", _raise_setup)

        db_path = tmp_path / "experiments.db"
        result = _invoke_train(tmp_path, monkeypatch, db_path)

        assert result.exit_code != 0, result.output
        run = _the_run(db_path)
        assert run["status"] == "failed", run
        assert run["error_message"] is not None
        assert "RuntimeError" in run["error_message"]
        assert "simulated tokenizer/model load failure" in run["error_message"]

    def test_a_config_only_failure_before_trainer_setup_also_marks_failed(
        self, tmp_path, monkeypatch
    ):
        """The gap covered every setup step, not only trainer_wrapper.setup —
        pin one earlier in the sequence (--tracker resolution) too, so the
        fix is proven at more than the one call site it was written against."""
        from soup_cli.utils import trackers as trackers_mod

        def _raise_resolve(**kwargs):
            raise ValueError("simulated --tracker resolution failure")

        monkeypatch.setattr(trackers_mod, "resolve_report_to", _raise_resolve)

        db_path = tmp_path / "experiments.db"
        result = _invoke_train(tmp_path, monkeypatch, db_path)

        assert result.exit_code != 0, result.output
        run = _the_run(db_path)
        assert run["status"] == "failed", run
        assert run["error_message"] is not None


class TestRunRecordsItsOwnPid:
    """Without a pid, _reconcile_orphaned_run (#401) has nothing to check
    liveness against, so a SIGKILL'd `soup train` process is never rescued."""

    def test_the_run_is_registered_with_the_current_process_pid(self, tmp_path, monkeypatch):
        from soup_cli.trainer.sft import SFTTrainerWrapper

        def _raise_setup(self, dataset):
            raise RuntimeError("stop before real training - pid is what's under test")

        monkeypatch.setattr(SFTTrainerWrapper, "setup", _raise_setup)

        db_path = tmp_path / "experiments.db"
        _invoke_train(tmp_path, monkeypatch, db_path)

        run = _the_run(db_path)
        assert run["pid"] == os.getpid(), run
