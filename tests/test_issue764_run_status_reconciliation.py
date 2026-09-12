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
        fix is proven at more than the one call site it was written against.

        This site is reached via ``raise typer.Exit(...) from exc``, not a
        bare raise — asserting the real message, not just non-None, is what
        catches recording "Exit: " instead of the actual reason (review
        finding on #767: typer.Exit carries no message of its own, and the
        except here originally formatted the caught exception directly
        rather than unwrapping to __cause__)."""
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
        message = run["error_message"]
        assert "ValueError" in message, message
        assert "simulated --tracker resolution failure" in message, message


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


class TestDescribeExceptionForTracker:
    """Second review round on #767: the first fix over-corrected.

    Unwrapping to ``__cause__`` fixes the three ``typer.Exit`` sites, but
    doing it unconditionally also rewrites ordinary training-phase
    failures — a plain ``raise X from Y`` deep inside a library then
    stores Y (the mechanism) and discards X (the operator-facing reason),
    which is *less* informative than before this whole fix existed. The
    unwrap must be gated to ``typer.Exit`` specifically.
    """

    def test_a_chained_typer_exit_unwraps_to_the_real_cause(self):
        import typer

        from soup_cli.commands.train import _describe_exception_for_tracker

        try:
            try:
                raise ValueError("simulated --tracker resolution failure")
            except ValueError as exc:
                raise typer.Exit(code=2) from exc
        except typer.Exit as e:
            message = _describe_exception_for_tracker(e)

        assert "ValueError" in message
        assert "simulated --tracker resolution failure" in message
        assert "Exit:" not in message

    def test_a_bare_typer_exit_records_the_exit_code_not_a_blank_reason(self):
        """The hub-cache containment refusal raises `typer.Exit(code=1)`
        with no `from exc` at all — there is no cause to unwrap, so the
        fix has to record *something* other than the old "Exit: "."""
        import typer

        from soup_cli.commands.train import _describe_exception_for_tracker

        try:
            raise typer.Exit(code=1)
        except typer.Exit as e:
            message = _describe_exception_for_tracker(e)

        assert message != "Exit: "
        assert "1" in message

    def test_an_ordinary_chained_exception_keeps_the_outer_message(self):
        """Regression pin: a plain `raise X from Y` (not a typer.Exit) must
        keep reporting X, the operator-facing reason, not Y, the inner
        mechanism — unwrapping unconditionally inverts #764's own point,
        that a run dying at 'Loading tokenizer' should read differently
        from one that diverged at step 900."""
        from soup_cli.commands.train import _describe_exception_for_tracker

        try:
            try:
                raise OSError("permission denied: /home/u/.ssh/id_rsa")
            except OSError as exc:
                raise RuntimeError(
                    "failed to load tokenizer for meta-llama/Llama-3.1-8B"
                ) from exc
        except RuntimeError as e:
            message = _describe_exception_for_tracker(e)

        assert "RuntimeError" in message
        assert "failed to load tokenizer" in message
        assert "permission denied" not in message


class TestFailRunRedactsAndCapsTheErrorMessage:
    """error_message is stored verbatim otherwise: a token embedded in an
    exception (a failed hub download's URL, an auth header) would sit in
    plaintext in a locally-readable database, and an unbounded string from
    a runaway stack trace could bloat the runs table."""

    def test_a_token_shaped_string_is_redacted(self, tmp_path):
        from soup_cli.experiment.tracker import ExperimentTracker

        db_path = tmp_path / "experiments.db"
        tracker = ExperimentTracker(db_path=db_path)
        run_id = tracker.start_run(
            config_dict={"base": "x"}, device="cpu", device_name="cpu", gpu_info={},
        )
        tracker.fail_run(
            run_id,
            error="RuntimeError: failed to download: "
            "token=hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345",
        )

        stored = tracker.get_run(run_id)["error_message"]
        assert "hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345" not in stored
        assert "<redacted>" in stored

    def test_an_oversized_message_is_capped(self, tmp_path):
        from soup_cli.experiment.tracker import (
            _MAX_ERROR_MESSAGE_CHARS,
            ExperimentTracker,
        )

        db_path = tmp_path / "experiments.db"
        tracker = ExperimentTracker(db_path=db_path)
        run_id = tracker.start_run(
            config_dict={"base": "x"}, device="cpu", device_name="cpu", gpu_info={},
        )
        tracker.fail_run(run_id, error="x" * (_MAX_ERROR_MESSAGE_CHARS * 3))

        stored = tracker.get_run(run_id)["error_message"]
        assert len(stored) <= _MAX_ERROR_MESSAGE_CHARS + len("...(truncated)")

    def test_a_short_message_is_left_alone(self, tmp_path):
        """Control: redaction/capping must not corrupt an ordinary message."""
        from soup_cli.experiment.tracker import ExperimentTracker

        db_path = tmp_path / "experiments.db"
        tracker = ExperimentTracker(db_path=db_path)
        run_id = tracker.start_run(
            config_dict={"base": "x"}, device="cpu", device_name="cpu", gpu_info={},
        )
        tracker.fail_run(run_id, error="ValueError: bad config")

        assert tracker.get_run(run_id)["error_message"] == "ValueError: bad config"
