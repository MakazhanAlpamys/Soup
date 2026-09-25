"""#1225 — `soup runs clean` keeps one checkpoint whole, even without a loss to rank by.

`runs clean` strips optimizer and scheduler state (or, with `--no-keep-weights`,
deletes whole checkpoints) from every checkpoint except the best one, which it
chose as the checkpoint at exactly the step of the run's lowest per-step loss.

Two kinds of run were left with no best checkpoint, so everything was pruned:

* a run with no recorded loss at all. The tracker used to store a fabricated
  0.0 for the Trainer's end-of-run summary, which happened to pick the final
  checkpoint; since #1225 it stores NULL, and a run shorter than
  `logging_steps` has no loss at any step;
* a run whose lowest loss fell between two checkpoints, which is usual when
  logs come every 10 steps and checkpoints every 500.

The checkpoint kept whole is now the one with the lowest recorded loss among
the checkpoints, and the latest checkpoint when none has a recorded loss.
"""

import math
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.experiment.tracker import ExperimentTracker

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_CKPT_FILES = ("model.safetensors", "optimizer.pt", "scheduler.pt")
_FALLBACK = "no checkpoint has a recorded loss"


def _plain(text: str) -> str:
    """Strip Rich ANSI escapes and collapse whitespace (Rich wraps too)."""
    return " ".join(_ANSI_RE.sub("", text).split())


@pytest.fixture
def make_run(tmp_path: Path, monkeypatch):
    """A tracked run under a temp cwd with the given checkpoints and metric rows."""
    db_path = tmp_path / "experiments.db"
    monkeypatch.setenv("SOUP_DB_PATH", str(db_path))
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    trackers = []

    def _make(checkpoint_steps, rows):
        out_dir = workdir / "output"
        for step in checkpoint_steps:
            ckpt = out_dir / f"checkpoint-{step}"
            ckpt.mkdir(parents=True)
            for name in _CKPT_FILES:
                (ckpt / name).write_text(f"{name} at step {step}")
        tracker = ExperimentTracker(db_path=db_path)
        trackers.append(tracker)
        run_id = tracker.start_run(config_dict={}, device="cpu", device_name="CPU", gpu_info={})
        tracker.finish_run(
            run_id=run_id,
            initial_loss=0.0,
            final_loss=0.0,
            total_steps=max(checkpoint_steps),
            duration_secs=1.0,
            output_dir=str(out_dir),
        )
        for step, loss in rows:
            tracker.log_metrics(run_id, step=step, loss=loss)
        return run_id, out_dir

    yield _make
    for tracker in trackers:
        tracker.close()


def _clean(run_id, *extra):
    result = runner.invoke(app, ["runs", "clean", run_id, *extra])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    return _plain(result.output)


def _whole(out_dir: Path, step: int) -> bool:
    return all((out_dir / f"checkpoint-{step}" / name).exists() for name in _CKPT_FILES)


def _stripped(out_dir: Path, step: int) -> bool:
    ckpt = out_dir / f"checkpoint-{step}"
    return (
        (ckpt / "model.safetensors").exists()
        and not (ckpt / "optimizer.pt").exists()
        and not (ckpt / "scheduler.pt").exists()
    )


# The rows a run shorter than `logging_steps` leaves: only the Trainer's
# end-of-run summary, which carries no loss.
_NO_LOSS = [(3, None)]


class TestARunWithoutARecordedLoss:
    def test_keeps_its_latest_checkpoint_whole_and_says_so(self, make_run) -> None:
        run_id, out_dir = make_run([2, 3], _NO_LOSS)
        output = _clean(run_id, "--force")
        assert _whole(out_dir, 3)
        assert _stripped(out_dir, 2)
        assert f"{_FALLBACK}; keeping the latest, checkpoint-3, whole" in output

    def test_no_keep_weights_deletes_every_checkpoint_but_the_latest(self, make_run) -> None:
        run_id, out_dir = make_run([2, 3], _NO_LOSS)
        output = _clean(run_id, "--no-keep-weights", "--force")
        assert not (out_dir / "checkpoint-2").exists()
        assert _whole(out_dir, 3)
        assert _FALLBACK in output

    def test_the_dry_run_lists_only_the_other_checkpoint(self, make_run) -> None:
        run_id, out_dir = make_run([2, 3], _NO_LOSS)
        output = _clean(run_id, "--no-keep-weights", "--dry-run")
        # Rich wraps the long temporary path, so count the entries rather
        # than parse them; the note names the one that is kept
        assert output.count("Delete dir:") == 1, output
        assert "keeping the latest, checkpoint-3, whole" in output
        assert _whole(out_dir, 2) and _whole(out_dir, 3)

    def test_a_single_checkpoint_is_left_alone(self, make_run) -> None:
        run_id, out_dir = make_run([3], _NO_LOSS)
        output = _clean(run_id, "--no-keep-weights", "--force")
        assert _whole(out_dir, 3)
        assert "No disposable checkpoint files found" in output

    def test_losses_logged_only_between_checkpoints_keep_the_latest(self, make_run) -> None:
        run_id, out_dir = make_run([5, 10], [(3, 1.5), (7, 0.9)])
        output = _clean(run_id, "--force")
        assert _whole(out_dir, 10)
        assert _stripped(out_dir, 5)
        assert _FALLBACK in output


class TestTheBestCheckpointIsChosenAmongCheckpoints:
    def test_a_lowest_loss_between_checkpoints_keeps_the_best_checkpoint(self, make_run) -> None:
        # the run's lowest loss (step 15) is not a checkpoint; checkpoint-20
        # has the lowest loss of the two checkpoints
        run_id, out_dir = make_run([10, 20], [(10, 2.0), (15, 0.5), (20, 1.0)])
        output = _clean(run_id, "--no-keep-weights", "--force")
        assert _whole(out_dir, 20)
        assert not (out_dir / "checkpoint-10").exists()
        assert _FALLBACK not in output

    def test_the_best_checkpoint_need_not_be_the_latest(self, make_run) -> None:
        run_id, out_dir = make_run([10, 20], [(10, 0.5), (20, 1.0)])
        _clean(run_id, "--force")
        assert _whole(out_dir, 10)
        assert _stripped(out_dir, 20)


class TestCheckpointToKeep:
    @pytest.mark.parametrize(
        ("rows", "steps", "expected"),
        [
            ([{"step": 10, "loss": 2.0}, {"step": 20, "loss": 1.0}], {10, 20}, (20, False)),
            ([{"step": 10, "loss": 2.0}, {"step": 15, "loss": 0.5}], {10, 20}, (10, False)),
            ([{"step": 3, "loss": None}], {2, 3}, (3, True)),
            ([], {2, 3}, (3, True)),
            ([{"step": 10, "loss": math.inf}, {"step": 20, "loss": 1.0}], {10, 20}, (20, False)),
            ([{"step": 10, "loss": math.inf}], {10, 20}, (20, True)),
            ([{"step": 10, "loss": 1.0}], set(), (None, False)),
        ],
        ids=[
            "lowest-at-a-checkpoint",
            "lowest-between-checkpoints",
            "only-a-null-row",
            "no-rows",
            "infinite-loss-is-not-best",
            "only-an-infinite-loss",
            "no-checkpoints",
        ],
    )
    def test_the_choice(self, rows, steps, expected) -> None:
        from soup_cli.commands.runs import _checkpoint_to_keep

        assert _checkpoint_to_keep(rows, steps) == expected
