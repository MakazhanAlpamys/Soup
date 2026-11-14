"""#2724 — `soup runs clean` when several runs share one output directory.

Training twice into `./output` records two runs at the same directory. `clean()`
used to handle one run at a time: each pass globbed every checkpoint in the
directory, kept only that run's own best, and queued the rest. So run A's pass
deleted run B's best and the other way round, every path was listed, counted
and deleted once per run (ending in "Failed to delete" warnings with exit 0),
and `runs clean B` pruned run A's best.

Each directory is now pruned once, and the checkpoints kept whole there are the
best of every run recorded at it, whether or not that run is being cleaned.
"""

import re
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.experiment.tracker import ExperimentTracker

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_CKPT_FILES = ("model.safetensors", "optimizer.pt", "scheduler.pt")
_STEPS = (5, 7, 21)
_FALLBACK = "no checkpoint has a recorded loss"
_FAILED = "Failed to delete"

# Run A's lowest loss at a checkpoint is at step 7, run B's at step 21.
_A_ROWS = [(5, 1.0), (7, 0.5), (21, 0.9)]
_B_ROWS = [(5, 1.0), (7, 0.9), (21, 0.3)]
# The rows a run shorter than `logging_steps` leaves (#1225): no loss at all.
_NO_LOSS = [(21, None)]
# Run C, in a directory of its own, has its lowest loss at step 5.
_C_ROWS = [(5, 0.2), (7, 0.6), (21, 0.4)]


def _plain(text: str) -> str:
    """Strip Rich ANSI escapes and collapse whitespace (Rich wraps too)."""
    return " ".join(_ANSI_RE.sub("", text).split())


@pytest.fixture
def shared_dir(tmp_path: Path, monkeypatch):
    """Checkpoints 5, 7 and 21 in one output directory, and a way to record runs.

    A run is recorded at the shared directory unless another one is named; that
    directory gets its own checkpoints 5, 7 and 21.
    """
    db_path = tmp_path / "experiments.db"
    monkeypatch.setenv("SOUP_DB_PATH", str(db_path))
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    # Wide enough that Rich never wraps a listed path.
    monkeypatch.setattr("soup_cli.commands.runs.console", Console(width=2000))
    out_dir = workdir / "output"
    _make_checkpoints(out_dir)
    tracker = ExperimentTracker(db_path=db_path)

    def _record(run_id, rows, run_dir=None):
        if run_dir is None:
            run_dir = out_dir
        elif not run_dir.exists():
            _make_checkpoints(run_dir)
        tracker.start_run(
            config_dict={}, device="cpu", device_name="CPU", gpu_info={}, run_id=run_id
        )
        tracker.finish_run(
            run_id=run_id,
            initial_loss=0.0,
            final_loss=0.0,
            total_steps=max(_STEPS),
            duration_secs=1.0,
            output_dir=str(run_dir),
        )
        for step, loss in rows:
            tracker.log_metrics(run_id, step=step, loss=loss)

    yield _record, out_dir
    tracker.close()


def _make_checkpoints(out_dir: Path) -> None:
    for step in _STEPS:
        ckpt = out_dir / f"checkpoint-{step}"
        ckpt.mkdir(parents=True)
        for name in _CKPT_FILES:
            (ckpt / name).write_text(f"{name} at step {step}")


def _clean(*args):
    result = runner.invoke(app, ["runs", "clean", *args])
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


def _listed(output: str) -> list[str]:
    """The paths a dry run lists, one entry per listing."""
    return re.findall(r"Delete (?:dir|file): (\S+)", output)


def _counts(output: str) -> tuple[int, int]:
    match = re.search(r"from (\d+) files and (\d+) directories", output)
    assert match, output
    return int(match.group(1)), int(match.group(2))


@pytest.fixture
def two_runs(shared_dir):
    record, out_dir = shared_dir
    record("run_A", _A_ROWS)
    record("run_B", _B_ROWS)
    return out_dir


class TestCleaningAll:
    def test_no_keep_weights_keeps_both_bests_whole(self, two_runs) -> None:
        output = _clean("--all", "--no-keep-weights", "--force")
        assert _whole(two_runs, 7)
        assert _whole(two_runs, 21)
        assert not (two_runs / "checkpoint-5").exists()
        assert _FAILED not in output

    def test_keep_weights_leaves_optimizer_and_scheduler_in_both_bests(self, two_runs) -> None:
        output = _clean("--all", "--force")
        assert _whole(two_runs, 7)
        assert _whole(two_runs, 21)
        assert _stripped(two_runs, 5)
        assert _FAILED not in output

    def test_no_keep_weights_dry_run_lists_each_directory_once(self, two_runs) -> None:
        output = _clean("--all", "--no-keep-weights", "--dry-run")
        listed = _listed(output)
        assert len(listed) == len(set(listed)), listed
        assert [Path(p).name for p in listed] == ["checkpoint-5"]
        assert _counts(output) == (0, 1)

    def test_keep_weights_dry_run_lists_each_file_once(self, two_runs) -> None:
        output = _clean("--all", "--dry-run")
        listed = _listed(output)
        assert len(listed) == len(set(listed)), listed
        assert sorted(Path(p).name for p in listed) == ["optimizer.pt", "scheduler.pt"]
        assert all(Path(p).parent.name == "checkpoint-5" for p in listed), listed
        assert _counts(output) == (2, 0)


class TestARunInAnotherDirectory:
    """The bests of the shared directory protect nothing in another directory."""

    @pytest.fixture
    def three_runs(self, shared_dir, tmp_path):
        record, out_dir = shared_dir
        record("run_A", _A_ROWS)
        record("run_B", _B_ROWS)
        other_dir = tmp_path / "work" / "other"
        record("run_C", _C_ROWS, run_dir=other_dir)
        return out_dir, other_dir

    def test_no_keep_weights_deletes_its_checkpoints_but_its_best(self, three_runs) -> None:
        out_dir, other_dir = three_runs
        output = _clean("--all", "--no-keep-weights", "--force")
        assert _whole(other_dir, 5)
        assert not (other_dir / "checkpoint-7").exists()
        assert not (other_dir / "checkpoint-21").exists()
        assert _whole(out_dir, 7)
        assert _whole(out_dir, 21)
        assert not (out_dir / "checkpoint-5").exists()
        assert _FAILED not in output

    def test_keep_weights_strips_its_checkpoints_but_its_best(self, three_runs) -> None:
        out_dir, other_dir = three_runs
        output = _clean("--all", "--force")
        assert _whole(other_dir, 5)
        assert _stripped(other_dir, 7)
        assert _stripped(other_dir, 21)
        assert _whole(out_dir, 7)
        assert _whole(out_dir, 21)
        assert _stripped(out_dir, 5)
        assert _FAILED not in output


class TestCleaningOneRun:
    @pytest.mark.parametrize(
        ("run_id", "other_best"), [("run_A", "checkpoint-21"), ("run_B", "checkpoint-7")]
    )
    @pytest.mark.parametrize("mode", ["--keep-weights", "--no-keep-weights"])
    def test_dry_run_does_not_list_the_other_runs_best(
        self, two_runs, run_id, other_best, mode
    ) -> None:
        output = _clean(run_id, mode, "--dry-run")
        listed = _listed(output)
        assert listed, output
        assert not any(other_best in Path(p).parts for p in listed), listed

    @pytest.mark.parametrize("run_id", ["run_A", "run_B"])
    def test_no_keep_weights_leaves_both_bests_whole(self, two_runs, run_id) -> None:
        output = _clean(run_id, "--no-keep-weights", "--force")
        assert _whole(two_runs, 7)
        assert _whole(two_runs, 21)
        assert not (two_runs / "checkpoint-5").exists()
        assert _FAILED not in output


class TestFallbackNote:
    def test_printed_once_for_each_run_being_cleaned(self, shared_dir) -> None:
        record, out_dir = shared_dir
        record("run_A", _NO_LOSS)
        record("run_B", _NO_LOSS)
        output = _clean("--all", "--no-keep-weights", "--force")
        assert output.count(_FALLBACK) == 2
        assert "run_A: " + _FALLBACK in output
        assert "run_B: " + _FALLBACK in output
        assert _whole(out_dir, 21)
        assert not (out_dir / "checkpoint-5").exists()
        assert not (out_dir / "checkpoint-7").exists()
        assert _FAILED not in output

    def test_not_printed_for_a_run_that_is_not_being_cleaned(self, shared_dir) -> None:
        record, out_dir = shared_dir
        record("run_A", _A_ROWS)
        record("run_B", _NO_LOSS)
        output = _clean("run_A", "--no-keep-weights", "--force")
        assert _FALLBACK not in output
        # Run B keeps its latest checkpoint; run A keeps its best.
        assert _whole(out_dir, 7)
        assert _whole(out_dir, 21)
        assert not (out_dir / "checkpoint-5").exists()
