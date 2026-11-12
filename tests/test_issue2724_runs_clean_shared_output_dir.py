"""#2724 — `soup runs clean` keeps every run's best checkpoint in a shared output directory.

Two runs trained into one ``output_dir`` share its ``checkpoint-*``
directories. `runs clean` handled each run alone: it chose that run's best
checkpoint and queued every other checkpoint in the directory. With ``--all``
each run queued the other run's best, so no best checkpoint survived, and a
checkpoint both runs queued was listed, counted and deleted twice (the second
attempt printed "Failed to delete"). Cleaning one run deleted the other run's
best as well.

Checkpoints are now grouped by resolved directory, the checkpoint kept for
every run recorded there is protected, and each path is handled once. A
directory with a single run is cleaned as before.
"""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.experiment.tracker import ExperimentTracker

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_ENTRY_RE = re.compile(r"Delete (file|dir):\s+(.+?)\s*$")
_SUMMARY_RE = re.compile(r"from (\d+) files and (\d+) directories")
_CKPT_FILES = ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt")
_STEPS = (5, 7, 21)
# Run A's lowest loss is at checkpoint-21, run B's at checkpoint-7.
_LOSSES_A = {5: 2.0, 7: 1.5, 21: 0.5}
_LOSSES_B = {5: 2.0, 7: 0.4, 21: 0.9}
_CKPT_5_STATE = ["checkpoint-5/optimizer.pt", "checkpoint-5/scheduler.pt"]


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch):
    """Rich wraps long paths at the terminal width; the dry-run parser wants one per line."""
    monkeypatch.setattr("soup_cli.commands.runs.console", Console(width=500))


@pytest.fixture
def tracker(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "experiments.db"
    monkeypatch.setenv("SOUP_DB_PATH", str(db_path))
    monkeypatch.chdir(tmp_path)
    tracker = ExperimentTracker(db_path=db_path)
    yield tracker
    tracker.close()


def _make_checkpoints(out_dir: Path) -> None:
    for step in _STEPS:
        ckpt = out_dir / f"checkpoint-{step}"
        ckpt.mkdir(parents=True)
        for name in _CKPT_FILES:
            (ckpt / name).write_text(f"{name} at step {step}")


def _record_run(tracker: ExperimentTracker, output_dir: str, losses: dict) -> str:
    run_id = tracker.start_run(config_dict={}, device="cpu", device_name="CPU", gpu_info={})
    for step, loss in sorted(losses.items()):
        tracker.log_metrics(run_id, step=step, loss=loss)
    tracker.finish_run(
        run_id=run_id,
        initial_loss=losses[min(losses)],
        final_loss=losses[max(losses)],
        total_steps=max(_STEPS),
        duration_secs=1.0,
        output_dir=output_dir,
    )
    return run_id


@pytest.fixture
def shared(tracker, tmp_path: Path):
    """Runs A and B recorded at one absolute output directory."""
    out_dir = tmp_path / "output"
    _make_checkpoints(out_dir)
    run_a = _record_run(tracker, str(out_dir), _LOSSES_A)
    run_b = _record_run(tracker, str(out_dir), _LOSSES_B)
    return SimpleNamespace(out_dir=out_dir, run_a=run_a, run_b=run_b)


def _clean(*args: str) -> str:
    result = runner.invoke(app, ["runs", "clean", *args])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    return _ANSI_RE.sub("", result.output)


def _listed(output: str) -> list:
    """The dry run's entries, as (kind, path, path from the checkpoint directory on)."""
    entries = []
    for line in output.splitlines():
        match = _ENTRY_RE.search(line)
        if match:
            kind, raw = match.groups()
            path = Path(raw)
            tail = path.name if kind == "dir" else f"{path.parent.name}/{path.name}"
            entries.append((kind, raw, tail))
    return entries


def _tails(output: str) -> list:
    return sorted(tail for _, _, tail in _listed(output))


def _whole(out_dir: Path, step: int) -> bool:
    return all((out_dir / f"checkpoint-{step}" / name).exists() for name in _CKPT_FILES)


def _stripped(out_dir: Path, step: int) -> bool:
    ckpt = out_dir / f"checkpoint-{step}"
    return (
        (ckpt / "adapter_model.safetensors").exists()
        and not (ckpt / "optimizer.pt").exists()
        and not (ckpt / "scheduler.pt").exists()
    )


def _untouched(out_dir: Path) -> bool:
    return all(_whole(out_dir, step) for step in _STEPS)


_FLAGS = pytest.mark.parametrize(
    ("flags", "expected"),
    [([], _CKPT_5_STATE), (["--no-keep-weights"], ["checkpoint-5"])],
    ids=["keep-weights", "no-keep-weights"],
)


class TestCleaningAllRuns:
    def test_no_keep_weights_keeps_both_bests_whole(self, shared) -> None:
        output = _clean("--all", "--no-keep-weights", "--force")
        assert "Failed to delete" not in output
        assert not (shared.out_dir / "checkpoint-5").exists()
        assert _whole(shared.out_dir, 7), "run B's best was deleted"
        assert _whole(shared.out_dir, 21), "run A's best was deleted"

    def test_keep_weights_keeps_the_state_of_both_bests(self, shared) -> None:
        output = _clean("--all", "--force")
        assert "Failed to delete" not in output
        assert _stripped(shared.out_dir, 5)
        assert _whole(shared.out_dir, 7), "run B's best lost its optimizer state"
        assert _whole(shared.out_dir, 21), "run A's best lost its optimizer state"

    @_FLAGS
    def test_the_dry_run_lists_each_path_once(self, shared, flags, expected) -> None:
        output = _clean("--all", "--dry-run", *flags)
        entries = _listed(output)
        paths = [raw for _, raw, _ in entries]
        assert len(paths) == len(set(paths)), paths
        assert _tails(output) == expected
        summary = _SUMMARY_RE.search(output)
        assert summary is not None, output
        files = sum(kind == "file" for kind, _, _ in entries)
        dirs = sum(kind == "dir" for kind, _, _ in entries)
        assert (int(summary.group(1)), int(summary.group(2))) == (files, dirs)
        assert _untouched(shared.out_dir)

    def test_differently_spelled_paths_to_one_directory_are_one_directory(
        self, tracker, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "output"
        _make_checkpoints(out_dir)
        _record_run(tracker, str(out_dir), _LOSSES_A)
        _record_run(tracker, str(out_dir / ".." / "output"), _LOSSES_B)
        output = _clean("--all", "--no-keep-weights", "--force")
        assert "Failed to delete" not in output
        assert not (out_dir / "checkpoint-5").exists()
        assert _whole(out_dir, 7) and _whole(out_dir, 21)


class TestCleaningOneRun:
    @_FLAGS
    def test_the_dry_run_leaves_the_other_runs_best_alone(self, shared, flags, expected) -> None:
        output = _clean(shared.run_a, "--dry-run", *flags)
        tails = _tails(output)
        assert not [tail for tail in tails if tail.startswith("checkpoint-7")], tails
        assert tails == expected
        assert _untouched(shared.out_dir)

    def test_no_keep_weights_keeps_the_other_runs_best(self, shared) -> None:
        output = _clean(shared.run_b, "--no-keep-weights", "--force")
        assert "Failed to delete" not in output
        assert not (shared.out_dir / "checkpoint-5").exists()
        assert _whole(shared.out_dir, 21), "run A's best was deleted"
        assert _whole(shared.out_dir, 7)

    def test_the_other_runs_latest_is_kept_when_it_has_no_recorded_loss(
        self, tracker, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "output"
        _make_checkpoints(out_dir)
        run_b = _record_run(tracker, str(out_dir), _LOSSES_B)
        # A run shorter than logging_steps records no loss; its latest is kept (#1225)
        _record_run(tracker, str(out_dir), {21: None})
        _clean(run_b, "--no-keep-weights", "--force")
        assert not (out_dir / "checkpoint-5").exists()
        assert _whole(out_dir, 7) and _whole(out_dir, 21)


class TestASingleRunDirectory:
    """A directory with one run is cleaned as before, beside a shared one."""

    @pytest.fixture
    def solo(self, shared, tracker, tmp_path: Path):
        out_dir = tmp_path / "solo"
        _make_checkpoints(out_dir)
        return SimpleNamespace(out_dir=out_dir, run=_record_run(tracker, str(out_dir), _LOSSES_B))

    def test_the_dry_run_lists_every_other_checkpoint(self, solo) -> None:
        output = _clean(solo.run, "--dry-run")
        assert _tails(output) == [
            "checkpoint-21/optimizer.pt",
            "checkpoint-21/scheduler.pt",
            *_CKPT_5_STATE,
        ]
        assert "from 4 files and 0 directories" in output

    def test_no_keep_weights_keeps_only_its_best(self, solo, shared) -> None:
        output = _clean(solo.run, "--no-keep-weights", "--force")
        assert "Failed to delete" not in output
        assert _whole(solo.out_dir, 7)
        assert not (solo.out_dir / "checkpoint-5").exists()
        assert not (solo.out_dir / "checkpoint-21").exists()
        assert _untouched(shared.out_dir)

    def test_cleaning_all_runs_keeps_only_its_best(self, solo) -> None:
        _clean("--all", "--no-keep-weights", "--force")
        assert _whole(solo.out_dir, 7)
        assert not (solo.out_dir / "checkpoint-5").exists()
        assert not (solo.out_dir / "checkpoint-21").exists()
