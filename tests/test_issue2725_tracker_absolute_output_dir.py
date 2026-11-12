"""#2725 — the tracker stores a run's output_dir as an absolute path.

With the default ``output: ./output`` the trainer hands ``finish_run`` the
relative string ``"output"``, which was stored verbatim. Every reader then
resolved it against the directory the reader ran in, not the one the run
trained in:

* ``soup runs --cwd-only`` listed every default-output run from any directory;
* ``soup runs clean`` pruned the checkpoints in the current directory's
  ``./output``, possibly another project's, choosing what to keep from a
  different run's losses;
* ``soup runs curriculum-curve`` read the current directory's history.

``finish_run`` now stores ``os.path.abspath(output_dir)``. A row recorded
before the fix still holds the relative value; the readers skip it rather
than guess, and ``clean`` says so.
"""

import json
import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands.runs import _filter_runs_by_cwd
from soup_cli.experiment.tracker import ExperimentTracker

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_CKPT_FILES = ("model.safetensors", "optimizer.pt", "scheduler.pt")
_NO_RUNS = "No runs found."
_LEGACY = "was recorded as a relative path"


def _plain(text: str) -> str:
    """Strip Rich ANSI escapes and collapse whitespace (Rich wraps too)."""
    return " ".join(_ANSI_RE.sub("", text).split())


@pytest.fixture
def db(tmp_path: Path, monkeypatch) -> Path:
    db_path = tmp_path / "experiments.db"
    monkeypatch.setenv("SOUP_DB_PATH", str(db_path))
    return db_path


def _project(root: Path, name: str) -> Path:
    """A project directory whose ./output holds checkpoint-1 and checkpoint-2."""
    project = root / name
    for step in (1, 2):
        ckpt = project / "output" / f"checkpoint-{step}"
        ckpt.mkdir(parents=True)
        for file_name in _CKPT_FILES:
            (ckpt / file_name).write_text("x" * 64, encoding="utf-8")
    return project


def _finished_run(db: Path, output_dir) -> str:
    """A completed run whose lower loss is at step 2, so clean keeps checkpoint-2."""
    tracker = ExperimentTracker(db_path=db)
    try:
        run_id = tracker.start_run(config_dict={}, device="cpu", device_name="CPU", gpu_info={})
        tracker.finish_run(
            run_id=run_id,
            initial_loss=1.0,
            final_loss=0.4,
            total_steps=2,
            duration_secs=1.0,
            output_dir=output_dir,
        )
        tracker.log_metrics(run_id, step=1, loss=0.9)
        tracker.log_metrics(run_id, step=2, loss=0.4)
    finally:
        tracker.close()
    return run_id


def _stored_output_dir(db: Path, run_id: str):
    tracker = ExperimentTracker(db_path=db)
    try:
        return tracker.get_run(run_id)["output_dir"]
    finally:
        tracker.close()


def _legacy_run(db: Path, output_dir: str) -> str:
    """A run as the tracker recorded it before #2725: output_dir stored verbatim."""
    run_id = _finished_run(db, "placeholder")
    with closing(sqlite3.connect(str(db))) as conn, conn:
        conn.execute("UPDATE runs SET output_dir = ? WHERE run_id = ?", (output_dir, run_id))
    assert _stored_output_dir(db, run_id) == output_dir
    return run_id


def _untouched(project: Path) -> bool:
    return all(
        (project / "output" / f"checkpoint-{step}" / name).exists()
        for step in (1, 2)
        for name in _CKPT_FILES
    )


def _invoke(*args: str) -> str:
    result = runner.invoke(app, ["runs", *args])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    return _plain(result.output)


class TestFinishRunStoresAnAbsolutePath:
    @pytest.mark.parametrize("given", ["output", "./output"])
    def test_a_relative_output_dir_resolves_against_the_training_directory(
        self, db, tmp_path, monkeypatch, given
    ) -> None:
        project_x = _project(tmp_path, "projX")
        monkeypatch.chdir(project_x)
        run_id = _finished_run(db, given)

        stored = _stored_output_dir(db, run_id)
        assert os.path.isabs(stored), stored
        assert stored == os.path.abspath(project_x / "output")
        assert os.path.realpath(stored) == os.path.realpath(project_x / "output")

    def test_an_absolute_output_dir_is_kept(self, db, tmp_path) -> None:
        given = str(tmp_path / "abs" / "output")
        run_id = _finished_run(db, given)
        assert _stored_output_dir(db, run_id) == os.path.abspath(given)

    @pytest.mark.parametrize("given", ["", None])
    def test_an_empty_output_dir_is_stored_as_given(
        self, db, tmp_path, monkeypatch, given
    ) -> None:
        # abspath("") is the current directory; an absent value must stay absent
        monkeypatch.chdir(tmp_path)
        run_id = _finished_run(db, given)
        assert _stored_output_dir(db, run_id) == given


class TestReadersDoNotResolveAgainstTheirOwnDirectory:
    def test_clean_from_another_project_leaves_it_untouched(
        self, db, tmp_path, monkeypatch
    ) -> None:
        project_x = _project(tmp_path, "projX")
        project_y = _project(tmp_path, "projY")
        monkeypatch.chdir(project_x)
        run_id = _finished_run(db, "output")

        monkeypatch.chdir(project_y)
        _invoke("clean", run_id, "--no-keep-weights", "--force")

        assert _untouched(project_y)
        assert _untouched(project_x)

    def test_cwd_only_from_an_unrelated_directory_lists_no_run(
        self, db, tmp_path, monkeypatch
    ) -> None:
        project_x = _project(tmp_path, "projX")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(project_x)
        _finished_run(db, "output")

        monkeypatch.chdir(elsewhere)
        assert _NO_RUNS in _invoke("--cwd-only")


class TestALegacyRelativeRowIsSkipped:
    def test_clean_skips_it_with_one_notice_and_leaves_cwd_untouched(
        self, db, tmp_path, monkeypatch
    ) -> None:
        project_y = _project(tmp_path, "projY")
        run_id = _legacy_run(db, "output")
        monkeypatch.chdir(project_y)

        output = _invoke("clean", run_id, "--no-keep-weights", "--force")

        assert _untouched(project_y)
        assert f"{run_id}: skipped; its output dir output {_LEGACY}" in output
        assert output.count(_LEGACY) == 1
        assert "Delete" not in output

    def test_dry_run_lists_nothing_from_cwd(self, db, tmp_path, monkeypatch) -> None:
        project_y = _project(tmp_path, "projY")
        run_id = _legacy_run(db, "output")
        monkeypatch.chdir(project_y)

        output = _invoke("clean", run_id, "--no-keep-weights", "--dry-run")

        assert "Delete dir:" not in output
        assert _LEGACY in output

    def test_cwd_only_does_not_claim_it(self, db, tmp_path, monkeypatch) -> None:
        project_y = _project(tmp_path, "projY")
        _legacy_run(db, "output")
        monkeypatch.chdir(project_y)

        assert _NO_RUNS in _invoke("--cwd-only")

    def test_filter_runs_by_cwd_drops_it(self, tmp_path, monkeypatch) -> None:
        project_x = _project(tmp_path, "projX")
        project_y = _project(tmp_path, "projY")
        monkeypatch.chdir(project_y)
        runs = [
            {"run_id": "legacy", "output_dir": "output"},
            {"run_id": "here", "output_dir": str(project_y / "output")},
            {"run_id": "there", "output_dir": str(project_x / "output")},
        ]

        kept = _filter_runs_by_cwd(runs, os.getcwd())

        assert [run["run_id"] for run in kept] == ["here"]

    def test_curriculum_curve_does_not_read_cwd_history(
        self, db, tmp_path, monkeypatch
    ) -> None:
        project_y = tmp_path / "projY"
        (project_y / "output").mkdir(parents=True)
        (project_y / "output" / "curriculum_history.jsonl").write_text(
            json.dumps({"step": 100, "weights": [0.5, 0.5]}) + "\n", encoding="utf-8"
        )
        run_id = _legacy_run(db, "output")
        monkeypatch.chdir(project_y)

        result = runner.invoke(app, ["runs", "curriculum-curve", run_id])

        assert result.exit_code == 1, result.output
        output = _plain(result.output)
        assert _LEGACY in output
        assert "--history" in output


class TestAnAbsoluteOutputDirUnderCwdStillWorks:
    def test_cwd_only_lists_it(self, db, tmp_path, monkeypatch) -> None:
        project_x = _project(tmp_path, "projX")
        monkeypatch.chdir(project_x)
        _finished_run(db, "output")

        output = _invoke("--cwd-only")

        assert _NO_RUNS not in output
        assert "Training Runs" in output

    def test_clean_prunes_its_checkpoints(self, db, tmp_path, monkeypatch) -> None:
        project_x = _project(tmp_path, "projX")
        monkeypatch.chdir(project_x)
        run_id = _finished_run(db, "output")

        output = _invoke("clean", run_id, "--no-keep-weights", "--force")

        assert not (project_x / "output" / "checkpoint-1").exists()
        assert all(
            (project_x / "output" / "checkpoint-2" / name).exists() for name in _CKPT_FILES
        )
        assert _LEGACY not in output

    def test_curriculum_curve_reads_its_history(self, db, tmp_path, monkeypatch) -> None:
        project_x = tmp_path / "projX"
        (project_x / "output").mkdir(parents=True)
        (project_x / "output" / "curriculum_history.jsonl").write_text(
            json.dumps({"step": 100, "weights": [0.5, 0.5]}) + "\n", encoding="utf-8"
        )
        monkeypatch.chdir(project_x)
        run_id = _finished_run(db, "output")

        output = _invoke("curriculum-curve", run_id)

        assert "B0" in output and "B1" in output
