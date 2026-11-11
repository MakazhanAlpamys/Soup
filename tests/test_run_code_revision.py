"""The training working tree's startup identity is recorded, never inferred later."""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from soup_cli.experiment.tracker import ExperimentTracker
from tests.cli_runner import CliRunner
from tests.conftest import strip_ansi


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True,
        text=True, encoding="utf-8", timeout=10,
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path: Path, monkeypatch) -> Path:
    if shutil.which("git") is None:
        pytest.skip("Git executable unavailable")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "no-global-config"))
    root = tmp_path / "project"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.invalid")
    (root / "code.py").write_text("original\n", encoding="utf-8")
    (root / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "initial")
    return root


def _capture(root: Path):
    from soup_cli.utils.code_revision import capture_code_revision

    return capture_code_revision(root)


def test_clean_subdirectory_and_detached_head(repository):
    expected = _git(repository, "rev-parse", "HEAD")
    nested = repository / "ignored" / "subdirectory"
    nested.mkdir(parents=True)
    assert _capture(nested).commit == expected
    assert _capture(nested).dirty is False
    _git(repository, "checkout", "--detach", "-q")
    assert _capture(repository).commit == expected
    assert _capture(repository).dirty is False


@pytest.mark.parametrize("change", ["unstaged", "staged", "untracked"])
def test_all_uncommitted_change_types_are_dirty(repository, change):
    expected = _git(repository, "rev-parse", "HEAD")
    changed = repository / ("new.py" if change == "untracked" else "code.py")
    changed.write_text("changed\n", encoding="utf-8")
    if change == "staged":
        _git(repository, "add", "code.py")
    captured = _capture(repository)
    assert captured.commit == expected
    assert captured.dirty is True


def test_worktree_is_its_own_checkout(repository, tmp_path):
    linked = tmp_path / "linked"
    _git(repository, "worktree", "add", "--detach", str(linked), "HEAD")
    (repository / "code.py").write_text("other checkout dirty\n", encoding="utf-8")
    assert _capture(linked).commit == _git(linked, "rev-parse", "HEAD")
    assert _capture(linked).dirty is False
    assert _capture(repository).dirty is True


def test_git_routing_environment_does_not_substitute_another_checkout(repository, monkeypatch):
    (repository / "code.py").write_text("dirty\n", encoding="utf-8")
    monkeypatch.setenv("GIT_WORK_TREE", str(repository / "ignored"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(repository / "not-the-index"))
    captured = _capture(repository)
    assert captured.commit == _git(repository, "rev-parse", "HEAD")
    assert captured.dirty is True
    assert not (repository / "not-the-index").exists()


def test_non_repo_and_unborn_head_are_unknown(repository, tmp_path):
    assert _capture(tmp_path).commit is None
    assert _capture(tmp_path).dirty is None
    unborn = tmp_path / "unborn"
    unborn.mkdir()
    _git(unborn, "init", "-q")
    assert _capture(unborn).commit is None
    assert _capture(unborn).dirty is None


@pytest.mark.parametrize("failure", [FileNotFoundError("git"),
                                     subprocess.TimeoutExpired("git", 2)])
def test_probe_failure_does_not_abort_training(monkeypatch, tmp_path, failure):
    from soup_cli.utils import code_revision

    def fail(*args, **kwargs):
        assert 0 < kwargs["timeout"] <= 2
        assert kwargs["env"]["GIT_OPTIONAL_LOCKS"] == "0"
        raise failure

    monkeypatch.setattr(code_revision.subprocess, "run", fail)
    captured = code_revision.capture_code_revision(tmp_path)
    assert captured.commit is None and captured.dirty is None


@pytest.mark.parametrize("outputs", [
    [(0, "not-a-commit"), (0, ""), (0, "a" * 40)],
    [(0, "a" * 40), (1, ""), (0, "a" * 40)],
    [(0, "a" * 40), (0, ""), (0, "b" * 40)],
])
def test_bad_commit_failed_status_or_moving_head_stays_unknown(monkeypatch, tmp_path, outputs):
    from soup_cli.utils import code_revision

    results = iter(outputs)

    def run(argv, **kwargs):
        code, value = next(results)
        if code and kwargs["check"]:
            raise subprocess.CalledProcessError(code, argv)
        return subprocess.CompletedProcess(argv, code, value, "")

    monkeypatch.setattr(code_revision.subprocess, "run", run)
    captured = code_revision.capture_code_revision(tmp_path)
    assert captured.commit is None and captured.dirty is None


def test_deadline_covers_all_git_commands_not_each_command(monkeypatch, tmp_path):
    from soup_cli.utils import code_revision

    clock = {"now": 0.0}
    timeouts = []

    def run(argv, **kwargs):
        timeouts.append(kwargs["timeout"])
        clock["now"] += 0.75
        return subprocess.CompletedProcess(argv, 0, "a" * 40, "")

    monkeypatch.setattr(code_revision.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(code_revision.subprocess, "run", run)
    captured = code_revision.capture_code_revision(tmp_path)
    assert captured.commit is None and captured.dirty is None
    assert timeouts == [2.0, 1.25, 0.5]


def test_sha256_git_object_id_is_not_truncated(monkeypatch, tmp_path):
    from soup_cli.utils import code_revision

    outputs = iter(["a" * 64, "", "a" * 64])
    monkeypatch.setattr(
        code_revision.subprocess, "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 0, next(outputs), ""),
    )
    captured = code_revision.capture_code_revision(tmp_path)
    assert captured.commit == "a" * 64
    assert captured.dirty is False


def test_legacy_database_is_upgraded_without_guessing_a_historical_commit(tmp_path):
    from soup_cli.experiment.tracker import _SCHEMA_SQL

    db = tmp_path / "legacy.db"
    legacy = "\n".join(
        line for line in _SCHEMA_SQL.splitlines()
        if "code_commit" not in line and "code_dirty" not in line
    )
    with sqlite3.connect(db) as conn:
        conn.executescript(legacy)
        conn.execute(
            "INSERT INTO runs(run_id, created_at, config_json) VALUES ('old', '2020', '{}')",
        )
    for _ in range(2):
        tracker = ExperimentTracker(db)
        try:
            row = tracker.get_run("old")
            assert row["code_commit"] is None
            assert row["code_dirty"] is None
        finally:
            tracker.close()


def test_recorded_revision_persists_and_restarted_run_clears_stale_values(tmp_path):
    db = tmp_path / "runs.db"
    tracker = ExperimentTracker(db)
    run_id = tracker.start_run({"base": "model"}, "cpu", "CPU", {})
    tracker.record_code_revision(run_id, commit="a" * 40, dirty=True)
    tracker.close()
    tracker = ExperimentTracker(db)
    try:
        assert tracker.get_run(run_id)["code_commit"] == "a" * 40
        assert tracker.get_run(run_id)["code_dirty"] == 1
        tracker.start_run({"base": "model"}, "cpu", "CPU", {}, run_id=run_id)
        assert tracker.get_run(run_id)["code_commit"] is None
        assert tracker.get_run(run_id)["code_dirty"] is None
    finally:
        tracker.close()


@pytest.mark.parametrize("commit,dirty", [("not-a-sha", True), (True, False),
                                         ("a" * 40, "false"), (None, False)])
def test_malformed_provenance_is_not_saved(tmp_path, commit, dirty):
    tracker = ExperimentTracker(tmp_path / "runs.db")
    try:
        run_id = tracker.start_run({"base": "model"}, "cpu", "CPU", {})
        with pytest.raises(ValueError):
            tracker.record_code_revision(run_id, commit=commit, dirty=dirty)
        assert tracker.get_run(run_id)["code_commit"] is None
        assert tracker.get_run(run_id)["code_dirty"] is None
    finally:
        tracker.close()


def test_show_and_compare_display_stored_values_even_outside_the_repository(tmp_path, monkeypatch):
    from soup_cli.cli import app
    from soup_cli.commands import runs

    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setattr(runs, "console", Console(width=200))
    tracker = ExperimentTracker()
    first = tracker.start_run({"base": "model"}, "cpu", "CPU", {})
    second = tracker.start_run({"base": "model"}, "cpu", "CPU", {})
    tracker.record_code_revision(first, commit="a" * 40, dirty=False)
    tracker.record_code_revision(second, commit="b" * 40, dirty=True)
    tracker.close()
    monkeypatch.chdir(tmp_path)
    for argv in (["runs", "show", first, "--no-plot"], ["runs", "compare", first, second]):
        result = CliRunner().invoke(app, argv)
        assert result.exit_code == 0, (result.output, result.exception)
        text = strip_ansi(result.output)
        assert "a" * 40 in text
        assert "clean" in text
        if "compare" in argv:
            assert "b" * 40 in text and "dirty" in text


def test_old_run_displays_unknown_rather_than_the_current_checkout(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "runs.db"))
    tracker = ExperimentTracker()
    run_id = tracker.start_run({"base": "model"}, "cpu", "CPU", {})
    tracker.close()
    result = CliRunner().invoke(app, ["runs", "show", run_id, "--no-plot"])
    text = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 0, (text, result.exception)
    assert "Code commit: unknown" in text
    assert "Code state: unknown" in text


def test_train_records_startup_cwd_before_loading_or_writing_artifacts(
    repository, tmp_path, monkeypatch,
):
    from soup_cli.cli import app
    from soup_cli.commands import train

    (repository / "data.jsonl").write_text(
        '{"instruction":"question", "output":"example"}\n', encoding="utf-8",
    )
    config = repository / "soup.yaml"
    config.write_text(
        "base: ./base\ndata: {train: data.jsonl, format: alpaca, val_split: 0}\n"
        "training: {quantization: none, batch_size: 1}\noutput: ./trained\n",
        encoding="utf-8",
    )
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "recipe")
    expected = _git(repository, "rev-parse", "HEAD")
    monkeypatch.chdir(repository)
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("SOUP_NO_TELEMETRY", "1")
    monkeypatch.setattr(train, "detect_device", lambda **kw: ("cpu", "CPU"))
    monkeypatch.setattr(train, "get_gpu_info", lambda **kw: {"memory_total": "N/A"})

    def load(*args, **kwargs):
        (repository / "code.py").write_text("changed during loading\n", encoding="utf-8")
        _git(repository, "commit", "-qam", "loader side effect")
        return {"train": [{"text": "example"}]}

    def run(**kwargs):
        output = repository / "trained"
        output.mkdir()
        (output / "adapter_model.safetensors").write_bytes(b"test artifact, not a model")
        return dict(initial_loss=1.0, final_loss=0.5, total_steps=1, duration_secs=0.1,
                    output_dir=str(output))

    monkeypatch.setattr(train, "load_dataset", load)
    monkeypatch.setattr(
        "soup_cli.trainer.dispatch.build_trainer",
        lambda *args, **kw: SimpleNamespace(setup=lambda data: None, train=run),
    )
    result = CliRunner().invoke(app, ["train", "--config", str(config), "--yes"])
    assert result.exit_code == 0, (result.output, result.exception)
    tracker = ExperimentTracker()
    try:
        recorded = tracker.list_runs()[0]
        assert recorded["status"] == "completed"
        assert recorded["code_commit"] == expected
        assert recorded["code_dirty"] == 0
        assert _git(repository, "rev-parse", "HEAD") != expected
        assert json.loads(recorded["config_json"])["base"] == "./base"
        assert (repository / "trained" / "adapter_model.safetensors").exists()
    finally:
        tracker.close()
