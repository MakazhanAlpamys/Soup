"""The training working tree's startup identity is recorded, never inferred later."""

from __future__ import annotations

import io
import json
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
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


@pytest.fixture
def git_processes(monkeypatch, tmp_path):
    """Exercise bounded pipe reads and cleanup, without running a fake executable."""
    from soup_cli.utils import code_revision

    calls = []
    monkeypatch.setattr(code_revision, "find_tool", lambda name: str(tmp_path / "git"))

    def install(outputs):
        results = iter(outputs)

        def popen(argv, **kwargs):
            code, value = next(results)

            class Pipe(io.BytesIO):
                def read(self, size=-1):
                    assert 0 < size <= 66, "Git output must have a bounded read"
                    self.requests.append(size)
                    return super().read(size)

            pipe = Pipe(value.encode("utf-8"))
            pipe.requests = []

            class Process:
                stdout = pipe
                returncode = None
                killed = False

                def poll(self):
                    return self.returncode

                def wait(self, timeout=None):
                    self.returncode = -9 if self.killed else code
                    return self.returncode

                def kill(self):
                    self.killed = True

            process = Process()
            calls.append((argv, kwargs, process))
            return process

        monkeypatch.setattr(code_revision.subprocess, "Popen", popen)
        return calls

    return install


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

    monkeypatch.setattr(code_revision, "find_tool", lambda name: str(tmp_path / "git"))

    def fail(*args, **kwargs):
        assert kwargs["env"]["GIT_OPTIONAL_LOCKS"] == "0"
        raise failure

    monkeypatch.setattr(code_revision.subprocess, "Popen", fail)
    captured = code_revision.capture_code_revision(tmp_path)
    assert captured.commit is None and captured.dirty is None


@pytest.mark.parametrize("outputs", [
    [(0, "not-a-commit"), (0, ""), (0, "a" * 40)],
    [(0, "a" * 100000)],
    [(0, "a" * 40), (1, ""), (0, "a" * 40)],
    [(0, "a" * 40), (0, ""), (0, "b" * 40)],
])
def test_bad_commit_failed_status_or_moving_head_stays_unknown(git_processes, tmp_path, outputs):
    from soup_cli.utils import code_revision

    git_processes(outputs)
    captured = code_revision.capture_code_revision(tmp_path)
    assert captured.commit is None and captured.dirty is None


def test_deadline_covers_all_git_commands_not_each_command(monkeypatch, tmp_path):
    from soup_cli.utils import code_revision

    monkeypatch.setattr(code_revision, "find_tool", lambda name: str(tmp_path / "git"))
    clock = {"now": 0.0}
    deadlines = []

    def output(argv, env, deadline, limit):
        deadlines.append((deadline, deadline - clock["now"]))
        clock["now"] += 0.75
        return b"a" * 40 if limit > 1 else b" "

    monkeypatch.setattr(code_revision.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(code_revision, "_bounded_git_output", output)
    captured = code_revision.capture_code_revision(tmp_path)
    assert captured.commit is None and captured.dirty is None
    assert deadlines == [(2.0, 2.0), (2.0, 1.25), (2.0, 0.5)]


def test_sha256_git_object_id_is_not_truncated(git_processes, tmp_path):
    from soup_cli.utils import code_revision

    git_processes([(0, "a" * 64), (0, ""), (0, "a" * 64)])
    captured = code_revision.capture_code_revision(tmp_path)
    assert captured.commit == "a" * 64
    assert captured.dirty is False


def test_missing_git_does_not_start_a_subprocess(monkeypatch, tmp_path):
    from soup_cli.utils import code_revision

    monkeypatch.setattr(code_revision, "find_tool", lambda name: None)
    monkeypatch.setattr(
        code_revision.subprocess, "Popen", lambda *a, **kw: pytest.fail("Git is unavailable"),
    )
    captured = code_revision.capture_code_revision(tmp_path)
    assert captured.commit is None and captured.dirty is None


def test_git_is_resolved_from_absolute_path_entries_not_the_current_directory(
    tmp_path, monkeypatch, git_processes,
):
    from soup_cli.utils import code_revision

    monkeypatch.chdir(tmp_path)
    binary = "git.EXE" if os.name == "nt" else "git"
    (tmp_path / binary).write_text("current-directory stand-in", encoding="utf-8")
    installed = tmp_path / "bin"
    installed.mkdir()
    trusted = installed / binary
    trusted.write_text("PATH stand-in; never executed by this test", encoding="utf-8")
    trusted.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join(["", ".", str(installed)]))
    monkeypatch.setenv("PATHEXT", ".EXE")
    calls = git_processes([(0, "a" * 40), (0, ""), (0, "a" * 40)])
    # Restore the real resolver; the process double records the selected program.
    from soup_cli.utils.tool_paths import find_tool

    monkeypatch.setattr(code_revision, "find_tool", find_tool)
    assert code_revision.capture_code_revision().commit == "a" * 40
    assert all(argv[0] == str(trusted) for argv, _, _ in calls)


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


@pytest.mark.parametrize("config_location", ["startup", "other-repository"])
def test_train_records_startup_cwd_before_loading_or_writing_artifacts(
    repository, tmp_path, monkeypatch, config_location,
):
    from soup_cli.cli import app
    from soup_cli.commands import train

    (repository / "data.jsonl").write_text(
        '{"instruction":"question", "output":"example"}\n', encoding="utf-8",
    )
    config_directory = repository
    if config_location == "other-repository":
        config_directory = tmp_path / "config-project"
        config_directory.mkdir()
        _git(config_directory, "init", "-q")
    config = config_directory / "soup.yaml"
    config.write_text(
        f"base: ./base\ndata: {{train: '{(repository / 'data.jsonl').as_posix()}', "
        "format: alpaca, val_split: 0}\ntraining: {quantization: none, batch_size: 1}\n"
        f"output: '{(repository / 'trained').as_posix()}'\n",
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


def test_repository_fsmonitor_hook_is_never_executed_and_index_is_unchanged(repository):
    marker = repository / "ignored" / "hook-ran"
    marker.parent.mkdir()
    hook = marker.parent / "monitor.py"
    hook.write_text(
        "from pathlib import Path\nimport sys\n"
        f"Path({str(marker)!r}).write_text('called')\n"
        "sys.stdout.buffer.write(b'token\\0')\n",
        encoding="utf-8",
    )
    command = " ".join(shlex.quote(str(path).replace("\\", "/")) for path in (sys.executable, hook))
    _git(repository, "config", "core.fsmonitor", command)
    # Establish that this repository really would execute its configured hook.
    _git(repository, "status", "--porcelain=v1")
    assert marker.exists()
    marker.unlink()
    (repository / "code.py").write_text("uncommitted change\n", encoding="utf-8")
    index = repository / ".git" / "index"
    before = (index.read_bytes(), index.stat().st_mtime_ns)

    captured = _capture(repository)

    assert captured.commit == _git(repository, "rev-parse", "HEAD")
    assert captured.dirty is True
    assert not marker.exists()
    assert (index.read_bytes(), index.stat().st_mtime_ns) == before
    assert not (repository / ".git" / "index.lock").exists()


def test_dubious_ownership_remains_unknown_without_a_safe_directory_override(
    repository, monkeypatch,
):
    monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
    captured = _capture(repository)
    assert captured.commit is None and captured.dirty is None


@pytest.mark.parametrize("commit", [
    "a" * 39, "a" * 41, "a" * 63, "a" * 65, "g" * 40,
    "[bold]not-a-commit[/bold]", "\x1b[31m" + "a" * 40,
])
def test_database_commit_is_validated_again_before_show_and_compare(
    tmp_path, monkeypatch, commit,
):
    from soup_cli.cli import app
    from soup_cli.commands import runs

    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setattr(runs, "console", Console(width=240, force_terminal=True))
    tracker = ExperimentTracker()
    try:
        first = tracker.start_run({"base": "model"}, "cpu", "CPU", {})
        second = tracker.start_run({"base": "model"}, "cpu", "CPU", {})
        tracker.record_code_revision(second, commit="b" * 64, dirty=False)
        with sqlite3.connect(tracker.db_path) as conn:
            conn.execute(
                "UPDATE runs SET code_commit=?, code_dirty=0 WHERE run_id=?", (commit, first),
            )
    finally:
        tracker.close()
    for argv in (["runs", "show", first, "--no-plot"], ["runs", "compare", first, second]):
        result = CliRunner().invoke(app, argv)
        text = " ".join(strip_ansi(result.output).split())
        assert result.exit_code == 0, (text, result.exception)
        assert commit not in result.output
        if "show" in argv:
            assert "Code commit: unknown" in text
            assert "Code state: unknown" in text
        else:
            assert "b" * 64 in text


@pytest.mark.parametrize("filename", ["soup.yaml", "output/adapter.safetensors"])
def test_untracked_recipe_and_output_are_dirty(repository, filename):
    path = repository / filename
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"untracked")
    assert _capture(repository).dirty is True


def test_git_arguments_and_stored_fields_are_limited_to_the_two_observations(
    tmp_path, git_processes,
):
    from soup_cli.utils.code_revision import capture_code_revision

    calls = git_processes([(0, "a" * 40), (0, ""), (0, "a" * 40)])
    captured = capture_code_revision(tmp_path)
    prefix = [str(tmp_path / "git"), "-c", "core.fsmonitor=false", "-C", str(tmp_path)]
    assert [argv for argv, _, _ in calls] == [
        [*prefix, "rev-parse", "--verify", "HEAD"],
        [*prefix, "status", "--porcelain=v1", "--untracked-files=normal",
         "--ignore-submodules=none"],
        [*prefix, "rev-parse", "--verify", "HEAD"],
    ]
    assert all(kwargs["env"]["GIT_OPTIONAL_LOCKS"] == "0" for _, kwargs, _ in calls)
    assert vars(captured) == {"commit": "a" * 40, "dirty": False}
    tracker = ExperimentTracker(tmp_path / "runs.db")
    try:
        run_id = tracker.start_run({"base": "model"}, "cpu", "CPU", {})
        tracker.record_code_revision(run_id, commit=captured.commit, dirty=captured.dirty)
        with sqlite3.connect(tracker.db_path) as conn:
            columns = [row[1] for row in conn.execute("PRAGMA table_info(runs)")]
            assert [name for name in columns if name.startswith("code_")] == [
                "code_commit", "code_dirty",
            ]
            assert conn.execute(
                "SELECT code_commit, code_dirty FROM runs WHERE run_id=?", (run_id,),
            ).fetchone() == ("a" * 40, 0)
    finally:
        tracker.close()


def test_status_reads_one_byte_and_stops_without_consuming_the_file_list(tmp_path, git_processes):
    calls = git_processes([(0, "a" * 40), (0, " M " + "private-name\n" * 100000),
                          (0, "a" * 40)])
    captured = _capture(tmp_path)
    assert captured.commit == "a" * 40 and captured.dirty is True
    status = calls[1][2]
    assert status.stdout.requests == [1]
    assert status.killed


def test_blocked_status_read_is_cancelled_by_the_total_deadline(tmp_path, monkeypatch):
    from soup_cli.utils import code_revision

    class BlockedPipe:
        def read(self, size):
            assert size == 1
            assert stopped.wait(1), "the timeout must stop the child and unblock the pipe"
            return b""

        def close(self):
            pass

    class BlockedProcess:
        stdout = BlockedPipe()
        returncode = None

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9
            stopped.set()

        def wait(self, timeout=None):
            return self.returncode

    stopped = threading.Event()
    process = BlockedProcess()
    monkeypatch.setattr(code_revision.subprocess, "Popen", lambda *args, **kw: process)
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        code_revision._bounded_git_output(["unused"], {}, started + 0.03, 1)
    assert stopped.is_set()
    assert time.monotonic() - started < 0.8
