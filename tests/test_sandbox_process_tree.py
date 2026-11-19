"""The sandbox ends every process the checked code started, not only the direct child.

``_run_sandboxed_subprocess`` backs the RLVR ``code_exec`` reward, ``soup serve``'s
python and bash tools, the agent-eval sandbox and the recipe runner. It used to call
``Popen.kill()`` on the direct child only, so a process the code started (a subprocess,
a shell's ``&``, a double fork) kept running with no time limit, kept the sandbox's output
pipes open, and on Windows kept the sandbox directory in use, so removing the directory
raised ``PermissionError`` and aborted the caller.

Every process these tests start is short-lived by construction: a helper grandchild sleeps
at most ``_HELPER_LIFETIME_SECONDS`` and each test kills the ones it recorded in a
``finally``, so a failing test cannot leave anything running.
"""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from soup_cli.trainer import rewards
from soup_cli.trainer.rewards import SandboxProcessResult, _run_sandboxed_subprocess
from soup_cli.utils import process_tree
from soup_cli.utils.process_liveness import process_is_alive
from soup_cli.utils.process_tree import ProcessTree, popen_kwargs

_HELPER_LIFETIME_SECONDS = 20
_GONE_WITHIN_SECONDS = 5.0
_posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")

_SLEEPER = f"import time; time.sleep({_HELPER_LIFETIME_SECONDS})"
_TERM_IGNORING_SLEEPER = (
    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    f"time.sleep({_HELPER_LIFETIME_SECONDS})"
)


def _is_running(pid: int) -> bool:
    if not process_is_alive(pid):
        return False
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as handle:
            return handle.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def _wait_until(predicate, timeout: float = _GONE_WITHIN_SECONDS) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _read_pid(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="ascii").strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


class _Helpers:
    """Pid files the sandboxed code writes, and the cleanup of anything left running."""

    def __init__(self, tmp_path: Path) -> None:
        self.dir = tmp_path
        self._recorded: list[Path] = []

    def pidfile(self, name: str) -> Path:
        path = self.dir / f"{name}.pid"
        self._recorded.append(path)
        return path

    def pids(self) -> list[int]:
        found = (_read_pid(path) for path in self._recorded)
        return [pid for pid in found if pid is not None]

    def kill_leftovers(self) -> None:
        for pid in self.pids():
            if _is_running(pid):
                try:
                    os.kill(pid, signal.SIGKILL if sys.platform != "win32" else signal.SIGTERM)
                except OSError:
                    pass


@pytest.fixture()
def helpers(tmp_path: Path):
    box = _Helpers(tmp_path)
    try:
        yield box
    finally:
        box.kill_leftovers()


@pytest.fixture()
def sandbox_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Where ``soup-sandbox-*`` directories are created, so leftovers can be listed."""
    root = tmp_path / "sbx"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


def _leftover_sandbox_dirs(root: Path) -> list[Path]:
    return list(root.glob("soup-sandbox-*"))


def _neutralise_first_cleanup(monkeypatch):
    """Make ``TemporaryDirectory.cleanup`` do nothing, as if the directory were held."""

    def hold(self):
        self._finalizer.detach()

    monkeypatch.setattr(tempfile.TemporaryDirectory, "cleanup", hold)


def _held_directory_rmtree(monkeypatch, refusals):
    """Make ``shutil.rmtree`` refuse the way a directory some process still holds does.

    ``refusals`` calls are refused (``None``: every call); later calls really remove the
    directory. A refused call that was asked to ignore errors returns quietly, as the real
    one does. Returns a dict whose ``calls`` counts the calls.
    """
    state = {"calls": 0}
    real = shutil.rmtree

    def refusing(path, ignore_errors=False, *args, **kwargs):
        state["calls"] += 1
        if refusals is None or state["calls"] <= refusals:
            if ignore_errors:
                return None
            raise PermissionError(13, "in use by another process")
        return real(path, ignore_errors, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", refusing)
    return state


def _run(code: str, **kwargs) -> SandboxProcessResult:
    kwargs.setdefault("timeout_seconds", 3)
    return _run_sandboxed_subprocess([sys.executable, "-c", code], None, **kwargs)


def _spawn_grandchild_code(
    pidfile: Path, program: str = _SLEEPER, tail: str = "", inherit_pipes: bool = False
) -> str:
    """Child code: start a grandchild running ``program``, record its pid, run ``tail``.

    ``inherit_pipes`` hands the grandchild the child's stdout and stderr, so it holds the
    sandbox's output pipes open the way a shell background job does.
    """
    pipes = ", stdout=sys.stdout, stderr=sys.stderr" if inherit_pipes else ""
    return (
        "import subprocess, sys, time\n"
        f"grandchild = subprocess.Popen([sys.executable, '-c', {program!r}]{pipes})\n"
        f"open({str(pidfile)!r}, 'w').write(str(grandchild.pid))\n"
        f"{tail}\n"
    )


def _assert_all_gone(helpers: _Helpers) -> None:
    pids = helpers.pids()
    assert pids, "the sandboxed code never recorded a helper pid"
    for pid in pids:
        assert _wait_until(lambda pid=pid: not _is_running(pid)), (
            f"process {pid} started by the sandboxed code is still running"
        )


class TestTimeout:
    def test_grandchild_does_not_outlive_the_timeout(self, helpers, sandbox_tmp):
        pidfile = helpers.pidfile("grandchild")
        code = _spawn_grandchild_code(pidfile, tail="time.sleep(60)")

        result = _run(code)

        assert result.timed_out is True
        _assert_all_gone(helpers)

    def test_sandbox_directory_is_removed_after_a_timeout(self, helpers, sandbox_tmp):
        pidfile = helpers.pidfile("grandchild")
        code = _spawn_grandchild_code(pidfile, tail="time.sleep(60)")

        result = _run(code)

        assert result.timed_out is True
        assert _wait_until(lambda: not _leftover_sandbox_dirs(sandbox_tmp)), (
            _leftover_sandbox_dirs(sandbox_tmp)
        )

    @_posix_only
    def test_children_that_ignore_sigterm_are_ended(self, helpers, sandbox_tmp):
        pidfile = helpers.pidfile("grandchild")
        code = (
            "import signal\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            + _spawn_grandchild_code(pidfile, _TERM_IGNORING_SLEEPER, "time.sleep(60)")
        )

        result = _run(code)

        assert result.timed_out is True
        _assert_all_gone(helpers)

    @_posix_only
    def test_double_forked_orphan_in_the_same_group_is_ended(self, helpers, sandbox_tmp):
        pidfile = helpers.pidfile("orphan")
        code = (
            "import os, time\n"
            "pid = os.fork()\n"
            "if pid == 0:\n"
            "    if os.fork() == 0:\n"
            f"        open({str(pidfile)!r}, 'w').write(str(os.getpid()))\n"
            f"        time.sleep({_HELPER_LIFETIME_SECONDS})\n"
            "    os._exit(0)\n"
            "os.waitpid(pid, 0)\n"
            "time.sleep(60)\n"
        )

        result = _run(code)

        assert result.timed_out is True
        pid = _read_pid(pidfile)
        assert pid is not None and pid != os.getpid()
        _assert_all_gone(helpers)

    @_posix_only
    def test_grandchild_of_a_shell_background_job_is_ended(self, helpers, sandbox_tmp):
        pidfile = helpers.pidfile("background")
        command = (
            f"{shlex.quote(sys.executable)} -c {shlex.quote(_SLEEPER)} & "
            f"echo $! > {shlex.quote(str(pidfile))}; sleep 60"
        )

        result = _run_sandboxed_subprocess(["/bin/sh", "-c", command], None, timeout_seconds=1)

        assert result.timed_out is True
        _assert_all_gone(helpers)

    def test_a_timeout_result_is_unchanged(self, sandbox_tmp):
        result = _run("import time; time.sleep(60)")

        assert result == SandboxProcessResult(None, "", "", timed_out=True)


class TestNormalExit:
    def test_output_and_return_code_are_unchanged(self, sandbox_tmp):
        result = _run("import sys; print('hello'); print('warn', file=sys.stderr)")

        assert result == SandboxProcessResult(0, "hello", "warn")

    def test_nonzero_exit_is_reported(self, sandbox_tmp):
        result = _run("import sys; sys.stderr.write('boom'); sys.exit(3)")

        assert (result.returncode, result.stderr, result.timed_out) == (3, "boom", False)

    def test_straggler_started_by_a_child_that_then_exits_is_ended(self, helpers, sandbox_tmp):
        pidfile = helpers.pidfile("straggler")
        code = _spawn_grandchild_code(pidfile, tail="print('done')", inherit_pipes=True)

        result = _run(code, timeout_seconds=10)

        assert (result.returncode, result.stdout, result.timed_out) == (0, "done", False)
        _assert_all_gone(helpers)

    def test_readers_do_not_wait_for_a_straggler_that_holds_the_pipes(
        self, helpers, sandbox_tmp, monkeypatch
    ):
        # The tree is ended before the readers are joined, so a straggler that holds the
        # output pipes cannot keep them from end-of-file for the length of the join.
        monkeypatch.setattr(rewards, "_READER_JOIN_SECONDS", 30.0)
        pidfile = helpers.pidfile("straggler")
        code = _spawn_grandchild_code(pidfile, tail="print('done')", inherit_pipes=True)

        started = time.monotonic()
        result = _run(code, timeout_seconds=10)
        elapsed = time.monotonic() - started

        assert result.stdout == "done"
        assert elapsed < _HELPER_LIFETIME_SECONDS / 2, elapsed
        _assert_all_gone(helpers)

    def test_sandbox_directory_is_removed_after_a_straggler(self, helpers, sandbox_tmp):
        pidfile = helpers.pidfile("straggler")
        code = _spawn_grandchild_code(pidfile, tail="print('done')")

        _run(code, timeout_seconds=10)

        assert _wait_until(lambda: not _leftover_sandbox_dirs(sandbox_tmp)), (
            _leftover_sandbox_dirs(sandbox_tmp)
        )

    def test_no_straggler_means_nothing_else_happens(self, sandbox_tmp):
        result = _run("print(40 + 2)", timeout_seconds=10)

        assert (result.returncode, result.stdout) == (0, "42")
        assert _leftover_sandbox_dirs(sandbox_tmp) == []


class TestOutputFlood:
    def test_grandchild_does_not_outlive_an_output_flood(self, helpers, sandbox_tmp):
        pidfile = helpers.pidfile("grandchild")
        code = _spawn_grandchild_code(
            pidfile, tail="sys.stdout.write('x' * 200000); sys.stdout.flush(); time.sleep(60)"
        )

        result = _run(code, timeout_seconds=10, max_output_bytes=1000)

        assert result.output_exceeded is True
        _assert_all_gone(helpers)


class TestInterruptedCaller:
    def test_an_exception_while_collecting_still_ends_the_whole_tree(
        self, helpers, sandbox_tmp, monkeypatch
    ):
        pidfile = helpers.pidfile("grandchild")
        code = _spawn_grandchild_code(pidfile, tail="time.sleep(60)")

        def interrupted(proc, tree, max_output_bytes, timeout_seconds):
            assert _wait_until(lambda: _read_pid(pidfile) is not None, 10.0)
            raise KeyboardInterrupt

        monkeypatch.setattr(rewards, "_collect_sandbox_result", interrupted)

        with pytest.raises(KeyboardInterrupt):
            _run(code)

        _assert_all_gone(helpers)
        assert _wait_until(lambda: not _leftover_sandbox_dirs(sandbox_tmp))


class TestLaunch:
    def test_a_directory_still_held_after_the_first_attempt_is_retried(
        self, sandbox_tmp, monkeypatch
    ):
        # The first removal attempt (TemporaryDirectory.cleanup) is the one a just-ended
        # process that still holds the directory defeats: it is neutralised here, and the
        # retry has to remove the directory.
        _neutralise_first_cleanup(monkeypatch)
        removed: list = []
        real_remove = rewards._remove_directory

        def recording(path, wait_seconds):
            removed.append((Path(path).parent, wait_seconds))
            real_remove(path, wait_seconds)

        monkeypatch.setattr(rewards, "_remove_directory", recording)

        _run("print('x')", timeout_seconds=10)

        assert removed == [(sandbox_tmp, rewards._DIRECTORY_RELEASE_SECONDS)]
        assert _leftover_sandbox_dirs(sandbox_tmp) == []

    def test_a_directory_that_stays_held_cannot_abort_the_caller(self, sandbox_tmp, monkeypatch):
        _neutralise_first_cleanup(monkeypatch)
        monkeypatch.setattr(rewards, "_DIRECTORY_RELEASE_SECONDS", 0.1)
        held = _held_directory_rmtree(monkeypatch, refusals=None)

        result = _run("print('x')", timeout_seconds=10)

        assert (result.returncode, result.stdout) == (0, "x")
        assert held["calls"] >= 2  # retried, then given up on
        assert len(_leftover_sandbox_dirs(sandbox_tmp)) == 1  # left behind, not raised

    def test_the_first_removal_attempt_cannot_raise(self, sandbox_tmp, monkeypatch):
        seen: dict = {}
        real = tempfile.TemporaryDirectory

        class Recording(real):
            def __init__(self, *args, **kwargs):
                seen.update(kwargs)
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(tempfile, "TemporaryDirectory", Recording)

        _run("print('x')", timeout_seconds=10)

        assert seen.get("ignore_cleanup_errors") is True

    @_posix_only
    def test_the_first_attempt_is_enough_when_nothing_holds_the_directory(
        self, sandbox_tmp, monkeypatch
    ):
        # (POSIX only: a Windows process that has just ended may still hold its directory
        # for a moment, which is what the retry is for.)
        retried: list = []
        monkeypatch.setattr(rewards, "_remove_directory", lambda path, wait: retried.append(path))

        _run("print('x')", timeout_seconds=10)

        assert retried == []
        assert _leftover_sandbox_dirs(sandbox_tmp) == []

    def test_the_first_attempt_resets_read_only_files_the_code_left_behind(self, sandbox_tmp):
        code = (
            "import os, stat\n"
            "os.mkdir('locked')\n"
            "open('locked/file.txt', 'w').write('x')\n"
            "os.chmod('locked/file.txt', stat.S_IREAD)\n"
            "os.chmod('locked', stat.S_IREAD | stat.S_IEXEC)\n"
        )

        result = _run(code, timeout_seconds=10)

        assert result.returncode == 0, result
        assert _leftover_sandbox_dirs(sandbox_tmp) == []


class TestRemoveDirectory:
    def test_removes_a_directory_with_files(self, tmp_path):
        target = tmp_path / "held"
        (target / "sub").mkdir(parents=True)
        (target / "sub" / "file.txt").write_text("x", encoding="utf-8")

        rewards._remove_directory(str(target), 1.0)

        assert not target.exists()

    def test_a_missing_directory_is_not_an_error(self, tmp_path):
        rewards._remove_directory(str(tmp_path / "never-created"), 1.0)

    def test_removal_is_retried_while_the_directory_is_held(self, tmp_path, monkeypatch):
        target = tmp_path / "held"
        target.mkdir()
        held = _held_directory_rmtree(monkeypatch, refusals=3)

        rewards._remove_directory(str(target), 5.0)

        assert not target.exists()
        assert held["calls"] == 4

    def test_a_directory_that_stays_held_is_left_behind_without_raising(
        self, tmp_path, monkeypatch
    ):
        target = tmp_path / "held"
        target.mkdir()
        held = _held_directory_rmtree(monkeypatch, refusals=None)

        rewards._remove_directory(str(target), 0.1)

        assert target.exists()
        assert held["calls"] >= 2

    def test_the_tree_is_waited_for_before_the_directory_is_removed(
        self, sandbox_tmp, monkeypatch
    ):
        # On Windows a job's processes release the sandbox directory a moment after the
        # job is terminated; the directory is removed only after the wait.
        waited: list = []
        real_wait = ProcessTree.wait_gone

        def recording(self, timeout):
            waited.append(timeout)
            return real_wait(self, timeout)

        monkeypatch.setattr(ProcessTree, "wait_gone", recording)

        _run("print('x')", timeout_seconds=10)

        assert waited == [rewards._PROCESS_TREE_GONE_SECONDS]

    @_posix_only
    def test_child_starts_as_a_session_leader(self, sandbox_tmp, monkeypatch):
        seen: dict = {}
        real_popen = subprocess.Popen

        def recording_popen(*args, **kwargs):
            seen.update(kwargs)
            return real_popen(*args, **kwargs)

        monkeypatch.setattr(subprocess, "Popen", recording_popen)

        _run("print('x')", timeout_seconds=10)

        assert seen.get("start_new_session") is True

    def test_launch_failure_is_still_reported(self, sandbox_tmp):
        result = _run_sandboxed_subprocess(
            [str(sandbox_tmp / "no-such-program-here")], None, timeout_seconds=1
        )

        assert result.launch_failed is True
        assert result.returncode is None
        assert _leftover_sandbox_dirs(sandbox_tmp) == []

    def test_a_failure_to_set_up_the_tree_does_not_leave_the_child_running(
        self, sandbox_tmp, monkeypatch
    ):
        started: list = []
        real_popen = subprocess.Popen

        def recording_popen(*args, **kwargs):
            started.append(real_popen(*args, **kwargs))
            return started[-1]

        def broken_tree(process):
            raise RuntimeError("no tree")

        monkeypatch.setattr(subprocess, "Popen", recording_popen)
        monkeypatch.setattr(rewards, "ProcessTree", broken_tree)
        try:
            with pytest.raises(RuntimeError, match="no tree"):
                _run(f"import time; time.sleep({_HELPER_LIFETIME_SECONDS})", timeout_seconds=10)

            (child,) = started
            assert child.wait(timeout=_GONE_WITHIN_SECONDS) is not None
        finally:
            for child in started:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=_GONE_WITHIN_SECONDS)
                for stream in (child.stdout, child.stderr):
                    if stream is not None:
                        stream.close()


class TestProcessTree:
    def _start(self, code: str) -> subprocess.Popen:
        return subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **popen_kwargs(),
        )

    def test_popen_kwargs_per_platform(self):
        if sys.platform == "win32":
            assert popen_kwargs() == {}
        else:
            assert popen_kwargs() == {"start_new_session": True}

    @pytest.mark.skipif(sys.platform != "win32", reason="job objects are Windows only")
    def test_child_is_held_by_a_kill_on_close_job_on_windows(self):
        proc = self._start(f"import time; time.sleep({_HELPER_LIFETIME_SECONDS})")
        tree = ProcessTree(proc)
        try:
            assert tree.has_job is True
            api = process_tree._job_api()
            limits = api.extended_limits()
            queried = api.kernel32.QueryInformationJobObject(
                tree._job,
                9,  # JobObjectExtendedLimitInformation
                api.ctypes.byref(limits),
                api.ctypes.sizeof(limits),
                None,
            )
            assert queried
            kill_on_close = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            assert limits.BasicLimitInformation.LimitFlags & kill_on_close
        finally:
            tree.kill()
            tree.wait_gone(_GONE_WITHIN_SECONDS)
            tree.close()
            proc.wait(timeout=_GONE_WITHIN_SECONDS)
            proc.stdout.close()
            proc.stderr.close()

    @pytest.mark.skipif(sys.platform != "win32", reason="job objects are Windows only")
    def test_wait_gone_reports_a_live_job_and_then_an_empty_one(self):
        proc = self._start(f"import time; time.sleep({_HELPER_LIFETIME_SECONDS})")
        tree = ProcessTree(proc)
        try:
            assert tree.wait_gone(0.2) is False

            tree.kill()

            assert tree.wait_gone(_GONE_WITHIN_SECONDS) is True
        finally:
            tree.kill()
            tree.close()
            proc.wait(timeout=_GONE_WITHIN_SECONDS)
            proc.stdout.close()
            proc.stderr.close()

    @_posix_only
    def test_wait_gone_has_nothing_to_wait_for_on_posix(self):
        proc = self._start("pass")
        tree = ProcessTree(proc)
        try:
            proc.wait(timeout=10)
            assert tree.has_job is False
            assert tree.wait_gone(0.0) is True
        finally:
            tree.close()
            proc.stdout.close()
            proc.stderr.close()

    def test_kill_is_idempotent_and_survives_an_exited_child(self):
        proc = self._start("pass")
        tree = ProcessTree(proc)
        try:
            proc.wait(timeout=10)
            tree.kill()
            tree.kill()
            assert tree.wait_gone(_GONE_WITHIN_SECONDS) is True
        finally:
            tree.close()
            tree.close()
            proc.stdout.close()
            proc.stderr.close()

    def test_kill_ends_a_grandchild_and_the_child(self, helpers):
        pidfile = helpers.pidfile("grandchild")
        proc = self._start(_spawn_grandchild_code(pidfile, tail="time.sleep(60)"))
        tree = ProcessTree(proc)
        try:
            assert _wait_until(lambda: _read_pid(pidfile) is not None, 10.0)

            tree.kill()
            proc.wait(timeout=_GONE_WITHIN_SECONDS)

            _assert_all_gone(helpers)
            assert tree.wait_gone(_GONE_WITHIN_SECONDS) is True
        finally:
            tree.kill()
            tree.close()
            proc.stdout.close()
            proc.stderr.close()
