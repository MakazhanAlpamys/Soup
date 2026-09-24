"""In-memory plan capabilities and isolated MCP command execution."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import stat
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from soup_cli.experiment.tracker import ExperimentTracker, generate_run_id
from soup_cli.utils.paths import (
    atomic_write_text,
    enforce_under_cwd_and_no_symlink,
    open_no_follow,
)
from soup_cli.utils.process_liveness import process_is_alive as _pid_is_alive

logger = logging.getLogger(__name__)

TOKEN_TTL_SECONDS = 5 * 60
DEFAULT_MAX_TREE_FILES = 10_000
DEFAULT_MAX_TREE_BYTES = 10 * 1024 * 1024 * 1024  # 10 GiB

# The one-active-execution cap is gated on this persisted status + a liveness
# check so it survives a server restart (issue #402): the in-memory slot resets
# to None on restart, but a child launched by a prior server is still recorded
# as _STATUS_RUNNING. A recorded run only counts as active while its pid is
# alive, so a stale record whose process is gone never blocks execution forever.
_STATUS_RUNNING = "running"
# launch_run() inserts this status with pid=NULL before Popen(); mark_running()
# only upgrades to _STATUS_RUNNING (with a pid) once Popen() returns. A crash
# in between leaves a row stuck here with no pid to check liveness against.
_STATUS_LAUNCHING = "launching"


class ExecutionError(ValueError):
    """Safe execution failure intended for translation to an MCP tool error."""


@dataclass(frozen=True)
class ProtectedFile:
    path: str
    digest: str


# Digest recorded for a planned input that did not exist at plan time (a hub id,
# a built-in reward name, a file the run would create). A sha256 hex digest can
# never equal it, and revalidation requires the path to still be absent.
ABSENT_DIGEST = "absent"


def absent_marker(path: str, field: str) -> ProtectedFile:
    """Record that planned input ``field`` at ``path`` does not exist."""
    del field  # kept for signature symmetry with digest_file
    return ProtectedFile(path=os.path.realpath(path), digest=ABSENT_DIGEST)


@dataclass
class PendingPlan:
    token: str
    kind: str
    argv: tuple[str, ...]
    display_command: str
    cwd: str
    protected_files: tuple[ProtectedFile, ...]
    created_at: float
    expires_at: float
    run_id: str = ""
    consumed: bool = False


def _open_binary_no_follow(path: str):
    """Open ``path`` for binary reading, refusing a symlink at open time.

    ``enforce_under_cwd_and_no_symlink`` is an ``lstat`` check, so a symlink
    swapped in between that check and the read would silently redirect the
    digest — the TOCTOU window the plan/execute split exists to close.

    This delegates to :func:`soup_cli.utils.paths.open_no_follow` (#820) rather
    than passing ``O_NOFOLLOW`` itself. The patch-release backport could not:
    that helper landed after the tag the release was cut from, so the
    released copy carries a bare flag and is therefore UNGUARDED ON WINDOWS,
    where ``os.O_NOFOLLOW`` does not exist. The shared helper closes that half
    with a pre-open ``lstat`` and a post-open ``fstat`` cross-check, so this
    port is strictly stronger than what shipped.

    ``O_BINARY`` (Windows-only) keeps the read free of CRLF translation, so a
    digest is the same on every platform.

    ``OSError`` propagates to ``digest_file``'s handler, which maps it to the
    path-free ``ExecutionError``.
    """
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    return os.fdopen(open_no_follow(path, flags), "rb")


def digest_file(
    path: str,
    field: str,
    *,
    max_files: int = DEFAULT_MAX_TREE_FILES,
    max_bytes: int = DEFAULT_MAX_TREE_BYTES,
) -> ProtectedFile:
    """Return a cwd-contained, non-symlink file or directory tree's canonical path and digest."""
    try:
        enforce_under_cwd_and_no_symlink(path, field)
        real = os.path.realpath(path)
        if os.path.isdir(real):
            entries: list[tuple[str, str]] = []
            total_files = 0
            total_bytes = 0

            for root, dirs, files in os.walk(real, topdown=True, followlinks=False):
                for d in dirs:
                    dir_path = os.path.join(root, d)
                    enforce_under_cwd_and_no_symlink(dir_path, field)

                for f in files:
                    file_path = os.path.join(root, f)
                    enforce_under_cwd_and_no_symlink(file_path, field)
                    st = os.lstat(file_path)
                    if not stat.S_ISREG(st.st_mode):
                        raise ExecutionError(f"{field} contains non-regular file: {file_path}")
                    total_files += 1
                    if total_files > max_files:
                        raise ExecutionError(
                            f"{field} exceeds maximum file count limit ({max_files})"
                        )

                    rel_path = os.path.relpath(file_path, real)
                    rel_posix = Path(rel_path).as_posix()

                    file_hasher = hashlib.sha256()
                    with _open_binary_no_follow(file_path) as handle:
                        while chunk := handle.read(65536):
                            total_bytes += len(chunk)
                            if total_bytes > max_bytes:
                                raise ExecutionError(
                                    f"{field} exceeds maximum byte limit ({max_bytes} bytes)"
                                )
                            file_hasher.update(chunk)
                    entries.append((rel_posix, file_hasher.hexdigest()))

            # Sort entries deterministically by relative path
            entries.sort(key=lambda item: item[0])
            tree_hasher = hashlib.sha256()
            for rel_posix, f_digest in entries:
                tree_hasher.update(f"{rel_posix}\0{f_digest}\n".encode("utf-8"))
            digest = tree_hasher.hexdigest()
        else:
            st = os.lstat(real)
            if not stat.S_ISREG(st.st_mode):
                raise ExecutionError(f"{field} is not a regular file")
            hasher = hashlib.sha256()
            total_bytes = 0
            # Open ``path`` AS GIVEN, never ``real``: ``os.path.realpath``
            # RESOLVES a symlink, so opening the resolved path means
            # ``O_NOFOLLOW`` can never fire and a link swapped in at ``path``
            # after the lstat guard would be silently followed and its target
            # digested — the very TOCTOU window this reader exists to close.
            # The digest of an ordinary file is unaffected (same inode, same
            # bytes), and the RESOLVED path is still what gets recorded below,
            # because that is what ``_revalidate`` compares against.
            with _open_binary_no_follow(path) as handle:
                while chunk := handle.read(65536):
                    total_bytes += len(chunk)
                    if total_bytes > max_bytes:
                        raise ExecutionError(
                            f"{field} exceeds maximum byte limit ({max_bytes} bytes)"
                        )
                    hasher.update(chunk)
            digest = hasher.hexdigest()
    except ExecutionError:
        raise
    except (OSError, ValueError) as exc:
        raise ExecutionError(f"{field} is unavailable for execution") from exc
    return ProtectedFile(path=real, digest=digest)


class ExecutionManager:
    """Per-stdio-server plan store, one-job cap, and subprocess watcher."""

    def __init__(self, *, cwd: str | None = None, ttl_seconds: int = TOKEN_TTL_SECONDS) -> None:
        self.cwd = os.path.realpath(cwd or os.getcwd())
        self.ttl_seconds = ttl_seconds
        self._plans: dict[str, PendingPlan] = {}
        self._active_run_id: str | None = None
        self._lock = threading.Lock()

    def allocate_run_id(self) -> str:
        return generate_run_id()

    def snapshot_config(self, run_id: str, content: str) -> str:
        """Write the exact validated config content to .soup/mcp-runs/<run_id>/config.yaml."""
        run_dir = Path(self.cwd) / ".soup" / "mcp-runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        snapshot_file = str(run_dir / "config.yaml")
        atomic_write_text(content, snapshot_file, field="config snapshot")
        return os.path.realpath(snapshot_file)

    def issue(
        self,
        *,
        kind: str,
        argv: list[str],
        display_command: str,
        protected_files: tuple[ProtectedFile, ...] = (),
        run_id: str | None = None,
    ) -> str:
        now = time.monotonic()
        token = secrets.token_urlsafe(32)
        allocated_run_id = run_id or generate_run_id()
        with self._lock:
            self._cleanup(now)
            self._plans[token] = PendingPlan(
                token=token,
                kind=kind,
                argv=tuple(argv),
                display_command=display_command,
                cwd=self.cwd,
                protected_files=protected_files,
                created_at=now,
                expires_at=now + self.ttl_seconds,
                run_id=allocated_run_id,
            )
        return token

    def _live_persisted_run(self) -> str | None:
        """Return the run_id of a persisted run that may still be active.

        This is what makes the one-active-execution cap survive a server
        restart (issue #402): the tracker records every MCP-launched child, so a
        freshly-started server (empty in-memory slot) still sees a prior child
        that is genuinely training. A _STATUS_RUNNING record only counts while
        its pid is alive, since a stale one whose process is gone must not
        permanently block execution. A _STATUS_LAUNCHING record carries no pid
        to check, so it blocks unconditionally rather than being read as free
        capacity (issue #505): the crash window it represents can leave
        a real child running with nothing to verify it against.
        Returns None (never raises) if the tracker is unreadable, so a tracker
        problem degrades to the in-memory-only behaviour rather than wedging.
        """
        try:
            # Every launching/running row, not the 50-row list_runs() window.
            runs = ExperimentTracker().list_active_execution_runs()
        except Exception:
            return None
        for run in runs:
            status = run.get("status")
            if status == _STATUS_LAUNCHING:
                return run.get("run_id")
            if status != _STATUS_RUNNING:
                continue
            pid = run.get("pid")
            if pid is not None and _pid_is_alive(pid):
                return run.get("run_id")
        return None

    def execute(self, *, token: str, kind: str) -> dict:
        if not isinstance(token, str) or not token or len(token) > 4096:
            raise ExecutionError("'confirmation_token' must be a non-empty string")
        with self._lock:
            now = time.monotonic()
            self._cleanup(now)
            plan = self._plans.get(token)
            if plan is None:
                raise ExecutionError("confirmation token is unknown or expired")
            if plan.kind != kind:
                raise ExecutionError("confirmation token is not valid for this execution tool")
            if plan.consumed:
                raise ExecutionError("confirmation token has already been consumed")
            if self._active_run_id is not None:
                raise ExecutionError("an execution is already active for this MCP server")
            # Survive a restart: a child launched by a prior server (in-memory
            # slot lost) is still recorded as running. Gate on liveness so the
            # machine never double-books a training, while a dead record frees.
            if self._live_persisted_run() is not None:
                raise ExecutionError("an execution is already active on this machine")
            self._revalidate(plan)
            # Consumption and capacity acquisition occur before Popen. A failed
            # spawn deliberately requires a fresh plan rather than enabling replay.
            plan.consumed = True
            run_id = plan.run_id or generate_run_id()
            self._active_run_id = run_id

        log_path = self._log_path(run_id)
        digest = hashlib.sha256(
            json.dumps(plan.argv, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        env = os.environ.copy()
        env["SOUP_MCP_RUN_ID"] = run_id

        try:
            tracker = ExperimentTracker()
            tracker.launch_run(
                run_id=run_id,
                kind=kind,
                config_dict={"mcp_argv": list(plan.argv)},
                command_digest=digest,
                log_path=log_path,
            )
            try:
                # open_no_follow (#820): a file symlink planted at the run-log
                # path itself is refused rather than followed, so the child's
                # output cannot be redirected to the link's target. Mode 0o666
                # keeps plain open()'s permissions (umask still applies); the
                # OSError on refusal maps to the path-free ExecutionError below.
                log_handle = os.fdopen(
                    open_no_follow(
                        log_path,
                        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                        0o666,
                    ),
                    "ab",
                )
                process = subprocess.Popen(  # noqa: S603 - internal argv, no shell
                    list(plan.argv),
                    cwd=plan.cwd,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    shell=False,
                )
            finally:
                if "log_handle" in locals():
                    # Popen duplicates the OS handle; the parent does not retain it.
                    log_handle.close()
        except Exception as exc:
            try:
                ExperimentTracker().finish_execution(run_id, status="spawn_failed", exit_code=None)
            except Exception:
                pass
            with self._lock:
                self._active_run_id = None
            if isinstance(exc, ExecutionError):
                raise
            raise ExecutionError("could not spawn execution subprocess") from exc
        try:
            tracker.mark_running(run_id, pid=process.pid)
            threading.Thread(
                target=self._watch,
                args=(process, run_id),
                daemon=True,
                name=f"soup-mcp-{run_id}",
            ).start()
        except Exception as exc:
            # No record of a live pid and no watcher: stop the child rather
            # than leave it running unsupervised, then free the slot.
            self._stop_child(process, run_id)
            try:
                ExperimentTracker().finish_execution(run_id, status="spawn_failed", exit_code=None)
            except Exception:
                pass
            with self._lock:
                if self._active_run_id == run_id:
                    self._active_run_id = None
            raise ExecutionError(
                "could not record the execution; the child process was stopped"
            ) from exc
        return {"run_id": run_id, "status": "running", "pid": process.pid, "log_path": log_path}

    @staticmethod
    def _stop_child(process: subprocess.Popen, run_id: str | None = None) -> None:
        """Terminate, then kill after 10 s; never raises.

        Every failure here leaves a child running that nothing supervises any
        more, so each swallowed exception is logged at debug with the run_id
        and pid: the contract stays "never raises", but a leaked process is no
        longer invisible to whoever reads the log afterwards.
        """
        pid = getattr(process, "pid", None)
        try:
            process.terminate()
        except Exception:
            logger.debug(
                "mcp execution %s: terminate() failed for pid %s", run_id, pid, exc_info=True
            )
        try:
            process.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            logger.debug(
                "mcp execution %s: pid %s still alive 10s after terminate(); killing",
                run_id,
                pid,
            )
        except Exception:
            logger.debug(
                "mcp execution %s: wait() after terminate() failed for pid %s; "
                "giving up without kill()",
                run_id,
                pid,
                exc_info=True,
            )
            return
        try:
            process.kill()
        except Exception:
            logger.debug(
                "mcp execution %s: kill() failed for pid %s", run_id, pid, exc_info=True
            )
        try:
            process.wait(timeout=10)
        except Exception:
            logger.debug(
                "mcp execution %s: pid %s did not reap after kill(); it may still be running",
                run_id,
                pid,
                exc_info=True,
            )

    def _watch(self, process: subprocess.Popen, run_id: str) -> None:
        try:
            exit_code = process.wait()
            tracker = ExperimentTracker()
            tracker.finish_execution(
                run_id,
                status="completed" if exit_code == 0 else "failed",
                exit_code=exit_code,
            )
        finally:
            with self._lock:
                if self._active_run_id == run_id:
                    self._active_run_id = None

    def _revalidate(self, plan: PendingPlan) -> None:
        if os.path.realpath(os.getcwd()) != plan.cwd:
            raise ExecutionError("server working directory changed; create a new plan")
        for protected in plan.protected_files:
            if protected.digest == ABSENT_DIGEST:
                if os.path.lexists(protected.path):
                    raise ExecutionError("planned input changed; create a new plan")
                continue
            current = digest_file(protected.path, "planned input")
            if current.path != protected.path or not secrets.compare_digest(
                current.digest, protected.digest
            ):
                raise ExecutionError("planned input changed; create a new plan")

    def _cleanup(self, now: float) -> None:
        expired = [token for token, plan in self._plans.items() if plan.expires_at <= now]
        for token in expired:
            del self._plans[token]

    def _log_path(self, run_id: str) -> str:
        root = Path(self.cwd) / ".soup" / "mcp-runs"
        root.mkdir(parents=True, exist_ok=True)
        return str(root / f"{run_id}.log")
