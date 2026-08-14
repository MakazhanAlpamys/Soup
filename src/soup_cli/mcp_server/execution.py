"""In-memory plan capabilities and isolated MCP command execution."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from soup_cli.experiment.tracker import ExperimentTracker, generate_run_id
from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

TOKEN_TTL_SECONDS = 5 * 60


class ExecutionError(ValueError):
    """Safe execution failure intended for translation to an MCP tool error."""


@dataclass(frozen=True)
class ProtectedFile:
    path: str
    digest: str


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
    consumed: bool = False


def digest_file(path: str, field: str) -> ProtectedFile:
    """Return a cwd-contained, non-symlink file's canonical path and digest."""
    try:
        enforce_under_cwd_and_no_symlink(path, field)
        real = os.path.realpath(path)
        if os.path.isdir(real):
            stat_result = os.stat(real)
            digest = hashlib.sha256(
                f"dir:{stat_result.st_mtime_ns}:{stat_result.st_size}".encode("ascii")
            ).hexdigest()
        else:
            with open(real, "rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
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

    def issue(
        self,
        *,
        kind: str,
        argv: list[str],
        display_command: str,
        protected_files: tuple[ProtectedFile, ...] = (),
    ) -> str:
        now = time.monotonic()
        token = secrets.token_urlsafe(32)
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
            )
        return token

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
            self._revalidate(plan)
            # Consumption and capacity acquisition occur before Popen. A failed
            # spawn deliberately requires a fresh plan rather than enabling replay.
            plan.consumed = True
            run_id = generate_run_id()
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
                log_handle = open(log_path, "ab")
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
        tracker.mark_running(run_id, pid=process.pid)
        threading.Thread(
            target=self._watch,
            args=(process, run_id),
            daemon=True,
            name=f"soup-mcp-{run_id}",
        ).start()
        return {"run_id": run_id, "status": "running", "pid": process.pid, "log_path": log_path}

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
