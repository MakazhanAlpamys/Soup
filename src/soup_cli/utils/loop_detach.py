"""The PID file and the log of a detached ``soup loop watch``.

``soup loop watch --detach`` spawns a background child. This module is its bookkeeping,
next to the state file (``.soup/loop.yaml`` by default):

* ``.soup/loop.pid`` -- a JSON object ``{"pid": <int>, "started_at": <UTC ISO 8601>,
  "log": <path of the log, relative to the project directory>}``. JSON rather than a bare
  pid so a reader (the Soup Zero desktop app, a script) gets the start time and the log
  location from one file; ``jq .pid .soup/loop.pid`` is the one-liner.
* ``.soup/loop.log`` -- the child's stdout and stderr, appended to across runs.

A PID file names a *live* daemon only while its pid answers the liveness check
(:func:`soup_cli.utils.process_liveness.process_is_alive`); a file whose process is gone,
or that does not parse, is *stale* and the next ``--detach`` replaces it. Known limitation,
inherited from that helper: a recycled pid reads as running, so a stale file whose pid has
since been given to an unrelated process blocks ``--detach`` until it is deleted.

Torch-free and import-light: ``soup loop status`` imports it.
"""

from __future__ import annotations

import errno
import json
import os
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import BinaryIO, Optional

from soup_cli.utils.loop_state import default_state_path
from soup_cli.utils.paths import (
    atomic_write_text,
    enforce_under_cwd_and_no_symlink,
    open_no_follow,
)
from soup_cli.utils.process_liveness import process_is_alive

PID_FILENAME = "loop.pid"
LOG_FILENAME = "loop.log"

#: Set to ``1`` in the environment of the child that ``--detach`` spawns. Only a child
#: that carries it removes the PID file when it exits; a plain foreground ``watch`` never
#: touches the file.
DETACHED_ENV = "SOUP_LOOP_DETACHED"

_PID_FIELD = "PID file"
_LOG_FIELD = "log file"
#: A PID file is a few dozen bytes; anything larger is not one of ours.
_MAX_PID_FILE_BYTES = 4096
#: Windows ``OpenProcess`` takes a DWORD and ``ctypes`` wraps a larger value silently
#: (2**32 + 1 would ask about pid 1), so a pid beyond 31 bits is not a pid.
_MAX_PID = 2**31 - 1

#: States of :class:`DaemonInfo` (``_VALID`` is internal: a file that holds a valid pid).
ABSENT = "absent"
RUNNING = "running"
STALE = "stale"
_VALID = "valid"


@dataclass(frozen=True)
class DaemonInfo:
    """What the PID file says about the daemon."""

    state: str
    pid: Optional[int] = None
    #: For a stale file, why: names the dead pid, or says the file is not a PID file.
    reason: str = ""


def pid_file_path(state_path: Optional[str] = None) -> str:
    """The PID file, in the directory of the state file (default ``.soup/loop.yaml``)."""
    state = default_state_path() if state_path is None else state_path
    return os.path.join(os.path.dirname(state), PID_FILENAME)


def log_file_path(state_path: Optional[str] = None) -> str:
    """The detached child's log, in the directory of the state file."""
    state = default_state_path() if state_path is None else state_path
    return os.path.join(os.path.dirname(state), LOG_FILENAME)


def display_path(path: str) -> str:
    """``path`` as shown to the user: relative to the working directory, ``/`` separators."""
    try:
        shown = os.path.relpath(path)
    except ValueError:  # a different drive on Windows: no relative form exists
        shown = path
    return shown.replace(os.sep, "/")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_pid(raw: bytes) -> Optional[int]:
    """The pid in a PID file's bytes, or ``None`` when they are not a valid record."""
    if len(raw) > _MAX_PID_FILE_BYTES:
        return None
    try:
        record = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    if not isinstance(record, dict):
        return None
    pid = record.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or not 0 < pid <= _MAX_PID:
        return None
    return pid


def _load(path: str) -> "tuple[str, Optional[int]]":
    """Read the PID file without touching a process: ``(ABSENT|STALE|_VALID, pid)``.

    ``_VALID`` only means "holds a valid pid"; the caller asks about liveness.
    Raises ``ValueError`` for a path that is outside the working directory, a symlink, a
    reparse point or not a regular file: those are refused, never read or replaced.
    """
    enforce_under_cwd_and_no_symlink(path, _PID_FIELD)
    try:
        status = os.lstat(path)
    except FileNotFoundError:
        return ABSENT, None
    except OSError as exc:
        raise ValueError(f"{_PID_FIELD} unreadable: {type(exc).__name__}") from exc
    if not stat.S_ISREG(status.st_mode):
        raise ValueError(f"{_PID_FIELD} must be a regular file")
    try:
        fd = open_no_follow(path, os.O_RDONLY)
    except FileNotFoundError:
        return ABSENT, None
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(f"{_PID_FIELD} must not be a symlink (TOCTOU defence)") from exc
        return STALE, None
    with os.fdopen(fd, "rb") as handle:
        raw = handle.read(_MAX_PID_FILE_BYTES + 1)
    pid = _parse_pid(raw)
    return (STALE, None) if pid is None else (_VALID, pid)


def inspect_pid_file(path: Optional[str] = None) -> DaemonInfo:
    """Whether the PID file names a running daemon, a dead one, or nothing.

    Raises ``ValueError`` for a PID file path that must be refused (see :func:`_load`).
    """
    target = pid_file_path() if path is None else path
    state, pid = _load(target)
    if state == ABSENT:
        return DaemonInfo(ABSENT)
    if pid is None:
        return DaemonInfo(STALE, reason="not a valid PID file")
    if process_is_alive(pid):
        return DaemonInfo(RUNNING, pid)
    return DaemonInfo(STALE, pid, reason=f"pid {pid} is not running")


def describe_daemon() -> str:
    """The ``daemon`` row of ``soup loop status`` (plain text, not Rich markup)."""
    try:
        info = inspect_pid_file()
    except ValueError:
        return "not running (unusable PID file)"
    if info.state == RUNNING:
        return f"running (pid {info.pid}), log {display_path(log_file_path())}"
    if info.state == STALE:
        return "not running (stale pid file)"
    return "not running"


def open_log(path: Optional[str] = None) -> BinaryIO:
    """Open the log for the child's stdout and stderr: created if missing, appended to.

    Refuses a path outside the working directory, a symlink or reparse point (also as a
    parent directory) and a hard link; raises ``ValueError`` for those. The caller closes
    its handle once the child has been spawned: the child holds its own duplicate.
    """
    target = log_file_path() if path is None else path
    enforce_under_cwd_and_no_symlink(target, _LOG_FIELD)
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    try:
        fd = open_no_follow(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
            check_parent=True,
            refuse_hardlink=True,
        )
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(
                f"{_LOG_FIELD} or its directory must not be a symlink or junction"
            ) from exc
        if exc.errno == errno.EMLINK:
            raise ValueError(f"{_LOG_FIELD} must not be a hard link") from exc
        raise
    try:
        return os.fdopen(fd, "ab")
    except Exception:
        os.close(fd)
        raise


def write_pid_file(pid: int, path: Optional[str] = None) -> str:
    """Atomically record the daemon's ``pid``; returns the path written."""
    if isinstance(pid, bool) or not isinstance(pid, int) or not 0 < pid <= _MAX_PID:
        raise ValueError(f"pid must be an int in [1, {_MAX_PID}], got {pid!r}")
    target = pid_file_path() if path is None else path
    record = {
        "pid": pid,
        "started_at": _utc_now_iso(),
        "log": display_path(log_file_path()),
    }
    atomic_write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        target,
        prefix=".loop_pid_",
        field=_PID_FIELD,
    )
    return target


def remove_pid_file_if_own(own_pid: int, path: Optional[str] = None) -> bool:
    """Remove the PID file, but only if it still holds ``own_pid``. Never raises.

    A newer daemon's file, an unreadable file and a symlink are all left alone. Returns
    whether a file was removed.
    """
    target = pid_file_path() if path is None else path
    try:
        state, pid = _load(target)
        if state != _VALID or pid != own_pid:
            return False
        os.unlink(target)
    except (OSError, ValueError):
        return False
    return True
