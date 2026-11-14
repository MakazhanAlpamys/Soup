"""The PID file and the log of a detached ``soup loop watch``.

``soup loop watch --detach`` spawns a background child. This module is its bookkeeping,
next to the state file (``.soup/loop.yaml`` by default):

* ``.soup/loop.pid`` -- a JSON object ``{"pid": <int>, "started_at": <UTC ISO 8601>,
  "log": <path of the log, relative to the project directory>}``. JSON rather than a bare
  pid so a reader (a desktop app, a script) gets the start time and the log
  location from one file; ``jq .pid .soup/loop.pid`` is the one-liner.
* ``.soup/loop.log`` -- the child's stdout and stderr, appended to across runs. It is
  rotated at :data:`MAX_LOG_BYTES` (10 MB), keeping one old file, ``.soup/loop.log.1``:
  :func:`rotate_log_if_needed` does it when ``--detach`` starts, :func:`rotate_open_log`
  does it inside the running child at each watch cycle.

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
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import BinaryIO, Optional

from soup_cli.utils.loop_state import default_state_path
from soup_cli.utils.paths import (
    atomic_write_text,
    enforce_under_cwd_and_no_symlink,
    open_no_follow,
    refuse_linked_dirs,
)
from soup_cli.utils.process_liveness import process_is_alive
from soup_cli.utils.terminal import strip_control

PID_FILENAME = "loop.pid"
LOG_FILENAME = "loop.log"

#: The log is rotated once it reaches this many bytes: at detach time by the parent, and at
#: each watch cycle by the running child. One old file is kept, ``loop.log.1``.
MAX_LOG_BYTES = 10 * 1024 * 1024
ROTATED_SUFFIX = ".1"

#: Set to ``1`` in the environment of the child that ``--detach`` spawns. Only a child
#: that carries it removes the PID file when it exits; a plain foreground ``watch`` never
#: touches the file.
DETACHED_ENV = "SOUP_LOOP_DETACHED"

_PID_FIELD = "PID file"
_LOG_FIELD = "log file"
_ROTATED_FIELD = "rotated log file"
#: A PID file is a few dozen bytes; anything larger is not one of ours.
_MAX_PID_FILE_BYTES = 4096
#: Windows ``OpenProcess`` takes a DWORD and ``ctypes`` wraps a larger value silently
#: (2**32 + 1 would ask about pid 1), so a pid beyond 31 bits is not a pid.
_MAX_PID = 2**31 - 1
#: How long a claimant waits for an unparsable PID file to turn into a valid one (seconds).
_CLAIM_WRITE_GRACE_SEC = 0.05

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


def detached_popen_kwargs(os_name: Optional[str] = None) -> dict:
    """``subprocess.Popen`` arguments that cut the child loose from the parent's terminal.

    POSIX: ``start_new_session=True`` (a new session, so the terminal's SIGHUP and Ctrl+C never
    reach it). Windows: ``DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`` (no console to close,
    and Ctrl+C is not delivered to the new group). Either way stdin is the null device.
    """
    kwargs: dict = {"stdin": subprocess.DEVNULL}
    if (os.name if os_name is None else os_name) == "nt":
        kwargs["creationflags"] = (
            getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        )
    else:
        kwargs["start_new_session"] = True
    return kwargs


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


def rotated_log_path(path: Optional[str] = None) -> str:
    """The one old log kept next to ``path`` (default: the log): ``loop.log.1``."""
    target = log_file_path() if path is None else path
    return target + ROTATED_SUFFIX


def _limit(max_bytes: Optional[int]) -> int:
    """``max_bytes`` or, when ``None``, the module limit (read at call time)."""
    limit = MAX_LOG_BYTES if max_bytes is None else max_bytes
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError(f"max_bytes must be a positive int, got {limit!r}")
    return limit


def _checked_rotation_target(path: str, field: str) -> "Optional[os.stat_result]":
    """``lstat`` of a file the rotation touches, after the refusals :func:`open_log` makes.

    ``None`` when the file does not exist. Raises ``ValueError`` for a path outside the
    working directory, a symlink or reparse point (also as a parent directory), a hard
    link, or anything that is not a regular file.
    """
    enforce_under_cwd_and_no_symlink(path, field)
    try:
        refuse_linked_dirs(os.path.dirname(os.path.abspath(path)))
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(
                f"{field} or its directory must not be a symlink or junction"
            ) from exc
        raise
    try:
        status = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError(f"{field} unreadable: {type(exc).__name__}") from exc
    if not stat.S_ISREG(status.st_mode):
        raise ValueError(f"{field} must be a regular file")
    if status.st_nlink > 1:
        raise ValueError(f"{field} must not be a hard link")
    return status


def rotate_log_if_needed(path: Optional[str] = None, *, max_bytes: Optional[int] = None) -> bool:
    """Parent side, before the spawn: move a log of ``max_bytes`` or more to ``loop.log.1``.

    ``max_bytes`` defaults to :data:`MAX_LOG_BYTES`; a smaller value is for tests. An
    existing ``loop.log.1`` is replaced. Returns whether the log was moved (the caller then
    opens a fresh one). Raises ``ValueError`` for a log or an old file that must be refused
    (see :func:`_checked_rotation_target`) and ``OSError`` if the move fails.
    """
    limit = _limit(max_bytes)
    target = log_file_path() if path is None else path
    status = _checked_rotation_target(target, _LOG_FIELD)
    if status is None or status.st_size < limit:
        return False
    rotated = rotated_log_path(target)
    _checked_rotation_target(rotated, _ROTATED_FIELD)
    os.replace(target, rotated)
    return True


def _flush_standard_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (AttributeError, OSError, ValueError):
            pass


def _says(exc: BaseException) -> str:
    """``exc`` on one line, without control characters, for a line written to the log."""
    return " ".join(strip_control(f"{type(exc).__name__}: {exc}").split())


def rotate_open_log(
    path: Optional[str] = None,
    *,
    max_bytes: Optional[int] = None,
    fds: "tuple[int, ...]" = (1, 2),
) -> bool:
    """Child side, at each watch cycle: swap a full log for a fresh one without stopping.

    ``fds`` are the descriptors that write the log (the child's stdout and stderr). Does
    nothing, and returns ``False``, unless every one of them is open and points at the very
    file ``path`` (default: the log) names and that file has reached ``max_bytes``
    (default :data:`MAX_LOG_BYTES`): a child whose output was redirected elsewhere, or a
    terminal, is left alone. Then:

    1. flush ``sys.stdout`` / ``sys.stderr``;
    2. point every fd at the null device, so this process holds no handle on the log
       (Windows cannot rename an open file);
    3. move the log to ``loop.log.1``, replacing an older one;
    4. open a fresh log with :func:`open_log` and point every fd at it.

    Past that check it never raises and never leaves an fd closed: a step that fails is
    reported in one line written to the last fd, whatever it points at by then, and the
    rotation carries on. (A ``max_bytes`` that is not a positive int is a caller bug and
    raises ``ValueError``; the daemon passes none.)
    If the move fails the fds are pointed back at the same log (nothing is lost); if no log
    can be opened they stay on the null device. Returns whether the log was moved.
    """
    limit = _limit(max_bytes)
    target = log_file_path() if path is None else path
    try:
        if not fds:
            return False
        reached = os.lstat(target)
        for fd in fds:
            current = os.fstat(fd)
            if not os.path.samestat(current, reached):
                return False
        if os.fstat(fds[0]).st_size < limit:
            return False
    except (OSError, ValueError):
        return False  # a closed fd, a missing log, a path this process may not look at

    rotated = rotated_log_path(target)
    problems: "list[str]" = []
    moved = False
    devnull: Optional[int] = None
    _flush_standard_streams()
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        for fd in fds:
            os.dup2(devnull, fd)
    except OSError as exc:
        problems.append(f"could not release the log ({_says(exc)})")
    try:
        _checked_rotation_target(rotated, _ROTATED_FIELD)
        os.replace(target, rotated)
        moved = True
    except (OSError, ValueError) as exc:
        problems.append(f"could not move the log to {LOG_FILENAME}{ROTATED_SUFFIX} ({_says(exc)})")
    for candidate in (target, rotated) if moved else (target,):
        try:
            handle = open_log(candidate)
        except (OSError, ValueError) as exc:
            problems.append(f"could not reopen {os.path.basename(candidate)} ({_says(exc)})")
            continue
        try:
            for fd in fds:
                os.dup2(handle.fileno(), fd)
        except OSError as exc:
            problems.append(f"could not redirect the output ({_says(exc)})")
        finally:
            handle.close()
        break
    if devnull is not None and devnull not in fds:
        try:
            os.close(devnull)
        except OSError:
            pass
    if problems:
        what = "log rotation incomplete" if moved else "log rotation skipped"
        line = f"soup loop: {what}: " + "; ".join(problems) + "\n"
        try:
            os.write(fds[-1], line.encode("utf-8", "replace"))
        except OSError:
            pass
    return moved


class PidFileHeldError(Exception):
    """A live process already holds the PID file: ``pid`` (None when it cannot be read yet)."""

    def __init__(self, pid: Optional[int], path: str) -> None:
        super().__init__(f"{_PID_FIELD} {path} is held by pid {pid}")
        self.pid = pid
        self.path = path


def _pid_record(pid: int) -> bytes:
    record = {
        "pid": pid,
        "started_at": _utc_now_iso(),
        "log": display_path(log_file_path()),
    }
    return (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _remove_stale(path: str, seen: DaemonInfo) -> None:
    """Unlink the PID file ``seen`` described, unless it has changed since (never raises)."""
    try:
        state, pid = _load(path)
        if state == ABSENT:
            return
        if state == _VALID and (pid != seen.pid or process_is_alive(pid)):
            return  # another claimant replaced it, or its pid is alive after all
        os.unlink(path)
    except (OSError, ValueError):
        return


def claim_pid_file(path: Optional[str] = None) -> Optional[DaemonInfo]:
    """Claim the PID file for this process, atomically, before a daemon is spawned.

    The file is created with ``O_CREAT | O_EXCL`` and holds THIS process's pid, so a second
    ``--detach`` running at the same moment finds a live holder and is refused instead of
    also seeing "no file" and spawning a second daemon. The caller replaces the pid with the
    child's through :func:`write_pid_file` once it has spawned, or removes the claim
    (:func:`remove_pid_file_if_own`) when the spawn fails.

    A file that already exists is inspected: a live pid raises :class:`PidFileHeldError`; a
    stale one (dead pid, or unparsable after a short wait for a writer still in the middle of
    its claim) is removed and the claim retried. Returns the stale file's description when one
    was replaced, else ``None``. Raises ``ValueError`` for a path that must be refused
    (outside the working directory, a symlink, a reparse point, not a regular file).
    """
    target = pid_file_path() if path is None else path
    replaced: Optional[DaemonInfo] = None
    for _attempt in range(4):
        try:
            fd = open_no_follow(
                target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, check_parent=True
            )
        except FileExistsError:
            info = inspect_pid_file(target)
            if info.state == STALE and info.pid is None:
                # The file may be a claim whose writer has not written its pid yet.
                time.sleep(_CLAIM_WRITE_GRACE_SEC)
                info = inspect_pid_file(target)
            if info.state == RUNNING:
                raise PidFileHeldError(info.pid, target) from None
            if info.state == STALE:
                replaced = info
                _remove_stale(target, info)
            continue  # stale one removed, or the file vanished: claim again
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ValueError(f"{_PID_FIELD} must not be a symlink (TOCTOU defence)") from exc
            if os.path.lexists(target):
                # A directory (Windows answers "access denied" rather than "exists") or
                # another file that cannot be opened: name the refusal.
                _load(target)
                raise ValueError(f"{_PID_FIELD} cannot be opened: {type(exc).__name__}") from exc
            raise
        with os.fdopen(fd, "wb") as handle:
            handle.write(_pid_record(os.getpid()))
        return replaced
    raise ValueError(f"{_PID_FIELD} is being claimed by another process; try again")


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
