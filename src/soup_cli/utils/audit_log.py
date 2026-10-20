"""HIPAA/SOC2-shaped JSONL audit log (v0.59.0 Part D).

Every Soup command appends one line to ``~/.soup/audit.jsonl`` (override
via ``SOUP_AUDIT_LOG_PATH``). Lines are JSON objects with a fixed set of
keys so Splunk / ELK can ingest them without a custom parser.

PII redaction reuses the v0.40.3 #33 ``_SECRET_RE`` policy: ``hf_*`` /
``sk-*`` / ``Bearer …`` tokens are masked as ``<redacted>``. Before a record
reaches this module, ``cli._emit_audit_event`` masks the values of credential
options by option name (``utils/argv_redaction.py``); this value-pattern layer
applies on top of that. Rotation at
100 MiB by default — operators wanting longer retention should run
``logrotate``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import stat
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Callable, Optional, Sequence, Tuple

from soup_cli.utils.paths import open_no_follow

_LOG = logging.getLogger(__name__)

# Mirrors v0.40.3 #33 TraceLogWriter._SECRET_RE policy.
_SECRET_RE = re.compile(
    r"hf_[A-Za-z0-9_]{8,}"        # HF tokens
    r"|sk-[A-Za-z0-9_\-]{16,}"     # OpenAI / Anthropic style
    r"|Bearer\s+[A-Za-z0-9_\-]{8,}"  # bearer header style
)
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")

_MAX_CMD = 64
_MAX_HOST = 128
_MAX_OPERATOR = 128
_MAX_ARG_LEN = 1024
_MAX_ARGS = 256
_DEFAULT_CAP_BYTES = 100 * 1024 * 1024  # 100 MiB

# Masks the args of one stored line: ``(command, args) -> args``.
ArgMask = Callable[[str, list], Sequence[str]]

# Lines written by older versions get the same option-name masking as new ones, once.
# The marker sits next to the log; bump the version to run the pass again.
_MASK_MARKER_SUFFIX = ".masked-v1"
# A pass that failed leaves ``<log>.masked-v1.tried`` holding the UTC time of the attempt;
# the next one waits for that to be a day old, so a log that cannot be replaced
# (another process holds it open) is not re-read by every command.
_MASK_TRIED_SUFFIX = ".masked-v1.tried"
_MASK_RETRY_AFTER_SECONDS = 24 * 60 * 60
# The pass writes its replacement to ``.audit-mask-*.tmp`` beside the log; one left by a
# process that was killed is removed by the next pass once it is this old.
_MASK_TMP_PREFIX = ".audit-mask-"
_MASK_TMP_SUFFIX = ".tmp"
_MASK_STALE_TMP_SECONDS = 60 * 60
_MASK_MAX_LINE_BYTES = 4 * 1024 * 1024  # a longer line is copied through unread
_MASK_COPY_BLOCK = 1024 * 1024
_MASK_MAX_ROUNDS = 5  # the first read of the file plus the re-reads for lines added meanwhile


@dataclass(frozen=True)
class AuditEvent:
    """One audit record. JSON-serialised one-per-line."""

    timestamp: str
    command: str
    args: Tuple[str, ...]
    exit_code: int
    host_id: str
    operator_id: str

    def __post_init__(self) -> None:
        for value, name, max_len in (
            (self.timestamp, "timestamp", 64),
            (self.command, "command", _MAX_CMD),
            (self.host_id, "host_id", _MAX_HOST),
            (self.operator_id, "operator_id", _MAX_OPERATOR),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty str")
            if "\x00" in value:
                raise ValueError(f"{name} must not contain null bytes")
            if len(value) > max_len:
                raise ValueError(f"{name} too long ({len(value)} > {max_len})")
        if not isinstance(self.args, tuple):
            raise ValueError("args must be a tuple")
        if len(self.args) > _MAX_ARGS:
            raise ValueError(f"too many args ({len(self.args)} > {_MAX_ARGS})")
        for arg in self.args:
            if not isinstance(arg, str):
                raise ValueError("args[*] must be str")
            if "\x00" in arg:
                raise ValueError("args[*] must not contain null bytes")
            if len(arg) > _MAX_ARG_LEN:
                raise ValueError(f"args[*] too long (> {_MAX_ARG_LEN})")
        if isinstance(self.exit_code, bool):
            raise ValueError("exit_code must be int, not bool")
        if not isinstance(self.exit_code, int):
            raise ValueError("exit_code must be int")

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "command": self.command,
            "args": list(self.args),
            "exit_code": self.exit_code,
            "host_id": self.host_id,
            "operator_id": self.operator_id,
        }


def _redact_str(value: str) -> str:
    return _SECRET_RE.sub("<redacted>", value)


def redact_event(ev: AuditEvent) -> AuditEvent:
    """Return a new ``AuditEvent`` with secrets masked in every string field.

    Security review HIGH fix: previously only ``args`` was redacted. An
    operator_id or host_id containing ``Bearer …`` / ``hf_…`` would persist
    verbatim. We now walk every string field (mirrors v0.34.0 ``crash.py``
    policy of recursive secret redaction).
    """
    return replace(
        ev,
        command=_redact_str(ev.command),
        args=tuple(_redact_str(a) for a in ev.args),
        host_id=_redact_str(ev.host_id),
        operator_id=_redact_str(ev.operator_id),
    )


def _check_symlink_at(path: str) -> bool:
    """Return True when ``path`` exists AND is a symlink (TOCTOU-safe).

    Uses ``os.lstat`` directly inside try/except FileNotFoundError instead of
    ``lexists``-then-``lstat`` — closes the race between the existence check
    and the stat (security review HIGH fix; mirrors v0.33.0 #22 / v0.55.0).
    """
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError:
        # Other OSError (perm, etc.) - treat as "do not touch".
        return True
    return stat.S_ISLNK(st.st_mode)


def rotate_if_needed(path: str, *, cap_bytes: int = _DEFAULT_CAP_BYTES) -> bool:
    """Rotate ``<path>`` -> ``<path>.1`` when file size exceeds ``cap_bytes``.

    Symlink at the backup path is rejected via direct ``os.lstat`` (no lexists
    race — security review HIGH fix). Returns True when rotation happened.
    """
    if isinstance(cap_bytes, bool) or not isinstance(cap_bytes, int):
        raise ValueError("cap_bytes must be int")
    if cap_bytes <= 0:
        raise ValueError("cap_bytes must be > 0")
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty str")
    try:
        size = os.path.getsize(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        _LOG.warning("rotate_if_needed: getsize(%s) failed: %s", path, exc)
        return False
    if size <= cap_bytes:
        return False
    backup = path + ".1"
    if _check_symlink_at(backup):
        # Refuse to overwrite a symlink — TOCTOU defence.
        _LOG.warning(
            "rotate_if_needed: refusing to rotate, backup path %s is a symlink",
            backup,
        )
        return False
    # If a regular backup exists, remove it.
    try:
        if os.path.isfile(backup):
            os.unlink(backup)
    except OSError as exc:
        _LOG.warning("rotate_if_needed: unlink(%s) failed: %s", backup, exc)
        return False
    try:
        os.replace(path, backup)
    except OSError as exc:
        _LOG.warning("rotate_if_needed: replace(%s) failed: %s", path, exc)
        return False
    return True


def _validate_log_path_override(override: str) -> Optional[str]:
    """Validate the ``SOUP_AUDIT_LOG_PATH`` env override.

    Per v0.36.0 ``SOUP_BATCH_CACHE_PATH`` / v0.54.0 ``SOUP_ADVISE_HISTORY_PATH``
    policy: rejects null bytes / control chars / overlong values, and refuses
    paths outside ``$HOME / $CWD / $TMPDIR``. Returns the validated path or
    ``None`` when invalid (caller then falls back to the safe default).
    """
    if not isinstance(override, str) or not override:
        return None
    if _CTRL_RE.search(override):
        _LOG.warning(
            "SOUP_AUDIT_LOG_PATH contains null/control chars; falling back to default"
        )
        return None
    if len(override) > 4096:
        _LOG.warning("SOUP_AUDIT_LOG_PATH too long; falling back to default")
        return None
    try:
        realpath = os.path.realpath(override)
    except (OSError, ValueError):
        return None
    home = os.path.realpath(os.path.expanduser("~"))
    cwd = os.path.realpath(os.getcwd())
    tmpdir = os.path.realpath(tempfile.gettempdir())
    for allowed in (home, cwd, tmpdir):
        try:
            common = os.path.commonpath([realpath, allowed])
        except ValueError:
            continue
        if common == allowed:
            return override
    _LOG.warning(
        "SOUP_AUDIT_LOG_PATH %r outside $HOME / $CWD / $TMPDIR; "
        "falling back to default",
        override,
    )
    return None


def default_log_path() -> str:
    """Resolve the audit log path (env override first, else ``~/.soup/audit.jsonl``).

    The env override goes through ``_validate_log_path_override`` so callers
    cannot smuggle a system file (``/etc/cron.d``) through the override.

    POSIX ``os.environ.get`` raises ``ValueError`` when the value contains
    embedded null bytes; we catch and fall back to the safe default.
    """
    try:
        override = os.environ.get("SOUP_AUDIT_LOG_PATH")
    except ValueError:
        # Env value contains a null byte — POSIX rejects on read.
        override = None
    if override:
        validated = _validate_log_path_override(override)
        if validated is not None:
            return validated
    home = os.path.expanduser("~")
    return os.path.join(home, ".soup", "audit.jsonl")


# Backwards-compatible alias for tests / internal callers.
_default_log_path = default_log_path


def _mask_line(raw: bytes, arg_mask: ArgMask) -> Optional[bytes]:
    """Return the re-masked form of one stored line, or ``None`` to keep it as it is.

    Only a line that parses as an audit record whose ``args`` change is rebuilt;
    anything else (blank, malformed, unexpected shape, already masked) stays
    byte for byte.
    """
    body = raw.rstrip(b"\r\n")
    ending = raw[len(body):]
    try:
        record = json.loads(body)
    except (ValueError, RecursionError):
        return None
    if not isinstance(record, dict):
        return None
    command = record.get("command")
    args = record.get("args")
    if not isinstance(command, str) or not isinstance(args, list):
        return None
    if not all(isinstance(arg, str) for arg in args):
        return None
    masked = list(arg_mask(command, args))
    if masked == args:
        return None
    record["args"] = masked
    return json.dumps(record).encode("utf-8") + ending


def _copy_head(src, out, count: int) -> None:
    """Copy the first ``count`` bytes of ``src`` to ``out``; ``src`` keeps its position."""
    resume = src.tell()
    src.seek(0)
    remaining = count
    while remaining:
        block = src.read(min(_MASK_COPY_BLOCK, remaining))
        if not block:
            raise OSError("audit log shrank while its older lines were being masked")
        out.write(block)
        remaining -= len(block)
    src.seek(resume)


def _mask_file_lines(path: str, arg_mask: ArgMask) -> None:
    """Apply ``arg_mask`` to the args of every line of ``path``, rewriting it only if needed.

    Streams line by line; the replacement is written to a temp file in the same
    directory and swapped in with ``os.replace``, so the log is either the old
    file or the complete new one. A missing file or a symlink is skipped. Raises
    ``OSError`` (file untouched) when the file cannot be read or replaced.
    """
    if _check_symlink_at(path):
        return
    try:
        fd = open_no_follow(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    except FileNotFoundError:
        return
    parent = os.path.dirname(os.path.abspath(path)) or "."
    out = None
    tmp_path = None
    try:
        with os.fdopen(fd, "rb") as src:
            info = os.fstat(src.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise OSError(f"{path!r} is not a regular file")
            if not info.st_mode & stat.S_IWRITE:
                raise PermissionError(f"{path!r} is read-only; left as it is")
            offset = 0
            skipping = False
            rounds = 0
            while True:
                # A round reads up to the size seen at its start. Lines another
                # process appends meanwhile are picked up by the next round.
                round_end = os.fstat(src.fileno()).st_size
                if round_end <= offset:
                    break
                rounds += 1
                if rounds > _MASK_MAX_ROUNDS:
                    raise OSError("audit log kept growing while its older lines were masked")
                while offset < round_end:
                    raw = src.readline(_MASK_MAX_LINE_BYTES)
                    if not raw:
                        raise OSError("audit log shrank while its older lines were masked")
                    whole = raw.endswith(b"\n")
                    if skipping or (not whole and len(raw) == _MASK_MAX_LINE_BYTES):
                        new = None
                        skipping = not whole
                    else:
                        new = _mask_line(raw, arg_mask)
                    if new is not None and out is None:
                        tmp_fd, tmp_path = tempfile.mkstemp(
                            dir=parent, prefix=_MASK_TMP_PREFIX, suffix=_MASK_TMP_SUFFIX
                        )
                        out = os.fdopen(tmp_fd, "wb")
                        _copy_head(src, out, offset)
                    if out is not None:
                        out.write(raw if new is None else new)
                    offset += len(raw)
        if out is None:
            return
        out.flush()
        os.fsync(out.fileno())
        out.close()
        out = None
        try:
            os.chmod(tmp_path, stat.S_IMODE(info.st_mode))
        except OSError as exc:
            _LOG.debug("audit-log mask: chmod of the temp file failed: %s", exc)
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if out is not None:
            out.close()
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _failed_attempt_is_recent(tried_path: str) -> bool:
    """Whether the record at ``tried_path`` names a time less than a day ago.

    A missing, unreadable, malformed or future-dated record is not recent. It is
    never read through a symlink.
    """
    try:
        fd = open_no_follow(tried_path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    except OSError:
        return False
    try:
        raw = os.read(fd, 64)
    except OSError:
        return False
    finally:
        os.close(fd)
    try:
        stamp = datetime.fromisoformat(raw.decode("ascii").strip())
    except ValueError:
        return False
    if stamp.tzinfo is None:
        return False
    age = (_utc_now() - stamp).total_seconds()
    return 0 <= age < _MASK_RETRY_AFTER_SECONDS


def _record_failed_attempt(tried_path: str) -> None:
    """Write the UTC time of a failed pass to ``tried_path``; never raises."""
    try:
        fd = open_no_follow(tried_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, (_utc_now().isoformat() + "\n").encode("ascii"))
        finally:
            os.close(fd)
    except OSError as exc:
        _LOG.debug("audit-log mask: could not record the failed attempt: %s", exc)


def _remove_stale_temp_files(parent: str) -> None:
    """Remove ``.audit-mask-*.tmp`` regular files in ``parent`` older than an hour.

    These are what a pass leaves behind when its process is killed before the
    swap. Only a file with exactly that prefix and suffix is considered, and it
    is looked at without following links, so a link or directory of that name
    is left as it is. Never raises.
    """
    try:
        names = os.listdir(parent)
    except OSError:
        return
    cutoff = time.time() - _MASK_STALE_TMP_SECONDS
    for name in names:
        if not (name.startswith(_MASK_TMP_PREFIX) and name.endswith(_MASK_TMP_SUFFIX)):
            continue
        full = os.path.join(parent, name)
        try:
            info = os.lstat(full)
            if stat.S_ISREG(info.st_mode) and info.st_mtime < cutoff:
                os.unlink(full)
        except OSError as exc:
            _LOG.debug("audit-log mask: stale temp file %r kept: %s", name, exc)


def mask_older_lines(path: str, arg_mask: ArgMask) -> bool:
    """Give the lines already in ``path`` (and ``path.1``) the masking new lines get.

    Lines written by older versions carry option values as typed. ``arg_mask``
    is the masking applied to new lines; it is run over the ``args`` of every
    stored line once, then a marker file ``<path>.masked-v1`` is written and
    later calls return at once. Best-effort: any failure (read-only file, file
    locked by another process, full disk) leaves the log untouched, writes no
    marker and never raises. The failure is recorded in ``<path>.masked-v1.tried``
    and the next attempt waits until that is a day old, so a log that cannot be
    replaced is not read again by every command; a pass that succeeds removes it.

    Returns True when the log is done (marker present), False after a failure
    or while waiting out a failed attempt.
    """
    marker = path + _MASK_MARKER_SUFFIX
    tried = path + _MASK_TRIED_SUFFIX
    try:
        if os.path.exists(marker):
            return True
        if _failed_attempt_is_recent(tried):
            return False
        _remove_stale_temp_files(os.path.dirname(os.path.abspath(path)) or ".")
        for name in (path, path + ".1"):
            _mask_file_lines(name, arg_mask)
        fd = open_no_follow(marker, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, b"1\n")
        finally:
            os.close(fd)
    except Exception as exc:  # noqa: BLE001 — never raise into the command being run
        _LOG.debug("audit-log mask of older lines failed, will retry later: %s", exc)
        _record_failed_attempt(tried)
        return False
    try:
        os.unlink(tried)
    except FileNotFoundError:
        pass
    except OSError as exc:
        _LOG.debug("audit-log mask: could not remove the failed-attempt record: %s", exc)
    return True


def append_audit_event(
    ev: AuditEvent,
    path: Optional[str] = None,
    *,
    cap_bytes: int = _DEFAULT_CAP_BYTES,
    redact: bool = True,
    arg_mask: Optional[ArgMask] = None,
) -> None:
    """Append one audit record to ``path``. Rotates at ``cap_bytes``.

    When ``arg_mask`` is given, lines written by older versions are first
    brought to the same masking, once (see :func:`mask_older_lines`).

    Atomic-ish append: opens with ``O_APPEND | O_CREAT`` (and ``O_NOFOLLOW``
    on POSIX — security review HIGH fix). Concurrent writers can interleave
    on POSIX without truncation; on Windows the OS does not guarantee atomic
    append for buffered writes, so very high concurrency may interleave
    bytes within a line.
    """
    # Explicit `is None` check (project policy since v0.40.6 — empty-string is
    # a distinct operator error, NOT silent missing).
    if path is None:
        target = default_log_path()
    elif not isinstance(path, str):
        raise ValueError(f"path must be str, got {type(path).__name__}")
    elif not path:
        raise ValueError("path must be a non-empty str")
    else:
        target = path
    if "\x00" in target:
        raise ValueError("path must not contain null bytes")
    parent = os.path.dirname(os.path.abspath(target)) or "."
    os.makedirs(parent, exist_ok=True)
    if arg_mask is not None:
        mask_older_lines(target, arg_mask)
    # Best-effort rotation before writing the new line.
    rotate_if_needed(target, cap_bytes=cap_bytes)
    line = json.dumps((redact_event(ev) if redact else ev).to_dict()) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    mode = 0o600
    fd = open_no_follow(target, flags, mode)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)
    if os.name != "nt":
        try:
            current = stat.S_IMODE(os.stat(target).st_mode)
            if current != 0o600:
                os.chmod(target, 0o600)
        except OSError as exc:
            _LOG.debug("audit-log chmod failed: %s", exc)


def read_audit_tail(path: Optional[str] = None, *, limit: int = 50) -> list[dict]:
    """Read the last ``limit`` audit records (newest last)."""
    if isinstance(limit, bool):
        raise ValueError("limit must be int")
    if not isinstance(limit, int) or limit < 1 or limit > 100_000:
        raise ValueError("limit must be an int in [1, 100000]")
    target = path or default_log_path()
    if not os.path.isfile(target):
        return []
    out: list[dict] = []
    try:
        with open(target, encoding="utf-8") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    out.append(json.loads(raw))
                except (ValueError, TypeError):
                    continue
    except OSError:
        return []
    return out[-limit:]
