"""HIPAA/SOC2-shaped JSONL audit log (v0.59.0 Part D).

Every Soup command appends one line to ``~/.soup/audit.jsonl`` (override
via ``SOUP_AUDIT_LOG_PATH``). Lines are JSON objects with a fixed set of
keys so Splunk / ELK can ingest them without a custom parser.

PII redaction reuses the v0.40.3 #33 ``_SECRET_RE`` policy: ``hf_*`` /
``sk-*`` / ``Bearer …`` tokens are masked as ``<redacted>``. Rotation at
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
from dataclasses import dataclass, replace
from typing import Optional, Tuple

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
                raise ValueError(f"{name} too int ({len(value)} > {max_len})")
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
                raise ValueError(f"args[*] too int (> {_MAX_ARG_LEN})")
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
        _LOG.warning("SOUP_AUDIT_LOG_PATH too int; falling back to default")
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


def append_audit_event(
    ev: AuditEvent,
    path: Optional[str] = None,
    *,
    cap_bytes: int = _DEFAULT_CAP_BYTES,
    redact: bool = True,
) -> None:
    """Append one audit record to ``path``. Rotates at ``cap_bytes``.

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
    # Best-effort rotation before writing the new line.
    rotate_if_needed(target, cap_bytes=cap_bytes)
    line = json.dumps((redact_event(ev) if redact else ev).to_dict()) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    # O_NOFOLLOW: refuse to follow a symlink at the target path.
    # Not available on Windows.
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow:
        flags |= nofollow
    mode = 0o600
    fd = os.open(target, flags, mode)
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
