"""Cross-project verdict history (v0.54.0 Part C).

Stores accepted/rejected verdicts + outcomes as a local JSONL log so
`soup advise compare` can render prior decisions. Separate from
``soup_cli/registry`` (which tracks trained model artifacts) — verdicts
are decision records, not training artifacts.

Path: ``~/.soup/advise_history.jsonl`` (override via
``SOUP_ADVISE_HISTORY_PATH``). The override must resolve under one of
``{$HOME, $CWD, tempfile.gettempdir()}`` — same containment policy as
v0.36.0 ``SOUP_BATCH_CACHE_PATH``.
"""

from __future__ import annotations

import json
import math
import os
import stat
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import List, Mapping, Optional

from soup_cli.utils.advise import CHOICES, TASK_CATEGORIES, Verdict

# Bounds — defence against unbounded growth.
_MAX_HISTORY_ROWS = 10_000
_MAX_FILE_BYTES = 16 * 1024 * 1024  # 16 MiB
_MAX_NOTES_CHARS = 4096
# Per-line cap on history reads — defence against an adversarially-crafted
# history file with a single multi-MiB JSON object (security-review MEDIUM
# fix — matches v0.40.3 TraceLogWriter policy).
_MAX_HISTORY_LINE_BYTES = 64 * 1024


@dataclass(frozen=True)
class HistoryEntry:
    """One verdict-decision row persisted to the history log."""

    timestamp: str  # ISO-8601 UTC
    project: str  # cwd basename at write time
    choice: str
    task_category: str
    confidence: float
    reason: str
    reverse_when: str
    accepted: bool  # did the user accept the verdict?
    outcome: Optional[float]  # post-hoc measured outcome in [-1, 1] (or None)
    notes: str = ""


def history_path() -> str:
    """Resolve the history log path, with env override + containment.

    Mirrors v0.36.0 ``SOUP_BATCH_CACHE_PATH`` containment: the override
    must sit under ``$HOME`` / ``$CWD`` / ``tempfile.gettempdir()``;
    out-of-bounds values fall through to the default.
    """
    default = os.path.join(os.path.expanduser("~"), ".soup", "advise_history.jsonl")
    override = os.environ.get("SOUP_ADVISE_HISTORY_PATH")
    if not override:
        return default
    if "\x00" in override:
        return default
    try:
        candidate_real = os.path.realpath(override)
    except (OSError, ValueError):
        return default
    safe_bases = [
        os.path.realpath(os.path.expanduser("~")),
        os.path.realpath(os.getcwd()),
        os.path.realpath(tempfile.gettempdir()),
    ]
    for base in safe_bases:
        try:
            if os.path.commonpath([candidate_real, base]) == base:
                return candidate_real
        except ValueError:
            continue
    return default


def _validate_notes(notes: str) -> str:
    if not isinstance(notes, str):
        raise TypeError("notes must be a string")
    if "\x00" in notes:
        raise ValueError("notes must not contain NUL")
    # Strip CRLF so a crafted note can't break the JSONL line layout when
    # operators inspect the file via tools that split on bare \n.
    cleaned = notes.replace("\r", "").replace("\n", " ")
    if len(cleaned) > _MAX_NOTES_CHARS:
        raise ValueError(f"notes exceeds {_MAX_NOTES_CHARS} characters")
    return cleaned


def _project_name() -> str:
    """Project label = basename of cwd (truncated to keep history compact)."""
    name = os.path.basename(os.path.realpath(os.getcwd())) or "soup-project"
    return name[:128]


def record_verdict(
    verdict: Verdict,
    *,
    accepted: bool,
    outcome: Optional[float] = None,
    notes: str = "",
    path: Optional[str] = None,
) -> HistoryEntry:
    """Append a verdict to the history log under an exclusive file lock.

    Raises ``TypeError`` / ``ValueError`` on shape errors so the CLI layer
    can surface them as friendly messages.

    Note: the ``path`` keyword is for trusted-internal callers (tests +
    `commands/advise.py`). Operator-supplied paths arrive via the
    ``SOUP_ADVISE_HISTORY_PATH`` env var which IS containment-checked in
    :func:`history_path`. Direct ``path=`` callers bypass that gate by
    design — do not expose this kwarg through any user-facing surface.
    """
    if not isinstance(verdict, Verdict):
        raise TypeError("verdict must be a Verdict instance")
    if not isinstance(accepted, bool):
        raise TypeError("accepted must be a bool")
    if outcome is not None:
        if isinstance(outcome, bool):
            raise TypeError("outcome must not be bool")
        if not isinstance(outcome, (int, float)):
            raise TypeError("outcome must be a number or None")
        if not math.isfinite(outcome):
            raise ValueError("outcome must be finite")
        if not (-1.0 <= outcome <= 1.0):
            raise ValueError("outcome must be in [-1, 1]")
    if verdict.choice not in CHOICES:
        raise ValueError(f"verdict.choice must be one of {CHOICES}")
    if verdict.task_category not in TASK_CATEGORIES:
        raise ValueError(
            f"verdict.task_category must be one of {TASK_CATEGORIES}"
        )
    cleaned_notes = _validate_notes(notes)

    entry = HistoryEntry(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        project=_project_name(),
        choice=verdict.choice,
        task_category=verdict.task_category,
        confidence=float(verdict.confidence),
        reason=verdict.reason,
        reverse_when=verdict.reverse_when,
        accepted=accepted,
        outcome=outcome,
        notes=cleaned_notes,
    )

    target = path if path is not None else history_path()
    if not isinstance(target, str) or not target:
        raise ValueError("history path must be a non-empty string")
    if "\x00" in target:
        raise ValueError("history path must not contain NUL")

    # Symlink rejection on the RAW path BEFORE realpath (TOCTOU defence —
    # mirrors v0.53.7 #106 policy). FileNotFoundError is fine: we're about
    # to create the file.
    try:
        if stat.S_ISLNK(os.lstat(target).st_mode):
            raise ValueError("history path must not be a symlink")
    except FileNotFoundError:
        pass

    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)

    payload = json.dumps(asdict(entry), ensure_ascii=False)
    _append_with_lock(target, payload + "\n")
    # Best-effort POSIX 600 perms — matches v0.26.0 registry.db policy.
    try:
        os.chmod(target, 0o600)
    except OSError:
        pass
    return entry


def _append_with_lock(target: str, line: str) -> None:
    """Append a line to ``target`` under a best-effort exclusive file lock.

    POSIX: ``fcntl.flock(LOCK_EX)``. Windows: separate ``<target>.lock``
    file with ``msvcrt.locking`` — this avoids the seek-position pitfall
    where unlocking the data file at offset 0 wouldn't match the locked
    range (security-review MEDIUM fix). When locking is unavailable on
    the host, the append proceeds without a lock (POSIX-only writes
    < PIPE_BUF stay atomic at the OS level for append-mode anyway).
    """
    lock_handle = None
    lock_kind: Optional[str] = None
    if os.name == "nt":
        try:
            import msvcrt  # type: ignore[import-not-found]

            lock_path = target + ".lock"
            lock_handle = open(lock_path, "a+")
            try:
                msvcrt.locking(lock_handle.fileno(), msvcrt.LK_LOCK, 1)
                lock_kind = "nt"
            except OSError:
                lock_handle.close()
                lock_handle = None
        except (ImportError, OSError):
            lock_handle = None
    try:
        with open(target, "a", encoding="utf-8") as fh:
            posix_locked = False
            if os.name == "posix":
                try:
                    import fcntl  # type: ignore[import-not-found]

                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                    posix_locked = True
                except (ImportError, OSError):
                    posix_locked = False
            try:
                fh.seek(0, os.SEEK_END)
                fh.write(line)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
            finally:
                if posix_locked:
                    try:
                        import fcntl  # type: ignore[import-not-found]

                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass
    finally:
        if lock_kind == "nt" and lock_handle is not None:
            try:
                import msvcrt  # type: ignore[import-not-found]

                # Seek back to the byte that was locked before unlocking, so
                # msvcrt sees the same range it locked initially.
                lock_handle.seek(0)
                msvcrt.locking(lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            try:
                lock_handle.close()
            except OSError:
                pass


def load_history(
    *, limit: int = 100, path: Optional[str] = None
) -> List[HistoryEntry]:
    """Read the most recent ``limit`` entries, newest first.

    Returns an empty list when the file does not exist. Malformed lines are
    skipped silently (a corrupted line from a partial write must NOT crash
    `soup advise compare`).
    """
    if isinstance(limit, bool):
        raise TypeError("limit must not be bool")
    if not isinstance(limit, int):
        raise TypeError("limit must be int")
    if not (1 <= limit <= _MAX_HISTORY_ROWS):
        raise ValueError(f"limit must be in [1, {_MAX_HISTORY_ROWS}]")

    target = path if path is not None else history_path()
    if not isinstance(target, str) or not target:
        raise ValueError("history path must be a non-empty string")
    if "\x00" in target:
        raise ValueError("history path must not contain NUL")
    try:
        if stat.S_ISLNK(os.lstat(target).st_mode):
            raise ValueError("history path must not be a symlink")
    except FileNotFoundError:
        return []

    if os.path.getsize(target) > _MAX_FILE_BYTES:
        raise ValueError(
            f"history file exceeds {_MAX_FILE_BYTES} bytes; rotate or trim"
        )

    entries: List[HistoryEntry] = []
    with open(target, "r", encoding="utf-8") as fh:
        for line in fh:
            if len(line) > _MAX_HISTORY_LINE_BYTES:
                # Oversize line — likely corruption or attack; skip rather
                # than feed it to json.loads (memory-DoS defence).
                continue
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            try:
                entries.append(_row_to_entry(row))
            except (KeyError, TypeError, ValueError):
                continue
    # Newest first.
    entries.reverse()
    return entries[:limit]


def _row_to_entry(row: Mapping[str, object]) -> HistoryEntry:
    choice = row.get("choice")
    if not isinstance(choice, str) or choice not in CHOICES:
        raise ValueError("unknown choice in history row")
    task_category = row.get("task_category")
    if not isinstance(task_category, str) or task_category not in TASK_CATEGORIES:
        raise ValueError("unknown task_category in history row")
    confidence_raw = row.get("confidence", 0.0)
    if isinstance(confidence_raw, bool):
        raise ValueError("confidence must not be bool")
    confidence = float(confidence_raw) if isinstance(confidence_raw, (int, float)) else 0.0
    outcome_raw = row.get("outcome")
    outcome: Optional[float]
    if outcome_raw is None:
        outcome = None
    elif isinstance(outcome_raw, bool):
        raise ValueError("outcome must not be bool")
    elif isinstance(outcome_raw, (int, float)):
        outcome = float(outcome_raw)
    else:
        outcome = None
    return HistoryEntry(
        timestamp=str(row.get("timestamp", "")),
        project=str(row.get("project", "")),
        choice=str(choice),
        task_category=str(task_category),
        confidence=confidence,
        reason=str(row.get("reason", "")),
        reverse_when=str(row.get("reverse_when", "")),
        accepted=bool(row.get("accepted", False)),
        outcome=outcome,
        notes=str(row.get("notes", "")),
    )


def summarise_history(entries: List[HistoryEntry]) -> Mapping[str, int]:
    """Aggregate choice counts for the compare panel."""
    counts: dict[str, int] = {choice: 0 for choice in CHOICES}
    for entry in entries:
        if entry.choice in counts:
            counts[entry.choice] += 1
    return counts


# Backward-compatible alias (American spelling) — kept for forward compat
# with external scripts; new code should use ``summarise_history``.
summarize_history = summarise_history
