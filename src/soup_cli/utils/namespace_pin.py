"""Namespace-pin verification (v0.60.0 Part D — anti-AI-Jacking).

Threat model: an attacker watches a popular repo on HuggingFace, waits until
the original owner deletes it (or the repo expires), then re-creates the same
``owner/name`` with malicious weights. Anyone with ``soup`` pinned to that
namespace silently pulls poison on next ``soup train`` / ``soup download``.

Defence: trust-on-first-use. The first time Soup sees a ``owner/name`` repo
it records the author + created_at fingerprint. On every subsequent load,
``verify_namespace`` compares the current author + created_at to the
recorded pin. Mismatch → operator confirmation required via
``--allow-namespace-shift <new-author>``. The opt-in must name the new
author explicitly — a free-for-all ``--allow-namespace-shift`` flag would
defeat the whole control.

SQLite-backed; survives across runs. The DB path can be overridden via
``SOUP_NAMESPACE_PIN_DB`` (validated through the same containment policy
as ``SOUP_AUDIT_LOG_PATH``). Default: ``~/.soup/namespace_pin.db``.

Public surface:
- ``NamespacePin`` frozen dataclass.
- ``NamespacePinStore`` SQLite wrapper.
- ``record_repo_first_seen`` + ``verify_namespace`` decision helpers.
- ``NamespaceVerifyReport`` frozen dataclass.
"""

from __future__ import annotations

import os
import re
import sqlite3
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,127}/[A-Za-z0-9][A-Za-z0-9._\-]{0,127}$")
_AUTHOR_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,127}$")
_MAX_REPO_LEN = 256
_MAX_AUTHOR_LEN = 128
_MAX_TIMESTAMP_LEN = 64

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS namespace_pins (
    repo_id     TEXT PRIMARY KEY,
    author      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    first_seen  TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class NamespacePin:
    """One repo's trust-on-first-use fingerprint."""

    repo_id: str
    author: str
    created_at: str
    first_seen: str

    def __post_init__(self) -> None:
        if not isinstance(self.repo_id, str) or not self.repo_id:
            raise ValueError("repo_id must be non-empty str")
        if "\x00" in self.repo_id:
            raise ValueError("repo_id must not contain null bytes")
        if len(self.repo_id) > _MAX_REPO_LEN:
            raise ValueError(f"repo_id too long (> {_MAX_REPO_LEN} chars)")
        if not isinstance(self.author, str) or not self.author:
            raise ValueError("author must be non-empty str")
        if "\x00" in self.author:
            raise ValueError("author must not contain null bytes")
        if len(self.author) > _MAX_AUTHOR_LEN:
            raise ValueError(f"author too long (> {_MAX_AUTHOR_LEN} chars)")
        for ts_value, ts_name in (
            (self.created_at, "created_at"),
            (self.first_seen, "first_seen"),
        ):
            if not isinstance(ts_value, str) or not ts_value:
                raise ValueError(f"{ts_name} must be non-empty str")
            if "\x00" in ts_value:
                raise ValueError(f"{ts_name} must not contain null bytes")
            if len(ts_value) > _MAX_TIMESTAMP_LEN:
                raise ValueError(f"{ts_name} too long (> {_MAX_TIMESTAMP_LEN})")


@dataclass(frozen=True)
class NamespaceVerifyReport:
    """Outcome of ``verify_namespace``."""

    repo_id: str
    ok: bool
    reason: str
    recorded: Optional[NamespacePin]


def _validate_repo_id(repo_id: object) -> str:
    if not isinstance(repo_id, str) or not repo_id:
        raise ValueError("repo_id must be non-empty str")
    if "\x00" in repo_id:
        raise ValueError("repo_id must not contain null bytes")
    if len(repo_id) > _MAX_REPO_LEN:
        raise ValueError(f"repo_id too long (> {_MAX_REPO_LEN} chars)")
    return repo_id


def _validate_author_override(value: object) -> str:
    """Validate the explicit ``--allow-namespace-shift`` author kwarg.

    Strict: must be a string matching the author regex. ``True`` /
    ``False`` / ``None`` are rejected so callers cannot smuggle a
    free-for-all bypass via boolean coercion (security review fix).
    """
    if isinstance(value, bool):
        raise TypeError(
            "allow_namespace_shift must be str (new author name), not bool"
        )
    if not isinstance(value, str) or not value:
        raise ValueError("allow_namespace_shift must be non-empty str")
    if "\x00" in value:
        raise ValueError("allow_namespace_shift must not contain null bytes")
    if not _AUTHOR_RE.match(value):
        raise ValueError("allow_namespace_shift must match author regex")
    return value


class NamespacePinStore:
    """SQLite wrapper. Single-connection, single-thread."""

    def __init__(self, path: str) -> None:
        if not isinstance(path, str) or not path:
            raise ValueError("path must be non-empty str")
        if "\x00" in path:
            raise ValueError("path must not contain null bytes")
        # Reject paths outside $HOME / $CWD / $TMPDIR — defends against a
        # caller smuggling /etc/passwd through the public constructor
        # (security-review HIGH fix; mirrors v0.36.0 SOUP_BATCH_CACHE_PATH
        # / v0.54.0 / v0.59.0 audit-log policy).
        if _validate_db_path_override(path) is None:
            raise ValueError(
                f"path {os.path.basename(path)!r} must stay under "
                "$HOME / $CWD / $TMPDIR"
            )
        parent = os.path.dirname(os.path.realpath(path)) or "."
        os.makedirs(parent, exist_ok=True)
        # TOCTOU: reject a pre-placed symlink at the DB path BEFORE
        # sqlite3.connect (which would happily write through the link).
        if os.path.lexists(path):
            link_st = os.lstat(path)
            if stat.S_ISLNK(link_st.st_mode):
                raise ValueError("DB path is a symlink (TOCTOU defence)")
        self._path = path
        self._lock_path = path + ".lock"
        self._conn = sqlite3.connect(path)
        # v0.71.2 #191 — concurrent-writer support. WAL lets readers proceed
        # while one writer holds the DB, and busy_timeout makes a competing
        # writer retry (instead of raising "database is locked" instantly).
        # Both are best-effort: WAL is a no-op on :memory: DBs, and a failed
        # PRAGMA must not break the store.
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
        except sqlite3.Error:
            pass
        self._conn.executescript(_SCHEMA_SQL)
        # POSIX-only 0600 perms for the DB file.
        if os.name != "nt":
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "NamespacePinStore":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def get(self, repo_id: str) -> Optional[NamespacePin]:
        _validate_repo_id(repo_id)
        cur = self._conn.execute(
            "SELECT repo_id, author, created_at, first_seen "
            "FROM namespace_pins WHERE repo_id = ?",
            (repo_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return NamespacePin(
            repo_id=row[0],
            author=row[1],
            created_at=row[2],
            first_seen=row[3],
        )

    @contextmanager
    def _cross_process_lock(self):
        """Best-effort cross-process exclusive lock around get+insert (#191).

        POSIX: ``fcntl.flock(LOCK_EX)`` on a ``<db>.lock`` file. Windows:
        ``msvcrt.locking`` on the same sidecar. Mirrors the v0.54.0
        ``advise_history._append_with_lock`` pattern (separate lock file so the
        unlock byte-range matches the locked one). When locking is unavailable
        (exotic FS), the write still proceeds — WAL + busy_timeout already
        serialise concurrent writers at the SQLite level.
        """
        handle = None
        kind: Optional[str] = None
        try:
            handle = open(self._lock_path, "a+")
        except OSError:
            handle = None
        if handle is not None:
            try:
                if os.name == "nt":
                    import msvcrt  # type: ignore[import-not-found]

                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                    kind = "nt"
                else:
                    import fcntl  # type: ignore[import-not-found]

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    kind = "posix"
            except (ImportError, OSError):
                kind = None
        try:
            yield
        finally:
            if handle is not None:
                try:
                    if kind == "nt":
                        import msvcrt  # type: ignore[import-not-found]

                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    elif kind == "posix":
                        import fcntl  # type: ignore[import-not-found]

                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except (ImportError, OSError):
                    # Best-effort unlock; ImportError can't happen here (the
                    # lock side already imported successfully) but catching it
                    # guarantees handle.close() still runs (code-review L2).
                    pass
                handle.close()

    def put(self, pin: NamespacePin) -> None:
        """Insert a new pin or update author/created_at on an existing one.

        ``first_seen`` is preserved across updates — the first time a repo
        was observed is the trust anchor and MUST NOT be overwritten by a
        later successful verify (security review fix). To replace the
        author / created_at fingerprint, callers use ``put`` with a
        ``NamespacePin`` whose ``first_seen`` matches the existing record
        OR with the operator-confirmed opt-in path inside
        ``verify_namespace``.

        The get+insert runs under a cross-process lock (#191) so a concurrent
        first-insert from another process cannot clobber the trust anchor.
        """
        if not isinstance(pin, NamespacePin):
            raise TypeError("pin must be NamespacePin")
        with self._cross_process_lock():
            # Preserve the original first_seen across overwrites — read INSIDE
            # the lock so a racing writer's row is observed (closes the
            # first-insert TOCTOU).
            existing = self.get(pin.repo_id)
            first_seen = existing.first_seen if existing is not None else pin.first_seen
            self._conn.execute(
                "INSERT OR REPLACE INTO namespace_pins "
                "(repo_id, author, created_at, first_seen) VALUES (?, ?, ?, ?)",
                (pin.repo_id, pin.author, pin.created_at, first_seen),
            )
            self._conn.commit()


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _is_backward(current: str, recorded: str) -> bool:
    """Return True when ``current`` represents a moment strictly before ``recorded``.

    Parses ISO-8601 timestamps with ``datetime.fromisoformat`` for offset-aware
    comparison; falls back to lexicographic compare when either string fails
    to parse (defence-in-depth for legacy / hand-edited rows).
    """
    try:
        return datetime.fromisoformat(current) < datetime.fromisoformat(recorded)
    except ValueError:
        return current < recorded


def _created_at_differs(current: str, recorded: str) -> bool:
    """Return True when ``current`` is a DIFFERENT moment than ``recorded``.

    A pinned repo's ``created_at`` is immutable, so ANY drift — forward OR
    backward — means the namespace was re-created (the AI-Jacking / repo-
    recreation attack this pin targets). The previous gate only flagged a
    *backward* jump, but a recreated repo gets a *later* timestamp and sailed
    straight through. Offset-aware equality via ``fromisoformat``; falls back
    to string inequality when either value fails to parse (legacy rows).
    """
    try:
        return datetime.fromisoformat(current) != datetime.fromisoformat(recorded)
    except ValueError:
        return current != recorded


def record_repo_first_seen(
    store: NamespacePinStore,
    *,
    repo_id: str,
    author: str,
    created_at: str,
) -> NamespacePin:
    """Trust-on-first-use record. Returns the (possibly pre-existing) pin.

    If the repo was already pinned, returns the existing pin without
    overwriting. Operators who want to refresh use ``verify_namespace``
    with ``allow_namespace_shift=<author>``.
    """
    _validate_repo_id(repo_id)
    existing = store.get(repo_id)
    if existing is not None:
        return existing
    pin = NamespacePin(
        repo_id=repo_id,
        author=author,
        created_at=created_at,
        first_seen=_now_iso(),
    )
    store.put(pin)
    return pin


def verify_namespace(
    store: NamespacePinStore,
    *,
    repo_id: str,
    current_author: str,
    current_created_at: str,
    allow_namespace_shift: Optional[str] = None,
) -> NamespaceVerifyReport:
    """Decide whether ``repo_id`` can be loaded.

    Decision matrix:

    - Unknown repo: trust-on-first-use. Record and return ok=True.
    - Known repo, author + created_at match: ok=True.
    - Known repo, author changed: ok=False unless
      ``allow_namespace_shift`` names the new author exactly (case-sensitive).
    - Known repo, created_at jumped backward: ok=False unless
      ``allow_namespace_shift`` matches the current author. A backward
      jump is a strong signal of namespace re-creation.

    When the opt-in path fires, the recorded pin is UPDATED to the new
    author + created_at so subsequent loads pass without re-prompting.

    Args:
        store: open ``NamespacePinStore``.
        repo_id: ``owner/name`` HuggingFace repo id.
        current_author: author reported by the Hub today.
        current_created_at: created_at reported by the Hub today (ISO 8601).
        allow_namespace_shift: explicit author name to accept on shift.
            ``None`` means strict (refuse on shift). A bool is rejected so
            callers cannot smuggle a free-for-all bypass.

    Returns:
        ``NamespaceVerifyReport``.
    """
    _validate_repo_id(repo_id)
    if not isinstance(current_author, str) or not current_author:
        raise ValueError("current_author must be non-empty str")
    if not isinstance(current_created_at, str) or not current_created_at:
        raise ValueError("current_created_at must be non-empty str")
    override: Optional[str] = None
    if allow_namespace_shift is not None:
        override = _validate_author_override(allow_namespace_shift)

    existing = store.get(repo_id)
    if existing is None:
        # Trust on first use.
        pin = NamespacePin(
            repo_id=repo_id,
            author=current_author,
            created_at=current_created_at,
            first_seen=_now_iso(),
        )
        store.put(pin)
        return NamespaceVerifyReport(
            repo_id=repo_id,
            ok=True,
            reason="trust on first use (no prior pin)",
            recorded=pin,
        )

    author_match = existing.author == current_author
    # Flag ANY created_at drift (forward or backward): a pinned repo's creation
    # time is immutable, so a changed timestamp means the namespace was
    # re-created. The old `_is_backward`-only check missed the repo-recreation
    # attack (which produces a LATER timestamp).
    created_at_changed = _created_at_differs(current_created_at, existing.created_at)

    if author_match and not created_at_changed:
        return NamespaceVerifyReport(
            repo_id=repo_id,
            ok=True,
            reason="namespace matches recorded pin",
            recorded=existing,
        )

    # Mismatch — either author changed or created_at drifted (in either
    # direction). Author comparison is case-insensitive: HF Hub authors are
    # case-insensitive, and a user typing the wrong case should not
    # fail the gate (security-review LOW fix).
    if override is not None and override.lower() == current_author.lower():
        # Operator-confirmed shift. Update the pin (preserves first_seen).
        new_pin = NamespacePin(
            repo_id=repo_id,
            author=current_author,
            created_at=current_created_at,
            first_seen=existing.first_seen,
        )
        store.put(new_pin)
        return NamespaceVerifyReport(
            repo_id=repo_id,
            ok=True,
            reason="namespace shift opt-in accepted",
            recorded=new_pin,
        )

    if not author_match:
        reason = (
            f"author changed: recorded {existing.author!r}, "
            f"current {current_author!r}; pass "
            f"`--allow-namespace-shift {current_author}` to override"
        )
    else:
        reason = (
            "created_at jumped backward (likely namespace re-creation); "
            f"recorded {existing.created_at!r}, current {current_created_at!r}; "
            f"pass `--allow-namespace-shift {current_author}` to override"
        )
    return NamespaceVerifyReport(
        repo_id=repo_id,
        ok=False,
        reason=reason,
        recorded=existing,
    )


def _validate_db_path_override(override: str) -> Optional[str]:
    """Apply v0.36.0 ``SOUP_BATCH_CACHE_PATH`` / v0.54.0 / v0.59.0 policy."""
    if not isinstance(override, str) or not override:
        return None
    if "\x00" in override:
        return None
    if len(override) > 4096:
        return None
    # Control character rejection
    if any(ord(c) < 0x20 or ord(c) == 0x7f for c in override):
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
    return None


def default_db_path() -> str:
    """Resolve the pin DB path (env override first, else ``~/.soup/namespace_pin.db``)."""
    try:
        override = os.environ.get("SOUP_NAMESPACE_PIN_DB")
    except ValueError:
        override = None
    if override:
        validated = _validate_db_path_override(override)
        if validated is not None:
            return validated
    return os.path.join(os.path.expanduser("~"), ".soup", "namespace_pin.db")
