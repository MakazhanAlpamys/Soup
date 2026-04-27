"""SQLite-backed registry store.

Schema (see Part A of v0.26.0 plan):

    registry_entries(id, name, tag, base_model, task, run_id, config_json,
                     config_hash, data_hash, entry_hash, created_at, notes)
    registry_artifacts(id, entry_id, kind, path, sha256, size_bytes,
                       created_at)
    registry_lineage(id, child_id, parent_id, relation, created_at)
    registry_tags(entry_id, tag)

The module is pure-Python / stdlib only — no heavy imports, so it stays
fast for CLI use.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from soup_cli.registry.hashing import hash_config, hash_entry, hash_file
from soup_cli.utils.constants import SOUP_DIR

if TYPE_CHECKING:
    from soup_cli.experiment.tracker import ExperimentTracker


class AmbiguousRefError(ValueError):
    """Raised when a reference matches more than one registry entry."""

REGISTRY_DB_FILENAME = "registry.db"

_VALID_KINDS = frozenset(
    {
        "adapter", "merged", "gguf", "awq", "gptq", "onnx", "dataset", "config",
        "eval_results", "tensorrt",
    }
)
_VALID_RELATIONS = frozenset(
    {"forked_from", "merged_from", "evaluated_with", "promoted_from"}
)

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-.]{0,127}$")
_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-.]{0,63}$")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS registry_entries (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    base_model   TEXT NOT NULL,
    task         TEXT NOT NULL,
    run_id       TEXT,
    config_json  TEXT NOT NULL,
    config_hash  TEXT NOT NULL,
    data_hash    TEXT,
    entry_hash   TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    notes        TEXT
);

CREATE TABLE IF NOT EXISTS registry_artifacts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id   TEXT NOT NULL REFERENCES registry_entries(id) ON DELETE CASCADE,
    kind       TEXT NOT NULL,
    path       TEXT NOT NULL,
    sha256     TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS registry_lineage (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    child_id   TEXT NOT NULL REFERENCES registry_entries(id) ON DELETE CASCADE,
    parent_id  TEXT NOT NULL REFERENCES registry_entries(id) ON DELETE CASCADE,
    relation   TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(child_id, parent_id, relation)
);

CREATE TABLE IF NOT EXISTS registry_tags (
    entry_id TEXT NOT NULL REFERENCES registry_entries(id) ON DELETE CASCADE,
    tag      TEXT NOT NULL,
    PRIMARY KEY (entry_id, tag)
);

CREATE INDEX IF NOT EXISTS idx_reg_entries_name ON registry_entries(name);
CREATE INDEX IF NOT EXISTS idx_reg_entries_run_id ON registry_entries(run_id);
CREATE INDEX IF NOT EXISTS idx_reg_artifacts_entry ON registry_artifacts(entry_id);
CREATE INDEX IF NOT EXISTS idx_reg_lineage_child ON registry_lineage(child_id);
CREATE INDEX IF NOT EXISTS idx_reg_lineage_parent ON registry_lineage(parent_id);
CREATE INDEX IF NOT EXISTS idx_reg_tags_tag ON registry_tags(tag);
"""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_name(name: str) -> None:
    """Raise ValueError if ``name`` is invalid for registry use."""
    if not name:
        raise ValueError("registry name cannot be empty")
    if len(name) > 128:
        raise ValueError("registry name too long (max 128 chars)")
    if "\x00" in name:
        raise ValueError("registry name contains invalid characters (null byte)")
    if not _NAME_RE.match(name):
        raise ValueError(
            f"registry name '{name}' is invalid - use letters, digits, "
            "'_', '-', '.' only; must start with alphanumeric"
        )


def validate_tag(tag: str) -> None:
    """Raise ValueError if ``tag`` is invalid."""
    if not tag:
        raise ValueError("registry tag cannot be empty")
    if len(tag) > 64:
        raise ValueError("registry tag too long (max 64 chars)")
    if "\x00" in tag:
        raise ValueError("registry tag contains invalid characters (null byte)")
    if not _TAG_RE.match(tag):
        raise ValueError(
            f"registry tag '{tag}' is invalid - use letters, digits, "
            "'_', '-', '.' only; must start with alphanumeric"
        )


def _is_under(path: Path, base: Path) -> bool:
    """Whether ``path`` is inside ``base`` using realpath + commonpath.

    Uses ``os.path.realpath`` + ``os.path.commonpath`` rather than
    ``Path.resolve() + relative_to()`` to survive Windows 8.3 short names.
    """
    try:
        resolved_path = os.path.realpath(str(path))
        resolved_base = os.path.realpath(str(base))
    except (OSError, ValueError):
        return False
    if os.name == "nt":
        resolved_path = resolved_path.lower()
        resolved_base = resolved_base.lower()
    try:
        common = os.path.commonpath([resolved_path, resolved_base])
    except ValueError:
        return False
    return common == resolved_base


def _escape_like(value: str) -> str:
    """Escape SQLite LIKE wildcards ``%`` and ``_`` in user-supplied strings."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def _default_db_path() -> Path:
    env = os.environ.get("SOUP_REGISTRY_DB_PATH")
    if env:
        return Path(env)
    soup_dir = Path.home() / SOUP_DIR
    soup_dir.mkdir(parents=True, exist_ok=True)
    return soup_dir / REGISTRY_DB_FILENAME


def _generate_entry_id() -> str:
    """Generate a short, collision-resistant entry id."""
    ts = datetime.now().strftime("%Y%m%d")
    return f"reg_{ts}_{secrets.token_hex(6)}"


class RegistryStore:
    """SQLite-backed store for registry entries, artifacts, lineage, tags.

    Usable as a context manager: ``with RegistryStore() as store: ...``
    guarantees the connection is closed even on exception paths.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else _default_db_path()
        # Cwd captured at construction time so containment checks remain
        # deterministic even if the process chdirs later.
        self._cwd_snapshot = Path.cwd()
        self._conn: Optional[sqlite3.Connection] = None
        try:
            self._ensure_schema()
            self._tighten_perms()
        except Exception:
            self.close()
            raise

    # -- connection handling ------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript(_SCHEMA_SQL)
        conn.commit()

    def _tighten_perms(self) -> None:
        """Set 600 permissions on the DB file on POSIX systems."""
        if os.name == "posix" and self.db_path.exists():
            try:
                os.chmod(self.db_path, 0o600)
            except OSError:
                pass  # best-effort — file may be on a filesystem without perms

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def __enter__(self) -> "RegistryStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # -- push / get / list --------------------------------------------------

    def push(
        self,
        *,
        name: str,
        tag: str,
        base_model: str,
        task: str,
        run_id: Optional[str],
        config: dict[str, Any],
        data_path: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> str:
        """Create a new registry entry. Returns the new entry's id."""
        validate_name(name)
        validate_tag(tag)
        if not base_model:
            raise ValueError("base_model is required")
        if not task:
            raise ValueError("task is required")

        entry_id = _generate_entry_id()
        now = datetime.now().isoformat()

        config_hash = hash_config(config)
        data_hash = hash_file(data_path) if data_path is not None else None
        entry_hash = hash_entry(
            config=config, data_path=data_path, base_model=base_model,
        )

        conn = self._get_conn()
        conn.execute(
            """INSERT INTO registry_entries
               (id, name, base_model, task, run_id, config_json,
                config_hash, data_hash, entry_hash, created_at, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry_id, name, base_model, task, run_id,
                json.dumps(config, default=str, sort_keys=True),
                config_hash, data_hash, entry_hash, now, notes,
            ),
        )
        conn.execute(
            "INSERT INTO registry_tags (entry_id, tag) VALUES (?, ?)",
            (entry_id, tag),
        )
        conn.commit()
        return entry_id

    def add_tag(self, entry_id: str, tag: str) -> None:
        validate_tag(tag)
        if self.get(entry_id) is None:
            raise ValueError(f"registry entry not found: {entry_id}")
        conn = self._get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO registry_tags (entry_id, tag) VALUES (?, ?)",
            (entry_id, tag),
        )
        conn.commit()

    def get(self, entry_id: str) -> Optional[dict]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM registry_entries WHERE id = ?", (entry_id,),
        ).fetchone()
        if row is None:
            return None
        return self._hydrate(dict(row))

    def _hydrate(self, row: dict) -> dict:
        """Attach tags + artifacts to an entry row."""
        conn = self._get_conn()
        tags = [
            r[0] for r in conn.execute(
                "SELECT tag FROM registry_tags WHERE entry_id = ? ORDER BY tag",
                (row["id"],),
            ).fetchall()
        ]
        row["tags"] = tags
        return row

    def list(
        self,
        *,
        name: Optional[str] = None,
        tag: Optional[str] = None,
        base: Optional[str] = None,
        task: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Return entries filtered by name/tag/base/task."""
        conn = self._get_conn()
        clauses: list[str] = []
        params: list[object] = []

        if name:
            clauses.append("e.name = ?")
            params.append(name)
        if base:
            clauses.append("e.base_model = ?")
            params.append(base)
        if task:
            clauses.append("e.task = ?")
            params.append(task)
        if tag:
            clauses.append(
                "e.id IN (SELECT entry_id FROM registry_tags WHERE tag = ?)"
            )
            params.append(tag)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            f"SELECT e.* FROM registry_entries e {where} "
            "ORDER BY e.created_at DESC, e.id DESC LIMIT ?"
        )
        params.append(int(limit))

        rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._hydrate(dict(r)) for r in rows]

    def search(self, query: str, *, limit: int = 50) -> list[dict]:
        """Case-insensitive search across name, base_model, task, notes.

        User query is parameterised AND ``%`` / ``_`` wildcards are escaped
        so that a query like ``%`` does not match every row.
        """
        if not query:
            return []
        pattern = f"%{_escape_like(query.lower())}%"
        conn = self._get_conn()
        rows = conn.execute(
            r"""SELECT * FROM registry_entries
                WHERE LOWER(name) LIKE ? ESCAPE '\'
                   OR LOWER(base_model) LIKE ? ESCAPE '\'
                   OR LOWER(task) LIKE ? ESCAPE '\'
                   OR LOWER(COALESCE(notes, '')) LIKE ? ESCAPE '\'
                ORDER BY created_at DESC LIMIT ?""",
            (pattern, pattern, pattern, pattern, int(limit)),
        ).fetchall()
        return [self._hydrate(dict(r)) for r in rows]

    def delete(self, entry_id: str) -> bool:
        """Delete an entry. FK ``ON DELETE CASCADE`` removes child rows."""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM registry_entries WHERE id = ?", (entry_id,),
        )
        conn.commit()
        return cursor.rowcount > 0

    # -- resolve ------------------------------------------------------------

    def resolve(self, ref: str) -> Optional[str]:
        """Resolve a reference (id, prefix, name:tag, or registry://id) to an id.

        Returns ``None`` on miss. Raises :class:`AmbiguousRefError` when a
        prefix matches more than one entry.
        """
        if not ref:
            return None
        if ref.startswith("registry://"):
            ref = ref[len("registry://"):]

        conn = self._get_conn()

        # Exact id match
        row = conn.execute(
            "SELECT id FROM registry_entries WHERE id = ?", (ref,),
        ).fetchone()
        if row:
            return row["id"]

        # name:tag form — pick most recent
        if ":" in ref:
            name, _, tag = ref.partition(":")
            try:
                validate_name(name)
                validate_tag(tag)
            except ValueError:
                return None
            row = conn.execute(
                """SELECT e.id FROM registry_entries e
                   JOIN registry_tags t ON t.entry_id = e.id
                   WHERE e.name = ? AND t.tag = ?
                   ORDER BY e.created_at DESC, e.rowid DESC LIMIT 1""",
                (name, tag),
            ).fetchone()
            return row["id"] if row else None

        # Prefix — must be unambiguous. Escape LIKE wildcards so a query
        # like "%" doesn't match everything.
        pattern = f"{_escape_like(ref)}%"
        rows = conn.execute(
            r"SELECT id FROM registry_entries "
            r"WHERE id LIKE ? ESCAPE '\' ORDER BY created_at DESC",
            (pattern,),
        ).fetchall()
        if len(rows) == 1:
            return rows[0]["id"]
        if len(rows) > 1:
            raise AmbiguousRefError(
                f"reference '{ref}' matches {len(rows)} entries - "
                "supply a longer prefix or the full id"
            )
        return None

    # -- artifacts ----------------------------------------------------------

    def add_artifact(
        self,
        *,
        entry_id: str,
        kind: str,
        path: str,
        enforce_cwd: bool = True,
    ) -> int:
        """Attach an artifact to an entry. Returns the artifact row id.

        ``enforce_cwd`` (on by default) rejects files outside the cwd captured
        at store construction time. Pass ``enforce_cwd=False`` explicitly when
        registering artifacts from known-safe locations (e.g. writer-supplied
        absolute paths the library produced itself).
        """
        if kind not in _VALID_KINDS:
            raise ValueError(
                f"unknown artifact kind '{kind}'. "
                f"Allowed: {sorted(_VALID_KINDS)}"
            )
        if self.get(entry_id) is None:
            raise ValueError(f"registry entry not found: {entry_id}")

        artifact_path = Path(path)
        if not artifact_path.exists() or not artifact_path.is_file():
            raise FileNotFoundError(f"artifact not found: {path}")

        if enforce_cwd and not _is_under(artifact_path, self._cwd_snapshot):
            raise ValueError(
                f"artifact '{path}' is outside cwd - refusing to register"
            )

        digest = hash_file(str(artifact_path))
        size = artifact_path.stat().st_size
        now = datetime.now().isoformat()
        # Store the realpath (not Path.resolve) for Windows 8.3 short-name
        # consistency with the containment check above.
        canonical = os.path.realpath(str(artifact_path))
        conn = self._get_conn()
        cursor = conn.execute(
            """INSERT INTO registry_artifacts
               (entry_id, kind, path, sha256, size_bytes, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (entry_id, kind, canonical, digest, size, now),
        )
        conn.commit()
        return cursor.lastrowid

    def get_artifacts(self, entry_id: str) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM registry_artifacts WHERE entry_id = ? "
            "ORDER BY created_at ASC",
            (entry_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- lineage ------------------------------------------------------------

    def add_lineage(
        self, *, child_id: str, parent_id: str, relation: str,
    ) -> None:
        """Record a ``child → parent`` lineage edge.

        Rejects self-references and indirect cycles (e.g. A→B→A) so BFS
        walks in :meth:`get_ancestors` and :meth:`get_descendants` always
        terminate on a DAG.
        """
        if relation not in _VALID_RELATIONS:
            raise ValueError(
                f"unknown lineage relation '{relation}'. "
                f"Allowed: {sorted(_VALID_RELATIONS)}"
            )
        if child_id == parent_id:
            raise ValueError("lineage cannot reference self")
        if self.get(child_id) is None:
            raise ValueError(f"child entry not found: {child_id}")
        if self.get(parent_id) is None:
            raise ValueError(f"parent entry not found: {parent_id}")
        # Cycle check: if parent already has child as an ancestor, adding this
        # edge would close a loop. Walk from parent_id upward and reject if
        # child_id is reachable.
        for anc in self.get_ancestors(parent_id):
            if anc["id"] == child_id:
                raise ValueError(
                    "lineage would introduce a cycle "
                    f"({child_id} -> {parent_id} -> ... -> {child_id})"
                )
        now = datetime.now().isoformat()
        conn = self._get_conn()
        conn.execute(
            """INSERT OR IGNORE INTO registry_lineage
               (child_id, parent_id, relation, created_at)
               VALUES (?, ?, ?, ?)""",
            (child_id, parent_id, relation, now),
        )
        conn.commit()

    def get_ancestors(self, entry_id: str, *, max_depth: int = 10) -> list[dict]:
        """BFS walk upwards from entry_id across lineage parents."""
        conn = self._get_conn()
        seen: set[str] = set()
        frontier = [entry_id]
        ancestors: list[dict] = []
        for _ in range(max_depth):
            if not frontier:
                break
            placeholders = ",".join("?" * len(frontier))
            rows = conn.execute(
                f"""SELECT e.*, l.relation, l.child_id AS via_child
                    FROM registry_lineage l
                    JOIN registry_entries e ON e.id = l.parent_id
                    WHERE l.child_id IN ({placeholders})""",
                tuple(frontier),
            ).fetchall()
            new_frontier: list[str] = []
            for row in rows:
                pid = row["id"]
                if pid in seen:
                    continue
                seen.add(pid)
                ancestors.append(self._hydrate(dict(row)))
                new_frontier.append(pid)
            frontier = new_frontier
        return ancestors

    def get_descendants(self, entry_id: str, *, max_depth: int = 10) -> list[dict]:
        """BFS walk downwards from entry_id across lineage children."""
        conn = self._get_conn()
        seen: set[str] = set()
        frontier = [entry_id]
        descendants: list[dict] = []
        for _ in range(max_depth):
            if not frontier:
                break
            placeholders = ",".join("?" * len(frontier))
            rows = conn.execute(
                f"""SELECT e.*, l.relation
                    FROM registry_lineage l
                    JOIN registry_entries e ON e.id = l.child_id
                    WHERE l.parent_id IN ({placeholders})""",
                tuple(frontier),
            ).fetchall()
            new_frontier: list[str] = []
            for row in rows:
                cid = row["id"]
                if cid in seen:
                    continue
                seen.add(cid)
                descendants.append(self._hydrate(dict(row)))
                new_frontier.append(cid)
            frontier = new_frontier
        return descendants

    def list_by_name(self, name: str) -> list[dict]:
        """All entries with a given name, oldest first (chronological)."""
        validate_name(name)
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM registry_entries WHERE name = ? "
            "ORDER BY created_at ASC",
            (name,),
        ).fetchall()
        return [self._hydrate(dict(r)) for r in rows]

    # -- tracker integration -----------------------------------------------

    def register_from_run(
        self,
        tracker: "ExperimentTracker",
        run_id: str,
        *,
        name: str,
        tag: str,
        notes: Optional[str] = None,
    ) -> str:
        """Create a registry entry from an existing training run."""
        run = tracker.get_run(run_id)
        if run is None:
            raise ValueError(f"run not found: {run_id}")

        import json as _json

        try:
            config = _json.loads(run.get("config_json") or "{}")
        except (TypeError, ValueError):
            config = {}

        base_model = run.get("base_model") or config.get("base") or "unknown"
        task = run.get("task") or config.get("task") or "sft"

        return self.push(
            name=name,
            tag=tag,
            base_model=base_model,
            task=task,
            run_id=run.get("run_id") or run_id,
            config=config,
            data_path=None,
            notes=notes,
        )

    def get_eval_results(self, entry_id: str) -> list[dict]:
        """Read eval rows attached via :class:`ExperimentTracker` by run_id.

        Registry stores entries; evals live in the experiments DB, linked
        by ``run_id``. This helper does the join on the caller's behalf.
        """
        entry = self.get(entry_id)
        if entry is None or not entry.get("run_id"):
            return []
        from soup_cli.experiment.tracker import ExperimentTracker

        return ExperimentTracker().get_eval_results(entry["run_id"])
