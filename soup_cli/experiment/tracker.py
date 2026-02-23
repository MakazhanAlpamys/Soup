"""Experiment tracking — stores runs in local SQLite.

Usage:
    tracker = ExperimentTracker()
    run_id = tracker.start_run(config_dict, device, device_name, gpu_info)
    tracker.log_metrics(run_id, step=10, loss=2.3, lr=1e-5)
    tracker.finish_run(run_id, initial_loss=2.5, final_loss=0.8, ...)
"""

from __future__ import annotations

import json
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from soup_cli.utils.constants import EXPERIMENTS_DB, SOUP_DIR

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    experiment_name TEXT,
    created_at      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'running',
    config_json     TEXT NOT NULL,
    device          TEXT,
    device_name     TEXT,
    gpu_memory      TEXT,
    initial_loss    REAL,
    final_loss      REAL,
    total_steps     INTEGER,
    duration_secs   REAL,
    output_dir      TEXT,
    base_model      TEXT,
    task            TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT NOT NULL REFERENCES runs(run_id),
    step      INTEGER NOT NULL,
    epoch     REAL,
    loss      REAL,
    lr        REAL,
    grad_norm REAL,
    speed     REAL,
    gpu_mem   TEXT,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_results (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT REFERENCES runs(run_id),
    model_path   TEXT NOT NULL,
    benchmark    TEXT NOT NULL,
    score        REAL NOT NULL,
    details_json TEXT,
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_metrics_run_id ON metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_eval_run_id ON eval_results(run_id);
"""


def _get_db_path() -> Path:
    """Return path to experiments DB, creating parent dir if needed."""
    import os

    # Allow override via env var (useful for tests and CI)
    env_path = os.environ.get("SOUP_DB_PATH")
    if env_path:
        return Path(env_path)

    soup_dir = Path.home() / SOUP_DIR
    soup_dir.mkdir(parents=True, exist_ok=True)
    return soup_dir / EXPERIMENTS_DB


def generate_run_id() -> str:
    """Generate a unique, sortable run ID: run_YYYYMMDD_HHMMSS_xxxxxxxx."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(4)
    return f"run_{ts}_{suffix}"


class ExperimentTracker:
    """SQLite-backed experiment tracker for training runs and evaluations."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or _get_db_path()
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        conn = self._get_conn()
        conn.executescript(_SCHEMA_SQL)
        conn.commit()

    def start_run(
        self,
        config_dict: dict,
        device: str,
        device_name: str,
        gpu_info: dict,
        experiment_name: Optional[str] = None,
    ) -> str:
        """Insert a new run and return its run_id."""
        run_id = generate_run_id()
        now = datetime.now().isoformat()
        config_json = json.dumps(config_dict, default=str)

        base_model = config_dict.get("base", "")
        task = config_dict.get("task", "sft")
        gpu_memory = gpu_info.get("memory_total", "")

        conn = self._get_conn()
        conn.execute(
            """INSERT INTO runs
               (run_id, experiment_name, created_at, status, config_json,
                device, device_name, gpu_memory, base_model, task)
               VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?, ?)""",
            (
                run_id, experiment_name, now, config_json,
                device, device_name, gpu_memory, base_model, task,
            ),
        )
        conn.commit()
        return run_id

    def log_metrics(
        self,
        run_id: str,
        step: int,
        epoch: float = 0.0,
        loss: float = 0.0,
        lr: float = 0.0,
        grad_norm: float = 0.0,
        speed: float = 0.0,
        gpu_mem: str = "",
    ) -> None:
        """Log a single metrics row for the given run."""
        now = datetime.now().isoformat()
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO metrics
               (run_id, step, epoch, loss, lr, grad_norm, speed, gpu_mem, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, step, epoch, loss, lr, grad_norm, speed, gpu_mem, now),
        )
        conn.commit()

    def finish_run(
        self,
        run_id: str,
        initial_loss: float,
        final_loss: float,
        total_steps: int,
        duration_secs: float,
        output_dir: str,
    ) -> None:
        """Mark run as completed and fill summary fields."""
        conn = self._get_conn()
        conn.execute(
            """UPDATE runs SET
               status = 'completed',
               initial_loss = ?, final_loss = ?,
               total_steps = ?, duration_secs = ?, output_dir = ?
               WHERE run_id = ?""",
            (initial_loss, final_loss, total_steps, duration_secs, output_dir, run_id),
        )
        conn.commit()

    def fail_run(self, run_id: str) -> None:
        """Mark run as failed."""
        conn = self._get_conn()
        conn.execute("UPDATE runs SET status = 'failed' WHERE run_id = ?", (run_id,))
        conn.commit()

    def list_runs(self, limit: int = 50) -> list[dict]:
        """Return list of runs ordered by created_at desc."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]

    def get_run(self, run_id: str) -> Optional[dict]:
        """Get full details of a single run. Supports prefix matching."""
        conn = self._get_conn()
        # Try exact match first
        row = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()

        if row is None:
            # Try prefix match
            rows = conn.execute(
                "SELECT * FROM runs WHERE run_id LIKE ? ORDER BY created_at DESC",
                (f"{run_id}%",),
            ).fetchall()
            if len(rows) == 1:
                row = rows[0]
            elif len(rows) > 1:
                return None  # ambiguous prefix

        return dict(row) if row else None

    def get_metrics(self, run_id: str) -> list[dict]:
        """Get all metric rows for a run, ordered by step."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM metrics WHERE run_id = ? ORDER BY step", (run_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def save_eval_result(
        self,
        model_path: str,
        benchmark: str,
        score: float,
        details: dict,
        run_id: Optional[str] = None,
    ) -> None:
        """Save an evaluation result."""
        now = datetime.now().isoformat()
        details_json = json.dumps(details, default=str)
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO eval_results
               (run_id, model_path, benchmark, score, details_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (run_id, model_path, benchmark, score, details_json, now),
        )
        conn.commit()

    def get_eval_results(self, run_id: Optional[str] = None) -> list[dict]:
        """Get eval results, optionally filtered by run_id."""
        conn = self._get_conn()
        if run_id:
            rows = conn.execute(
                "SELECT * FROM eval_results WHERE run_id = ? ORDER BY created_at DESC",
                (run_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM eval_results ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_run(self, run_id: str) -> bool:
        """Delete a run and its metrics. Returns True if found."""
        conn = self._get_conn()
        conn.execute("DELETE FROM metrics WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM eval_results WHERE run_id = ?", (run_id,))
        cursor = conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        conn.commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
