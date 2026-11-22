"""Schema order for metrics in fresh databases and databases upgraded from v0.77.2."""

import sqlite3
from contextlib import closing
from pathlib import Path

from soup_cli.experiment.tracker import ExperimentTracker

# The metrics definition from v0.77.2, before kind and per-step reward columns.
_METRICS_V0772 = """
CREATE TABLE metrics (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT NOT NULL REFERENCES runs(run_id),
    step      INTEGER NOT NULL,
    epoch     REAL,
    loss      REAL,
    val_loss  REAL,
    val_reward REAL,
    lr        REAL,
    grad_norm REAL,
    speed     REAL,
    gpu_mem   TEXT,
    timestamp TEXT NOT NULL
)
"""
_RUN_ID = "legacy-run"
_TIMESTAMP = "2026-01-01T00:00:00"
_OLD_METRIC = {
    "id": 1,
    "run_id": _RUN_ID,
    "step": 4,
    "epoch": 1.5,
    "loss": 0.25,
    "val_loss": 0.5,
    "val_reward": 0.75,
    "lr": 0.125,
    "grad_norm": 2.0,
    "speed": 8.0,
    "gpu_mem": "256 MiB",
    "timestamp": _TIMESTAMP,
}
_NEW_METRIC_FIELDS = {"kind": "train", "reward": 0.75, "reward_std": 0.125}


def _metric_columns(path: Path) -> list[str]:
    with closing(sqlite3.connect(path)) as conn:
        return [row[1] for row in conn.execute("PRAGMA table_info(metrics)")]


def _seed_metric(path: Path, metric: dict, *, schema: str | None = None) -> None:
    with closing(sqlite3.connect(path)) as conn, conn:
        if schema is not None:
            conn.execute(schema)
        columns = ", ".join(metric)
        placeholders = ", ".join("?" for _ in metric)
        conn.execute(
            f"INSERT INTO metrics ({columns}) VALUES ({placeholders})", tuple(metric.values())
        )


def test_upgraded_metrics_table_matches_fresh_order_and_preserves_legacy_values(tmp_path):
    fresh_path = tmp_path / "fresh.db"
    legacy_path = tmp_path / "legacy.db"
    _seed_metric(legacy_path, _OLD_METRIC, schema=_METRICS_V0772)
    old_columns = _metric_columns(legacy_path)

    with closing(ExperimentTracker(fresh_path)), closing(ExperimentTracker(legacy_path)) as tracker:
        upgraded_columns = _metric_columns(legacy_path)
        assert upgraded_columns == [*old_columns, "kind", "reward", "reward_std"]
        assert _metric_columns(fresh_path) == upgraded_columns
        expected = _OLD_METRIC | dict.fromkeys(_NEW_METRIC_FIELDS)
        assert tracker.get_metrics(_RUN_ID) == [expected]

    with closing(ExperimentTracker(legacy_path)) as tracker:
        assert _metric_columns(legacy_path) == upgraded_columns
        assert tracker.get_metrics(_RUN_ID) == [expected]


def test_get_metrics_reads_values_by_name_in_both_column_orders(tmp_path):
    upgraded_path = tmp_path / "upgraded.db"
    released_path = tmp_path / "v0780.db"
    # v0.78.0 placed kind before timestamp in newly created databases.
    released_schema = _METRICS_V0772.replace(
        "    timestamp TEXT NOT NULL",
        "    kind      TEXT,\n    timestamp TEXT NOT NULL,\n    reward REAL,\n    reward_std REAL",
    )
    expected = _OLD_METRIC | _NEW_METRIC_FIELDS
    _seed_metric(released_path, expected, schema=released_schema)
    _seed_metric(upgraded_path, _OLD_METRIC, schema=_METRICS_V0772)
    with closing(ExperimentTracker(upgraded_path)) as upgraded:
        with closing(sqlite3.connect(upgraded_path)) as conn, conn:
            conn.execute(
                "UPDATE metrics SET kind = ?, reward = ?, reward_std = ? WHERE id = ?",
                (*_NEW_METRIC_FIELDS.values(), _OLD_METRIC["id"]),
            )
        with closing(ExperimentTracker(released_path)) as released:
            released_columns = _metric_columns(released_path)
            assert released_columns.index("kind") < released_columns.index("timestamp")
            assert _metric_columns(upgraded_path) != released_columns
            assert upgraded.get_metrics(_RUN_ID) == released.get_metrics(_RUN_ID) == [expected]
