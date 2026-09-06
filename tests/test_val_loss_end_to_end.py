"""Validation loss must reach the sinks — on BOTH backends (#23 follow-up).

Found while bridging MLX's `on_val_loss_report`, but it is not an MLX gap.
`SoupTrainerCallback.on_log` reads `logs["loss"]` and never `logs["eval_loss"]`,
so an evaluation step leaves `_last_loss` untouched and **re-reports the stale
training number to every sink at the same step**. The evaluated value has never
existed anywhere in this project:

* `TrainingDisplay.update()` reads only `grad_norm`, `speed`, `gpu_mem` from
  ``**kwargs`` and drops the rest;
* the ``metrics`` table has no column for it;
* ``TrainEvent`` has no field for it.

So wiring a producer alone would have shipped a value that three sinks discard —
the collected-and-read-by-nothing shape this repo has already found in #363 and
#659. The sinks come first here, which is why this file starts with the schema.

The migration is the part that can hurt people: ``~/.soup/experiments.db``
exists on every machine that has ever run ``soup train``, so an ``ALTER TABLE``
that assumes a fresh schema would crash the CLI for every existing user.
"""

from __future__ import annotations

import sqlite3

import pytest

#: The ``metrics`` DDL exactly as it shipped BEFORE this change, copied from
#: `_SCHEMA_SQL` rather than hand-approximated, so the migration fixture is
#: genuinely the old shape and not a guess at it.
_PRE_VAL_LOSS_METRICS_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    status TEXT
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
"""


def _legacy_db(path) -> None:
    """A database in the pre-change schema, carrying one real metrics row."""
    conn = sqlite3.connect(str(path))
    conn.executescript(_PRE_VAL_LOSS_METRICS_DDL)
    conn.execute("INSERT INTO runs (run_id, status) VALUES ('old-run', 'completed')")
    conn.execute(
        "INSERT INTO metrics (run_id, step, epoch, loss, lr, grad_norm, speed,"
        " gpu_mem, timestamp) VALUES ('old-run', 7, 0.5, 1.25, 1e-4, 0.9, 3.0,"
        " '2.0 GB', '2026-01-01T00:00:00')"
    )
    conn.commit()
    conn.close()


def _columns(path, table: str) -> set[str]:
    conn = sqlite3.connect(str(path))
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    finally:
        conn.close()


class TestTheMigration:
    """An existing user's DB must upgrade in place, not crash or lose rows."""

    def test_a_legacy_db_gains_the_column(self, tmp_path):
        from soup_cli.experiment.tracker import ExperimentTracker

        db = tmp_path / "experiments.db"
        _legacy_db(db)
        assert "val_loss" not in _columns(db, "metrics"), "fixture must be pre-change"

        ExperimentTracker(db_path=str(db)).init_db()

        assert "val_loss" in _columns(db, "metrics")

    def test_the_legacy_row_survives_and_reads_null(self, tmp_path):
        """Existing rows must say "not recorded", not a fabricated 0.0.

        Same argument as `grad_norm` being absent rather than a plausible zero
        on the MLX path: a value nobody measured must not read as one somebody
        did.
        """
        from soup_cli.experiment.tracker import ExperimentTracker

        db = tmp_path / "experiments.db"
        _legacy_db(db)
        ExperimentTracker(db_path=str(db)).init_db()

        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM metrics WHERE step = 7").fetchone()
        conn.close()

        assert row["loss"] == pytest.approx(1.25), "pre-existing data must survive"
        assert row["gpu_mem"] == "2.0 GB"
        assert row["val_loss"] is None, "unmeasured must be NULL, never 0.0"

    def test_migrating_twice_is_a_no_op(self, tmp_path):
        from soup_cli.experiment.tracker import ExperimentTracker

        db = tmp_path / "experiments.db"
        _legacy_db(db)
        ExperimentTracker(db_path=str(db)).init_db()
        ExperimentTracker(db_path=str(db)).init_db()   # must not raise

        assert "val_loss" in _columns(db, "metrics")

    def test_the_second_migration_issues_no_alter_at_all(self, tmp_path):
        """Idempotent by PRAGMA, **not** by catching a duplicate-column error.

        The discriminating test. Without it, hard-coding the PRAGMA back to the
        `runs` table survives every other assertion here: the column check then
        always misses, the ALTER fires on every startup, and idempotency comes
        from the `except sqlite3.OperationalError` handler instead. That still
        "works" while turning a guarded migration into an exception-driven one
        on every single `soup train` invocation.

        Uses sqlite3's own `set_trace_callback` rather than patching
        `Connection.execute`, which is an immutable type and cannot be
        monkeypatched — my first attempt at this test failed for that reason,
        not because the migration was wrong.
        """
        from soup_cli.experiment.tracker import ExperimentTracker

        db = tmp_path / "experiments.db"
        _legacy_db(db)
        ExperimentTracker(db_path=str(db)).init_db()          # first: must ALTER

        tracker = ExperimentTracker(db_path=str(db))
        conn = tracker._get_conn()
        statements: list[str] = []
        conn.set_trace_callback(statements.append)
        tracker.init_db()                                     # second: must not
        conn.set_trace_callback(None)

        alters = [q for q in statements if "ALTER TABLE" in q.upper()]
        assert alters == [], (
            "the second migration re-issued ALTER TABLE and relied on the "
            f"duplicate-column handler to swallow it: {alters}"
        )
        assert any("PRAGMA table_info(metrics)" in q for q in statements), (
            "the metrics table was never inspected, so the guard cannot be "
            "checking the right table"
        )

    def test_a_fresh_db_has_the_column_without_migrating(self, tmp_path):
        from soup_cli.experiment.tracker import ExperimentTracker

        db = tmp_path / "fresh.db"
        ExperimentTracker(db_path=str(db)).init_db()

        assert "val_loss" in _columns(db, "metrics")

    def test_control_an_unmigrated_db_cannot_serve_the_new_read(self, tmp_path):
        """The control: without the migration the new path must FAIL.

        If this passed on a legacy DB, the migration would be unnecessary and
        every test above would be proving nothing.
        """
        db = tmp_path / "legacy.db"
        _legacy_db(db)

        conn = sqlite3.connect(str(db))
        with pytest.raises(sqlite3.OperationalError, match="val_loss"):
            conn.execute("SELECT val_loss FROM metrics").fetchall()
        conn.close()


class TestTheTrackerStoresIt:
    def test_log_metrics_persists_val_loss(self, tmp_path):
        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker(db_path=str(tmp_path / "e.db"))
        tracker.init_db()
        conn = tracker._get_conn()
        conn.execute(
            "INSERT INTO runs (run_id, created_at, status, config_json) "
            "VALUES ('r1', '2026-01-01T00:00:00', 'running', '{}')"
        )
        conn.commit()

        tracker.log_metrics(run_id="r1", step=10, epoch=1.0, loss=2.5, val_loss=0.75)

        row = conn.execute("SELECT loss, val_loss FROM metrics WHERE step=10").fetchone()
        assert row["loss"] == pytest.approx(2.5)
        assert row["val_loss"] == pytest.approx(0.75)

    def test_a_train_only_row_stores_null_not_zero(self, tmp_path):
        """Most rows are training steps with no evaluation attached."""
        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker(db_path=str(tmp_path / "e.db"))
        tracker.init_db()
        conn = tracker._get_conn()
        conn.execute(
            "INSERT INTO runs (run_id, created_at, status, config_json) "
            "VALUES ('r1', '2026-01-01T00:00:00', 'running', '{}')"
        )
        conn.commit()

        tracker.log_metrics(run_id="r1", step=11, loss=2.5)

        row = conn.execute("SELECT val_loss FROM metrics WHERE step=11").fetchone()
        assert row["val_loss"] is None


class TestTheEventCarriesIt:
    def test_train_event_has_a_val_loss_field(self):
        from soup_cli.utils.sse_train_stream import TrainEvent

        event = TrainEvent(type="metric", step=5, loss=2.0, val_loss=0.5)
        assert event.val_loss == pytest.approx(0.5)

    def test_val_loss_defaults_to_none(self):
        from soup_cli.utils.sse_train_stream import TrainEvent

        assert TrainEvent(type="metric", step=5, loss=2.0).val_loss is None


class TestTheDisplayCarriesItAsItsOwnSeries:
    def test_val_loss_does_not_overwrite_the_training_loss(self):
        """The whole reason this is a separate field.

        Routing validation loss through `loss=` would replace the training
        curve with a different series at the same step — worse than showing
        nothing, because the panel would look right while lying.
        """
        from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig
        from soup_cli.monitoring.display import TrainingDisplay

        cfg = SoupConfig(
            base="m", task="sft",
            data=DataConfig(train="t.jsonl", format="chatml"),
            training=TrainingConfig(),
            output="./o",
        )
        display = TrainingDisplay(cfg)
        display.update(step=10, epoch=1.0, loss=2.5, lr=1e-4, val_loss=0.75)

        assert display.loss == pytest.approx(2.5), "training loss must be untouched"
        assert display.val_loss == pytest.approx(0.75)

    def test_val_loss_is_none_until_an_evaluation_happens(self):
        from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig
        from soup_cli.monitoring.display import TrainingDisplay

        cfg = SoupConfig(
            base="m", task="sft",
            data=DataConfig(train="t.jsonl", format="chatml"),
            training=TrainingConfig(),
            output="./o",
        )
        display = TrainingDisplay(cfg)
        display.update(step=1, epoch=0.1, loss=3.0, lr=1e-4)

        assert display.val_loss is None


class _RecordingDisplay:
    def __init__(self):
        self.calls = []

    def start(self, *a, **k):
        pass

    def update(self, **kwargs):
        self.calls.append(kwargs)

    def stop(self):
        pass


class _RecordingTracker:
    def __init__(self):
        self.calls = []

    def log_metrics(self, **kwargs):
        self.calls.append(kwargs)


class _State:
    global_step = 10
    epoch = 1.0
    log_history: list = []


class TestTheTransformersProducer:
    """`SoupTrainerCallback.on_log` must read `eval_loss`.

    This is the probe @MakazhanAlpamys used to disprove my original premise,
    kept as a test. Before the fix it answered False: an eval log left
    `_last_loss` untouched and re-reported the stale training number, so the
    evaluated value reached no sink at all.
    """

    def _fire(self):
        from soup_cli.monitoring.callback import SoupTrainerCallback

        display, tracker = _RecordingDisplay(), _RecordingTracker()
        cb = SoupTrainerCallback(display, tracker=tracker, run_id="r1")
        cb.on_log(object(), _State(), object(),
                  logs={"loss": 2.5, "learning_rate": 1e-4})
        cb.on_log(object(), _State(), object(),
                  logs={"eval_loss": 0.75, "eval_runtime": 1.2})
        return display, tracker

    def test_the_eval_loss_reaches_both_sinks(self):
        display, tracker = self._fire()

        assert any(c.get("val_loss") == pytest.approx(0.75) for c in display.calls), (
            "the evaluated loss never reached the display — this is the exact "
            "state the fix exists to change"
        )
        assert any(c.get("val_loss") == pytest.approx(0.75) for c in tracker.calls)

    def test_the_training_loss_is_not_replaced_by_it(self):
        """The discriminating assertion.

        A 'fix' that routed eval_loss through `loss=` would satisfy the test
        above and silently replace the training curve with a different series
        at the same step.
        """
        display, _ = self._fire()

        assert all(c["loss"] == pytest.approx(2.5) for c in display.calls), (
            "training loss must survive an evaluation step untouched"
        )

    def test_a_train_only_log_reports_no_val_loss(self):
        """Reject-everything control: val_loss must not appear from nowhere."""
        from soup_cli.monitoring.callback import SoupTrainerCallback

        display = _RecordingDisplay()
        cb = SoupTrainerCallback(display)
        cb.on_log(object(), _State(), object(),
                  logs={"loss": 2.5, "learning_rate": 1e-4})

        assert display.calls[0].get("val_loss") is None
