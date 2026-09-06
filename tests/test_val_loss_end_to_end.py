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


def _legacy_db(path, *, rows: int = 5) -> None:
    """A database in the pre-change schema, populated the way a real one is.

    Built from the **live** ``_SCHEMA_SQL`` with the new column stripped, rather
    than a hand-written approximation, so the fixture cannot drift away from the
    schema it is pretending to be an older version of. The ``runs`` row carries
    the columns ``list_runs()`` actually reads — an earlier version of this
    fixture was a two-column stub, which meant the migration was exercised but
    the read path afterwards was not.
    """
    import json
    import re
    from pathlib import Path

    from soup_cli.experiment import tracker as _tracker_mod

    live = re.search(r'_SCHEMA_SQL = """(.*?)"""', Path(_tracker_mod.__file__).read_text(
        encoding="utf-8"), re.S).group(1)
    pre_change = live.replace("    val_loss  REAL,\n", "")
    assert "val_loss" not in pre_change, "fixture must genuinely predate the column"

    conn = sqlite3.connect(str(path))
    conn.executescript(pre_change)
    conn.execute(
        "INSERT INTO runs (run_id, created_at, status, config_json, base_model, task,"
        " initial_loss, final_loss, total_steps) VALUES"
        " ('old-run', '2026-01-01T00:00:00', 'completed', ?, 'Qwen/Qwen2.5-0.5B',"
        " 'sft', 3.0, 1.0, ?)",
        (json.dumps({"base": "Qwen/Qwen2.5-0.5B", "task": "sft"}), rows),
    )
    for step in range(1, rows + 1):
        conn.execute(
            "INSERT INTO metrics (run_id, step, epoch, loss, lr, grad_norm, speed,"
            " gpu_mem, timestamp) VALUES ('old-run', ?, ?, ?, 1e-4, 0.9, 3.0,"
            " '2.0 GB', ?)",
            (step, step / rows, 3.0 - step * 0.25, f"2026-01-01T00:0{step}:00"),
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
        rows = [dict(r) for r in conn.execute("SELECT * FROM metrics ORDER BY step")]
        conn.close()

        assert len(rows) == 5, "every pre-existing row must survive the migration"
        assert [r["step"] for r in rows] == [1, 2, 3, 4, 5]
        assert rows[0]["loss"] == pytest.approx(2.75), "pre-existing data must survive"
        assert all(r["gpu_mem"] == "2.0 GB" for r in rows)
        assert all(r["val_loss"] is None for r in rows), (
            "unmeasured must be NULL, never 0.0"
        )

    def test_the_read_path_still_works_after_migrating(self, tmp_path):
        """`soup runs show` reads through `list_runs()`; migrating must not
        break it. Previously nothing committed exercised the read at all."""
        from soup_cli.experiment.tracker import ExperimentTracker

        db = tmp_path / "experiments.db"
        _legacy_db(db)
        tracker = ExperimentTracker(db_path=str(db))
        tracker.init_db()

        runs = tracker.list_runs()
        assert [r["run_id"] for r in runs] == ["old-run"]
        assert runs[0]["base_model"] == "Qwen/Qwen2.5-0.5B"
        assert runs[0]["status"] == "completed"

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


class TestItReachesTheWireNotJustTheDataclass:
    """Review finding on #713: `val_loss` was on `TrainEvent` and absent from
    `_ALLOWED_KEYS`, so `to_payload` filtered it out and the SSE stream never
    carried it. The PR was written to close a value that is collected and read
    by nothing, and the sink verification stopped one layer above where the
    value actually died.
    """

    def test_val_loss_survives_to_payload(self):
        from soup_cli.utils.sse_train_stream import TrainEvent, to_payload

        payload = to_payload(TrainEvent(type="metric", step=10, loss=2.5, val_loss=0.75))
        assert payload.get("val_loss") == pytest.approx(0.75)

    def test_val_loss_is_on_the_serialised_wire_frame(self):
        """The end of the pipe, not the middle of it."""
        from soup_cli.utils.sse_train_stream import TrainEvent, format_sse_frame

        frame = format_sse_frame(TrainEvent(type="metric", step=10, loss=2.5, val_loss=0.75))
        assert '"val_loss":0.75' in frame.replace(" ", "")

    def test_every_dataclass_field_is_serialisable(self):
        """The guard that stops the next field drifting the same way.

        A field can be added to `TrainEvent` and silently never reach a client,
        because `to_payload` filters against a separately-maintained set. This
        ties the two together so the omission is a test failure rather than a
        quiet drop.
        """
        from soup_cli.utils.sse_train_stream import _ALLOWED_KEYS, TrainEvent

        fields = set(TrainEvent.__dataclass_fields__)
        assert fields <= _ALLOWED_KEYS, (
            f"TrainEvent fields absent from _ALLOWED_KEYS and therefore dropped "
            f"before the wire: {sorted(fields - _ALLOWED_KEYS)}"
        )


class TestStickyForThePanelPerCallForTheRecord:
    """Review finding on #713: the sticky value was persisted as well as shown.

    At a realistic cadence that writes a measurement on every step between
    evaluations — 9 stored points for 2 real ones — which inflates n for
    anything reading the series back, including `soup eval`'s paired bootstrap.
    """

    def _cb(self):
        from soup_cli.monitoring.callback import SoupTrainerCallback

        display, tracker = _RecordingDisplay(), _RecordingTracker()
        return SoupTrainerCallback(display, tracker=tracker, run_id="r1"), display, tracker

    def test_a_training_step_after_an_evaluation_persists_null(self):
        cb, _, tracker = self._cb()
        cb.on_log(object(), _State(), object(), logs={"loss": 2.5})
        cb.on_log(object(), _State(), object(), logs={"eval_loss": 0.9})
        cb.on_log(object(), _State(), object(), logs={"loss": 2.4})   # no eval

        stored = [c.get("val_loss") for c in tracker.calls]
        assert stored[-1] is None, (
            "a training step with no evaluation must persist NULL, not the "
            f"carried-forward value: {stored}"
        )
        assert stored.count(0.9) == 1, "exactly one row per actual evaluation"

    def test_the_panel_keeps_showing_the_last_measured_value(self):
        """Sticky, and the mutation that made it non-sticky passed 311 tests."""
        cb, display, _ = self._cb()
        cb.on_log(object(), _State(), object(), logs={"eval_loss": 0.9})
        cb.on_log(object(), _State(), object(), logs={"loss": 2.4})   # no eval

        assert display.calls[-1].get("val_loss") == pytest.approx(0.9), (
            "the panel row must not blink out between evaluations"
        )

    def test_the_two_sinks_genuinely_disagree(self):
        """The discriminating assertion: one call, two different values.

        If display and tracker ever receive the same thing on a non-eval step,
        one of the two behaviours has been lost.
        """
        cb, display, tracker = self._cb()
        cb.on_log(object(), _State(), object(), logs={"eval_loss": 0.9})
        cb.on_log(object(), _State(), object(), logs={"loss": 2.4})

        assert display.calls[-1].get("val_loss") == pytest.approx(0.9)
        assert tracker.calls[-1].get("val_loss") is None


class TestThePanelActuallyRendersIt:
    def test_the_val_loss_row_appears_in_the_rendered_panel(self):
        """`if False:` around the render line passed the whole suite."""
        from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig
        from soup_cli.monitoring.display import TrainingDisplay

        cfg = SoupConfig(
            base="m", task="sft",
            data=DataConfig(train="t.jsonl", format="chatml"),
            training=TrainingConfig(), output="./o",
        )
        display = TrainingDisplay(cfg)
        display.update(step=10, epoch=1.0, loss=2.5, lr=1e-4, val_loss=0.75)

        from rich.console import Console
        console = Console(file=__import__("io").StringIO(), width=100)
        console.print(display._render())
        rendered = console.file.getvalue()

        assert "Val loss" in rendered, f"the row is not rendered: {rendered!r}"
        assert "0.75" in rendered

    def test_no_val_loss_row_before_any_evaluation(self):
        """Reject-everything control: the row must not appear from nowhere."""
        from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig
        from soup_cli.monitoring.display import TrainingDisplay

        cfg = SoupConfig(
            base="m", task="sft",
            data=DataConfig(train="t.jsonl", format="chatml"),
            training=TrainingConfig(), output="./o",
        )
        display = TrainingDisplay(cfg)
        display.update(step=1, epoch=0.1, loss=3.0, lr=1e-4)

        from rich.console import Console
        console = Console(file=__import__("io").StringIO(), width=100)
        console.print(display._render())

        assert "Val loss" not in console.file.getvalue()


class TestTheDisplayIsStickyOnItsOwn:
    """Stickiness lives in TWO places and only one was pinned.

    `SoupTrainerCallback` carries `_last_val_loss` and passes it on every call,
    so a test driven through the callback keeps passing even if
    `TrainingDisplay` itself stops being sticky. This drives the display
    directly, which is the only way to tell the two apart.
    """

    def _display(self):
        from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig
        from soup_cli.monitoring.display import TrainingDisplay

        cfg = SoupConfig(
            base="m", task="sft",
            data=DataConfig(train="t.jsonl", format="chatml"),
            training=TrainingConfig(), output="./o",
        )
        return TrainingDisplay(cfg)

    def test_an_update_without_val_loss_keeps_the_last_one(self):
        display = self._display()
        display.update(step=5, epoch=0.5, loss=2.0, lr=1e-4, val_loss=0.9)
        display.update(step=6, epoch=0.6, loss=1.9, lr=1e-4)      # no val_loss

        assert display.val_loss == pytest.approx(0.9), (
            "the display must carry the last measured value forward on its own, "
            "not rely on the callback re-supplying it"
        )

    def test_an_explicit_none_also_keeps_the_last_one(self):
        """The callback passes val_loss=None on non-eval steps once the record
        path stopped fabricating, so None must be 'no news', not 'clear it'."""
        display = self._display()
        display.update(step=5, epoch=0.5, loss=2.0, lr=1e-4, val_loss=0.9)
        display.update(step=6, epoch=0.6, loss=1.9, lr=1e-4, val_loss=None)

        assert display.val_loss == pytest.approx(0.9)

    def test_a_new_measurement_replaces_the_old_one(self):
        """Control: sticky must not mean frozen."""
        display = self._display()
        display.update(step=5, epoch=0.5, loss=2.0, lr=1e-4, val_loss=0.9)
        display.update(step=10, epoch=1.0, loss=1.8, lr=1e-4, val_loss=0.7)

        assert display.val_loss == pytest.approx(0.7)


class TestTheCallbackPushesItToTheStream:
    """The producer side of the wire, distinct from `to_payload` being correct."""

    def test_the_pushed_event_carries_the_measured_val_loss(self, monkeypatch):
        import soup_cli.utils.train_event_buffer as buf
        from soup_cli.monitoring.callback import SoupTrainerCallback

        pushed = []
        monkeypatch.setattr(buf, "push_train_event", lambda e: pushed.append(e))

        cb = SoupTrainerCallback(_RecordingDisplay())
        cb.on_log(object(), _State(), object(), logs={"eval_loss": 0.75})

        assert pushed, "no SSE event was pushed at all"
        assert pushed[-1].val_loss == pytest.approx(0.75), (
            "the event reached the buffer without the value it exists to carry"
        )

    def test_a_non_eval_step_pushes_none_not_the_carried_value(self, monkeypatch):
        """Same rule as the database: the stream records measurements."""
        import soup_cli.utils.train_event_buffer as buf
        from soup_cli.monitoring.callback import SoupTrainerCallback

        pushed = []
        monkeypatch.setattr(buf, "push_train_event", lambda e: pushed.append(e))

        cb = SoupTrainerCallback(_RecordingDisplay())
        cb.on_log(object(), _State(), object(), logs={"eval_loss": 0.75})
        cb.on_log(object(), _State(), object(), logs={"loss": 2.0})

        assert pushed[-1].val_loss is None
