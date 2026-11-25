"""Durable run status storage on the existing SQLite tracker row."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import UTC, datetime, timedelta
from threading import Barrier

import pytest

from soup_cli.experiment import tracker as tracker_module
from soup_cli.experiment.tracker import ExperimentTracker, RunStatusStorageError
from soup_cli.utils.command_result import MAX_COMMAND_RESULT_BYTES
from soup_cli.utils.run_status import RunStatus, encode_run_status

_OWNER = "private-owner"
_START = datetime(2026, 11, 25, tzinfo=UTC)
_COLUMNS = "run_status_owner, run_status_state, run_status_json"


def _snapshot(state="launching", *, run_id="run-one", seconds=0, step=None):
    at = _START + timedelta(seconds=seconds)
    return RunStatus(
        schema="soup.run_status",
        schema_version=1,
        soup_version="0.78.0",
        run_id=run_id,
        command="train",
        started_by="cli",
        state=state,
        started_at=_START,
        updated_at=at,
        finished_at=at if state == "done" else None,
        step=step,
    )


def _start(tracker, run_id="run-one"):
    return tracker.start_run({}, "cpu", "CPU", {}, run_id=run_id)


def _row(path, run_id="run-one"):
    with closing(sqlite3.connect(path)) as conn:
        return conn.execute(
            f"SELECT {_COLUMNS} FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()


def _legacy_db(path, *, partial=False):
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'running',
                config_json TEXT NOT NULL,
                total_steps INTEGER
            );
            INSERT INTO runs VALUES ('legacy', '2026-01-02T00:00:00', 'completed', '{}', 3);
            INSERT INTO runs VALUES ('earlier', '2026-01-01T00:00:00', 'running', '{}', 2);
            """
        )
        if partial:
            conn.execute("ALTER TABLE runs ADD COLUMN run_status_state TEXT")


@pytest.mark.parametrize("partial", [False, True])
def test_legacy_migration_is_nullable_and_preserves_existing_rows(tmp_path, partial):
    path = tmp_path / "legacy.sqlite"
    _legacy_db(path, partial=partial)
    with closing(ExperimentTracker(path)) as tracker:
        tracker.init_db()
        tracker.init_db()
        newer, older = tracker.list_runs()
        assert (newer["run_id"], newer["status"], newer["total_steps"]) == (
            "legacy", "completed", 3
        )
        assert older["run_id"] == "earlier"
        assert tracker.get_run_status("legacy") is None
    assert _row(path, "legacy") == (None, None, None)
    assert _row(path, "earlier") == (None, None, None)


def test_two_connections_can_open_the_same_legacy_layout(tmp_path):
    path = tmp_path / "legacy.sqlite"
    _legacy_db(path)
    barrier = Barrier(2, timeout=8)

    def read_legacy():
        barrier.wait()
        with closing(ExperimentTracker(path)) as tracker:
            return tracker.get_run("legacy")["total_steps"]

    with ThreadPoolExecutor(max_workers=2) as executor:
        left = executor.submit(read_legacy)
        right = executor.submit(read_legacy)
        assert (left.result(), right.result()) == (3, 3)
    assert _row(path, "legacy") == (None, None, None)


def test_status_reads_by_exact_id_and_legacy_views_exclude_internal_columns(tmp_path):
    path = tmp_path / "fresh.sqlite"
    with closing(ExperimentTracker(path)) as tracker:
        assert _start(tracker) == "run-one"
        assert tracker.get_run_status("run-one") is None
        snapshot = _snapshot()
        assert tracker.initialize_run_status(_OWNER, snapshot)
        assert tracker.get_run_status("run-one") == snapshot
        assert tracker.get_run_status("run") is None
        assert tracker.get_run("run")["run_id"] == "run-one"
        assert not tracker.initialize_run_status(_OWNER, _snapshot(run_id="missing"))

        def select(rows):
            (run,) = rows
            assert not any(key.startswith("run_status_") for key in run)
            return [run]

        for run in (
            tracker.get_run("run-one"),
            tracker.list_runs()[0],
            tracker.list_runs(select_runs=select)[0],
            tracker.list_active_execution_runs()[0],
        ):
            assert not any(key.startswith("run_status_") for key in run)
            assert _OWNER not in str(run)
            assert run["status"] == "running"


@pytest.mark.parametrize(
    ("stored_state", "payload"),
    [
        ("running", "private malformed content"),
        ("running", "x" * (MAX_COMMAND_RESULT_BYTES + 1)),
        ("running", None),
        (None, "incomplete"),
        ("running", sqlite3.Binary(b"not-text")),
    ],
    ids=["malformed", "oversize", "missing", "partial", "blob"],
)
def test_corrupt_snapshot_has_fixed_diagnostic(tmp_path, stored_state, payload):
    path = tmp_path / "corrupt.sqlite"
    with closing(ExperimentTracker(path)) as tracker:
        _start(tracker)
        with closing(sqlite3.connect(path)) as conn, conn:
            conn.execute(
                f"UPDATE runs SET ({_COLUMNS}) = (?, ?, ?)", (_OWNER, stored_state, payload)
            )
        with pytest.raises(RunStatusStorageError, match="^stored run status is invalid$"):
            tracker.get_run_status("run-one")
        assert tracker.get_run("run-one")["run_id"] == "run-one"


def test_legacy_views_do_not_load_invalid_private_text(tmp_path):
    path = tmp_path / "invalid-text.sqlite"
    with closing(ExperimentTracker(path)) as tracker:
        _start(tracker)
        with closing(sqlite3.connect(path)) as conn, conn:
            conn.execute(
                """UPDATE runs SET run_status_owner = ?, run_status_state = 'launching',
                   run_status_json = CAST(X'FF' AS TEXT)""",
                (_OWNER,),
            )
        assert tracker.get_run("run-one")["run_id"] == "run-one"
        assert tracker.list_runs()[0]["run_id"] == "run-one"
        assert tracker.list_active_execution_runs()[0]["run_id"] == "run-one"
        with pytest.raises(RunStatusStorageError, match="^stored run status is invalid$"):
            tracker.get_run_status("run-one")


@pytest.mark.parametrize(
    ("run_id", "state"),
    [("different", "launching"), ("run-one", "running")],
)
def test_stored_snapshot_must_match_row_identity_and_state(tmp_path, run_id, state):
    path = tmp_path / "mismatch.sqlite"
    with closing(ExperimentTracker(path)) as tracker:
        _start(tracker)
        payload = encode_run_status(_snapshot(run_id=run_id)).decode("utf-8")
        with closing(sqlite3.connect(path)) as conn, conn:
            conn.execute(f"UPDATE runs SET ({_COLUMNS}) = (?, ?, ?)", (_OWNER, state, payload))
        with pytest.raises(RunStatusStorageError, match="^stored run status is invalid$"):
            tracker.get_run_status("run-one")


def test_state_and_wire_change_together_with_an_exact_owner_and_revision(tmp_path):
    path = tmp_path / "status.sqlite"
    with closing(ExperimentTracker(path)) as tracker:
        _start(tracker)
        launching = _snapshot()
        running = _snapshot("running", seconds=1, step=0)
        assert not tracker.initialize_run_status(_OWNER, running)
        assert tracker.initialize_run_status(_OWNER, launching)
        assert not tracker.initialize_run_status("another-owner", launching)
        old_row = _row(path)
        assert not tracker.claim_run_status("another-owner", launching, running)
        assert not tracker.claim_run_status(_OWNER, launching, _snapshot("running", seconds=0))
        assert _row(path) == old_row
        assert tracker.claim_run_status(_OWNER, launching, running)
        assert _row(path) == (_OWNER, "running", encode_run_status(running).decode("utf-8"))

        advanced = _snapshot("running", seconds=2, step=3)
        assert tracker.update_run_status(_OWNER, running, advanced)
        old_row = _row(path)
        for expected, update in (
            (running, _snapshot("running", seconds=3, step=4)),
            (advanced, _snapshot("running", seconds=3, step=2)),
            (advanced, _snapshot("running", seconds=3)),
            (advanced, _snapshot("running", seconds=1, step=4)),
            (advanced, _snapshot("running", seconds=2, step=4)),
            (advanced, _snapshot("launching", seconds=3, step=4)),
            (advanced, _snapshot("done", seconds=3, step=4)),
            (advanced, _snapshot("running", run_id="different", seconds=3, step=4)),
        ):
            assert not tracker.update_run_status(_OWNER, expected, update)
        assert not tracker.update_run_status("another-owner", advanced, advanced)
        assert _row(path) == old_row

        finalizing = _snapshot("finalizing", seconds=3, step=3)
        done = _snapshot("done", seconds=4, step=3)
        assert tracker.update_run_status(_OWNER, advanced, finalizing)
        assert tracker.update_run_status(_OWNER, finalizing, done)
        assert not tracker.update_run_status(_OWNER, done, _snapshot("done", seconds=5, step=3))
        assert tracker.get_run_status("run-one") == done
        assert tracker.get_run("run-one")["status"] == "running"


def test_two_writers_cannot_both_commit_against_the_same_status_revision(tmp_path):
    path = tmp_path / "racing.sqlite"
    launching = _snapshot()
    running = _snapshot("running", seconds=1, step=0)
    with closing(ExperimentTracker(path)) as tracker:
        _start(tracker)
        assert tracker.initialize_run_status(_OWNER, launching)
        assert tracker.claim_run_status(_OWNER, launching, running)

    barrier = Barrier(2, timeout=8)

    def commit(step):
        with closing(ExperimentTracker(path)) as tracker:
            candidate = _snapshot("running", seconds=2, step=step)
            barrier.wait()
            return candidate, tracker.update_run_status(_OWNER, running, candidate)

    with ThreadPoolExecutor(max_workers=2) as executor:
        left = executor.submit(commit, 1)
        right = executor.submit(commit, 2)
        (winner,) = [candidate for candidate, won in (left.result(), right.result()) if won]
    with closing(ExperimentTracker(path)) as tracker:
        assert tracker.get_run_status("run-one") == winner


def test_a_failed_sqlite_update_rolls_back_both_state_and_snapshot(tmp_path):
    path = tmp_path / "rollback.sqlite"
    with closing(ExperimentTracker(path)) as tracker:
        _start(tracker)
        launching = _snapshot()
        running = _snapshot("running", seconds=1)
        assert tracker.initialize_run_status(_OWNER, launching)
        with closing(sqlite3.connect(path)) as conn, conn:
            conn.execute(
                """
                CREATE TRIGGER fail_status BEFORE UPDATE OF run_status_json ON runs
                BEGIN SELECT RAISE(ABORT, 'forced failure'); END
                """
            )
        before = _row(path)
        with pytest.raises(sqlite3.IntegrityError, match="forced failure"):
            tracker.claim_run_status(_OWNER, launching, running)
        assert _row(path) == before
        with closing(sqlite3.connect(path)) as conn, conn:
            conn.execute("DROP TRIGGER fail_status")
        assert tracker.claim_run_status(_OWNER, launching, running)
        assert tracker.get_run_status("run-one") == running


def _prior_status_schema(path):
    with closing(ExperimentTracker(path)) as tracker:
        _start(tracker)
    with closing(sqlite3.connect(path)) as conn:
        for column in _COLUMNS.split(", "):
            conn.execute(f"ALTER TABLE runs DROP COLUMN {column}")


def _read_legacy_views(tracker):
    run = tracker.get_run("run-one")
    assert run["run_id"] == "run-one"
    assert tracker.list_runs() == [run]
    assert tracker.list_active_execution_runs() == [run]


def _expect_upgrade_error(tracker, reason):
    launching = _snapshot()
    running = _snapshot("running", seconds=1)
    later = _snapshot("running", seconds=2)
    for method, args in (
        (tracker.get_run_status, ("run-one",)),
        (tracker.initialize_run_status, (_OWNER, launching)),
        (tracker.claim_run_status, (_OWNER, launching, running)),
        (tracker.update_run_status, (_OWNER, running, later)),
    ):
        with pytest.raises(RunStatusStorageError, match=f"could not be upgraded.*{reason}"):
            method(*args)


def test_read_only_legacy_database_preserves_read_views(tmp_path, monkeypatch):
    path = tmp_path / "read-only.sqlite"
    _prior_status_schema(path)
    connect = sqlite3.connect

    def connect_read_only(database):
        assert database == str(path)
        return connect(f"{path.as_uri()}?mode=ro", uri=True)

    # Read-only is a connection property. chmod can leave the WAL sidecars
    # read-only even after the main file's permissions are restored.
    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", connect_read_only)
        with closing(ExperimentTracker(path)) as tracker:
            _read_legacy_views(tracker)
            _expect_upgrade_error(tracker, "read-only")

    with closing(ExperimentTracker(path)) as tracker:
        assert tracker.get_run_status("run-one") is None
        _read_legacy_views(tracker)


@pytest.mark.parametrize("journal_mode", ["wal", "delete"])
def test_locked_legacy_database_preserves_read_views_and_can_retry(
    tmp_path, monkeypatch, journal_mode
):
    path = tmp_path / "locked-legacy.sqlite"
    _prior_status_schema(path)
    connect = sqlite3.connect

    def connect_with_short_timeout(*args, **kwargs):
        return connect(*args, timeout=0.01, **kwargs)

    with closing(connect(path)) as writer:
        assert writer.execute(f"PRAGMA journal_mode={journal_mode}").fetchone() == (
            journal_mode,
        )
        with writer:
            writer.execute("UPDATE runs SET pid = 42")
        writer.execute("BEGIN IMMEDIATE")
        monkeypatch.setattr(sqlite3, "connect", connect_with_short_timeout)
        monkeypatch.setattr(tracker_module, "_process_is_alive", lambda pid: False)
        with closing(ExperimentTracker(path)) as tracker:
            run = tracker.get_run("run-one")
            assert (run["run_id"], run["status"]) == ("run-one", "terminated")
            assert tracker.list_runs() == [run]
            assert tracker.list_active_execution_runs() == [run]
            assert writer.execute("SELECT status FROM runs").fetchone() == ("running",)
            _expect_upgrade_error(tracker, "locked")
            writer.rollback()
            tracker.init_db()
            assert tracker.get_run_status("run-one") is None
            assert tracker.get_run("run-one")["status"] == "terminated"
            assert tracker.list_active_execution_runs() == []


def test_corrupt_database_keeps_native_open_error_and_leaves_file_untouched(tmp_path):
    path = tmp_path / "corrupt.sqlite"
    contents = b"not a SQLite database" * 64
    path.write_bytes(contents)

    with closing(sqlite3.connect(path)) as conn:
        with pytest.raises(sqlite3.DatabaseError) as native:
            conn.execute("PRAGMA journal_mode=WAL")
    with pytest.raises(sqlite3.DatabaseError) as tracker:
        ExperimentTracker(path)

    assert type(tracker.value) is type(native.value)
    assert str(tracker.value) == str(native.value)
    assert path.read_bytes() == contents
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]


def test_status_reads_report_a_fixed_error_when_the_database_is_corrupt(tmp_path):
    path = tmp_path / "replaced.sqlite"
    with closing(ExperimentTracker(path)) as tracker:
        tracker.close()
        contents = b"not a SQLite database" * 64
        path.write_bytes(contents)

        with pytest.raises(
            RunStatusStorageError,
            match=r"^experiment database is corrupt\. Restore it from a backup\.$",
        ):
            tracker.get_run_status("run-one")

    assert path.read_bytes() == contents


def test_run_status_upgrade_rolls_back_if_a_later_alter_fails(tmp_path, monkeypatch):
    path = tmp_path / "upgrade-rollback.sqlite"
    _prior_status_schema(path)
    connect = sqlite3.connect
    second_column = _COLUMNS.split(", ")[1]

    class InterruptedMigration(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if f"ALTER TABLE runs ADD COLUMN {second_column}" in sql:
                error = sqlite3.DatabaseError("unexpected private diagnostic")
                error.sqlite_errorcode = sqlite3.SQLITE_CORRUPT
                raise error
            return super().execute(sql, parameters)

    def connect_with_interruption(*args, **kwargs):
        return connect(*args, factory=InterruptedMigration, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect_with_interruption)
    with closing(ExperimentTracker(path)) as tracker:
        _read_legacy_views(tracker)
        with pytest.raises(
            RunStatusStorageError,
            match=r"^experiment database is corrupt\. Restore it from a backup\.$",
        ):
            tracker.get_run_status("run-one")
    with closing(connect(path)) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
        assert not columns.intersection(_COLUMNS.split(", "))


def test_status_writes_report_actionable_error_after_busy_timeout(tmp_path):
    path = tmp_path / "write-lock.sqlite"
    with closing(ExperimentTracker(path)) as tracker, closing(sqlite3.connect(path)) as writer:
        _start(tracker)
        tracker._get_conn().execute("PRAGMA busy_timeout=10")
        before = _row(path)
        launching = _snapshot()

        writer.execute("BEGIN IMMEDIATE")
        with pytest.raises(
            RunStatusStorageError, match=r"database is locked\. Retry after the other writer"
        ):
            tracker.initialize_run_status(_OWNER, launching)
        assert _row(path) == before
        writer.rollback()

        assert tracker.initialize_run_status(_OWNER, launching)
        before = _row(path)
        running = _snapshot("running", seconds=1)
        writer.execute("BEGIN IMMEDIATE")
        with pytest.raises(
            RunStatusStorageError, match=r"database is locked\. Retry after the other writer"
        ):
            tracker.claim_run_status(_OWNER, launching, running)
        assert _row(path) == before
        writer.rollback()
        assert tracker.claim_run_status(_OWNER, launching, running)
