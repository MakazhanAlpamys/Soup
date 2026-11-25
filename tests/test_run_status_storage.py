"""Durable run status storage on the existing SQLite tracker row."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import UTC, datetime, timedelta
from threading import Barrier

import pytest

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
