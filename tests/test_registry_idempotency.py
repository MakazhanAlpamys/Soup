"""``RegistryStore.push(idempotency_key=...)``: a retried registration finds its entry (#1763)."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.registry.store import (
    IdempotencyConflictError,
    RegistryStore,
    validate_idempotency_key,
)
from tests.conftest import strip_ansi

runner = CliRunner()

CONFIG = {"base": "llama", "task": "sft", "lr": 2e-4}

# registry_entries and registry_tags as every soup-cli before #1763 created them.
_LEGACY_SCHEMA = """
CREATE TABLE registry_entries (
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
CREATE TABLE registry_tags (
    entry_id TEXT NOT NULL REFERENCES registry_entries(id) ON DELETE CASCADE,
    tag      TEXT NOT NULL,
    PRIMARY KEY (entry_id, tag)
);
"""


@pytest.fixture()
def store(tmp_path):
    with RegistryStore(db_path=tmp_path / "reg.db") as s:
        yield s


def _push(store: RegistryStore, key=None, **overrides) -> str:
    request = {
        "name": "chat", "tag": "v1", "base_model": "llama", "task": "sft",
        "run_id": "run_1", "config": CONFIG,
    }
    request.update(overrides)
    return store.push(idempotency_key=key, **request)


def _counts(db: Path) -> tuple[int, int]:
    conn = sqlite3.connect(str(db))
    try:
        entries = conn.execute("SELECT COUNT(*) FROM registry_entries").fetchone()[0]
        tags = conn.execute("SELECT COUNT(*) FROM registry_tags").fetchone()[0]
    finally:
        conn.close()
    return entries, tags


def _columns(db: Path) -> list[str]:
    conn = sqlite3.connect(str(db))
    try:
        return [row[1] for row in conn.execute("PRAGMA table_info(registry_entries)")]
    finally:
        conn.close()


def _legacy_db(db: Path) -> None:
    """A database written by today's soup-cli: two twin entries and one tag each."""
    conn = sqlite3.connect(str(db))
    try:
        conn.executescript(_LEGACY_SCHEMA)
        for entry_id in ("reg_old_1", "reg_old_2"):
            conn.execute(
                "INSERT INTO registry_entries (id, name, base_model, task, run_id, config_json,"
                " config_hash, data_hash, entry_hash, created_at, notes)"
                " VALUES (?, 'chat', 'llama', 'sft', 'run_old', '{}', 'c', NULL, 'e',"
                " '2026-01-01T00:00:00', NULL)",
                (entry_id,),
            )
            conn.execute("INSERT INTO registry_tags VALUES (?, 'v1')", (entry_id,))
        conn.commit()
    finally:
        conn.close()


# -- repeat and conflict ------------------------------------------------------


def test_a_repeated_push_returns_the_same_entry_and_writes_nothing(store):
    first = _push(store, key="job-42")

    second = _push(store, key="job-42")

    assert second == first
    assert _counts(store.db_path) == (1, 1)
    assert store.get(first)["idempotency_key"] == "job-42"


def test_the_entry_is_found_by_its_key(store):
    entry_id = _push(store, key="job-42")

    found = store.get_by_idempotency_key("job-42")

    assert found["id"] == entry_id
    assert found["tags"] == ["v1"]
    assert store.get_by_idempotency_key("job-43") is None


@pytest.mark.parametrize(
    ("field", "override"),
    [
        ("name", {"name": "other"}),
        ("tag", {"tag": "v2"}),
        ("base_model", {"base_model": "qwen"}),
        ("task", {"task": "dpo"}),
        ("run_id", {"run_id": "run_2"}),
        ("config_hash", {"config": {**CONFIG, "lr": 1e-4}}),
    ],
)
def test_the_same_key_for_a_different_request_is_refused(store, field, override):
    first = _push(store, key="job-42")

    with pytest.raises(IdempotencyConflictError) as excinfo:
        _push(store, key="job-42", **override)

    message = str(excinfo.value)
    assert first in message
    assert field in message
    assert _counts(store.db_path) == (1, 1)


def test_a_different_data_file_under_the_same_key_is_refused(store, tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text": "a"}\n', encoding="utf-8")
    first = _push(store, key="job-42", data_path=str(data))
    assert _push(store, key="job-42", data_path=str(data)) == first

    data.write_text('{"text": "b"}\n', encoding="utf-8")
    with pytest.raises(IdempotencyConflictError, match="data_hash"):
        _push(store, key="job-42", data_path=str(data))
    with pytest.raises(IdempotencyConflictError, match="data_hash"):
        _push(store, key="job-42")
    assert _counts(store.db_path) == (1, 1)


def test_the_conflict_is_a_value_error(store):
    _push(store, key="job-42")

    with pytest.raises(ValueError):
        _push(store, key="job-42", run_id="run_2")


def test_notes_are_not_part_of_the_request(store):
    first = _push(store, key="job-42", notes="first try")

    assert _push(store, key="job-42", notes="retry") == first
    assert store.get(first)["notes"] == "first try"


def test_a_tag_added_later_does_not_break_the_retry(store):
    first = _push(store, key="job-42")
    store.add_tag(first, "prod")

    assert _push(store, key="job-42") == first
    assert store.get(first)["tags"] == ["prod", "v1"]


def test_pushes_without_a_key_never_collide(store):
    first = _push(store)
    second = _push(store)

    assert first != second
    assert _counts(store.db_path) == (2, 2)
    assert store.get(first)["idempotency_key"] is None


def test_different_keys_make_different_entries(store):
    assert _push(store, key="job-42") != _push(store, key="job-43")
    assert _counts(store.db_path) == (2, 2)


def test_a_deleted_entry_frees_its_key(store):
    first = _push(store, key="job-42")
    store.delete(first)

    second = _push(store, key="job-42")

    assert second != first
    assert _counts(store.db_path) == (1, 1)


# -- key format -----------------------------------------------------------------


@pytest.mark.parametrize("key", ["a", "x" * 128, "run:20261015.a_b-c", "0"])
def test_valid_keys(key):
    validate_idempotency_key(key)


@pytest.mark.parametrize(
    "key",
    ["", "x" * 129, "a b", "a/b", "a\\b", "a\n", "\na", "a\x00", "ключ", "a;b", "a%b", 42],
)
def test_invalid_keys_are_refused_before_anything_is_written(store, key):
    with pytest.raises(ValueError, match="idempotency key"):
        validate_idempotency_key(key)
    with pytest.raises(ValueError, match="idempotency key"):
        _push(store, key=key)
    with pytest.raises(ValueError, match="idempotency key"):
        store.get_by_idempotency_key(key)
    assert _counts(store.db_path) == (0, 0)


# -- schema upgrade ---------------------------------------------------------------


def test_a_new_database_has_the_column_and_the_unique_index(tmp_path):
    db = tmp_path / "reg.db"
    RegistryStore(db_path=db).close()

    assert "idempotency_key" in _columns(db)
    conn = sqlite3.connect(str(db))
    try:
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'idx_reg_entries_idempotency_key'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert "UNIQUE" in sql.upper()
    assert "WHERE idempotency_key IS NOT NULL" in sql


def test_control_a_legacy_database_has_no_column(tmp_path):
    """If this failed, the upgrade tests below would not be testing an upgrade."""
    db = tmp_path / "legacy.db"
    _legacy_db(db)

    assert "idempotency_key" not in _columns(db)


def test_a_legacy_database_is_upgraded_in_place(tmp_path):
    db = tmp_path / "legacy.db"
    _legacy_db(db)
    conn = sqlite3.connect(str(db))
    before = conn.execute("SELECT * FROM registry_entries ORDER BY id").fetchall()
    conn.close()

    with RegistryStore(db_path=db) as store:
        assert "idempotency_key" in _columns(db)
        old = store.get("reg_old_1")
        assert old["idempotency_key"] is None
        assert old["tags"] == ["v1"]
        first = _push(store, key="job-42")
        assert _push(store, key="job-42") == first

    conn = sqlite3.connect(str(db))
    after = conn.execute(
        "SELECT * FROM registry_entries WHERE id LIKE 'reg_old_%' ORDER BY id"
    ).fetchall()
    conn.close()
    width = len(before[0])
    assert [row[:width] for row in after] == before  # every legacy value is unchanged
    assert [row[width:] for row in after] == [(None, None, None, None)] * 2
    assert _counts(db) == (3, 3)


class _RecordingConn:
    """Records every statement passed to ``execute``, including ones that then fail.

    sqlite3's trace callback misses a statement refused while it is prepared, such as an
    ALTER that adds a column the table already has, so it can't tell a guarded upgrade
    from one that relies on the duplicate-column handler.
    """

    def __init__(self, conn: sqlite3.Connection, statements: list[str]):
        self._conn = conn
        self._statements = statements

    def execute(self, sql, *args):
        self._statements.append(sql)
        return self._conn.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def test_opening_an_upgraded_database_again_issues_no_alter(tmp_path, monkeypatch):
    db = tmp_path / "legacy.db"
    _legacy_db(db)
    real_get_conn = RegistryStore._get_conn
    statements: list[str] = []
    monkeypatch.setattr(
        RegistryStore, "_get_conn", lambda self: _RecordingConn(real_get_conn(self), statements),
    )

    RegistryStore(db_path=db).close()  # first open: must ALTER
    assert any("ALTER TABLE" in q.upper() for q in statements)
    statements.clear()
    RegistryStore(db_path=db).close()  # second: must not

    assert [q for q in statements if "ALTER TABLE" in q.upper()] == []
    assert any("PRAGMA table_info(registry_entries)" in q for q in statements)


class _Rows(list):
    def fetchall(self):
        return list(self)


class _StaleColumnsConn:
    """A connection whose column check misses a column another process just added."""

    def __init__(self, conn: sqlite3.Connection, db: Path):
        self._conn = conn
        self._db = db

    def execute(self, sql, *args):
        if sql.startswith("PRAGMA table_info(registry_entries)"):
            stale = self._conn.execute(sql, *args).fetchall()
            other = sqlite3.connect(str(self._db))
            other.execute("ALTER TABLE registry_entries ADD COLUMN idempotency_key TEXT")
            other.commit()
            other.close()
            return _Rows(stale)
        return self._conn.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def test_a_column_added_by_another_process_mid_upgrade_is_tolerated(tmp_path, monkeypatch):
    db = tmp_path / "legacy.db"
    _legacy_db(db)
    real_get_conn = RegistryStore._get_conn

    def racing_get_conn(self):
        conn = real_get_conn(self)
        return _StaleColumnsConn(conn, self.db_path)

    monkeypatch.setattr(RegistryStore, "_get_conn", racing_get_conn)
    store = RegistryStore(db_path=db)
    monkeypatch.setattr(RegistryStore, "_get_conn", real_get_conn)
    try:
        assert _columns(db).count("idempotency_key") == 1
        first = _push(store, key="job-42")
        assert _push(store, key="job-42") == first
    finally:
        store.close()


# -- two writers ------------------------------------------------------------------


def test_a_key_taken_between_the_lookup_and_the_insert_returns_that_entry(tmp_path, monkeypatch):
    """The insert loses to a writer that committed after this push looked the key up."""
    db = tmp_path / "reg.db"
    with RegistryStore(db_path=db) as winner, RegistryStore(db_path=db) as loser:
        lookups: list[str] = []
        real_find = RegistryStore._find_by_key

        def find_misses_first(key):
            lookups.append(key)
            return None if len(lookups) == 1 else real_find(loser, key)

        monkeypatch.setattr(loser, "_find_by_key", find_misses_first)
        first = _push(winner, key="job-42")

        assert _push(loser, key="job-42") == first
        assert lookups == ["job-42", "job-42"]

        lookups.clear()
        with pytest.raises(IdempotencyConflictError):
            _push(loser, key="job-42", run_id="run_2")
        assert lookups == ["job-42", "job-42"]
    assert _counts(db) == (1, 1)


def test_two_stores_pushing_the_same_key_at_once_leave_one_entry(tmp_path):
    db = tmp_path / "reg.db"
    RegistryStore(db_path=db).close()
    writers = 4
    barrier = threading.Barrier(writers)
    results: list[str] = []
    errors: list[BaseException] = []

    def push_once():
        try:
            with RegistryStore(db_path=db) as store:
                barrier.wait()
                results.append(_push(store, key="job-42"))
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=push_once) for _ in range(writers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(results) == writers
    assert len(set(results)) == 1
    assert _counts(db) == (1, 1)


# -- run registration and the CLI ---------------------------------------------------


def _finished_run() -> str:
    from soup_cli.experiment.tracker import ExperimentTracker

    tracker = ExperimentTracker()
    try:
        run_id = tracker.start_run(
            config_dict=CONFIG, device="cpu", device_name="cpu", gpu_info={},
        )
        tracker.finish_run(run_id, 2.0, 0.5, 100, 60.0, "out")
    finally:
        tracker.close()
    return run_id


@pytest.fixture()
def work(tmp_path, monkeypatch):
    """A working directory with isolated experiment and registry databases."""
    cwd = tmp_path / "work"
    cwd.mkdir()
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "exp.db"))
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))
    monkeypatch.chdir(cwd)
    return tmp_path


def test_register_from_run_passes_the_key_through(work):
    from soup_cli.experiment.tracker import ExperimentTracker

    run_id = _finished_run()
    tracker = ExperimentTracker()
    try:
        with RegistryStore() as store:
            first = store.register_from_run(
                tracker, run_id, name="chat", tag="v1", idempotency_key="job-42",
            )
            again = store.register_from_run(
                tracker, run_id, name="chat", tag="v1", idempotency_key="job-42",
            )
            assert again == first
            assert store.get(first)["idempotency_key"] == "job-42"
            with pytest.raises(IdempotencyConflictError):
                store.register_from_run(
                    tracker, run_id, name="chat", tag="v2", idempotency_key="job-42",
                )
    finally:
        tracker.close()
    assert _counts(work / "reg.db") == (1, 1)


def _cli_push(run_id: str, *extra: str):
    return runner.invoke(
        app, ["registry", "push", "--run-id", run_id, "--name", "chat", "--tag", "v1", *extra],
    )


def test_cli_retry_with_the_same_key_reports_the_existing_entry(work):
    run_id = _finished_run()

    first = _cli_push(run_id, "--idempotency-key", "job-42")
    second = _cli_push(run_id, "--idempotency-key", "job-42")

    assert first.exit_code == 0, (first.output, repr(first.exception))
    assert second.exit_code == 0, (second.output, repr(second.exception))
    assert "Registered" in strip_ansi(first.output)
    with RegistryStore() as store:
        entry = store.get_by_idempotency_key("job-42")
    out = strip_ansi(second.output)
    assert "Already registered" in out
    assert entry["id"] in out
    assert _counts(work / "reg.db") == (1, 1)


def test_cli_same_key_for_a_different_tag_fails_naming_the_entry(work):
    run_id = _finished_run()
    assert _cli_push(run_id, "--idempotency-key", "job-42").exit_code == 0

    result = runner.invoke(
        app,
        ["registry", "push", "--run-id", run_id, "--name", "chat", "--tag", "v2",
         "--idempotency-key", "job-42"],
    )

    assert result.exit_code == 1
    out = strip_ansi(result.output)
    assert "job-42" in out and "tag" in out
    assert "Traceback" not in out
    assert _counts(work / "reg.db") == (1, 1)


def test_cli_invalid_key_is_refused_before_any_database_is_opened(work):
    result = _cli_push("run_x", "--idempotency-key", "bad key")

    assert result.exit_code == 1
    assert "idempotency key" in strip_ansi(result.output)
    assert not (work / "reg.db").exists()
    assert not (work / "exp.db").exists()


def test_cli_without_a_key_is_unchanged(work):
    run_id = _finished_run()

    assert _cli_push(run_id).exit_code == 0
    assert _cli_push(run_id).exit_code == 0

    with RegistryStore() as store:
        entries = store.list(run_id=run_id)
    assert len(entries) == 2
    assert {e["idempotency_key"] for e in entries} == {None}
