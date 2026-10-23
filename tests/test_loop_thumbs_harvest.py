"""#1440 (producer half) — the loop must learn from `soup serve --record-thumbs`.

`soup serve --record-thumbs PATH` stores every `POST /v1/thumbs` as a
`(prompt, response, thumb)` row in a local-RL SQLite, and
`local_rl.harvest_dpo_pairs` already turns those rows into DPO pairs. The
pre-wired loop harvest read only `soup serve --trace-log` directories, whose
records carry no thumbs signal, so a loop next to a serve that recorded thumbs
harvested nothing.

The harvest now also reads a thumbs database named by `SOUP_LOOP_THUMBS_DB` or
by `soup loop init --thumbs-db PATH` (persisted in `.soup/loop.yaml`).
"""

from __future__ import annotations

import json
import os
import sqlite3

import pytest
from typer.testing import CliRunner

from tests.conftest import strip_ansi


def _record(db: str, *rows) -> None:
    from soup_cli.utils import local_rl

    if not os.path.exists(db):
        local_rl.init_local_rl_db(db)
    for prompt, response, thumb in rows:
        local_rl.record_thumb(db_path=db, prompt=prompt, response=response, thumb=thumb)


def _write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _read_jsonl(path) -> list:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _state(**kwargs):
    from soup_cli.utils.loop_state import LoopState

    # A registry ref, so the served model is never mistaken for a trace dir.
    return LoopState(served_model="registry://dummy", eval_suite="e", baseline="b", **kwargs)


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SOUP_LOOP_TRACE_DIR", raising=False)
    monkeypatch.delenv("SOUP_LOOP_THUMBS_DB", raising=False)
    return tmp_path


def test_the_issue_reproduction_harvests_the_recorded_pair(cwd, monkeypatch):
    """One up and one down thumb for the same prompt, no trace directory at all."""
    from soup_cli.utils.loop_stages import harvest_from_traces

    _record("thumbs.db", ("Q", "Good", "up"), ("Q", "Bad", "down"))
    monkeypatch.setenv("SOUP_LOOP_THUMBS_DB", "thumbs.db")

    result = harvest_from_traces(_state())

    assert result["pairs_harvested"] == 1, result
    assert result["thumbs_pairs"] == 1, result
    pairs = _read_jsonl(result["pairs_path"])
    assert pairs == [
        {"prompt": "Q", "chosen": "Good", "rejected": "Bad", "source": "soup-serve-thumbs"}
    ]


def test_the_database_named_in_loop_state_is_read(cwd):
    from soup_cli.utils.loop_stages import harvest_from_traces

    _record("thumbs.db", ("Q", "Good", "up"), ("Q", "Bad", "down"))

    result = harvest_from_traces(_state(thumbs_db="thumbs.db"))

    assert result["pairs_harvested"] == 1, result


def test_the_environment_overrides_loop_state(cwd, monkeypatch):
    """Same precedence as the trace directory: env first, then loop.yaml."""
    from soup_cli.utils.loop_stages import harvest_from_traces

    _record("a.db", ("Q", "Good", "up"), ("Q", "Bad", "down"))
    _record("b.db", ("Q2", "Yes", "up"), ("Q2", "No", "down"))
    monkeypatch.setenv("SOUP_LOOP_THUMBS_DB", "b.db")

    pairs = _read_jsonl(harvest_from_traces(_state(thumbs_db="a.db"))["pairs_path"])

    assert [(p["prompt"], p["chosen"]) for p in pairs] == [("Q2", "Yes")]


def test_trace_and_thumbs_pairs_are_unioned_without_duplicates(cwd, monkeypatch):
    from soup_cli.utils.loop_stages import harvest_from_traces

    def _trace(i, prompt, good, bad):
        return [
            {"id": f"{i}a", "prompt": prompt, "output": good, "signal": "thumbs_up"},
            {"id": f"{i}b", "prompt": prompt, "output": bad, "signal": "thumbs_down"},
        ]

    _write_jsonl(
        cwd / "traces" / "t.jsonl",
        _trace(1, "Q", "Good", "Bad") + _trace(2, "R", "G2", "B2") + _trace(3, "Q2", "Yes", "No"),
    )
    _record(
        "thumbs.db",
        ("Q", "Good", "up"), ("Q", "Bad2", "down"),    # same prompt+chosen, other rejected: kept
        ("R", "G3", "up"), ("R", "B2", "down"),        # same prompt+rejected, other chosen: kept
        ("Q2", "Yes", "up"), ("Q2", "No", "down"),     # exact duplicate of a trace pair: dropped
        ("Q3", "A", "up"), ("Q3", "B", "down"),        # new prompt: kept
    )
    monkeypatch.setenv("SOUP_LOOP_TRACE_DIR", str(cwd / "traces"))
    monkeypatch.setenv("SOUP_LOOP_THUMBS_DB", "thumbs.db")

    result = harvest_from_traces(_state())

    assert result["traces_collected"] == 6, result
    assert result["thumbs_pairs"] == 4, result  # pairs READ from the database, duplicates included
    assert result["pairs_harvested"] == 6, result
    rows = _read_jsonl(result["pairs_path"])
    # Trace pairs first, then the thumbs pairs the union kept (order decides the val_split tail).
    assert [p["source"] for p in rows] == ["thumbs_up"] * 3 + ["soup-serve-thumbs"] * 3, rows
    got = sorted((p["prompt"], p["chosen"], p["rejected"], p["source"]) for p in rows)
    assert got == [
        ("Q", "Good", "Bad", "thumbs_up"),
        ("Q", "Good", "Bad2", "soup-serve-thumbs"),
        ("Q2", "Yes", "No", "thumbs_up"),  # the duplicate keeps the trace row's source
        ("Q3", "A", "B", "soup-serve-thumbs"),
        ("R", "G2", "B2", "thumbs_up"),
        ("R", "G3", "B2", "soup-serve-thumbs"),
    ]


def test_thumbs_pairs_is_reported_when_nothing_was_harvested(cwd, monkeypatch):
    """Zero-pair path: a database with only up-thumbs reads 0 pairs, and says so."""
    from soup_cli.utils.loop_stages import harvest_from_traces

    _record("thumbs.db", ("Q", "Good", "up"))
    monkeypatch.setenv("SOUP_LOOP_THUMBS_DB", "thumbs.db")

    result = harvest_from_traces(_state())

    assert result["pairs_harvested"] == 0, result
    assert result["pairs_path"] is None, result
    assert result["thumbs_pairs"] == 0, result


@pytest.mark.parametrize("via", ["env", "loop.yaml"])
def test_a_configured_database_that_does_not_exist_fails_the_stage_by_name(
    cwd, monkeypatch, via
):
    """Maintainer ruling: missing fails the stage naming the field, never a silent 0."""
    from soup_cli.utils.loop_stages import harvest_from_traces

    if via == "env":
        monkeypatch.setenv("SOUP_LOOP_THUMBS_DB", "not-yet.db")
        state, field = _state(), "SOUP_LOOP_THUMBS_DB"
    else:
        state, field = _state(thumbs_db="not-yet.db"), "thumbs_db"

    with pytest.raises(ValueError, match=f"^{field}: no database file at 'not-yet.db'"):
        harvest_from_traces(state)
    assert not os.path.exists(os.path.join(".soup-loops", "pairs"))


def test_a_directory_at_the_database_path_is_refused_too(cwd, monkeypatch):
    from soup_cli.utils.loop_stages import harvest_from_traces

    os.mkdir("thumbs.db")
    monkeypatch.setenv("SOUP_LOOP_THUMBS_DB", "thumbs.db")

    with pytest.raises(ValueError, match="SOUP_LOOP_THUMBS_DB: no database file"):
        harvest_from_traces(_state())


def test_a_database_outside_the_project_is_refused_by_name(cwd, monkeypatch, tmp_path_factory):
    """The file EXISTS, so only the containment rule can be what refuses it."""
    from soup_cli.utils.loop_stages import harvest_from_traces

    outside = tmp_path_factory.mktemp("elsewhere") / "thumbs.db"
    sqlite3.connect(str(outside)).close()
    monkeypatch.setenv("SOUP_LOOP_THUMBS_DB", str(outside))

    with pytest.raises(
        ValueError,
        match=r"^SOUP_LOOP_THUMBS_DB: db_path 'thumbs.db' must stay under cwd; "
        r"move the database under the project directory$",
    ):
        harvest_from_traces(_state())


@pytest.mark.requires_symlink
def test_a_symlinked_database_is_refused_by_name(cwd):
    from soup_cli.utils.loop_stages import harvest_from_traces

    _record("real.db", ("Q", "Good", "up"), ("Q", "Bad", "down"))
    os.symlink("real.db", "link.db")

    with pytest.raises(
        ValueError,
        match=r"^thumbs_db: db_path must not be a symlink.*; "
        r"move the database under the project directory$",
    ):
        harvest_from_traces(_state(thumbs_db="link.db"))


def test_a_refusal_that_is_not_about_location_gets_no_move_advice(cwd, monkeypatch):
    from soup_cli.utils.loop_stages import checked_thumbs_db

    monkeypatch.setenv("SOUP_LOOP_THUMBS_DB", "a\nb.db")

    with pytest.raises(
        ValueError,
        match=r"^SOUP_LOOP_THUMBS_DB: db_path must not contain newline / carriage return$",
    ):
        checked_thumbs_db(_state())


def test_without_a_thumbs_database_the_harvest_is_unchanged(cwd):
    """Control: no env, no loop.yaml field, no trace dir -> the old zero result."""
    from soup_cli.utils.loop_stages import harvest_from_traces

    result = harvest_from_traces(_state())

    assert result["pairs_harvested"] == 0, result
    assert result["pairs_path"] is None, result
    assert result["traces_collected"] == 0, result


def test_loop_init_persists_thumbs_db_and_an_old_state_file_still_loads(cwd):
    from soup_cli.cli import app
    from soup_cli.utils.loop_state import default_state_path, read_state

    runner = CliRunner()
    result = runner.invoke(app, [
        "loop", "init", "registry://m", "--eval", "suite.yaml", "--baseline", "b",
        "--thumbs-db", "thumbs.db",
    ])
    assert result.exit_code == 0, strip_ansi(result.output)
    assert read_state().thumbs_db == "thumbs.db"

    # A loop.yaml written before the field existed loads with thumbs_db unset.
    path = default_state_path()
    data = json.loads(open(path, encoding="utf-8").read())
    del data["thumbs_db"]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    assert read_state().thumbs_db is None


def test_loop_init_refuses_a_thumbs_db_outside_the_project(cwd, tmp_path_factory):
    from soup_cli.cli import app

    outside = tmp_path_factory.mktemp("elsewhere") / "thumbs.db"
    result = CliRunner().invoke(app, [
        "loop", "init", "registry://m", "--eval", "suite.yaml", "--baseline", "b",
        "--thumbs-db", str(outside),
    ])
    text = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 2, text
    # The option exists and the refusal is the containment rule, not a usage error.
    assert "No such option" not in text, text
    assert "--thumbs-db" in text and "under cwd" in text, text
    assert not os.path.exists(os.path.join(".soup", "loop.yaml"))


def test_thumbs_db_is_the_last_field_so_positional_fields_keep_their_meaning():
    """Adding the field must not shift `traces_collected` and the rest for positional callers."""
    import dataclasses

    from soup_cli.utils.loop_state import LoopState

    names = [f.name for f in dataclasses.fields(LoopState)]
    assert names[-1] == "thumbs_db", names
    state = LoopState("m", "e", "b", "stopped", False, 7)
    assert state.traces_collected == 7
    assert state.thumbs_db is None


def _make_bad_db(kind: str, path: str = "thumbs.db") -> None:
    if kind == "text":
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("this is not a database\n" * 20)
    else:  # a valid SQLite file with no thumbs table (empty, or another Soup .db)
        sqlite3.connect(path).close()


@pytest.mark.parametrize("kind", ["text", "empty-sqlite"])
@pytest.mark.parametrize("via", ["env", "loop.yaml"])
def test_a_file_that_is_not_a_thumbs_database_fails_the_stage_by_name(
    cwd, monkeypatch, via, kind
):
    from soup_cli.utils.loop_stages import harvest_from_traces

    _make_bad_db(kind)
    if via == "env":
        monkeypatch.setenv("SOUP_LOOP_THUMBS_DB", "thumbs.db")
        state, field = _state(), "SOUP_LOOP_THUMBS_DB"
    else:
        state, field = _state(thumbs_db="thumbs.db"), "thumbs_db"

    with pytest.raises(
        ValueError,
        match=rf"^{field}: 'thumbs.db' is not a readable thumbs database "
        r"\(\w*Error\); soup serve --record-thumbs creates one$",
    ):
        harvest_from_traces(state)
    assert not os.path.exists(os.path.join(".soup-loops", "pairs"))


def test_a_locked_database_is_not_refused_by_the_check(cwd, monkeypatch):
    """A running serve can hold the lock briefly; that is not a bad database."""
    import time

    from soup_cli.utils.loop_stages import checked_thumbs_db

    _record("thumbs.db", ("Q", "Good", "up"))
    monkeypatch.setenv("SOUP_LOOP_THUMBS_DB", "thumbs.db")
    holder = sqlite3.connect("thumbs.db", isolation_level=None)
    try:
        holder.execute("BEGIN EXCLUSIVE")
        started = time.monotonic()
        assert checked_thumbs_db(_state()) == "thumbs.db"
        assert time.monotonic() - started < 3, "the check must not wait out sqlite's 5 s default"
    finally:
        holder.close()


@pytest.mark.parametrize("via", ["env", "loop.yaml"])
def test_a_sqlite_failure_mid_harvest_names_the_field(cwd, monkeypatch, via):
    """The file passed the check, then sqlite failed while reading (lock, corruption)."""
    import soup_cli.utils.local_rl as local_rl
    from soup_cli.utils.loop_stages import harvest_from_traces

    _record("thumbs.db", ("Q", "Good", "up"), ("Q", "Bad", "down"))
    if via == "env":
        monkeypatch.setenv("SOUP_LOOP_THUMBS_DB", "thumbs.db")
        state, field = _state(), "SOUP_LOOP_THUMBS_DB"
    else:
        state, field = _state(thumbs_db="thumbs.db"), "thumbs_db"

    def _boom(_path):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(local_rl, "harvest_dpo_pairs", _boom)

    with pytest.raises(
        ValueError,
        match=rf"^{field}: 'thumbs.db' could not be read \(OperationalError: database is locked\)$",
    ):
        harvest_from_traces(state)


def test_control_characters_in_a_configured_path_never_reach_the_message(cwd, monkeypatch):
    from soup_cli.utils.loop_stages import checked_thumbs_db

    monkeypatch.setenv("SOUP_LOOP_THUMBS_DB", "a\x1b[2Jb.db")

    with pytest.raises(ValueError) as info:
        checked_thumbs_db(_state())

    assert "\x1b[2J" not in str(info.value)
    assert "no database file at" in str(info.value)


def _stored_state_with(**fields):
    from soup_cli.utils.loop_state import default_state_path, read_state

    _init_loop(True)
    path = default_state_path()
    data = json.loads(open(path, encoding="utf-8").read())
    data.update(fields)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return read_state


@pytest.mark.parametrize("bad", ["", 123, "x" * 513])
def test_read_state_refuses_a_bad_thumbs_db(cwd, bad):
    read_state = _stored_state_with(thumbs_db=bad)

    with pytest.raises(ValueError, match="thumbs_db"):
        read_state()


def test_an_empty_environment_variable_is_treated_as_unset(cwd, monkeypatch):
    """Like SOUP_LOOP_TRACE_DIR: empty falls through to loop.yaml, or to nothing."""
    from soup_cli.utils.loop_stages import harvest_from_traces

    _record("a.db", ("Q", "Good", "up"), ("Q", "Bad", "down"))
    monkeypatch.setenv("SOUP_LOOP_THUMBS_DB", "")

    assert harvest_from_traces(_state(thumbs_db="a.db"))["pairs_harvested"] == 1
    assert harvest_from_traces(_state())["pairs_harvested"] == 0


# --- `soup loop watch` refuses a bad thumbs database before it starts --------


def _init_loop(pre_wired: bool, *extra: str) -> None:
    from soup_cli.cli import app

    args = ["loop", "init", "registry://m", "--eval", "suite.yaml", "--baseline", "b"]
    if pre_wired:
        args.append("--pre-wired")
    result = CliRunner().invoke(app, args + list(extra))
    assert result.exit_code == 0, strip_ansi(result.output)


def _watch(*extra: str):
    from soup_cli.cli import app

    result = CliRunner().invoke(
        app, ["loop", "watch", "--max-iterations", "1", "--poll-interval", "1", *extra]
    )
    return result, " ".join(strip_ansi(result.output).split())


@pytest.mark.parametrize("how", ["init --pre-wired", "watch --pre-wired"])
def test_watch_refuses_a_missing_database_before_any_iteration(cwd, monkeypatch, how):
    from soup_cli.utils.loop_state import read_state

    _init_loop(pre_wired=how.startswith("init"))
    monkeypatch.setenv("SOUP_LOOP_THUMBS_DB", "not-yet.db")

    result, text = _watch(*(["--pre-wired"] if how.startswith("watch") else []))

    assert result.exit_code == 2, text
    assert "invalid thumbs database" in text, text
    assert "SOUP_LOOP_THUMBS_DB: no database file at 'not-yet.db'" in text, text
    assert "Traceback" not in text, text
    assert not os.path.exists(".soup-loops"), "no iteration may run"
    assert read_state().status == "stopped"


def test_watch_refuses_the_loop_yaml_field_and_an_outside_path_by_name(
    cwd, monkeypatch, tmp_path_factory
):
    _init_loop(True, "--thumbs-db", "gone.db")
    result, text = _watch()
    assert result.exit_code == 2, text
    assert "thumbs_db: no database file at 'gone.db'" in text, text

    outside = tmp_path_factory.mktemp("elsewhere") / "thumbs.db"
    monkeypatch.setenv("SOUP_LOOP_THUMBS_DB", str(outside))
    result, text = _watch()
    assert result.exit_code == 2, text
    assert "SOUP_LOOP_THUMBS_DB:" in text and "under cwd" in text, text


def test_watch_detach_refuses_before_spawning(cwd, monkeypatch):
    import soup_cli.commands.loop as loop_cmd

    spawned = []
    monkeypatch.setattr(loop_cmd.subprocess, "Popen", lambda *a, **k: spawned.append(a))
    _init_loop(True)
    monkeypatch.setenv("SOUP_LOOP_THUMBS_DB", "not-yet.db")

    result, text = _watch("--detach")

    assert result.exit_code == 2, text
    assert spawned == []


def test_watch_with_a_recorded_database_harvests_through_the_cli(cwd, monkeypatch):
    """Control: an existing database passes the check and the iteration records the pair."""
    import soup_cli.utils.loop_stages as ls
    from soup_cli.utils.loop_iteration import list_iterations, read_iteration

    class _Failed:
        returncode = 1  # the train subprocess is not under test here

    monkeypatch.setattr(ls, "_TRAIN_RUNNER", lambda *a, **k: _Failed())
    _record("thumbs.db", ("Q", "Good", "up"), ("Q", "Bad", "down"))
    _init_loop(True, "--thumbs-db", "thumbs.db")

    result, text = _watch()

    assert result.exit_code == 0, text
    assert "stub stages never read it" not in text, text
    ids = list_iterations()
    assert len(ids) == 1, ids
    assert read_iteration(ids[0]).pairs_harvested == 1


def test_watch_without_pre_wired_stages_does_not_check_the_database(cwd, monkeypatch):
    """Control: the stub harvest never reads it, so it is not refused there."""
    _init_loop(False)
    monkeypatch.setenv("SOUP_LOOP_THUMBS_DB", "not-yet.db")

    result, text = _watch()

    assert result.exit_code == 0, text
    assert "invalid thumbs database" not in text, text


@pytest.mark.parametrize("kind", ["text", "empty-sqlite"])
def test_watch_refuses_a_file_that_is_not_a_thumbs_database(cwd, monkeypatch, kind):
    from soup_cli.utils.loop_state import read_state

    _make_bad_db(kind)
    _init_loop(True)
    monkeypatch.setenv("SOUP_LOOP_THUMBS_DB", "thumbs.db")

    result, text = _watch()

    assert result.exit_code == 2, text
    assert "invalid thumbs database" in text, text
    assert "SOUP_LOOP_THUMBS_DB: 'thumbs.db' is not a readable thumbs database" in text, text
    assert "Traceback" not in text, text
    assert not os.path.exists(".soup-loops"), "no iteration may run"
    assert read_state().status == "stopped"


@pytest.mark.parametrize("kind", ["text", "empty-sqlite"])
def test_watch_detach_refuses_a_file_that_is_not_a_thumbs_database(cwd, monkeypatch, kind):
    import soup_cli.commands.loop as loop_cmd

    spawned = []
    monkeypatch.setattr(loop_cmd.subprocess, "Popen", lambda *a, **k: spawned.append(a))
    _make_bad_db(kind)
    _init_loop(True)
    monkeypatch.setenv("SOUP_LOOP_THUMBS_DB", "thumbs.db")

    result, text = _watch("--detach")

    assert result.exit_code == 2, text
    assert "not a readable thumbs database" in text, text
    assert spawned == []


def _make_wrong_columns_db(columns: str = "x", path: str = "thumbs.db") -> None:
    """A hand-made ``thumbs`` table that lacks some columns the harvest reads."""
    conn = sqlite3.connect(path)
    conn.execute(f"CREATE TABLE thumbs({columns})")
    conn.commit()
    conn.close()


@pytest.mark.parametrize("missing", ["prompt", "response", "thumb", "ts"])
def test_watch_refuses_a_thumbs_table_without_the_harvest_columns(cwd, monkeypatch, missing):
    kept = [c for c in ("prompt", "response", "thumb", "ts") if c != missing]
    _make_wrong_columns_db(", ".join(kept))
    _init_loop(True)
    monkeypatch.setenv("SOUP_LOOP_THUMBS_DB", "thumbs.db")

    result, text = _watch()

    assert result.exit_code == 2, text
    assert "is not a readable thumbs database" in text, text
    assert "Traceback" not in text, text
    assert not os.path.exists(".soup-loops"), "no iteration may run"


def test_watch_detach_refuses_a_thumbs_table_without_the_harvest_columns(cwd, monkeypatch):
    import soup_cli.commands.loop as loop_cmd

    spawned = []
    monkeypatch.setattr(loop_cmd.subprocess, "Popen", lambda *a, **k: spawned.append(a))
    _make_wrong_columns_db()
    _init_loop(True)
    monkeypatch.setenv("SOUP_LOOP_THUMBS_DB", "thumbs.db")

    result, text = _watch("--detach")

    assert result.exit_code == 2, text
    assert "is not a readable thumbs database" in text, text
    assert spawned == []


def test_watch_still_accepts_a_database_made_by_init_local_rl_db(cwd, monkeypatch):
    """Control: the real schema passes the column-list probe."""
    from soup_cli.utils import local_rl

    local_rl.init_local_rl_db("thumbs.db")
    _init_loop(True)
    monkeypatch.setenv("SOUP_LOOP_THUMBS_DB", "thumbs.db")

    result, text = _watch()

    assert "not a readable thumbs database" not in text, text
    assert result.exit_code == 0, text


def test_watch_does_not_let_a_control_sequence_in_the_path_reach_the_terminal(cwd, monkeypatch):
    _init_loop(True)
    monkeypatch.setenv("SOUP_LOOP_THUMBS_DB", "a\x1b[2Jb.db")

    result, _text = _watch()

    assert result.exit_code == 2, result.output
    assert "\x1b[2J" not in result.output  # the sequence itself, not "any ESC" (Rich adds its own)
    assert "no database file at" in " ".join(strip_ansi(result.output).split())


def test_init_does_not_let_a_control_sequence_in_the_path_reach_the_terminal(cwd):
    from soup_cli.cli import app

    result = CliRunner().invoke(app, [
        "loop", "init", "registry://m", "--eval", "suite.yaml", "--baseline", "b",
        "--thumbs-db", "../a\x1b[2Jb.db",
    ])

    assert result.exit_code == 2, result.output
    assert "\x1b[2J" not in result.output


@pytest.mark.parametrize("via", ["env", "loop.yaml"])
def test_watch_warns_when_the_stub_stages_will_not_read_the_database(cwd, monkeypatch, via):
    if via == "env":
        _init_loop(False)
        monkeypatch.setenv("SOUP_LOOP_THUMBS_DB", "not-yet.db")
    else:
        _init_loop(False, "--thumbs-db", "not-yet.db")

    result, text = _watch()

    assert result.exit_code == 0, text
    assert "stub stages never read it" in text and "--pre-wired" in text, text
    # An existing loop cannot be re-initialised without --force, which resets it.
    assert "loop init" not in text, text


@pytest.mark.parametrize("env", [None, ""])
def test_watch_does_not_warn_without_a_thumbs_database(cwd, monkeypatch, env):
    _init_loop(False)
    if env is not None:
        monkeypatch.setenv("SOUP_LOOP_THUMBS_DB", env)

    result, text = _watch()

    assert result.exit_code == 0, text
    assert "--pre-wired" not in text, text


# --- a Rich tag in a refused path stays literal (#1440) ----------------------


def test_init_refusal_keeps_rich_tags_in_the_path_literal(cwd):
    from soup_cli.cli import app

    result = CliRunner().invoke(app, [
        "loop", "init", "registry://m", "--eval", "suite.yaml", "--baseline", "b",
        "--thumbs-db", "../db[bold]x.db",
    ])
    plain = " ".join(strip_ansi(result.output).split())

    assert result.exit_code == 2, plain
    assert "invalid --thumbs-db:" in plain, plain
    assert "db[bold]x.db" in plain, plain  # the brackets survive; nothing is parsed as markup
    assert "'dbx.db'" not in plain, plain


def test_watch_refusal_keeps_rich_tags_in_the_path_literal(cwd, monkeypatch):
    (cwd / "db[bold]x.db").write_text("not a database", encoding="utf-8")
    _init_loop(True)
    monkeypatch.setenv("SOUP_LOOP_THUMBS_DB", "db[bold]x.db")

    result, text = _watch()

    assert result.exit_code == 2, text
    assert "invalid thumbs database:" in text, text
    assert "db[bold]x.db" in text, text  # the brackets survive; nothing is parsed as markup
    assert "'dbx.db'" not in text, text
