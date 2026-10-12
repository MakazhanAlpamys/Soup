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
import sys

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

    _write_jsonl(
        cwd / "traces" / "t.jsonl",
        [
            {"id": "1", "prompt": "Q", "output": "Good", "signal": "thumbs_up"},
            {"id": "2", "prompt": "Q", "output": "Bad", "signal": "thumbs_down"},
        ],
    )
    # The same verdict arrives twice (trace log and database) plus one more prompt.
    _record("thumbs.db", ("Q", "Good", "up"), ("Q", "Bad", "down"),
            ("Q2", "Yes", "up"), ("Q2", "No", "down"))
    monkeypatch.setenv("SOUP_LOOP_TRACE_DIR", str(cwd / "traces"))
    monkeypatch.setenv("SOUP_LOOP_THUMBS_DB", "thumbs.db")

    result = harvest_from_traces(_state())

    assert result["traces_collected"] == 2, result
    assert result["pairs_harvested"] == 2, result
    rows = _read_jsonl(result["pairs_path"])
    got = sorted((p["prompt"], p["chosen"], p["rejected"]) for p in rows)
    assert got == [("Q", "Good", "Bad"), ("Q2", "Yes", "No")]


def test_a_database_the_serve_has_not_created_yet_is_zero_thumbs_not_an_error(cwd, monkeypatch):
    """`soup loop watch` may start before `soup serve --record-thumbs` creates the file."""
    from soup_cli.utils.loop_stages import harvest_from_traces

    monkeypatch.setenv("SOUP_LOOP_THUMBS_DB", "not-yet.db")

    result = harvest_from_traces(_state())

    assert result["pairs_harvested"] == 0, result
    assert result["thumbs_pairs"] == 0, result
    assert result["thumbs_db_missing"] is True, result
    assert result["pairs_path"] is None, result


def test_a_database_outside_the_project_is_refused_by_name(cwd, monkeypatch, tmp_path_factory):
    from soup_cli.utils.loop_stages import harvest_from_traces

    outside = tmp_path_factory.mktemp("elsewhere") / "thumbs.db"
    monkeypatch.setenv("SOUP_LOOP_THUMBS_DB", str(outside))

    with pytest.raises(ValueError, match="SOUP_LOOP_THUMBS_DB"):
        harvest_from_traces(_state())


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_a_symlinked_database_is_refused_by_name(cwd):
    from soup_cli.utils.loop_stages import harvest_from_traces

    _record("real.db", ("Q", "Good", "up"), ("Q", "Bad", "down"))
    os.symlink("real.db", "link.db")

    with pytest.raises(ValueError, match="thumbs_db"):
        harvest_from_traces(_state(thumbs_db="link.db"))


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
