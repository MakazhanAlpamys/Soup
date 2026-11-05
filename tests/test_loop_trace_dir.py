"""The pre-wired harvest reads the trace directory it was given, and nothing else.

``_resolve_trace_dir`` used to try the served model path when it was a local
directory, then ``.soup-loops/traces``: a model directory that held ``*.jsonl``
files (a dataset next to the weights) was harvested as serve logs. Now only
``soup loop init --trace-dir`` (saved as ``trace_dir`` in ``loop.yaml``) or
``SOUP_LOOP_TRACE_DIR`` names it.
"""

from __future__ import annotations

import json
import os
import re

import pytest
from typer.testing import CliRunner

import soup_cli.utils.loop_stages as ls
from soup_cli.utils.loop_state import LoopState, read_state

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _clean(text: str) -> str:
    return " ".join(_ANSI.sub("", text).split())


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    for name in ("SOUP_LOOP_TRACE_DIR", "SOUP_LOOP_THUMBS_DB", "SOUP_LOOP_SERVE_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def _write_traces(directory) -> None:
    """One prompt with a thumbs-up and a thumbs-down answer = one preference pair."""
    os.makedirs(directory, exist_ok=True)
    rows = [
        {"id": "a", "prompt": "Q", "response": "good", "signal": "thumbs_up"},
        {"id": "b", "prompt": "Q", "response": "bad", "signal": "thumbs_down"},
    ]
    with open(os.path.join(directory, "serve.jsonl"), "w", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in rows))


def _state(**kwargs) -> LoopState:
    kwargs.setdefault("served_model", "./model")
    kwargs.setdefault("base_model", "org/base")
    return LoopState(eval_suite="e", baseline="b", **kwargs)


def _invoke(*args: str):
    from soup_cli.commands.loop import app

    result = runner.invoke(app, list(args))
    return result, _clean(result.output)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_configured_trace_dir_is_harvested(cwd):
    _write_traces("logs")
    out = ls.harvest_from_traces(_state(trace_dir="logs"))
    assert out["traces_collected"] == 2
    assert out["pairs_harvested"] == 1


def test_served_model_directory_is_never_read_as_traces(cwd):
    _write_traces("model")
    out = ls.harvest_from_traces(_state(served_model="model"))
    assert out["traces_collected"] == 0
    assert out["pairs_harvested"] == 0


def test_old_implicit_directory_is_not_read_without_configuration(cwd):
    _write_traces(os.path.join(".soup-loops", "traces"))
    out = ls.harvest_from_traces(_state())
    assert out["traces_collected"] == 0


def test_environment_overrides_loop_yaml(cwd, monkeypatch):
    _write_traces("env-logs")
    os.makedirs("yaml-logs")
    monkeypatch.setenv("SOUP_LOOP_TRACE_DIR", "env-logs")
    assert ls._resolve_trace_dir(_state(trace_dir="yaml-logs")) == "env-logs"


def test_missing_configured_directory_harvests_nothing(cwd):
    assert ls._resolve_trace_dir(_state(trace_dir="not-yet")) is None


# ---------------------------------------------------------------------------
# trace_dir_notice (checked before a pre-wired watch)
# ---------------------------------------------------------------------------


def test_notice_is_silent_for_an_existing_directory(cwd):
    os.makedirs("logs")
    assert ls.trace_dir_notice(_state(trace_dir="logs")) is None


def test_notice_warns_when_the_directory_does_not_exist_yet(cwd):
    notice = ls.trace_dir_notice(_state(trace_dir="logs"))
    assert notice.startswith("trace_dir: no conversation logs at")
    assert "soup serve --trace-log" in notice


def test_notice_refuses_a_file_by_field(cwd, monkeypatch):
    open("logs.jsonl", "w").close()
    with pytest.raises(ValueError, match="trace_dir: .* is not a directory"):
        ls.trace_dir_notice(_state(trace_dir="logs.jsonl"))
    monkeypatch.setenv("SOUP_LOOP_TRACE_DIR", "logs.jsonl")
    with pytest.raises(ValueError, match="SOUP_LOOP_TRACE_DIR: .* is not a directory"):
        ls.trace_dir_notice(_state())


def test_notice_names_the_old_implicit_directory(cwd):
    os.makedirs(os.path.join(".soup-loops", "traces"))
    notice = ls.trace_dir_notice(_state())
    assert ".soup-loops/traces is not read unless it is configured" in notice
    assert "--trace-dir .soup-loops/traces" in notice


def test_notice_is_silent_when_nothing_is_configured_or_left_over(cwd):
    assert ls.trace_dir_notice(_state()) is None


# ---------------------------------------------------------------------------
# soup loop init / status / watch
# ---------------------------------------------------------------------------


def test_init_saves_trace_dir_and_status_shows_it(cwd):
    os.makedirs("logs")
    result, text = _invoke(
        "init", "./model", "--eval", "e", "--baseline", "b", "--trace-dir", "logs"
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert read_state().trace_dir == "logs"
    assert "trace_dir: logs" in text
    status, status_text = _invoke("status")
    assert status.exit_code == 0, (status.output, repr(status.exception))
    assert re.search(r"trace_dir\W+logs", status_text), status_text


def test_init_without_trace_dir_says_not_set(cwd):
    result, text = _invoke("init", "./model", "--eval", "e", "--baseline", "b")
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert read_state().trace_dir is None
    assert "trace_dir: not set" in text


def test_init_warns_for_a_directory_that_does_not_exist_yet(cwd):
    result, text = _invoke(
        "init", "./model", "--eval", "e", "--baseline", "b", "--trace-dir", "logs"
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "no conversation logs at logs yet" in text
    assert read_state().trace_dir == "logs"


@pytest.mark.parametrize(
    ("value", "message"),
    [("", "--trace-dir must not be empty"), ("logs.jsonl", "is not a directory")],
)
def test_init_refuses_a_bad_trace_dir(cwd, value, message):
    open("logs.jsonl", "w").close()
    result, text = _invoke(
        "init", "./model", "--eval", "e", "--baseline", "b", "--trace-dir", value
    )
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert message in text
    assert not os.path.exists(os.path.join(".soup", "loop.yaml"))


def test_watch_pre_wired_refuses_a_trace_dir_that_is_a_file(cwd, monkeypatch):
    _invoke("init", "./model", "--eval", "e", "--baseline", "b", "--base-model", "org/b")
    open("logs.jsonl", "w").close()
    monkeypatch.setenv("SOUP_LOOP_TRACE_DIR", "logs.jsonl")
    result, text = _invoke("watch", "--pre-wired", "--max-iterations", "1", "--poll-interval", "1")
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "invalid trace directory: SOUP_LOOP_TRACE_DIR:" in text
    assert read_state().iteration_count == 0


def test_watch_pre_wired_warns_about_the_old_implicit_directory(cwd, monkeypatch):
    _invoke("init", "./model", "--eval", "e", "--baseline", "b", "--base-model", "org/b")
    os.makedirs(os.path.join(".soup-loops", "traces"))
    calls = []
    monkeypatch.setattr("soup_cli.commands.loop.watch", lambda cfg: calls.append(cfg) or (
        read_state(), 0
    ))
    result, text = _invoke("watch", "--pre-wired", "--max-iterations", "1", "--poll-interval", "1")
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert ".soup-loops/traces is not read unless it is configured" in text
    assert len(calls) == 1


def test_state_file_without_trace_dir_still_loads(cwd):
    os.makedirs(".soup")
    legacy = {"served_model": "./m", "eval_suite": "e", "baseline": "b", "status": "stopped"}
    with open(os.path.join(".soup", "loop.yaml"), "w", encoding="utf-8") as fh:
        json.dump(legacy, fh)
    assert read_state().trace_dir is None


def test_trace_dir_field_is_validated():
    with pytest.raises(ValueError, match="trace_dir must not be empty"):
        _state(trace_dir="")
