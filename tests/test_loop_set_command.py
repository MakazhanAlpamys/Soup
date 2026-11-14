"""``soup loop set <key> <value>`` changes one setting of an existing loop without a reset.

``soup loop init`` on an existing loop needs ``--force`` and builds a new state: production,
canary, cursors, counters and the version counters go back to their start. The migration notes
for ``--base-model``, ``--trace-dir`` and ``--verify`` therefore name this command: it changes
only the one key, under the state lock, and leaves everything else (and the version log) alone.
"""

from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.loop_state import read_state

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _plain(text: str) -> str:
    return " ".join(_ANSI.sub("", text).split())


def _invoke(*args: str):
    result = runner.invoke(app, ["loop", *args])
    return result, _plain(result.output)


@pytest.fixture
def loop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result, _ = _invoke("init", "org/base", "--eval", "e.json", "--baseline", "b")
    assert result.exit_code == 0, (result.output, repr(result.exception))
    # a loop that has run: something every reset would lose
    from dataclasses import replace

    from soup_cli.utils.loop_state import write_state

    write_state(replace(
        read_state(), production_name="v2", production_adapter="v2", production_version=2,
        latest_version=2, iteration_count=7, attempt_cursor_ts=1_800_000_000.0,
        harvest_cursor_ts=1_799_999_000.0, canary_active="cand", canary_traffic_pct=5.0,
        canary_rollout_id="r1",
    ))
    return tmp_path


def _untouched(before, after, *changed: str) -> None:
    for name in (
        "production_name", "production_adapter", "production_version", "latest_version",
        "iteration_count", "attempt_cursor_ts", "harvest_cursor_ts", "canary_active",
        "canary_rollout_id", "status", "served_model", "eval_suite", "baseline",
        "base_model", "trace_dir", "verify",
    ):
        if name not in changed:
            assert getattr(after, name) == getattr(before, name), name


class TestSet:
    def test_base_model(self, loop):
        before = read_state()

        result, out = _invoke("set", "base_model", "org/other-base")

        assert result.exit_code == 0, (out, repr(result.exception))
        after = read_state()
        assert after.base_model == "org/other-base"
        _untouched(before, after, "base_model")

    def test_trace_dir(self, loop):
        (loop / "traces").mkdir()
        before = read_state()

        result, out = _invoke("set", "trace_dir", "traces")

        assert result.exit_code == 0, (out, repr(result.exception))
        after = read_state()
        assert after.trace_dir == "traces"
        _untouched(before, after, "trace_dir")

    def test_verify_accepts_the_dashed_spelling(self, loop):
        before = read_state()

        result, out = _invoke("set", "verify", "eval-gate")

        assert result.exit_code == 0, (out, repr(result.exception))
        after = read_state()
        assert after.verify == "eval_gate"
        _untouched(before, after, "verify")

    def test_the_version_log_is_not_touched(self, loop):
        versions = loop / ".soup-loops" / "versions.jsonl"
        before = versions.read_bytes() if versions.exists() else None

        _invoke("set", "base_model", "org/other-base")

        assert (versions.read_bytes() if versions.exists() else None) == before


class TestRefusals:
    def test_an_unknown_key_names_the_settable_ones(self, loop):
        result, out = _invoke("set", "production_name", "x")

        assert result.exit_code == 2
        assert "base_model, trace_dir, verify" in out
        assert read_state().production_name == "v2"

    def test_an_empty_base_model(self, loop):
        result, out = _invoke("set", "base_model", "   ")

        assert result.exit_code == 2 and "base_model must not be empty" in out
        assert read_state().base_model == "org/base"

    def test_a_trace_dir_that_is_a_file(self, loop):
        (loop / "afile").write_text("x", encoding="utf-8")

        result, out = _invoke("set", "trace_dir", "afile")

        assert result.exit_code == 2 and "is not a directory" in out

    def test_an_unknown_verify_method_names_the_choices(self, loop):
        result, out = _invoke("set", "verify", "vibes")

        assert result.exit_code == 2
        assert "ship" in out and "eval_gate" in out
        assert read_state().verify != "vibes"

    def test_a_control_byte_in_the_value_is_not_echoed(self, loop):
        result, _ = _invoke("set", "verify", "\x1b]0;x\x07")

        assert result.exit_code == 2
        assert "\x1b" not in result.output and "\x07" not in result.output


def test_set_takes_the_state_lock(loop, monkeypatch):
    import soup_cli.utils.loop_state as loop_state

    entered = []
    real = loop_state.state_lock
    monkeypatch.setattr(
        loop_state, "state_lock", lambda path=None, **kw: entered.append(path) or real(path, **kw)
    )

    result, _ = _invoke("set", "base_model", "org/other-base")

    assert result.exit_code == 0 and entered
