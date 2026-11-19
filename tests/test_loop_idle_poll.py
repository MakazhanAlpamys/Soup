"""An idle poll writes no iteration line and no iter-* directory (review 5b, N2).

Every poll that found nothing to train (``not_enough_new_data``, a wait, a hold) used to write a
``skipped`` line to ``iterations.jsonl`` and an ``iter-*`` manifest directory: 1,440 of each a day
at a 60 s poll. Now a cycle that trained nothing, failed nothing and touched no canary writes
neither, bumps no counter (so it does not use up ``max_runs_per_day``), and ``loop.yaml`` keeps one
field, ``last_idle``, with the time and the reason of the latest idle poll (``soup loop status``
shows it as ``last_idle_poll``). An iteration that DID something keeps its line (review 5, F4): a
failed train, a DON'T SHIP verdict, a deploy retry.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import replace

import pytest
from typer.testing import CliRunner

from soup_cli.commands.loop import app as loop_app
from soup_cli.utils.loop_daemon import WatchConfig, run_once, watch
from soup_cli.utils.loop_log import LoopLog
from soup_cli.utils.loop_state import LoopState, read_state, write_state

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
IDLE_REASON = "not_enough_new_data: 0 new edits (needs 8) and 0 new rated answers (needs 30)"


def _plain(text: str) -> str:
    return " ".join(_ANSI.sub("", text).replace("│", " ").split())


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SOUP_LOOP_SERVE_ENDPOINT", raising=False)
    return tmp_path


def _state(**kwargs) -> LoopState:
    return LoopState(
        served_model="base", eval_suite="e", baseline="b", status="running", **kwargs
    )


def _config(root, **fns) -> WatchConfig:
    def idle_train(state, harvested):
        return {"run_id": None, "skipped": True, "adapter_path": None, "notes": IDLE_REASON}

    return WatchConfig(
        poll_interval_sec=1.0, max_iterations=fns.pop("max_iterations", 1),
        state_path=str(root / ".soup" / "loop.yaml"),
        canary_stats_path=str(root / ".soup" / "canary-stats.json"),
        iteration_dir=str(root / ".soup-loops"),
        harvest_fn=fns.pop("harvest_fn", lambda state: {"pairs_harvested": 0}),
        train_fn=fns.pop("train_fn", idle_train),
        gate_fn=fns.pop("gate_fn", lambda state, ctx: {"gate_verdict": "SKIPPED"}),
        deploy_fn=fns.pop("deploy_fn", lambda state, ctx: {"deployed": False}),
        cost_fn=lambda state: 0.0,
        **fns,
    )


def _lines(root, name="iterations.jsonl"):
    path = root / ".soup-loops" / name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _iter_dirs(root):
    base = root / ".soup-loops"
    if not base.exists():
        return []
    return sorted(name for name in os.listdir(base) if name.startswith("iter-"))


class TestIdlePollsWriteNothing:
    def test_fifty_idle_polls_add_no_line_and_no_directory(self, cwd):
        state = _state()
        log = LoopLog(str(cwd / ".soup-loops"))
        config = _config(cwd)

        state, _, _ = run_once(state, config, log=log)  # the first cycle writes the seed line
        lines_after_seed = len(_lines(cwd))
        for _ in range(50):
            state, _, decision = run_once(state, config, log=log)
            assert decision.proceed is False

        assert len(_lines(cwd)) == lines_after_seed == 1  # only the seed ("initial deploy")
        assert not (cwd / ".soup-loops" / "progress.jsonl").exists()

    def test_idle_polls_through_watch_leave_no_iteration_directory(self, cwd):
        write_state(_state(), str(cwd / ".soup" / "loop.yaml"))

        watch(_config(cwd, max_iterations=3))

        assert _iter_dirs(cwd) == []
        assert [line["outcome"] for line in _lines(cwd)] == ["promoted"]  # the seed only

    def test_an_idle_poll_bumps_no_counter_so_it_never_uses_up_the_day(self, cwd):
        state = _state(max_runs_per_day=2)
        config = _config(cwd)

        for _ in range(10):
            state, _, _ = run_once(state, config, log=LoopLog(str(cwd / ".soup-loops")))

        assert (state.iteration_count, state.runs_today) == (0, 0)
        assert state.last_iteration_id is None

    def test_the_returned_record_still_names_the_reason(self, cwd):
        _, record, decision = run_once(_state(), _config(cwd))

        assert IDLE_REASON in record.notes
        assert decision.reason == f"idle poll: {IDLE_REASON}"

    def test_last_idle_keeps_the_time_and_the_reason_of_the_latest_poll(self, cwd):
        state, _, _ = run_once(_state(), _config(cwd))

        assert set(state.last_idle) == {"at", "reason"}
        assert state.last_idle["reason"] == IDLE_REASON
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", state.last_idle["at"])

    def test_a_reason_is_one_clean_line_of_at_most_300_characters(self, cwd):
        def train(state, harvested):
            return {"run_id": None, "skipped": True, "notes": "a\nb\x1b[31m" + "c" * 500}

        state, _, _ = run_once(_state(), _config(cwd, train_fn=train))

        reason = state.last_idle["reason"]
        assert len(reason) == 300 and "\x1b" not in reason and "\n" not in reason

    def test_the_notes_of_the_other_stages_are_kept_with_the_skip_reason(self, cwd):
        """A skip note is not lost when the deploy stage also reports one (#1909)."""
        config = _config(cwd, deploy_fn=lambda s, ctx: {"deployed": False, "notes": "deploy-hi"})

        state, _, _ = run_once(_state(), config)

        assert state.last_idle["reason"] == f"{IDLE_REASON} | deploy-hi"

    def test_no_reason_given_says_nothing_to_train(self, cwd):
        def train(state, harvested):
            return {"run_id": None, "skipped": True}

        state, _, _ = run_once(_state(), _config(cwd, train_fn=train))

        assert state.last_idle["reason"] == "nothing to train"


class TestWhatDidSomethingStillGetsItsLine:
    def test_a_failed_train_gets_a_line_and_a_manifest(self, cwd):
        write_state(_state(), str(cwd / ".soup" / "loop.yaml"))

        def failing(state, harvested):
            return {"run_id": None, "skipped": False, "adapter_path": None, "failed": True,
                    "notes": "train failed (exit code 1)"}

        watch(_config(cwd, harvest_fn=lambda s: {"pairs_harvested": 3}, train_fn=failing))

        state = read_state(str(cwd / ".soup" / "loop.yaml"))
        assert [line["outcome"] for line in _lines(cwd)] == ["promoted", "failed"]
        assert len(_iter_dirs(cwd)) == 1
        assert (state.iteration_count, state.runs_today) == (1, 1)
        assert state.last_idle is None

    def test_a_dont_ship_verdict_gets_a_line(self, cwd):
        trained = {"run_id": "run-1", "skipped": False, "adapter_path": "out/a"}
        config = _config(
            cwd, harvest_fn=lambda s: {"pairs_harvested": 3}, train_fn=lambda s, h: trained,
            gate_fn=lambda s, t: {"gate_verdict": "MAJOR", "notes": "regression"},
        )

        run_once(_state(), config, log=LoopLog(str(cwd / ".soup-loops")))

        assert [line["outcome"] for line in _lines(cwd)] == ["promoted", "skipped"]
        assert _lines(cwd)[-1]["verdict"]["decision"] == "DON'T SHIP"

    def test_a_cycle_that_retries_a_refused_deploy_gets_a_line(self, cwd):
        trained = {"run_id": "run-1", "skipped": False, "adapter_path": "out/a"}
        config = _config(
            cwd, harvest_fn=lambda s: {"pairs_harvested": 3}, train_fn=lambda s, h: trained,
            gate_fn=lambda s, t: {"gate_verdict": "OK"},
            deploy_fn=lambda s, o: {"deployed": False, "failed": True, "notes": "serve restarting"},
        )
        state = _state()

        state, _, decision = run_once(state, config, log=LoopLog(str(cwd / ".soup-loops")))
        state, _, retried = run_once(state, config, log=LoopLog(str(cwd / ".soup-loops")))

        assert decision.proceed and retried.proceed
        assert state.pending_deploy is not None
        assert [line["outcome"] for line in _lines(cwd)].count("failed") >= 1

    def test_a_running_idle_loop_after_real_work_does_not_erase_the_real_lines(self, cwd):
        log = LoopLog(str(cwd / ".soup-loops"))
        state = _state()
        failing = {"run_id": None, "skipped": False, "adapter_path": None, "failed": True,
                   "notes": "train failed (exit code 1)"}
        state, _, _ = run_once(
            state,
            _config(cwd, harvest_fn=lambda s: {"pairs_harvested": 3},
                    train_fn=lambda s, h: failing),
            log=log,
        )
        before = len(_lines(cwd))

        for _ in range(5):
            state, _, _ = run_once(state, _config(cwd), log=log)

        assert len(_lines(cwd)) == before
        assert state.last_idle is not None


class TestStatusAndState:
    def test_status_has_the_last_idle_poll_row(self, cwd):
        state = replace(
            _state(), last_idle={"at": "2026-11-17T00:00:00+00:00", "reason": IDLE_REASON}
        )
        write_state(state, str(cwd / ".soup" / "loop.yaml"))

        result = CliRunner().invoke(loop_app, ["status"])

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert f"last_idle_poll 2026-11-17T00:00:00+00:00: {IDLE_REASON}" in _plain(result.output)

    def test_status_has_no_row_before_the_first_idle_poll(self, cwd):
        write_state(_state(), str(cwd / ".soup" / "loop.yaml"))

        result = CliRunner().invoke(loop_app, ["status"])

        assert "last_idle_poll" not in _plain(result.output)

    def test_it_round_trips_and_is_the_last_field(self, cwd):
        path = str(cwd / ".soup" / "loop.yaml")
        value = {"at": "2026-11-17T00:00:00+00:00", "reason": "r"}
        write_state(replace(_state(), last_idle=value), path)

        assert read_state(path).last_idle == value
        assert list(LoopState.__dataclass_fields__)[-2:] == ["last_idle", "failed_stage"]

    @pytest.mark.parametrize(
        "bad",
        [
            "x", [], {}, {"at": "t"}, {"at": "t", "reason": "r", "x": 1}, {"at": 5, "reason": "r"},
            {"at": "t", "reason": ""}, {"at": "t", "reason": "a\x1bb"},
            {"at": "t", "reason": "x" * 301}, {"at": "t" * 65, "reason": "r"},
        ],
    )
    def test_a_bad_value_is_refused(self, bad):
        with pytest.raises((ValueError, TypeError)):
            _state(last_idle=bad)
