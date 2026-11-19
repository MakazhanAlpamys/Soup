"""A promotion the server refused ends the cycle there: nothing is trained on top of it.

A canary that passed its answers is promoted by telling the server. When the server refuses
(a 401, a restarted server that no longer holds the adapter), the canary stays. The cycle used
to go on all the same: harvest, train (hours of card time), gate, then the deploy stage refused
("canary X is still running; not starting another"), the batch counted as consumed and the data
cursor moved past rows no candidate was ever deployed for. That repeated on every poll while the
server kept refusing and new feedback kept arriving. Now the cycle ends at the refusal: it is
recorded with its reason (``last_failure``, the progress log), no stage runs, no counter moves
and no cursor advances. The next poll tries the promotion again.
"""

from __future__ import annotations

import json
import os

import pytest

from soup_cli.utils.canary_router import CanaryPolicy, record_bucket_outcome
from soup_cli.utils.loop_daemon import WatchConfig, run_once, watch
from soup_cli.utils.loop_log import LoopLog
from soup_cli.utils.loop_state import LoopState, read_state, write_state

REFUSAL = "canary v2 passed but was not promoted: 401"


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _state(**kwargs) -> LoopState:
    return LoopState(
        served_model="./base", base_model="org/base", eval_suite="e", baseline="b",
        status="running", canary_active="v2", canary_traffic_pct=5.0, canary_rollout_id="r1",
        canary_adapter="adapters/v2", **kwargs,
    )


def _passing_canary(state: LoopState) -> None:
    policy = CanaryPolicy(stable=state.stable_name(), canary="v2", traffic_pct=5.0)
    stats = os.path.join(".soup", "canary-stats.json")
    for bucket, count in (("stable", 40), ("canary", 30)):
        for _ in range(count):
            record_bucket_outcome(policy, bucket, True, rollout_id="r1", path=stats)


class Stages:
    """Counting stages: a call means the cycle went on past the refusal."""

    def __init__(self, accept: "list[bool] | None" = None) -> None:
        self.calls: "list[str]" = []
        self._accept = list(accept or [False])

    def harvest(self, state):
        self.calls.append("harvest")
        return {"pairs_harvested": 3, "cursor_ts": 1_700_000_000.0}

    def train(self, state, harvested):
        self.calls.append("train")
        return {"run_id": "run-1", "skipped": False, "adapter_path": "adapters/new"}

    def gate(self, state, trained):
        self.calls.append("gate")
        return {"gate_verdict": "OK"}

    def deploy(self, state, trained):
        self.calls.append("deploy")
        return {"deployed": False, "failed": True, "notes": "canary v2 is still running"}

    def promote(self, state):
        self.calls.append("promote")
        accepted = self._accept.pop(0) if len(self._accept) > 1 else self._accept[0]
        return {"promoted": True} if accepted else {"promoted": False, "notes": "401"}

    def config(self, root=None, **kwargs) -> WatchConfig:
        paths = {}
        if root is not None:
            paths = {
                "state_path": str(root / ".soup" / "loop.yaml"),
                "iteration_dir": str(root / ".soup-loops"),
            }
        return WatchConfig(
            max_iterations=kwargs.pop("max_iterations", 1), poll_interval_sec=1.0,
            canary_stats_path=os.path.join(".soup", "canary-stats.json"),
            harvest_fn=self.harvest, train_fn=self.train, gate_fn=self.gate,
            deploy_fn=self.deploy, promote_fn=self.promote, cost_fn=lambda state: 0.0,
            **paths, **kwargs,
        )


def _lines(root, name):
    path = root / ".soup-loops" / name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_a_refused_promotion_runs_no_other_stage(cwd):
    state = _state()
    _passing_canary(state)
    stages = Stages()

    new_state, record, decision = run_once(state, stages.config())

    assert stages.calls == ["promote"]
    assert decision.proceed is False
    assert new_state.canary_active == "v2"
    assert REFUSAL in record.notes


def test_a_refused_promotion_advances_no_cursor_and_no_counter(cwd):
    state = _state()
    _passing_canary(state)

    new_state, _, _ = run_once(state, Stages().config())

    assert new_state.attempt_cursor_ts == state.attempt_cursor_ts
    assert new_state.canary_cursor_ts == state.canary_cursor_ts
    assert (new_state.iteration_count, new_state.runs_today) == (0, 0)
    assert new_state.pending_deploy is None
    assert new_state.production_name is None


def test_the_refusal_is_recorded_with_its_reason(cwd):
    state = _state()
    _passing_canary(state)
    log = LoopLog(str(cwd / ".soup-loops"))

    new_state, record, decision = run_once(state, Stages().config(), log=log)

    assert new_state.last_failure == REFUSAL
    assert decision.reason == REFUSAL
    errors = [line for line in _lines(cwd, "progress.jsonl") if line.get("event") == "error"]
    assert [(line["stage"], line["error"]) for line in errors] == [("promote", "promote_refused")]
    assert REFUSAL in errors[0]["message"]


def test_a_refused_promotion_writes_no_iteration_line_and_no_manifest(cwd):
    write_state(_state(), str(cwd / ".soup" / "loop.yaml"))
    _passing_canary(_state())
    stages = Stages()

    watch(stages.config(cwd, max_iterations=3))

    assert stages.calls == ["promote"] * 3  # tried again every poll, nothing else ran
    # Only the seed line ("initial deploy", written once when the version log starts).
    assert [line["outcome"] for line in _lines(cwd, "iterations.jsonl")] == ["promoted"]
    assert not [name for name in os.listdir(cwd / ".soup-loops") if name.startswith("iter-")]
    saved = read_state(str(cwd / ".soup" / "loop.yaml"))
    assert saved.canary_active == "v2"
    assert saved.last_failure == REFUSAL
    assert saved.attempt_cursor_ts is None


def test_feedback_that_keeps_arriving_is_not_trained_on_while_the_server_refuses(cwd):
    state = _state()
    _passing_canary(state)
    stages = Stages()
    log = LoopLog(str(cwd / ".soup-loops"))

    for _ in range(5):
        state, _, _ = run_once(state, stages.config(), log=log)

    assert "train" not in stages.calls and "harvest" not in stages.calls
    assert state.attempt_cursor_ts is None


def test_when_the_server_accepts_a_later_promotion_the_cycle_goes_on_as_before(cwd):
    state = _state()
    _passing_canary(state)
    stages = Stages(accept=[False, True])

    state, _, _ = run_once(state, stages.config())
    assert stages.calls == ["promote"]
    state, record, decision = run_once(state, stages.config())

    assert state.canary_active is None and state.production_name == "v2"
    assert stages.calls[1:] == ["promote", "harvest", "train", "gate", "deploy"]
    assert decision.proceed is True
    assert "canary v2 promoted to production" in record.notes


def test_the_refusal_note_is_gone_once_the_promotion_goes_through(cwd):
    state = _state()
    _passing_canary(state)
    stages = Stages(accept=[False, True])

    state, _, _ = run_once(state, stages.config())
    assert state.last_failure == REFUSAL
    state, _, _ = run_once(state, stages.config())

    assert state.last_failure != REFUSAL
