"""The attempt cursor moves only when a batch was consumed by a decided verdict.

A trained candidate whose deploy failed, or whose verdict was skipped (``soup ship`` crashed),
used to advance ``attempt_cursor_ts`` anyway, so the loop waited for a whole new batch and the
candidate was lost. Now:

* a SKIPPED verdict keeps the batch (cursor unchanged);
* DON'T SHIP is a decided verdict (cursor moves, the reason is recorded);
* SHIP deployed moves the cursor;
* a failed deploy keeps the trained candidate (``pending_deploy`` in loop.yaml) and the next
  cycle retries only the deploy, at most ``DEPLOY_RETRY_LIMIT`` times in all; then the
  iteration is recorded as failed with the reason and the batch is consumed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from soup_cli.utils.loop_daemon import DEPLOY_RETRY_LIMIT, WatchConfig, watch
from soup_cli.utils.loop_state import LoopState, read_state, write_state

CURSOR = 100.0
SHIPPED = {
    "deployed": True, "canary_active": "cand", "canary_adapter": "out/a1",
    "canary_traffic_pct": 5.0, "canary_rollout_id": "r1",
}
DEPLOY_REFUSED = {"deployed": False, "failed": True, "notes": "HTTP 503: serve restarting"}


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


class Cycle:
    """Stub stages that remember how often each ran and what the deploy was given."""

    def __init__(self, root: Path, *, gate, deploy, cycles: int = 1) -> None:
        self.harvests = 0
        self.trains = 0
        self.deploys: list = []
        self.records: list = []
        self._gate = gate
        self._deploy = deploy
        write_state(
            LoopState(served_model="base", eval_suite="e.json", baseline="b", status="running"),
            str(root / ".soup" / "loop.yaml"),
        )
        self.config = WatchConfig(
            poll_interval_sec=1.0, max_iterations=cycles,
            state_path=str(root / ".soup" / "loop.yaml"),
            iteration_dir=str(root / ".soup-loops"),
            harvest_fn=self.harvest, train_fn=self.train, gate_fn=self.gate,
            deploy_fn=self.deploy, cost_fn=lambda state: 0.0,
            on_iteration=self.records.append,
        )
        self.path = str(root / ".soup" / "loop.yaml")

    def harvest(self, state):
        self.harvests += 1
        return {"pairs_harvested": 3, "traces_collected": 3, "cursor_ts": CURSOR}

    def train(self, state, harvest):
        self.trains += 1
        return {"run_id": f"run-{self.trains}", "skipped": False, "adapter_path": "out/a1"}

    def gate(self, state, train_out):
        return dict(self._gate)

    def deploy(self, state, outs):
        self.deploys.append(dict(outs))
        result = self._deploy
        return dict(result(len(self.deploys)) if callable(result) else result)

    def run(self) -> LoopState:
        watch(self.config)
        return read_state(self.path)


class TestDecidedVerdicts:
    def test_a_skipped_verdict_keeps_the_batch(self, cwd):
        cycle = Cycle(
            cwd, deploy={"deployed": False},
            gate={"gate_verdict": "SKIPPED", "notes": "verdict_skipped: soup ship crashed"},
        )

        state = cycle.run()

        assert state.attempt_cursor_ts is None
        assert state.pending_deploy is None

    def test_dont_ship_consumes_the_batch(self, cwd):
        cycle = Cycle(cwd, gate={"gate_verdict": "MAJOR"}, deploy={"deployed": False})

        assert cycle.run().attempt_cursor_ts == CURSOR

    def test_ship_deployed_consumes_the_batch(self, cwd):
        cycle = Cycle(cwd, gate={"gate_verdict": "OK"}, deploy=SHIPPED)

        state = cycle.run()

        assert state.attempt_cursor_ts == CURSOR and state.canary_active == "cand"
        assert state.pending_deploy is None


class TestFailedDeploy:
    def test_the_first_failure_keeps_the_candidate_and_the_batch(self, cwd):
        cycle = Cycle(cwd, gate={"gate_verdict": "OK"}, deploy=DEPLOY_REFUSED)

        state = cycle.run()

        assert state.attempt_cursor_ts is None
        assert state.pending_deploy is not None
        assert state.pending_deploy["attempts"] == 1
        assert state.pending_deploy["train"]["adapter_path"] == "out/a1"
        assert "retried" in cycle.records[0].notes
        assert "serve restarting" in cycle.records[0].notes

    def test_the_next_cycle_retries_only_the_deploy_and_then_consumes_the_batch(self, cwd):
        cycle = Cycle(
            cwd, gate={"gate_verdict": "OK"}, cycles=2,
            deploy=lambda call: DEPLOY_REFUSED if call == 1 else SHIPPED,
        )

        state = cycle.run()

        assert (cycle.harvests, cycle.trains) == (1, 1), "the candidate was trained again"
        assert len(cycle.deploys) == 2
        assert cycle.deploys[1]["adapter_path"] == "out/a1"
        assert cycle.deploys[1]["gate_verdict"] == "OK"
        assert state.pending_deploy is None
        assert state.attempt_cursor_ts == CURSOR and state.canary_active == "cand"
        assert cycle.records[1].shipped is True

    def test_a_deploy_that_never_works_is_recorded_as_failed_after_the_limit(self, cwd):
        cycle = Cycle(
            cwd, gate={"gate_verdict": "OK"}, cycles=DEPLOY_RETRY_LIMIT,
            deploy=DEPLOY_REFUSED,
        )

        state = cycle.run()

        assert len(cycle.deploys) == DEPLOY_RETRY_LIMIT
        assert (cycle.harvests, cycle.trains) == (1, 1)
        assert state.pending_deploy is None
        assert state.attempt_cursor_ts == CURSOR
        last = cycle.records[-1].notes
        assert f"deploy failed after {DEPLOY_RETRY_LIMIT} attempts" in last
        assert "serve restarting" in last
        assert state.last_failure is not None and "serve restarting" in state.last_failure

    def test_the_limit_is_a_small_named_constant(self):
        assert DEPLOY_RETRY_LIMIT == 3
