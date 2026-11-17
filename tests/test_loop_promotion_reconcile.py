"""A promotion is written to loop.yaml when it happens, and a half-done one is finished (N1).

Review 5b: at 23:41:57Z the daemon promoted a canary (the server switched, ``versions.jsonl`` got
version 2) and the same cycle went on into the next harvest, train and ship; ``loop.yaml`` was only
written when that cycle ended. A kill in that hour left ``loop.yaml`` on version 1 with the canary
running while the log said version 2 and the server served the adapter.

The order is now the server switch, then ``loop.yaml`` (atomic, under the lock, at the promotion),
then the log lines. ``loop.yaml`` decides. A kill between two writes, in either order, is finished
when ``soup loop watch`` starts (``utils/loop_reconcile.py``), and ``soup loop status`` names the
mismatch in its ``promotion`` row without writing anything.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace

import pytest
from typer.testing import CliRunner

import soup_cli.utils.loop_daemon as loop_daemon
import soup_cli.utils.loop_reconcile as reconcile
import soup_cli.utils.loop_stages as ls
from soup_cli.commands.loop import app as loop_app
from soup_cli.utils.canary_router import CanaryPolicy, record_bucket_outcome
from soup_cli.utils.loop_daemon import WatchConfig, watch
from soup_cli.utils.loop_log import LoopLog
from soup_cli.utils.loop_state import LoopState, read_state, update_state, write_state

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _plain(text: str) -> str:
    return " ".join(_ANSI.sub("", text).replace("│", " ").split())


class _Killed(BaseException):
    """A process kill: not an ``Exception``, so no handler of the daemon sees it."""


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SOUP_LOOP_SERVE_ENDPOINT", raising=False)
    return tmp_path


class World:
    """A loop with a stub pipeline in a temp project: a canary to start, make healthy, promote."""

    def __init__(self, root) -> None:
        self.root = root
        self.path = str(root / ".soup" / "loop.yaml")
        self.stats = str(root / ".soup" / "canary-stats.json")
        self.promoted_on_server: list = []
        write_state(
            LoopState(served_model="base", eval_suite="e", baseline="b", status="running"),
            self.path,
        )

    def config(self, *, max_iterations=1, train_fn=None, **fns) -> WatchConfig:
        def promote(state):
            self.promoted_on_server.append(state.canary_active)
            return {"promoted": True}

        return WatchConfig(
            poll_interval_sec=1.0, max_iterations=max_iterations, state_path=self.path,
            canary_stats_path=self.stats, iteration_dir=str(self.root / ".soup-loops"),
            harvest_fn=fns.get("harvest_fn") or (
                lambda state: {"pairs_harvested": 3, "traces_collected": 3}
            ),
            train_fn=train_fn or (
                lambda state, h: {"run_id": "run-1", "skipped": False, "adapter_path": "out/a1"}
            ),
            gate_fn=lambda state, t: {"gate_verdict": "OK"},
            deploy_fn=lambda state, o: {
                "deployed": True, "canary_active": "cand", "canary_adapter": "out/a1",
                "canary_traffic_pct": 5.0, "canary_rollout_id": "r1",
            },
            cost_fn=lambda state: 0.0, promote_fn=promote,
        )

    def idle(self, **kwargs) -> WatchConfig:
        """A pipeline with nothing to train: the cycle only decides about the running canary."""
        return self.config(
            harvest_fn=lambda state: {"pairs_harvested": 0, "traces_collected": 0},
            train_fn=lambda state, h: {"run_id": None, "skipped": True, "adapter_path": None},
            **kwargs,
        )

    def start_canary(self) -> None:
        watch(self.config())  # trains and starts the canary "cand"
        assert read_state(self.path).canary_active == "cand"

    def make_healthy(self) -> None:
        policy = CanaryPolicy(stable="base", canary="cand", traffic_pct=5.0)
        for _ in range(30):
            for bucket in ("stable", "canary"):
                record_bucket_outcome(policy, bucket, True, rollout_id="r1", path=self.stats)

    def state(self) -> LoopState:
        return read_state(self.path)

    def log(self) -> LoopLog:
        return LoopLog(str(self.root / ".soup-loops"))

    def lines(self, kind: str) -> list:
        try:
            with open(self.log().path(kind), encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]
        except FileNotFoundError:
            return []

    def promote_versions(self) -> list:
        return [line for line in self.lines("version") if line.get("cause") == "promote"]


@pytest.fixture
def world(cwd):
    return World(cwd)


def _killed_train(state, harvested):
    raise _Killed("killed at the start of the next training run")


class TestLoopYamlIsWrittenAtThePromotion:
    def test_a_kill_in_the_next_train_leaves_loop_yaml_promoted(self, world):
        world.start_canary()
        world.make_healthy()

        with pytest.raises(_Killed):
            watch(world.config(train_fn=_killed_train))

        state = world.state()
        assert (state.production_name, state.production_version) == ("cand", 2)
        assert state.latest_version == 2 and state.canary_active is None
        assert state.canary_iteration is None
        assert [line["to_version"] for line in world.promote_versions()] == [2]
        assert world.promoted_on_server == ["cand"]

    def test_the_log_agrees_with_loop_yaml_after_the_kill(self, world):
        world.start_canary()
        world.make_healthy()
        with pytest.raises(_Killed):
            watch(world.config(train_fn=_killed_train))

        assert reconcile.detect(world.state(), world.log()) is None

    def test_a_rollback_made_during_the_rest_of_the_cycle_is_not_undone_by_its_end(self, world):
        world.start_canary()
        world.make_healthy()

        def train_then_a_person_rolls_back(state, harvested):
            result = CliRunner().invoke(loop_app, ["rollback"])
            assert result.exit_code == 0, (result.output, repr(result.exception))
            return {"run_id": None, "skipped": True, "adapter_path": None}

        watch(world.config(train_fn=train_then_a_person_rolls_back))

        state = world.state()
        assert state.production_name is None and state.production_version == 1
        assert state.latest_version == 2  # the number is never reused

    def test_a_promotion_the_server_refused_writes_nothing(self, world):
        world.start_canary()
        world.make_healthy()
        config = replace(
            world.idle(), promote_fn=lambda state: {"promoted": False, "notes": "401"}
        )

        watch(config)

        state = world.state()
        assert state.canary_active == "cand" and state.production_version == 1
        assert world.promote_versions() == []


class TestKilledBetweenTwoWrites:
    def test_killed_after_the_server_switch_and_before_loop_yaml(self, world, monkeypatch):
        """Server first, then loop.yaml: a kill between leaves the canary running, no log line."""
        world.start_canary()
        world.make_healthy()
        with monkeypatch.context() as patched:
            patched.setattr(
                loop_daemon, "merge_state",
                lambda *args, **kwargs: (_ for _ in ()).throw(_Killed("before loop.yaml")),
            )
            with pytest.raises(_Killed):
                watch(world.idle())

        assert world.promoted_on_server == ["cand"]  # the server did switch
        assert world.state().canary_active == "cand" and world.state().production_version == 1
        assert world.promote_versions() == []  # the log is written after loop.yaml: nothing yet
        watch(world.idle())  # the restart: the canary is still healthy, the promotion is repeated

        state = world.state()
        assert (state.production_name, state.production_version) == ("cand", 2)
        assert [line["to_version"] for line in world.promote_versions()] == [2]
        assert world.promoted_on_server == ["cand", "cand"]  # the switch is idempotent

    def test_killed_after_loop_yaml_and_before_the_log(self, world, monkeypatch):
        world.start_canary()
        world.make_healthy()

        def killed(*args, **kwargs):
            raise _Killed("before the log lines")

        with monkeypatch.context() as patched:
            patched.setattr(loop_daemon, "_log_promotion", killed)
            with pytest.raises(_Killed):
                watch(world.idle())

        state = world.state()
        assert (state.production_name, state.production_version) == ("cand", 2)
        assert world.promote_versions() == []
        found = reconcile.detect(state, world.log())
        assert found.kind == "log_behind" and found.version == 2 and found.adapter == "cand"

        watch(world.idle(max_iterations=0))  # the restart finishes it before the first cycle

        (line,) = world.promote_versions()
        assert (line["to_version"], line["served_as"]) == (2, "cand")
        assert line["reason"].startswith("recovered: loop.yaml held this promotion")
        closing = [x for x in world.lines("iteration") if x["outcome"] == "promoted"]
        assert any(x["to_version"] == 2 for x in closing)
        assert reconcile.detect(world.state(), world.log()) is None
        assert any(
            "recovered the log line of version 2" in (x.get("message") or "")
            for x in world.lines("progress")
        )

    def test_finishing_twice_adds_nothing(self, world, monkeypatch):
        world.start_canary()
        world.make_healthy()
        with monkeypatch.context() as patched:
            patched.setattr(
                loop_daemon, "_log_promotion",
                lambda *args, **kwargs: (_ for _ in ()).throw(_Killed("x")),
            )
            with pytest.raises(_Killed):
                watch(world.idle())

        first = reconcile.finish(world.path, world.log())
        second = reconcile.finish(world.path, world.log())

        assert first and first[0].startswith("finished an unfinished promotion")
        assert second == []
        assert len(world.promote_versions()) == 1

    def test_the_log_first_and_loop_yaml_second_the_order_of_an_older_daemon(self, world):
        """The log has version 2, loop.yaml still runs the canary (the state review 5b found)."""
        world.start_canary()
        before = world.state()
        iteration_id = before.canary_iteration["iteration_id"]
        world.log().version({
            "schema_version": 1, "id": "v-2", "ts": "2026-11-16T23:41:57+00:00",
            "from_version": 1, "to_version": 2, "cause": "promote", "actor": "loop",
            "iteration_id": iteration_id, "served_as": "cand", "alias": None,
            "reason": "canary cand promoted to production", "source": "soup-loop",
        })

        found = reconcile.detect(before, world.log())
        assert found.kind == "yaml_behind" and found.version == 2 and found.adapter == "cand"
        sentences = reconcile.finish(world.path, world.log())

        state = world.state()
        assert sentences == [
            "finished an unfinished promotion: loop.yaml now names version 2 (cand) as "
            "production, as versions.jsonl already did"
        ]
        assert (state.production_name, state.production_version, state.latest_version) == (
            "cand", 2, 2
        )
        assert state.production_adapter == "out/a1" and state.canary_active is None
        assert state.canary_iteration is None and state.canary_rollout_id is None
        assert state.previous_production_version == 1
        assert reconcile.detect(state, world.log()) is None

    def test_a_mismatch_it_cannot_explain_is_reported_and_not_touched(self, world):
        world.start_canary()
        world.log().version({
            "schema_version": 1, "id": "v-9", "ts": "2026-11-16T23:41:57+00:00",
            "from_version": 1, "to_version": 2, "cause": "promote", "actor": "loop",
            "iteration_id": None, "served_as": "someone-else", "alias": None,
            "reason": "x", "source": "soup-loop",
        })
        before = open(world.path, "rb").read()

        found = reconcile.detect(world.state(), world.log())
        sentences = reconcile.finish(world.path, world.log())

        assert found.kind == "unexplained"
        assert sentences == [found.text]
        assert "not a half-done promotion this tool can finish" in found.text
        assert open(world.path, "rb").read() == before

    def test_loop_yaml_that_moved_meanwhile_is_not_overwritten(self, world, monkeypatch):
        world.start_canary()
        found = reconcile.Mismatch("yaml_behind", 2, "text", adapter="cand")
        update_state(lambda fresh: replace(fresh, canary_active="other"), world.path)
        monkeypatch.setattr(reconcile, "detect", lambda state, log=None: found)

        sentences = reconcile.finish(world.path, world.log())

        assert sentences == ["text (loop.yaml changed meanwhile: not touched)"]
        assert world.state().canary_active == "other" and world.state().production_version == 1


class TestStatusNamesTheMismatch:
    def _status(self, world):
        result = CliRunner().invoke(loop_app, ["status"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        return _plain(result.output)

    def test_the_log_ahead_of_loop_yaml(self, world):
        world.start_canary()
        world.log().version({
            "schema_version": 1, "id": "v-2", "ts": "2026-11-16T23:41:57+00:00",
            "from_version": 1, "to_version": 2, "cause": "promote", "actor": "loop",
            "iteration_id": None, "served_as": "cand", "alias": None, "reason": "x",
            "source": "soup-loop",
        })

        text = self._status(world)

        assert (
            "promotion versions.jsonl records version 2 (cand) as production, but loop.yaml "
            "still names version 1 and cand as the running canary: an unfinished promotion; "
            "the next soup loop watch start finishes it"
        ) in text

    def test_loop_yaml_ahead_of_the_log(self, world, monkeypatch):
        world.start_canary()
        world.make_healthy()
        with monkeypatch.context() as patched:
            patched.setattr(
                loop_daemon, "_log_promotion",
                lambda *args, **kwargs: (_ for _ in ()).throw(_Killed("x")),
            )
            with pytest.raises(_Killed):
                watch(world.idle())

        text = self._status(world)

        assert (
            "promotion loop.yaml names version 2 (cand) as production, but versions.jsonl ends "
            "at version 1: the next soup loop watch start writes the missing line"
        ) in text

    def test_status_writes_nothing(self, world):
        world.start_canary()
        world.log().version({
            "schema_version": 1, "id": "v-2", "ts": "2026-11-16T23:41:57+00:00",
            "from_version": 1, "to_version": 2, "cause": "promote", "actor": "loop",
            "iteration_id": None, "served_as": "cand", "alias": None, "reason": "x",
            "source": "soup-loop",
        })
        yaml_before = open(world.path, "rb").read()
        log_before = open(world.log().path("version"), "rb").read()

        self._status(world)

        assert open(world.path, "rb").read() == yaml_before
        assert open(world.log().path("version"), "rb").read() == log_before

    def test_a_consistent_loop_has_no_promotion_row(self, world):
        world.start_canary()
        world.make_healthy()
        watch(world.idle())

        assert " promotion " not in self._status(world)

    def test_a_loop_with_no_version_log_has_none(self, world):
        assert " promotion " not in self._status(world)
        assert reconcile.detect(world.state(), world.log()) is None


class TestOtherEdges:
    def test_a_torn_and_a_foreign_line_in_the_log_do_not_break_the_comparison(self, world):
        world.start_canary()
        with open(world.log().path("version"), "a", encoding="utf-8") as handle:
            handle.write('{"half"\n[1]\nnot json\n')

        assert reconcile.detect(world.state(), world.log()) is None

    def test_promoted_state_is_what_the_daemon_applies(self, world):
        world.start_canary()
        state = world.state()

        promoted = reconcile.promoted_state(state)

        assert promoted.production_name == "cand" and promoted.canary_active is None
        assert promoted.production_version == 2 and promoted.latest_version == 2
        assert promoted.previous_production_version == 1


class TestTheServerServesAVersionLoopYamlDoesNotName:
    """The server switches first, so a kill before loop.yaml leaves the server ahead of it."""

    def _note(self, monkeypatch, health, **fields):
        monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", "http://127.0.0.1:8000")
        monkeypatch.setattr(ls, "_HEALTH_GETTER", lambda endpoint: health)
        state = LoopState(
            served_model="base", eval_suite="e", baseline="b", status="running", **fields
        )
        return ls.server_adapter_note(state)

    def test_an_unpromoted_canary_that_the_server_serves_is_named(self, cwd, monkeypatch):
        note = self._note(
            monkeypatch, {"active_adapter": "cand"}, canary_active="cand", canary_traffic_pct=5.0
        )

        assert note == (
            "the server serves the canary cand as production, but loop.yaml has not promoted it "
            "(an unfinished promotion): the next cycle with a healthy canary finishes it, or "
            "run soup loop rollback"
        )

    def test_an_unpromoted_canary_the_server_does_not_serve_says_nothing(self, cwd, monkeypatch):
        assert self._note(
            monkeypatch, {"active_adapter": None}, canary_active="cand", canary_traffic_pct=5.0
        ) is None

    def test_an_unreachable_server_with_only_a_canary_says_nothing(self, cwd, monkeypatch):
        assert self._note(
            monkeypatch, None, canary_active="cand", canary_traffic_pct=5.0
        ) is None

    def test_some_other_adapter_active_with_no_production_is_named(self, cwd, monkeypatch):
        note = self._note(
            monkeypatch, {"active_adapter": "x"}, canary_active="cand", canary_traffic_pct=5.0
        )

        assert note == (
            "the server has the adapter x active, but loop.yaml names no production adapter "
            "(the served model)"
        )

    def test_with_a_production_the_canary_served_as_production_is_named(self, cwd, monkeypatch):
        note = self._note(
            monkeypatch, {"active_adapter": "cand"}, production_name="old",
            canary_active="cand", canary_traffic_pct=5.0,
        )

        assert note == (
            "serving the canary cand as production, not production old (an unfinished "
            "promotion): the next cycle with a healthy canary finishes it"
        )

    def test_status_shows_it(self, world, monkeypatch):
        world.start_canary()
        monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", "http://127.0.0.1:8000")
        monkeypatch.setattr(ls, "_HEALTH_GETTER", lambda endpoint: {"active_adapter": "cand"})

        result = CliRunner().invoke(loop_app, ["status"])

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "unfinished promotion" in _plain(result.output)
