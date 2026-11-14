"""``loop.yaml`` is never overwritten with a stale copy.

``soup loop watch`` reads the state, runs an iteration (training can take hours) and used to
write its in-memory copy back: a ``soup loop rollback`` / ``pause`` / ``canary`` made meanwhile
was undone. Now every writer goes through one lock (``loop_state.state_lock``), the daemon
re-reads the file under it and applies ONLY the fields its iteration changed
(``loop_state.merge_state``), and the write is atomic.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import fields, replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

import soup_cli.utils.loop_state as loop_state
from soup_cli.cli import app
from soup_cli.utils.loop_daemon import WatchConfig, watch
from soup_cli.utils.loop_state import (
    LoopState,
    StateMergeError,
    merge_state,
    read_state,
    state_lock,
    update_state,
    write_state,
)

runner = CliRunner()


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _path(root: Path) -> str:
    return str(root / ".soup" / "loop.yaml")


def _promoted_state() -> LoopState:
    return LoopState(
        served_model="base", eval_suite="evals.yaml", baseline="prod", status="running",
        production_name="v2", production_adapter="v2", production_version=2, latest_version=2,
        previous_production_name="v1", previous_production_adapter="v1",
        previous_production_version=1,
    )


def _config(root: Path, *, during_train) -> WatchConfig:
    """A watch config whose training stage runs ``during_train`` (the second writer)."""

    def train(state, harvest):
        during_train()
        return {"run_id": "run-1", "adapter_path": "out", "notes": "trained"}

    return WatchConfig(
        poll_interval_sec=1.0, max_iterations=1, state_path=_path(root),
        iteration_dir=str(root / ".soup-loops"),
        harvest_fn=lambda state: {"pairs_harvested": 3, "traces_collected": 3},
        train_fn=train,
        gate_fn=lambda state, train_out: {"gate_verdict": "OK"},
        deploy_fn=lambda state, outs: {"deployed": False},
        cost_fn=lambda state: 0.0,
    )


# ---------------------------------------------------------------------------
# The blocker: a second writer during an iteration survives the daemon's write
# ---------------------------------------------------------------------------


class TestSecondWriterDuringAnIteration:
    def test_a_rollback_made_during_the_iteration_survives(self, cwd):
        write_state(_promoted_state(), _path(cwd))

        def rollback():
            update_state(
                lambda fresh: replace(
                    fresh, production_name="v1", production_adapter="v1",
                    previous_production_name=None, previous_production_adapter=None,
                    production_version=1,
                ),
                _path(cwd),
            )

        watch(_config(cwd, during_train=rollback))

        final = read_state(_path(cwd))
        assert (final.production_name, final.production_adapter) == ("v1", "v1")
        assert final.production_version == 1
        assert final.previous_production_name is None

    def test_the_iterations_own_fields_are_stored(self, cwd):
        write_state(_promoted_state(), _path(cwd))

        watch(_config(cwd, during_train=lambda: update_state(
            lambda fresh: replace(fresh, production_name="v1"), _path(cwd)
        )))

        final = read_state(_path(cwd))
        assert final.iteration_count == 1
        assert final.last_iteration_id is not None
        assert final.traces_collected == 3 and final.pairs_distilled == 3
        assert final.runs_today == 1

    def test_a_pause_made_during_the_iteration_is_not_undone(self, cwd):
        write_state(_promoted_state(), _path(cwd))

        watch(_config(cwd, during_train=lambda: update_state(
            lambda fresh: fresh.with_status("paused"), _path(cwd)
        )))

        assert read_state(_path(cwd)).status == "paused"

    def test_a_canary_started_during_the_iteration_survives(self, cwd):
        write_state(_promoted_state(), _path(cwd))

        def canary():
            update_state(
                lambda fresh: replace(
                    fresh, canary_active="cand", canary_traffic_pct=5.0,
                    canary_rollout_id="rollout-9", canary_started_ts=1_800_000_000.0,
                ),
                _path(cwd),
            )

        watch(_config(cwd, during_train=canary))

        final = read_state(_path(cwd))
        assert final.canary_active == "cand" and final.canary_rollout_id == "rollout-9"

    def test_without_a_second_writer_the_iteration_writes_exactly_as_before(self, cwd):
        write_state(_promoted_state(), _path(cwd))

        watch(_config(cwd, during_train=lambda: None))

        final = read_state(_path(cwd))
        assert final.iteration_count == 1 and final.production_name == "v2"


# ---------------------------------------------------------------------------
# merge_state
# ---------------------------------------------------------------------------


class TestMergeState:
    def _states(self):
        base = _promoted_state()
        return base, replace(base)

    def test_only_the_fields_the_iteration_changed_are_applied(self, cwd):
        base, new = self._states()
        write_state(replace(base, production_name="v1", baseline="by-hand"), _path(cwd))
        new = replace(new, iteration_count=4, last_iteration_id="iter-x")

        merged = merge_state(base, new, _path(cwd))

        assert merged.iteration_count == 4 and merged.last_iteration_id == "iter-x"
        assert (merged.production_name, merged.baseline) == ("v1", "by-hand")
        assert read_state(_path(cwd)).production_name == "v1"

    def test_a_field_changed_by_both_takes_the_iterations_value(self, cwd):
        base, new = self._states()
        write_state(replace(base, canary_active="by-hand"), _path(cwd))
        new = replace(new, canary_active="by-iteration")

        assert merge_state(base, new, _path(cwd)).canary_active == "by-iteration"

    def test_a_status_changed_by_someone_else_is_kept(self, cwd):
        base, new = self._states()
        write_state(base.with_status("paused"), _path(cwd))
        new = replace(new, status="running", iteration_count=1)

        merged = merge_state(base, new, _path(cwd))

        assert merged.status == "paused" and merged.iteration_count == 1

    def test_a_status_only_the_iteration_changed_is_applied(self, cwd):
        base, new = self._states()
        write_state(base, _path(cwd))

        assert merge_state(base, new.with_status("stopped"), _path(cwd)).status == "stopped"

    def test_every_loop_state_field_is_either_merged_or_kept(self, cwd):
        base, new = self._states()
        write_state(base, _path(cwd))
        changed = {}
        for item in fields(LoopState):
            if item.name in ("created_at", "updated_at"):
                continue
            value = getattr(base, item.name)
            if isinstance(value, bool) or value is None or isinstance(value, (dict, list)):
                continue
            if isinstance(value, (int, float)):
                changed[item.name] = value + 1
        changed.pop("production_version", None)
        new = replace(new, **changed)

        merged = merge_state(base, new, _path(cwd))

        for key, val in changed.items():
            assert getattr(merged, key) == val, key


# ---------------------------------------------------------------------------
# The lock
# ---------------------------------------------------------------------------


class TestStateLock:
    def test_the_lock_file_is_gone_after_the_block(self, cwd):
        write_state(_promoted_state(), _path(cwd))

        with state_lock(_path(cwd)):
            assert Path(_path(cwd) + ".lock").exists()

        assert not Path(_path(cwd) + ".lock").exists()

    def test_the_lock_is_released_when_the_block_raises(self, cwd):
        write_state(_promoted_state(), _path(cwd))

        with pytest.raises(RuntimeError):
            with state_lock(_path(cwd)):
                raise RuntimeError("boom")

        assert not Path(_path(cwd) + ".lock").exists()

    def test_a_lock_held_by_a_live_process_times_out_with_a_named_error(self, cwd, monkeypatch):
        write_state(_promoted_state(), _path(cwd))
        lock = Path(_path(cwd) + ".lock")
        lock.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
        monkeypatch.setattr(loop_state, "process_is_alive", lambda pid: True)

        with pytest.raises(TimeoutError, match="loop.yaml is locked"):
            with state_lock(_path(cwd), timeout=0.2):
                pass

        assert lock.exists()  # a live holder's lock is never removed

    def test_a_lock_of_a_dead_process_is_recovered(self, cwd, monkeypatch):
        write_state(_promoted_state(), _path(cwd))
        lock = Path(_path(cwd) + ".lock")
        lock.write_text(json.dumps({"pid": 424242}), encoding="utf-8")
        monkeypatch.setattr(loop_state, "process_is_alive", lambda pid: False)

        with state_lock(_path(cwd), timeout=2.0):
            assert json.loads(lock.read_text(encoding="utf-8"))["pid"] == os.getpid()

        assert not lock.exists()

    def test_an_unparsable_lock_is_recovered_once_it_is_old(self, cwd, monkeypatch):
        write_state(_promoted_state(), _path(cwd))
        lock = Path(_path(cwd) + ".lock")
        lock.write_text("garbage", encoding="utf-8")
        old = lock.stat().st_mtime - 3600
        os.utime(lock, (old, old))

        with state_lock(_path(cwd), timeout=2.0):
            pass

        assert not lock.exists()

    def test_two_threads_never_hold_the_lock_together(self, cwd):
        write_state(_promoted_state(), _path(cwd))
        inside = []
        overlap = []

        def worker():
            for _ in range(5):
                with state_lock(_path(cwd), timeout=10.0):
                    inside.append(1)
                    if len(inside) > 1:
                        overlap.append(len(inside))
                    inside.pop()

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert overlap == []

    def test_concurrent_updates_all_land(self, cwd):
        write_state(_promoted_state(), _path(cwd))

        def bump():
            for _ in range(5):
                update_state(
                    lambda fresh: replace(fresh, iteration_count=fresh.iteration_count + 1),
                    _path(cwd),
                )

        threads = [threading.Thread(target=bump) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert read_state(_path(cwd)).iteration_count == 20


# ---------------------------------------------------------------------------
# Every writer goes through the lock
# ---------------------------------------------------------------------------


class TestEveryWriterTakesTheLock:
    @pytest.fixture
    def locked(self, cwd, monkeypatch):
        entered = []
        real = loop_state.state_lock

        def spy(path=None, **kwargs):
            entered.append(path)
            return real(path, **kwargs)

        monkeypatch.setattr(loop_state, "state_lock", spy)
        result = runner.invoke(
            app, ["loop", "init", "m", "--eval", "e.json", "--baseline", "b"]
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert entered, "init wrote loop.yaml without the lock"
        loop_state.update_state(lambda state: state.with_status("running"))
        entered.clear()
        return entered

    def test_pause_and_resume(self, locked):
        assert runner.invoke(app, ["loop", "pause"]).exit_code == 0
        assert len(locked) >= 1
        locked.clear()
        assert runner.invoke(app, ["loop", "resume"]).exit_code == 0
        assert len(locked) >= 1

    def test_canary(self, locked):
        result = runner.invoke(app, ["loop", "canary", "cand-adapter"])

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert len(locked) >= 1

    def test_rollback_that_only_withdraws_a_canary(self, locked):
        assert runner.invoke(app, ["loop", "canary", "cand-adapter"]).exit_code == 0
        locked.clear()

        result = runner.invoke(app, ["loop", "rollback"])

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert len(locked) >= 1

    def test_pause_keeps_a_field_another_writer_changed_since_it_read_the_state(self, cwd):
        runner.invoke(app, ["loop", "init", "m", "--eval", "e.json", "--baseline", "b"])
        update_state(lambda state: state.with_status("running"), _path(cwd))
        real_read = loop_state.read_state

        injections = []

        def read_then_change(path=None):
            state = real_read(path)
            # another writer lands right after each read this command makes through its own
            # module (a stale copy of the file is what a careless writer would save)
            injections.append(1)
            real_write = loop_state._write_state_unlocked
            real_write(replace(state, production_name=f"by-another-writer-{len(injections)}"), path)
            return state

        import soup_cli.commands.loop as loop_cmd

        original = loop_cmd.read_state
        loop_cmd.read_state = read_then_change
        try:
            result = runner.invoke(app, ["loop", "pause"])
        finally:
            loop_cmd.read_state = original

        assert result.exit_code == 0, (result.output, repr(result.exception))
        final = real_read(_path(cwd))
        assert final.status == "paused"
        assert final.production_name == f"by-another-writer-{len(injections)}"


# ---------------------------------------------------------------------------
# merge_state never writes a stale copy (python review P4)
# ---------------------------------------------------------------------------


class TestMergeStateNeverWritesAStaleCopy:
    def _conflict(self, cwd):
        """The file moved on (a manual promotion to version 5) while the iteration ran; the
        iteration's own ``latest_version=3`` cannot be applied next to it."""
        base = _promoted_state()
        write_state(
            replace(base, production_name="v5", production_version=5, latest_version=5),
            _path(cwd),
        )
        new = replace(base, latest_version=3, iteration_count=1, last_iteration_id="iter-x")
        return base, new

    def test_a_change_that_conflicts_does_not_bring_the_stale_copy_back(self, cwd):
        base, new = self._conflict(cwd)

        merged = merge_state(base, new, _path(cwd))

        on_disk = read_state(_path(cwd))
        assert on_disk == merged
        assert (on_disk.production_name, on_disk.production_version) == ("v5", 5)
        assert on_disk.latest_version == 5

    def test_the_fields_that_do_apply_are_still_written(self, cwd):
        base, new = self._conflict(cwd)

        merged = merge_state(base, new, _path(cwd))

        assert merged.iteration_count == 1 and merged.last_iteration_id == "iter-x"

    def test_the_field_that_could_not_be_applied_is_named(self, cwd):
        base, new = self._conflict(cwd)
        skipped: list = []

        merge_state(base, new, _path(cwd), skipped=skipped)

        assert skipped == ["latest_version"]

    def test_nothing_is_reported_when_every_change_applies(self, cwd):
        base = _promoted_state()
        write_state(base, _path(cwd))
        skipped: list = []

        merge_state(base, replace(base, iteration_count=2), _path(cwd), skipped=skipped)

        assert skipped == []

    def test_a_missing_file_fails_with_a_named_error_and_writes_nothing(self, cwd):
        base = _promoted_state()
        assert not Path(_path(cwd)).exists()

        with pytest.raises(StateMergeError, match="loop.yaml.*not written"):
            merge_state(base, replace(base, iteration_count=1), _path(cwd))

        assert not Path(_path(cwd)).exists()

    def test_an_unreadable_file_fails_with_a_named_error_and_is_left_alone(self, cwd):
        base = _promoted_state()
        write_state(base, _path(cwd))
        Path(_path(cwd)).write_text("{ not json", encoding="utf-8")

        with pytest.raises(StateMergeError, match="not valid JSON"):
            merge_state(base, replace(base, iteration_count=1), _path(cwd))

        assert Path(_path(cwd)).read_text(encoding="utf-8") == "{ not json"
        assert not Path(_path(cwd) + ".lock").exists()


# ---------------------------------------------------------------------------
# Taking over a dead holder's lock is atomic (python review P3)
# ---------------------------------------------------------------------------


class TestStaleLockTakeover:
    DEAD_PID = 424242

    def _stale_lock(self, cwd, monkeypatch) -> Path:
        write_state(_promoted_state(), _path(cwd))
        lock = Path(_path(cwd) + ".lock")
        lock.write_text(json.dumps({"pid": self.DEAD_PID}), encoding="utf-8")
        monkeypatch.setattr(loop_state, "process_is_alive", lambda pid: pid != self.DEAD_PID)
        return lock

    def test_two_waiters_that_both_saw_the_dead_holder_never_both_take_the_lock(
        self, cwd, monkeypatch
    ):
        self._stale_lock(cwd, monkeypatch)
        real_holder = loop_state._lock_holder
        rendezvous = threading.Barrier(2, timeout=10)
        seen = threading.local()

        def holder(path):
            result = real_holder(path)
            if not getattr(seen, "judged", False):
                seen.judged = True
                rendezvous.wait()  # both waiters have judged the lock abandoned by now
            return result

        monkeypatch.setattr(loop_state, "_lock_holder", holder)
        real_take_over = loop_state._take_over_stale
        take_over_calls, calls_guard = [], threading.Lock()
        first_inside = threading.Event()

        def take_over(lock_path):
            with calls_guard:
                order = len(take_over_calls)
                take_over_calls.append(1)
            if order == 1:
                # the slower waiter acts on its OLD judgement: only after the first one has
                # replaced the dead lock and is inside the critical section
                first_inside.wait(10)
            return real_take_over(lock_path)

        monkeypatch.setattr(loop_state, "_take_over_stale", take_over)
        inside, overlaps, entered = [], [], []
        second_in = threading.Event()

        def worker():
            with state_lock(_path(cwd), timeout=10.0):
                first_inside.set()
                inside.append(1)
                entered.append(1)
                if len(inside) > 1:
                    overlaps.append(len(inside))
                    second_in.set()
                second_in.wait(0.6)  # stay inside while the other waiter would come in
                inside.pop()

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)

        assert overlaps == []
        assert len(entered) == 2  # the second waiter did get the lock, after the first left

    def test_a_dead_holders_lock_is_still_taken_over(self, cwd, monkeypatch):
        lock = self._stale_lock(cwd, monkeypatch)

        with state_lock(_path(cwd), timeout=5.0):
            assert json.loads(lock.read_text(encoding="utf-8"))["pid"] == os.getpid()

        assert not lock.exists()

    def test_a_takeover_that_finds_the_lock_alive_again_leaves_it(self, cwd, monkeypatch):
        """The re-check under the takeover guard: the lock was dead when the waiter looked,
        a live process owns it by the time the waiter holds the guard."""
        lock = self._stale_lock(cwd, monkeypatch)
        lock.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")

        assert loop_state._take_over_stale(str(lock)) is True
        assert lock.exists()

    def test_a_takeover_does_not_wait_for_another_takeover_in_progress(self, cwd, monkeypatch):
        lock = self._stale_lock(cwd, monkeypatch)
        guard_fd = os.open(str(lock) + ".takeover", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            assert loop_state._try_kernel_lock(guard_fd) is True

            assert loop_state._take_over_stale(str(lock)) is False
            assert lock.exists()  # the loser leaves the lock to the winner
        finally:
            os.close(guard_fd)
