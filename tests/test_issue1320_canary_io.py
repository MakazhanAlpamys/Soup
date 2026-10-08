"""Canary hot-path IO frequency, durability, and rollout isolation."""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from soup_cli.utils import canary_router, loop_state
from soup_cli.utils.canary_router import BufferedCanaryOutcomes, CanaryPolicy, CanaryStateCache
from soup_cli.utils.loop_state import LoopState, write_state
from tests.test_issue815_canary_runtime import (
    _serve_with_canary,
    _watch_once,
    _write_canary_state,
)


@pytest.fixture
def buffers(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    instances = []

    def create(**kwargs):
        instance = BufferedCanaryOutcomes(str(tmp_path / "stats.json"), **kwargs)
        instances.append(instance)
        return instance

    yield create
    for instance in instances:
        instance.close()


def test_unchanged_policy_is_parsed_once_and_missing_policy_is_not_read(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "loop.yaml"
    calls = []
    original = loop_state.read_state

    def read(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(loop_state, "read_state", read)
    cache = CanaryStateCache(str(path))
    for _ in range(100):
        assert cache.get() is None
    assert calls == []
    state = LoopState(served_model="base", eval_suite="e", baseline="b")
    write_state(state, str(path))
    for _ in range(100):
        assert cache.get().served_model == "base"
        assert cache.get().canary_active is None
    assert len(calls) == 1


def test_atomic_replacement_with_same_mtime_and_size_invalidates_policy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "loop.yaml"
    state = LoopState(served_model="base", eval_suite="e", baseline="b",
                      canary_active="candidate", canary_traffic_pct=50.0)
    write_state(state, str(path))
    cache = CanaryStateCache(str(path))
    assert cache.get().canary_traffic_pct == 50.0
    before = path.stat()
    write_state(replace(state, canary_traffic_pct=10.0), str(path))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size
    assert path.stat().st_ino != before.st_ino
    assert cache.get().canary_traffic_pct == 10.0
    path.unlink()
    assert cache.get() is None


def test_concurrent_outcomes_preserve_counts_with_batched_writes(buffers, monkeypatch):
    writes = []
    original = canary_router.atomic_write_text

    def write(*args, **kwargs):
        writes.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(canary_router, "atomic_write_text", write)
    buffer = buffers(interval=60.0)
    policy = CanaryPolicy("base", "candidate", 50.0)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda index: buffer.record(policy, "canary" if index % 2 else "stable",
                                                 index % 3 != 0, rollout_id="r"), range(257)))
    assert len(writes) == 4  # Four complete 64-sample batches; one sample pending.
    buffer.close()
    assert len(writes) == 5
    stats = canary_router.read_bucket_stats(stable="base", canary="candidate", rollout_id="r",
                                           path=buffer.path).snapshot()
    expected = {name: 0 for name in stats}
    for index in range(257):
        bucket = "canary" if index % 2 else "stable"
        expected[f"{bucket}_{'ok' if index % 3 else 'major'}"] += 1
    assert dict(stats) == expected


def test_idle_partial_batch_flushes_without_another_request(buffers, monkeypatch):
    written = threading.Event()
    original = canary_router.atomic_write_text

    def write(*args, **kwargs):
        result = original(*args, **kwargs)
        written.set()
        return result

    monkeypatch.setattr(canary_router, "atomic_write_text", write)
    buffer = buffers(interval=0.02)
    buffer.record(CanaryPolicy("base", "candidate", 50.0), "stable", True, rollout_id="r")
    assert written.wait(2.0)
    stats = canary_router.read_bucket_stats(stable="base", canary="candidate", rollout_id="r",
                                           path=buffer.path)
    assert stats.stable_ok == 1


def test_promotion_change_keeps_rollout_counters_separate(buffers):
    buffer = buffers(interval=60.0)
    policy = CanaryPolicy("base", "candidate", 50.0)
    for _ in range(10):
        buffer.record(policy, "canary", False, rollout_id="old")
    buffer.record(policy, "stable", True, rollout_id="new")
    buffer.close()
    stats = canary_router.read_bucket_stats(stable="base", canary="candidate", rollout_id="new",
                                           path=buffer.path)
    assert dict(stats.snapshot()) == {"stable_ok": 1, "stable_major": 0,
                                      "canary_ok": 0, "canary_major": 0}


def test_failed_flush_retains_the_batch_for_retry(buffers, monkeypatch):
    buffer = buffers(batch_size=2, interval=60.0)
    policy = CanaryPolicy("base", "candidate", 50.0)
    original = canary_router.atomic_write_text
    buffer.record(policy, "canary", True, rollout_id="r")
    with monkeypatch.context() as patch:
        def fail(*args, **kwargs):
            raise OSError("disk unavailable")
        patch.setattr(canary_router, "atomic_write_text", fail)
        with pytest.raises(OSError):
            buffer.record(policy, "stable", False, rollout_id="r")
    buffer.flush()
    assert canary_router.atomic_write_text is original
    stats = canary_router.read_bucket_stats(stable="base", canary="candidate", rollout_id="r",
                                           path=buffer.path)
    assert stats.canary_ok == 1 and stats.stable_major == 1


def test_watch_rollback_is_visible_to_the_next_keyed_request(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state_path, stats_path = _write_canary_state(tmp_path, traffic_pct=100.0, rollout_id="r")
    client, observed = _serve_with_canary(monkeypatch, state_path, stats_path,
                                         adapters=("candidate",), fail_on="candidate")
    with client:
        for index in range(30):
            response = client.post("/v1/chat/completions", json={"messages": [{"role": "user",
                                   "content": "hi"}], "conversation_id": f"k{index}"})
            assert response.status_code == 500
        client.app.state.canary_outcomes.flush()
        state, _ = _watch_once(tmp_path, state_path, stats_path)
        assert state.canary_active is None
        assert client.post("/v1/chat/completions", json={"messages": [{"role": "user",
                           "content": "hi"}], "conversation_id": "k0"}).status_code == 200
        assert observed[-1] is None


def test_orderly_app_shutdown_flushes_the_tail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state_path, stats_path = _write_canary_state(tmp_path, traffic_pct=100.0, rollout_id="r")
    client, _ = _serve_with_canary(monkeypatch, state_path, stats_path, adapters=("candidate",))
    with client:
        response = client.post("/v1/chat/completions", json={"messages": [{"role": "user",
                               "content": "hi"}], "conversation_id": "k"})
        assert response.status_code == 200
        assert not stats_path.exists()
    assert canary_router.read_bucket_stats(stable="base", canary="candidate", rollout_id="r",
                                          path=str(stats_path)).canary_ok == 1


def test_invalid_state_warning_is_emitted_once(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    state_path, stats_path = _write_canary_state(tmp_path, traffic_pct=100.0, rollout_id="r")
    state_path.write_text("invalid json", encoding="utf-8")
    client, _ = _serve_with_canary(monkeypatch, state_path, stats_path, adapters=("candidate",))
    with client:
        for _ in range(5):
            assert client.post("/v1/chat/completions", json={"messages": [{"role": "user",
                               "content": "hi"}], "conversation_id": "k"}).status_code == 200
    assert sum(record.message == "canary policy unavailable" for record in caplog.records) == 1


def test_late_old_rollout_completion_cannot_replace_new_statistics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state_path, stats_path = _write_canary_state(tmp_path, traffic_pct=100.0, rollout_id="old")
    client, _ = _serve_with_canary(monkeypatch, state_path, stats_path, adapters=("candidate",))
    import soup_cli.commands.serve as serve

    def generate(*args, **kwargs):
        state = loop_state.read_state(str(state_path))
        if state.canary_rollout_id == "old":
            write_state(replace(state, canary_rollout_id="new"), str(state_path))
        return "ok", 1, 1

    monkeypatch.setattr(serve, "_generate_response", generate)
    request = {"messages": [{"role": "user", "content": "hi"}], "conversation_id": "k"}
    with client:
        assert client.post("/v1/chat/completions", json=request).status_code == 200
        client.app.state.canary_outcomes.flush()
        assert not stats_path.exists()
        assert client.post("/v1/chat/completions", json=request).status_code == 200
    stats = canary_router.read_bucket_stats(stable="base", canary="candidate", rollout_id="new",
                                           path=str(stats_path))
    assert stats.canary_ok == 1


@pytest.mark.parametrize("active", [True, False])
def test_live_keyed_requests_parse_an_unchanged_policy_only_once(tmp_path, monkeypatch, active):
    monkeypatch.chdir(tmp_path)
    state_path, stats_path = _write_canary_state(tmp_path, traffic_pct=100.0, rollout_id="r")
    if not active:
        state = loop_state.read_state(str(state_path))
        write_state(replace(state, canary_active=None, canary_traffic_pct=None,
                            canary_rollout_id=None), str(state_path))
    calls = []
    original = loop_state.read_state

    def read(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(loop_state, "read_state", read)
    client, _ = _serve_with_canary(monkeypatch, state_path, stats_path, adapters=("candidate",))
    with client:
        for _ in range(10):
            response = client.post("/v1/chat/completions", json={"messages": [{"role": "user",
                                   "content": "hi"}], "conversation_id": "k"})
            assert response.status_code == 200
        assert len(calls) == 1


def test_pending_superseded_batch_does_not_replace_current_stats(buffers, tmp_path, monkeypatch):
    state_path = tmp_path / "loop.yaml"
    state = LoopState(served_model="base", eval_suite="e", baseline="b",
                      canary_active="candidate", canary_traffic_pct=50.0, canary_rollout_id="old")
    write_state(state, str(state_path))
    buffer = buffers(interval=60.0, state_cache=CanaryStateCache(str(state_path)))
    policy = CanaryPolicy("base", "candidate", 50.0)
    buffer.record(policy, "canary", False, rollout_id="old")
    write_state(replace(state, canary_rollout_id="new"), str(state_path))
    canary_router.record_bucket_outcome(policy, "canary", True, rollout_id="new", path=buffer.path)
    buffer.close()
    stats = canary_router.read_bucket_stats(stable="base", canary="candidate", rollout_id="new",
                                           path=buffer.path)
    assert stats.canary_ok == 1 and stats.canary_major == 0


def test_in_place_edit_with_same_size_and_inode_invalidates_policy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "loop.yaml"
    state = LoopState(served_model="base", eval_suite="e", baseline="b",
                      canary_active="candidate", canary_traffic_pct=50.0)
    write_state(state, str(path))
    cache = CanaryStateCache(str(path))
    assert cache.get().canary_traffic_pct == 50.0
    before = path.stat()
    text = path.read_text(encoding="utf-8")
    with open(path, "r+", encoding="utf-8", newline="") as handle:
        handle.write(text.replace("50.0", "10.0"))
    after = path.stat()
    assert (after.st_ino, after.st_size) == (before.st_ino, before.st_size)
    if after.st_mtime_ns == before.st_mtime_ns:
        pytest.skip("filesystem timestamp too coarse for this check")
    assert cache.get().canary_traffic_pct == 10.0


def test_exactly_one_batch_is_persisted_without_waiting_for_the_next_outcome(buffers):
    buffer = buffers(batch_size=64, interval=60.0)
    policy = CanaryPolicy("base", "candidate", 50.0)
    for _ in range(63):
        buffer.record(policy, "stable", True, rollout_id="r")
    assert not os.path.exists(buffer.path)
    buffer.record(policy, "stable", True, rollout_id="r")
    stats = canary_router.read_bucket_stats(stable="base", canary="candidate", rollout_id="r",
                                           path=buffer.path)
    assert stats.stable_ok == 64


def test_the_interval_runs_from_the_first_pending_outcome(buffers, monkeypatch):
    created = []

    class FakeTimer:
        def __init__(self, interval, function):
            created.append(interval)
            self.daemon = False

        def start(self):
            pass

        def cancel(self):
            pass

    monkeypatch.setattr(canary_router.threading, "Timer", FakeTimer)
    buffer = buffers(interval=2.0)
    policy = CanaryPolicy("base", "candidate", 50.0)
    for _ in range(3):
        buffer.record(policy, "stable", True, rollout_id="r")
    assert created == [2.0]


def test_timed_flush_retries_after_a_transient_write_error(buffers, monkeypatch):
    original = canary_router.atomic_write_text
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise PermissionError("target locked")
        return original(*args, **kwargs)

    monkeypatch.setattr(canary_router, "atomic_write_text", flaky)
    buffer = buffers(interval=0.05)
    buffer.record(CanaryPolicy("base", "candidate", 50.0), "canary", False, rollout_id="r")
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and not os.path.exists(buffer.path):
        time.sleep(0.02)
    stats = canary_router.read_bucket_stats(stable="base", canary="candidate", rollout_id="r",
                                           path=buffer.path)
    assert len(calls) >= 2 and stats.canary_major == 1


def test_policy_warning_returns_after_recovery(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    state_path, stats_path = _write_canary_state(tmp_path, traffic_pct=100.0, rollout_id="r")
    valid = state_path.read_text(encoding="utf-8")
    client, _ = _serve_with_canary(monkeypatch, state_path, stats_path, adapters=("candidate",))
    request = {"messages": [{"role": "user", "content": "hi"}], "conversation_id": "k"}
    with client:
        for text in ("invalid json", valid, "invalid again"):
            state_path.write_text(text, encoding="utf-8")
            assert client.post("/v1/chat/completions", json=request).status_code == 200
    assert sum(record.message == "canary policy unavailable" for record in caplog.records) == 2


def test_unloaded_adapter_warning_has_no_exception_traceback(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    state_path, stats_path = _write_canary_state(tmp_path, traffic_pct=100.0, rollout_id="r")
    client, _ = _serve_with_canary(monkeypatch, state_path, stats_path, adapters=("other",))
    with client:
        assert client.post("/v1/chat/completions", json={"messages": [{"role": "user",
                           "content": "hi"}], "conversation_id": "k"}).status_code == 200
    warning = next(record for record in caplog.records if "is not loaded" in record.message)
    assert not warning.exc_info


def test_outage_refuses_outcomes_past_the_pending_batch_until_recovery(buffers, monkeypatch):
    buffer = buffers(batch_size=2, interval=60.0)
    policy = CanaryPolicy("base", "candidate", 50.0)

    def fail(*args, **kwargs):
        raise OSError("disk unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(canary_router, "atomic_write_text", fail)
        buffer.record(policy, "stable", True, rollout_id="r")
        with pytest.raises(OSError):
            buffer.record(policy, "stable", True, rollout_id="r")  # batch full, write fails
        for _ in range(5):
            with pytest.raises(OSError):
                buffer.record(policy, "stable", True, rollout_id="r")  # refused, not counted
    buffer.flush()
    stats = canary_router.read_bucket_stats(stable="base", canary="candidate", rollout_id="r",
                                           path=buffer.path)
    assert stats.stable_ok == 2  # exactly the one pending batch survives the outage


def test_late_old_completion_does_not_force_a_flush_of_the_current_batch(buffers, tmp_path):
    state_path = tmp_path / "loop.yaml"
    state = LoopState(served_model="base", eval_suite="e", baseline="b",
                      canary_active="candidate", canary_traffic_pct=50.0,
                      canary_rollout_id="new")
    write_state(state, str(state_path))
    buffer = buffers(interval=60.0, state_cache=CanaryStateCache(str(state_path)))
    policy = CanaryPolicy("base", "candidate", 50.0)
    buffer.record(policy, "canary", True, rollout_id="new")
    buffer.record(policy, "canary", False, rollout_id="old")  # late completion, superseded
    assert not os.path.exists(buffer.path)  # dropped on arrival: nothing flushed early
    buffer.close()
    stats = canary_router.read_bucket_stats(stable="base", canary="candidate", rollout_id="new",
                                           path=buffer.path)
    assert stats.canary_ok == 1 and stats.canary_major == 0
