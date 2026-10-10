"""Regression tests for the live canary path from issue #815."""

from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from soup_cli.utils.canary_router import (
    CanaryPolicy,
    read_bucket_stats,
    record_bucket_outcome,
)
from soup_cli.utils.loop_daemon import WatchConfig, watch
from soup_cli.utils.loop_state import LoopState, read_state, write_state


class _AdapterModel:
    def __init__(self) -> None:
        self.current = None

    def set_adapter(self, name: str) -> None:
        self.current = name

    @contextmanager
    def disable_adapter(self):
        previous = self.current
        self.current = None
        try:
            yield
        finally:
            self.current = previous


def test_five_percent_policy_routes_live_serve_requests(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state_path = tmp_path / ".soup" / "loop.yaml"
    stats_path = tmp_path / ".soup" / "canary-stats.json"
    write_state(
        LoopState(
            served_model="base",
            eval_suite="evals.yaml",
            baseline="prod",
            canary_active="candidate",
            canary_traffic_pct=5.0,
            canary_rollout_id="rollout-1",
        ),
        str(state_path),
    )

    import soup_cli.commands.serve as serve

    model = _AdapterModel()
    observed = []

    def generate(current_model, *args, **kwargs):
        observed.append(current_model.current)
        return "ok", 1, 1

    monkeypatch.setattr(serve, "_generate_response", generate)
    app = serve._create_app(
        model_obj=model,
        tokenizer=object(),
        device="cpu",
        model_name="base",
        max_tokens_default=8,
        adapter_map={"candidate": "unused"},
        peft_adapter_names={"candidate"},
        canary_state_path=str(state_path),
        canary_stats_path=str(stats_path),
    )
    client = TestClient(app)

    request_count = 400
    for index in range(request_count):
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "conversation_id": f"conversation-{index}",
            },
        )
        assert response.status_code == 200

    canary_count = observed.count("candidate")
    assert 10 <= canary_count <= 30
    assert observed.count(None) == request_count - canary_count
    client.app.state.canary_outcomes.close()


def test_failing_live_canary_requests_reach_the_rollout_stats(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state_path = tmp_path / ".soup" / "loop.yaml"
    stats_path = tmp_path / ".soup" / "canary-stats.json"
    write_state(
        LoopState(
            served_model="base",
            eval_suite="evals.yaml",
            baseline="prod",
            canary_active="candidate",
            canary_traffic_pct=50.0,
            canary_rollout_id="rollout-bridge",
        ),
        str(state_path),
    )

    import soup_cli.commands.serve as serve

    model = _AdapterModel()
    observed = []

    def generate(current_model, *args, **kwargs):
        observed.append(current_model.current)
        if current_model.current == "candidate":
            raise RuntimeError("bad canary")
        return "ok", 1, 1

    monkeypatch.setattr(serve, "_generate_response", generate)
    app = serve._create_app(
        model_obj=model,
        tokenizer=object(),
        device="cpu",
        model_name="base",
        max_tokens_default=8,
        adapter_map={"candidate": "unused"},
        peft_adapter_names={"candidate"},
        canary_state_path=str(state_path),
        canary_stats_path=str(stats_path),
    )
    client = TestClient(app)

    statuses = []
    for index in range(200):
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "conversation_id": f"bridge-{index}",
            },
        )
        statuses.append(response.status_code)

    canary_count = observed.count("candidate")
    stable_count = observed.count(None)
    client.app.state.canary_outcomes.close()
    stats = read_bucket_stats(
        stable="base",
        canary="candidate",
        rollout_id="rollout-bridge",
        path=str(stats_path),
    )
    assert statuses.count(500) == canary_count
    assert statuses.count(200) == stable_count
    assert stats.canary_major == canary_count
    assert stats.canary_ok == 0
    assert stats.stable_ok == stable_count
    assert stats.stable_major == 0


def test_watch_persists_rollback_for_major_live_stats(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state_path = tmp_path / ".soup" / "loop.yaml"
    stats_path = tmp_path / ".soup" / "canary-stats.json"
    write_state(
        LoopState(
            served_model="base",
            eval_suite="evals.yaml",
            baseline="prod",
            status="running",
            canary_active="candidate",
            canary_traffic_pct=5.0,
            canary_autoroll_on_regress=True,
            canary_rollout_id="rollout-1",
        ),
        str(state_path),
    )
    persisted = read_state(str(state_path))
    policy = CanaryPolicy(stable="base", canary="candidate", traffic_pct=5.0)
    for _ in range(30):
        record_bucket_outcome(
            policy,
            "stable",
            True,
            rollout_id=persisted.canary_rollout_id,
            path=str(stats_path),
        )
        record_bucket_outcome(
            policy,
            "canary",
            False,
            rollout_id=persisted.canary_rollout_id,
            path=str(stats_path),
        )

    final_state, iterations = watch(
        WatchConfig(
            poll_interval_sec=1.0,
            max_iterations=1,
            state_path=str(state_path),
            canary_stats_path=str(stats_path),
            iteration_dir=str(tmp_path / ".soup-loops"),
        )
    )

    persisted = read_state(str(state_path))
    assert iterations == 1
    assert final_state.canary_active is None
    assert persisted.canary_active is None
    assert persisted.canary_traffic_pct == 0.0


@pytest.mark.parametrize("sample_count", [5, 30])
def test_watch_keeps_a_non_regressing_canary_active(tmp_path, monkeypatch, sample_count):
    monkeypatch.chdir(tmp_path)
    state_path = tmp_path / ".soup" / "loop.yaml"
    stats_path = tmp_path / ".soup" / "canary-stats.json"
    write_state(
        LoopState(
            served_model="base",
            eval_suite="evals.yaml",
            baseline="prod",
            status="running",
            canary_active="candidate",
            canary_traffic_pct=5.0,
            canary_autoroll_on_regress=True,
            canary_rollout_id="rollout-healthy",
        ),
        str(state_path),
    )
    policy = CanaryPolicy(stable="base", canary="candidate", traffic_pct=5.0)
    for _ in range(sample_count):
        record_bucket_outcome(
            policy,
            "stable",
            True,
            rollout_id="rollout-healthy",
            path=str(stats_path),
        )
        record_bucket_outcome(
            policy,
            "canary",
            True,
            rollout_id="rollout-healthy",
            path=str(stats_path),
        )

    final_state, iterations = watch(
        WatchConfig(
            poll_interval_sec=1.0,
            max_iterations=1,
            state_path=str(state_path),
            canary_stats_path=str(stats_path),
            iteration_dir=str(tmp_path / ".soup-loops"),
        )
    )

    persisted = read_state(str(state_path))
    assert iterations == 1
    assert final_state.canary_active == "candidate"
    assert final_state.canary_traffic_pct == 5.0
    assert persisted.canary_active == "candidate"
    assert persisted.canary_traffic_pct == 5.0
    assert persisted.canary_rollout_id == "rollout-healthy"


def _write_canary_state(tmp_path, *, traffic_pct, rollout_id):
    state_path = tmp_path / ".soup" / "loop.yaml"
    write_state(
        LoopState(
            served_model="base",
            eval_suite="evals.yaml",
            baseline="prod",
            status="running",
            canary_active="candidate",
            canary_traffic_pct=traffic_pct,
            canary_autoroll_on_regress=True,
            canary_rollout_id=rollout_id,
        ),
        str(state_path),
    )
    return state_path, tmp_path / ".soup" / "canary-stats.json"


def _record(stats_path, rollout_id, *, stable_ok, canary_ok, count):
    policy = CanaryPolicy(stable="base", canary="candidate", traffic_pct=5.0)
    for _ in range(count):
        record_bucket_outcome(
            policy, "stable", stable_ok, rollout_id=rollout_id, path=str(stats_path)
        )
        record_bucket_outcome(
            policy, "canary", canary_ok, rollout_id=rollout_id, path=str(stats_path)
        )


def _watch_once(tmp_path, state_path, stats_path):
    return watch(
        WatchConfig(
            poll_interval_sec=1.0,
            max_iterations=1,
            state_path=str(state_path),
            canary_stats_path=str(stats_path),
            iteration_dir=str(tmp_path / ".soup-loops"),
        )
    )


def _serve_with_canary(monkeypatch, state_path, stats_path, adapters, fail_on=None):
    import soup_cli.commands.serve as serve

    observed = []

    def generate(current_model, *args, **kwargs):
        observed.append(current_model.current)
        if fail_on is not None and current_model.current == fail_on:
            raise RuntimeError("bad canary")
        return "ok", 1, 1

    monkeypatch.setattr(serve, "_generate_response", generate)
    app = serve._create_app(
        model_obj=_AdapterModel(),
        tokenizer=object(),
        device="cpu",
        model_name="base",
        max_tokens_default=8,
        adapter_map={name: "unused" for name in adapters},
        peft_adapter_names=set(adapters),
        canary_state_path=str(state_path),
        canary_stats_path=str(stats_path),
    )
    return TestClient(app, base_url="http://127.0.0.1"), observed


def test_samples_from_an_earlier_rollout_never_count(tmp_path, monkeypatch):
    """Re-promoting the same adapter names must start from zero evidence."""
    monkeypatch.chdir(tmp_path)
    state_path, stats_path = _write_canary_state(
        tmp_path, traffic_pct=5.0, rollout_id="rollout-new"
    )
    _record(stats_path, "rollout-old", stable_ok=True, canary_ok=False, count=30)

    final_state, _ = _watch_once(tmp_path, state_path, stats_path)

    assert final_state.canary_active == "candidate"
    assert read_state(str(state_path)).canary_active == "candidate"


def test_an_undersampled_failing_canary_is_not_rolled_back(tmp_path, monkeypatch):
    """Five failed canary samples are UNKNOWN (fewer than 30), not MAJOR."""
    monkeypatch.chdir(tmp_path)
    state_path, stats_path = _write_canary_state(
        tmp_path, traffic_pct=5.0, rollout_id="rollout-few"
    )
    _record(stats_path, "rollout-few", stable_ok=True, canary_ok=False, count=5)

    final_state, _ = _watch_once(tmp_path, state_path, stats_path)

    persisted = read_state(str(state_path))
    assert final_state.canary_active == "candidate"
    assert persisted.canary_active == "candidate"
    assert persisted.canary_rollout_id == "rollout-few"


def test_each_canary_promotion_gets_a_fresh_rollout_id(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    init = runner.invoke(app, ["loop", "init", "base", "--eval", "e", "--baseline", "b"])
    assert init.exit_code == 0, (init.output, repr(init.exception))
    rollout_ids = []
    for _ in range(2):
        result = runner.invoke(app, ["loop", "canary", "candidate", "--traffic", "5%"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        rollout_ids.append(read_state().canary_rollout_id)
    assert all(rollout_ids)
    assert rollout_ids[0] != rollout_ids[1]


def test_keyed_stable_traffic_is_the_base_even_after_a_manual_cutover(
    tmp_path, monkeypatch
):
    """The stable bucket is what the stats call stable, not an activated adapter."""
    monkeypatch.chdir(tmp_path)
    state_path, stats_path = _write_canary_state(
        tmp_path, traffic_pct=50.0, rollout_id="rollout-cutover"
    )
    client, observed = _serve_with_canary(
        monkeypatch, state_path, stats_path, adapters=("candidate", "other")
    )
    assert client.post("/v1/adapters/activate/other").status_code == 200

    for index in range(100):
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "conversation_id": f"cutover-{index}",
            },
        )
        assert response.status_code == 200

    assert "other" not in observed
    assert 0 < observed.count("candidate") < 100
    client.app.state.canary_outcomes.close()
    stats = read_bucket_stats(
        stable="base", canary="candidate", rollout_id="rollout-cutover", path=str(stats_path)
    )
    assert stats.stable_ok == observed.count(None)


def test_failing_live_canary_stream_requests_reach_the_rollout_stats(
    tmp_path, monkeypatch
):
    """A streamed generation error still answers 200; the stats file is its only record."""
    monkeypatch.chdir(tmp_path)
    state_path, stats_path = _write_canary_state(
        tmp_path, traffic_pct=50.0, rollout_id="rollout-stream"
    )
    client, observed = _serve_with_canary(
        monkeypatch, state_path, stats_path, adapters=("candidate",), fail_on="candidate"
    )

    for index in range(100):
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "conversation_id": f"stream-{index}",
                "stream": True,
            },
        )
        assert response.status_code == 200

    canary_count = observed.count("candidate")
    client.app.state.canary_outcomes.close()
    stats = read_bucket_stats(
        stable="base", canary="candidate", rollout_id="rollout-stream", path=str(stats_path)
    )
    assert canary_count > 0
    assert stats.canary_major == canary_count
    assert stats.canary_ok == 0
    assert stats.stable_ok == observed.count(None)
    assert stats.stable_major == 0


def test_an_empty_public_adapter_field_keeps_the_manual_activation(tmp_path, monkeypatch):
    """An empty public adapter field must preserve the manual activation."""
    monkeypatch.chdir(tmp_path)
    client, observed = _serve_with_canary(
        monkeypatch,
        tmp_path / ".soup" / "loop.yaml",
        tmp_path / ".soup" / "canary-stats.json",
        adapters=("chat",),
    )
    assert client.post("/v1/adapters/activate/chat").status_code == 200

    for stream in (False, True):
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "adapter": "",
                "stream": stream,
            },
        )
        assert response.status_code == 200

    assert observed == ["chat", "chat"]
