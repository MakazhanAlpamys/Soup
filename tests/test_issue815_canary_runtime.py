"""Regression tests for the live canary path from issue #815."""

from contextlib import contextmanager

from fastapi.testclient import TestClient

from soup_cli.utils.canary_router import CanaryPolicy, record_bucket_outcome
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
