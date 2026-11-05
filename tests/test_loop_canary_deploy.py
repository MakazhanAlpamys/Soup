"""A loop deploy is a canary, a passing canary is promoted, a rollback is honest.

The pre-wired deploy stage used to POST ``/v1/adapters/activate/<name>``: a
100% cutover, refused with 404 for an adapter the server had not loaded at
startup (every freshly trained one), and no canary was ever started. A
rollback cleared the canary in ``loop.yaml`` but never gave the previous
adapter back. Now the deploy loads the adapter into the running server
(``/v1/adapters/load``) and starts a canary on ``deploy_traffic_pct`` (5% by
default) of keyed traffic; the next watch cycle promotes it once 30 canary
answers say OK (and only then points unkeyed traffic at it), or withdraws it
on MAJOR; ``soup loop rollback`` returns to the previous production version.
"""

from __future__ import annotations

import json
import os
import re

import pytest
from typer.testing import CliRunner

import soup_cli.utils.loop_stages as ls
from soup_cli.utils.canary_router import CanaryPolicy, record_bucket_outcome
from soup_cli.utils.loop_daemon import WatchConfig, run_once
from soup_cli.utils.loop_state import LoopState, read_state, write_state

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _clean(text: str) -> str:
    return " ".join(_ANSI.sub("", text).split())


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", "http://127.0.0.1:8000")
    monkeypatch.setenv(ls.TOOL_AUTH_TOKEN_ENV, "t" * 32)
    for name in ("SOUP_LOOP_TRACE_DIR", "SOUP_LOOP_THUMBS_DB"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(ls, "_DEPLOY_POSTER", None)
    (tmp_path / "tasks.jsonl").write_text('{"prompt": "2+2", "expected": "4"}\n', "utf-8")
    return tmp_path


def _state(**kwargs) -> LoopState:
    kwargs.setdefault("served_model", "./base")
    kwargs.setdefault("base_model", "org/base")
    kwargs.setdefault("eval_suite", "tasks.jsonl")
    kwargs.setdefault("verify", "ship")
    kwargs.setdefault("status", "running")
    return LoopState(baseline="b", **kwargs)


def _loader(status=200, detail="", calls=None):
    def post(endpoint, name, path):
        if calls is not None:
            calls.append((endpoint, name, path))
        return status, detail

    return post


def _deploy(state, adapter="adapters/loop-train-1a2b3c4d"):
    return ls.deploy_to_canary(state, {"gate_verdict": "OK", "adapter_path": adapter})


# ---------------------------------------------------------------------------
# The deploy stage
# ---------------------------------------------------------------------------


def test_deploy_loads_the_adapter_and_starts_a_canary(cwd, monkeypatch):
    calls = []
    monkeypatch.setattr(ls, "_LOAD_POSTER", _loader(calls=calls))
    out = _deploy(_state())
    assert out["deployed"] is True
    assert out["canary_active"] == "loop-train-1a2b3c4d"
    assert out["canary_adapter"] == "adapters/loop-train-1a2b3c4d"
    assert out["canary_traffic_pct"] == 5.0
    assert re.fullmatch(r"[0-9a-f]{32}", out["canary_rollout_id"])
    assert out["canary_verdict"] is None
    assert out["notes"] == "canary loop-train-1a2b3c4d: 5% of keyed traffic"
    endpoint, name, path = calls[0]
    assert endpoint == "http://127.0.0.1:8000" and name == "loop-train-1a2b3c4d"
    assert path == os.path.abspath("adapters/loop-train-1a2b3c4d")


def test_deploy_uses_the_configured_traffic_share(cwd, monkeypatch):
    monkeypatch.setattr(ls, "_LOAD_POSTER", _loader())
    assert _deploy(_state(deploy_traffic_pct=12.5))["canary_traffic_pct"] == 12.5


def test_deploy_never_activates_the_adapter_for_all_traffic(cwd, monkeypatch):
    def no_activate(endpoint, name):
        raise AssertionError("a deploy must not cut over 100% of traffic")

    monkeypatch.setattr(ls, "_post_activate", no_activate)
    monkeypatch.setattr(ls, "_LOAD_POSTER", _loader())
    assert _deploy(_state())["deployed"] is True


def test_a_re_run_finds_the_adapter_already_loaded(cwd, monkeypatch):
    monkeypatch.setattr(ls, "_LOAD_POSTER", _loader(409, "adapter name already loaded"))
    assert _deploy(_state())["deployed"] is True


@pytest.mark.parametrize(
    ("status", "detail", "note"),
    [
        (401, "", "adapter load refused (401): the serve endpoint rejected the tool token"),
        (404, "", "start soup serve with --adapter-dir"),
        (403, "", "the adapter is outside the server's --adapter-dir"),
        (409, "runtime adapter limit reached (16)", "HTTP 409: runtime adapter limit reached"),
        (0, "ConnectError", "no response from the serve endpoint (ConnectError)"),
    ],
)
def test_a_refused_load_names_why_and_deploys_nothing(cwd, monkeypatch, status, detail, note):
    monkeypatch.setattr(ls, "_LOAD_POSTER", _loader(status, detail))
    out = _deploy(_state())
    assert out["deployed"] is False
    assert note in out["notes"]
    assert "canary_active" not in out


def test_a_missing_token_is_named(cwd, monkeypatch):
    monkeypatch.delenv(ls.TOOL_AUTH_TOKEN_ENV)
    monkeypatch.setattr(ls, "_LOAD_POSTER", _loader(401))
    assert "needs the serve tool token" in _deploy(_state())["notes"]


def test_no_second_canary_while_one_runs(cwd, monkeypatch):
    calls = []
    monkeypatch.setattr(ls, "_LOAD_POSTER", _loader(calls=calls))
    out = _deploy(_state(canary_active="loop-train-00000000", canary_traffic_pct=5.0))
    assert out["deployed"] is False
    assert out["notes"] == "canary loop-train-00000000 is still running; not starting another"
    assert calls == []


def test_the_real_loader_posts_name_and_path_with_the_token(cwd, monkeypatch):
    import httpx

    sent = {}

    def post(url, **kwargs):
        sent.update(url=url, **kwargs)
        return httpx.Response(200, json={"name": "v2", "status": "loaded"})

    monkeypatch.setattr(httpx, "post", post)
    status, _ = ls._post_load("http://127.0.0.1:8000", "v2", "/abs/v2")
    assert status == 200
    assert sent["url"] == "http://127.0.0.1:8000/v1/adapters/load"
    assert sent["json"] == {"name": "v2", "path": "/abs/v2"}
    assert sent["headers"] == {"Authorization": "Bearer " + "t" * 32}
    assert sent["trust_env"] is False


def test_the_real_loader_returns_the_server_detail(cwd, monkeypatch):
    import httpx

    monkeypatch.setattr(
        httpx, "post", lambda url, **kw: httpx.Response(409, json={"detail": "x\x1b[2Jy"})
    )
    status, detail = ls._post_load("http://127.0.0.1:8000", "v2", "/abs/v2")
    assert (status, detail) == (409, "x [2Jy")


# ---------------------------------------------------------------------------
# run_once: start, wait, promote, roll back
# ---------------------------------------------------------------------------


def _config(**kwargs) -> WatchConfig:
    kwargs.setdefault("max_iterations", 1)
    kwargs.setdefault("canary_stats_path", os.path.join(".soup", "canary-stats.json"))
    return WatchConfig(**kwargs)


def _record(state, bucket_ok: dict, rollout="r1"):
    """Record stable/canary outcomes for the state's active canary."""
    policy = CanaryPolicy(stable=state.stable_name(), canary=state.canary_active, traffic_pct=5.0)
    for bucket, (ok, bad) in bucket_ok.items():
        for _ in range(ok):
            record_bucket_outcome(policy, bucket, True, rollout_id=rollout,
                                  path=os.path.join(".soup", "canary-stats.json"))
        for _ in range(bad):
            record_bucket_outcome(policy, bucket, False, rollout_id=rollout,
                                  path=os.path.join(".soup", "canary-stats.json"))


def _canary_state(**kwargs) -> LoopState:
    return _state(
        canary_active="v2", canary_traffic_pct=5.0, canary_rollout_id="r1",
        canary_adapter="adapters/v2", **kwargs,
    )


def test_a_deployed_canary_is_written_to_loop_yaml(cwd):
    deploy_out = {
        "deployed": True, "canary_verdict": None, "canary_active": "v2",
        "canary_adapter": "adapters/v2", "canary_traffic_pct": 5.0, "canary_rollout_id": "r9",
    }
    config = _config(
        harvest_fn=lambda s: {"pairs_harvested": 1},
        train_fn=lambda s, c: {"run_id": "r", "skipped": False, "adapter_path": "adapters/v2"},
        gate_fn=lambda s, c: {"gate_verdict": "OK"},
        deploy_fn=lambda s, c: deploy_out,
    )
    state, record, _ = run_once(_state(), config)
    assert record.shipped is True
    assert (state.canary_active, state.canary_adapter) == ("v2", "adapters/v2")
    assert (state.canary_traffic_pct, state.canary_rollout_id) == (5.0, "r9")


def test_a_running_canary_pauses_new_iterations(cwd):
    state = _canary_state()
    _record(state, {"stable": (40, 0), "canary": (12, 0)})
    trained = []
    config = _config(train_fn=lambda s, c: trained.append(1) or {"skipped": True})
    new_state, record, decision = run_once(state, config)
    assert decision.proceed is False
    assert trained == []
    assert record.notes == "canary-wait: v2 has 12 canary answers (verdict at 30)"
    assert new_state.canary_active == "v2"


def test_a_passing_canary_is_promoted(cwd):
    state = _canary_state(production_name="v1", production_adapter="adapters/v1")
    _record(state, {"stable": (40, 0), "canary": (30, 0)})
    told = []
    config = _config(promote_fn=lambda s: told.append(s.canary_active) or {"promoted": True})
    new_state, record, _ = run_once(state, config)
    assert told == ["v2"]
    assert (new_state.production_name, new_state.production_adapter) == ("v2", "adapters/v2")
    assert new_state.previous_production_name == "v1"
    assert new_state.previous_production_adapter == "adapters/v1"
    assert new_state.canary_active is None and new_state.canary_adapter is None
    assert "canary v2 promoted to production" in record.notes


def test_a_promotion_the_server_refused_keeps_the_canary(cwd):
    state = _canary_state()
    _record(state, {"stable": (40, 0), "canary": (30, 0)})
    config = _config(promote_fn=lambda s: {"promoted": False, "notes": "401"})
    new_state, record, _ = run_once(state, config)
    assert new_state.canary_active == "v2"
    assert new_state.production_name is None
    assert "canary v2 passed but was not promoted: 401" in record.notes


def test_a_regressing_canary_is_withdrawn_and_production_untouched(cwd):
    state = _canary_state(production_name="v1", production_adapter="adapters/v1")
    _record(state, {"stable": (40, 0), "canary": (20, 10)})
    promoted = []
    config = _config(promote_fn=lambda s: promoted.append(1) or {"promoted": True})
    new_state, record, _ = run_once(state, config)
    assert promoted == []
    assert record.rolled_back is True
    assert new_state.canary_active is None and new_state.canary_adapter is None
    assert (new_state.production_name, new_state.production_adapter) == ("v1", "adapters/v1")
    assert "canary v2 rolled back (MAJOR)" in record.notes


def test_the_router_stable_side_follows_the_promoted_adapter(cwd):
    assert _state().stable_name() == "./base"
    assert _state(production_name="v2").stable_name() == "v2"


def test_promote_points_unkeyed_traffic_at_the_canary(cwd, monkeypatch):
    seen = []
    monkeypatch.setattr(ls, "_post_activate", lambda ep, name: seen.append((ep, name)) or True)
    assert ls.promote_canary(_canary_state()) == {"promoted": True, "notes": ""}
    assert seen == [("http://127.0.0.1:8000", "v2")]


def test_promote_without_an_endpoint_changes_the_state_and_says_so(cwd, monkeypatch):
    monkeypatch.delenv("SOUP_LOOP_SERVE_ENDPOINT")
    out = ls.promote_canary(_canary_state())
    assert out == {
        "promoted": True,
        "notes": "the server was not told (no SOUP_LOOP_SERVE_ENDPOINT)",
    }


def test_promote_refuses_a_public_endpoint(cwd, monkeypatch):
    monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", "https://example.com")
    out = ls.promote_canary(_canary_state())
    assert out["promoted"] is False and "loopback/LAN" in out["notes"]


# ---------------------------------------------------------------------------
# soup loop rollback / init --canary-traffic / status
# ---------------------------------------------------------------------------


def _invoke(*args: str):
    from soup_cli.commands.loop import app

    result = runner.invoke(app, list(args))
    return result, _clean(result.output)


def test_rollback_returns_the_previous_adapter(cwd, monkeypatch):
    write_state(_state(
        production_name="v2", production_adapter="adapters/v2",
        previous_production_name="v1", previous_production_adapter="adapters/v1",
    ))
    seen = []
    monkeypatch.setattr(ls, "_post_activate", lambda ep, name: seen.append(name) or True)
    result, text = _invoke("rollback")
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert seen == ["v1"]
    state = read_state()
    assert (state.production_name, state.production_adapter) == ("v1", "adapters/v1")
    assert state.previous_production_name is None
    assert "rolled back: v2 -> v1" in text


def test_rollback_to_the_served_model_deactivates(cwd, monkeypatch):
    write_state(_state(production_name="v2", production_adapter="adapters/v2",
                       canary_active="v3", canary_traffic_pct=5.0, canary_rollout_id="r"))
    seen = []
    monkeypatch.setattr(ls, "_post_deactivate", lambda ep: seen.append(ep) or True)
    result, text = _invoke("rollback")
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert seen == ["http://127.0.0.1:8000"]
    state = read_state()
    assert state.production_name is None and state.production_adapter is None
    assert state.canary_active is None
    assert "rolled back: v2 -> the served model" in text


def test_rollback_refuses_when_nothing_was_promoted(cwd):
    write_state(_state())
    result, text = _invoke("rollback")
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "nothing to roll back to" in text


def test_a_refused_rollback_leaves_loop_yaml_unchanged(cwd, monkeypatch):
    write_state(_state(production_name="v2", previous_production_name="v1"))
    monkeypatch.setattr(ls, "_post_activate", lambda ep, name: False)
    result, text = _invoke("rollback")
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "rollback failed: the serve endpoint did not switch the adapter" in text
    assert read_state().production_name == "v2"


def test_init_saves_the_canary_traffic_and_status_shows_it(cwd):
    result, _ = _invoke(
        "init", "./base", "--eval", "tasks.jsonl", "--baseline", "b", "--canary-traffic", "10%"
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert read_state().deploy_traffic_pct == 10.0
    _, status = _invoke("status")
    assert re.search(r"canary_traffic\W+10% per new canary", status), status


@pytest.mark.parametrize("value", ["0", "0%", "101", "lots"])
def test_init_refuses_a_bad_canary_traffic(cwd, value):
    result, _ = _invoke(
        "init", "./base", "--eval", "tasks.jsonl", "--baseline", "b", "--canary-traffic", value
    )
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert not os.path.exists(os.path.join(".soup", "loop.yaml"))


def test_state_fields_are_validated():
    with pytest.raises(ValueError, match=r"deploy_traffic_pct must be in \(0, 100\]"):
        _state(deploy_traffic_pct=0)
    with pytest.raises(ValueError, match="production_name must not be empty"):
        _state(production_name="")


def test_old_state_files_load_with_the_defaults(cwd):
    os.makedirs(".soup")
    with open(os.path.join(".soup", "loop.yaml"), "w", encoding="utf-8") as fh:
        json.dump({"served_model": "./m", "eval_suite": "e", "baseline": "b"}, fh)
    state = read_state()
    assert state.deploy_traffic_pct == 5.0
    assert state.production_name is None and state.stable_name() == "./m"
