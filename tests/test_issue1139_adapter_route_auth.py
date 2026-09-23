"""#1139 — the adapter hot-swap honours the serve tool token, and soup loop can send it.

`/v1/adapters/activate` and `/deactivate` depended only on the Host/Origin guard,
so on a non-loopback bind a server started with `--tool-auth-token` still let
anyone whose Host header named the bind switch the served adapter. They now
require the token whenever one is configured, and `soup loop watch` gains
`--tool-auth-token` (falling back to `SOUP_TOOL_AUTH_TOKEN`) so its deploy
stage can present it.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

TOKEN = "s3cret-tool-token"
LAN_HOST = "192.168.1.5"


def _client(host: str, auth_token: str | None) -> TestClient:
    from soup_cli.commands.serve import _create_app

    app = _create_app(
        model_obj=MagicMock(),
        tokenizer=MagicMock(),
        device="cpu",
        model_name="test",
        max_tokens_default=512,
        adapter_map={"chat": "./x"},
        auth_token=auth_token,
        host=host,
    )
    # The Host header names the bind, so the Host/Origin guard lets the request through.
    return TestClient(app, base_url=f"http://{host}")


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.parametrize("host", [LAN_HOST, "127.0.0.1"], ids=["lan", "loopback"])
def test_a_configured_token_gates_activate_and_deactivate(host):
    client = _client(host, TOKEN)

    for path in ("/v1/adapters/activate/chat", "/v1/adapters/deactivate"):
        assert client.post(path).status_code == 401, f"{path} without the token"
        assert client.post(path, headers=_bearer("wrong")).status_code == 401, (
            f"{path} with the wrong token"
        )
        assert client.post(path, headers=_bearer(TOKEN)).status_code == 200, (
            f"{path} with the token"
        )


def test_the_loopback_default_without_a_token_is_unchanged():
    client = _client("127.0.0.1", None)

    resp = client.post("/v1/adapters/activate/chat")
    assert resp.status_code == 200
    assert resp.json()["active"] == "chat"
    assert client.post("/v1/adapters/deactivate").status_code == 200


def test_the_token_does_not_gate_the_inference_routes():
    """The inference routes stay unguarded so a reverse proxy can front them."""
    client = _client(LAN_HOST, TOKEN)

    assert client.get("/v1/models").status_code == 200
    assert client.get("/v1/adapters").status_code == 200


def test_the_host_guard_still_runs_before_the_token_check():
    client = _client(LAN_HOST, TOKEN)

    resp = client.post(
        "/v1/adapters/activate/chat",
        headers={**_bearer(TOKEN), "Host": "evil.example"},
    )
    assert resp.status_code == 421


# ---------------------------------------------------------------------------
# client side: the loop's deploy stage
# ---------------------------------------------------------------------------


@pytest.fixture
def deploy_env(tmp_path, monkeypatch):
    import soup_cli.utils.loop_stages as ls
    from soup_cli.utils.loop_state import LoopState

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", f"https://{LAN_HOST}:8000")
    monkeypatch.delenv(ls.TOOL_AUTH_TOKEN_ENV, raising=False)
    monkeypatch.setattr(ls, "_DEPLOY_POSTER", None)

    def deploy():
        state = LoopState(
            served_model="m", eval_suite="e", baseline="b", status="running"
        )
        return ls.deploy_to_canary(
            state, {"gate_verdict": "OK", "adapter_path": "adapters/chat"}
        )

    return deploy


def _route_httpx_to(monkeypatch, client: TestClient) -> list:
    """Send the deploy stage's ``httpx.post`` to a real app through ``client``."""
    import httpx

    sent = []

    def post(url, headers=None, timeout=None):
        sent.append(dict(headers or {}))
        return client.post(url, headers=headers or {})

    monkeypatch.setattr(httpx, "post", post)
    return sent


def test_the_loop_token_reaches_a_server_started_with_it(deploy_env, monkeypatch):
    import soup_cli.utils.loop_stages as ls

    sent = _route_httpx_to(monkeypatch, _client(LAN_HOST, TOKEN))
    monkeypatch.setenv(ls.TOOL_AUTH_TOKEN_ENV, TOKEN)

    out = deploy_env()

    assert out["deployed"] is True
    assert sent == [_bearer(TOKEN)]


@pytest.mark.parametrize("token", [None, "wrong"], ids=["missing", "wrong"])
def test_a_refused_activate_names_the_credential(deploy_env, monkeypatch, token):
    import soup_cli.utils.loop_stages as ls

    _route_httpx_to(monkeypatch, _client(LAN_HOST, TOKEN))
    if token:
        monkeypatch.setenv(ls.TOOL_AUTH_TOKEN_ENV, token)

    out = deploy_env()

    assert out["deployed"] is False
    assert "401" in out["notes"]
    assert "--tool-auth-token" in out["notes"]
    assert "SOUP_TOOL_AUTH_TOKEN" in out["notes"]
    expected = "rejected the tool token" if token else "requires its tool token"
    assert expected in out["notes"]


def test_no_token_sends_no_authorization_header(deploy_env, monkeypatch):
    sent = _route_httpx_to(monkeypatch, _client("127.0.0.1", None))
    monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", "http://127.0.0.1:8000")

    out = deploy_env()

    assert out["deployed"] is True
    assert sent == [{}]


# ---------------------------------------------------------------------------
# the CLI option
# ---------------------------------------------------------------------------


def test_watch_detach_hands_the_token_to_the_child_through_the_environment(
    tmp_path, monkeypatch
):
    import os

    from typer.testing import CliRunner

    import soup_cli.commands.loop as loop_cmd
    from soup_cli.cli import app
    from soup_cli.utils.loop_stages import TOOL_AUTH_TOKEN_ENV

    monkeypatch.chdir(tmp_path)
    # recorded so monkeypatch restores the real value after the command sets it
    monkeypatch.setenv(TOOL_AUTH_TOKEN_ENV, "placeholder")
    spawned = {}

    class _Proc:
        pid = 4242

    def popen(argv, **kwargs):
        spawned["argv"] = list(argv)
        spawned["env"] = dict(os.environ) if kwargs.get("env") is None else kwargs["env"]
        return _Proc()

    monkeypatch.setattr(loop_cmd.subprocess, "Popen", popen)
    runner = CliRunner()
    runner.invoke(app, ["loop", "init", "m", "--eval", "e", "--baseline", "b"])

    result = runner.invoke(app, ["loop", "watch", "--detach", "--tool-auth-token", TOKEN])

    assert result.exit_code == 0, result.output
    assert TOKEN not in spawned["argv"], "the token must not appear in the child's argv"
    assert spawned["env"][TOOL_AUTH_TOKEN_ENV] == TOKEN


def test_the_token_is_masked_in_the_audit_log():
    from soup_cli.cli import _mask_audit_args

    masked = _mask_audit_args("loop", ["watch", "--tool-auth-token", TOKEN])

    assert TOKEN not in " ".join(masked)
    assert "--tool-auth-token" in masked
