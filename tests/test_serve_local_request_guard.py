"""soup serve: tool and adapter routes check Host and Origin."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


def _app(**kwargs):
    from soup_cli.commands.serve import _create_app

    base = dict(
        model_obj=MagicMock(),
        tokenizer=MagicMock(),
        device="cpu",
        model_name="test-model",
        max_tokens_default=256,
        adapter_map={"chat": "/fake/path/chat"},
    )
    base.update(kwargs)
    return _create_app(**base)


GUARDED = [
    ("post", "/v1/adapters/activate/chat", None),
    ("post", "/v1/adapters/deactivate", None),
    ("post", "/v1/tools/python", {"code": "print(1)"}),
    ("post", "/v1/tools/bash", {"command": "echo 1"}),
    ("post", "/v1/tools/web_search", {"query": "x"}),
    ("post", "/v1/thumbs", {"prompt": "p", "response": "r", "thumb": "up"}),
]


@pytest.mark.parametrize(("method", "path", "body"), GUARDED)
def test_foreign_host_refused(method, path, body):
    client = TestClient(_app(), base_url="http://evil.example:8000")
    resp = getattr(client, method)(path, json=body)
    assert resp.status_code == 421, (path, resp.status_code, resp.text)


@pytest.mark.parametrize(("method", "path", "body"), GUARDED)
def test_foreign_origin_refused(method, path, body):
    client = TestClient(_app(), base_url="http://127.0.0.1:8000")
    resp = getattr(client, method)(path, json=body, headers={"Origin": "http://evil.example"})
    assert resp.status_code == 403, (path, resp.status_code, resp.text)


def test_loopback_adapter_activation_still_works():
    client = TestClient(_app(), base_url="http://127.0.0.1:8000")
    resp = client.post("/v1/adapters/activate/chat")
    assert resp.status_code == 200, resp.text
    assert resp.json()["active"] == "chat"


def test_loopback_origin_allowed():
    client = TestClient(_app(), base_url="http://localhost:8000")
    resp = client.post("/v1/adapters/deactivate", headers={"Origin": "http://127.0.0.1:3000"})
    assert resp.status_code == 200, resp.text


def test_inference_routes_not_host_checked():
    client = TestClient(_app(), base_url="http://proxy.example")
    assert client.get("/health").status_code == 200
    assert client.get("/v1/models").status_code == 200


def test_foreign_host_refused_even_with_valid_token():
    client = TestClient(_app(auth_token="s3cret-token"), base_url="http://evil.example")
    resp = client.post(
        "/v1/tools/python",
        json={"code": "print(1)"},
        headers={"Authorization": "Bearer s3cret-token"},
    )
    assert resp.status_code == 421


def test_token_still_required_on_loopback_when_set():
    client = TestClient(_app(auth_token="s3cret-token"), base_url="http://127.0.0.1")
    resp = client.post(
        "/v1/tools/python",
        json={"code": "print(1)"},
        headers={"Authorization": "Bearer wrong"},
    )
    assert resp.status_code == 401


def test_token_compare_is_constant_time():
    import inspect

    from soup_cli.commands import serve

    src = inspect.getsource(serve._create_app)
    assert "compare_digest" in src
    assert "authorization != expected" not in src
