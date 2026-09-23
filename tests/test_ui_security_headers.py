"""soup ui response headers and script integrity."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import soup_cli.ui.app as ui_mod  # noqa: E402
from soup_cli.ui.app import create_app, get_auth_token  # noqa: E402

INDEX = Path(__file__).resolve().parents[1] / "src" / "soup_cli" / "ui" / "static" / "index.html"
CHART_URL = "https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"


def _directives(policy: str) -> dict[str, list[str]]:
    out = {}
    for part in policy.split(";"):
        tokens = part.split()
        if tokens:
            out[tokens[0]] = tokens[1:]
    return out


def _assert_plain_headers(resp):
    assert resp.headers.get("x-content-type-options") == "nosniff", dict(resp.headers)
    assert resp.headers.get("referrer-policy") == "no-referrer"
    assert resp.headers.get("x-frame-options") == "DENY"


@pytest.fixture
def client():
    return TestClient(create_app())


@pytest.mark.parametrize("path", ["/", "/static/app.js", "/static/safe_html.js", "/api/system"])
def test_headers_on_every_response(client, path):
    resp = client.get(path)
    policy = resp.headers.get("content-security-policy")
    assert policy, (path, dict(resp.headers))
    assert policy == ui_mod.CONTENT_SECURITY_POLICY
    _assert_plain_headers(resp)


def test_security_headers_mapping_is_exactly_four_and_read_only():
    assert set(ui_mod.SECURITY_HEADERS) == {
        "Content-Security-Policy",
        "X-Content-Type-Options",
        "Referrer-Policy",
        "X-Frame-Options",
    }
    with pytest.raises(TypeError):
        ui_mod.SECURITY_HEADERS["X-Frame-Options"] = "SAMEORIGIN"  # type: ignore[index]


def test_policy_forbids_inline_script(client):
    d = _directives(client.get("/").headers["content-security-policy"])
    assert d["script-src"] == ["'self'", CHART_URL]
    for directive in ("script-src", "default-src"):
        assert "'unsafe-inline'" not in d.get(directive, [])
        assert "'unsafe-eval'" not in d.get(directive, [])
    assert d["object-src"] == ["'none'"]
    assert d["base-uri"] == ["'none'"]
    assert d["frame-ancestors"] == ["'none'"]
    assert d["connect-src"] == ["'self'"]
    assert d["default-src"] == ["'self'"]
    assert d["form-action"] == ["'self'"]


def test_headers_on_error_responses(client):
    resp = client.get("/api/does-not-exist")
    assert resp.status_code == 404
    assert resp.headers.get("content-security-policy")
    _assert_plain_headers(resp)


def test_headers_on_unauthorized_response(client):
    resp = client.post("/api/auth/ticket")
    assert resp.status_code == 401
    assert resp.headers.get("content-security-policy")


def test_headers_on_body_size_rejection(client):
    resp = client.post(
        "/api/data/inspect",
        content=b"x" * (64 * 1024),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 413
    assert resp.headers.get("content-security-policy")
    _assert_plain_headers(resp)


def test_headers_on_event_stream():
    ui_mod._train_process = None
    ui_mod._train_log_buffer = None
    client = TestClient(create_app())
    with client.stream(
        "GET", "/api/train/logs", headers={"Authorization": f"Bearer {get_auth_token()}"}
    ) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        assert resp.headers.get("content-security-policy")
        body = b"".join(resp.iter_bytes())
    assert b"done" in body


def test_loopback_docs_keep_working_without_the_page_policy(client):
    # FastAPI's interactive docs page runs an inline bootstrap script, so the
    # UI policy would blank it. The docs exist only on a loopback bind.
    for path in ("/docs", "/redoc"):
        resp = client.get(path)
        assert resp.status_code == 200, path
        assert "content-security-policy" not in resp.headers, path
        _assert_plain_headers(resp)


def test_docs_path_on_public_bind_is_absent_and_still_carries_policy(monkeypatch):
    monkeypatch.setattr(ui_mod, "_auth_token", "a" * 43)
    public = TestClient(create_app(host="0.0.0.0"))
    resp = public.get("/docs")
    assert resp.status_code == 404
    assert resp.headers.get("content-security-policy")


def _run_middleware(downstream_messages):
    sent = []

    async def app(scope, receive, send):
        for message in downstream_messages:
            await send(message)

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b""}

    middleware = ui_mod._SecurityHeadersMiddleware(app)
    asyncio.run(middleware({"type": "http", "path": "/x", "headers": []}, receive, send))
    return sent


def test_middleware_forwards_each_body_chunk_without_buffering():
    messages = [
        {"type": "http.response.start", "status": 200, "headers": []},
        {"type": "http.response.body", "body": b"a", "more_body": True},
        {"type": "http.response.body", "body": b"b", "more_body": False},
    ]
    sent = _run_middleware(messages)
    assert [m["type"] for m in sent] == [m["type"] for m in messages]
    assert [m.get("body") for m in sent[1:]] == [b"a", b"b"]
    names = {k.decode().lower() for k, _ in sent[0]["headers"]}
    assert {"content-security-policy", "x-frame-options"} <= names


def test_middleware_does_not_overwrite_an_existing_header():
    sent = _run_middleware([
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"x-frame-options", b"SAMEORIGIN")],
        },
        {"type": "http.response.body", "body": b""},
    ])
    values = [v for k, v in sent[0]["headers"] if k.lower() == b"x-frame-options"]
    assert values == [b"SAMEORIGIN"]


def test_middleware_is_pure_asgi():
    from starlette.middleware.base import BaseHTTPMiddleware

    assert not issubclass(ui_mod._SecurityHeadersMiddleware, BaseHTTPMiddleware)


def test_chart_script_has_integrity():
    text = INDEX.read_text(encoding="utf-8")
    tag = re.search(r"<script[^>]*chart\.umd\.min\.js[^>]*>", text).group(0)
    assert CHART_URL in tag
    assert re.search(r'integrity="sha384-[A-Za-z0-9+/=]{64}"', tag), tag
    assert 'crossorigin="anonymous"' in tag
