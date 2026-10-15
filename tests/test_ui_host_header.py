"""Loopback UI Host validation before routes, authentication and body reads."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from soup_cli.ui.app import SECURITY_HEADERS, _HostAllowlistMiddleware, create_app, get_auth_token
from tests.ui_helpers import ui_client

PORT = 7860
BASE_URL = f"http://127.0.0.1:{PORT}"


def _client(**kwargs) -> TestClient:
    return ui_client(**kwargs)


def _bearer_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {get_auth_token()}"}


def _raw_request(headers: list[tuple[bytes, bytes]], **scope_overrides) -> list[dict]:
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "scheme": "http",
        "method": "GET", "path": "/api/health", "raw_path": b"/api/health",
        "query_string": b"", "root_path": "", "headers": headers,
        "client": ("127.0.0.1", 54321), "server": ("127.0.0.1", PORT),
        **scope_overrides,
    }
    sent = []

    async def receive():
        raise AssertionError("A refused request must not read its body")

    async def send(message):
        sent.append(message)

    asyncio.run(create_app()(scope, receive, send))
    return sent


class TestLoopbackHostAllowlist:
    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "[::1]"])
    def test_served_names(self, host):
        assert _client().get("/api/health", headers={"Host": f"{host}:{PORT}"}).status_code == 200

    @pytest.mark.parametrize("host", ["::1", "localhost", "127.0.0.2"])
    def test_other_loopback_binds_serve_the_bound_host_and_aliases(self, host):
        client = _client(host=host)
        bound_name = f"[{host}]" if ":" in host else host
        for name in (bound_name, "127.0.0.1", "localhost", "[::1]"):
            assert client.get("/api/health", headers={"Host": f"{name}:{PORT}"}).status_code == 200

    @pytest.mark.parametrize("host", ["::1", "0:0:0:0:0:0:0:1"])
    @pytest.mark.parametrize("port", [80, 8080])
    def test_ipv6_client_preserves_the_bound_host_and_port(self, host, port):
        # Use explicit Host headers so TestClient need not parse an IPv6 base URL.
        client = _client(host=host, port=port)
        assert client.get("/api/health", headers={"Host": f"[{host}]:{port}"}).status_code == 200
        assert client.get("/api/health", headers={"Host": f"[{host}]:7860"}).status_code == 403

    @pytest.mark.parametrize("value", ["[0:0:0:0:0:0:0:1]:7860", "LOCALHOST:7860"])
    def test_ip_literal_and_hostname_normalization(self, value):
        assert _client().get("/api/health", headers={"Host": value}).status_code == 200

    @pytest.mark.parametrize("name", ["127.0.0.1", "localhost", "[::1]"])
    def test_default_http_port_is_served_on_port_80(self, name):
        client = TestClient(create_app(port=80), base_url="http://127.0.0.1")
        assert client.get("/api/health", headers={"Host": name}).status_code == 200
        assert client.get("/api/health", headers={"Host": f"{name}:80"}).status_code == 200
        assert client.get("/api/health", headers={"Host": f"{name}:{PORT}"}).status_code == 403

    def test_nondefault_bound_port_is_served(self):
        client = _client(port=8080)
        assert client.get("/api/health", headers={"Host": "localhost:8080"}).status_code == 200
        assert client.get("/api/health", headers={"Host": "localhost:7860"}).status_code == 403


class TestForeignHostRefused:
    @pytest.mark.parametrize("path", [
        "/", "/static/app.js", "/openapi.json", "/docs", "/redoc",
        "/docs/oauth2-redirect", "/api/health", "/api/train/status", "/unknown",
    ])
    def test_foreign_host_is_refused_on_every_route(self, path):
        resp = _client().get(path, headers={"Host": "evil.example.com:7860"})
        assert resp.status_code == 403
        assert resp.json() == {"detail": "Host not allowed"}
        for name, value in SECURITY_HEADERS.items():
            assert resp.headers[name] == value

    def test_foreign_host_bearer_post_is_refused(self):
        client = _client()
        assert client.post("/api/auth/ticket", headers=_bearer_headers()).status_code == 200
        resp = client.post(
            "/api/auth/ticket", headers={"Host": "evil.example.com:7860", **_bearer_headers()},
        )
        assert resp.status_code == 403
        assert resp.json() == {"detail": "Host not allowed"}

    def test_foreign_host_cookie_post_is_refused(self):
        client = _client()
        assert client.get(f"/?token={get_auth_token()}", follow_redirects=False).status_code == 303
        assert client.post("/api/auth/ticket", headers={
            "Host": f"127.0.0.1:{PORT}", "Origin": BASE_URL,
        }).status_code == 200
        resp = client.post("/api/auth/ticket", headers={
            "Host": "evil.example.com:7860", "Origin": "http://evil.example.com:7860",
        })
        assert resp.status_code == 403
        assert resp.json() == {"detail": "Host not allowed"}

    @pytest.mark.parametrize("content_length", [None, b"2097152"])
    def test_foreign_host_is_refused_before_the_body_cap_reads_any_body(self, content_length):
        headers = [(b"host", b"evil.example.com:7860"), (b"content-type", b"application/json")]
        if content_length is not None:
            headers.append((b"content-length", content_length))
        sent = _raw_request(
            headers, method="POST", path="/api/chat/send", raw_path=b"/api/chat/send",
        )
        assert sent[0]["status"] == 403

    @pytest.mark.parametrize("value", ["127.0.0.1:9999", "127.0.0.1", "[::1]", ""])
    def test_wrong_port_or_missing_port_or_empty_host_is_refused(self, value):
        resp = _client().get("/api/health", headers={"Host": value})
        assert resp.status_code == 403
        assert resp.json() == {"detail": "Host not allowed"}

    def test_missing_host_is_refused(self):
        assert _raw_request([(b"accept", b"*/*")])[0]["status"] == 403

    @pytest.mark.parametrize("value", [
        "localhost:", "localhost:abc", "localhost:7860:7860", "::1:7860", "[::1",
        "[::1]x7860", "[::1]:", "[::1]:abc", "[:]:1:7860", "localhost.:7860",
        "[localhost]:7860", "[127.0.0.1]:7860",
        "localhost:0", "localhost:65536",
    ])
    def test_malformed_host_is_refused(self, value):
        assert _client().get("/api/health", headers={"Host": value}).status_code == 403

    @pytest.mark.parametrize("value", [b"localhost:\xb2", b"[::1]:\xb2"])
    def test_non_ascii_port_is_refused(self, value):
        assert _raw_request([(b"host", value)])[0]["status"] == 403

    @pytest.mark.parametrize("values", [
        (b"evil.example.com:7860", b"127.0.0.1:7860"),
        (b"127.0.0.1:7860", b"evil.example.com:7860"),
        (b"127.0.0.1:7860", b"127.0.0.1:7860"),
    ])
    def test_duplicate_host_headers_are_refused(self, values):
        assert _raw_request([(b"host", values[0]), (b"Host", values[1])])[0]["status"] == 403


class TestPublicBindUnchanged:
    @pytest.mark.parametrize("host", ["192.168.1.5", "0.0.0.0"])
    def test_public_bind_keeps_bearer_authentication(self, host):
        client = _client(host=host)
        headers = {"Host": "anything.example.com:7860"}
        assert client.get("/api/train/status", headers=headers).status_code == 401
        response = client.get("/api/train/status", headers={**headers, **_bearer_headers()})
        assert response.status_code == 200


class TestASGIScopes:
    def test_lifespan_messages_pass_through(self):
        received = [{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}]
        scope = {"type": "lifespan"}
        seen = []
        sent = []

        async def app(actual_scope, receive, send):
            assert actual_scope is scope
            seen.append(await receive())
            await send({"type": "lifespan.startup.complete"})
            seen.append(await receive())
            await send({"type": "lifespan.shutdown.complete"})

        async def receive():
            return received[len(seen)]

        async def send(message):
            sent.append(message)

        middleware = _HostAllowlistMiddleware(app, frozenset())
        asyncio.run(middleware(scope, receive, send))
        assert seen == received
        assert sent == [
            {"type": "lifespan.startup.complete"}, {"type": "lifespan.shutdown.complete"},
        ]

    def test_websocket_scopes_are_closed_without_calling_downstream(self):
        sent = []

        async def app(scope, receive, send):
            raise AssertionError("The HTTP-only UI gate must not pass a WebSocket to a route")

        async def receive():
            return {"type": "websocket.connect"}

        async def send(message):
            sent.append(message)

        middleware = _HostAllowlistMiddleware(app, frozenset())
        asyncio.run(middleware({"type": "websocket"}, receive, send))
        assert sent == [{"type": "websocket.close", "code": 1008}]
