"""#1829 — ``soup ui`` refuses requests whose ``Host`` is not the served name.

On a loopback bind the server has exactly three names: the bound host with
the bound port, plus ``localhost`` / ``127.0.0.1`` / ``[::1]`` with the same
port. A request whose ``Host`` names anything else is answered 403 by the
``_HostAllowlistMiddleware`` before any route runs, which also closes the
hole the #1191 session-cookie check left open: it compared ``Origin``
against the request's own ``Host``, so any request naming the same host on
both sides passed. ``--public`` binds are unchanged (the Bearer token stays
the gate there).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from soup_cli.ui.app import create_app, get_auth_token

PORT = 7860
BASE_URL = f"http://127.0.0.1:{PORT}"


def _client(**kwargs) -> TestClient:
    return TestClient(create_app(**kwargs), base_url=BASE_URL)


def _bearer_headers() -> dict:
    return {"Authorization": f"Bearer {get_auth_token()}"}


class TestLoopbackHostAllowlist:
    def test_bound_host_is_served(self):
        resp = _client().get("/", headers={"Host": f"127.0.0.1:{PORT}"})
        assert resp.status_code == 200

    def test_localhost_alias_is_served(self):
        resp = _client().get("/", headers={"Host": f"localhost:{PORT}"})
        assert resp.status_code == 200

    def test_ipv6_loopback_alias_is_served(self):
        resp = _client().get("/", headers={"Host": f"[::1]:{PORT}"})
        assert resp.status_code == 200

    def test_ipv6_bind_serves_the_bracketed_name(self):
        client = TestClient(
            create_app(host="::1", port=PORT), base_url=f"http://[::1]:{PORT}"
        )
        resp = client.get("/", headers={"Host": f"[::1]:{PORT}"})
        assert resp.status_code == 200


class TestForeignHostRefused:
    def test_foreign_host_get_is_refused(self):
        resp = _client().get(
            "/api/train/status", headers={"Host": "evil.example.com:7860"}
        )
        assert resp.status_code == 403
        assert resp.json()["detail"] == "Host not allowed"

    def test_foreign_host_bearer_post_is_refused(self):
        resp = _client().post(
            "/api/auth/ticket",
            headers={
                "Host": "evil.example.com:7860",
                "Authorization": f"Bearer {get_auth_token()}",
            },
        )
        assert resp.status_code == 403

    def test_foreign_host_cookie_post_is_refused(self):
        client = _client()
        assert client.get(f"/?token={get_auth_token()}", follow_redirects=False).status_code == 303
        resp = client.post(
            "/api/auth/ticket",
            headers={
                "Host": "evil.example.com:7860",
                "Origin": "http://evil.example.com:7860",
            },
        )
        assert resp.status_code == 403
        assert resp.json()["detail"] == "Host not allowed"

    def test_right_name_with_the_wrong_port_is_refused(self):
        resp = _client().get("/api/train/status", headers={"Host": "127.0.0.1:9999"})
        assert resp.status_code == 403

    def test_host_without_a_port_is_refused(self):
        resp = _client().get("/api/train/status", headers={"Host": "127.0.0.1"})
        assert resp.status_code == 403

    def test_empty_host_is_refused(self):
        resp = _client().get("/api/train/status", headers={"Host": ""})
        assert resp.status_code == 403

    def test_missing_host_is_refused(self):
        # A raw ASGI request can omit the header entirely.
        import asyncio

        client = _client()
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "scheme": "http",
            "method": "GET",
            "path": "/api/train/status",
            "raw_path": b"/api/train/status",
            "query_string": b"",
            "root_path": "",
            "headers": [(b"accept", b"*/*")],
            "client": ("127.0.0.1", 54321),
            "server": ("127.0.0.1", PORT),
        }
        seen = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            seen.append(message)

        async def call():
            await client.app(scope, receive, send)
            return next(
                m["status"] for m in seen if m.get("type") == "http.response.start"
            )

        assert asyncio.run(call()) == 403

    def test_ip_literal_spelling_normalization(self):
        # `[127.0.0.1]` normalizes to the same name and is served.
        resp = _client().get("/", headers={"Host": f"[127.0.0.1]:{PORT}"})
        assert resp.status_code == 200


class TestPublicBindUnchanged:
    def test_public_bind_does_not_install_the_allowlist(self):
        client = TestClient(
            create_app(host="192.168.1.5", port=PORT),
            base_url="http://192.168.1.5:7860",
        )
        resp = client.get(
            "/api/train/status",
            headers={"Host": "anything.example.com:7860", **_bearer_headers()},
        )
        assert resp.status_code != 403
