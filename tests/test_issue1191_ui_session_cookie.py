"""#1191 — `soup ui` keeps the session across reloads with an HttpOnly cookie.

`GET /?token=<token>` checks the token, sets a random session id in an HttpOnly
cookie and redirects to the same URL without `token`. Cookie-authenticated requests
other than GET/HEAD must carry an `Origin` naming this exact server, because every
port of 127.0.0.1 is one site and SameSite=Strict does not stop another local app.
The Bearer header keeps working unchanged for scripted clients.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit

import pytest
from fastapi.testclient import TestClient

from soup_cli.ui import app as ui_app
from soup_cli.ui.app import create_app, get_auth_token

PORT = 7860
COOKIE = f"soup_ui_session_{PORT}"
# TestClient sends `Host: testserver` over http.
SAME_ORIGIN = "http://testserver"


def _client(host: str = "127.0.0.1") -> TestClient:
    return TestClient(create_app(host=host, port=PORT))


def _sign_in(client: TestClient, query: str = ""):
    return client.get(f"/?token={get_auth_token()}{query}", follow_redirects=False)


def _signed_in(host: str = "127.0.0.1") -> TestClient:
    client = _client(host)
    assert _sign_in(client).status_code == 303
    return client


def _set_cookie_headers(resp) -> list[str]:
    return resp.headers.get_list("set-cookie")


class TestExchange:
    def test_token_url_sets_a_session_cookie_and_redirects(self):
        resp = _sign_in(_client())
        assert resp.status_code == 303
        (header,) = _set_cookie_headers(resp)
        assert header.startswith(f"{COOKIE}=")

    def test_cookie_flags(self):
        (header,) = _set_cookie_headers(_sign_in(_client()))
        attrs = [part.strip().lower() for part in header.split(";")[1:]]
        assert "httponly" in attrs
        assert "samesite=strict" in attrs
        assert "path=/" in attrs
        # A session cookie: it ends with the browser session.
        assert not any(a.startswith(("max-age", "expires")) for a in attrs)
        # `soup ui` serves plain HTTP.
        assert "secure" not in attrs

    def test_cookie_value_is_a_random_id_not_the_token(self):
        client = _client()
        first = _sign_in(client).cookies[COOKIE]
        second = _sign_in(client).cookies[COOKIE]
        token = get_auth_token()
        assert token not in first and first not in token
        assert len(first) >= 43  # token_urlsafe(32)
        assert first != second

    def test_cookie_name_carries_the_port(self):
        client = TestClient(create_app(host="127.0.0.1", port=8123))
        resp = client.get(f"/?token={get_auth_token()}", follow_redirects=False)
        assert "soup_ui_session_8123" in resp.cookies

    def test_redirect_drops_token_and_keeps_other_params(self):
        resp = _sign_in(_client(), "&tab=chat&run=a&run=b")
        location = urlsplit(resp.headers["location"])
        assert location.path == "/"
        assert parse_qsl(location.query) == [("tab", "chat"), ("run", "a"), ("run", "b")]

    def test_redirect_without_other_params_has_no_query(self):
        assert _sign_in(_client()).headers["location"] == "/"

    @pytest.mark.parametrize("token", ["wrong-token", "", "tést"])
    def test_wrong_token_sets_nothing_and_serves_the_page(self, token):
        resp = _client().get(f"/?token={token}", follow_redirects=False)
        assert resp.status_code == 200
        assert "<html" in resp.text.lower()
        assert not _set_cookie_headers(resp)

    def test_plain_page_sets_nothing(self):
        resp = _client().get("/", follow_redirects=False)
        assert resp.status_code == 200
        assert not _set_cookie_headers(resp)


class TestReloadKeepsTheSession:
    def test_cookie_alone_authenticates_a_read(self):
        client = _signed_in()
        assert client.get("/api/train/status").status_code == 200

    def test_no_cookie_is_still_refused(self):
        assert _client().get("/api/train/status").status_code == 401

    def test_unknown_cookie_is_refused(self):
        client = _client()
        client.cookies.set(COOKIE, "not-a-live-session-id")
        assert client.get("/api/train/status").status_code == 401

    def test_cookie_opens_the_sse_stream(self):
        resp = _signed_in().get("/api/train/logs")
        assert resp.status_code == 200
        assert "event: done" in resp.text

    def test_cookie_can_still_fetch_an_sse_ticket(self):
        resp = _signed_in().post("/api/auth/ticket", headers={"Origin": SAME_ORIGIN})
        assert resp.status_code == 200
        assert resp.json()["ticket"]

    def test_restart_ends_every_session(self):
        old = _signed_in()
        session_id = old.cookies[COOKIE]
        restarted = _client()
        restarted.cookies.set(COOKIE, session_id)
        assert restarted.get("/api/train/status").status_code == 401

    def test_oldest_session_is_dropped_past_the_cap(self, monkeypatch):
        monkeypatch.setattr(ui_app, "_MAX_UI_SESSIONS", 2)
        client = _client()
        first = _sign_in(client).cookies[COOKIE]
        _sign_in(client)
        _sign_in(client)
        probe = TestClient(client.app)
        probe.cookies.set(COOKIE, first)
        assert probe.get("/api/train/status").status_code == 401


class TestOriginCheck:
    def test_same_origin_cookie_post_is_allowed(self):
        resp = _signed_in().post("/api/auth/ticket", headers={"Origin": SAME_ORIGIN})
        assert resp.status_code == 200

    @pytest.mark.parametrize(
        "origin",
        [
            "http://testserver:9999",  # another port of the same host
            "https://testserver",  # another scheme
            "http://evil.example",
            "null",
        ],
    )
    def test_cross_origin_cookie_post_is_refused(self, origin):
        resp = _signed_in().post("/api/auth/ticket", headers={"Origin": origin})
        assert resp.status_code == 403

    def test_cookie_post_without_origin_is_refused(self):
        assert _signed_in().post("/api/auth/ticket").status_code == 403

    def test_cookie_delete_without_origin_is_refused(self):
        assert _signed_in().delete("/api/runs/some-run").status_code == 403

    def test_bearer_post_without_origin_still_works(self):
        resp = _client().post(
            "/api/auth/ticket", headers={"Authorization": f"Bearer {get_auth_token()}"}
        )
        assert resp.status_code == 200

    def test_bearer_post_with_a_foreign_origin_still_works(self):
        # Scripted clients are unchanged: the Bearer header is not sent automatically.
        resp = _signed_in().post(
            "/api/auth/ticket",
            headers={
                "Authorization": f"Bearer {get_auth_token()}",
                "Origin": "http://testserver:9999",
            },
        )
        assert resp.status_code == 200


class TestPublicBind:
    def test_no_cookie_and_no_redirect_on_a_non_loopback_bind(self):
        # The phone page's script reads `?token=` itself, so it must stay in the URL.
        resp = _sign_in(_client("0.0.0.0"))
        assert resp.status_code == 200
        assert not _set_cookie_headers(resp)

    def test_cookie_is_not_accepted_on_a_non_loopback_bind(self):
        client = _client("0.0.0.0")
        client.cookies.set(COOKIE, "anything")
        assert client.get("/api/train/status").status_code == 401


class TestNoCredentialedCors:
    @pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0"])
    def test_no_allow_credentials_on_any_response(self, host):
        client = _signed_in() if host == "127.0.0.1" else _client(host)
        origin = f"http://127.0.0.1:{PORT}"
        responses = [
            client.get("/api/train/status", headers={"Origin": origin}),
            client.get("/api/runs/x", headers={"Origin": "http://127.0.0.1:1"}),
            client.options(
                "/api/auth/ticket",
                headers={"Origin": origin, "Access-Control-Request-Method": "POST"},
            ),
            _sign_in(client),
        ]
        for resp in responses:
            assert "access-control-allow-credentials" not in resp.headers

def _spy_compare_digest(monkeypatch) -> list:
    seen = []
    real = ui_app.secrets.compare_digest

    def spy(a, b):
        seen.append((a, b))
        return real(a, b)

    monkeypatch.setattr(ui_app.secrets, "compare_digest", spy)
    return seen


class TestConstantTimeComparisons:
    def test_exchange_compares_the_token_in_constant_time(self, monkeypatch):
        seen = _spy_compare_digest(monkeypatch)
        assert _sign_in(_client()).status_code == 303
        presented = f"Bearer {get_auth_token()}".encode("utf-8")
        assert any(a == presented and isinstance(b, bytes) for a, b in seen), seen

    def test_session_cookie_is_compared_in_constant_time(self, monkeypatch):
        client = _signed_in()
        session_id = client.cookies[COOKIE].encode("utf-8")
        seen = _spy_compare_digest(monkeypatch)
        assert client.get("/api/train/status").status_code == 200
        assert any(a == session_id and isinstance(b, bytes) for a, b in seen), seen


class TestRedirectStaysOnThisServer:
    def test_hostile_parameters_never_become_the_redirect_target(self):
        hostile = [
            ("next", "//evil.example/"),
            ("redirect", "http://evil.example/"),
            ("url", "\\\\evil.example"),
            ("x", "a\r\nSet-Cookie: pwn=1"),
        ]
        resp = _sign_in(_client(), "&" + urlencode(hostile))
        assert resp.status_code == 303
        parts = urlsplit(resp.headers["location"])
        assert parts.scheme == "" and parts.netloc == "", resp.headers["location"]
        assert parts.path == "/"
        assert parse_qsl(parts.query) == hostile
        assert all("pwn" not in c for c in _set_cookie_headers(resp))
