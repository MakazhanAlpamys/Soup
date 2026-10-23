"""soup serve: tool and adapter routes check Host and Origin."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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


TOKEN = "s3cret-token"

# The three adapter routes: enumerating the loaded adapters and switching the
# one every subsequent completion runs against. On a non-loopback bind these
# carry no browser Origin (curl, requests, any script), so the Host/Origin
# guard passes and the bearer token is the only thing left standing.
ADAPTER_ROUTES = [
    ("get", "/v1/adapters"),
    ("post", "/v1/adapters/activate/chat"),
    ("post", "/v1/adapters/deactivate"),
]


def _lan_client(**kwargs):
    """A server bound to 0.0.0.0, reached over the LAN by a non-browser client."""
    return TestClient(_app(host="0.0.0.0", **kwargs), base_url="http://10.0.0.7:8000")


@pytest.mark.parametrize(("method", "path"), ADAPTER_ROUTES)
def test_adapter_routes_refuse_missing_token_on_wildcard_bind(method, path):
    client = _lan_client(auth_token=TOKEN)
    resp = getattr(client, method)(path)
    assert resp.status_code == 401, (path, resp.status_code, resp.text)


@pytest.mark.parametrize(("method", "path"), ADAPTER_ROUTES)
def test_adapter_routes_refuse_wrong_token_on_wildcard_bind(method, path):
    client = _lan_client(auth_token=TOKEN)
    resp = getattr(client, method)(path, headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401, (path, resp.status_code, resp.text)


@pytest.mark.parametrize(("method", "path"), ADAPTER_ROUTES)
def test_adapter_routes_accept_correct_token_on_wildcard_bind(method, path):
    client = _lan_client(auth_token=TOKEN)
    resp = getattr(client, method)(path, headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 200, (path, resp.status_code, resp.text)


def test_authorised_activation_on_wildcard_bind_takes_effect():
    client = _lan_client(auth_token=TOKEN)
    auth = {"Authorization": f"Bearer {TOKEN}"}
    resp = client.post("/v1/adapters/activate/chat", headers=auth)
    assert resp.status_code == 200, resp.text
    assert resp.json()["active"] == "chat", resp.text

    listing = client.get("/v1/adapters", headers=auth)
    assert listing.status_code == 200, listing.text
    assert listing.json()["active"] == "chat", listing.text

    cleared = client.post("/v1/adapters/deactivate", headers=auth)
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["active"] is None, cleared.text


def test_unauthorised_activation_on_wildcard_bind_does_not_take_effect():
    client = _lan_client(auth_token=TOKEN)
    refused = client.post("/v1/adapters/activate/chat")
    assert refused.status_code == 401, refused.text

    listing = client.get("/v1/adapters", headers={"Authorization": f"Bearer {TOKEN}"})
    assert listing.status_code == 200, listing.text
    assert listing.json()["active"] is None, listing.text


@pytest.mark.parametrize(("method", "path"), ADAPTER_ROUTES)
def test_adapter_routes_open_on_loopback_without_token(method, path):
    """The default `soup serve` case: loopback bind, no --tool-auth-token."""
    client = TestClient(_app(), base_url="http://127.0.0.1:8000")
    resp = getattr(client, method)(path)
    assert resp.status_code == 200, (path, resp.status_code, resp.text)


def test_token_compare_is_constant_time():
    import inspect

    from soup_cli.commands import serve

    src = inspect.getsource(serve._create_app)
    assert "compare_digest" in src
    assert "authorization != expected" not in src


# ----------------------------------------------------------------------
# The generation routes: Origin-only, because a reverse proxy legitimately
# fronts them under its own hostname while a browser always sends Origin on
# a cross-site request.
# ----------------------------------------------------------------------


class TestCheckBrowserOrigin:
    """Unit level: the narrow decision function itself."""

    @pytest.mark.parametrize("bind", ["127.0.0.1", "localhost", "::1", "0.0.0.0", "192.168.1.5"])
    def test_no_origin_is_allowed_on_any_bind(self, bind):
        from soup_cli.utils.local_request_guard import check_browser_origin

        assert check_browser_origin(bind, "proxy.example:8000", None) is None

    @pytest.mark.parametrize("host", ["proxy.example:443", "evil.example", "", None])
    def test_host_alone_never_refuses(self, host):
        """Deliberately NOT a Host check — that is what would break a proxy."""
        from soup_cli.utils.local_request_guard import check_browser_origin

        assert check_browser_origin("127.0.0.1", host, None) is None

    @pytest.mark.parametrize(
        "origin",
        ["http://evil.example", "null", "file://x", "http://127.0.0.1.evil.example"],
    )
    def test_foreign_origin_403(self, origin):
        from soup_cli.utils.local_request_guard import check_browser_origin

        assert check_browser_origin("127.0.0.1", "127.0.0.1:8000", origin) == (
            403,
            "Origin not allowed",
        )

    def test_loopback_origin_allowed(self):
        from soup_cli.utils.local_request_guard import check_browser_origin

        assert check_browser_origin("127.0.0.1", "127.0.0.1:8000", "http://localhost:5173") is None

    def test_wildcard_bind_compares_origin_to_host(self):
        from soup_cli.utils.local_request_guard import check_browser_origin

        assert check_browser_origin("0.0.0.0", "10.0.0.2:8000", "http://10.0.0.2:8000") is None
        assert check_browser_origin("0.0.0.0", "10.0.0.2:8000", "http://evil.example") == (
            403,
            "Origin not allowed",
        )


CHAT_BODY = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}
MESSAGES_BODY = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 16,
}
GENERATION = [
    ("/v1/chat/completions", CHAT_BODY),
    ("/v1/messages", MESSAGES_BODY),
]


def _generation_client(base_url="http://127.0.0.1:8000", **kwargs):
    return TestClient(_app(**kwargs), base_url=base_url)


@pytest.mark.parametrize(("path", "body"), GENERATION)
def test_generation_routes_refuse_foreign_origin(path, body):
    """A page the operator merely visits must not be able to drive generation."""
    with patch("soup_cli.commands.serve._generate_response") as mock_gen:
        mock_gen.return_value = ("hello world", 3, 2)
        resp = _generation_client().post(
            path, json=body, headers={"Origin": "http://evil.example"}
        )
    assert resp.status_code == 403, (path, resp.status_code, resp.text)
    assert mock_gen.call_count == 0, f"{path} generated before the Origin was checked"


@pytest.mark.parametrize(("path", "body"), GENERATION)
def test_generation_routes_allow_loopback_origin(path, body):
    with patch("soup_cli.commands.serve._generate_response") as mock_gen:
        mock_gen.return_value = ("hello world", 3, 2)
        resp = _generation_client().post(
            path, json=body, headers={"Origin": "http://localhost:5173"}
        )
    assert resp.status_code == 200, (path, resp.status_code, resp.text)


@pytest.mark.parametrize(("path", "body"), GENERATION)
def test_generation_routes_allow_missing_origin(path, body):
    with patch("soup_cli.commands.serve._generate_response") as mock_gen:
        mock_gen.return_value = ("hello world", 3, 2)
        resp = _generation_client().post(path, json=body)
    assert resp.status_code == 200, (path, resp.status_code, resp.text)


@pytest.mark.parametrize(("path", "body"), GENERATION)
def test_generation_routes_allow_reverse_proxy_shape(path, body):
    """Foreign Host, no Origin: a reverse proxy in front of the server.

    This is the property the Origin-only check exists to preserve — a Host
    check here would refuse every proxied deployment.
    """
    with patch("soup_cli.commands.serve._generate_response") as mock_gen:
        mock_gen.return_value = ("hello world", 3, 2)
        resp = _generation_client(base_url="http://proxy.example").post(path, json=body)
    assert resp.status_code == 200, (path, resp.status_code, resp.text)


def test_adapter_listing_refuses_foreign_origin_on_tokenless_loopback():
    client = TestClient(_app(), base_url="http://127.0.0.1:8000")
    resp = client.get("/v1/adapters", headers={"Origin": "http://evil.example"})
    assert resp.status_code == 403, resp.text


def test_adapter_listing_without_origin_still_works():
    client = TestClient(_app(), base_url="http://127.0.0.1:8000")
    resp = client.get("/v1/adapters")
    assert resp.status_code == 200, resp.text
    assert resp.json()["adapters"][0]["name"] == "chat"


def test_adapter_listing_not_host_checked():
    """The adapters route gains Origin, not Host — the proxy shape survives."""
    client = TestClient(_app(), base_url="http://proxy.example")
    assert client.get("/v1/adapters").status_code == 200


@pytest.mark.parametrize("path", ["/health", "/metrics", "/v1/models"])
def test_unguarded_routes_stay_unguarded(path):
    """These carry no generation cost and no adapter names."""
    client = TestClient(_app(), base_url="http://proxy.example")
    resp = client.get(path, headers={"Origin": "http://evil.example"})
    assert resp.status_code == 200, (path, resp.status_code, resp.text)
