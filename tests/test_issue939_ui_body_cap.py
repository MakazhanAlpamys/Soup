"""Regression tests for issue #939: cap /api/chat/send and /api/data/inspect bodies."""

import asyncio

import pytest


def _auth_headers():
    """Return auth headers with the current UI token."""
    from soup_cli.ui.app import get_auth_token
    return {"Authorization": f"Bearer {get_auth_token()}"}


@pytest.mark.parametrize(
    ("endpoint", "payload"),
    [
        (
            "/api/chat/send",
            {
                "messages": [{"role": "user", "content": "x" * (2 * 1024 * 1024)}],
                "endpoint": "http://evil.com:8000",
            },
        ),
        (
            "/api/data/inspect",
            {"path": "/does/not/exist", "padding": "x" * (2 * 1024 * 1024)},
        ),
    ],
)
def test_oversized_body_is_413_before_route_runs(
    monkeypatch: pytest.MonkeyPatch, endpoint: str, payload: dict
) -> None:
    """Both routes must stop before their own validation is reached."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from soup_cli.ui.app import create_app

    called = False

    def fail_if_called(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("route validation must not run on an oversized body")

    if endpoint == "/api/chat/send":
        # A too-large body already fails the SSRF check (evil.com), so a 400
        # here would mean the cap did not fire first. chat_send imports
        # urlparse locally, so patch the source it re-imports from.
        monkeypatch.setattr("urllib.parse.urlparse", fail_if_called)
    else:
        from soup_cli.utils import paths

        monkeypatch.setattr(paths, "is_under_cwd", fail_if_called)

    with TestClient(create_app()) as client:
        response = client.post(endpoint, json=payload, headers=_auth_headers())

    assert response.status_code == 413, response.text
    assert response.json() == {"detail": "Request body too large"}
    assert called is False


@pytest.mark.parametrize(
    ("path", "body", "expected_status"),
    [
        ("/api/chat/send", b"12345", 204),
        ("/api/chat/send", b"123456", 413),
        ("/api/data/inspect", b"12345", 204),
        ("/api/data/inspect", b"123456", 413),
    ],
)
def test_body_cap_exact_boundary(path: str, body: bytes, expected_status: int) -> None:
    """Exactly the configured limit is admitted; one byte more is refused."""
    from soup_cli.ui.app import _RequestBodySizeLimitMiddleware

    sent: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict) -> None:
        sent.append(message)

    async def downstream(scope, receive_body, send_response) -> None:
        await receive_body()
        await send_response({"type": "http.response.start", "status": 204})
        await send_response({"type": "http.response.body", "body": b""})

    scope = {"type": "http", "method": "POST", "path": path, "headers": []}
    middleware = _RequestBodySizeLimitMiddleware(downstream, limits={path: 5})

    asyncio.run(middleware(scope, receive, send))

    assert sent[0]["status"] == expected_status


def test_oversized_content_length_rejects_without_reading_body() -> None:
    """The header fast path must stop before the first receive call."""
    from soup_cli.ui.app import _RequestBodySizeLimitMiddleware

    sent: list[dict] = []

    async def receive() -> dict:
        raise AssertionError("body was read despite oversized Content-Length")

    async def send(message: dict) -> None:
        sent.append(message)

    async def downstream(scope, receive_body, send_response) -> None:
        raise AssertionError("downstream app was called")

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/data/inspect",
        "headers": [(b"content-length", b"6")],
    }
    middleware = _RequestBodySizeLimitMiddleware(
        downstream, limits={"/api/data/inspect": 5}
    )

    asyncio.run(middleware(scope, receive, send))

    assert sent[0]["status"] == 413


def test_malformed_content_length_falls_through_to_stream_check() -> None:
    """A non-numeric header is not trusted; the byte-count check still applies."""
    from soup_cli.ui.app import _RequestBodySizeLimitMiddleware

    sent: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"123456", "more_body": False}

    async def send(message: dict) -> None:
        sent.append(message)

    async def downstream(scope, receive_body, send_response) -> None:
        raise AssertionError("downstream app was called")

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/data/inspect",
        "headers": [(b"content-length", b"not-a-number")],
    }
    middleware = _RequestBodySizeLimitMiddleware(
        downstream, limits={"/api/data/inspect": 5}
    )

    asyncio.run(middleware(scope, receive, send))

    assert sent[0]["status"] == 413


def test_body_cap_counts_chunks_when_content_length_is_understated() -> None:
    """The actual ASGI bytes, not a client-controlled header, enforce the cap."""
    from soup_cli.ui.app import _RequestBodySizeLimitMiddleware

    sent: list[dict] = []
    downstream_called = False
    chunks = iter([
        {"type": "http.request", "body": b"1234", "more_body": True},
        {"type": "http.request", "body": b"5678", "more_body": False},
    ])

    async def receive() -> dict:
        return next(chunks)

    async def send(message: dict) -> None:
        sent.append(message)

    async def downstream(scope, receive_body, send_response) -> None:
        nonlocal downstream_called
        downstream_called = True

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/chat/send",
        "headers": [(b"content-length", b"1")],
    }
    middleware = _RequestBodySizeLimitMiddleware(
        downstream, limits={"/api/chat/send": 5}
    )

    asyncio.run(middleware(scope, receive, send))

    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 413
    assert downstream_called is False


def test_path_outside_the_table_is_not_capped() -> None:
    """A route with no entry in the limits table passes straight through."""
    from soup_cli.ui.app import _RequestBodySizeLimitMiddleware

    downstream_called = False

    async def receive() -> dict:
        return {"type": "http.request", "body": b"x" * 100, "more_body": False}

    async def send(message: dict) -> None:
        pass

    async def downstream(scope, receive_body, send_response) -> None:
        nonlocal downstream_called
        downstream_called = True

    scope = {"type": "http", "method": "POST", "path": "/api/runs", "headers": []}
    middleware = _RequestBodySizeLimitMiddleware(downstream, limits={"/api/chat/send": 5})

    asyncio.run(middleware(scope, receive, send))

    assert downstream_called is True
