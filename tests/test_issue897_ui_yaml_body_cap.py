"""Regression tests for issue #897: cap Web UI YAML request bodies."""

import asyncio
from collections.abc import Iterator

import pytest

from soup_cli.recipes.catalog import RECIPES


@pytest.fixture
def ui_client() -> Iterator[tuple[object, dict[str, str]]]:
    """Yield an authenticated Web UI test client."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from soup_cli.ui.app import create_app, get_auth_token

    with TestClient(create_app()) as client:
        yield client, {"Authorization": f"Bearer {get_auth_token()}"}


@pytest.mark.parametrize(
    ("endpoint", "payload"),
    [
        ("/api/config/validate", {"yaml": "x" * (2 * 1024 * 1024)}),
        ("/api/train/start", {"config_yaml": "x" * (2 * 1024 * 1024)}),
        (
            "/api/config/from-form",
            {"base": "x" * (2 * 1024 * 1024), "data": {"train": "data.jsonl"}},
        ),
    ],
)
def test_oversized_yaml_request_is_413_before_safe_load(
    ui_client: tuple[object, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
    payload: dict[str, object],
) -> None:
    """All YAML entry points must stop before YAML parsing is reached."""
    from soup_cli.config import loader

    called = False

    def fail_if_called(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("yaml.safe_load must not parse an oversized request")

    monkeypatch.setattr(loader.yaml, "safe_load", fail_if_called)
    client, headers = ui_client

    response = client.post(endpoint, json=payload, headers=headers)  # type: ignore[attr-defined]

    assert response.status_code == 413, response.text
    assert response.json() == {"detail": "Request body too large"}
    assert called is False


def test_body_cap_counts_chunks_when_content_length_is_understated() -> None:
    """The actual ASGI bytes, not a client-controlled header, enforce the cap."""
    from soup_cli.ui.app import _RequestBodySizeLimitMiddleware

    sent: list[dict[str, object]] = []
    downstream_called = False
    chunks = iter([
        {"type": "http.request", "body": b"1234", "more_body": True},
        {"type": "http.request", "body": b"5678", "more_body": False},
    ])

    async def receive() -> dict[str, object]:
        return next(chunks)

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    async def downstream(scope, receive_body, send_response) -> None:
        nonlocal downstream_called
        downstream_called = True

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/config/validate",
        "headers": [(b"content-length", b"1")],
    }
    middleware = _RequestBodySizeLimitMiddleware(
        downstream,
        limits={"/api/config/validate": 5},
    )

    asyncio.run(middleware(scope, receive, send))

    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 413
    assert downstream_called is False


@pytest.mark.parametrize(
    "endpoint",
    [
        "/api/config/validate",
        "/api/train/start",
        "/api/config/from-form",
    ],
)
def test_yaml_body_cap_runs_before_json_parsing(
    ui_client: tuple[object, dict[str, str]],
    endpoint: str,
) -> None:
    """The same malformed JSON is parsed below the cap and refused above it."""
    client, auth_headers = ui_client
    headers = {**auth_headers, "Content-Type": "application/json"}
    small_invalid_json = b'{"unterminated":"' + (b"x" * 48)
    oversized_invalid_json = b'{"unterminated":"' + (b"x" * (1024 * 1024))

    small_response = client.post(  # type: ignore[attr-defined]
        endpoint,
        content=small_invalid_json,
        headers=headers,
    )
    oversized_response = client.post(  # type: ignore[attr-defined]
        endpoint,
        content=oversized_invalid_json,
        headers=headers,
    )

    assert small_response.status_code == 422
    assert oversized_response.status_code == 413


@pytest.mark.parametrize(
    "endpoint",
    [
        "/api/config/validate",
        "/api/train/start",
        "/api/config/from-form",
    ],
)
def test_yaml_routes_keep_the_1mib_ceiling_not_the_8kib_one(
    ui_client: tuple[object, dict[str, str]],
    endpoint: str,
) -> None:
    """Bodies well above 8 KiB but below 1 MiB must still reach parsing."""
    client, auth_headers = ui_client
    body = b'{"unterminated":"' + (b"x" * (900 * 1024))

    response = client.post(  # type: ignore[attr-defined]
        endpoint,
        content=body,
        headers={**auth_headers, "Content-Type": "application/json"},
    )

    assert response.status_code == 422


def test_oversized_content_length_rejects_without_reading_body() -> None:
    """The header fast path must stop before the first receive call."""
    from soup_cli.ui.app import _RequestBodySizeLimitMiddleware

    sent: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        raise AssertionError("body was read despite oversized Content-Length")

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    async def downstream(scope, receive_body, send_response) -> None:
        raise AssertionError("downstream app was called")

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/config/validate",
        "headers": [(b"content-length", b"6")],
    }

    middleware = _RequestBodySizeLimitMiddleware(
        downstream,
        limits={"/api/config/validate": 5},
    )
    asyncio.run(middleware(scope, receive, send))

    assert sent[0]["status"] == 413


@pytest.mark.parametrize(("body", "expected_status"), [(b"12345", 204), (b"123456", 413)])
def test_body_cap_exact_boundary(body: bytes, expected_status: int) -> None:
    """Exactly the configured limit is admitted; one byte more is refused."""
    from soup_cli.ui.app import _RequestBodySizeLimitMiddleware

    sent: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    async def downstream(scope, receive_body, send_response) -> None:
        await receive_body()
        await send_response({"type": "http.response.start", "status": 204})
        await send_response({"type": "http.response.body", "body": b""})

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/config/validate",
        "headers": [],
    }

    middleware = _RequestBodySizeLimitMiddleware(
        downstream,
        limits={"/api/config/validate": 5},
    )
    asyncio.run(middleware(scope, receive, send))

    assert sent[0]["status"] == expected_status


@pytest.mark.parametrize(("name", "recipe"), RECIPES.items(), ids=RECIPES)
def test_every_shipped_recipe_still_validates_through_web_ui(
    ui_client: tuple[object, dict[str, str]], name: str, recipe: object
) -> None:
    """The body ceiling must leave every catalog recipe usable."""
    client, headers = ui_client
    response = client.post(  # type: ignore[attr-defined]
        "/api/config/validate",
        json={"yaml": recipe.yaml_str},  # type: ignore[attr-defined]
        headers=headers,
    )

    assert response.status_code == 200, f"{name}: {response.text}"
    assert response.json()["valid"] is True, f"{name}: {response.json()}"
