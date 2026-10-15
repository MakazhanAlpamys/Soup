"""Build Web UI test clients using the address served by create_app."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


def ui_client(app=None, *, host: str = "127.0.0.1", port: int = 7860, **kwargs) -> TestClient:
    """Use the served address; IPv6 binds need explicit Host headers on requests."""
    from fastapi.testclient import TestClient

    from soup_cli.ui.app import create_app

    if app is None:
        app = create_app(host=host, port=port)
    # Starlette 1.6 TestClient cannot parse an IPv6 netloc.
    url_host = "127.0.0.1" if ":" in host else host
    return TestClient(app, base_url=f"http://{url_host}:{port}", **kwargs)
