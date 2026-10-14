"""Build Web UI test clients using the address served by create_app."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


def ui_client(app=None, *, host: str = "127.0.0.1", port: int = 7860, **kwargs) -> TestClient:
    """Use the served address; host/port must match a supplied app's bind."""
    from fastapi.testclient import TestClient

    from soup_cli.ui.app import create_app

    if app is None:
        app = create_app(host=host, port=port)
    url_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return TestClient(app, base_url=f"http://{url_host}:{port}", **kwargs)
