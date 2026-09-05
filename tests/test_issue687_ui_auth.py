"""Tests for Issue #687: Web UI authentication on read APIs and SSE streams."""

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from soup_cli.cli import app as cli_app
from soup_cli.ui.app import create_app, get_auth_token


def _auth_headers():
    return {"Authorization": f"Bearer {get_auth_token()}"}


class TestReadAuthentication:
    """Verify that private read endpoints require authentication."""

    @pytest.mark.parametrize(
        "endpoint",
        [
            "/api/runs",
            "/api/runs/compare?ids=dummy",
            "/api/runs/dummy_id",
            "/api/runs/dummy_id/metrics",
            "/api/runs/dummy_id/eval",
            "/api/system",
            "/api/templates",
            "/api/train/status",
            "/api/train/progress",
            "/api/config/schema",
            "/api/recipes",
            "/api/tool-outputs",
        ],
    )
    def test_unauthenticated_reads_return_401(self, endpoint):
        """Unauthenticated requests to read APIs must return 401."""
        client = TestClient(create_app())
        resp = client.get(endpoint)
        assert resp.status_code == 401, f"{endpoint} allowed unauthenticated access"

    @pytest.mark.parametrize(
        "endpoint",
        [
            "/api/runs",
            "/api/system",
            "/api/templates",
            "/api/train/status",
            "/api/config/schema",
            "/api/recipes",
            "/api/tool-outputs",
        ],
    )
    def test_bad_token_reads_return_401(self, endpoint):
        """Requests with an invalid Bearer token must return 401."""
        client = TestClient(create_app())
        resp = client.get(endpoint, headers={"Authorization": "Bearer invalid_secret_token"})
        assert resp.status_code == 401

    def test_authenticated_reads_succeed(self):
        """Authenticated requests with valid Bearer token return 200."""
        client = TestClient(create_app())
        headers = _auth_headers()

        resp = client.get("/api/system", headers=headers)
        assert resp.status_code == 200

        resp = client.get("/api/templates", headers=headers)
        assert resp.status_code == 200

        resp = client.get("/api/train/status", headers=headers)
        assert resp.status_code == 200

        resp = client.get("/api/config/schema", headers=headers)
        assert resp.status_code == 200

        resp = client.get("/api/recipes", headers=headers)
        assert resp.status_code == 200

        resp = client.get("/api/tool-outputs", headers=headers)
        assert resp.status_code == 200


class TestPublicEndpoints:
    """Verify that public endpoints remain reachable without authentication."""

    def test_index_is_public(self):
        client = TestClient(create_app())
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_health_is_public(self):
        client = TestClient(create_app())
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_static_files_are_public(self):
        client = TestClient(create_app())
        resp = client.get("/static/app.js")
        assert resp.status_code == 200


class TestSSEAuthenticationAndTickets:
    """Verify SSE streaming authentication via single-use tickets and headers."""

    @pytest.mark.parametrize(
        "endpoint",
        [
            "/api/train/logs",
            "/api/train/metrics/live",
            "/api/train/stream",
        ],
    )
    def test_unauthenticated_sse_returns_401(self, endpoint):
        """SSE endpoints without auth must return 401."""
        client = TestClient(create_app())
        resp = client.get(endpoint)
        assert resp.status_code == 401

    def test_bearer_header_on_sse_succeeds(self):
        """SSE endpoints accept direct Authorization Bearer header."""
        client = TestClient(create_app())
        resp = client.get("/api/train/logs", headers=_auth_headers())
        assert resp.status_code == 200

    def test_durable_token_in_query_rejected(self):
        """?token= query param must NOT be accepted on SSE endpoints (#687)."""
        client = TestClient(create_app())
        token = get_auth_token()
        resp = client.get(f"/api/train/logs?token={token}")
        assert resp.status_code == 401

    def test_auth_ticket_exchange_and_single_use(self):
        """POST /api/auth/ticket issues ticket, ticket is single-use only."""
        client = TestClient(create_app())

        # Unauthenticated ticket request fails
        resp = client.post("/api/auth/ticket")
        assert resp.status_code == 401

        # Authenticated ticket request succeeds
        resp = client.post("/api/auth/ticket", headers=_auth_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert "ticket" in data
        ticket = data["ticket"]
        assert len(ticket) > 20

        # First connection with ticket succeeds
        resp_sse = client.get(f"/api/train/logs?ticket={ticket}")
        assert resp_sse.status_code == 200

        # Reusing the same ticket fails with 401 (single use)
        resp_reused = client.get(f"/api/train/logs?ticket={ticket}")
        assert resp_reused.status_code == 401

    def test_invalid_ticket_rejected(self):
        """Nonexistent ticket fails with 401."""
        client = TestClient(create_app())
        resp = client.get("/api/train/logs?ticket=nonexistent-ticket-value")
        assert resp.status_code == 401


class TestNonLoopbackBindingProtection:
    """Verify that binding to non-loopback with auth disabled refuses startup."""

    def test_create_app_refuses_non_loopback_with_no_auth(self):
        """create_app raises ValueError on non-loopback host if no_auth=True."""
        with pytest.raises(ValueError, match="Binding non-loopback host"):
            create_app(host="0.0.0.0", no_auth=True)

        with pytest.raises(ValueError, match="Binding non-loopback host"):
            create_app(host="192.168.1.15", no_auth=True)

    def test_create_app_allows_loopback_with_no_auth(self):
        """create_app allows loopback hosts when no_auth=True."""
        app = create_app(host="127.0.0.1", no_auth=True)
        client = TestClient(app)
        # Auth not required
        resp = client.get("/api/system")
        assert resp.status_code == 200

    def test_cli_ui_refuses_startup_on_non_loopback_with_no_auth(self):
        """soup ui --host 0.0.0.0 --no-auth exits with code 2."""
        runner = CliRunner()
        result = runner.invoke(cli_app, ["ui", "--host", "0.0.0.0", "--no-auth"])
        assert result.exit_code == 2
        assert "binding non-loopback host" in result.output.lower()

    def test_cli_ui_refuses_public_with_no_auth(self):
        """soup ui --public --no-auth exits with code 2."""
        runner = CliRunner()
        result = runner.invoke(cli_app, ["ui", "--public", "--no-auth"])
        assert result.exit_code == 2
        assert "binding non-loopback host" in result.output.lower()
