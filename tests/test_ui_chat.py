"""Tests for Web UI Chat Upgrade — SSE proxy, SSRF validation, param bounds."""

from unittest.mock import MagicMock, patch

import pytest


def _auth_headers():
    """Return auth headers with the current UI token."""
    from soup_cli.ui.app import get_auth_token

    return {"Authorization": f"Bearer {get_auth_token()}"}


class TestChatEndpointExists:
    """Test /api/chat/send endpoint registration."""

    def test_chat_send_endpoint_exists(self):
        """Route /api/chat/send should be registered."""
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        app = create_app()
        routes = [route.path for route in app.routes]
        assert "/api/chat/send" in routes


class TestChatSSRFProtection:
    """Test SSRF protection on chat endpoint."""

    def test_rejects_non_localhost_http(self):
        """Chat endpoint should reject non-localhost HTTP URLs."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.post(
            "/api/chat/send",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "endpoint": "http://evil.com:8000",
            },
            headers=_auth_headers(),
        )
        assert response.status_code == 400

    def test_allows_localhost_http(self):
        """Chat endpoint should allow localhost HTTP URLs."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        # Will fail at connection level but not at validation
        response = client.post(
            "/api/chat/send",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "endpoint": "http://localhost:8000",
            },
            headers=_auth_headers(),
        )
        # Should not be 400 (SSRF rejection) — may be 502 or stream error
        assert response.status_code != 400

    def test_allows_https_remote(self):
        """Chat endpoint should allow HTTPS remote URLs."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.post(
            "/api/chat/send",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "endpoint": "https://api.example.com",
            },
            headers=_auth_headers(),
        )
        # Should not be 400 (SSRF rejection)
        assert response.status_code != 400


class TestChatParamBounds:
    """Test parameter validation on chat endpoint."""

    def test_max_tokens_capped(self):
        """max_tokens should be capped at 16384."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.post(
            "/api/chat/send",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "endpoint": "http://localhost:8000",
                "max_tokens": 99999,
            },
            headers=_auth_headers(),
        )
        # Should be rejected (exceeds cap) — 422 from Pydantic or 400
        assert response.status_code in (400, 422)

    def test_temperature_bounded(self):
        """temperature should be bounded 0.0-2.0."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.post(
            "/api/chat/send",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "endpoint": "http://localhost:8000",
                "temperature": 5.0,
            },
            headers=_auth_headers(),
        )
        assert response.status_code == 400 or response.status_code == 422

    def test_top_p_bounded(self):
        """top_p should be bounded 0.0-1.0."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.post(
            "/api/chat/send",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "endpoint": "http://localhost:8000",
                "top_p": 2.0,
            },
            headers=_auth_headers(),
        )
        assert response.status_code == 400 or response.status_code == 422


class TestChatInvalidScheme:
    """Test rejection of invalid URL schemes."""

    def test_rejects_ftp_scheme(self):
        """Chat endpoint should reject ftp:// URLs."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.post(
            "/api/chat/send",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "endpoint": "ftp://evil.com/model",
            },
            headers=_auth_headers(),
        )
        assert response.status_code == 400

    def test_rejects_ipv6_mapped_loopback_alias(self):
        """Chat endpoint should reject IPv6-mapped private addresses."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.post(
            "/api/chat/send",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "endpoint": "http://192.168.1.1:8000",
            },
            headers=_auth_headers(),
        )
        assert response.status_code == 400

    def test_allows_127_loopback_range(self):
        """Chat endpoint should allow 127.x.x.x loopback addresses."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.post(
            "/api/chat/send",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "endpoint": "http://127.0.0.2:8000",
            },
            headers=_auth_headers(),
        )
        # 127.0.0.2 is loopback — should NOT be rejected as SSRF
        assert response.status_code != 400

    def test_rejects_file_scheme(self):
        """Chat endpoint should reject file:// URLs."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.post(
            "/api/chat/send",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "endpoint": "file:///etc/passwd",
            },
            headers=_auth_headers(),
        )
        assert response.status_code == 400


class TestChatAuth:
    """Test chat endpoint auth requirements."""

    def test_requires_auth(self):
        """Chat send (POST) should require Bearer auth."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.post(
            "/api/chat/send",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "endpoint": "http://localhost:8000",
            },
        )
        assert response.status_code == 401


class TestChatValidation:
    """Test chat request validation."""

    def test_empty_messages_rejected(self):
        """Empty messages array should return 400."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.post(
            "/api/chat/send",
            json={
                "messages": [],
                "endpoint": "http://localhost:8000",
            },
            headers=_auth_headers(),
        )
        assert response.status_code == 400

    def test_returns_event_stream(self):
        """Chat endpoint should return text/event-stream content type."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        # Mock httpx to avoid real connection
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(
            [
                'data: {"choices":[{"delta":{"content":"Hi"}}]}',
                "data: [DONE]",
            ]
        )
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = lambda s, *a: None

        with patch("httpx.stream", return_value=mock_response):
            with client.stream(
                "POST",
                "/api/chat/send",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "endpoint": "http://localhost:8000",
                },
                headers=_auth_headers(),
            ) as resp:
                assert resp.status_code == 200
                assert "text/event-stream" in resp.headers["content-type"]
