"""Tests for issue #2194: /v1/tools/python diagnostics and 401 WWW-Authenticate header.

Acceptance criteria:
- Tests for a syntax error, a raised exception, a non-zero sys.exit, an output overflow
  and a timeout, each asserting its own timed_out, exit_code and stderr.
- The 401 tests assert the WWW-Authenticate: Bearer header.
- Removing the header fails the 401 test.
- The tests fake the sandbox so CI on any OS runs them.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from soup_cli.commands.serve import _create_app
from soup_cli.trainer.rewards import SandboxProcessResult

TOKEN = "valid-test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _make_client(host: str = "127.0.0.1", auth_token: str | None = TOKEN) -> TestClient:
    app = _create_app(
        model_obj=MagicMock(),
        tokenizer=MagicMock(),
        device="cpu",
        model_name="test-model",
        max_tokens_default=64,
        host=host,
        auth_token=auth_token,
    )
    return TestClient(app, base_url=f"http://{host}:8000")


@pytest.fixture
def fake_isolation(monkeypatch):
    """Enable namespaces strategy so /v1/tools/python doesn't 501 on OS check."""
    from soup_cli.trainer import rewards

    monkeypatch.setattr(rewards, "_get_isolation_strategy", lambda: "namespaces")


class TestPythonToolDiagnostics:
    def test_syntax_error_returns_nonzero_exit_code_and_stderr(self, monkeypatch, fake_isolation):
        from soup_cli.trainer import rewards

        def _fake_sandbox(code: str, strict_namespaces: bool = False) -> SandboxProcessResult:
            err = (
                '  File "<string>", line 1\n'
                "    def foo(\n"
                "           ^\n"
                "SyntaxError: was never closed"
            )
            return SandboxProcessResult(
                returncode=1,
                stdout="",
                stderr=err,
                timed_out=False,
            )

        monkeypatch.setattr(rewards, "_run_code_sandbox_process", _fake_sandbox)

        client = _make_client()
        resp = client.post("/v1/tools/python", json={"code": "def foo("}, headers=AUTH)

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["timed_out"] is False
        assert data["exit_code"] == 1
        assert "SyntaxError" in data["stderr"]
        assert data["stdout"] == ""

    def test_raised_exception_returns_traceback_in_stderr(self, monkeypatch, fake_isolation):
        from soup_cli.trainer import rewards

        def _fake_sandbox(code: str, strict_namespaces: bool = False) -> SandboxProcessResult:
            return SandboxProcessResult(
                returncode=1,
                stdout="",
                stderr="Traceback (most recent call last):\nValueError: custom runtime failure",
                timed_out=False,
            )

        monkeypatch.setattr(rewards, "_run_code_sandbox_process", _fake_sandbox)

        client = _make_client()
        resp = client.post(
            "/v1/tools/python",
            json={"code": 'raise ValueError("custom runtime failure")'},
            headers=AUTH,
        )

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["timed_out"] is False
        assert data["exit_code"] == 1
        assert "ValueError: custom runtime failure" in data["stderr"]
        assert data["stdout"] == ""

    def test_nonzero_sys_exit_preserves_exit_code(self, monkeypatch, fake_isolation):
        from soup_cli.trainer import rewards

        def _fake_sandbox(code: str, strict_namespaces: bool = False) -> SandboxProcessResult:
            return SandboxProcessResult(
                returncode=42,
                stdout="exiting early\n",
                stderr="",
                timed_out=False,
            )

        monkeypatch.setattr(rewards, "_run_code_sandbox_process", _fake_sandbox)

        client = _make_client()
        resp = client.post(
            "/v1/tools/python",
            json={"code": "import sys; print('exiting early'); sys.exit(42)"},
            headers=AUTH,
        )

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["timed_out"] is False
        assert data["exit_code"] == 42
        assert data["stdout"] == "exiting early\n"
        assert data["stderr"] == ""

    def test_output_overflow_reports_overflow_and_not_timed_out(self, monkeypatch, fake_isolation):
        from soup_cli.trainer import rewards

        def _fake_sandbox(code: str, strict_namespaces: bool = False) -> SandboxProcessResult:
            return SandboxProcessResult(
                returncode=1,
                stdout="",
                stderr="sandbox output exceeded limit",
                output_exceeded=True,
                timed_out=False,
            )

        monkeypatch.setattr(rewards, "_run_code_sandbox_process", _fake_sandbox)

        client = _make_client()
        resp = client.post(
            "/v1/tools/python",
            json={"code": "print('x' * 1000000)"},
            headers=AUTH,
        )

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["timed_out"] is False
        assert data["exit_code"] == 1
        assert data["stderr"] == "sandbox output exceeded limit"
        assert data["stdout"] == ""

    def test_real_timeout_reports_timed_out_flag(self, monkeypatch, fake_isolation):
        from soup_cli.trainer import rewards

        def _fake_sandbox(code: str, strict_namespaces: bool = False) -> SandboxProcessResult:
            return SandboxProcessResult(
                returncode=None,
                stdout="",
                stderr="",
                timed_out=True,
            )

        monkeypatch.setattr(rewards, "_run_code_sandbox_process", _fake_sandbox)

        client = _make_client()
        resp = client.post(
            "/v1/tools/python",
            json={"code": "import time; time.sleep(10)"},
            headers=AUTH,
        )

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["timed_out"] is True
        assert data["exit_code"] == 124
        assert data["stdout"] == ""
        assert data["stderr"] == ""

    def test_real_sandbox_returns_syntax_error_when_available(self):
        from soup_cli.trainer.rewards import _get_isolation_strategy

        if _get_isolation_strategy() == "best-effort":
            pytest.skip("Host does not provide OS-level sandbox isolation")

        client = _make_client()
        resp = client.post("/v1/tools/python", json={"code": "def foo("}, headers=AUTH)
        if resp.status_code == 501:
            pytest.skip("OS isolation cannot be initialized in this test environment")

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["timed_out"] is False
        assert data["exit_code"] != 0
        assert "SyntaxError" in data["stderr"]


class TestWWWAuthenticateHeaderOn401:
    def test_missing_token_on_loopback_code_tools_sends_www_authenticate(self):
        client = _make_client(host="127.0.0.1", auth_token=None)
        resp = client.post("/v1/tools/python", json={"code": "print(1)"})

        assert resp.status_code == 401
        assert resp.headers.get("www-authenticate") == "Bearer"

    def test_invalid_bearer_token_sends_www_authenticate(self):
        client = _make_client(host="127.0.0.1", auth_token=TOKEN)
        resp = client.post(
            "/v1/tools/python",
            json={"code": "print(1)"},
            headers={"Authorization": "Bearer wrong-token"},
        )

        assert resp.status_code == 401
        assert resp.headers.get("www-authenticate") == "Bearer"

    def test_non_loopback_without_auth_sends_www_authenticate(self):
        client = _make_client(host="0.0.0.0", auth_token=None)
        resp = client.post(
            "/v1/tools/python",
            json={"code": "print(1)"},
        )

        assert resp.status_code == 401
        assert resp.headers.get("www-authenticate") == "Bearer"

    def test_bash_route_also_sends_www_authenticate_on_401(self):
        client = _make_client(host="127.0.0.1", auth_token=TOKEN)
        resp = client.post(
            "/v1/tools/bash",
            json={"command": "echo 1"},
            headers={"Authorization": "Bearer wrong-token"},
        )

        assert resp.status_code == 401
        assert resp.headers.get("www-authenticate") == "Bearer"
