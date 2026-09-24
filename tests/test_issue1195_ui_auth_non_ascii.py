"""Tests for Issue #1195: Non-ASCII Authorization header crash in soup ui.

Starlette decodes header bytes >= 0x80 as Latin-1 strings. Python's
secrets.compare_digest (hmac.compare_digest) raises TypeError when given
non-ASCII str arguments, which previously escaped FastAPI dependencies as an
unhandled 500 Internal Server Error, logging a 67-line traceback and bypassing
_SecurityHeadersMiddleware.

Comparing UTF-8 encoded bytes resolves the crash and returns 401 with full
security headers attached.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import soup_cli.ui.app as ui_mod  # noqa: E402
from soup_cli.ui.app import (  # noqa: E402
    _bearer_matches,
    create_app,
    get_auth_token,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_UI_APP_PATH = _REPO_ROOT / "src" / "soup_cli" / "ui" / "app.py"
_SERVE_PATH = _REPO_ROOT / "src" / "soup_cli" / "commands" / "serve.py"
_MCP_SERVER_PATH = _REPO_ROOT / "src" / "soup_cli" / "mcp_server" / "server.py"


def _assert_security_headers(resp) -> None:
    """Assert the four security headers are present on the response."""
    assert resp.headers.get("content-security-policy") == ui_mod.CONTENT_SECURITY_POLICY
    assert resp.headers.get("x-content-type-options") == "nosniff"
    assert resp.headers.get("referrer-policy") == "no-referrer"
    assert resp.headers.get("x-frame-options") == "DENY"


class TestNonAsciiAuthorization:
    """Verify that non-ASCII Authorization headers return 401, not 500."""

    @pytest.fixture
    def client(self):
        # Default raise_server_exceptions=True ensures unhandled TypeErrors fail tests
        return TestClient(create_app(), raise_server_exceptions=True)

    @pytest.mark.parametrize(
        "endpoint,method",
        [
            ("/api/system", "GET"),
            ("/api/auth/ticket", "POST"),
            ("/api/train/logs", "GET"),
        ],
    )
    @pytest.mark.parametrize(
        "header_value",
        [
            # U+00E9 as UTF-8 (c3 a9)
            pytest.param(b"Bearer \xc3\xa9", id="utf8_accent_e"),
            # U+00E9 as one Latin-1 byte (e9)
            pytest.param(b"Bearer \xe9", id="latin1_byte_e"),
            # Cyrillic word 'test' as UTF-8 (d1 82 d0 b5 d1 81 d1 82)
            pytest.param(b"Bearer \xd1\x82\xd0\xb5\xd1\x81\xd1\x82", id="utf8_cyrillic_test"),
            # Valid token followed by c3 a9
            pytest.param(
                b"Bearer " + get_auth_token().encode("ascii") + b"\xc3\xa9",
                id="valid_token_followed_by_utf8",
            ),
            # Non-ASCII with no Bearer prefix
            pytest.param(b"\xc3\xa9", id="no_bearer_prefix"),
        ],
    )
    def test_non_ascii_authorization_returns_401_with_security_headers(
        self, client, endpoint, method, header_value
    ):
        """Non-ASCII headers must return 401 with all security headers attached."""
        headers = {"Authorization": header_value}
        if method == "GET":
            resp = client.get(endpoint, headers=headers)
        elif method == "POST":
            resp = client.post(endpoint, headers=headers)
        else:
            pytest.fail(f"Unsupported method {method}")

        assert resp.status_code == 401, (
            f"Expected 401 for {endpoint} with header {header_value!r}, got {resp.status_code}"
        )
        _assert_security_headers(resp)


class TestHappyPathParity:
    """Verify that valid tokens and SSE tickets continue to succeed."""

    @pytest.fixture
    def client(self):
        return TestClient(create_app(), raise_server_exceptions=True)

    def test_valid_token_returns_200_on_read_endpoint(self, client):
        token = get_auth_token()
        resp = client.get("/api/system", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        _assert_security_headers(resp)

    def test_valid_token_issues_ticket(self, client):
        token = get_auth_token()
        resp = client.post("/api/auth/ticket", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert "ticket" in data
        assert len(data["ticket"]) > 20

    def test_valid_ticket_opens_sse_endpoint(self, client):
        token = get_auth_token()
        ticket_resp = client.post("/api/auth/ticket", headers={"Authorization": f"Bearer {token}"})
        assert ticket_resp.status_code == 200
        ticket = ticket_resp.json()["ticket"]

        sse_resp = client.get(f"/api/train/logs?ticket={ticket}")
        assert sse_resp.status_code == 200


class TestBearerMatchesUnit:
    """Unit tests for the _bearer_matches helper."""

    def test_bearer_matches_valid_token(self):
        valid_header = f"Bearer {get_auth_token()}"
        assert _bearer_matches(valid_header) is True

    def test_bearer_matches_invalid_ascii_header(self):
        assert _bearer_matches("Bearer wrong-token") is False
        assert _bearer_matches("Bearer ") is False
        assert _bearer_matches("") is False

    def test_bearer_matches_non_ascii_str_does_not_raise(self):
        # Must return False rather than raising TypeError on non-ASCII str
        assert _bearer_matches("Bearer \u00e9") is False
        assert _bearer_matches("\u00e9") is False
        assert _bearer_matches("Bearer \u0442\u0435\u0441\u0442") is False
        assert _bearer_matches(f"Bearer {get_auth_token()}\u00e9") is False


def find_auth_compare_digest_violations(tree: ast.AST) -> list[str]:
    """Scan an AST for compare_digest calls in auth checks that compare str."""
    violations: list[str] = []

    # Map variable names to their assigning expressions within each function/scope
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        assignments: dict[str, list[ast.AST]] = {}
        for inner in ast.walk(node):
            if isinstance(inner, ast.Assign):
                for target in inner.targets:
                    if isinstance(target, ast.Name):
                        assignments.setdefault(target.id, []).append(inner.value)

        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            is_compare_digest = False
            if isinstance(inner.func, ast.Attribute) and inner.func.attr == "compare_digest":
                is_compare_digest = True
            elif isinstance(inner.func, ast.Name) and inner.func.id == "compare_digest":
                is_compare_digest = True

            if not is_compare_digest or len(inner.args) < 2:
                continue

            for arg_idx, arg in enumerate(inner.args[:2]):
                # Explicit .encode() call is safe bytes
                if (
                    isinstance(arg, ast.Call)
                    and isinstance(arg.func, ast.Attribute)
                    and arg.func.attr == "encode"
                ):
                    continue

                # Explicit bytes constant is safe
                if isinstance(arg, ast.Constant) and isinstance(arg.value, bytes):
                    continue

                # Explicit str constant or f-string is a violation
                if isinstance(arg, ast.JoinedStr) or (
                    isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                ):
                    violations.append(
                        f"Line {inner.lineno}: compare_digest arg {arg_idx} is str literal/f-string"
                    )
                    continue

                # For names, inspect assignments in the scope
                if isinstance(arg, ast.Name):
                    val_nodes = assignments.get(arg.id, [])
                    for val in val_nodes:
                        if isinstance(val, ast.JoinedStr):
                            violations.append(
                                f"Line {inner.lineno}: arg '{arg.id}' assigned from f-string"
                            )
                        elif isinstance(val, ast.Constant) and isinstance(val.value, str):
                            violations.append(
                                f"Line {inner.lineno}: arg '{arg.id}' assigned from str literal"
                            )
                        elif (
                            isinstance(val, ast.Call)
                            and isinstance(val.func, ast.Attribute)
                            and val.func.attr == "get"
                        ):
                            # e.g. request.headers.get(...) returns str
                            violations.append(
                                f"Line {inner.lineno}: arg '{arg.id}' assigned from headers.get()"
                            )

    return violations


class TestCompareDigestGuard:
    """Guard test ensuring Bearer token comparisons use bytes, not str."""

    def test_guard_accepts_existing_auth_checks(self):
        """The guard must pass on all three Bearer authentication sites in the repo."""
        for path in [_UI_APP_PATH, _SERVE_PATH, _MCP_SERVER_PATH]:
            assert path.is_file(), f"Expected source file at {path}"
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            violations = find_auth_compare_digest_violations(tree)
            assert not violations, f"Violations found in {path.name}: {violations}"

    def test_guard_fires_on_str_comparison_control(self):
        """Control test: the guard must fire when compare_digest compares unencoded str."""
        buggy_auth_code = """
def _verify_token(request):
    auth = request.headers.get("Authorization", "")
    expected = f"Bearer {_auth_token}"
    if not secrets.compare_digest(auth, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")
"""
        tree = ast.parse(buggy_auth_code, filename="<control>")
        violations = find_auth_compare_digest_violations(tree)
        assert len(violations) >= 2, (
            f"Expected guard to flag both auth and expected as str violations, got: {violations}"
        )
