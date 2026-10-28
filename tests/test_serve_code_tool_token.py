"""``soup serve``: the code-execution tool routes take ``--tool-auth-token`` on every bind.

``/v1/tools/python`` and ``/v1/tools/bash`` answer 401 unless the server was started
with ``--tool-auth-token`` (or ``SOUP_TOOL_AUTH_TOKEN``) and the request sends
``Authorization: Bearer <token>``. They already did so beyond loopback; they now do so
on a loopback bind too. ``/v1/tools/web_search``, ``/v1/thumbs`` and the adapter routes
keep their rule: a token only when one is configured.

The token check comes after the Host/Origin check and before the request's content is
used or a sandbox starts, so a refused request runs nothing. An empty token, from the
option, the environment or ``_create_app``, counts as no token.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from soup_cli.cli import app as cli_app  # noqa: E402
from tests.conftest import strip_ansi  # noqa: E402

runner = CliRunner()

TOKEN = "s3cret-tool-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}

# The exact refusal a client reads when the server has no token configured.
TOKEN_REQUIRED = (
    "Authentication required: /v1/tools/python and /v1/tools/bash need a token on "
    "every bind. Start soup serve with --tool-auth-token <secret> (or set "
    "SOUP_TOOL_AUTH_TOKEN) and send 'Authorization: Bearer <secret>'."
)
BAD_BEARER = "Invalid or missing bearer token"

CODE_ROUTES = [
    ("/v1/tools/python", {"code": "print(1)"}),
    ("/v1/tools/bash", {"command": "echo 1"}),
]
CODE_ROUTE_IDS = ["python", "bash"]
LOOPBACK_BINDS = ["127.0.0.1", "localhost", "::1"]


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output, safe to substring-match."""
    return " ".join(strip_ansi(text).split())


def _app(**kwargs):
    from soup_cli.commands.serve import _create_app

    base = dict(
        model_obj=MagicMock(),
        tokenizer=MagicMock(),
        device="cpu",
        model_name="test-model",
        max_tokens_default=64,
    )
    base.update(kwargs)
    return _create_app(**base)


@pytest.fixture
def sandbox_calls(monkeypatch):
    """Stand in for both sandboxes and record every run they are asked for.

    The isolation strategy is pinned to one that lets both routes run, so what is
    measured here is the token check alone, on any host.
    """
    from soup_cli.trainer import rewards

    calls: list[tuple[str, str]] = []

    def _fake_python(code, **_options):
        calls.append(("python", code))
        return rewards.SandboxProcessResult(returncode=0, stdout="ran\n", stderr="")

    def _fake_bash(command, **_options):
        calls.append(("bash", command))
        return rewards.SandboxProcessResult(returncode=0, stdout="ran\n", stderr="")

    monkeypatch.setattr(rewards, "_get_isolation_strategy", lambda: "namespaces")
    monkeypatch.setattr(rewards, "_run_code_sandbox", lambda *a, **k: "ran\n")
    monkeypatch.setattr(rewards, "_run_code_sandbox_process", _fake_python)
    monkeypatch.setattr(rewards, "_run_bash_sandbox", _fake_bash)
    return calls


class TestNoTokenConfigured:
    @pytest.mark.parametrize("bind", LOOPBACK_BINDS)
    @pytest.mark.parametrize(("path", "body"), CODE_ROUTES, ids=CODE_ROUTE_IDS)
    def test_code_routes_refuse_on_a_loopback_bind(self, bind, path, body, sandbox_calls):
        client = TestClient(_app(host=bind), base_url="http://127.0.0.1:8000")

        resp = client.post(path, json=body)

        assert resp.status_code == 401, (path, resp.text)
        assert resp.headers.get("www-authenticate") == "Bearer"
        assert resp.json()["detail"] == TOKEN_REQUIRED
        assert sandbox_calls == []

    @pytest.mark.parametrize(("path", "body"), CODE_ROUTES, ids=CODE_ROUTE_IDS)
    def test_code_routes_refuse_beyond_loopback(self, path, body, sandbox_calls):
        client = TestClient(_app(host="0.0.0.0"), base_url="http://10.0.0.7:8000")

        resp = client.post(path, json=body)

        assert resp.status_code == 401, (path, resp.text)
        assert resp.headers.get("www-authenticate") == "Bearer"
        assert resp.json()["detail"] == TOKEN_REQUIRED
        assert sandbox_calls == []

    @pytest.mark.parametrize(("path", "body"), CODE_ROUTES, ids=CODE_ROUTE_IDS)
    def test_a_bearer_header_does_not_stand_in_for_a_server_token(
        self, path, body, sandbox_calls
    ):
        client = TestClient(_app(), base_url="http://127.0.0.1:8000")

        resp = client.post(path, json=body, headers=AUTH)

        assert resp.status_code == 401, (path, resp.text)
        assert resp.headers.get("www-authenticate") == "Bearer"
        assert resp.json()["detail"] == TOKEN_REQUIRED
        assert sandbox_calls == []

    @pytest.mark.parametrize(
        "header",
        [None, AUTH, {"Authorization": "Bearer"}],
        ids=["no-header", "bearer", "empty-bearer"],
    )
    @pytest.mark.parametrize(("path", "body"), CODE_ROUTES, ids=CODE_ROUTE_IDS)
    def test_an_empty_token_is_no_token(self, header, path, body, sandbox_calls):
        client = TestClient(_app(auth_token=""), base_url="http://127.0.0.1:8000")

        resp = client.post(path, json=body, headers=header)

        assert resp.status_code == 401, (path, resp.text)
        assert resp.headers.get("www-authenticate") == "Bearer"
        assert resp.json()["detail"] == TOKEN_REQUIRED
        assert sandbox_calls == []

    @pytest.mark.parametrize(("path", "_body"), CODE_ROUTES, ids=CODE_ROUTE_IDS)
    def test_the_token_is_checked_before_the_body_content(self, path, _body, sandbox_calls):
        client = TestClient(_app(), base_url="http://127.0.0.1:8000")

        resp = client.post(path, json={})

        assert resp.status_code == 401, (path, resp.text)
        assert resp.json()["detail"] == TOKEN_REQUIRED

    @pytest.mark.parametrize(("path", "body"), CODE_ROUTES, ids=CODE_ROUTE_IDS)
    def test_the_token_is_checked_before_the_isolation_requirement(
        self, path, body, sandbox_calls, monkeypatch
    ):
        from soup_cli.trainer import rewards

        # A host without OS-level isolation answers 501 once a request is let in.
        monkeypatch.setattr(rewards, "_get_isolation_strategy", lambda: "best-effort")
        client = TestClient(_app(), base_url="http://127.0.0.1:8000")

        resp = client.post(path, json=body)

        assert resp.status_code == 401, (path, resp.text)
        assert resp.json()["detail"] == TOKEN_REQUIRED
        assert sandbox_calls == []

    @pytest.mark.parametrize(("path", "body"), CODE_ROUTES, ids=CODE_ROUTE_IDS)
    def test_the_host_check_still_answers_first(self, path, body, sandbox_calls):
        client = TestClient(_app(), base_url="http://evil.example:8000")

        resp = client.post(path, json=body)

        assert resp.status_code == 421, (path, resp.text)
        assert sandbox_calls == []

    def test_web_search_keeps_its_rule_on_loopback(self):
        from soup_cli.utils.server_tools import WebSearchConfig

        def fake_backend(query, max_results, allowlist):
            return [{"url": "https://example.com/a", "snippet": "s"}]

        client = TestClient(
            _app(
                web_search_config=WebSearchConfig(domain_allowlist=("example.com",)),
                web_search_backend=fake_backend,
            ),
            base_url="http://127.0.0.1:8000",
        )

        resp = client.post("/v1/tools/web_search", json={"query": "q"})

        assert resp.status_code == 200, resp.text
        assert [r["url"] for r in resp.json()["results"]] == ["https://example.com/a"]

    def test_adapter_listing_keeps_its_rule_on_loopback(self):
        client = TestClient(
            _app(adapter_map={"chat": "/fake/chat"}), base_url="http://127.0.0.1:8000"
        )

        resp = client.get("/v1/adapters")

        assert resp.status_code == 200, resp.text
        assert [a["name"] for a in resp.json()["adapters"]] == ["chat"]


class TestTokenConfigured:
    @pytest.mark.parametrize("bind", ["127.0.0.1", "0.0.0.0"])
    @pytest.mark.parametrize(
        "header",
        [None, "Bearer wrong", TOKEN, f"Bearer {TOKEN}x", f"bearer {TOKEN}"],
        ids=["missing", "wrong", "no-scheme", "suffix", "lowercase-scheme"],
    )
    @pytest.mark.parametrize(("path", "body"), CODE_ROUTES, ids=CODE_ROUTE_IDS)
    def test_code_routes_refuse_a_bad_bearer(self, bind, header, path, body, sandbox_calls):
        client = TestClient(_app(host=bind, auth_token=TOKEN), base_url="http://127.0.0.1:8000")
        headers = {"Authorization": header} if header is not None else {}

        resp = client.post(path, json=body, headers=headers)

        assert resp.status_code == 401, (path, resp.text)
        assert resp.json()["detail"] == BAD_BEARER
        assert sandbox_calls == []

    @pytest.mark.parametrize("bind", LOOPBACK_BINDS + ["0.0.0.0"])
    @pytest.mark.parametrize(("path", "body"), CODE_ROUTES, ids=CODE_ROUTE_IDS)
    def test_code_routes_run_with_the_right_bearer(self, bind, path, body, sandbox_calls):
        client = TestClient(_app(host=bind, auth_token=TOKEN), base_url="http://127.0.0.1:8000")

        resp = client.post(path, json=body, headers=AUTH)

        assert resp.status_code == 200, (path, resp.text)
        assert resp.json()["stdout"] == "ran\n"
        assert [route for route, _ in sandbox_calls] == [path.rsplit("/", 1)[1]]

    def test_with_the_right_bearer_the_body_is_checked(self, sandbox_calls):
        client = TestClient(_app(auth_token=TOKEN), base_url="http://127.0.0.1:8000")

        resp = client.post("/v1/tools/python", json={}, headers=AUTH)

        assert resp.status_code == 400, resp.text
        assert sandbox_calls == []


class TestCommandLine:
    def test_help_says_the_code_routes_need_the_token_on_every_bind(self, monkeypatch):
        monkeypatch.setenv("COLUMNS", "200")

        result = runner.invoke(cli_app, ["serve", "--help"])

        assert result.exit_code == 0, (result.output, repr(result.exception))
        out = _plain(result.output.replace("│", " "))
        assert (
            "Without it, /v1/tools/python and /v1/tools/bash answer 401 on every bind "
            "and a non-loopback --host is refused." in out
        )
        assert "SOUP_TOOL_AUTH_TOKEN" in out
        assert "Strongly recommended whenever --host is not loopback" not in out

    def test_the_env_var_supplies_the_token(self, monkeypatch):
        monkeypatch.setenv("COLUMNS", "400")
        monkeypatch.setenv("SOUP_TOOL_AUTH_TOKEN", "env-token-value-123")

        # The model path does not exist, so the command stops before a server
        # starts; the bind check runs before that.
        result = runner.invoke(
            cli_app, ["serve", "--model", "no-such-model-dir", "--host", "0.0.0.0"]
        )

        out = _plain(result.output)
        assert "requires --tool-auth-token" not in out, out
        assert "Note: bound to '0.0.0.0', not loopback" in out, out
        assert "env-token-value-123" not in out

    @pytest.mark.parametrize(
        ("env", "option"),
        [("", []), (None, ["--tool-auth-token", ""])],
        ids=["empty-env-var", "empty-option"],
    )
    def test_an_empty_token_is_no_token_at_start(self, env, option, monkeypatch):
        monkeypatch.setenv("COLUMNS", "400")
        if env is not None:
            monkeypatch.setenv("SOUP_TOOL_AUTH_TOKEN", env)

        result = runner.invoke(
            cli_app, ["serve", "--model", "no-such-model-dir", "--host", "0.0.0.0", *option]
        )

        assert result.exit_code == 2, (result.output, repr(result.exception))
        out = _plain(result.output)
        assert "binding non-loopback host '0.0.0.0' requires --tool-auth-token <secret>" in out
        assert "Note: bound to" not in out

    def test_without_the_env_var_a_non_loopback_bind_is_still_refused(self, monkeypatch):
        monkeypatch.setenv("COLUMNS", "400")
        monkeypatch.delenv("SOUP_TOOL_AUTH_TOKEN", raising=False)

        result = runner.invoke(
            cli_app, ["serve", "--model", "no-such-model-dir", "--host", "0.0.0.0"]
        )

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert (
            "binding non-loopback host '0.0.0.0' requires --tool-auth-token <secret>"
            in _plain(result.output)
        )


class TestTheServingDocs:
    """docs/serving-and-export.md says what the two code routes can reach and how to
    choose and carry the token."""

    def _section(self) -> str:
        from pathlib import Path

        text = (Path(__file__).parents[1] / "docs" / "serving-and-export.md").read_text(
            encoding="utf-8"
        )
        start = text.index("## Server-Side Tool Endpoints")
        end = text.index("**What `--tool-auth-token` does not cover.**")
        return " ".join(text[start:end].split())

    def test_both_code_routes_say_they_run_as_the_server_user(self):
        section = self._section()
        assert "world-readable system files" not in section
        assert (
            "the code runs as the server's user, with the network cut and a scrubbed "
            "environment, and can read and write every file that user can"
        ) in section
        assert (
            "the command runs as the server's user and can read and write every file that "
            "user can"
        ) in section
        assert "in a container, a VM or under a dedicated unprivileged account" in section

    def test_signalling_other_processes_is_said_for_linux_only(self):
        section = self._section()
        # Unqualified, the sentence would claim the macOS child can signal every
        # process of the user; its profile allows signals to the run itself only.
        assert "configs) and send signals to that user's processes" not in section
        assert "user can and send signals to that user's processes" not in section
        assert (
            "(SSH keys, Hugging Face tokens, configs) and, on Linux, send signals to that "
            "user's processes. On macOS the profile allows reading every file, writing in "
            "the temporary directories and sending signals only to the run itself."
        ) in section
        assert (
            "every file that user can and, on Linux, send signals to that user's processes "
            "(on macOS the same profile limits writes to the temporary directories and "
            "signals to the run itself); only the network is cut."
        ) in section

    def test_the_token_advice_names_a_long_random_token_and_loopback_or_tls(self):
        section = self._section()
        assert 'python -c "import secrets; print(secrets.token_urlsafe(32))"' in section
        assert "keep the server on loopback or behind a proxy that terminates TLS" in section
