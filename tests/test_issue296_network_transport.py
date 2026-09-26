"""Issue #296 — ``soup mcp serve`` network transports (SSE / streamable HTTP).

v1 shipped stdio only. These tests pin the three things that make a listener
safe to add: it binds loopback by default, every HTTP request carries a Bearer
token or is refused, and the SDK's DNS-rebinding protection is switched on.

The ASGI-level tests drive the app through ``httpx.ASGITransport`` so no socket
is opened; ``TestSseEndToEnd`` is the one test that binds a real ephemeral port,
because "a scripted SSE client smoke (initialize -> list_tools -> call)" is an
acceptance criterion of the issue and an in-memory transport cannot show it.
"""

from __future__ import annotations

import asyncio
import re

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app as cli_app

_ANSI_RE = re.compile(r"\[[0-9;]*m")


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed help output (mirrors test_v07302.py).

    Typer styles each option NAME through Rich, so on a colour-capable stream
    `--transport` arrives with escapes inside it and a raw substring match
    cannot hit. Rich disables colour on Windows, so the raw version passes
    locally and fails on the Linux/macOS cells -- the exact asymmetry
    test_cli_help_assertions_are_ansi_safe.py exists to catch.
    """
    return " ".join(_ANSI_RE.sub("", text).split())

TOKEN = "T" * 43
WRONG = "W" * 43


def _run(coro):
    return asyncio.run(coro)


def _build(transport="sse", *, auth_token=TOKEN, host="127.0.0.1", port=8765):
    from soup_cli.mcp_server.server import build_asgi_app

    return build_asgi_app(
        transport=transport,
        allow_mutating=False,
        allow_execute=False,
        auth_token=auth_token,
        host=host,
        port=port,
    )


async def _request(app, method, path, *, headers=None, port=8765):
    """Drive the app in-process. ``port`` must match the port the app was built
    with, or the Host check refuses the request with 421 before anything else.
    """
    import httpx

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url=f"http://127.0.0.1:{port}"
    ) as client:
        return await client.request(method, path, headers=headers or {})


@pytest.fixture(autouse=True)
def _need_mcp():
    # Mirrors TestServerRoundTrip in test_v07128.py: the SDK only exists with
    # the [mcp] extra, so a partial install skips instead of erroring.
    pytest.importorskip("mcp")


class TestBearerGate:
    """No token, no service — on every network transport and every route."""

    @pytest.mark.parametrize(
        "transport,method,path",
        [
            ("sse", "GET", "/sse"),
            ("sse", "POST", "/messages/"),
            ("http", "POST", "/mcp"),
            ("http", "GET", "/mcp"),
        ],
    )
    def test_missing_authorization_is_401(self, transport, method, path):
        resp = _run(_request(_build(transport), method, path))
        assert resp.status_code == 401

    @pytest.mark.parametrize("transport,path", [("sse", "/messages/"), ("http", "/mcp")])
    def test_wrong_token_is_401(self, transport, path):
        resp = _run(
            _request(
                _build(transport),
                "POST",
                path,
                headers={"Authorization": f"Bearer {WRONG}"},
            )
        )
        assert resp.status_code == 401

    def test_malformed_authorization_header_is_401(self):
        # Right token, wrong scheme: "Bearer" is not optional.
        resp = _run(
            _request(
                _build("sse"), "POST", "/messages/", headers={"Authorization": TOKEN}
            )
        )
        assert resp.status_code == 401

    def test_correct_token_reaches_the_transport(self):
        # 400 is the SDK's own "session_id is required" answer on /messages/ --
        # the point is that it is NOT 401, i.e. the request got past the gate.
        resp = _run(
            _request(
                _build("sse"),
                "POST",
                "/messages/",
                headers={"Authorization": f"Bearer {TOKEN}"},
            )
        )
        assert resp.status_code == 400


class TestDnsRebindingProtection:
    """The gate a Bearer token cannot be: a browser attaches no Authorization
    header, but a page the operator merely visits can still reach a loopback
    port. The SDK's Host check is what refuses it.

    Both requests carry a JSON Content-Type on purpose -- the SDK validates
    Content-Type before Host on a POST, so without it the 400 for the missing
    header would mask whatever the Host check would have done.
    """

    _JSON = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

    def test_foreign_host_header_is_refused(self):
        # Authenticated on purpose: proves the Host check is a second gate
        # behind auth, not something auth happens to mask.
        resp = _run(
            _request(
                _build("sse"),
                "POST",
                "/messages/",
                headers={**self._JSON, "Host": "evil.example"},
            )
        )
        assert resp.status_code == 421

    def test_bound_host_and_port_are_allowed(self):
        resp = _run(
            _request(
                _build("sse", host="127.0.0.1", port=8765),
                "POST",
                "/messages/",
                headers={**self._JSON, "Host": "127.0.0.1:8765"},
            )
        )
        assert resp.status_code != 421

    def test_wildcard_bind_cannot_pin_a_host(self):
        from soup_cli.mcp_server.server import allowed_hosts_for, is_wildcard_host

        assert is_wildcard_host("0.0.0.0")
        assert allowed_hosts_for("0.0.0.0", 8765) == ["*"]
        assert "127.0.0.1:8765" in allowed_hosts_for("127.0.0.1", 8765)
        assert "localhost:8765" in allowed_hosts_for("127.0.0.1", 8765)

    def test_ipv6_literal_is_bracketed(self):
        # A client talking to ::1 sends `Host: [::1]:8765`. Joining with a bare
        # colon yields `::1:8765`, which matches nothing and would refuse every
        # IPv6 loopback client with a 421.
        from soup_cli.mcp_server.server import allowed_hosts_for

        hosts = allowed_hosts_for("::1", 8765)
        assert "[::1]:8765" in hosts
        assert "::1" in hosts
        assert "::1:8765" not in hosts


class TestAuthTokenResolution:
    def test_generated_token_is_urlsafe_and_valid(self):
        from soup_cli.commands.mcp import _resolve_auth_token
        from soup_cli.utils.qr_url import validate_token

        token = _resolve_auth_token(None)
        assert validate_token(token) == token
        assert len(token) >= 16

    def test_generated_tokens_differ_between_calls(self):
        from soup_cli.commands.mcp import _resolve_auth_token

        assert _resolve_auth_token(None) != _resolve_auth_token(None)

    def test_operator_token_is_preserved(self):
        from soup_cli.commands.mcp import _resolve_auth_token

        assert _resolve_auth_token(TOKEN) == TOKEN

    def test_short_token_is_rejected(self):
        from soup_cli.commands.mcp import _resolve_auth_token

        with pytest.raises(ValueError):
            _resolve_auth_token("tooshort")


class TestCliSurface:
    def test_network_only_flags_are_rejected_under_stdio(self):
        # Silently ignoring a flag that cannot apply is the failure mode this
        # guards against -- stdio has no host, no port and no auth.
        runner = CliRunner()
        for flag, value in (("--auth-token", TOKEN), ("--port", "9000"), ("--host", "0.0.0.0")):
            result = runner.invoke(cli_app, ["mcp", "serve", "--transport", "stdio", flag, value])
            assert result.exit_code != 0, (flag, result.output)
            assert "stdio" in result.output.lower(), (flag, result.output)

    def test_unknown_transport_is_rejected(self):
        result = CliRunner().invoke(cli_app, ["mcp", "serve", "--transport", "carrier-pigeon"])
        assert result.exit_code != 0

    # Same set test_cli_subprocess.py::TestUnicodeSafety guards. Not "pure
    # ASCII": Typer draws its help panels with box characters, which are fine.
    # These six are the ones that break a cp1252 Windows terminal.
    PROBLEMATIC_CHARS = ["→", "—", "‘", "’", "“", "”"]

    def test_help_lists_the_transports(self):
        result = CliRunner().invoke(cli_app, ["mcp", "serve", "--help"])
        assert result.exit_code == 0, result.output
        help_text = _plain(result.output)
        assert "--transport" in help_text
        for name in ("stdio", "sse", "http"):
            assert name in help_text

    def test_help_stays_terminal_safe(self):
        result = CliRunner().invoke(cli_app, ["mcp", "serve", "--help"])
        for char in self.PROBLEMATIC_CHARS:
            assert char not in _plain(result.output), repr(char)


class TestTransportListsAgree:
    def test_cli_list_matches_the_server_module(self):
        # The CLI keeps its own tuple on purpose: --transport must be validated
        # before the SDK is imported, so a missing [mcp] extra still produces a
        # flag error rather than an install message. This pins the two lists
        # together so the deliberate duplication cannot drift into a real one.
        from soup_cli.commands.mcp import _TRANSPORTS
        from soup_cli.mcp_server.server import NETWORK_TRANSPORTS

        assert _TRANSPORTS == ("stdio",) + NETWORK_TRANSPORTS

    def test_advertised_paths_come_from_the_server_module(self):
        from soup_cli.mcp_server.server import HTTP_PATH, MESSAGE_PATH, SSE_PATH

        assert (SSE_PATH, MESSAGE_PATH, HTTP_PATH) == ("/sse", "/messages/", "/mcp")


class TestBuildAsgiApp:
    def test_unknown_transport_raises(self):
        with pytest.raises(ValueError):
            _build("carrier-pigeon")

    def test_invalid_auth_token_raises(self):
        with pytest.raises(ValueError):
            _build("sse", auth_token="short")

    def test_stdio_is_not_an_asgi_transport(self):
        with pytest.raises(ValueError):
            _build("stdio")


# ---------------------------------------------------------------------------
# Real socket. The acceptance criterion asks for "a scripted SSE client smoke
# (initialize -> list_tools -> call)", and an in-process ASGI transport cannot
# show that a client which speaks SSE over a real connection is served.
# ---------------------------------------------------------------------------


class _UvicornThread:
    """Run the app on an ephemeral port for the duration of a `with` block."""

    def __init__(self, app, port):
        import uvicorn

        self._server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        )
        self._thread = None

    def __enter__(self):
        import threading
        import time

        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self._server.started:
                return self
            time.sleep(0.05)
        raise RuntimeError("uvicorn did not start within 30s")

    def __exit__(self, *exc):
        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=30)
        return False


def _free_port():
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def _sse_session(port, token):
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    url = f"http://127.0.0.1:{port}/sse"
    async with sse_client(url, headers={"Authorization": f"Bearer {token}"}) as (
        read_stream,
        write_stream,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool("recipes_search", {"query": "qwen"})
            return tools, result


class TestSseEndToEnd:
    def test_initialize_list_tools_and_call_over_a_real_socket(self):
        port = _free_port()
        app = _build("sse", port=port)
        with _UvicornThread(app, port):
            tools, result = _run(_sse_session(port, TOKEN))

        from soup_cli.mcp_server.registry import build_registry

        names = {tool.name for tool in tools.tools}
        # Parity with stdio is the acceptance criterion: SSE must serve the
        # SAME registry, not a subset. Compared against build_registry rather
        # than a hardcoded count so adding a tool cannot make this drift.
        expected = {
            spec.name
            for spec in build_registry(allow_mutating=False, allow_execute=False)
        }
        assert names == expected
        assert "recipes_search" in names
        assert result.content and result.content[0].text.strip().startswith("{")

    def test_wrong_token_cannot_open_a_session(self):
        port = _free_port()
        app = _build("sse", port=port)
        with _UvicornThread(app, port):
            with pytest.raises(Exception):
                _run(_sse_session(port, WRONG))


class TestTransportEntryPoints:
    """`run_sse_server` / `run_http_server` are the entry points #296 names.

    These pin which app each one hands to uvicorn without binding a socket.
    They are characterization tests for two one-line wrappers, not red-first
    tests -- their value is that the wrappers stop being uncovered names.
    """

    @staticmethod
    def _capture(monkeypatch):
        import uvicorn

        seen = {}

        def _fake_run(app, **kwargs):
            seen["app"] = app
            seen["kwargs"] = kwargs

        monkeypatch.setattr(uvicorn, "run", _fake_run)
        return seen

    @staticmethod
    def _authed(app, path, port):
        return _run(
            _request(
                app,
                "POST",
                path,
                port=port,
                headers={
                    "Authorization": f"Bearer {TOKEN}",
                    "Content-Type": "application/json",
                },
            )
        )

    def test_run_sse_server_serves_the_sse_routes(self, monkeypatch):
        from soup_cli.mcp_server.server import run_sse_server

        seen = self._capture(monkeypatch)
        run_sse_server(
            allow_mutating=False,
            allow_execute=False,
            auth_token=TOKEN,
            host="127.0.0.1",
            port=1234,
        )
        assert seen["kwargs"]["host"] == "127.0.0.1"
        assert seen["kwargs"]["port"] == 1234
        # 400 is the SSE transport answering; 404 would mean the route is absent.
        assert self._authed(seen["app"], "/messages/", 1234).status_code == 400
        assert _run(_request(seen["app"], "GET", "/sse", port=1234)).status_code == 401

    def test_run_http_server_serves_the_streamable_route(self, monkeypatch):
        from soup_cli.mcp_server.server import run_http_server

        seen = self._capture(monkeypatch)
        run_http_server(
            allow_mutating=False,
            allow_execute=False,
            auth_token=TOKEN,
            host="127.0.0.1",
            port=4321,
        )
        assert seen["kwargs"]["port"] == 4321
        # The SSE routes do not exist on the streamable-HTTP app.
        assert self._authed(seen["app"], "/messages/", 4321).status_code == 404
        assert _run(_request(seen["app"], "POST", "/mcp", port=4321)).status_code == 401

    def test_entry_points_validate_before_binding(self, monkeypatch):
        from soup_cli.mcp_server.server import run_sse_server

        seen = self._capture(monkeypatch)
        with pytest.raises(ValueError):
            run_sse_server(
                allow_mutating=False,
                allow_execute=False,
                auth_token="short",
                host="127.0.0.1",
                port=1234,
            )
        assert "app" not in seen  # never reached uvicorn
class TestExecutionIsStdioOnly:
    """--allow-execute must not reach a network listener (#296 review).

    The pre-fix code passed ``allow_execute`` straight through to
    ``run_network_server``, and the banner still said "execution disabled"
    because the comment predated #297 -- so a listener one Bearer token from
    the network could spawn real training / export processes while announcing
    that it could not.

    These tests assert BEHAVIOUR, not the tool table. ``build_registry``
    always LISTS train_execute / export_execute and swaps only the handler,
    so a test comparing tool names passes with the gate hardwired open --
    which is exactly how this went uncovered.
    """

    @pytest.mark.parametrize("transport", ["sse", "http"])
    def test_cli_refuses_execute_on_a_network_transport(self, transport):
        result = CliRunner().invoke(
            cli_app, ["mcp", "serve", "--transport", transport, "--allow-execute"]
        )
        assert result.exit_code == 2, (result.output, repr(result.exception))
        plain = _plain(result.output)
        # Name the flag and the transport: an assertion on the exit code alone
        # passes when a DIFFERENT guard fires (e.g. the unknown-transport one).
        assert "--allow-execute" in plain, plain
        assert "stdio" in plain, plain

    def test_cli_still_allows_execute_over_stdio(self, monkeypatch):
        """Control: the refusal is targeted, not a blanket kill of the flag."""
        seen = {}

        def _fake(**kwargs):
            seen.update(kwargs)

        monkeypatch.setattr("soup_cli.mcp_server.server.run_stdio_server", _fake)
        result = CliRunner().invoke(cli_app, ["mcp", "serve", "--allow-execute"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert seen == {"allow_mutating": True, "allow_execute": True}, seen

    def test_stdio_banner_does_not_claim_execution_is_disabled(self, monkeypatch):
        """The stale banner was the half that made the hole quiet."""
        monkeypatch.setattr(
            "soup_cli.mcp_server.server.run_stdio_server", lambda **kw: None
        )
        result = CliRunner().invoke(cli_app, ["mcp", "serve", "--allow-execute"])
        plain = _plain(result.output)
        assert "execution disabled" not in plain.lower(), plain
        assert "execution ENABLED" in plain, plain

    def test_build_asgi_app_refuses_execute_structurally(self):
        """A direct caller must not be able to bypass the CLI guard."""
        from soup_cli.mcp_server.server import build_asgi_app

        with pytest.raises(ValueError, match="allow_execute"):
            build_asgi_app(
                transport="sse",
                allow_mutating=True,
                allow_execute=True,
                auth_token=TOKEN,
            )

    @pytest.mark.parametrize("entry", ["run_sse_server", "run_http_server"])
    def test_entry_points_refuse_execute_before_binding(self, entry, monkeypatch):
        import soup_cli.mcp_server.server as srv

        def _boom(*a, **kw):  # pragma: no cover - must never be reached
            raise AssertionError("uvicorn.run reached with allow_execute=True")

        import uvicorn

        monkeypatch.setattr(uvicorn, "run", _boom)
        with pytest.raises(ValueError, match="allow_execute"):
            getattr(srv, entry)(
                allow_mutating=True,
                allow_execute=True,
                auth_token=TOKEN,
            )


class TestExecuteGateIsCoveredByBehaviour:
    """Pins the gate the name-comparing tests cannot see.

    Hardwiring ``allow_execute=True`` inside build_registry leaves the tool
    NAMES identical, so only calling a handler distinguishes the two states.
    """

    @staticmethod
    def _handler(name, *, allow_execute):
        from soup_cli.mcp_server.registry import build_registry

        specs = {
            spec.name: spec
            for spec in build_registry(
                allow_mutating=True, allow_execute=allow_execute
            )
        }
        return specs[name]

    @pytest.mark.parametrize("name", ["train_execute", "export_execute"])
    def test_tool_names_alone_cannot_tell_the_states_apart(self, name):
        """The control that explains why the behavioural tests below exist."""
        from soup_cli.mcp_server.registry import build_registry

        off = {s.name for s in build_registry(allow_mutating=True, allow_execute=False)}
        on = {s.name for s in build_registry(allow_mutating=True, allow_execute=True)}
        assert name in off and name in on
        assert off == on

    @pytest.mark.parametrize("name", ["train_execute", "export_execute"])
    def test_execute_handler_refuses_when_the_gate_is_off(self, name):
        from soup_cli.mcp_server.registry import McpToolError

        spec = self._handler(name, allow_execute=False)
        with pytest.raises(McpToolError, match="--allow-execute"):
            spec.handler({"confirmation_token": "x" * 43})

    @pytest.mark.parametrize("name", ["train_execute", "export_execute"])
    def test_execute_handler_is_live_when_the_gate_is_on(self, name):
        """Must NOT be the disabled-flag refusal.

        A bogus token still fails -- but on the token, which is the enabled
        path. Asserting "raises" alone would pass in both states.
        """
        from soup_cli.mcp_server.registry import McpToolError

        spec = self._handler(name, allow_execute=True)
        with pytest.raises(McpToolError) as exc:
            spec.handler({"confirmation_token": "x" * 43})
        assert "--allow-execute" not in str(exc.value), str(exc.value)

    @pytest.mark.parametrize("name", ["train_execute", "export_execute"])
    def test_gate_is_off_even_when_an_execution_manager_is_supplied(self, name):
        """The combination the production path actually uses.

        ``run_stdio_server`` builds an ``ExecutionManager()`` unconditionally
        and passes it whether or not --allow-execute was given, so the handler
        choice must turn on ``allow_execute`` -- not merely on an execution
        manager being present. Dropping ``allow_execute and`` from that
        condition passed every other test in this file while making a plain
        ``soup mcp serve`` serve live execution handlers.
        """
        from soup_cli.mcp_server.execution import ExecutionManager
        from soup_cli.mcp_server.registry import McpToolError, build_registry

        specs = {
            spec.name: spec
            for spec in build_registry(
                allow_mutating=True,
                allow_execute=False,
                execution=ExecutionManager(),
            )
        }
        with pytest.raises(McpToolError, match="--allow-execute"):
            specs[name].handler({"confirmation_token": "x" * 43})

    def test_stdio_entry_point_passes_an_execution_manager_unconditionally(
        self, monkeypatch
    ):
        """Control: pins WHY the test above is the one that matters.

        If run_stdio_server ever stops handing an ExecutionManager to a
        gate-off registry, the test above becomes vacuous and should be
        re-justified rather than silently kept.
        """
        import soup_cli.mcp_server.server as srv

        seen = {}

        def _capture(**kwargs):
            seen.update(kwargs)
            return []

        monkeypatch.setattr(srv, "build_registry", _capture)
        monkeypatch.setattr(srv, "build_server", lambda specs: None)
        monkeypatch.setattr(srv.anyio, "run", lambda *a, **kw: None)
        srv.run_stdio_server(allow_mutating=False, allow_execute=False)
        assert seen["allow_execute"] is False, seen
        assert seen["execution"] is not None, seen
