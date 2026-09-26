"""MCP server wiring for ``soup mcp serve`` (v0.71.28; network transports #296).

This is the ONLY module that imports the ``mcp`` SDK — importing it therefore
requires the ``[mcp]`` extra. The pure tool table lives in
:mod:`soup_cli.mcp_server.registry` (no SDK dependency, fully unit-testable).

:func:`build_server` is transport-agnostic; ``stdio`` (v1) and the two
network transports added for #296 (``sse`` / streamable ``http``) all wrap
the same server object. Everything the network transports need beyond the
SDK -- starlette, uvicorn -- is already a hard dependency of ``mcp``.
"""

from __future__ import annotations

import inspect
import json
import secrets
import sys
from contextlib import redirect_stdout
from typing import List

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from soup_cli.mcp_server.execution import ExecutionManager
from soup_cli.mcp_server.registry import McpToolError, ToolSpec, _sanitize, build_registry

SERVER_NAME = "soup"


class _DispatchError(Exception):
    """A tool call that failed, carrying an already-sanitized message."""


def _tool_descriptors(specs: List[ToolSpec]) -> List[types.Tool]:
    """The advertised tool table. Identical on both mcp majors."""
    return [
        types.Tool(
            name=spec.name,
            title=spec.title,
            description=spec.description,
            inputSchema=spec.input_schema,
            annotations=spec.annotations,
        )
        for spec in specs
    ]


def _dispatch_tool(by_name, name: str, arguments: dict) -> str:
    """Run one tool and return its pretty-printed JSON text.

    Raises :class:`_DispatchError` with a path-free, C0/ESC-free message for
    every failure mode, so the two SDK adapters below only have to decide how
    to *shape* an error, never what it says.
    """
    spec = by_name.get(name)
    if spec is None:
        raise _DispatchError("unknown tool")
    try:
        # Any core that prints (e.g. a Rich warning) must not corrupt the
        # JSON-RPC stdout channel - send stray stdout to stderr for the
        # duration of the (synchronous) handler call. Serialization stays
        # INSIDE the try so a non-JSON-serializable result also becomes a
        # sanitized error (never a raw TypeError the SDK would echo).
        with redirect_stdout(sys.stderr):
            result = spec.handler(arguments or {})
        return json.dumps(_sanitize(result), indent=2, ensure_ascii=False)
    except McpToolError as exc:
        # _sanitize the message too so the C0/ESC guarantee is structural,
        # not just a convention every handler must remember (security-review).
        raise _DispatchError(_sanitize(str(exc))) from None
    except Exception as exc:  # never leak a stack trace / path to the client
        raise _DispatchError(f"internal error ({type(exc).__name__})") from None


def _uses_callback_handlers() -> bool:
    """True on mcp 2.x, which registers handlers through ``Server(...)``.

    #322 — 2.0.0 removed the ``@server.list_tools()`` / ``@server.call_tool()``
    decorators in favour of ``on_list_tools=`` / ``on_call_tool=`` constructor
    arguments. Asked of the constructor rather than of ``mcp.__version__``: a
    version table is the thing that goes stale, and this repo has been bitten
    by exactly that twice (see ``trainer/_trl_compat.py``).
    """
    return "on_list_tools" in inspect.signature(Server.__init__).parameters


def build_server(specs: List[ToolSpec]) -> Server:
    """Build a low-level MCP :class:`Server` that dispatches to ``specs``.

    Each tool result is a plain ``dict`` from the handler; it is sanitized
    (C0/ESC-stripped) and returned as a single pretty-printed JSON
    ``TextContent`` block. Handler failures become an ``isError`` result with a
    path-free message so the server survives bad calls.

    Supports both mcp majors (#322). The dispatch logic is shared; only the two
    adapters below differ, because 2.x hands its handlers a request context and
    wants a ``*Result`` object where 1.x wanted a bare list.
    """
    by_name = {spec.name: spec for spec in specs}

    if _uses_callback_handlers():  # mcp 2.x
        async def _on_list_tools(ctx, params) -> types.ListToolsResult:
            return types.ListToolsResult(tools=_tool_descriptors(specs))

        async def _on_call_tool(ctx, params) -> types.CallToolResult:
            try:
                text = _dispatch_tool(by_name, params.name, params.arguments or {})
            except _DispatchError as exc:
                # Shaped explicitly rather than raised: on 2.x an exception out
                # of a handler becomes a JSON-RPC error, which is a different
                # thing on the wire from a tool that ran and reported failure.
                # 1.x produced the latter, and the tests assert it.
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=str(exc))],
                    isError=True,
                )
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=text)]
            )

        return Server(
            SERVER_NAME,
            on_list_tools=_on_list_tools,
            on_call_tool=_on_call_tool,
        )

    server: Server = Server(SERVER_NAME)  # mcp 1.x

    @server.list_tools()
    async def _list_tools() -> List[types.Tool]:
        return _tool_descriptors(specs)

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict) -> List[types.TextContent]:
        try:
            text = _dispatch_tool(by_name, name, arguments or {})
        except _DispatchError as exc:
            # 1.x turns a raised ValueError into the isError result itself.
            raise ValueError(str(exc)) from None
        return [types.TextContent(type="text", text=text)]

    return server


def run_stdio_server(*, allow_mutating: bool, allow_execute: bool) -> None:
    """Run the MCP server over stdio until the client disconnects."""
    execution = ExecutionManager()
    server = build_server(build_registry(
        allow_mutating=allow_mutating,
        allow_execute=allow_execute,
        execution=execution,
    ))

    async def _main() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    anyio.run(_main)


# --- Network transports (#296) ---------------------------------------------

DEFAULT_NETWORK_PORT = 8765
DEFAULT_NETWORK_HOST = "127.0.0.1"
SSE_PATH = "/sse"
MESSAGE_PATH = "/messages/"
HTTP_PATH = "/mcp"
NETWORK_TRANSPORTS = ("sse", "http")
# Binds that accept from every interface. The Host a client will send is not
# knowable for these, so rebinding protection cannot be pinned to one name.
_WILDCARD_HOSTS = frozenset({"", "0.0.0.0", "::", "[::]", "*"})


def is_wildcard_host(host: str) -> bool:
    """True when ``host`` binds every interface rather than one address."""
    return str(host).strip().lower() in _WILDCARD_HOSTS


def allowed_hosts_for(host: str, port: int) -> List[str]:
    """Host header values accepted by the DNS-rebinding check.

    A wildcard bind degrades to ``["*"]`` — with no single advertised name
    there is nothing to pin, and pinning the wrong one would reject every real
    client. ``soup mcp serve`` warns loudly in that case rather than pretending
    the check is still doing work.
    """
    if is_wildcard_host(host):
        return ["*"]
    names = [host]
    if host in ("127.0.0.1", "::1", "localhost"):
        names = ["127.0.0.1", "localhost", "::1"]
    # An IPv6 literal is bracketed in a Host header (`[::1]:8765`); joining it
    # with a bare colon would produce `::1:8765`, which matches nothing and
    # would 421 every IPv6 loopback client.
    authorities = [
        f"[{name}]:{port}" if ":" in name else f"{name}:{port}" for name in names
    ]
    return authorities + names


def _security_settings(host: str, port: int):
    """SDK transport-security settings (mcp >= 1.10) for this bind.

    DNS-rebinding protection is what stops a page the operator merely *visits*
    from driving a loopback MCP server through their browser; the Bearer gate
    alone would not, since a browser attaches no Authorization header but the
    request still reaches the port.
    """
    from mcp.server.transport_security import TransportSecuritySettings

    hosts = allowed_hosts_for(host, port)
    origins = ["*"] if hosts == ["*"] else [
        f"{scheme}://{name}" for name in hosts for scheme in ("http", "https")
    ]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=hosts,
        allowed_origins=origins,
    )


class _BearerAuthMiddleware:
    """Refuse any HTTP request without ``Authorization: Bearer <token>``.

    Pure ASGI rather than a starlette ``BaseHTTPMiddleware`` subclass: the SSE
    transport streams for the lifetime of a session, and ``BaseHTTPMiddleware``
    buffers through a response body it does not own. Non-HTTP scopes
    (``lifespan``, ``websocket``) pass straight through — gating ``lifespan``
    would stop the streamable-HTTP session manager from ever starting.
    """

    def __init__(self, app, token: str) -> None:
        self._app = app
        self._expected = f"Bearer {token}".encode()

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return
        supplied = b""
        for key, value in scope.get("headers", ()):
            if key.lower() == b"authorization":
                supplied = value
                break
        # compare_digest over the whole header, so the scheme is checked in
        # constant time along with the token.
        if not secrets.compare_digest(supplied, self._expected):
            from starlette.responses import PlainTextResponse

            response = PlainTextResponse(
                "Unauthorized",
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        await self._app(scope, receive, send)


def _sse_app(server: Server, security):
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.responses import Response
    from starlette.routing import Mount, Route

    sse = SseServerTransport(MESSAGE_PATH, security_settings=security)

    async def _handle_sse(request):
        # ``request._send`` is how the SDK's own examples drive this transport:
        # connect_sse needs the raw ASGI send, which Request does not expose.
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as (read_stream, write_stream):
            await server.run(
                read_stream, write_stream, server.create_initialization_options()
            )
        return Response(status_code=200)

    return Starlette(
        routes=[
            Route(SSE_PATH, endpoint=_handle_sse, methods=["GET"]),
            Mount(MESSAGE_PATH, app=sse.handle_post_message),
        ]
    )


def _http_app(server: Server, security):
    from contextlib import asynccontextmanager

    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.routing import Mount

    manager = StreamableHTTPSessionManager(
        app=server,
        json_response=False,
        stateless=False,
        security_settings=security,
    )

    async def _handle(scope, receive, send) -> None:
        await manager.handle_request(scope, receive, send)

    @asynccontextmanager
    async def _lifespan(app):
        # The manager refuses requests until its task group is running, so it
        # is owned by the app lifespan rather than by the first request.
        async with manager.run():
            yield

    return Starlette(routes=[Mount(HTTP_PATH, app=_handle)], lifespan=_lifespan)


def build_asgi_app(
    *,
    transport: str,
    allow_mutating: bool,
    allow_execute: bool,
    auth_token: str,
    host: str = DEFAULT_NETWORK_HOST,
    port: int = DEFAULT_NETWORK_PORT,
):
    """Build the authenticated ASGI app for a network transport.

    Split out from :func:`run_sse_server` / :func:`run_http_server` so the auth
    and rebinding behaviour is testable through an in-process ASGI transport
    without binding a socket.

    Raises:
        ValueError: for a transport that is not ``sse`` / ``http`` (``stdio``
            included — it is not an ASGI transport), or a token that is not
            urlsafe-base64 shaped.
            Also for ``allow_execute=True``: gated execution (#297) spawns
            real processes and is stdio-only, so no network transport may
            construct a registry that can execute.
    """
    from soup_cli.utils.qr_url import validate_token

    if transport not in NETWORK_TRANSPORTS:
        raise ValueError(
            f"transport must be one of {NETWORK_TRANSPORTS}, got {transport!r}"
        )
    # Refused here, not only in the CLI: a direct caller of build_asgi_app /
    # run_network_server must not be able to put an executing registry behind
    # a listener either. The CLI guard gives the operator a readable message;
    # this one makes the property structural.
    if allow_execute:
        raise ValueError(
            "allow_execute is not available over a network transport - gated "
            "execution spawns real training / export processes and is stdio-only"
        )
    token = validate_token(auth_token)

    server = build_server(build_registry(
        allow_mutating=allow_mutating,
        allow_execute=allow_execute,
        execution=ExecutionManager(),
    ))
    security = _security_settings(host, port)
    inner = _sse_app(server, security) if transport == "sse" else _http_app(server, security)
    return _BearerAuthMiddleware(inner, token)


def run_network_server(
    *,
    transport: str,
    allow_mutating: bool,
    allow_execute: bool,
    auth_token: str,
    host: str = DEFAULT_NETWORK_HOST,
    port: int = DEFAULT_NETWORK_PORT,
) -> None:
    """Serve MCP over ``sse`` or ``http`` until interrupted."""
    import uvicorn

    app = build_asgi_app(
        transport=transport,
        allow_mutating=allow_mutating,
        allow_execute=allow_execute,
        auth_token=auth_token,
        host=host,
        port=port,
    )
    uvicorn.run(app, host=host, port=port, log_level="warning")


def run_sse_server(**kwargs) -> None:
    """Serve MCP over HTTP+SSE (GET ``/sse``, POST ``/messages/``)."""
    run_network_server(transport="sse", **kwargs)


def run_http_server(**kwargs) -> None:
    """Serve MCP over the streamable-HTTP transport (``/mcp``)."""
    run_network_server(transport="http", **kwargs)
