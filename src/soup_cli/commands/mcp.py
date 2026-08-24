"""soup mcp - Model Context Protocol server (v0.71.28; network transports #296).

``soup mcp serve`` exposes Soup's read-only commands (plus two plan-only
mutating tools) to any MCP client (Claude Code / Cursor / Cline / Continue).
stdio remains the default and is unchanged; ``--transport sse`` and
``--transport http`` add a local listener for remote or multi-client setups.
The heavy ``mcp`` SDK is behind the ``[mcp]`` extra and imported lazily, so this
command errors friendly-ly when the extra is missing.

All user-facing help text stays ASCII (Windows terminal / cp1252 safety;
mirrors the ``test_help_output_is_ascii_safe`` guard).
"""

from __future__ import annotations

import secrets
from typing import Optional

import typer

app = typer.Typer(
    no_args_is_help=True,
    help="Model Context Protocol server - drive Soup from any MCP client.",
)

_STDIO = "stdio"
_TRANSPORTS = ("stdio", "sse", "http")
_LOOPBACK = ("127.0.0.1", "::1", "localhost")


def _resolve_auth_token(auth_token: Optional[str]) -> str:
    """Return the operator's token (validated), or mint a fresh one.

    Reuses ``utils.qr_url.validate_token`` so ``soup ui`` and ``soup mcp serve``
    agree on what a token is (16-128 urlsafe-base64 chars) instead of growing a
    second bespoke format.
    """
    from soup_cli.utils.qr_url import validate_token

    if auth_token is None:
        return secrets.token_urlsafe(32)
    return validate_token(auth_token)


@app.command()
def serve(
    allow_mutating: bool = typer.Option(
        False,
        "--allow-mutating",
        help=(
            "Enable the plan-only mutating tools (train_start / export). Even "
            "when enabled they only render the command that would run - v1 "
            "never executes training or export. Off by default: those tools "
            "refuse."
        ),
    ),
    allow_execute: bool = typer.Option(
        False,
        "--allow-execute",
        help=(
            "Reserve the execution gate for future MCP tools. Implies "
            "--allow-mutating, but train_start and export remain plan-only "
            "and never execute in this version."
        ),
    ),
    transport: str = typer.Option(
        _STDIO,
        "--transport",
        help=(
            "stdio (default), sse, or http. stdio speaks JSON-RPC over the "
            "process pipes and is what a client that spawns Soup wants. sse "
            "and http open a local listener instead, for remote or "
            "multi-client setups, and require a Bearer token."
        ),
    ),
    host: Optional[str] = typer.Option(
        None,
        "--host",
        help=(
            "Interface to bind for --transport sse/http. Defaults to "
            "127.0.0.1. Binding anything else exposes the server to the "
            "network, where the Bearer token is the only gate."
        ),
    ),
    port: Optional[int] = typer.Option(
        None,
        "--port",
        help="Port to bind for --transport sse/http. Defaults to 8765.",
    ),
    auth_token: Optional[str] = typer.Option(
        None,
        "--auth-token",
        help=(
            "Override the auto-generated Bearer token (16-128 urlsafe "
            "base64 chars) for --transport sse/http. When omitted, a fresh "
            "token is generated each startup and printed."
        ),
    ),
) -> None:
    """Start the MCP server (read-only tools by default).

    Under the default stdio transport, stdout is the JSON-RPC channel - all
    human-facing output goes to stderr. Wire it into a client, e.g. `.mcp.json`:

        {"mcpServers": {"soup": {"command": "soup", "args": ["mcp", "serve"]}}}

    The sse / http transports bind 127.0.0.1 and require
    'Authorization: Bearer <token>' on every request.
    """
    from rich.console import Console
    from rich.markup import escape

    # stderr-only: stdout is reserved for the MCP JSON-RPC stream.
    console = Console(stderr=True)

    if transport not in _TRANSPORTS:
        console.print(
            f"[red]--transport must be one of: {', '.join(_TRANSPORTS)}[/] "
            f"(got '{escape(str(transport))}')"
        )
        raise typer.Exit(2)

    # Flags that cannot mean anything under stdio are an error, not a silently
    # ignored argument: stdio has no listener to bind and nothing to authorize.
    inapplicable = [
        name
        for name, value in (
            ("--host", host),
            ("--port", port),
            ("--auth-token", auth_token),
        )
        if value is not None
    ]
    if transport == _STDIO and inapplicable:
        console.print(
            f"[red]{', '.join(inapplicable)} apply to --transport sse/http only[/] - "
            "stdio has no listener to bind and nothing to authorize."
        )
        raise typer.Exit(2)

    # Execution over a network transport is refused outright rather than
    # warned about. --allow-execute spawns real training / export processes
    # (#297); behind a listener that is one Bearer token away from the
    # network, a leaked token would mean process execution, not just plan
    # disclosure. stdio is a pipe to a client the operator already started,
    # which is a different trust boundary. This combination has never
    # shipped, so refusing it takes nothing away.
    if transport != _STDIO and allow_execute:
        console.print(
            "[red]--allow-execute is refused with --transport "
            f"{escape(str(transport))}[/] - execution is available over stdio "
            "only. A network listener is one Bearer token away from spawning "
            "real training / export processes; run the executing server over "
            "stdio, or drop --allow-execute to serve plan-only tools here."
        )
        raise typer.Exit(2)

    # Only what stdio needs. Pulling the network symbols in here as well would
    # make the default path depend on names it never uses, and any stand-in
    # module that provides `run_stdio_server` alone would fail to import.
    try:
        from soup_cli.mcp_server.server import run_stdio_server
    except ImportError:
        # NB: escape the '[' in 'soup-cli[mcp]' so Rich prints it literally
        # instead of parsing '[mcp]' as a (dropped) markup tag.
        console.print(
            "[red]The MCP server needs the 'mcp' SDK.[/] "
            "Install it with: [bold]pip install \"soup-cli\\[mcp]\"[/]"
        )
        raise typer.Exit(1) from None

    # --allow-execute is the stronger opt-in and implies --allow-mutating.
    # It reaches here only under stdio: the network branch refused above.
    allow_mutating = allow_mutating or allow_execute
    if allow_execute:
        mode = "mutating tools ENABLED (execution ENABLED)"
    elif allow_mutating:
        mode = "mutating tools ENABLED (plan-only)"
    else:
        mode = "read-only"

    if transport == _STDIO:
        console.print(
            f"[dim]soup mcp serve - stdio transport - {mode}. Waiting for a client...[/]"
        )
        run_stdio_server(allow_mutating=allow_mutating, allow_execute=allow_execute)
        return

    try:
        token = _resolve_auth_token(auth_token)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]--auth-token:[/] {escape(str(exc))}")
        raise typer.Exit(2) from exc

    try:
        from soup_cli.mcp_server.server import (
            DEFAULT_NETWORK_HOST,
            DEFAULT_NETWORK_PORT,
            HTTP_PATH,
            SSE_PATH,
            is_wildcard_host,
            run_network_server,
        )
    except ImportError:
        console.print(
            "[red]--transport sse/http needs a newer 'mcp' SDK.[/] "
            "Upgrade with: [bold]pip install -U \"soup-cli\\[mcp]\"[/] "
            "(streamable HTTP landed in mcp 1.8.0, rebinding protection in 1.10.0)"
        )
        raise typer.Exit(1) from None

    bind_host = DEFAULT_NETWORK_HOST if host is None else host
    bind_port = DEFAULT_NETWORK_PORT if port is None else port

    # Route from the server module, so the printed URL cannot drift from it.
    path = SSE_PATH if transport == "sse" else HTTP_PATH
    console.print(
        f"[dim]soup mcp serve - {transport} transport - {mode}[/]\n"
        f"[dim]URL:[/]   http://{bind_host}:{bind_port}{path}\n"
        f"[dim]Token:[/] {token}\n"
        f"[dim]Every request needs: Authorization: Bearer {token}[/]"
    )
    if is_wildcard_host(bind_host) or bind_host not in _LOOPBACK:
        console.print(
            f"[yellow]WARNING:[/] bound to {escape(bind_host)}, not loopback - this "
            "server is reachable from the network and the Bearer token is the "
            "only thing standing in front of it."
        )
    if is_wildcard_host(bind_host):
        console.print(
            "[yellow]WARNING:[/] a wildcard bind has no single Host to pin, so "
            "DNS-rebinding protection degrades to accepting any Host header."
        )

    run_network_server(
        transport=transport,
        allow_mutating=allow_mutating,
        allow_execute=allow_execute,
        auth_token=token,
        host=bind_host,
        port=bind_port,
    )
