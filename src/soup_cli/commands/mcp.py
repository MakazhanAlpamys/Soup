"""soup mcp — Model Context Protocol server (v0.71.28).

``soup mcp serve`` exposes Soup's read-only commands (plus two plan-only
mutating tools) to any MCP client (Claude Code / Cursor / Cline / Continue)
over stdio. The heavy ``mcp`` SDK is behind the ``[mcp]`` extra and imported
lazily, so this command errors friendly-ly when the extra is missing.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    no_args_is_help=True,
    help="Model Context Protocol server — drive Soup from any MCP client.",
)


@app.command()
def serve(
    allow_mutating: bool = typer.Option(
        False,
        "--allow-mutating",
        help=(
            "Enable the plan-only mutating tools (train_start / export). Even "
            "when enabled they only render the command that would run — v1 "
            "never executes training or export. Off by default: those tools "
            "refuse."
        ),
    ),
) -> None:
    """Start the stdio MCP server (read-only tools by default).

    stdout is the JSON-RPC channel — all human-facing output goes to stderr.
    Wire it into a client, e.g. `.mcp.json`:

        {"mcpServers": {"soup": {"command": "soup", "args": ["mcp", "serve"]}}}
    """
    from rich.console import Console

    # stderr-only: stdout is reserved for the MCP JSON-RPC stream.
    console = Console(stderr=True)

    try:
        from soup_cli.mcp_server.server import run_stdio_server
    except ImportError:
        console.print(
            "[red]The MCP server needs the 'mcp' SDK.[/] "
            "Install it with: [bold]pip install 'soup-cli[mcp]'[/]"
        )
        raise typer.Exit(1) from None

    mode = "mutating tools ENABLED (plan-only)" if allow_mutating else "read-only"
    console.print(f"[dim]soup mcp serve — stdio transport — {mode}. Waiting for a client...[/]")
    run_stdio_server(allow_mutating=allow_mutating)
