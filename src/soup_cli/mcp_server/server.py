"""MCP stdio server wiring for ``soup mcp serve`` (v0.71.28).

This is the ONLY module that imports the ``mcp`` SDK — importing it therefore
requires the ``[mcp]`` extra. The pure tool table lives in
:mod:`soup_cli.mcp_server.registry` (no SDK dependency, fully unit-testable).
"""

from __future__ import annotations

import json
import sys
from contextlib import redirect_stdout
from typing import List

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from soup_cli.mcp_server.registry import McpToolError, ToolSpec, _sanitize, build_registry

SERVER_NAME = "soup"


def build_server(specs: List[ToolSpec]) -> Server:
    """Build a low-level MCP :class:`Server` that dispatches to ``specs``.

    Each tool result is a plain ``dict`` from the handler; it is sanitized
    (C0/ESC-stripped) and returned as a single pretty-printed JSON
    ``TextContent`` block. Handler failures become an ``isError`` result with a
    path-free message so the server survives bad calls.
    """
    by_name = {spec.name: spec for spec in specs}
    server: Server = Server(SERVER_NAME)

    @server.list_tools()
    async def _list_tools() -> List[types.Tool]:
        return [
            types.Tool(
                name=spec.name,
                title=spec.title,
                description=spec.description,
                inputSchema=spec.input_schema,
            )
            for spec in specs
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict) -> List[types.TextContent]:
        spec = by_name.get(name)
        if spec is None:
            # The SDK stringifies this into an isError result; keep it generic.
            raise ValueError("unknown tool")
        try:
            # Any core that prints (e.g. a Rich warning) must not corrupt the
            # JSON-RPC stdout channel — send stray stdout to stderr for the
            # duration of the (synchronous) handler call.
            with redirect_stdout(sys.stderr):
                result = spec.handler(arguments or {})
        except McpToolError as exc:
            raise ValueError(str(exc)) from None
        except Exception as exc:  # never leak a stack trace / path to the client
            raise ValueError(f"internal error ({type(exc).__name__})") from None
        text = json.dumps(_sanitize(result), indent=2, ensure_ascii=False)
        return [types.TextContent(type="text", text=text)]

    return server


def run_stdio_server(*, allow_mutating: bool) -> None:
    """Run the MCP server over stdio until the client disconnects."""
    server = build_server(build_registry(allow_mutating=allow_mutating))

    async def _main() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    anyio.run(_main)
