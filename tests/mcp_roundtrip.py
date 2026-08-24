"""A client/server MCP session that works on both mcp 1.x and 2.x (#322).

Not named ``test_*`` on purpose: pytest must not collect it.

``mcp.shared.memory.create_connected_server_and_client_session`` was removed in
2.0.0, and every round-trip test in ``test_v07128.py`` was built on it. The
lower-level primitive it was built from — ``create_client_server_memory_streams``
— survives on both majors with the same shape, so this helper rebuilds the
session on top of that instead of branching on a version.

That is the point: a version branch here would have to be revisited at 3.0, and
the thing it would be branching on is not the thing that changed. What changed
is one convenience wrapper; the transport underneath it did not.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import anyio


@asynccontextmanager
async def connected_session(server, *, raise_exceptions: bool = False):
    """Yield a ``ClientSession`` talking to ``server`` over in-memory streams.

    Mirrors what the removed helper did: run the server in a task group against
    one end of a memory stream pair, hand the client the other end, and cancel
    the server task on exit so a hung handler cannot wedge the suite.
    """
    from mcp.client.session import ClientSession
    from mcp.shared.memory import create_client_server_memory_streams

    async with create_client_server_memory_streams() as (client_streams, server_streams):
        client_read, client_write = client_streams
        server_read, server_write = server_streams

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(
                lambda: server.run(
                    server_read,
                    server_write,
                    server.create_initialization_options(),
                    raise_exceptions=raise_exceptions,
                )
            )
            try:
                async with ClientSession(client_read, client_write) as session:
                    await session.initialize()
                    yield session
            finally:
                # The server task never returns on its own -- it waits for the
                # stream to close. Without this the task group blocks forever.
                task_group.cancel_scope.cancel()


def is_error(result) -> bool:
    """Read a ``CallToolResult``'s error flag on either mcp major (#322).

    1.x exposes the wire name ``isError``; 2.x exposes ``is_error``. The value
    means the same thing on both, so the tests ask for the meaning rather than
    for a spelling. Raising on neither is deliberate: silently returning False
    would turn "the SDK renamed the field again" into "no tool call ever
    failed", which is the failure mode a test suite must not have.
    """
    for attribute in ("isError", "is_error"):
        if hasattr(result, attribute):
            return bool(getattr(result, attribute))
    raise AttributeError(
        f"{type(result).__name__} exposes neither isError nor is_error; "
        "the mcp SDK renamed the field again (#322)"
    )


def input_schema(tool) -> dict:
    """Read an advertised ``Tool``'s schema on either mcp major (#322).

    Same rename as :func:`is_error`: the wire name is ``inputSchema`` and 2.x
    exposes it as ``input_schema``. Raises rather than returning ``{}`` for the
    same reason -- an empty schema would read as "the tool advertises nothing"
    instead of "the field moved".
    """
    for attribute in ("inputSchema", "input_schema"):
        if hasattr(tool, attribute):
            return getattr(tool, attribute)
    raise AttributeError(
        f"{type(tool).__name__} exposes neither inputSchema nor input_schema; "
        "the mcp SDK renamed the field again (#322)"
    )
