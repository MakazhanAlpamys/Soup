"""Soup MCP server (v0.71.28).

``soup mcp serve`` makes the ungated tools callable by default over stdio,
SSE, and Streamable HTTP. Startup flags enable planning and stdio execution.

The package is split so the tool *table* (:mod:`registry`) is pure Python
with NO dependency on the ``mcp`` SDK — it is fully unit-testable on the
light core install. Only :mod:`server` imports the SDK, lazily.
"""
