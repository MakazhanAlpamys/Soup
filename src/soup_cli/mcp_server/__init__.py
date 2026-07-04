"""Soup MCP server (v0.71.28).

``soup mcp serve`` exposes Soup's read-only commands (plus two plan-only
mutating tools) to any Model Context Protocol client over stdio.

The package is split so the tool *table* (:mod:`registry`) is pure Python
with NO dependency on the ``mcp`` SDK — it is fully unit-testable on the
light core install. Only :mod:`server` imports the SDK, lazily.
"""
