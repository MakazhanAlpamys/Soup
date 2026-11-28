"""Issue #3301 — pin the MCP dataset path refusal across transports.

`data_inspect` and `data_validate` share one local data path guard. When the
supplied absolute path names a file outside the server working directory, the
guard returns the tool error `'data' must be a readable file under the working
directory`. Clients rely on that exact wording, so these tests pin it over
actual stdio, SSE and Streamable HTTP connections — not just the in-memory
transport — plus a success control for a valid file inside the working
directory on each transport.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from tests.test_issue296_network_transport import (
    TOKEN,
    _build,
    _free_port,
    _UvicornThread,
)

ERROR = "'data' must be a readable file under the working directory"

_ROWS = [
    {"instruction": "Summarize this article", "output": "A short summary."},
    {"instruction": "Translate to French", "output": "Bonjour le monde."},
    {"instruction": "Write a poem about spring", "output": "Petals fall softly."},
    {"instruction": "Explain gravity", "output": "Mass attracts mass."},
    {"instruction": "List three fruits", "output": "Apple, pear, plum."},
]

TOOLS = ["data_inspect", "data_validate"]
TRANSPORTS = ["stdio", "sse", "http"]


def _run(coro):
    return asyncio.run(coro)


def _loopback_client_factory(**kwargs):
    """httpx client factory for loopback tests: never consult proxy env vars.

    The SDK's default factory builds a client that reads proxy env vars, and
    this VM's proxy entries break httpx's URL parsing. Loopback needs no proxy.
    The client is built from the httpx the installed SDK itself uses (plain
    httpx on 1.x, httpx2 on 2.x), so the timeout object the SDK passes in
    matches the client's own types.
    """
    from mcp.shared import _httpx_utils

    httpx = getattr(_httpx_utils, "httpx2", None) or _httpx_utils.httpx
    # follow_redirects stays on: the 1.x transport relies on the client to
    # follow the server's /mcp -> /mcp/ redirect, and the 2.x transports do
    # not consult the setting.
    return httpx.AsyncClient(trust_env=False, follow_redirects=True, **kwargs)


@pytest.fixture(autouse=True)
def _need_mcp():
    # Mirrors test_v07128.py: the SDK only exists with the [mcp] extra.
    pytest.importorskip("mcp")


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """A working directory and a sibling directory outside it."""
    work = tmp_path / "work"
    outside = tmp_path / "outside"
    work.mkdir()
    outside.mkdir()
    _write_jsonl(outside / "data.jsonl", _ROWS)
    _write_jsonl(work / "data.jsonl", _ROWS)
    monkeypatch.chdir(work)
    return work, outside


def _call_stdio(work, tool, args):
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    repo_src = Path(__file__).resolve().parents[1] / "src"
    existing = [
        p
        for p in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if p and Path(p).is_absolute()
    ]
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "soup_cli", "mcp", "serve"],
        env={"PYTHONPATH": os.pathsep.join([str(repo_src), *existing])},
        cwd=str(work),
    )

    async def _go():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(tool, args)

    return _run(_go())


async def _sse_session(port, token, tool, args):
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    url = f"http://127.0.0.1:{port}/sse"
    async with sse_client(
        url,
        headers={"Authorization": f"Bearer {token}"},
        httpx_client_factory=_loopback_client_factory,
    ) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(tool, args)


async def _http_session(port, token, tool, args):
    from contextlib import asynccontextmanager

    from mcp import ClientSession

    @asynccontextmanager
    async def _streams():
        # Capability probe (#322): 2.x renamed streamablehttp_client to
        # streamable_http_client, moved headers onto an httpx client, and
        # dropped get_session_id from the yield. Ask the installed SDK which
        # shape it has instead of branching on a version number.
        from mcp.client import streamable_http as _sh

        url = f"http://127.0.0.1:{port}/mcp"
        if hasattr(_sh, "streamable_http_client"):
            import httpx

            # Loopback only: never consult proxy env vars (httpx would parse
            # them and can fail on entries it does not understand).
            async with httpx.AsyncClient(
                headers={"Authorization": f"Bearer {token}"}, trust_env=False
            ) as http_client:
                async with _sh.streamable_http_client(
                    url, http_client=http_client
                ) as (read, write):
                    yield read, write
        else:
            async with _sh.streamablehttp_client(
                url,
                headers={"Authorization": f"Bearer {token}"},
                httpx_client_factory=_loopback_client_factory,
            ) as (read, write, _session_id):
                yield read, write

    async with _streams() as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(tool, args)


def _call(transport, dirs, tool, args):
    work, _outside = dirs
    if transport == "stdio":
        return _call_stdio(work, tool, args)
    port = _free_port()
    app = _build(transport, port=port)
    with _UvicornThread(app, port):
        if transport == "sse":
            return _run(_sse_session(port, TOKEN, tool, args))
        return _run(_http_session(port, TOKEN, tool, args))


@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("transport", TRANSPORTS)
def test_refusal_is_exact_across_transports(dirs, tool, transport):
    from tests.mcp_roundtrip import is_error

    _work, outside = dirs
    result = _call(transport, dirs, tool, {"data": str(outside / "data.jsonl")})
    assert is_error(result) is True
    assert len(result.content) == 1
    text = result.content[0].text
    assert text == ERROR
    assert str(outside / "data.jsonl") not in text


@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("transport", TRANSPORTS)
def test_valid_file_inside_workdir_succeeds(dirs, tool, transport):
    from tests.mcp_roundtrip import is_error

    work, _outside = dirs
    result = _call(transport, dirs, tool, {"data": str(work / "data.jsonl")})
    assert is_error(result) is False
    payload = json.loads(result.content[0].text)
    if tool == "data_inspect":
        assert payload["total"] == 5
    else:
        assert "issues" in payload and "valid_rows" in payload
