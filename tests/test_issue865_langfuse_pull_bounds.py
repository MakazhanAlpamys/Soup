"""#865 — bounds and filtering for `soup ingest --source langfuse --pull`.

Three findings from the #859 review, each reproduced here before it was fixed:

1. ``--pull``'s timeout reached the socket, not the call: a server that sent one
   byte before every timeout expired held the command open indefinitely.
2. ``type=GENERATION`` was only a query parameter, so a server that ignored it
   would have spans and tool calls written out as training rows.
3. A server repeating one pagination cursor was only stopped by ``--max-pages``,
   after spending the whole budget on identical requests.

The credential, redirect, host-validation and atomic-output guarantees from #859
are pinned in ``tests/test_issue204_langfuse_pull.py``; this file covers what
changed on top of them.
"""

from __future__ import annotations

import http.server
import json
import re
import threading
import time
from datetime import timedelta
from urllib.parse import parse_qs, urlsplit

import pytest

_PUBLIC = "pk-lf-SOUP865PUBLIC-1a2b"
_SECRET = "sk-lf-SOUP865SECRET-3c4d"

# The drip server sends a byte every _DRIP_GAP for _DRIP_CHUNKS: each gap is well
# inside the timeout, the total is far past it.
_TIMEOUT = 0.4
_DRIP_GAP = 0.1
_DRIP_CHUNKS = 20
_EPSILON = 0.6  # scheduling slack on a loaded CI runner


def _creds(host="https://cloud.langfuse.com"):
    from soup_cli.utils.ingest_pull import LangfuseCredentials

    return LangfuseCredentials(public_key=_PUBLIC, secret_key=_SECRET, host=host)


def _response(status=200, body=b"", headers=None):
    from soup_cli.utils.ingest_pull import HttpResponse

    return HttpResponse(status=status, headers=dict(headers or {}), body=body)


def _page(observations, cursor=None):
    meta = {"cursor": cursor} if cursor else {}
    return _response(body=json.dumps({"data": observations, "meta": meta}).encode())


def _observation(index, kind="GENERATION"):
    return {
        "id": f"obs-{index}",
        "type": kind,
        "input": f"q{index}",
        "output": f"a{index}",
        "model": "gpt-4o-mini",
    }


class _Transport:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = []

    def __call__(self, url, credentials, timeout):
        self.calls.append(url)
        return self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]


def _pull(transport, **kwargs):
    from soup_cli.utils.ingest_pull import pull_langfuse_generations

    kwargs.setdefault("since", timedelta(days=1))
    kwargs.setdefault("sleep", lambda seconds: None)
    return list(pull_langfuse_generations(_creds(), transport=transport, **kwargs))


@pytest.fixture
def drip_server(monkeypatch):
    for scheme in ("HTTP", "HTTPS", "ALL"):
        for name in (f"{scheme}_PROXY", f"{scheme.lower()}_proxy"):
            monkeypatch.delenv(name, raising=False)

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):  # noqa: N802 — http.server API
            route = urlsplit(self.path).path
            try:
                if route == "/drip-401":
                    # A 401 whose body arrives too slowly to read.
                    self.send_response(401)
                    self.send_header("Content-Length", str(_DRIP_CHUNKS))
                    self.end_headers()
                    for _ in range(_DRIP_CHUNKS):
                        time.sleep(_DRIP_GAP)
                        self.wfile.write(b"x")
                        self.wfile.flush()
                elif route == "/drip":
                    self.send_response(200)
                    self.send_header("Content-Length", str(_DRIP_CHUNKS))
                    self.end_headers()
                    for _ in range(_DRIP_CHUNKS):
                        time.sleep(_DRIP_GAP)
                        self.wfile.write(b"x")
                        self.wfile.flush()
                else:
                    body = b'{"data": [], "meta": {}}'
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


# ============================================================
# 1. A total deadline, not a per-socket-operation timeout
# ============================================================


class TestDeadline:
    def test_a_drip_feed_cannot_outlast_the_timeout(self, drip_server):
        """Each gap is inside the socket timeout, so only a wall-clock deadline
        ends this: the whole response takes _DRIP_CHUNKS * _DRIP_GAP."""
        from soup_cli.utils.ingest_pull import PullError, _urllib_transport

        started = time.monotonic()
        raised = None
        try:
            _urllib_transport(f"{drip_server}/drip", _creds(), _TIMEOUT)
        except PullError as exc:  # what the deadline must produce
            raised = exc
        elapsed = time.monotonic() - started

        assert elapsed < _TIMEOUT + _EPSILON, (
            f"a {_TIMEOUT}s timeout allowed a {elapsed:.2f}s call "
            f"({_DRIP_CHUNKS} chunks {_DRIP_GAP}s apart)"
        )
        assert raised is not None and "timed out" in str(raised).lower()

    def test_a_drip_fed_error_body_still_reports_its_status(self, drip_server):
        """The deadline raises PullError, which is not an OSError: without it in
        the HTTPError branch, a slow 401 body loses the credentials message and
        surfaces as a timeout instead."""
        from soup_cli.utils.ingest_pull import PullError, _decode_page, _urllib_transport

        started = time.monotonic()
        response = _urllib_transport(f"{drip_server}/drip-401", _creds(), _TIMEOUT)
        elapsed = time.monotonic() - started

        assert (response.status, response.body) == (401, b"")
        assert elapsed < _TIMEOUT + _EPSILON
        with pytest.raises(PullError) as info:
            _decode_page(response, _creds())
        assert "LANGFUSE_PUBLIC_KEY" in str(info.value)

    def test_control_a_prompt_response_still_arrives(self, drip_server):
        """CONTROL — the deadline must not break a normal read."""
        from soup_cli.utils.ingest_pull import _urllib_transport

        response = _urllib_transport(f"{drip_server}/ok", _creds(), _TIMEOUT)

        assert (response.status, json.loads(response.body)["data"]) == (200, [])


# ============================================================
# 2. GENERATION filtered client-side, not only by query parameter
# ============================================================


class TestGenerationFilter:
    def _mixed_page(self):
        return _page([
            _observation(1, "GENERATION"),
            _observation(2, "SPAN"),
            _observation(3, "TOOL"),
            _observation(4, "GENERATION"),
        ])

    def test_only_generations_become_rows(self):
        """A server that ignores `type=GENERATION` must not get spans and tool
        calls written out as training rows."""
        rows = _pull(_Transport(self._mixed_page()))

        assert [row["id"] for row in rows] == ["obs-1", "obs-4"]

    def test_control_the_request_still_asks_the_server_for_generations(self):
        """CONTROL — the client-side check is defence in depth, not a replacement:
        dropping the query parameter would fetch every observation type."""
        transport = _Transport(self._mixed_page())

        _pull(transport)

        query = parse_qs(urlsplit(transport.calls[0]).query)
        assert query["type"] == ["GENERATION"]

    def test_the_cli_reports_them_as_skipped(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.utils import ingest_pull

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", _PUBLIC)
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", _SECRET)
        for name in ("LANGFUSE_HOST", "LANGFUSE_BASE_URL", "SOUP_TELEMETRY"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(ingest_pull, "_urllib_transport", _Transport(self._mixed_page()))

        result = CliRunner().invoke(
            app, ["ingest", "--source", "langfuse", "--pull", "--output", "out.jsonl"]
        )

        assert result.exit_code == 0, result.output
        output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "Wrote 2 traces from langfuse (2 generations pulled)" in output
        assert "2 observation(s) were not generations and were skipped" in output
        rows = (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["trace_id"] for line in rows] == ["obs-1", "obs-4"]

    def test_non_generation_observations_are_counted(self):
        from soup_cli.utils.ingest_pull import PullStats, pull_langfuse_generations

        stats = PullStats()
        rows = list(
            pull_langfuse_generations(
                _creds(),
                since=timedelta(days=1),
                transport=_Transport(self._mixed_page()),
                stats=stats,
            )
        )

        assert len(rows) == 2
        assert (stats.observations, stats.generations, stats.skipped_not_generation) == (4, 2, 2)


# ============================================================
# 3. A repeated cursor stops the pull instead of burning the budget
# ============================================================


class TestCursorLoop:
    def test_a_repeated_cursor_stops_at_the_second_page(self):
        from soup_cli.utils.ingest_pull import PullError

        transport = _Transport(_page([_observation(1)], cursor="same-cursor"))

        with pytest.raises(PullError) as info:
            _pull(transport, max_pages=100)

        assert len(transport.calls) == 2, "must not spend the rest of --max-pages"
        assert "cursor" in str(info.value).lower()

    def test_control_distinct_cursors_still_page(self):
        """CONTROL — the loop detector must not stop ordinary pagination."""
        transport = _Transport(
            _page([_observation(1)], cursor="c1"),
            _page([_observation(2)], cursor="c2"),
            _page([_observation(3)]),
        )

        rows = _pull(transport)

        assert [row["id"] for row in rows] == ["obs-1", "obs-2", "obs-3"]
        assert len(transport.calls) == 3
