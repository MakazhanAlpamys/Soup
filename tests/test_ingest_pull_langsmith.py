"""#204 — `soup ingest --source langsmith --pull`: a live pull from LangSmith.

The LangSmith slice of #204, after the Langfuse one (#859, #865). It has its own
module, ``utils/ingest_pull_langsmith.py``: LangSmith's request is a POST with a
JSON body, an ``X-API-Key`` header and the cursor in the body, so sharing the
Langfuse GET transport would mean editing the Langfuse code. ``ingest_pull.py``
is used as it is: ``parse_since``, ``_parse_retry_after``, ``_read_capped`` and
the error classes. ``TestSharedHelpers`` reaches each of those through the
LangSmith path, so a change in ``ingest_pull.py`` that breaks LangSmith fails
here and not only in the Langfuse files.

What this file pins, as the Langfuse files do for Langfuse:

- one row per LLM run: ``run_type`` is sent as a filter and checked again on
  each run, so a server that ignores it cannot turn chain or tool runs into
  training rows;
- the API key never reaches an artefact: stdout / stderr under
  ``--verbose --log-level debug``, the output file, the audit log, a webhook
  payload, or a traceback that renders locals;
- ``LANGSMITH_ENDPOINT`` goes through the existing SSRF validator, is
  HTTPS-only, and a private / loopback address needs ``--allow-private-host``;
- every loop is bounded, and a cap stops loudly instead of truncating;
- without ``--pull`` nothing changes and no pull module is imported.

The fake transports below do not prove the wire shape. That is pinned by the
recorded page of #2868 (``tests/fixtures/issue2868_langsmith_runs_v2.json``):
real ``POST /api/v2/runs/query`` items from a LangSmith Cloud project, which go
through the pull and then through ``parse_langsmith``.
"""

from __future__ import annotations

import http.server
import io
import json
import logging
import os
import subprocess
import sys
import threading
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from tests._loopback_http import LoopbackHTTPServer

_KEY = "lsv2_pt_SOUPPLANTEDLANGSMITHKEY_0123456789abcdef"
_TOKENS = (_KEY,)
_PROJECT = "soup-pull-test"
_PROJECT_ID = "00000000-0000-4000-8000-000000000001"
_HOST = "https://api.smith.langchain.com"

_NOW = datetime(2026, 11, 15, 12, 0, 0, tzinfo=timezone.utc)

_RECORDED_V2 = Path(__file__).resolve().parent / "fixtures" / "issue2868_langsmith_runs_v2.json"


def _settings(host=_HOST, project=_PROJECT):
    from soup_cli.utils.ingest_pull_langsmith import LangSmithSettings

    return LangSmithSettings(api_key=_KEY, host=host, project=project)


def _response(status=200, body=b"", headers=None):
    from soup_cli.utils.ingest_pull import HttpResponse

    return HttpResponse(status=status, headers=dict(headers or {}), body=body)


def _json(payload, status=200, headers=None):
    return _response(status, json.dumps(payload).encode(), headers)


def _sessions(project=_PROJECT, project_id=_PROJECT_ID):
    return _json([{"id": project_id, "name": project}])


def _page(runs, cursor=None):
    return _json({"items": runs, "next_cursor": cursor})


def _run(index, run_type="llm"):
    """Control-flow filler only: the wire shape is pinned by the recorded page."""
    return {
        "id": f"run-{index}",
        "run_type": run_type,
        "inputs": {"messages": [{"role": "user", "content": f"q{index}"}]},
        "outputs": {"choices": [{"message": {"role": "assistant", "content": f"a{index}"}}]},
    }


class _Transport:
    """Answers the project lookup, then replays scripted runs pages; records every call."""

    def __init__(self, *pages, sessions=None):
        self._sessions = sessions if sessions is not None else _sessions()
        self._pages = list(pages)
        self.calls = []

    def __call__(self, method, url, body, settings, timeout):
        self.calls.append(
            {"method": method, "url": url, "body": body, "settings": settings, "timeout": timeout}
        )
        if urlsplit(url).path.endswith("/sessions"):
            item = self._sessions.pop(0) if isinstance(self._sessions, list) else self._sessions
        else:
            item = self._pages.pop(0) if len(self._pages) > 1 else self._pages[0]
        if isinstance(item, BaseException):
            raise item
        return item

    def query_calls(self):
        return [call for call in self.calls if call["url"].endswith("/runs/query")]

    def query(self, index):
        return json.loads(self.query_calls()[index]["body"])


def _pull(transport, **kwargs):
    from soup_cli.utils.ingest_pull_langsmith import pull_langsmith_runs

    kwargs.setdefault("since", timedelta(days=1))
    kwargs.setdefault("now", _NOW)
    kwargs.setdefault("sleep", lambda seconds: None)
    return list(pull_langsmith_runs(_settings(), transport=transport, **kwargs))


# ============================================================
# The request: project lookup, then Runs API v2, LLM runs, with the fields
# ============================================================


class TestRequest:
    def test_resolves_the_project_then_queries_llm_runs_with_the_fields(self):
        """v2 returns only ``id`` unless ``selects`` asks for more (seen on a LangSmith
        Cloud account on 2026-11-15, and in langsmith-py 0.14.17's ``query_v2``)."""
        transport = _Transport(_page([_run(1)]))

        _pull(transport, since=timedelta(hours=6), timeout=12.5)

        lookup, query = transport.calls
        parts = urlsplit(lookup["url"])
        assert (lookup["method"], lookup["body"]) == ("GET", None)
        assert f"{parts.scheme}://{parts.netloc}{parts.path}" == f"{_HOST}/api/v1/sessions"
        assert parse_qs(parts.query) == {"name": [_PROJECT], "limit": ["1"]}
        assert query["method"] == "POST"
        assert query["url"] == f"{_HOST}/api/v2/runs/query"
        body = transport.query(0)
        assert body["project_ids"] == [_PROJECT_ID]
        assert body["run_type"] == "LLM"
        assert {"ID", "RUN_TYPE", "INPUTS", "OUTPUTS", "FEEDBACK_STATS"} <= set(body["selects"])
        # Without min_start_time the API covers only the last day, whatever --since says.
        assert body["min_start_time"] == "2026-11-15T06:00:00Z"
        assert body["max_start_time"] == "2026-11-15T12:00:00Z"
        assert 1 <= body["page_size"] <= 1000
        assert "cursor" not in body
        assert {call["timeout"] for call in transport.calls} == {12.5}

    def test_a_project_that_does_not_exist_stops_before_any_runs_query(self):
        from soup_cli.utils.ingest_pull import PullError

        transport = _Transport(_page([_run(1)]), sessions=_json([]))

        with pytest.raises(PullError) as info:
            _pull(transport)

        assert "LANGSMITH_PROJECT" in str(info.value) and _PROJECT in str(info.value)
        assert transport.query_calls() == []

    def test_a_lookup_answer_for_another_project_is_not_taken(self):
        """The name is a query parameter; the answer is checked like run_type is."""
        from soup_cli.utils.ingest_pull import PullError

        transport = _Transport(_page([_run(1)]), sessions=_sessions(project="someone-else"))

        with pytest.raises(PullError):
            _pull(transport)

        assert transport.query_calls() == []

    @pytest.mark.parametrize(
        "body",
        [
            b"not json",
            b"{}",
            b'[{"name": "soup-pull-test"}]',
            b'[{"name": "soup-pull-test", "id": 7}]',
        ],
        ids=["not-json", "not-a-list", "no-id", "id-not-a-string"],
    )
    def test_an_unexpected_lookup_answer_raises(self, body):
        from soup_cli.utils.ingest_pull import PullError

        with pytest.raises(PullError):
            _pull(_Transport(_page([]), sessions=_response(200, body=body)))


# ============================================================
# Only LLM runs become rows
# ============================================================


class TestRunTypeFilter:
    def test_only_llm_runs_become_rows(self):
        from soup_cli.utils.ingest_pull_langsmith import LangSmithPullStats

        stats = LangSmithPullStats()
        transport = _Transport(
            _page([_run(1), _run(2, "chain"), _run(3, "tool"), _run(4, "LLM"), {"id": "x"}, "junk"])
        )

        rows = _pull(transport, stats=stats)

        assert [row["id"] for row in rows] == ["run-1", "run-4"]
        assert (stats.pages, stats.runs, stats.llm_runs, stats.skipped_not_llm) == (1, 6, 2, 4)

    def test_control_the_request_still_asks_the_server_for_llm_runs(self):
        transport = _Transport(_page([_run(1)]))

        _pull(transport)

        assert transport.query(0)["run_type"] == "LLM"


# ============================================================
# Pagination and caps
# ============================================================


class TestPagination:
    def test_follows_the_cursor_until_it_is_absent(self):
        transport = _Transport(
            _page([_run(1), _run(2)], cursor="c1"), _page([_run(3)], cursor="c2"), _page([_run(4)])
        )

        rows = _pull(transport)

        assert [row["id"] for row in rows] == ["run-1", "run-2", "run-3", "run-4"]
        assert len(transport.query_calls()) == 3
        assert [transport.query(i).get("cursor") for i in range(3)] == [None, "c1", "c2"]
        windows = {
            (transport.query(i)["min_start_time"], transport.query(i)["max_start_time"])
            for i in range(3)
        }
        assert len(windows) == 1  # fixed before the first request
        assert sum(call["url"].endswith("/sessions?name=soup-pull-test&limit=1")
                   for call in transport.calls) == 1

    def test_page_cap_stops_loudly_while_results_are_pending(self):
        from soup_cli.utils.ingest_pull import PullLimitError

        transport = _Transport(_page([_run(1)], cursor="c1"), _page([_run(2)], cursor="c2"))

        with pytest.raises(PullLimitError) as info:
            _pull(transport, max_pages=2)

        assert len(transport.query_calls()) == 2
        assert (info.value.pages, info.value.rows) == (2, 2)
        assert "--max-pages" in str(info.value) and "--since" in str(info.value)

    def test_control_reaching_the_cap_on_the_last_page_is_not_an_error(self):
        transport = _Transport(_page([_run(1)], cursor="c1"), _page([_run(2)]))

        assert len(_pull(transport, max_pages=2)) == 2

    def test_row_cap_stops_loudly(self, monkeypatch):
        from soup_cli.utils import ingest_pull_langsmith
        from soup_cli.utils.ingest_pull import PullLimitError

        monkeypatch.setattr(ingest_pull_langsmith, "_MAX_INGEST_LINES", 3)
        transport = _Transport(_page([_run(1), _run(2)], cursor="c1"), _page([_run(3), _run(4)]))

        with pytest.raises(PullLimitError):
            _pull(transport)

    def test_a_repeated_cursor_stops_at_the_second_page(self):
        from soup_cli.utils.ingest_pull import PullError, PullLimitError

        transport = _Transport(_page([_run(1)], cursor="same"))

        with pytest.raises(PullError) as info:
            _pull(transport, max_pages=50)

        assert not isinstance(info.value, PullLimitError)
        assert len(transport.query_calls()) == 2
        assert "cursor" in str(info.value)

    @pytest.mark.parametrize("max_pages", [0, -1, True, 10_001])
    def test_max_pages_outside_bounds_is_rejected(self, max_pages):
        with pytest.raises((TypeError, ValueError)):
            _pull(_Transport(_page([])), max_pages=max_pages)

    @pytest.mark.parametrize(
        "body",
        [
            b"not json",
            b"[]",
            b'{"next_cursor": null}',
            b'{"items": {}}',
            b'{"items": [], "next_cursor": 5}',
        ],
        ids=["not-json", "list", "no-items", "items-not-list", "cursor-not-string"],
    )
    def test_an_unexpected_page_raises_instead_of_yielding_nothing(self, body):
        from soup_cli.utils.ingest_pull import PullError

        with pytest.raises(PullError):
            _pull(_Transport(_response(200, body=body)))


# ============================================================
# 429: honour Retry-After, with a ceiling, and give up loudly
# ============================================================


class TestRateLimit:
    def _run(self, *responses):
        sleeps = []
        transport = _Transport(*responses)
        rows = _pull(transport, sleep=sleeps.append)
        return rows, sleeps, transport

    @pytest.mark.parametrize("header", ["Retry-After", "retry-after"])
    def test_retry_after_seconds_is_honoured(self, header):
        rows, sleeps, _ = self._run(_response(429, headers={header: "7"}), _page([_run(1)]))

        assert sleeps == [7.0]
        assert len(rows) == 1

    def test_retry_after_is_capped(self):
        from soup_cli.utils.ingest_pull import RETRY_AFTER_CEILING_SECONDS

        _, sleeps, _ = self._run(_response(429, headers={"Retry-After": "86400"}), _page([]))

        assert sleeps == [RETRY_AFTER_CEILING_SECONDS]

    def test_backoff_without_retry_after_grows_and_stays_under_the_ceiling(self):
        from soup_cli.utils.ingest_pull import MAX_RATE_LIMIT_RETRIES, RETRY_AFTER_CEILING_SECONDS

        _, sleeps, _ = self._run(*([_response(429)] * MAX_RATE_LIMIT_RETRIES + [_page([])]))

        assert len(sleeps) == MAX_RATE_LIMIT_RETRIES
        assert sleeps == sorted(sleeps) and max(sleeps) <= RETRY_AFTER_CEILING_SECONDS

    def test_gives_up_loudly_after_the_retry_budget(self):
        from soup_cli.utils.ingest_pull import MAX_RATE_LIMIT_RETRIES, PullError

        sleeps = []
        transport = _Transport(_response(429, headers={"Retry-After": "1"}))

        with pytest.raises(PullError) as info:
            _pull(transport, sleep=sleeps.append)

        assert len(sleeps) == MAX_RATE_LIMIT_RETRIES
        assert len(transport.query_calls()) == MAX_RATE_LIMIT_RETRIES + 1
        assert "429" in str(info.value) and "LangSmith" in str(info.value)

    def test_the_project_lookup_is_retried_too(self):
        limited = _response(429, headers={"Retry-After": "2"})
        sleeps = []
        transport = _Transport(_page([_run(1)]), sessions=[limited, _sessions()])

        rows = _pull(transport, sleep=sleeps.append)

        assert sleeps == [2.0] and len(rows) == 1


# ============================================================
# Errors: named, bounded, never carrying the key
# ============================================================


class TestErrors:
    @pytest.mark.parametrize("status", [401, 403])
    @pytest.mark.parametrize("where", ["lookup", "query"])
    def test_a_rejected_key_names_the_variable_not_its_value(self, status, where):
        from soup_cli.utils.ingest_pull import PullError

        denied = _response(status, body=b'{"detail":"Invalid token"}')
        transport = (
            _Transport(_page([]), sessions=denied) if where == "lookup" else _Transport(denied)
        )

        with pytest.raises(PullError) as info:
            _pull(transport)

        assert "LANGSMITH_API_KEY" in str(info.value)
        assert not any(token in str(info.value) for token in _TOKENS)

    def test_a_server_that_echoes_the_request_is_scrubbed(self):
        from soup_cli.utils.ingest_pull import PullError

        echo = f"bad request; X-API-Key: {_KEY}; again {_KEY}".encode()

        with pytest.raises(PullError) as info:
            _pull(_Transport(_response(500, body=echo)))

        assert "500" in str(info.value)
        assert not any(token in str(info.value) for token in _TOKENS)

    def test_a_redirect_is_refused_not_followed(self):
        from soup_cli.utils.ingest_pull import PullError

        transport = _Transport(_response(307, headers={"Location": "https://elsewhere.example"}))

        with pytest.raises(PullError) as info:
            _pull(transport)

        assert len(transport.query_calls()) == 1
        assert "redirect" in str(info.value).lower() and "LANGSMITH_ENDPOINT" in str(info.value)


# ============================================================
# The helpers used from ingest_pull.py, reached through the LangSmith path
# ============================================================


class TestSharedHelpers:
    """Each test fails if the helper it names stops working for LangSmith."""

    def test_parse_since_sets_the_window_from_the_cli(self, pull_env, monkeypatch):
        transport = _Transport(_page([]))
        _use_transport(monkeypatch, transport)

        result = _invoke(["--source", "langsmith", "--pull", "--since", "30m", "-o", "o.jsonl"])

        assert result.exit_code == 0, result.output
        body = transport.query(0)
        start = datetime.strptime(body["min_start_time"], "%Y-%m-%dT%H:%M:%SZ")
        end = datetime.strptime(body["max_start_time"], "%Y-%m-%dT%H:%M:%SZ")
        assert end - start == timedelta(minutes=30)

    def test_parse_since_is_the_function_called(self, pull_env, monkeypatch):
        from soup_cli.utils import ingest_pull

        seen = []
        real = ingest_pull.parse_since
        monkeypatch.setattr(ingest_pull, "parse_since", lambda v: seen.append(v) or real(v))
        _use_transport(monkeypatch, _Transport(_page([])))

        result = _invoke(["--source", "langsmith", "--pull", "--since", "2h", "-o", "o.jsonl"])

        assert result.exit_code == 0, result.output
        assert seen == ["2h"]

    def test_retry_after_as_an_http_date_goes_through_parse_retry_after(self, monkeypatch):
        from soup_cli.utils import ingest_pull

        seen = []
        real = ingest_pull._parse_retry_after
        monkeypatch.setattr(
            ingest_pull, "_parse_retry_after", lambda value: seen.append(value) or real(value)
        )
        sleeps = []
        date = "Wed, 21 Oct 2099 07:28:00 GMT"

        _pull(_Transport(_response(429, headers={"Retry-After": date}), _page([])),
              sleep=sleeps.append)

        assert seen == [date]
        assert sleeps == [ingest_pull.RETRY_AFTER_CEILING_SECONDS]

    def test_the_body_is_read_through_read_capped(self, loopback, monkeypatch):
        from soup_cli.utils import ingest_pull
        from soup_cli.utils.ingest_pull_langsmith import _urllib_transport

        base, _ = loopback
        seen = []
        real = ingest_pull._read_capped

        def _spy(stream, **kwargs):
            seen.append(kwargs)
            return real(stream, **kwargs)

        monkeypatch.setattr(ingest_pull, "_read_capped", _spy)

        response = _urllib_transport("GET", f"{base}/ok", None, _settings(), 5.0)

        assert response.status == 200
        assert len(seen) == 1 and seen[0]["timeout"] == 5.0

    def test_the_size_cap_applies_and_names_langsmith(self, loopback, monkeypatch):
        from soup_cli.utils import ingest_pull
        from soup_cli.utils.ingest_pull_langsmith import _urllib_transport

        base, _ = loopback
        monkeypatch.setattr(ingest_pull, "_MAX_RESPONSE_BYTES", 1024)

        with pytest.raises(ingest_pull.PullError) as info:
            _urllib_transport("GET", f"{base}/big", None, _settings(), 5.0)

        assert "LangSmith" in str(info.value) and "Langfuse" not in str(info.value)


# ============================================================
# The real transport, against a loopback server
# ============================================================

_SLOW_ROUTE_CEILING_SECONDS = 10.0


@pytest.fixture
def slow_route():
    return types.SimpleNamespace(release=threading.Event(), answered=threading.Event())


@pytest.fixture
def loopback(monkeypatch, slow_route):
    for scheme in ("HTTP", "HTTPS", "ALL"):
        for name in (f"{scheme}_PROXY", f"{scheme.lower()}_proxy"):
            monkeypatch.delenv(name, raising=False)
    hits = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, status, body=b"", headers=()):
            self.send_response(status)
            for key, value in headers:
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _handle(self):
            route = urlsplit(self.path).path
            length = int(self.headers.get("Content-Length") or 0)
            hits.append({
                "method": self.command,
                "route": route,
                "api_key": self.headers.get("X-API-Key"),
                "content_type": self.headers.get("Content-Type"),
                "body": self.rfile.read(length) if length else b"",
            })
            try:
                if route.startswith("/redirect"):
                    self._send(int(route[-3:]), headers=[("Location", "/landing")])
                elif route == "/limited":
                    self._send(429, b'{"detail":"slow down"}', [("Retry-After", "7")])
                elif route == "/slow":
                    slow_route.release.wait(_SLOW_ROUTE_CEILING_SECONDS)
                    slow_route.answered.set()
                    self._send(200, b'{"items": [], "next_cursor": null}')
                elif route == "/big":
                    self._send(200, b"x" * 4096)
                else:
                    self._send(200, b'{"items": [], "next_cursor": null}')
            except ConnectionError:
                pass

        do_GET = _handle  # noqa: N815 — http.server API
        do_POST = _handle  # noqa: N815 — http.server API

    server = LoopbackHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", hits
    finally:
        slow_route.release.set()
        server.shutdown()
        server.server_close()


class TestUrllibTransport:
    def test_posts_json_with_the_api_key_header(self, loopback):
        from soup_cli.utils.ingest_pull_langsmith import _urllib_transport

        base, hits = loopback

        url = f"{base}/api/v2/runs/query"

        response = _urllib_transport("POST", url, b'{"a": 1}', _settings(), 5.0)

        assert response.status == 200
        assert hits == [{
            "method": "POST", "route": "/api/v2/runs/query", "api_key": _KEY,
            "content_type": "application/json", "body": b'{"a": 1}',
        }]

    @pytest.mark.parametrize(
        ("method", "status"), [("GET", 307), ("GET", 302), ("POST", 302), ("POST", 303)]
    )
    def test_does_not_follow_a_redirect_with_the_key_attached(self, loopback, method, status):
        """Pairs urllib's default handler would follow (it leaves a POST 307 alone)."""
        from soup_cli.utils.ingest_pull_langsmith import _urllib_transport

        base, hits = loopback
        body = b"{}" if method == "POST" else None

        response = _urllib_transport(method, f"{base}/redirect{status}", body, _settings(), 5.0)

        assert response.status == status
        assert [hit["route"] for hit in hits] == [f"/redirect{status}"]

    def test_an_http_error_status_comes_back_as_a_response_with_headers(self, loopback):
        from soup_cli.utils.ingest_pull_langsmith import _urllib_transport

        base, _ = loopback

        response = _urllib_transport("GET", f"{base}/limited", None, _settings(), 5.0)

        assert response.status == 429
        assert {k.lower(): v for k, v in response.headers.items()}["retry-after"] == "7"

    def test_an_error_body_that_cannot_be_read_keeps_the_status(self, loopback, monkeypatch):
        from soup_cli.utils import ingest_pull
        from soup_cli.utils.ingest_pull_langsmith import _urllib_transport

        base, _ = loopback

        def _stalls(stream, **kwargs):
            raise TimeoutError("timed out")

        monkeypatch.setattr(ingest_pull, "_read_capped", _stalls)

        response = _urllib_transport("GET", f"{base}/limited", None, _settings(), 5.0)

        assert (response.status, response.body) == (429, b"")

    def test_times_out(self, loopback, slow_route):
        from soup_cli.utils.ingest_pull import PullError
        from soup_cli.utils.ingest_pull_langsmith import _urllib_transport

        base, _ = loopback

        with pytest.raises(PullError) as info:
            _urllib_transport("POST", f"{base}/slow", b"{}", _settings(), 0.3)

        assert "timed out" in str(info.value).lower()
        assert not slow_route.answered.is_set(), "the client gave up only after the server answered"

    def test_a_refused_connection_is_a_pull_error(self):
        from soup_cli.utils.ingest_pull import PullError
        from soup_cli.utils.ingest_pull_langsmith import _urllib_transport

        with pytest.raises(PullError) as info:
            _urllib_transport("GET", "http://127.0.0.1:9/", None, _settings(), 2.0)

        assert "LangSmith" in str(info.value)

    def test_the_key_does_not_appear_in_a_traceback_that_renders_locals(self, monkeypatch):
        import urllib.request

        from rich.console import Console
        from rich.traceback import Traceback

        from soup_cli.utils.ingest_pull_langsmith import _urllib_transport

        def _explode(self, *args, **kwargs):
            raise RuntimeError("socket layer exploded")

        monkeypatch.setattr(urllib.request.OpenerDirector, "open", _explode)

        with pytest.raises(RuntimeError) as info:
            _urllib_transport("POST", f"{_HOST}/api/v2/runs/query", b"{}", _settings(), 5.0)

        console = Console(record=True, width=240, file=io.StringIO())
        console.print(
            Traceback.from_exception(
                info.type, info.value, info.tb, show_locals=True,
                locals_max_string=100_000, locals_max_length=100_000,
            )
        )
        rendered = console.export_text()

        assert "socket layer exploded" in rendered and "_urllib_transport" in rendered  # CONTROL
        assert _leaked_tokens(rendered) == []

    def test_settings_repr_hides_the_key(self):
        settings = _settings()

        assert _KEY not in repr(settings) and _KEY not in str(settings)
        assert _PROJECT in repr(settings)


# ============================================================
# LANGSMITH_ENDPOINT and the other settings
# ============================================================


class TestHost:
    def test_default_is_langsmith_cloud(self):
        from soup_cli.utils.ingest_pull_langsmith import (
            LANGSMITH_DEFAULT_HOST,
            resolve_langsmith_host,
        )

        assert resolve_langsmith_host({}) == LANGSMITH_DEFAULT_HOST == _HOST
        assert resolve_langsmith_host({"LANGSMITH_ENDPOINT": ""}) == _HOST

    @pytest.mark.parametrize(
        ("endpoint", "expected"),
        [
            ("https://eu.api.smith.langchain.com/", "https://eu.api.smith.langchain.com"),
            ("https://langsmith.example.com/api/v1", "https://langsmith.example.com"),
            ("https://langsmith.example.com/api", "https://langsmith.example.com"),
            ("https://example.com/langsmith/api/v1/", "https://example.com/langsmith"),
        ],
    )
    def test_the_endpoint_is_read_like_the_sdk(self, endpoint, expected):
        """langsmith-py 0.14.17 strips a trailing ``/api/v1`` or ``/api`` the same way."""
        from soup_cli.utils.ingest_pull_langsmith import resolve_langsmith_host

        assert resolve_langsmith_host({"LANGSMITH_ENDPOINT": endpoint}) == expected

    def test_the_shared_webhook_validator_is_the_gate(self, monkeypatch):
        from soup_cli.utils import webhooks
        from soup_cli.utils.ingest_pull_langsmith import resolve_langsmith_host

        seen = []
        real = webhooks.validate_webhook_url

        def _spy(url, *, allow_private_hosts=False):
            seen.append((url, allow_private_hosts))
            return real(url, allow_private_hosts=allow_private_hosts)

        monkeypatch.setattr(webhooks, "validate_webhook_url", _spy)

        resolve_langsmith_host({"LANGSMITH_ENDPOINT": "https://10.0.0.5"}, allow_private_host=True)

        assert seen == [("https://10.0.0.5", True)]

    @pytest.mark.parametrize(
        "host",
        [
            "http://api.smith.langchain.com",
            "https://169.254.169.254",
            "https://127.1",
            "https://[::1]:1984",
            "ftp://api.smith.langchain.com",
            "https://api.smith.langchain.com/?x=1",
            "https://api.smith.langchain.com/#frag",
            "https://api.smith.langchain.com\n.evil",
        ],
    )
    def test_refused_by_default(self, host):
        from soup_cli.utils.ingest_pull_langsmith import resolve_langsmith_host

        with pytest.raises(ValueError):
            resolve_langsmith_host({"LANGSMITH_ENDPOINT": host})

    @pytest.mark.parametrize(
        "host", ["https://10.0.0.5", "https://192.168.1.20:1984", "https://localhost", "https://127.0.0.1:1984"]
    )
    def test_private_and_loopback_https_need_the_explicit_opt_in(self, host):
        from soup_cli.utils.ingest_pull_langsmith import resolve_langsmith_host

        with pytest.raises(ValueError) as info:
            resolve_langsmith_host({"LANGSMITH_ENDPOINT": host})
        assert "--allow-private-host" in str(info.value)

        assert resolve_langsmith_host({"LANGSMITH_ENDPOINT": host}, allow_private_host=True) == host

    @pytest.mark.parametrize("host", ["http://localhost:1984", "http://10.0.0.5"])
    def test_plain_http_stays_refused_even_with_the_opt_in(self, host):
        from soup_cli.utils.ingest_pull_langsmith import resolve_langsmith_host

        with pytest.raises(ValueError):
            resolve_langsmith_host({"LANGSMITH_ENDPOINT": host}, allow_private_host=True)

    def test_credentials_embedded_in_the_url_are_refused_without_echoing_them(self):
        from soup_cli.utils.ingest_pull_langsmith import resolve_langsmith_host

        with pytest.raises(ValueError) as info:
            resolve_langsmith_host({"LANGSMITH_ENDPOINT": f"https://user:{_KEY}@api.smith.langchain.com"})

        assert _KEY not in str(info.value)


class TestSettings:
    def test_loads_key_project_and_host(self):
        from soup_cli.utils.ingest_pull_langsmith import load_langsmith_settings

        eu = "https://eu.api.smith.langchain.com"
        settings = load_langsmith_settings(
            {"LANGSMITH_API_KEY": _KEY, "LANGSMITH_PROJECT": _PROJECT, "LANGSMITH_ENDPOINT": eu}
        )

        assert (settings.api_key, settings.project, settings.host) == (_KEY, _PROJECT, eu)

    @pytest.mark.parametrize(
        ("environ", "missing"),
        [
            ({}, ["LANGSMITH_API_KEY", "LANGSMITH_PROJECT"]),
            ({"LANGSMITH_PROJECT": _PROJECT}, ["LANGSMITH_API_KEY"]),
            ({"LANGSMITH_API_KEY": _KEY, "LANGSMITH_PROJECT": ""}, ["LANGSMITH_PROJECT"]),
        ],
    )
    def test_missing_settings_are_named_and_the_key_never_echoed(self, environ, missing):
        from soup_cli.utils.ingest_pull import PullError
        from soup_cli.utils.ingest_pull_langsmith import load_langsmith_settings

        with pytest.raises(PullError) as info:
            load_langsmith_settings(environ)

        for name in missing:
            assert name in str(info.value)
        assert _KEY not in str(info.value)


# ============================================================
# CLI
# ============================================================


@pytest.fixture
def pull_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LANGSMITH_API_KEY", _KEY)
    monkeypatch.setenv("LANGSMITH_PROJECT", _PROJECT)
    for name in ("LANGSMITH_ENDPOINT", "SOUP_TELEMETRY"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def _invoke(args):
    from typer.testing import CliRunner

    from soup_cli.cli import app

    return CliRunner().invoke(app, ["ingest", *args])


def _plain(text):
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _use_transport(monkeypatch, transport):
    from soup_cli.utils import ingest_pull_langsmith

    monkeypatch.setattr(ingest_pull_langsmith, "_urllib_transport", transport)
    monkeypatch.setattr(ingest_pull_langsmith.time, "sleep", lambda seconds: None)


class TestCli:
    def test_pull_writes_parsed_rows(self, pull_env, monkeypatch):
        _use_transport(monkeypatch, _Transport(_page([_run(1)], cursor="c1"), _page([_run(2)])))

        result = _invoke(["--source", "langsmith", "--pull", "--since", "1d", "-o", "out.jsonl"])

        assert result.exit_code == 0, result.output
        lines = (pull_env / "out.jsonl").read_text("utf-8").splitlines()
        rows = [json.loads(line) for line in lines]
        assert [(r["trace_id"], r["prompt"], r["output"], r["source"]) for r in rows] == [
            ("run-1", "q1", "a1", "langsmith"),
            ("run-2", "q2", "a2", "langsmith"),
        ]
        assert "Wrote 2 traces from langsmith (2 LLM runs pulled)" in _plain(result.output)

    def test_skipped_runs_are_reported(self, pull_env, monkeypatch):
        _use_transport(
            monkeypatch,
            _Transport(_page([_run(1), _run(2, "chain"), {"id": "x", "run_type": "llm"}])),
        )

        result = _invoke(["--source", "langsmith", "--pull", "--output", "out.jsonl"])

        assert result.exit_code == 0, result.output
        output = _plain(result.output)
        assert "Wrote 1 traces from langsmith (2 LLM runs pulled)" in output
        assert "1 LLM run(s) had no input or no output" in output
        assert "1 run(s) were not LLM runs" in output

    def test_page_cap_exits_nonzero_and_leaves_the_previous_output_untouched(
        self, pull_env, monkeypatch
    ):
        (pull_env / "out.jsonl").write_text("previous run\n", encoding="utf-8")
        _use_transport(
            monkeypatch, _Transport(_page([_run(1)], cursor="c1"), _page([_run(2)], cursor="c2"))
        )

        result = _invoke(["--source", "langsmith", "--pull", "--max-pages", "2", "-o", "out.jsonl"])

        assert result.exit_code == 1
        assert "--max-pages" in result.output and "Nothing was written" in result.output
        assert (pull_env / "out.jsonl").read_text(encoding="utf-8") == "previous run\n"
        assert [path.name for path in pull_env.iterdir()] == ["out.jsonl"]

    @pytest.mark.parametrize("missing", ["LANGSMITH_API_KEY", "LANGSMITH_PROJECT"])
    def test_missing_settings_exit_before_any_request(self, pull_env, monkeypatch, missing):
        transport = _Transport(_page([]))
        _use_transport(monkeypatch, transport)
        monkeypatch.delenv(missing)

        result = _invoke(["--source", "langsmith", "--pull"])

        assert result.exit_code == 1
        assert missing in result.output
        assert transport.calls == []

    def test_private_host_is_refused_before_any_request_unless_allowed(self, pull_env, monkeypatch):
        transport = _Transport(_page([]))
        _use_transport(monkeypatch, transport)
        monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://10.0.0.5/api/v1")

        refused = _invoke(["--source", "langsmith", "--pull"])
        allowed = _invoke(
            ["--source", "langsmith", "--pull", "--allow-private-host", "--output", "out.jsonl"]
        )

        assert refused.exit_code == 2
        assert "--allow-private-host" in refused.output
        assert allowed.exit_code == 0, allowed.output
        assert transport.calls[-1]["url"] == "https://10.0.0.5/api/v2/runs/query"

    @pytest.mark.parametrize("source", ["helicone", "openpipe", "otel", "openai-stored"])
    def test_pull_stays_refused_for_the_other_sources(self, pull_env, source):
        result = _invoke(["--source", source, "--pull"])

        assert result.exit_code == 2
        assert "langfuse" in result.output.lower() and "langsmith" in result.output.lower()

    def test_pull_and_logs_are_mutually_exclusive(self, pull_env):
        (pull_env / "ls.jsonl").write_text('{"inputs":"x","outputs":"y"}\n', encoding="utf-8")

        result = _invoke(["--source", "langsmith", "--pull", "--logs", "ls.jsonl"])

        assert result.exit_code == 2
        assert "--logs" in result.output and "--pull" in result.output

    def test_bad_since_is_a_usage_error(self, pull_env, monkeypatch):
        transport = _Transport(_page([]))
        _use_transport(monkeypatch, transport)

        result = _invoke(["--source", "langsmith", "--pull", "--since", "forever"])

        assert result.exit_code == 2
        assert transport.calls == []

    def test_the_pii_panel_names_the_langsmith_settings(self, pull_env):
        (pull_env / "ls.jsonl").write_text('{"inputs":"x","outputs":"y"}\n', encoding="utf-8")

        result = _invoke(["--source", "langsmith", "--logs", "ls.jsonl", "--output", "out.jsonl"])

        assert result.exit_code == 0, result.output
        output = " ".join(_plain(result.output).split())
        for name in ("LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT"):
            assert name in output

    def test_resolve_auth_env_still_needs_only_the_key(self, monkeypatch):
        """LANGSMITH_PROJECT and LANGSMITH_ENDPOINT are settings, not credentials."""
        from soup_cli.utils.ingest_sources import resolve_auth_env

        for name in ("LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT"):
            monkeypatch.delenv(name, raising=False)
        assert resolve_auth_env("langsmith") is None

        monkeypatch.setenv("LANGSMITH_API_KEY", _KEY)
        assert resolve_auth_env("langsmith") == _KEY


# ============================================================
# Without --pull: no LangSmith pull code loaded, no socket
# ============================================================

_IMPORT_PROBE = r"""
import json, socket, sys

attempts = []

def _refuse(*args, **kwargs):
    attempts.append("network")
    raise OSError("network disabled by the #204 probe")

socket.socket.connect = _refuse
socket.create_connection = _refuse
socket.getaddrinfo = _refuse

if sys.argv[1] == "control":
    import soup_cli.utils.ingest_pull_langsmith  # noqa: F401

from typer.testing import CliRunner
from soup_cli.cli import app

result = CliRunner().invoke(
    app, ["ingest", "--source", "langsmith", "--logs", "ls.jsonl", "--output", "out.jsonl"]
)
watched = {
    "langsmith", "httpx", "soup_cli.utils.ingest_pull", "soup_cli.utils.ingest_pull_langsmith",
}
loaded = sorted(
    name for name in sys.modules if name in watched or name.split(".")[0] in {"langsmith", "httpx"}
)
print(json.dumps({"exit": result.exit_code, "attempts": attempts, "loaded": loaded}))
"""


def _probe(tmp_path, mode):
    run = {"id": "r1", "inputs": {"messages": [{"role": "user", "content": "x"}]},
           "outputs": {"generations": [[{"text": "y"}]]}}
    (tmp_path / "ls.jsonl").write_text(json.dumps(run) + "\n", encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if key != "SOUP_TELEMETRY"}
    env["SOUP_NO_AUDIT_LOG"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE, mode],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_local_export_loads_no_pull_code_and_opens_no_socket(tmp_path):
    report = _probe(tmp_path, "local")

    assert report == {"exit": 0, "attempts": [], "loaded": []}
    assert json.loads((tmp_path / "out.jsonl").read_text("utf-8"))["trace_id"] == "r1"


def test_control_the_probe_sees_the_pull_module_when_it_is_imported(tmp_path):
    report = _probe(tmp_path, "control")

    assert report["exit"] == 0
    assert "soup_cli.utils.ingest_pull_langsmith" in report["loaded"]


# ============================================================
# The recorded real page: pull -> parse_langsmith
# ============================================================

_RECORDED_ROWS = [
    ("01a1ddd9-a9e4-7bee-a428-8c3f2ae2c763", "Name another primary color.",
     "Blue is also a primary color.", "none"),
    ("01a1ddd9-a9e3-733e-8d88-4b705105d1de", "Translate 'hello' to Spanish.", "Hola.", "none"),
    ("b11a9e24-4a58-4596-bd2d-cd06fad7a788", "Name a primary color.", "Red.", "thumbs_down"),
    ("ca77fb5c-6d6c-479e-9128-3b616a18ad97", "What is the capital of France?", "Paris.",
     "thumbs_up"),
]


class TestRecordedRealResponse:
    """``POST /api/v2/runs/query`` as LangSmith Cloud answered it on 2026-11-15 for
    every run type (the reporter's #2868 recording, which #2942 added as a fixture):
    four LLM runs, one chain run and one tool run."""

    def test_the_page_flows_through_the_pull_and_parse_langsmith(self):
        from soup_cli.utils.ingest_pull_langsmith import LangSmithPullStats
        from soup_cli.utils.ingest_sources import parse_langsmith

        stats = LangSmithPullStats()
        pulled = _pull(_Transport(_response(200, body=_RECORDED_V2.read_bytes())), stats=stats)
        records = [
            (r.trace_id, r.prompt, r.output, r.signal) for r in parse_langsmith(pulled)
        ]

        assert records == _RECORDED_ROWS
        assert (stats.runs, stats.llm_runs, stats.skipped_not_llm) == (6, 4, 2)

    def test_the_chain_and_tool_runs_are_not_pulled(self):
        page = json.loads(_RECORDED_V2.read_text(encoding="utf-8"))["items"]
        dropped = {run["id"] for run in page if run["run_type"] != "llm"}

        pulled = _pull(_Transport(_response(200, body=_RECORDED_V2.read_bytes())))

        assert {run["run_type"] for run in page} == {"llm", "chain", "tool"}
        assert len(dropped) == 2
        assert dropped.isdisjoint(run["id"] for run in pulled)

    def test_the_page_is_the_last_one(self):
        """``next_cursor`` is null on the recorded page, so the pull makes one query."""
        transport = _Transport(_response(200, body=_RECORDED_V2.read_bytes()))

        _pull(transport)

        assert json.loads(_RECORDED_V2.read_text("utf-8"))["next_cursor"] is None
        assert len(transport.query_calls()) == 1


# ============================================================
# A planted key never reaches an artefact
# ============================================================


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.lines = []

    def emit(self, record):
        self.lines.append(self.format(record))


def _scenario(name):
    from soup_cli.utils.ingest_pull import PullError

    echo = f"upstream said: X-API-Key: {_KEY}".encode()
    unreachable = PullError("could not reach LangSmith at https://api.smith.langchain.com")
    denied = _response(401, body=b'{"detail":"Invalid token"}')
    return {
        "success": (_Transport(_page([_run(1)], cursor="c1"), _page([_run(2)])), 0),
        "project_id_echoes_key": (
            _Transport(_page([_run(1)]), sessions=_sessions(project_id=f"proj-{_KEY}")), 0
        ),
        "unauthorized": (_Transport(denied), 1),
        "unauthorized_lookup": (_Transport(_page([]), sessions=denied), 1),
        "server_echoes_request": (_Transport(_response(500, body=echo)), 1),
        "page_cap": (_Transport(_page([_run(1)], cursor="c1"), _page([_run(2)], cursor="c2")), 1),
        "rate_limited": (_Transport(_response(429, headers={"Retry-After": "1"})), 1),
        "network_error": (_Transport(unreachable), 1),
        "unexpected_exception": (_Transport(RuntimeError("transport bug")), 1),
    }[name]


_SCENARIOS = (
    "success", "project_id_echoes_key", "unauthorized", "unauthorized_lookup",
    "server_echoes_request", "page_cap", "rate_limited", "network_error", "unexpected_exception",
)


@pytest.mark.parametrize("name", _SCENARIOS)
def test_a_planted_key_never_reaches_any_artefact(name, pull_env, monkeypatch, capsys):
    """Runs through ``soup``'s real entry point (``run()``), so the audit log and
    the ``--verbose`` friendly-error traceback are part of the surface."""
    from soup_cli import cli
    from soup_cli.utils import webhooks

    transport, expected_exit = _scenario(name)
    _use_transport(monkeypatch, transport)
    monkeypatch.setenv("SOUP_AUDIT_LOG_PATH", str(pull_env / "audit.jsonl"))
    monkeypatch.delenv("SOUP_NO_AUDIT_LOG", raising=False)
    payloads = []
    monkeypatch.setattr(
        webhooks, "send_webhooks", lambda payload, **kwargs: payloads.append(dict(payload)) or []
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "soup", "--verbose", "--log-level", "debug",
            "ingest", "--source", "langsmith", "--pull", "--since", "1d", "--max-pages", "2",
            "--output", "out.jsonl", "--slack-url", "https://hooks.slack.com/services/T0/B0/X0",
        ],
    )
    collected = _Collect()
    loggers = (logging.getLogger("soup"), logging.getLogger())
    saved = [(logger, logger.level, list(logger.handlers), logger.propagate) for logger in loggers]
    for logger in loggers:
        logger.addHandler(collected)
    try:
        with pytest.raises(SystemExit) as exit_info:
            cli.run()
    finally:
        for logger, level, handlers, propagate in saved:
            logger.setLevel(level)
            logger.handlers[:] = handlers
            logger.propagate = propagate

    captured = capsys.readouterr()
    artefacts = {
        "stdout": captured.out,
        "stderr": captured.err,
        "logs": "\n".join(collected.lines),
        "webhook": json.dumps(payloads),
    }
    for path in sorted(pull_env.rglob("*")):
        if path.is_file():
            artefacts[f"file:{path.name}"] = path.read_bytes().decode("utf-8", errors="replace")

    assert exit_info.value.code == expected_exit, captured.out
    # CONTROLS: each surface this test claims to scan was actually produced.
    assert "file:audit.jsonl" in artefacts
    assert any("langsmith pull" in line for line in collected.lines)
    if name == "success":
        assert "file:out.jsonl" in artefacts and payloads
    if name == "project_id_echoes_key":
        # The id from the sessions answer reached the debug line, scrubbed.
        assert any("for project proj-*** from" in line for line in collected.lines)
    if name == "unexpected_exception":
        streams = _plain(captured.out + captured.err)
        assert "Full Traceback" in streams and "transport bug" in streams
    leaks = {where: found for where, text in artefacts.items() if (found := _leaked_tokens(text))}
    assert leaks == {}


def _leaked_tokens(text):
    """Also looks with ANSI codes, whitespace and box-drawing characters removed, and
    counts a 20-character prefix as a leak, as the Langfuse scan does."""
    import re

    plain = _plain(text)
    compact = re.sub(r"[\s│╭╮╰╯─]+", "", plain)
    views = (text, plain, compact)
    return [
        token
        for token in _TOKENS
        if any(needle in view for needle in (token, token[:20]) for view in views)
    ]


def test_control_the_leak_scan_sees_a_key_rich_highlighted_and_folded():
    from rich.console import Console
    from rich.panel import Panel

    console = Console(file=io.StringIO(), width=24, force_terminal=True, highlight=True)
    console.print(Panel(f"X-API-Key: {_KEY}"))
    rendered = console.file.getvalue()

    assert _KEY not in rendered  # rich really did split it
    assert _leaked_tokens(rendered) == [_KEY]
