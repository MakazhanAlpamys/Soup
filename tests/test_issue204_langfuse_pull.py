"""#204 — `soup ingest --source langfuse --pull`: a live pull from Langfuse Cloud.

`soup ingest` has always parsed an offline JSONL export and never touched the
network. This slice adds an opt-in pull for Langfuse only (the other four
providers stay open on #204), and it is the first time the command makes a
network call, so most of this file pins what must NOT happen:

- without ``--pull`` nothing changes: same bytes, no pull module, no httpx, no
  socket (checked in a clean subprocess, because this test process has long
  since imported half the ecosystem);
- the key pair never reaches an artefact: stdout / stderr under
  ``--verbose --log-level debug``, the output file, any file left in the
  working directory, the audit log, a webhook payload, or a traceback that
  renders locals;
- ``LANGFUSE_HOST`` goes through the existing SSRF validator, is HTTPS-only,
  and a private / loopback address needs ``--allow-private-host``;
- every loop is bounded, and a cap stops loudly instead of truncating.

What the fake transports below do NOT prove is the wire shape: a response
hand-shaped from the docs is exactly how a pull passes its tests and then
yields zero rows against the real API. The parser agreement is pinned by the
recorded response from a real Hobby-cloud run instead (see the fixture test).
"""

from __future__ import annotations

import base64
import http.server
import io
import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

_PUBLIC = "pk-lf-SOUPPLANTEDPUBLIC-5b1d"
_SECRET = "sk-lf-SOUPPLANTEDSECRET-9e27"
_BASIC = base64.b64encode(f"{_PUBLIC}:{_SECRET}".encode()).decode()
_TOKENS = (_PUBLIC, _SECRET, _BASIC)

_NOW = datetime(2026, 9, 11, 12, 0, 0, tzinfo=timezone.utc)


def _creds(host="https://cloud.langfuse.com"):
    from soup_cli.utils.ingest_pull import LangfuseCredentials

    return LangfuseCredentials(public_key=_PUBLIC, secret_key=_SECRET, host=host)


def _response(status=200, body=b"", headers=None):
    from soup_cli.utils.ingest_pull import HttpResponse

    return HttpResponse(status=status, headers=dict(headers or {}), body=body)


def _page(observations, cursor=None):
    meta = {"cursor": cursor} if cursor else {}
    return _response(body=json.dumps({"data": observations, "meta": meta}).encode())


def _obs(index):
    """Control-flow filler only — the wire shape is pinned by the recorded fixture."""
    return {"id": f"obs-{index}", "type": "GENERATION", "input": f"q{index}", "output": f"a{index}"}


class _Transport:
    """Replays scripted responses and records every call it receives."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = []

    def __call__(self, url, credentials, timeout):
        self.calls.append({"url": url, "credentials": credentials, "timeout": timeout})
        item = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        if isinstance(item, BaseException):
            raise item
        return item

    def query(self, index):
        parsed = parse_qs(urlsplit(self.calls[index]["url"]).query)
        return {key: values[0] for key, values in parsed.items()}


def _pull(transport, **kwargs):
    from soup_cli.utils.ingest_pull import pull_langfuse_generations

    kwargs.setdefault("since", timedelta(days=1))
    kwargs.setdefault("now", _NOW)
    kwargs.setdefault("sleep", lambda seconds: None)
    return list(pull_langfuse_generations(_creds(), transport=transport, **kwargs))


# ============================================================
# The request: Observations API v2, generations, with input/output
# ============================================================


class TestRequest:
    def test_targets_observations_v2_generations_and_asks_for_io(self):
        """``fields`` defaults to ``core,basic`` server-side, which carries NO
        input/output — a pull that forgot ``io`` returns rows the parser drops."""
        from soup_cli.utils.ingest_pull import LANGFUSE_OBSERVATIONS_PATH

        transport = _Transport(_page([_obs(1)]))

        _pull(transport, since=timedelta(hours=6), timeout=12.5)

        call = transport.calls[0]
        parts = urlsplit(call["url"])
        query = transport.query(0)
        assert f"{parts.scheme}://{parts.netloc}" == "https://cloud.langfuse.com"
        assert parts.path == LANGFUSE_OBSERVATIONS_PATH == "/api/public/v2/observations"
        assert query["type"] == "GENERATION"
        assert "io" in query["fields"].split(",")
        assert "parseIoAsJson" not in query  # deprecated: true is a 400
        assert query["fromStartTime"] == "2026-09-11T06:00:00Z"
        assert query["toStartTime"] == "2026-09-11T12:00:00Z"
        assert 1 <= int(query["limit"]) <= 1000
        assert call["timeout"] == 12.5


# ============================================================
# Pagination and caps
# ============================================================


class TestPagination:
    def test_follows_the_cursor_until_it_is_absent(self):
        from soup_cli.utils.ingest_pull import LANGFUSE_OBSERVATIONS_PATH

        transport = _Transport(
            _page([_obs(1), _obs(2)], cursor="c1"),
            _page([_obs(3)], cursor="c2"),
            _page([_obs(4)]),
        )

        rows = _pull(transport)

        assert [row["id"] for row in rows] == ["obs-1", "obs-2", "obs-3", "obs-4"]
        assert len(transport.calls) == 3
        assert "cursor" not in transport.query(0)
        assert transport.query(1)["cursor"] == "c1"
        assert transport.query(2)["cursor"] == "c2"
        # Every page stays on v2 (`/api/public/traces` leaves Langfuse Cloud on
        # 2026-11-16) and on the window fixed before the first request.
        windows = {(transport.query(i)["fromStartTime"], transport.query(i)["toStartTime"])
                   for i in range(3)}
        paths = {urlsplit(call["url"]).path for call in transport.calls}
        assert len(windows) == 1
        assert paths == {LANGFUSE_OBSERVATIONS_PATH}

    def test_page_cap_stops_loudly_while_results_are_pending(self):
        from soup_cli.utils.ingest_pull import PullLimitError

        transport = _Transport(_page([_obs(1)], cursor="more"))

        with pytest.raises(PullLimitError) as info:
            _pull(transport, max_pages=2)

        assert len(transport.calls) == 2
        assert info.value.pages == 2
        assert info.value.rows == 2
        message = str(info.value)
        assert "--max-pages" in message and "--since" in message

    def test_control_reaching_the_cap_on_the_last_page_is_not_an_error(self):
        """The cap must fire on *pending* results, not on the page count alone."""
        transport = _Transport(_page([_obs(1)], cursor="c1"), _page([_obs(2)]))

        rows = _pull(transport, max_pages=2)

        assert len(rows) == 2

    def test_row_cap_stops_loudly(self, monkeypatch):
        from soup_cli.utils import ingest_pull

        monkeypatch.setattr(ingest_pull, "_MAX_INGEST_LINES", 3)
        transport = _Transport(_page([_obs(1), _obs(2)], cursor="c1"), _page([_obs(3), _obs(4)]))

        with pytest.raises(ingest_pull.PullLimitError):
            _pull(transport)

    @pytest.mark.parametrize("max_pages", [0, -1, True, 10_001])
    def test_max_pages_outside_bounds_is_rejected(self, max_pages):
        with pytest.raises((TypeError, ValueError)):
            _pull(_Transport(_page([])), max_pages=max_pages)


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
        rows, sleeps, _ = self._run(_response(429, headers={header: "7"}), _page([_obs(1)]))

        assert sleeps == [7.0]
        assert len(rows) == 1

    def test_retry_after_is_capped(self):
        from soup_cli.utils.ingest_pull import RETRY_AFTER_CEILING_SECONDS

        _, sleeps, _ = self._run(_response(429, headers={"Retry-After": "86400"}), _page([]))

        assert sleeps == [RETRY_AFTER_CEILING_SECONDS]

    def test_retry_after_http_date_is_honoured_and_capped(self):
        from soup_cli.utils.ingest_pull import RETRY_AFTER_CEILING_SECONDS

        _, sleeps, _ = self._run(
            _response(429, headers={"Retry-After": "Wed, 21 Oct 2099 07:28:00 GMT"}), _page([])
        )

        assert sleeps == [RETRY_AFTER_CEILING_SECONDS]

    @pytest.mark.parametrize("value", ["-5", "soon", ""])
    def test_unusable_retry_after_falls_back_to_bounded_backoff(self, value):
        from soup_cli.utils.ingest_pull import RETRY_AFTER_CEILING_SECONDS

        _, sleeps, _ = self._run(_response(429, headers={"Retry-After": value}), _page([]))

        assert len(sleeps) == 1
        assert 0 < sleeps[0] <= RETRY_AFTER_CEILING_SECONDS

    def test_backoff_without_retry_after_grows_and_stays_under_the_ceiling(self):
        from soup_cli.utils.ingest_pull import MAX_RATE_LIMIT_RETRIES, RETRY_AFTER_CEILING_SECONDS

        responses = [_response(429)] * MAX_RATE_LIMIT_RETRIES + [_page([])]
        _, sleeps, _ = self._run(*responses)

        assert len(sleeps) == MAX_RATE_LIMIT_RETRIES
        assert sleeps == sorted(sleeps)
        assert max(sleeps) <= RETRY_AFTER_CEILING_SECONDS

    def test_gives_up_loudly_after_the_retry_budget(self):
        from soup_cli.utils.ingest_pull import MAX_RATE_LIMIT_RETRIES, PullError

        sleeps = []
        transport = _Transport(_response(429, headers={"Retry-After": "1"}))

        with pytest.raises(PullError) as info:
            _pull(transport, sleep=sleeps.append)

        assert len(sleeps) == MAX_RATE_LIMIT_RETRIES
        assert len(transport.calls) == MAX_RATE_LIMIT_RETRIES + 1
        assert "429" in str(info.value)


# ============================================================
# Errors: named, bounded, never carrying the credentials
# ============================================================


class TestErrors:
    @pytest.mark.parametrize("status", [401, 403])
    def test_rejected_credentials_name_the_variables_not_their_values(self, status):
        from soup_cli.utils.ingest_pull import PullError

        with pytest.raises(PullError) as info:
            _pull(_Transport(_response(status, body=b'{"message":"Invalid credentials"}')))

        message = str(info.value)
        assert "LANGFUSE_PUBLIC_KEY" in message and "LANGFUSE_SECRET_KEY" in message
        assert not any(token in message for token in _TOKENS)

    def test_a_server_that_echoes_the_request_is_scrubbed(self):
        """Defence in depth: the message may quote the server's error, so it is
        scrubbed of the key pair and of the Basic token built from it."""
        from soup_cli.utils.ingest_pull import PullError

        echo = f"bad request; Authorization: Basic {_BASIC}; key={_SECRET}; pk={_PUBLIC}".encode()

        with pytest.raises(PullError) as info:
            _pull(_Transport(_response(500, body=echo)))

        assert "500" in str(info.value)
        assert not any(token in str(info.value) for token in _TOKENS)

    def test_a_redirect_is_refused_not_followed(self):
        from soup_cli.utils.ingest_pull import PullError

        transport = _Transport(_response(302, headers={"Location": "https://elsewhere.example"}))

        with pytest.raises(PullError) as info:
            _pull(transport)

        assert len(transport.calls) == 1
        assert "redirect" in str(info.value).lower()

    @pytest.mark.parametrize(
        "body",
        [b"not json", b"[]", b'{"meta": {}}', b'{"data": {}, "meta": {}}'],
        ids=["not-json", "list", "no-data", "data-not-list"],
    )
    def test_an_unexpected_body_raises_instead_of_yielding_nothing(self, body):
        from soup_cli.utils.ingest_pull import PullError

        with pytest.raises(PullError):
            _pull(_Transport(_response(200, body=body)))


# ============================================================
# The real transport, against a loopback server
# ============================================================


@pytest.fixture
def loopback(monkeypatch):
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

        def do_GET(self):  # noqa: N802 — http.server API
            route = urlsplit(self.path).path
            hits.append({"route": route, "authorization": self.headers.get("Authorization")})
            try:
                if route == "/redirect":
                    self._send(302, headers=[("Location", "/landing")])
                elif route == "/limited":
                    self._send(429, b'{"message":"slow down"}', [("Retry-After", "7")])
                elif route == "/slow":
                    time.sleep(1.5)
                    self._send(200, b'{"data": [], "meta": {}}')
                elif route == "/big":
                    self._send(200, b"x" * 4096)
                else:
                    self._send(200, b'{"data": [], "meta": {}}')
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", hits
    finally:
        server.shutdown()
        server.server_close()


class TestUrllibTransport:
    def test_sends_basic_auth_built_from_the_key_pair(self, loopback):
        from soup_cli.utils.ingest_pull import _urllib_transport

        base, hits = loopback

        response = _urllib_transport(f"{base}/ok", _creds(), 5.0)

        assert response.status == 200
        assert hits[0]["authorization"] == f"Basic {_BASIC}"

    def test_does_not_follow_a_redirect_with_the_credentials_attached(self, loopback):
        """urllib follows redirects by default and copies ``Authorization`` to
        the new location — including a different host."""
        from soup_cli.utils.ingest_pull import _urllib_transport

        base, hits = loopback

        response = _urllib_transport(f"{base}/redirect", _creds(), 5.0)

        assert response.status == 302
        assert [hit["route"] for hit in hits] == ["/redirect"]

    def test_an_http_error_status_comes_back_as_a_response_with_headers(self, loopback):
        from soup_cli.utils.ingest_pull import _urllib_transport

        base, _ = loopback

        response = _urllib_transport(f"{base}/limited", _creds(), 5.0)

        assert response.status == 429
        assert {key.lower(): value for key, value in response.headers.items()}["retry-after"] == "7"

    def test_an_error_body_that_cannot_be_read_keeps_the_status(self, loopback, monkeypatch):
        from soup_cli.utils import ingest_pull

        base, _ = loopback

        def _stalls(stream):
            raise TimeoutError("timed out")

        monkeypatch.setattr(ingest_pull, "_read_capped", _stalls)

        response = ingest_pull._urllib_transport(f"{base}/limited", _creds(), 5.0)

        assert (response.status, response.body) == (429, b"")

    def test_times_out(self, loopback):
        from soup_cli.utils.ingest_pull import PullError, _urllib_transport

        base, _ = loopback
        started = time.monotonic()

        with pytest.raises(PullError) as info:
            _urllib_transport(f"{base}/slow", _creds(), 0.3)

        assert time.monotonic() - started < 1.4
        assert "timed out" in str(info.value).lower()

    def test_a_refused_connection_is_a_pull_error(self):
        from soup_cli.utils.ingest_pull import PullError, _urllib_transport

        with pytest.raises(PullError):
            _urllib_transport("http://127.0.0.1:9/", _creds(), 2.0)

    def test_the_response_body_is_size_capped(self, loopback, monkeypatch):
        from soup_cli.utils import ingest_pull

        base, _ = loopback
        monkeypatch.setattr(ingest_pull, "_MAX_RESPONSE_BYTES", 1024)

        with pytest.raises(ingest_pull.PullError):
            ingest_pull._urllib_transport(f"{base}/big", _creds(), 5.0)

    def test_credentials_do_not_appear_in_a_traceback_that_renders_locals(self, monkeypatch):
        """Typer's default pretty exceptions render frame locals
        (``pretty_exceptions_show_locals=True`` in 0.20); ``soup``'s ``run()``
        catches first, but the transport must not depend on that."""
        import urllib.request

        from rich.console import Console
        from rich.traceback import Traceback

        from soup_cli.utils.ingest_pull import _urllib_transport

        def _explode(self, *args, **kwargs):
            raise RuntimeError("socket layer exploded")

        monkeypatch.setattr(urllib.request.OpenerDirector, "open", _explode)

        url = "https://cloud.langfuse.com/api/public/v2/observations"
        with pytest.raises(RuntimeError) as info:
            _urllib_transport(url, _creds(), 5.0)

        console = Console(record=True, width=240, file=io.StringIO())
        # Rich truncates locals at 80 chars by default, which would print most of
        # the Basic token and still dodge an exact match: render untruncated.
        console.print(
            Traceback.from_exception(
                info.type,
                info.value,
                info.tb,
                show_locals=True,
                locals_max_string=100_000,
                locals_max_length=100_000,
            )
        )
        rendered = console.export_text()

        assert "socket layer exploded" in rendered and "_urllib_transport" in rendered  # CONTROL
        assert _leaked_tokens(rendered) == []

    def test_credentials_repr_hides_the_key_pair(self):
        creds = _creds()

        assert not any(token in repr(creds) or token in str(creds) for token in _TOKENS)


# ============================================================
# LANGFUSE_HOST: the existing SSRF validator, HTTPS-only, private opt-in
# ============================================================


class TestHost:
    def test_langfuse_host_is_read_and_trailing_slash_dropped(self):
        from soup_cli.utils.ingest_pull import resolve_langfuse_host

        assert resolve_langfuse_host({"LANGFUSE_HOST": "https://us.cloud.langfuse.com/"}) == (
            "https://us.cloud.langfuse.com"
        )

    def test_base_url_wins_over_host_like_the_sdk(self):
        """langfuse-python reads LANGFUSE_BASE_URL first, then the older LANGFUSE_HOST."""
        from soup_cli.utils.ingest_pull import resolve_langfuse_host

        environ = {
            "LANGFUSE_BASE_URL": "https://us.cloud.langfuse.com",
            "LANGFUSE_HOST": "https://cloud.langfuse.com",
        }

        assert resolve_langfuse_host(environ) == "https://us.cloud.langfuse.com"

    def test_unset_or_empty_variables_fall_back_to_langfuse_cloud(self):
        from soup_cli.utils.ingest_pull import LANGFUSE_DEFAULT_HOST, resolve_langfuse_host

        both_empty = {"LANGFUSE_BASE_URL": "", "LANGFUSE_HOST": ""}
        base_url_empty = {"LANGFUSE_BASE_URL": "", "LANGFUSE_HOST": "https://us.cloud.langfuse.com"}

        assert resolve_langfuse_host({}) == LANGFUSE_DEFAULT_HOST == "https://cloud.langfuse.com"
        assert resolve_langfuse_host(both_empty) == LANGFUSE_DEFAULT_HOST
        assert resolve_langfuse_host(base_url_empty) == "https://us.cloud.langfuse.com"

    def test_the_shared_webhook_validator_is_the_gate(self, monkeypatch):
        from soup_cli.utils import webhooks
        from soup_cli.utils.ingest_pull import resolve_langfuse_host

        seen = []
        real = webhooks.validate_webhook_url

        def _spy(url, *, allow_private_hosts=False):
            seen.append((url, allow_private_hosts))
            return real(url, allow_private_hosts=allow_private_hosts)

        monkeypatch.setattr(webhooks, "validate_webhook_url", _spy)

        resolve_langfuse_host({"LANGFUSE_HOST": "https://10.0.0.5"}, allow_private_host=True)

        assert seen == [("https://10.0.0.5", True)]

    @pytest.mark.parametrize(
        "host",
        [
            # Private and loopback HTTPS hosts are in the opt-in test below, which
            # also checks the hint and that the flag admits them.
            "http://cloud.langfuse.com",
            "https://169.254.169.254",
            "https://127.1",
            "https://[::1]:3000",
            "ftp://cloud.langfuse.com",
            "https://cloud.langfuse.com/?x=1",
            "https://cloud.langfuse.com/#frag",
            "https://cloud.langfuse.com\n.evil",
        ],
    )
    def test_refused_by_default(self, host):
        from soup_cli.utils.ingest_pull import resolve_langfuse_host

        with pytest.raises(ValueError):
            resolve_langfuse_host({"LANGFUSE_HOST": host})

    @pytest.mark.parametrize(
        "host",
        [
            "https://10.0.0.5",
            "https://192.168.1.20:3000",
            "https://localhost",
            "https://127.0.0.1:3000",
        ],
    )
    def test_private_and_loopback_https_need_the_explicit_opt_in(self, host):
        from soup_cli.utils.ingest_pull import resolve_langfuse_host

        with pytest.raises(ValueError) as info:
            resolve_langfuse_host({"LANGFUSE_HOST": host})
        assert "--allow-private-host" in str(info.value)

        assert resolve_langfuse_host({"LANGFUSE_HOST": host}, allow_private_host=True) == host

    @pytest.mark.parametrize("host", ["http://localhost:3000", "http://10.0.0.5"])
    def test_plain_http_stays_refused_even_with_the_opt_in(self, host):
        from soup_cli.utils.ingest_pull import resolve_langfuse_host

        with pytest.raises(ValueError):
            resolve_langfuse_host({"LANGFUSE_HOST": host}, allow_private_host=True)

    def test_credentials_embedded_in_the_url_are_refused_without_echoing_them(self):
        from soup_cli.utils.ingest_pull import resolve_langfuse_host

        embedded = f"https://{_PUBLIC}:{_SECRET}@cloud.langfuse.com"
        with pytest.raises(ValueError) as info:
            resolve_langfuse_host({"LANGFUSE_HOST": embedded})

        assert not any(token in str(info.value) for token in _TOKENS)


class TestCredentials:
    def test_loads_the_pair_and_the_host(self):
        from soup_cli.utils.ingest_pull import load_langfuse_credentials

        us = "https://us.cloud.langfuse.com"
        creds = load_langfuse_credentials(
            {"LANGFUSE_PUBLIC_KEY": _PUBLIC, "LANGFUSE_SECRET_KEY": _SECRET, "LANGFUSE_HOST": us}
        )

        assert (creds.public_key, creds.secret_key, creds.host) == (_PUBLIC, _SECRET, us)

    @pytest.mark.parametrize(
        ("environ", "missing"),
        [
            ({}, ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]),
            ({"LANGFUSE_PUBLIC_KEY": _PUBLIC}, ["LANGFUSE_SECRET_KEY"]),
            ({"LANGFUSE_SECRET_KEY": _SECRET, "LANGFUSE_PUBLIC_KEY": ""}, ["LANGFUSE_PUBLIC_KEY"]),
        ],
    )
    def test_missing_keys_are_named_and_values_never_echoed(self, environ, missing):
        from soup_cli.utils.ingest_pull import PullError, load_langfuse_credentials

        with pytest.raises(PullError) as info:
            load_langfuse_credentials(environ)

        for name in missing:
            assert name in str(info.value)
        assert not any(token in str(info.value) for token in _TOKENS)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("30m", timedelta(minutes=30)),
        ("24h", timedelta(hours=24)),
        ("7d", timedelta(days=7)),
        ("365d", timedelta(days=365)),
    ],
)
def test_parse_since_accepts_bounded_windows(value, expected):
    from soup_cli.utils.ingest_pull import parse_since

    assert parse_since(value) == expected


@pytest.mark.parametrize(
    "value", ["", "7", "7w", "0d", "-1d", "366d", "1.5h", " 7d", "7d\n", True, 7, None]
)
def test_parse_since_rejects_everything_else(value):
    from soup_cli.utils.ingest_pull import parse_since

    with pytest.raises((TypeError, ValueError)):
        parse_since(value)


# ============================================================
# CLI
# ============================================================


@pytest.fixture
def pull_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", _PUBLIC)
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", _SECRET)
    for name in ("LANGFUSE_HOST", "LANGFUSE_BASE_URL", "SOUP_TELEMETRY"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def _invoke(args):
    from typer.testing import CliRunner

    from soup_cli.cli import app

    return CliRunner().invoke(app, ["ingest", *args])


def _plain(text):
    """Rich highlights numbers with ANSI codes that split phrases like ``1 generation``."""
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _use_transport(monkeypatch, transport):
    from soup_cli.utils import ingest_pull

    monkeypatch.setattr(ingest_pull, "_urllib_transport", transport)
    monkeypatch.setattr(ingest_pull.time, "sleep", lambda seconds: None)


class TestCli:
    def test_pull_writes_parsed_rows(self, pull_env, monkeypatch):
        transport = _Transport(_page([_obs(1)], cursor="c1"), _page([_obs(2)]))
        _use_transport(monkeypatch, transport)

        result = _invoke(
            ["--source", "langfuse", "--pull", "--since", "1d", "--output", "out.jsonl"]
        )

        assert result.exit_code == 0, result.output
        lines = (pull_env / "out.jsonl").read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines]
        assert [(row["trace_id"], row["prompt"], row["output"], row["source"]) for row in rows] == [
            ("obs-1", "q1", "a1", "langfuse"),
            ("obs-2", "q2", "a2", "langfuse"),
        ]
        assert "Wrote 2 traces from langfuse (2 generations pulled)" in _plain(result.output)

    def test_generations_without_io_warn_instead_of_silently_writing_nothing(
        self, pull_env, monkeypatch
    ):
        _use_transport(monkeypatch, _Transport(_page([{"id": "obs-1", "type": "GENERATION"}])))

        result = _invoke(["--source", "langfuse", "--pull", "--output", "out.jsonl"])

        assert result.exit_code == 0, result.output
        output = _plain(result.output)
        assert "0 traces" in output and "1 generations pulled" in output
        assert "1 generation(s) had no input or no output" in output

    def test_page_cap_exits_nonzero_and_leaves_the_previous_output_untouched(
        self, pull_env, monkeypatch
    ):
        """The output streams to a staging file, so a pull stopped on page N neither
        truncates last run's file nor leaves the staging file behind."""
        (pull_env / "out.jsonl").write_text("previous run\n", encoding="utf-8")
        _use_transport(monkeypatch, _Transport(_page([_obs(1)], cursor="more")))

        result = _invoke(
            ["--source", "langfuse", "--pull", "--max-pages", "2", "--output", "out.jsonl"]
        )

        assert result.exit_code == 1
        assert "--max-pages" in result.output and "Nothing was written" in result.output
        assert (pull_env / "out.jsonl").read_text(encoding="utf-8") == "previous run\n"
        assert [path.name for path in pull_env.iterdir()] == ["out.jsonl"]

    def test_missing_credentials_exit_before_any_request(self, pull_env, monkeypatch):
        transport = _Transport(_page([]))
        _use_transport(monkeypatch, transport)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY")

        result = _invoke(["--source", "langfuse", "--pull"])

        assert result.exit_code == 1
        assert "LANGFUSE_SECRET_KEY" in result.output
        assert transport.calls == []

    def test_private_host_is_refused_before_any_request_unless_allowed(self, pull_env, monkeypatch):
        transport = _Transport(_page([]))
        _use_transport(monkeypatch, transport)
        monkeypatch.setenv("LANGFUSE_HOST", "https://10.0.0.5")

        refused = _invoke(["--source", "langfuse", "--pull"])
        allowed = _invoke(
            ["--source", "langfuse", "--pull", "--allow-private-host", "--output", "out.jsonl"]
        )

        assert refused.exit_code == 2
        assert "--allow-private-host" in refused.output
        assert allowed.exit_code == 0, allowed.output
        assert len(transport.calls) == 1
        assert transport.calls[0]["url"].startswith("https://10.0.0.5/api/public/v2/observations?")

    @pytest.mark.parametrize(
        "source", ["langsmith", "helicone", "openpipe", "otel", "openai-stored"]
    )
    def test_pull_is_langfuse_only_for_now(self, pull_env, source):
        result = _invoke(["--source", source, "--pull"])

        assert result.exit_code == 2
        assert "langfuse" in result.output.lower()

    def test_pull_and_logs_are_mutually_exclusive(self, pull_env):
        (pull_env / "lf.jsonl").write_text('{"input":"x","output":"y"}\n', encoding="utf-8")

        result = _invoke(["--source", "langfuse", "--pull", "--logs", "lf.jsonl"])

        assert result.exit_code == 2
        assert "No such option" not in result.output  # not Click refusing an unknown flag
        assert "--logs" in result.output and "--pull" in result.output

    def test_one_of_logs_or_pull_is_required(self, pull_env):
        result = _invoke(["--source", "langfuse"])

        assert result.exit_code == 2
        assert "--logs" in result.output and "--pull" in result.output

    @pytest.mark.parametrize(
        "extra",
        [["--since", "1d"], ["--max-pages", "3"], ["--allow-private-host"]],
        ids=["since", "max-pages", "allow-private-host"],
    )
    def test_pull_only_options_without_pull_are_rejected(self, pull_env, extra):
        (pull_env / "lf.jsonl").write_text('{"input":"x","output":"y"}\n', encoding="utf-8")

        result = _invoke(["--source", "langfuse", "--logs", "lf.jsonl", *extra])

        assert result.exit_code == 2
        assert "No such option" not in result.output  # not Click refusing an unknown flag
        assert extra[0] in result.output and "--pull" in result.output

    def test_bad_since_is_a_usage_error(self, pull_env, monkeypatch):
        transport = _Transport(_page([]))
        _use_transport(monkeypatch, transport)

        result = _invoke(["--source", "langfuse", "--pull", "--since", "forever"])

        assert result.exit_code == 2
        assert transport.calls == []


# ============================================================
# The advisory the local-export path prints
# ============================================================


class TestAuthAdvisory:
    def test_the_pii_panel_names_the_key_pair_langfuse_reads(self, pull_env):
        """``LANGFUSE_KEY`` was invented by #204's issue text; Langfuse reads a
        key pair, so the hint told people to set a variable nothing reads."""
        (pull_env / "lf.jsonl").write_text('{"input":"x","output":"y"}\n', encoding="utf-8")

        result = _invoke(["--source", "langfuse", "--logs", "lf.jsonl", "--output", "out.jsonl"])

        assert result.exit_code == 0, result.output
        assert "LANGFUSE_PUBLIC_KEY" in result.output and "LANGFUSE_SECRET_KEY" in result.output
        assert "LANGFUSE_KEY" not in result.output

    def test_resolve_auth_env_needs_both_halves_of_the_pair(self, monkeypatch):
        from soup_cli.utils.ingest_sources import resolve_auth_env

        for name in ("LANGFUSE_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
            monkeypatch.delenv(name, raising=False)

        monkeypatch.setenv("LANGFUSE_KEY", "legacy")
        assert resolve_auth_env("langfuse") is None

        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", _PUBLIC)
        assert resolve_auth_env("langfuse") is None

        monkeypatch.setenv("LANGFUSE_SECRET_KEY", _SECRET)
        assert resolve_auth_env("langfuse") is not None


# ============================================================
# Without --pull: byte-identical, and no network code loaded
# ============================================================

# Captured from `soup ingest --source langfuse --logs lf.jsonl` on f14fa9f,
# before this change, over the input below.
_GOLDEN_INPUT = (
    '{"id":"t1","input":"What is 2+2?","output":"4","model":"gpt-4o-mini","score":1}\n'
    '{"id":"t2","input":{"messages":[{"role":"user","content":"Привет"}]},'
    '"output":{"content":"Здравствуйте"},"rating":"down"}\n'
    '{"input":"no output here"}\n'
    "not-json\n"
    '{"id":"t4","input":[{"role":"user","content":"list shape"}],"output":"ok","score":0}\n'
)
_GOLDEN_OUTPUT = (
    '{"trace_id": "t1", "prompt": "What is 2+2?", "output": "4", "source": "langfuse", '
    '"signal": "thumbs_up", "metadata": {"model": "gpt-4o-mini"}}\n'
    '{"trace_id": "t2", "prompt": "Привет", "output": "Здравствуйте", "source": "langfuse", '
    '"signal": "thumbs_down", "metadata": {}}\n'
    '{"trace_id": "t4", "prompt": "list shape", "output": "ok", "source": "langfuse", '
    '"signal": "none", "metadata": {}}\n'
).encode("utf-8")


def test_local_export_output_is_byte_identical(pull_env):
    (pull_env / "lf.jsonl").write_text(_GOLDEN_INPUT, encoding="utf-8")

    result = _invoke(["--source", "langfuse", "--logs", "lf.jsonl", "--output", "out.jsonl"])

    assert result.exit_code == 0, result.output
    # The local path writes in text mode, so each line ends in os.linesep (CRLF on
    # Windows, as before this change). JSON escapes newlines inside values, so the
    # golden's only b"\n" bytes are line ends.
    expected = _GOLDEN_OUTPUT.replace(b"\n", os.linesep.encode())
    assert (pull_env / "out.jsonl").read_bytes() == expected


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
    import soup_cli.utils.ingest_pull  # noqa: F401

from typer.testing import CliRunner
from soup_cli.cli import app

result = CliRunner().invoke(
    app, ["ingest", "--source", "langfuse", "--logs", "lf.jsonl", "--output", "out.jsonl"]
)
watched = {"langfuse", "httpx", "soup_cli.utils.ingest_pull"}
loaded = sorted(
    name for name in sys.modules if name in watched or name.split(".")[0] in {"langfuse", "httpx"}
)
print(json.dumps({"exit": result.exit_code, "attempts": attempts, "loaded": loaded}))
"""


def _probe(tmp_path, mode):
    (tmp_path / "lf.jsonl").write_text('{"id":"t1","input":"x","output":"y"}\n', encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if key != "SOUP_TELEMETRY"}
    env["SOUP_NO_AUDIT_LOG"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE, mode],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_local_export_loads_no_pull_code_and_opens_no_socket(tmp_path):
    report = _probe(tmp_path, "local")

    assert report == {"exit": 0, "attempts": [], "loaded": []}


def test_control_the_probe_sees_the_pull_module_when_it_is_imported(tmp_path):
    """CONTROL — without this, a probe that could not see imports would pass."""
    report = _probe(tmp_path, "control")

    assert report["exit"] == 0
    assert "soup_cli.utils.ingest_pull" in report["loaded"]


# ============================================================
# The recorded real response: adapter -> parse_langfuse
# ============================================================

# One real Observations API v2 page, fetched from a Langfuse Cloud Hobby project
# with exactly the query `--pull` sends (fields=core,io,model) after seeding known
# generations with langfuse-python 4.15.2. Sanitised: projectId, internalModelId
# and modelId replaced; no cursor (a last page has `"meta": {}`).
_RECORDED_PAGE = (
    Path(__file__).resolve().parent / "fixtures" / "issue204_langfuse_observations_v2_page.json"
)


def _recorded_rows():
    return _pull(_Transport(_response(200, body=_RECORDED_PAGE.read_bytes())))


class TestRecordedRealResponse:
    def test_the_recorded_page_is_sanitised(self):
        text = _RECORDED_PAGE.read_text(encoding="utf-8")
        rows = json.loads(text)["data"]

        assert {row["projectId"] for row in rows} == {"project-redacted"}
        assert {row["internalModelId"] for row in rows} == {"model-definition-redacted"}
        assert {row["modelId"] for row in rows} == {"model-definition-redacted"}
        for marker in ("pk-lf-", "sk-lf-", "Basic ", "Authorization", "/home/", "@", "cursor"):
            assert marker not in text

    def test_the_page_flows_through_the_adapter_and_parse_langfuse(self):
        from soup_cli.utils.ingest_sources import parse_langfuse

        records = [record.to_dict() for record in parse_langfuse(_recorded_rows())]
        page = json.loads(_RECORDED_PAGE.read_text(encoding="utf-8"))["data"]
        trace_of = {row["id"]: row["traceId"] for row in page}

        # One row per GENERATION: the first two rows are the two LLM calls of one
        # agent trace, whose agent span and tool call yield no row.
        assert trace_of["44d5516f74ed7738"] == trace_of["09c25553e670e9a1"]
        model = {"model": "gpt-4o-mini"}
        assert records == [  # API order: newest startTime first
            {"trace_id": "44d5516f74ed7738", "prompt": "How tall is the Eiffel Tower?\n330 m",
             "output": "The Eiffel Tower is 330 m tall.", "source": "langfuse", "signal": "none",
             "metadata": model},
            {"trace_id": "09c25553e670e9a1",
             "prompt": "How tall is the Eiffel Tower? Decide whether to search.",
             "output": "search: Eiffel Tower height", "source": "langfuse", "signal": "none",
             "metadata": model},
            {"trace_id": "e83a0c5fd3d47039", "prompt": "Translate 'hello' to Spanish.",
             "output": "hola", "source": "langfuse", "signal": "none", "metadata": model},
            {"trace_id": "8dd24f84d2bf7735",
             "prompt": "You are terse.\nWhat is the capital of France?",
             "output": "Paris.", "source": "langfuse", "signal": "none", "metadata": model},
        ]

    def test_the_generation_without_output_is_pulled_but_not_written(self):
        from soup_cli.utils.ingest_sources import parse_langfuse

        rows = _recorded_rows()
        written = {record.trace_id for record in parse_langfuse(rows)}

        assert len(rows) == 5
        assert {row["id"] for row in rows} - written == {"66f27680a0fe7516"}

    def test_control_the_raw_page_alone_gives_json_text_as_the_prompt(self):
        """CONTROL: the real API hands structured input back as JSON text, so a
        parser fed the page without the adapter writes that text as the prompt."""
        from soup_cli.utils.ingest_sources import parse_langfuse

        page = json.loads(_RECORDED_PAGE.read_text(encoding="utf-8"))["data"]
        raw = {record.trace_id: record for record in parse_langfuse(page)}

        assert raw["8dd24f84d2bf7735"].prompt.startswith('[{"role": "system"')

    def test_control_a_bare_message_list_would_keep_only_the_system_prompt(self):
        """CONTROL for the ``{"messages": [...]}`` hand-off: ``_coerce_str`` keeps
        only the first message of a bare list."""
        from soup_cli.utils.ingest_sources import parse_langfuse

        page = json.loads(_RECORDED_PAGE.read_text(encoding="utf-8"))["data"]
        chat = next(row for row in page if row["id"] == "8dd24f84d2bf7735")
        bare = {
            "id": chat["id"],
            "input": json.loads(chat["input"]),
            "output": json.loads(chat["output"]),
        }

        assert next(parse_langfuse([bare])).prompt == "You are terse."


# ============================================================
# paths.atomic_write_lines: the streaming writer behind --pull
# ============================================================


class TestAtomicWriteLines:
    def test_streams_every_line_into_the_target(self, tmp_path, monkeypatch):
        from soup_cli.utils.paths import atomic_write_lines

        monkeypatch.chdir(tmp_path)

        atomic_write_lines((f"row {index}\n" for index in range(3)), "out.jsonl")

        assert (tmp_path / "out.jsonl").read_text(encoding="utf-8") == "row 0\nrow 1\nrow 2\n"
        assert [path.name for path in tmp_path.iterdir()] == ["out.jsonl"]

    def test_an_error_mid_stream_keeps_the_old_target_and_drops_the_staging_file(
        self, tmp_path, monkeypatch
    ):
        from soup_cli.utils.paths import atomic_write_lines

        monkeypatch.chdir(tmp_path)
        (tmp_path / "out.jsonl").write_text("old\n", encoding="utf-8")

        def _lines():
            yield "new 1\n"
            raise RuntimeError("source failed on page 2")

        with pytest.raises(RuntimeError):
            atomic_write_lines(_lines(), "out.jsonl")

        assert (tmp_path / "out.jsonl").read_text(encoding="utf-8") == "old\n"
        assert [path.name for path in tmp_path.iterdir()] == ["out.jsonl"]

    def test_refuses_a_path_outside_cwd(self, tmp_path, monkeypatch):
        from soup_cli.utils.paths import atomic_write_lines

        monkeypatch.chdir(tmp_path)

        with pytest.raises(ValueError):
            atomic_write_lines(iter(["x\n"]), str(tmp_path.parent / "escape.jsonl"))

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink")
    def test_refuses_a_symlinked_target(self, tmp_path, monkeypatch):
        from soup_cli.utils.paths import atomic_write_lines

        monkeypatch.chdir(tmp_path)
        (tmp_path / "real.jsonl").write_text("keep\n", encoding="utf-8")
        (tmp_path / "out.jsonl").symlink_to(tmp_path / "real.jsonl")

        with pytest.raises(ValueError):
            atomic_write_lines(iter(["x\n"]), "out.jsonl")

        assert (tmp_path / "real.jsonl").read_text(encoding="utf-8") == "keep\n"


# ============================================================
# Planted credentials never reach an artefact
# ============================================================


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.lines = []

    def emit(self, record):
        self.lines.append(self.format(record))


def _scenario(name):
    from soup_cli.utils.ingest_pull import PullError

    echo = f"upstream said: Authorization: Basic {_BASIC} ({_SECRET})".encode()
    unreachable = PullError("could not reach Langfuse at https://cloud.langfuse.com")
    denied = _response(401, body=b'{"message":"Invalid credentials"}')
    return {
        "success": (_Transport(_page([_obs(1)], cursor="c1"), _page([_obs(2)])), 0),
        "unauthorized": (_Transport(denied), 1),
        "server_echoes_request": (_Transport(_response(500, body=echo)), 1),
        "page_cap": (_Transport(_page([_obs(1)], cursor="more")), 1),
        "rate_limited": (_Transport(_response(429, headers={"Retry-After": "1"})), 1),
        "network_error": (_Transport(unreachable), 1),
        "unexpected_exception": (_Transport(RuntimeError("transport bug")), 1),
    }[name]


_SCENARIOS = (
    "success",
    "unauthorized",
    "server_echoes_request",
    "page_cap",
    "rate_limited",
    "network_error",
    "unexpected_exception",
)


@pytest.mark.parametrize("name", _SCENARIOS)
def test_planted_credentials_never_reach_any_artefact(name, pull_env, monkeypatch, capsys):
    """Runs through ``soup``'s real entry point (``run()``), so the audit log
    and the ``--verbose`` friendly-error traceback are part of the surface."""
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
            "ingest", "--source", "langfuse", "--pull", "--since", "1d", "--max-pages", "2",
            "--output", "out.jsonl", "--slack-url", "https://hooks.slack.com/services/T0/B0/X0",
        ],
    )
    collected = _Collect()
    loggers = (logging.getLogger("soup"), logging.getLogger())
    # `--log-level debug` reconfigures both loggers process-wide; put them back.
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
    assert any("langfuse pull" in line for line in collected.lines)  # a debug line
    if name == "success":
        assert "file:out.jsonl" in artefacts and payloads
    if name == "unexpected_exception":  # the friendly-error console writes to stderr
        streams = _plain(captured.out + captured.err)
        assert "Full Traceback" in streams and "transport bug" in streams
    leaks = {
        where: found for where, text in artefacts.items() if (found := _leaked_tokens(text))
    }
    assert leaks == {}


def _leaked_tokens(text):
    """Rich can highlight part of a token or fold it across a panel border, so a
    plain substring check would miss it: also look with ANSI codes stripped and
    with whitespace and box-drawing characters removed. A truncated token is
    still a leak (the first 80 chars of the Basic token decode to nearly the
    whole key pair), so a 20-char prefix counts too."""
    import re

    plain = _plain(text)
    compact = re.sub(r"[\s│╭╮╰╯─]+", "", plain)
    views = (text, plain, compact)
    return [
        token
        for token in _TOKENS
        if any(needle in view for needle in (token, token[:20]) for view in views)
    ]


def test_control_the_leak_scan_sees_a_token_rich_highlighted_and_folded():
    """CONTROL for the scan itself: render a key through a narrow Rich panel with
    highlighting on, and it must still be found."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console(file=io.StringIO(), width=24, force_terminal=True, highlight=True)
    console.print(Panel(f"Authorization: Basic {_BASIC} key={_SECRET}"))
    rendered = console.file.getvalue()

    assert _SECRET not in rendered or _BASIC not in rendered  # rich really did split one
    assert set(_leaked_tokens(rendered)) >= {_SECRET, _BASIC}
