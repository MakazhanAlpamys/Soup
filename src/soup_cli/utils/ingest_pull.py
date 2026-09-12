"""Live Langfuse pull for ``soup ingest --source langfuse --pull`` (#204).

The only network path in ``soup ingest``. Every other invocation parses an
offline export; ``commands/ingest.py`` imports this module only when ``--pull``
is passed, so the local-export path loads no HTTP code.

Design decisions, each pinned by ``tests/test_issue204_langfuse_pull.py``:

- **Standard-library HTTPS, not the ``langfuse`` SDK.** The pull is one
  read-only GET with cursor pagination. The SDK would add OpenTelemetry, wrapt,
  backoff and httpx to the install and would own the retry, timeout, redirect
  and error-formatting behaviour, which is exactly what decides whether a key
  leaks or a loop runs away.
- **Observations API v2.** ``GET /api/public/traces`` is deprecated and is
  removed from Langfuse Cloud on 2026-11-16 (langfuse-python 4.15.2 marks
  ``api.trace.list`` accordingly).
- **One row per GENERATION observation.** A generation is the unit that carries
  a model, the exact input it was given and the output it produced. Rows are
  handed to the existing ``parse_langfuse``, so parsing stays in one place.
- **Credentials stay in memory.** They come from the environment only (never a
  flag, so never in the audit log's argv), the ``Authorization`` header is
  built inline where no local variable holds it, reprs are redacted, and any
  text quoted from a server is scrubbed of the key pair.
- **Bounded.** A wall-clock deadline per request (#865: the socket timeout alone
  let a drip-feeding server hold the call open indefinitely), capped response
  size, a page cap and a row cap that stop with an error instead of truncating,
  a repeated pagination cursor that stops the pull, and a 429 retry budget that
  honours ``Retry-After`` up to a ceiling.
- **Only generations become rows.** ``type=GENERATION`` is sent as a query
  parameter and checked again on each observation (#865), so a server that
  ignores the filter cannot turn spans or tool calls into training rows.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterator, Mapping, Optional
from urllib.parse import urlencode, urlsplit

from soup_cli.utils.ingest_sources import _MAX_INGEST_LINES

LANGFUSE_DEFAULT_HOST = "https://cloud.langfuse.com"
LANGFUSE_OBSERVATIONS_PATH = "/api/public/v2/observations"
# `core` is always returned; the server's default adds `basic`, which carries no
# input/output — without `io` every row would be dropped by the parser.
LANGFUSE_FIELDS = "core,io,model"

PAGE_LIMIT = 100
DEFAULT_MAX_PAGES = 100
MAX_PAGES_LIMIT = 10_000
DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_RATE_LIMIT_RETRIES = 5
RETRY_AFTER_CEILING_SECONDS = 60.0

_BACKOFF_BASE_SECONDS = 2.0
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
# Read in chunks so the deadline is checked while a slow server is still sending.
_READ_CHUNK_BYTES = 256 * 1024
_GENERATION_TYPE = "GENERATION"
_MAX_ERROR_DETAIL_CHARS = 200
_SINCE_RE = re.compile(r"([1-9][0-9]{0,5})([mhd])")
_SINCE_UNITS = {"m": "minutes", "h": "hours", "d": "days"}
_MAX_SINCE = timedelta(days=365)
# langfuse-python's own precedence: LANGFUSE_BASE_URL, then the older LANGFUSE_HOST.
_HOST_ENV_VARS = ("LANGFUSE_BASE_URL", "LANGFUSE_HOST")
_KEY_ENV_VARS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
_PRIVATE_HOST_HINT = " — pass --allow-private-host for a self-hosted Langfuse you trust"

_LOG = logging.getLogger("soup.ingest.pull")


class PullError(RuntimeError):
    """A pull failed. Messages name settings, never their values."""


class PullLimitError(PullError):
    """A cap was reached while results were still pending; nothing is written."""

    def __init__(self, message: str, *, pages: int, rows: int) -> None:
        super().__init__(message)
        self.pages = pages
        self.rows = rows


@dataclass(frozen=True, repr=False)
class LangfuseCredentials:
    """The key pair and the validated base URL. The repr never shows the keys."""

    public_key: str
    secret_key: str
    host: str

    def __repr__(self) -> str:
        return f"LangfuseCredentials(host={self.host!r}, keys=<redacted>)"

    __str__ = __repr__


@dataclass(frozen=True, repr=False)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes

    def __repr__(self) -> str:
        # The body may quote the request back; keep it out of reprs.
        return f"HttpResponse(status={self.status}, body_bytes={len(self.body)})"


@dataclass
class PullStats:
    """What a pull saw, for the summary line the CLI prints (#865)."""

    pages: int = 0
    observations: int = 0
    generations: int = 0
    skipped_not_generation: int = 0


Transport = Callable[[str, LangfuseCredentials, float], HttpResponse]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def parse_since(value: object) -> timedelta:
    """Parse ``--since`` (``30m`` / ``24h`` / ``7d``) into a window of at most 365 days."""
    if isinstance(value, bool) or not isinstance(value, str):
        raise TypeError("--since must be a string such as 30m, 24h or 7d")
    match = _SINCE_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"--since {value!r} must look like 30m, 24h or 7d")
    window = timedelta(**{_SINCE_UNITS[match.group(2)]: int(match.group(1))})
    if window > _MAX_SINCE:
        raise ValueError("--since must be at most 365d")
    return window


def resolve_langfuse_host(
    environ: Mapping[str, str], *, allow_private_host: bool = False
) -> str:
    """Return the validated Langfuse base URL (default: Langfuse Cloud EU).

    HTTPS only, because the key pair travels with every request. The address
    goes through the shared SSRF validator (``utils/webhooks.py``); a private,
    link-local or loopback address additionally needs ``allow_private_host``.
    Error messages never echo the URL, which may embed credentials.
    """
    name, raw = next(
        ((var, environ[var]) for var in _HOST_ENV_VARS if environ.get(var)), (None, "")
    )
    if name is None:
        return LANGFUSE_DEFAULT_HOST
    if not raw.lower().startswith("https://"):
        raise ValueError(
            f"{name} must be an https:// URL — the key pair is sent with every request"
        )

    from soup_cli.utils import webhooks

    try:
        url = webhooks.validate_webhook_url(raw, allow_private_hosts=allow_private_host)
    except (TypeError, ValueError) as exc:
        detail = str(exc).replace("webhook URL", "the URL")
        hint = _PRIVATE_HOST_HINT if not allow_private_host and _is_private(raw) else ""
        raise ValueError(f"{name} is not an allowed Langfuse host: {detail}{hint}") from None

    parts = urlsplit(url)
    if parts.username is not None or parts.password is not None:
        raise ValueError(
            f"{name} must not embed credentials; set LANGFUSE_PUBLIC_KEY and "
            "LANGFUSE_SECRET_KEY instead"
        )
    if parts.query or parts.fragment:
        raise ValueError(f"{name} must be a base URL with no query string or fragment")
    if not allow_private_host and _is_loopback(parts.hostname or ""):
        raise ValueError(f"{name} points at a loopback address{_PRIVATE_HOST_HINT}")
    return url


def load_langfuse_credentials(
    environ: Mapping[str, str], *, allow_private_host: bool = False
) -> LangfuseCredentials:
    """Read the key pair and host from ``environ``.

    Raises :class:`PullError` naming any missing variable, or ``ValueError`` for
    a host that is not allowed.
    """
    missing = [var for var in _KEY_ENV_VARS if not environ.get(var)]
    if missing:
        raise PullError(
            f"--pull needs {' and '.join(_KEY_ENV_VARS)} (missing: {', '.join(missing)}); "
            "create a key pair under the Langfuse project's Settings -> API Keys"
        )
    host = resolve_langfuse_host(environ, allow_private_host=allow_private_host)
    return LangfuseCredentials(
        public_key=environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=environ["LANGFUSE_SECRET_KEY"],
        host=host,
    )


def _is_loopback(host: str) -> bool:
    from soup_cli.utils.net_guard import LOOPBACK_HOSTS, parse_ip_literal

    clean = host.lower().rstrip(".")
    if clean in LOOPBACK_HOSTS:
        return True
    address = parse_ip_literal(clean) if clean else None
    return address is not None and address.is_loopback


def _is_private(url: str) -> bool:
    from soup_cli.utils.net_guard import is_private_or_link_local

    try:
        host = urlsplit(url).hostname or ""
    except ValueError:
        return False
    return bool(host) and (_is_loopback(host) or is_private_or_link_local(host))


# ---------------------------------------------------------------------------
# The pull
# ---------------------------------------------------------------------------


def pull_langfuse_generations(
    credentials: LangfuseCredentials,
    *,
    since: timedelta,
    max_pages: int = DEFAULT_MAX_PAGES,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    now: Optional[datetime] = None,
    transport: Optional[Transport] = None,
    sleep: Optional[Callable[[float], None]] = None,
    stats: Optional[PullStats] = None,
) -> Iterator[Any]:
    """Yield one ``parse_langfuse``-shaped row per GENERATION in the window.

    The window ``[now - since, now)`` is fixed before the first request so
    later pages cannot drift. Only observations whose own ``type`` is
    ``GENERATION`` are yielded, whatever the server did with the query
    parameter (#865); the rest are counted in ``stats``. Raises
    :class:`PullLimitError` when ``max_pages`` or the ingest row cap is reached
    with results still pending, and :class:`PullError` for any other failure,
    including a repeated pagination cursor.
    """
    if isinstance(max_pages, bool) or not isinstance(max_pages, int):
        raise TypeError("max_pages must be an int")
    if not 1 <= max_pages <= MAX_PAGES_LIMIT:
        raise ValueError(f"max_pages must be between 1 and {MAX_PAGES_LIMIT}")
    send = transport if transport is not None else _urllib_transport
    pause = sleep if sleep is not None else time.sleep
    until = (now if now is not None else datetime.now(timezone.utc)).astimezone(timezone.utc)
    window = {
        "type": "GENERATION",
        "fields": LANGFUSE_FIELDS,
        "limit": str(PAGE_LIMIT),
        "fromStartTime": _iso(until - since),
        "toStartTime": _iso(until),
    }
    _LOG.debug(
        "langfuse pull: %s%s from %s to %s",
        credentials.host,
        LANGFUSE_OBSERVATIONS_PATH,
        window["fromStartTime"],
        window["toStartTime"],
    )

    counters = stats if stats is not None else PullStats()
    cursor: Optional[str] = None
    seen_cursors: set[str] = set()
    pages = 0
    rows = 0
    while True:
        query = dict(window, cursor=cursor) if cursor else window
        url = f"{credentials.host}{LANGFUSE_OBSERVATIONS_PATH}?{urlencode(query)}"
        response = _get_with_backoff(send, url, credentials, timeout, pause)
        pages += 1
        counters.pages = pages
        observations, cursor = _decode_page(response, credentials)
        _LOG.debug(
            "langfuse pull: page %d, %d observation(s), %s",
            pages, len(observations), "more pending" if cursor else "last page",
        )
        for observation in observations:
            counters.observations += 1
            if not _is_generation(observation):
                counters.skipped_not_generation += 1
                continue
            rows += 1
            if rows > _MAX_INGEST_LINES:
                raise PullLimitError(
                    f"stopped at the {_MAX_INGEST_LINES}-row ingest cap with more generations "
                    "pending; narrow --since",
                    pages=pages,
                    rows=rows - 1,
                )
            counters.generations += 1
            yield _to_parser_row(observation)
        if not cursor:
            return
        if cursor in seen_cursors:
            raise PullError(
                f"Langfuse returned a pagination cursor it had already sent, after {pages} "
                "page(s); stopping rather than repeating the same request"
            )
        seen_cursors.add(cursor)
        if pages >= max_pages:
            raise PullLimitError(
                f"stopped after {pages} page(s) and {rows} generation(s) with more results "
                f"pending; raise --max-pages (up to {MAX_PAGES_LIMIT}) or narrow --since",
                pages=pages,
                rows=rows,
            )


def _get_with_backoff(
    send: Transport,
    url: str,
    credentials: LangfuseCredentials,
    timeout: float,
    pause: Callable[[float], None],
) -> HttpResponse:
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        response = send(url, credentials, timeout)
        if response.status != 429:
            return response
        if attempt == MAX_RATE_LIMIT_RETRIES:
            break
        delay = _retry_delay(response.headers, attempt)
        _LOG.debug("langfuse pull: HTTP 429, retrying in %.1fs", delay)
        pause(delay)
    raise PullError(
        f"Langfuse kept answering HTTP 429 (rate limited) after {MAX_RATE_LIMIT_RETRIES} "
        "retries; wait and re-run, or narrow --since"
    )


def _retry_delay(headers: Mapping[str, str], attempt: int) -> float:
    seconds = _parse_retry_after(_header(headers, "retry-after"))
    if seconds is None:
        seconds = _BACKOFF_BASE_SECONDS * (2**attempt)
    return float(min(seconds, RETRY_AFTER_CEILING_SECONDS))


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    """``Retry-After`` is delay-seconds or an HTTP-date (RFC 9110 §10.2.3)."""
    if not value:
        return None
    text = value.strip()
    if text.isdigit():
        return float(text)
    from email.utils import parsedate_to_datetime

    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def _decode_page(
    response: HttpResponse, credentials: LangfuseCredentials
) -> tuple[list, Optional[str]]:
    status = response.status
    if 300 <= status < 400:
        raise PullError(
            f"Langfuse answered HTTP {status} with a redirect; refusing to follow it with "
            "credentials attached — point LANGFUSE_HOST at the final URL"
        )
    if status in (401, 403):
        raise PullError(
            f"Langfuse rejected the key pair (HTTP {status}); check LANGFUSE_PUBLIC_KEY and "
            "LANGFUSE_SECRET_KEY, and that LANGFUSE_HOST is the project's region"
        )
    if status != 200:
        raise PullError(
            f"Langfuse answered HTTP {status}: {_error_detail(response.body, credentials)}"
        )
    try:
        payload = json.loads(response.body)
    except ValueError:  # includes UnicodeDecodeError
        raise PullError("Langfuse answered HTTP 200 with a body that is not JSON") from None
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise PullError(
            "Langfuse answered HTTP 200 without a `data` list; the Observations API v2 "
            "response shape may have changed"
        )
    meta = payload.get("meta")
    cursor = meta.get("cursor") if isinstance(meta, dict) else None
    if cursor is not None and not isinstance(cursor, str):
        raise PullError("Langfuse answered HTTP 200 with a non-string pagination cursor")
    return data, cursor or None


def _is_generation(observation: Any) -> bool:
    """``type=GENERATION`` is a request parameter; this is the check on the answer."""
    if not isinstance(observation, dict):
        return False
    kind = observation.get("type")
    return isinstance(kind, str) and kind.strip().upper() == _GENERATION_TYPE


def _to_parser_row(observation: Any) -> Any:
    """Shape one observation the way ``parse_langfuse`` reads a trace export."""
    if not isinstance(observation, dict):
        return observation
    return {
        "id": observation.get("id"),
        "input": _decode_io(observation.get("input")),
        "output": _decode_io(observation.get("output")),
        "model": observation.get("model"),
    }


def _decode_io(value: Any) -> Any:
    """Undo the JSON encoding Observations API v2 applies to structured input/output.

    Recorded against Langfuse Cloud: plain text comes back as-is, but a chat
    input comes back as a string holding JSON, e.g.
    ``'[{"role": "system", ...}, {"role": "user", ...}]'``. A decoded message list
    is handed on as ``{"messages": [...]}``, the shape ``parse_langfuse`` joins —
    given a bare list, its ``_coerce_str`` keeps only the first message, which
    for a chat generation is the system prompt.
    """
    if not isinstance(value, str):
        return value
    try:
        decoded = json.loads(value)
    except ValueError:
        return value
    if isinstance(decoded, list) and decoded and all(
        isinstance(message, dict) and "role" in message for message in decoded
    ):
        return {"messages": decoded}
    if isinstance(decoded, (dict, list)):
        return decoded
    return value


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


def _urllib_transport(url: str, credentials: LangfuseCredentials, timeout: float) -> HttpResponse:
    """One GET with Basic auth. Never follows redirects; bounded in time and size."""
    import http.client
    import urllib.error
    import urllib.request

    from soup_cli import __version__

    deadline = time.monotonic() + timeout
    opener = urllib.request.build_opener(_refuse_redirects_handler())
    # Built inline: no local variable of this frame holds the header value, so a
    # traceback that renders locals cannot show it.
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": _basic_authorization(credentials),
            "Accept": "application/json",
            "User-Agent": f"soup-cli/{__version__} (ingest --pull)",
        },
        method="GET",
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            return HttpResponse(
                status=response.status,
                headers=dict(response.headers.items()),
                body=_read_capped(response, deadline=deadline, timeout=timeout),
            )
    except urllib.error.HTTPError as exc:
        try:
            # The status is what the caller acts on; an error body that cannot be
            # read in time must not turn a clean "HTTP 401" into a raw traceback.
            # PullError belongs here too: since #865 the deadline and the size cap
            # raise it from _read_capped, and it is not an OSError.
            body = _read_capped(exc, deadline=deadline, timeout=timeout)
        except (OSError, http.client.HTTPException, PullError):
            body = b""
        finally:
            exc.close()
        headers = dict(exc.headers.items()) if exc.headers is not None else {}
        return HttpResponse(status=exc.code, headers=headers, body=body)
    except (OSError, http.client.HTTPException, ValueError) as exc:
        reason = getattr(exc, "reason", exc)
        raise PullError(
            _scrub(
                f"could not reach Langfuse at {_origin(url)}: {type(reason).__name__}: {reason}",
                credentials,
            )
        ) from None


def _refuse_redirects_handler():
    """urllib copies ``Authorization`` onto a redirect, even to another host."""
    import urllib.request

    class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
            return None

    return _RefuseRedirects()


def _basic_authorization(credentials: LangfuseCredentials) -> str:
    pair = f"{credentials.public_key}:{credentials.secret_key}".encode("utf-8")
    return "Basic " + base64.b64encode(pair).decode("ascii")


def _read_capped(stream: Any, *, deadline: float, timeout: float) -> bytes:
    """Read the body under both the size cap and the wall-clock deadline (#865).

    ``read1`` returns what has arrived instead of blocking for a full chunk, so
    a server drip-feeding bytes inside the socket timeout is still cut off at
    the deadline. A read already in flight can overshoot by at most the socket
    timeout, which is what bounds a server that stops sending entirely.
    """
    read = getattr(stream, "read1", None) or stream.read
    chunks: list[bytes] = []
    size = 0
    while True:
        if time.monotonic() >= deadline:
            raise PullError(f"the request to Langfuse timed out after {timeout:g}s")
        chunk = read(_READ_CHUNK_BYTES)
        if not chunk:
            return b"".join(chunks)
        size += len(chunk)
        if size > _MAX_RESPONSE_BYTES:
            raise PullError(f"Langfuse response exceeded {_MAX_RESPONSE_BYTES} bytes")
        chunks.append(chunk)


def _origin(url: str) -> str:
    parts = urlsplit(url)
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{parts.hostname}{port}"


def _header(headers: Mapping[str, str], name: str) -> Optional[str]:
    for key, value in headers.items():
        if key.lower() == name:
            return value
    return None


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _error_detail(body: bytes, credentials: LangfuseCredentials) -> str:
    # Scrub the whole window before truncating, so no cut can expose part of a key.
    text = body[: 64 * 1024].decode("utf-8", errors="replace")
    text = " ".join(_scrub(text, credentials).split())
    return text[:_MAX_ERROR_DETAIL_CHARS] or "(empty body)"


def _scrub(text: str, credentials: LangfuseCredentials) -> str:
    token = _basic_authorization(credentials)[len("Basic "):]
    for secret in (token, credentials.secret_key, credentials.public_key):
        if secret:
            text = text.replace(secret, "***")
    return text


__all__ = [
    "DEFAULT_MAX_PAGES",
    "DEFAULT_TIMEOUT_SECONDS",
    "LANGFUSE_DEFAULT_HOST",
    "LANGFUSE_FIELDS",
    "LANGFUSE_OBSERVATIONS_PATH",
    "MAX_PAGES_LIMIT",
    "MAX_RATE_LIMIT_RETRIES",
    "RETRY_AFTER_CEILING_SECONDS",
    "HttpResponse",
    "LangfuseCredentials",
    "PullError",
    "PullLimitError",
    "PullStats",
    "load_langfuse_credentials",
    "parse_since",
    "pull_langfuse_generations",
    "resolve_langfuse_host",
]
