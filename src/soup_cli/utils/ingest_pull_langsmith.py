"""Live LangSmith pull for ``soup ingest --source langsmith --pull`` (#204).

The LangSmith slice of #204, after the Langfuse one (``utils/ingest_pull.py``).
``commands/ingest.py`` imports this module only when ``--pull`` is passed with
``--source langsmith``, so the local-export path loads no HTTP code.

Design decisions, each pinned by ``tests/test_ingest_pull_langsmith.py``:

- **Its own module, the Langfuse code unchanged.** LangSmith's runs query is a
  POST with a JSON body, an ``X-API-Key`` header and the cursor in the body;
  the Langfuse transport is one GET with Basic auth. What is provider-neutral is
  used from ``ingest_pull`` as it is: ``parse_since``, the ``Retry-After``
  parsing, the capped and deadline-bound read, the response and error classes,
  and the retry and page limits. They are reached through the module
  (``ingest_pull._read_capped``), so a change there is seen here.
- **Standard-library HTTPS, not the ``langsmith`` SDK**, for the reasons in
  ``ingest_pull``: the SDK would own the retry, timeout, redirect and
  error-formatting behaviour, which is what decides whether a key leaks or a
  loop runs away.
- **Runs API v2.** ``POST /api/v2/runs/query`` with ``project_ids``, so the
  project named by ``LANGSMITH_PROJECT`` is first resolved to its id with
  ``GET /api/v1/sessions?name=``.

  Seen on a LangSmith Cloud account on 2026-11-15 (an observation, not
  documentation): the sessions lookup returns a list with the project's ``id``;
  v2 returns only ``id`` unless ``selects`` asks for the fields; the filter
  takes ``run_type: "LLM"`` and the runs come back as ``"llm"``; without
  ``min_start_time`` it covers only the last day, so the window is always sent;
  a last page has ``next_cursor: null``.

  Read in langsmith-py 0.14.17's ``query_v2`` only, not seen on the account (no
  second page was returned there): the next page is asked for with ``cursor``
  in the body, ``max_start_time`` bounds the window, and ``page_size`` is 1 to
  1000.
- **One row per LLM run.** An LLM run holds the exact model input and output;
  chain and tool runs are not model calls. ``run_type`` is sent as a filter and
  checked again on each run, so a server that ignores the filter cannot turn
  them into training rows. Runs are handed to ``parse_langsmith`` as they come.
- **The key stays in memory.** It comes from the environment only (never a
  flag, so never in the audit log's argv), the ``X-API-Key`` header is built
  inline where no local variable holds it, reprs are redacted, and any text
  quoted from a server is scrubbed of it.
- **Bounded** like the Langfuse pull: a wall-clock deadline per request, a
  capped response size, a page cap and a row cap that stop with an error
  instead of truncating, a repeated cursor that stops the pull, and a 429 retry
  budget that honours ``Retry-After`` up to a ceiling.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterator, Mapping, Optional
from urllib.parse import urlencode, urlsplit

from soup_cli.utils import ingest_pull
from soup_cli.utils.ingest_pull import (
    DEFAULT_MAX_PAGES,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_PAGES_LIMIT,
    MAX_RATE_LIMIT_RETRIES,
    RETRY_AFTER_CEILING_SECONDS,
    HttpResponse,
    PullError,
    PullLimitError,
)
from soup_cli.utils.ingest_sources import _MAX_INGEST_LINES

LANGSMITH_DEFAULT_HOST = "https://api.smith.langchain.com"
LANGSMITH_SESSIONS_PATH = "/api/v1/sessions"
LANGSMITH_RUNS_QUERY_PATH = "/api/v2/runs/query"
# What parse_langsmith reads, plus what the pull checks; without `selects` v2 returns `id` only.
LANGSMITH_SELECTS = ("ID", "NAME", "RUN_TYPE", "START_TIME", "INPUTS", "OUTPUTS", "FEEDBACK_STATS")

PAGE_SIZE = 100

_BACKOFF_BASE_SECONDS = 2.0
_LLM_RUN_TYPE = "llm"
_MAX_ERROR_DETAIL_CHARS = 200
_HOST_ENV_VAR = "LANGSMITH_ENDPOINT"
_KEY_ENV_VAR = "LANGSMITH_API_KEY"
_PROJECT_ENV_VAR = "LANGSMITH_PROJECT"
# langsmith-py strips these from LANGSMITH_ENDPOINT to get the base the API paths hang off.
_API_SUFFIXES = ("/api/v1", "/api")
_PRIVATE_HOST_HINT = " — pass --allow-private-host for a self-hosted LangSmith you trust"

_LOG = logging.getLogger("soup.ingest.pull")


@dataclass(frozen=True, repr=False)
class LangSmithSettings:
    """The API key, the validated base URL and the project name. The repr never shows the key."""

    api_key: str
    host: str
    project: str

    def __repr__(self) -> str:
        return f"LangSmithSettings(host={self.host!r}, project={self.project!r}, key=<redacted>)"

    __str__ = __repr__


@dataclass
class LangSmithPullStats:
    """What a pull saw, for the summary line the CLI prints."""

    pages: int = 0
    runs: int = 0
    llm_runs: int = 0
    skipped_not_llm: int = 0


Transport = Callable[[str, str, Optional[bytes], LangSmithSettings, float], HttpResponse]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def resolve_langsmith_host(
    environ: Mapping[str, str], *, allow_private_host: bool = False
) -> str:
    """Return the validated LangSmith base URL (default: LangSmith Cloud, US).

    ``LANGSMITH_ENDPOINT`` is read the way langsmith-py reads it: a trailing
    ``/api/v1`` or ``/api`` is dropped, since the API paths are added here.
    HTTPS only, because the key travels with every request; the address goes
    through the shared SSRF validator (``utils/webhooks.py``), and a private,
    link-local or loopback address additionally needs ``allow_private_host``.
    Error messages never echo the URL, which may embed credentials.
    """
    raw = environ.get(_HOST_ENV_VAR) or ""
    if not raw:
        return LANGSMITH_DEFAULT_HOST
    name = _HOST_ENV_VAR
    if not raw.lower().startswith("https://"):
        raise ValueError(f"{name} must be an https:// URL — the API key is sent with every request")

    from soup_cli.utils import webhooks

    try:
        url = webhooks.validate_webhook_url(raw, allow_private_hosts=allow_private_host)
    except (TypeError, ValueError) as exc:
        detail = str(exc).replace("webhook URL", "the URL")
        hint = _PRIVATE_HOST_HINT if not allow_private_host and ingest_pull._is_private(raw) else ""
        raise ValueError(f"{name} is not an allowed LangSmith host: {detail}{hint}") from None

    parts = urlsplit(url)
    if parts.username is not None or parts.password is not None:
        raise ValueError(f"{name} must not embed credentials; set {_KEY_ENV_VAR} instead")
    if parts.query or parts.fragment:
        raise ValueError(f"{name} must be a base URL with no query string or fragment")
    if not allow_private_host and ingest_pull._is_loopback(parts.hostname or ""):
        raise ValueError(f"{name} points at a loopback address{_PRIVATE_HOST_HINT}")
    for suffix in _API_SUFFIXES:
        if url.endswith(suffix):
            return url[: -len(suffix)]
    return url


def load_langsmith_settings(
    environ: Mapping[str, str], *, allow_private_host: bool = False
) -> LangSmithSettings:
    """Read the key, the project and the host from ``environ``.

    Raises :class:`PullError` naming any missing variable, or ``ValueError`` for
    a host that is not allowed.
    """
    missing = [var for var in (_KEY_ENV_VAR, _PROJECT_ENV_VAR) if not environ.get(var)]
    if missing:
        raise PullError(
            f"--pull needs {_KEY_ENV_VAR} and {_PROJECT_ENV_VAR} (missing: {', '.join(missing)}); "
            "create an API key in LangSmith's settings, and set "
            f"{_PROJECT_ENV_VAR} to the tracing project's name"
        )
    host = resolve_langsmith_host(environ, allow_private_host=allow_private_host)
    return LangSmithSettings(
        api_key=environ[_KEY_ENV_VAR], host=host, project=environ[_PROJECT_ENV_VAR]
    )


# ---------------------------------------------------------------------------
# The pull
# ---------------------------------------------------------------------------


def pull_langsmith_runs(
    settings: LangSmithSettings,
    *,
    since: timedelta,
    max_pages: int = DEFAULT_MAX_PAGES,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    now: Optional[datetime] = None,
    transport: Optional[Transport] = None,
    sleep: Optional[Callable[[float], None]] = None,
    stats: Optional[LangSmithPullStats] = None,
) -> Iterator[Any]:
    """Yield each LLM run of the project in the window, shaped as ``parse_langsmith`` reads it.

    The window ``[now - since, now]`` is fixed before the first request so later
    pages cannot drift. Only runs whose own ``run_type`` is ``llm`` are yielded,
    whatever the server did with the filter; the rest are counted in ``stats``.
    Raises :class:`PullLimitError` when ``max_pages`` or the ingest row cap is
    reached with results still pending, and :class:`PullError` for any other
    failure, including an unknown project and a repeated pagination cursor.
    """
    if isinstance(max_pages, bool) or not isinstance(max_pages, int):
        raise TypeError("max_pages must be an int")
    if not 1 <= max_pages <= MAX_PAGES_LIMIT:
        raise ValueError(f"max_pages must be between 1 and {MAX_PAGES_LIMIT}")
    send = transport if transport is not None else _urllib_transport
    pause = sleep if sleep is not None else time.sleep
    until = (now if now is not None else datetime.now(timezone.utc)).astimezone(timezone.utc)

    project_id = _resolve_project_id(send, settings, timeout, pause)
    window = {
        "project_ids": [project_id],
        "run_type": "LLM",
        "min_start_time": _iso(until - since),
        "max_start_time": _iso(until),
        "page_size": PAGE_SIZE,
        "selects": list(LANGSMITH_SELECTS),
    }
    url = f"{settings.host}{LANGSMITH_RUNS_QUERY_PATH}"
    _LOG.debug(
        "langsmith pull: %s for project %s from %s to %s",
        url, _scrub(project_id, settings), window["min_start_time"], window["max_start_time"],
    )

    counters = stats if stats is not None else LangSmithPullStats()
    cursor: Optional[str] = None
    seen_cursors: set[str] = set()
    pages = 0
    rows = 0
    while True:
        body = dict(window, cursor=cursor) if cursor else window
        response = _send_with_backoff(
            send, "POST", url, json.dumps(body).encode("utf-8"), settings, timeout, pause
        )
        pages += 1
        counters.pages = pages
        runs, cursor = _decode_page(response, settings)
        _LOG.debug(
            "langsmith pull: page %d, %d run(s), %s",
            pages, len(runs), "more pending" if cursor else "last page",
        )
        for run in runs:
            counters.runs += 1
            if not _is_llm_run(run):
                counters.skipped_not_llm += 1
                continue
            rows += 1
            if rows > _MAX_INGEST_LINES:
                raise PullLimitError(
                    f"stopped at the {_MAX_INGEST_LINES}-row ingest cap with more LLM runs "
                    "pending; narrow --since",
                    pages=pages,
                    rows=rows - 1,
                )
            counters.llm_runs += 1
            yield run
        if not cursor:
            return
        if cursor in seen_cursors:
            raise PullError(
                f"LangSmith returned a pagination cursor it had already sent, after {pages} "
                "page(s); stopping rather than repeating the same request"
            )
        seen_cursors.add(cursor)
        if pages >= max_pages:
            raise PullLimitError(
                f"stopped after {pages} page(s) and {rows} LLM run(s) with more results "
                f"pending; raise --max-pages (up to {MAX_PAGES_LIMIT}) or narrow --since",
                pages=pages,
                rows=rows,
            )


def _resolve_project_id(
    send: Transport, settings: LangSmithSettings, timeout: float, pause: Callable[[float], None]
) -> str:
    """Find the id of the project named ``settings.project``; v2 queries by id only."""
    query = urlencode({"name": settings.project, "limit": "1"})
    url = f"{settings.host}{LANGSMITH_SESSIONS_PATH}?{query}"
    _LOG.debug("langsmith pull: looking up project %r at %s", settings.project, settings.host)
    response = _send_with_backoff(send, "GET", url, None, settings, timeout, pause)
    payload = _decode_json(response, settings)
    if not isinstance(payload, list):
        raise PullError(
            "LangSmith answered the project lookup without a list; the sessions API "
            "response shape may have changed"
        )
    # The name is a query parameter; the answer is checked, as run_type is.
    for project in payload:
        if isinstance(project, dict) and project.get("name") == settings.project:
            project_id = project.get("id")
            if not isinstance(project_id, str) or not project_id:
                raise PullError("LangSmith answered the project lookup without a project id")
            return project_id
    raise PullError(
        f"LangSmith has no tracing project named {settings.project!r} for this key; check "
        f"{_PROJECT_ENV_VAR} and that the key belongs to the project's workspace"
    )


def _send_with_backoff(
    send: Transport,
    method: str,
    url: str,
    body: Optional[bytes],
    settings: LangSmithSettings,
    timeout: float,
    pause: Callable[[float], None],
) -> HttpResponse:
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        response = send(method, url, body, settings, timeout)
        if response.status != 429:
            return response
        if attempt == MAX_RATE_LIMIT_RETRIES:
            break
        delay = _retry_delay(response.headers, attempt)
        _LOG.debug("langsmith pull: HTTP 429, retrying in %.1fs", delay)
        pause(delay)
    raise PullError(
        f"LangSmith kept answering HTTP 429 (rate limited) after {MAX_RATE_LIMIT_RETRIES} "
        "retries; wait and re-run, or narrow --since"
    )


def _retry_delay(headers: Mapping[str, str], attempt: int) -> float:
    seconds = ingest_pull._parse_retry_after(_header(headers, "retry-after"))
    if seconds is None:
        seconds = _BACKOFF_BASE_SECONDS * (2**attempt)
    return float(min(seconds, RETRY_AFTER_CEILING_SECONDS))


def _decode_json(response: HttpResponse, settings: LangSmithSettings) -> Any:
    status = response.status
    if 300 <= status < 400:
        raise PullError(
            f"LangSmith answered HTTP {status} with a redirect; refusing to follow it with the "
            f"API key attached — point {_HOST_ENV_VAR} at the final URL"
        )
    if status in (401, 403):
        raise PullError(
            f"LangSmith rejected the API key (HTTP {status}); check {_KEY_ENV_VAR}, and that "
            f"{_HOST_ENV_VAR} is the workspace's region"
        )
    if status != 200:
        raise PullError(
            f"LangSmith answered HTTP {status}: {_error_detail(response.body, settings)}"
        )
    try:
        return json.loads(response.body)
    except ValueError:  # includes UnicodeDecodeError
        raise PullError("LangSmith answered HTTP 200 with a body that is not JSON") from None


def _decode_page(response: HttpResponse, settings: LangSmithSettings) -> tuple[list, Optional[str]]:
    payload = _decode_json(response, settings)
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise PullError(
            "LangSmith answered HTTP 200 without an `items` list; the Runs API v2 response "
            "shape may have changed"
        )
    cursor = payload.get("next_cursor")
    if cursor is not None and not isinstance(cursor, str):
        raise PullError("LangSmith answered HTTP 200 with a non-string pagination cursor")
    return items, cursor or None


def _is_llm_run(run: Any) -> bool:
    """``run_type: "LLM"`` is a request field; this is the check on the answer."""
    if not isinstance(run, dict):
        return False
    kind = run.get("run_type")
    return isinstance(kind, str) and kind.strip().lower() == _LLM_RUN_TYPE


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


def _urllib_transport(
    method: str, url: str, body: Optional[bytes], settings: LangSmithSettings, timeout: float
) -> HttpResponse:
    """One request with the API key. Never follows redirects; bounded in time and size."""
    import http.client
    import urllib.error
    import urllib.request

    from soup_cli import __version__

    deadline = time.monotonic() + timeout
    opener = urllib.request.build_opener(ingest_pull._refuse_redirects_handler())
    headers = {
        "Accept": "application/json",
        "User-Agent": f"soup-cli/{__version__} (ingest --pull)",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    # Added on the request, not kept in a local: a traceback that renders this
    # frame's locals shows ``headers`` and ``request``, and neither holds the key.
    request.add_unredirected_header("X-API-Key", settings.api_key)
    try:
        with opener.open(request, timeout=timeout) as response:
            return HttpResponse(
                status=response.status,
                headers=dict(response.headers.items()),
                body=_read(response, deadline=deadline, timeout=timeout),
            )
    except urllib.error.HTTPError as exc:
        try:
            # The status is what the caller acts on; an error body that cannot be
            # read in time must not turn a clean "HTTP 401" into a raw traceback.
            error_body = _read(exc, deadline=deadline, timeout=timeout)
        except (OSError, http.client.HTTPException, PullError):
            error_body = b""
        finally:
            exc.close()
        error_headers = dict(exc.headers.items()) if exc.headers is not None else {}
        return HttpResponse(status=exc.code, headers=error_headers, body=error_body)
    except (OSError, http.client.HTTPException, ValueError) as exc:
        reason = getattr(exc, "reason", exc)
        raise PullError(
            _scrub(
                f"could not reach LangSmith at {ingest_pull._origin(url)}: "
                f"{type(reason).__name__}: {reason}",
                settings,
            )
        ) from None


def _read(stream: Any, *, deadline: float, timeout: float) -> bytes:
    """``ingest_pull._read_capped``, with its errors naming LangSmith."""
    try:
        return ingest_pull._read_capped(stream, deadline=deadline, timeout=timeout)
    except PullError as exc:
        raise PullError(str(exc).replace("Langfuse", "LangSmith")) from None


def _header(headers: Mapping[str, str], name: str) -> Optional[str]:
    for key, value in headers.items():
        if key.lower() == name:
            return value
    return None


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _error_detail(body: bytes, settings: LangSmithSettings) -> str:
    # Scrub the whole window before truncating, so no cut can expose part of the key.
    text = body[: 64 * 1024].decode("utf-8", errors="replace")
    text = " ".join(_scrub(text, settings).split())
    return text[:_MAX_ERROR_DETAIL_CHARS] or "(empty body)"


def _scrub(text: str, settings: LangSmithSettings) -> str:
    if settings.api_key:
        text = text.replace(settings.api_key, "***")
    return text


__all__ = [
    "LANGSMITH_DEFAULT_HOST",
    "LANGSMITH_RUNS_QUERY_PATH",
    "LANGSMITH_SELECTS",
    "LANGSMITH_SESSIONS_PATH",
    "PAGE_SIZE",
    "LangSmithPullStats",
    "LangSmithSettings",
    "load_langsmith_settings",
    "pull_langsmith_runs",
    "resolve_langsmith_host",
]
