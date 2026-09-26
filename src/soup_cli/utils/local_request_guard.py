"""Host and Origin checks for routes of a locally bound HTTP server.

A server bound to a loopback or LAN address can still receive requests that a
browser sends on behalf of a page from another site: the page names some other
hostname, the name resolves to the bound address, and the request arrives with
that other hostname in its ``Host`` header (and the page's site in ``Origin``).
Checking both headers against the bound address refuses such requests before
the route runs.

Hostnames are compared WITHOUT ports. The port in ``Host`` is whatever the
client used to reach the server, which differs from the bind port behind a port
forward or a container mapping, and an ``Origin`` port names the page's server
(a dev server on :5173 calling the API on :8000 is an ordinary local setup).
The hostname is the part that says which site the request is for.

Dependency-free on purpose: imported by ``soup serve`` at app construction, and
must not pull in the optional ``mcp`` SDK or any heavy dependency.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

LOOPBACK_HOSTNAMES: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
WILDCARD_HOSTS: frozenset[str] = frozenset({"", "0.0.0.0", "::", "[::]", "*"})

_HOST_NOT_ALLOWED = (421, "Host not allowed")
_ORIGIN_NOT_ALLOWED = (403, "Origin not allowed")

_BRACKETED_RE = re.compile(r"^\[([^\[\]]+)\](?::(\d+))?$")
_PLAIN_RE = re.compile(r"^([^:\[\]]+)(?::(\d+))?$")


def hostname_from_host_header(value: str | None) -> str | None:
    """Return the lowercased hostname of a ``Host`` header, without its port.

    ``None`` for a missing, empty, whitespace-containing or malformed value.
    """
    if not value or any(ch.isspace() for ch in value):
        return None
    match = _BRACKETED_RE.match(value) or _PLAIN_RE.match(value)
    if match is None:
        return None
    return match.group(1).lower()


def hostname_from_origin(value: str | None) -> str | None:
    """Return the lowercased hostname of an http(s) ``Origin`` header, else ``None``."""
    if not value:
        return None
    try:
        parts = urlsplit(value)
        hostname = parts.hostname
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not hostname:
        return None
    return hostname.lower()


def _normalise_bind(bind_host: str) -> str:
    bind = bind_host.strip().lower()
    if bind.startswith("[") and bind.endswith("]"):
        bind = bind[1:-1]
    return bind


def _allowed_hostnames(bind: str) -> frozenset[str]:
    """The hostnames that name an already-normalised, non-wildcard bind."""
    return LOOPBACK_HOSTNAMES if bind in LOOPBACK_HOSTNAMES else frozenset({bind})


def _check_origin(
    bind: str, host_header: str | None, origin_header: str | None
) -> tuple[int, str] | None:
    """The ``Origin`` half of the policy, for an already-normalised bind.

    A missing ``Origin`` is acceptable: only a browser is obliged to send one.
    """
    if origin_header is None:
        return None
    origin_name = hostname_from_origin(origin_header)
    if bind in WILDCARD_HOSTS:
        # No single name to compare against, so the page must name whatever
        # hostname this very request was addressed to.
        if origin_name is None or origin_name != hostname_from_host_header(host_header):
            return _ORIGIN_NOT_ALLOWED
        return None
    if origin_name not in _allowed_hostnames(bind):
        return _ORIGIN_NOT_ALLOWED
    return None


def check_local_request(
    bind_host: str, host_header: str | None, origin_header: str | None
) -> tuple[int, str] | None:
    """Decide whether a request's ``Host`` / ``Origin`` name the bound server.

    Returns ``None`` when the request is acceptable, otherwise a
    ``(status_code, reason)`` pair: 421 for a foreign or missing ``Host``,
    403 for a foreign ``Origin``.

    A wildcard bind (``0.0.0.0``, ``::``) has no single name to compare ``Host``
    against, so only ``Origin`` is checked there, and it must match ``Host``.
    """
    bind = _normalise_bind(bind_host)
    if bind in WILDCARD_HOSTS:
        return _check_origin(bind, host_header, origin_header)
    if hostname_from_host_header(host_header) not in _allowed_hostnames(bind):
        return _HOST_NOT_ALLOWED
    return _check_origin(bind, host_header, origin_header)


def check_browser_origin(
    bind_host: str, host_header: str | None, origin_header: str | None
) -> tuple[int, str] | None:
    """Decide on ``Origin`` alone, ignoring ``Host``.

    The narrower half of :func:`check_local_request`, for routes where the
    full check would cost more than it buys. No ``Origin`` header means the
    request is allowed — curl, an SDK and a reverse proxy all send none — and
    an ``Origin`` that is present must name the bound server by exactly the
    rule :func:`check_local_request` already applies.

    ``Host`` is deliberately NOT checked here. An inference route is the one a
    reverse proxy legitimately fronts under some other hostname, so refusing a
    foreign ``Host`` would refuse every proxied deployment; a browser, by
    contrast, always sends ``Origin`` on a cross-site request, which is the
    case this is here to refuse. Returns 403 for a foreign ``Origin``, never
    421.
    """
    return _check_origin(_normalise_bind(bind_host), host_header, origin_header)
