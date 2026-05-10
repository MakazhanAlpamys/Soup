"""v0.44.0 Part A — Phone-visible URL + QR code helper.

Pure-Python URL builder; QR rendering lazy-imports `qrcode` so the dep stays
optional. Validation enforces strict scheme + host shape so we never paste an
unsafe URL into the terminal.
"""

from __future__ import annotations

import ipaddress
import re
from typing import Optional
from urllib.parse import urlparse

# Loopback hosts on which plain HTTP is allowed.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

# Token regex — 16-128 chars of urlsafe base64.
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_\-]{16,128}$")


def validate_token(token: str) -> str:
    """Reject a token that isn't urlsafe-base64-shaped.

    Mirrors `secrets.token_urlsafe(N)` output: 16..128 chars of `A-Za-z0-9_-`.
    """
    if not isinstance(token, str):
        raise TypeError("token must be str")
    if not _TOKEN_RE.match(token):
        raise ValueError(
            "token must be 16-128 urlsafe-base64 chars (A-Z, a-z, 0-9, '_', '-')"
        )
    return token


def _host_is_private_ip(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(ip.is_private and not ip.is_loopback)


def build_phone_url(
    *,
    scheme: str,
    host: str,
    port: int,
    token: str,
    path: str = "/",
) -> str:
    """Build the URL that a phone scans.

    Restrictions:
    - scheme must be 'http' or 'https'
    - plain http only allowed on loopback hosts (LAN exposure must be https)
    - port in [1, 65535] (rejects bool, non-int)
    - path must start with '/'
    """
    if scheme not in ("http", "https"):
        raise ValueError("scheme must be http or https")
    if not isinstance(host, str) or not host or "\x00" in host:
        raise ValueError("host must be a non-empty NUL-free str")
    if isinstance(port, bool) or not isinstance(port, int):
        raise ValueError("port must be int")
    if not (1 <= port <= 65535):
        raise ValueError("port must be in [1, 65535]")
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("path must start with '/'")
    if scheme == "http" and host not in _LOOPBACK_HOSTS:
        raise ValueError(
            "plain http only allowed on loopback (localhost / 127.0.0.1 / ::1); "
            "use https for LAN exposure"
        )
    validate_token(token)
    # IPv6 literals must be bracketed in URLs (RFC 3986). Detect by ":" not
    # being a port-only character.
    host_for_url = f"[{host}]" if ":" in host else host
    # Token goes in the query string so the server (FastAPI / static page
    # auth) actually sees it on inbound requests. URL fragments (`#…`) are
    # client-side only — the v0.44.0 first-cut had this wrong.
    separator = "?" if "?" not in path else "&"
    url = f"{scheme}://{host_for_url}:{port}{path}{separator}token={token}"
    parsed = urlparse(url)
    # `urlparse` lowercases hostnames and strips IPv6 brackets; compare
    # against the canonical (unbracketed) host.
    if parsed.scheme != scheme or (parsed.hostname or "").lower() != host.lower():
        raise ValueError("constructed URL failed round-trip validation")
    return url


def render_qr_ascii(url: str) -> Optional[str]:
    """Render `url` as an ASCII QR code. Returns None if `qrcode` is missing.

    Caller decides whether to print or display the result.
    """
    if not isinstance(url, str) or not url:
        raise ValueError("url must be a non-empty str")
    try:
        import qrcode  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        # error_correction=L is fine — terminals print fixed-size cells.
        qr = qrcode.QRCode(border=1, error_correction=qrcode.constants.ERROR_CORRECT_L)
        qr.add_data(url)
        qr.make(fit=True)
    except (ValueError, AttributeError):
        return None
    matrix = qr.get_matrix()
    # Two-row-per-line block rendering keeps QR square in a typical terminal.
    lines = []
    rows = len(matrix)
    for row_idx in range(0, rows, 2):
        cells = []
        for col_idx in range(len(matrix[row_idx])):
            top = matrix[row_idx][col_idx]
            bot = matrix[row_idx + 1][col_idx] if row_idx + 1 < rows else False
            if top and bot:
                cells.append("█")
            elif top and not bot:
                cells.append("▀")
            elif not top and bot:
                cells.append("▄")
            else:
                cells.append(" ")
        lines.append("".join(cells))
    return "\n".join(lines)
