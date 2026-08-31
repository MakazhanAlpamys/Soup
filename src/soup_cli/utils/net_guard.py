"""#616 — shared SSRF loopback/private-host predicate.

``_is_private_or_link_local`` and ``_LOOPBACK_HOSTS`` were copied across
``utils/hf.py`` and ``utils/hubs.py`` (and the loopback set alone into
``utils/loop_stages.py`` and ``utils/qr_url.py``) — the same shape that
already cost this repo a fix landing in one copy and not the others three
times before (#372, #392, #424). This module is the one definition; the four
call sites import it under their original private names so no caller
changes.
"""

from __future__ import annotations

import ipaddress

# Loopback hosts that may legitimately use plain HTTP (dev / self-hosted).
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def is_private_or_link_local(host: str) -> bool:
    """Whether ``host`` is a private / link-local / loopback IP.

    Handles canonical IPv4/IPv6 via :func:`ipaddress.ip_address` **and**
    abbreviated / decimal / hex / octal IPv4 forms (e.g. ``127.1``,
    ``2130706433``, ``0x7f000001``, ``0177.0.0.1``) via the platform C
    library :func:`socket.inet_aton`.  The latter is a pure in-process
    string parser — no DNS lookup is performed.
    """
    import socket  # noqa: PLC0415 — lazy import (stdlib, negligible cost)

    clean_host = host.rstrip(".")
    try:
        addr = ipaddress.ip_address(clean_host)
        return addr.is_private or addr.is_link_local or addr.is_loopback
    except ValueError:
        pass
    # Fallback: C-level inet_aton accepts abbreviated/integer/hex/octal
    # IPv4 representations that Python's ipaddress module rejects.
    try:
        canonical = socket.inet_ntoa(socket.inet_aton(clean_host))
        addr = ipaddress.ip_address(canonical)
        return addr.is_private or addr.is_link_local or addr.is_loopback
    except (OSError, ValueError):
        # Hostname — we don't resolve DNS here (the SDK does), so fall
        # back to "treat as public". A malicious DNS record pointing to
        # a private IP is out of scope for this local-tool threat model.
        return False
