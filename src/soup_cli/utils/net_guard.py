"""#616 — shared SSRF loopback/private-host predicate.

``_is_private_or_link_local`` / ``_LOOPBACK_HOSTS`` shaped guards were copied
across six call sites: the predicate itself in ``utils/hf.py``, ``utils/
hubs.py`` and ``utils/webhooks.py``, the loopback set alone into ``utils/
loop_stages.py`` and ``utils/qr_url.py``, and a differently-named variant
(``_is_private_ip``) plus its own loopback set in ``utils/tracing.py``. This
is the same shape that already cost this repo a fix landing in one copy and
not the others three times before (#372, #392, #424). All six now import
this one definition under their original private names, so no caller
signature changes.

The three source predicates did not agree, and reconciling them is a real
behaviour change, not a rename — documented here rather than silently
absorbed:

- ``hf.py`` / ``hubs.py`` / ``loop_stages.py`` / ``qr_url.py`` checked only
  ``is_private``, ``is_link_local``, ``is_loopback``. They now also reject
  reserved and multicast ranges (previously accepted).
- ``webhooks.py`` already rejected reserved/multicast; folding it in changes
  nothing for its callers.
- ``tracing.py``'s ``_is_private_ip`` had no fallback for abbreviated /
  decimal / hex / octal IPv4 forms (the #604 fix). Its OTLP endpoint
  validator was reachable with e.g. ``https://127.1:4317`` or
  ``https://2130706433:4317``. It now gets the same protection as every
  other endpoint validator in this repo — closing a real bypass, not just
  deduplicating.
"""

from __future__ import annotations

import ipaddress

# Loopback hosts that may legitimately use plain HTTP (dev / self-hosted).
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def is_private_or_link_local(host: str) -> bool:
    """Whether ``host`` is a private / link-local / loopback / reserved /
    multicast / unspecified IP — the union of every guard's policy in this
    repo, since consolidating stops being safe the moment a caller's
    predicate is quietly narrower than the shared one it now uses.

    Handles canonical IPv4/IPv6 via :func:`ipaddress.ip_address` **and**
    abbreviated / decimal / hex / octal IPv4 forms (e.g. ``127.1``,
    ``2130706433``, ``0x7f000001``, ``0177.0.0.1``) via the platform C
    library :func:`socket.inet_aton`.  The latter is a pure in-process
    string parser — no DNS lookup is performed.
    """
    import socket  # noqa: PLC0415 — lazy import (stdlib, negligible cost)

    def _rejects(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
        return (
            addr.is_private
            or addr.is_link_local
            or addr.is_loopback
            or addr.is_unspecified
            or addr.is_reserved
            or addr.is_multicast
        )

    clean_host = host.rstrip(".")
    try:
        return _rejects(ipaddress.ip_address(clean_host))
    except ValueError:
        pass
    # Fallback: C-level inet_aton accepts abbreviated/integer/hex/octal
    # IPv4 representations that Python's ipaddress module rejects.
    try:
        canonical = socket.inet_ntoa(socket.inet_aton(clean_host))
        return _rejects(ipaddress.ip_address(canonical))
    except (OSError, ValueError):
        # Hostname — we don't resolve DNS here (the SDK does), so fall
        # back to "treat as public". A malicious DNS record pointing to
        # a private IP is out of scope for this local-tool threat model.
        return False
