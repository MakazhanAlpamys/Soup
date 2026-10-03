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

A seventh copy of the *parsing* half (not the whole predicate) lived in
``loop_stages._endpoint_is_local``: same abbreviated-IPv4 gap as ``tracing.py``,
found after the six above were already folded in. That function's OR-policy
is deliberately narrower than ``is_private_or_link_local`` — it does not
treat reserved/multicast ranges as "local enough to deploy to", and widening
it to match would be a real policy change to what the deploy-canary surface
trusts, not a rename. ``parse_ip_literal`` below is the shared parsing step
(canonical + abbreviated/decimal/hex/octal IPv4, no DNS); ``loop_stages.py``
builds its own narrower OR-chain on top of it rather than calling
``is_private_or_link_local`` directly.
"""

from __future__ import annotations

import ipaddress

# Loopback hosts that may legitimately use plain HTTP (dev / self-hosted).
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _ascii_spellings(host: str) -> list[str]:
    """``host`` plus the ASCII forms an HTTP client may turn it into.

    A client IDNA-encodes a non-ASCII host before it resolves it: urllib and
    :mod:`socket` through the stdlib codec (IDNA 2003, whose nameprep step
    NFKC-folds and drops characters such as the soft hyphen), httpx through the
    ``idna`` package (IDNA 2008). Both treat U+3002, U+FF0E and U+FF61 as label
    separators, so text that is not an IP literal as written -- ``10.0.0.1``
    with U+3002 IDEOGRAPHIC FULL STOP in place of each dot, for instance -- can
    be one once encoded. A form that fails to encode is skipped.
    """
    if host.isascii():
        return [host]
    spellings = [host]
    try:
        spellings.append(host.encode("idna").decode("ascii"))
    except UnicodeError:
        pass
    try:
        import idna  # noqa: PLC0415 — the encoder httpx uses; optional here

        spellings.append(idna.encode(host.lower()).decode("ascii"))
    except (ImportError, UnicodeError):  # idna.IDNAError is a UnicodeError
        pass
    return spellings


def parse_ip_literal(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Parse ``host`` as an IP literal, or ``None`` if it isn't one.

    Handles canonical IPv4/IPv6 via :func:`ipaddress.ip_address` **and**
    abbreviated / decimal / hex / octal IPv4 forms (e.g. ``127.1``,
    ``2130706433``, ``0x7f000001``, ``0177.0.0.1``) via :func:`inet_aton`, a
    pure-Python port of the BSD parser, so the answer is the same on Linux,
    macOS and Windows (the C library's differs: Windows refuses every spelling
    of ``255.255.255.255`` except the dotted quad, #1550). No DNS lookup is
    performed. A hostname (``"localhost"``, ``"evil.example.com"``) returns
    ``None`` rather than being resolved.

    A non-ASCII host is also read in the ASCII forms a client would connect to
    (see :func:`_ascii_spellings`), and the first form that is an IP literal is
    returned, so ``10.0.0.1`` written with ideographic full stops parses as
    ``10.0.0.1``.
    """
    for spelling in _ascii_spellings(host):
        addr = _parse_ascii_ip_literal(spelling)
        if addr is not None:
            return addr
    return None


def _parse_ascii_ip_literal(
    host: str,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    clean_host = host.rstrip(".")
    try:
        return ipaddress.ip_address(clean_host)
    except ValueError:
        pass
    # Fallback: the BSD inet_aton grammar accepts abbreviated/integer/hex/octal
    # IPv4 representations that Python's ipaddress module rejects.
    packed = inet_aton(clean_host)
    if packed is None:
        return None
    return ipaddress.IPv4Address(packed)


# Upper bound of the last part of a 1-, 2-, 3- or 4-part BSD inet_aton
# spelling: that part fills every byte the earlier parts did not.
_INET_ATON_LAST_PART_MAX = {1: 0xFFFFFFFF, 2: 0xFFFFFF, 3: 0xFFFF, 4: 0xFF}

# What C's ``isspace()`` calls whitespace in the C locale: the C parsers stop
# there and read only the text before it.
_C_SPACE = frozenset(" \t\n\v\f\r")


# The digits each base accepts. ``int(digits, base)`` is not a substitute: it
# also takes its own radix prefix, so ``int("0xa", 16)`` would let a second
# prefix through, and it takes "_" separators and surrounding space.
_INET_ATON_DIGITS = {
    10: frozenset("0123456789"),
    8: frozenset("01234567"),
    16: frozenset("0123456789abcdefABCDEF"),
}


def _inet_aton_part(part: str) -> int | None:
    """One dot-separated part, read like C ``strtoul(part, &end, 0)`` would."""
    if not part or not part[0].isdigit() or not part.isascii():
        return None
    base = 10
    digits = part
    if part[0] == "0" and len(part) > 1:
        if part[1] in "xX":
            base, digits = 16, part[2:]
        else:
            base, digits = 8, part[1:]
    if not digits or not set(digits) <= _INET_ATON_DIGITS[base]:
        return None  # includes a bare "0x", which needs at least one hex digit
    return int(digits, base)


def inet_aton(text: str) -> int | None:
    """Parse ``text`` with the BSD ``inet_aton`` grammar, in pure Python.

    Returns the address as an int, or ``None`` if ``text`` is not an IPv4
    spelling. Accepts one to four ``.``-separated parts; each part is a C
    number (``0x`` hex, leading-``0`` octal, else decimal, no sign); the last
    part fills every byte the earlier parts left (``127.1`` is ``127.0.0.1``,
    ``1.65535`` is ``1.0.255.255``, ``2130706433`` is ``127.0.0.1``). Unlike
    the C library's, the result does not depend on the platform: Windows
    refuses ``0xffffffff``, ``4294967295`` and ``0377.0377.0377.0377`` where
    glibc reads ``255.255.255.255`` (#1550).

    Like the C parsers, the text ends at the first ASCII whitespace and only
    what precedes it is read, so ``"10.0.0.1 x"`` is ``10.0.0.1`` while
    ``"10.0.0.1x"`` is not an address at all.
    """
    for i, ch in enumerate(text):
        if ch in _C_SPACE:
            text = text[:i]
            break
    parts = text.split(".")
    if not 1 <= len(parts) <= 4:
        return None
    values = []
    for part in parts:
        value = _inet_aton_part(part)
        if value is None:
            return None
        values.append(value)
    *head, last = values
    if any(v > 0xFF for v in head) or last > _INET_ATON_LAST_PART_MAX[len(values)]:
        return None
    packed = 0
    for v in head:
        packed = (packed << 8) | v
    return (packed << (8 * (5 - len(values)))) | last


# IPv6 prefixes that carry an IPv4 address inside them. The inner address is
# what the packet reaches, so it is classified too, whatever the interpreter's
# ``ipaddress`` tables say about the outer prefix (6to4 counts as private only
# from 3.10.15 / 3.11.10 / 3.12.4, #1550).
_SIXTOFOUR = ipaddress.IPv6Network("2002::/16")
_TEREDO = ipaddress.IPv6Network("2001::/32")
_NAT64 = ipaddress.IPv6Network("64:ff9b::/96")


def embedded_ipv4(addr: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    """The IPv4 address an IPv6 transition address embeds, or ``None``.

    IPv4-mapped ``::ffff:a.b.c.d`` (low 32 bits), 6to4 ``2002::/16`` (bits
    16-47), NAT64 ``64:ff9b::/96`` (low 32 bits) and Teredo ``2001::/32``
    (low 32 bits, inverted). Unwrapped here by hand rather than through
    ``IPv6Address.sixtofour`` / ``.teredo`` so the set does not depend on the
    interpreter either.
    """
    bits = int(addr)
    if addr.ipv4_mapped is not None:
        return addr.ipv4_mapped
    if addr in _SIXTOFOUR:
        return ipaddress.IPv4Address((bits >> 80) & 0xFFFFFFFF)
    if addr in _NAT64:
        return ipaddress.IPv4Address(bits & 0xFFFFFFFF)
    if addr in _TEREDO:
        return ipaddress.IPv4Address((~bits) & 0xFFFFFFFF)
    return None


def is_private_or_link_local(host: str) -> bool:
    """Whether ``host`` is a private / link-local / loopback / reserved /
    multicast / unspecified IP — the union of every guard's policy in this
    repo, since consolidating stops being safe the moment a caller's
    predicate is quietly narrower than the shared one it now uses.

    Also any address ``ipaddress`` does not consider globally reachable
    (``not is_global``) — that adds RFC 6598 shared address space,
    ``100.64.0.0/10``, which is neither ``is_private`` nor ``is_global`` —
    and the deprecated IPv6 site-local block ``fec0::/10``, which
    ``ipaddress`` still reports as global. ``is_reserved`` stays: NAT64
    ``64:ff9b::/96`` IS ``is_global``.
    """
    addr = parse_ip_literal(host)
    if addr is None:
        # Hostname — we don't resolve DNS here (the SDK does), so fall
        # back to "treat as public". A malicious DNS record pointing to
        # a private IP is out of scope for this local-tool threat model.
        return False
    return is_non_public_address(addr)


def is_non_public_address(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """:func:`is_private_or_link_local` for an already-parsed address.

    An IPv6 address that embeds an IPv4 address (see :func:`embedded_ipv4`)
    is non-public when either the outer or the inner address is.
    """
    if isinstance(addr, ipaddress.IPv6Address):
        inner = embedded_ipv4(addr)
        if inner is not None and is_non_public_address(inner):
            return True
    return (
        addr.is_private
        or addr.is_link_local
        or addr.is_loopback
        or addr.is_unspecified
        or addr.is_reserved
        or addr.is_multicast
        or not addr.is_global
        or getattr(addr, "is_site_local", False)
    )


def refuse_private_ip_literal(host: str | None, *, label: str) -> None:
    """Raise ``ValueError`` if an outbound endpoint's host is a non-public IP.

    ``host`` is a URL's ``hostname``. It is refused when it is an IP literal, in
    any spelling :func:`parse_ip_literal` accepts (IPv4-mapped IPv6 included),
    that :func:`is_private_or_link_local` classifies as non-public, on every
    scheme. Loopback stays allowed, because a server on the same machine
    (Ollama, vLLM, ``soup serve``) is a supported target. A hostname passes
    unchanged and is NOT resolved, so this narrows which addresses a URL can
    name directly; it does not make internal services unreachable by name. An
    empty or missing host passes too: the caller's scheme check owns that case.
    ``label`` names the setting in the message.
    """
    clean = (host or "").strip().lower().rstrip(".")
    if clean.startswith("[") and clean.endswith("]"):
        clean = clean[1:-1]
    if not clean or clean in LOOPBACK_HOSTS:
        return
    addr = parse_ip_literal(clean)
    if addr is None:
        return
    mapped = getattr(addr, "ipv4_mapped", None)
    if (mapped if mapped is not None else addr).is_loopback:
        return
    if is_non_public_address(addr):
        raise ValueError(
            f"{label}: private/link-local/reserved IP hosts are not allowed (SSRF protection)"
        )
