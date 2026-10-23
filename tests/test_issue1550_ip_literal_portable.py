"""#1550: IP-literal parsing does not depend on the platform or the interpreter.

``parse_ip_literal`` fell back to the C library's ``inet_aton``, which differs
by platform: Windows refuses ``0xffffffff``, ``4294967295`` and
``0377.0377.0377.0377`` (every spelling of ``255.255.255.255`` except the
dotted quad) that glibc reads, so those spellings passed the guard there. And
IPv6 forms that embed an IPv4 address were classified only through the
interpreter's ``ipaddress`` tables: 6to4 ``2002::/16`` counts as private only
from 3.10.15 / 3.11.10 / 3.12.4.

The parser is now pure Python, the embedded IPv4 address is unwrapped and
classified by hand, and ``refuse_private_ip_literal`` parses once.
"""

from __future__ import annotations

import ipaddress

import pytest

from soup_cli.utils.net_guard import (
    is_private_or_link_local,
    parse_ip_literal,
    refuse_private_ip_literal,
)

# ``inet_aton`` and ``embedded_ipv4`` are new in this change; they are imported
# inside the tests that need them so the file still collects on an older tree
# and the other tests measure what that tree gets wrong.

# (spelling, dotted quad) -- the same answer is expected on Linux, macOS and
# Windows, which is what the CI matrix checks.
INET_ATON_TABLE = [
    ("255.255.255.255", "255.255.255.255"),
    ("0xffffffff", "255.255.255.255"),  # Windows C library: refused
    ("0XFFFFFFFF", "255.255.255.255"),
    ("4294967295", "255.255.255.255"),  # Windows C library: refused
    ("0377.0377.0377.0377", "255.255.255.255"),  # Windows C library: refused
    ("037777777777", "255.255.255.255"),
    ("127.1", "127.0.0.1"),
    ("127.0.1", "127.0.0.1"),
    ("2130706433", "127.0.0.1"),
    ("0x7f000001", "127.0.0.1"),
    ("0x7f.1", "127.0.0.1"),
    ("0177.0.0.1", "127.0.0.1"),
    ("0177.1", "127.0.0.1"),
    ("1.2.3", "1.2.0.3"),
    ("1.65535", "1.0.255.255"),
    ("1.256", "1.0.1.0"),
    ("1.0x10000", "1.1.0.0"),
    ("10.0x0.1", "10.0.0.1"),
    ("0", "0.0.0.0"),
    ("00", "0.0.0.0"),
    ("167772161", "10.0.0.1"),
    ("0xa000001", "10.0.0.1"),
    ("012.0.0.1", "10.0.0.1"),
    ("0xC0.0xA8.0x1.0x1", "192.168.1.1"),
    # The text ends at the first ASCII whitespace; the prefix is the address.
    ("10.0.0.1 x", "10.0.0.1"),
    ("10.1 x", "10.0.0.1"),
    ("167772161 x", "10.0.0.1"),
    ("10.0.0.1\x0bx", "10.0.0.1"),  # vertical tab
    ("10.0.0.1 ", "10.0.0.1"),
    ("127.0.0.1 x", "127.0.0.1"),  # still loopback
    ("1 ", "0.0.0.1"),
]

NOT_INET_ATON = [
    "",
    ".1",
    "1..2",
    "1.2.3.4.5",
    "4294967296",  # one past the top of the address space
    "1.2.3.256",
    "1.2.65536",
    "256.1",
    "08.1.1.1",  # 8 is not an octal digit
    "0x1g",
    "+1",
    "-1",
    " 1",  # leading space is not skipped: the prefix is empty
    " ",
    "1_0",  # int() would take the separator; C does not
    "0_x",
    "١",  # ARABIC-INDIC DIGIT ONE: int() would take it; C does not
    "localhost",
    "evil.example.com",
    "1.2.3.4a",
    "10.0.0.1x",  # no whitespace to end the text: the whole string is read
    # A radix prefix is written once. int(digits, base) would take a second
    # one, and a bare "0x" needs at least one hex digit after it.
    "0x",
    "0x.1",
    "0x0xa.1",  # int("0xa", 16)
    "00o12.0.0.1",  # int("0o12", 8)
    "0X0X1",
]


@pytest.mark.parametrize(("spelling", "dotted"), INET_ATON_TABLE)
def test_inet_aton_reads_every_spelling_the_same_everywhere(spelling: str, dotted: str) -> None:
    """The pure-Python parser gives one answer on every OS."""
    from soup_cli.utils.net_guard import inet_aton

    packed = inet_aton(spelling)
    assert packed is not None, spelling
    assert str(ipaddress.IPv4Address(packed)) == dotted


@pytest.mark.parametrize("spelling", NOT_INET_ATON)
def test_inet_aton_refuses_what_is_not_an_address(spelling: str) -> None:
    """Nothing outside the BSD grammar is read as an address."""
    from soup_cli.utils.net_guard import inet_aton

    assert inet_aton(spelling) is None


@pytest.mark.parametrize(("spelling", "dotted"), INET_ATON_TABLE)
def test_parse_ip_literal_uses_the_portable_parser(spelling: str, dotted: str) -> None:
    """``parse_ip_literal`` reaches the same dotted quad through the fallback."""
    assert parse_ip_literal(spelling) == ipaddress.IPv4Address(dotted)


@pytest.mark.parametrize(
    "spelling",
    ["0xffffffff", "4294967295", "0377.0377.0377.0377", "037777777777"],
)
def test_broadcast_spellings_are_refused_on_every_platform(spelling: str) -> None:
    """The spellings Windows' C library let through are refused like the dotted quad."""
    assert is_private_or_link_local(spelling)
    with pytest.raises(ValueError, match="SSRF"):
        refuse_private_ip_literal(spelling, label="x")


def test_parse_ip_literal_does_not_use_the_c_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """The platform ``socket.inet_aton`` is not consulted at all."""
    import socket

    def boom(_text: str) -> bytes:
        raise AssertionError("socket.inet_aton must not be called")

    monkeypatch.setattr(socket, "inet_aton", boom)
    assert parse_ip_literal("0xffffffff") == ipaddress.IPv4Address("255.255.255.255")
    assert parse_ip_literal("127.1") == ipaddress.IPv4Address("127.0.0.1")
    assert parse_ip_literal("example.com") is None


# --- IPv6 forms that embed an IPv4 address ---------------------------------

EMBEDDED = [
    pytest.param("2002:0a00:0001::", "10.0.0.1", id="6to4"),
    pytest.param("64:ff9b::a00:1", "10.0.0.1", id="nat64"),
    # Teredo: the client IPv4 address is the inverted low 32 bits
    # (10.0.0.1 -> f5ff:fffe).
    pytest.param("2001:0:4136:e378:8000:63bf:f5ff:fffe", "10.0.0.1", id="teredo"),
    pytest.param("::ffff:10.0.0.1", "10.0.0.1", id="v4-mapped"),
]


@pytest.mark.parametrize(("literal", "inner"), EMBEDDED)
def test_embedded_ipv4_is_unwrapped_by_hand(literal: str, inner: str) -> None:
    """The inner address comes out of the right bits of each prefix."""
    from soup_cli.utils.net_guard import embedded_ipv4

    assert embedded_ipv4(ipaddress.IPv6Address(literal)) == ipaddress.IPv4Address(inner)


@pytest.mark.parametrize("literal", ["2606:4700::1111", "2001:db8::1", "fd00::1", "::1"])
def test_other_ipv6_embeds_nothing(literal: str) -> None:
    """Prefixes outside the four transition forms carry no inner address."""
    from soup_cli.utils.net_guard import embedded_ipv4

    assert embedded_ipv4(ipaddress.IPv6Address(literal)) is None


@pytest.fixture
def old_ipaddress_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make IPv6 look the way an interpreter with the pre-2024 tables saw it.

    Every outer-prefix classification says "public", so only the unwrapped
    inner address can refuse the literal (as the mapped-loopback test in
    ``test_outbound_endpoint_ip_literals.py`` does for ``is_loopback``).
    """
    v6 = ipaddress.IPv6Address
    for name in ("is_private", "is_reserved", "is_link_local", "is_multicast", "is_site_local"):
        monkeypatch.setattr(v6, name, property(lambda self: False))
    monkeypatch.setattr(v6, "is_global", property(lambda self: True))


@pytest.mark.usefixtures("old_ipaddress_tables")
@pytest.mark.parametrize(("literal", "inner"), EMBEDDED)
def test_embedded_private_ipv4_is_refused_on_every_interpreter(literal: str, inner: str) -> None:
    """6to4, NAT64 and Teredo around 10.0.0.1 are refused even with old tables."""
    assert is_private_or_link_local(literal)
    with pytest.raises(ValueError, match="SSRF"):
        refuse_private_ip_literal(literal, label="x")
    with pytest.raises(ValueError, match="SSRF"):
        refuse_private_ip_literal(f"[{literal}]", label="x")


@pytest.mark.usefixtures("old_ipaddress_tables")
def test_old_tables_fixture_really_lets_the_outer_prefix_through() -> None:
    """Guard on the fixture: with the tables patched, the outer prefix alone says public."""
    addr = ipaddress.IPv6Address("2002:0a00:0001::")
    assert not addr.is_private and addr.is_global


def test_embedded_public_ipv4_does_not_refuse_by_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unwrapping only adds an inner address: NAT64 around 8.8.8.8 follows the outer prefix."""
    v6 = ipaddress.IPv6Address
    for name in ("is_private", "is_reserved", "is_link_local", "is_multicast", "is_site_local"):
        monkeypatch.setattr(v6, name, property(lambda self: False))
    monkeypatch.setattr(v6, "is_global", property(lambda self: True))
    assert not is_private_or_link_local("64:ff9b::808:808")


def test_refuse_parses_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """``refuse_private_ip_literal`` parses the host a single time."""
    from soup_cli.utils import net_guard

    calls: list[str] = []
    real = net_guard.parse_ip_literal

    def counting(host: str):
        calls.append(host)
        return real(host)

    monkeypatch.setattr(net_guard, "parse_ip_literal", counting)
    with pytest.raises(ValueError):
        net_guard.refuse_private_ip_literal("10.0.0.1", label="x")
    assert calls == ["10.0.0.1"]


# --- hosts ending in a character the stdlib IDNA codec folds to a space ------

# The stdlib ``idna`` codec (nameprep, NFKC) turns U+00A0 NO-BREAK SPACE and
# U+3000 IDEOGRAPHIC SPACE into an ASCII space, so such a host is read like the
# same host ending in " ": the address before the space, as the C parsers do.
IDNA_FOLDED_SPACE = [
    pytest.param("10.0.0.1\u00a0", "10.0.0.1", id="nbsp"),
    pytest.param("10.0.0.1\u3000", "10.0.0.1", id="ideographic-space"),
    pytest.param("10.1\u00a0", "10.0.0.1", id="nbsp-short-form"),
    pytest.param("167772161\u3000", "10.0.0.1", id="ideographic-space-decimal"),
    pytest.param("10.0.0.1\u00a0x", "10.0.0.1", id="nbsp-then-text"),
    pytest.param("10.0.0.1\u3000x", "10.0.0.1", id="ideographic-space-then-text"),
]


@pytest.mark.parametrize(("host", "prefix"), IDNA_FOLDED_SPACE)
def test_idna_folded_trailing_space_reads_as_prefix(host: str, prefix: str) -> None:
    """A host ending in U+00A0 / U+3000 reads as the address before it."""
    # The premise: the codec a client encodes the host with turns each of the
    # two characters into an ASCII space and changes nothing else.
    folded = host.replace("\u00a0", " ").replace("\u3000", " ")
    assert host.encode("idna").decode("ascii") == folded
    assert parse_ip_literal(host) == ipaddress.IPv4Address(prefix)
    assert is_private_or_link_local(host)
    with pytest.raises(ValueError, match="SSRF"):
        refuse_private_ip_literal(host, label="x")


@pytest.mark.parametrize("host", ["127.0.0.1\u00a0", "127.0.0.1\u3000x"])
def test_idna_folded_trailing_space_keeps_loopback(host: str) -> None:
    """The same reading keeps a loopback address allowed."""
    assert parse_ip_literal(host) == ipaddress.IPv4Address("127.0.0.1")
    refuse_private_ip_literal(host, label="x")


@pytest.mark.parametrize(
    "text",
    [
        "10.0.0.1\x1cx",  # str.isspace() says yes; C isspace() says no
        "10.0.0.1\x1fx",
        "10.0.0.1\x85",
        "10.0.0.1\u00a0",  # folded to a space by the codec, not by the parser
        "10.0.0.1\u3000",
        "10.0.0.1\u2028",
    ],
)
def test_only_c_whitespace_ends_the_text(text: str) -> None:
    """Only the six C whitespace characters end the text; after any other, it is no address."""
    from soup_cli.utils.net_guard import inet_aton

    assert inet_aton(text) is None
