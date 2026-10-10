"""#1550 follow-up: a host written as numbers only is an IPv4 literal or it is refused.

``parse_ip_literal`` reads IPv4 spellings with one strict grammar on every
platform (#1562). C libraries do not all agree with that grammar on text that
looks like a number but is not an address under it: a number too large for its
position (``4294967296``, ``10.0.0.256``), a ``0x`` with no hex digit after it
(``10.0x.0.1``), an ``8`` in an octal part (``08.0.0.1``), a fifth part. Some
of them refuse such text and some read an address from it, so what a host like
that names would depend on the platform.

The two refusing predicates, ``is_private_or_link_local`` and
``refuse_private_ip_literal``, therefore never treat a host made only of
numeric labels as a hostname: it is a valid IPv4 literal, classified as before,
or it is refused. The parser is unchanged. It still answers ``None`` for such
text, so a caller that asks whether an address is loopback or local gets the
same answer as before.
"""

from __future__ import annotations

import ipaddress

import pytest
from pydantic import ValidationError

from soup_cli.utils import net_guard
from soup_cli.utils.net_guard import (
    inet_aton,
    is_private_or_link_local,
    parse_ip_literal,
    refuse_private_ip_literal,
)

_NUMERIC = (
    "an all-numeric host that is not a valid IPv4 address is not allowed (SSRF protection); "
    "use a valid IPv4 address or a hostname"
)
# How the non-public refusal starts. What follows it is not pinned in this file.
_NON_PUBLIC = "private/link-local/reserved IP hosts are not allowed (SSRF protection)"

# Every label is a number and the whole is not an IPv4 literal under the strict
# grammar. Plain ASCII, lower case, no whitespace: these go through URLs too.
NUMERIC_NOT_AN_ADDRESS = [
    # A number too large for its position.
    pytest.param("4294967296", id="one-part-2pow32"),
    pytest.param("4462739457", id="one-part-above-2pow32"),
    pytest.param("0x10a000001", id="one-part-hex-above-2pow32"),
    pytest.param("040000000000", id="one-part-octal-2pow32"),
    pytest.param("18446744073709719777", id="one-part-above-2pow64"),
    pytest.param("0x1000000000000000a", id="one-part-hex-above-2pow64"),
    pytest.param("10.18446744073709551617", id="last-part-above-2pow64"),
    pytest.param("10.4294967297", id="last-part-above-2pow32"),
    pytest.param("10.16777216", id="last-part-above-24-bits"),
    pytest.param("10.0.65536", id="last-part-above-16-bits"),
    pytest.param("10.0.0.256", id="last-part-above-8-bits"),
    pytest.param("256.0.0.1", id="first-part-above-8-bits"),
    pytest.param("4294967306.0.0.1", id="first-part-above-2pow32"),
    pytest.param("18446744073709551626.0.0.1", id="first-part-above-2pow64"),
    # A 0x prefix with no hex digit after it.
    pytest.param("0x.1", id="empty-hex-first-part"),
    pytest.param("0x.0", id="empty-hex-then-zero"),
    pytest.param("10.0x.0.1", id="empty-hex-inner-part"),
    pytest.param("0x.0x.0x.1", id="empty-hex-three-parts"),
    pytest.param("10.0x", id="empty-hex-last-part"),
    pytest.param("0x", id="bare-0x"),
    # An 8 or a 9 in a part that starts with 0.
    pytest.param("08.0.0.1", id="octal-with-8"),
    pytest.param("012.0.0.09", id="octal-with-9-last"),
    pytest.param("09", id="one-part-octal-with-9"),
    # More than four parts.
    pytest.param("1.2.3.4.5", id="five-parts"),
    pytest.param("10.0.0.0.1", id="five-parts-with-zero"),
]

# The same hosts in the other forms the predicates accept: a trailing dot,
# upper case, text after C whitespace, and the non-ASCII forms a client folds
# to ASCII before it connects (see ``net_guard._ascii_spellings``).
NUMERIC_NOT_AN_ADDRESS_OTHER_FORMS = [
    pytest.param("4462739457.", id="trailing-dot"),
    pytest.param("10.0x.0.1.", id="empty-hex-trailing-dot"),
    pytest.param("0X10A000001", id="upper-case-hex"),
    pytest.param("0X.1", id="upper-case-empty-hex"),
    pytest.param("4462739457 x", id="space-then-text"),
    pytest.param("10.0x.0.1\tx", id="tab-then-text"),
    pytest.param("4462739457\x0bx", id="vertical-tab-then-text"),
    pytest.param("0x.1 ", id="trailing-space"),
    pytest.param(
        "\uff14\uff14\uff16\uff12\uff17\uff13\uff19\uff14\uff15\uff17", id="fullwidth-digits"
    ),
    pytest.param("10\u300218446744073709551617", id="ideographic-full-stop"),
    pytest.param("0x\uff0e1", id="fullwidth-full-stop"),
    pytest.param("4462739457\u3000x", id="ideographic-space-then-text"),
    pytest.param("44627\u00ad39457", id="soft-hyphen"),
]

# At least one label is not a number: these are names, whatever else is in them.
HOSTNAMES = [
    pytest.param("api.example.com", id="plain"),
    pytest.param("cbc.ca", id="hex-letters-only"),
    pytest.param("db.de", id="hex-letters-only-short"),
    pytest.param("deadbeef", id="one-label-hex-letters"),
    pytest.param("cafe.babe", id="two-labels-hex-letters"),
    pytest.param("3f4e5a6b7c8d", id="hex-digits-no-prefix"),
    pytest.param("1a", id="digit-then-letter"),
    pytest.param("1e100.net", id="digit-led-domain"),
    pytest.param("0x.org", id="0x-under-a-name"),
    pytest.param("0xabc.de", id="hex-number-under-a-name"),
    pytest.param("0xdead.beef", id="hex-number-then-letters"),
    pytest.param("10.0.0.1.nip.io", id="address-under-a-name"),
    pytest.param("1.2.3.4.example.com", id="numbers-under-a-name"),
    pytest.param("4294967296.example.com", id="large-number-under-a-name"),
    pytest.param("0x10a000001.internal", id="large-hex-under-a-name"),
    pytest.param("10-0-0-1", id="hyphens"),
    pytest.param("1_0", id="underscore"),
    pytest.param("0b1010.0.0.1", id="binary-prefix"),
    pytest.param("0o12.0.0.1", id="python-octal-prefix"),
    pytest.param("0x0xa.1", id="two-hex-prefixes"),
    pytest.param("0x1g", id="hex-with-g"),
    pytest.param("10.0.0.1x", id="letter-after-address"),
    pytest.param("4462739457x", id="letter-after-number"),
    pytest.param("1..2", id="empty-inner-label"),
    pytest.param(".1", id="empty-first-label"),
    pytest.param("+1", id="plus-sign"),
    pytest.param("-1", id="minus-sign"),
    pytest.param("10.0.0.1%20x", id="percent-escape"),
    pytest.param("4462739457\x1cx", id="not-c-whitespace"),
    pytest.param("m\u00fcnchen.example", id="idn"),
    pytest.param("\u0661\u0660.0.0.1", id="arabic-indic-digits"),
]

LOOPBACK = [
    "localhost",
    "LOCALHOST",
    "localhost.",
    "127.0.0.1",
    "::1",
    "[::1]",
    "::ffff:127.0.0.1",
    "127.1",
    "127.0.1",
    "2130706433",
    "0x7f000001",
    "0x7f.1",
    "0177.0.0.1",
    "127.0.0.1 x",
    "127\u30020\u30020\u30021",
]

# Numeric spellings the strict grammar does read keep their own classification.
PUBLIC_NUMERIC = ["8.8.8.8", "134744072", "0x8080808", "010.010.010.010", "1.1"]
NON_PUBLIC_NUMERIC = ["10.0.0.1", "167772161", "0xa.1", "012.0.0.1", "0xffffffff", "0"]


def _refusal(host: str) -> str:
    with pytest.raises(ValueError) as excinfo:
        refuse_private_ip_literal(host, label="x")
    return str(excinfo.value)


# --- the rule ----------------------------------------------------------------


@pytest.mark.parametrize("host", NUMERIC_NOT_AN_ADDRESS + NUMERIC_NOT_AN_ADDRESS_OTHER_FORMS)
def test_numeric_host_that_is_not_an_address_is_refused(host: str) -> None:
    """Both predicates refuse it, although the parser says it is no address."""
    assert parse_ip_literal(host) is None  # the premise
    assert is_private_or_link_local(host) is True
    assert _refusal(host) == f"x: {_NUMERIC}"


@pytest.mark.parametrize("host", ["4462739457", "0x.1", "10.0x.0.1"])
def test_numeric_host_is_refused_in_brackets_too(host: str) -> None:
    """``refuse_private_ip_literal`` takes the brackets off first, as it does for IPv6."""
    assert _refusal(f"[{host}]") == f"x: {_NUMERIC}"


def test_the_refusal_does_not_repeat_the_host() -> None:
    """The message is fixed text: a caller may show it without echoing the endpoint."""
    message = _refusal("4462739457")
    assert message == f"x: {_NUMERIC}"
    assert "4462739457" not in message
    # The label is the only part that varies.
    with pytest.raises(ValueError) as excinfo:
        refuse_private_ip_literal("4462739457", label="judge URL")
    assert str(excinfo.value) == f"judge URL: {_NUMERIC}"


def test_refuse_still_parses_once_for_a_numeric_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """The extra check reuses the one parse; it does not parse again."""
    calls: list[str] = []
    real = net_guard.parse_ip_literal

    def counting(host: str):
        calls.append(host)
        return real(host)

    monkeypatch.setattr(net_guard, "parse_ip_literal", counting)
    with pytest.raises(ValueError, match="all-numeric"):
        net_guard.refuse_private_ip_literal("4462739457", label="x")
    assert calls == ["4462739457"]


# --- what the rule leaves alone ---------------------------------------------


@pytest.mark.parametrize("host", HOSTNAMES)
def test_host_with_a_label_that_is_not_a_number_stays_a_hostname(host: str) -> None:
    """A name is not refused, even one made only of digits, dots and a-f."""
    assert parse_ip_literal(host) is None
    assert is_private_or_link_local(host) is False
    refuse_private_ip_literal(host, label="x")


@pytest.mark.parametrize("host", ["", ".", "..", "\t"])
def test_host_with_nothing_in_it_is_not_numeric(host: str) -> None:
    """No label at all is not a number either; an empty host is the caller's case."""
    assert is_private_or_link_local(host) is False
    refuse_private_ip_literal(host, label="x")


def test_leading_space_is_not_skipped() -> None:
    """The text before the first C whitespace is empty, so there is no number to read."""
    assert parse_ip_literal(" 4462739457") is None
    assert is_private_or_link_local(" 4462739457") is False


@pytest.mark.parametrize("host", LOOPBACK)
def test_loopback_stays_allowed(host: str) -> None:
    """Loopback names and every loopback spelling the parser reads still pass."""
    refuse_private_ip_literal(host, label="x")


@pytest.mark.parametrize("host", PUBLIC_NUMERIC)
def test_public_numeric_spelling_stays_public(host: str) -> None:
    """A numeric host that IS an address is classified by that address."""
    assert parse_ip_literal(host) is not None
    assert is_private_or_link_local(host) is False
    refuse_private_ip_literal(host, label="x")


@pytest.mark.parametrize("host", NON_PUBLIC_NUMERIC)
def test_non_public_numeric_spelling_keeps_its_own_message(host: str) -> None:
    """A non-public address is still reported as one, not as an invalid number."""
    assert parse_ip_literal(host) is not None
    assert is_private_or_link_local(host) is True
    message = _refusal(host)
    assert message.startswith(f"x: {_NON_PUBLIC}")
    assert "all-numeric" not in message


# --- the parser and the callers that only read it are unchanged --------------


@pytest.mark.parametrize("host", NUMERIC_NOT_AN_ADDRESS)
def test_the_parser_still_reads_no_address(host: str) -> None:
    """The rule lives in the refusing predicates; the grammar did not get wider."""
    assert inet_aton(host) is None
    assert parse_ip_literal(host) is None


@pytest.mark.parametrize("host", NUMERIC_NOT_AN_ADDRESS)
@pytest.mark.parametrize("scheme", ["https", "http"])
def test_numeric_host_is_not_local_or_loopback(scheme: str, host: str) -> None:
    """Callers that ask "is this address local?" keep saying no."""
    from soup_cli.utils.ingest_pull import _is_loopback
    from soup_cli.utils.loop_stages import _endpoint_is_local

    assert _endpoint_is_local(f"{scheme}://{host}") is False
    assert _endpoint_is_local(f"{scheme}://{host}:8000/v1") is False
    assert _is_loopback(host) is False


# --- through the callers ------------------------------------------------------

GATE_HOSTS = ["4462739457", "0x.1", "10.0x.0.1", "10.0.0.256"]


@pytest.mark.parametrize("host", GATE_HOSTS)
def test_vllm_url_refuses_a_numeric_host(host: str) -> None:
    from soup_cli.data.providers.vllm import validate_vllm_url

    with pytest.raises(ValueError) as excinfo:
        validate_vllm_url(f"https://{host}")
    assert str(excinfo.value) == f"vLLM URL: {_NUMERIC}"
    validate_vllm_url("https://api.example.com")


@pytest.mark.parametrize("host", GATE_HOSTS)
def test_judge_api_base_refuses_a_numeric_host(host: str) -> None:
    from soup_cli.eval.judge import validate_judge_api_base

    with pytest.raises(ValueError) as excinfo:
        validate_judge_api_base(f"https://{host}/v1")
    assert str(excinfo.value) == f"judge URL: {_NUMERIC}"
    validate_judge_api_base("https://api.example.com/v1")


@pytest.mark.parametrize("host", GATE_HOSTS)
def test_gate_suite_refuses_a_numeric_judge_host(host: str) -> None:
    from soup_cli.eval.gate import GateTask

    def task(judge_model: str) -> GateTask:
        return GateTask(
            type="judge", name="t", threshold=0.5, prompts="p.jsonl", judge_model=judge_model
        )

    with pytest.raises(ValidationError, match="judge_model URL: an all-numeric host"):
        task(f"https://{host}/m")
    assert task("https://api.example.com/m").judge_model == "https://api.example.com/m"


@pytest.mark.parametrize("host", GATE_HOSTS)
@pytest.mark.parametrize("scheme", ["https", "http"])
def test_online_dpo_judge_refuses_a_numeric_host(scheme: str, host: str) -> None:
    from soup_cli.config.schema import TrainingConfig

    with pytest.raises(ValidationError, match="online_dpo_judge: an all-numeric host"):
        TrainingConfig(online_dpo_judge=f"{scheme}://{host}/m")
    url = "https://api.example.com/m"
    assert TrainingConfig(online_dpo_judge=url).online_dpo_judge == url


@pytest.mark.parametrize("host", GATE_HOSTS)
def test_webhook_refuses_a_numeric_host_unless_private_hosts_are_allowed(host: str) -> None:
    from soup_cli.utils.webhooks import validate_webhook_url

    url = f"https://{host}/hook"
    with pytest.raises(ValueError, match="private/link-local/reserved hosts are not allowed"):
        validate_webhook_url(url)
    # The opt-in that admits a private address admits this host as well.
    assert validate_webhook_url(url, allow_private_hosts=True) == url
    assert (
        validate_webhook_url("https://hooks.example.com/hook") == "https://hooks.example.com/hook"
    )


@pytest.mark.parametrize("host", GATE_HOSTS)
def test_otlp_endpoint_refuses_a_numeric_host(host: str) -> None:
    from soup_cli.utils.tracing import validate_otlp_endpoint

    with pytest.raises(ValueError, match="is a private / link-local IP"):
        validate_otlp_endpoint(f"https://{host}:4317")
    assert (
        validate_otlp_endpoint("https://otel.example.com:4317") == "https://otel.example.com:4317"
    )


@pytest.mark.parametrize("host", GATE_HOSTS)
def test_langfuse_host_check_counts_a_numeric_host_as_private(host: str) -> None:
    from soup_cli.utils.ingest_pull import _is_private

    assert _is_private(f"https://{host}") is True
    assert _is_private("https://cloud.example.com") is False


@pytest.mark.parametrize("host", GATE_HOSTS)
def test_telemetry_refuses_a_numeric_host_without_a_lookup(
    monkeypatch: pytest.MonkeyPatch, host: str
) -> None:
    """The static tier decides: the host never reaches the resolver."""
    from soup_cli.utils import trackers

    looked_up: list[str] = []

    def resolver(name: str, **_kwargs: object) -> list[str]:
        looked_up.append(name)
        return ["93.184.216.34"]

    monkeypatch.setattr(trackers, "_resolve_host_ips", resolver)
    assert trackers._telemetry_endpoint_is_safe(f"https://{host}/i/v0/e/") is False
    assert looked_up == []
    # The same resolver is reached, and its public answer accepted, for a name.
    assert trackers._telemetry_endpoint_is_safe("https://telemetry.example.com/i/v0/e/") is True
    assert looked_up == ["telemetry.example.com"]


def test_embedded_and_plain_addresses_are_untouched() -> None:
    """Sanity: the rule does not reach hosts the parser already reads."""
    assert parse_ip_literal("0xa.1") == ipaddress.IPv4Address("10.0.0.1")
    assert parse_ip_literal("::ffff:10.0.0.1") is not None
    assert _refusal("::ffff:10.0.0.1").startswith(f"x: {_NON_PUBLIC}")


# --- a number longer than int() converts --------------------------------------

# int() refuses decimal text longer than sys.get_int_max_str_digits() with a
# ValueError. Such a number is far out of range: it is no address, and the
# parser says so with None, as it does for every other text it does not read.
LONG_DECIMAL = [
    pytest.param("9" * 4301, id="one-part-4301-digits"),
    pytest.param("9" * 20000, id="one-part-20000-digits"),
    pytest.param("10." + "9" * 5000, id="last-part-5000-digits"),
    pytest.param("9" * 5000 + ".0.0.1", id="first-part-5000-digits"),
]


@pytest.fixture
def default_int_limit():
    """Pin the interpreter's default limit, whatever the environment set."""
    import sys

    previous = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(4300)
    yield
    sys.set_int_max_str_digits(previous)


@pytest.mark.usefixtures("default_int_limit")
@pytest.mark.parametrize("host", LONG_DECIMAL)
def test_a_number_longer_than_int_converts_is_no_address_and_no_error(host: str) -> None:
    """The parser answers None; it does not let int()'s ValueError out."""
    assert inet_aton(host) is None
    assert parse_ip_literal(host) is None


@pytest.mark.usefixtures("default_int_limit")
@pytest.mark.parametrize("host", LONG_DECIMAL)
def test_a_very_long_numeric_host_is_refused_like_any_other(host: str) -> None:
    """Both predicates give their usual answer, with the usual message."""
    assert is_private_or_link_local(host) is True
    assert _refusal(host) == f"x: {_NUMERIC}"


@pytest.mark.usefixtures("default_int_limit")
@pytest.mark.parametrize("host", LONG_DECIMAL)
def test_a_very_long_numeric_host_does_not_raise_in_the_callers(
    monkeypatch: pytest.MonkeyPatch, host: str
) -> None:
    """Callers that return a bool keep returning one."""
    from soup_cli.utils import trackers
    from soup_cli.utils.ingest_pull import _is_loopback, _is_private
    from soup_cli.utils.loop_stages import _endpoint_is_local

    def resolver(name: str, **_kwargs: object) -> list[str]:
        raise AssertionError("the resolver must not be reached")

    monkeypatch.setattr(trackers, "_resolve_host_ips", resolver)
    assert _endpoint_is_local(f"https://{host}") is False
    assert _is_loopback(host) is False
    assert _is_private(f"https://{host}") is True
    assert trackers._telemetry_endpoint_is_safe(f"https://{host}/i/v0/e/") is False


@pytest.mark.usefixtures("default_int_limit")
def test_a_long_run_of_zeros_before_a_valid_number_still_reads() -> None:
    """Length alone refuses nothing: octal and hex text of any length converts."""
    assert inet_aton("0" * 5000 + "12") == 10
    assert inet_aton("0x" + "0" * 5000 + "a") == 10
    assert inet_aton("0x" + "f" * 5000) is None
    assert inet_aton("0" + "7" * 5000) is None


# --- a host that is one space ---------------------------------------------------

# Windows' C library reads a host of exactly one space as the unspecified
# address, so while the parser used that library the refusing predicates called
# such a host non-public there. The pure-Python parser says it is no address:
# nothing precedes the whitespace. The predicates keep the old answer, on every
# platform, and nothing else about whitespace changes.
_ONE_SPACE = (
    "a host that is a single space is not allowed (SSRF protection); "
    "use a valid IPv4 address or a hostname"
)

# One space, in the forms the predicates read a host in: without trailing dots,
# and in the ASCII form a client folds a non-ASCII host to.
ONE_SPACE = [
    pytest.param(" ", id="space"),
    pytest.param(" .", id="space-trailing-dot"),
    pytest.param(" ..", id="space-two-trailing-dots"),
    pytest.param("\u00a0", id="nbsp"),
    pytest.param("\u3000", id="ideographic-space"),
    pytest.param("\u00a0.", id="nbsp-trailing-dot"),
]

# Hosts next to it that the C library never read as an address either.
NOT_ONE_SPACE = [
    pytest.param("", id="empty"),
    pytest.param("\t", id="tab"),
    pytest.param("  ", id="two-spaces"),
    pytest.param(" \t", id="space-tab"),
    pytest.param("\x0b", id="vertical-tab"),
    pytest.param(" x", id="space-then-letter"),
    pytest.param(". ", id="dot-then-space"),
    pytest.param(".. ", id="two-dots-then-space"),
    pytest.param(" 1", id="space-then-number"),
    pytest.param(" 10.0.0.1", id="space-then-address"),
    pytest.param("\n10.0.0.1", id="newline-then-address"),
]


@pytest.mark.parametrize("host", ONE_SPACE)
def test_a_one_space_host_counts_as_non_public(host: str) -> None:
    """The predicate keeps the answer it gave on Windows; the parser reads nothing."""
    assert parse_ip_literal(host) is None
    assert is_private_or_link_local(host) is True


@pytest.mark.parametrize("host", ["\u00a0", "\u3000"])
def test_the_two_folded_spaces_are_one_space_to_a_client(host: str) -> None:
    """The premise of the non-ASCII rows: the stdlib codec folds each to one space."""
    assert host.encode("idna").decode("ascii") == " "


@pytest.mark.parametrize("host", NOT_ONE_SPACE)
def test_other_whitespace_hosts_are_not_refused_by_the_predicate(host: str) -> None:
    """Only the one-space host was ever read as an address; the rule adds nothing else."""
    assert is_private_or_link_local(host) is False


def test_trailing_space_after_an_address_still_reads_as_the_address() -> None:
    """That is the parser's whitespace rule, and it is unchanged."""
    assert is_private_or_link_local("10.0.0.1 ") is True
    assert is_private_or_link_local("8.8.8.8 ") is False


@pytest.mark.parametrize("host", [None, "", " ", "\t", " .", "\u00a0", "  "])
def test_refuse_still_passes_a_host_that_is_empty_once_stripped(host: str | None) -> None:
    """``refuse_private_ip_literal`` strips first; an empty host is the caller's case."""
    refuse_private_ip_literal(host, label="x")


def test_refuse_refuses_the_one_space_host_in_brackets() -> None:
    """Brackets keep the space from being stripped, so the host reaches the check."""
    message = _refusal("[ ]")
    assert message == f"x: {_ONE_SPACE}"
    assert "[" not in message


@pytest.mark.parametrize("host", [" 1", " 10.0.0.1", "\n10.0.0.1", "10.0.0.1 "])
def test_refuse_reads_the_address_inside_surrounding_whitespace(host: str) -> None:
    """Stripped, then read: refused as the non-public address it is, as before."""
    assert _refusal(host).startswith(f"x: {_NON_PUBLIC}")


@pytest.mark.parametrize("host", ONE_SPACE)
def test_the_parser_and_the_local_checks_do_not_read_a_one_space_host(host: str) -> None:
    """The rule is in the refusing predicates only."""
    from soup_cli.utils.ingest_pull import _is_loopback
    from soup_cli.utils.loop_stages import _endpoint_is_local

    assert inet_aton(host) is None
    assert parse_ip_literal(host) is None
    assert _is_loopback(host) is False
    assert _endpoint_is_local(f"https://{host}") is False


def test_webhook_refuses_a_one_space_host_unless_private_hosts_are_allowed() -> None:
    from soup_cli.utils.webhooks import validate_webhook_url

    url = "https:// /hook"
    with pytest.raises(ValueError, match="private/link-local/reserved hosts are not allowed"):
        validate_webhook_url(url)
    assert validate_webhook_url(url, allow_private_hosts=True) == url


def test_otlp_endpoint_refuses_a_one_space_host() -> None:
    from soup_cli.utils.tracing import validate_otlp_endpoint

    with pytest.raises(ValueError, match="is a private / link-local IP"):
        validate_otlp_endpoint("https:// :4317")


def test_langfuse_host_check_counts_a_one_space_host_as_private() -> None:
    from soup_cli.utils.ingest_pull import _is_private

    assert _is_private("https:// ") is True
    assert _is_private("https://  ") is False


def test_telemetry_refuses_a_one_space_host_without_a_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The static tier decides, as it did while the parser used the C library on Windows."""
    from soup_cli.utils import trackers

    looked_up: list[str] = []

    def resolver(name: str, **_kwargs: object) -> list[str]:
        looked_up.append(name)
        return ["93.184.216.34"]

    monkeypatch.setattr(trackers, "_resolve_host_ips", resolver)
    assert trackers._telemetry_endpoint_is_safe("https:// /i/v0/e/") is False
    assert looked_up == []
