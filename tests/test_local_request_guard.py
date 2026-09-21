"""Host / Origin checks for local server routes."""

from __future__ import annotations

import pytest

from soup_cli.utils.local_request_guard import (
    check_local_request,
    hostname_from_host_header,
    hostname_from_origin,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("127.0.0.1:8000", "127.0.0.1"),
        ("LOCALHOST", "localhost"),
        ("localhost:8000", "localhost"),
        ("[::1]:8000", "::1"),
        ("[::1]", "::1"),
        ("evil.example:8000", "evil.example"),
        ("", None),
        (None, None),
        ("[::1", None),
        ("host:abc", None),
        ("a b:80", None),
    ],
)
def test_hostname_from_host_header(value, expected):
    assert hostname_from_host_header(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("http://localhost:3000", "localhost"),
        ("https://127.0.0.1", "127.0.0.1"),
        ("http://[::1]:9", "::1"),
        ("http://evil.example", "evil.example"),
        ("null", None),
        ("file://x", None),
        ("", None),
        (None, None),
    ],
)
def test_hostname_from_origin(value, expected):
    assert hostname_from_origin(value) == expected


class TestLoopbackBind:
    @pytest.mark.parametrize("bind", ["127.0.0.1", "localhost", "::1", "[::1]"])
    @pytest.mark.parametrize("host", ["127.0.0.1:8000", "localhost:8000", "[::1]:8000"])
    def test_loopback_host_no_origin_ok(self, bind, host):
        assert check_local_request(bind, host, None) is None

    def test_loopback_origin_ok(self):
        assert (
            check_local_request("127.0.0.1", "127.0.0.1:8000", "http://localhost:5173") is None
        )

    @pytest.mark.parametrize(
        "host", ["evil.example:8000", "evil.example", "127.0.0.1.evil.example", "", None]
    )
    def test_foreign_or_missing_host_421(self, host):
        assert check_local_request("127.0.0.1", host, None) == (421, "Host not allowed")

    @pytest.mark.parametrize(
        "origin",
        ["http://evil.example", "null", "file://x", "http://127.0.0.1.evil.example"],
    )
    def test_foreign_origin_403(self, origin):
        assert check_local_request("127.0.0.1", "127.0.0.1:8000", origin) == (
            403,
            "Origin not allowed",
        )


class TestSpecificBind:
    def test_bound_address_ok(self):
        assert check_local_request("192.168.1.5", "192.168.1.5:8000", None) is None

    def test_loopback_name_not_allowed_for_lan_bind(self):
        assert check_local_request("192.168.1.5", "localhost:8000", None) == (
            421,
            "Host not allowed",
        )


class TestWildcardBind:
    @pytest.mark.parametrize("bind", ["0.0.0.0", "::", "[::]"])
    def test_any_host_without_origin_ok(self, bind):
        assert check_local_request(bind, "anything.example:8000", None) is None

    def test_same_origin_ok(self):
        assert check_local_request("0.0.0.0", "10.0.0.2:8000", "http://10.0.0.2:8000") is None

    def test_cross_origin_403(self):
        assert check_local_request("0.0.0.0", "10.0.0.2:8000", "http://evil.example") == (
            403,
            "Origin not allowed",
        )
