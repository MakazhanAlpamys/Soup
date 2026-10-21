"""Tests for Issue #2054: soup ui must not send the browser to a bind-any address.

``soup ui --public`` binds to ``0.0.0.0`` and used to print and open
``http://0.0.0.0:<port>/?token=...``. Windows cannot connect to ``0.0.0.0``, so
the tab never loaded; ``--host ::`` gave the malformed ``http://:::<port>``. The
printed and opened URL now use loopback for an unspecified bind, while the
server keeps the bind host.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from tests.conftest import strip_ansi

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from soup_cli.cli import app  # noqa: E402

_TOKEN = "abcdefghijklmnopqrstuvwx"
_LAN_IP = "192.168.1.23"


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", strip_ansi(text))


def _immediate_thread(target=None, daemon=None, **kwargs):
    thread = MagicMock()
    thread.start.side_effect = lambda: target() if target else None
    return thread


def _lan_socket(*args, **kwargs):
    sock = MagicMock()
    sock.__enter__.return_value.getsockname.return_value = (_LAN_IP, 0)
    return sock


def _run_ui(*args: str) -> tuple[object, list[str], str]:
    """Invoke ``soup ui`` with the browser thread inline; return (result, opened, bind host)."""
    opened: list[str] = []

    def _open(url, *a, **kw):
        opened.append(url)
        return True

    with (
        patch("uvicorn.run") as run,
        patch("webbrowser.open", side_effect=_open),
        patch("threading.Thread", side_effect=_immediate_thread),
        patch("time.sleep"),
        patch("socket.socket", side_effect=_lan_socket),
    ):
        result = CliRunner().invoke(app, ["ui", "--port", "7861", "--auth-token", _TOKEN, *args])
    bind_host = run.call_args.kwargs["host"] if run.called else ""
    return result, opened, bind_host


class TestUnspecifiedBindOpensLoopback:
    @pytest.mark.parametrize(
        ("args", "bind_host", "login_url"),
        [
            (("--public",), "0.0.0.0", f"http://127.0.0.1:7861/?token={_TOKEN}"),
            (("--host", "0.0.0.0"), "0.0.0.0", f"http://127.0.0.1:7861/?token={_TOKEN}"),
            (("--host", "::"), "::", f"http://[::1]:7861/?token={_TOKEN}"),
        ],
    )
    def test_printed_and_opened_url_use_loopback(self, args, bind_host, login_url):
        result, opened, bound = _run_ui(*args)

        assert result.exit_code == 0, (result.output, repr(result.exception))
        output = _collapse(result.output)
        assert f"URL: {login_url}" in output
        assert opened == [login_url]
        assert "0.0.0.0:7861" not in output
        assert bound == bind_host

    def test_public_phone_url_keeps_the_lan_address(self):
        result, _, _ = _run_ui("--public")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert f"http://{_LAN_IP}:7861/?token={_TOKEN}" in _collapse(result.output)


class TestOtherHostsUnchanged:
    @pytest.mark.parametrize("host", ["127.0.0.1", "192.168.1.5", "localhost"])
    def test_named_host_is_used_as_given(self, host):
        login_url = f"http://{host}:7861/?token={_TOKEN}"
        result, opened, bound = _run_ui("--host", host)

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert f"URL: {login_url}" in _collapse(result.output)
        assert opened == [login_url]
        assert bound == host

    def test_ipv6_loopback_is_bracketed(self):
        login_url = f"http://[::1]:7861/?token={_TOKEN}"
        result, opened, bound = _run_ui("--host", "::1")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert f"URL: {login_url}" in _collapse(result.output)
        assert opened == [login_url]
        assert bound == "::1"
