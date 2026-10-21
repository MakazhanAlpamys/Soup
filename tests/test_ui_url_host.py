"""The URL ``soup ui`` prints and opens names a host the browser can reach.

The server binds to the host it is given, but a bind-any address is not a
destination: ``0.0.0.0`` (``--public`` or ``--host 0.0.0.0``) and ``::`` are
printed and opened as ``127.0.0.1`` and ``[::1]``, since Windows cannot connect
to ``0.0.0.0``. IPv6 literals are bracketed, and the printed line is escaped so
that Rich does not read a host such as ``[fd00::1]`` as a style tag. The app,
``uvicorn.run`` and the ``--public`` phone URL keep the bind host or the LAN
address. See #2054.
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
from soup_cli.ui.app import create_app  # noqa: E402

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


def _run_ui(*args: str) -> tuple[object, list[str], str, str]:
    """Invoke ``soup ui`` with the browser thread inline.

    Returns the result, the opened URLs, the host given to ``uvicorn.run`` and the
    host given to ``create_app``.
    """
    opened: list[str] = []

    def _open(url, *a, **kw):
        opened.append(url)
        return True

    with (
        patch("uvicorn.run") as run,
        patch("soup_cli.ui.app.create_app", wraps=create_app) as made,
        patch("webbrowser.open", side_effect=_open),
        patch("threading.Thread", side_effect=_immediate_thread),
        patch("time.sleep"),
        patch("socket.socket", side_effect=_lan_socket),
    ):
        result = CliRunner().invoke(app, ["ui", "--port", "7861", "--auth-token", _TOKEN, *args])
    bind_host = run.call_args.kwargs["host"] if run.called else ""
    app_host = made.call_args.kwargs["host"] if made.called else ""
    return result, opened, bind_host, app_host


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
        result, opened, bound, app_host = _run_ui(*args)

        assert result.exit_code == 0, (result.output, repr(result.exception))
        output = _collapse(result.output)
        assert f"URL: {login_url}" in output
        assert opened == [login_url]
        assert "0.0.0.0:7861" not in output
        assert bound == bind_host
        assert app_host == bind_host

    def test_public_phone_url_keeps_the_lan_address(self):
        result, _, _, _ = _run_ui("--public")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert f"http://{_LAN_IP}:7861/?token={_TOKEN}" in _collapse(result.output)


class TestOtherHostsUnchanged:
    @pytest.mark.parametrize("host", ["127.0.0.1", "192.168.1.5", "localhost"])
    def test_named_host_is_used_as_given(self, host):
        login_url = f"http://{host}:7861/?token={_TOKEN}"
        result, opened, bound, app_host = _run_ui("--host", host)

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert f"URL: {login_url}" in _collapse(result.output)
        assert opened == [login_url]
        assert bound == app_host == host

    @pytest.mark.parametrize("host", ["::1", "fd00::1", "abcd::1"])
    def test_ipv6_literal_is_bracketed_in_the_printed_url(self, host):
        # fd00::1 and abcd::1 start with a letter, so unescaped "[fd00::1]" is a Rich tag
        login_url = f"http://[{host}]:7861/?token={_TOKEN}"
        result, opened, bound, app_host = _run_ui("--host", host)

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert f"URL: {login_url}" in _collapse(result.output)
        assert opened == [login_url]
        assert bound == app_host == host
