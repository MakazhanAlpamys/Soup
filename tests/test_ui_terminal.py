"""Web UI startup receipts show the host literally and pass the original bind onward."""

import sys
from types import SimpleNamespace

import pytest
from rich.console import Console

from soup_cli.cli import app
from tests.conftest import make_cli_runner, strip_ansi

pytest.importorskip("fastapi")

runner = make_cli_runner()
TOKEN = "test_token_0123456789"
HOSTS = [
    "localhost",
    "node[red].test",
    "node[/].test",
    "node:warning:.test",
    "node\\[.test",
    "node\\\\.test",
    "node\x9b\u202e.test",
]


@pytest.fixture
def server(monkeypatch):
    calls = []
    application = object()
    monkeypatch.setattr("soup_cli.ui.app.get_auth_token", lambda: TOKEN)
    monkeypatch.setattr("soup_cli.ui.app.create_app", lambda **kwargs: application)
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(run=lambda app, **kwargs: calls.append((app, kwargs))),
    )
    monkeypatch.setattr("soup_cli.commands.ui.console", Console(width=500, height=100))
    return calls, application


@pytest.mark.parametrize("host", HOSTS)
def test_startup_panel_and_original_bind(server, host):
    calls, application = server

    result = runner.invoke(app, ["ui", "--host", host, "--no-browser"])

    assert result.exit_code == 0, result.output
    assert calls == [(application, {"host": host, "port": 7860, "log_level": "warning"})]
    body = "\n".join(
        line.strip(" │") for line in strip_ansi(result.output).splitlines() if line.startswith("│")
    )
    shown = host.replace("\x9b\u202e", "")
    assert body == (
        f"URL:    http://{shown}:7860/?token={TOKEN}\n"
        f"Token:  {TOKEN}\n\nAPI endpoints require:\n"
        f"Authorization: Bearer {TOKEN}\n\nPages:\n"
        "Dashboard      - View experiments, loss charts, system info\n"
        "New Training   - Create config from templates, start training\n"
        "Data Explorer  - Browse and inspect datasets\n"
        "Model Chat     - Chat with a running inference server\n\nPress Ctrl+C to stop."
    )


@pytest.mark.parametrize("host", HOSTS[1:])
def test_refused_public_host_is_literal_and_never_started(monkeypatch, server, host):
    calls, _ = server
    monkeypatch.setattr("soup_cli.ui.app.get_auth_token", lambda: None)

    result = runner.invoke(app, ["ui", "--host", host, "--no-browser"])

    assert result.exit_code == 2
    assert not calls
    assert strip_ansi(result.output).strip() == (
        f"Error: binding non-loopback host '{host.replace(chr(155) + chr(8238), '')}' "
        "requires a valid authentication token."
    )


def _coloured(monkeypatch):
    monkeypatch.setattr(
        "soup_cli.commands.ui.console",
        Console(
            width=500, height=100, force_terminal=True, no_color=False, color_system="standard"
        ),
    )


def test_url_token_pages_and_shortcut_keep_their_styles(monkeypatch, server):
    _coloured(monkeypatch)

    result = runner.invoke(app, ["ui", "--no-browser"])

    assert result.exit_code == 0
    for value in (
        f"http://127.0.0.1:7860/?token={TOKEN}",
        TOKEN,
        "Dashboard",
        "New Training",
        "Data Explorer",
        "Model Chat",
        "Ctrl+C",
    ):
        assert f"\x1b[1m{value}\x1b[0m" in result.output
    assert f"\x1b[2mAuthorization: Bearer {TOKEN}\x1b[0m" in result.output


def test_the_authentication_refusal_keeps_its_red_label(monkeypatch, server):
    _coloured(monkeypatch)
    calls, _ = server
    monkeypatch.setattr("soup_cli.ui.app.get_auth_token", lambda: None)

    result = runner.invoke(app, ["ui", "--host", "node:warning:.test", "--no-browser"])

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert not calls
    assert "\x1b[31mError:\x1b[0m binding non-loopback host 'node:warning:.test'" in result.output
