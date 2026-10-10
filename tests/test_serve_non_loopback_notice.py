"""``soup serve`` says what ``--tool-auth-token`` covers when it binds beyond loopback.

A non-loopback ``--host`` is refused without ``--tool-auth-token``, which
reads as if the token put the whole server behind a credential. It does not:

* on the transformers backend the token is checked on the tool, thumbs and
  adapter routes only; the generation routes take none;
* the vLLM, SGLang and MII apps have no such routes and are never handed the
  token, so no route checks it there;
* on a wildcard bind there is no single name to compare ``Host`` against, so
  the routes that do compare it check only ``Origin`` against ``Host``.

None of that changes here. The command now prints it at startup, and the
``--host`` help says that the token is required and what it is checked on.
"""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import serve as serve_cmd
from soup_cli.utils.local_request_guard import WILDCARD_HOSTS, is_wildcard_bind
from tests.conftest import strip_ansi

runner = CliRunner()

LOOPBACK = ["127.0.0.1", "localhost", "::1"]
# One named interface: a LAN address, a hostname, an IPv6 literal.
NAMED = ["192.168.1.20", "10.0.0.5", "gpu-box.internal", "fe80::1", "[fe80::1]"]
# Every spelling the request guard treats as "all interfaces".
WILDCARD = ["0.0.0.0", "::", "[::]", "*", ""]
OTHER_BACKENDS = ["vllm", "sglang", "mii"]

TOKEN_SCOPE = "--tool-auth-token is checked on the tool, thumbs and adapter routes only"
OPEN_ROUTES = "answer every client that can reach this port without a token"
NO_HOST_CHECK = "the Host header is not compared"


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output, safe to substring-match."""
    return " ".join(strip_ansi(text).split())


def _rendered(notices: list[str]) -> str:
    """The notices as the command prints them, markup applied, on a wide console."""
    buffer = StringIO()
    console = Console(file=buffer, width=400, color_system=None)
    for notice in notices:
        console.print(f"[yellow]Note:[/] {notice}")
    return " ".join(buffer.getvalue().split())


class TestWildcardSpellings:
    def test_the_test_list_is_the_guards_own_set(self):
        # If the guard learns a new spelling, the cases below must cover it.
        assert set(WILDCARD) == set(WILDCARD_HOSTS)

    @pytest.mark.parametrize("host", WILDCARD + ["  0.0.0.0 ", "[::]"])
    def test_wildcard_binds_are_recognised(self, host):
        assert is_wildcard_bind(host) is True

    @pytest.mark.parametrize("host", LOOPBACK + NAMED)
    def test_a_single_address_is_not_a_wildcard(self, host):
        assert is_wildcard_bind(host) is False


class TestNotices:
    @pytest.mark.parametrize("backend", ["transformers", *OTHER_BACKENDS])
    @pytest.mark.parametrize("host", LOOPBACK)
    def test_a_loopback_bind_gets_no_notice(self, host, backend):
        assert serve_cmd._non_loopback_notices(host, backend) == []

    @pytest.mark.parametrize("host", NAMED)
    def test_a_named_interface_gets_the_token_scope_only(self, host):
        notices = serve_cmd._non_loopback_notices(host, "transformers")

        assert len(notices) == 1
        out = _rendered(notices)
        assert f"bound to '{host}', not loopback" in out
        assert TOKEN_SCOPE in out
        assert "/v1/chat/completions, /v1/messages" in out
        assert OPEN_ROUTES in out
        assert NO_HOST_CHECK not in out

    @pytest.mark.parametrize("host", WILDCARD)
    def test_a_wildcard_bind_also_says_host_is_not_compared(self, host):
        notices = serve_cmd._non_loopback_notices(host, "transformers")

        assert len(notices) == 2
        out = _rendered(notices)
        assert TOKEN_SCOPE in out
        assert OPEN_ROUTES in out
        assert NO_HOST_CHECK in out
        assert "check only that an Origin header, when sent, names the same host as Host" in out

    @pytest.mark.parametrize("backend", OTHER_BACKENDS + ["VLLM", "SGLang"])
    @pytest.mark.parametrize("host", ["192.168.1.20", "0.0.0.0"])
    def test_other_backends_are_told_that_no_route_checks_the_token(self, host, backend):
        notices = serve_cmd._non_loopback_notices(host, backend)

        # One notice even on a wildcard bind: these apps compare Host nowhere,
        # so there is no Host rule to describe.
        assert len(notices) == 1
        out = _rendered(notices)
        assert f"the {backend.lower()} backend has no tool or adapter routes" in out
        assert "no route checks --tool-auth-token" in out
        assert OPEN_ROUTES in out
        assert TOKEN_SCOPE not in out

    @pytest.mark.parametrize(
        "host",
        ["[bold]lan[/bold]", "[/]", "[red]x", "lan\x1b[31mred", "lan\x9b31m"],
        ids=["tag-pair", "bare-close", "unclosed", "esc-csi", "c1-csi"],
    )
    def test_the_host_is_shown_literally(self, host):
        out = _rendered(serve_cmd._non_loopback_notices(host, "transformers"))

        assert "\x1b" not in out
        assert "\x9b" not in out
        visible = host.replace("\x1b", "").replace("\x9b", "")
        assert f"bound to '{visible}', not loopback" in out


class TestTheCommandPrintsThem:
    def _serve(self, *args: str):
        # The model path does not exist, so the command stops long before a
        # server starts; the notices are printed before that.
        return runner.invoke(app, ["serve", "--model", "no-such-model-dir", *args])

    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.20"])
    def test_a_non_loopback_bind_with_a_token_prints_the_scope(self, host, monkeypatch):
        monkeypatch.setenv("COLUMNS", "400")

        result = self._serve("--host", host, "--tool-auth-token", "s3cret-token-value")

        out = _plain(result.output)
        assert f"Note: bound to '{host}', not loopback" in out
        assert TOKEN_SCOPE in out
        assert (NO_HOST_CHECK in out) is (host == "0.0.0.0")
        assert "s3cret-token-value" not in out

    def test_the_backend_option_reaches_the_notice(self, monkeypatch):
        monkeypatch.setenv("COLUMNS", "400")

        result = self._serve(
            "--host", "0.0.0.0", "--tool-auth-token", "s3cret-token-value", "--backend", "sglang"
        )

        out = _plain(result.output)
        assert "the sglang backend has no tool or adapter routes" in out
        assert TOKEN_SCOPE not in out

    def test_a_loopback_bind_prints_no_notice(self, monkeypatch):
        monkeypatch.setenv("COLUMNS", "400")

        result = self._serve("--tool-auth-token", "s3cret-token-value")

        out = _plain(result.output)
        assert "not loopback" not in out
        assert TOKEN_SCOPE not in out

    def test_without_a_token_the_bind_is_still_refused_first(self, monkeypatch):
        # Unchanged behaviour: exit 2 and the existing message, no notice.
        monkeypatch.setenv("COLUMNS", "400")

        result = self._serve("--host", "0.0.0.0")

        assert result.exit_code == 2, (result.output, repr(result.exception))
        out = _plain(result.output)
        assert "binding non-loopback host '0.0.0.0' requires --tool-auth-token <secret>" in out
        assert "Note: bound to" not in out


class TestHostHelp:
    def test_help_says_the_token_is_required_and_what_it_is_checked_on(self, monkeypatch):
        monkeypatch.setenv("COLUMNS", "200")

        result = runner.invoke(app, ["serve", "--help"])

        assert result.exit_code == 0, (result.output, repr(result.exception))
        # The help table wraps a long description; drop its borders before joining.
        out = _plain(result.output.replace("│", " "))
        assert "Any other host needs --tool-auth-token" in out
        assert "checked on the tool, thumbs and adapter routes only" in out
        assert "should be paired with" not in out
