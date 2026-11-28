"""soup mcp serve and soup mcp runs reconcile show typed hosts, transports and run ids as typed."""

import re
import sys
import types

import pytest
import rich.console
from typer.testing import CliRunner

import soup_cli.experiment.tracker as tracker_module
from soup_cli.cli import app
from tests.conftest import strip_ansi

runner = CliRunner()

_CONTROLS = "\x9b\u202e"

# One spelling per way the old code went wrong: a tag, a closing tag with
# nothing to close (a MarkupError), an emoji code, and a tag with control
# characters, so the strip and the escape are pinned separately. A backslash before
# an unmatched bracket and a trailing pair of backslashes are what markup rewrites.
_VALUES = [
    "x[red]n",
    "[/]n",
    pytest.param("n:warning:", id="emoji-code"),
    pytest.param("x\\[n", id="backslash-bracket"),
    pytest.param("n\\\\", id="trailing-backslashes"),
    f"x[bold]n{_CONTROLS}",
]


@pytest.fixture
def server_calls(monkeypatch):
    """Stand in for the MCP server module, so no SDK is needed and nothing listens."""
    calls = []
    fake = types.ModuleType("soup_cli.mcp_server.server")
    fake.run_stdio_server = lambda **kwargs: calls.append(("stdio", kwargs))
    fake.run_network_server = lambda **kwargs: calls.append(("network", kwargs))
    fake.DEFAULT_NETWORK_HOST = "127.0.0.1"
    fake.DEFAULT_NETWORK_PORT = 8765
    fake.HTTP_PATH = "/mcp"
    fake.SSE_PATH = "/sse"
    fake.is_wildcard_host = lambda host: host in ("0.0.0.0", "::")
    monkeypatch.setitem(sys.modules, "soup_cli.mcp_server.server", fake)
    return calls


def _console_factory(monkeypatch, **extra):
    """The commands build their own ``Console`` when they run; give it a fixed width."""
    real = rich.console.Console

    def make(*args, **kwargs):
        kwargs.setdefault("width", 500)
        kwargs.setdefault("height", 100)
        kwargs.update(extra)
        return real(*args, **kwargs)

    monkeypatch.setattr(rich.console, "Console", make)


@pytest.fixture(autouse=True)
def _plain_console(monkeypatch):
    monkeypatch.setenv("COLUMNS", "500")
    _console_factory(monkeypatch)


def _coloured(monkeypatch):
    _console_factory(
        monkeypatch, force_terminal=True, no_color=False, color_system="standard"
    )


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _shown(value: str) -> str:
    return "".join(ch for ch in value if ch not in _CONTROLS)


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


def _serve(*args: str):
    return runner.invoke(app, ["mcp", "serve", *args])


# --- serve -----------------------------------------------------------------


@pytest.mark.parametrize("value", _VALUES)
def test_a_refused_transport_is_shown_as_typed(server_calls, value):
    result = _serve("--transport", value)

    assert result.exit_code == 2
    text = _plain(result)
    assert f"--transport must be one of: stdio, sse, http (got '{_shown(value)}')" in text
    assert _not_shown(text)
    assert server_calls == []


@pytest.mark.parametrize("value", _VALUES)
def test_the_url_line_shows_the_host_as_typed(server_calls, value):
    result = _serve("--transport", "sse", "--host", value, "--port", "9000")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"URL: http://{_shown(value)}:9000/sse" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_the_loopback_warning_shows_the_host_as_typed(server_calls, value):
    result = _serve("--transport", "http", "--host", value)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"WARNING: bound to {_shown(value)}, not loopback - this server is reachable" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_the_host_the_server_binds_is_exactly_what_was_typed(server_calls, value):
    result = _serve("--transport", "sse", "--host", value)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    [(kind, kwargs)] = server_calls
    assert kind == "network"
    assert kwargs["host"] == value


def test_a_loopback_host_has_no_warning_and_keeps_the_banner(server_calls):
    result = _serve("--transport", "sse")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert "WARNING" not in text
    lines = strip_ansi(result.output).splitlines()
    assert lines[0] == "soup mcp serve - sse transport - ungated tools (some can write state)"
    assert lines[1] == "URL:   http://127.0.0.1:8765/sse"
    assert re.fullmatch(r"Token: [A-Za-z0-9_-]{43}", lines[2])
    assert lines[3].startswith("Every request needs: Authorization: Bearer ")


# --- reconcile -------------------------------------------------------------


class _Tracker:
    removed: list = []
    error: Exception = None

    def expunge_stale_launching_runs(self, *, older_than_seconds):
        if self.error is not None:
            raise self.error
        return list(self.removed)


@pytest.fixture
def tracker(monkeypatch):
    class Tracker(_Tracker):
        removed = []
        error = None

    monkeypatch.setattr(tracker_module, "ExperimentTracker", Tracker)
    return Tracker


def _reconcile():
    return runner.invoke(app, ["mcp", "runs", "reconcile", "--expunge-launching"])


@pytest.mark.parametrize("value", _VALUES)
def test_a_live_run_is_named_as_stored(tracker, value):
    tracker.error = tracker_module.ActiveLaunchingRunError(f"run-{value}", 4242)

    result = _reconcile()

    assert result.exit_code == 1
    text = _plain(result)
    expected = f"Refusing to expunge launching run run-{_shown(value)}: PID 4242 is still alive."
    assert expected in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_the_expunged_runs_are_named_as_stored(tracker, value):
    tracker.removed = [f"first-{value}", f"second-{value}"]

    result = _reconcile()

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    shown = _shown(value)
    assert f"Expunged launching run: first-{shown} Expunged launching run: second-{shown}" in text
    assert _not_shown(text)


# --- styles ----------------------------------------------------------------


def test_the_banner_url_and_token_keep_their_dim_labels(monkeypatch, server_calls):
    _coloured(monkeypatch)

    result = _serve("--transport", "sse", "--host", "127.0.0.1", "--port", "9001")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "\x1b[2msoup mcp serve - sse transport - ungated tools" in result.output
    # The URL keeps the underlined blue that Rich's highlighter gave it before.
    assert "\x1b[2mURL:\x1b[0m   \x1b[4;94mhttp://127.0.0.1:9001/sse\x1b[0m\n" in result.output
    assert re.search(r"\x1b\[2mToken:\x1b\[0m [A-Za-z0-9_-]{43}\n", result.output)
    assert re.search(
        r"\x1b\[2mEvery request needs: Authorization: Bearer \S{43}\x1b\[0m", result.output
    )


def test_the_warning_label_is_yellow_and_the_transport_refusal_red(monkeypatch, server_calls):
    _coloured(monkeypatch)

    warned = _serve("--transport", "sse", "--host", "example.test")
    refused = _serve("--transport", "nope")

    assert warned.exit_code == 0, (warned.output, repr(warned.exception))
    assert refused.exit_code == 2
    assert "\x1b[33mWARNING:\x1b[0m bound to example.test, not loopback" in warned.output
    expected = "\x1b[31m--transport must be one of: stdio, sse, http\x1b[0m (got 'nope')"
    assert expected in refused.output


def test_the_reconcile_lines_keep_their_colours(monkeypatch, tracker):
    _coloured(monkeypatch)
    tracker.removed = ["run-1"]
    removed = _reconcile()
    tracker.error = tracker_module.ActiveLaunchingRunError("run-2", 77)
    refused = _reconcile()

    assert "\x1b[32mExpunged launching run:\x1b[0m run-1" in removed.output
    assert (
        "\x1b[31mRefusing to expunge launching run\x1b[0m \x1b[1mrun-2\x1b[0m: PID 77 is still"
    ) in refused.output
