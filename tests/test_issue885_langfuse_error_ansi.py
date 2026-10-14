"""#885 — Langfuse error bodies must not reach the terminal with ANSI escapes intact.

A 500 response body is untrusted input. ``_error_detail`` collapses whitespace
but leaves ESC/NUL/BEL bytes alone, so ``commands/ingest.py`` must route the
rendered ``PullError`` through the shared ``utils/terminal.py:for_terminal``
helper (C0 strip + markup escape) instead of bare ``rich.markup.escape``.
"""

from __future__ import annotations

from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils import ingest_pull
from soup_cli.utils.ingest_pull import LangfuseCredentials, PullError, _error_detail
from tests.conftest import strip_ansi

HOSTILE = b"err \x1b[2J\x1b[31mFAKE ERROR\x07\x00 tail"


def test_pull_error_body_renders_inert(monkeypatch, tmp_path) -> None:
    """The injected erase-display/colour sequences never reach stdout intact."""
    creds = LangfuseCredentials(
        public_key="pk", secret_key="sk", host="https://langfuse.example.com"
    )
    detail = _error_detail(HOSTILE, creds)
    message = f"Langfuse answered HTTP 500: {detail}"

    def _raise_pull_error(*args, **kwargs):
        raise PullError(message)

    monkeypatch.setattr(
        ingest_pull,
        "load_langfuse_credentials",
        lambda environ, allow_private_host=False: creds,
    )
    monkeypatch.setattr(ingest_pull, "pull_langfuse_generations", _raise_pull_error)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner(env={"FORCE_COLOR": "1", "TERM": "xterm-256color", "COLUMNS": "300"})
    result = runner.invoke(app, ["ingest", "--source", "langfuse", "--pull", "-o", "out.jsonl"])

    assert result.exit_code != 0
    assert "\x1b[2J" not in result.output
    assert "\x00" not in result.output
    assert "\x07" not in result.output
    assert "HTTP 500" in strip_ansi(result.output)
    assert "FAKE ERROR" in result.output


def test_stripped_body_renders_on_a_colour_console(monkeypatch) -> None:
    """Even with colour genuinely on, no payload sequence survives the strip.

    The CLI-level test above may render without colour. This one renders the
    same production expression with explicit colour settings, regardless of
    the runner's NO_COLOR/TERM, and rejects the payload's control bytes.
    """
    import re
    from io import StringIO

    from rich.console import Console

    from soup_cli.utils.terminal import for_terminal

    creds = LangfuseCredentials(
        public_key="pk", secret_key="sk", host="https://langfuse.example.com"
    )
    message = f"Langfuse answered HTTP 500: {_error_detail(HOSTILE, creds)}"

    # TERM=dumb also overrides Rich's explicit width. Keep the override local
    # to this test, and opt this console into colour even under NO_COLOR.
    monkeypatch.setenv("TERM", "xterm-256color")
    out = StringIO()
    console = Console(
        file=out, force_terminal=True, color_system="standard", no_color=False, width=300
    )
    # This is the exact production expression from commands/ingest.py.
    console.print(f"[red]{for_terminal(message)}[/]")
    rendered = out.getvalue()

    # Control: the console really did colour, so the assertions below are not
    # trivially true because nothing was styled.
    assert "\x1b[" in rendered
    # Erase-display is payload-only; Rich has no reason to emit it.
    assert "\x1b[2J" not in rendered
    # C0 bytes stripped by for_terminal().
    assert "\x00" not in rendered
    assert "\x07" not in rendered
    # The server's message itself is preserved, not dropped. Rich's own
    # ReprHighlighter restyles numbers (``500`` -> ``\x1b[1;31m500\x1b[0m``),
    # splitting "HTTP 500" with SGR codes that are Rich's own, not payload --
    # so compare against the SGR-stripped text here. The absence assertions
    # above already ran against the raw bytes.
    plain = re.sub(r"\x1b\[[0-9;]*m", "", rendered)
    assert "HTTP 500" in plain
    assert "FAKE ERROR" in plain


def test_credentials_pull_error_body_renders_inert(monkeypatch, tmp_path) -> None:
    """The credentials PullError site (:246) strips control bytes too."""
    creds = LangfuseCredentials(
        public_key="pk", secret_key="sk", host="https://langfuse.example.com"
    )
    message = f"Langfuse credentials: {_error_detail(HOSTILE, creds)}"

    def _raise_pull_error(*args, **kwargs):
        raise PullError(message)

    monkeypatch.setattr(ingest_pull, "load_langfuse_credentials", _raise_pull_error)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner(env={"FORCE_COLOR": "1", "TERM": "xterm-256color", "COLUMNS": "300"})
    result = runner.invoke(app, ["ingest", "--source", "langfuse", "--pull", "-o", "out.jsonl"])

    assert result.exit_code == 1
    assert "\x1b[2J" not in result.output
    assert "\x00" not in result.output
    assert "\x07" not in result.output
    assert "Langfuse credentials" in result.output
    assert "FAKE ERROR" in result.output
