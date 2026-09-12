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
    assert "\x1b[31m" not in result.output
    assert "\x00" not in result.output
    assert "\x07" not in result.output
    assert "HTTP 500" in result.output
    assert "FAKE ERROR" in result.output
