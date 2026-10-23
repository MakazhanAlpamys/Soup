"""Tests for Issue #1194: soup ui must hand the auth token to the browser.

The tab ``soup ui`` opens used to be ``http://host:port`` with no ``?token=``, so
the first paint was Unauthorized. The launch URL and the printed URL now
carry the token. ``app.js`` already reads ``?token=`` and strips it.
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
_LOGIN = f"http://127.0.0.1:7860/?token={_TOKEN}"


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", strip_ansi(text))


def _immediate_thread(target=None, daemon=None, **kwargs):
    thread = MagicMock()
    thread.start.side_effect = lambda: target() if target else None
    return thread


def _run_ui(*extra: str) -> tuple[object, list[str]]:
    """Invoke ``soup ui`` with the browser thread run inline, so any open() is recorded."""
    opened: list[str] = []

    def _open(url, *args, **kwargs):
        opened.append(url)
        return True

    with (
        patch("uvicorn.run"),
        patch("webbrowser.open", side_effect=_open),
        patch("threading.Thread", side_effect=_immediate_thread),
        patch("time.sleep"),
    ):
        result = CliRunner().invoke(
            app,
            ["ui", "--host", "127.0.0.1", "--port", "7860", "--auth-token", _TOKEN, *extra],
        )
    return result, opened


class TestUiLoginUrl:
    def test_ui_opens_login_url_with_token(self):
        result, opened = _run_ui()

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert _LOGIN in _collapse(result.output)
        assert opened == [_LOGIN]

    def test_no_browser_prints_login_url_and_does_not_open(self):
        result, opened = _run_ui("--no-browser")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert _LOGIN in _collapse(result.output)
        assert opened == []
