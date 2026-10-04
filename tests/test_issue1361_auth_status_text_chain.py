"""Tests for #1361: the auth phrasing check must read the whole exception chain.

`_auth_status` (#1268) reads a 401/403 from the exception object by walking
`__cause__`/`__context__`, but its text fallback searches only `str(exc)`, the
outermost message. A wrapper that says nothing about status, such as
`RuntimeError("Failed to load adapter")` around a text-only
`401 Client Error: Unauthorized`, therefore printed the wrapper's message:

    RuntimeError("Failed to load adapter")  <-  __cause__
        Exception("401 Client Error: Unauthorized for url: ...")

The fallback now walks the chain too, exception-major, so the outermost
exception that carries a status phrase decides, and a 403 wrapper of a text-only
401 still reads as the 403 it says.
"""

import urllib.error
from io import StringIO
from unittest.mock import patch

from rich.console import Console

from soup_cli.utils.errors import format_friendly_error

_401_TEXT = "401 Client Error: Unauthorized for url: https://huggingface.co/org/model"
_403_TEXT = "Client error '403 Forbidden' for url 'https://huggingface.co/org/model'"


def _render(exc: Exception) -> str:
    buf = StringIO()
    console = Console(file=buf, stderr=False)
    with patch("soup_cli.utils.errors.console", console):
        format_friendly_error(exc, verbose=False)
    return buf.getvalue()


def test_a_wrapped_text_only_401_cause_prints_authentication_failed():
    exc = RuntimeError("Failed to load adapter")
    exc.__cause__ = Exception(_401_TEXT)
    out = _render(exc)
    assert "Authentication failed." in out
    assert "Failed to load adapter" not in out


def test_a_wrapped_text_only_403_cause_prints_access_denied():
    exc = RuntimeError("Failed to load adapter")
    exc.__cause__ = Exception("403 Client Error: Forbidden for url: https://huggingface.co/x")
    out = _render(exc)
    assert "Access denied." in out
    assert "Failed to load adapter" not in out


def test_an_implicitly_chained_text_only_401_prints_authentication_failed():
    """`raise ... from` sets `__cause__`; a `raise` inside an `except` sets `__context__`."""
    try:
        try:
            raise Exception(_401_TEXT)
        except Exception:
            raise RuntimeError("Failed to load adapter")
    except RuntimeError as exc:
        out = _render(exc)
    assert "Authentication failed." in out
    assert "Failed to load adapter" not in out


def test_the_outer_phrasing_decides_when_a_403_wraps_a_401():
    """Exception-major, not pattern-major: the 401 pattern comes first in the table, so a
    pattern-major walk would let the inner 401 phrase beat the outer 403 message."""
    exc = RuntimeError(_403_TEXT)
    exc.__cause__ = Exception(_401_TEXT)
    out = _render(exc)
    assert "Access denied." in out
    assert "Authentication failed." not in out


def test_the_inner_phrasing_still_decides_when_the_outer_has_none():
    exc = RuntimeError("Failed to load adapter")
    inner = ValueError("the checkpoint is gone")
    inner.__cause__ = Exception(_403_TEXT)
    exc.__cause__ = inner
    out = _render(exc)
    assert "Access denied." in out


class _QuietHTTPError(urllib.error.HTTPError):
    """A urllib error whose message states no status, so only `.code` can classify it."""

    def __str__(self) -> str:
        return "download refused"


def test_a_wrapped_code_only_error_is_still_read_from_its_code():
    # urllib's own HTTPError always renders "HTTP Error 403: ...", which the text
    # fallback now reads anywhere in the chain, so a plain wrapped HTTPError no
    # longer shows whether the `.code` probe works. This one says nothing in text.
    cause = _QuietHTTPError("https://x/y", 403, "Forbidden", None, None)
    try:
        raise RuntimeError("download failed") from cause
    except RuntimeError as exc:
        out = _render(exc)
    assert "Access denied." in out


class _UnprintableError(Exception):
    def __str__(self) -> str:
        raise RuntimeError("__str__ failed")


def test_an_unprintable_wrapped_exception_does_not_break_the_formatter():
    # The text fallback now calls str() on every wrapped exception; one whose own
    # __str__ raises must not turn the friendly error into a second traceback.
    exc = RuntimeError("Failed to load adapter")
    exc.__cause__ = _UnprintableError()
    out = _render(exc)
    assert "Failed to load adapter" in out
