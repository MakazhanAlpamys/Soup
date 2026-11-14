"""User-provided strings in friendly diagnostics and tracebacks are literal."""

from io import StringIO

import pytest
from rich.console import Console

from soup_cli.utils import errors
from tests.conftest import strip_ansi


@pytest.mark.parametrize("known", [False, True], ids=["fallback", "matched"])
@pytest.mark.parametrize(
    "value,shown",
    [
        (":warning:", ":warning:"),
        ("[red]name", "[red]name"),
        ("x\\[n", "x\\[n"),
        ("x\x9b\u202en", "xn"),
    ],
)
def test_error_and_verbose_traceback_keep_literal_message(monkeypatch, known, value, shown):
    stream = StringIO()
    monkeypatch.setattr(
        errors,
        "console",
        Console(
            file=stream,
            width=300,
            height=100,
            force_terminal=True,
            no_color=False,
            color_system="standard",
        ),
    )
    prefix = "No such file or directory: " if known else "model "
    kind = "RuntimeError" if known else "Failure:warning:x\\[n\u202e"
    exc_type = type(kind, (RuntimeError,), {})
    try:
        raise exc_type(prefix + value)
    except RuntimeError as exc:
        errors.format_friendly_error(exc, verbose=True)

    rendered = stream.getvalue()
    plain = strip_ansi(rendered)
    displayed_type = kind.replace("\u202e", "")
    expected = prefix + shown
    expected_summary = expected if known else f"{displayed_type}: {expected}"
    assert plain.startswith(f"\nError: {expected_summary}\n")
    assert "\x1b[1;31mError:\x1b[0m " in rendered
    if known:
        assert "\x1b[32mFix:\x1b[0m Check the file path." in rendered
    else:
        assert "\x1b[2mRun with --verbose for the full traceback.\x1b[0m" in rendered
    assert "Full Traceback" in plain
    assert f"{displayed_type}: {expected}" in plain.split("Full Traceback", 1)[1]
    assert "\x9b" not in plain and "\u202e" not in plain
