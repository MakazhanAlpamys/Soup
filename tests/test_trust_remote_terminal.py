"""The explicit remote-code warning identifies the original model literally."""

from io import StringIO

import pytest
from rich.console import Console

from soup_cli.utils.trust_remote import resolve_trust_remote_code
from tests.conftest import strip_ansi


@pytest.mark.parametrize(
    "model,shown",
    [
        ("owner/:warning:", "owner/:warning:"),
        ("owner/[red]n", "owner/[red]n"),
        ("owner/x\\[n", "owner/x\\[n"),
        ("owner/n\\\\", "owner/n\\\\"),
        ("owner/x\x9b\u202en", "owner/xn"),
    ],
)
def test_opt_in_warning_keeps_model_text_and_styles(model, shown):
    stream = StringIO()
    console = Console(
        file=stream,
        width=300,
        height=100,
        force_terminal=True,
        no_color=False,
        color_system="standard",
    )

    assert resolve_trust_remote_code(model, requested=True, console=console) is True

    output = stream.getvalue()
    assert "\x1b[33m--trust-remote-code is enabled for\x1b[0m " in output
    assert f"\x1b[1m{shown}\x1b[0m" in output
    body = [
        line.removeprefix("│ ").removesuffix("│").rstrip()
        for line in strip_ansi(output).splitlines()
        if line.startswith("│")
    ]
    assert body == [
        f"--trust-remote-code is enabled for {shown}",
        "This will execute Python code shipped in the model repo. "
        "Only proceed if you trust the source.",
    ]
