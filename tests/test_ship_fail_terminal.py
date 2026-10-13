"""``ship._fail`` strips control bytes as well as escaping Rich markup (#1869).

``_verdict_live`` forwards an exception raised while rendering a Hub model's chat
template, which is text the user did not write. ``escape`` alone left a raw ESC
or C1 byte to reach the terminal.
"""

from __future__ import annotations

from io import StringIO

import pytest
import typer
from rich.console import Console

from soup_cli.commands import ship


def test_fail_strips_control_bytes_and_keeps_markup_literal(monkeypatch):
    out = StringIO()
    monkeypatch.setattr(ship, "console", Console(file=out, color_system=None, width=200))

    with pytest.raises(typer.Exit) as info:
        ship._fail("a" + chr(0x1B) + "b" + chr(0x9B) + "c [red]d", 3)

    assert info.value.exit_code == 3
    text = out.getvalue()
    assert chr(0x1B) not in text
    assert chr(0x9B) not in text
    assert text == "Error: abc [red]d\n"
