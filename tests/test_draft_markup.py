from __future__ import annotations

import re
from io import StringIO

from rich.console import Console

from soup_cli.utils.draft import AcceptanceReport, render_draft_panel


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", _ANSI_RE.sub("", text or ""))


def _report(**overrides: str) -> AcceptanceReport:
    values = dict(
        target="org/[red]EVIL[/red]",
        draft="org/[blue]DRAFT[/blue]",
        n_prompts=1,
        n_generated_tokens=10,
        acceptance_rate=0.75,
        verdict="STRONG",
        tok_s_plain=10.0,
        tok_s_assisted=12.0,
        speedup=1.2,
        num_assistant_tokens=5,
        soup_version="0.75.0",
    )
    values.update(overrides)
    return AcceptanceReport(**values)


def test_draft_panel_escapes_user_controlled_model_names() -> None:
    buf = StringIO()
    Console(file=buf, force_terminal=True, width=100).print(
        render_draft_panel(_report())
    )

    output = _plain(buf.getvalue())
    assert "org/[red]EVIL[/red]" in output
    assert "org/[blue]DRAFT[/blue]" in output
