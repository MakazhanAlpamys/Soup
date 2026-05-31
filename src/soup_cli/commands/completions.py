"""`soup completions <shell>` — emit a bash / zsh / fish completion script.

Sourceable script: `eval "$(soup completions bash)"` adds tab-completion
for the `soup` command in the current shell. Designed for `eval`
consumption — every output goes to stdout exactly once, no panels or
banners.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markup import escape

from soup_cli.utils.completions import render_completion_script

# Console with stderr so the script body (stdout) stays clean for eval.
_err = Console(stderr=True)


def completions_cmd(
    shell: str = typer.Argument(
        ...,
        help="One of: bash, zsh, fish.",
    ),
) -> None:
    """Render a shell completion script for `soup` (v0.64.0)."""
    try:
        text = render_completion_script(shell)
    except (TypeError, ValueError) as exc:
        _err.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc
    # Plain stdout — the script gets sourced by `eval`.
    typer.echo(text)


__all__ = ["completions_cmd"]
