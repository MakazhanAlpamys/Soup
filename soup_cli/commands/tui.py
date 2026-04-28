"""soup tui — full-screen Textual dashboard for live + historical runs.

The Textual app itself is constructed inside ``run_tui`` so the import is
lazy and the rest of the CLI keeps working when ``textual`` isn't
installed. The user-facing error mirrors the pattern used by ``soup ui``
and the optional ``[tui]`` install extra.
"""

from __future__ import annotations

import typer
from rich.console import Console

console = Console()


def _missing_dep_panel() -> str:
    return (
        "[red]Textual is not installed.[/]\n\n"
        "[bold]Install:[/] pip install 'soup-cli[tui]'\n"
        "[dim]Or directly:[/] pip install textual"
    )


def tui(
    refresh: float = typer.Option(
        1.0,
        "--refresh",
        help="Refresh interval in seconds (0.25 - 10.0).",
    ),
    limit: int = typer.Option(
        50,
        "--limit",
        help="Maximum runs to load into the dashboard.",
    ),
):
    """Full-screen terminal dashboard: list runs, inspect metrics, tail logs."""
    if refresh < 0.25 or refresh > 10.0:
        console.print("[red]--refresh must be between 0.25 and 10.0[/]")
        raise typer.Exit(2)
    if limit < 1 or limit > 1000:
        console.print("[red]--limit must be between 1 and 1000[/]")
        raise typer.Exit(2)

    try:
        from soup_cli.tui_app import SoupTuiApp
    except ImportError:
        console.print(_missing_dep_panel())
        raise typer.Exit(1) from None

    app = SoupTuiApp(refresh_secs=refresh, run_limit=limit)
    app.run()
