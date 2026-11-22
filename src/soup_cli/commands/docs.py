"""Documentation search and reference command for Soup CLI."""

from __future__ import annotations

import json
import sys

import typer
from rich.console import Console
from rich.table import Table

from soup_cli.utils.terminal import for_terminal

console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    no_args_is_help=True,
    help="Documentation search & offline reference.",
)


@app.command(name="search")
def search(
    query: str = typer.Argument(..., help="Search query (e.g. 'safetensors export')"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to return"),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: 'table' (default) or 'json'",
    ),
) -> None:
    """Search Soup documentation for flags, configs, training tasks, and guides."""
    from soup_cli.docs.retriever import search_docs

    if top_k < 1:
        err_console.print(
            f"[bold red]Error:[/bold red] --top-k must be greater than or equal to 1, got {top_k}."
        )
        raise typer.Exit(code=2)

    fmt = format.lower()
    if fmt not in ("table", "json"):
        fmt_term = for_terminal(format)
        err_console.print(
            f"[bold red]Error:[/bold red] Unknown --format: '{fmt_term}'. Choose 'table' or 'json'."
        )
        raise typer.Exit(code=2)

    try:
        results = search_docs(query, top_k=top_k)
    except (RuntimeError, ValueError, FileNotFoundError, OSError) as err:
        err_console.print(f"[bold red]Error:[/bold red] {for_terminal(str(err))}")
        raise typer.Exit(code=1)

    if fmt == "json":
        sys.stdout.write(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        return

    if not results:
        q_term = for_terminal(query)
        console.print(f"[dim]No matching documentation found for:[/dim] [italic]{q_term}[/italic]")
        return

    table = Table(title=f"Documentation Search: '{for_terminal(query)}'")
    table.add_column("Doc", style="cyan", no_wrap=True)
    table.add_column("Section", style="bold green")
    table.add_column("Score", style="yellow", justify="right")
    table.add_column("Excerpt", style="dim")

    for r in results:
        excerpt = (r.get("text") or "").replace("\n", " ").strip()
        if len(excerpt) > 120:
            excerpt = excerpt[:117] + "..."
        table.add_row(
            for_terminal(r.get("doc_name", "")),
            for_terminal(r.get("section", "")),
            f"{r.get('score', 0.0):.4f}",
            for_terminal(excerpt),
        )

    console.print(table)
