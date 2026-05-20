"""`soup prune-prompt` — strip shared system-prompt prefix from training data.

Detect a static system-prompt prefix across all rows of a trace JSONL,
strip it from training data so the FT model internalises it (OpenPipe's
signature trick — shipped OSS for v0.63.0 Part B).
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from soup_cli.utils.prune_prompt import prune_traces, validate_min_frequency

console = Console()


def prune_prompt_cmd(
    input_path: str = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input JSONL with {prompt, output} rows (e.g. from `soup ingest`).",
    ),
    output_path: str = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output JSONL with the shared prefix stripped from `prompt`.",
    ),
    min_frequency: float = typer.Option(
        0.95,
        "--min-frequency",
        help="Prefix must appear in >= this fraction of rows to be stripped (0.0 - 1.0).",
    ),
) -> None:
    """Detect + strip a shared system-prompt prefix (v0.63.0 Part B)."""
    try:
        validate_min_frequency(min_frequency)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    try:
        report = prune_traces(
            input_path,
            output_path=output_path,
            min_frequency=min_frequency,
        )
    except FileNotFoundError:
        console.print(f"[red]Input not found: {escape(input_path)}[/]")
        raise typer.Exit(1) from None
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc

    if not report.prefix:
        console.print(
            Panel(
                f"No prefix found at >= {report.min_frequency:.0%} threshold.\n"
                f"Rows in: {report.rows_total}",
                title="prune-prompt",
                border_style="yellow",
            )
        )
        return

    snippet = report.prefix if len(report.prefix) <= 200 else report.prefix[:200] + "..."
    console.print(
        Panel(
            f"[green]Stripped prefix ({report.prefix_chars} chars) from "
            f"{report.rows_pruned}/{report.rows_total} rows.[/]\n\n"
            f"[dim]Prefix:[/] {escape(snippet)}",
            title="prune-prompt",
            border_style="green",
        )
    )


__all__ = ["prune_prompt_cmd"]
