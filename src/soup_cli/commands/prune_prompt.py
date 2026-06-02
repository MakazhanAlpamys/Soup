"""`soup prune-prompt` — strip shared system-prompt prefix from training data.

Detect a static system-prompt prefix across all rows of a trace JSONL,
strip it from training data so the FT model internalises it (OpenPipe's
signature trick — shipped OSS for v0.63.0 Part B).
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from soup_cli.commands._webhook_cli import emit_webhooks, validate_webhook_flags
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
    tokenizer: Optional[str] = typer.Option(
        None,
        "--tokenizer",
        help=(
            "Optional HF tokenizer (model id or local path) for token-aware "
            "prefix detection. Default: whitespace-character level."
        ),
    ),
    slack_url: Optional[str] = typer.Option(
        None, "--slack-url",
        help="Optional Slack webhook URL — POSTed on completion. SSRF-validated.",
    ),
    discord_url: Optional[str] = typer.Option(
        None, "--discord-url",
        help="Optional Discord webhook URL — POSTed on completion. SSRF-validated.",
    ),
) -> None:
    """Detect + strip a shared system-prompt prefix (v0.63.0 Part B)."""
    try:
        validate_min_frequency(min_frequency)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    slack_url, discord_url = validate_webhook_flags(
        slack_url, discord_url, console=console
    )

    try:
        report = prune_traces(
            input_path,
            output_path=output_path,
            min_frequency=min_frequency,
            tokenizer=tokenizer,
        )
    except FileNotFoundError:
        console.print(f"[red]Input not found: {escape(input_path)}[/]")
        raise typer.Exit(1) from None
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc

    payload = {
        "command": "prune-prompt",
        "prefix_found": bool(report.prefix),
        "prefix_chars": report.prefix_chars,
        "rows_pruned": report.rows_pruned,
        "rows_total": report.rows_total,
    }

    if not report.prefix:
        console.print(
            Panel(
                f"No prefix found at >= {report.min_frequency:.0%} threshold.\n"
                f"Rows in: {report.rows_total}",
                title="prune-prompt",
                border_style="yellow",
            )
        )
        emit_webhooks(slack_url, discord_url, payload=payload, console=console)
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
    emit_webhooks(slack_url, discord_url, payload=payload, console=console)


__all__ = ["prune_prompt_cmd"]
