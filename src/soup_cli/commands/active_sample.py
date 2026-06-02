"""`soup data active-sample` — active-learning sampler (v0.63.0 Part C)."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from soup_cli.commands._webhook_cli import emit_webhooks, validate_webhook_flags
from soup_cli.utils.active_sampler import sample_uncertain_rows, validate_budget

console = Console()


def active_sample(
    input_path: str = typer.Option(
        ..., "--input", "-i", help="JSONL of trace rows with rm_score or rm_scores fields.",
    ),
    output: str = typer.Option(
        "active_samples.jsonl", "--output", "-o",
        help="Output JSONL of the top-uncertainty rows (default: active_samples.jsonl).",
    ),
    budget: int = typer.Option(
        100, "--budget",
        help="Max rows to surface for human review (1 - 100_000).",
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
    """Surface the most uncertain prod traces for human review."""
    try:
        validate_budget(budget)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    slack_url, discord_url = validate_webhook_flags(
        slack_url, discord_url, console=console
    )

    try:
        plan = sample_uncertain_rows(
            input_path,
            output_path=output,
            budget=budget,
        )
    except FileNotFoundError:
        console.print(f"[red]Input not found: {escape(input_path)}[/]")
        raise typer.Exit(1) from None
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc

    console.print(
        Panel(
            f"[green]Selected {plan.rows_selected}/{plan.rows_in} rows[/]\n"
            f"Mean uncertainty: {plan.mean_uncertainty:.3f}\n"
            f"Budget: {plan.budget}",
            title="active-sample",
            border_style="green",
        )
    )

    emit_webhooks(
        slack_url,
        discord_url,
        payload={
            "command": "active-sample",
            "rows_selected": plan.rows_selected,
            "rows_in": plan.rows_in,
            "mean_uncertainty": plan.mean_uncertainty,
            "budget": plan.budget,
        },
        console=console,
    )


__all__ = ["active_sample"]
