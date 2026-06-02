"""`soup ab` — mSPRT A/B harness (v0.63.0 Part D)."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from soup_cli.commands._webhook_cli import emit_webhooks, validate_webhook_flags
from soup_cli.utils.ab_test import (
    MsprtConfig,
    run_msprt,
    validate_metric_name,
)

console = Console()


def ab(
    input_path: str = typer.Option(
        ..., "--input", "-i", help="JSONL with {arm, <metric>} per row.",
    ),
    metric: str = typer.Option(
        ..., "--metric",
        help="Metric: latency | judge_score | retry_rate.",
    ),
    alpha: float = typer.Option(
        0.05, "--alpha", help="Type-I error (false positive) rate (0, 1).",
    ),
    beta: float = typer.Option(
        0.20, "--beta", help="Type-II error (false negative) rate (0, 1).",
    ),
    effect_size: float = typer.Option(
        0.1, "--effect-size",
        help="Minimum detectable difference in means.",
    ),
    slack_url: Optional[str] = typer.Option(
        None, "--slack-url",
        help=(
            "Optional Slack webhook URL — POSTed on a reject_h0 / accept_h0 "
            "decision (not on continue). SSRF-validated."
        ),
    ),
    discord_url: Optional[str] = typer.Option(
        None, "--discord-url",
        help=(
            "Optional Discord webhook URL — POSTed on a reject_h0 / accept_h0 "
            "decision (not on continue). SSRF-validated."
        ),
    ),
) -> None:
    """Sequential A/B test with early-stop guarantees (mSPRT)."""
    try:
        canonical = validate_metric_name(metric)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    slack_url, discord_url = validate_webhook_flags(
        slack_url, discord_url, console=console
    )

    try:
        cfg = MsprtConfig(
            metric=canonical, alpha=alpha, beta=beta, effect_size=effect_size,
        )
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    try:
        verdict = run_msprt(input_path, config=cfg)
    except FileNotFoundError:
        console.print(f"[red]Input not found: {escape(input_path)}[/]")
        raise typer.Exit(1) from None
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc

    decision_colour = {
        "reject_h0": "green",
        "accept_h0": "yellow",
        "continue": "cyan",
    }[verdict.decision]

    table = Table(title=f"mSPRT verdict — {escape(canonical)}", border_style=decision_colour)
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("decision", f"[bold]{verdict.decision}[/]")
    table.add_row("log_likelihood_ratio", f"{verdict.log_likelihood_ratio:.4f}")
    table.add_row("n_control", str(verdict.n_control))
    table.add_row("n_treatment", str(verdict.n_treatment))
    table.add_row("mean_control", f"{verdict.mean_control:.4f}")
    table.add_row("mean_treatment", f"{verdict.mean_treatment:.4f}")
    console.print(table)

    if verdict.decision == "reject_h0":
        console.print(
            Panel(
                "[green]Significant difference detected. Promote / rollback "
                "via `soup loop canary` (v0.58).[/]",
                border_style="green",
            )
        )
    elif verdict.decision == "accept_h0":
        console.print(
            Panel(
                "[yellow]No significant difference. Treatment is not "
                "distinguishable from control at the configured effect size.[/]",
                border_style="yellow",
            )
        )
    else:
        console.print(
            Panel(
                "[cyan]Insufficient evidence. Collect more samples and re-run.[/]",
                border_style="cyan",
            )
        )

    # Webhook only fires on a terminal decision (reject_h0 / accept_h0) —
    # a `continue` verdict carries no actionable signal (issue #207).
    if verdict.decision != "continue":
        emit_webhooks(
            slack_url,
            discord_url,
            payload={
                "command": "ab",
                "metric": canonical,
                "decision": verdict.decision,
                "log_likelihood_ratio": verdict.log_likelihood_ratio,
                "n_control": verdict.n_control,
                "n_treatment": verdict.n_treatment,
                "mean_control": verdict.mean_control,
                "mean_treatment": verdict.mean_treatment,
            },
            console=console,
        )


__all__ = ["ab"]
