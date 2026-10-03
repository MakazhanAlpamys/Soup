"""`soup ab` — mSPRT A/B harness (v0.63.0 Part D)."""

from __future__ import annotations

import math
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from soup_cli.commands._webhook_cli import emit_webhooks, validate_webhook_flags
from soup_cli.utils.ab_test import (
    ACCEPT_HOLD_SPREAD_RATIO,
    HIGHER_IS_BETTER,
    PRIOR_SCALE_ROWS,
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
        0.05, "--alpha",
        help=(
            "Type-I error (false positive) rate (0, 1) of the two-sided test, kept for a "
            "test re-run after every new row, at any number of rows."
        ),
    ),
    beta: float = typer.Option(
        0.20, "--beta",
        help=(
            "Type-II error (false negative) rate (0, 1). Not the power: a power "
            "of 0.95 is --beta 0.05. alpha + beta must stay below 1, or the "
            "test accepts H0 on no evidence either way."
        ),
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

    # The test is two-sided (#1227): a reject_h0 carries the direction, read
    # through the metric's polarity, and only a worse treatment is a rollback.
    worse = verdict.direction == "worse"
    polarity = "higher is better" if HIGHER_IS_BETTER[canonical] else "lower is better"
    decision_colour = {
        "reject_h0": "red" if worse else "green",
        "accept_h0": "yellow",
        "continue": "cyan",
    }[verdict.decision]

    table = Table(title=f"mSPRT verdict — {escape(canonical)}", border_style=decision_colour)
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("decision", f"[bold]{verdict.decision}[/]")
    table.add_row("direction", verdict.direction or "n/a")
    table.add_row("prior_scale", f"first {PRIOR_SCALE_ROWS} rows per arm")
    table.add_row("log_likelihood_ratio", f"{verdict.log_likelihood_ratio:.4f}")
    table.add_row("n_control", str(verdict.n_control))
    table.add_row("n_treatment", str(verdict.n_treatment))
    table.add_row("mean_control", f"{verdict.mean_control:.4f}")
    table.add_row("mean_treatment", f"{verdict.mean_treatment:.4f}")
    # #1524 - the two spreads behind the hold, when it is what kept the verdict
    # at `continue`. Shown next to the statistic so the reader can see that the
    # accept boundary was already crossed.
    if verdict.accept_held:
        table.add_row("accept_held", "[bold]true[/]")
        table.add_row("held_out_rows", str(verdict.held_out_rows))
        table.add_row("held_out_spread", f"{verdict.held_out_spread:.4g}")
        table.add_row("tested_spread", f"{verdict.tested_spread:.4g}")
    console.print(table)

    if verdict.accept_held:
        # #1524 - the accept boundary is already crossed; the only thing keeping
        # this at `continue` is the 3x spread rule. The held-out rows are fixed,
        # so "collect more samples" cannot help here — name the ratio and point
        # at the rows that actually set it.
        ratio = (
            verdict.tested_spread / verdict.held_out_spread
            if verdict.held_out_spread
            else math.inf
        )
        console.print(
            Panel(
                "[cyan]Held back: the tested rows spread "
                f"{ratio:.3g} times the held-out ones "
                f"(held_out_spread {verdict.held_out_spread:.4g}, "
                f"tested_spread {verdict.tested_spread:.4g}, "
                f"limit {ACCEPT_HOLD_SPREAD_RATIO:g}x), so accept_h0 is "
                f"withheld at log_likelihood_ratio {verdict.log_likelihood_ratio:.4f}. "
                f"The first {verdict.held_out_rows} rows of each arm set the scale of "
                "--effect-size and barely vary (a warm cache, or a judge that gives "
                "the same score early). Those rows are fixed, so more samples of the "
                "same kind will not release the hold: drop them, or reorder the input "
                "so the first rows vary, and re-run.[/]",
                border_style="cyan",
            )
        )
    elif verdict.decision == "reject_h0" and worse:
        console.print(
            Panel(
                "[red]Significant difference detected: the treatment is worse than "
                f"control on {escape(canonical)} ({polarity}). Recommend rollback: "
                "keep serving the control.[/]",
                border_style="red",
            )
        )
    elif verdict.decision == "reject_h0":
        console.print(
            Panel(
                "[green]Significant difference detected: the treatment is better than "
                f"control on {escape(canonical)} ({polarity}). Promote via "
                "`soup loop canary` (v0.58).[/]",
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
    elif min(verdict.n_control, verdict.n_treatment) < PRIOR_SCALE_ROWS + 2:
        console.print(
            Panel(
                f"[cyan]Not enough rows yet: the first {PRIOR_SCALE_ROWS} rows of each arm "
                "set the scale of --effect-size, and the test needs 2 more per arm after "
                f"them (control has {verdict.n_control}, treatment {verdict.n_treatment}). "
                "Collect more samples and re-run.[/]",
                border_style="cyan",
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
                "direction": verdict.direction,
                "log_likelihood_ratio": verdict.log_likelihood_ratio,
                "n_control": verdict.n_control,
                "n_treatment": verdict.n_treatment,
                "mean_control": verdict.mean_control,
                "mean_treatment": verdict.mean_treatment,
                # #1524 - added keys, nothing renamed: a consumer reading the
                # old payload still finds every field it had.
                "accept_held": verdict.accept_held,
                "held_out_spread": verdict.held_out_spread,
                "tested_spread": verdict.tested_spread,
                "held_out_rows": verdict.held_out_rows,
            },
            console=console,
        )


__all__ = ["ab"]
