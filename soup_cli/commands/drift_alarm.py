"""`soup drift-alarm` — online-eval drift alarm (v0.63.0 Part E)."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from soup_cli.utils.drift_alarm import (
    post_webhook,
    run_drift_check,
    validate_threshold,
    validate_webhook_url,
)

console = Console()


def drift_alarm(
    reference_path: str = typer.Option(
        ..., "--reference", help="JSONL of FT-time reference outputs.",
    ),
    live_path: str = typer.Option(
        ..., "--live", help="JSONL of live production outputs.",
    ),
    threshold: float = typer.Option(
        0.2, "--threshold",
        help="Drift threshold (KL divergence). Default 0.2.",
    ),
    slack_url: Optional[str] = typer.Option(
        None, "--slack-url",
        help="Optional Slack webhook URL — POSTed on drift. SSRF-validated.",
    ),
    discord_url: Optional[str] = typer.Option(
        None, "--discord-url",
        help="Optional Discord webhook URL — POSTed on drift. SSRF-validated.",
    ),
) -> None:
    """Compute KL divergence between FT-time reference + live token distributions."""
    try:
        validate_threshold(threshold)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    if slack_url is not None:
        try:
            slack_url = validate_webhook_url(slack_url)
        except (TypeError, ValueError) as exc:
            console.print(f"[red]--slack-url: {escape(str(exc))}[/]")
            raise typer.Exit(2) from exc
    if discord_url is not None:
        try:
            discord_url = validate_webhook_url(discord_url)
        except (TypeError, ValueError) as exc:
            console.print(f"[red]--discord-url: {escape(str(exc))}[/]")
            raise typer.Exit(2) from exc

    try:
        report = run_drift_check(
            reference_path=reference_path,
            live_path=live_path,
            threshold=threshold,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]File not found: {escape(str(exc))}[/]")
        raise typer.Exit(1) from None
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc

    border = "red" if report.drift_detected else "green"
    verdict = "DRIFT" if report.drift_detected else "OK"
    panel_body = (
        f"[bold]{verdict}[/]\n"
        f"KL: {report.kl_divergence:.4f}  threshold: {report.threshold:.4f}\n"
        f"n_reference: {report.n_reference}  n_live: {report.n_live}"
    )
    if report.top_drift_tokens:
        top_str = ", ".join(
            f"{escape(t)}={d:.3f}" for t, d in report.top_drift_tokens[:3]
        )
        panel_body += f"\nTop drift tokens: {top_str}"
    console.print(Panel(panel_body, title="drift-alarm", border_style=border))

    if report.drift_detected:
        payload = {
            "kl": report.kl_divergence,
            "threshold": report.threshold,
            "n_reference": report.n_reference,
            "n_live": report.n_live,
            "top_drift_tokens": [
                [t, d] for t, d in report.top_drift_tokens
            ],
        }
        sent = []
        if slack_url:
            sent.append(("slack", post_webhook(url=slack_url, payload=payload)))
        if discord_url:
            sent.append(("discord", post_webhook(url=discord_url, payload=payload)))
        for label, ok in sent:
            colour = "green" if ok else "yellow"
            console.print(
                f"[{colour}]{label} webhook: {'delivered' if ok else 'failed'}[/]"
            )
        # Drift detected -> non-zero exit so the operator's cron / Loop
        # can flag it. Matches v0.55 / v0.56 gate convention.
        raise typer.Exit(3)


__all__ = ["drift_alarm"]
