"""Shared CLI glue for the --slack-url / --discord-url webhook flags (v0.71.5 #207).

Keeps Typer + Rich Console concerns in the commands layer (``utils/webhooks``
stays import-light + framework-free). Used by ``ingest`` / ``prune-prompt`` /
``ab`` / ``data active-sample`` so the validate-then-deliver pattern is defined
once.
"""

from __future__ import annotations

from typing import Optional, Tuple

import typer
from rich.console import Console
from rich.markup import escape


def validate_webhook_flags(
    slack_url: Optional[str],
    discord_url: Optional[str],
    *,
    console: Console,
) -> Tuple[Optional[str], Optional[str]]:
    """Validate webhook URLs at the CLI boundary (``typer.Exit(2)`` on bad).

    Returns the (canonical) URLs. Mirrors the v0.63.0 ``drift-alarm``
    early-rejection pattern so a typo'd / SSRF-y URL fails fast with a
    friendly message instead of being silently swallowed at delivery time.
    """
    from soup_cli.utils.webhooks import validate_webhook_url

    out = []
    for label, value in (("--slack-url", slack_url), ("--discord-url", discord_url)):
        if value is None:
            out.append(None)
            continue
        try:
            out.append(validate_webhook_url(value))
        except (TypeError, ValueError) as exc:
            console.print(f"[red]{label}: {escape(str(exc))}[/]")
            raise typer.Exit(2) from exc
    return out[0], out[1]


def emit_webhooks(
    slack_url: Optional[str],
    discord_url: Optional[str],
    *,
    payload: dict,
    console: Console,
) -> None:
    """POST the completion payload to any configured webhooks (best-effort).

    Never raises (delegates to the never-raising
    :func:`soup_cli.utils.webhooks.send_webhooks`). Prints a per-target
    delivered/failed line so the operator sees whether the alert landed.
    """
    if slack_url is None and discord_url is None:
        return
    from soup_cli.utils.webhooks import send_webhooks

    for label, ok in send_webhooks(
        payload, slack_url=slack_url, discord_url=discord_url
    ):
        colour = "green" if ok else "yellow"
        console.print(
            f"[{colour}]{label} webhook: {'delivered' if ok else 'failed'}[/]"
        )


__all__ = ["emit_webhooks", "validate_webhook_flags"]
