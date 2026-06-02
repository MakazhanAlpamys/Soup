"""`soup ingest` — universal trace importer (v0.63.0 Part A).

Imports production traces from Langfuse / LangSmith / Helicone / OpenPipe /
OpenTelemetry / OpenAI Stored Completions JSONL exports and emits a
normalised JSONL stream that downstream tools (`soup data from-traces`,
`soup loop watch`) can consume.

Composes with v0.26.0 Trace-to-Preference: the emitted records share the
same prompt/output/signal vocabulary so the existing pair-builder works
unchanged after a thin shim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from soup_cli.commands._webhook_cli import emit_webhooks, validate_webhook_flags
from soup_cli.utils import ingest_sources as _ingest_sources
from soup_cli.utils.ingest_sources import (
    SUPPORTED_INGEST_SOURCES,
    ingest_traces,
    resolve_auth_env,
    validate_source_name,
)
from soup_cli.utils.paths import is_under_cwd

console = Console()


def ingest(
    source: str = typer.Option(
        ...,
        "--source",
        help=(
            "Trace source: langfuse | langsmith | helicone | openpipe | "
            "otel | openai-stored"
        ),
    ),
    logs: str = typer.Option(
        ...,
        "--logs",
        help="Path to JSONL trace export (one event per line).",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output JSONL (default: traces.jsonl in cwd).",
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
    """Import production traces from a SaaS observability vendor (v0.63.0).

    Reads an offline JSONL export and writes a normalised trace stream.
    No network calls — operators export from their SaaS dashboard or via
    that vendor's official API, then point ``soup ingest`` at the file.
    """
    try:
        canonical = validate_source_name(source)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    slack_url, discord_url = validate_webhook_flags(
        slack_url, discord_url, console=console
    )

    if not is_under_cwd(logs):
        console.print(f"[red]--logs '{escape(logs)}' is outside cwd — refusing[/]")
        raise typer.Exit(1)
    logs_path = Path(logs)
    if not logs_path.exists():
        console.print(f"[red]--logs not found: {escape(logs)}[/]")
        raise typer.Exit(1)

    output_path = Path(output) if output else Path("traces.jsonl")
    if not is_under_cwd(output_path):
        console.print(
            f"[red]--output '{escape(str(output_path))}' is outside cwd — refusing[/]"
        )
        raise typer.Exit(1)

    # PII reminder — matches v0.26.0 Part C policy.
    console.print(
        Panel(
            "[yellow]Traces may contain sensitive user data (PII).[/]\n"
            "Review the output before sharing or uploading to external systems.\n"
            f"Auth env var for this source: "
            f"[bold]{escape(_env_label(canonical))}[/]",
            title="PII reminder",
            border_style="yellow",
        )
    )

    auth_value = resolve_auth_env(canonical)
    if auth_value is None:
        console.print(
            "[dim]No auth env var set — this CLI parses the local export "
            "only (no SaaS pull).[/]"
        )

    count = 0
    with open(output_path, "w", encoding="utf-8") as out_fh:
        for record in ingest_traces(source=canonical, path=str(logs_path)):
            out_fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            count += 1

    console.print(
        f"[green]Wrote {count} traces from {escape(canonical)} -> "
        f"{escape(output_path.name)}[/]"
    )

    emit_webhooks(
        slack_url,
        discord_url,
        payload={
            "command": "ingest",
            "source": canonical,
            "traces_written": count,
            "auth_env_set": auth_value is not None,
        },
        console=console,
    )


def _env_label(source: str) -> str:
    """Return the env-var name that authenticates ``source``.

    Single source of truth: ``ingest_sources._AUTH_ENV``. Avoids the
    drift hazard of duplicating the table here (code-review MEDIUM fix
    v0.63.0).
    """
    return _ingest_sources._AUTH_ENV.get(source, "(unset)")


__all__ = ["ingest", "SUPPORTED_INGEST_SOURCES"]
