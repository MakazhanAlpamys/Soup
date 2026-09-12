"""`soup ingest` — universal trace importer (v0.63.0 Part A).

Imports production traces from Langfuse / LangSmith / Helicone / OpenPipe /
OpenTelemetry / OpenAI Stored Completions JSONL exports and emits a
normalised JSONL stream that downstream tools (`soup data from-traces`,
`soup loop watch`) can consume.

Composes with v0.26.0 Trace-to-Preference: the emitted records share the
same prompt/output/signal vocabulary so the existing pair-builder works
unchanged after a thin shim.

``--source langfuse --pull`` (#204) fetches from the Langfuse API instead of
reading an export; ``utils/ingest_pull.py`` is imported only on that path.
"""

from __future__ import annotations

import json
import os
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

_DEFAULT_SINCE = "7d"


def ingest(
    source: str = typer.Option(
        ...,
        "--source",
        help=(
            "Trace source: langfuse | langsmith | helicone | openpipe | "
            "otel | openai-stored"
        ),
    ),
    logs: Optional[str] = typer.Option(
        None,
        "--logs",
        help="Path to JSONL trace export (one event per line). Required unless --pull.",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output JSONL (default: traces.jsonl in cwd).",
    ),
    pull: bool = typer.Option(
        False,
        "--pull",
        help=(
            "Fetch generations live from the Langfuse API instead of reading "
            "--logs (--source langfuse only). Reads LANGFUSE_PUBLIC_KEY and "
            "LANGFUSE_SECRET_KEY, plus LANGFUSE_HOST for another region or a "
            "self-hosted instance."
        ),
    ),
    since: Optional[str] = typer.Option(
        None,
        "--since",
        help="With --pull: how far back to fetch, e.g. 30m, 24h, 7d (default 7d, max 365d).",
    ),
    max_pages: Optional[int] = typer.Option(
        None,
        "--max-pages",
        min=1,
        max=10_000,
        help=(
            "With --pull: stop with an error, writing nothing, if results are still "
            "pending after this many pages of 100 generations (default 100)."
        ),
    ),
    allow_private_host: bool = typer.Option(
        False,
        "--allow-private-host",
        help=(
            "With --pull: allow LANGFUSE_HOST to be a private or loopback address "
            "(self-hosted Langfuse). HTTPS is still required."
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
    """Import production traces from a SaaS observability vendor (v0.63.0).

    Reads an offline JSONL export and writes a normalised trace stream,
    making no network calls — operators export from their SaaS dashboard or
    via that vendor's official API, then point ``soup ingest`` at the file.
    ``--source langfuse --pull`` fetches from the Langfuse API instead (#204).
    """
    try:
        canonical = validate_source_name(source)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    slack_url, discord_url = validate_webhook_flags(
        slack_url, discord_url, console=console
    )

    if pull:
        _pull(
            canonical,
            logs=logs,
            output=output,
            since=since,
            max_pages=max_pages,
            allow_private_host=allow_private_host,
            slack_url=slack_url,
            discord_url=discord_url,
        )
        return

    pull_only = [
        flag
        for flag, given in (
            ("--since", since is not None),
            ("--max-pages", max_pages is not None),
            ("--allow-private-host", allow_private_host),
        )
        if given
    ]
    if pull_only:
        console.print(f"[red]{', '.join(pull_only)}: only meaningful with --pull[/]")
        raise typer.Exit(2)
    if logs is None:
        console.print("[red]Pass --logs <export.jsonl>, or --pull to fetch from Langfuse.[/]")
        raise typer.Exit(2)

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

    _print_pii_reminder(canonical)

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


def _pull(
    canonical: str,
    *,
    logs: Optional[str],
    output: Optional[str],
    since: Optional[str],
    max_pages: Optional[int],
    allow_private_host: bool,
    slack_url: Optional[str],
    discord_url: Optional[str],
) -> None:
    """``--pull``: fetch GENERATION observations from Langfuse and write them parsed (#204)."""
    from soup_cli.utils.paths import atomic_write_lines, enforce_under_cwd_and_no_symlink

    if logs is not None:
        console.print(
            "[red]--pull and --logs are mutually exclusive: fetch live or read an export.[/]"
        )
        raise typer.Exit(2)
    if canonical != "langfuse":
        console.print(
            f"[red]--pull is only available for --source langfuse so far; export "
            f"{escape(canonical)} traces to JSONL and pass --logs instead.[/]"
        )
        raise typer.Exit(2)

    output_path = Path(output) if output else Path("traces.jsonl")
    try:
        enforce_under_cwd_and_no_symlink(str(output_path), "--output")
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc

    from soup_cli.utils import ingest_pull

    since_text = since if since is not None else _DEFAULT_SINCE
    try:
        window = ingest_pull.parse_since(since_text)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc
    try:
        credentials = ingest_pull.load_langfuse_credentials(
            os.environ, allow_private_host=allow_private_host
        )
    except ingest_pull.PullError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from None
    except ValueError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from None

    _print_pii_reminder(canonical)
    console.print(
        f"[dim]Pulling GENERATION observations from {escape(credentials.host)} "
        f"for the last {escape(since_text)}...[/]"
    )

    stats = ingest_pull.PullStats()
    written = 0

    def _lines():
        nonlocal written
        generations = ingest_pull.pull_langfuse_generations(
            credentials,
            since=window,
            max_pages=max_pages if max_pages is not None else ingest_pull.DEFAULT_MAX_PAGES,
            stats=stats,
        )
        for record in _ingest_sources.parse_langfuse(generations):
            written += 1
            yield json.dumps(record.to_dict(), ensure_ascii=False) + "\n"

    try:
        # Streams to a staging file; the target only appears once the pull finished.
        atomic_write_lines(_lines(), str(output_path), field="--output")
    except ingest_pull.PullError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        console.print(f"[red]Nothing was written to {escape(output_path.name)}.[/]")
        raise typer.Exit(1) from None

    console.print(
        f"[green]Wrote {written} traces from langfuse ({stats.generations} generations "
        f"pulled) -> {escape(output_path.name)}[/]"
    )
    if stats.generations > written:
        console.print(
            f"[yellow]{stats.generations - written} generation(s) had no input or no output "
            "and were skipped.[/]"
        )
    if stats.skipped_not_generation:
        console.print(
            f"[yellow]{stats.skipped_not_generation} observation(s) were not generations "
            "and were skipped.[/]"
        )

    emit_webhooks(
        slack_url,
        discord_url,
        payload={
            "command": "ingest",
            "source": canonical,
            "traces_written": written,
            "auth_env_set": True,
            "generations_pulled": stats.generations,
        },
        console=console,
    )


def _print_pii_reminder(canonical: str) -> None:
    """PII reminder — matches v0.26.0 Part C policy."""
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


def _env_label(source: str) -> str:
    """Return the env-var names that authenticate ``source``.

    Single source of truth: ``ingest_sources._AUTH_ENV``. Avoids the
    drift hazard of duplicating the table here (code-review MEDIUM fix
    v0.63.0).
    """
    return " + ".join(_ingest_sources._AUTH_ENV.get(source, ("(unset)",)))


__all__ = ["ingest", "SUPPORTED_INGEST_SOURCES"]
