"""`soup ingest` — universal trace importer (v0.63.0 Part A).

Imports production traces from Langfuse / LangSmith / Helicone / OpenPipe /
OpenTelemetry / OpenAI Stored Completions JSONL exports and emits a
normalised JSONL stream that downstream tools (`soup data from-traces`,
`soup loop watch`) can consume.

Composes with v0.26.0 Trace-to-Preference: the emitted records share the
same prompt/output/signal vocabulary so the existing pair-builder works
unchanged after a thin shim.

``--source langfuse --pull`` and ``--source langsmith --pull`` (#204) fetch
from the vendor's API instead of reading an export; ``utils/ingest_pull.py`` and
``utils/ingest_pull_langsmith.py`` are imported only on that path.
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
from rich.text import Text

from soup_cli.commands._common import refuse_output_over_input
from soup_cli.commands._webhook_cli import emit_webhooks, validate_webhook_flags
from soup_cli.utils import ingest_sources as _ingest_sources
from soup_cli.utils.ingest_sources import (
    SUPPORTED_INGEST_SOURCES,
    ingest_traces,
    resolve_auth_env,
    validate_source_name,
)
from soup_cli.utils.paths import atomic_write_lines, is_under_cwd
from soup_cli.utils.terminal import strip_control, terminal_text

console = Console()

_DEFAULT_SINCE = "7d"


# The typed --source and --output, a bad --since, the Langfuse host and the text
# of a refused or failed pull come from outside the program, so every line that
# carries them is printed as ``Text``: shown as typed, never read as markup or
# ``:name:`` emoji, control characters removed (#2489).


def _refuse(message: object) -> None:
    console.print(terminal_text(message, "red"))


class _LogsUnreadableError(Exception):
    """The --logs export could not be read; the message names it and why."""


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
            "Fetch live from the vendor's API instead of reading --logs "
            "(--source langfuse or langsmith). Langfuse: LANGFUSE_PUBLIC_KEY and "
            "LANGFUSE_SECRET_KEY, plus LANGFUSE_HOST for another region or a "
            "self-hosted instance. LangSmith: LANGSMITH_API_KEY and "
            "LANGSMITH_PROJECT, plus LANGSMITH_ENDPOINT for another region or a "
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
            "pending after this many pages of 100 generations or LLM runs (default 100)."
        ),
    ),
    allow_private_host: bool = typer.Option(
        False,
        "--allow-private-host",
        help=(
            "With --pull: allow LANGFUSE_HOST or LANGSMITH_ENDPOINT to be a private "
            "or loopback address (self-hosted). HTTPS is still required."
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
    ``--source langfuse --pull`` and ``--source langsmith --pull`` fetch from the
    vendor's API instead (#204).
    """
    try:
        canonical = validate_source_name(source)
    except (TypeError, ValueError) as exc:
        _refuse(exc)
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
        console.print(
            "[red]Pass --logs <export.jsonl>, or --pull to fetch from Langfuse or LangSmith.[/]"
        )
        raise typer.Exit(2)

    if not is_under_cwd(logs):
        console.print(
            Text(f"--logs '{strip_control(logs)}' is outside cwd — refusing", style="red")
        )
        raise typer.Exit(1)
    logs_path = Path(logs)
    if not logs_path.exists():
        console.print(Text(f"--logs not found: {strip_control(logs)}", style="red"))
        raise typer.Exit(1)
    if logs_path.is_dir():
        console.print(
            Text(f"--logs '{strip_control(logs)}' is a directory, not an export file", style="red")
        )
        raise typer.Exit(1)

    output_path = Path(output) if output else Path("traces.jsonl")
    if not is_under_cwd(output_path):
        console.print(
            Text(f"--output '{strip_control(output_path)}' is outside cwd — refusing", style="red")
        )
        raise typer.Exit(1)

    # Refuse before anything is opened for writing: the default --output
    # (traces.jsonl) is also a plausible --logs, and the write would empty it.
    refuse_output_over_input(
        [("--logs", str(logs_path))],
        [("--output", str(output_path))],
        console=console,
    )

    _print_pii_reminder(canonical)

    auth_value = resolve_auth_env(canonical)
    if auth_value is None:
        console.print(
            "[dim]No auth env var set — this CLI parses the local export "
            "only (no SaaS pull).[/]"
        )

    count = 0

    def _lines():
        nonlocal count
        try:
            for record in ingest_traces(source=canonical, path=str(logs_path)):
                count += 1
                yield json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
        except UnicodeDecodeError as exc:
            raise _LogsUnreadableError(
                f"--logs '{logs}' is not valid UTF-8 (byte {exc.start})"
            ) from exc
        except OSError as exc:
            reason = exc.strerror or type(exc).__name__
            raise _LogsUnreadableError(f"--logs '{logs}' cannot be read: {reason}") from exc

    # Streamed into a staging file; the target is replaced only after the export
    # has been read completely, so a failed read leaves an earlier output intact.
    try:
        atomic_write_lines(_lines(), str(output_path), field="--output")
    except _LogsUnreadableError as exc:
        console.print(Text(strip_control(exc), style="red"))
        raise typer.Exit(1) from exc

    console.print(
        Text(
            f"Wrote {count} traces from {canonical} -> {strip_control(output_path.name)}",
            style="green",
        )
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
    """``--pull``: fetch from the vendor's API and write the rows parsed (#204)."""
    from soup_cli.utils.paths import atomic_write_lines, enforce_under_cwd_and_no_symlink

    if logs is not None:
        console.print(
            "[red]--pull and --logs are mutually exclusive: fetch live or read an export.[/]"
        )
        raise typer.Exit(2)
    if canonical not in ("langfuse", "langsmith"):
        console.print(
            f"[red]--pull is only available for --source langfuse and langsmith so far; "
            f"export {escape(canonical)} traces to JSONL and pass --logs instead.[/]"
        )
        raise typer.Exit(2)

    output_path = Path(output) if output else Path("traces.jsonl")
    try:
        enforce_under_cwd_and_no_symlink(str(output_path), "--output")
    except (TypeError, ValueError) as exc:
        _refuse(exc)
        raise typer.Exit(1) from exc

    from soup_cli.utils import ingest_pull

    since_text = since if since is not None else _DEFAULT_SINCE
    try:
        window = ingest_pull.parse_since(since_text)
    except (TypeError, ValueError) as exc:
        _refuse(exc)
        raise typer.Exit(2) from exc
    pages = max_pages if max_pages is not None else ingest_pull.DEFAULT_MAX_PAGES
    if canonical == "langsmith":
        _pull_langsmith(
            window,
            since_text=since_text,
            max_pages=pages,
            output_path=output_path,
            allow_private_host=allow_private_host,
            slack_url=slack_url,
            discord_url=discord_url,
        )
        return

    try:
        credentials = ingest_pull.load_langfuse_credentials(
            os.environ, allow_private_host=allow_private_host
        )
    except ingest_pull.PullError as exc:
        _refuse(exc)
        raise typer.Exit(1) from None
    except ValueError as exc:
        _refuse(exc)
        raise typer.Exit(2) from None

    _print_pii_reminder(canonical)
    console.print(
        terminal_text(
            f"Pulling GENERATION observations from {credentials.host} "
            f"for the last {since_text}...",
            "dim",
        )
    )

    stats = ingest_pull.PullStats()
    written = 0

    def _lines():
        nonlocal written
        generations = ingest_pull.pull_langfuse_generations(
            credentials,
            since=window,
            max_pages=pages,
            stats=stats,
        )
        for record in _ingest_sources.parse_langfuse(generations):
            written += 1
            yield json.dumps(record.to_dict(), ensure_ascii=False) + "\n"

    try:
        # Streams to a staging file; the target only appears once the pull finished.
        atomic_write_lines(_lines(), str(output_path), field="--output")
    except ingest_pull.PullError as exc:
        _refuse(exc)
        _refuse(f"Nothing was written to {output_path.name}.")
        raise typer.Exit(1) from None

    console.print(
        terminal_text(
            f"Wrote {written} traces from langfuse ({stats.generations} generations "
            f"pulled) -> {output_path.name}",
            "green",
        )
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


def _pull_langsmith(
    window,
    *,
    since_text: str,
    max_pages: int,
    output_path: Path,
    allow_private_host: bool,
    slack_url: Optional[str],
    discord_url: Optional[str],
) -> None:
    """``--source langsmith --pull``: fetch the project's LLM runs and write them parsed."""
    from soup_cli.utils import ingest_pull_langsmith
    from soup_cli.utils.ingest_pull import PullError

    try:
        settings = ingest_pull_langsmith.load_langsmith_settings(
            os.environ, allow_private_host=allow_private_host
        )
    except PullError as exc:
        console.print(terminal_text(str(exc), "red"))
        raise typer.Exit(1) from None
    except ValueError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from None

    _print_pii_reminder("langsmith")
    console.print(
        Text(
            f"Pulling LLM runs of project {strip_control(settings.project)!r} from "
            f"{strip_control(settings.host)} for the last {strip_control(since_text)}...",
            style="dim",
        )
    )

    stats = ingest_pull_langsmith.LangSmithPullStats()
    written = 0

    def _lines():
        nonlocal written
        runs = ingest_pull_langsmith.pull_langsmith_runs(
            settings, since=window, max_pages=max_pages, stats=stats
        )
        for record in _ingest_sources.parse_langsmith(runs):
            written += 1
            yield json.dumps(record.to_dict(), ensure_ascii=False) + "\n"

    try:
        # Streams to a staging file; the target only appears once the pull finished.
        atomic_write_lines(_lines(), str(output_path), field="--output")
    except PullError as exc:
        console.print(terminal_text(str(exc), "red"))
        console.print(f"[red]Nothing was written to {escape(output_path.name)}.[/]")
        raise typer.Exit(1) from None

    console.print(
        f"[green]Wrote {written} traces from langsmith ({stats.llm_runs} LLM runs "
        f"pulled) -> {escape(output_path.name)}[/]"
    )
    if stats.llm_runs > written:
        console.print(
            f"[yellow]{stats.llm_runs - written} LLM run(s) had no input or no output "
            "and were skipped.[/]"
        )
    if stats.skipped_not_llm:
        console.print(
            f"[yellow]{stats.skipped_not_llm} run(s) were not LLM runs and were skipped.[/]"
        )

    emit_webhooks(
        slack_url,
        discord_url,
        payload={
            "command": "ingest",
            "source": "langsmith",
            "traces_written": written,
            "auth_env_set": True,
            "llm_runs_pulled": stats.llm_runs,
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
    label = " + ".join(_ingest_sources._AUTH_ENV.get(source, ("(unset)",)))
    settings = _ingest_sources._PULL_SETTINGS_ENV.get(source)
    if settings:
        label += f" (--pull also reads {', '.join(settings)})"
    return label


__all__ = ["ingest", "SUPPORTED_INGEST_SOURCES"]
