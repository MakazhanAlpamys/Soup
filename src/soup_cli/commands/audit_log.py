"""soup audit-log — HIPAA/SOC2 JSONL audit log CLI (v0.59.0 Part D)."""

from __future__ import annotations

import json
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from soup_cli.utils.audit_log import default_log_path, read_audit_tail, rotate_if_needed

console = Console()

app = typer.Typer(
    no_args_is_help=True,
    help="HIPAA/SOC2-shaped JSONL audit log (v0.59.0).",
)


@app.command("tail")
def tail_cmd(
    limit: int = typer.Option(50, "--limit", help="Max records to show (1-100000)."),
    path: Optional[str] = typer.Option(
        None, "--path",
        help="Override audit log path (default: $SOUP_AUDIT_LOG_PATH "
             "or ~/.soup/audit.jsonl).",
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Emit raw JSONL instead of a table.",
    ),
) -> None:
    """Show the most recent audit records."""
    try:
        records = read_audit_tail(path, limit=limit)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]Invalid arguments: {escape(str(exc))}[/]")
        raise typer.Exit(2)
    if json_out:
        for r in records:
            console.print(json.dumps(r))
        return
    if not records:
        console.print("[dim]No audit records.[/]")
        return
    table = Table(title="soup audit-log tail")
    for header in ("timestamp", "command", "exit", "operator", "host"):
        table.add_column(header)
    for r in records:
        table.add_row(
            escape(str(r.get("timestamp", ""))),
            escape(str(r.get("command", ""))),
            escape(str(r.get("exit_code", ""))),
            escape(str(r.get("operator_id", ""))),
            escape(str(r.get("host_id", ""))),
        )
    console.print(table)


@app.command("rotate")
def rotate_cmd(
    path: Optional[str] = typer.Option(
        None, "--path", help="Override audit log path.",
    ),
    cap_mb: int = typer.Option(
        100, "--cap-mb", min=1, max=10000, help="Rotation cap in MiB.",
    ),
) -> None:
    """Force a rotation pass at the current cap."""
    target = path if path is not None else default_log_path()
    try:
        rotated = rotate_if_needed(target, cap_bytes=cap_mb * 1024 * 1024)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]Rotate failed: {escape(str(exc))}[/]")
        raise typer.Exit(2)
    if rotated:
        console.print(f"[green]Rotated {escape(target)} -> {escape(target)}.1[/]")
    else:
        console.print("[dim]No rotation needed.[/]")
