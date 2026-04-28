"""soup why — explain anomalies in the most recent (or named) training run."""

from __future__ import annotations

import json
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

console = Console()

_SEVERITY_COLOR = {"critical": "red", "warning": "yellow", "info": "cyan"}


def why(
    run_id: Optional[str] = typer.Argument(
        None,
        help="Run ID (or prefix). Defaults to the most recent run.",
    ),
):
    """Explain training anomalies for a run in plain English.

    Heuristic — looks at the SQLite-stored metric series and config and
    surfaces common pathologies (NaN loss, plateaus, divergence, bad LR).
    """
    from soup_cli.experiment.tracker import ExperimentTracker
    from soup_cli.utils.why import diagnose

    tracker = ExperimentTracker()
    if run_id is None:
        runs = tracker.list_runs(limit=1)
        if not runs:
            console.print("[yellow]No runs found yet. Train a model first.[/]")
            raise typer.Exit(1)
        target = runs[0]
    else:
        target = tracker.get_run(run_id)
        if target is None:
            console.print(f"[red]Run not found:[/] {escape(run_id)}")
            raise typer.Exit(1)

    metrics = tracker.get_metrics(target["run_id"])
    config = None
    raw_config = target.get("config_json")
    if raw_config:
        try:
            config = json.loads(raw_config)
        except (json.JSONDecodeError, TypeError):
            config = None

    findings = diagnose(metrics, config)

    header = (
        f"Run: [bold]{escape(target['run_id'])}[/]\n"
        f"Status: {target.get('status') or '-'} | "
        f"Steps: {target.get('total_steps') or '-'} | "
        f"Loss: {target.get('initial_loss') or '-'} → {target.get('final_loss') or '-'}"
    )
    console.print(Panel(header, title="soup why"))

    if not findings:
        console.print(
            "[green]No anomalies detected.[/] Loss curve, gradient norms, and "
            "learning rate are within typical ranges for this run."
        )
        return

    table = Table(title="Findings", show_lines=True)
    table.add_column("Severity", style="bold", no_wrap=True)
    table.add_column("Category", no_wrap=True)
    table.add_column("Diagnosis")
    table.add_column("Suggestion")

    for finding in findings:
        color = _SEVERITY_COLOR.get(finding.severity, "white")
        table.add_row(
            f"[{color}]{finding.severity}[/]",
            escape(finding.category),
            escape(finding.message),
            escape(finding.suggestion),
        )
    console.print(table)
