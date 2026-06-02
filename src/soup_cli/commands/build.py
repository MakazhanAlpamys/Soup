"""soup build — dbt-for-SFT DAG of dataset transforms (v0.69.0 Part A).

Reads a YAML manifest, validates the model DAG, prints the topological plan,
and (with ``--dry-run``) exits cleanly. Without ``--dry-run`` the live runner
(v0.71.6 #231) materialises each model in topological order — ``table`` /
``view`` / ``incremental`` kinds with a SQLite state store that re-runs the
transform on only the added/changed rows.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

console = Console()


def build_cmd(
    config: str = typer.Argument(..., help="Path to build manifest (YAML)"),
    output_dir: str = typer.Option(
        "build_out",
        "--output-dir",
        "-o",
        help="Directory (under cwd) for materialised model JSONL + state store.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate the manifest + print the topological plan; do not execute.",
    ),
) -> None:
    """Validate a build DAG and (optionally) materialise it.

    ``--dry-run`` only validates + prints the topological plan. Otherwise the
    DAG is materialised under ``--output-dir`` (live as of v0.71.6 #231).
    """
    from soup_cli.utils.build_dag import (
        load_build_yaml,
        render_plan_table,
        run_build,
    )

    try:
        plan = load_build_yaml(config)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    rendered = render_plan_table(plan)
    console.print(
        Panel(
            escape(rendered),
            title=f"soup build — {escape(config)}",
        )
    )

    if dry_run:
        return

    try:
        result = run_build(plan, output_dir=output_dir)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    table = Table(title=f"Materialised — {escape(result.output_dir)}")
    table.add_column("Model", style="bold")
    table.add_column("Kind")
    table.add_column("Rows in", justify="right")
    table.add_column("Rows out", justify="right")
    table.add_column("Transformed", justify="right")
    table.add_column("Diff")
    for model in result.models:
        if model.diff is not None:
            diff_str = (
                f"+{model.diff.added} ~{model.diff.changed} "
                f"-{model.diff.removed} ={model.diff.unchanged}"
            )
        else:
            diff_str = "—"
        table.add_row(
            escape(model.name),
            escape(model.kind),
            str(model.rows_in),
            str(model.rows_out),
            str(model.transform_calls),
            diff_str,
        )
    console.print(table)
