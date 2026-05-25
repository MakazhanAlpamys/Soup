"""soup build — dbt-for-SFT DAG of dataset transforms (v0.69.0 Part A).

Reads a YAML manifest, validates the model DAG, prints the topological plan,
and (with ``--dry-run``) exits cleanly. The live runner that actually
materialises each model is deferred to v0.69.1 — invoking ``soup build`` WITHOUT
``--dry-run`` exits with code 3 and a deferred-live advisory (matches v0.61.0 /
v0.62.0 / v0.68.0 distinct-exit-code-for-deferred policy).
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

console = Console()


def build_cmd(
    config: str = typer.Argument(..., help="Path to build manifest (YAML)"),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate the manifest + print the topological plan; do not execute.",
    ),
) -> None:
    """Validate a build DAG and (optionally) execute it.

    The live materialiser is deferred to v0.69.1; today only ``--dry-run``
    produces a meaningful result.
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

    # Live runner deferred to v0.69.1 — same distinct-exit-3 policy as
    # v0.61.0 / v0.62.0 / v0.68.0 deferred-live stubs.
    try:
        run_build(plan)
    except NotImplementedError as exc:
        console.print(
            Panel(
                f"[yellow]{escape(str(exc))}[/]",
                title="Live build deferred",
            )
        )
        raise typer.Exit(3) from exc
