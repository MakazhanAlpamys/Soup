"""soup iterative-dpo — Iterative DPO loop driver — v0.70.0 Part E.

Sample → RM-score → re-pair → retrain over N rounds. The live runner
is deferred to v0.70.1; v0.70.0 ships the schema + ``--plan-only``
renderer that prints the canonical per-round artifacts.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

console = Console()

app = typer.Typer(
    no_args_is_help=True,
    help="Iterative DPO loop driver (v0.70.0 Part E)",
)


@app.callback(invoke_without_command=True)
def main(
    base_model: str = typer.Option(..., "--base-model", help="HF id / local path"),
    reward_model: str = typer.Option(..., "--reward-model", help="RM HF id / path"),
    prompts: str = typer.Option(..., "--prompts", help="Source prompts JSONL"),
    output_dir: str = typer.Option(..., "--output-dir", help="Output dir"),
    rounds: int = typer.Option(3, "--rounds", help="Number of iterative-DPO rounds"),
    pairs_per_round: int = typer.Option(
        500,
        "--pairs-per-round",
        help="Number of (chosen, rejected) pairs per round",
    ),
    plan_only: bool = typer.Option(
        False,
        "--plan-only",
        help="Print the resolved plan and exit (no training).",
    ),
):
    """Render the iterative-DPO plan and (in v0.70.1) execute it."""
    from soup_cli.utils.iterative_dpo import (
        build_iterative_dpo_plan,
        run_iterative_dpo,
    )

    try:
        plan = build_iterative_dpo_plan(
            base_model=base_model,
            reward_model=reward_model,
            prompts_path=prompts,
            output_dir=output_dir,
            rounds=rounds,
            pairs_per_round=pairs_per_round,
        )
    except (ValueError, TypeError) as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=2) from exc

    table = Table(title="Iterative-DPO plan")
    table.add_column("Round")
    table.add_column("Pairs path")
    table.add_column("Adapter path")
    table.add_column("Pairs")
    for r in plan.rounds:
        table.add_row(
            str(r.round_index),
            escape(r.pairs_path),
            escape(r.adapter_path),
            str(r.pairs_count),
        )
    console.print(table)

    if plan_only:
        console.print(
            Panel(
                "[green]Plan rendered[/green] (--plan-only). To execute, "
                "drop the flag once v0.70.1 ships.",
                title="Iterative DPO",
            )
        )
        raise typer.Exit(code=0)

    try:
        run_iterative_dpo(plan)
    except NotImplementedError as exc:
        console.print(
            Panel(
                "[yellow]Live runner deferred to v0.70.1.[/yellow] "
                f"{escape(str(exc))}",
                title="Iterative DPO",
            )
        )
        raise typer.Exit(code=3) from exc
