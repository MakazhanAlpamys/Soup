"""v0.44.0 Part D — `soup delinearize-llama4` command (schema stub)."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from soup_cli.utils.delinearize_llama4 import is_llama4_model, plan_delinearize

console = Console()


def delinearize_llama4(
    source_dir: str = typer.Argument(
        ...,
        help="Llama 4 checkpoint directory (containing *.safetensors).",
    ),
    target_dir: str = typer.Option(
        ...,
        "--target",
        "-o",
        help="Destination directory for the delinearized weights (under cwd).",
    ),
    model_id: str = typer.Option(
        None,
        "--model-id",
        help="Optional model id; warn if it doesn't look like a Llama 4 model.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Acknowledge that live runtime lands in v0.44.1 (plan-only now).",
    ),
) -> None:
    """Plan Llama 4 expert-weight delinearization for export.

    v0.44.0 ships the planner; live torch-side reshape lands in v0.44.1.
    """
    if model_id is not None and not is_llama4_model(model_id):
        console.print(
            f"[yellow]model id {escape(model_id)} doesn't match the Llama 4 "
            "naming pattern - proceed only if you're sure.[/]"
        )
    try:
        plan = plan_delinearize(source_dir, target_dir)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(code=2) from exc
    body = (
        f"Weight files:    {len(plan.weight_files)}\n"
        f"Source dir:      {escape(plan.source_dir)}\n"
        f"Target dir:      {escape(plan.target_dir)}\n\n"
        "Live delinearization runtime lands in v0.44.1 — plan-only for now."
    )
    console.print(
        Panel(body, title="Llama 4 Delinearization Plan", border_style="cyan")
    )
    if not yes:
        console.print(
            "[yellow]Pass --yes to acknowledge the deferred runtime.[/]"
        )
        raise typer.Exit(code=0)
