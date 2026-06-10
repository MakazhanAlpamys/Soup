"""`soup delinearize-llama4` — Llama 4 expert-weight reshape.

v0.44.0 shipped the planner; v0.71.21 (#97) lifts the live torch runtime
(load each shard → reshape fused 2-D expert weights to 3-D → atomic write
to ``--target``). ``--plan-only`` keeps the old render-and-exit flow.
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from soup_cli.utils.delinearize_llama4 import (
    is_llama4_model,
    plan_delinearize,
    run_delinearize,
)

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
    num_experts: Optional[int] = typer.Option(
        None,
        "--num-experts",
        help=(
            "Expert count for the reshape. Defaults to "
            "config.json's num_local_experts."
        ),
    ),
    plan_only: bool = typer.Option(
        False,
        "--plan-only",
        help="Render the plan and exit without reshaping (v0.44.0 flow).",
    ),
) -> None:
    """Delinearize Llama 4 fused expert weights to 3-D form.

    Live since v0.71.21 (#97): reshapes ``...experts.gate_up_proj`` /
    ``...experts.down_proj`` from ``[E*dim_in, dim_out]`` to
    ``[E, dim_in, dim_out]`` and copies JSON sidecars so the target stays
    a loadable checkpoint.
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
    console.print(
        Panel(
            f"Weight files:    {len(plan.weight_files)}\n"
            f"Source dir:      {escape(plan.source_dir)}\n"
            f"Target dir:      {escape(plan.target_dir)}",
            title="Llama 4 Delinearization Plan",
            border_style="cyan",
        )
    )
    if plan_only:
        return

    try:
        result = run_delinearize(plan, num_experts=num_experts)
    except (TypeError, ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(code=2) from exc
    except ImportError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(code=1) from exc

    console.print(
        Panel(
            f"Files written:     {len(result.files_written)}\n"
            f"Reshaped keys:     {result.reshaped_keys}\n"
            f"Already 3-D:       {result.already_3d_keys}\n"
            f"Passthrough keys:  {result.passthrough_keys}\n"
            f"Sidecars copied:   {result.sidecars_copied}\n"
            f"Target dir:        {escape(result.target_dir)}",
            title="Llama 4 Delinearization — done",
            border_style="green",
        )
    )
