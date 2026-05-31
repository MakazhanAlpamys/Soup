"""v0.44.0 Part D — `soup merge-sharded-fsdp-weights` command.

Schema-only stub: discovers + validates FSDP shards and prints the planned
operation. Live consolidation lands in v0.44.1.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from soup_cli.utils.fsdp_consolidate import plan_consolidation

console = Console()


def merge_sharded_fsdp_weights(
    shard_dir: str = typer.Argument(
        ...,
        help="Directory containing pytorch_model_fsdp_*.bin shard files.",
    ),
    output: str = typer.Option(
        ...,
        "--output",
        "-o",
        help="Destination .safetensors file path (under cwd).",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Acknowledge that live consolidation lands in v0.44.1 (plan-only now).",
    ),
) -> None:
    """Plan a consolidation of FSDP shard files into a single safetensors file.

    v0.44.0 ships the planner; live torch-side consolidation lands in v0.44.1.
    """
    try:
        plan = plan_consolidation(shard_dir, output)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(code=2) from exc
    body = (
        f"Shards found:    {len(plan.shard_files)}\n"
        f"Source dir:      {escape(plan.shard_dir)}\n"
        f"Output (target): {escape(plan.output_path)}\n\n"
        "Live consolidation runtime lands in v0.44.1 — this is a plan-only run."
    )
    console.print(Panel(body, title="FSDP Consolidation Plan", border_style="cyan"))
    if not yes:
        console.print(
            "[yellow]Pass --yes to acknowledge the deferred runtime "
            "and exit cleanly.[/]"
        )
        raise typer.Exit(code=0)
