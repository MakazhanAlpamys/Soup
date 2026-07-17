"""`soup merge-sharded-fsdp-weights` command.

v0.44.0 shipped the planner; v0.71.14 (#96) lifts the live torch-side
consolidation: by default the command loads each FSDP shard and writes a single
consolidated ``.safetensors``. Pass ``--plan-only`` to print the plan without
writing anything (single-process — no multi-GPU needed to MERGE).
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from soup_cli.utils.fsdp_consolidate import consolidate_shards, plan_consolidation

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
    plan_only: bool = typer.Option(
        False,
        "--plan-only",
        help="Print the consolidation plan and exit without writing anything.",
    ),
) -> None:
    """Consolidate FSDP shard files into a single safetensors file.

    Streams each ``pytorch_model_fsdp_*.bin`` shard (loaded via
    ``torch.load(weights_only=True)`` — no arbitrary pickle exec), unions the
    per-shard parameter fragments, and writes one ``.safetensors`` atomically.
    """
    try:
        plan = plan_consolidation(shard_dir, output)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(code=2) from exc

    if plan_only:
        body = (
            f"Shards found:    {len(plan.shard_files)}\n"
            f"Source dir:      {escape(plan.shard_dir)}\n"
            f"Output (target): {escape(plan.output_path)}\n\n"
            "Plan-only run — pass without --plan-only to write the file."
        )
        console.print(
            Panel(body, title="FSDP Consolidation Plan", border_style="cyan")
        )
        raise typer.Exit(code=0)

    console.print(
        f"[dim]Consolidating {len(plan.shard_files)} shard(s)...[/]"
    )
    try:
        result = consolidate_shards(plan)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(code=2) from exc
    except ImportError as exc:
        console.print(
            "[red]torch + safetensors are required for consolidation. "
            "Install with: [bold]pip install \"soup-cli\\[train]\"[/][/]"
        )
        raise typer.Exit(code=1) from exc

    body = (
        f"Tensors written: {result.num_tensors}\n"
        f"Shards merged:   {result.num_shards}\n"
        f"Size:            {result.total_bytes / 1e6:.2f} MB\n"
        f"Output:          {escape(result.output_path)}"
    )
    console.print(
        Panel(body, title="[bold green]FSDP Consolidation Done[/]", border_style="green")
    )
