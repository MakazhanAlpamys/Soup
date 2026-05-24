"""``soup apple-adapter`` CLI — v0.68.0 Part D.

HF / PEFT <-> MLX <-> Apple Foundation Models adapter conversion + signing.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from soup_cli.utils.apple_adapter import (
    SUPPORTED_ADAPTER_DIRECTIONS,
    build_apple_adapter_plan,
    convert_apple_adapter,
)

console = Console()


def apple_adapter_cmd(
    source: str = typer.Argument(..., help="Source adapter directory"),
    direction: str = typer.Option(
        "hf-to-mlx",
        "--direction",
        help="Allowed: " + ", ".join(sorted(SUPPORTED_ADAPTER_DIRECTIONS)),
    ),
    output: str = typer.Option(..., "--output", "-o", help="Output directory"),
    sign: bool = typer.Option(
        False,
        "--sign/--no-sign",
        help="Sign converted adapter via v0.60 Merkle-root signing.",
    ),
    plan_only: bool = typer.Option(
        False, "--plan-only", help="Render plan + exit 0 (no live conversion)."
    ),
) -> None:
    """Convert / sign HF / MLX / Apple FoundationModels adapters."""
    try:
        plan = build_apple_adapter_plan(
            source_dir=source,
            output_dir=output,
            direction=direction,
            sign=sign,
        )
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    console.print(
        Panel(
            f"Source:    [bold]{escape(plan.source_dir)}[/]\n"
            f"Output:    [bold]{escape(plan.output_dir)}[/]\n"
            f"Direction: [bold]{escape(plan.direction)}[/]\n"
            f"Sign:      [bold]{plan.sign}[/]",
            title="soup apple-adapter — plan",
        )
    )

    if plan_only:
        return

    try:
        convert_apple_adapter(plan)
    except NotImplementedError as exc:
        console.print(
            Panel(
                f"[yellow]{escape(str(exc))}[/]",
                title="Live apple-adapter deferred",
            )
        )
        raise typer.Exit(3) from exc
