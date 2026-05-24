"""``soup compile-tools`` CLI — v0.68.0 Part C.

Generate tool schemas + descriptions optimized via textual gradients.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from soup_cli.utils.compile_tools import (
    SUPPORTED_TOOL_OPTIMIZERS,
    build_tool_compile_plan,
    run_tool_compile,
)

console = Console()


def compile_tools_cmd(
    spec: str = typer.Argument(..., help="OpenAPI / MCP / GraphQL spec path"),
    eval_suite: str = typer.Option(
        ..., "--eval", help="Path to tool-call eval JSONL"
    ),
    optimizer: str = typer.Option(
        "textgrad",
        "--optimizer",
        help="Allowed: " + ", ".join(sorted(SUPPORTED_TOOL_OPTIMIZERS)),
    ),
    output: str = typer.Option(
        "compiled_tools.json", "--output", "-o", help="Output schema path"
    ),
    plan_only: bool = typer.Option(
        False, "--plan-only", help="Render plan + exit 0 (no live optimise)."
    ),
) -> None:
    """Compile / optimise tool schemas via textual-gradient methods."""
    try:
        plan = build_tool_compile_plan(
            spec_path=spec,
            eval_suite_path=eval_suite,
            optimizer=optimizer,
            output_path=output,
        )
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    console.print(
        Panel(
            f"Spec:      [bold]{escape(plan.spec_path)}[/]\n"
            f"Eval:      [bold]{escape(plan.eval_suite_path)}[/]\n"
            f"Optimizer: [bold]{escape(plan.optimizer)}[/]\n"
            f"Output:    [bold]{escape(plan.output_path)}[/]",
            title="soup compile-tools — plan",
        )
    )

    if plan_only:
        return

    try:
        run_tool_compile(plan)
    except NotImplementedError as exc:
        console.print(
            Panel(
                f"[yellow]{escape(str(exc))}[/]",
                title="Live compile-tools deferred",
            )
        )
        raise typer.Exit(3) from exc
