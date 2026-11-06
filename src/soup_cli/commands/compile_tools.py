"""``soup compile-tools`` CLI — v0.68.0 Part C.

Generate tool schemas + descriptions optimized via textual gradients.
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from soup_cli.utils.compile_tools import (
    SUPPORTED_TOOL_OPTIMIZERS,
    build_tool_compile_plan,
    run_tool_compile,
)
from soup_cli.utils.terminal import for_terminal

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
    task_lm: Optional[str] = typer.Option(
        None, "--task-lm", help="Task language model for GEPA optimizer"
    ),
    reflection_lm: Optional[str] = typer.Option(
        None, "--reflection-lm", help="Reflection language model for GEPA optimizer"
    ),
    engine: Optional[str] = typer.Option(
        None, "--engine", help="Backward engine for TextGrad optimizer (e.g. gpt-4o)"
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
            task_lm=task_lm,
            reflection_lm=reflection_lm,
            engine=engine,
        )
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{for_terminal(exc)}[/]", emoji=False)
        raise typer.Exit(2) from exc

    plan_details = [
        f"Spec:      [bold]{for_terminal(plan.spec_path)}[/]",
        f"Eval:      [bold]{for_terminal(plan.eval_suite_path)}[/]",
        f"Optimizer: [bold]{escape(plan.optimizer)}[/]",
        f"Output:    [bold]{for_terminal(plan.output_path)}[/]",
    ]
    if plan.task_lm:
        plan_details.append(f"Task LM:     [bold]{for_terminal(plan.task_lm)}[/]")
    if plan.reflection_lm:
        plan_details.append(f"Reflect LM:  [bold]{for_terminal(plan.reflection_lm)}[/]")
    if plan.engine:
        plan_details.append(f"Engine:      [bold]{for_terminal(plan.engine)}[/]")

    console.print(
        Panel(
            Text.from_markup("\n".join(plan_details), emoji=False),
            title="soup compile-tools — plan",
        )
    )

    if plan_only:
        return

    try:
        n = run_tool_compile(plan)
    except ImportError as exc:
        console.print(
            Panel(
                Text.from_markup(f"[yellow]{for_terminal(exc)}[/]", emoji=False),
                title="compile-tools — missing dependency",
            )
        )
        raise typer.Exit(2) from exc
    except (TypeError, ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{for_terminal(exc)}[/]", emoji=False)
        raise typer.Exit(2) from exc
    except Exception as exc:
        console.print(f"[red]{for_terminal(exc)}[/]", emoji=False)
        raise typer.Exit(2) from exc

    console.print(
        Panel(
            Text.from_markup(
                f"Tools:  [bold]{n}[/]\n"
                f"Output: [bold]{for_terminal(plan.output_path)}[/]",
                emoji=False,
            ),
            title="soup compile-tools — done",
        )
    )
