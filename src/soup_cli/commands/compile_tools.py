"""``soup compile-tools`` CLI — v0.68.0 Part C.

Generate tool schemas + descriptions optimized via textual gradients.
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from soup_cli.utils.compile_tools import (
    SUPPORTED_TOOL_OPTIMIZERS,
    build_tool_compile_plan,
    run_tool_compile,
)
from soup_cli.utils.terminal import strip_control

console = Console()


def _detail(label: str, value: object) -> Text:
    # A Text is not parsed as markup and has no emoji codes, so the value is shown as typed.
    return Text.assemble(label, (strip_control(value), "bold"))


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
        console.print(Text(strip_control(exc), style="red"))
        raise typer.Exit(2) from exc

    plan_details = [
        _detail("Spec:      ", plan.spec_path),
        _detail("Eval:      ", plan.eval_suite_path),
        _detail("Optimizer: ", plan.optimizer),
        _detail("Output:    ", plan.output_path),
    ]
    if plan.task_lm:
        plan_details.append(_detail("Task LM:     ", plan.task_lm))
    if plan.reflection_lm:
        plan_details.append(_detail("Reflect LM:  ", plan.reflection_lm))
    if plan.engine:
        plan_details.append(_detail("Engine:      ", plan.engine))

    console.print(
        Panel(
            Text("\n").join(plan_details),
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
                Text(strip_control(exc), style="yellow"),
                title="compile-tools — missing dependency",
            )
        )
        raise typer.Exit(2) from exc
    except (TypeError, ValueError, FileNotFoundError) as exc:
        console.print(Text(strip_control(exc), style="red"))
        raise typer.Exit(2) from exc
    except Exception as exc:
        console.print(Text(strip_control(exc), style="red"))
        raise typer.Exit(2) from exc

    console.print(
        Panel(
            Text.assemble(
                "Tools:  ",
                (str(n), "bold"),
                "\nOutput: ",
                (strip_control(plan.output_path), "bold"),
            ),
            title="soup compile-tools — done",
        )
    )
