"""``soup distill-prompt`` CLI — v0.68.0 Part B.

Distill prompt-heavy traces into a small FT plan.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from soup_cli.utils.prompt_distill import (
    SUPPORTED_DISTILL_STRATEGIES,
    build_distill_prompt_plan,
    prepare_distill_dataset,
)

console = Console()


def distill_prompt_cmd(
    traces: str = typer.Option(..., "--traces", help="Path to traces JSONL"),
    teacher: str = typer.Option(..., "--teacher", help="Teacher model id"),
    student: str = typer.Option(..., "--student", help="Student model id"),
    strategy: str = typer.Option(
        "sft",
        "--strategy",
        help="Distill strategy. Allowed: "
        + ", ".join(sorted(SUPPORTED_DISTILL_STRATEGIES)),
    ),
    output: str = typer.Option(
        "distilled.jsonl", "--output", "-o", help="Output JSONL path"
    ),
    plan_only: bool = typer.Option(
        False, "--plan-only", help="Render plan + exit 0 (no live preparation)."
    ),
) -> None:
    """Distill prompt-heavy traces into a small FT plan."""
    try:
        plan = build_distill_prompt_plan(
            traces_path=traces,
            teacher=teacher,
            student=student,
            strategy=strategy,
            output_path=output,
        )
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    console.print(
        Panel(
            f"Traces:    [bold]{escape(plan.traces_path)}[/]\n"
            f"Teacher:   [bold]{escape(plan.teacher)}[/]\n"
            f"Student:   [bold]{escape(plan.student)}[/]\n"
            f"Strategy:  [bold]{escape(plan.strategy)}[/]\n"
            f"Output:    [bold]{escape(plan.output_path)}[/]",
            title="soup distill-prompt — plan",
        )
    )

    if plan_only:
        return

    try:
        prepare_distill_dataset(plan)
    except NotImplementedError as exc:
        console.print(
            Panel(
                f"[yellow]{escape(str(exc))}[/]",
                title="Live distill-prompt deferred",
            )
        )
        raise typer.Exit(3) from exc
