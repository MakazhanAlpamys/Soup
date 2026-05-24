"""``soup compile`` — DSPy / GEPA prompt-program compiler CLI (v0.68.0 Part A).

Renders a ``CompilePlan`` panel and, when ``--plan-only`` is omitted,
invokes the deferred-live ``run_compile`` (raises NotImplementedError
with a v0.68.1 marker). Mirrors v0.61.0 / v0.62.0 stub-then-live CLI policy.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from soup_cli.utils.paths import atomic_write_text
from soup_cli.utils.prompt_compile import (
    SUPPORTED_PROMPT_OPTIMIZERS,
    build_compile_plan,
    run_compile,
)

console = Console()


def compile_cmd(
    program: str = typer.Argument(..., help="Path to DSPy / GEPA program (.py)"),
    eval_suite: str = typer.Option(
        ..., "--eval", help="Path to eval-suite JSON / JSONL file"
    ),
    optimizer: str = typer.Option(
        "mipro",
        "--optimizer",
        help=(
            "Optimizer name. Allowed: "
            + ", ".join(sorted(SUPPORTED_PROMPT_OPTIMIZERS))
        ),
    ),
    max_iters: int = typer.Option(10, "--max-iters", help="Maximum optimizer iterations"),
    output: str = typer.Option(
        "compiled_program.py", "--output", "-o", help="Output program path"
    ),
    plan_only: bool = typer.Option(
        False,
        "--plan-only",
        help="Render the resolved plan + exit 0 (no live compile).",
    ),
) -> None:
    """Compile a DSPy / GEPA prompt program against an eval suite."""
    try:
        plan = build_compile_plan(
            program_path=program,
            eval_suite_path=eval_suite,
            optimizer=optimizer,
            max_iters=max_iters,
            output_path=output,
        )
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    console.print(
        Panel(
            f"Program:     [bold]{escape(plan.program_path)}[/]\n"
            f"Eval suite:  [bold]{escape(plan.eval_suite_path)}[/]\n"
            f"Optimizer:   [bold]{escape(plan.optimizer)}[/]\n"
            f"Max iters:   [bold]{plan.max_iters}[/]\n"
            f"Output:      [bold]{escape(plan.output_path)}[/]",
            title="soup compile — plan",
        )
    )

    if plan_only:
        return

    try:
        result = run_compile(plan)
    except NotImplementedError as exc:
        console.print(
            Panel(
                f"[yellow]{escape(str(exc))}[/]",
                title="Live compile deferred",
            )
        )
        raise typer.Exit(3) from exc
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    atomic_write_text(result.program_text, plan.output_path, field="output_path")
    console.print(
        Panel(
            f"Output:      [bold]{escape(plan.output_path)}[/]\n"
            f"Score:       [bold]{result.score:.3f}[/]\n"
            f"Iterations:  [bold]{result.iterations}[/]\n"
            f"Converged:   [bold]{result.converged}[/]",
            title="soup compile — done",
        )
    )
