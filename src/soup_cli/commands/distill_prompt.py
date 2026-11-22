"""``soup distill-prompt`` CLI — v0.68.0 Part B.

Distill prompt-heavy traces into a small FT plan.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from soup_cli.utils.prompt_distill import (
    SUPPORTED_DISTILL_STRATEGIES,
    NoTracePromptsError,
    TraceRowStats,
    build_distill_prompt_plan,
    prepare_distill_dataset,
)
from soup_cli.utils.terminal import strip_control

console = Console()


def _detail(label: str, value: object) -> Text:
    # A Text is not parsed as markup and has no emoji codes, so the value is shown as typed.
    return Text.assemble(label, (strip_control(value), "bold"))


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
    provider: str = typer.Option(
        "ollama",
        "--provider",
        help="Teacher/student provider: ollama / anthropic / vllm.",
    ),
    base_url: str = typer.Option(
        None, "--base-url", help="Provider base URL (ollama / vllm)."
    ),
    temperature: float = typer.Option(
        0.0, "--temperature", help="Sampling temperature for teacher/student."
    ),
    max_rows: int = typer.Option(
        None, "--max-rows", help="Cap the number of distilled rows."
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
        console.print(Text(strip_control(exc), style="red"))
        raise typer.Exit(2) from exc

    console.print(
        Panel(
            Text("\n").join(
                [
                    _detail("Traces:    ", plan.traces_path),
                    _detail("Teacher:   ", plan.teacher),
                    _detail("Student:   ", plan.student),
                    _detail("Strategy:  ", plan.strategy),
                    _detail("Provider:  ", provider),
                    _detail("Output:    ", plan.output_path),
                ]
            ),
            title="soup distill-prompt — plan",
        )
    )

    if plan_only:
        return

    from soup_cli.utils.data_forge import ForgeJudgeStats, ProviderCallError

    stats = ForgeJudgeStats()
    trace_stats = TraceRowStats()
    try:
        n = prepare_distill_dataset(
            plan,
            provider=provider,
            base_url=base_url,
            temperature=temperature,
            max_rows=max_rows,
            stats=stats,
            trace_stats=trace_stats,
        )
    except NoTracePromptsError as exc:
        # #1854: the input is at fault — no provider call was made and the
        # output path was left as it was.
        console.print(f"[red]No prompts found in traces:[/] {escape(str(exc))}")
        raise typer.Exit(3) from exc
    except ProviderCallError as exc:
        # #1274: every teacher/student call failed — fail loudly instead of
        # writing an empty dataset and reporting "done" (mirrors #1221).
        console.print(
            Text.assemble(("No usable rows produced:", "red"), " ", strip_control(exc))
        )
        raise typer.Exit(1) from exc
    except (TypeError, ValueError) as exc:
        console.print(Text(strip_control(exc), style="red"))
        raise typer.Exit(2) from exc
    except ImportError as exc:
        console.print(
            Panel(
                Text(strip_control(exc), style="yellow"),
                title="distill-prompt — missing dependency",
            )
        )
        raise typer.Exit(2) from exc

    body = Text()
    title = "soup distill-prompt — done"
    if stats.failures:
        body.append("Provider calls: ")
        body.append(f"{stats.failures} of {stats.calls} failed", style="bold yellow")
        body.append("\n")
        title = "soup distill-prompt — done with provider failures"
    body.append("Rows:   ")
    body.append(str(n), style="bold")
    if trace_stats.skipped:
        body.append(" ")
        body.append(
            f"({trace_stats.skipped} of {trace_stats.read} trace rows "
            "skipped: no prompt found)",
            style="yellow",
        )
    body.append("\nOutput: ")
    body.append(strip_control(plan.output_path), style="bold")
    console.print(Panel(body, title=title))
