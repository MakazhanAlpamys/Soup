"""v0.61.0 Parts C/D/E — `soup edit` command group.

* ``soup edit set`` — surgical ROME / MEMIT / AlphaEdit (Part C).
* ``soup edit diff`` — knowledge-injection diff visualizer (Part E).
* Sequential edit governor (Part D) is consulted by both subcommands.
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

console = Console()

app = typer.Typer(
    name="edit",
    help=(
        "Knowledge editing (ROME / MEMIT / AlphaEdit) - patch facts "
        "without re-training (v0.61.0)."
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.command(name="set")
def set_edit(
    base: str = typer.Option(
        ..., "--base", "-b",
        help="Base model HF id or local path.",
    ),
    method: str = typer.Option(
        "rome", "--method", "-m",
        help="Edit method: rome / memit / alphaedit.",
    ),
    subject: str = typer.Option(
        ..., "--subject", "-s",
        help='Prefix sentence (e.g. "Paris is the capital of France").',
    ),
    target: str = typer.Option(
        ..., "--target", "-t",
        help='New completion target (e.g. "Lyon").',
    ),
    layer: Optional[int] = typer.Option(
        None, "--layer", "-l",
        help="MLP layer index to edit (defaults to method-specific recommended layer).",
    ),
    plan_only: bool = typer.Option(
        False, "--plan-only",
        help="Print the resolved EditPlan and exit without applying (deferred to v0.61.1).",
    ),
    registry_id: Optional[str] = typer.Option(
        None, "--registry-id",
        help="Optional Registry entry id to attach the edited model as a child (v0.61.1).",
    ),
) -> None:
    """Apply a single surgical knowledge edit."""
    from soup_cli.utils.knowledge_edit import apply_edit, build_edit_plan

    # Validate the optional --registry-id BEFORE building the plan so a
    # crafted id can't crash the v0.61.1 registry attach (review HIGH H4).
    if registry_id is not None:
        if not isinstance(registry_id, str) or not registry_id:
            console.print("[red]Invalid --registry-id:[/] must be non-empty")
            raise typer.Exit(2)
        if "\x00" in registry_id:
            console.print("[red]Invalid --registry-id:[/] null bytes not allowed")
            raise typer.Exit(2)
        if len(registry_id) > 256:
            console.print("[red]Invalid --registry-id:[/] >256 chars")
            raise typer.Exit(2)

    try:
        plan = build_edit_plan(
            base=base,
            method=method,
            subject=subject,
            target=target,
            layer=layer,
        )
    except (TypeError, ValueError) as exc:
        console.print(f"[red]Invalid edit request:[/] {escape(str(exc))}")
        raise typer.Exit(2) from exc

    # Render the resolved plan.
    body = (
        f"[bold]Method:[/] {escape(plan.method)}\n"
        f"[bold]Base:[/]   {escape(plan.base)}\n"
        f"[bold]Layer:[/]  {plan.layer}\n"
        f"[bold]Subject:[/] {escape(plan.subject)}\n"
        f"[bold]Target:[/]  {escape(plan.target)}\n\n"
        f"[dim]{escape(plan.spec.description)}[/]"
    )
    console.print(Panel(body, title="EditPlan", border_style="cyan"))

    if registry_id is not None:
        console.print(
            f"[dim]Will attach as Registry child of {escape(registry_id)} "
            f"once v0.61.1 lands.[/]"
        )

    if plan_only:
        console.print("[green]Plan-only mode — exiting without applying.[/]")
        return

    # Live apply: raises NotImplementedError with explicit v0.61.1 marker.
    # Exit code 3 distinguishes "deferred / not yet shipped" from "exit 2
    # = validation rejection" (matches v0.56.0 diagnose strict-mode policy).
    try:
        apply_edit(plan)
    except NotImplementedError as exc:
        console.print(
            Panel(
                f"[yellow]{escape(str(exc))}[/]\n\n"
                f"Re-run with [bold]--plan-only[/] to validate the request "
                f"and exit 0 until v0.61.1 lands the live kernel.",
                title="Live edit deferred",
                border_style="yellow",
            )
        )
        raise typer.Exit(3) from exc


@app.command(name="diff")
def diff_edit(
    before: str = typer.Argument(
        ..., help="Registry run id of the model BEFORE the edit.",
    ),
    after: str = typer.Argument(
        ..., help="Registry run id of the model AFTER the edit.",
    ),
    probe_file: Optional[str] = typer.Option(
        None, "--probes",
        help=(
            "Optional JSONL file with probe prompts. Each row should "
            "have a 'prompt' field. Capped at 1000 rows."
        ),
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="Where to write the rendered diff JSON.",
    ),
    top_k: int = typer.Option(
        10, "--top-k", "-k",
        help="Number of changed facts to surface (1-100).",
    ),
) -> None:
    """Knowledge-injection diff: facts changed between before / after.

    Schema-only in v0.61.0 — actual model loading + generation is the
    v0.61.1 deliverable. This release validates inputs + renders a
    placeholder diff table so the CLI surface is stable.
    """
    from soup_cli.utils.edit_diff import build_diff_report, render_diff_table

    try:
        report = build_diff_report(
            before_run_id=before,
            after_run_id=after,
            probe_file=probe_file,
            top_k=top_k,
        )
    except (TypeError, ValueError, FileNotFoundError) as exc:
        console.print(f"[red]Cannot build diff:[/] {escape(str(exc))}")
        raise typer.Exit(2) from exc

    render_diff_table(report, console)

    if output is not None:
        from soup_cli.utils.edit_diff import write_diff_report

        try:
            write_diff_report(report, output)
        except (TypeError, ValueError, OSError) as exc:
            console.print(f"[red]Cannot write diff:[/] {escape(str(exc))}")
            raise typer.Exit(2) from exc
        console.print(f"Wrote diff -> {escape(output)}")
