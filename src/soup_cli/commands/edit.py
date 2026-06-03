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
    output: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="Directory to save the edited model / GRACE codebook (cwd-contained).",
    ),
    device: Optional[str] = typer.Option(
        None, "--device",
        help="torch device (cpu / cuda). Defaults to CUDA when available.",
    ),
    use_governor: bool = typer.Option(
        True, "--governor/--no-governor",
        help="Consult the sequential-edit governor (refuse on norm blowup).",
    ),
    plan_only: bool = typer.Option(
        False, "--plan-only",
        help="Print the resolved EditPlan and exit without applying.",
    ),
    registry_id: Optional[str] = typer.Option(
        None, "--registry-id",
        help="Optional Registry entry id to attach the edited model as a child.",
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

    if plan_only:
        console.print("[green]Plan-only mode — exiting without applying.[/]")
        return

    # Load a persisted governor so sequential edits across separate `soup edit
    # set` runs accumulate (#196 / #197). Best-effort — a missing / unreadable
    # governor DB never blocks an edit.
    governor = None
    store = None
    if use_governor:
        try:
            from soup_cli.utils.edit_governor import (
                EditGovernorStore,
                default_governor_db_path,
                load_governor,
                save_governor,
            )

            store = EditGovernorStore(default_governor_db_path())
            governor = load_governor(store, plan.base)
        except (ValueError, OSError) as exc:
            console.print(f"[dim]Governor disabled ({escape(str(exc))}).[/]")
            governor = None
            store = None

    from soup_cli.utils.edit_governor import GovernedEditError

    try:
        result = apply_edit(
            plan,
            output_dir=output,
            governor=governor,
            device=device,
        )
    except GovernedEditError as exc:
        console.print(
            Panel(
                f"[red]Edit refused by governor:[/] {escape(str(exc))}",
                title="Sequential-edit governor",
                border_style="red",
            )
        )
        if store is not None:
            store.close()
        raise typer.Exit(2) from exc
    except (ValueError, RuntimeError, OSError) as exc:
        console.print(f"[red]Edit failed:[/] {escape(str(exc))}")
        if store is not None:
            store.close()
        raise typer.Exit(2) from exc

    # Persist the updated governor state so the next edit sees the new count.
    if governor is not None and store is not None:
        try:
            save_governor(store, governor)
        except (ValueError, OSError):
            pass
        finally:
            store.close()

    console.print(
        Panel(
            f"[bold]Method:[/] {escape(result.method)}\n"
            f"[bold]Layers edited:[/] {list(result.layers_edited)}\n"
            f"[bold]Norm delta:[/] {result.norm_delta:.4f}\n"
            f"[bold]Target prob:[/] {result.target_prob_before:.4f} -> "
            f"{result.target_prob_after:.4f}\n"
            + (
                f"[bold]Saved:[/] {escape(result.output_dir)}\n"
                if result.output_dir
                else ""
            )
            + (
                f"[bold]Governor:[/] count="
                f"{governor.edit_count} verdict={escape(governor.last_verdict)}"
                if governor is not None
                else "[dim]Governor disabled.[/]"
            ),
            title="Edit applied",
            border_style="green",
        )
    )

    if registry_id is not None and result.output_dir is not None:
        from soup_cli.registry.attach import attach_artifact

        kind = "grace_codebook" if result.method == "grace" else "edited_model"
        try:
            attach_artifact(registry_id, path=result.output_dir, kind=kind)
            console.print(
                f"[green]Attached {kind} to Registry entry "
                f"{escape(registry_id)}.[/]"
            )
        except (ValueError, FileNotFoundError) as exc:
            console.print(
                f"[yellow]Could not attach to Registry:[/] {escape(str(exc))}"
            )


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
    before_model: Optional[str] = typer.Option(
        None, "--before-model",
        help="Model path / HF id of the BEFORE model (enables live generation).",
    ),
    after_model: Optional[str] = typer.Option(
        None, "--after-model",
        help="Model path / HF id of the AFTER model (enables live generation).",
    ),
    device: Optional[str] = typer.Option(
        None, "--device",
        help="torch device (cpu / cuda). Defaults to CUDA when available.",
    ),
) -> None:
    """Knowledge-injection diff: facts changed between before / after.

    Pass both ``--before-model`` and ``--after-model`` (plus ``--probes``) to
    generate the diff LIVE (v0.71.9 #194): each probe is run through both
    models and changed completions are surfaced. Without model paths the
    report is a validated placeholder.
    """
    from soup_cli.utils.edit_diff import build_diff_report, render_diff_table

    try:
        report = build_diff_report(
            before_run_id=before,
            after_run_id=after,
            probe_file=probe_file,
            top_k=top_k,
            before_model=before_model,
            after_model=after_model,
            device=device,
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
