"""v0.71.10 #200 — ``soup ra-dit`` one-shot two-stage RA-DIT orchestrator.

Chains the RA-DIT retriever (stage 1, embedding/contrastive) and generator
(stage 2, RAFT-style SFT) in a single invocation, **recording** the trained
retriever as the generator's paired retriever (it writes the retriever's
output dir into the generator config's ``training.ra_dit_retriever_model``).
The generator is trained RAFT-style to be robust to retrieved distractors;
the recorded retriever is the one used at deploy/serve time for the actual
retrieval step (per the Meta RA-DIT recipe) — stage-2 training does not fuse
the retriever weights. Mirrors the v0.62.0 Part B schema; this command lifts
the deferred "live orchestration" note.

``--plan-only`` validates both config paths + renders the resolved plan
without training. The live path runs each stage as a subprocess
(``soup train --config <yaml> --yes``) via :mod:`soup_cli.utils.ra_dit_run`.
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

console = Console()

app = typer.Typer(
    name="ra-dit",
    help=(
        "RA-DIT (Retrieval-Augmented Dual Instruction Tuning) one-shot "
        "orchestrator: train the retriever then the generator, recording the "
        "trained retriever as the generator's paired retriever (v0.71.10)."
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.callback(invoke_without_command=True)
def main(
    retriever_config: str = typer.Option(
        ..., "--retriever-config", "-r",
        help="Stage-1 embedding/contrastive recipe (cwd-contained).",
    ),
    generator_config: str = typer.Option(
        ..., "--generator-config", "-g",
        help="Stage-2 RAFT-SFT recipe (cwd-contained).",
    ),
    retriever_model: Optional[str] = typer.Option(
        None, "--retriever-model",
        help="Manual retriever-model override; skips Registry auto-link.",
    ),
    timeout: int = typer.Option(
        6 * 60 * 60, "--timeout",
        help="Per-stage hard timeout in seconds (60..21600).",
    ),
    plan_only: bool = typer.Option(
        False, "--plan-only",
        help="Validate config paths + render the plan; skip training.",
    ),
) -> None:
    """Run (or plan) a two-stage RA-DIT pipeline."""
    from soup_cli.utils.ra_dit_run import (
        resolve_retriever_for_generator,
        run_ra_dit,
        validate_ra_dit_config_path,
    )

    # Validate both config paths up front (containment + existence) so a typo
    # fails fast with a clear message — for plan-only AND live.
    try:
        retr_path = validate_ra_dit_config_path("--retriever-config", retriever_config)
        gen_path = validate_ra_dit_config_path("--generator-config", generator_config)
    except (TypeError, ValueError, FileNotFoundError) as exc:
        console.print(f"[red]Invalid ra-dit config:[/] {escape(str(exc))}")
        raise typer.Exit(2) from exc

    if plan_only:
        # Preview the retriever link without touching the Registry write path.
        if retriever_model is not None:
            link_preview = (
                f"manual override: {escape(retriever_model)}"
            )
        else:
            resolved, advisory = resolve_retriever_for_generator(None)
            link_preview = escape(advisory)
        console.print(
            Panel(
                f"[bold]Stage 1 (retriever):[/] {escape(retr_path)}\n"
                f"[bold]Stage 2 (generator):[/] {escape(gen_path)}\n"
                f"[bold]Retriever link:[/] {link_preview}\n"
                f"[bold]Per-stage timeout:[/] {timeout}s",
                title="soup ra-dit (plan-only)",
                border_style="cyan",
            )
        )
        console.print(
            "[green]Plan-only mode — config paths validated. "
            "Drop --plan-only to run both stages.[/]"
        )
        return

    try:
        result = run_ra_dit(
            retriever_config,
            generator_config,
            retriever_model=retriever_model,
            timeout_seconds=timeout,
        )
    except (TypeError, ValueError, FileNotFoundError, RuntimeError) as exc:
        console.print(f"[red]RA-DIT run failed:[/] {escape(str(exc))}")
        raise typer.Exit(2) from exc

    link_note = "auto-linked" if result.autolinked else "manual override"
    console.print(
        Panel(
            f"[bold]Retriever output:[/] {escape(result.retriever_output)}\n"
            f"[bold]Generator output:[/] {escape(result.generator_output)}\n"
            f"[bold]Retriever model used:[/] "
            f"{escape(result.retriever_model_used)} ({link_note})",
            title="RA-DIT pipeline complete",
            border_style="green",
        )
    )
