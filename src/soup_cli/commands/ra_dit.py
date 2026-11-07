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
from rich.panel import Panel
from rich.text import Text

from soup_cli.utils.terminal import strip_control

console = Console()


def _line(label: str, value: object) -> Text:
    # A Text is not parsed as markup and has no emoji codes, so the value is shown as typed.
    return Text.assemble((label, "bold"), " ", strip_control(value))

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
        load_ra_dit_configs,
        run_ra_dit,
        validate_ra_dit_config_path,
    )

    # Validate both config paths up front (containment + existence) so a typo
    # fails fast with a clear message — for plan-only AND live.
    try:
        retr_path = validate_ra_dit_config_path("--retriever-config", retriever_config)
        gen_path = validate_ra_dit_config_path("--generator-config", generator_config)
    except (TypeError, ValueError, FileNotFoundError) as exc:
        console.print(Text.assemble(("Invalid ra-dit config:", "red"), " ", strip_control(exc)))
        raise typer.Exit(2) from exc

    if plan_only:
        try:
            _, _, retriever_output, generator_output = load_ra_dit_configs(
                retr_path,
                gen_path,
            )
        except (TypeError, ValueError, FileNotFoundError) as exc:
            console.print(
                Text.assemble(("Invalid ra-dit config:", "red"), " ", strip_control(exc))
            )
            raise typer.Exit(2) from exc

        if retriever_model is not None:
            link_preview = f"manual override: {retriever_model}"
        else:
            link_preview = f"auto-link to stage-1 output: {retriever_output}"

        console.print(
            Panel(
                Text("\n").join(
                    [
                        _line("Stage 1 (retriever):", retr_path),
                        _line("Retriever output:", retriever_output),
                        _line("Stage 2 (generator):", gen_path),
                        _line("Generator output:", generator_output),
                        _line("Retriever link:", link_preview),
                        _line("Per-stage timeout:", f"{timeout}s"),
                    ]
                ),
                title="soup ra-dit (plan-only)",
                border_style="cyan",
            )
        )
        console.print(
            "[green]Plan-only mode — config paths and output directories "
            "validated. Drop --plan-only to run both stages.[/]"
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
        console.print(Text.assemble(("RA-DIT run failed:", "red"), " ", strip_control(exc)))
        raise typer.Exit(2) from exc

    link_note = "auto-linked" if result.autolinked else "manual override"
    console.print(
        Panel(
            Text("\n").join(
                [
                    _line("Retriever output:", result.retriever_output),
                    _line("Generator output:", result.generator_output),
                    _line("Retriever model used:", f"{result.retriever_model_used} ({link_note})"),
                ]
            ),
            title="RA-DIT pipeline complete",
            border_style="green",
        )
    )
