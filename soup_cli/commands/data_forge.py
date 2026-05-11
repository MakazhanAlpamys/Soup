"""soup data forge — Synthetic Data Forge CLI (v0.47.0 Part A)."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

console = Console()


def _default_judge(prompt: str) -> dict:
    """Deterministic offline judge used when no provider is configured.

    Live judge integration (Ollama / Anthropic / vLLM via v0.20.0
    providers) is wired through a ``--judge-provider`` flag in v0.47.1.
    """
    # Echo a short stand-in answer; the active-pruning step will still
    # score these against the source chunk so we don't keep duplicates.
    head = prompt.splitlines()[0] if prompt else ""
    return {"text": f"Synthesised answer (offline): {head[:80]}"}


def forge(
    docs: str = typer.Option(
        ..., "--docs", "-d",
        help="Directory of documents (txt/md/json/jsonl, under cwd).",
    ),
    task: str = typer.Option(
        "sft", "--task", "-t",
        help="Forge task: sft | preference | tool.",
    ),
    target_rows: int = typer.Option(
        100, "--target-rows", "-r",
        min=1, max=1_000_000,
        help="Maximum number of synthesised rows.",
    ),
    teacher: str = typer.Option(
        "local-judge", "--teacher",
        help="Label recorded in provenance for the judge backend.",
    ),
    output: str = typer.Option(
        "forge_dataset.jsonl", "--output", "-o",
        help="JSONL output path (under cwd).",
    ),
    provenance: str = typer.Option(
        "forge_provenance.json", "--provenance",
        help="Provenance manifest output path (under cwd).",
    ),
    uncertainty_threshold: float = typer.Option(
        0.0, "--uncertainty-threshold",
        min=0.0, max=1.0,
        help="Minimum Jaccard-distance score required to keep a synthesised row.",
    ),
    max_chunk_chars: int = typer.Option(
        1000, "--max-chunk-chars",
        min=1, max=64_000,
        help="Maximum chars per document chunk before judge call.",
    ),
):
    """Run the multi-stage synthetic data pipeline with provenance."""
    from soup_cli.utils.data_forge import (
        build_forge_plan,
        discover_documents,
        synthesise_forge_rows,
        write_forge_dataset,
        write_provenance,
    )

    try:
        plan = build_forge_plan(
            docs_dir=docs,
            task=task,
            target_rows=target_rows,
            teacher=teacher,
            uncertainty_threshold=uncertainty_threshold,
        )
        # Discover once; plan.num_docs already attests the directory is
        # non-empty and contained, so this scan is the cached enumeration.
        doc_paths = discover_documents(docs)
    except (TypeError, ValueError, FileNotFoundError) as exc:
        console.print(f"[red]Plan failed:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    rows = synthesise_forge_rows(
        doc_paths,
        task=plan.task,
        target_rows=plan.target_rows,
        judge=_default_judge,
        teacher=plan.teacher,
        uncertainty_threshold=plan.uncertainty_threshold,
        max_chunk_chars=max_chunk_chars,
    )

    if not rows:
        console.print(
            "[yellow]No rows produced.[/] Try a lower --uncertainty-threshold "
            "or check your --docs directory."
        )
        raise typer.Exit(1)

    try:
        dataset_path = write_forge_dataset(rows, output)
        manifest_path = write_provenance(rows, provenance)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]Write failed:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    console.print(
        Panel(
            f"Task:        [bold]{escape(plan.task)}[/]\n"
            f"Docs scanned:[bold] {plan.num_docs}[/]\n"
            f"Rows kept:   [bold]{len(rows)}[/] / target {plan.target_rows}\n"
            f"Teacher:     [bold]{escape(plan.teacher)}[/]\n"
            f"Threshold:   [bold]{plan.uncertainty_threshold:.2f}[/]\n"
            f"Dataset:     [bold]{escape(dataset_path)}[/]\n"
            f"Provenance:  [bold]{escape(manifest_path)}[/]",
            title="[bold green]Data Forge — synth complete[/]",
        )
    )
    console.print(
        "[dim]The built-in judge is the deterministic offline stub. "
        "Live Ollama / Anthropic / vLLM judge providers wire through "
        "--judge-provider in v0.47.1.[/]"
    )
