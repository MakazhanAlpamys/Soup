"""soup data forge — Synthetic Data Forge CLI (v0.47.0 Part A / v0.53.7 #111)."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

console = Console()


def _default_judge(prompt: str) -> dict:
    """Deterministic offline judge used when no provider is configured.

    Live judge integration (Ollama / Anthropic / vLLM via v0.20.0
    providers) is wired through ``--judge-provider`` in v0.53.7 #111.
    """
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
    judge_provider: Optional[str] = typer.Option(
        None, "--judge-provider",
        help=(
            "v0.53.7 #111: live judge backend. One of "
            "ollama / anthropic / vllm. Omit for the deterministic "
            "offline stub."
        ),
    ),
    judge_model: str = typer.Option(
        "llama3.1", "--judge-model",
        help="Model name for the live judge provider (default llama3.1).",
    ),
    judge_base_url: Optional[str] = typer.Option(
        None, "--judge-base-url",
        help=(
            "Override base URL for Ollama (localhost-only) / vLLM "
            "(scheme allowlist + loopback). Ignored for Anthropic."
        ),
    ),
    hub: str = typer.Option(
        "hf", "--hub",
        help=(
            "Teacher hub: hf (default) / modelscope / modelers. When non-HF "
            "and --teacher is a repo id (owner/name), the teacher is "
            "pre-fetched from that hub (v0.71.5 #157)."
        ),
    ),
):
    """Run the multi-stage synthetic data pipeline with provenance.

    v0.53.7 #111: ``--judge-provider {ollama, anthropic, vllm}`` swaps the
    offline echo stub for a real judge backend. Ollama is localhost-only,
    Anthropic reads ``ANTHROPIC_API_KEY`` from env, vLLM is
    scheme-allowlist + loopback validated.
    """
    from soup_cli.utils.data_forge import (
        JUDGE_PROVIDERS,
        build_forge_plan,
        discover_documents,
        make_judge_provider_fn,
        synthesise_forge_rows,
        write_forge_dataset,
        write_provenance,
    )
    from soup_cli.utils.hubs import validate_hub_name

    try:
        hub_canonical = validate_hub_name(hub)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    effective_teacher = teacher
    # v0.71.5 #157 — non-HF hub + repo-id teacher → pre-fetch the teacher from
    # that hub and record the resolved local path in provenance. HF (default)
    # is a no-op: the teacher stays a provenance label.
    if hub_canonical != "hf":
        if "/" in teacher:
            from soup_cli.utils.hubs import prefetch_model_from_hub

            try:
                effective_teacher = prefetch_model_from_hub(
                    teacher, hub_canonical, console=console
                )
            except ImportError as exc:
                console.print(f"[red]{escape(str(exc))}[/]")
                raise typer.Exit(1) from exc
            except (TypeError, ValueError) as exc:
                console.print(
                    f"[red]Teacher pre-fetch failed:[/] {escape(str(exc))}"
                )
                raise typer.Exit(1) from exc
        else:
            # Non-HF hub requested but the teacher is not a routable repo id
            # (owner/name) — warn loudly instead of silently ignoring --hub
            # (code-review MEDIUM fix v0.71.5 #157).
            console.print(
                f"[yellow]--hub {escape(hub_canonical)} ignored:[/] --teacher "
                f"{escape(teacher)!r} is not a repo id (owner/name), so there "
                "is nothing to pre-fetch."
            )

    judge_fn = _default_judge
    if judge_provider is not None:
        canonical = judge_provider.strip().lower()
        if canonical not in JUDGE_PROVIDERS:
            console.print(
                f"[red]Unknown --judge-provider:[/] {escape(judge_provider)}. "
                f"Choose from {sorted(JUDGE_PROVIDERS)}."
            )
            raise typer.Exit(2)
        try:
            judge_fn = make_judge_provider_fn(
                canonical,
                model=judge_model,
                base_url=judge_base_url,
            )
        except (TypeError, ValueError, ImportError) as exc:
            console.print(
                f"[red]Failed to build judge backend:[/] {escape(str(exc))}"
            )
            raise typer.Exit(1) from exc
        # Record the live backend in provenance.
        if teacher == "local-judge":
            effective_teacher = f"{canonical}:{judge_model}"

    try:
        plan = build_forge_plan(
            docs_dir=docs,
            task=task,
            target_rows=target_rows,
            teacher=effective_teacher,
            uncertainty_threshold=uncertainty_threshold,
        )
        doc_paths = discover_documents(docs)
    except (TypeError, ValueError, FileNotFoundError) as exc:
        console.print(f"[red]Plan failed:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    rows = synthesise_forge_rows(
        doc_paths,
        task=plan.task,
        target_rows=plan.target_rows,
        judge=judge_fn,
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
    if judge_provider is None:
        console.print(
            "[dim]The built-in judge is the deterministic offline stub. "
            "Pass --judge-provider {ollama, anthropic, vllm} to wire a "
            "live backend (v0.53.7 #111).[/]"
        )
    else:
        console.print(
            f"[dim]Live judge backend: [bold]{escape(judge_provider)}[/] "
            f"(model: {escape(judge_model)}).[/]"
        )
