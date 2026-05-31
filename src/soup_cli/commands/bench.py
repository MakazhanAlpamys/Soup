"""soup bench -- simple measuring tool for model speed and memory."""

import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def bench(
    model: str = typer.Argument(
        ...,
        help="Path to model (LoRA adapter or full model) to benchmark",
    ),
    base: Optional[str] = typer.Option(
        None,
        "--base",
        "-b",
        help="Base model for LoRA adapter (auto-detected if not set)",
    ),
    max_tokens: int = typer.Option(
        128,
        "--max-tokens",
        help="Maximum tokens to generate per prompt",
    ),
    num_prompts: int = typer.Option(
        3,
        "--num-prompts",
        "-n",
        help="Number of prompts to run for averaging (ignored when --prompts-file is set)",
    ),
    prompts_file: Optional[str] = typer.Option(
        None,
        "--prompts-file",
        help="Path to custom prompts file (.txt or .jsonl)",
    ),
    p50: bool = typer.Option(
        False,
        "--p50",
        help="Print p50 (median) per-prompt latency. v0.53.9 #26.",
    ),
    p95: bool = typer.Option(
        False,
        "--p95",
        help="Print p95 per-prompt latency. v0.53.9 #26.",
    ),
    backend: str = typer.Option(
        "auto",
        "--backend",
        help=(
            "Inference backend hint: auto (default) | transformers | mlx. "
            "v0.53.9 #28."
        ),
    ),
) -> None:
    """Run an inference benchmark (speed and memory) on a loaded model."""
    import torch

    from soup_cli.commands.infer import _generate, _load_model, _resolve_model_source
    from soup_cli.utils.gpu import detect_device

    # Resolve local-path-or-HF-id (#N7).
    try:
        model_kind, model_ref = _resolve_model_source(model)
    except FileNotFoundError as exc:
        console.print(
            f"[red]{exc}[/]\n"
            "[dim]If you meant a HuggingFace repo, use the form "
            "'owner/repo-name' (no leading './').[/]"
        )
        raise typer.Exit(1) from exc
    model_path = Path(model_ref)
    if model_kind == "hf":
        console.print(
            f"[dim]Local path not found; treating {model_ref!r} as a HF repo id.[/]"
        )

    device, _ = detect_device()

    # v0.53.9 #28 — backend auto-detect.
    from soup_cli.utils.backend_detect import SUPPORTED_BACKENDS, detect_backend

    backend_lower = (backend or "auto").strip().lower()
    if backend_lower == "auto":
        backend_resolved = detect_backend(str(model_path))
        console.print(
            f"[dim]Backend auto-detected:[/] [bold]{backend_resolved}[/]"
        )
    else:
        if backend_lower not in SUPPORTED_BACKENDS:
            console.print(
                f"[red]Unknown --backend:[/] {backend} "
                f"(expected: auto | {' | '.join(sorted(SUPPORTED_BACKENDS))})"
            )
            raise typer.Exit(2)
        backend_resolved = backend_lower

    if device == "cpu":
        console.print(
            "[yellow]Warning:[/] Running on CPU. Inference speed is typically "
            "10-100x slower than GPU -- results will not reflect production TPS."
        )

    if prompts_file:
        import json
        import os as _os

        from soup_cli.utils.paths import is_under_cwd

        if not is_under_cwd(prompts_file):
            console.print(
                "[red]Security Error:[/] Prompts file must stay under the "
                "current working directory."
            )
            raise typer.Exit(1)
        # v0.53.9 review fix M2 — reject symlinked prompts file on the
        # RAW path BEFORE realpath resolution (mirrors v0.53.7 #106 policy).
        import stat as _stat

        try:
            _st = _os.lstat(prompts_file)
        except OSError:
            console.print(f"[red]Prompts file not found:[/] {prompts_file}")
            raise typer.Exit(1)
        if _stat.S_ISLNK(_st.st_mode):
            console.print(
                "[red]Prompts file must not be a symlink.[/]"
            )
            raise typer.Exit(1)
        p_path = Path(_os.path.realpath(prompts_file))

        if not p_path.is_file():
            console.print(f"[red]Prompts file not found:[/] {p_path}")
            raise typer.Exit(1)

        prompts = []
        try:
            if p_path.suffix == ".jsonl":
                with open(p_path, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            if "prompt" in data:
                                prompts.append(data["prompt"])
            else:
                with open(p_path, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            prompts.append(line)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            console.print(f"[red]Failed to read prompts file:[/] {exc}")
            raise typer.Exit(1) from exc

        if not prompts:
            console.print("[red]No prompts found in file.[/]")
            raise typer.Exit(1)

        # When --prompts-file is provided, run all prompts in the file; --num-prompts is ignored.
    else:
        prompts = [
            "Explain the theory of relativity briefly.",
            "Write a short Python function to calculate fibonacci numbers.",
            "What are the main consequences of the Industrial Revolution?",
            "Compose a poem about a wandering space traveler.",
            "Describe how a database index works under the hood.",
        ]

    actual_num_prompts = len(prompts) if prompts_file else num_prompts

    console.print(
        Panel(
            f"Model:    [bold]{model_path}[/]\n"
            f"Device:   [bold]{device}[/]\n"
            f"Prompts:  [bold]{actual_num_prompts}[/]\n"
            f"Tokens/P: [bold]{max_tokens}[/]",
            title="Benchmarking Configuration",
        )
    )

    console.print("[dim]Loading model to measure resource usage...[/]")

    # Reset peak stats before load -- "Max VRAM" reflects total footprint
    # (model load + inference), i.e. what users need for deployment planning.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_load = time.time()
    try:
        model_obj, tokenizer = _load_model(str(model_path), base, device)
    except (OSError, ImportError, RuntimeError, ValueError) as exc:
        console.print(f"[red]Failed to load model:[/] {exc}")
        raise typer.Exit(1) from exc

    load_time = time.time() - start_load
    console.print(f"[green]Model loaded in {load_time:.2f}s.[/]\n")

    test_prompts = (prompts * (actual_num_prompts // len(prompts) + 1))[:actual_num_prompts]

    # Warmup run: first inference includes CUDA kernel JIT compilation,
    # which would skew the average. Discarded from timing.
    console.print("[dim]Warmup run (discarded from timing)...[/]")
    warmup_messages = [{"role": "user", "content": test_prompts[0]}]
    _generate(
        model_obj, tokenizer, warmup_messages,
        max_tokens=min(max_tokens, 32), temperature=0.0,
    )

    total_tokens = 0
    total_latency = 0.0
    per_prompt_latencies: list[float] = []

    console.print(f"[bold]Running {len(test_prompts)} test inferences...[/]")

    for i, prompt_text in enumerate(test_prompts):
        messages = [{"role": "user", "content": prompt_text}]
        start_time = time.time()

        _, token_count = _generate(
            model_obj, tokenizer, messages,
            max_tokens=max_tokens, temperature=0.0,
        )

        latency = time.time() - start_time
        total_tokens += token_count
        total_latency += latency
        per_prompt_latencies.append(latency)
        console.print(f"  [dim]Prompt {i + 1}: {token_count} tokens in {latency:.2f}s[/]")

    avg_tps = total_tokens / total_latency if total_latency > 0 else 0

    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

    table = Table(title="Inference Benchmark Results")
    table.add_column("Backend", style="cyan")
    table.add_column("TPS (Avg)", style="green", justify="right")
    table.add_column("Latency (Total)", style="yellow", justify="right")
    table.add_column("Max VRAM", style="magenta", justify="right")

    vram_str = f"{peak_vram_gb:.2f} GB" if torch.cuda.is_available() else "N/A"
    table.add_row(
        backend_resolved.capitalize(),
        f"{avg_tps:.2f}",
        f"{total_latency:.2f}s",
        vram_str,
    )

    console.print()
    console.print(table)

    # v0.53.9 #26 — tail-latency percentiles.
    if (p50 or p95) and per_prompt_latencies:
        from soup_cli.utils.tail_latency import summarise_latency

        summary = summarise_latency(per_prompt_latencies)
        pct_table = Table(title="Per-prompt latency")
        pct_table.add_column("Statistic", style="cyan")
        pct_table.add_column("Latency (s)", style="green", justify="right")
        pct_table.add_row("count", str(summary.count))
        if summary.mean is not None:
            pct_table.add_row("mean", f"{summary.mean:.3f}")
        if p50 and summary.p50 is not None:
            pct_table.add_row("p50", f"{summary.p50:.3f}")
        if p95 and summary.p95 is not None:
            pct_table.add_row("p95", f"{summary.p95:.3f}")
        # p99 is a natural superset of p95 — only print when p95 was requested.
        if p95 and summary.p99 is not None:
            pct_table.add_row("p99", f"{summary.p99:.3f}")
        console.print()
        console.print(pct_table)
