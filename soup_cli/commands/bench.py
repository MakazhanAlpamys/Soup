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
        help="Number of prompts to run for averaging",
    ),
) -> None:
    """Run an inference benchmark (speed and memory) on a loaded model."""
    import torch

    from soup_cli.commands.infer import _generate, _load_model
    from soup_cli.utils.gpu import detect_device

    model_path = Path(model)
    if not model_path.exists():
        console.print(f"[red]Model not found: {model_path}[/]")
        raise typer.Exit(1)

    device, _ = detect_device()

    if device == "cpu":
        console.print(
            "[yellow]Warning:[/] Running on CPU. Inference speed is typically "
            "10-100x slower than GPU -- results will not reflect production TPS."
        )

    console.print(
        Panel(
            f"Model:    [bold]{model_path}[/]\n"
            f"Device:   [bold]{device}[/]\n"
            f"Prompts:  [bold]{num_prompts}[/]\n"
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

    prompts = [
        "Explain the theory of relativity briefly.",
        "Write a short Python function to calculate fibonacci numbers.",
        "What are the main consequences of the Industrial Revolution?",
        "Compose a poem about a wandering space traveler.",
        "Describe how a database index works under the hood.",
    ]
    test_prompts = (prompts * (num_prompts // len(prompts) + 1))[:num_prompts]

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

    console.print(f"[bold]Running {num_prompts} test inferences...[/]")

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
        "Transformers",
        f"{avg_tps:.2f}",
        f"{total_latency:.2f}s",
        vram_str,
    )

    console.print()
    console.print(table)
