"""soup profile — estimate memory, speed, and GPU requirements before training."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from soup_cli.utils.gpu import model_size_from_name
from soup_cli.utils.profiler import (
    GPU_MEMORY,
    estimate_speed,
    estimate_total,
    recommend_batch_size,
    recommend_gpu,
)

console = Console()


def profile(
    config: str = typer.Option(
        "soup.yaml", "--config", "-c", help="Path to soup.yaml config file"
    ),
    gpu: str = typer.Option(
        None, "--gpu", "-g",
        help=(
            "Target GPU for recommendations "
            "(e.g., rtx3090, rtx4090, a100, h100). Auto-detects if not set."
        ),
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output as JSON for scripting"
    ),
):
    """Estimate memory, speed, and GPU requirements BEFORE training."""
    from soup_cli.config.loader import load_config

    config_path = Path(config)
    if not config_path.exists():
        console.print(f"[red]Config file not found:[/] {config}")
        raise typer.Exit(1)

    cfg = load_config(config_path)

    # Determine model size
    model_params_b = model_size_from_name(cfg.base)

    # Determine batch size (use 4 as default estimate for "auto")
    batch_size = cfg.training.batch_size
    if batch_size == "auto":
        batch_size = 4
    else:
        batch_size = int(batch_size)

    # Resolve GPU memory
    gpu_memory_gb = _resolve_gpu_memory(gpu)

    # Compute profile
    result = estimate_total(
        model_name=cfg.base,
        model_params_b=model_params_b,
        quantization=cfg.training.quantization,
        lora_r=cfg.training.lora.r,
        lora_alpha=cfg.training.lora.alpha,
        batch_size=batch_size,
        seq_len=cfg.data.max_length,
        optimizer=cfg.training.optimizer,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
    )

    # Speed estimates
    tokens_per_sec = estimate_speed(
        model_params_b, cfg.training.quantization, batch_size
    )
    samples_per_sec = tokens_per_sec / max(cfg.data.max_length, 1)

    # Batch size recommendation
    recommended_bs = recommend_batch_size(result["total_memory_gb"], gpu_memory_gb)

    # GPU recommendations
    compatible_gpus = recommend_gpu(result["total_memory_gb"])

    # Add speed/time to result
    result["tokens_per_sec"] = round(tokens_per_sec, 1)
    result["samples_per_sec"] = round(samples_per_sec, 2)
    result["recommended_batch_size"] = recommended_bs
    result["compatible_gpus"] = compatible_gpus
    result["gpu_memory_gb"] = gpu_memory_gb

    if json_output:
        console.print(json.dumps(result, indent=2))
        return

    _render_profile(result, cfg, gpu_memory_gb)


def _resolve_gpu_memory(gpu: str | None) -> float:
    """Resolve GPU memory in GB from flag or auto-detection."""
    if gpu is not None:
        gpu_key = gpu.lower().replace(" ", "").replace("-", "")
        if gpu_key not in GPU_MEMORY:
            valid = ", ".join(sorted(GPU_MEMORY.keys()))
            console.print(
                f"[red]Unknown GPU:[/] {gpu}\n"
                f"[dim]Valid options: {valid}[/]"
            )
            raise typer.Exit(1)
        return float(GPU_MEMORY[gpu_key])

    # Auto-detect
    try:
        from soup_cli.utils.gpu import get_gpu_info

        info = get_gpu_info()
        mem_bytes = info.get("memory_total_bytes", 0)
        if mem_bytes > 0:
            return mem_bytes / (1024**3)
    except (ImportError, RuntimeError, OSError):
        pass

    # Default to 24 GB (common consumer GPU)
    return 24.0


def _render_profile(result: dict, cfg, gpu_memory_gb: float) -> None:
    """Render Rich profile output."""
    # Model info
    model_info = (
        f"Model:     [bold]{cfg.base}[/]\n"
        f"Params:    [bold]{result['model_params_b']:.1f}B[/] "
        f"(trainable: {result['trainable_params']:,.0f} with LoRA r={cfg.training.lora.r})\n"
        f"Quantization: [bold]{result['quantization']}[/]"
    )
    if result["gradient_checkpointing"]:
        model_info += "\nGradient checkpointing: [bold green]enabled[/]"

    # Memory breakdown table
    mem_table = Table(show_header=False, box=None, padding=(0, 2))
    mem_table.add_column("Component", style="bold")
    mem_table.add_column("Memory", justify="right")
    mem_table.add_row("Model", f"~{result['model_memory_gb']:.1f} GB")
    mem_table.add_row("LoRA", f"~{result['lora_memory_gb']:.1f} GB")
    mem_table.add_row("Optimizer", f"~{result['optimizer_memory_gb']:.1f} GB")
    mem_table.add_row(
        f"Activations (bs={result['batch_size']}, seq={result['seq_len']})",
        f"~{result['activation_memory_gb']:.1f} GB",
    )
    mem_table.add_row("Overhead", f"~{result['overhead_gb']:.1f} GB")
    mem_table.add_row("─" * 20, "─" * 10)
    mem_table.add_row("[bold]Total[/]", f"[bold]~{result['total_memory_gb']:.1f} GB[/]")

    # Speed info
    speed_info = (
        f"Tokens/sec: ~{result['tokens_per_sec']:,.0f}\n"
        f"Samples/sec: ~{result['samples_per_sec']:.1f}"
    )

    # Recommendations
    recs = []
    fits = result["total_memory_gb"] <= gpu_memory_gb
    if fits:
        recs.append(
            f"[green]✓[/] Fits in {gpu_memory_gb:.0f} GB VRAM"
        )
    else:
        recs.append(
            f"[red]✗[/] Does NOT fit in {gpu_memory_gb:.0f} GB VRAM "
            f"(need ~{result['total_memory_gb']:.0f} GB)"
        )

    recs.append(
        f"[green]✓[/] Recommended batch_size: {result['recommended_batch_size']}"
    )

    if result["total_memory_gb"] > 24 and not result["gradient_checkpointing"]:
        recs.append(
            "[yellow]⚠[/] Consider gradient_checkpointing: true for memory savings"
        )

    if result["total_memory_gb"] > 40:
        recs.append(
            "[yellow]⚠[/] Consider DeepSpeed ZeRO-3 or FSDP for distributed training"
        )

    # Compatible GPUs (show top 5)
    gpu_list = result["compatible_gpus"][:5]

    console.print(Panel(model_info, title="[bold]Training Profile[/]"))
    console.print()
    console.print("[bold]GPU Memory Estimate:[/]")
    console.print(mem_table)
    console.print()
    console.print(Panel(speed_info, title="Speed Estimate"))
    console.print()
    console.print(Panel("\n".join(recs), title="Recommendations"))
    console.print()

    if gpu_list:
        console.print("[bold]Compatible GPUs:[/]")
        for gpu_name in gpu_list:
            console.print(f"  • {gpu_name}")
