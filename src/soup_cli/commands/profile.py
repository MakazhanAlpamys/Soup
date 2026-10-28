"""soup profile — estimate memory, speed, and GPU requirements before training."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from soup_cli.utils.profiler import (
    ASSUMED_GPU_MEMORY_GB,
    GPU_MEMORY,
    estimate_speed,
    normalize_gpu_key,
    profile_from_config,
    recommend_batch_size,
    recommend_gpu,
)
from soup_cli.utils.terminal import for_terminal
from soup_cli.utils.vram_estimate import VRAM_SAFETY_MARGIN, fit_verdict

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

    # Resolve GPU memory
    gpu_memory_gb, gpu_memory_source = _resolve_gpu_memory(gpu)

    # The train pre-flight's estimate, from the model's own shape (#2049). An assumed
    # card is not one to resolve `auto` checkpointing on, so it prices as full then.
    result = profile_from_config(
        cfg, gpu_memory_gb=None if gpu_memory_source == "assumed" else gpu_memory_gb
    )
    if result is None:
        console.print(
            "[red]Cannot estimate this run:[/] batch_size, max_length or the model's "
            "size (over 1,000B parameters) is outside the range the estimate supports."
        )
        raise typer.Exit(1)
    model_params_b = result["model_params_b"]
    batch_size = result["batch_size"]

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
    # "flag" (--gpu), "detected" (a device was found) or "assumed" (neither:
    # the number is a default, not a measurement).
    result["gpu_memory_source"] = gpu_memory_source
    # "fits", "tight" (fits, but not with the train pre-flight's margin) or
    # "does not fit". A card's GB are binary: an "rtx3060laptop" 6 is 6144 MiB.
    result["fit"] = fit_verdict(result["total_memory_bytes"], gpu_memory_gb * 1024**3)

    if json_output:
        # Machine-readable JSON: raw stdout, not console.print — Rich folds
        # long lines and parses ``[...]`` as markup when piped (issue #1468).
        typer.echo(json.dumps(result, indent=2))
        return

    _render_profile(result, cfg, gpu_memory_gb, gpu_memory_source)


def _resolve_gpu_memory(gpu: str | None) -> tuple[float, str]:
    """Resolve GPU memory in GB, and where the number came from.

    The source is ``"flag"``, ``"detected"`` or ``"assumed"``; an assumed
    value must not be presented as a measured fit.
    """
    if gpu is not None:
        gpu_key = normalize_gpu_key(gpu)
        if gpu_key not in GPU_MEMORY:
            valid = ", ".join(sorted(GPU_MEMORY.keys()))
            console.print(
                f"[red]Unknown GPU:[/] {gpu}\n"
                f"[dim]Valid options: {valid}[/]"
            )
            raise typer.Exit(1)
        return float(GPU_MEMORY[gpu_key]), "flag"

    # Auto-detect
    try:
        from soup_cli.utils.gpu import get_gpu_info

        info = get_gpu_info()
        mem_bytes = info.get("memory_total_bytes", 0)
        if mem_bytes > 0:
            return mem_bytes / (1024**3), "detected"
    except (ImportError, RuntimeError, OSError):
        pass

    # Nothing detected and no --gpu: a common consumer card, reported as assumed.
    return ASSUMED_GPU_MEMORY_GB, "assumed"


def _render_profile(result: dict, cfg, gpu_memory_gb: float, gpu_memory_source: str) -> None:
    """Render Rich profile output."""
    # Model info
    if result.get("peft") == "full":
        trainable_str = f"(trainable: {result['trainable_params']:,.0f}, full fine-tuning)"
    else:
        targets = ", ".join(result.get("lora_targets", []))
        trainable_str = (
            f"(trainable: {result['trainable_params']:,.0f} with LoRA r={cfg.training.lora.r}"
            + (f" on {for_terminal(targets)})" if targets else ")")
        )

    model_info = (
        f"Model:     [bold]{cfg.base}[/]\n"
        f"Params:    [bold]{result['model_params_b']:.1f}B[/] {trainable_str}\n"
        f"Quantization: [bold]{result['quantization']}[/]"
    )
    if result["gradient_checkpointing_tier"] != "off":
        model_info += (
            "\nGradient checkpointing: "
            f"[bold green]enabled[/] ({result['gradient_checkpointing_tier']})"
        )

    # Memory breakdown table
    mem_table = Table(show_header=False, box=None, padding=(0, 2))
    mem_table.add_column("Component", style="bold")
    mem_table.add_column("Memory", justify="right")
    mem_table.add_row("Model", f"~{result['model_memory_gb']:.1f} GB")
    if result.get("peft") != "full":
        mem_table.add_row("LoRA", f"~{result['lora_memory_gb']:.1f} GB")
    # A LoRA run's adapter gradients are usually too small to show.
    if result.get("peft") == "full" or result["gradients_memory_gb"] >= 0.05:
        mem_table.add_row("Gradients", f"~{result['gradients_memory_gb']:.1f} GB")
    mem_table.add_row("Optimizer", f"~{result['optimizer_memory_gb']:.1f} GB")
    mem_table.add_row(
        f"Activations (bs={result['batch_size']}, seq={result['seq_len']})",
        f"~{result['activation_memory_gb']:.1f} GB",
    )
    if result.get("vocab_size"):
        mem_table.add_row(
            f"Logits and loss (vocab {result['vocab_size']:,})",
            f"~{result['logits_memory_gb']:.1f} GB",
        )
    else:
        mem_table.add_row("Logits and loss", "not counted (vocabulary unknown)")
    if result.get("dequant_memory_gb"):
        mem_table.add_row("4-bit dequantization", f"~{result['dequant_memory_gb']:.1f} GB")
    mem_table.add_row("Overhead", f"~{result['overhead_gb']:.1f} GB")
    mem_table.add_row("-" * 20, "-" * 10)
    mem_table.add_row("[bold]Total[/]", f"[bold]~{result['total_memory_gb']:.1f} GB[/]")

    # Speed info. estimate_speed is an A100 lookup that does not depend on the GPU.
    speed_info = (
        f"Tokens/sec: ~{result['tokens_per_sec']:,.0f}\n"
        f"Samples/sec: ~{result['samples_per_sec']:.1f}\n"
        "[dim]Typical A100 throughput, not specific to your GPU.[/]"
    )

    # Recommendations
    recs = []
    fit = result["fit"]
    margin = f"{VRAM_SAFETY_MARGIN:.0%}"
    if gpu_memory_source == "assumed":
        verdict = {
            "fits": "would fit",
            "tight": f"would be tight (not with the train pre-flight's {margin} margin)",
        }.get(fit, "would NOT fit")
        recs.append(
            f"[yellow]?[/] No GPU detected: {verdict} in an assumed "
            f"{gpu_memory_gb:.0f} GB (need ~{result['total_memory_gb']:.0f} GB). "
            "Pass --gpu for a real verdict."
        )
    elif fit == "fits":
        recs.append(
            f"[green]OK[/] Fits in {gpu_memory_gb:.0f} GB VRAM"
        )
    elif fit == "tight":
        recs.append(
            f"[yellow]~[/] Tight: fits in {gpu_memory_gb:.0f} GB VRAM, but not with "
            f"the train pre-flight's {margin} margin "
            f"(need ~{result['total_memory_gb']:.1f} GB)"
        )
    else:
        recs.append(
            f"[red]X[/] Does NOT fit in {gpu_memory_gb:.0f} GB VRAM "
            f"(need ~{result['total_memory_gb']:.0f} GB)"
        )

    if gpu_memory_source == "assumed":
        recs.append(
            f"[yellow]?[/] batch_size {result['recommended_batch_size']} assumes "
            f"{gpu_memory_gb:.0f} GB VRAM"
        )
    else:
        recs.append(
            f"[green]OK[/] Recommended batch_size: {result['recommended_batch_size']}"
        )

    if result["total_memory_gb"] > 24 and result["gradient_checkpointing_tier"] == "off":
        recs.append(
            "[yellow]![/] Consider gradient_checkpointing: true for memory savings"
        )

    if result["total_memory_gb"] > 40:
        recs.append(
            "[yellow]![/] Consider DeepSpeed ZeRO-3 or FSDP for distributed training"
        )

    # Compatible GPUs (show top 5)
    gpu_list = result["compatible_gpus"][:5]

    console.print(Panel(model_info, title="[bold]Training Profile[/]"))
    console.print()
    console.print("[bold]GPU Memory Estimate:[/]")
    console.print(mem_table)
    for line in result.get("notes", []):
        console.print(f"[dim]{escape(line)}[/]")
    console.print()
    console.print(Panel(speed_info, title="Speed Estimate"))
    console.print()
    console.print(Panel("\n".join(recs), title="Recommendations"))
    console.print()

    if gpu_list:
        console.print("[bold]Compatible GPUs:[/]")
        for gpu_name in gpu_list:
            console.print(f"  - {gpu_name}")
