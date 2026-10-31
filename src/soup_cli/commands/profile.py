"""soup profile — estimate memory, speed, and GPU requirements before training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from soup_cli.utils.profiler import (
    GPU_MEMORY,
    MLX_WORKING_SET_SOURCE,
    MODEL_QUERY_CHECKPOINTING,
    MODEL_QUERY_METHODS,
    ModelShapeUnavailableError,
    detected_gpu_memory,
    finish_profile,
    normalize_gpu_key,
    profile_from_config,
    profile_from_model,
)
from soup_cli.utils.terminal import for_terminal
from soup_cli.utils.vram_estimate import VRAM_SAFETY_MARGIN

console = Console()

_DEFAULT_CONFIG = "soup.yaml"


def profile(
    config: Optional[str] = typer.Option(
        None, "--config", "-c",
        help=f"Path to soup.yaml config file (default {_DEFAULT_CONFIG})",
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
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help=(
            "A Hugging Face model id or a local model directory, priced without a "
            "soup.yaml from its config.json (works offline once cached)."
        ),
    ),
    seq: Optional[int] = typer.Option(
        None, "--seq", min=64, max=1_048_576,
        help="With --model: sequence length (default 2048)",
    ),
    batch: Optional[int] = typer.Option(
        None, "--batch", min=1, max=1024, help="With --model: batch size (default 4)",
    ),
    method: Optional[str] = typer.Option(
        None, "--method", help="With --model: lora, qlora (default) or full",
    ),
    lora_r: Optional[int] = typer.Option(
        None, "--lora-r", min=1, max=4096, help="With --model: LoRA rank (default 64)",
    ),
    optimizer: Optional[str] = typer.Option(
        None, "--optimizer", help="With --model: optimizer (default adamw_torch)",
    ),
    gradient_checkpointing: Optional[str] = typer.Option(
        None, "--gradient-checkpointing",
        help="With --model: off (default), full, medium, selective or auto",
    ),
):
    """Estimate memory, speed, and GPU requirements BEFORE training."""
    if model is not None:
        if config is not None:
            console.print("[red]--model and --config cannot be used together.[/]")
            raise typer.Exit(1)
        method = _choice("--method", method, "qlora", MODEL_QUERY_METHODS)
        if method == "full" and lora_r is not None:
            console.print(
                "[red]--lora-r only applies with --method lora or qlora;[/] a full "
                "fine-tune trains no adapters."
            )
            raise typer.Exit(1)
        _profile_model(
            model,
            gpu=gpu,
            json_output=json_output,
            seq=2048 if seq is None else seq,
            batch=4 if batch is None else batch,
            method=method,
            lora_r=64 if lora_r is None else lora_r,
            optimizer=_choice("--optimizer", optimizer, "adamw_torch", _optimizers()),
            gradient_checkpointing=_choice(
                "--gradient-checkpointing", gradient_checkpointing, "off",
                MODEL_QUERY_CHECKPOINTING,
            ),
        )
        return
    query = {
        "--seq": seq,
        "--batch": batch,
        "--method": method,
        "--lora-r": lora_r,
        "--optimizer": optimizer,
        "--gradient-checkpointing": gradient_checkpointing,
    }
    for flag, value in query.items():
        if value is not None:
            console.print(
                f"[red]{flag} only applies with --model;[/] with a config the run "
                "comes from soup.yaml."
            )
            raise typer.Exit(1)

    from soup_cli.config.loader import load_config

    config = config or _DEFAULT_CONFIG
    config_path = Path(config)
    if not config_path.exists():
        console.print(f"[red]Config file not found:[/] {for_terminal(config)}")
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
    finish_profile(
        result,
        seq_len=cfg.data.max_length,
        gpu_memory_gb=gpu_memory_gb,
        gpu_memory_source=gpu_memory_source,
    )
    _emit(result, json_output, gpu_memory_gb, gpu_memory_source)


def _optimizers() -> frozenset:
    from soup_cli.utils.hardware_fit import _VALID_OPTIMIZERS

    return _VALID_OPTIMIZERS


def _choice(flag: str, value: Optional[str], default: str, valid) -> str:
    """``value`` if it is one of ``valid`` (a closed set, safe to list), else exit 1."""
    if value is None:
        return default
    if value not in valid:
        console.print(
            f"[red]Unknown {flag}:[/] {for_terminal(value)}\n"
            f"[dim]Valid options: {', '.join(sorted(valid))}[/]"
        )
        raise typer.Exit(1)
    return value


def _profile_model(
    model: str,
    *,
    gpu: Optional[str],
    json_output: bool,
    seq: int,
    batch: int,
    method: str,
    lora_r: int,
    optimizer: str,
    gradient_checkpointing: str,
) -> None:
    """``soup profile --model``: the config's estimate, from the model alone (#2049)."""
    gpu_memory_gb, gpu_memory_source = _resolve_gpu_memory(gpu)
    try:
        result = profile_from_model(
            model,
            method=method,
            seq_len=seq,
            batch_size=batch,
            lora_r=lora_r,
            optimizer=optimizer,
            gradient_checkpointing=gradient_checkpointing,
            gpu_memory_gb=None if gpu_memory_source == "assumed" else gpu_memory_gb,
        )
    except ModelShapeUnavailableError as exc:
        console.print(
            f"[red]Cannot read {for_terminal(model)}:[/] {exc}. Nothing was estimated: "
            "--model needs the model's own config.json, not a size guessed from its name."
        )
        raise typer.Exit(1) from None
    except (ValueError, TypeError):
        console.print(
            "[red]Cannot estimate this run:[/] the model's size (over 1,000B "
            "parameters) is outside the range the estimate supports."
        )
        raise typer.Exit(1) from None
    finish_profile(
        result,
        seq_len=seq,
        gpu_memory_gb=gpu_memory_gb,
        gpu_memory_source=gpu_memory_source,
    )
    _emit(result, json_output, gpu_memory_gb, gpu_memory_source)


def _emit(result: dict, json_output: bool, gpu_memory_gb: float, gpu_memory_source: str) -> None:
    if json_output:
        # Machine-readable JSON: raw stdout, not console.print — Rich folds
        # long lines and parses ``[...]`` as markup when piped (issue #1468).
        typer.echo(json.dumps(result, indent=2))
        return

    _render_profile(result, gpu_memory_gb, gpu_memory_source)


def _resolve_gpu_memory(gpu: str | None) -> tuple[float, str]:
    """Resolve GPU memory in GB, and where the number came from.

    The source is ``"flag"``, ``"detected"``, ``"mlx working set"`` or
    ``"assumed"``; an assumed value must not be presented as a measured fit.
    """
    if gpu is not None:
        gpu_key = normalize_gpu_key(gpu)
        if gpu_key not in GPU_MEMORY:
            valid = ", ".join(sorted(GPU_MEMORY.keys()))
            console.print(
                f"[red]Unknown GPU:[/] {for_terminal(gpu)}\n"
                f"[dim]Valid options: {valid}[/]"
            )
            raise typer.Exit(1)
        return float(GPU_MEMORY[gpu_key]), "flag"

    return detected_gpu_memory()


def _render_profile(result: dict, gpu_memory_gb: float, gpu_memory_source: str) -> None:
    """Render Rich profile output."""
    # Model info
    if result.get("peft") == "full":
        trainable_str = f"(trainable: {result['trainable_params']:,.0f}, full fine-tuning)"
    else:
        targets = ", ".join(result.get("lora_targets", []))
        trainable_str = (
            f"(trainable: {result['trainable_params']:,.0f} with LoRA r={result['lora_r']}"
            + (f" on {for_terminal(targets)})" if targets else ")")
        )

    model_info = (
        f"Model:     [bold]{for_terminal(result['model_name'])}[/]\n"
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
    if result.get("optimizer_step_memory_gb"):
        mem_table.add_row(
            "Optimizer step (over the pass)", f"~{result['optimizer_step_memory_gb']:.1f} GB"
        )
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
    # On Apple Silicon the limit is Metal's working set, not a card's VRAM.
    where = "of Metal's working set" if gpu_memory_source == MLX_WORKING_SET_SOURCE else "VRAM"
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
            f"[green]OK[/] Fits in {gpu_memory_gb:.0f} GB {where}"
        )
    elif fit == "tight":
        recs.append(
            f"[yellow]~[/] Tight: fits in {gpu_memory_gb:.0f} GB {where}, but not with "
            f"the train pre-flight's {margin} margin "
            f"(need ~{result['total_memory_gb']:.1f} GB)"
        )
    else:
        recs.append(
            f"[red]X[/] Does NOT fit in {gpu_memory_gb:.0f} GB {where} "
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
