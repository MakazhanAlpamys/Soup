"""soup autopilot — zero-config fine-tuning (Part H of v0.25.0)."""

from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from soup_cli.autopilot.analyzer import (
    analyze_dataset,
    analyze_hardware,
    analyze_model,
)
from soup_cli.autopilot.decisions import GOAL_TO_TASK, parse_gpu_budget
from soup_cli.autopilot.generate_config import build_soup_config, write_yaml

console = Console()


def _is_under_cwd(path: Path) -> bool:
    """Check whether ``path`` resolves inside the current working directory.

    Uses ``os.path.realpath`` for both sides instead of ``Path.resolve()``.
    On Windows + Python 3.9, ``Path.resolve()`` occasionally leaves 8.3 short
    names (e.g. ``C:\\Users\\RUNNER~1``) in one of the two paths but not the
    other, making ``relative_to`` fail even when the paths refer to the same
    location. ``realpath`` handles the short-name expansion consistently.
    """
    try:
        resolved = os.path.realpath(str(path))
        cwd = os.path.realpath(str(Path.cwd()))
    except (OSError, ValueError):
        return False
    if os.name == "nt":
        resolved = resolved.lower()
        cwd = cwd.lower()
    try:
        common = os.path.commonpath([resolved, cwd])
    except ValueError:
        return False
    return common == cwd


def autopilot_cmd(
    model: str = typer.Option(..., "--model", "-m", help="Base model (HF model id)"),
    data: str = typer.Option(..., "--data", "-d", help="Dataset path (JSONL)"),
    goal: str = typer.Option(
        ..., "--goal", "-g",
        help=(
            "Goal: chat | reasoning | code | classification | tool-calling | "
            "alignment | domain-adapt"
        ),
    ),
    gpu_budget: str = typer.Option(
        "", "--gpu-budget", help="VRAM budget, e.g. 24GB (default: auto-detect)",
    ),
    output: str = typer.Option(
        "soup.yaml", "--output", "-o", help="Output config path",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show decisions but don't write the config",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip confirmation prompts",
    ),
) -> None:
    """Autopilot: give model+data+goal, Soup picks optimal hyperparameters."""
    if goal not in GOAL_TO_TASK:
        console.print(
            f"[red]Unknown goal '{goal}'. "
            f"Options: {', '.join(sorted(GOAL_TO_TASK.keys()))}[/]"
        )
        raise typer.Exit(1)

    # Path traversal protection — data must stay under cwd
    data_path = Path(data)
    if not _is_under_cwd(data_path):
        console.print("[red]Data path must be under the current working directory.[/]")
        raise typer.Exit(1)
    data_path = Path(os.path.realpath(str(data_path)))

    if not data_path.exists():
        console.print(f"[red]Data file not found: {data_path}[/]")
        raise typer.Exit(1)

    # GPU budget
    vram_gb: float | None = None
    if gpu_budget:
        try:
            vram_gb = parse_gpu_budget(gpu_budget)
        except ValueError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(1)

    # Analyze
    console.print("[dim]Analyzing dataset, model, and hardware...[/]")
    try:
        dataset_profile = analyze_dataset(str(data_path))
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1)
    model_profile = analyze_model(model)
    hardware_profile = analyze_hardware()

    effective_vram = vram_gb if vram_gb is not None else max(hardware_profile.vram_gb, 8.0)

    try:
        cfg = build_soup_config(
            model=model,
            data_path=str(data_path),
            goal=goal,
            vram_gb=effective_vram,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1)

    # Render decision summary
    console.print(
        Panel(
            f"Dataset:    {data_path.name}\n"
            f"  format:   {dataset_profile.format}\n"
            f"  samples:  {dataset_profile.samples:,}\n"
            f"  tokens:   avg {dataset_profile.avg_tokens}, p95 {dataset_profile.p95_tokens}\n"
            f"  quality:  {dataset_profile.quality}\n\n"
            f"Model:      {model_profile.name}\n"
            f"  params:   {model_profile.params_b:.1f}B\n"
            f"  context:  {model_profile.context}\n\n"
            f"Hardware:   {hardware_profile.gpu_name}\n"
            f"  vram:     {hardware_profile.vram_gb:.1f}GB (budget: {effective_vram:.1f}GB)\n\n"
            f"Goal:       {goal} -> task: {cfg.task}",
            title="Soup Autopilot",
        )
    )
    console.print(
        Panel(
            f"Quantization: {cfg.training.quantization}\n"
            f"PEFT: LoRA r={cfg.training.lora.r}, alpha={cfg.training.lora.alpha}\n"
            f"Batch size: {cfg.training.batch_size} x grad_accum "
            f"{cfg.training.gradient_accumulation_steps}\n"
            f"Learning rate: {cfg.training.lr}\n"
            f"Epochs: {cfg.training.epochs}\n"
            f"Max length: {cfg.data.max_length}\n"
            f"Flash Attention: {cfg.training.use_flash_attn}\n"
            f"Liger Kernel: {cfg.training.use_liger}\n"
            f"Forgetting detection: {cfg.training.forgetting_detection}\n"
            f"Checkpoint intelligence: {cfg.training.checkpoint_intelligence}",
            title="Autopilot Decisions",
        )
    )

    if dry_run:
        console.print("[dim]--dry-run: config not written.[/]")
        return

    # Path traversal protection for output
    output_raw = Path(output)
    if not _is_under_cwd(output_raw):
        console.print("[red]Output path must be under the current working directory.[/]")
        raise typer.Exit(1)
    output_path = Path(os.path.realpath(str(output_raw)))

    if output_path.exists() and not yes:
        if not typer.confirm(f"{output_path} exists. Overwrite?"):
            raise typer.Exit()

    write_yaml(cfg, output_path)
    console.print(f"[green]Config written:[/] {output_path}")
