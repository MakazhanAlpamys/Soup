"""soup train — the main training command."""

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from soup_cli.config.loader import load_config
from soup_cli.data.loader import load_dataset
from soup_cli.monitoring.display import TrainingDisplay
from soup_cli.trainer.sft import SFTTrainerWrapper
from soup_cli.utils.gpu import detect_device, get_gpu_info

console = Console()


def train(
    config: str = typer.Option(
        "soup.yaml",
        "--config",
        "-c",
        help="Path to soup.yaml config file",
    ),
    name: str = typer.Option(
        None,
        "--name",
        "-n",
        help="Experiment name (auto-generated if not set)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate config and data without training",
    ),
):
    """Start training from a soup.yaml config."""
    config_path = Path(config)
    if not config_path.exists():
        console.print(f"[red]Config not found: {config_path}[/]")
        console.print("Run [bold]soup init[/] to create one.")
        raise typer.Exit(1)

    # Load & validate config
    console.print(f"[dim]Loading config from {config_path}...[/]")
    cfg = load_config(config_path)

    # Detect hardware
    device, device_name = detect_device()
    gpu_info = get_gpu_info()
    console.print(
        Panel(
            f"Device: [bold]{device_name}[/]\n"
            f"Memory: [bold]{gpu_info['memory_total']}[/]\n"
            f"Model:  [bold]{cfg.base}[/]\n"
            f"Task:   [bold]{cfg.task}[/]\n"
            f"LoRA:   [bold]r={cfg.training.lora.r}, alpha={cfg.training.lora.alpha}[/]\n"
            f"Quant:  [bold]{cfg.training.quantization}[/]",
            title="Training Setup",
        )
    )

    if dry_run:
        console.print("[yellow]Dry run — validating data...[/]")
        dataset = load_dataset(cfg.data)
        console.print(f"[green]Data OK:[/] {len(dataset['train'])} train samples")
        if "val" in dataset:
            console.print(f"[green]Val:[/] {len(dataset['val'])} samples")
        console.print("[green]Config valid. Ready to train![/]")
        raise typer.Exit()

    # Load data
    console.print("[dim]Loading dataset...[/]")
    dataset = load_dataset(cfg.data)
    console.print(f"[green]Loaded:[/] {len(dataset['train'])} train samples")

    # Build trainer based on task type
    console.print("[dim]Setting up model + trainer...[/]")
    if cfg.task == "dpo":
        from soup_cli.trainer.dpo import DPOTrainerWrapper

        trainer_wrapper = DPOTrainerWrapper(cfg, device=device)
    else:
        trainer_wrapper = SFTTrainerWrapper(cfg, device=device)
    trainer_wrapper.setup(dataset)

    # Train with live display
    display = TrainingDisplay(cfg, device_name=device_name)
    console.print("[bold green]Training started![/]\n")
    result = trainer_wrapper.train(display=display)

    # Report
    console.print(
        Panel(
            f"Loss: [bold]{result['initial_loss']:.4f} → {result['final_loss']:.4f}[/]\n"
            f"Duration: [bold]{result['duration']}[/]\n"
            f"Output: [bold]{result['output_dir']}[/]\n\n"
            f"Quick test:  [bold]soup chat --model {result['output_dir']}[/]\n"
            f"Push to HF:  [bold]soup push --model {result['output_dir']}[/]",
            title="[bold green]Training Complete![/]",
        )
    )
