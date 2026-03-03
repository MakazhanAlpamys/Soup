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
    resume: str = typer.Option(
        None,
        "--resume",
        "-r",
        help="Resume from checkpoint: path to checkpoint dir, or 'auto' for latest",
    ),
    wandb: bool = typer.Option(
        False,
        "--wandb",
        help="Enable Weights & Biases logging",
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

    # --- Resolve resume checkpoint (fail fast before heavy operations) ---
    resume_from = None
    if resume:
        resume_from = _resolve_checkpoint(resume, cfg.output, cfg.experiment_name)
        if resume_from:
            console.print(f"[green]Resuming from:[/] {resume_from}")
        else:
            console.print("[red]No checkpoint found to resume from.[/]")
            raise typer.Exit(1)

    # --- W&B setup (fail fast if wandb not installed) ---
    if wandb:
        try:
            import wandb as _wandb  # noqa: F401

            console.print("[green]W&B logging enabled[/]")
        except ImportError:
            console.print(
                "[red]wandb not installed.[/]\n"
                "Run: [bold]pip install wandb[/]"
            )
            raise typer.Exit(1)

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

    # Start experiment tracking
    from soup_cli.experiment.tracker import ExperimentTracker

    tracker = ExperimentTracker()
    experiment_name = cfg.experiment_name or name
    run_id = tracker.start_run(
        config_dict=cfg.model_dump(),
        device=device,
        device_name=device_name,
        gpu_info=gpu_info,
        experiment_name=experiment_name,
    )
    console.print(f"[dim]Run ID: {run_id}[/]")

    # Build trainer based on task type
    report_to = "wandb" if wandb else "none"
    console.print("[dim]Setting up model + trainer...[/]")
    if cfg.task == "dpo":
        from soup_cli.trainer.dpo import DPOTrainerWrapper

        trainer_wrapper = DPOTrainerWrapper(cfg, device=device, report_to=report_to)
    else:
        trainer_wrapper = SFTTrainerWrapper(cfg, device=device, report_to=report_to)
    trainer_wrapper.setup(dataset)

    # Train with live display and experiment tracking
    display = TrainingDisplay(cfg, device_name=device_name)
    console.print("[bold green]Training started![/]\n")

    try:
        result = trainer_wrapper.train(
            display=display, tracker=tracker, run_id=run_id,
            resume_from_checkpoint=resume_from,
        )

        # Save completion to tracker
        tracker.finish_run(
            run_id=run_id,
            initial_loss=result["initial_loss"],
            final_loss=result["final_loss"],
            total_steps=result["total_steps"],
            duration_secs=result["duration_secs"],
            output_dir=result["output_dir"],
        )
    except Exception:
        tracker.fail_run(run_id)
        raise

    # Report
    console.print(
        Panel(
            f"Loss: [bold]{result['initial_loss']:.4f} → {result['final_loss']:.4f}[/]\n"
            f"Duration: [bold]{result['duration']}[/]\n"
            f"Output: [bold]{result['output_dir']}[/]\n"
            f"Run ID: [bold]{run_id}[/]\n\n"
            f"Quick test:  [bold]soup chat --model {result['output_dir']}[/]\n"
            f"Push to HF:  [bold]soup push --model {result['output_dir']}[/]\n"
            f"Merge LoRA:  [bold]soup merge --adapter {result['output_dir']}[/]\n"
            f"Export GGUF: [bold]soup export --model {result['output_dir']}[/]\n"
            f"Run details: [bold]soup runs show {run_id}[/]",
            title="[bold green]Training Complete![/]",
        )
    )


def _resolve_checkpoint(resume: str, output_dir: str, experiment_name: str = None) -> str:
    """Resolve the checkpoint path from --resume argument.

    If resume == "auto", find the latest checkpoint in the output directory.
    Otherwise, treat it as a direct path to a checkpoint directory.
    """
    if resume.lower() == "auto":
        base = Path(output_dir)
        if experiment_name:
            base = base / experiment_name

        if not base.exists():
            return None

        checkpoints = sorted(
            [d for d in base.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda d: int(d.name.split("-")[-1]) if d.name.split("-")[-1].isdigit() else 0,
        )
        if checkpoints:
            return str(checkpoints[-1])
        return None

    # Direct path
    checkpoint_path = Path(resume)
    if checkpoint_path.exists() and checkpoint_path.is_dir():
        return str(checkpoint_path)
    return None
