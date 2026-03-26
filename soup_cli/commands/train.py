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
    tensorboard: bool = typer.Option(
        False,
        "--tensorboard",
        help="Enable TensorBoard logging (logs to output_dir/runs/)",
    ),
    deepspeed: str = typer.Option(
        None,
        "--deepspeed",
        help="Enable DeepSpeed: zero2, zero3, zero2_offload, or path to config JSON",
    ),
    fsdp: str = typer.Option(
        None,
        "--fsdp",
        help="Enable FSDP2: fsdp_full_shard, fsdp_shard_grad, or fsdp_full_offload",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt",
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

    # --- Validate logging flags ---
    if wandb and tensorboard:
        console.print(
            "[red]Cannot use --wandb and --tensorboard together. Pick one.[/]"
        )
        raise typer.Exit(1)

    # --- TensorBoard setup ---
    if tensorboard:
        try:
            import tensorboard  # noqa: F401

            console.print("[green]TensorBoard logging enabled[/]")
        except ImportError:
            console.print(
                "[red]TensorBoard not installed.[/]\n"
                "Run: [bold]pip install tensorboard[/]"
            )
            raise typer.Exit(1)

    # --- W&B setup (fail fast if wandb not installed) ---
    if wandb:
        try:
            import wandb as _wandb  # noqa: F401

            console.print("[green]W&B logging enabled[/]")
        except ImportError:
            console.print(
                "[red]wandb not installed.[/]\n"
                "Run: [bold]pip install 'soup-cli[wandb]'[/]"
            )
            raise typer.Exit(1)
        except Exception as wandb_err:
            console.print(
                f"[red]wandb import error:[/] {wandb_err}\n"
                "Try: [bold]pip install 'wandb>=0.15.0,<0.18.0'[/]"
            )
            raise typer.Exit(1)

    # --- DeepSpeed setup ---
    ds_config_path = None
    if deepspeed:
        ds_config_path = _resolve_deepspeed(deepspeed)
        if ds_config_path:
            console.print(f"[green]DeepSpeed enabled:[/] {deepspeed}")

    # --- FSDP2 setup ---
    fsdp_kwargs = None
    if fsdp:
        from soup_cli.utils.fsdp import FSDP_CONFIGS, get_fsdp_training_args

        if fsdp not in FSDP_CONFIGS:
            console.print(
                f"[red]Invalid FSDP preset: {fsdp}[/]\n"
                f"Options: {', '.join(FSDP_CONFIGS.keys())}"
            )
            raise typer.Exit(1)
        fsdp_kwargs = get_fsdp_training_args(fsdp)
        console.print(f"[green]FSDP2 enabled:[/] {fsdp}")

    # Detect hardware
    device, device_name = detect_device()
    gpu_info = get_gpu_info()

    # Auto-disable quantization on CPU (bitsandbytes doesn't support CPU)
    if device == "cpu" and cfg.training.quantization in ("4bit", "8bit"):
        console.print(
            f"[yellow]Warning: {cfg.training.quantization} quantization is not "
            "supported on CPU. Switching to quantization: none.[/]"
        )
        cfg.training.quantization = "none"

    backend_label = cfg.backend
    if cfg.backend == "unsloth":
        backend_label = "unsloth [green](fast mode)[/]"

    quant_label = cfg.training.quantization
    if cfg.training.quantization_aware:
        quant_label += " + QAT"

    console.print(
        Panel(
            f"Device:  [bold]{device_name}[/]\n"
            f"Memory:  [bold]{gpu_info['memory_total']}[/]\n"
            f"Model:   [bold]{cfg.base}[/]\n"
            f"Task:    [bold]{cfg.task}[/]\n"
            f"Backend: [bold]{backend_label}[/]\n"
            f"LoRA:    [bold]r={cfg.training.lora.r}, alpha={cfg.training.lora.alpha}[/]\n"
            f"Quant:   [bold]{quant_label}[/]",
            title="Training Setup",
        )
    )

    # Validate GaLore configuration
    if cfg.training.use_galore:
        from soup_cli.utils.galore import validate_galore_config

        galore_errors = validate_galore_config(
            cfg.training.use_galore, cfg.training.quantization, cfg.backend,
        )
        for err in galore_errors:
            console.print(f"[red]GaLore error:[/] {err}")
        if galore_errors:
            raise typer.Exit(1)

    # Validate QAT configuration
    if cfg.training.quantization_aware:
        from soup_cli.utils.qat import validate_qat_config

        qat_errors = validate_qat_config(
            cfg.training.quantization, cfg.backend, cfg.modality,
        )
        for err in qat_errors:
            console.print(f"[red]QAT error:[/] {err}")
        if qat_errors:
            raise typer.Exit(1)

    # Validate FSDP configuration
    if fsdp:
        from soup_cli.utils.fsdp import validate_fsdp_config

        fsdp_errors = validate_fsdp_config(
            fsdp_preset=fsdp,
            deepspeed_config=ds_config_path,
            backend=cfg.backend,
            device=device,
        )
        for err in fsdp_errors:
            console.print(f"[red]FSDP error:[/] {err}")
        if fsdp_errors:
            raise typer.Exit(1)

    # Validate Liger Kernel configuration
    if cfg.training.use_liger:
        from soup_cli.utils.liger import validate_liger_config

        liger_errors = validate_liger_config(
            cfg.training.use_liger, cfg.backend, device,
        )
        for err in liger_errors:
            console.print(f"[red]Liger error:[/] {err}")
        if liger_errors:
            raise typer.Exit(1)

    # Validate FlashAttention configuration
    if cfg.training.use_flash_attn:
        from soup_cli.utils.flash_attn import validate_flash_attn_config

        fa_errors = validate_flash_attn_config(
            cfg.training.use_flash_attn, cfg.backend, device,
        )
        for err in fa_errors:
            console.print(f"[red]FlashAttention error:[/] {err}")
        if fa_errors:
            raise typer.Exit(1)

    # Validate Ring FlashAttention configuration
    if cfg.training.use_ring_attention:
        from soup_cli.utils.ring_attention import validate_ring_attention_config

        ring_errors = validate_ring_attention_config(
            cfg.training.use_ring_attention, device, cfg.data.max_length,
        )
        for err in ring_errors:
            console.print(f"[red]Ring Attention error:[/] {err}")
        if ring_errors:
            raise typer.Exit(1)

    # Validate long-context configuration
    if cfg.training.rope_scaling_type:
        from soup_cli.utils.long_context import validate_long_context_config

        ctx_errors = validate_long_context_config(
            cfg.data.max_length,
            cfg.training.rope_scaling_type,
            cfg.training.gradient_checkpointing,
        )
        for err in ctx_errors:
            console.print(f"[yellow]Long-context warning:[/] {err}")

    # Suggest unsloth if available but not being used
    if cfg.backend == "transformers":
        from soup_cli.utils.unsloth import is_unsloth_available

        if is_unsloth_available():
            console.print(
                "[dim]Tip: unsloth is installed. Add [bold]backend: unsloth[/dim]"
                "[dim] to soup.yaml for 2-5x faster training.[/]"
            )

    if not dry_run and not yes:
        if not typer.confirm("Start training?", default=True):
            console.print("[yellow]Cancelled.[/]")
            raise typer.Exit()

    if dry_run:
        console.print("[yellow]Dry run - validating data...[/]")
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
    if wandb:
        report_to = "wandb"
    elif tensorboard:
        report_to = "tensorboard"
    else:
        report_to = "none"
    console.print("[dim]Setting up model + trainer...[/]")
    trainer_kwargs = {
        "device": device,
        "report_to": report_to,
        "deepspeed_config": ds_config_path,
        "fsdp_config": fsdp_kwargs,
    }
    if cfg.task == "dpo":
        from soup_cli.trainer.dpo import DPOTrainerWrapper

        trainer_wrapper = DPOTrainerWrapper(cfg, **trainer_kwargs)
    elif cfg.task == "grpo":
        from soup_cli.trainer.grpo import GRPOTrainerWrapper

        trainer_wrapper = GRPOTrainerWrapper(cfg, **trainer_kwargs)
    elif cfg.task == "ppo":
        from soup_cli.trainer.ppo import PPOTrainerWrapper

        trainer_wrapper = PPOTrainerWrapper(cfg, **trainer_kwargs)
    elif cfg.task == "kto":
        from soup_cli.trainer.kto import KTOTrainerWrapper

        trainer_wrapper = KTOTrainerWrapper(cfg, **trainer_kwargs)
    elif cfg.task == "orpo":
        from soup_cli.trainer.orpo import ORPOTrainerWrapper

        trainer_wrapper = ORPOTrainerWrapper(cfg, **trainer_kwargs)
    elif cfg.task == "simpo":
        from soup_cli.trainer.simpo import SimPOTrainerWrapper

        trainer_wrapper = SimPOTrainerWrapper(cfg, **trainer_kwargs)
    elif cfg.task == "ipo":
        from soup_cli.trainer.ipo import IPOTrainerWrapper

        trainer_wrapper = IPOTrainerWrapper(cfg, **trainer_kwargs)
    elif cfg.task == "reward_model":
        from soup_cli.trainer.reward_model import RewardModelTrainerWrapper

        trainer_wrapper = RewardModelTrainerWrapper(cfg, **trainer_kwargs)
    elif cfg.task == "pretrain":
        from soup_cli.trainer.pretrain import PretrainTrainerWrapper

        trainer_wrapper = PretrainTrainerWrapper(cfg, **trainer_kwargs)
    else:
        trainer_wrapper = SFTTrainerWrapper(cfg, **trainer_kwargs)
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
            f"Loss: [bold]{result['initial_loss']:.4f} -> {result['final_loss']:.4f}[/]\n"
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


def _resolve_deepspeed(deepspeed: str) -> str:
    """Resolve DeepSpeed config: named preset or path to JSON file."""
    from soup_cli.utils.deepspeed import CONFIGS, write_deepspeed_config

    # Named preset
    if deepspeed in CONFIGS:
        return write_deepspeed_config(deepspeed)

    # Path to config file
    ds_path = Path(deepspeed)
    if ds_path.exists() and ds_path.suffix == ".json":
        return str(ds_path)

    console.print(
        f"[red]Invalid DeepSpeed config: {deepspeed}[/]\n"
        f"Options: {', '.join(CONFIGS.keys())} or path to JSON file."
    )
    raise typer.Exit(1)


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
