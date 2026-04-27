"""soup train — the main training command."""

from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape as markup_escape
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
        help=(
            "Enable DeepSpeed: zero2, zero3, zero2_offload, zero++ (ZeRO++), "
            "or path to config JSON"
        ),
    ),
    fsdp: str = typer.Option(
        None,
        "--fsdp",
        help="Enable FSDP2: full_shard, shard_grad, or full_offload",
    ),
    gpus: str = typer.Option(
        None,
        "--gpus",
        help="Number of GPUs for distributed training ('auto' or integer)",
    ),
    gate: str = typer.Option(
        None,
        "--gate",
        help=(
            "Enable eval-gated training with a suite file "
            "(shortcut for training.eval_gate.enabled=true + suite=<path>)"
        ),
    ),
    push_as: str = typer.Option(
        None,
        "--push-as",
        help=(
            "Auto-push each save_steps checkpoint to HF Hub as "
            "'checkpoint-<step>' branch of the given repo (e.g. user/my-model)"
        ),
    ),
    hf_resume: bool = typer.Option(
        False,
        "--hf-resume",
        help=(
            "Download the latest checkpoint branch from the --push-as repo "
            "and resume from it. Requires --push-as."
        ),
    ),
    find_lr: bool = typer.Option(
        False,
        "--find-lr",
        help=(
            "LR range finder (v0.32.0): run a short geometric LR sweep, write "
            "a JSON report with the recommended LR, then exit without training."
        ),
    ),
    find_lr_start: float = typer.Option(
        1e-7,
        "--find-lr-start",
        help="LR range finder: starting LR (default 1e-7)",
    ),
    find_lr_end: float = typer.Option(
        1e-1,
        "--find-lr-end",
        help="LR range finder: ending LR (default 1e-1)",
    ),
    find_lr_steps: int = typer.Option(
        100,
        "--find-lr-steps",
        help="LR range finder: number of sweep steps (default 100)",
    ),
    find_lr_output: str = typer.Option(
        "lr_finder.json",
        "--find-lr-output",
        help="LR range finder: JSON report path (default ./lr_finder.json)",
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

    # --- LR range finder fast path ---
    if find_lr:
        from soup_cli.utils.lr_finder import (
            compute_lr_schedule,
            save_lr_finder_report,
        )

        try:
            schedule = compute_lr_schedule(
                start_lr=find_lr_start,
                end_lr=find_lr_end,
                num_steps=find_lr_steps,
            )
        except ValueError as exc:
            console.print(f"[red]Invalid --find-lr range:[/] {exc}")
            raise typer.Exit(1) from exc
        # v0.33.0 #56: live LR-sweep training loop. Falls back to a
        # synthetic curve only when the real loop cannot run (no torch /
        # config load failure) so users still get a parseable report.
        losses_for_report = _run_live_lr_sweep_or_synth(
            config_path, schedule,
        )
        try:
            save_lr_finder_report(schedule, losses_for_report, find_lr_output)
        except ValueError as exc:
            console.print(f"[red]Invalid --find-lr-output:[/] {exc}")
            raise typer.Exit(1) from exc
        console.print(f"[green]LR finder report written to:[/] {find_lr_output}")
        raise typer.Exit(0)

    # Load & validate config
    console.print(f"[dim]Loading config from {config_path}...[/]")
    cfg = load_config(config_path)

    # --- --push-as / --hf-resume validation ---
    if push_as:
        from soup_cli.utils.hf import validate_repo_id

        try:
            validate_repo_id(push_as)
        except ValueError as exc:
            console.print(f"[red]Invalid --push-as repo id:[/] {exc}")
            raise typer.Exit(1) from exc
    if hf_resume and not push_as:
        console.print("[red]--hf-resume requires --push-as <repo>[/]")
        raise typer.Exit(1)

    # --- Eval-gate shortcut: --gate <path> sets training.eval_gate ---
    if gate:
        from soup_cli.config.schema import EvalGateConfig
        from soup_cli.eval.gate import load_suite

        try:
            # Validate the suite path up-front (path containment + parse).
            load_suite(gate)
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[red]Invalid --gate suite: {exc}[/]")
            raise typer.Exit(1) from exc
        cfg.training.eval_gate = EvalGateConfig(enabled=True, suite=gate)
        console.print(f"[green]Eval gate enabled[/] with suite: {gate}")

    # --- Resolve resume checkpoint (fail fast before heavy operations) ---
    resume_from = None
    if resume:
        resume_from = _resolve_checkpoint(resume, cfg.output, cfg.experiment_name)
        if resume_from:
            console.print(f"[green]Resuming from:[/] {resume_from}")
        else:
            console.print("[red]No checkpoint found to resume from.[/]")
            raise typer.Exit(1)

    # --- HF auto-resume: pull latest checkpoint branch into output dir ---
    if hf_resume and push_as and resume_from is None:
        from soup_cli.monitoring.hf_push import prepare_hf_resume
        from soup_cli.utils.hf import resolve_endpoint, resolve_token

        try:
            hf_endpoint = resolve_endpoint()
        except ValueError as exc:
            console.print(f"[red]--hf-resume: {exc}[/]")
            raise typer.Exit(1) from exc
        hf_token = resolve_token()
        if hf_token is None:
            console.print(
                "[yellow]--hf-resume: no HF token available; skipping auto-resume[/]"
            )
        else:
            local_ckpt = prepare_hf_resume(
                repo_id=push_as,
                output_dir=cfg.output,
                token=hf_token,
                endpoint=hf_endpoint,
            )
            if local_ckpt:
                resume_from = local_ckpt
                console.print(f"[green]Resumed from HF:[/] {local_ckpt}")
            else:
                console.print(
                    "[yellow]--hf-resume: no checkpoint branch found; starting fresh[/]"
                )

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

    # --- Multi-GPU topology + --gpus resolution ---
    num_gpus = None
    if gpus:
        from soup_cli.utils.topology import detect_topology, resolve_num_gpus

        try:
            num_gpus = resolve_num_gpus(gpus)
        except ValueError as exc:
            console.print(f"[red]Invalid --gpus:[/] {exc}")
            raise typer.Exit(1) from exc
        topo = detect_topology()
        if num_gpus is not None and num_gpus < 1:
            # --gpus auto on CPU / no-CUDA box — explicit, not silent.
            console.print(
                "[yellow]--gpus auto detected 0 GPUs; continuing as a "
                "single-process CPU run.[/]"
            )
        elif num_gpus is not None and num_gpus > 1:
            from soup_cli.utils.launcher import format_advice, is_in_distributed

            if not is_in_distributed():
                safe_config = markup_escape(config)
                console.print(
                    Panel(
                        markup_escape(
                            format_advice(num_gpus, ["soup", "train", "-c", safe_config])
                        ),
                        title="[yellow]Multi-GPU launch required[/]",
                    )
                )
                console.print(
                    f"[dim]Detected topology: {topo['gpu_count']} GPUs, "
                    f"{topo['interconnect']}[/]"
                )
                console.print(
                    "[dim]Note: carry any additional flags (e.g. --fsdp, "
                    "--deepspeed, --wandb) over to the accelerate command.[/]"
                )
                raise typer.Exit(1)
            console.print(
                f"[green]Distributed run detected[/] "
                f"({num_gpus} procs, {topo['interconnect']} interconnect)"
            )
            # Apply NCCL env hints. All current keys (``NCCL_P2P_DISABLE`` /
            # ``NCCL_IB_DISABLE`` / ``NCCL_NVLS_ENABLE``) are rank-idempotent
            # string literals so it is safe to run on every rank. If a
            # rank-sensitive key is ever added to ``suggest_nccl_env``, this
            # loop must be gated to ``LOCAL_RANK == 0``. ``setdefault`` keeps
            # user / launcher overrides winning over our suggestions.
            from soup_cli.utils.topology import suggest_nccl_env

            for key, val in suggest_nccl_env(
                gpu_count=num_gpus, interconnect=topo["interconnect"]
            ).items():
                os.environ.setdefault(key, val)

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

    # Validate FSDP2 + torch.compile (v0.27.0 Part D)
    if cfg.training.use_fsdp2_compile:
        from soup_cli.utils.fsdp import validate_fsdp2_compile_config

        compile_errors = validate_fsdp2_compile_config(
            use_compile=cfg.training.use_fsdp2_compile,
            fsdp_preset=fsdp,
            backend=cfg.backend,
            device=device,
            deepspeed_config=ds_config_path,
        )
        for err in compile_errors:
            console.print(f"[red]FSDP2 + torch.compile error:[/] {err}")
        if compile_errors:
            raise typer.Exit(1)

    # Validate pipeline parallelism (v0.27.0 Part F)
    if cfg.training.parallelism == "pipeline":
        from soup_cli.utils.pipeline import validate_pipeline_config

        pp_errors = validate_pipeline_config(
            parallelism=cfg.training.parallelism,
            pipeline_stages=cfg.training.pipeline_stages,
            device=device,
            gpu_count=gpu_info.get("gpu_count", 0),
        )
        for err in pp_errors:
            console.print(f"[red]Pipeline parallel error:[/] {err}")
        if pp_errors:
            raise typer.Exit(1)
        console.print(
            Panel(
                (
                    f"Pipeline parallelism is configured "
                    f"({cfg.training.pipeline_stages} stages) but live "
                    f"execution wiring ships in v0.27.1. Your config is "
                    f"validated and the trainer will run in data-parallel "
                    f"mode for now."
                ),
                title="[yellow]Pipeline parallelism (deferred execution)[/]",
                border_style="yellow",
            )
        )

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
    elif cfg.task == "embedding":
        from soup_cli.trainer.embedding import EmbeddingTrainerWrapper

        trainer_wrapper = EmbeddingTrainerWrapper(cfg, **trainer_kwargs)
    else:
        trainer_wrapper = SFTTrainerWrapper(cfg, **trainer_kwargs)
    trainer_wrapper.setup(dataset)

    # --- HF auto-push callback (Part B of v0.29.0) ---
    if push_as:
        from soup_cli.monitoring.hf_push import build_push_callback

        push_cb = build_push_callback(
            repo_id=push_as,
            output_dir=cfg.output,
            private=False,
        )
        if push_cb is None:
            console.print(
                "[yellow]--push-as: no HF token available; skipping auto-push[/]"
            )
        else:
            hf_trainer = getattr(trainer_wrapper, "trainer", None)
            if hf_trainer is not None and hasattr(hf_trainer, "add_callback"):
                hf_trainer.add_callback(push_cb)
                console.print(
                    f"[green]HF auto-push enabled[/] -> {push_as} "
                    "(one branch per save_steps)"
                )
            else:
                console.print(
                    "[yellow]--push-as: trainer does not expose add_callback; "
                    "auto-push disabled for this run[/]"
                )

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


def _run_live_lr_sweep_or_synth(
    config_path: str, schedule: list[float],
) -> list[float]:
    """v0.33.0 #56 — try to run an in-process LR sweep; fall back to a
    synthetic curve when prerequisites are missing.

    Falls back when:
      - torch / transformers / datasets are not importable
      - config load fails
      - dataset cannot be tokenized into a small in-memory loader
    The fallback curve descends 60% then diverges so the recommended-LR
    extraction in :func:`find_optimal_lr` still produces sensible output.
    """
    try:
        cfg = load_config(config_path)
    except Exception as exc:  # noqa: BLE001 — fall back rather than abort
        console.print(
            f"[yellow]--find-lr: config load failed ({exc}); "
            f"writing synthetic curve.[/]"
        )
        return _synth_lr_curve(len(schedule))

    try:
        return _live_lr_sweep_from_config(cfg, schedule)
    except Exception as exc:  # noqa: BLE001 — informative fallback
        console.print(
            f"[yellow]--find-lr: live sweep unavailable ({exc}); "
            f"writing synthetic curve.[/]"
        )
        return _synth_lr_curve(len(schedule))


def _synth_lr_curve(n: int) -> list[float]:
    descend_until = max(1, int(n * 0.6))
    out: list[float] = []
    for i in range(n):
        if i < descend_until:
            out.append(3.0 - 2.0 * (i / descend_until))
        else:
            tail = (i - descend_until) / max(1, n - descend_until)
            out.append(1.0 + 8.0 * tail * tail)
    return out


def _live_lr_sweep_from_config(cfg, schedule: list[float]) -> list[float]:
    """Build a tiny in-process loop: load model + tokenizer + a slice of
    the train dataset, then call :func:`run_lr_sweep`."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from soup_cli.data.loader import load_local
    from soup_cli.utils.lr_finder import run_lr_sweep

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base, trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base, trust_remote_code=False,
    ).to(device)
    model.train()

    dataset = load_local(cfg.data.train, cfg.data.format)
    rows = list(dataset)[: max(2, len(schedule))]
    if not rows:
        raise RuntimeError("training dataset is empty")

    def _tokenize(row):
        text = row.get("text") or row.get("prompt") or ""
        if not text and "messages" in row:
            text = " ".join(m.get("content", "") for m in row["messages"])
        enc = tokenizer(
            text or " ", return_tensors="pt", truncation=True,
            max_length=min(cfg.data.max_length or 256, 256),
            padding="max_length",
        )
        enc["labels"] = enc["input_ids"].clone()
        return {k: v.squeeze(0) for k, v in enc.items()}

    def _batched_loader():
        for row in rows:
            tok = _tokenize(row)
            yield {k: v.unsqueeze(0) for k, v in tok.items()}

    return run_lr_sweep(
        model=model,
        dataloader=_batched_loader(),
        schedule=schedule,
        optimizer_factory=lambda params: torch.optim.AdamW(params, lr=schedule[0]),
        device=device,
    )
