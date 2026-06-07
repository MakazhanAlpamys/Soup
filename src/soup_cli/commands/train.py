"""soup train — the main training command."""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.markup import escape as markup_escape
from rich.panel import Panel

from soup_cli.config.loader import load_config
from soup_cli.data.loader import load_dataset
from soup_cli.monitoring.display import TrainingDisplay
from soup_cli.trainer.sft import SFTTrainerWrapper
from soup_cli.utils.gpu import detect_device, get_gpu_info

if TYPE_CHECKING:  # pragma: no cover - type hints only, no runtime import
    from soup_cli.utils.energy import EnergyMeasurement

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
    tracker: str = typer.Option(
        None,
        "--tracker",
        help=(
            "Experiment tracker: mlflow / swanlab / trackio (v0.43.0). "
            "Mutually exclusive with --wandb / --tensorboard."
        ),
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
    no_reexec: bool = typer.Option(
        False,
        "--no-reexec",
        help=(
            "When --gpus N>1, print the accelerate launch command instead "
            "of auto-reexec under it (v0.33.0 #37 default behaviour: reexec)"
        ),
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
    trust_remote_code: bool = typer.Option(
        False,
        "--trust-remote-code",
        help=(
            "Allow loading models that ship custom Python via auto_map. "
            "Default deny (v0.36.0). Only enable if you trust the source."
        ),
    ),
    echo_trap_tokenizer_aware: bool = typer.Option(
        False,
        "--echo-trap-tokenizer-aware",
        help=(
            "Use tokenizer-id n-grams for echo-trap scoring. Requires "
            "training.echo_trap_enabled=true on grpo/ppo."
        ),
    ),
    profile_run: bool = typer.Option(
        False,
        "--profile",
        help=(
            "Record a torch.profiler trace (Chrome trace JSON) during early "
            "training steps. Output: <output>/profiles/<run_id>.trace.json"
        ),
    ),
    diagnose_gate: str = typer.Option(
        None,
        "--diagnose-gate",
        help=(
            "After training, run `soup diagnose` against the supplied evidence "
            "JSON (or scratch evidence). Refuses to mark the run successful "
            "if any of the 6 v0.56.0 failure modes returns MAJOR."
        ),
    ),
    annex_xi: str = typer.Option(
        None,
        "--annex-xi",
        help=(
            "After training, render an EU AI Act Annex XI/XII auto-doc to the "
            "given output path (cwd-contained). Markdown body now; PDF in v0.59.1."
        ),
    ),
    repro_receipt: str = typer.Option(
        None,
        "--repro-receipt",
        help=(
            "After training, write an SR 11-7-style reproducibility receipt "
            "(seeds + kernel versions + GPU + OS) to the given path. v0.59.0."
        ),
    ),
    capture_activations: str = typer.Option(
        None,
        "--capture-activations",
        help=(
            "After training, capture residual-stream activations at the named "
            "decoder layer (e.g. model.layers.5) on --capture-prompts and write "
            "them to <output>/activations/activations.json for soup probe "
            "sae-diff / sleeper. v0.71.8 #219."
        ),
    ),
    capture_prompts: str = typer.Option(
        None,
        "--capture-prompts",
        help=(
            "JSONL (or .txt) of prompts to run for --capture-activations "
            "(one prompt per line; 'prompt'/'text' field or raw text)."
        ),
    ),
    track_energy: bool = typer.Option(
        False,
        "--track-energy",
        help=(
            "Measure the training window's energy + CO2 via codecarbon "
            "(offline; requires `pip install soup-cli[carbon]`). Feeds the "
            "kWh / CO2 into --annex-xi. v0.71.3."
        ),
    ),
    energy_country: str = typer.Option(
        "USA",
        "--energy-country",
        help=(
            "ISO 3166-1 alpha-3 country code for the CO2 grid-intensity "
            "estimate used by --track-energy (default USA)."
        ),
    ),
    energy_out: str = typer.Option(
        None,
        "--energy-out",
        help=(
            "Write the --track-energy measurement to this JSON file (cwd-"
            "contained) so `soup bom emit --energy <path>` can consume it. "
            "v0.71.15."
        ),
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

    # --- RA-DIT generator-stage auto-link (v0.71.10 #200) ---
    # When a generator stage has no retriever model set, splice in the latest
    # RA-DIT retriever output from the Registry. A manual value always wins.
    if getattr(cfg.training, "ra_dit_stage", None) == "generator":
        from soup_cli.utils.ra_dit_run import autolink_generator_retriever

        advisory = autolink_generator_retriever(cfg)
        if advisory:
            # `advisory` embeds a Registry-derived `output` path — escape it
            # before printing into the Rich-markup console (security MEDIUM).
            console.print(f"[yellow]RA-DIT:[/] {markup_escape(advisory)}")

    # --- Echo-trap tokenizer-aware shortcut ---
    if echo_trap_tokenizer_aware:
        if not cfg.training.echo_trap_enabled:
            console.print(
                "[red]--echo-trap-tokenizer-aware requires "
                "training.echo_trap_enabled=true[/]"
            )
            raise typer.Exit(1)
        cfg.training.echo_trap_tokenizer_aware = True
        console.print("[green]Echo-trap tokenizer-aware scoring enabled[/]")

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

    # --- v0.38.0 Quant Menu × multi-GPU compatibility check ---
    from soup_cli.utils.quant_menu import check_quant_distributed_compat

    quant_problems = check_quant_distributed_compat(
        quantization=cfg.training.quantization,
        deepspeed=deepspeed,
        fsdp=bool(fsdp),
        bnb_4bit_quant_storage=cfg.training.bnb_4bit_quant_storage,
    )
    if quant_problems:
        hard = [p for p in quant_problems if not p.lower().startswith("warning")]
        warn = [p for p in quant_problems if p.lower().startswith("warning")]
        for problem in hard:
            console.print(f"[red]Quant compat:[/] {problem}")
        for problem in warn:
            console.print(f"[yellow]{problem}[/]")
        if hard:
            raise typer.Exit(1)

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
            from soup_cli.utils.launcher import (
                build_accelerate_argv,
                format_advice,
                is_in_distributed,
            )

            if not is_in_distributed():
                # v0.33.0 #37 — auto-reexec under accelerate launch unless
                # --no-reexec was passed. Reexec uses os.execvp so the new
                # accelerate process replaces this process; no leftover PID
                # tree, stdio passes through unchanged.
                if no_reexec:
                    safe_config = markup_escape(config)
                    console.print(
                        Panel(
                            markup_escape(
                                format_advice(
                                    num_gpus,
                                    ["soup", "train", "-c", safe_config],
                                )
                            ),
                            title="[yellow]Multi-GPU launch required[/]",
                        )
                    )
                    console.print(
                        f"[dim]Detected topology: {topo['gpu_count']} GPUs, "
                        f"{topo['interconnect']}[/]"
                    )
                    raise typer.Exit(1)

                # Reconstruct argv. Pass through critical flags so the
                # reexec'd run sees what the user typed.
                script_args: list[str] = [
                    sys.executable, "-m", "soup_cli.cli", "train",
                    "--config", config, "--no-reexec",
                ]
                if fsdp:
                    script_args.extend(["--fsdp", fsdp])
                if deepspeed:
                    script_args.extend(["--deepspeed", deepspeed])
                if resume:
                    script_args.extend(["--resume", resume])
                if wandb:
                    script_args.append("--wandb")
                if tensorboard:
                    script_args.append("--tensorboard")
                if echo_trap_tokenizer_aware:
                    script_args.append("--echo-trap-tokenizer-aware")
                if yes:
                    script_args.append("--yes")
                argv = build_accelerate_argv(
                    num_processes=num_gpus, script_args=script_args,
                )
                console.print(
                    f"[green]Auto-reexec under accelerate "
                    f"({num_gpus} GPUs, {topo['interconnect']})[/]"
                )
                console.print(f"[dim]argv: {' '.join(argv)}[/]")
                # os.execvp replaces the current process — does not return.
                # On Windows execvp creates a new process and returns the
                # child's return code; we don't loop because the parent
                # also exits via Typer.
                try:
                    os.execvp(argv[0], argv)
                except OSError as exc:
                    console.print(
                        f"[red]accelerate launch failed:[/] {exc}\n"
                        "Use [bold]--no-reexec[/] to fall back to printing "
                        "the launch command for manual execution."
                    )
                    raise typer.Exit(1) from exc
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

    # v0.53.2 review-fix: classifier-family tasks train a sequence-classification
    # head, not a causal-LM LoRA — render "head" instead of LoRA r/alpha.
    classifier_family = ("classifier", "reranker", "cross_encoder")
    if cfg.task in classifier_family:
        # v0.71.12 #146 — render BOTH the head line AND a LoRA line when the
        # opt-in classifier LoRA path is active.
        head_line = (
            f"Head:    [bold]num_labels={cfg.training.num_labels}, "
            f"kind={cfg.training.classifier_kind}[/]"
        )
        if getattr(cfg.training, "classifier_lora", False) and cfg.training.lora.r > 0:
            peft_line = (
                head_line
                + f"\nLoRA:    [bold]r={cfg.training.lora.r}, "
                + f"alpha={cfg.training.lora.alpha} (SEQ_CLS)[/]"
            )
        else:
            peft_line = head_line
    else:
        peft_line = (
            f"LoRA:    [bold]r={cfg.training.lora.r}, "
            f"alpha={cfg.training.lora.alpha}[/]"
        )
    console.print(
        Panel(
            f"Device:  [bold]{device_name}[/]\n"
            f"Memory:  [bold]{gpu_info['memory_total']}[/]\n"
            f"Model:   [bold]{cfg.base}[/]\n"
            f"Task:    [bold]{cfg.task}[/]\n"
            f"Backend: [bold]{backend_label}[/]\n"
            f"{peft_line}\n"
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

    # Capture the --tracker CLI value BEFORE the local ExperimentTracker
    # shadows it (v0.43.0 review fix — name-collision regression).
    tracker_backend = tracker

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
    from soup_cli.utils.trackers import resolve_report_to

    try:
        report_to = resolve_report_to(
            wandb=wandb, tensorboard=tensorboard, tracker=tracker_backend
        )
    except ValueError as exc:
        from rich.markup import escape as _esc

        console.print(f"[red]{_esc(str(exc))}[/]")
        raise typer.Exit(code=2) from exc
    console.print("[dim]Setting up model + trainer...[/]")
    # v0.53.8 #130 — pre-fetch model from non-HF hub into a local cache and
    # rewrite cfg.base to point at the local snapshot. The trainer wrappers
    # still use transformers.from_pretrained, which reads HF Hub by default;
    # by snapshotting first we keep every wrapper unchanged.
    hub_name = getattr(cfg.training, "hub", "hf") or "hf"
    if hub_name != "hf":
        import re as _re

        from rich.markup import escape as _markup_escape

        from soup_cli.utils.hubs import download_repo
        from soup_cli.utils.paths import is_under_cwd

        # Sanitise cache subdir name — strip every path-separator and
        # `..` segment so a crafted ``base: ../../etc`` cannot escape the
        # cache root (Windows ``\\`` and POSIX ``/`` both blocked).
        safe_slug = _re.sub(r"[^A-Za-z0-9._-]+", "__", cfg.base).strip("._-") or "model"
        cache_dir = (Path.cwd() / ".soup_hub_cache" / safe_slug).resolve()
        if not is_under_cwd(str(cache_dir)):
            console.print(
                "[red]Resolved hub cache dir escaped the current working "
                "directory; refusing to download.[/]"
            )
            raise typer.Exit(code=1)
        try:
            # Idempotency: if the cache dir already has a config.json, skip
            # the re-download (modelscope/openmind-hub also short-circuit on
            # match but having an explicit probe lets us print a clear hint).
            existing_cfg = cache_dir / "config.json"
            if existing_cfg.is_file():
                local_path = str(cache_dir)
                console.print(
                    f"[dim]Using cached snapshot at {local_path}[/]"
                )
            else:
                local_path = download_repo(
                    hub_name,
                    cfg.base,
                    local_dir=str(cache_dir),
                )
                console.print(
                    f"[dim]Fetched {cfg.base} from hub={hub_name} → "
                    f"{local_path}[/]"
                )
            # Use ``model_copy(update=...)`` so the Pydantic field
            # validators on ``base`` rerun (matches v0.33.0 #47 / v0.40.0
            # Part B immutability policy).
            cfg = cfg.model_copy(update={"base": local_path})
        except ImportError as exc:
            console.print(f"[red]{_markup_escape(str(exc))}[/]")
            raise typer.Exit(code=1) from exc

    # v0.53.8 #89 — friendly missing-dep advisory for `--tracker <name>`.
    if report_to and report_to not in ("none", "wandb", "tensorboard"):
        from soup_cli.utils.trackers import tracker_missing_dep_message

        msg = tracker_missing_dep_message(report_to)
        if msg:
            console.print(f"[yellow]{msg}[/]")
    trainer_kwargs = {
        "device": device,
        "report_to": report_to,
        "deepspeed_config": ds_config_path,
        "fsdp_config": fsdp_kwargs,
    }
    # v0.40.4 #63 — every transformer-backend trainer now threads
    # --trust-remote-code through the wrapper (closes the v0.36.0 Part B gap).
    trainer_kwargs = dict(trainer_kwargs, trust_remote_code=trust_remote_code)
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
    elif cfg.task == "bco":
        from soup_cli.trainer.bco import BCOTrainerWrapper

        trainer_wrapper = BCOTrainerWrapper(cfg, **trainer_kwargs)
    elif cfg.task == "preference":
        from soup_cli.trainer.preference import PreferenceTrainerWrapper

        trainer_wrapper = PreferenceTrainerWrapper(cfg, **trainer_kwargs)
    elif cfg.task == "reward_model":
        from soup_cli.trainer.reward_model import RewardModelTrainerWrapper

        trainer_wrapper = RewardModelTrainerWrapper(cfg, **trainer_kwargs)
    elif cfg.task == "pretrain":
        from soup_cli.trainer.pretrain import PretrainTrainerWrapper

        trainer_wrapper = PretrainTrainerWrapper(cfg, **trainer_kwargs)
    elif cfg.task == "embedding":
        from soup_cli.trainer.embedding import EmbeddingTrainerWrapper

        trainer_wrapper = EmbeddingTrainerWrapper(cfg, **trainer_kwargs)
    elif cfg.task == "distill":
        # v0.53.2 #133 — knowledge distillation (student + frozen teacher).
        from soup_cli.trainer.distill import DistillTrainerWrapper

        trainer_wrapper = DistillTrainerWrapper(cfg, **trainer_kwargs)
    elif cfg.task == "prm":
        # v0.53.11 #126 — Process Reward Model trainer.
        from soup_cli.trainer.prm import PRMTrainerWrapper

        trainer_wrapper = PRMTrainerWrapper(cfg, **trainer_kwargs)
    elif cfg.task in ("classifier", "reranker", "cross_encoder"):
        # v0.53.2 #132 — sequence-classification head.
        from soup_cli.trainer.classifier import ClassifierTrainerWrapper

        trainer_wrapper = ClassifierTrainerWrapper(cfg, **trainer_kwargs)
    elif cfg.task == "unlearn":
        # v0.71.9 #193 — NPO / SimNPO / RMU unlearning.
        from soup_cli.trainer.unlearn import UnlearnTrainerWrapper

        trainer_wrapper = UnlearnTrainerWrapper(cfg, **trainer_kwargs)
    elif cfg.task == "moe_lora_routing":
        # v0.71.12 #222 — MoLE per-token routing over N frozen task LoRAs.
        from soup_cli.trainer.mole_routing import MoleRoutingTrainerWrapper

        trainer_wrapper = MoleRoutingTrainerWrapper(cfg, **trainer_kwargs)
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

    profiler_ctx = contextlib.nullcontext()
    if profile_run:
        from soup_cli.utils.profiling import profile_training

        profiler_ctx = profile_training(output_dir=Path(cfg.output), run_id=run_id)
        console.print(
            "[cyan]--profile:[/] writing torch.profiler trace to "
            f"{cfg.output}/profiles/{run_id}.trace.json (early-steps window)"
        )

    # v0.71.3 #180 — optional codecarbon energy/CO2 measurement around the
    # training window. Lazy-built; a graceful no-op when codecarbon is absent.
    energy_ctx = contextlib.nullcontext()
    energy_tracker = None
    if track_energy:
        try:
            from soup_cli.utils.energy import EnergyTracker

            energy_tracker = EnergyTracker(country_iso_code=energy_country)
            energy_ctx = energy_tracker
        except ValueError as exc:
            console.print(f"[yellow]--track-energy disabled:[/] {exc}")
            energy_tracker = None
            energy_ctx = contextlib.nullcontext()

    try:
        with profiler_ctx, energy_ctx:
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
    except Exception as exc:
        tracker.fail_run(run_id)
        # v0.34.0 Part D — write a .crash bundle next to the run for triage.
        try:
            from soup_cli.utils.crash import build_crash_bundle, write_crash_bundle

            metrics = tracker.get_metrics(run_id)
            bundle = build_crash_bundle(
                error=exc,
                config=cfg.model_dump() if hasattr(cfg, "model_dump") else None,
                metrics=metrics,
                run_id=run_id,
                output_dir=getattr(cfg, "output", None),
            )
            crash_path = write_crash_bundle(bundle)
            console.print(
                f"[yellow]Crash bundle written:[/] {crash_path}\n"
                "[dim]Attach this file when reporting the failure.[/]"
            )
        except Exception as crash_err:
            # Never let the crash reporter mask the original error, but tell
            # the user the bundle is missing so they don't hunt for it.
            console.print(
                f"[dim]Could not write crash bundle: {crash_err}[/]"
            )
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

    # --- v0.56.0 --diagnose-gate: post-training failure-mode check ---
    if diagnose_gate and _should_run_diagnose_gate_on_rank():
        try:
            _run_diagnose_gate(
                diagnose_gate, run_id, cfg.base, result["output_dir"]
            )
        except typer.Exit:
            raise
        except (OSError, ValueError) as exc:
            console.print(
                f"[red]--diagnose-gate failed:[/] {type(exc).__name__}: {exc}"
            )
            raise typer.Exit(1) from exc

    # --- v0.71.3 #180 --track-energy: print the measured energy/CO2 -------
    energy_measurement = (
        energy_tracker.measurement if energy_tracker is not None else None
    )
    if track_energy:
        if energy_measurement is not None:
            console.print(
                f"[cyan]--track-energy:[/] {energy_measurement.energy_kwh:.4f} kWh"
                f" / {energy_measurement.co2_kg:.4f} kg CO2"
                f" (grid {energy_measurement.grid_intensity_g_per_kwh:.0f} g/kWh,"
                f" PUE {energy_measurement.pue})"
            )
        else:
            console.print(
                "[yellow]--track-energy:[/] no reading "
                "(install `pip install soup-cli[carbon]`)"
            )

    # --- v0.71.15 #244 --energy-out: persist for `soup bom emit --energy` -
    if energy_out and _should_run_diagnose_gate_on_rank():
        if energy_measurement is not None:
            try:
                _write_energy_json(energy_out, energy_measurement)
                console.print(
                    "[cyan]--energy-out:[/] wrote "
                    f"{os.path.basename(energy_out)} "
                    "(feed to `soup bom emit --energy`)"
                )
            except (OSError, ValueError) as exc:
                console.print(
                    f"[yellow]--energy-out skipped:[/] {type(exc).__name__}: {exc}"
                )
        else:
            console.print(
                "[yellow]--energy-out skipped:[/] no energy reading "
                "(set --track-energy + install `pip install soup-cli[carbon]`)"
            )

    # --- v0.59.0 --annex-xi: Annex XI/XII auto-doc -----------------------
    if annex_xi and _should_run_diagnose_gate_on_rank():
        try:
            _write_annex_xi(annex_xi, run_id, cfg, energy=energy_measurement)
        except typer.Exit:
            raise
        except (OSError, ValueError) as exc:
            console.print(
                f"[yellow]--annex-xi skipped:[/] {type(exc).__name__}: {exc}"
            )

    # --- v0.59.0 --repro-receipt: SR 11-7 receipt ------------------------
    if repro_receipt and _should_run_diagnose_gate_on_rank():
        try:
            _write_repro_receipt(repro_receipt, run_id, cfg)
        except typer.Exit:
            raise
        except (OSError, ValueError) as exc:
            console.print(
                f"[yellow]--repro-receipt skipped:[/] {type(exc).__name__}: {exc}"
            )

    # --- v0.71.8 #219 --capture-activations: SAE-diff-ready snapshot ------
    if capture_activations and _should_run_diagnose_gate_on_rank():
        try:
            written = _capture_activations(
                capture_activations,
                capture_prompts,
                cfg.base,
                result["output_dir"],
                trust_remote_code=trust_remote_code,
            )
            console.print(f"[green]--capture-activations[/] -> {written}")
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            console.print(
                f"[yellow]--capture-activations skipped:[/] "
                f"{type(exc).__name__}: {exc}"
            )


def _write_annex_xi(out_path: str, run_id: str, cfg, *, energy=None) -> None:
    """Render an Annex XI doc using values from the resolved soup.yaml.

    v0.71.3 #180: the optional ``energy`` measurement populates the kWh / CO2
    fields. v0.71.3 #184: the top crawled domains are auto-extracted from the
    training JSONL. v0.71.3 #181: a ``.pdf`` output path renders a PDF.
    """
    from datetime import datetime, timezone

    from soup_cli import __version__
    from soup_cli.utils.annex_xi import (
        AnnexXIData,
        load_top_domains_from_jsonl,
        write_annex_doc,
    )

    modality = getattr(cfg, "modality", "text") or "text"
    energy_kwh = float(getattr(energy, "energy_kwh", 0.0)) if energy is not None else 0.0
    co2_kg = float(getattr(energy, "co2_kg", 0.0)) if energy is not None else 0.0
    train_path = str(getattr(cfg.data, "train", "") or "")
    # #184 — best-effort extract the top crawled domains from the training data.
    top_domains = load_top_domains_from_jsonl(train_path)
    fmt = "pdf" if out_path.lower().endswith(".pdf") else "markdown"
    data = AnnexXIData(
        model_name=str(getattr(cfg, "output", run_id) or run_id),
        base_model=str(cfg.base),
        task=str(cfg.task),
        dataset_summary=train_path,
        modalities=(modality,),
        train_compute_flops=0.0,
        train_energy_kwh=energy_kwh,
        train_co2_kg=co2_kg,
        top_domains=top_domains,
        soup_version=__version__,
        run_id=run_id,
        created_at=datetime.now(tz=timezone.utc).isoformat(),
    )
    written = write_annex_doc(data, "xi", out_path, fmt=fmt)
    console.print(f"[green]--annex-xi[/] -> {written}")


def _write_repro_receipt(out_path: str, run_id: str, cfg) -> None:
    """Render an SR 11-7 receipt from the resolved soup.yaml."""
    from soup_cli.utils.repro_receipt import build_repro_receipt, write_repro_receipt

    seeds: dict[str, int] = {}
    seed = getattr(cfg.training, "seed", None)
    if isinstance(seed, int) and not isinstance(seed, bool):
        seeds["torch"] = seed
        seeds["numpy"] = seed
        seeds["python"] = seed
    receipt = build_repro_receipt(seeds=seeds, run_id=run_id)
    written = write_repro_receipt(receipt, out_path)
    console.print(f"[green]--repro-receipt[/] -> {written}")


def _load_capture_prompts(canonical_path: str) -> list[str]:
    """Read prompts for --capture-activations (JSONL objects, JSON strings, or raw lines)."""
    import json

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(canonical_path, flags)
    # File-size cap on the SAME fd (no TOCTOU) bounds a pathological
    # newline-free multi-GB line before the `for line in handle` read.
    if os.fstat(fd).st_size > 64 * 1024 * 1024:
        os.close(fd)
        raise ValueError("--capture-prompts file exceeds 64 MiB")
    prompts: list[str] = []
    with os.fdopen(fd, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                prompts.append(line)  # raw-text line
            else:
                if isinstance(obj, str) and obj:
                    prompts.append(obj)
                elif isinstance(obj, dict):
                    for key in ("prompt", "text", "instruction", "input", "question"):
                        val = obj.get(key)
                        if isinstance(val, str) and val:
                            prompts.append(val)
                            break
            if len(prompts) >= 256:
                break
    return prompts


def _capture_activations(
    layer: str,
    prompts_path: str,
    base: str,
    output_dir: str,
    *,
    trust_remote_code: bool = False,
) -> str:
    """Capture residual-stream activations from the trained model (v0.71.8 #219).

    Loads the trained model at ``output_dir`` (or ``base`` + adapter when the
    output is a LoRA adapter), runs ``prompts_path`` through a forward hook on
    ``layer``, and writes the per-token activations to
    ``<output_dir>/activations/activations.json`` in the
    ``{"activations": [[...]], "layer", "num_tokens", "hidden_dim"}`` shape that
    ``soup probe sae-diff`` / ``sleeper`` consume directly.
    """
    import json

    from soup_cli.utils import live_eval
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    if not prompts_path:
        raise ValueError("--capture-activations requires --capture-prompts <jsonl>")
    if not isinstance(layer, str) or not layer.strip():
        raise ValueError("--capture-activations layer must be a non-empty string")
    canonical = enforce_under_cwd_and_no_symlink(prompts_path, "--capture-prompts")
    prompts = _load_capture_prompts(canonical)
    if not prompts:
        raise ValueError("--capture-prompts has no usable prompts")

    # When the output is a LoRA adapter dir, load base + adapter; else load the
    # full fine-tuned model from output_dir directly.
    is_adapter = os.path.isfile(os.path.join(output_dir, "adapter_config.json"))
    has_full = os.path.isfile(os.path.join(output_dir, "config.json"))
    if not is_adapter and not has_full:
        raise ValueError(
            f"output dir {os.path.basename(output_dir)} has neither "
            "adapter_config.json nor config.json — cannot capture activations"
        )
    adapter = output_dir if is_adapter else None
    model_id = base if adapter else output_dir

    model, tokenizer, dev = live_eval.load_model_and_tokenizer(
        model_id, adapter=adapter, trust_remote_code=trust_remote_code
    )
    acts = live_eval.extract_layer_activations(
        model, tokenizer, prompts, layer=layer, device=dev, pool="none"
    )

    acts_dir = os.path.join(output_dir, "activations")
    # Refuse a pre-placed `activations` symlink so the write cannot be
    # redirected outside output_dir (defence-in-depth — output_dir is config-
    # derived/trusted but the symlink check is cheap).
    if os.path.islink(acts_dir):
        raise ValueError("activations subdir is a symlink — refusing to write")
    os.makedirs(acts_dir, exist_ok=True)
    out_path = os.path.join(acts_dir, "activations.json")
    payload = {
        "layer": layer,
        "num_tokens": int(acts.shape[0]),
        "hidden_dim": int(acts.shape[1]),
        "activations": acts.tolist(),
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return out_path


def _write_energy_json(path: str, measurement: "EnergyMeasurement") -> None:
    """Persist an ``EnergyMeasurement`` as JSON for ``soup bom emit --energy``.

    Writes exactly the five fields ``EnergyMeasurement(**parsed)`` expects so
    the producer (``soup train --track-energy --energy-out``) and the consumer
    (``soup bom emit --energy`` — v0.71.3 #256) round-trip cleanly. Atomic +
    cwd-contained + symlink-rejected via the shared helper (v0.71.15 #244).
    """
    import json
    from dataclasses import asdict

    from soup_cli.utils.paths import atomic_write_text

    payload = json.dumps(asdict(measurement), indent=2)
    atomic_write_text(payload, path, field="--energy-out")


def _should_run_diagnose_gate_on_rank() -> bool:
    """Return True only for the single chief worker in a distributed launch.

    The diagnose gate (and the --annex-xi / --repro-receipt / capture hooks)
    write one report and, on a shared filesystem, read one shared output dir
    -- so they should fire exactly once per *cluster*, not once per node.

    Resolution (v0.71.15 #170 — fixes the v0.56.0 limitation where a shared-FS
    multi-node run fired the gate once per node):
      * If RANK is set (multi-node / torchrun launch) -> gate only on the
        global chief, RANK == 0. On a single-node torchrun this is equivalent
        to LOCAL_RANK == 0 (both 0 only for the one chief process).
      * Else fall back to LOCAL_RANK == 0 (plain single-node accelerate launch
        that doesn't export RANK) so a one-box multi-GPU run still gates once.

    Defaults to True (run gate) on any parse error: a malformed env var is
    safer to over-run than to silently skip.
    """
    rank = os.environ.get("RANK")
    if rank:  # non-empty -> multi-node / torchrun sets a global RANK
        try:
            return int(rank) == 0
        except ValueError:
            return True
    try:
        return int(os.environ.get("LOCAL_RANK", "0")) == 0
    except ValueError:
        return True


def _run_diagnose_gate(
    evidence_path: str, run_id: str, base: str, adapter: str
) -> None:
    """Post-training failure-mode gate (v0.56.0).

    Loads a JSON ``evidence`` file with optional per-mode scores and
    refuses to mark the run successful if any mode comes back MAJOR.
    Missing modes fall back to a neutral OK score so partial evidence
    still produces a useful report card. The train command only calls
    this helper on the chief worker (RANK==0 in a multi-node launch, else
    LOCAL_RANK==0) so distributed runs execute the gate once per cluster,
    not once per worker (v0.71.15 #170).
    """
    import json

    from soup_cli.utils.diagnose.report import FAILURE_MODES, FailureScore
    from soup_cli.utils.diagnose.runner import build_report
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    enforce_under_cwd_and_no_symlink(evidence_path, "--diagnose-gate evidence")
    # 16 MiB cap on evidence JSON (security review HIGH — symmetric with
    # `commands/diagnose._MAX_EVIDENCE_BYTES`; prevents `/dev/zero` /
    # multi-GB symlink-pointed OOM at json.load time).
    if os.path.getsize(evidence_path) > 16 * 1024 * 1024:
        raise ValueError(
            "--diagnose-gate evidence exceeds 16 MiB"
        )
    with open(evidence_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("evidence file must contain a JSON object")
    raw_scores = payload.get("scores") or {}
    if not isinstance(raw_scores, dict):
        raise ValueError("evidence.scores must be an object")

    from soup_cli.utils.diagnose.report import classify_score

    scores: dict = {}
    for mode in FAILURE_MODES:
        entry = raw_scores.get(mode)
        if entry is None:
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"scores.{mode} must be an object")
        score = entry.get("score", 1.0)
        verdict = entry.get("verdict") or classify_score(score)
        scores[mode] = FailureScore(
            mode=mode,
            score=float(score),
            verdict=verdict,
            evidence=str(entry.get("evidence", "supplied by --diagnose-gate")),
        )

    report = build_report(
        run_id=run_id, base=base, adapter=adapter, scores=scores
    )
    if report.overall == "MAJOR":
        console.print(
            "[red]--diagnose-gate: MAJOR regression in one or more modes.[/]"
        )
        for mode in FAILURE_MODES:
            sc = report.scores[mode]
            if sc.verdict == "MAJOR":
                console.print(f"  [red]MAJOR[/] {mode}: {sc.evidence}")
        raise typer.Exit(2)
    console.print(
        f"[green]--diagnose-gate: {report.overall}[/] across "
        f"{len(FAILURE_MODES)} modes."
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
    # v0.40.1 Part C / G12 — fix broken `load_local` import that previously
    # always fell through to the synthetic curve. The actual exported symbol
    # is ``load_raw_data`` (path-only loader) — we use that.
    from pathlib import Path as _Path

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from soup_cli.data.loader import load_raw_data
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

    dataset = load_raw_data(_Path(cfg.data.train))
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
