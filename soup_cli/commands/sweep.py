"""soup sweep — hyperparameter search over training configs."""

import itertools
import random
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from soup_cli.config.loader import load_config

console = Console()


def sweep(
    config: str = typer.Option(
        "soup.yaml",
        "--config",
        "-c",
        help="Path to base soup.yaml config file",
    ),
    param: list[str] = typer.Option(
        ...,
        "--param",
        "-p",
        help="Parameter to sweep: key=val1,val2,val3 (e.g., lr=1e-5,2e-5,5e-5)",
    ),
    strategy: str = typer.Option(
        "grid",
        "--strategy",
        "-s",
        help="Search strategy: grid, random",
    ),
    max_runs: Optional[int] = typer.Option(
        None,
        "--max-runs",
        help="Max number of runs (useful for random strategy)",
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Sweep experiment name prefix",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show planned runs without executing",
    ),
    early_stop: float = typer.Option(
        None,
        "--early-stop",
        help="Stop early if run's loss exceeds best loss by this factor (e.g. 1.5 = 50% worse)",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt",
    ),
):
    """Run hyperparameter sweep: grid or random search over training parameters."""
    config_path = Path(config)
    if not config_path.exists():
        console.print(f"[red]Config not found: {config_path}[/]")
        raise typer.Exit(1)

    if strategy not in ("grid", "random"):
        console.print(f"[red]Invalid strategy: {strategy}. Must be grid or random.[/]")
        raise typer.Exit(1)

    # Parse sweep parameters
    sweep_params = _parse_sweep_params(param)
    if not sweep_params:
        console.print("[red]No valid sweep parameters provided.[/]")
        raise typer.Exit(1)

    # Generate parameter combinations
    combinations = _generate_combinations(sweep_params, strategy, max_runs)

    console.print(
        Panel(
            f"Config:   [bold]{config_path}[/]\n"
            f"Strategy: [bold]{strategy}[/]\n"
            f"Params:   [bold]{', '.join(sweep_params.keys())}[/]\n"
            f"Runs:     [bold]{len(combinations)}[/]",
            title="Sweep Plan",
        )
    )

    # Show parameter table
    param_table = Table(title="Parameter Grid")
    param_table.add_column("Run", style="bold")
    for key in sweep_params:
        param_table.add_column(key)

    for idx, combo in enumerate(combinations):
        row_values = [str(combo[key]) for key in sweep_params]
        param_table.add_row(f"#{idx + 1}", *row_values)

    console.print(param_table)

    if dry_run:
        console.print("[yellow]Dry run - no training will be executed.[/]")
        raise typer.Exit()

    if not yes:
        if not typer.confirm(f"Start {len(combinations)} training run(s)?", default=True):
            console.print("[yellow]Cancelled.[/]")
            raise typer.Exit()

    # Execute sweep
    base_cfg = load_config(config_path)
    results = []
    best_loss = float("inf")
    skipped = 0

    for idx, combo in enumerate(combinations):
        run_name = f"{name or 'sweep'}_{idx + 1}"
        console.print(f"\n[bold]--- Run {idx + 1}/{len(combinations)}: {run_name} ---[/]")

        for key, val in combo.items():
            console.print(f"  {key} = {val}")

        try:
            result = _run_single(base_cfg, combo, run_name, config_path)
            final_loss = result.get("final_loss", 0)
            results.append({
                "name": run_name,
                "params": combo,
                "run_id": result.get("run_id", ""),
                "final_loss": final_loss,
                "duration": result.get("duration", ""),
                "status": "completed",
            })

            # Update best loss and check early stopping for remaining runs
            if final_loss and final_loss < best_loss:
                best_loss = final_loss

            if early_stop and final_loss and best_loss < float("inf"):
                if final_loss > best_loss * early_stop:
                    console.print(
                        f"[yellow]Loss {final_loss:.4f} exceeds threshold "
                        f"({best_loss:.4f} × {early_stop} = {best_loss * early_stop:.4f})[/]"
                    )
        except Exception as exc:
            console.print(f"[red]Run {run_name} failed: {exc}[/]")
            results.append({
                "name": run_name,
                "params": combo,
                "run_id": "",
                "final_loss": 0,
                "duration": "",
                "status": "failed",
            })

        # Early stopping: skip remaining runs if too many are poor
        if early_stop and len(results) >= 2:
            completed = [r for r in results if r["status"] == "completed" and r["final_loss"]]
            if completed:
                recent = completed[-1]
                if recent["final_loss"] > best_loss * early_stop:
                    remaining = len(combinations) - idx - 1
                    if remaining > 0:
                        skipped = remaining
                        console.print(
                            f"[yellow]Early stopping: skipping {remaining} remaining run(s). "
                            f"Last loss {recent['final_loss']:.4f} exceeded threshold.[/]"
                        )
                        break

    # Summary table
    _display_summary(results, sweep_params)

    if skipped:
        console.print(f"\n[yellow]Early stopping: {skipped} run(s) skipped.[/]")


def _parse_sweep_params(params: list[str]) -> dict[str, list]:
    """Parse sweep parameter strings into a dict of {key: [values]}."""
    result = {}
    for param_str in params:
        if "=" not in param_str:
            console.print(f"[yellow]Skipping invalid param: {param_str} (missing '=')[/]")
            continue

        key, values_str = param_str.split("=", 1)
        key = key.strip()
        values = []

        for val in values_str.split(","):
            val = val.strip()
            values.append(_parse_value(val))

        if values:
            result[key] = values

    return result


def _parse_value(val: str):
    """Parse a string value into the appropriate Python type."""
    # Bool
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    # None
    if val.lower() == "none":
        return None
    # Int
    try:
        return int(val)
    except ValueError:
        pass
    # Float (including scientific notation)
    try:
        return float(val)
    except ValueError:
        pass
    # String
    return val


def _generate_combinations(
    sweep_params: dict[str, list],
    strategy: str,
    max_runs: Optional[int],
) -> list[dict]:
    """Generate parameter combinations based on strategy."""
    keys = list(sweep_params.keys())
    value_lists = [sweep_params[k] for k in keys]

    if strategy == "grid":
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*value_lists)]
    elif strategy == "random":
        total_possible = 1
        for vals in value_lists:
            total_possible *= len(vals)

        num_runs = max_runs or min(total_possible, 10)
        num_runs = min(num_runs, total_possible)

        if num_runs >= total_possible:
            # Just do all of them
            combos = [dict(zip(keys, vals)) for vals in itertools.product(*value_lists)]
        else:
            seen = set()
            combos = []
            while len(combos) < num_runs:
                vals = tuple(random.choice(vals_list) for vals_list in value_lists)
                if vals not in seen:
                    seen.add(vals)
                    combos.append(dict(zip(keys, vals)))
    else:
        combos = []

    if max_runs and len(combos) > max_runs:
        combos = combos[:max_runs]

    return combos


def _set_nested_param(config_dict: dict, key: str, value) -> dict:
    """Set a nested parameter in a config dict using dot notation.

    Supports keys like: lr, lora.r, training.epochs, etc.
    Maps common short names to their full paths.
    """
    # Short name mappings
    shortcuts = {
        "lr": "training.lr",
        "epochs": "training.epochs",
        "batch_size": "training.batch_size",
        "lora_r": "training.lora.r",
        "lora_alpha": "training.lora.alpha",
        "lora_dropout": "training.lora.dropout",
        "quantization": "training.quantization",
        "warmup_ratio": "training.warmup_ratio",
        "weight_decay": "training.weight_decay",
        "gradient_accumulation_steps": "training.gradient_accumulation_steps",
        "max_grad_norm": "training.max_grad_norm",
        "optimizer": "training.optimizer",
        "scheduler": "training.scheduler",
        "val_split": "data.val_split",
        "max_length": "data.max_length",
        "dpo_beta": "training.dpo_beta",
        "kto_beta": "training.kto_beta",
        "grpo_beta": "training.grpo_beta",
        "num_generations": "training.num_generations",
        "reward_fn": "training.reward_fn",
        "ppo_epochs": "training.ppo_epochs",
        "ppo_clip_ratio": "training.ppo_clip_ratio",
        "ppo_kl_penalty": "training.ppo_kl_penalty",
        "reward_model": "training.reward_model",
        "orpo_beta": "training.orpo_beta",
        "simpo_gamma": "training.simpo_gamma",
        "cpo_alpha": "training.cpo_alpha",
        "ipo_tau": "training.ipo_tau",
        "loraplus_lr_ratio": "training.loraplus_lr_ratio",
        "use_dora": "training.lora.use_dora",
        "use_galore": "training.use_galore",
        "galore_rank": "training.galore_rank",
        "moe_lora": "training.moe_lora",
        "moe_aux_loss_coeff": "training.moe_aux_loss_coeff",
        "embedding_loss": "training.embedding_loss",
        "embedding_margin": "training.embedding_margin",
        "embedding_pooling": "training.embedding_pooling",
        "embedding_temperature": "training.embedding_temperature",
        "neftune_alpha": "training.neftune_alpha",
        "use_rslora": "training.lora.use_rslora",
        "backend": "backend",
    }

    full_key = shortcuts.get(key, key)
    parts = full_key.split(".")

    obj = config_dict
    for part in parts[:-1]:
        if part not in obj:
            obj[part] = {}
        obj = obj[part]
    obj[parts[-1]] = value

    return config_dict


def _run_single(base_cfg, params: dict, run_name: str, config_path: Path) -> dict:
    """Run a single training with modified parameters."""
    from soup_cli.config.schema import SoupConfig
    from soup_cli.data.loader import load_dataset
    from soup_cli.experiment.tracker import ExperimentTracker
    from soup_cli.monitoring.display import TrainingDisplay
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils.gpu import detect_device, get_gpu_info

    # Deep copy and modify config
    config_dict = base_cfg.model_dump()
    for key, val in params.items():
        _set_nested_param(config_dict, key, val)

    # Override experiment name
    config_dict["experiment_name"] = run_name
    cfg = SoupConfig(**config_dict)

    # Detect hardware
    device, device_name = detect_device()
    gpu_info = get_gpu_info()

    # Load data
    dataset = load_dataset(cfg.data)
    console.print(f"[dim]Loaded {len(dataset['train'])} train samples[/]")

    # Start tracking
    tracker = ExperimentTracker()
    run_id = tracker.start_run(
        config_dict=cfg.model_dump(),
        device=device,
        device_name=device_name,
        gpu_info=gpu_info,
        experiment_name=run_name,
    )

    # Build trainer
    if cfg.task == "dpo":
        from soup_cli.trainer.dpo import DPOTrainerWrapper

        trainer_wrapper = DPOTrainerWrapper(cfg, device=device)
    elif cfg.task == "kto":
        from soup_cli.trainer.kto import KTOTrainerWrapper

        trainer_wrapper = KTOTrainerWrapper(cfg, device=device)
    elif cfg.task == "grpo":
        from soup_cli.trainer.grpo import GRPOTrainerWrapper

        trainer_wrapper = GRPOTrainerWrapper(cfg, device=device)
    elif cfg.task == "ppo":
        from soup_cli.trainer.ppo import PPOTrainerWrapper

        trainer_wrapper = PPOTrainerWrapper(cfg, device=device)
    elif cfg.task == "orpo":
        from soup_cli.trainer.orpo import ORPOTrainerWrapper

        trainer_wrapper = ORPOTrainerWrapper(cfg, device=device)
    elif cfg.task == "simpo":
        from soup_cli.trainer.simpo import SimPOTrainerWrapper

        trainer_wrapper = SimPOTrainerWrapper(cfg, device=device)
    elif cfg.task == "ipo":
        from soup_cli.trainer.ipo import IPOTrainerWrapper

        trainer_wrapper = IPOTrainerWrapper(cfg, device=device)
    elif cfg.task == "reward_model":
        from soup_cli.trainer.reward_model import RewardModelTrainerWrapper

        trainer_wrapper = RewardModelTrainerWrapper(cfg, device=device)
    elif cfg.task == "pretrain":
        from soup_cli.trainer.pretrain import PretrainTrainerWrapper

        trainer_wrapper = PretrainTrainerWrapper(cfg, device=device)
    elif cfg.task == "embedding":
        from soup_cli.trainer.embedding import EmbeddingTrainerWrapper

        trainer_wrapper = EmbeddingTrainerWrapper(cfg, device=device)
    else:
        trainer_wrapper = SFTTrainerWrapper(cfg, device=device)
    trainer_wrapper.setup(dataset)

    # Train
    display = TrainingDisplay(cfg, device_name=device_name)
    try:
        result = trainer_wrapper.train(display=display, tracker=tracker, run_id=run_id)
        tracker.finish_run(
            run_id=run_id,
            initial_loss=result["initial_loss"],
            final_loss=result["final_loss"],
            total_steps=result["total_steps"],
            duration_secs=result["duration_secs"],
            output_dir=result["output_dir"],
        )
        result["run_id"] = run_id
        return result
    except Exception:
        tracker.fail_run(run_id)
        raise


def _display_summary(results: list[dict], sweep_params: dict[str, list]):
    """Display sweep results summary table."""
    table = Table(title="Sweep Results")
    table.add_column("Run", style="bold")
    for key in sweep_params:
        table.add_column(key)
    table.add_column("Final Loss", justify="right", style="green")
    table.add_column("Duration", justify="right")
    table.add_column("Status")

    # Sort by final loss (best first)
    sorted_results = sorted(results, key=lambda r: r.get("final_loss", float("inf")))

    for idx, res in enumerate(sorted_results):
        status_style = "green" if res["status"] == "completed" else "red"
        param_vals = [str(res["params"].get(k, "")) for k in sweep_params]
        loss_str = f"{res['final_loss']:.4f}" if res["final_loss"] else "-"
        best_marker = " [bold yellow]*[/]" if idx == 0 and res["status"] == "completed" else ""
        table.add_row(
            res["name"],
            *param_vals,
            f"{loss_str}{best_marker}",
            res.get("duration", "-"),
            f"[{status_style}]{res['status']}[/]",
        )

    console.print(table)

    # Best run
    completed = [r for r in sorted_results if r["status"] == "completed"]
    if completed:
        best = completed[0]
        console.print(
            f"\n[bold green]Best run:[/] {best['name']} "
            f"(loss: {best['final_loss']:.4f})"
        )
        for key, val in best["params"].items():
            console.print(f"  {key} = {val}")
        if best.get("run_id"):
            console.print(f"\n[dim]View details: soup runs show {best['run_id']}[/]")
