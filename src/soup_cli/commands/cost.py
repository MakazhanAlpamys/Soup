"""soup cost -- estimate training cost in USD."""

from __future__ import annotations

import contextlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer
from rich.console import Console
from rich.table import Table

from soup_cli.utils.gpu import model_size_from_name
from soup_cli.utils.profiler import (
    estimate_speed,
    estimate_training_time,
)
from soup_cli.utils.terminal import for_terminal

if TYPE_CHECKING:
    from soup_cli.config.schema import SoupConfig

console = Console()
err_console = Console(stderr=True)

# GPU pricing and speed multipliers (relative to A100).
# Pricing last updated: 2026-04. Actual rates change frequently --
# treat these as rough estimates; real costs can vary +/- 30%.
GPU_PRICING = [
    {"provider": "RunPod", "gpu": "A100 80G", "cost_per_hr": 1.89, "speed_mult": 1.0},
    {"provider": "Lambda", "gpu": "A100 40G", "cost_per_hr": 1.10, "speed_mult": 1.0},
    {"provider": "Modal", "gpu": "H100", "cost_per_hr": 5.92, "speed_mult": 2.5},
    {"provider": "Vast.ai", "gpu": "RTX 4090", "cost_per_hr": 0.35, "speed_mult": 0.7},
    {"provider": "CoreWeave", "gpu": "A100 80G", "cost_per_hr": 2.21, "speed_mult": 1.0},
]

_DEFAULT_DATASET_SIZE = 10000
_HISTORY_GPU_LABELS = {"A100 40G": "A100 40GB", "A100 80G": "A100 80GB"}


def _get_dataset_size(cfg) -> tuple[int, bool]:
    """Estimate training dataset size.

    Returns (size, is_estimated). `is_estimated=True` means we fell back to
    the default because the dataset could not be read; callers should warn.
    """
    from soup_cli.utils.eval_schedule import effective_val_split

    # #1223: the share `soup train` actually withholds, which is 0 for a task
    # that does not evaluate its split.
    split = 1.0 - effective_val_split(cfg)
    train_path = cfg.data.train

    # #443 — data.interleave: sum row counts across every local dataset
    # rather than treating cfg.data.train as one path.
    if isinstance(train_path, list):
        from soup_cli.data.loader import load_raw_data

        total = 0
        any_estimated = False
        for p in train_path:
            entry_path = Path(p)
            if entry_path.exists():
                try:
                    total += len(load_raw_data(entry_path))
                    continue
                except (OSError, ValueError, KeyError):
                    pass
            any_estimated = True
            total += _DEFAULT_DATASET_SIZE
        return int(total * split), any_estimated

    path = Path(train_path)

    # Local file
    if path.exists():
        from soup_cli.data.loader import load_raw_data
        try:
            data = load_raw_data(path)
            return int(len(data) * split), False
        except (OSError, ValueError, KeyError):
            pass

    # Match training's route for Hub names, including names such as OpenHermes-2.5.
    from soup_cli.data.loader import _classify_train_entry

    if _classify_train_entry(train_path) == "hub" and not (path.suffix and path.is_file()):
        try:
            from datasets import load_dataset_builder

            splits = load_dataset_builder(train_path).info.splits
            if splits is not None:
                size = splits["train"].num_examples
                return int(size * split), False
        except (OSError, ValueError, KeyError, ImportError):
            pass

    return _DEFAULT_DATASET_SIZE, True


def _estimate_from_history(
    cfg: SoupConfig,
    gpu_labels: set[str],
    dataset_size: int | None,
) -> tuple[dict | None, str | None]:
    from soup_cli.utils.cost_history import (
        MIN_MATCHES,
        estimate_from_step_times,
        fixed_sft_settings,
        matching_step_times,
    )

    if len(gpu_labels) != 1:
        return None, "Select a single GPU model with --gpu to use history."
    if cfg.task != "sft":
        return None, "History is available for SFT only."
    settings = fixed_sft_settings(cfg.model_dump())
    if settings is None:
        return None, (
            "History requires text SFT on Transformers or Unsloth, with fixed batch size "
            "and accumulation, "
            "and without packing, streaming, interleaving or replay."
        )
    if dataset_size is None or dataset_size < 1:
        return None, "The training row count is unavailable."
    gpu_label = next(iter(gpu_labels))
    recorded_label = _HISTORY_GPU_LABELS.get(gpu_label, gpu_label)

    from soup_cli.experiment.tracker import ExperimentTracker

    try:
        with contextlib.closing(ExperimentTracker()) as tracker:
            times = matching_step_times(tracker.list_runs(limit=-1), settings, recorded_label)
    except (OSError, sqlite3.Error):
        return None, "The local run history could not be read."
    planned_steps = settings.planned_steps(dataset_size, cfg.training.epochs)
    estimate = estimate_from_step_times(times, planned_steps)
    if estimate is None:
        return None, f"Found {len(times)} matching runs for {gpu_label}; need {MIN_MATCHES}."
    return estimate, None


def cost(
    config: str = typer.Option(
        "soup.yaml", "--config", "-c", help="Path to soup.yaml config file"
    ),
    gpu: Optional[str] = typer.Option(
        None, "--gpu", "-g",
        help="Filter by specific GPU (e.g., A100, H100, RTX 4090)",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output as JSON for scripting"
    ),
    history: bool = typer.Option(
        False, "--history", help="Use comparable local SFT runs with one GPU selected by --gpu."
    ),
) -> None:
    """Estimate training cost in USD across cloud providers."""
    from soup_cli.config.loader import load_config

    diagnostic_console = err_console if json_output else console
    config_path = Path(config)
    if not config_path.exists():
        diagnostic_console.print(
            f"[red]Config file not found:[/] {for_terminal(config)}", emoji=False
        )
        raise typer.Exit(1)

    # The loader reports refusals and advisories on stdout. Keep those out of
    # the document stream when --json is requested, including on exit 1.
    with contextlib.redirect_stdout(sys.stderr if json_output else sys.stdout):
        cfg = load_config(config_path)
    model_params_b = model_size_from_name(cfg.base)

    batch_size = cfg.training.batch_size
    if batch_size == "auto":
        batch_size = 4
    else:
        batch_size = int(batch_size)

    dataset_size, is_estimated = _get_dataset_size(cfg)
    epochs = cfg.training.epochs

    if is_estimated and not json_output:
        console.print(
            f"[yellow]Warning:[/] Could not read training dataset; using default "
            f"of {_DEFAULT_DATASET_SIZE:,} examples for the estimate."
        )

    # Base speed (A100)
    base_tokens_per_sec = estimate_speed(
        model_params_b, cfg.training.quantization, batch_size
    )
    base_samples_per_sec = base_tokens_per_sec / max(cfg.data.max_length, 1)

    results = []

    for provider_info in GPU_PRICING:
        if gpu:
            gpu_norm = gpu.lower().replace(" ", "")
            p_gpu_norm = provider_info["gpu"].lower().replace(" ", "")
            if gpu_norm not in p_gpu_norm:
                continue

        speed_mult = provider_info["speed_mult"]
        samples_per_sec = base_samples_per_sec * speed_mult

        duration_mins = estimate_training_time(dataset_size, epochs, samples_per_sec)
        duration_hrs = duration_mins / 60.0

        total_cost = duration_hrs * provider_info["cost_per_hr"]

        results.append({
            "provider": provider_info["provider"],
            "gpu": provider_info["gpu"],
            "cost_per_hr": provider_info["cost_per_hr"],
            "duration_hrs": duration_hrs,
            "total_cost": total_cost,
        })

    if not results:
        diagnostic_console.print(
            f"[red]No matching GPUs found for:[/] {for_terminal(gpu)}", emoji=False
        )
        raise typer.Exit(1)

    history_estimate = None
    fallback = None
    if history:
        history_estimate, fallback = _estimate_from_history(
            cfg,
            {result["gpu"] for result in results} if gpu else set(),
            dataset_size if not is_estimated else None,
        )
        for result in results:
            if history_estimate is not None:
                duration_hrs = (
                    history_estimate["median_seconds_per_step"]
                    * history_estimate["optimizer_steps"] / 3600
                )
                result.update(
                    estimate_source="history", history=history_estimate,
                    duration_hrs=duration_hrs, total_cost=duration_hrs * result["cost_per_hr"],
                )
            else:
                result.update(estimate_source="model", estimate_reason=fallback)

    if json_output:
        # Plain stdout, not console.print: Rich folds a document wider than the
        # console (80 columns when piped) and reads "[...]" as markup, so a long
        # record is split and a bracketed value such as "[v2]" is dropped or
        # raises MarkupError. Matches the other machine-readable paths (#1468).
        typer.echo(json.dumps(results, indent=2))
        return

    table = Table(title="Training Cost Estimate", title_justify="left", box=None, padding=(0, 2))
    table.add_column("Provider", style="cyan")
    table.add_column("GPU", style="green")
    table.add_column("$/hr", justify="right")
    table.add_column("Total Cost", justify="right")

    for r in results:
        cost_str = f"~${r['total_cost']:.2f}"
        if r["duration_hrs"] < 1.0:
            dur_str = "(<1h)"
        else:
            dur_str = f"({r['duration_hrs']:.0f}h)"
        table.add_row(
            r["provider"],
            r["gpu"],
            f"${r['cost_per_hr']:.2f}",
            f"{cost_str} [dim]{dur_str}[/dim]",
        )

    console.print(table)
    if history_estimate is not None:
        console.print(
            f"History estimate from {history_estimate['matches']} matching runs. "
            f"Median {history_estimate['median_seconds_per_step']:.2f} s/step, "
            f"observed range {history_estimate['min_seconds_per_step']:.2f} to "
            f"{history_estimate['max_seconds_per_step']:.2f} s/step, "
            f"relative spread {history_estimate['relative_spread']:.1%}, "
            f"confidence {history_estimate['confidence']}."
        )
        console.print(
            f"For {history_estimate['optimizer_steps']} estimated optimizer steps, "
            f"the range is {60 * history_estimate['min_duration_hrs']:.2f} to "
            f"{60 * history_estimate['max_duration_hrs']:.2f} min."
        )
    elif fallback is not None:
        console.print(f"Model-based estimate. {fallback}")
    console.print(
        "[dim]Note: estimates are approximate; actual costs can vary +/- 30% "
        "depending on region, spot/on-demand pricing, and workload variance.[/dim]"
    )
