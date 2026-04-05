"""soup runs — experiment tracking commands."""

from __future__ import annotations

import json
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

app = typer.Typer(no_args_is_help=False, invoke_without_command=True)


@app.callback(invoke_without_command=True)
def list_runs(
    ctx: typer.Context,
    limit: int = typer.Option(20, "--limit", "-l", help="Max runs to show"),
):
    """List all training runs."""
    if ctx.invoked_subcommand is not None:
        return

    from soup_cli.experiment.tracker import ExperimentTracker

    tracker = ExperimentTracker()
    runs = tracker.list_runs(limit=limit)

    if not runs:
        console.print("[dim]No runs found. Train a model with:[/] [bold]soup train[/]")
        raise typer.Exit()

    table = Table(title="Training Runs")
    table.add_column("Run ID", style="bold cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Model", max_width=30)
    table.add_column("Task")
    table.add_column("Status")
    table.add_column("Loss", justify="right")
    table.add_column("Steps", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Date", no_wrap=True)

    for run in runs:
        # Format status with color
        status = run["status"]
        if status == "completed":
            status_str = "[green]completed[/]"
        elif status == "failed":
            status_str = "[red]failed[/]"
        else:
            status_str = "[yellow]running[/]"

        # Format loss
        loss_str = ""
        if run.get("initial_loss") and run.get("final_loss"):
            loss_str = f"{run['initial_loss']:.3f} -> {run['final_loss']:.3f}"

        # Format duration
        duration_str = ""
        if run.get("duration_secs"):
            secs = run["duration_secs"]
            if secs >= 3600:
                duration_str = f"{secs / 3600:.1f}h"
            elif secs >= 60:
                duration_str = f"{secs / 60:.0f}m"
            else:
                duration_str = f"{secs:.0f}s"

        # Format date (just date + time, no seconds)
        date_str = run["created_at"][:16].replace("T", " ")

        # Shorten run_id for display
        short_id = run["run_id"]

        table.add_row(
            short_id,
            run.get("experiment_name") or "",
            run.get("base_model") or "",
            run.get("task") or "",
            status_str,
            loss_str,
            str(run.get("total_steps") or ""),
            duration_str,
            date_str,
        )

    console.print(table)


@app.command()
def show(
    run_id: str = typer.Argument(..., help="Run ID (or prefix) to show"),
    plot: bool = typer.Option(True, "--plot/--no-plot", help="Show loss curve"),
):
    """Show detailed info about a specific run, including loss curve."""
    from soup_cli.experiment.tracker import ExperimentTracker

    tracker = ExperimentTracker()
    run = tracker.get_run(run_id)

    if not run:
        console.print(f"[red]Run not found: {run_id}[/]")
        console.print("[dim]Use [bold]soup runs[/] to see all runs.[/]")
        raise typer.Exit(1)

    # Format status
    status = run["status"]
    if status == "completed":
        status_str = "[green]completed[/]"
    elif status == "failed":
        status_str = "[red]failed[/]"
    else:
        status_str = "[yellow]running[/]"

    # Format duration
    duration_str = "-"
    if run.get("duration_secs"):
        secs = run["duration_secs"]
        hours = int(secs // 3600)
        minutes = int((secs % 3600) // 60)
        duration_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"

    # Build info panel
    info_lines = [
        f"Run ID:     [bold]{run['run_id']}[/]",
        f"Name:       {run.get('experiment_name') or '-'}",
        f"Status:     {status_str}",
        f"Date:       {run['created_at'][:19].replace('T', ' ')}",
        "",
        f"Model:      [bold]{run.get('base_model') or '-'}[/]",
        f"Task:       {run.get('task') or '-'}",
        f"Device:     {run.get('device_name') or '-'} ({run.get('device') or '-'})",
        f"GPU Memory: {run.get('gpu_memory') or '-'}",
        "",
        f"Loss:       {_fmt_loss(run)}",
        f"Steps:      {run.get('total_steps') or '-'}",
        f"Duration:   {duration_str}",
        f"Output:     {run.get('output_dir') or '-'}",
    ]
    console.print(Panel("\n".join(info_lines), title="Run Details"))

    # Config section
    if run.get("config_json"):
        try:
            config = json.loads(run["config_json"])
            config_str = json.dumps(config, indent=2, default=str)
            # Truncate long configs
            if len(config_str) > 1500:
                config_str = config_str[:1500] + "\n..."
            console.print(Panel(config_str, title="Config"))
        except json.JSONDecodeError:
            pass

    # Eval results
    eval_results = tracker.get_eval_results(run_id=run["run_id"])
    if eval_results:
        eval_table = Table(title="Evaluation Results")
        eval_table.add_column("Benchmark", style="bold")
        eval_table.add_column("Score", justify="right")
        for result in eval_results:
            eval_table.add_row(result["benchmark"], f"{result['score']:.4f}")
        console.print(eval_table)

    # Loss curve
    if plot:
        metrics = tracker.get_metrics(run["run_id"])
        if metrics:
            _plot_loss_curve(metrics)


@app.command()
def compare(
    run_1: str = typer.Argument(..., help="First run ID (or prefix)"),
    run_2: str = typer.Argument(..., help="Second run ID (or prefix)"),
):
    """Compare two training runs side by side."""
    from soup_cli.experiment.tracker import ExperimentTracker

    tracker = ExperimentTracker()
    r1 = tracker.get_run(run_1)
    r2 = tracker.get_run(run_2)

    if not r1:
        console.print(f"[red]Run not found: {run_1}[/]")
        raise typer.Exit(1)
    if not r2:
        console.print(f"[red]Run not found: {run_2}[/]")
        raise typer.Exit(1)

    table = Table(title="Run Comparison")
    table.add_column("Metric", style="bold")
    table.add_column(r1["run_id"][:20], justify="right")
    table.add_column(r2["run_id"][:20], justify="right")

    rows = [
        ("Name", r1.get("experiment_name") or "-", r2.get("experiment_name") or "-"),
        ("Model", r1.get("base_model") or "-", r2.get("base_model") or "-"),
        ("Task", r1.get("task") or "-", r2.get("task") or "-"),
        ("Device", r1.get("device_name") or "-", r2.get("device_name") or "-"),
        ("Status", r1.get("status") or "-", r2.get("status") or "-"),
        ("Initial Loss", _fmt_float(r1.get("initial_loss")), _fmt_float(r2.get("initial_loss"))),
        ("Final Loss", _fmt_float(r1.get("final_loss")), _fmt_float(r2.get("final_loss"))),
        ("Steps", str(r1.get("total_steps") or "-"), str(r2.get("total_steps") or "-")),
        ("Duration", _fmt_duration(r1.get("duration_secs")),
         _fmt_duration(r2.get("duration_secs"))),
    ]

    # Add config comparison for key fields
    for run_data in [r1, r2]:
        if run_data.get("config_json"):
            try:
                run_data["_config"] = json.loads(run_data["config_json"])
            except json.JSONDecodeError:
                run_data["_config"] = {}

    c1 = r1.get("_config", {})
    c2 = r2.get("_config", {})

    training1 = c1.get("training", {})
    training2 = c2.get("training", {})
    rows.extend([
        ("Epochs", str(training1.get("epochs", "-")), str(training2.get("epochs", "-"))),
        ("Learning Rate", str(training1.get("lr", "-")), str(training2.get("lr", "-"))),
        ("Batch Size", str(training1.get("batch_size", "-")),
         str(training2.get("batch_size", "-"))),
        ("Quantization", str(training1.get("quantization", "-")),
         str(training2.get("quantization", "-"))),
    ])

    lora1 = training1.get("lora", {})
    lora2 = training2.get("lora", {})
    rows.extend([
        ("LoRA r", str(lora1.get("r", "-")), str(lora2.get("r", "-"))),
        ("LoRA alpha", str(lora1.get("alpha", "-")), str(lora2.get("alpha", "-"))),
    ])

    for label, val1, val2 in rows:
        # Highlight differences
        if val1 != val2:
            table.add_row(label, f"[yellow]{val1}[/]", f"[yellow]{val2}[/]")
        else:
            table.add_row(label, val1, val2)

    console.print(table)


@app.command()
def delete(
    run_id: str = typer.Argument(..., help="Run ID (or prefix) to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Delete a training run and its metrics."""
    from soup_cli.experiment.tracker import ExperimentTracker

    tracker = ExperimentTracker()
    run = tracker.get_run(run_id)

    if not run:
        console.print(f"[red]Run not found: {run_id}[/]")
        raise typer.Exit(1)

    if not force:
        if not typer.confirm(f"Delete run {run['run_id']}?"):
            raise typer.Exit()

    tracker.delete_run(run["run_id"])
    console.print(f"[green]Deleted run: {run['run_id']}[/]")


@app.command()
def clean(
    run_id: Optional[str] = typer.Argument(None, help="Run ID (or prefix) to clean. Omit if --all is used."),
    all_runs: bool = typer.Option(False, "--all", help="Cleanup all historical runs."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Estimate space savings without deleting."),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
    keep_weights: bool = typer.Option(
        True, "--keep-weights",
        help="Keep intermediate model weights but delete optimizer states."
    ),
):
    """Intelligently clean up redundant checkpoint files to reclaim disk space."""
    import shutil
    from pathlib import Path
    from soup_cli.experiment.tracker import ExperimentTracker

    tracker = ExperimentTracker()

    if all_runs:
        runs_to_clean = tracker.list_runs(limit=100000)
    elif run_id:
        run = tracker.get_run(run_id)
        if not run:
            console.print(f"[red]Run not found: {run_id}[/]")
            raise typer.Exit(1)
        runs_to_clean = [run]
    else:
        console.print("[red]Must specify RUN_ID or use --all[/]")
        console.print("[dim]Example: soup runs clean run_2026...[/]")
        raise typer.Exit(1)

    if not runs_to_clean:
        console.print("[yellow]No runs to clean.[/]")
        raise typer.Exit()

    total_bytes_to_reclaim = 0
    files_to_delete = []
    dirs_to_delete = []

    for run in runs_to_clean:
        out_dir_str = run.get("output_dir")
        if not out_dir_str:
            continue
        output_dir = Path(out_dir_str)
        if not output_dir.exists():
            continue

        metrics = tracker.get_metrics(run["run_id"])

        best_step = -1
        valid_metrics = [m for m in metrics if m.get("loss") is not None]
        if valid_metrics:
            best_metric = min(valid_metrics, key=lambda x: x["loss"])
            best_step = best_metric["step"]

        checkpoints = [d for d in output_dir.glob("checkpoint-*") if d.is_dir()]
        for ckpt in checkpoints:
            is_best = False
            try:
                step = int(ckpt.name.split("-")[-1])
                if step == best_step:
                    is_best = True
            except ValueError:
                pass

            if is_best:
                continue

            if keep_weights:
                for opt_file in ckpt.glob("optimizer*.pt"):
                    total_bytes_to_reclaim += opt_file.stat().st_size
                    files_to_delete.append(opt_file)
                for opt_file in ckpt.glob("optimizer*.safetensors"):
                    total_bytes_to_reclaim += opt_file.stat().st_size
                    files_to_delete.append(opt_file)
                for sch_file in ckpt.glob("scheduler.pt"):
                    total_bytes_to_reclaim += sch_file.stat().st_size
                    files_to_delete.append(sch_file)
            else:
                size = sum(f.stat().st_size for f in ckpt.rglob('*') if f.is_file())
                total_bytes_to_reclaim += size
                dirs_to_delete.append(ckpt)

    if total_bytes_to_reclaim == 0:
        console.print("[green]No disposable checkpoint files found. Storage is already optimized.[/]")
        raise typer.Exit()

    gb_to_reclaim = total_bytes_to_reclaim / (1024 ** 3)

    if dry_run:
        console.print(f"[bold]Dry Run:[/] Would reclaim [green]{gb_to_reclaim:.2f} GB[/] "
                      f"from {len(files_to_delete)} files "
                      f"and {len(dirs_to_delete)} directories.")
        for d in dirs_to_delete:
            console.print(f"  [red]Delete dir:[/]\t{d}")
        for f in files_to_delete:
            console.print(f"  [red]Delete file:[/]\t{f}")
        raise typer.Exit()

    if not force:
        console.print(f"Ready to reclaim [green]{gb_to_reclaim:.2f} GB[/] by pruning intermediate files.")
        if not typer.confirm("Do you want to proceed?"):
            raise typer.Exit()

    for f in files_to_delete:
        try:
            f.unlink()
        except OSError:
            pass
    for d in dirs_to_delete:
        try:
            shutil.rmtree(d)
        except OSError:
            pass

    console.print(f"[green]Successfully reclaimed {gb_to_reclaim:.2f} GB.[/]")


def _fmt_loss(run: dict) -> str:
    """Format loss as 'initial -> final'."""
    init = run.get("initial_loss")
    final = run.get("final_loss")
    if init is not None and final is not None:
        return f"{init:.4f} -> {final:.4f}"
    return "-"


def _fmt_float(val: Optional[float]) -> str:
    """Format a float or return '-'."""
    if val is not None:
        return f"{val:.4f}"
    return "-"


def _fmt_duration(secs: Optional[float]) -> str:
    """Format duration in seconds to human-readable string."""
    if secs is None:
        return "-"
    if secs >= 3600:
        return f"{secs / 3600:.1f}h"
    if secs >= 60:
        return f"{secs / 60:.0f}m"
    return f"{secs:.0f}s"


def _plot_loss_curve(metrics: list[dict]) -> None:
    """Render a loss-over-steps chart in the terminal using plotext."""
    try:
        import plotext as plt
    except ImportError:
        console.print(
            "[yellow]Install plotext for terminal charts:[/] "
            "[bold]pip install plotext[/]"
        )
        return

    steps = [m["step"] for m in metrics if m.get("loss")]
    losses = [m["loss"] for m in metrics if m.get("loss")]

    if not steps:
        console.print("[dim]No loss data to plot.[/]")
        return

    plt.clear_figure()
    plt.plot(steps, losses, label="loss")
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.theme("dark")
    plt.show()
