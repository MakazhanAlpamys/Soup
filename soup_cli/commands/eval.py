"""soup eval — evaluate models on standard benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def eval_model(
    model: str = typer.Option(
        ..., "--model", "-m",
        help="Path to model or LoRA adapter directory",
    ),
    benchmarks: str = typer.Option(
        "mmlu", "--benchmarks", "-b",
        help="Comma-separated benchmark names (mmlu, gsm8k, hellaswag, etc.)",
    ),
    num_fewshot: Optional[int] = typer.Option(
        None, "--fewshot", "-f",
        help="Number of few-shot examples (benchmark default if not set)",
    ),
    batch_size: int = typer.Option(
        8, "--batch-size",
        help="Batch size for evaluation",
    ),
    run_id: Optional[str] = typer.Option(
        None, "--run-id",
        help="Link results to an existing training run",
    ),
    device: Optional[str] = typer.Option(
        None, "--device",
        help="Device: cuda, mps, cpu. Auto-detected if not set.",
    ),
):
    """Evaluate a model on standard benchmarks (wraps lm-evaluation-harness)."""
    model_path = Path(model)
    if not model_path.exists():
        console.print(f"[red]Model path not found: {model_path}[/]")
        raise typer.Exit(1)

    # Check for LoRA adapter and resolve base model
    adapter_config = model_path / "adapter_config.json"
    model_arg = str(model_path)
    if adapter_config.exists():
        import json

        with open(adapter_config) as f:
            adapter_info = json.load(f)
        base_model = adapter_info.get("base_model_name_or_path", "")
        if base_model:
            model_arg = f"pretrained={base_model},peft={model_path},trust_remote_code=True"
            console.print(
                f"[dim]LoRA adapter detected. Base model: {base_model}[/]"
            )
        else:
            model_arg = f"pretrained={model_path},trust_remote_code=True"
    else:
        model_arg = f"pretrained={model_path},trust_remote_code=True"

    benchmark_list = [b.strip() for b in benchmarks.split(",")]
    console.print(
        f"[dim]Evaluating on: {', '.join(benchmark_list)}[/]"
    )

    # Lazy import lm_eval
    try:
        import lm_eval  # noqa: F401
    except ImportError:
        console.print(
            "[red]lm-eval not installed.[/]\n"
            "Install with: [bold]pip install 'soup-cli[eval]'[/]"
        )
        raise typer.Exit(1)

    # Detect device
    if not device:
        from soup_cli.utils.gpu import detect_device

        device, _ = detect_device()

    # Run evaluation
    console.print("[dim]Running evaluation (this may take a while)...[/]")
    results = _run_evaluation(
        model_arg=model_arg,
        tasks=benchmark_list,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
    )

    # Display results
    _display_results(results, benchmark_list)

    # Save to experiment tracker
    _save_results(results, str(model_path), benchmark_list, run_id)

    console.print("\n[green]Results saved to experiment tracker.[/]")
    if run_id:
        console.print(f"[dim]Linked to run: {run_id}[/]")


def _run_evaluation(
    model_arg: str,
    tasks: list[str],
    num_fewshot: Optional[int],
    batch_size: int,
    device: str,
) -> dict:
    """Run lm-evaluation-harness and return results dict."""
    import lm_eval

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_arg,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
    )
    return results


def _display_results(results: dict, benchmarks: list[str]) -> None:
    """Display evaluation results as a rich table."""
    table = Table(title="Evaluation Results")
    table.add_column("Benchmark", style="bold")
    table.add_column("Metric", style="dim")
    table.add_column("Score", justify="right", style="green")

    task_results = results.get("results", {})

    for benchmark in benchmarks:
        bench_data = task_results.get(benchmark, {})
        if not bench_data:
            table.add_row(benchmark, "—", "[red]not found[/]")
            continue

        # Try common metric names
        for metric_key in ["acc,none", "acc_norm,none", "exact_match,none", "em,none"]:
            if metric_key in bench_data:
                metric_name = metric_key.split(",")[0]
                score = bench_data[metric_key]
                table.add_row(benchmark, metric_name, f"{score:.4f}")
                break
        else:
            # Show first numeric result
            for key, val in bench_data.items():
                if isinstance(val, (int, float)) and not key.startswith("alias"):
                    metric_name = key.split(",")[0] if "," in key else key
                    table.add_row(benchmark, metric_name, f"{val:.4f}")
                    break
            else:
                table.add_row(benchmark, "—", "[yellow]no numeric result[/]")

    console.print(table)


def _save_results(
    results: dict,
    model_path: str,
    benchmarks: list[str],
    run_id: Optional[str],
) -> None:
    """Save evaluation results to the experiment tracker."""
    from soup_cli.experiment.tracker import ExperimentTracker

    tracker = ExperimentTracker()
    task_results = results.get("results", {})

    for benchmark in benchmarks:
        bench_data = task_results.get(benchmark, {})
        # Find the primary score
        score = 0.0
        for metric_key in ["acc,none", "acc_norm,none", "exact_match,none", "em,none"]:
            if metric_key in bench_data:
                score = bench_data[metric_key]
                break
        else:
            # Use first numeric value
            for val in bench_data.values():
                if isinstance(val, (int, float)):
                    score = val
                    break

        tracker.save_eval_result(
            model_path=model_path,
            benchmark=benchmark,
            score=score,
            details=bench_data,
            run_id=run_id,
        )
