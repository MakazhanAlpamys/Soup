"""soup eval — evaluation platform with benchmarks, custom evals, LLM judge, and more."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

app = typer.Typer(
    name="eval",
    help="Evaluate models: benchmarks, custom evals, LLM judge, leaderboard.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


# ─── soup eval benchmark ───


@app.command()
def benchmark(
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
    """Evaluate on standard benchmarks (wraps lm-evaluation-harness)."""
    model_path = Path(model)
    if not model_path.exists():
        console.print(f"[red]Model path not found: {model_path}[/]")
        raise typer.Exit(1)

    # Check for LoRA adapter and resolve base model
    adapter_config = model_path / "adapter_config.json"
    model_arg = str(model_path)
    if adapter_config.exists():
        adapter_info = json.loads(adapter_config.read_text())
        base_model = adapter_info.get("base_model_name_or_path", "")
        if base_model:
            model_arg = f"pretrained={base_model},peft={model_path}"
            console.print(
                f"[dim]LoRA adapter detected. Base model: {base_model}[/]"
            )
        else:
            model_arg = f"pretrained={model_path}"
    else:
        model_arg = f"pretrained={model_path}"

    benchmark_list = [b.strip() for b in benchmarks.split(",")]
    console.print(f"[dim]Evaluating on: {', '.join(benchmark_list)}[/]")

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

    console.print("[dim]Running evaluation (this may take a while)...[/]")
    results = _run_lm_eval(
        model_arg=model_arg,
        tasks=benchmark_list,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
    )

    _display_benchmark_results(results, benchmark_list)
    _save_benchmark_results(results, str(model_path), benchmark_list, run_id)
    console.print("\n[green]Results saved to experiment tracker.[/]")


# ─── soup eval custom ───


@app.command()
def custom(
    tasks: str = typer.Option(
        ..., "--tasks", "-t",
        help="Path to eval tasks JSONL file",
    ),
    model: str = typer.Option(
        ..., "--model", "-m",
        help="Path to model directory",
    ),
    run_id: Optional[str] = typer.Option(
        None, "--run-id",
        help="Link results to an existing training run",
    ),
    attach_to_registry: Optional[str] = typer.Option(
        None, "--attach-to-registry",
        help="Attach the eval JSON to a registry entry as kind=eval_results",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="Path for the eval JSON output (required with --attach-to-registry)",
    ),
):
    """Run custom evaluation tasks from a JSONL file."""
    from soup_cli.eval.custom import load_eval_tasks

    tasks_path = Path(tasks)
    model_path = Path(model)

    if not model_path.exists():
        console.print(f"[red]Model path not found: {model_path}[/]")
        raise typer.Exit(1)

    # Load and validate tasks
    try:
        eval_tasks = load_eval_tasks(tasks_path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1)

    console.print(
        f"[dim]Loaded {len(eval_tasks)} eval tasks from {tasks_path}[/]"
    )

    # Run evaluation with progress
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
    )

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    )

    results_list = []
    from soup_cli.eval.custom import _create_default_generator, score_task

    console.print("[dim]Loading model...[/]")
    generate_fn = _create_default_generator(str(model_path))

    with progress:
        task_bar = progress.add_task(
            "Evaluating...", total=len(eval_tasks),
        )
        for eval_task in eval_tasks:
            output = generate_fn(eval_task.prompt)
            result = score_task(eval_task, output)
            results_list.append(result)
            progress.advance(task_bar)

    from soup_cli.eval.custom import EvalResults

    eval_results = EvalResults(results=results_list)
    eval_results.compute()

    # Display results
    _display_custom_results(eval_results)

    # Save to tracker
    _save_custom_results(eval_results, str(model_path), run_id)
    console.print("\n[green]Results saved to experiment tracker.[/]")

    # v0.33.0 #35: optional registry attach
    if attach_to_registry:
        if not output:
            console.print(
                "[red]--attach-to-registry requires --output <json-path>[/]"
            )
            raise typer.Exit(1)
        from soup_cli.registry.attach import attach_artifact, write_eval_json

        payload = {
            "model": str(model_path),
            "tasks": str(tasks_path),
            "total": eval_results.total,
            "correct": eval_results.correct,
            "accuracy": eval_results.accuracy,
            "category_scores": eval_results.category_scores,
        }
        try:
            json_path = write_eval_json(output, payload=payload)
            attach_artifact(
                attach_to_registry, path=str(json_path), kind="eval_results",
            )
        except (ValueError, FileNotFoundError) as exc:
            console.print(f"[red]Registry attach failed:[/] {exc}")
            raise typer.Exit(1) from exc
        console.print(
            f"[green]Attached eval results to registry entry "
            f"'{attach_to_registry}' as eval_results.[/]"
        )


# ─── soup eval judge ───


@app.command()
def judge(
    target: str = typer.Option(
        ..., "--target",
        help="Path to JSONL with prompt+response pairs to evaluate",
    ),
    judge_model: str = typer.Option(
        "gpt-4o-mini", "--model", "-m",
        help="Judge model name",
    ),
    provider: str = typer.Option(
        "openai", "--provider",
        help="Judge provider: openai, server, ollama",
    ),
    rubric: Optional[str] = typer.Option(
        None, "--rubric", "-r",
        help="Path to rubric YAML file",
    ),
    api_base: Optional[str] = typer.Option(
        None, "--api-base",
        help="API base URL for judge model",
    ),
    run_id: Optional[str] = typer.Option(
        None, "--run-id",
        help="Link results to an existing training run",
    ),
):
    """Evaluate model outputs using LLM-as-a-judge."""
    from soup_cli.eval.judge import (
        JudgeEvaluator,
        load_rubric,
        validate_judge_api_base,
    )

    # Validate API base
    try:
        validate_judge_api_base(api_base)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1)

    # Load target data
    target_path = Path(target)
    if not target_path.exists():
        console.print(f"[red]Target file not found: {target_path}[/]")
        raise typer.Exit(1)

    items: list[dict] = []
    with open(target_path, encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                console.print(
                    f"[red]Invalid JSON on line {line_num}: {exc}[/]"
                )
                raise typer.Exit(1)
            if "prompt" not in row or "response" not in row:
                console.print(
                    f"[red]Line {line_num}: missing 'prompt' or 'response'[/]"
                )
                raise typer.Exit(1)
            items.append(row)

    # Load rubric
    rubric_data = None
    if rubric:
        try:
            rubric_data = load_rubric(Path(rubric))
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(1)

    console.print(
        f"[dim]Judging {len(items)} responses with {judge_model} "
        f"(provider: {provider})...[/]"
    )

    # Run judge evaluation
    try:
        evaluator = JudgeEvaluator(
            rubric=rubric_data,
            provider=provider,
            model=judge_model,
            api_base=api_base,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1)

    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
    )

    from soup_cli.eval.judge import JudgeResults

    judge_scores = []
    skipped_count = 0
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    )
    with progress:
        task_bar = progress.add_task("Judging...", total=len(items))
        for item in items:
            try:
                score = evaluator.evaluate(
                    prompt=item["prompt"],
                    response=item["response"],
                    category=item.get("category", "default"),
                )
                judge_scores.append(score)
            except (ValueError, OSError, KeyError) as exc:
                skipped_count += 1
                console.print(
                    f"[yellow]Warning: judge failed for prompt: {exc}[/]"
                )
            progress.advance(task_bar)

    if skipped_count > 0:
        console.print(
            f"[yellow]{skipped_count}/{len(items)} items skipped "
            f"due to judge errors.[/]"
        )

    if not judge_scores:
        console.print("[red]All items failed. Check judge configuration.[/]")
        raise typer.Exit(1)

    results = JudgeResults(scores=judge_scores)
    results.compute()

    _display_judge_results(results)

    # Save to tracker
    if run_id:
        from soup_cli.experiment.tracker import ExperimentTracker
        tracker = ExperimentTracker()
        tracker.save_eval_result(
            model_path=target,
            benchmark=f"judge:{judge_model}",
            score=results.overall_score,
            details={
                "criteria_averages": results.criteria_averages,
                "category_scores": results.category_scores,
                "total_evaluated": len(results.scores),
            },
            run_id=run_id,
        )
        console.print("[green]Results saved to experiment tracker.[/]")


# ─── soup eval auto ───


@app.command()
def auto(
    config: str = typer.Option(
        "soup.yaml", "--config", "-c",
        help="Path to soup.yaml config file",
    ),
    benchmarks: Optional[str] = typer.Option(
        None, "--benchmarks", "-b",
        help="Comma-separated benchmarks (overrides config)",
    ),
    custom_tasks: Optional[str] = typer.Option(
        None, "--tasks", "-t",
        help="Path to custom eval JSONL (overrides config)",
    ),
):
    """Run automatic evaluation using config from soup.yaml."""
    from soup_cli.config.loader import load_config

    config_path = Path(config)
    if not config_path.exists():
        console.print(f"[red]Config not found: {config_path}[/]")
        raise typer.Exit(1)

    soup_config = load_config(str(config_path))
    output_dir = Path(soup_config.output)

    if not output_dir.exists():
        console.print(f"[red]Output directory not found: {output_dir}[/]")
        console.print("[dim]Run soup train first to create a model.[/]")
        raise typer.Exit(1)

    eval_config = getattr(soup_config, "eval", None)

    # Determine benchmarks
    bench_list = []
    if benchmarks:
        bench_list = [b.strip() for b in benchmarks.split(",")]
    elif eval_config and hasattr(eval_config, "benchmarks"):
        bench_list = eval_config.benchmarks or []

    # Determine custom tasks
    tasks_file = custom_tasks
    if not tasks_file and eval_config and hasattr(eval_config, "custom_tasks"):
        tasks_file = eval_config.custom_tasks

    if not bench_list and not tasks_file:
        console.print(
            "[yellow]No benchmarks or custom tasks specified.[/]\n"
            "Use --benchmarks or --tasks, or add eval config to soup.yaml."
        )
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold]Auto-Eval[/]\n"
        f"Model: {output_dir}\n"
        f"Benchmarks: {', '.join(bench_list) if bench_list else 'none'}\n"
        f"Custom tasks: {tasks_file or 'none'}",
        title="Evaluation",
        border_style="blue",
    ))

    # Run standard benchmarks if specified
    if bench_list:
        console.print("\n[bold]Standard Benchmarks[/]")
        try:
            benchmark(
                model=str(output_dir),
                benchmarks=",".join(bench_list),
                num_fewshot=None,
                batch_size=8,
                run_id=None,
                device=None,
            )
        except SystemExit:
            console.print("[yellow]Benchmark eval skipped (see above).[/]")

    # Run custom eval if specified
    if tasks_file:
        console.print("\n[bold]Custom Evaluation[/]")
        try:
            custom(
                tasks=tasks_file,
                model=str(output_dir),
                run_id=None,
            )
        except SystemExit:
            console.print("[yellow]Custom eval skipped (see above).[/]")

    console.print("\n[green]Auto-eval complete.[/]")


# ─── soup eval compare ───


@app.command()
def compare(
    run1: str = typer.Argument(..., help="First run ID"),
    run2: str = typer.Argument(..., help="Second run ID"),
):
    """Compare eval results between two training runs."""
    from soup_cli.eval.leaderboard import compare_runs
    from soup_cli.experiment.tracker import ExperimentTracker

    tracker = ExperimentTracker()

    # Verify runs exist
    for rid in [run1, run2]:
        run_data = tracker.get_run(rid)
        if run_data is None:
            console.print(f"[red]Run not found: {rid}[/]")
            raise typer.Exit(1)

    comparison = compare_runs(tracker, run1, run2)

    if not comparison["comparisons"]:
        console.print("[yellow]No eval results found for these runs.[/]")
        console.print("[dim]Run soup eval benchmark --run-id ... first.[/]")
        raise typer.Exit(1)

    # Display comparison table
    table = Table(title=f"Eval Comparison: {run1} vs {run2}")
    table.add_column("Benchmark", style="bold")
    table.add_column("Run 1", justify="right")
    table.add_column("Run 2", justify="right")
    table.add_column("Delta", justify="right")

    for comp in comparison["comparisons"]:
        score_a = (
            f"{comp['run_1_score']:.4f}" if comp["run_1_score"] is not None
            else "-"
        )
        score_b = (
            f"{comp['run_2_score']:.4f}" if comp["run_2_score"] is not None
            else "-"
        )
        if comp["delta"] is not None:
            delta_val = comp["delta"]
            if delta_val > 0.01:
                delta_str = f"[green]+{delta_val:.4f}[/]"
            elif delta_val < -0.01:
                delta_str = f"[red]{delta_val:.4f}[/]"
            else:
                delta_str = f"{delta_val:.4f}"
        else:
            delta_str = "-"

        table.add_row(comp["benchmark"], score_a, score_b, delta_str)

    console.print(table)

    if comparison["has_regressions"]:
        console.print(
            f"\n[red bold]Regressions detected:[/] "
            f"{', '.join(comparison['regressions'])}"
        )


# ─── soup eval leaderboard ───


@app.command()
def leaderboard(
    sort_by: Optional[str] = typer.Option(
        None, "--sort-by", "-s",
        help="Sort by specific benchmark (default: average)",
    ),
    fmt: str = typer.Option(
        "table", "--format", "-f",
        help="Output format: table, json, csv",
    ),
):
    """Show local leaderboard across all evaluated models."""
    from soup_cli.eval.leaderboard import (
        build_leaderboard_from_tracker,
        export_leaderboard,
    )
    from soup_cli.experiment.tracker import ExperimentTracker

    tracker = ExperimentTracker()
    lb = build_leaderboard_from_tracker(tracker)

    if not lb.entries:
        console.print("[yellow]No eval results found.[/]")
        console.print(
            "[dim]Run soup eval benchmark or soup eval custom first.[/]"
        )
        raise typer.Exit(1)

    if fmt in ("json", "csv"):
        output = export_leaderboard(lb, fmt=fmt)
        console.print(output)
        return

    # Table format
    sorted_models = lb.get_sorted_models(sort_by=sort_by)

    # Collect all benchmarks
    all_benchmarks: set[str] = set()
    for _, scores, _ in sorted_models:
        all_benchmarks.update(scores.keys())
    benchmarks = sorted(all_benchmarks)

    table = Table(title="Eval Leaderboard")
    table.add_column("#", style="dim", width=3)
    table.add_column("Model", style="bold")
    for bench in benchmarks:
        table.add_column(bench, justify="right")
    table.add_column("Avg", justify="right", style="green")

    for rank, (model_name, scores, avg) in enumerate(sorted_models, 1):
        row = [str(rank), _short_model_name(model_name)]
        for bench in benchmarks:
            if bench in scores:
                row.append(f"{scores[bench]:.4f}")
            else:
                row.append("-")
        row.append(f"{avg:.4f}")
        table.add_row(*row)

    console.print(table)


# ─── soup eval human ───


@app.command()
def human(
    prompts_file: str = typer.Option(
        ..., "--input", "-i",
        help="Path to JSONL file with evaluation prompts",
    ),
    model_a: str = typer.Option(
        ..., "--model-a", "-a",
        help="Path to first model",
    ),
    model_b: str = typer.Option(
        ..., "--model-b", "-b",
        help="Path to second model",
    ),
    output: str = typer.Option(
        "human_eval_results.json", "--output", "-o",
        help="Output file for results",
    ),
):
    """Run human A/B evaluation between two models."""
    from soup_cli.eval.human import (
        HumanEvalResults,
        HumanJudgment,
        load_prompts,
        save_results,
    )

    # Load prompts
    try:
        prompts = load_prompts(Path(prompts_file))
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1)

    # Verify models exist
    for mpath in [model_a, model_b]:
        if not Path(mpath).exists():
            console.print(f"[red]Model not found: {mpath}[/]")
            raise typer.Exit(1)

    console.print(Panel(
        f"[bold]Human Evaluation[/]\n"
        f"Prompts: {len(prompts)}\n"
        f"Model A: {model_a}\n"
        f"Model B: {model_b}",
        title="A/B Comparison",
        border_style="blue",
    ))

    # Generate responses
    console.print("[dim]Loading models and generating responses...[/]")
    from soup_cli.eval.custom import _create_default_generator

    gen_a = _create_default_generator(model_a)
    gen_b = _create_default_generator(model_b)

    results = HumanEvalResults()

    for idx, prompt_data in enumerate(prompts):
        prompt_text = prompt_data["prompt"]
        resp_a = gen_a(prompt_text)
        resp_b = gen_b(prompt_text)

        # Display for human judgment
        console.print(f"\n[bold]--- Prompt {idx + 1}/{len(prompts)} ---[/]")
        console.print(Panel(prompt_text, title="Prompt", border_style="dim"))
        console.print(Panel(resp_a, title="[blue]Response A[/]"))
        console.print(Panel(resp_b, title="[magenta]Response B[/]"))

        # Get human choice
        while True:
            choice = console.input(
                "[bold]Winner? (a/b/tie/q to quit): [/]"
            ).strip().lower()
            if choice in ("a", "b", "tie", "q"):
                break
            console.print("[yellow]Enter a, b, tie, or q[/]")

        if choice == "q":
            console.print("[dim]Session ended early.[/]")
            break

        results.judgments.append(HumanJudgment(
            prompt=prompt_text,
            response_a=resp_a,
            response_b=resp_b,
            model_a=model_a,
            model_b=model_b,
            winner=choice,
        ))

    results.compute_ratings()

    # Display Elo ratings
    _display_elo_ratings(results)

    # Save results
    save_results(results, Path(output))
    console.print(f"\n[green]Results saved to {output}[/]")


# ─── Helper functions ───


def _run_lm_eval(
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


def _display_benchmark_results(results: dict, benchmarks: list[str]) -> None:
    """Display benchmark results as a rich table."""
    table = Table(title="Evaluation Results")
    table.add_column("Benchmark", style="bold")
    table.add_column("Metric", style="dim")
    table.add_column("Score", justify="right", style="green")

    task_results = results.get("results", {})
    for bench in benchmarks:
        bench_data = task_results.get(bench, {})
        if not bench_data:
            table.add_row(bench, "-", "[red]not found[/]")
            continue

        for metric_key in [
            "acc,none", "acc_norm,none", "exact_match,none", "em,none",
        ]:
            if metric_key in bench_data:
                metric_name = metric_key.split(",")[0]
                score = bench_data[metric_key]
                table.add_row(bench, metric_name, f"{score:.4f}")
                break
        else:
            for key, val in bench_data.items():
                if isinstance(val, (int, float)) and not key.startswith("alias"):
                    metric_name = key.split(",")[0] if "," in key else key
                    table.add_row(bench, metric_name, f"{val:.4f}")
                    break
            else:
                table.add_row(bench, "-", "[yellow]no numeric result[/]")

    console.print(table)


def _save_benchmark_results(
    results: dict,
    model_path: str,
    benchmarks: list[str],
    run_id: Optional[str],
) -> None:
    """Save benchmark results to the experiment tracker."""
    from soup_cli.experiment.tracker import ExperimentTracker

    tracker = ExperimentTracker()
    task_results = results.get("results", {})

    for bench in benchmarks:
        bench_data = task_results.get(bench, {})
        score = 0.0
        for metric_key in [
            "acc,none", "acc_norm,none", "exact_match,none", "em,none",
        ]:
            if metric_key in bench_data:
                score = bench_data[metric_key]
                break
        else:
            for val in bench_data.values():
                if isinstance(val, (int, float)):
                    score = val
                    break

        tracker.save_eval_result(
            model_path=model_path,
            benchmark=bench,
            score=score,
            details=bench_data,
            run_id=run_id,
        )


def _display_custom_results(eval_results: object) -> None:
    """Display custom eval results as a rich table."""
    table = Table(title="Custom Eval Results")
    table.add_column("Category", style="bold")
    table.add_column("Correct", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Accuracy", justify="right", style="green")

    for cat, cat_data in eval_results.category_scores.items():
        table.add_row(
            cat,
            str(cat_data["correct"]),
            str(cat_data["total"]),
            f"{cat_data['accuracy']:.2%}",
        )

    # Overall
    table.add_section()
    table.add_row(
        "[bold]Overall[/]",
        str(eval_results.correct),
        str(eval_results.total),
        f"[bold green]{eval_results.accuracy:.2%}[/]",
    )

    console.print(table)


def _save_custom_results(
    eval_results: object,
    model_path: str,
    run_id: Optional[str],
) -> None:
    """Save custom eval results to the experiment tracker."""
    from soup_cli.experiment.tracker import ExperimentTracker

    tracker = ExperimentTracker()
    tracker.save_eval_result(
        model_path=model_path,
        benchmark="custom",
        score=eval_results.accuracy,
        details={
            "total": eval_results.total,
            "correct": eval_results.correct,
            "category_scores": eval_results.category_scores,
        },
        run_id=run_id,
    )


def _display_judge_results(results: object) -> None:
    """Display judge evaluation results."""
    # Criteria averages
    table = Table(title="Judge Evaluation Results")
    table.add_column("Criterion", style="bold")
    table.add_column("Average Score", justify="right", style="green")

    for crit, avg in results.criteria_averages.items():
        table.add_row(crit, f"{avg:.2f}")

    table.add_section()
    table.add_row(
        "[bold]Overall[/]",
        f"[bold green]{results.overall_score:.2f}[/]",
    )

    console.print(table)

    # Category breakdown
    if len(results.category_scores) > 1:
        cat_table = Table(title="Scores by Category")
        cat_table.add_column("Category", style="bold")
        cat_table.add_column("Average", justify="right", style="green")

        for cat, score in results.category_scores.items():
            cat_table.add_row(cat, f"{score:.2f}")

        console.print(cat_table)


def _display_elo_ratings(results: object) -> None:
    """Display Elo ratings from human evaluation."""
    table = Table(title="Elo Ratings")
    table.add_column("Model", style="bold")
    table.add_column("Rating", justify="right", style="green")
    table.add_column("W", justify="right")
    table.add_column("L", justify="right")
    table.add_column("T", justify="right")

    for name, rating in sorted(
        results.ratings.items(),
        key=lambda item: item[1].rating,
        reverse=True,
    ):
        table.add_row(
            _short_model_name(name),
            f"{rating.rating:.0f}",
            str(rating.wins),
            str(rating.losses),
            str(rating.ties),
        )

    console.print(table)


def _short_model_name(path: str) -> str:
    """Shorten model path for display."""
    parts = Path(path).parts
    if len(parts) > 2:
        return "/".join(parts[-2:])
    return path


# ─── soup eval gate (v0.26.0 Part B) ───


@app.command(name="gate")
def gate_cmd(
    suite: str = typer.Option(
        ..., "--suite", "-s",
        help="Path to eval suite YAML (see evals/gate.yaml example)",
    ),
    baseline: Optional[str] = typer.Option(
        None, "--baseline", "-b",
        help="Baseline: registry://<id> or path to {name: score} JSON file",
    ),
    regression_threshold: float = typer.Option(
        0.05, "--regression-threshold",
        help="Max absolute drop vs baseline before regression fires (0.0-1.0)",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Model path or HF id to evaluate (required for live scoring)",
    ),
) -> None:
    """Run an eval-gate suite standalone (post-hoc verdict)."""
    if not 0.0 <= regression_threshold <= 1.0:
        console.print(
            "[red]--regression-threshold must be between 0.0 and 1.0[/]"
        )
        raise typer.Exit(1)

    from soup_cli.eval.gate import load_suite, resolve_baseline, run_gate

    try:
        eval_suite = load_suite(suite)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Cannot load suite:[/] {exc}")
        raise typer.Exit(1) from exc

    try:
        baseline_scores = resolve_baseline(baseline)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Cannot resolve baseline:[/] {exc}")
        raise typer.Exit(1) from exc

    # When --model is provided, build a transformers-backed generator.
    # Otherwise fall back to an empty-string stub for smoke runs.
    if model is None:
        console.print(
            "[yellow]No --model given; using stub generator "
            "(empty output per prompt) for a smoke run.[/]"
        )

        def _stub_generate(_: str) -> str:
            return ""

        generate_fn = _stub_generate
    else:
        from soup_cli.eval.quant_check import make_model_generator

        try:
            generate_fn = make_model_generator(model)
        except (OSError, ValueError, ImportError) as exc:
            console.print(
                f"[red]Failed to load --model '{model}':[/] {exc}"
            )
            raise typer.Exit(1) from exc

    result = run_gate(
        eval_suite, generate_fn=generate_fn, baseline=baseline_scores,
        regression_threshold=regression_threshold,
    )

    _print_gate_result(result)
    raise typer.Exit(0 if result.passed else 1)


# ─── soup eval quant-check (v0.26.0 Part D) ───


@app.command(name="quant-check")
def quant_check_cmd(
    before: str = typer.Option(
        ..., "--before",
        help="Before-quantization model path or registry://<id>",
    ),
    after: str = typer.Option(
        ..., "--after",
        help="After-quantization model path or registry://<id>",
    ),
    tasks: str = typer.Option(
        ..., "--tasks",
        help="JSONL eval tasks file (see 'soup eval custom')",
    ),
    fmt: str = typer.Option(
        "table", "--format",
        help="Output format: table | json | markdown",
    ),
) -> None:
    """Compare accuracy before vs after quantization on the same eval suite.

    Runs the same JSONL eval tasks through both models sequentially (memory
    safe) and renders a per-task delta with OK / MINOR / MAJOR verdicts.
    Wiring live model loading is post-v0.26.0; until then, this runs with a
    stub generator so the orchestration layer is still usable for tests and
    CI smoke-checks.
    """
    from soup_cli.eval.quant_check import (
        ensure_format,
        is_under_cwd,
        render,
        resolve_model_ref,
        run_quant_check,
        stub_generator,
    )

    try:
        ensure_format(fmt)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc

    resolved_before = resolve_model_ref(before)
    resolved_after = resolve_model_ref(after)
    if resolved_before is None:
        console.print(f"[red]Cannot resolve --before: {before}[/]")
        raise typer.Exit(1)
    if resolved_after is None:
        console.print(f"[red]Cannot resolve --after: {after}[/]")
        raise typer.Exit(1)

    for label, path_str in (("--before", resolved_before),
                            ("--after", resolved_after),
                            ("--tasks", tasks)):
        path_obj = Path(path_str)
        if not is_under_cwd(path_obj):
            console.print(
                f"[red]{label} '{path_str}' is outside cwd - refusing[/]"
            )
            raise typer.Exit(1)
        if not path_obj.exists() and label == "--tasks":
            console.print(f"[red]{label} not found: {path_str}[/]")
            raise typer.Exit(1)

    before_path = Path(resolved_before)
    after_path = Path(resolved_after)
    if not before_path.exists():
        console.print(f"[red]--before not found: {resolved_before}[/]")
        raise typer.Exit(1)
    if not after_path.exists():
        console.print(f"[red]--after not found: {resolved_after}[/]")
        raise typer.Exit(1)

    # Live model scoring: build transformers-backed generators per side.
    # Falls back to deterministic stubs if loading fails (e.g. missing deps),
    # so CI without GPUs can still smoke-test the orchestration layer.
    from soup_cli.eval.quant_check import make_model_generator

    try:
        before_gen = make_model_generator(resolved_before)
        after_gen = make_model_generator(resolved_after)
    except (OSError, ValueError, ImportError) as exc:
        console.print(
            f"[yellow]Live model load failed ({exc}); using deterministic stub.[/]"
        )
        before_gen = stub_generator("before")
        after_gen = stub_generator("after")

    result = run_quant_check(
        before_gen=before_gen,
        after_gen=after_gen,
        tasks_file=tasks,
    )
    rendered = render(result, fmt=fmt)
    if fmt == "table":
        console.print(rendered)
    else:
        # Plain text (markdown / json) — skip Rich markup interpretation so
        # pipe chars in markdown don't render as Rich tags.
        console.print(rendered, markup=False)


def _print_gate_result(result) -> None:
    """Render a GateResult as a Rich table + pass/fail panel."""
    table = Table(title="Eval gate")
    table.add_column("Task", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("Baseline", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Verdict")
    for row in result.task_results:
        score_text = (
            f"{row.score:.3f}" if row.score is not None
            else "[red]ERROR[/]"
        )
        verdict = (
            "[green]PASS[/]" if row.passed
            else (f"[red]FAIL ({row.error})[/]" if row.error else "[red]FAIL[/]")
        )
        table.add_row(
            row.name,
            score_text,
            f"{row.threshold:.3f}",
            f"{row.baseline:.3f}" if row.baseline is not None else "-",
            f"{row.delta:+.3f}" if row.delta is not None else "-",
            verdict,
        )
    console.print(table)
    verdict = "[green]GATE PASSED[/]" if result.passed else "[red]GATE FAILED[/]"
    if result.regression:
        verdict += " [yellow](regression vs baseline)[/]"
    console.print(Panel(verdict, border_style="green" if result.passed else "red"))
