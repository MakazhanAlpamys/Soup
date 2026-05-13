"""soup data mix — Data Mixing Optimizer CLI (v0.48.0 Part B — BETA).

Two modes:

* ``--optimize`` runs N short proxy training runs and writes a recipe.
* ``--apply <recipe.yaml>`` re-loads a previously written recipe and prints
  the spliceable ``data:`` block.

Live wiring of the proxy training loop into ``soup train`` is deferred to
v0.48.1 (matches the project's stub-then-live pattern). The CLI ships a
synthetic offline proxy so users can exercise the budget tracker, the
optimiser surface, and the recipe writer end-to-end without GPUs.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

console = Console()


def _offline_proxy(weights: Tuple[float, ...]) -> float:
    """Synthetic proxy loss for offline / CI use.

    Penalises mixtures that concentrate too much weight on a single dataset
    so the budget tracker / writer exercises a non-trivial search landscape.
    Live wiring (short ``soup train`` proxy run) lands in v0.48.1.
    """
    # Minimum at uniform mixture; quadratic penalty away from uniform.
    if not weights:
        return float("inf")
    n = len(weights)
    uniform = 1.0 / n
    return sum((w - uniform) ** 2 for w in weights)


def mix(
    optimize: bool = typer.Option(
        False, "--optimize",
        help="Run Bayesian search over dataset mixture weights.",
    ),
    apply_recipe: Optional[str] = typer.Option(
        None, "--apply",
        help="Re-print a previously written mix-recipe (path under cwd).",
    ),
    datasets: Optional[str] = typer.Option(
        None, "--datasets",
        help="Comma-separated list of dataset JSONL paths (>= 2, all under cwd).",
    ),
    budget: str = typer.Option(
        "1h", "--budget",
        help="Wall-clock cap: digits + optional s/m/h suffix (e.g. 1h, 30m).",
    ),
    num_probes: int = typer.Option(
        8, "--num-probes", "-n",
        min=1, max=256,
        help="Maximum number of proxy runs.",
    ),
    seed: int = typer.Option(
        42, "--seed",
        min=0, max=2**31 - 1,
        help="RNG seed for the optimiser.",
    ),
    output: str = typer.Option(
        "mix_recipe.yaml", "--output", "-o",
        help="YAML recipe output path (under cwd).",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite",
        help="Overwrite the output path when it exists.",
    ),
    live: bool = typer.Option(
        False, "--live",
        help=(
            "v0.53.5 #116 — run live short `soup train` proxy runs per "
            "candidate (requires --base-yaml). Defaults to the offline "
            "synthetic proxy."
        ),
    ),
    base_yaml: Optional[str] = typer.Option(
        None, "--base-yaml",
        help=(
            "Path to a base soup.yaml (under cwd) supplying base/task/training/output. "
            "Required with --live."
        ),
    ),
) -> None:
    """BETA: optimise per-dataset mixture weights against a proxy run."""
    from soup_cli.utils.data_mix import (
        build_optimization_plan,
        load_mix_recipe,
        render_mix_recipe_yaml,
        run_mix_optimizer,
        write_mix_recipe,
    )

    if optimize and apply_recipe is not None:
        console.print(
            "[red]Pick exactly one of --optimize or --apply.[/red]"
        )
        raise typer.Exit(code=2)
    if not optimize and apply_recipe is None:
        console.print(
            "[red]Pick one of --optimize or --apply.[/red]"
        )
        raise typer.Exit(code=2)

    if apply_recipe is not None:
        try:
            data_block = load_mix_recipe(apply_recipe)
        except (ValueError, FileNotFoundError, TypeError) as exc:
            console.print(f"[red]apply failed: {escape(str(exc))}[/red]")
            raise typer.Exit(code=2) from exc
        console.print(
            Panel.fit(
                f"[bold]Loaded recipe:[/bold] {escape(apply_recipe)}\n"
                f"[dim]Splice the following block into your soup.yaml[/dim]"
            )
        )
        # Round-trip via the renderer so the user sees the canonical shape.
        train_paths = data_block.get("train", []) if hasattr(
            data_block, "get"
        ) else []
        interleave = (
            data_block.get("interleave", {}) if hasattr(data_block, "get") else {}
        )
        probs = interleave.get("probs", []) if hasattr(interleave, "get") else []
        console.print("data:")
        console.print("  interleave:")
        console.print("    strategy: probs")
        console.print("    probs:")
        for p in probs:
            console.print(f"      - {float(p):.6f}")
        console.print("  train:")
        for path in train_paths:
            console.print(f"    - {escape(str(path))}")
        return

    if not datasets:
        console.print(
            "[red]--datasets is required when --optimize is set.[/red]"
        )
        raise typer.Exit(code=2)
    raw = [p.strip() for p in datasets.split(",") if p.strip()]
    try:
        plan = build_optimization_plan(
            raw, budget=budget, num_probes=num_probes, seed=seed
        )
    except (ValueError, TypeError) as exc:
        console.print(f"[red]plan validation failed: {escape(str(exc))}[/red]")
        raise typer.Exit(code=2) from exc

    console.print(
        Panel.fit(
            f"[bold]Mix Optimizer[/bold]  (BETA)\n"
            f"datasets: {len(plan.datasets)} | "
            f"probes: {plan.num_probes} | "
            f"budget: {plan.budget_seconds}s"
        )
    )

    if live:
        if not base_yaml:
            console.print(
                "[red]--live requires --base-yaml <path/to/soup.yaml>.[/red]"
            )
            raise typer.Exit(code=2)
        from soup_cli.utils.mix_proxy import proxy_run_for_weights

        per_candidate_timeout = max(
            60, min(plan.budget_seconds // max(plan.num_probes, 1), 30 * 60)
        )

        def _live_proxy(w: Tuple[float, ...]) -> float:
            return proxy_run_for_weights(
                w,
                list(plan.datasets),
                base_yaml,
                timeout_seconds=per_candidate_timeout,
            )

        proxy_callable = _live_proxy
    else:
        proxy_callable = _offline_proxy

    report = run_mix_optimizer(plan, proxy_callable)
    try:
        path = write_mix_recipe(report, output, overwrite=overwrite)
    except (ValueError, OSError) as exc:
        console.print(f"[red]write failed: {escape(str(exc))}[/red]")
        raise typer.Exit(code=2) from exc

    if math.isfinite(report.best_eval_loss):
        loss_str = f"{report.best_eval_loss:.6f}"
    else:
        loss_str = "n/a"
    console.print(
        f"[green]wrote recipe:[/green] {escape(path)}  "
        f"(best_loss={loss_str}, partial={report.partial})"
    )
    # Echo the recipe for grep-ability.
    console.print(render_mix_recipe_yaml(report))
