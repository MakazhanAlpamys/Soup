"""soup data mix — Data Mixing Optimizer CLI (v0.48.0 Part B — BETA).

Modes:

* ``--optimize --live --base-yaml soup.yaml`` runs short proxy training runs
  and writes a recipe only when at least one candidate has a measured loss.
* ``--optimize`` without ``--live`` uses a synthetic offline loss; it exercises
  the search and recipe writer without training or evaluating a model.
* ``--apply <recipe.yaml>`` re-loads a previously written recipe and prints
  the spliceable ``data:`` block.

Offline synthetic results do not establish which mixture improves a model.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from soup_cli.utils.terminal import for_terminal, strip_control

console = Console()


def _offline_proxy(weights: Tuple[float, ...]) -> float:
    """Synthetic proxy loss for offline / CI use.

    Penalises mixtures that concentrate too much weight on a single dataset
    so the budget tracker / writer exercises a non-trivial search landscape.
    This loss does not measure model quality or read dataset contents.
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
        help="Search mixture weights; synthetic offline results unless --live is set.",
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
            "Run live short `soup train` proxy runs per "
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
    """BETA: search mixtures with a synthetic offline proxy, or measured --live runs."""
    import yaml

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
        except (ValueError, FileNotFoundError, TypeError, yaml.YAMLError) as exc:
            # Most messages name the path with repr(), but a YAML error quotes the
            # source line as it is in the file, and PyYAML accepts U+202E there.
            console.print(f"[red]apply failed: {for_terminal(exc)}[/red]", emoji=False)
            raise typer.Exit(code=2) from exc

        train_value = data_block.get("train")
        interleave = data_block.get("interleave", {})
        probs = interleave.get("probs", []) if hasattr(interleave, "get") else []
        try:
            weights = [f"{float(weight):.6f}" for weight in probs]
        except (TypeError, ValueError, OverflowError) as exc:
            console.print(
                f"[red]apply failed: invalid data.interleave.probs in {for_terminal(apply_recipe)}"
                f" ({for_terminal(exc)})[/red]",
                emoji=False,
            )
            raise typer.Exit(code=2) from exc

        console.print(
            Panel.fit(
                Text.from_markup(
                    f"[bold]Loaded recipe:[/bold] {for_terminal(apply_recipe)}\n"
                    f"[dim]Splice the following block into your soup.yaml[/dim]",
                    emoji=False,
                )
            )
        )
        console.print("data:")
        console.print("  interleave:")
        console.print("    strategy: probs")
        console.print("    probs:")
        for weight in weights:
            console.print(f"      - {weight}")

        if isinstance(train_value, str):
            # Quote the legacy single-string shape as YAML, like the list below.
            scalar = yaml.safe_dump(
                train_value, default_style='"', allow_unicode=False, width=float("inf")
            ).removesuffix("\n")
            console.print(
                f"  train: {scalar}", markup=False, emoji=False, highlight=False, soft_wrap=True
            )
        else:
            # #443 — current recipes: data.train is the full dataset list,
            # index-aligned with data.interleave.probs, now that soup train
            # actually consumes the mixture. (Also matches pre-#330 recipes
            # on disk, which happened to be list-shaped by accident.)
            console.print("  train:")
            for path in train_value or []:
                # Quote every path and escape non-ASCII on the terminal while
                # keeping the original value when the snippet is parsed.
                scalar = yaml.safe_dump(
                    str(path), default_style='"', allow_unicode=False, width=float("inf")
                ).removesuffix("\n")
                console.print(
                    f"    - {scalar}", markup=False, emoji=False, highlight=False, soft_wrap=True
                )
        return

    if not datasets:
        console.print(
            "[red]--datasets is required when --optimize is set.[/red]"
        )
        raise typer.Exit(code=2)
    raw = [p.strip() for p in datasets.split(",") if p.strip()]
    from soup_cli.commands._common import refuse_output_over_input

    inputs = [("--datasets", path) for path in raw]
    if live and base_yaml:
        inputs.append(("--base-yaml", base_yaml))
    refuse_output_over_input(inputs, [("--output", output)], console=console)
    try:
        plan = build_optimization_plan(
            raw, budget=budget, num_probes=num_probes, seed=seed
        )
    except (ValueError, TypeError) as exc:
        console.print(
            f"[red]plan validation failed: {escape(str(exc))}[/red]", emoji=False
        )
        raise typer.Exit(code=2) from exc

    mode = "live training/evaluation" if live else "offline synthetic (no training or evaluation)"
    console.print(
        Panel.fit(
            f"[bold]Mix Optimizer[/bold]  (BETA)\n"
            f"mode: {mode}\n"
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
        from soup_cli.utils.mix_proxy import proxy_run_for_weights, validate_proxy_config

        try:
            validate_proxy_config(base_yaml)
        except (ValueError, TypeError, OSError, yaml.YAMLError) as exc:
            console.print(
                f"[red]live proxy validation failed: {for_terminal(exc)}[/red]", emoji=False
            )
            raise typer.Exit(code=2) from exc

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

    try:
        report = run_mix_optimizer(plan, proxy_callable)
    except (ValueError, TypeError, OSError) as exc:
        console.print(f"[red]optimization failed: {for_terminal(exc)}[/red]", emoji=False)
        raise typer.Exit(code=2) from exc
    if not report.candidates or not math.isfinite(report.best_eval_loss):
        advice = (
            "Check training and evaluation errors. If final-step validation is missing, "
            "remove training.eval_steps to evaluate each epoch."
            if live else "Check the synthetic proxy outputs, then retry."
        )
        console.print(
            f"[red]No valid proxy measurements. No recipe written. {advice}[/red]",
            emoji=False,
        )
        raise typer.Exit(code=2)
    try:
        path = write_mix_recipe(report, output, overwrite=overwrite)
    except (ValueError, OSError) as exc:
        console.print(f"[red]write failed: {escape(str(exc))}[/red]", emoji=False)
        raise typer.Exit(code=2) from exc

    loss_str = f"{report.best_eval_loss:.6f}"
    console.print(
        f"[green]wrote recipe:[/green] {for_terminal(path)}  "
        f"(best_loss={loss_str}, partial={report.partial})",
        emoji=False,
    )
    # Echo the recipe for grep-ability.
    # The recipe is YAML text with the dataset paths in it, not markup.
    console.print(strip_control(render_mix_recipe_yaml(report)), markup=False, emoji=False)
