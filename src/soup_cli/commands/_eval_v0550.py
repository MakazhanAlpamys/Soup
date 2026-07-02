"""v0.55.0 eval subcommands: design / discover / lock / coverage / gate-install.

Lives in its own module so the v0.26.0 eval.py file stays at a sane
length. ``register`` mutates the Typer app passed in.
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table


def register(app: typer.Typer, console: Console) -> None:
    """Attach v0.55.0 subcommands to ``app``."""

    @app.command(name="design")
    def design_cmd(
        data: str = typer.Argument(..., help="Training-data JSONL path"),
        goal: str = typer.Option(
            ..., "--goal", "-g",
            help="One-line goal description (e.g. 'better at SQL').",
        ),
        num_dimensions: int = typer.Option(
            5, "--num-dimensions", "-n",
            help="Number of eval dimensions to draft (1-20).",
        ),
        output: str = typer.Option(
            "evals/design.json", "--output", "-o",
            help="Where to write the rendered EvalDesign JSON.",
        ),
    ) -> None:
        """Draft an evaluation suite from training data + a one-line goal."""
        from soup_cli.utils.advise import load_advise_dataset
        from soup_cli.utils.eval_design import (
            design_evals_from_data,
            write_eval_design,
        )

        try:
            rows = load_advise_dataset(data)
        except (FileNotFoundError, ValueError, TypeError) as exc:
            console.print(f"[red]Cannot read dataset:[/] {escape(str(exc))}")
            raise typer.Exit(1) from exc

        try:
            design = design_evals_from_data(
                rows, goal=goal, num_dimensions=num_dimensions,
            )
        except (TypeError, ValueError) as exc:
            console.print(f"[red]Cannot build design:[/] {exc}")
            raise typer.Exit(2) from exc

        try:
            path = write_eval_design(design, output)
        except (ValueError, OSError) as exc:
            console.print(f"[red]Cannot write design:[/] {exc}")
            raise typer.Exit(1) from exc

        table = Table(title="Eval Design", show_header=True)
        table.add_column("Dimension")
        table.add_column("Scorer")
        table.add_column("Rubric", overflow="fold")
        for dim in design.dimensions:
            table.add_row(
                escape(dim.name), escape(dim.scorer_type), escape(dim.rubric),
            )
        console.print(table)
        console.print(
            f"[green]Wrote {len(design.dimensions)} dimensions[/] to "
            f"[cyan]{escape(path)}[/]"
        )

    @app.command(name="discover")
    def discover_cmd(
        data: str = typer.Argument(..., help="Training-data JSONL path"),
        base: Optional[str] = typer.Option(
            None, "--base",
            help="Base model id (recorded; consumed by `soup diagnose`).",
        ),
        num_clusters: int = typer.Option(
            5, "--num-clusters",
            help="Number of behavioural clusters to discover (1-64).",
        ),
        per_cluster: int = typer.Option(
            3, "--per-cluster",
            help="Held-out canaries to draw per cluster (1-64).",
        ),
        seed: int = typer.Option(0, "--seed", help="Deterministic seed."),
        output: str = typer.Option(
            "evals/canaries.json", "--output", "-o",
            help="Where to write the rendered CanarySet JSON.",
        ),
    ) -> None:
        """Discover a held-out canary set for regression detection."""
        from soup_cli.utils.advise import load_advise_dataset
        from soup_cli.utils.canary_discovery import (
            discover_canaries,
            write_canary_set,
        )

        try:
            rows = load_advise_dataset(data)
        except (FileNotFoundError, ValueError, TypeError) as exc:
            console.print(f"[red]Cannot read dataset:[/] {escape(str(exc))}")
            raise typer.Exit(1) from exc

        try:
            canary = discover_canaries(
                rows, base=base, num_clusters=num_clusters,
                per_cluster=per_cluster, seed=seed,
            )
        except (TypeError, ValueError) as exc:
            console.print(f"[red]Cannot discover canaries:[/] {exc}")
            raise typer.Exit(2) from exc

        try:
            path = write_canary_set(canary, output)
        except (ValueError, OSError) as exc:
            console.print(f"[red]Cannot write canary set:[/] {exc}")
            raise typer.Exit(1) from exc

        console.print(
            f"[green]Wrote {len(canary.held_out)} held-out + "
            f"{len(canary.adjacent_skills)} adjacent + "
            f"{len(canary.memorization_probes)} memorization probes[/] to "
            f"[cyan]{escape(path)}[/]"
        )

    @app.command(name="lock")
    def lock_cmd(
        design_path: str = typer.Argument(
            ..., help="Path to an EvalDesign JSON (from `soup eval design`)",
        ),
        output: str = typer.Option(
            "evals/locked.json", "--output", "-o",
            help="Where to write the canonicalised locked suite.",
        ),
        attach_to_registry: Optional[str] = typer.Option(
            None, "--attach-to-registry",
            help="Registry entry id or name to attach the locked suite to.",
        ),
    ) -> None:
        """Freeze an EvalDesign as a checksummed eval_suite artifact."""
        from soup_cli.utils.eval_design import load_eval_design
        from soup_cli.utils.eval_lock_coverage import lock_suite

        try:
            design = load_eval_design(design_path)
        except (FileNotFoundError, ValueError, TypeError) as exc:
            console.print(f"[red]Cannot load design:[/] {exc}")
            raise typer.Exit(1) from exc

        try:
            locked = lock_suite(design, output)
        except (ValueError, OSError) as exc:
            console.print(f"[red]Cannot lock suite:[/] {exc}")
            raise typer.Exit(1) from exc

        console.print(
            f"[green]Locked {locked.dimension_count} dimensions[/] "
            f"sha256=[dim]{locked.checksum[:16]}…[/] → "
            f"[cyan]{escape(locked.path)}[/]"
        )

        if attach_to_registry:
            try:
                from soup_cli.registry.attach import attach_artifact

                attach_artifact(
                    attach_to_registry,
                    kind="eval_suite",
                    path=locked.path,
                )
                console.print(
                    f"[green]Attached to registry entry "
                    f"{escape(attach_to_registry)}[/]"
                )
            except (ValueError, FileNotFoundError, ImportError) as exc:
                console.print(f"[yellow]Registry attach skipped:[/] {exc}")

    @app.command(name="coverage")
    def coverage_cmd(
        design_path: str = typer.Argument(
            ..., help="Path to an EvalDesign JSON",
        ),
        task_category: str = typer.Option(
            ..., "--task",
            help=(
                "Task category from v0.54.0 taxonomy: factual_lookup | "
                "style_shaping | format_conversion | reasoning | tool_use | "
                "summarization | classification."
            ),
        ),
    ) -> None:
        """Heuristic coverage / gap analysis for a locked or drafted suite."""
        from soup_cli.utils.eval_design import load_eval_design
        from soup_cli.utils.eval_lock_coverage import compute_coverage

        try:
            design = load_eval_design(design_path)
        except (FileNotFoundError, ValueError, TypeError) as exc:
            console.print(f"[red]Cannot load design:[/] {exc}")
            raise typer.Exit(1) from exc

        try:
            report = compute_coverage(design, task_category=task_category)
        except (TypeError, ValueError) as exc:
            console.print(f"[red]Cannot compute coverage:[/] {exc}")
            raise typer.Exit(2) from exc

        table = Table(
            title=f"Coverage — {escape(report.task_category)}",
            show_header=True,
        )
        table.add_column("Scorer")
        table.add_column("Dimensions", justify="right")
        for scorer, count in sorted(report.scorer_mix.items()):
            table.add_row(scorer, str(count))
        console.print(table)
        if report.missing_scorers:
            console.print(
                "[yellow]Missing scorers:[/] "
                + ", ".join(report.missing_scorers)
            )
        for rec in report.recommendations:
            console.print(f"[dim]•[/] {escape(rec)}")

    @app.command(name="against")
    def against_cmd(
        baseline_run_id: str = typer.Argument(
            ..., help="Baseline run id (from `soup runs list`).",
        ),
        candidate_run_id: str = typer.Option(
            ..., "--candidate",
            help="Candidate run id whose metrics are compared to the baseline.",
        ),
        metric: str = typer.Option(
            "task_accuracy", "--metric",
            help=(
                "Metric to check: task_accuracy | refusal_rate | "
                "format_validity | p95_latency_ms."
            ),
        ),
        n_samples: int = typer.Option(
            1000, "--n-samples",
            help="Paired-bootstrap samples (100-100000).",
        ),
        seed: int = typer.Option(0, "--seed", help="Deterministic seed."),
        json_only: bool = typer.Option(
            False, "--json-only",
            help="Suppress Rich output; emit a single JSON verdict line.",
        ),
        suite: str = typer.Option(
            None, "--suite",
            help=(
                "Locked eval suite (from `soup eval lock`) to validate as a gate "
                "precondition — a missing / unparseable suite BLOCKS the check "
                "(exit 1). cwd-contained. The generated pre-push hook passes "
                "$GATE_SUITE here so the locked suite is actually enforced."
            ),
        ),
    ) -> None:
        """Run-vs-run regression check (paired-bootstrap CI).

        Reads per-row metric series from the experiment tracker for both
        runs, runs ``decide_regression`` on the paired delta, and exits
        ``0`` when no regression is detected, ``1`` otherwise. Designed
        to be invoked from the pre-push hook generated by
        ``soup eval gate-install``.
        """
        import json as _json

        # The locked suite is a hard precondition when supplied: validate it
        # exists, is under cwd, and parses. Previously the hook wrote $GATE_SUITE
        # but never used it, so a deleted / tampered locked suite silently
        # passed the gate.
        if suite:
            from soup_cli.utils.eval_lock_coverage import load_locked_suite
            from soup_cli.utils.paths import is_under_cwd

            if not is_under_cwd(suite):
                console.print(f"[red]--suite is outside cwd:[/] {escape(str(suite))}")
                raise typer.Exit(1)
            try:
                load_locked_suite(suite)
            except (FileNotFoundError, ValueError, OSError) as exc:
                console.print(
                    f"[red]Locked eval suite invalid — gate blocked:[/] "
                    f"{escape(str(exc))}"
                )
                raise typer.Exit(1) from exc

        from soup_cli.experiment.tracker import ExperimentTracker
        from soup_cli.utils.eval_gate_hook import (
            GateThresholds,
            decide_regression,
        )

        tracker = ExperimentTracker()
        try:
            baseline_series = tracker.get_metric_series(
                baseline_run_id, metric,
            )
            candidate_series = tracker.get_metric_series(
                candidate_run_id, metric,
            )
        except AttributeError:
            # Older trackers (or the lazy import surface) may not expose
            # get_metric_series — print an actionable advisory.
            console.print(
                "[yellow]Per-row metric series not available in this "
                "tracker version — run-vs-run comparison deferred to "
                "v0.55.1. See README for context.[/]"
            )
            raise typer.Exit(2) from None
        except (FileNotFoundError, ValueError, KeyError) as exc:
            console.print(f"[red]Cannot fetch metric series:[/] {exc}")
            raise typer.Exit(1) from exc

        if not baseline_series or not candidate_series:
            console.print(
                f"[red]Empty series — baseline={len(baseline_series)} "
                f"candidate={len(candidate_series)}.[/]"
            )
            raise typer.Exit(1)

        try:
            verdict = decide_regression(
                metric=metric,
                baseline=baseline_series,
                candidate=candidate_series,
                thresholds=GateThresholds(),
                n_samples=n_samples,
                seed=seed,
            )
        except (TypeError, ValueError) as exc:
            console.print(f"[red]Regression check failed:[/] {exc}")
            raise typer.Exit(1) from exc

        if json_only:
            console.print(_json.dumps({
                "metric": metric,
                "regressed": verdict.regressed,
                "offenders": list(verdict.offenders),
                "ci_lower": verdict.ci_lower,
                "ci_upper": verdict.ci_upper,
                "delta_mean": verdict.delta_mean,
                "baseline_run_id": baseline_run_id,
                "candidate_run_id": candidate_run_id,
            }))
        else:
            color = "red" if verdict.regressed else "green"
            tag = "REGRESSED" if verdict.regressed else "OK"
            console.print(
                f"[{color}]{tag}[/] {escape(metric)} delta_mean="
                f"{verdict.delta_mean:+.4f} "
                f"ci=[{verdict.ci_lower:+.4f}, {verdict.ci_upper:+.4f}]"
            )
        raise typer.Exit(1 if verdict.regressed else 0)

    @app.command(name="gate-install")
    def gate_install_cmd(
        baseline_run_id: str = typer.Option(
            ..., "--baseline",
            help="Baseline run id the pre-push hook compares against.",
        ),
        suite_path: str = typer.Option(
            "evals/locked.json", "--suite",
            help="Path to the locked eval suite (cwd-contained).",
        ),
        hook_path: str = typer.Option(
            ".git/hooks/pre-push", "--hook-path",
            help="Hook target — usually .git/hooks/pre-push.",
        ),
        force: bool = typer.Option(
            False, "--force", help="Overwrite an existing hook.",
        ),
    ) -> None:
        """Install a pre-push regression gate (v0.55.0 Part D)."""
        from soup_cli.utils.eval_gate_hook import write_pre_push_hook

        try:
            path = write_pre_push_hook(
                baseline_run_id=baseline_run_id,
                suite_path=suite_path,
                hook_path=hook_path,
                overwrite=force,
            )
        except (TypeError, ValueError, OSError) as exc:
            console.print(f"[red]Cannot install hook:[/] {exc}")
            raise typer.Exit(1) from exc
        console.print(
            f"[green]Installed pre-push gate[/] → [cyan]{escape(path)}[/]"
        )
        console.print(
            f"[dim]Baseline: {escape(baseline_run_id)} • suite: "
            f"{escape(suite_path)}[/]"
        )
