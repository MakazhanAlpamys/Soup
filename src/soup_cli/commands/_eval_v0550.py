"""v0.55.0 eval subcommands: design / discover / lock / coverage / gate-install.

Lives in its own module so the v0.26.0 eval.py file stays at a sane
length. ``register`` mutates the Typer app passed in.
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from soup_cli.utils.exit_codes import (
    EXIT_GATE_FAILED,
    EXIT_OK,
    EXIT_RUNTIME_ERROR,
    EXIT_USAGE_ERROR,
    GateCommand,
)
from soup_cli.utils.terminal import strip_control, terminal_text

# Paths, the goal, registry and run ids, the metric and error text come from
# outside the program, so every line that carries them is printed as ``Text``:
# shown as typed, never read as markup or ``:name:`` emoji, control characters
# removed (#2489).


def _refuse(console: Console, label: str, exc: object) -> None:
    console.print(Text.assemble((label, "red"), " ", strip_control(exc)))


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
            _refuse(console, "Cannot read dataset:", exc)
            raise typer.Exit(1) from exc

        try:
            design = design_evals_from_data(
                rows, goal=goal, num_dimensions=num_dimensions,
            )
        except (TypeError, ValueError) as exc:
            _refuse(console, "Cannot build design:", exc)
            raise typer.Exit(2) from exc

        try:
            path = write_eval_design(design, output)
        except (ValueError, OSError) as exc:
            _refuse(console, "Cannot write design:", exc)
            raise typer.Exit(1) from exc

        table = Table(title="Eval Design", show_header=True)
        table.add_column("Dimension")
        table.add_column("Scorer")
        table.add_column("Rubric", overflow="fold")
        for dim in design.dimensions:
            table.add_row(
                terminal_text(dim.name),
                terminal_text(dim.scorer_type),
                terminal_text(dim.rubric),
            )
        console.print(table)
        console.print(Text.assemble(
            (f"Wrote {len(design.dimensions)} dimensions", "green"),
            " to ",
            (strip_control(path), "cyan"),
        ))

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
            _refuse(console, "Cannot read dataset:", exc)
            raise typer.Exit(1) from exc

        try:
            canary = discover_canaries(
                rows, base=base, num_clusters=num_clusters,
                per_cluster=per_cluster, seed=seed,
            )
        except (TypeError, ValueError) as exc:
            _refuse(console, "Cannot discover canaries:", exc)
            raise typer.Exit(2) from exc

        try:
            path = write_canary_set(canary, output)
        except (ValueError, OSError) as exc:
            _refuse(console, "Cannot write canary set:", exc)
            raise typer.Exit(1) from exc

        console.print(Text.assemble(
            (
                f"Wrote {len(canary.held_out)} held-out + "
                f"{len(canary.adjacent_skills)} adjacent + "
                f"{len(canary.memorization_probes)} memorization probes",
                "green",
            ),
            " to ",
            (strip_control(path), "cyan"),
        ))

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
            _refuse(console, "Cannot load design:", exc)
            raise typer.Exit(1) from exc

        try:
            locked = lock_suite(design, output)
        except (ValueError, OSError) as exc:
            _refuse(console, "Cannot lock suite:", exc)
            raise typer.Exit(1) from exc

        console.print(Text.assemble(
            (f"Locked {locked.dimension_count} dimensions", "green"),
            " sha256=",
            (f"{locked.checksum[:16]}…", "dim"),
            " → ",
            (strip_control(locked.path), "cyan"),
        ))

        if attach_to_registry:
            try:
                from soup_cli.registry.attach import attach_artifact

                attach_artifact(
                    attach_to_registry,
                    kind="eval_suite",
                    path=locked.path,
                )
                console.print(terminal_text(
                    f"Attached to registry entry {attach_to_registry}", "green",
                ))
            except (ValueError, FileNotFoundError, ImportError) as exc:
                console.print(Text.assemble(
                    ("Registry attach skipped:", "yellow"), " ", strip_control(exc),
                ))

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
            _refuse(console, "Cannot load design:", exc)
            raise typer.Exit(1) from exc

        try:
            report = compute_coverage(design, task_category=task_category)
        except (TypeError, ValueError) as exc:
            _refuse(console, "Cannot compute coverage:", exc)
            raise typer.Exit(2) from exc

        # A Text title is not given the table title style by itself.
        table = Table(
            title=terminal_text(
                f"Coverage — {report.task_category}", "table.title",
            ),
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
            console.print(Text.assemble(("•", "dim"), " ", strip_control(rec)))

    @app.command(name="against", cls=GateCommand)
    def against_cmd(
        baseline_run_id: str = typer.Argument(
            ..., help="Baseline run id (from `soup runs`).",
        ),
        candidate_run_id: str = typer.Option(
            ..., "--candidate",
            help="Candidate run id whose metrics are compared to the baseline.",
        ),
        metric: str = typer.Option(
            "task_accuracy", "--metric",
            help=(
                "Metric to check: task_accuracy | refusal_rate | format_validity | "
                "p95_latency_ms | custom | aider_polyglot | judge:<model> | "
                "benchmark:<lm-eval-task>."
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
                "(exit 3). cwd-contained. The generated pre-push hook passes "
                "$GATE_SUITE here so the locked suite is actually enforced."
            ),
        ),
    ) -> None:
        """Run-vs-run regression check (paired-bootstrap CI).

        Reads per-row metric series from the experiment tracker for both
        runs, runs ``decide_regression`` on the paired delta, and exits
        ``0`` when no regression is detected, ``2`` on regression, and ``3``
        on bad flag, missing suite or missing run data. Designed
        to be invoked from the pre-push hook generated by
        ``soup eval gate-install``.
        """
        import json as _json

        from soup_cli.utils.eval_gate_hook import (
            GateThresholds,
            decide_regression,
            resolve_gate_metric,
        )

        # Validate before constructing ExperimentTracker: an unknown metric is
        # a usage error, not an empty database query (#812).
        try:
            metric_spec = resolve_gate_metric(metric)
        except (TypeError, ValueError) as exc:
            _refuse(console, "Invalid --metric:", exc)
            raise typer.Exit(EXIT_USAGE_ERROR) from exc

        # The locked suite is a hard precondition when supplied: validate it
        # exists, is under cwd, and parses. Previously the hook wrote $GATE_SUITE
        # but never used it, so a deleted / tampered locked suite silently
        # passed the gate.
        if suite:
            from soup_cli.utils.eval_lock_coverage import load_locked_suite
            from soup_cli.utils.paths import is_under_cwd

            if not is_under_cwd(suite):
                _refuse(console, "--suite is outside cwd:", suite)
                raise typer.Exit(EXIT_USAGE_ERROR)
            try:
                load_locked_suite(suite)
            except (FileNotFoundError, ValueError, OSError) as exc:
                _refuse(console, "Locked eval suite invalid — gate blocked:", exc)
                raise typer.Exit(EXIT_USAGE_ERROR) from exc

        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker()
        try:
            candidates = [
                (
                    source_metric,
                    tracker.get_metric_series(baseline_run_id, source_metric),
                    tracker.get_metric_series(candidate_run_id, source_metric),
                )
                for source_metric in metric_spec.source_metrics
            ]
            if metric.startswith("judge:") and metric != metric_spec.name:
                source, baseline, candidate = candidates[0]
                # Older rows may still use the original label. Prefer the safe
                # label independently for each run so old and new runs can pair.
                candidates[0] = (
                    source,
                    baseline or tracker.get_metric_series(baseline_run_id, metric),
                    candidate or tracker.get_metric_series(candidate_run_id, metric),
                )
        except AttributeError:
            # Older trackers (or the lazy import surface) may not expose
            # get_metric_series — print an actionable advisory.
            console.print(
                "[yellow]Per-row metric series not available in this "
                "tracker version — run-vs-run comparison deferred to "
                "v0.55.1. See README for context.[/]"
            )
            raise typer.Exit(EXIT_RUNTIME_ERROR) from None
        except (FileNotFoundError, ValueError, KeyError) as exc:
            _refuse(console, "Cannot fetch metric series:", exc)
            raise typer.Exit(EXIT_USAGE_ERROR) from exc

        # Prefer the first source with a complete pair. If only one run has a
        # source, retain that partial pair so the diagnostic reports the true
        # side-specific counts rather than silently falling through.
        selected = next(
            (candidate for candidate in candidates if candidate[1] and candidate[2]),
            None,
        )
        if selected is None:
            selected = next(
                (candidate for candidate in candidates if candidate[1] or candidate[2]),
                candidates[0],
            )
        source_metric, baseline_series, candidate_series = selected

        if not baseline_series or not candidate_series:
            console.print(terminal_text(
                f"Comparison unavailable — no eval results or tracked metrics "
                f"for {source_metric}: "
                f"baseline {baseline_run_id}={len(baseline_series)}, "
                f"candidate {candidate_run_id}={len(candidate_series)}.",
                "red",
            ))
            raise typer.Exit(EXIT_USAGE_ERROR)

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
            _refuse(console, "Regression check failed:", exc)
            raise typer.Exit(EXIT_USAGE_ERROR) from exc

        sample_count = len(baseline_series)
        ci_degenerate = sample_count < 2
        if json_only:
            # Plain stdout, not console.print: Rich folds a document wider than
            # the console and reads "[...]" in a string as markup (#1468).
            typer.echo(_json.dumps({
                "metric": metric_spec.name,
                "source_metric": source_metric,
                "regressed": verdict.regressed,
                "offenders": list(verdict.offenders),
                "ci_lower": None if ci_degenerate else verdict.ci_lower,
                "ci_upper": None if ci_degenerate else verdict.ci_upper,
                "delta_mean": verdict.delta_mean,
                "sample_count": sample_count,
                "ci_degenerate": ci_degenerate,
                "baseline_run_id": baseline_run_id,
                "candidate_run_id": candidate_run_id,
            }))
        else:
            color = "red" if verdict.regressed else "green"
            tag = "REGRESSED" if verdict.regressed else "OK"
            if ci_degenerate:
                interval = "confidence interval unavailable (one aggregate result)"
            else:
                interval = (
                    f"ci=[{verdict.ci_lower:+.4f}, {verdict.ci_upper:+.4f}]"
                )
            console.print(Text.assemble(
                (tag, color),
                " ",
                strip_control(metric_spec.name),
                f" delta_mean={verdict.delta_mean:+.4f} {interval}",
            ))
        raise typer.Exit(EXIT_GATE_FAILED if verdict.regressed else EXIT_OK)

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
        """Install a pre-push regression gate for the selected baseline."""
        from soup_cli.utils.eval_gate_hook import write_pre_push_hook

        try:
            path = write_pre_push_hook(
                baseline_run_id=baseline_run_id,
                suite_path=suite_path,
                hook_path=hook_path,
                overwrite=force,
            )
        except (TypeError, ValueError, OSError) as exc:
            _refuse(console, "Cannot install hook:", exc)
            raise typer.Exit(1) from exc
        console.print(Text.assemble(
            ("Installed pre-push gate", "green"),
            " → ",
            (strip_control(path), "cyan"),
        ))
        console.print(terminal_text(
            f"Baseline: {baseline_run_id} • suite: {suite_path}", "dim",
        ))
