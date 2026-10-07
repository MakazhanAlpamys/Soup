"""v0.61.0 Part B — `soup eval unlearning` subcommand.

Computes Forget Quality + Model Utility + PrivLeak metrics from
operator-supplied evidence JSON and emits an :class:`UnlearnReport`.
Live model-driven evaluation lands in v0.61.1; this release writes a
neutral OK report when no evidence is supplied so the schema + output
shape are usable today.
"""

from __future__ import annotations

import json
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from soup_cli.utils.terminal import for_terminal


def register(app: typer.Typer, console: Console) -> None:
    """Attach v0.61.0 subcommands to ``app``."""

    @app.command(name="unlearning")
    def unlearning_cmd(
        run_id: str = typer.Argument(
            ..., help="Run identifier (e.g. registry id of the unlearn run).",
        ),
        benchmark: str = typer.Option(
            "tofu", "--benchmark", "-b",
            help="Benchmark: tofu / muse / wmdp.",
        ),
        evidence: Optional[str] = typer.Option(
            None, "--evidence", "-e",
            help=(
                "Path to a JSON file with pre-computed evidence. Schema: "
                "{forget_quality: {pre_loss, post_loss}, "
                "model_utility: {pre_acc, post_acc}, "
                "priv_leak: {mia_auc}}. Missing keys fall through to a "
                "neutral OK score."
            ),
        ),
        output: Optional[str] = typer.Option(
            None, "--output", "-o",
            help="Where to write the rendered UnlearnReport JSON.",
        ),
        attach_to_registry: Optional[str] = typer.Option(
            None, "--attach-to-registry",
            help=(
                "Registry entry id to attach the --output report to as an "
                "eval_results artifact (needs --output). Exits 1 when the "
                "attach fails."
            ),
        ),
    ) -> None:
        """Score an unlearn run on TOFU / MUSE / WMDP."""
        from soup_cli.utils.unlearning_eval import (
            load_evidence_file,
            run_unlearn_eval,
            validate_benchmark_name,
            write_unlearn_report,
        )

        try:
            bench = validate_benchmark_name(benchmark)
        except (TypeError, ValueError) as exc:
            console.print(f"[red]Invalid benchmark:[/] {for_terminal(exc)}")
            raise typer.Exit(2) from exc

        # The attach links the --output file, so without one nothing can be
        # attached: refuse before the eval runs.
        if attach_to_registry is not None and output is None:
            console.print(
                "[red]--attach-to-registry requires --output[/] "
                "(nothing written to attach)."
            )
            raise typer.Exit(2)

        evidence_data = None
        if evidence is not None:
            try:
                evidence_data = load_evidence_file(evidence)
            except json.JSONDecodeError as exc:
                # json.JSONDecodeError is a ValueError subclass — catch it
                # FIRST so we can label the message specifically (review
                # MEDIUM M10 — was unreachable after the broader except).
                console.print(
                    f"[red]Evidence file is not valid JSON:[/] {for_terminal(exc)}"
                )
                raise typer.Exit(2) from exc
            except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
                console.print(f"[red]Cannot read evidence:[/] {for_terminal(exc)}")
                raise typer.Exit(2) from exc

        try:
            report = run_unlearn_eval(
                run_id=run_id,
                benchmark=bench,
                evidence=evidence_data,
            )
        except (TypeError, ValueError) as exc:
            console.print(f"[red]Eval failed:[/] {for_terminal(exc)}")
            raise typer.Exit(2) from exc

        # Render the table.
        table = Table(title=f"Unlearn eval — {escape(bench)} — {for_terminal(run_id)}")
        table.add_column("metric", style="bold")
        table.add_column("score", justify="right")
        table.add_column("verdict")
        table.add_column("evidence")
        for m in report.metrics:
            colour = {
                "OK": "green",
                "MINOR": "yellow",
                "MAJOR": "red",
            }.get(m.verdict, "white")
            table.add_row(
                escape(m.name),
                f"{m.score:.3f}",
                f"[{colour}]{escape(m.verdict)}[/]",
                for_terminal(m.evidence),
            )
        console.print(table)
        overall_colour = {
            "OK": "green",
            "MINOR": "yellow",
            "MAJOR": "red",
        }.get(report.overall, "white")
        console.print(
            f"Overall: [{overall_colour}]{escape(report.overall)}[/]"
        )

        # Write the report.
        if output is not None:
            try:
                write_unlearn_report(report, output)
            except (TypeError, ValueError, OSError) as exc:
                console.print(f"[red]Cannot write report:[/] {for_terminal(exc)}")
                raise typer.Exit(2) from exc
            console.print(f"Wrote report -> {for_terminal(output)}")

        # Optional registry attach. Once requested it is part of the command's
        # success: a failure exits 1, with the report already on disk.
        if attach_to_registry is not None:
            assert output is not None  # narrowed by the early arg-check
            try:
                from soup_cli.registry.attach import attach_artifact

                attach_artifact(
                    entry_id=attach_to_registry,
                    path=output,
                    kind="eval_results",
                )
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]Registry attach failed:[/] {for_terminal(exc)}")
                raise typer.Exit(1) from exc
            console.print(
                f"Attached eval_results -> registry {for_terminal(attach_to_registry)}"
            )

        # Exit code: 0 on OK / MINOR; 2 on MAJOR (matches v0.56.0 diagnose
        # gate convention so CI scripts can chain `soup train` → `soup
        # eval unlearning` → exit-on-MAJOR).
        if report.overall == "MAJOR":
            raise typer.Exit(2)
