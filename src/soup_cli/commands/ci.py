"""soup ci — fine-tuning CI helpers (v0.71.35).

``soup ci init`` writes a GitHub Actions workflow that gates every PR on
``soup data validate`` -> ``soup expect`` -> ``soup ship --evidence``.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from soup_cli.utils.ci_workflow import write_soup_gate_workflow

console = Console()

app = typer.Typer(no_args_is_help=True, help="Fine-tuning CI helpers.")


@app.command("init")
def init(
    data: str = typer.Option(
        "./data/train.jsonl", "--data", help="Training data path (repo-relative)"
    ),
    suite: str = typer.Option(
        "expectations.yaml", "--suite", help="Expectations suite YAML (repo-relative)"
    ),
    evidence: str = typer.Option(
        "ship_evidence.json", "--evidence", help="Ship evidence JSON (repo-relative)"
    ),
    python_version: str = typer.Option(
        "3.11", "--python", help="Python version for the CI runner (e.g. 3.11)"
    ),
    branch: str = typer.Option("main", "--branch", help="Branch the workflow triggers on"),
    output: str = typer.Option(
        ".github/workflows/soup-gate.yml", "-o", "--output", help="Workflow output path"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite an existing workflow"),
) -> None:
    """Write a GitHub Actions fine-tuning gate workflow."""
    try:
        written = write_soup_gate_workflow(
            data_path=data,
            suite_path=suite,
            evidence_path=evidence,
            python_version=python_version,
            branch=branch,
            output_path=output,
            overwrite=force,
        )
    except (ValueError, TypeError, OSError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc

    console.print(
        Panel(
            f"Wrote [bold]{escape(written)}[/]\n\n"
            "The gate runs: [cyan]soup data validate[/] -> [cyan]soup expect[/] -> "
            "[cyan]soup ship --evidence[/].\n"
            "Edit the paths in the workflow to match your repo.",
            title="soup ci init",
            border_style="green",
        )
    )
