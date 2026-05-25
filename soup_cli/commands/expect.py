"""soup expect — Expectations suite for chat data (v0.69.0 Part B).

Runs a YAML suite of expectations against a JSONL dataset. Exit code 3 on
suite failure (matches v0.55 / v0.56 / v0.64 / v0.65 gate convention) so CI
pipelines can gate on dataset quality regressions without parsing output.
"""

from __future__ import annotations

import json
import os
from typing import List, Mapping

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

console = Console()

_MAX_DATA_BYTES = 1_073_741_824  # 1 GiB cap on input data
_MAX_DATA_ROWS = 1_000_000


def _load_jsonl_rows(data_path: str) -> List[Mapping[str, object]]:
    """Cwd-contained + symlink-rejected JSONL loader.

    Delegates to ``utils.paths.enforce_under_cwd_and_no_symlink`` (v0.59.0
    centralised TOCTOU helper) for the path-validation pre-check.
    """
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    if not isinstance(data_path, str) or not data_path:
        raise ValueError("data path must be a non-empty string")
    if not os.path.lexists(data_path):
        raise FileNotFoundError(data_path)
    enforce_under_cwd_and_no_symlink(data_path, "data path")
    real = os.path.realpath(data_path)
    if not os.path.isfile(real):
        raise FileNotFoundError(real)
    if os.path.getsize(real) > _MAX_DATA_BYTES:
        raise ValueError(f"data file exceeds {_MAX_DATA_BYTES} bytes")
    rows: List[Mapping[str, object]] = []
    skipped = 0
    with open(real, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if line_no > _MAX_DATA_ROWS:
                raise ValueError(f"data file exceeds {_MAX_DATA_ROWS} rows")
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
    if skipped:
        console.print(
            f"[yellow]Note: skipped {skipped} malformed JSONL line(s)[/]"
        )
    return rows


def expect_cmd(
    data: str = typer.Argument(..., help="Path to JSONL dataset"),
    suite: str = typer.Argument(..., help="Path to expectations suite YAML"),
) -> None:
    """Run an expectations suite against a JSONL dataset.

    Exit 0 = suite passed. Exit 2 = validation rejection. Exit 3 = suite failed.
    """
    from soup_cli.utils.expectations import load_suite_yaml, run_suite

    try:
        spec = load_suite_yaml(suite)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    try:
        rows = _load_jsonl_rows(data)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    try:
        report = run_suite(rows, spec)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    table = Table(title=f"soup expect — {escape(data)}")
    table.add_column("Expectation")
    table.add_column("Passed")
    table.add_column("Rows")
    table.add_column("Violations")
    for result in report.results:
        verdict = "[green]PASS[/]" if result.passed else "[red]FAIL[/]"
        table.add_row(
            escape(result.name),
            verdict,
            str(result.num_rows_checked),
            str(result.num_violations),
        )
    console.print(table)

    if not report.passed:
        for result in report.results:
            if not result.passed and result.details:
                console.print(
                    Panel(
                        "\n".join(escape(d) for d in result.details),
                        title=f"[red]Violations: {escape(result.name)}[/]",
                    )
                )
        raise typer.Exit(3)
