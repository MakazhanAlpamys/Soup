"""soup data clean — dataset cleaning and sanity repair CLI (Feature 3).

Applies deterministic data cleaning rules:
- Invisible control character and zero-width space sanitization
- Markdown code fence repair (closes unclosed ```)
- Canned AI preamble and disclaimer stripping ("As an AI...", "Certainly!...")
- JSON / tool-call argument repair (trailing commas, markdown wrappers)
- Empty and degenerate turn pruning
- Echo turn pruning
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from soup_cli.data.formats import detect_format
from soup_cli.data.loader import load_raw_data
from soup_cli.utils.data_clean import CleanReport, clean_dataset
from soup_cli.utils.paths import atomic_write_text, is_under_cwd

console = Console()


def _render_summary_table(report: CleanReport) -> None:
    """Render the breakdown of cleaning rules applied."""
    table = Table(title="Dataset Cleaning Breakdown", show_lines=True)
    table.add_column("Rule / Issue", style="bold")
    table.add_column("Affected Rows", justify="right")

    if not report.rule_counts:
        table.add_row("[green]No issues found[/]", "0")
    else:
        for rule_name, count in sorted(report.rule_counts.items(), key=lambda x: -x[1]):
            table.add_row(escape(rule_name), f"[cyan]{count}[/]")

    console.print(table)


def clean(
    path: str = typer.Argument(
        ...,
        help="Path to input dataset file (JSONL)",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to write cleaned dataset (default: <input>_cleaned.jsonl)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview modifications without writing output file",
    ),
    min_tokens: int = typer.Option(
        1,
        "--min-tokens",
        help="Minimum character/token length for assistant turns (shorter turns are dropped)",
    ),
    prune_echo: bool = typer.Option(
        False,
        "--prune-echo/--no-prune-echo",
        help="Prune assistant turns that merely echo the user prompt (opt-in)",
    ),
    strip_boilerplate: bool = typer.Option(
        False,
        "--strip-boilerplate/--no-strip-boilerplate",
        help="Strip canned AI disclaimers ('As an AI...', 'Certainly!...')",
    ),
    repair_code: bool = typer.Option(
        False,
        "--repair-code/--no-repair-code",
        help="Auto-close unclosed markdown ``` code fences in assistant completions",
    ),
    repair_json: bool = typer.Option(
        False,
        "--repair-json/--no-repair-json",
        help="Repair trailing commas and markdown fences in tool-call arguments",
    ),
    drop_invalid_json: bool = typer.Option(
        False,
        "--drop-invalid-json",
        help="Drop tool-calling rows with unrepairable JSON syntax errors",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output machine-readable JSON summary",
    ),
) -> None:
    """Clean and repair a fine-tuning dataset: control characters, whitespace, format sanity."""
    file_path = Path(path)
    if not file_path.exists():
        console.print(f"[red]File not found:[/] {file_path}")
        raise typer.Exit(1)

    if not is_under_cwd(file_path):
        console.print(
            f"[red]Input path must be under the current working directory:[/] {file_path}"
        )
        raise typer.Exit(1)

    try:
        data = load_raw_data(file_path)
    except Exception as exc:
        console.print(f"[red]Failed to load dataset:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    if not data:
        console.print("[red]Dataset is empty.[/]")
        raise typer.Exit(1)

    try:
        fmt = detect_format(data)
    except ValueError:
        fmt = "plaintext"

    if not json_output:
        console.print(
            f"[dim]Cleaning {len(data)} rows in [bold]{fmt}[/] format "
            f"({'DRY RUN' if dry_run else 'LIVE'})...[/]"
        )

    cleaned_data, report = clean_dataset(
        data,
        fmt,
        min_tokens=min_tokens,
        prune_echo=prune_echo,
        strip_ai_boilerplate=strip_boilerplate,
        repair_code=repair_code,
        repair_json=repair_json,
        drop_invalid_json=drop_invalid_json,
    )

    output_path = None
    if not dry_run:
        if output is None:
            output_path = file_path.with_name(f"{file_path.stem}_cleaned{file_path.suffix}")
        else:
            output_path = Path(output)

        if not is_under_cwd(output_path):
            console.print(
                f"[red]Output path must be under the current working directory:[/] {output_path}"
            )
            raise typer.Exit(1)

        if output_path.resolve() == file_path.resolve():
            console.print(
                f"[red]Output path cannot be the same as input path:[/] {output_path} "
                "(soup data clean never modifies its input file in place)"
            )
            raise typer.Exit(1)

        try:
            jsonl_lines = [json.dumps(row, ensure_ascii=False) for row in cleaned_data]
            content = "\n".join(jsonl_lines) + ("\n" if jsonl_lines else "")
            atomic_write_text(content, str(output_path))
        except Exception as exc:
            console.print(f"[red]Failed to write output file:[/] {escape(str(exc))}")
            raise typer.Exit(1) from exc

    if json_output:
        summary_payload = {
            "total_scanned": report.total_scanned,
            "total_clean": report.total_clean,
            "total_modified": report.total_modified,
            "total_dropped": report.total_dropped,
            "output_rows": len(cleaned_data),
            "rule_counts": report.rule_counts,
            "dry_run": dry_run,
        }
        console.print(json.dumps(summary_payload, indent=2))
        return

    # Render Rich header panel
    header_text = (
        f"Scanned: [bold]{report.total_scanned}[/] rows | "
        f"Modified: [yellow]{report.total_modified}[/] | "
        f"Dropped: [red]{report.total_dropped}[/] | "
        f"Clean: [green]{len(cleaned_data)}[/]"
    )
    console.print(Panel.fit(header_text, title="soup data clean"))

    _render_summary_table(report)

    if dry_run:
        console.print("\n[yellow]Dry-run mode:[/] No files were written.")
        return

    if output_path is not None:
        console.print(
            f"\n[bold green]✓ Cleaned dataset saved to:[/] {output_path} "
            f"({len(cleaned_data)} rows)"
        )
