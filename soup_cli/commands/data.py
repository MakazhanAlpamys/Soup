"""soup data — dataset inspection and tools."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from soup_cli.data.loader import load_raw_data
from soup_cli.data.validator import validate_and_stats

console = Console()

app = typer.Typer(no_args_is_help=True)


@app.command()
def inspect(
    path: str = typer.Argument(..., help="Path to dataset file (jsonl, csv, parquet)"),
    rows: int = typer.Option(5, "--rows", "-r", help="Number of sample rows to show"),
):
    """Inspect a dataset: show stats and sample rows."""
    file_path = Path(path)
    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/]")
        raise typer.Exit(1)

    console.print(f"[dim]Inspecting {file_path}...[/]\n")
    data = load_raw_data(file_path)
    stats = validate_and_stats(data)

    # Print stats
    stats_table = Table(title="Dataset Stats")
    stats_table.add_column("Metric", style="bold")
    stats_table.add_column("Value")
    stats_table.add_row("Total samples", str(stats["total"]))
    stats_table.add_row("Columns", ", ".join(stats["columns"]))
    stats_table.add_row("Avg length (chars)", str(stats["avg_length"]))
    stats_table.add_row("Min length", str(stats["min_length"]))
    stats_table.add_row("Max length", str(stats["max_length"]))
    stats_table.add_row("Empty fields", str(stats["empty_fields"]))
    stats_table.add_row("Duplicates", str(stats["duplicates"]))
    console.print(stats_table)

    # Print sample rows
    if rows > 0 and len(data) > 0:
        console.print(f"\n[bold]Sample rows ({min(rows, len(data))}):[/]")
        sample_table = Table(show_lines=True)
        for col in stats["columns"][:5]:  # max 5 columns
            sample_table.add_column(col, max_width=60)
        for row in data[: min(rows, len(data))]:
            values = [str(row.get(col, ""))[:60] for col in stats["columns"][:5]]
            sample_table.add_row(*values)
        console.print(sample_table)


@app.command()
def validate(
    path: str = typer.Argument(..., help="Path to dataset file"),
    format: str = typer.Option(
        "alpaca", "--format", "-f",
        help="Expected format: alpaca, sharegpt, chatml",
    ),
):
    """Validate dataset format and report issues."""
    file_path = Path(path)
    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/]")
        raise typer.Exit(1)

    data = load_raw_data(file_path)
    stats = validate_and_stats(data, expected_format=format)

    if stats["issues"]:
        console.print("[yellow]Issues found:[/]")
        for issue in stats["issues"]:
            console.print(f"  [yellow]![/] {issue}")
    else:
        console.print("[bold green]Dataset is valid![/]")

    valid = stats["valid_rows"]
    total = stats["total"]
    console.print(f"\n[green]{valid}/{total} rows valid for {format} format[/]")
