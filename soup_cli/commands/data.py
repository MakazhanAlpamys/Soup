"""soup data — dataset inspection and tools."""

from __future__ import annotations

import json
import random
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
    result = validate_and_stats(data)

    # Print stats
    stats_table = Table(title="Dataset Stats")
    stats_table.add_column("Metric", style="bold")
    stats_table.add_column("Value")
    stats_table.add_row("Total samples", str(result["total"]))
    stats_table.add_row("Columns", ", ".join(result["columns"]))
    stats_table.add_row("Avg length (chars)", str(result["avg_length"]))
    stats_table.add_row("Min length", str(result["min_length"]))
    stats_table.add_row("Max length", str(result["max_length"]))
    stats_table.add_row("Empty fields", str(result["empty_fields"]))
    stats_table.add_row("Duplicates", str(result["duplicates"]))
    console.print(stats_table)

    # Print sample rows
    if rows > 0 and len(data) > 0:
        console.print(f"\n[bold]Sample rows ({min(rows, len(data))}):[/]")
        sample_table = Table(show_lines=True)
        for col in result["columns"][:5]:  # max 5 columns
            sample_table.add_column(col, max_width=60)
        for row in data[: min(rows, len(data))]:
            values = [str(row.get(col, ""))[:60] for col in result["columns"][:5]]
            sample_table.add_row(*values)
        console.print(sample_table)


@app.command()
def validate(
    path: str = typer.Argument(..., help="Path to dataset file"),
    fmt: str = typer.Option(
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
    result = validate_and_stats(data, expected_format=fmt)

    if result["issues"]:
        console.print("[yellow]Issues found:[/]")
        for issue in result["issues"]:
            console.print(f"  [yellow]![/] {issue}")
    else:
        console.print("[bold green]Dataset is valid![/]")

    valid = result["valid_rows"]
    total = result["total"]
    console.print(f"\n[green]{valid}/{total} rows valid for {fmt} format[/]")


@app.command()
def convert(
    path: str = typer.Argument(..., help="Input dataset file"),
    to: str = typer.Option(
        ..., "--to", "-t",
        help="Target format: alpaca, sharegpt, chatml",
    ),
    output: str = typer.Option(
        None, "--output", "-o",
        help="Output file path (default: <input>_<format>.jsonl)",
    ),
):
    """Convert a dataset between formats (alpaca, sharegpt, chatml)."""
    from soup_cli.data.formats import (
        CONVERTIBLE_FORMATS,
        detect_format,
        format_to_messages,
        messages_to_format,
    )

    file_path = Path(path)
    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/]")
        raise typer.Exit(1)

    if to not in CONVERTIBLE_FORMATS:
        console.print(
            f"[red]Invalid target format: {to}[/]\n"
            f"Supported: {', '.join(CONVERTIBLE_FORMATS)}"
        )
        raise typer.Exit(1)

    data = load_raw_data(file_path)
    if not data:
        console.print("[red]Dataset is empty.[/]")
        raise typer.Exit(1)

    src_fmt = detect_format(data)
    console.print(f"[dim]Detected source format: {src_fmt}[/]")

    if src_fmt == to:
        console.print(f"[yellow]Source and target format are both '{to}'. Nothing to convert.[/]")
        raise typer.Exit()

    if src_fmt == "dpo":
        console.print("[red]Cannot convert DPO format (preference pairs are not conversations).[/]")
        raise typer.Exit(1)

    # Convert: source → messages → target
    converted = []
    failed = 0
    for row in data:
        messages = format_to_messages(row, src_fmt)
        if messages is None:
            failed += 1
            continue
        result = messages_to_format(messages, to)
        if result is None:
            failed += 1
            continue
        converted.append(result)

    if not converted:
        console.print("[red]All rows failed to convert.[/]")
        raise typer.Exit(1)

    # Determine output path
    if output is None:
        output = str(file_path.stem) + f"_{to}.jsonl"
    out_path = Path(output)

    _write_jsonl(out_path, converted)

    console.print(
        f"[green]Converted {len(converted)} rows:[/] {src_fmt} → {to}\n"
        f"Output: [bold]{out_path}[/]"
    )
    if failed > 0:
        console.print(f"[yellow]{failed} rows failed to convert.[/]")


@app.command()
def merge(
    files: list[str] = typer.Argument(..., help="Paths to dataset files to merge"),
    output: str = typer.Option(
        "merged.jsonl", "--output", "-o",
        help="Output file path",
    ),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle after merging"),
):
    """Merge multiple datasets into a single file."""
    all_data: list[dict] = []

    for file_str in files:
        file_path = Path(file_str)
        if not file_path.exists():
            console.print(f"[red]File not found: {file_path}[/]")
            raise typer.Exit(1)
        data = load_raw_data(file_path)
        console.print(f"[dim]Loaded {len(data)} rows from {file_path}[/]")
        all_data.extend(data)

    if not all_data:
        console.print("[red]No data loaded from any file.[/]")
        raise typer.Exit(1)

    if shuffle:
        random.shuffle(all_data)

    out_path = Path(output)
    _write_jsonl(out_path, all_data)

    console.print(
        f"[green]Merged {len(all_data)} rows from {len(files)} files.[/]\n"
        f"Output: [bold]{out_path}[/]"
    )


@app.command()
def dedup(
    path: str = typer.Argument(..., help="Path to dataset file"),
    output: str = typer.Option(
        None, "--output", "-o",
        help="Output file path (default: <input>_deduped.jsonl)",
    ),
    threshold: float = typer.Option(
        0.8, "--threshold",
        help="MinHash similarity threshold (0.0-1.0)",
    ),
    field: str = typer.Option(
        None, "--field", "-f",
        help="Field to hash (default: all text fields concatenated)",
    ),
):
    """Remove near-duplicate rows using MinHash (locality-sensitive hashing)."""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        console.print(
            "[red]datasketch not installed.[/]\n"
            "Install with: [bold]pip install 'soup-cli[data]'[/]"
        )
        raise typer.Exit(1)

    file_path = Path(path)
    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/]")
        raise typer.Exit(1)

    data = load_raw_data(file_path)
    if not data:
        console.print("[red]Dataset is empty.[/]")
        raise typer.Exit(1)

    console.print(f"[dim]Deduplicating {len(data)} rows (threshold={threshold})...[/]")

    # Build MinHash for each row
    num_perm = 128
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = []

    for idx, row in enumerate(data):
        if field:
            text = str(row.get(field, ""))
        else:
            text = " ".join(str(v) for v in row.values() if v)

        words = text.lower().split()
        shingles = set()
        for i in range(max(1, len(words) - 2)):
            shingles.add(" ".join(words[i: i + 3]))

        mhash = MinHash(num_perm=num_perm)
        for shingle in shingles:
            mhash.update(shingle.encode("utf-8"))

        minhashes.append(mhash)

        try:
            lsh.insert(str(idx), mhash)
        except ValueError:
            pass  # duplicate key, already inserted by LSH

    # Collect unique indices
    seen: set[int] = set()
    unique_indices = []
    for idx in range(len(data)):
        if idx in seen:
            continue
        unique_indices.append(idx)
        results = lsh.query(minhashes[idx])
        for dup_idx_str in results:
            seen.add(int(dup_idx_str))

    unique_data = [data[idx] for idx in unique_indices]
    removed = len(data) - len(unique_data)

    # Write output
    if output is None:
        output = str(file_path.stem) + "_deduped.jsonl"
    out_path = Path(output)

    _write_jsonl(out_path, unique_data)

    console.print(
        f"[green]Dedup complete:[/] {len(data)} → {len(unique_data)} rows "
        f"([red]-{removed}[/] duplicates)\n"
        f"Output: [bold]{out_path}[/]"
    )


@app.command()
def stats(
    path: str = typer.Argument(..., help="Path to dataset file"),
):
    """Extended dataset statistics: length distribution, token counts, languages."""
    from soup_cli.data.validator import extended_stats

    file_path = Path(path)
    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/]")
        raise typer.Exit(1)

    data = load_raw_data(file_path)
    if not data:
        console.print("[red]Dataset is empty.[/]")
        raise typer.Exit(1)

    ext_stats = extended_stats(data)

    # Basic info table
    info_table = Table(title=f"Extended Stats: {file_path.name}")
    info_table.add_column("Metric", style="bold")
    info_table.add_column("Value", justify="right")
    info_table.add_row("Total samples", str(ext_stats["total"]))
    info_table.add_row("", "")
    info_table.add_row("[bold]Length (chars)[/]", "")
    info_table.add_row("  p10", str(ext_stats["length_p10"]))
    info_table.add_row("  p25", str(ext_stats["length_p25"]))
    info_table.add_row("  p50 (median)", str(ext_stats["length_p50"]))
    info_table.add_row("  p75", str(ext_stats["length_p75"]))
    info_table.add_row("  p90", str(ext_stats["length_p90"]))
    info_table.add_row("", "")
    info_table.add_row("[bold]Tokens (approx)[/]", "")
    info_table.add_row("  Average", str(ext_stats["avg_tokens"]))
    info_table.add_row("  Min", str(ext_stats["min_tokens"]))
    info_table.add_row("  Max", str(ext_stats["max_tokens"]))

    if ext_stats["languages"]:
        info_table.add_row("", "")
        info_table.add_row("[bold]Languages (sample)[/]", "")
        for lang, count in sorted(
            ext_stats["languages"].items(), key=lambda x: -x[1]
        ):
            info_table.add_row(f"  {lang}", str(count))

    console.print(info_table)

    # Terminal histogram of lengths
    try:
        import plotext as plt

        lengths = ext_stats["lengths"]
        if lengths:
            plt.clear_figure()
            plt.hist(lengths, bins=30)
            plt.title("Text Length Distribution (chars)")
            plt.xlabel("Length")
            plt.ylabel("Count")
            plt.theme("dark")
            plt.show()
    except ImportError:
        console.print(
            "\n[dim]Install plotext for histograms:[/] [bold]pip install plotext[/]"
        )


def _write_jsonl(path: Path, data: list[dict]) -> None:
    """Write a list of dicts as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
