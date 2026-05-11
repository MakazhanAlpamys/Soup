"""soup data score / decontaminate / toxicity / langdetect / pii / educational
(v0.47.0 Part B).

Each subcommand reads JSONL, runs one of the pure-function quality
helpers from ``soup_cli/utils/data_score.py``, and writes an enriched
JSONL with per-row scores or a flagged subset.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

console = Console()


def _exit(msg: str, code: int = 1) -> None:
    console.print(f"[red]{escape(msg)}[/]")
    raise typer.Exit(code)


def _read_rows(path: str) -> List[Mapping[str, Any]]:
    from soup_cli.utils.data_score import load_jsonl_rows

    try:
        return load_jsonl_rows(path)
    except (TypeError, ValueError, FileNotFoundError) as exc:
        _exit(str(exc))
        raise  # unreachable; satisfies type checker


def _write_rows(rows: Iterable[Mapping[str, Any]], path: str) -> str:
    from soup_cli.utils.data_score import write_jsonl_rows

    try:
        return write_jsonl_rows(rows, path)
    except (TypeError, ValueError) as exc:
        _exit(str(exc))
        return ""  # unreachable


def score(
    input: str = typer.Option(..., "--input", "-i", help="JSONL input under cwd."),
    benchmarks: Optional[str] = typer.Option(
        None, "--benchmarks", "-b",
        help="Comma-separated benchmark names (e.g. mmlu,gsm8k).",
    ),
    threshold: float = typer.Option(
        0.8, "--threshold",
        min=0.0, max=1.0,
        help="Decontamination overlap threshold.",
    ),
):
    """Composite data-quality scorecard (PII + toxicity + lang + edu + dec)."""
    from soup_cli.utils.data_score import BENCHMARKS, compute_scorecard

    rows = _read_rows(input)
    bench_list: List[str] = []
    if benchmarks:
        for name in benchmarks.split(","):
            name = name.strip().lower()
            if not name:
                continue
            if name not in BENCHMARKS:
                _exit(f"unknown benchmark: {name!r}; choose from {sorted(BENCHMARKS)}")
            bench_list.append(name)

    try:
        report = compute_scorecard(
            rows,
            benchmarks=bench_list,
            decontaminate_texts=None,
            decontaminate_threshold=threshold,
        )
    except (TypeError, ValueError) as exc:
        _exit(str(exc))

    table = Table(title="Data Scorecard")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Total rows", str(report.total))
    table.add_row("PII flagged", str(report.pii_flagged))
    table.add_row("Toxic flagged", str(report.toxic_flagged))
    table.add_row("Edu mean", f"{report.educational_mean:.3f}")
    table.add_row("Decontaminated", str(report.decontaminated_removed))
    for lang, count in sorted(report.languages.items(), key=lambda kv: -kv[1]):
        table.add_row(f"lang:{escape(lang)}", str(count))
    console.print(table)


def decontaminate(
    input: str = typer.Option(..., "--input", "-i"),
    benchmarks: str = typer.Option(
        ..., "--benchmarks", "-b",
        help="Comma-separated benchmark names.",
    ),
    output: str = typer.Option(
        "clean.jsonl", "--output", "-o",
        help="Output JSONL of rows that survived decontamination.",
    ),
    threshold: float = typer.Option(0.8, "--threshold", min=0.0, max=1.0),
    n: int = typer.Option(8, "--n", min=1, max=32),
):
    """Drop rows that overlap public benchmarks (n-gram heuristic)."""
    from soup_cli.utils.data_score import BENCHMARKS, decontaminate_rows

    rows = _read_rows(input)
    bench_list: List[str] = []
    for name in benchmarks.split(","):
        name = name.strip().lower()
        if not name:
            continue
        if name not in BENCHMARKS:
            _exit(f"unknown benchmark: {name!r}; choose from {sorted(BENCHMARKS)}")
        bench_list.append(name)
    if not bench_list:
        _exit("--benchmarks must list at least one entry")

    # No bundled corpora in v0.47.0 — operators wire local benchmark JSONLs
    # in v0.47.1. For now we emit an empty-benchmark decontamination, which
    # is equivalent to a pass-through with the flag presence in audit logs.
    console.print(
        "[yellow]Note:[/] benchmark corpora are operator-supplied in v0.47.0 "
        "(--benchmarks names a label only). Pass benchmark text via a future "
        "--benchmark-file flag in v0.47.1 for actual filtering."
    )

    kept, removed = decontaminate_rows(rows, [], n=n, threshold=threshold)
    path = _write_rows(kept, output)
    console.print(
        Panel(
            f"Input:    [bold]{len(rows)}[/]\n"
            f"Kept:     [bold]{len(kept)}[/]\n"
            f"Removed:  [bold]{len(removed)}[/]\n"
            f"Output:   [bold]{escape(path)}[/]",
            title="[bold green]Decontamination[/]",
        )
    )


def toxicity(
    input: str = typer.Option(..., "--input", "-i"),
    output: str = typer.Option("toxicity.jsonl", "--output", "-o"),
    threshold: float = typer.Option(
        0.05, "--threshold", min=0.0, max=1.0,
        help="Rows scoring ≥ threshold are kept in the flagged JSONL.",
    ),
):
    """Score toxicity per row via the keyword baseline."""
    from soup_cli.utils.data_score import _extract_row_text, score_toxicity

    rows = _read_rows(input)
    out_rows = []
    for row in rows:
        try:
            text = _extract_row_text(row)
            s = score_toxicity(text) if text else 0.0
        except (TypeError, ValueError):
            s = 0.0
        if s >= threshold:
            out_rows.append({**row, "_toxicity": s})
    path = _write_rows(out_rows, output)
    console.print(
        Panel(
            f"Input:      [bold]{len(rows)}[/]\n"
            f"Flagged:    [bold]{len(out_rows)}[/]\n"
            f"Threshold:  [bold]{threshold:.3f}[/]\n"
            f"Output:     [bold]{escape(path)}[/]",
            title="[bold green]Toxicity[/]",
        )
    )


def langdetect(
    input: str = typer.Option(..., "--input", "-i"),
    output: str = typer.Option("langdetect.jsonl", "--output", "-o"),
):
    """Tag each row with a 2-letter language code (heuristic)."""
    from soup_cli.utils.data_score import _extract_row_text, detect_language

    rows = _read_rows(input)
    out_rows = []
    for row in rows:
        try:
            text = _extract_row_text(row)
            lang = detect_language(text) if text else "unknown"
        except (TypeError, ValueError):
            lang = "unknown"
        out_rows.append({**row, "_language": lang})
    path = _write_rows(out_rows, output)
    console.print(
        Panel(
            f"Input:   [bold]{len(rows)}[/]\n"
            f"Output:  [bold]{escape(path)}[/]",
            title="[bold green]Lang Detect[/]",
        )
    )


def pii(
    input: str = typer.Option(..., "--input", "-i"),
    output: str = typer.Option("pii.jsonl", "--output", "-o"),
):
    """Flag rows containing email / phone / SSN / credit-card patterns."""
    from soup_cli.utils.data_score import _extract_row_text, detect_pii

    rows = _read_rows(input)
    out_rows = []
    for row in rows:
        try:
            text = _extract_row_text(row)
            hits = detect_pii(text) if text else []
        except (TypeError, ValueError):
            hits = []
        if hits:
            out_rows.append({**row, "_pii_hits": hits})
    path = _write_rows(out_rows, output)
    console.print(
        Panel(
            f"Input:    [bold]{len(rows)}[/]\n"
            f"Flagged:  [bold]{len(out_rows)}[/]\n"
            f"Output:   [bold]{escape(path)}[/]",
            title="[bold green]PII[/]",
        )
    )


def educational(
    input: str = typer.Option(..., "--input", "-i"),
    output: str = typer.Option("educational.jsonl", "--output", "-o"),
):
    """Tag each row with an educational-value score [0, 1]."""
    from soup_cli.utils.data_score import _extract_row_text, score_educational_value

    rows = _read_rows(input)
    out_rows = []
    for row in rows:
        try:
            text = _extract_row_text(row)
            s = score_educational_value(text) if text else 0.0
        except (TypeError, ValueError):
            s = 0.0
        out_rows.append({**row, "_educational": s})
    path = _write_rows(out_rows, output)
    console.print(
        Panel(
            f"Input:   [bold]{len(rows)}[/]\n"
            f"Output:  [bold]{escape(path)}[/]",
            title="[bold green]Educational[/]",
        )
    )
