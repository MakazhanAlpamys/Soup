"""``soup data topics`` — topic map over training data (v0.71.36).

Thin Typer/Rich layer over the pure engine in ``utils/topics.py``.
BERTopic-lite: embed -> k-means -> c-TF-IDF labels -> coverage table.

Labels are EMERGENT unsupervised term clusters, not a classification
against a fixed ontology, and there is no join to ``soup eval coverage``
(that compares an eval suite's scorer mix to a task taxonomy — a different
axis). The help text and docs say so plainly rather than implying the
labels mean more than they do.

Exit 0 = report rendered, 1 = runtime error (bad path/args/model).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from soup_cli.data.loader import load_raw_data
from soup_cli.utils._eval_text import row_text
from soup_cli.utils.embed import DEFAULT_EMBED_MODEL, embed_texts
from soup_cli.utils.paths import atomic_write_text, is_under_cwd
from soup_cli.utils.topics import build_topic_report, kmeans, resolve_k

console = Console()


def _report_to_dict(report) -> dict:
    """JSON-safe rendering of a :class:`~soup_cli.utils.topics.TopicReport`."""
    return {
        "n_rows": report.n_rows,
        "n_clusters": report.n_clusters,
        "warnings": list(report.warnings),
        "topics": [
            {
                "label": topic.label,
                "terms": list(topic.terms),
                "size": topic.size,
                "fraction": topic.fraction,
            }
            for topic in report.topics
        ],
    }


def _parse_clusters(value: str) -> "int | str":
    """``"auto"`` stays a string; anything else must be an int."""
    if value.strip().lower() == "auto":
        return "auto"
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"--clusters must be an integer or 'auto', got {value!r}"
        ) from exc


def topics(
    path: str = typer.Argument(..., help="Path to dataset file (jsonl/json/csv)"),
    clusters: str = typer.Option(
        "auto", "--clusters", "-k",
        help="Number of topic clusters, or 'auto' (sqrt heuristic).",
    ),
    embed_model: str = typer.Option(
        DEFAULT_EMBED_MODEL, "--embed-model", help="Embedding model."
    ),
    device: str = typer.Option("auto", "--device", help="auto/cpu/cuda"),
    seed: int = typer.Option(0, "--seed", help="Clustering seed."),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Write the report as JSON."
    ),
):
    """Cluster a dataset into topics and show a coverage table.

    Labels are emergent term clusters, not a fixed taxonomy.
    """
    file_path = Path(path)
    if not file_path.exists():
        console.print(f"[red]File not found: {escape(str(file_path))}[/]")
        raise typer.Exit(1)
    if output is not None and not is_under_cwd(Path(output)):
        console.print(
            "[red]Output path is outside the working directory: "
            f"{escape(str(output))}[/]"
        )
        raise typer.Exit(1)

    rows = load_raw_data(file_path)
    if not rows:
        console.print("[red]Dataset is empty.[/]")
        raise typer.Exit(1)

    try:
        k_eff = resolve_k(len(rows), _parse_clusters(clusters))
    except (ValueError, TypeError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1)

    texts = [
        row_text(row) or " ".join(str(v) for v in row.values() if v)
        for row in rows
    ]
    console.print(
        f"[dim]Embedding {len(rows)} rows with {escape(embed_model)}...[/]"
    )
    try:
        vectors = embed_texts(texts, model_id=embed_model, device=device)
    except ImportError:
        console.print(
            "[red]soup data topics needs PyTorch + transformers.[/]\n"
            # \[train] escaped: Rich would eat the bracket and print a
            # command that installs WITHOUT the extra.
            "Install with: [bold]pip install 'soup-cli\\[train]'[/]"
        )
        raise typer.Exit(1)
    except (ValueError, TypeError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1)

    labels = kmeans(vectors, k=k_eff, seed=seed)
    report = build_topic_report(rows, labels, k=k_eff)

    table = Table(
        title=f"Topic map — {report.n_rows} rows, "
              f"{report.n_clusters} clusters"
    )
    table.add_column("Topic")
    table.add_column("Rows", justify="right")
    table.add_column("Coverage", justify="right")
    for topic in report.topics:
        table.add_row(
            escape(topic.label),
            str(topic.size),
            f"{topic.fraction * 100:.1f}%",
        )
    console.print(table)
    for warning in report.warnings:
        console.print(f"[yellow]{escape(warning)}[/]")
    console.print(
        "[dim]Labels are emergent term clusters, not a fixed taxonomy.[/]"
    )

    if output is not None:
        atomic_write_text(
            json.dumps(_report_to_dict(report), indent=2), output
        )
        console.print(f"[green]Report written:[/] [bold]{escape(output)}[/]")
