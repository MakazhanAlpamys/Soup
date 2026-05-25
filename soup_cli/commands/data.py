"""soup data — dataset inspection and tools."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Optional

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

    # Vision stats (if dataset contains images)
    _show_vision_stats(data)

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
        "auto", "--format", "-f",
        help="Expected format: auto, alpaca, sharegpt, chatml, dpo, kto, plaintext",
    ),
):
    """Validate dataset format and report issues."""
    file_path = Path(path)
    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/]")
        raise typer.Exit(1)

    data = load_raw_data(file_path)

    # Auto-detect format if not specified
    if fmt == "auto":
        from soup_cli.data.formats import detect_format

        try:
            fmt = detect_format(data)
            console.print(f"[dim]Auto-detected format: {fmt}[/]")
        except ValueError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(1)

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

    # Convert: source -> messages -> target
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
        f"[green]Converted {len(converted)} rows:[/] {src_fmt} -> {to}\n"
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
        f"[green]Dedup complete:[/] {len(data)} -> {len(unique_data)} rows "
        f"([red]-{removed}[/] duplicates)\n"
        f"Output: [bold]{out_path}[/]"
    )


@app.command(name="filter")
def filter_data(
    path: str = typer.Argument(..., help="Path to dataset file"),
    output: str = typer.Option(
        None, "--output", "-o",
        help="Output file path (default: <input>_filtered.jsonl)",
    ),
    perplexity: float = typer.Option(
        None, "--perplexity", "--ppl",
        help="Max perplexity threshold (rows above this are removed)",
    ),
    coherence: float = typer.Option(
        None, "--coherence", "--min-coherence",
        help="Min coherence threshold 0.0-1.0 (rows below this are removed)",
    ),
    perplexity_model: str = typer.Option(
        "gpt2", "--ppl-model",
        help="Model for perplexity scoring (default: gpt2)",
    ),
    field: str = typer.Option(
        None, "--field", "-f",
        help="Field to score (default: all text fields concatenated)",
    ),
    score_only: bool = typer.Option(
        False, "--score-only",
        help="Add scores to data without filtering (writes _scored.jsonl)",
    ),
):
    """Filter dataset by quality: perplexity and/or coherence scoring."""
    file_path = Path(path)
    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/]")
        raise typer.Exit(1)

    if perplexity is None and coherence is None and not score_only:
        console.print(
            "[red]Specify at least one filter: --perplexity, --coherence, or --score-only[/]"
        )
        raise typer.Exit(1)

    data = load_raw_data(file_path)
    if not data:
        console.print("[red]Dataset is empty.[/]")
        raise typer.Exit(1)

    console.print(f"[dim]Scoring {len(data)} rows...[/]")

    # Extract texts for scoring
    texts = []
    for row in data:
        if field and field in row:
            texts.append(str(row[field]))
        else:
            texts.append(" ".join(str(v) for v in row.values() if v))

    # Compute coherence scores (lightweight, always computed)
    from soup_cli.utils.quality import compute_coherence_score

    coherence_scores = compute_coherence_score(texts)

    # Compute perplexity scores (requires model, only if requested)
    perplexity_scores = None
    if perplexity is not None or score_only:
        try:
            from soup_cli.utils.quality import compute_perplexity_scores

            console.print(f"[dim]Computing perplexity with {perplexity_model}...[/]")
            perplexity_scores = compute_perplexity_scores(
                texts, model_name=perplexity_model,
            )
        except ImportError:
            console.print(
                "[yellow]torch/transformers not available for perplexity scoring. "
                "Skipping perplexity.[/]"
            )

    if score_only:
        # Add scores to each row and write output
        scored_data = []
        for idx, row in enumerate(data):
            scored_row = dict(row)
            scored_row["_coherence_score"] = coherence_scores[idx]
            if perplexity_scores is not None:
                scored_row["_perplexity_score"] = round(perplexity_scores[idx], 2)
            scored_data.append(scored_row)

        if output is None:
            output = str(file_path.stem) + "_scored.jsonl"
        out_path = Path(output)
        _write_jsonl(out_path, scored_data)
        console.print(
            f"[green]Scored {len(scored_data)} rows.[/]\n"
            f"Output: [bold]{out_path}[/]"
        )
        return

    # Filter
    kept = []
    removed = []
    for idx, row in enumerate(data):
        remove = False
        if perplexity is not None and perplexity_scores is not None:
            if perplexity_scores[idx] > perplexity:
                remove = True
        if coherence is not None and coherence_scores[idx] < coherence:
            remove = True

        if remove:
            removed.append(row)
        else:
            kept.append(row)

    if output is None:
        output = str(file_path.stem) + "_filtered.jsonl"
    out_path = Path(output)
    _write_jsonl(out_path, kept)

    console.print(
        f"[green]Filter complete:[/] {len(data)} -> {len(kept)} rows "
        f"([red]-{len(removed)}[/] removed)\n"
        f"Output: [bold]{out_path}[/]"
    )
    if perplexity is not None and perplexity_scores is not None:
        avg_ppl = sum(perplexity_scores) / len(perplexity_scores)
        console.print(f"Avg perplexity: [bold]{avg_ppl:.1f}[/] (threshold: {perplexity})")
    if coherence is not None:
        avg_coh = sum(coherence_scores) / len(coherence_scores)
        console.print(f"Avg coherence:  [bold]{avg_coh:.3f}[/] (threshold: {coherence})")


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
        import io
        import sys

        import plotext as plt

        lengths = ext_stats["lengths"]
        if lengths:
            # Force UTF-8 stdout on Windows to avoid UnicodeEncodeError
            # plotext uses box-drawing chars (U+2500 etc.) that cp1251/cp1252 can't encode
            original_stdout = sys.stdout
            needs_redirect = (
                sys.platform == "win32"
                and hasattr(sys.stdout, "encoding")
                and (sys.stdout.encoding or "").lower().replace("-", "") != "utf8"
            )
            if needs_redirect:
                try:
                    sys.stdout = io.TextIOWrapper(
                        sys.stdout.buffer, encoding="utf-8", errors="replace",
                    )
                except AttributeError:
                    pass  # no .buffer (e.g. in tests), keep original

            try:
                plt.clear_figure()
                plt.hist(lengths, bins=30)
                plt.title("Text Length Distribution (chars)")
                plt.xlabel("Length")
                plt.ylabel("Count")
                plt.theme("dark")
                plt.show()
            finally:
                sys.stdout = original_stdout
    except UnicodeEncodeError:
        console.print(
            "\n[dim]Histogram skipped (encoding issue).[/] "
            "Set PYTHONIOENCODING=utf-8 to enable."
        )
    except ImportError:
        console.print(
            "\n[dim]Install plotext for histograms:[/] [bold]pip install plotext[/]"
        )


def _show_vision_stats(data: list[dict]) -> None:
    """Show image statistics if dataset contains image fields."""
    if not data:
        return

    # Check if this is a vision dataset
    sample = data[0]
    if "image" not in sample:
        return

    total = len(data)
    has_image = sum(1 for row in data if row.get("image"))
    missing_image = total - has_image

    # Collect image file info
    extensions: dict[str, int] = {}
    existing = 0
    for row in data:
        img_path = row.get("image", "")
        if not img_path:
            continue
        ext = Path(img_path).suffix.lower()
        extensions[ext] = extensions.get(ext, 0) + 1
        if Path(img_path).exists():
            existing += 1

    vision_table = Table(title="Vision Stats")
    vision_table.add_column("Metric", style="bold")
    vision_table.add_column("Value")
    vision_table.add_row("Images referenced", str(has_image))
    vision_table.add_row("Missing image field", str(missing_image))
    vision_table.add_row("Images found on disk", str(existing))
    if extensions:
        ext_str = ", ".join(f"{ext} ({count})" for ext, count in sorted(extensions.items()))
        vision_table.add_row("Image formats", ext_str)
    console.print(vision_table)


def _write_jsonl(path: Path, data: list[dict]) -> None:
    """Write a list of dicts as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# --- Sampling strategies ---


def _sample_random(data: list[dict], num: int, seed: int | None = None) -> list[dict]:
    """Random sampling without replacement."""
    rng = random.Random(seed)
    num = min(num, len(data))
    return rng.sample(data, num)


def _sample_diverse(
    data: list[dict], num: int, seed: int | None = None
) -> list[dict]:
    """Cluster-based diverse sampling using TF-IDF + K-means.

    Falls back to random sampling if sklearn is not available.
    """
    num = min(num, len(data))
    if num >= len(data):
        return list(data)

    # Extract text representations
    texts = [
        " ".join(str(val) for val in row.values() if val) for row in data
    ]

    try:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(texts)

        num_clusters = min(num, len(data))
        kmeans = MiniBatchKMeans(
            n_clusters=num_clusters, random_state=seed or 0, n_init=3
        )
        labels = kmeans.fit_predict(tfidf_matrix)

        # Sample one item from each cluster (index-based dedup)
        chosen_indices: list[int] = []
        rng = random.Random(seed)
        for cluster_id in range(num_clusters):
            cluster_indices = [
                idx for idx, label in enumerate(labels) if label == cluster_id
            ]
            if cluster_indices:
                chosen_indices.append(rng.choice(cluster_indices))

        sampled = [data[idx] for idx in chosen_indices]

        # If we need more, fill randomly from remaining
        if len(sampled) < num:
            remaining_indices = list(set(range(len(data))) - set(chosen_indices))
            extra_indices = rng.sample(
                remaining_indices, min(num - len(sampled), len(remaining_indices))
            )
            sampled.extend(data[idx] for idx in extra_indices)

        return sampled[:num]

    except ImportError:
        # Fallback: simple length-based diversity (bucket by text length)
        rng = random.Random(seed)
        indexed = [(idx, len(texts[idx])) for idx in range(len(data))]
        indexed.sort(key=lambda pair: pair[1])
        # Evenly spaced picks across sorted list
        step = max(1, len(indexed) // num)
        picked_indices = [
            indexed[idx * step][0] for idx in range(min(num, len(indexed)))
        ]
        picked = [data[idx] for idx in picked_indices]
        # Fill remainder randomly
        if len(picked) < num:
            remaining_indices = list(set(range(len(data))) - set(picked_indices))
            extra_indices = rng.sample(
                remaining_indices, min(num - len(picked), len(remaining_indices))
            )
            picked.extend(data[idx] for idx in extra_indices)
        return picked[:num]


def _sample_hard(data: list[dict], num: int) -> list[dict]:
    """Sample hardest examples by text length (proxy for complexity).

    Longer texts tend to be more complex / challenging.
    """
    num = min(num, len(data))
    if num >= len(data):
        return list(data)

    # Score by total text length (proxy for difficulty)
    scored = []
    for row in data:
        text_len = sum(len(str(val)) for val in row.values() if val)
        scored.append((text_len, row))

    # Sort by length descending, take top N
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [row for _, row in scored[:num]]


@app.command(name="sample")
def sample_data(
    path: str = typer.Argument(..., help="Path to dataset file"),
    output: str = typer.Option(
        None, "--output", "-o",
        help="Output file path (default: <input>_sampled.jsonl)",
    ),
    num: int = typer.Option(
        None, "--n", "-n",
        help="Number of samples to select",
    ),
    pct: float = typer.Option(
        None, "--pct",
        help="Percentage of dataset to sample (0-100)",
    ),
    strategy: str = typer.Option(
        "random", "--strategy", "-s",
        help="Sampling strategy: random, diverse (TF-IDF + clusters), hard (by length)",
    ),
    seed: int = typer.Option(
        None, "--seed",
        help="Random seed for reproducibility",
    ),
):
    """Sample a subset of a dataset using various strategies."""
    file_path = Path(path)
    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/]")
        raise typer.Exit(1)

    if num is None and pct is None:
        console.print("[red]Specify either --n (count) or --pct (percentage).[/]")
        raise typer.Exit(1)

    if strategy not in ("random", "diverse", "hard"):
        console.print(
            f"[red]Unknown strategy: {strategy}[/]\n"
            "Supported: [bold]random[/], [bold]diverse[/], [bold]hard[/]"
        )
        raise typer.Exit(1)

    data = load_raw_data(file_path)
    if not data:
        console.print("[red]Dataset is empty.[/]")
        raise typer.Exit(1)

    # Compute sample count
    if pct is not None:
        sample_count = max(1, int(len(data) * pct / 100))
    else:
        sample_count = num

    # Apply strategy
    if strategy == "random":
        sampled = _sample_random(data, sample_count, seed=seed)
    elif strategy == "diverse":
        sampled = _sample_diverse(data, sample_count, seed=seed)
    elif strategy == "hard":
        sampled = _sample_hard(data, sample_count)
    else:
        sampled = _sample_random(data, sample_count, seed=seed)

    # Resolve output path (with path traversal protection on explicit --output)
    # v0.40.1 Part E — include the strategy in the default filename so
    # successive `random` / `diverse` / `hard` runs don't silently overwrite
    # each other.
    if output is None:
        out_path = file_path.parent / f"{file_path.stem}_sampled_{strategy}.jsonl"
    else:
        out_path = Path(output).resolve()
        cwd = Path.cwd().resolve()
        try:
            out_path.relative_to(cwd)
        except ValueError:
            console.print("[red]Output path must be under the current working directory.[/]")
            raise typer.Exit(1)

    _write_jsonl(out_path, sampled)

    console.print(
        f"[green]Sampled {len(sampled)} rows[/] from {len(data)} "
        f"(strategy: {strategy})\n"
        f"Output: [bold]{out_path}[/]"
    )


@app.command(name="split")
def split_data(
    path: str = typer.Argument(..., help="Path to dataset file"),
    val: int = typer.Option(
        None, "--val",
        help="Validation split: percentage (default) or absolute count (with --absolute)",
    ),
    test: int = typer.Option(
        None, "--test",
        help="Test split: percentage (default) or absolute count (with --absolute)",
    ),
    train: int = typer.Option(
        None, "--train",
        help=(
            "Train split (informational; the train remainder is implied by "
            "--val + --test). Accepted for command parity."
        ),
    ),
    absolute: bool = typer.Option(
        False, "--absolute",
        help="Treat --val/--test as absolute sample counts instead of percentages",
    ),
    seed: int = typer.Option(
        None, "--seed",
        help="Random seed for reproducible splits",
    ),
    stratify: str = typer.Option(
        None, "--stratify",
        help="Field name for stratified splitting (preserves category distribution)",
    ),
):
    """Split dataset into train/val/test files."""
    file_path = Path(path)
    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/]")
        raise typer.Exit(1)

    if val is None and test is None:
        console.print("[red]Specify at least one of --val or --test.[/]")
        raise typer.Exit(1)

    data = load_raw_data(file_path)
    if not data:
        console.print("[red]Dataset is empty.[/]")
        raise typer.Exit(1)

    total = len(data)

    # Calculate split sizes
    if absolute:
        val_count = val or 0
        test_count = test or 0
        if val_count + test_count >= total:
            console.print(
                f"[red]val ({val_count}) + test ({test_count}) >= dataset size ({total}).[/]"
            )
            raise typer.Exit(1)
    else:
        val_count = int(total * val / 100) if val else 0
        test_count = int(total * test / 100) if test else 0
        if val_count + test_count >= total:
            console.print(
                f"[red]Split sizes ({val_count} + {test_count}) >= dataset size ({total}).[/]"
            )
            raise typer.Exit(1)

    # Perform split
    if stratify:
        train_data, val_data, test_data = _stratified_split(
            data, val_count, test_count, stratify, seed=seed,
        )
    else:
        train_data, val_data, test_data = _random_split(
            data, val_count, test_count, seed=seed,
        )

    # Write output files
    stem = file_path.stem
    parent = file_path.parent

    train_path = parent / f"{stem}_train.jsonl"
    _write_jsonl(train_path, train_data)

    output_msg = (
        f"[green]Split {total} rows:[/]\n"
        f"  Train: {len(train_data)} -> [bold]{train_path}[/]"
    )

    if val_data:
        val_path = parent / f"{stem}_val.jsonl"
        _write_jsonl(val_path, val_data)
        output_msg += f"\n  Val:   {len(val_data)} -> [bold]{val_path}[/]"

    if test_data:
        test_path = parent / f"{stem}_test.jsonl"
        _write_jsonl(test_path, test_data)
        output_msg += f"\n  Test:  {len(test_data)} -> [bold]{test_path}[/]"

    console.print(output_msg)


def _random_split(
    data: list, val_count: int, test_count: int, seed: int | None = None,
) -> tuple:
    """Random split into train/val/test."""
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)

    test_indices = set(indices[:test_count])
    val_indices = set(indices[test_count:test_count + val_count])

    train_data = []
    val_data = []
    test_data = []

    for idx in range(len(data)):
        if idx in test_indices:
            test_data.append(data[idx])
        elif idx in val_indices:
            val_data.append(data[idx])
        else:
            train_data.append(data[idx])

    return train_data, val_data, test_data


def _stratified_split(
    data: list, val_count: int, test_count: int,
    stratify_field: str, seed: int | None = None,
) -> tuple:
    """Stratified split preserving category distribution."""
    # Group by stratify field
    groups: dict[str, list[int]] = {}
    for idx, row in enumerate(data):
        key = str(row.get(stratify_field, "unknown"))
        groups.setdefault(key, []).append(idx)

    rng = random.Random(seed)
    total = len(data)

    train_indices = []
    val_indices = []
    test_indices = []

    for key, indices in groups.items():
        rng.shuffle(indices)
        group_size = len(indices)
        group_frac = group_size / total

        group_val = round(val_count * group_frac) if val_count else 0
        group_test = round(test_count * group_frac) if test_count else 0

        # Ensure we don't take more than available
        group_val = min(group_val, group_size)
        group_test = min(group_test, group_size - group_val)

        test_indices.extend(indices[:group_test])
        val_indices.extend(indices[group_test:group_test + group_val])
        train_indices.extend(indices[group_test + group_val:])

    train_data = [data[idx] for idx in train_indices]
    val_data = [data[idx] for idx in val_indices]
    test_data = [data[idx] for idx in test_indices]

    return train_data, val_data, test_data


# ---------------------------------------------------------------------------
# HuggingFace Dataset Hub helpers
# ---------------------------------------------------------------------------


def list_datasets(search: str, sort: str = "downloads", limit: int = 20) -> list:
    """Search HuggingFace Hub for datasets. Returns list of DatasetInfo objects."""
    from huggingface_hub import HfApi

    api = HfApi()
    return list(api.list_datasets(search=search, sort=sort, limit=limit))


def _hf_dataset_info(dataset_id: str) -> dict:
    """Fetch metadata about a HuggingFace dataset."""
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        info = api.dataset_info(dataset_id)
    except Exception as exc:
        raise ValueError(f"Dataset not found: {dataset_id} — {exc}") from exc

    # Extract split sizes
    splits: dict[str, int] = {}
    if hasattr(info, "card_data") and info.card_data:
        ds_info = getattr(info.card_data, "dataset_info", None)
        if ds_info and isinstance(ds_info, dict):
            for config_data in ds_info.values():
                if isinstance(config_data, dict) and "splits" in config_data:
                    for split_name, split_data in config_data["splits"].items():
                        if isinstance(split_data, dict):
                            splits[split_name] = split_data.get("num_examples", 0)

    # Extract feature names
    features: list[str] = []
    if hasattr(info, "card_data") and info.card_data:
        ds_info = getattr(info.card_data, "dataset_info", None)
        if ds_info and isinstance(ds_info, dict):
            for config_data in ds_info.values():
                if isinstance(config_data, dict) and "features" in config_data:
                    feat_list = config_data["features"]
                    if isinstance(feat_list, list):
                        for feat in feat_list:
                            if isinstance(feat, dict) and "name" in feat:
                                features.append(feat["name"])
                    break

    return {
        "id": info.id,
        "description": getattr(info, "description", "") or "",
        "downloads": getattr(info, "downloads", 0) or 0,
        "likes": getattr(info, "likes", 0) or 0,
        "size_bytes": getattr(info, "size", None),
        "splits": splits,
        "features": features,
        "tags": list(info.tags) if info.tags else [],
    }


def _hf_download_dataset(
    dataset_id: str,
    split: str = "train",
    samples: int | None = None,
) -> list[dict]:
    """Download a dataset from HuggingFace Hub and return as list of dicts."""
    from datasets import load_dataset

    try:
        ds = load_dataset(
            dataset_id, split=split, streaming=True, trust_remote_code=False,
        )
    except Exception as exc:
        raise ValueError(f"Failed to load dataset {dataset_id}: {exc}") from exc

    rows: list[dict] = []
    for idx, row in enumerate(ds):
        if samples is not None and idx >= samples:
            break
        rows.append(dict(row))

    return rows


def _format_size_bytes(size_bytes: int | None) -> str:
    """Format byte count as human-readable string."""
    if size_bytes is None:
        return "unknown"
    if size_bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_idx = 0
    size = float(size_bytes)
    while size >= 1024 and unit_idx < len(units) - 1:
        size /= 1024
        unit_idx += 1
    if unit_idx == 0:
        return f"{int(size)} {units[unit_idx]}"
    return f"{size:.1f} {units[unit_idx]}"


def _format_count(count: int) -> str:
    """Format large numbers with K/M suffix."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)


# ---------------------------------------------------------------------------
# HuggingFace Dataset Hub CLI commands
# ---------------------------------------------------------------------------


@app.command(name="search")
def search_datasets(
    query: str = typer.Argument(..., help="Search query for HuggingFace datasets"),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum results to show"),
    sort: str = typer.Option(
        "downloads", "--sort", "-s",
        help="Sort by: downloads, likes, lastModified, trending, createdAt",
    ),
):
    """Search HuggingFace Hub for datasets."""
    valid_sorts = {"downloads", "likes", "lastModified", "trending", "createdAt"}
    if sort not in valid_sorts:
        console.print(
            f"[red]Invalid sort: {sort}[/]\n"
            f"Valid options: {', '.join(sorted(valid_sorts))}"
        )
        raise typer.Exit(1)

    try:
        datasets = list_datasets(search=query, sort=sort, limit=limit)
    except ImportError:
        console.print(
            "[red]huggingface_hub not available.[/]\n"
            "Install with: [bold]pip install huggingface-hub[/]"
        )
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]Search failed: {exc}[/]")
        raise typer.Exit(1)

    if not datasets:
        console.print(f"[yellow]No datasets found for '{query}'.[/]")
        return

    table = Table(title=f"HuggingFace Datasets: '{query}'")
    table.add_column("Dataset", style="bold cyan", max_width=45)
    table.add_column("Downloads", justify="right")
    table.add_column("Likes", justify="right")
    table.add_column("Tags", max_width=30)

    for ds_item in datasets[:limit]:
        ds_tags = getattr(ds_item, "tags", []) or []
        tag_str = ", ".join(ds_tags[:5])
        if len(ds_tags) > 5:
            tag_str += "..."
        table.add_row(
            ds_item.id,
            _format_count(getattr(ds_item, "downloads", 0) or 0),
            _format_count(getattr(ds_item, "likes", 0) or 0),
            tag_str,
        )

    console.print(table)
    console.print(f"[dim]Showing {min(limit, len(datasets))} results.[/]")


@app.command(name="preview")
def preview_dataset(
    dataset_id: str = typer.Argument(
        ..., help="HuggingFace dataset ID (e.g. teknium/OpenHermes-2.5)"
    ),
):
    """Preview a remote HuggingFace dataset: metadata, splits, features."""
    try:
        info = _hf_dataset_info(dataset_id)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1)
    except ImportError:
        console.print(
            "[red]huggingface_hub not available.[/]\n"
            "Install with: [bold]pip install huggingface-hub[/]"
        )
        raise typer.Exit(1)

    table = Table(title=f"Dataset: {info['id']}")
    table.add_column("Field", style="bold")
    table.add_column("Value", max_width=80)

    table.add_row("ID", info["id"])
    desc = info["description"]
    if len(desc) > 200:
        desc = desc[:200] + "..."
    table.add_row("Description", desc or "[dim]No description[/]")
    table.add_row("Downloads", _format_count(info["downloads"]))
    table.add_row("Likes", _format_count(info["likes"]))
    table.add_row("Size", _format_size_bytes(info["size_bytes"]))

    if info["splits"]:
        splits_str = ", ".join(
            f"{name} ({_format_count(count)})"
            for name, count in info["splits"].items()
        )
        table.add_row("Splits", splits_str)
    else:
        table.add_row("Splits", "[dim]Not available (use streaming to explore)[/]")

    if info["features"]:
        table.add_row("Features", ", ".join(info["features"]))

    if info["tags"]:
        table.add_row("Tags", ", ".join(info["tags"][:10]))

    console.print(table)


@app.command(name="download")
def download_dataset(
    dataset_id: str = typer.Argument(
        ..., help="HuggingFace dataset ID (e.g. teknium/OpenHermes-2.5)"
    ),
    output: str = typer.Option(
        None, "--output", "-o",
        help="Output file path (default: <dataset-name>.jsonl)",
    ),
    split: str = typer.Option(
        "train", "--split",
        help="Dataset split to download (e.g. train, test, train[:1000])",
    ),
    samples: int = typer.Option(
        None, "--samples", "-n",
        help="Max number of samples to download (streams, no full download)",
    ),
    fmt: str = typer.Option(
        None, "--format", "-f",
        help="Convert to Soup format after download: alpaca, sharegpt, chatml",
    ),
    trust_remote_code: bool = typer.Option(
        False,
        "--trust-remote-code",
        help=(
            "Allow datasets that ship custom Python loaders. Default deny "
            "(v0.36.0). Only enable if you trust the source."
        ),
    ),
    hub: str = typer.Option(
        "hf",
        "--hub",
        help=(
            "Source hub: hf (default) / modelscope / modelers. "
            "Non-HF hubs require the matching SDK (v0.53.8 #130)."
        ),
    ),
):
    """Download a HuggingFace dataset and save as JSONL."""
    # v0.53.8 #130 — validate --hub at the CLI boundary; only `hf` is wired
    # for live dataset download in this release (modelscope / modelers
    # dataset SDKs differ from snapshot_download; live wiring tracked for
    # v0.53.9). Non-HF hubs surface a friendly error.
    from soup_cli.utils.hubs import validate_hub_name

    try:
        hub_canonical = validate_hub_name(hub)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=2) from exc
    # v0.53.10 #153 — non-HF hub live dataset download. ModelScope uses
    # the ``MsDataset`` API (different shape from snapshot_download), so
    # we dispatch here instead of going through utils.hubs.download_repo.
    if hub_canonical != "hf":
        from soup_cli.utils.hubs import download_repo as _download_repo

        try:
            if hub_canonical == "modelscope":
                try:
                    from modelscope.msdatasets import (
                        MsDataset,  # type: ignore[import-not-found]
                    )
                except ImportError as exc:
                    console.print(
                        "[red]modelscope is not installed. "
                        "Install with: pip install modelscope[/]"
                    )
                    raise typer.Exit(1) from exc
                _ms_ds = MsDataset.load(  # noqa: F841 — touched for side effect
                    dataset_id, split=split
                )
                console.print(
                    f"[dim]ModelScope dataset {dataset_id} loaded; "
                    "use soup_cli.utils.hubs.download_repo for raw "
                    "snapshot download.[/]"
                )
            else:  # modelers
                try:
                    out_dir = _download_repo(
                        hub_canonical,
                        dataset_id,
                        local_dir=str(Path.cwd() / ".soup_hub_cache"
                                      / "datasets"
                                      / dataset_id.replace("/", "__")),
                        repo_type="dataset",
                    )
                except ImportError as exc:
                    console.print(f"[red]{exc}[/]")
                    raise typer.Exit(1) from exc
                console.print(
                    f"[green]Downloaded {dataset_id} from {hub_canonical} → "
                    f"{out_dir}[/]"
                )
        except (TypeError, ValueError) as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(2) from exc
        return
    max_download_samples = 1_000_000
    if samples is not None and samples > max_download_samples:
        console.print(
            f"[red]--samples cannot exceed {max_download_samples:,}.[/]"
        )
        raise typer.Exit(1)

    # Resolve output path
    if output is None:
        ds_name = dataset_id.split("/")[-1] if "/" in dataset_id else dataset_id
        # Strip embedded path separators to prevent traversal
        ds_name = Path(ds_name).name
        out_path = (Path.cwd() / f"{ds_name}.jsonl").resolve()
        cwd = Path.cwd().resolve()
        try:
            out_path.relative_to(cwd)
        except ValueError:
            console.print(
                "[red]Derived output path escapes working directory.[/]"
            )
            raise typer.Exit(1)
    else:
        out_path = Path(output).resolve()
        cwd = Path.cwd().resolve()
        try:
            out_path.relative_to(cwd)
        except ValueError:
            console.print(
                "[red]Output path must be under the current working directory.[/]"
            )
            raise typer.Exit(1)

    from rich.panel import Panel

    console.print(Panel(
        "[bold yellow]Warning:[/] Downloading this dataset may execute a "
        "remote dataset loading script from HuggingFace Hub.\n\n"
        "Only download datasets from sources you trust.",
        title="Remote Code Warning",
        border_style="yellow",
    ))
    console.print(f"[dim]Downloading {dataset_id} (split={split})...[/]")

    try:
        data = _hf_download_dataset(
            dataset_id, split=split, samples=samples,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1)
    except ImportError:
        console.print(
            "[red]datasets library not available.[/]\n"
            "Install with: [bold]pip install datasets[/]"
        )
        raise typer.Exit(1)

    if not data:
        console.print("[red]No data downloaded (dataset may be empty).[/]")
        raise typer.Exit(1)

    # Optional format conversion
    if fmt:
        from soup_cli.data.formats import (
            CONVERTIBLE_FORMATS,
            detect_format,
            format_to_messages,
            messages_to_format,
        )

        if fmt not in CONVERTIBLE_FORMATS:
            console.print(
                f"[red]Invalid format: {fmt}[/]\n"
                f"Supported: {', '.join(CONVERTIBLE_FORMATS)}"
            )
            raise typer.Exit(1)

        try:
            src_fmt = detect_format(data)
        except ValueError:
            src_fmt = None

        if src_fmt and src_fmt != fmt:
            converted = []
            for row in data:
                messages = format_to_messages(row, src_fmt)
                if messages is not None:
                    result = messages_to_format(messages, fmt)
                    if result is not None:
                        converted.append(result)
            if converted:
                data = converted
                console.print(
                    f"[dim]Converted {len(data)} rows to {fmt} format.[/]"
                )

    # Apply samples limit if data came from non-streaming path
    if samples is not None and len(data) > samples:
        data = data[:samples]

    _write_jsonl(out_path, data)
    console.print(
        f"[green]Downloaded {len(data)} rows.[/]\n"
        f"Output: [bold]{out_path}[/]"
    )


# ---------------------------------------------------------------------------
# Dataset registry CLI commands
# ---------------------------------------------------------------------------


def _get_registry_path() -> Path:
    """Get the default registry path (~/.soup/datasets.json)."""
    from soup_cli.utils.registry import _default_registry_path

    return _default_registry_path()


@app.command(name="augment")
def augment_data(
    input_path: str = typer.Option(..., "--input", "-i", help="Source JSONL file"),
    output_path: str = typer.Option(
        "augmented.jsonl", "--output", "-o", help="Output JSONL path"
    ),
    strategy: str = typer.Option(
        "rephrase", "--strategy", "-s",
        help="Augmentation strategy: rephrase, translate, style",
    ),
    provider: str = typer.Option(
        "ollama", "--provider", "-p",
        help="LLM provider: openai, ollama, anthropic, server, vllm",
    ),
    count: int = typer.Option(
        2, "--count", "-c", min=1, max=10,
        help="Augmentation multiplier (1-10)",
    ),
    lang: str = typer.Option(
        "", "--lang", help="Comma-separated target languages for translate",
    ),
    styles: str = typer.Option(
        "", "--styles", help="Comma-separated styles for style strategy",
    ),
    requests_per_minute: int = typer.Option(
        60, "--requests-per-minute", min=1, max=600,
        help="Rate limit for provider requests",
    ),
    dedup: bool = typer.Option(
        False, "--dedup", help="Deduplicate augmented + original data",
    ),
):
    """Augment a dataset via LLM (rephrase / translate / style)."""
    from soup_cli.data.augment import STRATEGIES

    if strategy not in STRATEGIES:
        console.print(
            f"[red]Unknown strategy: {strategy}. "
            f"Options: {', '.join(STRATEGIES.keys())}[/]"
        )
        raise typer.Exit(1)

    # Path traversal protection for input
    try:
        input_resolved = Path(input_path).resolve()
        input_resolved.relative_to(Path.cwd().resolve())
    except ValueError:
        console.print("[red]Input path must be under the current working directory.[/]")
        raise typer.Exit(1)

    if not input_resolved.exists():
        console.print(f"[red]Input file not found: {input_resolved}[/]")
        raise typer.Exit(1)

    # Path traversal protection for output
    try:
        output_resolved = Path(output_path).resolve()
        output_resolved.relative_to(Path.cwd().resolve())
    except ValueError:
        console.print("[red]Output path must be under the current working directory.[/]")
        raise typer.Exit(1)

    # Load data
    data = load_raw_data(input_resolved)
    if not data:
        console.print("[red]Input dataset is empty.[/]")
        raise typer.Exit(1)

    console.print(
        f"[dim]Loaded {len(data)} examples from {input_resolved.name}[/]"
    )

    # Load provider
    provider_instance = _load_augment_provider(provider, requests_per_minute)

    max_entries = 10
    max_entry_len = 32

    def _bounded_list(raw: str, field: str) -> list[str]:
        parts = [s.strip() for s in raw.split(",") if s.strip()]
        if len(parts) > max_entries:
            raise ValueError(
                f"--{field} accepts at most {max_entries} entries"
            )
        for entry in parts:
            if len(entry) > max_entry_len:
                raise ValueError(
                    f"--{field} entries must be <= {max_entry_len} chars"
                )
        return parts

    # Run strategy
    augment_fn = STRATEGIES[strategy]
    try:
        if strategy == "translate":
            target_langs = _bounded_list(lang, "lang")
            augmented = augment_fn(
                data, provider=provider_instance,
                languages=target_langs or None,
            )
        elif strategy == "style":
            target_styles = _bounded_list(styles, "styles")
            augmented = augment_fn(
                data, provider=provider_instance, styles=target_styles or None,
            )
        else:
            augmented = augment_fn(data, provider=provider_instance, count=count)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1)

    # Optional dedup
    if dedup:
        seen = set()
        deduped: list[dict] = []
        for row in data + augmented:
            key = json.dumps(row, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        final_rows = deduped
    else:
        final_rows = data + augmented

    # Write output
    output_resolved.parent.mkdir(parents=True, exist_ok=True)
    with open(output_resolved, "w", encoding="utf-8") as fh:
        for row in final_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    console.print(
        f"[green]Augmentation complete:[/] {len(data)} → {len(final_rows)} "
        f"({strategy} via {provider})\n"
        f"  Output: {output_resolved}"
    )


def _load_augment_provider(provider: str, rpm: int):
    """Construct an LLM provider instance with generate(prompt) method."""
    # Minimal provider abstraction — wraps existing sync clients.
    if provider == "ollama":
        from soup_cli.data.providers.ollama import OllamaProvider

        return OllamaProvider(model="llama3.1:8b", rate_limit_rpm=rpm)
    if provider == "anthropic":
        from soup_cli.data.providers.anthropic import AnthropicProvider

        return AnthropicProvider(model="claude-3-5-haiku-20241022", rate_limit_rpm=rpm)
    if provider == "vllm":
        from soup_cli.data.providers.vllm import VllmProvider

        return VllmProvider(base_url="http://localhost:8000/v1", rate_limit_rpm=rpm)
    raise ValueError(
        f"Unknown provider '{provider}'. Options: ollama, anthropic, vllm."
    )


@app.command(name="register")
def register_data(
    name_arg: Optional[str] = typer.Argument(
        None, help="Dataset name (positional alternative to --name)",
    ),
    path_arg: Optional[str] = typer.Argument(
        None, help="Path to dataset file (positional alternative to --path)",
    ),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Dataset name"),
    path: Optional[str] = typer.Option(
        None, "--path", "-p", help="Path to dataset file"
    ),
    fmt: str = typer.Option(
        "auto", "--format", "-f",
        help="Dataset format: alpaca, sharegpt, chatml, dpo, kto, auto",
    ),
):
    """Register a local dataset by name for use in soup.yaml.

    Accepts both ``--name X --path Y`` and positional ``X Y``.
    """
    from soup_cli.utils.registry import register_dataset

    # Resolve positional vs option, with conflict detection
    final_name = name if name is not None else name_arg
    final_path = path if path is not None else path_arg
    if final_name is None or final_path is None:
        console.print(
            "[red]Provide both name and path "
            "(positional `<name> <path>` or `--name --path`).[/]"
        )
        raise typer.Exit(2)
    if name is not None and name_arg is not None and name != name_arg:
        console.print("[red]Conflict: --name and positional name differ.[/]")
        raise typer.Exit(2)
    if path is not None and path_arg is not None and path != path_arg:
        console.print("[red]Conflict: --path and positional path differ.[/]")
        raise typer.Exit(2)

    # Path traversal protection — use os.path.realpath + commonpath
    # (project standard; Windows 8.3 short-name safe).
    import os as _os

    from soup_cli.utils.paths import is_under_cwd

    if not is_under_cwd(final_path):
        console.print(
            "[red]Dataset path must be under the current working directory.[/]"
        )
        raise typer.Exit(1)
    resolved = Path(_os.path.realpath(final_path))

    registry_path = _get_registry_path()

    try:
        register_dataset(final_name, str(resolved), fmt, registry_path=registry_path)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1)

    console.print(
        f"[green]Registered dataset '[bold]{final_name}[/bold]'[/]\n"
        f"  Path: {final_path}\n"
        f"  Format: {fmt}"
    )


@app.command(name="unregister")
def unregister_data(
    name_arg: Optional[str] = typer.Argument(
        None, help="Dataset name (positional alternative to --name)",
    ),
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Dataset name to remove"
    ),
):
    """Remove a dataset from the local registry.

    Accepts both ``--name X`` and positional ``X``.
    """
    from soup_cli.utils.registry import unregister_dataset

    final_name = name if name is not None else name_arg
    if final_name is None:
        console.print(
            "[red]Provide a dataset name "
            "(positional `<name>` or `--name`).[/]"
        )
        raise typer.Exit(2)
    if name is not None and name_arg is not None and name != name_arg:
        console.print("[red]Conflict: --name and positional name differ.[/]")
        raise typer.Exit(2)

    registry_path = _get_registry_path()
    removed = unregister_dataset(final_name, registry_path=registry_path)

    if removed:
        console.print(f"[green]Removed dataset '{final_name}' from registry.[/]")
    else:
        console.print(f"[red]Dataset '{final_name}' not found in registry.[/]")
        raise typer.Exit(1)


@app.command(name="from-traces")
def from_traces_cmd(
    logs: str = typer.Option(
        ..., "--logs", help="Path to JSONL trace log (or directory for soup-serve)",
    ),
    format: str = typer.Option(
        ..., "--format", help="Trace format: langchain | openai | soup-serve",
    ),
    signal: str = typer.Option(
        "thumbs_up", "--signal",
        help="Signal to extract pairs from: thumbs_up | regenerations | user_edit",
    ),
    output: str = typer.Option(
        "prefs.jsonl", "--output", "-o",
        help="Output path for preference pairs (JSONL)",
    ),
    judge: bool = typer.Option(
        False, "--judge",
        help="Filter pairs via LLM-as-a-judge confidence (v0.40.3 #33).",
    ),
    judge_provider: str = typer.Option(
        "openai", "--judge-provider",
        help="Judge backend: openai | server | ollama. Used with --judge.",
    ),
    judge_model: str = typer.Option(
        "gpt-4o-mini", "--judge-model",
        help="Judge model id (e.g. 'gpt-4o-mini', 'llama3', 'qwen2.5'). Used with --judge.",
    ),
    judge_api_base: Optional[str] = typer.Option(
        None, "--judge-api-base",
        help="Judge API base URL. SSRF-protected. Used with --judge.",
    ),
    min_confidence: float = typer.Option(
        0.7, "--min-confidence",
        help="Drop pairs with judge-confidence below this threshold (0.0 - 1.0).",
    ),
) -> None:
    """Harvest preference pairs from production traces (v0.26.0 Part C).

    Prominent reminder: traces may contain sensitive user data; review
    before sharing or uploading to external systems.
    """
    import json as _json

    from rich.markup import escape as _escape
    from rich.panel import Panel

    from soup_cli.data.traces import (
        SUPPORTED_FORMATS,
        SUPPORTED_SIGNALS,
        build_pairs,
        parse_langchain,
        parse_openai,
        parse_soup_serve,
    )
    from soup_cli.utils.paths import is_under_cwd as _under_cwd

    if format not in SUPPORTED_FORMATS:
        console.print(
            f"[red]Unknown format '{format}'. "
            f"Supported: {', '.join(SUPPORTED_FORMATS)}[/]"
        )
        raise typer.Exit(1)
    if signal not in SUPPORTED_SIGNALS:
        console.print(
            f"[red]Unknown signal '{signal}'. "
            f"Supported: {', '.join(SUPPORTED_SIGNALS)}[/]"
        )
        raise typer.Exit(1)

    logs_path = Path(logs)
    if not _under_cwd(logs_path):
        console.print(f"[red]--logs '{logs}' is outside cwd - refusing[/]")
        raise typer.Exit(1)
    if not logs_path.exists():
        console.print(f"[red]--logs not found: {logs}[/]")
        raise typer.Exit(1)

    output_path = Path(output)
    if not _under_cwd(output_path):
        console.print(f"[red]--output '{output}' is outside cwd - refusing[/]")
        raise typer.Exit(1)

    console.print(Panel(
        "[yellow]Traces may contain sensitive user data.[/]\n"
        "Review the output before sharing or uploading.",
        title="PII reminder", border_style="yellow",
    ))

    max_trace_lines = 100_000  # matches eval / human-eval caps in the project
    events: list[dict] = []
    if format == "soup-serve":
        trace_iter = parse_soup_serve(str(logs_path))
    else:
        if logs_path.is_file():
            with logs_path.open("r", encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, start=1):
                    if line_no > max_trace_lines:
                        console.print(
                            f"[yellow]--logs exceeds cap of {max_trace_lines} "
                            "lines; truncating.[/]"
                        )
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(_json.loads(line))
                    except _json.JSONDecodeError:
                        continue
        if format == "langchain":
            trace_iter = parse_langchain(events)
        else:  # openai
            trace_iter = parse_openai(events)

    pairs = list(build_pairs(trace_iter, signal=signal))

    if judge:
        # v0.40.3 (#33 (a)) — LLM-judge confidence filter.
        from soup_cli.data.traces.quality import judge_filter_pairs
        from soup_cli.eval.judge import VALID_PROVIDERS, JudgeEvaluator

        # Friendly early validation matches the existing CLI conventions —
        # fall through to the constructor only after the obvious typo is caught.
        if judge_provider not in VALID_PROVIDERS:
            console.print(
                f"[red]--judge-provider '{_escape(judge_provider)}' is invalid. "
                f"Choose: {', '.join(sorted(VALID_PROVIDERS))}[/]"
            )
            raise typer.Exit(1)

        try:
            judge_evaluator = JudgeEvaluator(
                provider=judge_provider,
                model=judge_model,
                api_base=judge_api_base,
            )
        except ValueError as exc:
            console.print(f"[red]--judge config error:[/] {_escape(str(exc))}")
            raise typer.Exit(1) from exc

        # Cost-shock warning: each pair → TWO judge calls (chosen + rejected).
        projected = len(pairs) * 2
        console.print(
            f"[yellow]Judge filter will issue ~{projected} backend calls "
            f"({len(pairs)} pairs × 2). Cost depends on provider and model. "
            f"Use --min-confidence to tune throughput.[/]"
        )

        try:
            filtered, report = judge_filter_pairs(
                pairs, judge=judge_evaluator, min_confidence=min_confidence,
            )
        except (TypeError, ValueError) as exc:
            console.print(f"[red]--judge runtime error:[/] {_escape(str(exc))}")
            raise typer.Exit(1) from exc

        console.print(
            f"[green]Judge filter:[/] kept={report.kept} dropped={report.dropped} "
            f"errors={report.errors} (min_confidence={min_confidence:.2f})"
        )
        pairs = filtered

    with output_path.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(_json.dumps(pair.to_jsonl_dict(), ensure_ascii=False) + "\n")

    console.print(
        f"[green]Wrote {len(pairs)} preference pair(s)[/] to "
        f"[cyan]{_escape(str(output_path))}[/]"
    )


@app.command(name="review")
def review_cmd(
    input_file: str = typer.Argument(
        ..., metavar="INPUT", help="Path to preference JSONL (chosen/rejected)",
    ),
    sample: int = typer.Option(
        10, "--sample", "-s",
        help="How many pairs to preview (1-100)",
    ),
) -> None:
    """Preview preference pairs for manual review."""
    import json as _json

    from rich.markup import escape as _escape
    from rich.panel import Panel

    sample = max(1, min(int(sample), 100))
    path = Path(input_file)
    if not path.exists():
        console.print(f"[red]File not found:[/] {input_file}")
        raise typer.Exit(1)

    shown = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if shown >= sample:
                break
            line = line.strip()
            if not line:
                continue
            try:
                entry = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            prompt = str(entry.get("prompt", ""))
            chosen = str(entry.get("chosen", ""))
            rejected = str(entry.get("rejected", ""))
            console.print(Panel(
                f"[bold cyan]Prompt:[/] {_escape(prompt[:400])}\n\n"
                f"[green]Chosen:[/] {_escape(chosen[:400])}\n\n"
                f"[red]Rejected:[/] {_escape(rejected[:400])}",
                title=f"Pair {shown + 1}",
                border_style="blue",
            ))
            shown += 1

    if shown == 0:
        console.print("[yellow]No pairs found in file.[/]")


@app.command(name="registry")
def list_registry():
    """List all registered datasets."""
    from soup_cli.utils.registry import load_registry

    registry_path = _get_registry_path()
    registry = load_registry(registry_path)

    if not registry:
        console.print("[yellow]No datasets registered.[/]")
        console.print(
            "[dim]Register with: "
            "soup data register --name my-data --path data.jsonl --format alpaca[/]"
        )
        return

    table = Table(title="Registered Datasets")
    table.add_column("Name", style="bold cyan")
    table.add_column("Path")
    table.add_column("Format")

    from rich.markup import escape

    for ds_name, ds_info in sorted(registry.items()):
        table.add_row(
            escape(ds_name),
            escape(ds_info.get("path", "")),
            escape(ds_info.get("format", "")),
        )

    console.print(table)


@app.command(name="push")
def push_dataset_cmd(
    input_path: str = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to local JSONL dataset file",
    ),
    hf_dataset: str = typer.Option(
        ...,
        "--hf-dataset",
        help="HuggingFace dataset repo id (e.g. user/my-dataset)",
    ),
    private: bool = typer.Option(
        False, "--private", help="Make the HF dataset repo private",
    ),
    commit_message: str = typer.Option(
        "Upload dataset with Soup CLI",
        "--message",
        help="Commit message for the dataset upload",
    ),
):
    """Upload a local JSONL dataset to HuggingFace Hub as a dataset repo."""
    from soup_cli.utils.hf import (
        get_hf_api,
        resolve_endpoint,
        resolve_token,
        validate_repo_id,
    )
    from soup_cli.utils.paths import is_under_cwd

    file_path = Path(input_path)
    if not file_path.exists():
        console.print(f"[red]Dataset file not found: {file_path}[/]")
        raise typer.Exit(1)
    if not file_path.is_file():
        console.print(f"[red]Expected a file, got a directory: {file_path}[/]")
        raise typer.Exit(1)
    if not is_under_cwd(file_path):
        console.print(
            "[red]Dataset path must stay under the current working directory.[/]"
        )
        raise typer.Exit(1)

    try:
        validate_repo_id(hf_dataset)
    except ValueError as exc:
        console.print(f"[red]Invalid --hf-dataset repo id:[/] {exc}")
        raise typer.Exit(1) from exc

    token = resolve_token()
    if token is None:
        console.print(
            "[red]No HuggingFace token found.[/]\n"
            "Set HF_TOKEN env var or run: [bold]huggingface-cli login[/]"
        )
        raise typer.Exit(1)

    try:
        endpoint = resolve_endpoint()
    except ValueError as exc:
        console.print(f"[red]HF_ENDPOINT invalid:[/] {exc}")
        raise typer.Exit(1) from exc

    try:
        api = get_hf_api(token=token, endpoint=endpoint)
    except ImportError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc

    # Sanitise commit message to a single short line — prevents multi-line
    # injection into HF commit history.
    safe_commit = (commit_message.splitlines()[0][:200] if commit_message else "")
    try:
        api.create_repo(
            repo_id=hf_dataset, repo_type="dataset",
            private=private, exist_ok=True,
        )
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=file_path.name,
            repo_id=hf_dataset,
            repo_type="dataset",
            commit_message=safe_commit,
        )
    except Exception as exc:
        console.print(f"[red]Upload failed:[/] {exc}")
        raise typer.Exit(1) from exc

    console.print(
        f"[green]Uploaded to[/] https://huggingface.co/datasets/{hf_dataset}"
    )


# --- v0.42.0 Part C / F: AOT preprocess + document ingestion ---------------

@app.command(name="preprocess")
def preprocess_dataset(
    config_path: str = typer.Argument(
        ..., help="Path to soup.yaml — uses data.train + tokenizer + max_length"
    ),
    output_dir: str = typer.Option(
        "./.soup-tokenized", "--output", "-o",
        help="Cache directory under cwd",
    ),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation prompt"),
) -> None:
    """AOT-tokenize a dataset and cache to disk for reuse across runs.

    v0.53.7 #86: live tokenize loop. Lazy-imports ``transformers``/``datasets``,
    iterates ``data.train``, renders each row through chat template (or raw
    text for pretrain), tokenizes with ``max_length=data.max_length`` +
    ``truncation=True``, writes Arrow shards under
    ``<output>/<cache_key>/`` via ``Dataset.save_to_disk``. Writes
    ``metadata.json`` with cache_key + row_count + tokenizer_name + max_length.

    Rows capped at 10M to defend against pathological dataset sizes.
    """
    import json as _json

    from soup_cli.config.loader import load_config
    from soup_cli.utils.data_pipeline import make_preprocess_cache_key
    from soup_cli.utils.paths import is_under_cwd

    cfg_real = os.path.realpath(config_path)
    if not is_under_cwd(cfg_real):
        console.print(
            f"[red]--config must stay under cwd[/] (got {config_path!r})"
        )
        raise typer.Exit(1)
    if not Path(cfg_real).is_file():
        console.print(f"[red]Config not found:[/] {config_path}")
        raise typer.Exit(1)

    cfg = load_config(cfg_real)
    out_real = os.path.realpath(output_dir)
    if not is_under_cwd(out_real):
        console.print(
            f"[red]--output must stay under cwd[/] (got {output_dir!r})"
        )
        raise typer.Exit(1)

    cache_key = make_preprocess_cache_key(
        dataset_path=cfg.data.train,
        tokenizer_name=cfg.base,
        max_length=cfg.data.max_length,
        format_name=cfg.data.format,
    )
    target = Path(out_real) / cache_key
    console.print(f"[cyan]Dataset:[/] {cfg.data.train}")
    console.print(f"[cyan]Tokenizer:[/] {cfg.base}")
    console.print(f"[cyan]max_length:[/] {cfg.data.max_length}")
    console.print(f"[cyan]Cache key:[/] {cache_key}")
    console.print(f"[cyan]Target:[/] {target}")

    if target.exists() and not yes:
        console.print(
            f"[yellow]Target already exists:[/] {target}\n"
            "[dim]Re-run with --yes to overwrite.[/]"
        )
        raise typer.Exit(0)

    # Lazy imports — defer heavy deps until the actual tokenize loop runs.
    try:
        from transformers import AutoTokenizer
    except ImportError:
        console.print(
            "[red]transformers not installed.[/] Run: pip install transformers"
        )
        raise typer.Exit(1) from None
    try:
        from datasets import Dataset
    except ImportError:
        console.print(
            "[red]datasets not installed.[/] Run: pip install datasets"
        )
        raise typer.Exit(1) from None
    from soup_cli.data.loader import load_dataset

    # DoS cap (10M rows).
    max_preprocess_rows = 10_000_000

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base, trust_remote_code=False
    )

    try:
        dataset = load_dataset(cfg.data)
    except (TypeError, ValueError, FileNotFoundError) as exc:
        console.print(f"[red]Failed to load dataset:[/] {exc}")
        raise typer.Exit(1) from exc

    raw_rows = dataset.get("train", []) if isinstance(dataset, dict) else []
    max_length = int(cfg.data.max_length)
    is_pretrain = cfg.task == "pretrain"

    rendered_rows: list[dict] = []
    for idx, row in enumerate(raw_rows):
        if idx >= max_preprocess_rows:
            console.print(
                f"[yellow]Reached row cap {max_preprocess_rows}, truncating.[/]"
            )
            break
        if is_pretrain:
            # Pretrain uses raw text.
            text = ""
            if isinstance(row, dict):
                text = row.get("text") or row.get("content") or ""
            if not isinstance(text, str):
                continue
            if not text:
                continue
        else:
            messages = row.get("messages") if isinstance(row, dict) else None
            if not messages or not getattr(tokenizer, "chat_template", None):
                continue
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            except Exception:  # noqa: BLE001 — tokenizer template errors vary
                continue
            if not isinstance(text, str) or not text:
                continue
        try:
            tokens = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding=False,
                return_attention_mask=True,
            )
        except Exception:  # noqa: BLE001 — tokenizer errors vary
            continue
        rendered_rows.append(
            {
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens.get(
                    "attention_mask", [1] * len(tokens["input_ids"])
                ),
            }
        )

    if not rendered_rows:
        console.print(
            "[red]No rows tokenized.[/] Check data.format and tokenizer "
            "chat_template."
        )
        raise typer.Exit(1)

    ds = Dataset.from_list(rendered_rows)
    target.parent.mkdir(parents=True, exist_ok=True)
    # v0.53.7 M-H: atomic write via temp dir + os.replace so a partial
    # write (e.g. SIGKILL mid-save) does not leave a corrupt dataset at
    # ``target``. save_to_disk overwrites silently; we route through a
    # sibling temp dir to defend against that too.
    import shutil as _shutil

    tmp_dir = target.parent / (".tmp_" + target.name)
    if tmp_dir.exists():
        _shutil.rmtree(tmp_dir, ignore_errors=True)
    try:
        ds.save_to_disk(str(tmp_dir))
        if target.exists():
            _shutil.rmtree(target, ignore_errors=True)
        os.replace(str(tmp_dir), str(target))
    except Exception:
        _shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    metadata = {
        "cache_key": cache_key,
        "row_count": len(rendered_rows),
        "tokenizer_name": cfg.base,
        "max_length": max_length,
        "format": cfg.data.format,
        "task": cfg.task,
        "soup_version": "0.53.7",
    }
    metadata_path = target / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        _json.dump(metadata, f, indent=2)

    console.print(
        f"[green]Wrote {len(rendered_rows)} tokenized rows to[/] {target}"
    )
    console.print(
        f"[dim]Set data.tokenized_path={target} + data.format=pre_tokenized "
        "in your soup.yaml to short-circuit tokenization on subsequent runs.[/]"
    )


@app.command(name="ingest")
def ingest_document(
    file: str = typer.Argument(..., help="PDF / DOCX / MD / TXT file"),
    output: str = typer.Option(
        "./ingested.jsonl", "--output", "-o", help="Output JSONL path"
    ),
) -> None:
    """Ingest a document into JSONL with one row per page / heading.

    Lazy-imports the per-format extractor so missing optional deps don't
    break the rest of `soup data --help`. Supported formats:
    - .pdf  → pypdf
    - .docx → python-docx
    - .md   → markdown
    - .txt  → built-in
    """
    import stat

    from soup_cli.utils.data_pipeline import detect_ingest_format
    from soup_cli.utils.paths import is_under_cwd

    # Reject symlinks at the input path (TOCTOU defence — mirrors v0.33.0 #22
    # prune_checkpoints policy).
    try:
        lst = os.lstat(file)
    except OSError as exc:
        console.print(f"[red]File not found:[/] {file}")
        raise typer.Exit(1) from exc
    if stat.S_ISLNK(lst.st_mode):
        console.print(
            f"[red]Input file must not be a symlink[/] (got {file!r})"
        )
        raise typer.Exit(1)

    in_real = os.path.realpath(file)
    if not is_under_cwd(in_real):
        console.print(f"[red]Input must stay under cwd[/] (got {file!r})")
        raise typer.Exit(1)
    if not Path(in_real).is_file():
        console.print(f"[red]File not found:[/] {file}")
        raise typer.Exit(1)

    out_real = os.path.realpath(output)
    if not is_under_cwd(out_real):
        console.print("[red]--output must stay under cwd[/]")
        raise typer.Exit(1)

    try:
        kind = detect_ingest_format(file)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc

    rows: list[dict] = []
    if kind == "txt":
        with open(in_real, encoding="utf-8") as f:
            text = f.read()
        rows.append({"text": text, "source": Path(in_real).name})
    elif kind == "markdown":
        # v0.53.7 #88 — heading-aware split. One row per ATX heading section
        # (``^#{1,6}\s``); preamble before the first heading is emitted as a
        # row with ``section=None`` + ``level=None``. Closes the v0.42.0 #88
        # known limitation.
        from soup_cli.utils.data_pipeline import split_markdown_by_headings

        with open(in_real, encoding="utf-8") as f:
            text = f.read()
        sections = split_markdown_by_headings(text)
        if not sections:
            # Empty markdown file — keep parity with v0.42.0 single-row shape.
            rows.append(
                {
                    "text": "",
                    "section": None,
                    "level": None,
                    "source": Path(in_real).name,
                }
            )
        else:
            for sec in sections:
                rows.append(
                    {
                        "text": sec["text"],
                        "section": sec["section"],
                        "level": sec["level"],
                        "source": Path(in_real).name,
                    }
                )
    elif kind == "pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            console.print(
                "[red]pypdf not installed.[/] Run: pip install pypdf"
            )
            raise typer.Exit(1) from None
        reader = PdfReader(in_real)
        for index, page in enumerate(reader.pages):
            rows.append({
                "text": page.extract_text() or "",
                "source": Path(in_real).name,
                "page": index,
            })
    elif kind == "docx":
        try:
            from docx import Document
        except ImportError:
            console.print(
                "[red]python-docx not installed.[/] Run: pip install python-docx"
            )
            raise typer.Exit(1) from None
        doc = Document(in_real)
        for index, para in enumerate(doc.paragraphs):
            text = para.text.strip()
            if text:
                rows.append({
                    "text": text,
                    "source": Path(in_real).name,
                    "para": index,
                })
    else:
        console.print(f"[red]Unhandled ingest kind:[/] {kind}")
        raise typer.Exit(1)

    Path(out_real).parent.mkdir(parents=True, exist_ok=True)
    with open(out_real, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    console.print(f"[green]Wrote {len(rows)} rows to[/] {output}")


@app.command(name="demo")
def demo_bundle(
    name: str = typer.Argument(
        None,
        help="Bundle name (alpaca_demo / sharegpt_demo / dpo_demo / grpo_demo). "
        "Omit to list available bundles.",
    ),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Destination JSONL path (defaults to ./<name>.jsonl).",
    ),
) -> None:
    """List or copy a bundled demo dataset (v0.43.0).

    `soup data demo` lists available bundles. `soup data demo <name>` copies
    the JSONL fixture to the current directory. Bundles are version-locked
    JSONL fixtures shipped under `examples/data/`.
    """
    from rich.markup import escape as _esc
    from rich.table import Table

    from soup_cli.utils.demo_bundles import (
        copy_bundle_to,
        get_bundle,
        list_bundles,
    )

    if name is None:
        table = Table(title="Available demo bundles", show_lines=False)
        table.add_column("Name", style="cyan")
        table.add_column("Format", style="green")
        table.add_column("Description")
        for bundle in list_bundles():
            table.add_row(bundle.name, bundle.format, bundle.description)
        console.print(table)
        console.print(
            "[dim]Run: soup data demo <name> --output <path>[/]"
        )
        return

    try:
        bundle = get_bundle(name)
    except ValueError as exc:
        console.print(f"[red]{_esc(str(exc))}[/]")
        raise typer.Exit(2) from exc

    target = output or f"./{bundle.name}.jsonl"
    try:
        written = copy_bundle_to(bundle.name, target)
    except FileExistsError as exc:
        console.print(f"[red]{_esc(str(exc))}[/]")
        raise typer.Exit(1) from exc
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{_esc(str(exc))}[/]")
        raise typer.Exit(1) from exc
    console.print(
        f"[green]Copied bundle '{bundle.name}' to[/] {_esc(written)}"
    )


@app.command(name="recipe")
def recipe(
    path: str = typer.Argument(..., help="Path to recipe.yaml under cwd"),
    execute: bool = typer.Option(
        False,
        "--execute",
        help=(
            "Run the validated DAG end-to-end (v0.53.7 #106 — live per-node "
            "execution: seed / llm_text / code / judge / validator / sampler)."
        ),
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output dir for sampler node (required with --execute).",
    ),
) -> None:
    """v0.45.0 Part E — Validate a Data Recipe DAG (live runner deferred)."""
    from rich.markup import escape as _escape

    from soup_cli.utils.recipe_dag import load_recipe_yaml

    try:
        dag = load_recipe_yaml(path)
    except FileNotFoundError as exc:
        console.print(f"[red]Recipe not found: {_escape(str(exc))}[/]")
        raise typer.Exit(1) from exc
    except (TypeError, ValueError) as exc:
        console.print(f"[red]Invalid recipe: {_escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    console.print(
        f"[green]Recipe validated.[/] {len(dag.nodes)} node(s), "
        f"{len(dag.edges)} edge(s)."
    )
    console.print(
        "Topological order: "
        + ", ".join(_escape(name) for name in dag.topo_order)
    )

    if execute:
        # `is None` guard — empty-string `--output ""` is a distinct
        # operator error that should NOT be silently mapped to "missing"
        # (matches v0.40.6 project policy on `is None` over falsy).
        if output is None:
            console.print(
                "[red]--execute requires --output <dir>[/]"
            )
            raise typer.Exit(2)
        # Defence-in-depth: enforce cwd containment at the CLI boundary
        # BEFORE handing off to run_recipe. Today run_recipe is a stub,
        # so this only protects against future v0.53.7 live-runner bugs —
        # the docstring contract on run_recipe.output_dir says
        # cwd-contained, and we never want the live runner to be the
        # first/only enforcement point.
        from soup_cli.utils.paths import is_under_cwd
        from soup_cli.utils.recipe_run import run_recipe

        if not output or not is_under_cwd(output):
            console.print(
                "[red]--output must be a non-empty path under the current directory[/]"
            )
            raise typer.Exit(2)

        # v0.53.7 #106: live runner. Per-node handlers + checkpoint + resume
        # land here; ``NotImplementedError`` is no longer raised on the live
        # surface but we keep the catch for defence-in-depth (a future schema
        # change might re-introduce a stub for an unknown node kind).
        try:
            result = run_recipe(dag, output_dir=output)
        except NotImplementedError as exc:
            console.print(f"[yellow]{_escape(str(exc))}[/]")
            raise typer.Exit(2) from exc
        except (TypeError, ValueError) as exc:
            console.print(f"[red]Recipe failed: {_escape(str(exc))}[/]")
            raise typer.Exit(1) from exc
        console.print(
            f"[green]Recipe executed.[/] "
            f"{len(result.get('completed_nodes', ()))} node(s) completed."
        )
        return

    console.print(
        "[dim]Re-run with --execute --output <dir> to run the DAG.[/]"
    )


# v0.69.0 Part C — Magpie synthetic data generator.
@app.command(name="gen-magpie")
def gen_magpie(
    base: str = typer.Option(..., "--base", help="Aligned chat-tuned base model id"),
    provider: str = typer.Option(..., "--provider", help="ollama | anthropic | vllm"),
    target: int = typer.Option(..., "--target", help="Target row count [1, 1_000_000]"),
    quality_filter: bool = typer.Option(
        True, "--quality-filter/--no-quality-filter", help="Apply v0.47 quality filter"
    ),
    plan_only: bool = typer.Option(
        False, "--plan-only", help="Validate + print plan; do not generate."
    ),
) -> None:
    """Magpie synthetic data generator (v0.69.0 Part C).

    Feeds the chat-template prefix only to ``--base`` and harvests user-side
    turns via ``--provider``. The live generator is deferred to v0.69.1; today
    only ``--plan-only`` produces a meaningful result.
    """
    from rich.markup import escape as _escape
    from rich.panel import Panel

    from soup_cli.utils.magpie import build_magpie_config, run_magpie

    try:
        cfg = build_magpie_config(
            base=base,
            provider=provider,
            target=target,
            quality_filter=quality_filter,
        )
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{_escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    console.print(
        Panel(
            f"Base model:     [bold]{_escape(cfg.base_model)}[/]\n"
            f"Provider:       [bold]{_escape(cfg.provider)}[/]\n"
            f"Target rows:    [bold]{cfg.target_rows}[/]\n"
            f"Quality filter: [bold]{cfg.quality_filter}[/]",
            title="soup data gen-magpie — plan",
        )
    )

    if plan_only:
        return

    try:
        run_magpie(cfg)
    except NotImplementedError as exc:
        console.print(
            Panel(
                f"[yellow]{_escape(str(exc))}[/]",
                title="Live magpie generator deferred",
            )
        )
        raise typer.Exit(3) from exc


# v0.69.0 Part D — Persona-Hub diversity sampler.
@app.command(name="persona-mix")
def persona_mix(
    prompts: str = typer.Option(..., "--prompts", help="JSONL file with one prompt per row"),
    n: int = typer.Option(..., "--n", help="Number of rows to emit [1, 1_000_000]"),
    output: str = typer.Option(..., "--output", help="Output JSONL path"),
    personas: str = typer.Option(
        "", "--personas", help="Optional JSONL of personas (defaults to bundled set)"
    ),
    styles: str = typer.Option(
        "", "--styles", help="Optional JSONL of styles (defaults to bundled set)"
    ),
    seed: int = typer.Option(0, "--seed", help="Deterministic seed"),
) -> None:
    """Sample a prompt × persona × style matrix (v0.69.0 Part D).

    Reads ``--prompts`` (JSONL with ``prompt`` field per row); writes
    ``--output`` JSONL with ``{prompt, persona, style}`` rows. When
    ``--personas`` / ``--styles`` are omitted the bundled diversity set is used.
    """
    import os
    import tempfile

    from rich.markup import escape as _escape
    from rich.panel import Panel

    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink
    from soup_cli.utils.persona_hub import (
        list_bundled_personas,
        list_bundled_styles,
        sample_persona_matrix,
    )

    max_jsonl_bytes = 100 * 1024 * 1024  # 100 MiB cap on field inputs
    max_values = 100_000  # cap matches persona_hub._MAX_LIST_LEN

    def _load_jsonl_field(path: str, field: str) -> list:
        # Delegate to centralised TOCTOU helper (v0.59.0).
        enforce_under_cwd_and_no_symlink(path, f"--{field}s path")
        if not os.path.lexists(path):
            raise FileNotFoundError(path)
        real = os.path.realpath(path)
        if not os.path.isfile(real):
            raise FileNotFoundError(real)
        if os.path.getsize(real) > max_jsonl_bytes:
            raise ValueError(
                f"--{field}s file exceeds {max_jsonl_bytes} bytes"
            )
        values: list = []
        with open(real, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                if len(values) >= max_values:
                    raise ValueError(
                        f"--{field}s file exceeds {max_values} entries"
                    )
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                val = row.get(field) if isinstance(row, dict) else None
                if isinstance(val, str) and val:
                    values.append(val)
        return values

    try:
        prompt_list = _load_jsonl_field(prompts, "prompt")
        if not prompt_list:
            raise ValueError("prompts file produced no rows with a 'prompt' field")
        persona_list = (
            _load_jsonl_field(personas, "persona")
            if personas
            else list(list_bundled_personas())
        )
        if not persona_list:
            raise ValueError("personas file produced no rows with a 'persona' field")
        style_list = (
            _load_jsonl_field(styles, "style")
            if styles
            else list(list_bundled_styles())
        )
        if not style_list:
            raise ValueError("styles file produced no rows with a 'style' field")

        # Output containment + TOCTOU symlink rejection at the write target.
        enforce_under_cwd_and_no_symlink(output, "--output")
        rows = sample_persona_matrix(
            prompts=prompt_list,
            personas=persona_list,
            styles=style_list,
            n=n,
            seed=seed,
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        console.print(f"[red]{_escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    # Atomic write via tempfile + os.replace.
    parent = os.path.dirname(os.path.realpath(output)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".soup.", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_path, output)
    except OSError:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    console.print(
        Panel(
            f"Rows written: [bold]{len(rows)}[/]\n"
            f"Output:       [bold]{_escape(output)}[/]",
            title="soup data persona-mix",
        )
    )


# v0.69.0 Part E — Brain-rot detector.
@app.command(name="brain-rot")
def brain_rot_cmd(
    data: str = typer.Argument(..., help="Path to JSONL dataset"),
    strict: bool = typer.Option(
        False, "--strict", help="Exit 3 on MAJOR verdict (CI gate mode)"
    ),
    max_major_fraction: float = typer.Option(
        0.25, "--max-major-fraction", help="Strict-mode MAJOR-row fraction cap [0, 1]"
    ),
) -> None:
    """Score a dataset for brain-rot per arXiv 2510.13928 (v0.69.0 Part E).

    Reports a per-row OK/MINOR/MAJOR verdict + an aggregate verdict. With
    ``--strict`` the command exits 3 when the MAJOR fraction exceeds
    ``--max-major-fraction`` (default 25%), so CI pipelines can refuse
    training on excessive slop.
    """
    import math as _math
    import os

    from rich.markup import escape as _escape
    from rich.table import Table

    from soup_cli.utils.brain_rot import (
        refuse_if_rotten,
        score_dataset_brain_rot,
    )
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    max_brain_rot_bytes = 1_073_741_824  # 1 GiB
    max_brain_rot_rows = 1_000_000

    # Validate --max-major-fraction at the CLI boundary so a flag of NaN /
    # inf / bool / out-of-range is rejected BEFORE the heavy scoring pass
    # (security review M2).
    if isinstance(max_major_fraction, bool):
        console.print("[red]--max-major-fraction must be a number, not bool[/]")
        raise typer.Exit(2)
    if not isinstance(max_major_fraction, (int, float)):
        console.print("[red]--max-major-fraction must be a number[/]")
        raise typer.Exit(2)
    if not _math.isfinite(float(max_major_fraction)):
        console.print("[red]--max-major-fraction must be finite[/]")
        raise typer.Exit(2)
    if not (0.0 <= float(max_major_fraction) <= 1.0):
        console.print("[red]--max-major-fraction must be in [0.0, 1.0][/]")
        raise typer.Exit(2)

    if not isinstance(data, str) or not data:
        console.print("[red]data path must be a non-empty string[/]")
        raise typer.Exit(2)
    try:
        enforce_under_cwd_and_no_symlink(data, "data path")
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{_escape(str(exc))}[/]")
        raise typer.Exit(2) from exc
    if not os.path.lexists(data):
        console.print(f"[red]data file not found: {_escape(data)}[/]")
        raise typer.Exit(2)
    real = os.path.realpath(data)
    if not os.path.isfile(real):
        console.print(f"[red]data file not found: {_escape(real)}[/]")
        raise typer.Exit(2)
    if os.path.getsize(real) > max_brain_rot_bytes:
        console.print(
            f"[red]data file exceeds {max_brain_rot_bytes} bytes[/]"
        )
        raise typer.Exit(2)

    rows: list = []
    with open(real, "r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            if line_no > max_brain_rot_rows:
                console.print(
                    f"[red]data file exceeds {max_brain_rot_rows} rows[/]"
                )
                raise typer.Exit(2)
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)

    report = score_dataset_brain_rot(rows)

    table = Table(title="Brain-rot report")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Rows scored", str(report.num_rows))
    table.add_row("Mean score", f"{report.mean_score:.3f}")
    table.add_row("Verdict OK", str(report.num_ok))
    table.add_row("Verdict MINOR", str(report.num_minor))
    table.add_row("Verdict MAJOR", str(report.num_major))
    table.add_row("Overall", report.overall_verdict)
    console.print(table)

    if strict:
        try:
            refuse_if_rotten(rows, max_major_fraction=max_major_fraction)
        except (TypeError, ValueError) as exc:
            console.print(f"[red]{_escape(str(exc))}[/]")
            raise typer.Exit(3) from exc
