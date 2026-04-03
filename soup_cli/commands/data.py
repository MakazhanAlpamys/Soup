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
        None, "--coherence",
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
    if output is None:
        out_path = file_path.parent / f"{file_path.stem}_sampled.jsonl"
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
        f"  Train: {len(train_data)} → [bold]{train_path}[/]"
    )

    if val_data:
        val_path = parent / f"{stem}_val.jsonl"
        _write_jsonl(val_path, val_data)
        output_msg += f"\n  Val:   {len(val_data)} → [bold]{val_path}[/]"

    if test_data:
        test_path = parent / f"{stem}_test.jsonl"
        _write_jsonl(test_path, test_data)
        output_msg += f"\n  Test:  {len(test_data)} → [bold]{test_path}[/]"

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
):
    """Download a HuggingFace dataset and save as JSONL."""
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


@app.command(name="register")
def register_data(
    name: str = typer.Option(..., "--name", "-n", help="Dataset name"),
    path: str = typer.Option(..., "--path", "-p", help="Path to dataset file"),
    fmt: str = typer.Option(
        "auto", "--format", "-f",
        help="Dataset format: alpaca, sharegpt, chatml, dpo, kto, auto",
    ),
):
    """Register a local dataset by name for use in soup.yaml."""
    from soup_cli.utils.registry import register_dataset

    # Path traversal protection
    resolved = Path(path).resolve()
    cwd = Path.cwd().resolve()
    try:
        resolved.relative_to(cwd)
    except ValueError:
        console.print(
            "[red]Dataset path must be under the current working directory.[/]"
        )
        raise typer.Exit(1)

    registry_path = _get_registry_path()

    try:
        register_dataset(name, str(resolved), fmt, registry_path=registry_path)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1)

    console.print(
        f"[green]Registered dataset '[bold]{name}[/bold]'[/]\n"
        f"  Path: {path}\n"
        f"  Format: {fmt}"
    )


@app.command(name="unregister")
def unregister_data(
    name: str = typer.Option(..., "--name", "-n", help="Dataset name to remove"),
):
    """Remove a dataset from the local registry."""
    from soup_cli.utils.registry import unregister_dataset

    registry_path = _get_registry_path()
    removed = unregister_dataset(name, registry_path=registry_path)

    if removed:
        console.print(f"[green]Removed dataset '{name}' from registry.[/]")
    else:
        console.print(f"[red]Dataset '{name}' not found in registry.[/]")
        raise typer.Exit(1)


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
