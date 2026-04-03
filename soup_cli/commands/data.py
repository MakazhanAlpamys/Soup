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
