"""Data loading from local files and HuggingFace."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from soup_cli.config.schema import DataConfig
from soup_cli.data.formats import (
    detect_format,
    format_to_messages,
    is_audio_format,
    is_vision_format,
)
from soup_cli.utils.paths import is_under_cwd

console = Console()

# File extensions we support
SUPPORTED_EXTENSIONS = {".jsonl", ".json", ".csv", ".parquet", ".txt"}


def load_raw_data(path: Path) -> list[dict]:
    """Load raw data from a file into list of dicts."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {ext}. Supported: {SUPPORTED_EXTENSIONS}")

    if ext == ".jsonl":
        return _load_jsonl(path)
    elif ext == ".json":
        return _load_json(path)
    elif ext == ".csv":
        return _load_csv(path)
    elif ext == ".parquet":
        return _load_parquet(path)
    elif ext == ".txt":
        return _load_txt(path)

    raise ValueError(f"Unsupported format: {ext}")


def _load_jsonl(path: Path) -> list[dict]:
    data = []
    # v0.40.1 Part E — auto-strip UTF-8 BOM (Windows users overwhelmingly
    # write JSONL via PowerShell `Out-File -Encoding utf8` which adds BOM).
    # The ``utf-8-sig`` codec consumes the BOM transparently if present.
    with open(path, encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                console.print(f"[yellow]Warning: invalid JSON on line {i + 1}: {e}[/]")
    return data


def _load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    raise ValueError("JSON file must contain a list of objects")


def _load_csv(path: Path) -> list[dict]:
    import csv

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _load_parquet(path: Path) -> list[dict]:
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Install pandas to read parquet files: pip install pandas pyarrow")
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")


def _load_txt(path: Path) -> list[dict]:
    """Load a plain text file as a list of {text: ...} dicts.

    Each non-empty line is treated as a separate document.
    Empty lines are skipped.
    """
    file_size = path.stat().st_size
    if file_size > 500 * 1024 * 1024:  # 500 MB
        console.print(
            f"[yellow]Warning: large text file ({file_size / 1024 / 1024:.0f} MB). "
            f"Consider splitting into smaller files or using JSONL format.[/]"
        )
    with open(path, encoding="utf-8") as f:
        content = f.read()

    # Split by double newline (paragraph/document separator) or treat each line as a doc
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    if not lines:
        console.print(f"[yellow]Warning: empty text file: {path}[/]")
        return []

    return [{"text": line} for line in lines]


def _load_replay_rows(data_config: DataConfig) -> list[dict]:
    """Load + normalize the replay file with its OWN format detection.

    The old dataset may be alpaca while the new one is sharegpt, so the
    replay file cannot inherit ``data_config.format``.

    It also gets its OWN media-containment pass. ``load_dataset`` runs
    :func:`_validate_vision_images` / :func:`_validate_audio_files` on the
    primary dataset, and the replay file is loaded here rather than there —
    so without this, a llava-shaped replay row's ``image`` value survives
    ``format_to_messages`` untouched and a traversal path would reach
    ``PIL.Image.open`` in the trainer. Media resolve against the REPLAY
    file's own directory unless an explicit dir is configured: the old
    dataset's images live with the old dataset.
    """
    replay_path = Path(data_config.replay)
    if not is_under_cwd(replay_path):
        raise ValueError(
            f"data.replay path is outside the working directory: {replay_path}"
        )
    if not replay_path.exists():
        raise FileNotFoundError(
            f"data.replay file not found: {replay_path}"
        )
    raw = load_raw_data(replay_path)
    fmt = detect_format(raw)
    rows = [format_to_messages(row, fmt) for row in raw]
    rows = [row for row in rows if row is not None]

    if is_vision_format(fmt):
        image_dir = (
            Path(data_config.image_dir)
            if data_config.image_dir
            else replay_path.parent
        )
        rows = _validate_vision_images(rows, image_dir)
    if is_audio_format(fmt):
        audio_dir = (
            Path(data_config.audio_dir)
            if data_config.audio_dir
            else replay_path.parent
        )
        rows = _validate_audio_files(rows, audio_dir)
    return rows


def _finalize(
    formatted: list[dict],
    data_config: DataConfig,
    *,
    val: list[dict] | None = None,
) -> dict:
    """Split train/val, then mix replay into train ONLY.

    Single exit point for every load path (local / remote / HF) so replay
    behaviour cannot drift between them — `soup sweep` and
    `soup train --dry-run` go through the same seam.

    Replay is mixed AFTER the split so val stays pure new-task: it is the
    yardstick for the task being learned. Old-task retention is measured
    externally with `soup eval custom` / `soup ship`, which adds no new
    eval machinery here.
    """
    if val is not None:
        result = {"train": formatted, "val": val}
    elif data_config.val_split > 0:
        split_idx = int(len(formatted) * (1 - data_config.val_split))
        result = {"train": formatted[:split_idx], "val": formatted[split_idx:]}
    else:
        result = {"train": formatted}

    if getattr(data_config, "replay", None):
        from soup_cli.utils.rehearsal import mix_replay

        replay_rows = _load_replay_rows(data_config)
        mixed, report = mix_replay(
            result["train"],
            replay_rows,
            ratio=data_config.replay_ratio,
            seed=data_config.replay_seed,
        )
        result["train"] = mixed
        console.print(
            f"[dim]Replay: +{report.n_replay} old rows interleaved "
            f"({report.ratio_actual * 100:.1f}% of {report.n_final})[/]"
        )
        if report.shortfall:
            console.print(
                f"[yellow]Replay pool too small: wanted {report.requested}, "
                f"used {report.n_replay} (short {report.shortfall}). Rows are "
                "NOT repeated.[/]"
            )
    return result


def load_dataset(data_config: DataConfig) -> dict:
    """Load dataset for training. Returns dict with 'train' and optionally 'val' keys.

    Supports:
    - Local files (.jsonl, .json, .csv, .parquet, .txt)
    - HuggingFace dataset names (auto-detected if no file extension)
    - Remote fsspec URIs (s3://, gs://, gcs://, az://, abfs://, abfss://, oci://) — v0.53.8 #85
    - A list of >= 2 local files combined via data.interleave — #443. Schema
      validation (SoupConfig._validate_interleave_compat) already guarantees
      data.interleave is set and every entry is a local file path whenever
      data.train is a list, so this branch never has to re-check that.
    """
    train_path = data_config.train

    if isinstance(train_path, list):
        return _load_interleaved_local_datasets(train_path, data_config)

    # v0.53.8 #85 — fsspec live remote loader. Schema accepts these URIs
    # since v0.42.0; live loader lands here. Lazy-imports fsspec + the
    # backend driver (s3fs / gcsfs / adlfs / ocifs) and surfaces a
    # friendly Rich panel naming the pip install when the driver is
    # missing.
    if _looks_like_remote_uri(train_path):
        return _load_remote_dataset(train_path, data_config)

    # Check if it's a HuggingFace dataset
    if not Path(train_path).suffix:
        return _load_hf_dataset(train_path, data_config)

    # Local file
    path = Path(train_path)
    raw_data = load_raw_data(path)

    # Detect or use specified format
    fmt = data_config.format
    if fmt == "auto":
        fmt = detect_format(raw_data)
        console.print(f"[dim]Auto-detected format: {fmt}[/]")

    # Convert to standard message format
    formatted = [format_to_messages(row, fmt) for row in raw_data]
    formatted = [r for r in formatted if r is not None]  # filter failed rows

    # Validate image paths for vision formats
    if is_vision_format(fmt):
        image_dir = Path(data_config.image_dir) if data_config.image_dir else path.parent
        formatted = _validate_vision_images(formatted, image_dir)

    # Validate audio paths for audio formats
    if is_audio_format(fmt):
        audio_dir = Path(data_config.audio_dir) if data_config.audio_dir else path.parent
        formatted = _validate_audio_files(formatted, audio_dir)

    # Split into train/val, then mix replay into train (v0.71.36).
    return _finalize(formatted, data_config)


def _load_one_local_dataset(train_path: str, data_config: DataConfig) -> list[dict]:
    """Load + format one local file — the same per-file pipeline the
    single-path branch of load_dataset() runs above, factored out so
    #443's interleave path (below) reuses it rather than re-deriving it.
    The single-path branch itself is left inline and untouched, so its
    output stays byte-identical by construction rather than by
    refactor-equivalence.
    """
    path = Path(train_path)
    raw_data = load_raw_data(path)

    fmt = data_config.format
    if fmt == "auto":
        fmt = detect_format(raw_data)
        console.print(f"[dim]Auto-detected format ({train_path}): {fmt}[/]")

    formatted = [format_to_messages(row, fmt) for row in raw_data]
    formatted = [r for r in formatted if r is not None]

    if is_vision_format(fmt):
        image_dir = Path(data_config.image_dir) if data_config.image_dir else path.parent
        formatted = _validate_vision_images(formatted, image_dir)
    if is_audio_format(fmt):
        audio_dir = Path(data_config.audio_dir) if data_config.audio_dir else path.parent
        formatted = _validate_audio_files(formatted, audio_dir)
    return formatted


def _cycle_to(rows: list[dict], target: int) -> list[dict]:
    """Deterministically repeat/truncate ``rows`` to exactly ``target`` rows
    via round-robin cycling. No RNG, no new seed knob — #443's scope does
    not authorize one, and the downstream HF Trainer's default sampler
    already shuffles every epoch, so row ORDER out of this function is not
    load-bearing; only per-source row COUNT is.
    """
    if target <= len(rows):
        return rows[:target]
    out = list(rows)
    i = 0
    while len(out) < target:
        out.append(rows[i % len(rows)])
        i += 1
    return out


def _apportion(probs: tuple[float, ...], total: int) -> list[int]:
    """Largest-remainder apportionment: integer per-dataset quotas summing
    to exactly ``total``, as close as possible to ``probs[i] * total``.
    Deterministic; ties break by dataset index.
    """
    raw = [p * total for p in probs]
    base = [int(x) for x in raw]
    remainder = total - sum(base)
    order = sorted(range(len(probs)), key=lambda i: (raw[i] - base[i]), reverse=True)
    for i in order[:remainder]:
        base[i] += 1
    return base


def _combine_interleaved(per_dataset_rows: list[list[dict]], spec) -> list[dict]:
    """Combine per-dataset row lists per an InterleaveSpec (#443).

    ``concat`` / ``under`` / ``over`` / ``probs`` control per-source row
    COUNT, not order — see _cycle_to's docstring for why order doesn't
    matter here.
    """
    sizes = [len(rows) for rows in per_dataset_rows]
    if any(n == 0 for n in sizes):
        raise ValueError(
            "data.interleave: every dataset in data.train must have "
            ">= 1 row after formatting"
        )
    if spec.strategy == "concat":
        combined: list[dict] = []
        for rows in per_dataset_rows:
            combined.extend(rows)
        return combined
    if spec.strategy == "under":
        target = min(sizes)
        combined = []
        for rows in per_dataset_rows:
            combined.extend(rows[:target])
        return combined
    if spec.strategy == "over":
        target = max(sizes)
        combined = []
        for rows in per_dataset_rows:
            combined.extend(_cycle_to(rows, target))
        return combined
    if spec.strategy == "probs":
        quotas = _apportion(spec.probs, sum(sizes))
        combined = []
        for rows, quota in zip(per_dataset_rows, quotas):
            combined.extend(_cycle_to(rows, quota))
        return combined
    raise AssertionError(f"unreachable interleave strategy {spec.strategy!r}")


def _load_interleaved_local_datasets(train_paths: list[str], data_config: DataConfig) -> dict:
    """#443 — load + combine every local dataset in a list-shaped
    data.train per data.interleave, then hand the combined rows to the
    existing, unmodified _finalize() — same seam every other loader uses.
    """
    from soup_cli.utils.data_pipeline import parse_interleave

    spec = parse_interleave(data_config.interleave, num_datasets=len(train_paths))
    if spec is None:
        # Defence-in-depth only: SoupConfig._validate_interleave_compat
        # already requires data.interleave whenever data.train is a list,
        # so a caller only reaches here via a hand-built DataConfig that
        # skipped SoupConfig validation.
        raise ValueError("data.train is a list but data.interleave is not set")

    per_dataset_rows = [_load_one_local_dataset(p, data_config) for p in train_paths]
    combined = _combine_interleaved(per_dataset_rows, spec)
    console.print(
        f"[dim]Interleaved {len(train_paths)} datasets "
        f"(strategy={spec.strategy}) -> {len(combined)} rows[/]"
    )
    return _finalize(combined, data_config)


def _validate_vision_images(data: list[dict], image_dir: Path) -> list[dict]:
    """Validate and resolve image paths in vision dataset rows.

    Each row must have an 'image' key with a filename or path. Resolves
    relative paths against image_dir and rejects path traversal — a crafted
    llava/sharegpt4v row like ``{"image": "/etc/passwd"}`` must not be handed
    to ``PIL.Image.open``. Mirrors :func:`_validate_audio_files` (the sibling
    audio path got this fix in v0.71.32; the vision path was missed).
    """
    from soup_cli.utils.paths import is_under

    valid = []
    missing = 0
    traversal = 0
    for row in data:
        if "image" not in row or not row["image"]:
            missing += 1
            continue
        image_path = Path(row["image"])
        if not image_path.is_absolute():
            image_path = image_dir / image_path
        # Path traversal protection: resolved path must stay under image_dir.
        # realpath + commonpath (is_under) — Path.is_relative_to() breaks on
        # Windows 8.3 short names.
        if not is_under(image_path, image_dir):
            traversal += 1
            continue
        valid.append({**row, "image": str(image_path.resolve())})

    if missing > 0:
        console.print(f"[yellow]Warning: {missing} rows skipped (missing image path)[/]")
    if traversal > 0:
        console.print(
            f"[yellow]Warning: {traversal} rows skipped "
            f"(image path outside {image_dir})[/]"
        )
    return valid


def _validate_audio_files(data: list[dict], audio_dir: Path) -> list[dict]:
    """Validate and resolve audio file paths in audio dataset rows.

    Each row must have an 'audio' key with a filename or path.
    Resolves relative paths against audio_dir. Rejects path traversal.
    """
    valid = []
    from soup_cli.utils.paths import is_under

    missing = 0
    traversal = 0
    for row in data:
        if "audio" not in row or not row["audio"]:
            missing += 1
            continue
        audio_path = Path(row["audio"])
        if not audio_path.is_absolute():
            audio_path = audio_dir / audio_path
        # Path traversal protection: resolved path must stay under audio_dir.
        # realpath + commonpath (is_under) — Path.is_relative_to() breaks on
        # Windows 8.3 short names.
        resolved = audio_path.resolve()
        if not is_under(audio_path, audio_dir):
            traversal += 1
            continue
        valid.append({**row, "audio": str(resolved)})

    if missing > 0:
        console.print(f"[yellow]Warning: {missing} rows skipped (missing audio path)[/]")
    if traversal > 0:
        console.print(
            f"[red]Warning: {traversal} rows skipped (audio path traversal blocked)[/]"
        )
    return valid


def _looks_like_remote_uri(value: str) -> bool:
    """Quick sniff for the fsspec scheme allowlist (v0.42.0 Part B)."""
    if not isinstance(value, str) or "://" not in value:
        return False
    from soup_cli.utils.data_pipeline import is_remote_uri

    return is_remote_uri(value)


def _load_remote_dataset(train_path: str, data_config: DataConfig) -> dict:
    """Load JSONL from a remote fsspec URI (s3 / gs / az / oci / etc.).

    Validates the URI via the v0.42.0 ``validate_remote_uri`` allowlist
    (bucket regex, no userinfo/query/fragment) BEFORE opening any
    connection — defends against URL injection into the fsspec backend.

    Streaming knobs (``data_config.streaming`` + ``buffer_size`` + ``shards``)
    are honoured via :func:`datasets.load_dataset` when present; otherwise
    the file is streamed as JSONL through :func:`fsspec.open`.
    """
    from soup_cli.utils.data_pipeline import (
        required_remote_package,
        validate_remote_uri,
    )

    canonical = validate_remote_uri(train_path)
    scheme = canonical.split("://", 1)[0]

    try:
        import fsspec  # type: ignore[import-not-found]
    except ImportError:
        from rich.panel import Panel

        pkg = required_remote_package(scheme) or scheme
        console.print(
            Panel(
                f"[bold yellow]Missing dependency:[/] reading from "
                f"[bold]{scheme}://[/] requires the [bold]{pkg}[/] package.\n\n"
                f"Install with:\n  [bold]pip install {pkg}[/]",
                title="Remote loader",
                border_style="yellow",
            )
        )
        raise

    # Cap on rows materialised from a remote URI — matches v0.24.0
    # ``soup data download --samples`` ceiling. Defends against OOM when a
    # crafted / oversized bucket object is pointed at via streaming +
    # eager-materialise.
    max_remote_rows = 1_000_000

    # Try the HF datasets streaming path first when the user opted in via
    # ``data.streaming=true`` — gives us free interleaving, shuffling, and
    # caching. Falls back to direct fsspec.open when datasets is missing or
    # rejects the URI.
    if data_config.streaming:
        try:
            from datasets import load_dataset as hf_load
        except ImportError as exc:
            raise ImportError(
                "data.streaming=true requires the 'datasets' package: "
                "pip install datasets"
            ) from exc
        ds = hf_load(
            "json",
            data_files=canonical,
            split="train",
            streaming=True,
        )
        buf = data_config.buffer_size
        if buf:
            ds = ds.shuffle(buffer_size=buf)
        # Eager materialise capped at max_remote_rows — emit a clear advisory
        # if the cap trips.
        raw_data: list[dict] = []
        for i, row in enumerate(ds):
            if i >= max_remote_rows:
                console.print(
                    f"[yellow]Remote dataset truncated at {max_remote_rows:,} "
                    f"rows (use a local split for larger jobs).[/]"
                )
                break
            raw_data.append(row)
    else:
        # Non-streaming: open once, read lines, decode JSON.
        raw_data = []
        with fsspec.open(canonical, mode="rt", encoding="utf-8-sig") as fh:
            for i, raw_line in enumerate(fh):
                if i >= max_remote_rows:
                    console.print(
                        f"[yellow]Remote dataset truncated at "
                        f"{max_remote_rows:,} rows.[/]"
                    )
                    break
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    raw_data.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    console.print(
                        f"[yellow]Warning: invalid JSON on line "
                        f"{i + 1}: {exc}[/]"
                    )

    fmt = data_config.format
    if fmt == "auto":
        fmt = detect_format(raw_data)
        console.print(f"[dim]Auto-detected format: {fmt}[/]")

    formatted = [format_to_messages(row, fmt) for row in raw_data]
    formatted = [r for r in formatted if r is not None]

    return _finalize(formatted, data_config)


def _load_hf_dataset(name: str, data_config: DataConfig) -> dict:
    """Load a dataset from HuggingFace Hub."""
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")

    console.print(f"[dim]Loading from HuggingFace: {name}[/]")
    ds = hf_load(name)

    if "train" not in ds:
        raise ValueError(f"Dataset {name} has no 'train' split")

    raw_data = [dict(row) for row in ds["train"]]
    fmt = data_config.format
    if fmt == "auto":
        fmt = detect_format(raw_data)

    formatted = [format_to_messages(row, fmt) for row in raw_data]
    formatted = [r for r in formatted if r is not None]

    if "validation" in ds:
        # The hub split wins over val_split; pass it through so _finalize
        # does not re-derive one from the train rows.
        val_data = [dict(row) for row in ds["validation"]]
        val_formatted = [format_to_messages(row, fmt) for row in val_data]
        return _finalize(
            formatted,
            data_config,
            val=[r for r in val_formatted if r is not None],
        )

    return _finalize(formatted, data_config)
