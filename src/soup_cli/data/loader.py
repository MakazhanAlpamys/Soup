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

# HF `datasets` builder name per suffix, for _load_interleaved_streaming_
# datasets (#459/#468 MEDIUM fix) — mirrors SUPPORTED_EXTENSIONS /
# load_raw_data's per-suffix dispatch so streaming reads the same file
# type the non-streaming local loader would, instead of always assuming
# JSON. Keys match SUPPORTED_EXTENSIONS exactly (one entry each).
_STREAMING_BUILDERS = {
    ".jsonl": "json",
    ".json": "json",
    ".csv": "csv",
    ".parquet": "parquet",
    ".txt": "text",
}


def _streaming_builder_for(entry: str) -> str:
    """Pick the HF ``datasets`` builder for one streaming interleave entry
    by file suffix, so flipping only ``data.streaming: true`` doesn't
    silently misparse a csv/txt/parquet file as JSON (#459/#468 MEDIUM).

    Call with the CANONICALIZED entry (post ``validate_remote_uri``), not
    the raw one — a query string surviving on the raw string could
    otherwise be mistaken for the suffix.
    """
    suffix = Path(entry).suffix.lower()
    try:
        return _STREAMING_BUILDERS[suffix]
    except KeyError:
        raise ValueError(
            f"data.streaming=true does not support '{suffix or '(no extension)'}' "
            f"data.train entries ({entry!r}) -- supported: "
            f"{sorted(_STREAMING_BUILDERS)}"
        ) from None


# Cap on rows materialised from a remote/streaming source — matches v0.24.0
# ``soup data download --samples`` ceiling. Defends against OOM when a
# crafted / oversized bucket object or hub dataset is pointed at via
# streaming + eager-materialise. Shared by _load_remote_dataset (v0.53.8
# #85) and _load_interleaved_streaming_datasets (#459) so the ceiling can't
# drift between the two streaming call sites.
MAX_REMOTE_ROWS = 1_000_000


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


def _classify_train_entry(value: str) -> str:
    """Classify one data.train list entry for interleave dispatch (#459).

    Returns ``'remote'`` / ``'hub'`` / ``'local'``. Mirrors the identical
    classification inline in SoupConfig._validate_interleave_compat — kept
    as two copies of the same three-line rule (suffix-in-SUPPORTED_EXTENSIONS
    check + "://"-in-entry check) rather than one shared function, since
    schema.py must stay import-light (no torch-adjacent deps) and loader.py
    already owns _looks_like_remote_uri; the rule itself, not the function,
    is the single source of truth, and both sites are covered by
    tests/test_issue459_interleave_streaming_hub.py's schema+loader pairs.

    A suffix must be one of SUPPORTED_EXTENSIONS to count as 'local' (#468
    review fix) — "any non-empty Path.suffix" previously misclassified any
    hub name with a version number (``teknium/OpenHermes-2.5``,
    ``mlfoundations/dclm-baseline-1.0``: Path.suffix is ``.5`` / ``.0``) as
    a local file, even though the hub-loading path itself handles these
    names correctly once actually dispatched there.

    Any ``"://"``-bearing entry classifies 'remote' regardless of scheme
    (#468 review fix) — the scheme allowlist is enforced downstream by
    validate_remote_uri, which refuses a non-allowlisted scheme BY NAME;
    classifying only allowlisted schemes as 'remote' let e.g. an
    ``https://...`` entry with a familiar suffix fall through to 'local'
    and reach hf_load unvalidated instead.
    """
    if _looks_like_remote_uri(value):
        return "remote"
    if Path(value).suffix.lower() in SUPPORTED_EXTENSIONS:
        return "local"
    return "hub"


def load_dataset(data_config: DataConfig) -> dict:
    """Load dataset for training. Returns dict with 'train' and optionally 'val' keys.

    Supports:
    - Local files (.jsonl, .json, .csv, .parquet, .txt)
    - HuggingFace dataset names (auto-detected if no file extension)
    - Remote fsspec URIs (s3://, gs://, gcs://, az://, abfs://, abfss://, oci://) — v0.53.8 #85
    - A list of >= 2 local files / remote URIs combined via data.interleave
      (#443 local files; #459 widened to remote URIs when data.streaming is
      set, and to a separate all-HF-hub-dataset-name list shape). Schema
      validation (SoupConfig._validate_interleave_compat) already
      guarantees data.interleave is set and every entry classifies
      consistently whenever data.train is a list, so this branch only has
      to pick which loader — not re-validate the shape.
    """
    train_path = data_config.train

    if isinstance(train_path, list):
        kinds = {_classify_train_entry(p) for p in train_path}
        if kinds == {"hub"}:
            return _load_interleaved_hub_datasets(train_path, data_config)
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

    #459 — data_config.streaming=true (list may then also contain remote
    URIs; schema already enforced that) delegates to
    _load_interleaved_streaming_datasets instead: HF's own
    interleave_datasets/concatenate_datasets replace this function's
    eager per-file combining. The eager body below is untouched and only
    ever reached when streaming is false, so its output stays
    byte-identical to pre-#459 (pinned by
    test_single_path_train_output_is_byte_identical_to_baseline's sibling
    interleave goldens in tests/test_issue443_interleave_wiring.py).
    """
    from soup_cli.utils.data_pipeline import parse_interleave

    spec = parse_interleave(data_config.interleave, num_datasets=len(train_paths))
    if spec is None:
        # Defence-in-depth only: SoupConfig._validate_interleave_compat
        # already requires data.interleave whenever data.train is a list,
        # so a caller only reaches here via a hand-built DataConfig that
        # skipped SoupConfig validation.
        raise ValueError("data.train is a list but data.interleave is not set")

    if data_config.streaming:
        return _load_interleaved_streaming_datasets(train_paths, data_config, spec)

    per_dataset_rows = [_load_one_local_dataset(p, data_config) for p in train_paths]
    combined = _combine_interleaved(per_dataset_rows, spec)
    console.print(
        f"[dim]Interleaved {len(train_paths)} datasets "
        f"(strategy={spec.strategy}) -> {len(combined)} rows[/]"
    )
    return _finalize(combined, data_config)


def _load_interleaved_streaming_datasets(
    train_paths: list[str], data_config: DataConfig, spec
) -> dict:
    """#459 — data.interleave + data.streaming=true: delegate combining to
    HF ``datasets.interleave_datasets`` / ``concatenate_datasets`` instead
    of reimplementing mixing over an unbounded/remote source we can't
    count ahead of time.

    Every entry (local file path or remote URI — schema guarantees no hub
    names reach here) becomes one streaming dataset via
    ``datasets.load_dataset(builder, data_files=entry, split="train",
    streaming=True)``, where ``builder`` is chosen per entry from its file
    suffix (:func:`_streaming_builder_for`) — the same
    ``SUPPORTED_EXTENSIONS`` set the non-streaming local path
    (:func:`load_raw_data`) dispatches on, so flipping only
    ``data.streaming: true`` doesn't silently reinterpret a csv/txt/parquet
    file as JSON. Suffix-based selection works identically for local paths
    and remote URIs, which is why the two can share one streaming list.

    Strategy -> HF call mapping (the "mean the same thing as the local
    path" decision #459 asks for; also documented in docs/data.md):

    - ``concat`` -> ``concatenate_datasets(streams)``: same as the local
      path's "extend every source's rows in order", exactly.
    - ``under``  -> ``interleave_datasets(streams,
      stopping_strategy="first_exhausted")``: local ``under`` truncates
      every source to ``min(sizes)``; streaming can't know sizes ahead of
      time, so stopping at the first exhausted stream gives the same
      *shape* of outcome (bounded by the smallest source) without an exact
      row-count guarantee.
    - ``over``   -> ``interleave_datasets(streams,
      stopping_strategy="all_exhausted")``: local ``over`` upsamples every
      source to ``max(sizes)`` by cycling; "all exhausted" recycles
      shorter streams until the longest is consumed once — same shape,
      not exact count.
    - ``probs``  -> ``interleave_datasets(streams,
      probabilities=list(spec.probs), stopping_strategy="first_exhausted")``:
      local ``probs`` apportions an EXACT ratio via largest-remainder;
      streaming samples per ``probabilities`` instead, which converges to
      the same ratio asymptotically rather than exactly. Tested by running
      one probs config through both paths and comparing proportions
      (not two tests that each pass alone) — see
      test_issue459_interleave_streaming_hub.py.

    Combined output is eagerly materialised up to MAX_REMOTE_ROWS (shared
    with _load_remote_dataset — one OOM ceiling, not two). Format
    detection runs ONCE on the combined raw rows when format="auto",
    mirroring _load_remote_dataset's existing single-detect behaviour —
    unlike the local-file interleave path's per-source auto-detect,
    streaming sources are assumed schema-homogeneous; mixing differently
    -formatted streams via streaming interleave is out of scope (use the
    non-streaming local path for that).

    Security: every entry classified as a remote URI is canonicalised
    through ``validate_remote_uri`` before it reaches ``hf_load`` — the
    same allowlist validator _load_remote_dataset already runs for a
    single URI (bucket regex, no userinfo / query / fragment; a query
    string is SSRF-adjacent since fsspec backends treat it as a config
    override). The schema-time classifier only checks for a ``"://"``
    (scheme-agnostic — #468 review fix), so this call is the one place
    that actually validates the URI shape AND enforces the scheme
    allowlist — without it, a crafted entry like
    ``s3://bucket/key.jsonl?endpoint=http://169.254.169.254`` would reach
    hf_load unvalidated, and a non-allowlisted-scheme entry like
    ``https://example.com/data.jsonl`` would reach it unvalidated too.
    """
    try:
        from datasets import concatenate_datasets, interleave_datasets
        from datasets import load_dataset as hf_load
    except ImportError as exc:
        raise ImportError(
            "data.streaming=true requires the 'datasets' package: "
            "pip install datasets"
        ) from exc

    from soup_cli.utils.data_pipeline import validate_remote_uri

    streams = []
    for entry in train_paths:
        load_entry = validate_remote_uri(entry) if _looks_like_remote_uri(entry) else entry
        builder = _streaming_builder_for(load_entry)
        ds = hf_load(builder, data_files=load_entry, split="train", streaming=True)
        buf = data_config.buffer_size
        if buf:
            ds = ds.shuffle(buffer_size=buf)
        streams.append(ds)

    if spec.strategy == "concat":
        combined_stream = concatenate_datasets(streams)
    elif spec.strategy == "under":
        combined_stream = interleave_datasets(streams, stopping_strategy="first_exhausted")
    elif spec.strategy == "over":
        combined_stream = interleave_datasets(streams, stopping_strategy="all_exhausted")
    elif spec.strategy == "probs":
        combined_stream = interleave_datasets(
            streams, probabilities=list(spec.probs), stopping_strategy="first_exhausted"
        )
    else:
        raise AssertionError(f"unreachable interleave strategy {spec.strategy!r}")

    raw_data: list[dict] = []
    for i, row in enumerate(combined_stream):
        if i >= MAX_REMOTE_ROWS:
            console.print(
                f"[yellow]Interleaved streaming dataset truncated at "
                f"{MAX_REMOTE_ROWS:,} rows (use non-streaming local files "
                "for larger jobs).[/]"
            )
            break
        raw_data.append(dict(row))

    fmt = data_config.format
    if fmt == "auto":
        fmt = detect_format(raw_data)
        console.print(f"[dim]Auto-detected format: {fmt}[/]")

    formatted = [format_to_messages(row, fmt) for row in raw_data]
    formatted = [r for r in formatted if r is not None]

    console.print(
        f"[dim]Streaming-interleaved {len(train_paths)} datasets "
        f"(strategy={spec.strategy}) -> {len(formatted)} rows[/]"
    )
    return _finalize(formatted, data_config)


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
    """Quick sniff for a URI-shaped data.train entry (contains '://').

    Deliberately scheme-agnostic (#468 review fix) — classification and
    dispatch only need to know "does this want to be treated as a remote
    source", not "is the scheme one we allow". Scheme-allowlist
    enforcement happens once, downstream, in validate_remote_uri, which
    refuses a non-allowlisted scheme BY NAME — narrowing this sniff to
    allowlisted schemes only (as it did before this fix, via is_remote_uri)
    let https/http/ftp (with a familiar file suffix) slip into 'local'
    classification and reach hf_load completely unvalidated, bypassing
    validate_remote_uri rather than being refused by it.
    """
    return isinstance(value, str) and "://" in value


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
        # Eager materialise capped at MAX_REMOTE_ROWS — emit a clear advisory
        # if the cap trips.
        raw_data: list[dict] = []
        for i, row in enumerate(ds):
            if i >= MAX_REMOTE_ROWS:
                console.print(
                    f"[yellow]Remote dataset truncated at {MAX_REMOTE_ROWS:,} "
                    f"rows (use a local split for larger jobs).[/]"
                )
                break
            raw_data.append(row)
    else:
        # Non-streaming: open once, read lines, decode JSON.
        raw_data = []
        with fsspec.open(canonical, mode="rt", encoding="utf-8-sig") as fh:
            for i, raw_line in enumerate(fh):
                if i >= MAX_REMOTE_ROWS:
                    console.print(
                        f"[yellow]Remote dataset truncated at "
                        f"{MAX_REMOTE_ROWS:,} rows.[/]"
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


def _load_one_hub_dataset(
    name: str, data_config: DataConfig
) -> tuple[list[dict], list[dict] | None]:
    """Load + format one HF-hub dataset name's 'train' (+ optional
    'validation') split. Factored out of _load_hf_dataset (#459) — same
    pattern #443 used for _load_one_local_dataset — so the hub interleave
    path below reuses this rather than re-deriving it. _load_hf_dataset's
    own single-name behaviour is left byte-identical by construction: it
    now just calls this and forwards straight to _finalize, same as before.
    """
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
        val_data = [dict(row) for row in ds["validation"]]
        val_formatted = [format_to_messages(row, fmt) for row in val_data]
        return formatted, [r for r in val_formatted if r is not None]

    return formatted, None


def _load_hf_dataset(name: str, data_config: DataConfig) -> dict:
    """Load a dataset from HuggingFace Hub.

    The hub split wins over val_split; passing it through as _finalize's
    ``val=`` means _finalize does not re-derive one from the train rows.
    """
    formatted, val_formatted = _load_one_hub_dataset(name, data_config)
    if val_formatted is not None:
        return _finalize(formatted, data_config, val=val_formatted)
    return _finalize(formatted, data_config)


def _load_interleaved_hub_datasets(train_names: list[str], data_config: DataConfig) -> dict:
    """#459 — data.train is a list of HF-hub dataset names combined via
    data.interleave. Reuses the SAME _combine_interleaved the local-file
    path (#443) uses — this is what makes the strategy names mean the same
    thing across both paths, by construction rather than by parallel
    reimplementation.

    Decided validation-split precedence (documented here + docs/data.md,
    per the issue's demand that this not be emergent): a hub entry's own
    'validation' split is honoured for the COMBINED result only when EVERY
    entry provides one — those are combined with the same spec and passed
    through as _finalize's val=. If only some entries provide one, it is
    ignored (warned, naming which entries had it) and data_config.val_split
    is applied to the combined train rows instead, exactly as if no entry
    had a hub split — a partial-hub-split mixture would otherwise silently
    be a smaller/differently-composed val set than a reader expects. If no
    entry provides one, behaviour is unchanged from today.
    """
    from soup_cli.utils.data_pipeline import parse_interleave

    spec = parse_interleave(data_config.interleave, num_datasets=len(train_names))
    if spec is None:
        # Defence-in-depth only — see _load_interleaved_local_datasets's
        # identical comment.
        raise ValueError("data.train is a list but data.interleave is not set")

    per_dataset_train: list[list[dict]] = []
    per_dataset_val: list[list[dict] | None] = []
    for name in train_names:
        train_rows, val_rows = _load_one_hub_dataset(name, data_config)
        per_dataset_train.append(train_rows)
        per_dataset_val.append(val_rows)

    combined_train = _combine_interleaved(per_dataset_train, spec)

    has_val = [v is not None for v in per_dataset_val]
    if all(has_val):
        combined_val = _combine_interleaved(
            [v for v in per_dataset_val if v is not None], spec
        )
        result = _finalize(combined_train, data_config, val=combined_val)
    else:
        if any(has_val):
            missing = [
                name for name, v in zip(train_names, per_dataset_val) if v is None
            ]
            console.print(
                "[yellow]data.interleave: only some HF-hub datasets provide "
                f"a 'validation' split (missing from {missing}) — ignoring "
                "every hub validation split and applying data.val_split to "
                "the combined train rows instead.[/]"
            )
        result = _finalize(combined_train, data_config)

    console.print(
        f"[dim]Interleaved {len(train_names)} HF-hub datasets "
        f"(strategy={spec.strategy}) -> {len(combined_train)} rows[/]"
    )
    return result
