"""Data loading from local files and HuggingFace."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from functools import partial
from itertools import islice
from pathlib import Path

from rich.console import Console

from soup_cli.config.schema import DataConfig
from soup_cli.data.formats import (
    MAX_DETECT_ROWS,
    detect_format,
    format_to_messages_with_reason,
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
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("Install pyarrow to read parquet files: pip install pyarrow")

    table = pq.read_table(path)

    # pandas stores an unnamed, non-default index (after df.sample() or a filter)
    # as a hidden __index_level_N__ column; pd.read_parquet restored it as the index.
    pandas_meta = table.schema.pandas_metadata or {}
    hidden = {
        name
        for name in pandas_meta.get("index_columns", [])
        if isinstance(name, str) and name.startswith("__index_level_")
    }

    if hidden:
        table = table.select(
            [name for name in table.column_names if name not in hidden]
        )
    return table.to_pylist()


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


@dataclass(frozen=True)
class LoadOutcome:
    """What the most recent :func:`load_dataset` call converted (#1217).

    ``fmt`` is the format the rows were converted as (resolved, never
    ``"auto"``); ``first_drop`` is ``(format, row index, reason, source)`` for
    the first row a converter dropped. ``soup train`` reads it to say why a
    load ended with zero training rows instead of printing "Ready to train".

    Frozen, and replaced (never mutated) as a load progresses, so a caller
    holding one keeps a snapshot that no later load can change.
    """

    fmt: str | None = None
    first_drop: tuple[str, int, str, str | None] | None = None


_last_load = LoadOutcome()


def last_load_outcome() -> LoadOutcome:
    """The :class:`LoadOutcome` of the most recent :func:`load_dataset` call."""
    return _last_load


# Tasks that require source columns preserved across dataset format normalisation (#1219).
# GRPO uses raw columns for custom reward functions (answer, expected, etc.).
# The classifier family (classifier, reranker, cross_encoder) uses source columns
# to retain label, paired text (text_a, text_b, question, answer), and metadata.
PRESERVE_SOURCE_TASKS: frozenset[str] = frozenset({
    "grpo",
    "classifier",
    "reranker",
    "cross_encoder",
})


def task_preserves_source_columns(task: str) -> bool:
    """Return whether task requires source dataset columns to be preserved."""
    return task in PRESERVE_SOURCE_TASKS


def data_config_for_task(data_config: DataConfig, task: str) -> DataConfig:
    """Resolve ``format: auto`` for task: cross_encoder (#1219).

    ``detect_format`` cannot see the task, and ``{question, answer}`` is also
    GSM8K's shape, so pair rows are recognised only when the task reads them.
    """
    if task == "cross_encoder" and data_config.format == "auto":
        return data_config.model_copy(update={"format": "cross_encoder"})
    return data_config


def _format_rows(
    raw_data: list[dict],
    fmt: str,
    *,
    preserve_source_columns: bool = False,
    source: str | None = None,
    kept_indices: list[int] | None = None,
) -> list[dict]:
    """Normalize rows, optionally retaining columns used by GRPO rewards.

    TRL forwards every non-prompt dataset column to custom reward functions.
    Soup's normalizers intentionally return only task-specific columns, which
    is correct for supervised and preference trainers but would discard GRPO
    references such as ``answer``, ``expected``, ``schema``, or custom
    metadata.  The opt-in keeps the default loader contract byte-for-byte
    unchanged for every other task.

    ``kept_indices``, when given, receives the ``raw_data`` index of every row
    that converted, in order, so a caller can map a formatted row back to the
    raw row it came from.
    """
    formatted: list[dict] = []
    dropped = 0
    first_drop: tuple[int, str] | None = None
    for index, raw_row in enumerate(raw_data):
        # #1181: the same drop decision as format_to_messages, plus the reason,
        # so a dropped row is counted and named instead of vanishing.
        normalized, reason = format_to_messages_with_reason(raw_row, fmt)
        if normalized is None:
            dropped += 1
            if first_drop is None:
                first_drop = (index, reason or "")
            continue
        if preserve_source_columns:
            normalized = {**raw_row, **normalized}
        formatted.append(normalized)
        if kept_indices is not None:
            kept_indices.append(index)
    global _last_load
    if dropped and first_drop is not None:
        _report_dropped_rows(dropped, len(raw_data), fmt, first_drop, source)
        if _last_load.first_drop is None:
            _last_load = replace(
                _last_load, first_drop=(fmt, first_drop[0], first_drop[1], source)
            )
    if _last_load.fmt is None:
        _last_load = replace(_last_load, fmt=fmt)
    return formatted


def _report_dropped_rows(
    dropped: int,
    total: int,
    fmt: str,
    first_drop: tuple[int, str],
    source: str | None,
) -> None:
    """One warning for the rows the format converter dropped (#1181).

    The drop is the loader's documented contract (one bad line must not abort a
    load); the silence was the defect. Same shape as the media paths' "rows
    skipped" lines, with the first row's index and the converter's own reason.
    """
    from rich.markup import escape

    index, reason = first_drop
    message = (
        f"[yellow]Warning: {dropped} of {total} rows dropped: they do not convert "
        f"as {escape(repr(fmt))}. First: row {index}: {escape(reason)}.[/]"
    )
    if source:
        message += (
            f"\n  List them all with: soup data validate {escape(source)} --format "
            f"{escape(fmt)}"
        )
    console.print(message)


def _load_replay_rows(
    data_config: DataConfig,
    *,
    preserve_source_columns: bool = False,
) -> list[dict]:
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
    rows = _format_rows(
        raw,
        fmt,
        preserve_source_columns=preserve_source_columns,
        source=str(replay_path),
    )

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
    preserve_source_columns: bool = False,
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
        train_rows, val_rows = _split_val(formatted, data_config.val_split)
        result = {"train": train_rows, "val": val_rows}
    else:
        result = {"train": formatted}

    if getattr(data_config, "replay", None):
        from soup_cli.utils.rehearsal import mix_replay

        replay_rows = _load_replay_rows(
            data_config,
            preserve_source_columns=preserve_source_columns,
        )
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
    classification in schema._classify_data_train_entry — kept
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


def load_dataset(
    data_config: DataConfig,
    *,
    preserve_source_columns: bool = False,
) -> dict:
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
    global _last_load
    _last_load = LoadOutcome()

    train_path = data_config.train

    if isinstance(train_path, list):
        kinds = {_classify_train_entry(p) for p in train_path}
        if kinds == {"hub"}:
            return _load_interleaved_hub_datasets(
                train_path,
                data_config,
                preserve_source_columns=preserve_source_columns,
            )
        return _load_interleaved_local_datasets(
            train_path,
            data_config,
            preserve_source_columns=preserve_source_columns,
        )

    # v0.53.8 #85 — fsspec live remote loader. Schema accepts these URIs
    # since v0.42.0; live loader lands here. Lazy-imports fsspec + the
    # backend driver (s3fs / gcsfs / adlfs / ocifs) and surfaces a
    # friendly Rich panel naming the pip install when the driver is
    # missing.
    if _looks_like_remote_uri(train_path):
        return _load_remote_dataset(
            train_path,
            data_config,
            preserve_source_columns=preserve_source_columns,
        )

    # Check if it's a HuggingFace dataset
    if not Path(train_path).suffix:
        return _load_hf_dataset(
            train_path,
            data_config,
            preserve_source_columns=preserve_source_columns,
        )

    # Local file
    path = Path(train_path)
    raw_data = load_raw_data(path)

    # Detect or use specified format
    fmt = data_config.format
    if fmt == "auto":
        fmt = detect_format(raw_data)
        console.print(f"[dim]Auto-detected format: {fmt}[/]")

    # Convert to standard message format
    formatted = _format_rows(
        raw_data,
        fmt,
        preserve_source_columns=preserve_source_columns,
        source=str(path),
    )

    # Validate image paths for vision formats
    if is_vision_format(fmt):
        image_dir = Path(data_config.image_dir) if data_config.image_dir else path.parent
        formatted = _validate_vision_images(formatted, image_dir)

    # Validate audio paths for audio formats
    if is_audio_format(fmt):
        audio_dir = Path(data_config.audio_dir) if data_config.audio_dir else path.parent
        formatted = _validate_audio_files(formatted, audio_dir)

    # Split into train/val, then mix replay into train (v0.71.36).
    return _finalize(
        formatted,
        data_config,
        preserve_source_columns=preserve_source_columns,
    )


def _load_one_local_dataset(
    train_path: str,
    data_config: DataConfig,
    *,
    preserve_source_columns: bool = False,
) -> list[dict]:
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

    formatted = _format_rows(
        raw_data,
        fmt,
        preserve_source_columns=preserve_source_columns,
        source=str(path),
    )

    if is_vision_format(fmt):
        image_dir = Path(data_config.image_dir) if data_config.image_dir else path.parent
        formatted = _validate_vision_images(formatted, image_dir)
    if is_audio_format(fmt):
        audio_dir = Path(data_config.audio_dir) if data_config.audio_dir else path.parent
        formatted = _validate_audio_files(formatted, audio_dir)
    return formatted


def _split_val(rows: list[dict], val_split: float) -> tuple[list[dict], list[dict]]:
    """Slice ``rows`` into (train, val) at the same fraction _finalize has
    always used, factored out so callers can carve val out of a row set
    that has not been through interleave oversampling yet (#680).
    """
    split_idx = int(len(rows) * (1 - val_split))
    return rows[:split_idx], rows[split_idx:]


def _split_val_per_source(
    rows: list[dict], val_split: float, strategy: str
) -> tuple[list[dict], list[dict]]:
    """``_split_val`` for a per-source carve-out (#680). An empty train side
    here would otherwise surface downstream as _combine_interleaved's generic
    "must have >= 1 row" error, which is misleading: the source did have
    rows, val_split just consumed all of them.
    """
    train_rows, val_rows = _split_val(rows, val_split)
    if rows and not train_rows:
        raise ValueError(
            f"data.interleave: data.val_split={val_split} leaves 0 training "
            f"rows for a source with {len(rows)} row(s) under strategy "
            f"'{strategy}', which carves val out per source. Use "
            "'concat'/'under', lower val_split, or drop the source."
        )
    return train_rows, val_rows


def _row_key(row: dict) -> str:
    """A stable identity for a formatted row, for duplicate detection.

    Rows are plain dicts of JSON-ish values, so a sorted-key dump is both
    hashable and order-insensitive. Falls back to ``repr`` for anything
    ``json`` refuses, which keeps an exotic value from turning a leak check
    into a crash.
    """
    import json

    try:
        return json.dumps(row, sort_keys=True, default=repr)
    except (TypeError, ValueError):
        return repr(sorted(row.items(), key=lambda kv: str(kv[0])))


def _split_val_deduplicated(
    rows: list[dict], val_split: float
) -> tuple[list[dict], list[dict]]:
    """Carve val out of ``rows`` so no val row has a duplicate left in train (#702).

    The streaming ``over`` path interleaves with
    ``stopping_strategy="all_exhausted"``, which recycles the shorter stream
    to exhaust the longer one. A row and its recycled copy are separate
    entries, so the positional tail slice ``_split_val`` takes could put one
    on each side -- validation loss on the recycled source then measures
    memorisation and reports a suspiciously good number rather than an error.

    A stream is not countable ahead of time, so #701's per-source carve-out
    does not transfer. The split is instead sized over *distinct* rows, and
    val is drawn from the tail of the distinct rows in stream order,
    preferring content that occurs exactly once: such a row has no copy to
    leave behind, so taking it costs train nothing. Content is keyed by
    value, so a recycled copy and a genuinely repeated source row are
    indistinguishable; only when too few unique rows exist is repeated
    content used, and then every other copy is withheld from train and the
    number withheld is reported rather than dropped silently.

    Raises ``ValueError`` when the split would leave no training rows, the
    same refusal ``_split_val_per_source`` makes on the eager path.
    """
    from collections import Counter

    keys = [_row_key(row) for row in rows]
    counts = Counter(keys)
    first_index: dict[str, int] = {}
    for index, key in enumerate(keys):
        first_index.setdefault(key, index)
    distinct = list(first_index)

    n_val = len(distinct) - int(len(distinct) * (1 - val_split))
    newest_first = distinct[::-1]
    unique = [key for key in newest_first if counts[key] == 1]
    repeated = [key for key in newest_first if counts[key] > 1]
    chosen = unique[:n_val]
    fallback = repeated[: n_val - len(chosen)]
    val_keys = set(chosen) | set(fallback)

    val_rows = [rows[first_index[key]] for key in sorted(val_keys, key=first_index.get)]
    train_rows = [row for row, key in zip(rows, keys) if key not in val_keys]

    if rows and not train_rows:
        raise ValueError(
            f"data.interleave: data.val_split={val_split} leaves 0 training rows "
            f"under streaming 'over', which splits over distinct rows: the "
            f"{len(rows)} materialised row(s) hold {len(distinct)} distinct "
            "row(s). Add distinct rows, set val_split: 0, or use 'concat'/'under'."
        )
    withheld = sum(counts[key] - 1 for key in fallback)
    if withheld:
        console.print(
            f"[yellow]Warning: data.val_split under streaming 'over' withheld "
            f"{withheld} duplicate row(s) from train: too few rows with unique "
            f"content to fill val, so {len(fallback)} val row(s) have copies "
            "that would otherwise have leaked into train.[/]"
        )
    return train_rows, val_rows


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


def _load_interleaved_local_datasets(
    train_paths: list[str],
    data_config: DataConfig,
    *,
    preserve_source_columns: bool = False,
) -> dict:
    """#443: load + combine every local dataset in a list-shaped
    data.train per data.interleave, then hand the combined rows to
    _finalize(): the same seam every other loader uses. #680: "over"/
    "probs" now carve val out per source before cycling (_split_val),
    since those are the strategies that pad a source with copies of its
    own rows; "concat"/"under" are unchanged from #443.

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
        return _load_interleaved_streaming_datasets(
            train_paths,
            data_config,
            spec,
            preserve_source_columns=preserve_source_columns,
        )

    per_dataset_rows = [
        _load_one_local_dataset(
            p,
            data_config,
            preserve_source_columns=preserve_source_columns,
        )
        for p in train_paths
    ]

    if data_config.val_split > 0 and spec.strategy in ("over", "probs"):
        # #680: these two strategies pad a source with copies of its own
        # rows, so splitting the padded result can put one copy in train
        # and another in val. Carve val out of each source first instead.
        per_dataset_train = []
        per_dataset_val = []
        for rows in per_dataset_rows:
            train_rows, val_rows = _split_val_per_source(
                rows, data_config.val_split, spec.strategy
            )
            per_dataset_train.append(train_rows)
            per_dataset_val.append(val_rows)
        combined_train = _combine_interleaved(per_dataset_train, spec)
        combined_val = [row for val_rows in per_dataset_val for row in val_rows]
        console.print(
            f"[dim]Interleaved {len(train_paths)} datasets "
            f"(strategy={spec.strategy}) -> {len(combined_train)} train, "
            f"{len(combined_val)} val rows[/]"
        )
        return _finalize(
            combined_train,
            data_config,
            val=combined_val,
            preserve_source_columns=preserve_source_columns,
        )

    combined = _combine_interleaved(per_dataset_rows, spec)
    console.print(
        f"[dim]Interleaved {len(train_paths)} datasets "
        f"(strategy={spec.strategy}) -> {len(combined)} rows[/]"
    )
    return _finalize(
        combined,
        data_config,
        preserve_source_columns=preserve_source_columns,
    )


def _load_interleaved_streaming_datasets(
    train_paths: list[str],
    data_config: DataConfig,
    spec,
    *,
    preserve_source_columns: bool = False,
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

    Media paths (vision / audio formats) resolve per entry, as on the eager
    path: against ``data.image_dir`` / ``data.audio_dir`` when set, else the
    entry's own directory (a remote entry has none). When the entries do not
    share that directory, each stream is tagged with its entry's index before
    the streams are combined -- lazily, through ``IterableDataset.map`` -- and
    the tag is popped as the rows materialise (:func:`_streams_need_source_tags`).
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

    tag_sources = _streams_need_source_tags(streams, data_config, train_paths)
    if tag_sources:
        streams = _tag_streams(streams)

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
    row_sources: list[int] = []
    for i, row in enumerate(combined_stream):
        if i >= MAX_REMOTE_ROWS:
            console.print(
                f"[yellow]Interleaved streaming dataset truncated at "
                f"{MAX_REMOTE_ROWS:,} rows (use non-streaming local files "
                "for larger jobs).[/]"
            )
            break
        raw_row = dict(row)
        if tag_sources:
            row_sources.append(raw_row.pop(_SOURCE_INDEX_KEY))
        raw_data.append(raw_row)

    fmt = data_config.format
    if fmt == "auto":
        fmt = detect_format(raw_data)
        console.print(f"[dim]Auto-detected format: {fmt}[/]")

    kept: list[int] = []
    formatted = _format_rows(
        raw_data,
        fmt,
        preserve_source_columns=preserve_source_columns,
        kept_indices=kept,
    )
    formatted = _resolve_streamed_media(
        formatted,
        [row_sources[position] for position in kept] if tag_sources else None,
        fmt,
        data_config,
        train_paths,
    )

    console.print(
        f"[dim]Streaming-interleaved {len(train_paths)} datasets "
        f"(strategy={spec.strategy}) -> {len(formatted)} rows[/]"
    )
    # `over` recycles the shorter stream, so the materialised rows contain
    # duplicates and the positional split in `_finalize` could put a row and
    # its copy on opposite sides (#702). The other three strategies never
    # duplicate a row here, so they keep the ordinary path.
    if spec.strategy == "over" and data_config.val_split > 0:
        train_rows, val_rows = _split_val_deduplicated(
            formatted, data_config.val_split
        )
        return _finalize(
            train_rows,
            data_config,
            val=val_rows,
            preserve_source_columns=preserve_source_columns,
        )

    return _finalize(
        formatted,
        data_config,
        preserve_source_columns=preserve_source_columns,
    )


def _validate_vision_images(data: list[dict], image_dir: Path) -> list[dict]:
    """Validate and resolve image paths in vision dataset rows.

    Each row must have an 'image' key with a filename or path, a decoded PIL
    image, or a datasets Image struct ({'bytes': ..., 'path': ...}). Resolves
    relative paths against image_dir and rejects path traversal — a crafted
    llava/sharegpt4v row like ``{"image": "/etc/passwd"}`` must not be handed
    to ``PIL.Image.open``. Mirrors :func:`_validate_audio_files` (the sibling
    audio path got this fix in v0.71.32; the vision path was missed).
    """
    return _resolve_media_column(data, "image", image_dir)


def _validate_audio_files(data: list[dict], audio_dir: Path) -> list[dict]:
    """Validate and resolve audio file paths in audio dataset rows.

    Each row must have an 'audio' key with a filename or path.
    Resolves relative paths against audio_dir. Rejects path traversal.
    """
    return _resolve_media_column(data, "audio", audio_dir)


# Row key of a media format -> the setting that names its base directory.
_MEDIA_DIR_SETTINGS = {"image": "data.image_dir", "audio": "data.audio_dir"}

# Set on streamed rows by _tag_source_index and popped as the rows
# materialise: the data.train index the row was read from.
_SOURCE_INDEX_KEY = "__soup_source_index__"


def _media_key(fmt: str) -> str | None:
    """The row key that holds a media path in ``fmt``; None for a text format."""
    if is_vision_format(fmt):
        return "image"
    if is_audio_format(fmt):
        return "audio"
    return None


def _media_dir_for(key: str, data_config: DataConfig, data_dir: Path | None) -> Path | None:
    """``data.image_dir`` / ``data.audio_dir`` when set, else ``data_dir``."""
    configured = data_config.image_dir if key == "image" else data_config.audio_dir
    return Path(configured) if configured else data_dir


def _entry_media_dir(key: str, data_config: DataConfig, entry: str) -> Path | None:
    """The base directory of one data.train entry's media; None for a remote
    entry when no directory is configured."""
    data_dir = None if _looks_like_remote_uri(entry) else Path(entry).parent
    return _media_dir_for(key, data_config, data_dir)


def _media_dir_required(source: str, key: str, *, plural: bool = False) -> ValueError:
    """The error for media paths that have no local directory to resolve against."""
    setting = _MEDIA_DIR_SETTINGS[key]
    if plural:
        subject = f"{source} are not local files, so their"
    else:
        subject = f"{source} is not a local file, so its"
    return ValueError(
        f"{subject} {key!r} paths have no directory to resolve against. "
        f"Set {setting} to the local directory that holds the {key} files: "
        f"relative paths resolve under it, and rows whose {key} is outside it "
        "are skipped."
    )


def _resolve_media_entries(
    rows: list[dict],
    key: str,
    media_dir: Path | None,
    *,
    source: str = "data.train",
    plural: bool = False,
) -> tuple[list[dict | None], int, int]:
    """Resolve each row's ``key`` media path against ``media_dir``.

    Returns one entry per row, in order -- the row with its path resolved,
    the row unchanged when its value is not a path at all (an image a Hub
    dataset has already decoded), or None for a skipped row -- plus how many
    rows were skipped for a missing path and for a path outside
    ``media_dir``. Every load path goes through here, so a row is treated
    the same way wherever it was read from.

    A relative path resolves against ``media_dir``. A network-share or
    device form (:func:`~soup_cli.utils.paths.is_network_or_device_path`), a
    value holding a NUL byte, and a ``bytes`` or path-object value count as
    outside and are refused before any filesystem call. ``media_dir`` is
    None for a remote URI or Hub dataset with no ``data.image_dir`` /
    ``data.audio_dir``: a string path there has nothing to resolve against,
    so the load stops and names the setting (``source``, ``plural``).
    """
    from soup_cli.utils.paths import is_network_or_device_path, is_under

    entries: list[dict | None] = []
    missing = 0
    outside = 0
    for row in rows:
        value = row.get(key)
        if not value:
            missing += 1
            entries.append(None)
            continue
        if isinstance(value, (bytes, os.PathLike)):
            # A reader takes these as a file name too, but they are not path
            # strings: skipped like a path outside, never handed on unchecked.
            outside += 1
            entries.append(None)
            continue
        if not isinstance(value, str):
            if isinstance(value, dict):
                # Struct with embedded bytes passes through unchanged
                if value.get("bytes"):
                    entries.append(row)
                    continue
                path_val = value.get("path")
                if path_val:
                    if isinstance(path_val, (bytes, os.PathLike)):
                        outside += 1
                        entries.append(None)
                        continue
                    if not isinstance(path_val, str):
                        missing += 1
                        entries.append(None)
                        continue
                    if media_dir is None:
                        raise _media_dir_required(source, key, plural=plural)
                    if "\x00" in path_val or is_network_or_device_path(path_val):
                        outside += 1
                        entries.append(None)
                        continue
                    media_path = Path(path_val)
                    if not media_path.is_absolute():
                        media_path = media_dir / media_path
                    if not is_under(media_path, media_dir):
                        outside += 1
                        entries.append(None)
                        continue
                    entries.append({**row, key: {**value, "path": str(media_path.resolve())}})
                    continue
            entries.append(row)
            continue
        if media_dir is None:
            raise _media_dir_required(source, key, plural=plural)
        if "\x00" in value or is_network_or_device_path(value):
            outside += 1
            entries.append(None)
            continue
        media_path = Path(value)
        if not media_path.is_absolute():
            media_path = media_dir / media_path
        # realpath + commonpath (is_under) -- Path.is_relative_to() breaks on
        # Windows 8.3 short names.
        if not is_under(media_path, media_dir):
            outside += 1
            entries.append(None)
            continue
        entries.append({**row, key: str(media_path.resolve())})
    return entries, missing, outside


def _warn_media_skips(key: str, missing: int, outside: int, media_dir: Path | None) -> None:
    """The "rows skipped" warnings of one media pass, worded as they always were.

    ``media_dir`` is None only for rows with no local directory, where the
    one value counted as outside without a string path is a ``bytes`` or
    path object.
    """
    if missing > 0:
        console.print(f"[yellow]Warning: {missing} rows skipped (missing {key} path)[/]")
    if outside > 0 and key == "image":
        if media_dir is None:
            reason = "image value is not a path string"
        else:
            reason = f"image path outside {media_dir}"
        console.print(f"[yellow]Warning: {outside} rows skipped ({reason})[/]")
    elif outside > 0:
        console.print(
            f"[red]Warning: {outside} rows skipped (audio path traversal blocked)[/]"
        )


def _resolve_media_column(
    rows: list[dict],
    key: str,
    media_dir: Path | None,
    *,
    source: str = "data.train",
    plural: bool = False,
) -> list[dict]:
    """:func:`_resolve_media_entries` plus its warnings; returns the rows kept."""
    entries, missing, outside = _resolve_media_entries(
        rows, key, media_dir, source=source, plural=plural
    )
    _warn_media_skips(key, missing, outside, media_dir)
    return [row for row in entries if row is not None]


def _resolve_remote_media(
    rows: list[dict], fmt: str, data_config: DataConfig, *, source: str
) -> list[dict]:
    """Media pass for rows that did not come from a local file.

    A remote URI or a Hub dataset has no local directory to stand in for the
    data file's, so a string media path resolves only against
    ``data.image_dir`` / ``data.audio_dir``. Rows of a text format are
    returned unchanged.
    """
    key = _media_key(fmt)
    if key is None:
        return rows
    return _resolve_media_column(rows, key, _media_dir_for(key, data_config, None), source=source)


def _streams_need_source_tags(
    streams: list, data_config: DataConfig, train_paths: list[str]
) -> bool:
    """Whether streamed rows must carry the index of the entry they came from.

    Only a media row needs it, and only when the entries do not share one base
    directory for its key: otherwise every row resolves against that base.
    ``format: auto`` is resolved after the rows materialise, from the first
    rows of the combined stream, which are among the first rows of each
    stream; the format found can only be a media format if one of those rows
    carries its key, so the streams are tagged when one does.
    """
    keys = [
        key
        for key in _MEDIA_DIR_SETTINGS
        if len({_entry_media_dir(key, data_config, entry) for entry in train_paths}) > 1
    ]
    if not keys:
        return False
    if data_config.format != "auto":
        return _media_key(data_config.format) in keys
    return any(
        key in row
        for stream in streams
        for row in islice(stream, MAX_DETECT_ROWS)
        for key in keys
    )


def _tag_source_index(row: dict, index: int, columns: tuple[str, ...] = ()) -> dict:
    """``IterableDataset.map`` callback: the data.train index of ``row``, plus
    None for each of ``columns`` the row lacks."""
    tag: dict = {name: None for name in columns if name not in row}
    tag[_SOURCE_INDEX_KEY] = index
    return tag


def _tag_streams(streams: list) -> list:
    """Tag every stream's rows with its data.train index.

    A stream whose features are known keeps them, plus the int64 tag, so
    ``datasets`` does not re-infer its types from the first rows. A row gets
    each column another stream declares and it lacks, as None: combining
    untagged streams fills those in the same way.
    """
    columns = tuple(
        dict.fromkeys(
            name for stream in streams if stream.features is not None for name in stream.features
        )
    )
    tagged = []
    for index, stream in enumerate(streams):
        tag = partial(_tag_source_index, index=index, columns=columns)
        if stream.features is None:
            tagged.append(stream.map(tag))
            continue
        from datasets import Features, Value

        features = Features({**stream.features, _SOURCE_INDEX_KEY: Value("int64")})
        tagged.append(stream.map(tag, features=features))
    return tagged


def _resolve_streamed_media(
    rows: list[dict],
    row_sources: list[int] | None,
    fmt: str,
    data_config: DataConfig,
    train_paths: list[str],
) -> list[dict]:
    """Media pass of the streaming interleave path: each row against its own entry.

    ``row_sources[i]`` is the data.train index ``rows[i]`` was streamed from.
    A local entry's default directory is its file's parent, as on the eager
    path; a remote entry has none. Each entry is resolved and warned about
    on its own, like the eager path's per-file warnings, and the rows keep
    their stream order. ``row_sources`` is None when the streams were not
    tagged: every entry then has the same base for this format's key.
    """
    key = _media_key(fmt)
    if key is None:
        return rows
    if row_sources is None:
        names = ", ".join(repr(entry) for entry in train_paths)
        return _resolve_media_column(
            rows,
            key,
            _entry_media_dir(key, data_config, train_paths[0]),
            source=f"data.train entries {names}",
            plural=True,
        )
    positions: dict[int, list[int]] = {}
    for position, source_index in enumerate(row_sources):
        positions.setdefault(source_index, []).append(position)
    resolved: list[dict | None] = list(rows)
    for source_index in sorted(positions):
        entry = train_paths[source_index]
        media_dir = _entry_media_dir(key, data_config, entry)
        group = positions[source_index]
        entries, missing, outside = _resolve_media_entries(
            [rows[position] for position in group],
            key,
            media_dir,
            source=f"data.train[{source_index}] {entry!r}",
        )
        _warn_media_skips(key, missing, outside, media_dir)
        for position, entry_row in zip(group, entries):
            resolved[position] = entry_row
    return [row for row in resolved if row is not None]


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


def _load_remote_dataset(
    train_path: str,
    data_config: DataConfig,
    *,
    preserve_source_columns: bool = False,
) -> dict:
    """Load JSONL from a remote fsspec URI (s3 / gs / az / oci / etc.).

    Validates the URI via the v0.42.0 ``validate_remote_uri`` allowlist
    (bucket regex, no userinfo/query/fragment) BEFORE opening any
    connection — defends against URL injection into the fsspec backend.

    Streaming knobs (``data_config.streaming`` + ``buffer_size`` + ``shards``)
    are honoured via :func:`datasets.load_dataset` when present; otherwise
    the file is streamed as JSONL through :func:`fsspec.open`.

    Image / audio paths resolve against ``data.image_dir`` /
    ``data.audio_dir`` (:func:`_resolve_remote_media`): a remote object has
    no local directory of its own.
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

    formatted = _format_rows(
        raw_data,
        fmt,
        preserve_source_columns=preserve_source_columns,
    )
    formatted = _resolve_remote_media(
        formatted, fmt, data_config, source=f"Remote dataset {canonical!r}"
    )

    return _finalize(
        formatted,
        data_config,
        preserve_source_columns=preserve_source_columns,
    )


def _rows_from_hub_split(split, data_config: DataConfig, *, shuffle: bool) -> list[dict]:
    """Materialise one hub split.

    Streaming matches `_load_remote_dataset`: optional shuffle buffer, then
    eager rows capped at MAX_REMOTE_ROWS. Non-streaming is the pre-#689
    list comprehension (no cap).
    """
    if data_config.streaming:
        buf = data_config.buffer_size
        if shuffle and buf:
            split = split.shuffle(buffer_size=buf)
        raw_data: list[dict] = []
        for i, row in enumerate(split):
            if i >= MAX_REMOTE_ROWS:
                console.print(
                    f"[yellow]Hub dataset truncated at {MAX_REMOTE_ROWS:,} "
                    f"rows (use a local split for larger jobs).[/]"
                )
                break
            raw_data.append(dict(row))
        return raw_data
    return [dict(row) for row in split]


def _load_one_hub_dataset(
    name: str,
    data_config: DataConfig,
    *,
    preserve_source_columns: bool = False,
) -> tuple[list[dict], list[dict] | None]:
    """Load + format one HF-hub dataset name's 'train' (+ optional
    'validation') split. Factored out of _load_hf_dataset (#459) — same
    pattern #443 used for _load_one_local_dataset — so the hub interleave
    path below reuses this rather than re-deriving it. _load_hf_dataset's
    own single-name behaviour is left byte-identical by construction: it
    now just calls this and forwards straight to _finalize, same as before.

    Image / audio paths in both splits resolve against ``data.image_dir`` /
    ``data.audio_dir`` (:func:`_resolve_remote_media`): a Hub dataset has
    no local directory of its own.
    """
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")

    console.print(f"[dim]Loading from HuggingFace: {name}[/]")
    if data_config.streaming:
        ds = hf_load(name, streaming=True)
    else:
        ds = hf_load(name)

    if "train" not in ds:
        raise ValueError(f"Dataset {name} has no 'train' split")

    raw_data = _rows_from_hub_split(ds["train"], data_config, shuffle=True)
    fmt = data_config.format
    if fmt == "auto":
        fmt = detect_format(raw_data)

    formatted = _format_rows(
        raw_data,
        fmt,
        preserve_source_columns=preserve_source_columns,
    )
    formatted = _resolve_remote_media(
        formatted, fmt, data_config, source=f"Hub dataset {name!r}"
    )

    if "validation" in ds:
        val_data = _rows_from_hub_split(
            ds["validation"], data_config, shuffle=False,
        )
        val_formatted = _format_rows(
            val_data,
            fmt,
            preserve_source_columns=preserve_source_columns,
        )
        val_formatted = _resolve_remote_media(
            val_formatted, fmt, data_config, source=f"Hub dataset {name!r}"
        )
        return formatted, val_formatted

    return formatted, None


def _load_hf_dataset(
    name: str,
    data_config: DataConfig,
    *,
    preserve_source_columns: bool = False,
) -> dict:
    """Load a dataset from HuggingFace Hub.

    The hub split wins over val_split; passing it through as _finalize's
    ``val=`` means _finalize does not re-derive one from the train rows.
    """
    formatted, val_formatted = _load_one_hub_dataset(
        name,
        data_config,
        preserve_source_columns=preserve_source_columns,
    )
    if val_formatted is not None:
        return _finalize(
            formatted,
            data_config,
            val=val_formatted,
            preserve_source_columns=preserve_source_columns,
        )
    return _finalize(
        formatted,
        data_config,
        preserve_source_columns=preserve_source_columns,
    )


def _load_interleaved_hub_datasets(
    train_names: list[str],
    data_config: DataConfig,
    *,
    preserve_source_columns: bool = False,
) -> dict:
    """#459 — data.train is a list of HF-hub dataset names combined via
    data.interleave. Reuses the SAME _combine_interleaved the local-file
    path (#443) uses — this is what makes the strategy names mean the same
    thing across both paths, by construction rather than by parallel
    reimplementation.

    Decided validation-split precedence (documented here + docs/data.md,
    per the issue's demand that this not be emergent): a hub entry's own
    'validation' split is honoured for the COMBINED result only when EVERY
    entry provides one: those are combined with the same spec and passed
    through as _finalize's val=. If only some entries provide one, it is
    ignored (warned, naming which entries had it) and data_config.val_split
    is derived instead, exactly as if no entry had a hub split. A
    partial-hub-split mixture would otherwise silently be a
    smaller/differently-composed val set than a reader expects. If no
    entry provides one, behaviour is unchanged from today. #680: that
    val_split fallback carves val out per source before over/probs pads
    it, same fix and same reason as _load_interleaved_local_datasets.
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
        train_rows, val_rows = _load_one_hub_dataset(
            name,
            data_config,
            preserve_source_columns=preserve_source_columns,
        )
        per_dataset_train.append(train_rows)
        per_dataset_val.append(val_rows)

    has_val = [v is not None for v in per_dataset_val]
    if all(has_val):
        combined_train = _combine_interleaved(per_dataset_train, spec)
        combined_val = _combine_interleaved(
            [v for v in per_dataset_val if v is not None], spec
        )
        result = _finalize(
            combined_train,
            data_config,
            val=combined_val,
            preserve_source_columns=preserve_source_columns,
        )
    else:
        if any(has_val):
            missing = [
                name for name, v in zip(train_names, per_dataset_val) if v is None
            ]
            if spec.strategy in ("over", "probs"):
                detail = "carving data.val_split out per source before oversampling"
            else:
                detail = "applying data.val_split to the combined train rows instead"
            console.print(
                "[yellow]data.interleave: only some HF-hub datasets provide "
                f"a 'validation' split (missing from {missing}) — ignoring "
                f"every hub validation split and {detail}.[/]"
            )
        if data_config.val_split > 0 and spec.strategy in ("over", "probs"):
            per_split = [
                _split_val_per_source(rows, data_config.val_split, spec.strategy)
                for rows in per_dataset_train
            ]
            combined_train = _combine_interleaved([t for t, _ in per_split], spec)
            combined_val = [row for _, v in per_split for row in v]
            result = _finalize(
                combined_train,
                data_config,
                val=combined_val,
                preserve_source_columns=preserve_source_columns,
            )
        else:
            combined_train = _combine_interleaved(per_dataset_train, spec)
            result = _finalize(
                combined_train,
                data_config,
                preserve_source_columns=preserve_source_columns,
            )

    console.print(
        f"[dim]Interleaved {len(train_names)} HF-hub datasets "
        f"(strategy={spec.strategy}) -> {len(combined_train)} rows[/]"
    )
    return result
