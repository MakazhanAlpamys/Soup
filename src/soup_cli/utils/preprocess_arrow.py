"""Bounded Arrow IPC production and validation for the existing saved-dataset layout."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable
from pathlib import Path

_BATCH_ROWS = 128
_BATCH_TOKENS = 65_536
_SHARD_ROWS = 1024
_SHARD_BYTES = 16 * 1024 * 1024


class PreparedDataset:
    """An on-disk producer result; publication copies it without mapping the corpus."""

    def __init__(self, rows: Iterable[dict], folder: Path) -> None:
        from datasets import Dataset, Features, Sequence, Value
        from datasets.arrow_writer import ArrowWriter

        self.folder = folder
        self.num_rows = 0
        self.features = Features({
            "input_ids": Sequence(Value("int32")), "labels": Sequence(Value("int64")),
            "attention_mask": Sequence(Value("int8")),
        })
        files: list[dict[str, str]] = []
        fingerprint = hashlib.sha256()
        batch: list[dict] = []
        writer = None
        shard_bytes = 0
        shard_rows = 0
        batch_tokens = 0

        def write_batch():
            nonlocal writer, shard_bytes, shard_rows, batch_tokens
            dataset = Dataset.from_list(batch, features=self.features)
            if not files:
                dataset.info.write_to_directory(str(folder))
            if writer is None:
                filename = f"data-{len(files):05d}.arrow"
                files.append({"filename": filename})
                writer = ArrowWriter(
                    path=str(folder / filename), features=self.features,
                    writer_batch_size=_BATCH_ROWS,
                )
            writer.write_table(dataset.data.table)
            fingerprint.update(dataset._fingerprint.encode())
            self.num_rows += len(batch)
            shard_bytes += dataset.data.table.nbytes
            shard_rows += len(batch)
            batch.clear()
            batch_tokens = 0
            if shard_bytes >= _SHARD_BYTES or shard_rows >= _SHARD_ROWS:
                completed, writer = writer, None
                try:
                    completed.finalize()
                finally:
                    completed.close()
                shard_bytes = 0
                shard_rows = 0

        try:
            for row in rows:
                batch.append(row)
                batch_tokens += len(row["input_ids"])
                if len(batch) >= _BATCH_ROWS or batch_tokens >= _BATCH_TOKENS:
                    write_batch()
            if batch:
                write_batch()
            if writer is not None:
                writer.finalize()
        finally:
            if writer is not None:
                writer.close()
        with (folder / "state.json").open("w", encoding="utf-8") as handle:
            json.dump({
                "_data_files": files, "_fingerprint": fingerprint.hexdigest()[:16],
                "_format_columns": None, "_format_kwargs": {}, "_format_type": None,
                "_output_all_columns": False, "_split": None,
            }, handle, indent=2)

    def save_to_disk(self, path: str) -> None:
        shutil.copytree(self.folder, path, dirs_exist_ok=True)


def validate_staged_arrow(staging: Path, row_count: int, features) -> None:
    """Read record batches with ordinary file handles, never whole-cache memory maps."""
    import pyarrow as arrow
    from datasets import DatasetInfo

    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    info = DatasetInfo.from_directory(str(staging))
    with (staging / "state.json").open(encoding="utf-8") as handle:
        state = json.load(handle)
    if info.features != features:
        raise ValueError("preprocess staged Arrow does not match")
    count = 0
    for entry in state["_data_files"]:
        path = staging / entry["filename"]
        enforce_under_cwd_and_no_symlink(str(path), "preprocess staged shard")
        if path.parent != staging:
            raise ValueError("preprocess staged shard must be inside staging")
        with arrow.OSFile(str(path), "rb") as handle:
            with arrow.ipc.open_stream(handle) as reader:
                if not reader.schema.equals(features.arrow_schema, check_metadata=False):
                    raise ValueError("preprocess staged Arrow does not match")
                for batch in reader:
                    batch.validate(full=True)
                    count += batch.num_rows
    if count != row_count:
        raise ValueError("preprocess staged Arrow does not match")
