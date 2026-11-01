"""Document-isolated next-token windows for native MLX continued pretraining."""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from soup_cli.config.schema import DataConfig


@dataclass(frozen=True)
class TokenWindow:
    """One document's tokens; the first is context, every remaining token is a target."""

    tokens: tuple[int, ...]
    document: int


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_pretrain_corpus(config: DataConfig) -> dict:
    """Use the existing local readers and diagnostics, splitting documents before windows."""
    from soup_cli.data.loader import SkippedRows, _format_rows, _split_val, load_raw_data

    started = time.perf_counter()

    def read(source: str) -> tuple[list[dict], dict]:
        path = Path(source)
        if path.suffix.lower() not in {".txt", ".jsonl"}:
            raise ValueError("MLX pretrain requires a local .txt or .jsonl corpus")
        skipped = SkippedRows()
        raw = load_raw_data(path, issues=skipped)
        rows = _format_rows(raw, "plaintext", source=str(path))
        return rows, {
            "source_documents": len(raw) + skipped.total,
            "accepted_documents": len(rows),
            "dropped_documents": len(raw) + skipped.total - len(rows),
            "source_sha256": file_sha256(path),
        }

    train, source_stats = read(config.train)
    validation = getattr(config, "validation", None)
    if validation is not None:
        val, validation_stats = read(validation)
    elif config.val_split:
        train, val = _split_val(train, config.val_split)
        validation_stats = None
    else:
        val, validation_stats = [], None
    if not train:
        raise ValueError("MLX pretrain: the training split has no usable raw-text documents")
    if (validation is not None or config.val_split > 0) and not val:
        raise ValueError("MLX pretrain: the required validation split has no usable documents")
    return {
        "train": train,
        "val": val,
        "pretrain_data_stats": {
            "source": source_stats,
            "validation_source": validation_stats,
            "train_documents": len(train),
            "validation_documents": len(val),
            "split_policy": "explicit file" if validation else "document-level before windows",
            "file_preparation_seconds": time.perf_counter() - started,
        },
    }


def prepare_windows(
    rows: list[dict], tokenizer: Any, max_length: int, *, split: str = "train",
) -> tuple[list[TokenWindow], dict]:
    """Honor tokenizer special tokens; append EOS once; retain every within-document pair."""
    if type(max_length) is not int or max_length < 2:
        raise ValueError("MLX pretrain max_length must be an integer of at least 2")
    windows: list[TokenWindow] = []
    tokenized = targetless = context_only = 0
    for index, row in enumerate(rows):
        text = row.get("text") if isinstance(row, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"MLX pretrain {split} document {index}: text must be nonempty")
        ids = list(tokenizer.encode(text, add_special_tokens=True))
        if any(type(token) is not int or token < 0 for token in ids):
            raise ValueError(f"MLX pretrain {split} document {index}: invalid token IDs")
        eos = tokenizer.eos_token_id
        if eos is not None and (not ids or ids[-1] != eos):
            ids.append(int(eos))
        tokenized += len(ids)
        context_only += bool(ids)
        if len(ids) < 2:
            targetless += 1
            continue
        for start in range(0, len(ids) - 1, max_length - 1):
            windows.append(TokenWindow(tuple(ids[start:start + max_length]), index))
    if not windows:
        raise ValueError(f"MLX pretrain {split} split has no supervised next-token targets")
    return windows, {
        "documents": len(rows),
        "targetless_documents": targetless,
        "tokenized_tokens": tokenized,
        "context_only_tokens": context_only,
        "supervised_tokens": sum(len(window.tokens) - 1 for window in windows),
        "windows": len(windows),
        "max_window_tokens": max(len(window.tokens) for window in windows),
    }


def batch_windows(
    windows: list[TokenWindow], pad_id: int, *, width: int | None = None,
) -> tuple[list[list[int]], list[list[int]], list[list[float]]]:
    """Shift exactly once, masking padding by position even when PAD equals EOS."""
    if not windows or any(len(window.tokens) < 2 for window in windows):
        raise ValueError("MLX pretrain batch must contain supervised targets")
    needed = max(len(window.tokens) - 1 for window in windows)
    width = needed if width is None else width
    if width < needed:
        raise ValueError("MLX pretrain batch width would truncate supervised targets")
    inputs, targets, masks = [], [], []
    for window in windows:
        tokens = list(window.tokens)
        count = len(tokens) - 1
        inputs.append(tokens[:-1] + [pad_id] * (width - count))
        targets.append(tokens[1:] + [pad_id] * (width - count))
        masks.append([1.0] * count + [0.0] * (width - count))
    return inputs, targets, masks


def plan_updates(
    windows: int, batch_size: int, accumulation: int, epochs: int, seed: int,
) -> list[tuple[int, tuple[tuple[int, ...], ...]]]:
    """Shuffle per epoch; consume every window and flush each epoch's partial update."""
    for name, value in {"windows": windows, "batch_size": batch_size,
                        "accumulation": accumulation, "epochs": epochs}.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"MLX pretrain {name} must be a positive integer")
    rng = random.Random(seed)
    plan = []
    for epoch in range(1, epochs + 1):
        order = list(range(windows))
        rng.shuffle(order)
        batches = [tuple(order[start:start + batch_size])
                   for start in range(0, windows, batch_size)]
        plan.extend((epoch, tuple(batches[start:start + accumulation]))
                    for start in range(0, len(batches), accumulation))
    return plan
