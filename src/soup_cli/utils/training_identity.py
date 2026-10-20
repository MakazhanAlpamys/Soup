"""Bounded startup identity of one local raw text-training source file.

This describes source bytes at capture time, not transformed examples consumed.
Unsupported, changing or incompletely read inputs have no digest.
"""

from __future__ import annotations

import hashlib
import os
import stat
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from soup_cli.config.schema import SoupConfig

MAX_HASH_BYTES = 256 * 1024 * 1024
HASH_TIMEOUT_SECONDS = 5.0
_CHUNK_BYTES = 1024 * 1024


def _identity(info: os.stat_result) -> tuple[int, ...]:
    """Metadata with the same meaning for path and descriptor APIs."""
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _signature(info: os.stat_result) -> tuple[int, ...]:
    """Compare ctime only within one API; Windows path/fd values can differ."""
    return _identity(info) + (info.st_ctime_ns,)


def capture_training_data_hash(
    config: SoupConfig, *, max_bytes: int = MAX_HASH_BYTES,
    timeout_seconds: float = HASH_TIMEOUT_SECONDS,
) -> str | None:
    """Hash complete supported bytes, or return unknown without changing training.

    The time budget is checked between bounded reads; it cannot interrupt a
    blocking filesystem operation. No network dataset is opened. Replaced or
    modified files are detected by descriptor/path metadata checks, which are
    not a filesystem snapshot or protection against deliberate timestamp replay.
    """
    from soup_cli.data.loader import SUPPORTED_EXTENSIONS

    data = config.data
    source = data.train
    if (config.modality != "text" or config.task in {"unlearn", "tts"}
            or not isinstance(source, str) or not source or "://" in source
            or source.startswith(("\\\\", "//"))
            or data.format == "pre_tokenized" or data.tokenized_path
            or data.replay or data.forget_set or data.retain_set
            or data.interleave is not None or data.streaming):
        return None
    if Path(source).suffix.lower() not in SUPPORTED_EXTENSIONS:
        return None
    if max_bytes < 0 or timeout_seconds <= 0:
        return None
    started = time.monotonic()
    descriptor = None
    try:
        initial = os.stat(source)
        if not stat.S_ISREG(initial.st_mode) or initial.st_size > max_bytes:
            return None
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(source, flags)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None  # the context owns it, including on early return
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode) or _identity(initial) != _identity(opened):
                return None
            digest = hashlib.sha256()
            count = 0
            while True:
                if time.monotonic() - started >= timeout_seconds:
                    return None
                chunk = handle.read(min(_CHUNK_BYTES, max_bytes - count + 1))
                if not chunk:
                    break
                count += len(chunk)
                if count > max_bytes:
                    return None
                digest.update(chunk)
            if (time.monotonic() - started >= timeout_seconds
                    or count != opened.st_size
                    or _signature(os.fstat(handle.fileno())) != _signature(opened)
                    or _signature(os.stat(source)) != _signature(initial)):
                return None
            if time.monotonic() - started >= timeout_seconds:
                return None
            return digest.hexdigest()
    except (OSError, ValueError):
        # A provenance gap must not introduce a new refusal or expose paths.
        return None
    finally:
        if descriptor is not None:
            os.close(descriptor)
