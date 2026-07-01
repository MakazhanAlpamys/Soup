"""Append-only JSONL request log for ``soup serve --trace-log``.

One JSON object per request with shape::

    {"prompt": ..., "response": ..., "latency_ms": ..., "tokens": ..., "ts": ...}

Hard rotation cap of ``cap_mb`` (default 100 MB). When the active file
exceeds the cap, it is renamed ``<path>.1`` (one backup retained) and a
new active file is started. The writer is thread-safe — FastAPI runs
sync handlers in a threadpool, and the request log is shared.

Path containment via shared :func:`soup_cli.utils.paths.is_under_cwd`
prevents ``--trace-log /etc/cron.d/x`` writes (mirrors v0.30.0 / v0.32.0
policy).

Added in v0.40.3 (#33 (b)).
"""

from __future__ import annotations

import json
import os
import re
import stat
import threading
import time
from pathlib import Path
from typing import Any, Optional

from soup_cli.utils.paths import is_under_cwd

# Mirror v0.34.0 crash.py policy: redact common secret shapes BEFORE writing
# request/response strings to disk. Operators frequently embed tokens in
# system prompts or chat-template stubs; logging them verbatim is a leak.
_SECRET_RE = re.compile(
    # `Bearer` token body excludes `.` so an end-of-sentence period is not
    # silently consumed (e.g. `Use Bearer abc12345xyz.` keeps the period).
    r"(?:hf_[A-Za-z0-9]{8,}"
    r"|sk-[A-Za-z0-9_-]{16,}"
    r"|Bearer\s+[A-Za-z0-9_\-]{8,})"
)


def _redact_secrets(text: str) -> str:
    """Replace common token shapes with ``<redacted>``."""
    if not isinstance(text, str):
        return text
    return _SECRET_RE.sub("<redacted>", text)


def _redact_value(value: Any) -> Any:
    """Recursively walk and redact any nested string secrets."""
    if isinstance(value, str):
        return _redact_secrets(value)
    if isinstance(value, dict):
        return {k: _redact_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_value(v) for v in value]
    return value


# Public alias so other subsystems (e.g. the v0.71.26 reward-hack mitigation
# log) reuse the ONE redaction implementation instead of duplicating the regex.
redact_value = _redact_value

DEFAULT_CAP_MB: int = 100

_MIN_CAP_MB: int = 1
_MAX_CAP_MB: int = 10_000


class TraceLogWriter:
    """Thread-safe append-only JSONL writer with size-based rotation."""

    def __init__(self, path: str, *, cap_mb: int = DEFAULT_CAP_MB) -> None:
        if isinstance(cap_mb, bool) or not isinstance(cap_mb, int):
            raise TypeError("cap_mb must be int")
        if not (_MIN_CAP_MB <= cap_mb <= _MAX_CAP_MB):
            raise ValueError(
                f"cap_mb must be in [{_MIN_CAP_MB}, {_MAX_CAP_MB}], got {cap_mb}"
            )
        if not isinstance(path, str) or not path or "\x00" in path:
            raise ValueError("path must be a non-empty string with no null bytes")
        if not is_under_cwd(path):
            raise ValueError(f"--trace-log path must stay under cwd: {path}")
        resolved = Path(os.path.realpath(path))
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self._path = resolved
        self._cap_bytes = cap_mb * 1024 * 1024
        # Single-process lock: protects rotate + write within ONE serve
        # process. Multi-worker (e.g. `--workers 4`) deployments writing to
        # the same path would race on rotation; the OS-level rename is
        # best-effort and one missed rotation cycle is acceptable for a
        # passive log. Documented limitation.
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def cap_bytes(self) -> int:
        return self._cap_bytes

    def record(
        self,
        *,
        prompt: str,
        response: str,
        latency_ms: float,
        tokens: int,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Append one entry. Never raises on serialisation issues."""
        entry: dict[str, Any] = {
            "ts": time.time(),
            "prompt": _redact_secrets(str(prompt)),
            "response": _redact_secrets(str(response)),
            "latency_ms": float(latency_ms),
            "tokens": int(tokens),
        }
        if extra:
            for key, value in extra.items():
                if key in entry:
                    continue
                # Defence in depth — redact secrets in caller-supplied
                # extras too (e.g. `extra={"system_prompt": ...}`).
                entry[str(key)] = _redact_value(value)
        try:
            line = json.dumps(entry, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return  # drop unserialisable entries silently — passive log
        line_bytes = (line + "\n").encode("utf-8")
        with self._lock:
            self._maybe_rotate(extra=len(line_bytes))
            try:
                with self._path.open("ab") as fh:
                    fh.write(line_bytes)
            except OSError:
                # Disk full / permissions — never crash the request handler.
                return

    def _maybe_rotate(self, *, extra: int) -> None:
        try:
            current = self._path.stat().st_size
        except OSError:
            return
        if current + extra <= self._cap_bytes:
            return
        backup = self._path.with_suffix(self._path.suffix + ".1")
        try:
            # Mirror v0.33.0 #22 TOCTOU policy: refuse to overwrite a symlink
            # at the backup path. An attacker who pre-places <log>.1 as a
            # symlink to /etc/cron.d/x would otherwise have the active log
            # renamed onto that target.
            try:
                backup_stat = os.lstat(backup)
                if stat.S_ISLNK(backup_stat.st_mode):
                    return  # refuse to overwrite a symlink
                backup.unlink()
            except FileNotFoundError:
                pass
            self._path.rename(backup)
        except OSError:
            # rename failure (Windows file-lock / antivirus / cross-volume)
            # is non-fatal: rotation is best-effort.
            return
