"""v0.44.0 Part A — Tool Outputs panel + tool-call timer.

Tracks tool invocations during a tool-calling SFT run. Pure-Python; the
`record_call` API can be plumbed from any trainer callback that observes
`tool_calls`.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Deque, List, Optional

# Bound the in-memory record buffer (Tool Outputs panel only shows latest N).
_MAX_RECORDS = 1000
_MAX_NAME_LEN = 128
_MAX_OUTPUT_LEN = 4096


def _validate_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("name must be str")
    if not name:
        raise ValueError("name must be non-empty")
    if "\x00" in name:
        raise ValueError("name contains NUL byte")
    if len(name) > _MAX_NAME_LEN:
        raise ValueError(f"name exceeds {_MAX_NAME_LEN} chars")
    return name


@dataclass(frozen=True)
class ToolCallRecord:
    """One tool invocation with timing and truncated output."""

    name: str
    started_ts: float
    duration_ms: float
    success: bool
    output_preview: str
    error: Optional[str] = None


@dataclass
class ToolOutputsBuffer:
    """Thread-safe, capped ring of `ToolCallRecord` entries.

    Uses `collections.deque(maxlen=_MAX_RECORDS)` so the buffer drops the
    oldest record on overflow without an O(N) list slice. The `records`
    field is kept as the public surface but exposed as a deque for the same
    reason.
    """

    records: Deque[ToolCallRecord] = field(
        default_factory=lambda: deque(maxlen=_MAX_RECORDS)
    )
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def record_call(
        self,
        *,
        name: str,
        started_ts: float,
        duration_ms: float,
        success: bool,
        output_preview: str,
        error: Optional[str] = None,
    ) -> ToolCallRecord:
        _validate_name(name)
        if isinstance(started_ts, bool) or not isinstance(started_ts, (int, float)):
            raise TypeError("started_ts must be a number")
        if not math.isfinite(float(started_ts)):
            raise ValueError("started_ts must be finite")
        if isinstance(duration_ms, bool) or not isinstance(
            duration_ms, (int, float)
        ):
            raise TypeError("duration_ms must be a number")
        if not math.isfinite(float(duration_ms)) or float(duration_ms) < 0.0:
            raise ValueError("duration_ms must be finite and >= 0")
        if not isinstance(success, bool):
            raise TypeError("success must be bool")
        if not isinstance(output_preview, str):
            raise TypeError("output_preview must be str")
        # Truncate to bound memory; never raise.
        truncated = output_preview[:_MAX_OUTPUT_LEN]
        if error is not None:
            if not isinstance(error, str):
                raise TypeError("error must be str or None")
            if len(error) > _MAX_OUTPUT_LEN:
                error = error[:_MAX_OUTPUT_LEN]
        record = ToolCallRecord(
            name=name,
            started_ts=float(started_ts),
            duration_ms=float(duration_ms),
            success=success,
            output_preview=truncated,
            error=error,
        )
        with self._lock:
            self.records.append(record)
        return record

    def snapshot(self, *, limit: Optional[int] = None) -> List[ToolCallRecord]:
        """Return a copy of the latest `limit` records (None = all)."""
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int):
                raise TypeError("limit must be int or None")
            if limit < 0:
                raise ValueError("limit must be >= 0")
        with self._lock:
            data = list(self.records)
        if limit is None:
            return data
        if limit == 0:
            return []
        return data[-limit:]

    def clear(self) -> None:
        with self._lock:
            self.records.clear()


class ToolCallTimer:
    """Context manager that times a tool invocation and records the result."""

    def __init__(self, buffer: ToolOutputsBuffer, *, name: str) -> None:
        self._buffer = buffer
        self._name = _validate_name(name)
        self._start_perf: float = 0.0
        self._start_wall: float = 0.0
        self._output: str = ""
        self._error: Optional[str] = None
        self._success: bool = True

    def set_output(self, text: str) -> None:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        self._output = text

    def set_error(self, text: str) -> None:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        self._error = text
        self._success = False

    def __enter__(self) -> "ToolCallTimer":
        self._start_perf = time.perf_counter()
        self._start_wall = time.time()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc: Optional[BaseException],
        tb: object,
    ) -> bool:
        if exc is not None:
            self._success = False
            self._error = (
                f"{exc_type.__name__ if exc_type else 'Exception'}: {exc}"
            )
        duration_ms = (time.perf_counter() - self._start_perf) * 1000.0
        self._buffer.record_call(
            name=self._name,
            started_ts=self._start_wall,
            duration_ms=duration_ms,
            success=self._success,
            output_preview=self._output,
            error=self._error,
        )
        # Don't suppress exceptions — `False` is explicit per project policy.
        return False


# v0.53.9 #100 — Module-level singleton consumed by the SFT trainer
# tool-calling callback and the FastAPI `/api/tool-outputs` endpoint.
_GLOBAL_TOOL_BUFFER: ToolOutputsBuffer = ToolOutputsBuffer()


def get_global_tool_buffer() -> ToolOutputsBuffer:
    """Return the process-wide tool-output buffer."""
    return _GLOBAL_TOOL_BUFFER


def reset_global_tool_buffer() -> None:
    """Replace the process-wide buffer (test hook)."""
    global _GLOBAL_TOOL_BUFFER
    _GLOBAL_TOOL_BUFFER = ToolOutputsBuffer()
