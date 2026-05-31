"""v0.53.9 #94 — Module-level training event buffer for SSE streaming.

Thread-safe deque of `TrainEvent` payloads. The `SoupTrainerCallback`
pushes events on log / save / evaluate; the FastAPI `/api/train/stream`
endpoint drains pending events as SSE frames.

Decoupled from FastAPI so callbacks can push without importing it.
"""

from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Any, Deque, List, Optional

from soup_cli.utils.sse_train_stream import TrainEvent

_MAX_EVENTS = 1000


class TrainEventBuffer:
    """Thread-safe ring buffer of `TrainEvent` objects.

    `push` appends one event (dropping the oldest on overflow);
    `drain` returns all events queued since the last drain. Consumers
    can also pass a `since` cursor for catch-up subscribers.
    """

    def __init__(self, *, maxlen: int = _MAX_EVENTS) -> None:
        if isinstance(maxlen, bool) or not isinstance(maxlen, int):
            raise TypeError("maxlen must be int")
        if maxlen <= 0:
            raise ValueError("maxlen must be > 0")
        self._events: Deque[TrainEvent] = deque(maxlen=maxlen)
        self._cursor: int = 0
        self._lock = Lock()

    def push(self, event: TrainEvent) -> int:
        """Append an event. Returns the event's monotonic cursor id."""
        if not isinstance(event, TrainEvent):
            raise TypeError("event must be TrainEvent")
        with self._lock:
            self._cursor += 1
            self._events.append(event)
            return self._cursor

    def drain(self) -> List[TrainEvent]:
        """Return every queued event and clear the buffer.

        Note: destructive — concurrent subscribers will starve each other.
        For multi-subscriber SSE, use `snapshot_since(cursor)` instead.
        """
        with self._lock:
            data = list(self._events)
            self._events.clear()
            return data

    def cursor(self) -> int:
        """Return the latest assigned cursor id (0 if empty)."""
        with self._lock:
            return self._cursor

    def snapshot_since(self, since: int) -> tuple[List[TrainEvent], int]:
        """Return events with cursor > `since`, plus the new cursor.

        Non-destructive — every subscriber gets its own catch-up view.
        Returns `(events, latest_cursor)`. When the buffer has rolled over
        past `since` (i.e. the requested cursor is older than the oldest
        retained event), returns every retained event.
        """
        if isinstance(since, bool) or not isinstance(since, int):
            raise TypeError("since must be int")
        with self._lock:
            latest = self._cursor
            events = list(self._events)
        if since >= latest:
            return [], latest
        # Cursor ids are monotonic; the i-th retained event has cursor
        # `latest - (len - 1 - i)`. Slice off the prefix the caller already saw.
        start = latest - len(events) + 1  # cursor id of events[0]
        offset = max(0, since - start + 1)
        return events[offset:], latest

    def snapshot(self, *, limit: Optional[int] = None) -> List[TrainEvent]:
        """Return a copy of the latest `limit` events without clearing."""
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int):
                raise TypeError("limit must be int or None")
            if limit < 0:
                raise ValueError("limit must be >= 0")
        with self._lock:
            data = list(self._events)
        if limit is None:
            return data
        if limit == 0:
            return []
        return data[-limit:]

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


# Module-level singleton consumed by SoupTrainerCallback and the FastAPI
# SSE endpoint. Tests can call `reset_global_buffer()` to start fresh.
_GLOBAL_BUFFER: TrainEventBuffer = TrainEventBuffer()


def get_global_buffer() -> TrainEventBuffer:
    """Return the process-wide event buffer."""
    return _GLOBAL_BUFFER


def reset_global_buffer() -> None:
    """Replace the process-wide buffer (test hook)."""
    global _GLOBAL_BUFFER
    _GLOBAL_BUFFER = TrainEventBuffer()


def push_train_event(event: Any) -> Optional[int]:
    """Push an event onto the global buffer. Returns the cursor id.

    Silently ignores non-TrainEvent input so a misbehaving callback can
    never crash the training loop. Use the typed `get_global_buffer().push`
    when you want explicit validation.
    """
    if not isinstance(event, TrainEvent):
        return None
    return _GLOBAL_BUFFER.push(event)
