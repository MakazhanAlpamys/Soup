"""Server-side metrics for the continuous-batching dashboard (v0.30.0).

Pure-Python, thread-safe via a single lock. Exposed as JSON via the
``/metrics`` endpoint and rendered as a Rich dashboard when ``--dashboard``
is set.
"""

from __future__ import annotations

import threading
from collections import deque
from contextlib import contextmanager
from typing import Any, Deque, Iterator, List

_MAX_LATENCY_SAMPLES = 1000


def _percentile(sorted_values: List[float], pct: float) -> float:
    """Nearest-rank percentile (0 <= pct <= 100). Returns 0.0 on empty."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = max(0, min(len(sorted_values) - 1, int(round((pct / 100) * (len(sorted_values) - 1)))))
    return sorted_values[idx]


class ServerMetrics:
    """Thread-safe inference server metrics.

    Tracks:
    - requests_total: total requests handled (success + failure)
    - tokens_generated_total: cumulative completion tokens
    - active_requests: currently in-flight requests
    - latency_p50_ms / latency_p95_ms: nearest-rank percentiles over the last
      ``_MAX_LATENCY_SAMPLES`` measurements.

    Latencies are kept in insertion-order in a bounded deque, sorted on
    snapshot read. Insert is O(1); snapshot is O(n log n) but only runs when
    /metrics is scraped.
    """

    __slots__ = (
        "_active",
        "_latencies",
        "_lock",
        "_requests",
        "_tokens",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: int = 0
        self._tokens: int = 0
        self._active: int = 0
        # Ring buffer: last _MAX_LATENCY_SAMPLES measurements, insertion-order.
        self._latencies: Deque[float] = deque(maxlen=_MAX_LATENCY_SAMPLES)

    def record_tokens(self, count: int) -> None:
        """Add to the cumulative completion-token counter.

        ``count == 0`` is allowed (valid no-op for empty-output generations).
        """
        if count < 0:
            raise ValueError(f"token count must be non-negative, got {count}")
        with self._lock:
            self._tokens += count

    def record_latency(self, latency_ms: float) -> None:
        """Append a latency sample; older samples automatically age out."""
        if latency_ms < 0:
            raise ValueError(f"latency must be non-negative, got {latency_ms}")
        with self._lock:
            self._latencies.append(float(latency_ms))

    @contextmanager
    def track_request(self) -> Iterator[None]:
        """Context manager: increments active on enter, decrements + counts
        on exit. Counts the request even if the body raised — lets the
        dashboard show true failure rate.
        """
        with self._lock:
            self._active += 1
        try:
            yield
        finally:
            with self._lock:
                self._active = max(0, self._active - 1)
                self._requests += 1

    def snapshot(self) -> dict[str, Any]:
        """Return a point-in-time copy of all metrics."""
        with self._lock:
            sample_snapshot = list(self._latencies)
            requests = self._requests
            tokens = self._tokens
            active = self._active
        # Sort outside the lock so scrapes don't block request handlers.
        sample_snapshot.sort()
        return {
            "requests_total": requests,
            "tokens_generated_total": tokens,
            "active_requests": active,
            "latency_p50_ms": _percentile(sample_snapshot, 50),
            "latency_p95_ms": _percentile(sample_snapshot, 95),
            "latency_samples": len(sample_snapshot),
        }
