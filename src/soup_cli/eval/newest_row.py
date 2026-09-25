"""Pick the newest eval_results row per benchmark.

Used by registry baselines, ``soup registry diff``, and ``soup eval compare``
so those call sites cannot drift apart on "oldest vs newest" again
(see #1224 / #1270).
"""

from __future__ import annotations

from typing import Any, Mapping


def _row_sort_key(row: Mapping[str, Any]) -> tuple[str, int]:
    """Sort key: newer ``created_at`` first; ties break by higher ``id`` (insertion)."""
    created = row.get("created_at") or ""
    raw_id = row.get("id")
    try:
        row_id = int(raw_id) if raw_id is not None else 0
    except (TypeError, ValueError):
        row_id = 0
    return (str(created), row_id)


def newest_row_per_benchmark(rows: list[dict]) -> dict[str, dict]:
    """Keep the newest scored row for each benchmark.

    Ordering is computed here (``created_at`` desc, then ``id`` desc) so callers
    need not depend on SQL ``ORDER BY``. The first surviving row per benchmark
    is the one that counts.
    """
    ordered = sorted(rows, key=_row_sort_key, reverse=True)
    newest: dict[str, dict] = {}
    for row in ordered:
        name = row.get("benchmark")
        if not name or row.get("score") is None or name in newest:
            continue
        newest[str(name)] = row
    return newest


def newest_score_per_benchmark(rows: list[dict]) -> dict[str, float]:
    """Newest score per benchmark (same selection rules as :func:`newest_row_per_benchmark`)."""
    return {name: float(row["score"]) for name, row in newest_row_per_benchmark(rows).items()}
