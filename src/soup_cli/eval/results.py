"""Helpers for selecting the latest rows from evaluation results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def _row_order(row: Mapping[str, Any]) -> tuple[str, int]:
    """Return the database ordering used to identify the newest row."""
    created_at = row.get("created_at")
    stamp = "" if created_at is None else str(created_at)
    raw_id = row.get("rowid", row.get("id", -1))
    try:
        insertion_id = int(raw_id)
    except (TypeError, ValueError):
        insertion_id = -1
    return stamp, insertion_id


def newest_eval_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    key_fields: Sequence[str] = ("benchmark",),
) -> list[dict[str, Any]]:
    """Keep the newest scored row for each evaluation key.

    Rows are ordered by ``created_at DESC, rowid DESC``. SQLite exposes the
    ``INTEGER PRIMARY KEY`` column as both ``id`` and ``rowid``; accepting both
    names keeps this helper usable with database rows and small test doubles.
    Rows with missing or empty key fields are ignored, matching the eval gate's
    previous ``if not name`` behavior.
    """
    newest: dict[tuple[Any, ...], dict[str, Any]] = {}
    ordered = sorted(rows, key=_row_order, reverse=True)
    for row in ordered:
        if row.get("score") is None:
            continue
        key = tuple(row.get(field) for field in key_fields)
        if any(value is None or value == "" for value in key) or key in newest:
            continue
        newest[key] = dict(row)
    return list(newest.values())
