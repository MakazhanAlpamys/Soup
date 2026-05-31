"""Curriculum learning utilities — sort datasets by difficulty for staged training."""

from __future__ import annotations

import json


def sort_by_length(data: list[dict]) -> list[dict]:
    """Sort dataset rows by text length (short → long).

    Supports both 'text' field and 'messages' format.
    """
    def _row_length(row: dict) -> int:
        if "text" in row:
            return len(str(row["text"]))
        if "messages" in row:
            return sum(len(str(msg.get("content", ""))) for msg in row["messages"])
        # Fallback: stringify entire row
        return len(json.dumps(row))

    return sorted(data, key=_row_length)


def create_buckets(data: list, num_buckets: int) -> list[list]:
    """Split sorted data into N roughly equal buckets.

    Args:
        data: Pre-sorted list (easy → hard).
        num_buckets: Number of difficulty stages.

    Returns:
        List of lists, each representing a difficulty bucket.
    """
    if num_buckets <= 0:
        return [data]

    total = len(data)
    if total == 0:
        return [[] for _ in range(num_buckets)]

    bucket_size = total // num_buckets
    remainder = total % num_buckets

    buckets = []
    start = 0
    for idx in range(num_buckets):
        # Distribute remainder across first `remainder` buckets
        extra = 1 if idx < remainder else 0
        end = start + bucket_size + extra
        buckets.append(data[start:end])
        start = end

    return buckets
