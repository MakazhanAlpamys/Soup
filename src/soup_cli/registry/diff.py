"""Registry diff helpers: config tree diff + eval delta."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from soup_cli.eval.newest_row import newest_score_per_benchmark


@dataclass(frozen=True)
class ConfigChange:
    path: str
    kind: str  # "added" | "removed" | "changed"
    left: Any = None
    right: Any = None


def _walk(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict into ``dot.path → leaf`` pairs."""
    flat: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key in sorted(obj):
            sub = _walk(obj[key], f"{prefix}.{key}" if prefix else str(key))
            flat.update(sub)
    else:
        flat[prefix] = obj
    return flat


def config_diff(left: dict, right: dict) -> list[ConfigChange]:
    """Return a list of :class:`ConfigChange` describing left → right."""
    left_flat = _walk(left)
    right_flat = _walk(right)
    changes: list[ConfigChange] = []

    all_paths = sorted(set(left_flat) | set(right_flat))
    for path in all_paths:
        if path in left_flat and path in right_flat:
            if left_flat[path] != right_flat[path]:
                changes.append(ConfigChange(
                    path=path, kind="changed",
                    left=left_flat[path], right=right_flat[path],
                ))
        elif path in right_flat:
            changes.append(ConfigChange(
                path=path, kind="added",
                left=None, right=right_flat[path],
            ))
        else:
            changes.append(ConfigChange(
                path=path, kind="removed",
                left=left_flat[path], right=None,
            ))
    return changes


def eval_delta(
    left: list[dict], right: list[dict],
) -> list[dict]:
    """Compute per-benchmark delta given two eval_results lists.

    When a side has multiple rows for one benchmark, the newest score wins
    (``created_at``, then ``id``) — same rule as the eval gate (#1270).
    """
    left_map = newest_score_per_benchmark(left)
    right_map = newest_score_per_benchmark(right)

    deltas: list[dict] = []
    for bench in sorted(set(left_map) | set(right_map)):
        left_score = left_map.get(bench)
        right_score = right_map.get(bench)
        if left_score is None or right_score is None:
            deltas.append({
                "benchmark": bench,
                "left": left_score,
                "right": right_score,
                "delta": None,
            })
        else:
            deltas.append({
                "benchmark": bench,
                "left": left_score,
                "right": right_score,
                "delta": float(right_score) - float(left_score),
            })
    return deltas
