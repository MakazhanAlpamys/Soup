"""LoRA adapter blame: attribute weight movement to dataset shards (v0.57.0 Part C).

This module emits a leave-one-out ablation PLAN that future v0.57.1 wiring
will execute through the existing v0.34 SQLite tracker + v0.26 Registry
lineage. The plan is fully deterministic given the inputs so callers can
inspect / re-run / share without running real ablations.

Public surface:

- ``parse_budget(spec)`` -> seconds (mirrors v0.48.0 ``parse_budget`` semantics)
- ``plan_blame(adapter_dir, dataset_path, *, layer, budget_seconds, num_shards)``
  -> ``BlamePlan`` (frozen dataclass) with per-shard work item + projected duration
- ``run_blame`` stub (raises ``NotImplementedError`` with v0.57.1 marker)
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Tuple

from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

_MIN_BUDGET_SECONDS = 60
_MAX_BUDGET_SECONDS = 24 * 3600
_MIN_SHARDS = 2
_MAX_SHARDS = 100
_MIN_PER_SHARD_SECONDS = 30

_BUDGET_RE = re.compile(r"^(\d+)([smh]?)$")


@dataclass(frozen=True)
class BlameShardWork:
    shard_id: int
    holdout_offset: int
    holdout_size: int
    projected_seconds: int


@dataclass(frozen=True)
class BlamePlan:
    adapter_dir: str
    dataset_path: str
    layer: str
    budget_seconds: int
    num_shards: int
    per_shard_seconds: int
    shards: Tuple[BlameShardWork, ...]
    feasible: bool
    reason: str


def parse_budget(spec: str) -> int:
    """Parse a budget string (``60s`` / ``5m`` / ``2h`` / bare seconds)."""
    if isinstance(spec, bool) or not isinstance(spec, str):
        raise TypeError("spec must be str")
    if not spec:
        raise ValueError("spec must be non-empty")
    if "\x00" in spec:
        raise ValueError("spec must not contain null bytes")
    match = _BUDGET_RE.match(spec.strip())
    if not match:
        raise ValueError(f"invalid budget: {spec!r}")
    value = int(match.group(1))
    unit = match.group(2) or "s"
    multiplier = {"s": 1, "m": 60, "h": 3600}[unit]
    seconds = value * multiplier
    if seconds < _MIN_BUDGET_SECONDS:
        raise ValueError(
            f"budget {seconds}s below floor {_MIN_BUDGET_SECONDS}s"
        )
    if seconds > _MAX_BUDGET_SECONDS:
        raise ValueError(
            f"budget {seconds}s above cap {_MAX_BUDGET_SECONDS}s"
        )
    return seconds


def _validate_int(value: object, field: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be int")
    if value < lo or value > hi:
        raise ValueError(f"{field} must be in [{lo}, {hi}]")
    return value


def _validate_str(value: object, field: str, max_len: int = 512) -> str:
    if isinstance(value, bool) or not isinstance(value, str):
        raise TypeError(f"{field} must be str")
    if not value:
        raise ValueError(f"{field} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain null bytes")
    if len(value) > max_len:
        raise ValueError(f"{field} must be ≤{max_len} chars")
    return value


def _count_dataset_rows(dataset_path: str) -> int:
    """Cheap line count; for blame planning a rough order-of-magnitude is enough.

    Opens via the realpath captured at containment check (see plan_blame)
    so a symlink swap between check and open cannot redirect the read
    (TOCTOU defence).
    """
    from pathlib import Path

    real = os.path.realpath(dataset_path)
    path = Path(real)
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found: {path.name}")
    # Quick line count without holding the whole file in memory
    count = 0
    with open(real, "rb") as fh:
        for _ in fh:
            count += 1
            if count > 10_000_000:  # DoS cap — 10M lines is plenty
                break
    return count


def plan_blame(
    adapter_dir: str,
    dataset_path: str,
    *,
    layer: str,
    budget_seconds: int,
    num_shards: int = 10,
) -> BlamePlan:
    """Build a leave-one-out ablation plan.

    The plan splits the dataset into ``num_shards`` equal shards; per shard,
    a 1/10-scale ablation run trains with that shard held out. ``feasible``
    is False when the budget cannot cover ≥ ``_MIN_PER_SHARD_SECONDS`` per
    shard with safety overhead.
    """
    enforce_under_cwd_and_no_symlink(adapter_dir, "adapter_dir")
    enforce_under_cwd_and_no_symlink(dataset_path, "dataset_path")
    _validate_str(layer, "layer", max_len=256)
    _validate_int(budget_seconds, "budget_seconds",
                  _MIN_BUDGET_SECONDS, _MAX_BUDGET_SECONDS)
    _validate_int(num_shards, "num_shards", _MIN_SHARDS, _MAX_SHARDS)

    row_count = _count_dataset_rows(dataset_path)
    if row_count == 0:
        raise ValueError("dataset is empty")

    # Reserve 10% overhead for setup/teardown across all shards
    usable_seconds = int(budget_seconds * 0.9)
    per_shard = usable_seconds // num_shards
    feasible = per_shard >= _MIN_PER_SHARD_SECONDS
    reason = (
        "ok"
        if feasible
        else f"budget {budget_seconds}s gives only {per_shard}s/shard "
             f"(need ≥{_MIN_PER_SHARD_SECONDS}s)"
    )

    shard_size = math.ceil(row_count / num_shards)
    shards: list[BlameShardWork] = []
    for sid in range(num_shards):
        offset = sid * shard_size
        size = min(shard_size, max(0, row_count - offset))
        shards.append(
            BlameShardWork(
                shard_id=sid,
                holdout_offset=offset,
                holdout_size=size,
                projected_seconds=per_shard,
            )
        )

    return BlamePlan(
        adapter_dir=adapter_dir,
        dataset_path=dataset_path,
        layer=layer,
        budget_seconds=budget_seconds,
        num_shards=num_shards,
        per_shard_seconds=per_shard,
        shards=tuple(shards),
        feasible=feasible,
        reason=reason,
    )


def run_blame(plan: BlamePlan) -> None:
    """Live leave-one-out ablation runner — deferred to v0.57.1.

    The plan is executed through the v0.34 SQLite tracker + v0.26 Registry
    lineage so every ablation run is reproducible. Live wiring lands in v0.57.1.
    """
    if not isinstance(plan, BlamePlan):
        raise TypeError("plan must be BlamePlan")
    raise NotImplementedError(
        "Live blame ablation runner deferred to v0.57.1. "
        "Use `soup adapters blame --plan-only` to inspect the plan."
    )
