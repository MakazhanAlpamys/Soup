"""v0.44.0 Part D — `soup merge-sharded-fsdp-weights` consolidator.

Schema-only stub: validates the shard-directory layout and returns an
operation plan. Live consolidation (loading each shard via torch + writing a
single `.safetensors`) is deferred to v0.44.1 to keep the v0.44.0 surface
small.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Tuple

from soup_cli.utils.paths import is_under_cwd

_SHARD_RE = re.compile(r"^pytorch_model_fsdp_\d+(_\d+)?\.bin$")
_MAX_SHARDS = 1024


@dataclass(frozen=True)
class ConsolidationPlan:
    """What `merge-sharded-fsdp-weights` would do.

    `shard_files` is a `tuple` so the frozen dataclass is genuinely
    immutable (matches v0.32.0 / v0.39.0 / v0.43.0 frozen-collection
    policy).
    """

    shard_dir: str
    shard_files: Tuple[str, ...]
    output_path: str


def discover_shards(shard_dir: str) -> List[str]:
    """List FSDP shard files in `shard_dir`. Returns sorted basenames."""
    if not isinstance(shard_dir, str):
        raise TypeError("shard_dir must be str")
    if not is_under_cwd(shard_dir):
        raise ValueError(
            f"shard_dir is outside cwd: {os.path.basename(shard_dir)}"
        )
    real = os.path.realpath(shard_dir)
    if not os.path.isdir(real):
        raise FileNotFoundError(f"shard_dir not found: {os.path.basename(real)}")
    shards = []
    for entry in sorted(os.listdir(real)):
        if _SHARD_RE.match(entry):
            shards.append(entry)
        if len(shards) > _MAX_SHARDS:
            raise RuntimeError(
                f"too many shards (>{_MAX_SHARDS}); refuse to plan"
            )
    return shards


def plan_consolidation(shard_dir: str, output_path: str) -> ConsolidationPlan:
    """Build a `ConsolidationPlan`. Raises on missing shards or bad output."""
    if not isinstance(output_path, str) or not output_path:
        raise ValueError("output_path must be non-empty str")
    if "\x00" in output_path:
        raise ValueError("output_path contains NUL byte")
    if not output_path.endswith(".safetensors"):
        raise ValueError("output_path must end in .safetensors")
    if not is_under_cwd(output_path):
        raise ValueError(
            f"output_path is outside cwd: {os.path.basename(output_path)}"
        )
    shards = discover_shards(shard_dir)
    if not shards:
        raise FileNotFoundError(
            "no FSDP shard files (pytorch_model_fsdp_*.bin) found in shard_dir"
        )
    return ConsolidationPlan(
        shard_dir=os.path.realpath(shard_dir),
        shard_files=tuple(shards),
        output_path=os.path.realpath(output_path),
    )
