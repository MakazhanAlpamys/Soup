"""v0.44.0 Part D — `soup merge-sharded-fsdp-weights` consolidator.

v0.44.0 shipped the planner (``plan_consolidation`` / ``discover_shards``).
v0.71.14 (#96) lifts the deferred runtime: ``consolidate_shards`` streams each
FSDP shard ``.bin`` (via ``torch.load(weights_only=True)`` — no arbitrary
pickle exec), unions the per-shard parameter fragments into one state-dict, and
writes a single ``.safetensors`` atomically.

Scope note: this handles per-rank **FULL_STATE_DICT** / disjoint-parameter
shards (each ``.bin`` holds complete parameters; the union of all shards is the
full model). DCP sharded-tensor checkpoints (``.distcp`` with per-rank tensor
SLICES that need concatenation from the torch.distributed.checkpoint metadata)
are out of scope — use ``accelerate merge-weights`` for those.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

from soup_cli.utils.paths import (
    atomic_write_bytes,
    enforce_under_cwd_and_no_symlink,
    is_under_cwd,
)

logger = logging.getLogger(__name__)

_SHARD_RE = re.compile(r"^pytorch_model_fsdp_\d+(_\d+)?\.bin$")
_MAX_SHARDS = 1024
# Defence-in-depth: a single shard fragment over 16 GiB is almost certainly
# a wrong file (or an attempt to OOM the box). Real per-rank shards are small.
_MAX_SHARD_BYTES = 16 * 1024**3


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


@dataclass(frozen=True)
class ConsolidationResult:
    """Outcome of a live shard consolidation."""

    output_path: str
    num_tensors: int
    num_shards: int
    total_bytes: int


def consolidate_shards(plan: ConsolidationPlan) -> ConsolidationResult:
    """Stream each FSDP shard and write a single consolidated ``.safetensors``.

    Loads one shard at a time via ``torch.load(weights_only=True)`` (no
    arbitrary pickle execution), unions the per-shard parameter fragments into
    a single CPU state-dict, and writes it atomically. Memory-friendly: the
    raw shard object is freed before the next is loaded (peak ≈ merged
    state-dict + one shard).

    Raises:
        TypeError: ``plan`` is not a :class:`ConsolidationPlan`.
        ValueError: a shard is not a state-dict, two shards disagree on a
            tensor's shape, the output path is a symlink, or no tensors were
            found across all shards.
    """
    import torch  # lazy — keep CLI startup fast
    from safetensors.torch import save as st_save

    if not isinstance(plan, ConsolidationPlan):
        raise TypeError(
            f"plan must be ConsolidationPlan, got {type(plan).__name__}"
        )
    # Re-validate the output path (symlink rejection + cwd containment) at
    # write time — TOCTOU defence, mirrors v0.59.0 atomic_write policy.
    enforce_under_cwd_and_no_symlink(plan.output_path, "output_path")

    merged: dict = {}
    duplicate_keys: List[str] = []
    for shard_name in plan.shard_files:
        shard_path = os.path.join(plan.shard_dir, shard_name)
        # The shard file is a direct child of the realpath'd shard_dir, but a
        # symlinked child could redirect torch.load to an arbitrary target —
        # reject symlinks + confirm containment before reading.
        enforce_under_cwd_and_no_symlink(shard_path, "shard")
        try:
            size = os.path.getsize(shard_path)
        except OSError as exc:
            raise ValueError(
                f"shard {shard_name} unreadable: {type(exc).__name__}"
            ) from exc
        if size > _MAX_SHARD_BYTES:
            raise ValueError(
                f"shard {shard_name} too large (>{_MAX_SHARD_BYTES} bytes); "
                "refuse to load"
            )
        try:
            state = torch.load(
                shard_path, map_location="cpu", weights_only=True
            )
        except Exception as exc:  # noqa: BLE001 — corrupt/non-torch shard
            # torch.load raises UnpicklingError / RuntimeError / EOFError /
            # BadZipFile on a corrupt or non-torch .bin. Surface a clean
            # ValueError (the CLI maps it to exit 2) instead of leaking the
            # raw pickle internals as a crash.
            raise ValueError(
                f"shard {shard_name} is not a valid torch checkpoint: "
                f"{type(exc).__name__}"
            ) from exc
        if not isinstance(state, dict):
            raise ValueError(
                f"shard {shard_name} is not a state-dict "
                f"(got {type(state).__name__})"
            )
        for key, tensor in state.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"shard {shard_name} has a non-string key {key!r}"
                )
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(
                    f"shard {shard_name} key {key!r} is not a tensor "
                    f"(got {type(tensor).__name__})"
                )
            if key in merged:
                if tuple(merged[key].shape) != tuple(tensor.shape):
                    raise ValueError(
                        f"shape conflict for {key!r}: "
                        f"{tuple(merged[key].shape)} vs {tuple(tensor.shape)}"
                    )
                # Same key + same shape across shards — FSDP replicates some
                # params; keep the first occurrence. Track it so a genuinely
                # sharded checkpoint (where ranks hold DIFFERENT values for the
                # same key) isn't silently corrupted without a trace.
                duplicate_keys.append(key)
                continue
            merged[key] = tensor.detach().to("cpu").contiguous()
        # Free the raw shard before loading the next (memory-friendly).
        del state

    if duplicate_keys:
        sample = ", ".join(duplicate_keys[:5])
        logger.warning(
            "%d key(s) appeared in more than one shard with the same shape; "
            "kept the first occurrence (e.g. %s). This is correct for "
            "replicated FULL_STATE_DICT params, but if your shards hold "
            "per-rank tensor SLICES they need concatenation — use "
            "`accelerate merge-weights` for DCP sharded checkpoints.",
            len(duplicate_keys),
            sample,
        )

    if not merged:
        raise ValueError("no tensors found across shards; nothing to write")

    payload = st_save(merged)
    written = atomic_write_bytes(payload, plan.output_path, field="output_path")
    return ConsolidationResult(
        output_path=written,
        num_tensors=len(merged),
        num_shards=len(plan.shard_files),
        total_bytes=len(payload),
    )
