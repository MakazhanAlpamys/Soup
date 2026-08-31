"""Per-rank FSDP sharding probe (STEP 20 of gate-h100-validation.md), repaired.

This is the probe that turned "the flag was accepted" into "FSDP engaged" at
0.5B: every FSDP arm reported 124,048,864 params/rank = 496,195,456 / 4
exactly, against a single-GPU control reporting the full 496,195,456 locally.

At 70B the original raised
``TypeError('FullyShardedDataParallel does not support len()')`` on all eight
ranks, so no sharding claim could be made from that session
(``gate-h100-validation.md`` L4805-4809).

The mechanism
-------------
That message is not FSDP's. FSDP defines no ``__len__``; a bare one raises
Python's ``object of type 'FullyShardedDataParallel' has no len()``. The
recorded text comes from ``torch._dynamo.eval_frame.OptimizedModule.__len__``,
which names the *wrapped* class:

    if isinstance(self._orig_mod, Sized):
        return len(self._orig_mod)
    raise TypeError(f"{type(self._orig_mod).__name__} does not support len()")

So the failing object was a ``torch.compile`` wrapper around FSDP -- one
indirection deeper than the 0.5B arm, because the 70B recipe shape includes
``use_fsdp2_compile``. ``OptimizedModule.__len__`` does not exist before torch
2.11, which is why this cannot be reproduced on the torch 2.5.1 dev box that
produced the repo's other records.

Two things follow, and the second is the one that matters:

1. Never measure length while walking. Iterate children and sum ``numel()``.
2. **Unwrap ``_orig_mod`` first.** A probe that only drops the ``len()`` call
   walks without error and still mis-attributes: it counts ``OptimizedModule``
   where the 0.5B record counts ``FullyShardedDataParallel``, and concludes FSDP
   was never engaged. A silent wrong number is worse than the crash it replaced.

Requirements
------------
- The accounting functions are pure and duck typed: no torch import, no CUDA,
  no process group. They are unit tested on CPU in
  ``tests/test_issue373_fsdp_sharding_probe.py``.
- Running it against a real model requires the multi-GPU box under test. It is
  designed to be installed as a ``sitecustomize`` on ``PYTHONPATH`` -- no repo
  edit -- exactly as the original was, or imported by the #41 recipe smoke-train
  so the sharding claim and the smoke claim come from one run.

Protocol
--------
Each rank reports ``local_parameter_numel`` of its own model. Sharding is
asserted, not eyeballed: ``local == total // world_size`` exactly, and the
single-GPU control must report the full count -- so a probe that always divides
fails the control instead of passing everything.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass

__all__ = [
    "ShardingVerdict",
    "local_parameter_numel",
    "module_class_histogram",
    "probe_rank",
    "sharding_verdict",
    "unwrap_compiled",
]


def unwrap_compiled(module):
    """Return the module underneath any stack of ``torch.compile`` wrappers.

    ``torch.compile`` returns an ``OptimizedModule`` holding the real module in
    ``_orig_mod``. Every class-identity question -- "is this FSDP?", "how many
    FSDP units are there?" -- must be asked of the wrapped module, or the answer
    describes the wrapper instead of the model.
    """
    seen = set()
    while True:
        orig = getattr(module, "_orig_mod", None)
        if orig is None or id(orig) in seen:
            return module
        seen.add(id(module))
        module = orig


def local_parameter_numel(module) -> int:
    """Total elements of the parameters physically resident on this rank.

    Uses ``numel()`` per parameter and never ``len()`` on a module: under
    ``torch.compile`` that raises, which is the defect this probe exists to
    survive.
    """
    return sum(parameter.numel() for parameter in unwrap_compiled(module).parameters())


def module_class_histogram(module) -> Counter:
    """Count module class names over the tree, unwrapping compile wrappers.

    The 0.5B record's evidence was "217x FullyShardedDataParallel". Reporting
    ``OptimizedModule`` there instead would silently answer a different question.
    """
    histogram: Counter = Counter()
    stack = [unwrap_compiled(module)]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        histogram[type(current).__name__] += 1
        for _name, child in current.named_children():
            stack.append(unwrap_compiled(child))
    return histogram


@dataclass(frozen=True)
class ShardingVerdict:
    local: int
    total: int
    world_size: int
    expected_local: int
    is_sharded: bool


def sharding_verdict(local: int, total: int, world_size: int) -> ShardingVerdict:
    """Assert the accounting rather than eyeballing it.

    ``world_size == 1`` is the control: it expects the *full* count, so a probe
    that unconditionally divides fails here instead of passing everything.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    expected_local = total // world_size
    return ShardingVerdict(
        local=local,
        total=total,
        world_size=world_size,
        expected_local=expected_local,
        is_sharded=(local == expected_local and total % world_size == 0),
    )


def probe_rank(module, total_parameters: int, world_size: int | None = None) -> dict:
    """One rank's line of the report."""
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local = local_parameter_numel(module)
    verdict = sharding_verdict(local, total_parameters, world_size)
    return {
        "rank": int(os.environ.get("RANK", "0")),
        "world_size": world_size,
        "local_parameters": local,
        "expected_local": verdict.expected_local,
        "is_sharded": verdict.is_sharded,
        "wrapper_classes": dict(module_class_histogram(module)),
    }
