"""Accelerate / torchrun launcher wrapper helpers (v0.27.0).

v0.27.0 is advisory only: when the user passes --gpus N>1 but soup is not
running under a launcher, we print the exact `accelerate launch` command to
run. Auto-reexec is deferred to v0.27.1 to keep the blast radius small.
"""

from __future__ import annotations

import os
import shlex
from typing import Sequence

VALID_MIXED_PRECISION = ("no", "fp16", "bf16", "fp8")
MAX_NUM_MACHINES = 256  # sanity cap, consistent with --gpus MAX_GPU_COUNT=128


def is_in_distributed() -> bool:
    """Return True if the current process was launched by torchrun / accelerate.

    Detection is based on the presence of either the torch.distributed standard
    variables (RANK + WORLD_SIZE) or any Accelerate-specific marker.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return True
    return any(
        key in os.environ
        for key in (
            "ACCELERATE_MIXED_PRECISION",
            "ACCELERATE_USE_DEEPSPEED",
            "ACCELERATE_USE_FSDP",
        )
    )


def build_accelerate_argv(
    num_processes: int,
    script_args: Sequence[str],
    mixed_precision: str | None = None,
    num_machines: int = 1,
) -> list[str]:
    """Build argv that wraps ``script_args`` with ``accelerate launch``.

    When ``num_processes == 1`` the launcher wrapper is skipped and
    ``script_args`` is returned unchanged.

    Args:
        num_processes: Total number of processes. Must be >= 1.
        script_args: The command to run (e.g. ``["soup", "train"]``).
        mixed_precision: One of ``no``, ``fp16``, ``bf16``, ``fp8``.
        num_machines: Number of nodes. Defaults to 1.

    Raises:
        ValueError: On invalid ``num_processes`` or ``mixed_precision``.
    """
    if not isinstance(num_processes, int) or num_processes < 1:
        raise ValueError(
            f"num_processes must be a positive integer (got {num_processes!r})."
        )
    if mixed_precision is not None and mixed_precision not in VALID_MIXED_PRECISION:
        raise ValueError(
            f"Invalid mixed_precision: {mixed_precision!r}. "
            f"Options: {', '.join(VALID_MIXED_PRECISION)}."
        )
    if num_machines < 1 or num_machines > MAX_NUM_MACHINES:
        raise ValueError(
            f"num_machines must be in [1, {MAX_NUM_MACHINES}] (got {num_machines})."
        )

    script_list = list(script_args)
    if num_processes == 1:
        return script_list

    argv: list[str] = ["accelerate", "launch", "--num_processes", str(num_processes)]
    if num_machines > 1:
        argv.extend(["--num_machines", str(num_machines)])
    if mixed_precision is not None:
        argv.extend(["--mixed_precision", mixed_precision])
    argv.extend(script_list)
    return argv


def format_advice(num_processes: int, script_args: Sequence[str]) -> str:
    """Human-readable hint telling the user the exact command to re-run."""
    cmd = build_accelerate_argv(num_processes=num_processes, script_args=script_args)
    quoted = " ".join(shlex.quote(arg) for arg in cmd)
    return (
        f"To train on {num_processes} GPUs, re-run under accelerate:\n\n"
        f"    {quoted}\n\n"
        f"(soup does not auto-re-exec in v0.27.0 — it prints this hint so you "
        f"stay in control of env vars and stdio.)"
    )
