"""v0.44.0 Part D — `soup delinearize-llama4` weight reshape stub.

Llama 4 ships with linearised expert weights that some downstream backends
expect in 3-D form. This module declares the planned reshape; live runtime
deferred to v0.44.1 (mirrors the project's stub-then-live pattern).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Tuple

from soup_cli.utils.paths import is_under_cwd

# Restrict to canonical Llama 4 model id shape; reject crafted names.
_LLAMA4_RE = re.compile(r"(?i)(?:^|[^a-z0-9])llama-?4(?:[^a-z0-9]|$)")


@dataclass(frozen=True)
class DelinearizePlan:
    """Planned weights to reshape, source path, target path.

    `weight_files` is a `tuple` for genuine immutability (matches the
    project frozen-collection policy).
    """

    source_dir: str
    target_dir: str
    weight_files: Tuple[str, ...]


def is_llama4_model(name: str) -> bool:
    """Return True iff `name` looks like a Llama 4 family model."""
    if not isinstance(name, str) or not name or "\x00" in name:
        return False
    return bool(_LLAMA4_RE.search(name))


def discover_weight_files(source_dir: str) -> List[str]:
    """List `.safetensors` weight files in `source_dir`."""
    if not isinstance(source_dir, str):
        raise TypeError("source_dir must be str")
    if not is_under_cwd(source_dir):
        raise ValueError(
            f"source_dir is outside cwd: {os.path.basename(source_dir)}"
        )
    real = os.path.realpath(source_dir)
    if not os.path.isdir(real):
        raise FileNotFoundError(
            f"source_dir not found: {os.path.basename(real)}"
        )
    files = sorted(
        entry for entry in os.listdir(real) if entry.endswith(".safetensors")
    )
    if not files:
        raise FileNotFoundError(
            "no .safetensors files found in source_dir"
        )
    return files


def plan_delinearize(source_dir: str, target_dir: str) -> DelinearizePlan:
    """Build a `DelinearizePlan`. Raises on bad inputs."""
    if not isinstance(target_dir, str) or not target_dir:
        raise ValueError("target_dir must be non-empty str")
    if "\x00" in target_dir:
        raise ValueError("target_dir contains NUL byte")
    if not is_under_cwd(target_dir):
        raise ValueError(
            f"target_dir is outside cwd: {os.path.basename(target_dir)}"
        )
    files = discover_weight_files(source_dir)
    return DelinearizePlan(
        source_dir=os.path.realpath(source_dir),
        target_dir=os.path.realpath(target_dir),
        weight_files=tuple(files),
    )
