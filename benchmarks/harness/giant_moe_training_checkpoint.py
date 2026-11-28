"""Atomic optimizer-step checkpoints for the giant MoE training harness."""

from __future__ import annotations

import math
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_CHECKPOINT_NAME = re.compile(r"checkpoint-([1-9][0-9]*)\.pt")
_FORMAT_VERSION = 1


def find_training_checkpoint(path: Path) -> Path:
    """Resolve a file or the newest published checkpoint in a directory."""
    if path.is_file():
        return path
    if path.is_dir():
        candidates = []
        for entry in path.iterdir():
            match = _CHECKPOINT_NAME.fullmatch(entry.name)
            if match and entry.is_file():
                candidates.append((int(match[1]), entry))
        if candidates:
            return max(candidates)[1]
    raise FileNotFoundError(f"no committed training checkpoint at {path}")


def save_training_checkpoint(
    directory: Path,
    parameters: Mapping[str, Any],
    optimizer: Any,
    *,
    identity: dict[str, Any],
    completed_steps: int,
    heldout_before: float,
    device: str,
) -> Path:
    """Publish the trainable parameters and optimizer state after a completed update."""
    import torch

    is_cuda = torch.device(device).type == "cuda"
    state = {
        "format": _FORMAT_VERSION,
        "identity": identity,
        "completed_steps": completed_steps,
        "heldout_before": heldout_before,
        "adapter": {name: param.detach().cpu() for name, param in parameters.items()},
        "optimizer": optimizer.state_dict(),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(device) if is_cuda else None,
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"checkpoint-{completed_steps}.pt"
    stream = tempfile.NamedTemporaryFile(
        "wb",
        dir=directory,
        prefix=f".checkpoint-{completed_steps}-",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(stream.name)
    try:
        with stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def restore_training_checkpoint(
    path: Path,
    parameters: Mapping[str, Any],
    optimizer: Any,
    *,
    identity: dict[str, Any],
    device: str,
) -> tuple[int, float]:
    """Restore a compatible run, returning its completed steps and original heldout loss."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("format") != _FORMAT_VERSION:
        raise ValueError(f"invalid training checkpoint format at {path}")
    if state.get("identity") != identity:
        raise ValueError(f"checkpoint run identity differs from the current run at {path}")
    completed = state.get("completed_steps")
    heldout_before = state.get("heldout_before")
    if type(completed) is not int or not 1 <= completed <= identity["steps"]:
        raise ValueError(f"checkpoint step is outside the run at {path}")
    if type(heldout_before) not in (int, float) or not math.isfinite(heldout_before):
        raise ValueError(f"invalid checkpoint heldout baseline at {path}")

    adapter = state.get("adapter")
    # The optimizer state is positional. Equal names in a different order are unsafe.
    if not isinstance(adapter, dict) or tuple(adapter) != tuple(parameters):
        raise ValueError(f"checkpoint adapter parameter order differs at {path}")
    for name, param in parameters.items():
        saved = adapter[name]
        if not isinstance(saved, torch.Tensor) or saved.shape != param.shape:
            raise ValueError(f"checkpoint adapter shape differs for {name}")
        if saved.dtype != param.dtype:
            raise ValueError(f"checkpoint adapter dtype differs for {name}")

    cpu_rng, cuda_rng = state.get("cpu_rng"), state.get("cuda_rng")
    is_cuda = torch.device(device).type == "cuda"
    has_device_rng = not is_cuda or isinstance(cuda_rng, torch.Tensor)
    if not isinstance(cpu_rng, torch.Tensor) or not has_device_rng:
        raise ValueError(f"missing checkpoint RNG state at {path}")
    optimizer_state = state.get("optimizer")
    if not isinstance(optimizer_state, dict):
        raise ValueError(f"missing checkpoint optimizer state at {path}")
    optimizer.load_state_dict(optimizer_state)
    with torch.no_grad():
        for name, param in parameters.items():
            param.copy_(adapter[name])
    torch.set_rng_state(cpu_rng)
    if is_cuda:
        torch.cuda.set_rng_state(cuda_rng, device=device)
    return completed, float(heldout_before)
