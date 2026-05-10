"""v0.43.0 Part C — Profiling extras.

  - `memory_snapshot_context`: torch.cuda.memory._record_memory_history wrapper
    that writes a `<run_id>.snapshot.pickle` to the profiles dir under cwd.
  - `enable_detect_anomaly`: thin context manager around
    `torch.autograd.set_detect_anomaly(True)`.
  - `nccl_bandwidth_check`: scaffold that returns the expected upper-bound
    bandwidth for a (gpu_pair, link) tuple. Live measurement deferred — we
    expose the reference table so `soup doctor --nccl` can warn when measured
    perf is well below expectation.

Containment, redaction, and exception-narrowing follow v0.34.0 `crash.py` /
v0.34.0 `profiling.py` policies.
"""
from __future__ import annotations

import contextlib
import math
import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterator, Mapping

from soup_cli.utils.paths import is_under_cwd

# Reference NVLink/PCIe bandwidth ceilings (GB/s, unidirectional).
# Source: NVIDIA published topology specs for H100/A100/V100.
_BANDWIDTH_REFERENCE: Mapping[tuple[str, str], float] = MappingProxyType({
    ("h100", "nvlink"): 450.0,    # NVLink 4 (18 links × 25 GB/s)
    ("h100", "pcie"): 64.0,       # PCIe Gen5 x16
    ("a100", "nvlink"): 300.0,    # NVLink 3 (12 links × 25 GB/s)
    ("a100", "pcie"): 32.0,       # PCIe Gen4 x16
    ("v100", "nvlink"): 150.0,    # NVLink 2 (6 links × 25 GB/s)
    ("v100", "pcie"): 16.0,       # PCIe Gen3 x16
    ("rtx4090", "pcie"): 64.0,
    ("rtx3090", "pcie"): 32.0,
})


def _validate_run_id(run_id: object) -> str:
    if not isinstance(run_id, str):
        raise ValueError("run_id must be a string")
    if not run_id:
        raise ValueError("run_id must not be empty")
    if "\x00" in run_id:
        raise ValueError("run_id must not contain null bytes")
    if run_id in {".", ".."}:
        raise ValueError(f"run_id contains forbidden token '{run_id}'")
    for ch in ("/", "\\"):
        if ch in run_id:
            raise ValueError(f"run_id must not contain path separator '{ch}'")
    return run_id


def resolve_snapshot_path(run_id: str, *, base_dir: str = "profiles") -> str:
    """Return realpath of `<cwd>/<base_dir>/<run_id>.snapshot.pickle`.

    Rejects values that escape cwd. Mirrors v0.34.0 `resolve_trace_path`.
    """
    rid = _validate_run_id(run_id)
    if not isinstance(base_dir, str) or not base_dir:
        raise ValueError("base_dir must be a non-empty string")
    if "\x00" in base_dir:
        raise ValueError("base_dir must not contain null bytes")
    # Reject path separators and `..` components in base_dir so an absolute
    # / parent-traversing path can never sneak through realpath on Windows
    # short-name systems (security review fix).
    if base_dir in {".", ".."}:
        raise ValueError("base_dir must not be '.' or '..'")
    if os.path.isabs(base_dir):
        raise ValueError("base_dir must be a relative path under cwd")
    parts = [p for p in base_dir.replace("\\", "/").split("/") if p]
    if any(p == ".." for p in parts):
        raise ValueError("base_dir must not contain '..' segments")
    target = os.path.realpath(os.path.join(base_dir, f"{rid}.snapshot.pickle"))
    if not is_under_cwd(target):
        raise ValueError("snapshot path must stay under cwd")
    return target


@contextlib.contextmanager
def memory_snapshot_context(
    run_id: str,
    *,
    base_dir: str = "profiles",
    max_entries: int = 100_000,
) -> Iterator[str | None]:
    """Record CUDA memory history; on exit, dump pickle + stop recording.

    Yields the snapshot path on success, or None when torch is missing /
    CUDA is unavailable / the recording API is missing. Never raises through
    the context exit when torch failures are missing-dep style — those map
    to a yielded None.
    """
    if isinstance(max_entries, bool) or not isinstance(max_entries, int):
        raise ValueError("max_entries must be a positive int")
    if max_entries < 1 or max_entries > 10_000_000:
        raise ValueError("max_entries must be in [1, 10_000_000]")

    path = resolve_snapshot_path(run_id, base_dir=base_dir)
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        yield None
        return
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        yield None
        return
    record = getattr(getattr(cuda, "memory", None), "_record_memory_history", None)
    dump = getattr(getattr(cuda, "memory", None), "_dump_snapshot", None)
    if record is None or dump is None:
        yield None
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Narrow the RuntimeError catch to only the entry call so a user-body
    # RuntimeError cannot trigger a double-yield in this generator
    # (review fix — the previous wide `except RuntimeError` would have
    # raised "generator already executing" on user-body failures).
    try:
        record(max_entries=max_entries)
    except RuntimeError:
        yield None
        return
    try:
        yield path
    finally:
        try:
            dump(path)
        finally:
            try:
                record(enabled=None)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                pass


@contextlib.contextmanager
def detect_anomaly_context() -> Iterator[bool]:
    """torch.autograd.set_detect_anomaly(True) wrapper.

    Yields True when activated, False when torch is missing.
    """
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        yield False
        return
    set_detect = getattr(getattr(torch, "autograd", None), "set_detect_anomaly", None)
    if set_detect is None:
        yield False
        return
    set_detect(True)
    try:
        yield True
    finally:
        set_detect(False)


@dataclass(frozen=True)
class BandwidthExpectation:
    gpu: str
    link: str
    expected_gb_per_sec: float


def expected_bandwidth(gpu: str, link: str) -> float | None:
    """Return reference bandwidth (GB/s) for `(gpu, link)`, or None."""
    if not isinstance(gpu, str) or not isinstance(link, str):
        return None
    return _BANDWIDTH_REFERENCE.get((gpu.lower(), link.lower()))


def nccl_bandwidth_check(
    *, gpu: str, link: str, measured_gb_per_sec: float
) -> dict:
    """Compare measured bandwidth against the reference table.

    Returns a dict with `expected`, `measured`, `ratio`, and `status`.

      - status="OK"     : ratio >= 0.80
      - status="MINOR"  : 0.50 <= ratio < 0.80
      - status="MAJOR"  : ratio < 0.50  (silent degradation likely)
      - status="UNKNOWN": no reference entry for (gpu, link)
    """
    if (
        isinstance(measured_gb_per_sec, bool)
        or not isinstance(measured_gb_per_sec, (int, float))
    ):
        raise ValueError("measured_gb_per_sec must be a number")
    if not math.isfinite(float(measured_gb_per_sec)):
        raise ValueError("measured_gb_per_sec must be finite")
    if measured_gb_per_sec < 0:
        raise ValueError("measured_gb_per_sec must be >= 0")

    expected = expected_bandwidth(gpu, link)
    if expected is None or expected <= 0:
        return {
            "expected_gb_per_sec": None,
            "measured_gb_per_sec": float(measured_gb_per_sec),
            "ratio": None,
            "status": "UNKNOWN",
        }
    ratio = float(measured_gb_per_sec) / expected
    if ratio >= 0.80:
        status = "OK"
    elif ratio >= 0.50:
        status = "MINOR"
    else:
        status = "MAJOR"
    return {
        "expected_gb_per_sec": expected,
        "measured_gb_per_sec": float(measured_gb_per_sec),
        "ratio": round(ratio, 4),
        "status": status,
    }
