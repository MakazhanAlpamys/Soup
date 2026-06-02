"""v0.44.0 Part A — `soup monitor` GPU live-monitor primitives.

Pure-Python helpers for parsing nvidia-smi CSV output and Apple Silicon
`powermetrics` output. Subprocess invocations use list args (no shell).
"""

from __future__ import annotations

import shutil
import subprocess  # noqa: S404 — list-args invocation only
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Bounds (defence-in-depth)
_NVIDIA_SMI_TIMEOUT_S = 5
_MAX_GPUS = 128


@dataclass(frozen=True)
class GpuSample:
    """One row of nvidia-smi output for a single GPU."""

    index: int
    name: str
    util_gpu_pct: Optional[float]
    util_mem_pct: Optional[float]
    mem_used_mb: Optional[float]
    mem_total_mb: Optional[float]
    temp_c: Optional[float]
    power_w: Optional[float]


def _parse_float_or_none(text: str) -> Optional[float]:
    cleaned = text.strip()
    if not cleaned or cleaned in {"[N/A]", "N/A", "[Not Supported]"}:
        return None
    # nvidia-smi suffixes units in some configs; keep numeric prefix only.
    head = cleaned.split()[0]
    try:
        return float(head)
    except (ValueError, TypeError):
        return None


def parse_nvidia_smi_csv(text: str) -> List[GpuSample]:
    """Parse `nvidia-smi --query-gpu=... --format=csv,noheader` output.

    Expected query order:
      index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw
    Lines that don't have exactly 8 columns are skipped silently.
    """
    if not isinstance(text, str):
        raise TypeError("text must be str")
    samples: List[GpuSample] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        cols = [col.strip() for col in line.split(",")]
        if len(cols) != 8:
            continue
        try:
            index = int(cols[0])
        except (ValueError, TypeError):
            continue
        if index < 0 or index >= _MAX_GPUS:
            continue
        # Reject embedded NUL byte in the GPU name (defence-in-depth).
        name = cols[1]
        if "\x00" in name:
            continue
        samples.append(
            GpuSample(
                index=index,
                name=name,
                util_gpu_pct=_parse_float_or_none(cols[2]),
                util_mem_pct=_parse_float_or_none(cols[3]),
                mem_used_mb=_parse_float_or_none(cols[4]),
                mem_total_mb=_parse_float_or_none(cols[5]),
                temp_c=_parse_float_or_none(cols[6]),
                power_w=_parse_float_or_none(cols[7]),
            )
        )
    return samples


def query_nvidia_smi() -> Tuple[bool, List[GpuSample]]:
    """Invoke nvidia-smi and return ok, samples. ok=False when smi is missing
    or returns a non-zero exit. Never raises."""
    smi_path = shutil.which("nvidia-smi")
    if smi_path is None:
        return False, []
    argv = [
        smi_path,
        "--query-gpu=index,name,utilization.gpu,utilization.memory,"
        "memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(  # noqa: S603 — list args, no shell
            argv,
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, []
    if result.returncode != 0:
        return False, []
    return True, parse_nvidia_smi_csv(result.stdout or "")


def detect_apple_silicon() -> bool:
    """Best-effort detection of Apple Silicon hardware (Mac M-series).

    Uses `platform.system()` + `platform.machine()` — the conditional logic
    here is intentionally simple to avoid the prior version's parser-priority
    bug where `if X if Y else Z:` produced a load-bearing-coincidence on
    every platform.
    """
    try:
        import platform
    except ImportError:
        return False
    if platform.system() != "Darwin":
        return False
    return platform.machine().lower() in {"arm64", "aarch64"}
