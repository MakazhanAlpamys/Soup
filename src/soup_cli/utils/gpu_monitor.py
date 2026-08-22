"""v0.44.0 Part A — `soup monitor` GPU live-monitor primitives.

Pure-Python helpers for parsing nvidia-smi CSV output and Apple Silicon
`powermetrics` output. Subprocess invocations use list args (no shell).
"""

from __future__ import annotations

import math
import os
import plistlib
import shutil
import subprocess  # noqa: S404 — list-args invocation only
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

# Bounds (defence-in-depth)
_NVIDIA_SMI_TIMEOUT_S = 5
_MAX_GPUS = 128
_MAX_POWERMETRICS_OUTPUT_BYTES = 1 * 1024 * 1024
_POWERMETRICS_PATH = "/usr/bin/powermetrics"
_SUDO_PATH = "/usr/bin/sudo"


@dataclass(frozen=True)
class GpuSample:
    """One row of GPU telemetry from NVIDIA or Apple tooling."""

    index: int
    name: str
    util_gpu_pct: Optional[float]
    util_mem_pct: Optional[float]
    mem_used_mb: Optional[float]
    mem_total_mb: Optional[float]
    temp_c: Optional[float]
    power_w: Optional[float]


class PowermetricsStatus(str, Enum):
    """Closed outcomes for one non-interactive ``powermetrics`` query."""

    OK = "ok"
    UNAVAILABLE = "unavailable"
    PERMISSION_DENIED = "permission_denied"
    FAILED = "failed"
    INVALID_OUTPUT = "invalid_output"


@dataclass(frozen=True)
class PowermetricsResult:
    """Result of one bounded Apple Silicon telemetry query."""

    status: PowermetricsStatus
    samples: tuple[GpuSample, ...] = ()


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
    """Invoke nvidia-smi and return (ok, samples). ok=False when smi is missing
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


def _finite_nonnegative(value: object) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _powermetrics_power_w(
    sample: Mapping[str, object],
    gpu: Mapping[str, object],
) -> Optional[float]:
    processor_obj = sample.get("processor")
    processor = processor_obj if isinstance(processor_obj, Mapping) else {}

    # Current macOS plist output carries average power in mW. Prefer it to
    # reconstructing power from the energy counter.
    for source in (processor, gpu, sample):
        power_mw = _finite_nonnegative(source.get("gpu_power"))
        if power_mw is not None:
            return power_mw / 1000.0

    elapsed_ns = _finite_nonnegative(sample.get("elapsed_ns"))
    if elapsed_ns is None or elapsed_ns <= 0:
        return None
    for source in (processor, gpu):
        energy_mj = _finite_nonnegative(source.get("gpu_energy"))
        if energy_mj is not None:
            elapsed_seconds = elapsed_ns / 1_000_000_000.0
            return energy_mj / elapsed_seconds / 1000.0
    return None


def _parse_powermetrics_sample(payload: bytes) -> Optional[GpuSample]:
    try:
        parsed = plistlib.loads(payload)
    except (plistlib.InvalidFileException, ValueError, TypeError, OverflowError):
        return None
    if not isinstance(parsed, Mapping):
        return None
    gpu_obj = parsed.get("gpu")
    if not isinstance(gpu_obj, Mapping):
        return None

    idle_ratio = _finite_nonnegative(gpu_obj.get("idle_ratio"))
    util_gpu_pct = None
    if idle_ratio is not None and idle_ratio <= 1.0:
        util_gpu_pct = (1.0 - idle_ratio) * 100.0

    hw_model = parsed.get("hw_model")
    if (
        isinstance(hw_model, str)
        and hw_model
        and "\x00" not in hw_model
        and len(hw_model) <= 128
    ):
        name = f"Apple Silicon ({hw_model})"
    else:
        name = "Apple Silicon"

    return GpuSample(
        index=0,
        name=name,
        util_gpu_pct=util_gpu_pct,
        util_mem_pct=None,
        mem_used_mb=None,
        mem_total_mb=None,
        temp_c=None,
        power_w=_powermetrics_power_w(parsed, gpu_obj),
    )


def parse_powermetrics_plist(payload: bytes) -> List[GpuSample]:
    """Parse bounded, NUL-separated ``powermetrics --format plist`` output.

    Apple emits one plist document per sample and prefixes later documents with
    a NUL byte. Malformed documents are skipped; NVIDIA-only memory and
    temperature fields remain unavailable instead of being guessed.
    """
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if len(payload) > _MAX_POWERMETRICS_OUTPUT_BYTES:
        return []

    samples: List[GpuSample] = []
    for document in payload.split(b"\x00"):
        document = document.strip()
        if not document:
            continue
        sample = _parse_powermetrics_sample(document)
        if sample is not None:
            samples.append(sample)
    return samples


def _effective_uid() -> int:
    getter = getattr(os, "geteuid", None)
    if getter is None:
        return -1
    return int(getter())


def query_powermetrics(interval_seconds: float) -> PowermetricsResult:
    """Collect one Apple GPU sample without prompting for credentials.

    Non-root callers use ``sudo -n`` so a cached sudo ticket works while a
    missing ticket fails immediately. The CLI can then tell the user to run
    ``sudo -v`` explicitly; Soup never reads a password.
    """
    if isinstance(interval_seconds, bool) or not isinstance(
        interval_seconds,
        (int, float),
    ):
        raise TypeError("interval_seconds must be a number")
    interval = float(interval_seconds)
    if not math.isfinite(interval) or not (0.25 <= interval <= 30.0):
        raise ValueError("interval_seconds must be finite and in [0.25, 30]")
    if not os.path.isfile(_POWERMETRICS_PATH):
        return PowermetricsResult(PowermetricsStatus.UNAVAILABLE)

    sample_rate_ms = int(round(interval * 1000.0))
    powermetrics_argv = [
        _POWERMETRICS_PATH,
        "--samplers",
        "gpu_power",
        "--sample-rate",
        str(sample_rate_ms),
        "--sample-count",
        "1",
        "--format",
        "plist",
        "--handle-invalid-values",
    ]
    if _effective_uid() == 0:
        argv = powermetrics_argv
    else:
        if not os.path.isfile(_SUDO_PATH):
            return PowermetricsResult(PowermetricsStatus.UNAVAILABLE)
        argv = [_SUDO_PATH, "-n", *powermetrics_argv]

    try:
        result = subprocess.run(  # noqa: S603 — fixed absolute tools, list args, no shell
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=False,
            timeout=interval + 5.0,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return PowermetricsResult(PowermetricsStatus.FAILED)
    if result.returncode != 0:
        stderr = (result.stderr or b"").decode("utf-8", errors="replace").lower()
        permission_markers = (
            "password is required",
            "a terminal is required",
            "must be invoked as the superuser",
            "not allowed to execute",
            "permission denied",
        )
        if any(marker in stderr for marker in permission_markers):
            return PowermetricsResult(PowermetricsStatus.PERMISSION_DENIED)
        return PowermetricsResult(PowermetricsStatus.FAILED)

    samples = parse_powermetrics_plist(result.stdout or b"")
    if not samples:
        return PowermetricsResult(PowermetricsStatus.INVALID_OUTPUT)
    return PowermetricsResult(PowermetricsStatus.OK, tuple(samples[-1:]))


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
