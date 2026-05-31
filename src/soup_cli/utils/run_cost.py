"""Per-run cost estimation (v0.34.0 Part B).

Maps a detected GPU device name onto an approximate hourly rate (USD), then
multiplies by run duration to produce an informational $ estimate stored on
the run row. Rates are rough mid-2026 RunPod-comparable spot prices and
should be treated as ballpark, not invoices.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# Order matters: longer / more-specific patterns first so "h100" doesn't
# eat "h100 sxm". Each entry is (compiled_regex, label, usd_per_hour).
_GPU_RATE_TABLE: Tuple[Tuple[re.Pattern[str], str, float], ...] = (
    (re.compile(r"h100\s*sxm", re.IGNORECASE), "H100 SXM", 4.50),
    (re.compile(r"h100", re.IGNORECASE), "H100", 4.11),
    (re.compile(r"a100\s*80", re.IGNORECASE), "A100 80GB", 2.05),
    (re.compile(r"a100", re.IGNORECASE), "A100 40GB", 1.10),
    (re.compile(r"a40", re.IGNORECASE), "A40", 0.55),
    (re.compile(r"a6000", re.IGNORECASE), "A6000", 0.79),
    (re.compile(r"l40s", re.IGNORECASE), "L40S", 1.19),
    (re.compile(r"l40", re.IGNORECASE), "L40", 0.99),
    (re.compile(r"rtx\s*4090", re.IGNORECASE), "RTX 4090", 0.35),
    (re.compile(r"rtx\s*3090", re.IGNORECASE), "RTX 3090", 0.22),
    (re.compile(r"rtx\s*a5000", re.IGNORECASE), "RTX A5000", 0.36),
    (re.compile(r"v100", re.IGNORECASE), "V100", 0.49),
    (re.compile(r"t4", re.IGNORECASE), "T4", 0.20),
)

# Bounds — defence against pathological inputs. A multi-day run is fine; a
# negative duration or an absurdly long one (> 1 year) is a bug elsewhere.
MAX_DURATION_SECS = 60 * 60 * 24 * 365


def lookup_gpu_rate(device_name: Optional[str]) -> Optional[Tuple[str, float]]:
    """Return (canonical_label, usd_per_hour) for a device name, or None.

    Matches case-insensitively against `_GPU_RATE_TABLE`. Returns None when
    the device is not in the table (CPU, MPS, unknown vendor).
    """
    if not device_name or not isinstance(device_name, str):
        return None
    if "\x00" in device_name:
        return None
    for pattern, label, rate in _GPU_RATE_TABLE:
        if pattern.search(device_name):
            return label, rate
    return None


def estimate_run_cost_usd(
    device_name: Optional[str],
    duration_secs: Optional[float],
    num_gpus: int = 1,
) -> Optional[float]:
    """Estimate per-run cost in USD; None when the GPU is not priced.

    Returns None for CPU / MPS / unknown devices so callers can render a
    "—" instead of a fabricated $0.00.
    """
    if duration_secs is None or duration_secs <= 0:
        return None
    if duration_secs > MAX_DURATION_SECS:
        duration_secs = float(MAX_DURATION_SECS)
    # bool is a subclass of int — must reject explicitly so True/False
    # don't sneak past the isinstance check (matches v0.30.0 Candidate).
    if isinstance(num_gpus, bool) or not isinstance(num_gpus, int) or num_gpus < 1:
        num_gpus = 1
    looked_up = lookup_gpu_rate(device_name)
    if looked_up is None:
        return None
    _, rate = looked_up
    hours = duration_secs / 3600.0
    return round(rate * hours * num_gpus, 4)


def format_cost_usd(cost: Optional[float]) -> str:
    """Render a cost for display: `$0.42`, `<$0.01`, or `—`."""
    if cost is None:
        return "—"
    if cost < 0.01:
        return "<$0.01"
    return f"${cost:.2f}"
