"""Per-run cost estimation (v0.34.0 Part B).

Maps a detected GPU device name onto an approximate hourly rate (USD), then
multiplies by run duration to produce an informational $ estimate stored on
the run row. Rates are rough mid-2026 RunPod-comparable spot prices and
should be treated as ballpark, not invoices.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# Two invariants keep this table correct:
#   * Anchoring: every pattern is \b-bounded so "a40" cannot match inside "RTX A4000".
#   * Order: first match wins, so a more specific row must sit above the general
#     one it also matches ("A100 80GB" before "A100", "RTX 4070 Ti" before "RTX 4070").
# Device names are what torch.cuda.get_device_name() reports. Rows must not claim
# precision the driver name lacks: "NVIDIA H100 80GB HBM3" (SXM5) and "H100 NVL" say
# nothing about SXM/PCIe, so they stay on the generic "H100" row. The "H100 SXM" row
# only fires on provider-supplied labels and must stay above "H100".
# Known limitation: MIG slices report the parent card's name and take its full rate.
# Every pattern must start and end with \b (tests enforce this): a missing trailing
# boundary would let "rtx 4090" match "RTX 40900".
# Each entry is (compiled_regex, label, usd_per_hour).
#
# Rates for rows marked (v0.34.0) are unchanged from the original table. Rows below
# are sourced as follows (all retrieved 2026-09-18):
#   RunPod Secure Cloud on-demand, https://www.runpod.io/pricing (updated 2026-09-13):
#     B200, H200, RTX PRO 6000, RTX 6000 Ada, L4, RTX 5090.
#   computeprices.com, Runpod Community tier (no Secure listing found): RTX A4000
#     https://computeprices.com/gpus/rtxa4000; RTX A4500 is the Runpod Secure listing
#     https://computeprices.com/gpus/rtxa4500.
#   getdeploying.com: A10G is the AWS on-demand whole-instance price (g5.xlarge,
#     $1.006/hr, host included: an upper bound, not a per-GPU rental rate);
#     RTX 4070 Ti / 4080 / 5080 are the Runpod price; RTX 4070 / 5070 / 5070 Ti have
#     no RunPod listing so the cross-provider median is used.
_GPU_RATE_TABLE: Tuple[Tuple[re.Pattern[str], str, float], ...] = (
    # Blackwell / Hopper datacentre
    (re.compile(r"\bb200\b", re.IGNORECASE), "B200", 6.79),
    (re.compile(r"\bh200\b", re.IGNORECASE), "H200", 4.59),
    (re.compile(r"\bh100\b[\s-]*sxm\d*\b", re.IGNORECASE), "H100 SXM", 4.50),  # (v0.34.0)
    (re.compile(r"\bh100\b", re.IGNORECASE), "H100", 4.11),  # (v0.34.0)
    (re.compile(r"\ba100\b.*\b80\s*gb\b", re.IGNORECASE), "A100 80GB", 2.05),  # (v0.34.0)
    (re.compile(r"\ba100\b", re.IGNORECASE), "A100 40GB", 1.10),  # (v0.34.0)
    # Ada / Ampere datacentre
    (re.compile(r"\bl40s\b", re.IGNORECASE), "L40S", 1.19),  # (v0.34.0)
    (re.compile(r"\bl40\b", re.IGNORECASE), "L40", 0.99),  # (v0.34.0)
    (re.compile(r"\bl4\b", re.IGNORECASE), "L4", 0.49),
    (re.compile(r"\ba40\b", re.IGNORECASE), "A40", 0.55),  # (v0.34.0)
    # $1.01 is a whole AWS g5.xlarge instance (host included), not a per-GPU rental
    # price like the other rows, so treat it as an upper bound.
    (re.compile(r"\ba10g\b", re.IGNORECASE), "A10G", 1.01),
    (re.compile(r"\bv100s?\b", re.IGNORECASE), "V100", 0.49),  # (v0.34.0)
    (re.compile(r"\bt4\b", re.IGNORECASE), "T4", 0.20),  # (v0.34.0)
    # Workstation RTX
    (re.compile(r"\brtx[\s-]*pro[\s-]*6000\b", re.IGNORECASE), "RTX PRO 6000", 2.09),
    (re.compile(r"\brtx\s*6000\b.*\bada\b", re.IGNORECASE), "RTX 6000 Ada", 0.84),
    (re.compile(r"\ba6000\b", re.IGNORECASE), "A6000", 0.79),  # (v0.34.0)
    (re.compile(r"\brtx\s*a5000\b", re.IGNORECASE), "RTX A5000", 0.36),  # (v0.34.0)
    (re.compile(r"\brtx\s*a4500\b", re.IGNORECASE), "RTX A4500", 0.25),
    (re.compile(r"\brtx\s*a4000\b", re.IGNORECASE), "RTX A4000", 0.17),
    # GeForce. SUPER / D suffixes price at the base card's rate.
    (re.compile(r"\brtx\s*5090\b", re.IGNORECASE), "RTX 5090", 0.99),
    (re.compile(r"\brtx\s*5080\b", re.IGNORECASE), "RTX 5080", 0.39),
    (re.compile(r"\brtx\s*5070\s*ti\b", re.IGNORECASE), "RTX 5070 Ti", 0.23),
    (re.compile(r"\brtx\s*5070\b", re.IGNORECASE), "RTX 5070", 0.17),
    (re.compile(r"\brtx\s*4090\b", re.IGNORECASE), "RTX 4090", 0.35),  # (v0.34.0)
    (re.compile(r"\brtx\s*4080\b", re.IGNORECASE), "RTX 4080", 0.27),
    (re.compile(r"\brtx\s*4070\s*ti\b", re.IGNORECASE), "RTX 4070 Ti", 0.19),
    (re.compile(r"\brtx\s*4070\b", re.IGNORECASE), "RTX 4070", 0.15),
    # "3090 Ti" has no separate rental tier and prices as a 3090.
    (re.compile(r"\brtx\s*3090\b", re.IGNORECASE), "RTX 3090", 0.22),  # (v0.34.0)
)

# Bounds — defence against pathological inputs. A multi-day run is fine; a
# negative duration or an absurdly long one (> 1 year) is a bug elsewhere.
MAX_DURATION_SECS = 60 * 60 * 24 * 365

# Real device names are well under 100 characters. The A100 80GB pattern uses `.*`,
# which is quadratic on a pathological input, so longer names are treated as unknown.
MAX_DEVICE_NAME_LEN = 256


def lookup_gpu_rate(device_name: Optional[str]) -> Optional[Tuple[str, float]]:
    """Return (canonical_label, usd_per_hour) for a device name, or None.

    Matches case-insensitively against `_GPU_RATE_TABLE`. Returns None when
    the device is not in the table (CPU, MPS, unknown vendor).
    """
    if not device_name or not isinstance(device_name, str):
        return None
    if "\x00" in device_name or len(device_name) > MAX_DEVICE_NAME_LEN:
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
