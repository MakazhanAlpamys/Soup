"""v0.44.0 Part A — EMA + p95/p99 tail-latency stats.

Pure-Python, no torch. Used by `runs show` and the live training dashboard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional

MAX_SAMPLES = 1_000_000  # DoS cap


def _is_real_number(value: object) -> bool:
    """Reject bool (subclass of int) AND non-finite floats."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def update_ema(prev: Optional[float], sample: float, alpha: float) -> float:
    """One-step exponential moving average update.

    `prev=None` initialises to `sample`. `alpha` is the weight on the new
    sample (0 < alpha <= 1). Smaller alpha → smoother EMA.
    """
    if not _is_real_number(sample):
        raise ValueError("sample must be a finite number")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("alpha must be a number")
    if not (0.0 < float(alpha) <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    sample_f = float(sample)
    if prev is None:
        return sample_f
    if not _is_real_number(prev):
        raise ValueError("prev must be a finite number or None")
    return float(alpha) * sample_f + (1.0 - float(alpha)) * float(prev)


def percentile(samples: Iterable[float], pct: float) -> Optional[float]:
    """Linear-interpolated percentile. `pct` is in [0, 100].

    Returns None on empty input. Rejects non-finite samples and bool.
    """
    if isinstance(pct, bool) or not isinstance(pct, (int, float)):
        raise ValueError("pct must be a number")
    if not (0.0 <= float(pct) <= 100.0):
        raise ValueError("pct must be in [0, 100]")
    materialised: List[float] = []
    for sample in samples:
        if not _is_real_number(sample):
            raise ValueError("samples must be finite numbers")
        materialised.append(float(sample))
        if len(materialised) > MAX_SAMPLES:
            raise ValueError(f"too many samples (>{MAX_SAMPLES})")
    if not materialised:
        return None
    materialised.sort()
    if len(materialised) == 1:
        return materialised[0]
    rank = (float(pct) / 100.0) * (len(materialised) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return materialised[lower]
    frac = rank - lower
    return materialised[lower] * (1.0 - frac) + materialised[upper] * frac


@dataclass(frozen=True)
class TailLatencySummary:
    count: int
    mean: Optional[float]
    p50: Optional[float]
    p95: Optional[float]
    p99: Optional[float]
    ema: Optional[float]


def summarise_latency(
    samples: Iterable[float],
    *,
    ema_alpha: float = 0.1,
) -> TailLatencySummary:
    """Compute mean / p50 / p95 / p99 + EMA over `samples`.

    Empty input returns a zero-count summary with all-None metrics.
    """
    materialised: List[float] = []
    ema: Optional[float] = None
    for sample in samples:
        if not _is_real_number(sample):
            raise ValueError("samples must be finite numbers")
        materialised.append(float(sample))
        ema = update_ema(ema, float(sample), ema_alpha)
        if len(materialised) > MAX_SAMPLES:
            raise ValueError(f"too many samples (>{MAX_SAMPLES})")
    if not materialised:
        return TailLatencySummary(0, None, None, None, None, None)
    return TailLatencySummary(
        count=len(materialised),
        mean=sum(materialised) / len(materialised),
        p50=percentile(materialised, 50.0),
        p95=percentile(materialised, 95.0),
        p99=percentile(materialised, 99.0),
        ema=ema,
    )
