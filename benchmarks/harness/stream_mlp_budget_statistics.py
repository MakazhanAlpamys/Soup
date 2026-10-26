"""Fixed global family: eight published contrasts, three new, one confirmation."""
from __future__ import annotations

import math
import statistics

CONTRASTS = ("mixed_vs_native", "mixed_vs_qkvo", "qkvo_vs_native")
FAMILY = 12


def summarize(reductions: list[float], *, seed: int = 42) -> dict:
    import numpy as np

    count = len(reductions)
    if count not in (5, 6) or any(not math.isfinite(value) for value in reductions):
        raise ValueError("five historical or six current finite paired reductions required")
    draws = np.median(np.asarray(reductions)[
        np.random.default_rng(seed).integers(0, count, size=(10000, count))
    ], axis=1)
    quantiles = [.05 / (2 * FAMILY), 1 - .05 / (2 * FAMILY)]
    ordinary = np.quantile(draws, [.025, .975], method="linear").tolist()
    corrected = np.quantile(draws, quantiles, method="linear").tolist()
    median = statistics.median(reductions)
    return dict(median_reduction_percent=median, ordinary_95=ordinary,
                family_adjusted_ci=corrected, adjusted_quantiles=quantiles,
                correction_family=FAMILY, reductions_percent=list(reductions),
                resamples=10000, seed=seed, count=count,
                positive_lower_bounds=ordinary[0] > 0 and corrected[0] > 0,
                passes_two_percent=median >= 2 and ordinary[0] > 0 and corrected[0] > 0)


def stage_a_decision(comparisons: dict) -> bool:
    if set(comparisons) != set(CONTRASTS):
        raise ValueError("all three prespecified contrasts required")
    if any(row["count"] != 6 or row["correction_family"] != FAMILY
           for row in comparisons.values()):
        raise ValueError("incomplete block or correction family")
    return (comparisons["mixed_vs_native"]["passes_two_percent"]
            and comparisons["mixed_vs_qkvo"]["positive_lower_bounds"])
