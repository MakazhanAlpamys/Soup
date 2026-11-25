"""The #2162 predeclared five-whole-pair bootstrap, with its two-cell family."""
from __future__ import annotations

import math
import statistics


def summarize_pairs(reductions: list[float]) -> dict:
    import numpy as np

    if len(reductions) != 5 or any(not math.isfinite(value) for value in reductions):
        raise ValueError("exactly five finite pair reductions are required")
    values = np.asarray(reductions, dtype=np.float64)
    indices = np.random.default_rng(42).integers(0, 5, size=(10000, 5))
    draws = np.median(values[indices], axis=1)
    ordinary = np.quantile(draws, [.025, .975], method="linear").tolist()
    corrected = np.quantile(draws, [.0125, .9875], method="linear").tolist()
    median = statistics.median(reductions)
    return dict(
        median_reduction_percent=median, ordinary_95=ordinary,
        family_adjusted_97_5=corrected, pair_reductions_percent=list(reductions),
        passes_rule=median >= 2 and ordinary[0] > 0 and corrected[0] > 0,
        n_pairs=5, resamples=10000, seed=42, correction_family_cells=2,
    )


def summarize_reallocation_blocks(reductions: list[float]) -> dict:
    """Separate follow-up: all six three-arm orders, three prespecified contrasts."""
    import numpy as np

    if len(reductions) != 6 or any(not math.isfinite(value) for value in reductions):
        raise ValueError("exactly six finite block reductions are required")
    values = np.asarray(reductions, dtype=np.float64)
    indices = np.random.default_rng(42).integers(0, 6, size=(10000, 6))
    draws = np.median(values[indices], axis=1)
    ordinary = np.quantile(draws, [.025, .975], method="linear").tolist()
    quantiles = [.05/6, 1-.05/6]
    corrected = np.quantile(draws, quantiles, method="linear").tolist()
    median = statistics.median(reductions)
    return dict(
        median_reduction_percent=median, ordinary_95=ordinary,
        family_adjusted_ci=corrected, adjusted_quantiles=quantiles,
        block_reductions_percent=list(reductions),
        passes_rule=median >= 2 and ordinary[0] > 0 and corrected[0] > 0,
        n_blocks=6, resamples=10000, seed=42, correction_family_contrasts=3,
    )
