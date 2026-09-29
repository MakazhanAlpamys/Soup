"""mSPRT A/B harness — sequential testing with early-stop guarantees.

v0.63.0 Part D — proper sequential statistics on (latency, judge_score,
retry_rate) instead of naive repeated Wald tests. Composes with v0.58
`soup loop canary` so a canary deploy can be promoted (or rolled back)
as soon as the evidence clears the threshold, not at a fixed sample size.

Why mSPRT and not a t-test:
- t-test inflates Type-I error if you peek at the data N times.
- mSPRT (Mixture Sequential Probability Ratio Test) controls Type-I + II
  errors *for any stopping time*. You can monitor live and stop as soon as
  the log-likelihood ratio crosses either decision boundary.

The test is two-sided (#1227). The statistic is a symmetric two-point
mixture: the average of Wald's likelihood ratios for a treatment-minus-control
difference of `+effect_size` and of `-effect_size`, averaged as ratios (in log
space, via logsumexp), never as log-ratios and never via `|z|`. Averaging keeps
what each ratio has: E[LR_n] = 1 under H0 at every n. The boundaries are
`log(beta/(1-alpha))` (accept H0) and `log((1-beta)/alpha)` (reject H0); with a
known variance the simulated Type-I rate of this test under peeking stays below
alpha (benchmarks/gate-1227-ab-burn-in.md). A `reject_h0` reports its direction,
`better` or `worse`, from the metric's polarity in `HIGHER_IS_BETTER`.

Three known limitations:
1. Single metric per pass — multi-metric correction (Bonferroni / Holm)
   is operator-controlled. The CLI accepts one metric at a time.
2. Assumes Gaussian-like data. For binary metrics (e.g. retry_rate as
   a boolean), the operator should pre-aggregate per-prompt rates so the
   resulting per-prompt averages are approximately Gaussian.
3. The variance is estimated from the rows, not known, and from a handful of
   rows the estimate is too noisy. So no verdict is given (`continue`) until
   each arm has `min_rows_per_arm(alpha)` rows; `BURN_IN_ROWS_BY_ALPHA` says
   how those were chosen and what Type-I rate remains. The calibration holds
   up to `CALIBRATED_HORIZON_ROWS` rows per arm and for alpha >= 0.01
   (`burn_in_is_calibrated`). A variance-robust statistic that would need no
   burn-in is #1265.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from soup_cli.utils.paths import is_under_cwd

SUPPORTED_METRICS: frozenset[str] = frozenset(
    {"latency", "judge_score", "retry_rate"}
)
# Which way is an improvement, per metric (#1227). Every supported metric must
# appear here (a test pins it), so a new metric cannot inherit a direction.
HIGHER_IS_BETTER: Mapping[str, bool] = MappingProxyType(
    {"judge_score": True, "latency": False, "retry_rate": False}
)
# Burn-in (#1227): rows per arm required before any verdict, by the Type-I level
# alpha. While either arm has fewer, the verdict is `continue` whatever the
# log-likelihood ratio says. The pooled variance is estimated from the rows, and
# from a handful of them it can come out far too small, so without a burn-in an
# operator who re-runs after every new pair would reject a true H0 up to 0.164
# of the time at alpha 0.05. Calibrated by a Monte-Carlo sweep of that peeking
# procedure: H0, a look after every pair, the first verdict stops the run,
# beta 0.20, 22 values of effect_size / sigma from 0.1 to 5, 100,000 runs per
# ratio, and a horizon of 1000 rows per arm (200 checked too). Each value is the
# smallest candidate in {20, 30, 40, 50, 60} whose Type-I rate stays within
# alpha + 3 binomial standard errors at every ratio, at both horizons, at every
# alpha measured in its range.
#   alpha >= 0.05       -> 30 rows. Alpha 0.05: worst 0.0512 at ratio 0.35
#                          (bound 0.0521); 20 rows reach 0.0534. Alpha 0.10:
#                          worst 0.0994 (bound 0.1029).
#   0.01 <= alpha < 0.05 -> 40 rows. Alpha 0.01: worst 0.01088 at ratio 0.4
#                          (bound 0.01094) in this sweep, but up to 0.0116 on
#                          other seeds: a floor near 0.0108 set by the
#                          estimated variance, which 50/60 rows do not lower
#                          (#1265); 30 rows reach 0.0114. Alpha 0.025:
#                          worst 0.0261 (bound 0.0265); 30 rows reach 0.0269.
#   alpha < 0.01        -> 40 rows, NOT calibrated: at alpha 0.005 the worst is
#                          0.0058, above its bound 0.0057. `soup ab` warns.
# With the true variance the same sweep stays below alpha (worst 0.0468 at 0.05,
# 0.0095 at 0.01). Record: benchmarks/gate-1227-ab-burn-in.md; script:
# benchmarks/harness/ab_burn_in_sweep.py; results and log:
# benchmarks/results/gate-1227-ab-burn-in/. Changing a value means re-running it.
BURN_IN_ROWS_BY_ALPHA: tuple[tuple[float, int], ...] = ((0.05, 30), (0.01, 40))
_BURN_IN_ROWS_UNCALIBRATED = 40
# The sweep followed runs up to this many rows per arm; past it the Type-I
# figures above are not established, and `soup ab` says so.
CALIBRATED_HORIZON_ROWS = 1000
_MAX_METRIC_NAME_LEN = 32
_MAX_SAMPLES_PER_ARM = 1_000_000
_VALID_DECISIONS: frozenset[str] = frozenset(
    {"continue", "reject_h0", "accept_h0"}
)
_VALID_DIRECTIONS: frozenset[str] = frozenset({"better", "worse"})


def validate_metric_name(name: object) -> str:
    """Validate + canonicalise an A/B test metric name."""
    if isinstance(name, bool):
        raise TypeError("metric must be str, not bool")
    if not isinstance(name, str):
        raise TypeError(f"metric must be str, got {type(name).__name__}")
    if not name:
        raise ValueError("metric must be non-empty")
    if "\x00" in name:
        raise ValueError("metric must not contain null bytes")
    if len(name) > _MAX_METRIC_NAME_LEN:
        raise ValueError(
            f"metric must be <= {_MAX_METRIC_NAME_LEN} chars, got {len(name)}"
        )
    canonical = name.lower().strip()
    if canonical not in SUPPORTED_METRICS:
        raise ValueError(
            f"unknown metric {name!r}; supported: {sorted(SUPPORTED_METRICS)}"
        )
    return canonical


def _require_unit_open(value: object, *, field: str) -> float:
    """Validate a float in the open interval (0, 1)."""
    if isinstance(value, bool):
        raise TypeError(f"{field} must be a number, not bool")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number, got {type(value).__name__}")
    f_val = float(value)
    if not math.isfinite(f_val):
        raise ValueError(f"{field} must be finite (no NaN / Inf)")
    if not (0.0 < f_val < 1.0):
        raise ValueError(f"{field} must be in (0.0, 1.0) exclusive, got {f_val}")
    return f_val


def min_rows_per_arm(alpha: float) -> int:
    """Rows each arm needs before `soup ab` gives any verdict at Type-I level ``alpha``.

    From ``BURN_IN_ROWS_BY_ALPHA``: 30 at alpha >= 0.05, 40 below. Below alpha
    0.01 the value is 40 but not calibrated; see :func:`burn_in_is_calibrated`.
    """
    value = _require_unit_open(alpha, field="alpha")
    for lowest_alpha, rows in BURN_IN_ROWS_BY_ALPHA:
        if value >= lowest_alpha:
            return rows
    return _BURN_IN_ROWS_UNCALIBRATED


def burn_in_is_calibrated(alpha: float) -> bool:
    """False below alpha 0.01, where Type-I control under peeking was not calibrated."""
    return _require_unit_open(alpha, field="alpha") >= BURN_IN_ROWS_BY_ALPHA[-1][0]


def _require_positive_finite(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be a number, not bool")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number, got {type(value).__name__}")
    f_val = float(value)
    if not math.isfinite(f_val):
        raise ValueError(f"{field} must be finite (no NaN / Inf)")
    if f_val <= 0.0:
        raise ValueError(f"{field} must be > 0, got {f_val}")
    return f_val


@dataclass(frozen=True)
class MsprtConfig:
    """Parameters for an mSPRT pass."""

    metric: str
    alpha: float = 0.05  # Type-I error rate
    beta: float = 0.20  # Type-II error rate
    effect_size: float = 0.1  # Minimum detectable difference in means

    def __post_init__(self) -> None:
        # Re-validate via canonicalisation so callers bypassing the factory
        # cannot smuggle through a non-canonical metric.
        object.__setattr__(self, "metric", validate_metric_name(self.metric))
        object.__setattr__(self, "alpha", _require_unit_open(self.alpha, field="alpha"))
        object.__setattr__(self, "beta", _require_unit_open(self.beta, field="beta"))
        object.__setattr__(
            self,
            "effect_size",
            _require_positive_finite(self.effect_size, field="effect_size"),
        )


@dataclass(frozen=True)
class MsprtVerdict:
    """Outcome of an mSPRT step.

    ``direction`` is ``"better"`` or ``"worse"`` (the treatment relative to
    control, by the metric's polarity) on a ``reject_h0``, and ``None`` on
    ``accept_h0`` / ``continue``.
    """

    decision: str
    log_likelihood_ratio: float
    n_control: int
    n_treatment: int
    mean_control: float
    mean_treatment: float
    direction: str | None = None

    def __post_init__(self) -> None:
        if self.decision not in _VALID_DECISIONS:
            raise ValueError(
                f"decision must be one of {sorted(_VALID_DECISIONS)}, "
                f"got {self.decision!r}"
            )
        if self.direction is not None and self.direction not in _VALID_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {sorted(_VALID_DIRECTIONS)} or None, "
                f"got {self.direction!r}"
            )
        if self.decision == "reject_h0" and self.direction is None:
            raise ValueError("a reject_h0 verdict must carry a direction (better / worse)")
        if self.decision != "reject_h0" and self.direction is not None:
            raise ValueError(
                f"only a reject_h0 verdict has a direction; got {self.direction!r} "
                f"with decision {self.decision!r}"
            )


def _validate_sample_list(samples: object, *, arm: str) -> list[float]:
    if not isinstance(samples, Sequence) or isinstance(samples, str):
        raise TypeError(
            f"{arm} samples must be a list/tuple, got {type(samples).__name__}"
        )
    out: list[float] = []
    for i, value in enumerate(samples):
        if isinstance(value, bool):
            raise TypeError(
                f"{arm}[{i}] must be number, not bool"
            )
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"{arm}[{i}] must be number, got {type(value).__name__}"
            )
        f_val = float(value)
        if not math.isfinite(f_val):
            raise ValueError(f"{arm}[{i}] must be finite (no NaN / Inf)")
        out.append(f_val)
        if len(out) >= _MAX_SAMPLES_PER_ARM:
            break
    return out


def _log_mean_exp(first: float, second: float) -> float:
    """``log((exp(first) + exp(second)) / 2)`` without overflow (two-term logsumexp)."""
    top = max(first, second)
    return top + math.log1p(math.exp(-abs(first - second))) - math.log(2.0)


def _direction(metric: str, diff: float) -> str:
    """``better`` / ``worse`` for a treatment-minus-control difference, by polarity."""
    treatment_higher = diff > 0.0
    return "better" if treatment_higher == HIGHER_IS_BETTER[metric] else "worse"


def _verdict_from_summary(
    config: MsprtConfig,
    *,
    n_control: int,
    n_treatment: int,
    mean_control: float,
    mean_treatment: float,
    pooled_variance: float,
) -> MsprtVerdict:
    """Decide from per-arm summaries (``n >= 2`` per arm, ``pooled_variance > 0``).

    The decision half of :func:`msprt_step`, split out so a peeking simulation
    can drive exactly this code from running sums instead of re-reading every
    row at every peek. Below ``min_rows_per_arm(config.alpha)`` rows in either
    arm the log-likelihood ratio is still reported, but the decision is
    ``continue``.
    """
    pooled_se = math.sqrt(pooled_variance * (1.0 / n_control + 1.0 / n_treatment))

    # Standardised effect size (z-statistic of the difference of means).
    diff = mean_treatment - mean_control
    z = diff / pooled_se

    # Two-sided SPRT (#1227). For a point alternative H1: delta = +d (d =
    # effect_size, in standardised units mu_h1) Wald's log-likelihood ratio is
    #
    # log(LR+_n) = z * mu_h1 * sqrt(n_eff / (n_eff + 1))
    #            - 0.5 * mu_h1**2 * n_eff / (n_eff + 1)
    #
    # and for H1: delta = -d the first term flips sign. With a known variance
    # E[LR_n] = 1 under H0 at every n for each ratio, and so for their average.
    # (The n_eff / (n_eff + 1) shrink moves the alternative with n, so neither
    # is exactly a martingale; the Type-I rate under peeking is measured, not
    # derived: benchmarks/gate-1227-ab-burn-in.md. Here the variance is also
    # estimated; see limitation 3 in the module docstring.)
    #
    # It must be the average of the RATIOS. The mean of the log-ratios is just
    # -drift (it ignores the data), and |z| in the one-sided formula is the
    # larger of the two log-ratios, whose expectation exceeds 1 and which
    # roughly doubles the Type-I rate under peeking.
    #
    # (Code-review CRITICAL fix v0.63.0: earlier draft used a malformed
    # mixture-prior LLR with the wrong sign on the log term, which drove
    # the LLR positive under H0 as n grew → unbounded Type-I error.)
    n_eff = (n_control * n_treatment) / (n_control + n_treatment)
    mu_h1 = config.effect_size / pooled_se  # in standardised units
    n_ratio = n_eff / (n_eff + 1.0)
    shift = z * mu_h1 * math.sqrt(n_ratio)
    drift = 0.5 * mu_h1**2 * n_ratio
    llr = _log_mean_exp(shift - drift, -shift - drift)

    upper = math.log((1.0 - config.beta) / config.alpha)
    lower = math.log(config.beta / (1.0 - config.alpha))

    direction = None
    if min(n_control, n_treatment) < min_rows_per_arm(config.alpha):
        decision = "continue"  # burn-in: see BURN_IN_ROWS_BY_ALPHA
    elif llr >= upper:
        decision = "reject_h0"
        direction = _direction(config.metric, diff)
    elif llr <= lower:
        decision = "accept_h0"
    else:
        decision = "continue"

    return MsprtVerdict(
        decision=decision,
        log_likelihood_ratio=llr,
        n_control=n_control,
        n_treatment=n_treatment,
        mean_control=mean_control,
        mean_treatment=mean_treatment,
        direction=direction,
    )


def msprt_step(
    config: MsprtConfig,
    *,
    control: Sequence[float],
    treatment: Sequence[float],
) -> MsprtVerdict:
    """Run a single two-sided mSPRT decision step.

    Returns ``MsprtVerdict`` with one of:
    - ``continue``: keep collecting samples; always the answer while either
      arm has fewer than ``min_rows_per_arm(config.alpha)`` rows (burn-in)
    - ``reject_h0``: difference is real (treatment != control), in either
      direction; ``direction`` says whether the treatment is ``better`` or
      ``worse`` than control, by the metric's polarity
    - ``accept_h0``: difference is not significant
    """
    ctrl = _validate_sample_list(control, arm="control")
    treat = _validate_sample_list(treatment, arm="treatment")

    n_c, n_t = len(ctrl), len(treat)
    mean_c = sum(ctrl) / n_c if n_c else 0.0
    mean_t = sum(treat) / n_t if n_t else 0.0

    if n_c < 2 or n_t < 2:
        return MsprtVerdict(
            decision="continue",
            log_likelihood_ratio=0.0,
            n_control=n_c,
            n_treatment=n_t,
            mean_control=mean_c,
            mean_treatment=mean_t,
        )

    # Pooled variance with Bessel correction.
    var_c = sum((x - mean_c) ** 2 for x in ctrl) / (n_c - 1)
    var_t = sum((x - mean_t) ** 2 for x in treat) / (n_t - 1)
    raw_pooled_var = ((n_c - 1) * var_c + (n_t - 1) * var_t) / (n_c + n_t - 2)
    # Degenerate (zero variance) — both arms are constant. If the means
    # are also identical, defer to ``continue`` (no information). If the
    # means differ, fall back to ``continue`` as well: with zero observed
    # variance the SPRT cannot bound Type-I error honestly. Operators
    # need real measurement noise to use sequential testing (code-review
    # LOW fix v0.63.0 — the equality check was previously dead under the
    # `max(_, 1e-9)` floor).
    if raw_pooled_var <= 0.0:
        return MsprtVerdict(
            decision="continue",
            log_likelihood_ratio=0.0,
            n_control=n_c,
            n_treatment=n_t,
            mean_control=mean_c,
            mean_treatment=mean_t,
        )
    return _verdict_from_summary(
        config,
        n_control=n_c,
        n_treatment=n_t,
        mean_control=mean_c,
        mean_treatment=mean_t,
        pooled_variance=raw_pooled_var,
    )


def run_msprt(
    input_path: str,
    *,
    config: MsprtConfig,
) -> MsprtVerdict:
    """Read a JSONL of {arm, <metric>} rows and run the mSPRT pass.

    Each row must have ``arm`` (``control`` or ``treatment``) and a numeric
    field matching ``config.metric``.
    """
    if not isinstance(input_path, str):
        raise TypeError(
            f"input_path must be str, got {type(input_path).__name__}"
        )
    if not input_path:
        raise ValueError("input_path must be non-empty")
    if "\x00" in input_path:
        raise ValueError("input_path must not contain null bytes")
    if not is_under_cwd(input_path):
        raise ValueError(f"input_path {input_path!r} is outside cwd")
    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)

    control: list[float] = []
    treatment: list[float] = []
    with open(input_path, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            arm = row.get("arm")
            value = row.get(config.metric)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            f_val = float(value)
            if not math.isfinite(f_val):
                continue
            if arm == "control" and len(control) < _MAX_SAMPLES_PER_ARM:
                control.append(f_val)
            elif arm == "treatment" and len(treatment) < _MAX_SAMPLES_PER_ARM:
                treatment.append(f_val)

    return msprt_step(config, control=control, treatment=treatment)


__all__ = [
    "BURN_IN_ROWS_BY_ALPHA",
    "CALIBRATED_HORIZON_ROWS",
    "HIGHER_IS_BETTER",
    "MsprtConfig",
    "MsprtVerdict",
    "SUPPORTED_METRICS",
    "burn_in_is_calibrated",
    "min_rows_per_arm",
    "msprt_step",
    "run_msprt",
    "validate_metric_name",
]
