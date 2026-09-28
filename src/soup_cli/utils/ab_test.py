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

The test is two-sided (#1227). Since #1265 the statistic is a Bayes factor
that mixes over the variance as well as the mean: the normal-inverse-gamma
mixture in its right-Haar limit (a flat prior on the common mean, 1/sigma on
the standard deviation) with a N(0, g) prior on the standardised difference
delta / sigma. It depends on the rows only through the two-sample t statistic,
so it is scale-invariant, and under H0 it is a test martingale whatever sigma
is. Ville's inequality then bounds the chance that it EVER reaches 1/alpha by
alpha, from the first rows on and at any horizon, with no burn-in.

`effect_size` is in the metric's units, but the prior needs it in standard
deviations. So the first `PRIOR_SCALE_ROWS` rows of each arm are held out:
their pooled standard deviation s0 sets g = (effect_size / s0) ** 2, and the
Bayes factor runs on the rows after them. The prior is fixed before those rows
are seen, which is what keeps the guarantee exact. The boundaries are
`log(1/alpha)` (reject H0) and `log(beta/(1-alpha))` (accept H0). A
`reject_h0` reports its direction, `better` or `worse`, from the metric's
polarity in `HIGHER_IS_BETTER`. Record: benchmarks/gate-1265-ab-nig.md.

Two known limitations:
1. Single metric per pass — multi-metric correction (Bonferroni / Holm)
   is operator-controlled. The CLI accepts one metric at a time.
2. Assumes Gaussian-like data. For binary metrics (e.g. retry_rate as
   a boolean), the operator should pre-aggregate per-prompt rates so the
   resulting per-prompt averages are approximately Gaussian.
"""

from __future__ import annotations

import json
import math
import os
import sys
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
# Rows per arm held out to put `effect_size` in standard deviations (#1265).
# Their pooled standard deviation sets the prior scale, and the Bayes factor
# runs on the rows after them. While both arms are still constant, more rows are
# held out until one varies. This is not a Type-I burn-in: the guarantee holds
# for any value. It trades power at small effect_size / sigma (fewer rows left)
# against power at large ones (a less noisy prior scale). Chosen by the sweep in
# benchmarks/gate-1265-ab-nig.md; changing it means re-running that sweep.
PRIOR_SCALE_ROWS = 5

# #1339 - the largest standardised effect whose square is still a float. The
# prior variance below squares effect_size / s0, and ** raises OverflowError
# where * would have returned inf, so the bound is checked rather than the
# result. Division does not raise: a near-constant held-out column would turn
# the ratio into inf and the Bayes factor into -inf, a silent accept_h0.
MAX_STANDARDISED_EFFECT = math.sqrt(sys.float_info.max)
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
        # #1339 - each rate is in (0, 1) on its own, but the accept boundary
        # log(beta / (1 - alpha)) is below 0, a Bayes factor of 1, only while
        # alpha + beta < 1. At or past 1 it accepts H0 on no evidence either
        # way, and on evidence for a difference: with alpha 0.05 / beta 0.95
        # (a power typed as beta) a true difference of effect_size ends in
        # accept_h0 in 0.96 of runs at 0.3 standard deviations and still 0.36
        # at 2 (1000-2000 runs each, a look after every pair). Under #1227's
        # statistic the same pairs crossed the boundaries instead, and rejected.
        if self.alpha + self.beta >= 1.0:
            raise ValueError(
                f"alpha + beta must be < 1.0, got alpha={self.alpha} + "
                f"beta={self.beta} = {self.alpha + self.beta}. At or above 1.0 "
                "the accept boundary log(beta / (1 - alpha)) is at or above 0, "
                "so the test accepts H0 when the rows show no evidence either "
                "way, or even evidence of a difference. beta is the Type-II "
                "error rate, not the power: a power of 0.95 is beta 0.05."
            )
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


def _direction(metric: str, diff: float) -> str:
    """``better`` / ``worse`` for a treatment-minus-control difference, by polarity."""
    treatment_higher = diff > 0.0
    return "better" if treatment_higher == HIGHER_IS_BETTER[metric] else "worse"


def _nig_log_bayes_factor(
    *,
    n_control: int,
    n_treatment: int,
    mean_difference: float,
    pooled_variance: float,
    prior_variance: float,
) -> float:
    """Log Bayes factor of the normal-inverse-gamma mixture (right-Haar limit).

    H1: the treatment-minus-control difference is delta = d * sigma with
    d ~ N(0, ``prior_variance``); H0: delta = 0. Both share a flat prior on the
    common mean and 1/sigma on sigma, which integrate out in closed form
    (Gönen, Johnson, Lu and Westfall, 2005). With the pooled two-sample t
    statistic, nu = n_c + n_t - 2 and n_eff = n_c * n_t / (n_c + n_t):

        log BF = -0.5 * log(1 + n_eff * g)
                 - (nu + 1) / 2 * [log1p(t**2 / (nu * (1 + n_eff * g)))
                                   - log1p(t**2 / nu)]

    It depends on the rows only through t, so under H0 its distribution does
    not depend on sigma, and as a ratio of two marginal likelihoods it is a
    martingale under H0 in the order the rows arrive. Ville's inequality then
    gives P(sup_n BF_n >= 1/alpha) <= alpha. Needs ``n >= 2`` per arm and
    ``pooled_variance > 0``; split out so the sweep in
    benchmarks/harness/ab_nig_sweep.py can check its vectorised copy
    against exactly this code.
    """
    nu = n_control + n_treatment - 2
    n_eff = (n_control * n_treatment) / (n_control + n_treatment)
    t_squared = mean_difference**2 * n_eff / pooled_variance
    spread = 1.0 + n_eff * prior_variance
    return -0.5 * math.log(spread) - 0.5 * (nu + 1) * (
        math.log1p(t_squared / (nu * spread)) - math.log1p(t_squared / nu)
    )


def _held_out_rows(control: Sequence[float], treatment: Sequence[float]) -> int | None:
    """Rows per arm that set the prior scale, or ``None`` if there are not enough yet.

    ``PRIOR_SCALE_ROWS``, or more while the held-out rows of both arms are still
    constant (their pooled standard deviation would be 0). The count depends
    only on the held-out rows themselves, so the rows after them are still
    untouched by the prior, and the guarantee is unchanged.
    """
    def first_change(values: Sequence[float]) -> int:
        return next((i for i, x in enumerate(values) if x != values[0]), len(values))

    shortest = min(len(control), len(treatment))
    held = max(PRIOR_SCALE_ROWS, min(first_change(control), first_change(treatment)) + 1)
    return held if held <= shortest else None


def _means_and_pooled_variance(
    control: Sequence[float], treatment: Sequence[float]
) -> tuple[float, float, float]:
    """Per-arm means and the Bessel-pooled variance (each arm needs ``>= 2`` rows)."""
    mean_c = sum(control) / len(control)
    mean_t = sum(treatment) / len(treatment)
    sum_sq = sum((x - mean_c) ** 2 for x in control) + sum(
        (x - mean_t) ** 2 for x in treatment
    )
    return mean_c, mean_t, sum_sq / (len(control) + len(treatment) - 2)


def _standardised_effect(
    config: MsprtConfig,
    control: Sequence[float],
    treatment: Sequence[float],
    *,
    scale_rows: int,
) -> float:
    """``effect_size`` over the pooled standard deviation of the first ``scale_rows``.

    Those are the held-out rows once there are enough of them, and until then
    every row so far, which are the first of them. Checking that early refuses an
    impossible ``--effect-size`` on the first run instead of after
    ``PRIOR_SCALE_ROWS`` rows (#1339). Returns 0.0 while the scale is not known yet
    (fewer than 2 rows per arm, or all of them equal); the caller does not use it
    then.
    """
    if scale_rows < 2:
        return 0.0
    _, _, variance = _means_and_pooled_variance(
        control[:scale_rows], treatment[:scale_rows]
    )
    if variance <= 0.0:
        return 0.0
    scale = math.sqrt(variance)
    effect = config.effect_size / scale
    if effect > MAX_STANDARDISED_EFFECT:
        raise ValueError(
            f"--effect-size {config.effect_size!r} is {effect:.3g} standard "
            f"deviations at a pooled standard deviation of {scale:.3g}, and the "
            f"test statistic overflows above {MAX_STANDARDISED_EFFECT:.3g}. "
            f"Lower --effect-size, or check the {config.metric!r} column: a "
            "near-constant column reaches this at any --effect-size."
        )
    return effect


def msprt_step(
    config: MsprtConfig,
    *,
    control: Sequence[float],
    treatment: Sequence[float],
) -> MsprtVerdict:
    """Run a single two-sided mSPRT decision step.

    Returns ``MsprtVerdict`` with one of:
    - ``continue``: keep collecting samples; always the answer until each arm
      has at least 2 rows past the ``PRIOR_SCALE_ROWS`` held-out ones
    - ``reject_h0``: difference is real (treatment != control), in either
      direction; ``direction`` says whether the treatment is ``better`` or
      ``worse`` than control, by the metric's polarity
    - ``accept_h0``: difference is not significant

    ``mean_control`` / ``mean_treatment`` and the row counts cover every row;
    the statistic and the direction use the rows after the held-out ones.
    """
    ctrl = _validate_sample_list(control, arm="control")
    treat = _validate_sample_list(treatment, arm="treatment")

    n_c, n_t = len(ctrl), len(treat)

    def verdict(llr: float = 0.0, decision: str = "continue", direction: str | None = None):
        return MsprtVerdict(
            decision=decision,
            log_likelihood_ratio=llr,
            n_control=n_c,
            n_treatment=n_t,
            mean_control=sum(ctrl) / n_c if n_c else 0.0,
            mean_treatment=sum(treat) / n_t if n_t else 0.0,
            direction=direction,
        )

    held = _held_out_rows(ctrl, treat)
    standardised_effect = _standardised_effect(
        config, ctrl, treat, scale_rows=held if held is not None else min(n_c, n_t)
    )
    if held is None or n_c - held < 2 or n_t - held < 2:
        return verdict()
    rest_c, rest_t = ctrl[held:], treat[held:]
    mean_c, mean_t, pooled_variance = _means_and_pooled_variance(rest_c, rest_t)
    # Both tested arms constant: t is 0/0, and without measurement noise
    # sequential testing cannot bound the Type-I error honestly (code-review LOW
    # fix v0.63.0), so keep collecting.
    if pooled_variance <= 0.0:
        return verdict()
    diff = mean_t - mean_c
    llr = _nig_log_bayes_factor(
        n_control=len(rest_c),
        n_treatment=len(rest_t),
        mean_difference=diff,
        pooled_variance=pooled_variance,
        prior_variance=standardised_effect**2,
    )
    if llr >= math.log(1.0 / config.alpha):
        return verdict(llr, "reject_h0", _direction(config.metric, diff))
    if llr <= math.log(config.beta / (1.0 - config.alpha)):
        return verdict(llr, "accept_h0")
    return verdict(llr)


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
    "HIGHER_IS_BETTER",
    "MAX_STANDARDISED_EFFECT",
    "MsprtConfig",
    "MsprtVerdict",
    "PRIOR_SCALE_ROWS",
    "SUPPORTED_METRICS",
    "msprt_step",
    "run_msprt",
    "validate_metric_name",
]
