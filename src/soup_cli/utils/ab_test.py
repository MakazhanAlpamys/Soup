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
are seen, which is what keeps the guarantee exact. It rejects H0 at
`log(1/alpha)`. A `reject_h0` reports its direction, `better` or `worse`, from
the metric's polarity in `HIGHER_IS_BETTER`. Record:
benchmarks/gate-1265-ab-nig.md.

Since #1418 it accepts H0 from the same mixture read the other way: the
differences it does not reject at level alpha form an always-valid confidence
sequence for the difference, and `accept_h0` comes once that sequence lies
inside (-effect_size, +effect_size). By the same Ville argument, a difference of
effect_size or more ends in `accept_h0` in at most an alpha share of runs,
however often the operator looks, so alpha bounds both wrong verdicts and beta
is retired. Record: benchmarks/gate-1418-ab-cs-accept.md.

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

from soup_cli.config.deprecation import warn_deprecated_value
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
# accept_h0 is held at continue while the tested rows' pooled standard deviation
# is more than this many times the held-out rows' (#1265 review). It was added
# because, with a saturated start, the prior scale was far too small and #1265's
# Bayes-factor accept boundary was crossed whether or not there was a real
# difference. #1418's confidence sequence covers at its level for any prior
# scale fixed before the tested rows, so it does not need the hold for its
# guarantee; whether the hold stays is #1524's question, and until then it is
# unchanged. On Gaussian rows the two spreads agree and it changes no verdict.
# Only accepts are held back, so Type-I is untouched.
ACCEPT_HOLD_SPREAD_RATIO = 3.0

def retired_beta_message(beta: str = "beta", alpha: str = "alpha") -> str:
    """What a passed ``beta`` is told (#1418), before the deadline clause.

    The CLI passes its flag names, ``--beta`` and ``--alpha``.
    """
    return (
        f"{beta} no longer does anything (#1418): soup ab now accepts H0 once the "
        f"confidence sequence for the difference at level {alpha} lies inside "
        f"+-effect_size, so {alpha} bounds both a wrong reject_h0 and a wrong "
        f"accept_h0. Remove {beta}; to make accept_h0 stricter, lower {alpha}."
    )


# #1339 - the largest standardised effect whose square is still a float.
MAX_STANDARDISED_EFFECT = math.sqrt(sys.float_info.max)
_MAX_METRIC_NAME_LEN = 32
_MAX_SAMPLES_PER_ARM = 1_000_000
# The prior variance enters the statistic as 1 + n_eff * effect**2, and n_eff
# grows with every row. So effect_size / s0 is checked against the largest n_eff
# the tool accepts (half of _MAX_SAMPLES_PER_ARM), not the row count so far:
# otherwise the same held-out rows would give a silent -inf accept_h0 at a few
# rows and be refused only later, and an operator who stops at the first
# verdict would never see the refusal. About 1.9e151. Any statistic that is
# still non-finite is refused in msprt_step before the boundaries are compared.
_MAX_PRIOR_EFFECT = math.sqrt(sys.float_info.max / (_MAX_SAMPLES_PER_ARM / 2))
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
    alpha: float = 0.05  # bounds both wrong reject_h0 and wrong accept_h0 (#1418)
    beta: float | None = None  # retired in #1418: ignored, with a warning
    effect_size: float = 0.1  # Minimum detectable difference in means

    def __post_init__(self) -> None:
        # Re-validate via canonicalisation so callers bypassing the factory
        # cannot smuggle through a non-canonical metric.
        object.__setattr__(self, "metric", validate_metric_name(self.metric))
        object.__setattr__(self, "alpha", _require_unit_open(self.alpha, field="alpha"))
        # #1418 retired beta: accept_h0 now comes from a confidence sequence at
        # level alpha, so alpha bounds both wrong verdicts and there is no
        # Type-II rate left to set. A beta that is passed is not used, and it
        # says so rather than being dropped silently. #1339's alpha + beta < 1
        # check guarded the old accept boundary and went with it.
        if self.beta is not None:
            warn_deprecated_value(retired_beta_message())
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
    # Squared with * so an overflow is inf, which msprt_step refuses, not a bare
    # OverflowError (** raises).
    t_squared = mean_difference * mean_difference * n_eff / pooled_variance
    spread = 1.0 + n_eff * prior_variance
    return -0.5 * math.log(spread) - 0.5 * (nu + 1) * (
        math.log1p(t_squared / (nu * spread)) - math.log1p(t_squared / nu)
    )


def _nig_confidence_half_width(
    *,
    n_control: int,
    n_treatment: int,
    pooled_variance: float,
    prior_variance: float,
    level: float,
) -> float:
    """Half-width of the always-valid confidence sequence for the difference (#1418).

    Shifting the treatment by a candidate difference delta0 leaves the pooled
    variance as it is and moves t to (mean_difference - delta0) / se, with
    se = sqrt(pooled_variance / n_eff). So the Bayes factor for "delta0 is the
    difference" is ``_nig_log_bayes_factor`` at that t, and it is a martingale
    when delta0 is the true difference. The differences it has not rejected at
    ``1 / level`` are mean_difference +- this half-width, and Ville's inequality
    keeps the true one inside at every look with probability at least
    1 - ``level``.

    The Bayes factor grows with t**2, and the edge has a closed form. With
    spread = 1 + n_eff * g and c = exp(-(2 log(1 / level) + log(spread)) / (nu + 1)),
    log BF = log(1 / level) where t**2 = nu (1 - c) / (c - 1 / spread). When
    c <= 1 / spread no t reaches it (too few rows for this prior): every
    difference is still in the sequence, and the half-width is inf.
    """
    nu = n_control + n_treatment - 2
    n_eff = (n_control * n_treatment) / (n_control + n_treatment)
    spread = 1.0 + n_eff * prior_variance
    c = math.exp(-(2.0 * math.log(1.0 / level) + math.log(spread)) / (nu + 1))
    if c <= 1.0 / spread:
        return math.inf
    t_squared = nu * (1.0 - c) / (c - 1.0 / spread)
    return math.sqrt(t_squared * pooled_variance / n_eff)


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


def _require_finite_means(
    metric: str,
    control: Sequence[float],
    treatment: Sequence[float],
    mean_c: float,
    mean_t: float,
) -> None:
    """#1384 - refuse an arm whose values sum past the largest float.

    Its mean is infinite however little the values spread, so the message names
    their size, not their spread.
    """
    if math.isfinite(mean_c) and math.isfinite(mean_t):
        return
    values = [*control, *treatment]
    raise ValueError(
        f"The {metric!r} column's values are too large for the test "
        f"statistic: they run from {min(values):.3g} to {max(values):.3g}, "
        "and an arm's sum overflows a float. Rescale the metric (divide its "
        "values by a large constant) and re-run."
    )


def _means_and_pooled_variance(
    metric: str, control: Sequence[float], treatment: Sequence[float]
) -> tuple[float, float, float]:
    """Per-arm means and the Bessel-pooled variance (each arm needs ``>= 2`` rows).

    Called for the held-out rows and for the tested rows, so the #1384 refusals
    cover both: an arm sum past the largest float, and deviations whose squares
    are.
    """
    mean_c = sum(control) / len(control)
    mean_t = sum(treatment) / len(treatment)
    _require_finite_means(metric, control, treatment, mean_c, mean_t)
    # Squared as d * d, not d ** 2: ** raises OverflowError where * returns inf,
    # and inf is refused below.
    sum_sq = sum((x - mean_c) * (x - mean_c) for x in control) + sum(
        (x - mean_t) * (x - mean_t) for x in treatment
    )
    variance = sum_sq / (len(control) + len(treatment) - 2)
    # #1384 - deviations past about 1.3e154 square past the largest float.
    if not math.isfinite(variance):
        values = [*control, *treatment]
        raise ValueError(
            f"The {metric!r} column's spread is above what the test "
            f"statistic can represent: its values run from {min(values):.3g} to "
            f"{max(values):.3g}, and their variance overflows a float. Rescale "
            "the metric (divide its values by a large constant) and re-run."
        )
    return mean_c, mean_t, variance


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
        config.metric, control[:scale_rows], treatment[:scale_rows]
    )
    if variance <= 0.0:
        # Rows that differ can still have a variance of 0.0 in floats (0 and
        # 1e-170). Held out, they would set a prior variance of 0 and a continue
        # that never ends, so only rows that are all equal wait for more.
        if len(set(control[:scale_rows])) > 1 or len(set(treatment[:scale_rows])) > 1:
            raise ValueError(
                f"The {config.metric!r} column's spread is below what the test "
                f"statistic can represent: its first {scale_rows} rows per arm "
                "differ, but their variance underflows to 0. Rescale the metric "
                "(multiply its values by a large constant) and re-run."
            )
        return 0.0
    scale = math.sqrt(variance)
    effect = config.effect_size / scale
    too_large = effect > _MAX_PRIOR_EFFECT
    # #1384 - a variance below the smallest normal float is a spread the
    # statistic cannot carry at any --effect-size: say so, and how to fix it.
    if too_large and variance < sys.float_info.min:
        raise ValueError(
            f"The {config.metric!r} column's spread is below what the test "
            f"statistic can represent: its first {scale_rows} rows per arm have a "
            f"pooled standard deviation of {scale:.3g}, which puts --effect-size "
            f"{config.effect_size!r} at {effect:.3g} standard deviations, past the "
            f"{_MAX_PRIOR_EFFECT:.3g} the statistic can carry. Rescale the "
            "metric (multiply its values by a large constant) and re-run."
        )
    if too_large:
        raise ValueError(
            f"--effect-size {config.effect_size!r} is {effect:.3g} standard "
            f"deviations at a pooled standard deviation of {scale:.3g}, and the "
            f"test statistic overflows above {_MAX_PRIOR_EFFECT:.3g}. "
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
    - ``accept_h0``: any difference is smaller than ``effect_size``: the
      confidence sequence at coverage 1 - ``alpha`` lies inside
      (-``effect_size``, +``effect_size``). ``log_likelihood_ratio`` is still
      the Bayes factor against "no difference", so on an accept it can be
      anything below ``log(1 / alpha)``, positive included

    ``mean_control`` / ``mean_treatment`` and the row counts cover every row;
    the statistic and the direction use the rows after the held-out ones.
    """
    ctrl = _validate_sample_list(control, arm="control")
    treat = _validate_sample_list(treatment, arm="treatment")

    n_c, n_t = len(ctrl), len(treat)
    if n_c and n_t:
        _require_finite_means(config.metric, ctrl, treat, sum(ctrl) / n_c, sum(treat) / n_t)

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
    mean_c, mean_t, pooled_variance = _means_and_pooled_variance(
        config.metric, rest_c, rest_t
    )
    # Both tested arms constant: t is 0/0, and without measurement noise
    # sequential testing cannot bound the Type-I error honestly (code-review LOW
    # fix v0.63.0), so keep collecting.
    if pooled_variance <= 0.0:
        return verdict()
    diff = mean_t - mean_c
    prior_variance = standardised_effect * standardised_effect
    llr = _nig_log_bayes_factor(
        n_control=len(rest_c),
        n_treatment=len(rest_t),
        mean_difference=diff,
        pooled_variance=pooled_variance,
        prior_variance=prior_variance,
    )
    # A non-finite statistic cannot be compared with the boundaries: nan crosses
    # neither (a continue that never ends) and -inf is a silent accept_h0.
    if not math.isfinite(llr):
        n_eff = len(rest_c) * len(rest_t) / (len(rest_c) + len(rest_t))
        if not math.isfinite(1.0 + n_eff * prior_variance):
            raise ValueError(
                f"--effect-size {config.effect_size!r} is {standardised_effect:.3g} "
                f"standard deviations of the {config.metric!r} column's held-out "
                f"rows, and at {len(rest_c)} + {len(rest_t)} tested rows the prior "
                "variance of the test statistic overflows a float. Lower "
                f"--effect-size, or check the {config.metric!r} column: a "
                "near-constant held-out column reaches this at any --effect-size."
            )
        raise ValueError(
            f"The {config.metric!r} column's tested rows are outside what the "
            f"test statistic can represent: the arm means differ by {diff:.3g} at "
            f"a pooled variance of {pooled_variance:.3g}, and the t statistic "
            "overflows a float. Check the column for a near-constant arm, or "
            "rescale the metric (divide its values by a large constant) if its "
            "values are very large, and re-run."
        )
    if llr >= math.log(1.0 / config.alpha):
        return verdict(llr, "reject_h0", _direction(config.metric, diff))
    half_width = _nig_confidence_half_width(
        n_control=len(rest_c),
        n_treatment=len(rest_t),
        pooled_variance=pooled_variance,
        prior_variance=prior_variance,
        level=config.alpha,
    )
    if abs(diff) + half_width < config.effect_size:
        # Held-out rows far tighter than the tested ones (a warm cache, a judge
        # that saturates early): hold the accept back until the spreads agree,
        # as #1265 shipped it. Why it is still here, and what decides whether
        # it stays, is at ACCEPT_HOLD_SPREAD_RATIO. A reject is never held
        # back, so the Type-I bound is unchanged.
        held_out_sd = config.effect_size / standardised_effect
        if math.sqrt(pooled_variance) > ACCEPT_HOLD_SPREAD_RATIO * held_out_sd:
            return verdict(llr)
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
    "ACCEPT_HOLD_SPREAD_RATIO",
    "HIGHER_IS_BETTER",
    "MAX_STANDARDISED_EFFECT",
    "MsprtConfig",
    "MsprtVerdict",
    "PRIOR_SCALE_ROWS",
    "SUPPORTED_METRICS",
    "msprt_step",
    "retired_beta_message",
    "run_msprt",
    "validate_metric_name",
]
