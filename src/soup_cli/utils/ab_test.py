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
space, via logsumexp), never as log-ratios and never via `|z|`. Each point
ratio is a martingale under H0 and so is their average, so the boundaries
`log(beta/(1-alpha))` (accept H0) and `log((1-beta)/alpha)` (reject H0) keep
their any-time meaning. A `reject_h0` reports its direction, `better` or
`worse`, from the metric's polarity in `HIGHER_IS_BETTER`.

Three known limitations:
1. Single metric per pass — multi-metric correction (Bonferroni / Holm)
   is operator-controlled. The CLI accepts one metric at a time.
2. Assumes Gaussian-like data. For binary metrics (e.g. retry_rate as
   a boolean), the operator should pre-aggregate per-prompt rates so the
   resulting per-prompt averages are approximately Gaussian.
3. The variance is estimated from the rows, not known. The martingale
   argument holds for a known variance; with only a few rows per arm the
   estimate is noisy enough to push the Type-I rate under peeking above
   `alpha` when `effect_size` is comparable to the row-to-row noise.
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
    row at every peek.
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
    # each LR is a martingale under H0 (E[LR_n] = 1), and so is their average,
    # which is what makes the statistic below keep Wald's boundaries at every
    # stopping time per the optional stopping theorem (here the variance is
    # estimated; see limitation 3 in the module docstring).
    #
    # It must be the average of the RATIOS. The mean of the log-ratios is just
    # -drift (it ignores the data), and |z| in the one-sided formula is the
    # larger of the two log-ratios, which is not a martingale and inflates
    # Type-I error under peeking.
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
    if llr >= upper:
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
    - ``continue``: keep collecting samples
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
    "HIGHER_IS_BETTER",
    "MsprtConfig",
    "MsprtVerdict",
    "SUPPORTED_METRICS",
    "msprt_step",
    "run_msprt",
    "validate_metric_name",
]
