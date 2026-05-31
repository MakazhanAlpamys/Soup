"""KL-divergence quant calibration (v0.43.0 Part B) + Judge calibration with
conformal abstention (v0.65.0 Part A).

The v0.43.0 surface (``kl_divergence`` / ``classify_kl_delta`` /
``run_calibration``) compares logits between baseline and quantized models on
a small fixed subset (default: 5-shot MMLU). Pure-math kernel ``kl_divergence``
operates on numpy arrays and is safe to call without torch. Live model loading
+ tokenization is the caller's responsibility — ``run_calibration`` accepts
pre-computed logit matrices so the same kernel works for any pair of models
the user can load.

The v0.65.0 Part A surface adds SCOPE/CJE-style bidirectional pairwise judging:
``PairwiseJudgement`` carries first/second/oracle winners for one prompt;
``fit_position_bias`` returns a coefficient in ``[-1, 1]`` measuring the
judge's preference for the first slot; ``conformal_threshold`` emits the
``alpha``-coverage threshold from a calibration set of judge confidence
scores; ``run_pairwise_calibration`` is the orchestrator returning a frozen
``JudgeCalibrationReport``; ``ensure_judge_calibrated`` is the production
gate that refuses to score with an uncalibrated judge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

# Allowed winner labels in a pairwise judgement.
_WINNER_VALUES = frozenset({"a", "b", "tie"})

# DoS / sanity caps.
_MAX_PAIRS = 50_000


@dataclass(frozen=True)
class CalibrationReport:
    """Per-prompt KL divergence + corpus mean.

    `delta_status` follows v0.26.0 Part D quant-check policy:
      - "OK"     : mean_kl < 0.05
      - "MINOR"  : 0.05 <= mean_kl < 0.20
      - "MAJOR"  : mean_kl >= 0.20
    """

    mean_kl: float
    per_prompt_kl: tuple[float, ...]
    delta_status: str
    num_prompts: int


def _softmax(logits: Sequence[float]) -> list[float]:
    if not logits:
        raise ValueError("logits must not be empty")
    max_l = max(logits)
    exps = [math.exp(x - max_l) for x in logits]
    s = sum(exps)
    if s <= 0:
        raise ValueError("softmax denominator non-positive")
    return [e / s for e in exps]


def kl_divergence(p_logits: Sequence[float], q_logits: Sequence[float]) -> float:
    """KL(P || Q) over discrete distributions derived from logits.

    Both inputs must be the same length and contain finite floats.
    Returns a non-negative float; uses natural log.
    """
    if len(p_logits) != len(q_logits):
        raise ValueError(
            f"p_logits ({len(p_logits)}) and q_logits "
            f"({len(q_logits)}) must have the same length"
        )
    for name, logits in (("p_logits", p_logits), ("q_logits", q_logits)):
        if not logits:
            raise ValueError(f"{name} must not be empty")
        for x in logits:
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                raise ValueError(f"{name} must contain only int/float")
            if not math.isfinite(float(x)):
                raise ValueError(f"{name} must be finite")
    p = _softmax(p_logits)
    q = _softmax(q_logits)
    kl = 0.0
    for pi, qi in zip(p, q):
        if pi <= 0:
            continue
        # qi could be ~0; floor with epsilon to avoid log(0).
        kl += pi * math.log(pi / max(qi, 1e-12))
    # Floating-point can produce tiny negatives; clamp.
    return max(0.0, kl)


def classify_kl_delta(mean_kl: float) -> str:
    """OK / MINOR / MAJOR thresholds (v0.26.0 Part D policy)."""
    if isinstance(mean_kl, bool) or not isinstance(mean_kl, (int, float)):
        raise ValueError("mean_kl must be a number")
    if not math.isfinite(float(mean_kl)) or mean_kl < 0:
        raise ValueError("mean_kl must be a finite non-negative number")
    if mean_kl < 0.05:
        return "OK"
    if mean_kl < 0.20:
        return "MINOR"
    return "MAJOR"


def run_calibration(
    baseline_logits: Sequence[Sequence[float]],
    quantized_logits: Sequence[Sequence[float]],
) -> CalibrationReport:
    """Run calibration on aligned baseline + quantized logit pairs.

    Each entry of `*_logits` is the next-token-logit row for one prompt
    (shape: (vocab,)). Both must have the same outer length.
    """
    base_list = list(baseline_logits)
    quant_list = list(quantized_logits)
    if len(base_list) != len(quant_list):
        raise ValueError(
            f"baseline_logits ({len(base_list)}) and quantized_logits "
            f"({len(quant_list)}) must have the same length"
        )
    if not base_list:
        raise ValueError("at least one prompt is required")
    if len(base_list) > 10_000:
        raise ValueError(
            f"too many prompts ({len(base_list)}); cap is 10000"
        )
    per_prompt = tuple(
        kl_divergence(b, q) for b, q in zip(base_list, quant_list)
    )
    mean = sum(per_prompt) / len(per_prompt)
    return CalibrationReport(
        mean_kl=mean,
        per_prompt_kl=per_prompt,
        delta_status=classify_kl_delta(mean),
        num_prompts=len(per_prompt),
    )


# ─── v0.65.0 Part A — Judge calibration with conformal abstention ───


def _validate_winner(value: object, field: str) -> str:
    """Validate a pairwise winner label ∈ {a, b, tie}."""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string, got {type(value).__name__}")
    if "\x00" in value:
        raise ValueError(f"{field} contains null byte")
    if value not in _WINNER_VALUES:
        raise ValueError(
            f"{field} must be one of {sorted(_WINNER_VALUES)}, got {value!r}"
        )
    return value


def _validate_prompt_id(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("prompt_id must be a string")
    if "\x00" in value:
        raise ValueError("prompt_id contains null byte")
    if not value:
        raise ValueError("prompt_id must not be empty")
    if len(value) > 256:
        raise ValueError("prompt_id exceeds 256 chars")
    return value


@dataclass(frozen=True)
class PairwiseJudgement:
    """One SCOPE/CJE bidirectional pairwise judgement.

    The same (model_a, model_b) pair is judged twice with positions swapped:
    ``first_winner`` is the verdict when model_a appears first;
    ``second_winner`` is the verdict when model_b appears first. ``oracle``
    is the ground-truth winner from a held-out oracle set (human or stronger
    LLM judge). All three values are labels in {a, b, tie}.
    """

    prompt_id: str
    first_winner: str
    second_winner: str
    oracle: str

    def __post_init__(self) -> None:
        # Bypass frozen-dataclass setattr via object.__setattr__ so validation
        # can normalise (or just raise) without mutating the user's view.
        object.__setattr__(self, "prompt_id", _validate_prompt_id(self.prompt_id))
        object.__setattr__(
            self, "first_winner",
            _validate_winner(self.first_winner, "first_winner"),
        )
        object.__setattr__(
            self, "second_winner",
            _validate_winner(self.second_winner, "second_winner"),
        )
        object.__setattr__(
            self, "oracle", _validate_winner(self.oracle, "oracle"),
        )


def fit_position_bias(judgements: Iterable[PairwiseJudgement]) -> float:
    """Estimate the judge's preference for the first slot.

    Returns a coefficient in ``[-1, 1]``:

    - 0.0 → no position bias (judge always agrees with itself across swaps)
    - +1.0 → judge always picks the first slot ("a") regardless of swap
    - -1.0 → judge always picks the second slot

    The metric is computed as the rate at which ``first_winner`` and
    ``second_winner`` disagree (proxy for position-dependent flipping).
    A perfectly-consistent judge returns 0.0.
    """
    if not isinstance(judgements, (list, tuple)):
        # Materialise iterables but reject non-iterables loudly.
        try:
            iter(judgements)
        except TypeError:
            raise TypeError(
                "judgements must be iterable of PairwiseJudgement"
            ) from None
        judgements = list(judgements)
    if not judgements:
        raise ValueError("judgements must not be empty")

    # Count flips weighted by direction. If the judge picks "first slot"
    # in both arrangements, that's a positive position bias of +1 for that
    # row. If it consistently picks the "second slot", that's -1.
    total = 0
    signed_sum = 0.0
    for j in judgements:
        if not isinstance(j, PairwiseJudgement):
            raise TypeError(
                "judgements must be PairwiseJudgement instances"
            )
        # For each row, compute +1 if judge picked first-slot label both
        # times, -1 if picked second-slot both times, 0 if mixed/tie.
        # "first" arrangement: a is in slot 1, b is in slot 2
        # "second" arrangement: b is in slot 1, a is in slot 2
        if j.first_winner == "tie" or j.second_winner == "tie":
            total += 1
            continue
        # First slot in arrangement 1 corresponds to label "a";
        # first slot in arrangement 2 corresponds to label "b".
        chose_first_slot_1 = j.first_winner == "a"
        chose_first_slot_2 = j.second_winner == "b"
        if chose_first_slot_1 and chose_first_slot_2:
            signed_sum += 1.0
        elif (not chose_first_slot_1) and (not chose_first_slot_2):
            signed_sum -= 1.0
        # Mixed = consistent across swaps → 0 contribution.
        total += 1

    if total == 0:
        return 0.0
    return max(-1.0, min(1.0, signed_sum / total))


def conformal_threshold(
    scores: Sequence[float],
    *,
    alpha: float,
) -> float:
    """Return the alpha-quantile of the calibration scores.

    With ``alpha=0.1`` and a well-calibrated set of judge confidence
    scores in ``[0, 1]``, predictions with confidence below the returned
    threshold should be abstained from at production time to preserve
    1-alpha coverage.

    Edge cases:
    - ``alpha=0.0`` → return ``min(scores)`` (abstain on nothing)
    - ``alpha=1.0`` → return ``max(scores)`` (abstain on everything below max)
    """
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("alpha must be a number")
    if not math.isfinite(float(alpha)):
        raise ValueError("alpha must be finite")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0.0, 1.0]")
    if not scores:
        raise ValueError("scores must not be empty")

    flat: list[float] = []
    for s in scores:
        if isinstance(s, bool) or not isinstance(s, (int, float)):
            raise ValueError("scores must contain only int/float")
        if not math.isfinite(float(s)):
            raise ValueError("scores must be finite")
        if not 0.0 <= s <= 1.0:
            raise ValueError("scores must be in range [0.0, 1.0]")
        flat.append(float(s))

    sorted_scores = sorted(flat)
    if alpha == 0.0:
        return sorted_scores[0]
    if alpha == 1.0:
        return sorted_scores[-1]
    # Type-1 quantile (lower interpolation).
    n = len(sorted_scores)
    idx = int(math.floor(alpha * n))
    idx = max(0, min(n - 1, idx))
    return sorted_scores[idx]


@dataclass(frozen=True)
class JudgeCalibrationReport:
    """Calibration verdict for a pairwise judge.

    ``position_bias`` ∈ [-1, 1] (0 = no bias).
    ``conformal_threshold`` ∈ [0, 1] (production gate threshold).
    ``agreement_rate`` ∈ [0, 1] (judge-vs-oracle agreement).
    ``num_pairs`` is the calibration set size.
    ``calibrated`` is False if calibration was rejected (e.g. too few pairs,
    extreme bias, etc.) and the judge MUST NOT be used in production.
    """

    position_bias: float
    conformal_threshold: float
    agreement_rate: float
    num_pairs: int
    calibrated: bool

    def __post_init__(self) -> None:
        for field, value in (
            ("position_bias", self.position_bias),
            ("conformal_threshold", self.conformal_threshold),
            ("agreement_rate", self.agreement_rate),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field} must be a number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{field} must be finite")
        if not -1.0 <= self.position_bias <= 1.0:
            raise ValueError("position_bias must be in [-1.0, 1.0]")
        if not 0.0 <= self.conformal_threshold <= 1.0:
            raise ValueError("conformal_threshold must be in [0.0, 1.0]")
        if not 0.0 <= self.agreement_rate <= 1.0:
            raise ValueError("agreement_rate must be in [0.0, 1.0]")
        if isinstance(self.num_pairs, bool) or not isinstance(self.num_pairs, int):
            raise ValueError("num_pairs must be int")
        if self.num_pairs < 0:
            raise ValueError("num_pairs must be non-negative")
        if not isinstance(self.calibrated, bool):
            raise ValueError("calibrated must be a bool")


def run_pairwise_calibration(
    judgements: Sequence[PairwiseJudgement],
    *,
    scores: Sequence[float],
    alpha: float = 0.1,
    min_agreement: float = 0.7,
    max_bias: float = 0.3,
) -> JudgeCalibrationReport:
    """End-to-end pairwise calibration.

    Fits position bias, computes conformal threshold from confidence
    ``scores``, measures agreement with the oracle, and returns a frozen
    report. ``calibrated=False`` is set when agreement falls below
    ``min_agreement`` or position bias exceeds ``max_bias``.
    """
    judgement_list = list(judgements)
    score_list = list(scores)
    if not judgement_list:
        raise ValueError("at least one calibration pairs entry is required")
    if len(judgement_list) != len(score_list):
        raise ValueError(
            f"judgements ({len(judgement_list)}) and scores "
            f"({len(score_list)}) must have the same length"
        )
    if len(judgement_list) > _MAX_PAIRS:
        raise ValueError(
            f"too many pairs ({len(judgement_list)}); cap is {_MAX_PAIRS}"
        )

    bias = fit_position_bias(judgement_list)
    threshold = conformal_threshold(score_list, alpha=alpha)

    # Agreement rate: how often the judge's verdict (treating "first" as the
    # primary arrangement) matches the oracle.
    agree = sum(1 for j in judgement_list if j.first_winner == j.oracle)
    agreement_rate = agree / len(judgement_list)

    calibrated = (
        agreement_rate >= min_agreement
        and abs(bias) <= max_bias
    )

    return JudgeCalibrationReport(
        position_bias=bias,
        conformal_threshold=threshold,
        agreement_rate=agreement_rate,
        num_pairs=len(judgement_list),
        calibrated=calibrated,
    )


def ensure_judge_calibrated(
    report: Optional[JudgeCalibrationReport],
    *,
    min_agreement: float = 0.7,
    max_bias: float = 0.3,
) -> None:
    """Production gate — raise ``RuntimeError`` if judge is not calibrated.

    Call this before using a judge in production scoring. Refuses on:
    - ``report is None`` (no calibration ran)
    - ``report.calibrated is False``
    - ``report.agreement_rate < min_agreement``
    - ``abs(report.position_bias) > max_bias``
    """
    if report is None:
        raise RuntimeError(
            "Judge is not calibrated. Run `soup eval design --calibrate "
            "<oracle-set>` first."
        )
    if not isinstance(report, JudgeCalibrationReport):
        raise TypeError(
            f"report must be JudgeCalibrationReport, got {type(report).__name__}"
        )
    if not report.calibrated:
        raise RuntimeError(
            "Judge calibration failed; refusing to score in production. "
            f"agreement={report.agreement_rate:.2f}, "
            f"bias={report.position_bias:+.2f}"
        )
    if report.agreement_rate < min_agreement:
        raise RuntimeError(
            f"Judge agreement {report.agreement_rate:.2f} below floor "
            f"{min_agreement:.2f} — refusing production use."
        )
    if abs(report.position_bias) > max_bias:
        raise RuntimeError(
            f"Judge position bias |{report.position_bias:+.2f}| above ceiling "
            f"{max_bias:.2f} — refusing production use."
        )
