"""v0.43.0 Part B — KL Divergence calibration framework (Unsloth Calibration_v3/v5).

Compares logits between baseline and quantized models on a small fixed subset
(default: 5-shot MMLU). Pure-math kernel `kl_divergence` operates on numpy
arrays and is safe to call without torch. Live model loading + tokenization
is the caller's responsibility — `run_calibration` accepts pre-computed logit
matrices so the same kernel works for any pair of models the user can load.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


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
