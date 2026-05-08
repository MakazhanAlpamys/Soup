"""Optional LLM-as-a-judge quality filter for harvested preference pairs.

Reuses the v0.19.0 judge backends (OpenAI / Ollama / vLLM-localhost) to
score each ``(chosen, rejected)`` pair on the judge's rubric scale
(default 1-5 helpfulness/accuracy/safety). Pairs whose normalised
``(chosen_score - rejected_score)`` confidence falls below the configured
threshold are dropped.

Added in v0.40.3 (#33 (a)) — closes the rest of the v0.26.1 plan.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass
from typing import Any, Iterable

from soup_cli.data.traces.pair_builder import PreferencePair

logger = logging.getLogger(__name__)

DEFAULT_MIN_CONFIDENCE: float = 0.7

_MIN_THRESHOLD: float = 0.0
_MAX_THRESHOLD: float = 1.0
# DoS cap: an operator-pointed --logs that produces 1M pairs would otherwise
# fan out into 2M judge invocations. Cap matches v0.26.0 trace-line cap.
_MAX_BATCH: int = 100_000


@dataclass(frozen=True)
class JudgeFilterReport:
    """Counts produced by :func:`judge_filter_pairs`."""

    kept: int
    dropped: int
    errors: int


def _validate_threshold(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("min_confidence must be a finite number")
    out = float(value)
    if math.isnan(out):
        raise ValueError("min_confidence must not be NaN")
    if not (_MIN_THRESHOLD <= out <= _MAX_THRESHOLD):
        raise ValueError(
            f"min_confidence must be in [{_MIN_THRESHOLD}, {_MAX_THRESHOLD}], "
            f"got {out}"
        )
    return out


def _normalise_score(raw: float, *, scale_min: int, scale_max: int) -> float:
    if scale_max <= scale_min:
        return 0.0
    return max(0.0, min(1.0, (float(raw) - scale_min) / (scale_max - scale_min)))


def judge_filter_pairs(
    pairs: Iterable[PreferencePair],
    *,
    judge: Any,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> tuple[list[PreferencePair], JudgeFilterReport]:
    """Filter ``pairs`` by LLM-judge confidence.

    The judge scores ``chosen`` and ``rejected`` independently. Each
    weighted score is normalised to ``[0, 1]`` against the rubric scale;
    pairs with ``chosen_norm - rejected_norm >= min_confidence`` are kept.
    Any per-pair exception from the judge backend (network error, parse
    failure) drops the pair and increments the report's ``errors`` field.
    """
    threshold = _validate_threshold(min_confidence)

    # Lazy materialise: peek at most _MAX_BATCH+1 to detect overflow without
    # eagerly buffering a malicious / pathologically large generator.
    pair_list = list(itertools.islice(pairs, _MAX_BATCH + 1))
    if len(pair_list) > _MAX_BATCH:
        raise ValueError(
            f"Too many pairs to judge in one call: > {_MAX_BATCH}"
        )

    scale = (getattr(judge, "rubric", None) or {}).get("scale", {"min": 1, "max": 5})
    scale_min = int(scale.get("min", 1))
    scale_max = int(scale.get("max", 5))

    kept: list[PreferencePair] = []
    dropped = 0
    errors = 0
    for pair in pair_list:
        try:
            chosen_score = float(judge.evaluate(pair.prompt, pair.chosen).weighted_score)
            rejected_score = float(
                judge.evaluate(pair.prompt, pair.rejected).weighted_score
            )
        except Exception as exc:  # noqa: BLE001 — judge backends raise many shapes
            # Mirror v0.33.0 #47 / v0.35.0 policy: log at DEBUG so production
            # silent-degradation is inspectable, but do not crash the harvest.
            logger.debug("judge backend error: %s", exc, exc_info=True)
            errors += 1
            continue
        c_norm = _normalise_score(chosen_score, scale_min=scale_min, scale_max=scale_max)
        r_norm = _normalise_score(
            rejected_score, scale_min=scale_min, scale_max=scale_max,
        )
        if (c_norm - r_norm) >= threshold:
            kept.append(pair)
        else:
            dropped += 1

    return kept, JudgeFilterReport(kept=len(kept), dropped=dropped, errors=errors)
