"""FailureReport + FailureScore frozen dataclasses + verdict taxonomy.

Same OK / MINOR / MAJOR taxonomy as v0.26.0 Part D Quant-Lobotomy. A
score in ``[0.85, 1.0]`` is OK; ``[0.60, 0.85)`` is MINOR; below 0.60
is MAJOR. Thresholds are constants so future tuning lands in one place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Tuple

FAILURE_MODES: Tuple[str, ...] = (
    "forgetting",
    "refusal",
    "format",
    "mode_collapse",
    "memorization",
    "contamination",
)

VERDICTS: Tuple[str, ...] = ("OK", "MINOR", "MAJOR")

# Thresholds (lower bound for verdict). Score >= 0.85 → OK; >= 0.60 → MINOR.
_OK_THRESHOLD: float = 0.85
_MINOR_THRESHOLD: float = 0.60

# Exposed read-only for callers that want to render the thresholds.
THRESHOLDS: Mapping[str, float] = MappingProxyType(
    {"ok": _OK_THRESHOLD, "minor": _MINOR_THRESHOLD}
)


def classify_score(score: float) -> str:
    """Map ``score in [0, 1]`` → ``OK`` / ``MINOR`` / ``MAJOR``.

    Rejects bool (subclass of int — matches project bool-as-int policy),
    non-finite, and out-of-range input loudly.
    """
    if isinstance(score, bool):
        raise TypeError("score must be float, not bool")
    if not isinstance(score, (int, float)):
        raise TypeError(f"score must be float, got {type(score).__name__}")
    value = float(score)
    if not math.isfinite(value):
        raise ValueError("score must be finite")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"score must be in [0, 1], got {value}")
    if value >= _OK_THRESHOLD:
        return "OK"
    if value >= _MINOR_THRESHOLD:
        return "MINOR"
    return "MAJOR"


@dataclass(frozen=True)
class FailureScore:
    """Per-mode score with an OK / MINOR / MAJOR verdict + evidence line."""

    mode: str
    score: float
    verdict: str
    evidence: str

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or not self.mode:
            raise ValueError("mode must be non-empty str")
        if self.mode not in FAILURE_MODES:
            raise ValueError(f"unknown failure mode {self.mode!r}")
        # Re-validate score / verdict via classify_score for consistency.
        expected = classify_score(self.score)
        if not isinstance(self.verdict, str) or self.verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}, got {self.verdict!r}")
        if self.verdict != expected:
            raise ValueError(
                f"verdict {self.verdict!r} disagrees with score "
                f"{self.score} (expected {expected!r})"
            )
        if not isinstance(self.evidence, str):
            raise TypeError("evidence must be str")
        if "\x00" in self.evidence:
            raise ValueError("evidence must not contain null bytes")
        if len(self.evidence) > 4096:
            raise ValueError("evidence too int (max 4096 chars)")


def overall_verdict(scores: Mapping[str, FailureScore]) -> str:
    """Worst-case across all modes; empty → ``OK``.

    Used as the headline badge value. MAJOR wins, then MINOR, then OK.
    """
    if not isinstance(scores, Mapping):
        raise TypeError("scores must be Mapping[str, FailureScore]")
    worst = "OK"
    rank = {"OK": 0, "MINOR": 1, "MAJOR": 2}
    for value in scores.values():
        if not isinstance(value, FailureScore):
            raise TypeError("every entry must be FailureScore")
        if rank[value.verdict] > rank[worst]:
            worst = value.verdict
    return worst


@dataclass(frozen=True)
class FailureReport:
    """Aggregated report card returned by ``soup.diagnose``.

    ``run_id`` / ``base`` / ``adapter`` are stored as plain strings (no
    path-containment here — the CLI boundary does that). ``scores`` is a
    read-only Mapping so callers cannot mutate post-construction.
    """

    run_id: str
    base: str
    adapter: str
    scores: Mapping[str, FailureScore]
    overall: str
    soup_version: str = ""
    extras: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        for attr in ("run_id", "base", "adapter", "soup_version"):
            value = getattr(self, attr)
            if not isinstance(value, str):
                raise TypeError(f"{attr} must be str")
            if "\x00" in value:
                raise ValueError(f"{attr} must not contain null bytes")
            if len(value) > 512:
                raise ValueError(f"{attr} too int (max 512 chars)")
        if not isinstance(self.scores, Mapping):
            raise TypeError("scores must be Mapping[str, FailureScore]")
        for key, value in self.scores.items():
            if key not in FAILURE_MODES:
                raise ValueError(f"unknown failure mode key {key!r}")
            if not isinstance(value, FailureScore):
                raise TypeError(f"scores[{key!r}] must be FailureScore")
            if value.mode != key:
                raise ValueError(
                    f"scores[{key!r}].mode={value.mode!r} mismatch"
                )
        if self.overall not in VERDICTS:
            raise ValueError(f"overall must be one of {VERDICTS}")
        if not isinstance(self.extras, Mapping):
            raise TypeError("extras must be Mapping[str, str]")
        # Freeze scores + extras to MappingProxyType so they round-trip
        # immutable through dict() / to_dict() without exposing a mutable view.
        object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))
        object.__setattr__(self, "extras", MappingProxyType(dict(self.extras)))

    def to_dict(self) -> dict:
        """Serialisable shape — used by ``write_report`` + badge renderer."""
        return {
            "run_id": self.run_id,
            "base": self.base,
            "adapter": self.adapter,
            "overall": self.overall,
            "soup_version": self.soup_version,
            "scores": {
                key: {
                    "mode": value.mode,
                    "score": value.score,
                    "verdict": value.verdict,
                    "evidence": value.evidence,
                }
                for key, value in self.scores.items()
            },
            "extras": dict(self.extras),
        }


def compose_report(
    *,
    run_id: str,
    base: str,
    adapter: str,
    scores: Mapping[str, FailureScore],
    soup_version: str = "",
    extras: Mapping[str, str] | None = None,
) -> FailureReport:
    """Build a ``FailureReport`` with the overall verdict computed for you."""
    overall = overall_verdict(scores)
    return FailureReport(
        run_id=run_id,
        base=base,
        adapter=adapter,
        scores=scores,
        overall=overall,
        soup_version=soup_version,
        extras=extras or {},
    )
