"""Auto-quant picker: try multiple quant formats, pick fastest-at-acceptable-quality.

Pure-Python decision engine. Actual model loading + eval is delegated to the
caller (v0.30.0 ships the picker + schema, trainer-side eval loop deferred
to v0.30.1 following the same pattern as v0.28.0 kernel_picker).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

_VALID_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_DEFAULT_ORDER = ("gguf", "awq", "gptq", "fp8", "none")


def default_candidate_order() -> tuple[str, ...]:
    """Canonical order of quant formats to try.

    GGUF first because it's the widest-deployable; AWQ/GPTQ next for 4-bit
    quality; FP8 only on Hopper+; 'none' (baseline) last so we always have
    a fallback to prove any quant is actually helping.
    """
    return _DEFAULT_ORDER


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not _VALID_NAME_RE.match(name):
        raise ValueError(
            f"candidate name must match {_VALID_NAME_RE.pattern}, got {name!r}"
        )


@dataclass(frozen=True)
class Candidate:
    """One candidate quant configuration + its measured score/latency."""

    name: str
    score: float  # quality in [0.0, 1.0]; higher is better
    latency_ms: float
    ok: bool  # False if eval crashed / threshold-invalid output

    def __post_init__(self) -> None:
        _validate_name(self.name)
        if not isinstance(self.score, (int, float)) or math.isnan(self.score):
            raise ValueError(f"score must be a finite float, got {self.score!r}")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0.0, 1.0], got {self.score}")
        if not isinstance(self.latency_ms, (int, float)) or math.isnan(self.latency_ms):
            raise ValueError(f"latency_ms must be a finite float, got {self.latency_ms!r}")
        if self.latency_ms < 0:
            raise ValueError(f"latency_ms must be non-negative, got {self.latency_ms}")
        if not isinstance(self.ok, bool):
            raise ValueError(f"ok must be a bool, got {type(self.ok).__name__}")


def pick_best(
    candidates: Iterable[Candidate],
    *,
    min_score: float = 0.90,
) -> Candidate:
    """Pick the fastest candidate whose score >= min_score.

    Tie-break: first-encountered (stable, matches v0.28.0 kernel_picker).
    Raises ValueError if no candidate passes the threshold.
    """
    if not 0.0 <= min_score <= 1.0:
        raise ValueError(f"min_score must be in [0.0, 1.0], got {min_score}")

    # Materialise so we can both filter and count from a generator input.
    all_candidates = list(candidates)
    pool = [c for c in all_candidates if c.ok and c.score >= min_score]
    if not pool:
        raise ValueError(
            f"no candidate passed min_score={min_score}; "
            f"ran {len(all_candidates)} candidates"
        )
    best = pool[0]
    for cand in pool[1:]:
        if cand.latency_ms < best.latency_ms:
            best = cand
    return best
