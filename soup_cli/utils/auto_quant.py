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
        # ``bool`` is a subclass of ``int``; reject it explicitly so a caller
        # passing ``score=True`` doesn't sneak past as 1.0.
        if (
            not isinstance(self.score, (int, float))
            or isinstance(self.score, bool)
            or math.isnan(self.score)
        ):
            raise ValueError(f"score must be a finite float, got {self.score!r}")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0.0, 1.0], got {self.score}")
        if (
            not isinstance(self.latency_ms, (int, float))
            or isinstance(self.latency_ms, bool)
            or math.isnan(self.latency_ms)
        ):
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


def evaluate_candidate(
    name: str, *,
    eval_fn,
    prompts,
    min_correct_fraction: float = 0.5,
) -> Candidate:
    """Run ``eval_fn(prompt) -> (response, correct_bool)`` over a small prompt
    set, time it, and produce a Candidate.

    Latency is mean per-prompt ms. Score is the fraction correct. ``ok`` is
    True when score >= ``min_correct_fraction``.

    Robust to ``eval_fn`` crashes — any prompt that raises sets ``ok=False``
    and continues so a single bad prompt doesn't disqualify a candidate that
    works on the rest.
    """
    import time as _time

    if not prompts:
        raise ValueError("evaluate_candidate requires at least one prompt")

    correct = 0
    total = 0
    completed = 0  # for honest latency mean (excludes crashed prompts)
    started = _time.perf_counter()
    crashed = False
    for prompt in prompts:
        total += 1
        try:
            _resp, hit = eval_fn(prompt)
        except Exception:  # noqa: BLE001 — surface as eval failure
            crashed = True
            continue
        completed += 1
        if hit:
            correct += 1
    elapsed_ms = (_time.perf_counter() - started) * 1000.0 / max(1, completed)
    score = correct / total
    return Candidate(
        name=name,
        score=score,
        latency_ms=elapsed_ms,
        ok=(not crashed) and (score >= min_correct_fraction),
    )


def run_auto_quant_picker(
    *, candidate_specs, prompts, min_score: float = 0.90,
) -> Candidate:
    """Run the full pick: evaluate each candidate, pick best by score+latency.

    ``candidate_specs`` is a sequence of ``(name, eval_fn)`` pairs. Each
    ``eval_fn`` takes a prompt and returns ``(response, correct_bool)``.

    Falls back to the highest-scoring candidate (regardless of threshold)
    when no candidate passes ``min_score``, so the server can still bind a
    port. The caller is expected to log the choice.
    """
    candidates = [
        evaluate_candidate(name, eval_fn=fn, prompts=prompts)
        for name, fn in candidate_specs
    ]
    try:
        return pick_best(candidates, min_score=min_score)
    except ValueError:
        # Soft fallback: pick the highest-scored candidate so the server
        # still has a valid choice. Documented as advisory in serve.py.
        ok_candidates = [c for c in candidates if c.ok]
        pool = ok_candidates or candidates
        return max(pool, key=lambda c: (c.score, -c.latency_ms))
