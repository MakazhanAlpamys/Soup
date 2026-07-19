"""Adversarial verifier probe — `soup reward stress` (v0.71.41).

Turn the v0.71.26 reward-hacking expertise on a reward VERIFIER itself: does it
pay out for degenerate completions (empty / length-padded / repetition /
sentinel-spam)? A correct deterministic verifier rejects all of them.

Pure, offline, NO top-level torch. Reuses the attack-kind vocabulary of
``reward_hack_control`` (``SHAPING_KINDS`` = length/repetition/sentinel) — as
ATTACKS rather than mitigations — plus the trivial ``empty`` case. Companion to
``reward_synth`` (v0.71.40): ``synth`` proves a verifier separates references
from *friendly* perturbations; ``stress`` asks the *adversarial* question a
reward-hacking model asks at train time.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from soup_cli.utils.reward_hack_control import _DEFAULT_SENTINEL

# Public re-export so the CLI does not reach into a private cross-module symbol.
DEFAULT_SENTINEL = _DEFAULT_SENTINEL

# ``empty`` + the three ``reward_hack_control.SHAPING_KINDS``.
ATTACKS: tuple[str, ...] = ("empty", "length", "repetition", "sentinel")

# > ``reward_hack_control._SHAPING_LENGTH_SAT`` (32) so a length-reward saturates.
_LENGTH_ATTACK_WORDS = 60
_REPETITION_COUNT = 40
_SENTINEL_COUNT = 20
# O(golds) forward calls per attack — cap the sampled golds to bound the run.
_MAX_STRESS_GOLDS = 200

DEFAULT_THRESHOLD = 0.5
DEFAULT_MAX_GAMEABLE = 0.0


# ---------------------------------------------------------------------------
# Frozen result types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AttackResult:
    kind: str
    n: int
    accepted: int
    accept_rate: float


@dataclass(frozen=True)
class StressReport:
    reference_accept: Optional[float]
    attacks: tuple[AttackResult, ...]
    gameability: float
    gameable: bool
    threshold: float
    max_gameable: float
    sentinel: str


# ---------------------------------------------------------------------------
# Attack generation
# ---------------------------------------------------------------------------
def _attack_text(kind: str, sentinel: str) -> str:
    if kind == "empty":
        return ""
    if kind == "length":
        # A long ramble with NO answer — games a length-based reward.
        return " ".join(["padding"] * _LENGTH_ATTACK_WORDS)
    if kind == "repetition":
        return " ".join(["loop"] * _REPETITION_COUNT)
    if kind == "sentinel":
        # Spam a magic token — games a "contains a sentinel" reward.
        return " ".join([sentinel] * _SENTINEL_COUNT)
    raise ValueError(f"unknown attack kind: {kind!r} (options: {', '.join(ATTACKS)})")


def generate_attacks(
    *, sentinel: str = _DEFAULT_SENTINEL, kinds: Sequence[str] = ATTACKS
) -> list[tuple[str, str]]:
    """Deterministic ``(kind, completion_text)`` junk a correct verifier must reject."""
    if isinstance(kinds, (str, bytes)):
        # A bare string is a Sequence of characters — iterating it would probe
        # per-letter. Force an explicit collection (mirrors reward_synth guards).
        raise TypeError("kinds must be a sequence of attack-kind strings, not a str")
    return [(kind, _attack_text(kind, sentinel)) for kind in kinds]


# ---------------------------------------------------------------------------
# Scoring + verdict
# ---------------------------------------------------------------------------
def _score_batch(
    reward_fn: Callable[..., Sequence[Any]],
    texts: Sequence[str],
    answers: Optional[Sequence[str]],
) -> list[float]:
    """Score completions built from ``texts``; supply ``answer=`` only when given.

    Validates the reward fn returned exactly one finite numeric score per
    completion. A short return (the classic case: a gold-requiring builtin like
    ``accuracy`` scored with no ``answer`` — its ``zip(completions, answers)``
    yields an empty list) is a hard error, NOT a silent "0 accepted" that would
    render a false "robust" verdict. A non-finite score means a broken verifier.
    """
    if not texts:
        return []
    completions = [[{"role": "assistant", "content": t}] for t in texts]
    kwargs = {"answer": list(answers)} if answers is not None else {}
    scores = list(reward_fn(completions, **kwargs))
    if len(scores) != len(texts):
        raise ValueError(
            f"reward target returned {len(scores)} score(s) for {len(texts)} "
            "completion(s) — this verifier likely needs --references to be probed "
            "meaningfully (its reward compares each completion against a gold answer)"
        )
    coerced: list[float] = []
    for s in scores:
        try:
            val = float(s)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"reward target returned a non-numeric score {s!r}") from exc
        if not math.isfinite(val):
            raise ValueError(f"reward target returned a non-finite score {val!r}")
        coerced.append(val)
    return coerced


def run_stress(
    reward_fn: Callable[..., Sequence[Any]],
    golds: Sequence[str],
    *,
    sentinel: str = _DEFAULT_SENTINEL,
    threshold: float = DEFAULT_THRESHOLD,
    max_gameable: float = DEFAULT_MAX_GAMEABLE,
    attacks: Sequence[str] = ATTACKS,
) -> StressReport:
    """Score adversarial junk completions and flag a gameable verifier.

    For each attack kind, one junk completion is scored per sampled gold against
    the REAL gold (so numeric/tool_call/json_schema verifiers get a valid
    ``answer=`` and still must reject the junk). ``gameability`` is the overall
    junk accept-rate; ``gameable`` iff it strictly exceeds ``max_gameable``.

    ``reference_accept`` (the golds scored as their own correct completions) is
    reported for context — a verifier that rejects everything is broken, a
    different problem — but does NOT set the verdict.

    No-gold fallback: with an empty ``golds`` each attack is scored once with no
    ``answer`` kwarg and ``reference_accept`` is ``None``.
    """
    sampled = list(golds)[:_MAX_STRESS_GOLDS]
    have_golds = bool(sampled)

    def _accepted(scores: Sequence[float]) -> int:
        # Scores are already validated finite + numeric by _score_batch.
        return sum(1 for s in scores if s >= threshold)

    reference_accept: Optional[float] = None
    if have_golds:
        ref_scores = _score_batch(reward_fn, sampled, sampled)
        reference_accept = _accepted(ref_scores) / len(sampled)

    results: list[AttackResult] = []
    total_accepted = total_n = 0
    for kind, junk in generate_attacks(sentinel=sentinel, kinds=attacks):
        if have_golds:
            texts: list[str] = [junk] * len(sampled)
            answers: Optional[list[str]] = list(sampled)
        else:
            texts = [junk]
            answers = None
        scores = _score_batch(reward_fn, texts, answers)
        accepted = _accepted(scores)
        n = len(texts)
        results.append(AttackResult(kind, n, accepted, accepted / n if n else 0.0))
        total_accepted += accepted
        total_n += n

    gameability = (total_accepted / total_n) if total_n else 0.0
    return StressReport(
        reference_accept=reference_accept,
        attacks=tuple(results),
        gameability=gameability,
        gameable=gameability > max_gameable,
        threshold=threshold,
        max_gameable=max_gameable,
        sentinel=sentinel,
    )
