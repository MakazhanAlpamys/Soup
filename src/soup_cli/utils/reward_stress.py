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

# Mirrors ``reward_hack_control._DEFAULT_SENTINEL``, deliberately NOT imported
# from it. That module resolves its ``TrainerCallback`` base at module scope, so
# importing it costs ~4.4s of transformers+torch — and this module sits on the
# light CLI path via ``soup reward stress``. Importing a heavy module for one
# string constant made `soup --help` 7x slower (v0.71.41 regression).
# The two values are pinned equal by a test; if that test ever goes red, the
# constant moved and this copy must follow.
DEFAULT_SENTINEL = "GOLD"
_DEFAULT_SENTINEL = DEFAULT_SENTINEL

CLASSIC_ATTACKS: tuple[str, ...] = ("empty", "length", "repetition", "sentinel")
STRUCTURE_ATTACKS: tuple[str, ...] = (
    "wrapped_junk",
    "answer_spray",
    "structure_without_content",
)
ATTACKS: tuple[str, ...] = CLASSIC_ATTACKS + STRUCTURE_ATTACKS

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
def generate_attack_variants(
    kind: str,
    *,
    sentinel: str = _DEFAULT_SENTINEL,
    gold: Optional[str] = None,
) -> list[str]:
    """Return multiple distinct completion strings for a given attack kind."""
    if kind == "empty":
        return ["   ", "\n\n", ""]
    if kind == "length":
        return [
            " ".join(["padding"] * _LENGTH_ATTACK_WORDS),
            " ".join(["ramble"] * 50),
            " ".join(["filler"] * 80),
        ]
    if kind == "repetition":
        return [
            " ".join(["loop"] * _REPETITION_COUNT),
            " ".join(["repeat"] * 45),
            " ".join(["again"] * 50),
        ]
    if kind == "sentinel":
        return [
            " ".join([sentinel] * _SENTINEL_COUNT),
            " ".join([sentinel] * 10),
            f"start {sentinel} middle {sentinel} end",
        ]
    if kind == "wrapped_junk":
        return [
            "<think>padding</think>padding",
            "<think></think>a",
            "<think>loop loop</think>loop",
            "<think>.</think>.",
            "#### padding",
            r"\boxed{padding}",
            '{"response": "padding", "answer": "junk"}',
        ]
    if kind == "answer_spray":
        target = "42" if gold is None else str(gold).strip()
        return [
            f"<think>options</think> 1 2 3 17 {target} 256 999",
            f"The candidate options considered were {target}, 987, 654, and 321.",
            f"Analysis indicates {target} is an option, while alternatives include 888 and 777.",
            f"Could be {target} or perhaps 111, 222, and 333.",
        ]
    if kind == "structure_without_content":
        return [
            '{"name": "", "arguments": {}, "result": null}',
            '{"type": "object", "properties": {}, "status": "pending"}',
            '{"tool": "none", "parameters": {"key": "placeholder"}}',
            '{"thought": "", "action": "none", "input": ""}',
        ]
    raise ValueError(f"unknown attack kind: {kind!r} (options: {', '.join(ATTACKS)})")


def _attack_text(kind: str, sentinel: str = _DEFAULT_SENTINEL) -> str:
    variants = generate_attack_variants(kind, sentinel=sentinel)
    return variants[-1] if variants else ""


def generate_attacks(
    *, sentinel: str = _DEFAULT_SENTINEL, kinds: Sequence[str] = ATTACKS
) -> list[tuple[str, str]]:
    """Deterministic ``(kind, completion_text)`` junk a correct verifier must reject."""
    if isinstance(kinds, (str, bytes)):
        # A bare string is a Sequence of characters — iterating it would probe
        # per-letter. Force an explicit collection (mirrors reward_synth guards).
        raise TypeError("kinds must be a sequence of attack-kind strings, not a str")
    attacks: list[tuple[str, str]] = []
    for kind in kinds:
        for text in generate_attack_variants(kind, sentinel=sentinel):
            attacks.append((kind, text))
    return attacks


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

    Each attack family generates distinct variant attempts (classic strings,
    wrapped scaffolds, answer-spray distractors, and empty structures).
    When gold references are supplied, junk variants are scored against real
    golds and ``answer_spray`` embeds the actual gold targets.

    ``gameability`` is the overall junk accept-rate; ``gameable`` iff it
    strictly exceeds ``max_gameable``.

    ``reference_accept`` (the golds scored as their own correct completions) is
    reported for context — a verifier that rejects everything is broken, a
    different problem — but does NOT set the verdict.

    No-gold fallback: with an empty ``golds`` each attack family's variants
    are scored once with no ``answer`` kwarg and ``reference_accept`` is ``None``.
    """
    if isinstance(attacks, (str, bytes)):
        raise TypeError("attacks must be a sequence of attack-kind strings, not a str")

    sampled = list(golds)[:_MAX_STRESS_GOLDS]
    have_golds = bool(sampled)

    def _accepted(scores: Sequence[float]) -> int:
        # Scores are already validated finite + numeric by _score_batch.
        return sum(1 for s in scores if s >= threshold)

    reference_accept: Optional[float] = None
    if have_golds:
        ref_scores = _score_batch(reward_fn, sampled, sampled)
        reference_accept = _accepted(ref_scores) / len(sampled)

    unique_kinds = list(dict.fromkeys(attacks))
    results: list[AttackResult] = []
    total_accepted = total_n = 0

    for kind in unique_kinds:
        kind_accepted = 0
        kind_n = 0

        if have_golds:
            if kind == "answer_spray":
                # Multiple templates; each template is tested across all sampled golds
                # in batches of len(sampled) so batch size matches golds cap.
                num_templates = len(
                    generate_attack_variants("answer_spray", sentinel=sentinel, gold="42")
                )
                for idx in range(num_templates):
                    texts = [
                        generate_attack_variants(
                            "answer_spray", sentinel=sentinel, gold=g
                        )[idx]
                        for g in sampled
                    ]
                    scores = _score_batch(reward_fn, texts, sampled)
                    kind_accepted += _accepted(scores)
                    kind_n += len(texts)
            else:
                variants = generate_attack_variants(kind, sentinel=sentinel, gold=None)
                for v in variants:
                    texts = [v] * len(sampled)
                    scores = _score_batch(reward_fn, texts, sampled)
                    kind_accepted += _accepted(scores)
                    kind_n += len(texts)
        else:
            texts = generate_attack_variants(kind, sentinel=sentinel, gold=None)
            scores = _score_batch(reward_fn, texts, None)
            kind_accepted = _accepted(scores)
            kind_n = len(texts)

        results.append(
            AttackResult(
                kind=kind,
                n=kind_n,
                accepted=kind_accepted,
                accept_rate=kind_accepted / kind_n if kind_n else 0.0,
            )
        )
        total_accepted += kind_accepted
        total_n += kind_n

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
