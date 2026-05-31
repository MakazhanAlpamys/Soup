"""``soup adapters bisect`` — binary search over training history (v0.67.0 Part F).

Given an ordered history of checkpoints (or dataset commits) and an
operator-supplied predicate that returns True when "eval passes",
binary-search for the FIRST checkpoint where the predicate flips to
False. The result is the regression boundary.

Composes with v0.66 Part B influence-blame: once the boundary is
found, ``soup adapters blame`` can attribute the regression to
specific dataset rows.

Public surface:

- ``BisectPlan`` / ``BisectStep`` / ``BisectResult`` frozen dataclasses
- ``build_bisect_plan(history)`` factory
- ``bisect_next_step(plan, lo, hi)`` pure midpoint kernel
- ``run_bisect(plan, *, eval_fn)`` end-to-end binary search

Closed verdict taxonomy:
- ``ALL_OK``      — every checkpoint passes (no regression found)
- ``BROKEN_AT``   — found first failing checkpoint
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

MAX_HISTORY = 4096


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def _check_checkpoint(value: object, field: str) -> str:
    if isinstance(value, bool):
        raise TypeError(f"{field} must not be bool")
    if not isinstance(value, str):
        raise TypeError(f"{field} entries must be str")
    if not value:
        raise ValueError(f"{field} entry must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{field} entry must not contain null bytes")
    if len(value) > 1024:
        raise ValueError(f"{field} entry too long")
    return value


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BisectPlan:
    """An ordered history of checkpoint identifiers to bisect over.

    The order is "oldest first" — index 0 is the earliest checkpoint
    and the last entry is the most recent.
    """

    history: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.history, tuple):
            raise TypeError("history must be tuple")
        if len(self.history) < 2:
            raise ValueError(
                "history must contain at least 2 entries"
            )
        if len(self.history) > MAX_HISTORY:
            raise ValueError(
                f"history length {len(self.history)} > {MAX_HISTORY}"
            )
        seen: set[str] = set()
        for entry in self.history:
            _check_checkpoint(entry, "history")
            if entry in seen:
                raise ValueError(
                    f"history must be unique (duplicate {entry!r})"
                )
            seen.add(entry)


@dataclass(frozen=True)
class BisectStep:
    """One probe result during a bisect run."""

    checkpoint: str
    ok: bool

    def __post_init__(self) -> None:
        _check_checkpoint(self.checkpoint, "checkpoint")
        if not isinstance(self.ok, bool):
            raise TypeError("ok must be bool")


@dataclass(frozen=True)
class BisectResult:
    """Outcome of a bisect run.

    ``first_broken`` is ``None`` when ``verdict='ALL_OK'``; otherwise
    it is the first failing checkpoint id.
    """

    first_broken: Optional[str]
    verdict: str
    steps: Tuple[BisectStep, ...]
    probes: int

    def __post_init__(self) -> None:
        valid_verdicts = ("ALL_OK", "BROKEN_AT")
        if self.verdict not in valid_verdicts:
            raise ValueError(
                f"verdict must be one of {valid_verdicts}"
            )
        if self.first_broken is not None:
            _check_checkpoint(self.first_broken, "first_broken")
        if not isinstance(self.steps, tuple):
            raise TypeError("steps must be tuple")
        for s in self.steps:
            if not isinstance(s, BisectStep):
                raise TypeError("steps entries must be BisectStep")
        if isinstance(self.probes, bool) or not isinstance(self.probes, int):
            raise TypeError("probes must be int")
        if self.probes < 0:
            raise ValueError("probes must be non-negative")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_bisect_plan(history: Sequence[str]) -> BisectPlan:
    """Build a frozen plan from a sequence of checkpoint ids."""
    if isinstance(history, str) or not isinstance(history, Sequence):
        raise TypeError("history must be a non-string sequence")
    return BisectPlan(history=tuple(history))


# ---------------------------------------------------------------------------
# Pure midpoint kernel
# ---------------------------------------------------------------------------


def bisect_next_step(plan: BisectPlan, *, lo: int, hi: int) -> int:
    """Return the midpoint index between lo and hi (inclusive).

    Pure: deterministic given the same inputs. Used internally by
    ``run_bisect`` but exposed for callers that want to drive the
    bisect loop themselves.
    """
    if not isinstance(plan, BisectPlan):
        raise TypeError("plan must be BisectPlan")
    if isinstance(lo, bool) or not isinstance(lo, int):
        raise TypeError("lo must be int")
    if isinstance(hi, bool) or not isinstance(hi, int):
        raise TypeError("hi must be int")
    if lo < 0 or hi >= len(plan.history):
        raise ValueError("lo/hi out of bounds")
    if lo > hi:
        raise ValueError("lo must be <= hi")
    return (lo + hi) // 2


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_bisect(
    plan: BisectPlan,
    *,
    eval_fn: Callable[[str], bool],
) -> BisectResult:
    """Find the first failing checkpoint via binary search.

    ``eval_fn(checkpoint_id)`` MUST return ``True`` when the checkpoint
    passes (no regression) and ``False`` when it fails. The bisect
    assumes monotonic regression: once a checkpoint fails, every later
    one also fails. Non-monotonic histories produce undefined results;
    operators wanting a sweep should use ``soup eval`` directly.
    """
    if not isinstance(plan, BisectPlan):
        raise TypeError("plan must be BisectPlan")
    if eval_fn is None or not callable(eval_fn):
        raise TypeError("eval_fn must be callable")

    n = len(plan.history)
    steps: list[BisectStep] = []

    # Probe both endpoints first to short-circuit "all OK" / "all broken".
    first = plan.history[0]
    first_ok = bool(eval_fn(first))
    steps.append(BisectStep(checkpoint=first, ok=first_ok))
    if not first_ok:
        # All broken from the very first checkpoint.
        return BisectResult(
            first_broken=first,
            verdict="BROKEN_AT",
            steps=tuple(steps),
            probes=len(steps),
        )

    last = plan.history[-1]
    last_ok = bool(eval_fn(last))
    steps.append(BisectStep(checkpoint=last, ok=last_ok))
    if last_ok:
        return BisectResult(
            first_broken=None,
            verdict="ALL_OK",
            steps=tuple(steps),
            probes=len(steps),
        )

    # Now standard binary search: find the LOWEST index that fails.
    # Invariant: plan.history[lo] passes; plan.history[hi] fails.
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        ckpt = plan.history[mid]
        ok = bool(eval_fn(ckpt))
        steps.append(BisectStep(checkpoint=ckpt, ok=ok))
        if ok:
            lo = mid
        else:
            hi = mid

    return BisectResult(
        first_broken=plan.history[hi],
        verdict="BROKEN_AT",
        steps=tuple(steps),
        probes=len(steps),
    )
