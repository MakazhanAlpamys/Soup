"""Iterative DPO loop driver — v0.70.0 Part E.

Sample → RM-score → re-pair → retrain over N rounds. Frozen plan +
per-round artifact tracking; the actual round orchestrator (which
would invoke ``soup train --task dpo`` between rounds) is deferred to
v0.70.1 (mirrors v0.68.0 local-rl nightly-train policy).

The plan models each round explicitly so the v0.70.1 runner can:
- skip rounds whose ``adapter_path`` already exists (resume),
- re-render pairs JSONL deterministically per round,
- track per-round pairs_count for the `runs replay` integration.

Security:
- Frozen dataclasses with per-field validation (matches v0.67.0
  CmaesPlan / v0.68.0 CompilePlan policy).
- Bool / null-byte / oversize / non-int rejection on every input.
- ``rounds`` tuple required (List would not be immutable under
  ``frozen=True``; matches v0.43 Part B / v0.61 Part E policy).
- Consecutive ``round_index`` invariant: rounds must be 0..N-1 with no
  gaps (defends against caller-side bugs that would silently skip
  rounds).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

_MIN_ROUNDS = 1
_MAX_ROUNDS = 100
_MIN_PAIRS_PER_ROUND = 10
_MAX_PAIRS_PER_ROUND = 1_000_000
_MAX_PATH_LEN = 4096


def validate_rounds(value: object) -> int:
    """Validate the ``rounds`` count: int in [1, 100], bool rejected."""
    if isinstance(value, bool):
        raise ValueError("rounds must not be bool")
    if not isinstance(value, int):
        raise ValueError(f"rounds must be int, got {type(value).__name__}")
    if value < _MIN_ROUNDS:
        raise ValueError(f"rounds must be >= {_MIN_ROUNDS}, got {value}")
    if value > _MAX_ROUNDS:
        raise ValueError(
            f"rounds={value} exceeds {_MAX_ROUNDS} cap"
        )
    return value


def validate_pairs_per_round(value: object) -> int:
    """Validate the per-round pair count.

    Range: ``[10, 1_000_000]``. Below 10 pairs the DPO gradient signal
    is too noisy to be meaningful; above 1M is a clear OOM / disk hazard.
    """
    if isinstance(value, bool):
        raise ValueError("pairs_per_round must not be bool")
    if not isinstance(value, int):
        raise ValueError(
            f"pairs_per_round must be int, got {type(value).__name__}"
        )
    if value < _MIN_PAIRS_PER_ROUND:
        raise ValueError(
            f"pairs_per_round must be >= {_MIN_PAIRS_PER_ROUND}, got {value}"
        )
    if value > _MAX_PAIRS_PER_ROUND:
        raise ValueError(
            f"pairs_per_round={value} exceeds {_MAX_PAIRS_PER_ROUND} cap"
        )
    return value


def _check_path(value: object, field: str) -> str:
    if isinstance(value, bool):
        raise ValueError(f"{field} must not be bool")
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{field} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain null bytes")
    if len(value) > _MAX_PATH_LEN:
        raise ValueError(f"{field} exceeds {_MAX_PATH_LEN} chars")
    return value


def _check_non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must not be bool")
    if not isinstance(value, int):
        raise TypeError(f"{field} must be int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{field} must be non-negative, got {value}")
    return value


@dataclass(frozen=True)
class IterativeDPORound:
    """Frozen per-round descriptor.

    - ``round_index``: 0-based, non-negative int (bool rejected).
    - ``prompts_path``: source prompts JSONL (sampled into pairs).
    - ``pairs_path``: where the round's chosen/rejected JSONL gets written.
    - ``adapter_path``: where the round's DPO adapter lands.
    - ``pairs_count``: number of pairs the round produced. Non-negative
      int (0 allowed for plan-only rendering).
    """

    round_index: int
    prompts_path: str
    pairs_path: str
    adapter_path: str
    pairs_count: int

    def __post_init__(self) -> None:
        _check_non_negative_int(self.round_index, "round_index")
        _check_path(self.prompts_path, "prompts_path")
        _check_path(self.pairs_path, "pairs_path")
        _check_path(self.adapter_path, "adapter_path")
        _check_non_negative_int(self.pairs_count, "pairs_count")


@dataclass(frozen=True)
class IterativeDPOPlan:
    """Frozen iterative-DPO plan.

    - ``base_model``: HF id / local path. Shape-validated.
    - ``reward_model``: HF id / local path of the RM used for scoring.
    - ``rounds``: tuple of :class:`IterativeDPORound`; must be
      consecutive (0..N-1 with no gaps).
    """

    base_model: str
    reward_model: str
    rounds: Tuple[IterativeDPORound, ...]

    def __post_init__(self) -> None:
        _check_path(self.base_model, "base_model")
        _check_path(self.reward_model, "reward_model")
        if not isinstance(self.rounds, tuple):
            raise TypeError(
                f"rounds must be a tuple, got {type(self.rounds).__name__}"
            )
        if len(self.rounds) < 1:
            raise ValueError("rounds must contain at least 1 round")
        for r in self.rounds:
            if not isinstance(r, IterativeDPORound):
                raise TypeError(
                    f"every rounds[] entry must be IterativeDPORound, "
                    f"got {type(r).__name__}"
                )
        for idx, r in enumerate(self.rounds):
            if r.round_index != idx:
                raise ValueError(
                    f"rounds must have consecutive round_index 0..N-1; "
                    f"rounds[{idx}].round_index={r.round_index}"
                )


def build_iterative_dpo_plan(
    *,
    base_model: str,
    reward_model: str,
    prompts_path: str,
    output_dir: str,
    rounds: int,
    pairs_per_round: int,
) -> IterativeDPOPlan:
    """Build a canonical :class:`IterativeDPOPlan` from operator inputs.

    Per-round paths follow the pattern
    ``<output_dir>/round-<NN>/{pairs.jsonl,adapter}``. The plan is
    cheap to construct; the v0.70.1 runner consumes it.
    """
    validate_rounds(rounds)
    validate_pairs_per_round(pairs_per_round)
    _check_path(base_model, "base_model")
    _check_path(reward_model, "reward_model")
    _check_path(prompts_path, "prompts_path")
    _check_path(output_dir, "output_dir")
    output_dir = output_dir.rstrip("/\\")
    per_round = []
    for i in range(rounds):
        per_round.append(
            IterativeDPORound(
                round_index=i,
                prompts_path=prompts_path,
                pairs_path=f"{output_dir}/round-{i:02d}/pairs.jsonl",
                adapter_path=f"{output_dir}/round-{i:02d}/adapter",
                pairs_count=pairs_per_round,
            )
        )
    return IterativeDPOPlan(
        base_model=base_model,
        reward_model=reward_model,
        rounds=tuple(per_round),
    )


def run_iterative_dpo(plan):
    """Execute the iterative-DPO loop. Deferred to v0.70.1.

    Validates plan type at the public boundary so misconfigured callers
    fail fast (mirrors v0.50.0 / v0.62.0 / v0.67.0 / v0.69.0 deferred-live
    policy).
    """
    if not isinstance(plan, IterativeDPOPlan):
        raise TypeError(
            f"plan must be IterativeDPOPlan, got {type(plan).__name__}"
        )
    raise NotImplementedError(
        "Live iterative-DPO loop runner is deferred to v0.70.1. "
        "v0.70.0 ships the schema + plan-only renderer only."
    )
