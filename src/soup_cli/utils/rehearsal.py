"""v0.71.36 — continual-learning rehearsal mix (Data Moat II).

PURE: given the new task's rows and an old task's rows, sample a slice of
the old and INTERLEAVE it into the new so fine-tuning on the new task does
not erase the previous one. No torch, no I/O.

Named ``rehearsal`` rather than ``replay`` because ``utils/replay.py`` is
already taken by v0.34.0's metric-history replay (``soup runs replay``) —
an unrelated feature. The user-facing flag is still ``--replay``; only the
module name differs.

Ratio semantics (locked): ``r`` is the fraction of the FINAL mixed set
that is replay, so ``n_replay = round(r/(1-r) * n_new)``. At r=0.1 over
1000 new rows: 111 replay -> 1111 total -> 10.0%. The naive ``r * n_new``
yields 9.09% and is wrong.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class ReplayReport:
    """What the mixer actually did — surfaced to the console + provenance."""

    n_new: int
    n_replay: int
    n_final: int
    requested: int
    shortfall: int
    ratio_actual: float


def _require_ratio(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"ratio must be a float, got {type(value).__name__}")
    num = float(value)
    if not math.isfinite(num):
        raise ValueError("ratio must be finite")
    if not (0.0 < num <= 0.5):
        raise ValueError(f"ratio must be in (0.0, 0.5], got {num}")
    return num


def _require_n_new(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"n_new must be int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"n_new must be >= 0, got {value}")
    return value


def resolve_replay_count(n_new: int, ratio: float) -> int:
    """How many replay rows to add so replay is ``ratio`` of the FINAL set.

    Solving ``n_replay / (n_new + n_replay) = r`` gives
    ``n_replay = r/(1-r) * n_new`` — NOT ``r * n_new``, which would leave
    replay at only ``r/(1+r)`` of the result (9.09% for a requested 10%).
    """
    rate = _require_ratio(ratio)
    rows = _require_n_new(n_new)
    if rows == 0:
        return 0
    return int(round(rate / (1.0 - rate) * rows))


def mix_replay(
    new_rows: Sequence[dict],
    replay_rows: Sequence[dict],
    *,
    ratio: float,
    seed: Optional[int] = None,
) -> tuple[list[dict], ReplayReport]:
    """Interleave a seeded sample of ``replay_rows`` into ``new_rows``.

    Interleaved, NOT appended. ``new + replay`` would put every replay row
    in one contiguous block at the end, so the model sees them all in the
    final steps — a second mini-finetune rather than rehearsal, which is
    the very failure replay exists to prevent.

    An undersized replay pool uses every row it has and reports the
    shortfall; it never upsamples, because repeating rows silently changes
    epoch semantics (a row seen twice per epoch is not the same experiment).

    ``seed=None`` means seed 0, not "random": a training run must reproduce.
    """
    new_list = list(new_rows)
    pool = list(replay_rows)
    requested = resolve_replay_count(len(new_list), ratio)
    take = min(requested, len(pool))

    if take == 0:
        # Nothing to add: return the new rows untouched. Shuffling here
        # would silently reorder training data as a side effect of pointing
        # --replay at an empty pool.
        return new_list, ReplayReport(
            n_new=len(new_list),
            n_replay=0,
            n_final=len(new_list),
            requested=requested,
            shortfall=requested,
            ratio_actual=0.0,
        )

    rng = random.Random(seed if seed is not None else 0)
    sampled = rng.sample(pool, take)
    mixed = new_list + sampled
    rng.shuffle(mixed)

    n_final = len(mixed)
    return mixed, ReplayReport(
        n_new=len(new_list),
        n_replay=take,
        n_final=n_final,
        requested=requested,
        shortfall=requested - take,
        ratio_actual=(take / n_final) if n_final else 0.0,
    )
