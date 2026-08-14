"""v0.71.36 — SemDeDup selection (Data Moat II).

PURE: takes an already-embedded ``(n, d)`` array of L2-normalized row
vectors and decides which rows to keep. No torch, no I/O — so the whole
decision is testable on CPU with hand-built vectors.

Greedy: walk rows in order; drop row ``i`` when its cosine against any
ALREADY-KEPT row is >= threshold. Vectors are L2-normalized upstream
(:func:`soup_cli.utils.embed.embed_texts`), so cosine is a plain dot
product.

Why this and not MinHash: MinHash near-dup detection is *lexical* — two
paraphrases with disjoint vocabulary score ~0 and both survive. Embedding
cosine is semantic, so it can catch them. Whether it actually beats MinHash
on real data is a measured claim, not an assumed one — see the release's
dedup gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # static types only — numpy stays a lazy runtime import
    from numpy.typing import NDArray

# O(n^2) in the worst case — refuse loudly rather than silently subsample.
_MAX_SEMDEDUP_ROWS = 50_000


@dataclass(frozen=True)
class DedupReport:
    """Which rows survived, which collided, and with what.

    ``pairs`` holds ``(dropped_idx, kept_idx, cosine)`` so a drop can be
    audited — it names the nearest kept row that caused it, which is what
    makes a MinHash-vs-semantic comparison checkable rather than a vibe.
    """

    kept: tuple[int, ...]
    dropped: tuple[int, ...]
    pairs: tuple[tuple[int, int, float], ...]
    threshold: float


def _require_threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"threshold must be a float, got {type(value).__name__}"
        )
    num = float(value)
    if not math.isfinite(num):
        raise ValueError("threshold must be finite")
    if not (0.0 <= num <= 1.0):
        raise ValueError(f"threshold must be in [0.0, 1.0], got {num}")
    return num


def greedy_semdedup(
    vectors: "Any | NDArray[Any]", *, threshold: float
) -> DedupReport:
    """Greedy cosine near-duplicate removal over L2-normalized ``vectors``.

    Rows are visited in order. The first row of any near-duplicate cluster
    is kept and the rest are dropped, so the output preserves input order
    and is deterministic.
    """
    import numpy as np

    thresh = _require_threshold(threshold)
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"vectors must be 2-D (n, d), got shape {arr.shape}")
    n_rows = arr.shape[0]
    if n_rows == 0:
        return DedupReport(kept=(), dropped=(), pairs=(), threshold=thresh)
    if n_rows > _MAX_SEMDEDUP_ROWS:
        raise ValueError(
            f"too many rows ({n_rows}); semantic dedup is O(n^2) and the cap "
            f"is {_MAX_SEMDEDUP_ROWS}. Sample first (`soup data sample`) — "
            "Soup refuses rather than silently subsampling."
        )

    kept: list[int] = []
    dropped: list[int] = []
    pairs: list[tuple[int, int, float]] = []
    for idx in range(n_rows):
        if not kept:
            kept.append(idx)
            continue
        sims = arr[kept] @ arr[idx]
        best = int(np.argmax(sims))
        best_sim = float(sims[best])
        if best_sim >= thresh:
            dropped.append(idx)
            pairs.append((idx, kept[best], best_sim))
        else:
            kept.append(idx)
    return DedupReport(
        kept=tuple(kept),
        dropped=tuple(dropped),
        pairs=tuple(pairs),
        threshold=thresh,
    )
