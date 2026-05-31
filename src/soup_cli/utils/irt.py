"""v0.65.0 Part E — IRT eval-cost optimizer.

1-parameter (Rasch) Item Response Theory model fit on per-item correctness
signals, plus a subset-selector that keeps high-information items so eval
bills can drop ~5-10x without losing ranking power.

The Rasch model says: P(correct | ability θ, difficulty β) =
σ(θ - β). With a single respondent's correctness across many items, we
can only fit β up to an additive constant — we centre by setting the
mean ability θ̂ to 0, then β̂_i = -log(p̂_i / (1 - p̂_i)). Pure-Python
math; no numpy / scipy needed for the v0.65.0 surface.

Information at θ=0 under Rasch: I(β) = σ(-β) · (1 - σ(-β)) = σ(-β) · σ(β).
Items with β near 0 carry the most information (50/50 questions are most
discriminating); items at the extremes (always right / always wrong) carry
little new information.
"""
from __future__ import annotations

import json
import logging
import math
import os
import stat
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

_LOG = logging.getLogger(__name__)

# Profile -> keep-fraction.
IRT_PROFILES: Mapping[str, float] = MappingProxyType({
    "full": 1.0,
    "small": 0.30,
    "tiny": 0.10,
})

# DoS / sanity caps.
_MAX_ROWS = 1_000_000
_MAX_ID_LEN = 256
_MAX_FILE_BYTES = 256 * 1024 * 1024  # 256 MiB
_EPSILON = 1e-3


def _validate_item_id(value: object, *, field: str = "item_id") -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be str")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain null bytes")
    if not value:
        raise ValueError(f"{field} must not be empty")
    if len(value) > _MAX_ID_LEN:
        raise ValueError(f"{field} too long")
    return value


@dataclass(frozen=True)
class ItemDifficulty:
    """Frozen per-item Rasch fit."""

    item_id: str
    difficulty: float
    info: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "item_id", _validate_item_id(self.item_id))
        for field, value in (
            ("difficulty", self.difficulty),
            ("info", self.info),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field} must be a number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{field} must be finite")
        if self.info < 0:
            raise ValueError("info must be non-negative")


@dataclass(frozen=True)
class IrtSubsetPlan:
    """Subset selection plan: which items to keep + approximate cost cut."""

    size: str
    item_ids: tuple[str, ...]
    total_items: int
    cost_ratio: float

    def __post_init__(self) -> None:
        if self.size not in IRT_PROFILES:
            raise ValueError(f"size must be one of {sorted(IRT_PROFILES)}")
        if not isinstance(self.item_ids, tuple):
            raise ValueError("item_ids must be a tuple")
        if (
            isinstance(self.total_items, bool)
            or not isinstance(self.total_items, int)
            or self.total_items < 0
        ):
            raise ValueError("total_items must be non-negative int")
        if len(self.item_ids) > self.total_items:
            raise ValueError("item_ids cannot exceed total_items")
        if isinstance(self.cost_ratio, bool) or not isinstance(
            self.cost_ratio, (int, float)
        ):
            raise ValueError("cost_ratio must be a number")
        if not math.isfinite(float(self.cost_ratio)):
            raise ValueError("cost_ratio must be finite")
        if not 0.0 <= self.cost_ratio <= 1.0:
            raise ValueError("cost_ratio must be in [0.0, 1.0]")

    def to_dict(self) -> dict:
        return {
            "size": self.size,
            "item_ids": list(self.item_ids),
            "total_items": self.total_items,
            "cost_ratio": self.cost_ratio,
        }


def _sigmoid(x: float) -> float:
    """Numerically-safe sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def fit_difficulty(rows: Sequence[Mapping[str, object]]) -> tuple[ItemDifficulty, ...]:
    """Fit per-item Rasch difficulty from a flat list of {item_id, correct}.

    The closed-form estimate when ability is centred at 0 is
    ``β̂_i = -log(p̂_i / (1 - p̂_i))`` where ``p̂_i`` is the empirical
    correct-rate for item ``i`` clipped to ``[ε, 1-ε]`` to keep the logit
    finite.
    """
    if not isinstance(rows, (list, tuple)):
        raise TypeError("rows must be a list/tuple of mappings")
    if not rows:
        raise ValueError("rows must not be empty")
    if len(rows) > _MAX_ROWS:
        raise ValueError(f"too many rows (cap {_MAX_ROWS})")

    correct_counts: dict[str, int] = {}
    total_counts: dict[str, int] = {}
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"rows[{idx}] must be a dict")
        if "item_id" not in row:
            raise ValueError(f"rows[{idx}] missing item_id")
        if "correct" not in row:
            raise ValueError(f"rows[{idx}] missing correct field")
        item_id = _validate_item_id(row["item_id"])
        correct = row["correct"]
        if not isinstance(correct, bool):
            raise ValueError(f"rows[{idx}].correct must be bool")
        correct_counts[item_id] = correct_counts.get(item_id, 0) + (1 if correct else 0)
        total_counts[item_id] = total_counts.get(item_id, 0) + 1

    results: list[ItemDifficulty] = []
    for item_id, total in total_counts.items():
        c = correct_counts.get(item_id, 0)
        p_hat = c / total
        p_clipped = max(_EPSILON, min(1.0 - _EPSILON, p_hat))
        # β̂ = -logit(p̂). High p (easy) -> negative β; low p (hard) -> positive.
        beta = -math.log(p_clipped / (1.0 - p_clipped))
        # Rasch info at θ=0: σ(-β) · σ(β) = p̂ · (1-p̂).
        info = p_clipped * (1.0 - p_clipped)
        results.append(ItemDifficulty(
            item_id=item_id, difficulty=beta, info=info,
        ))
    # Deterministic order: sort by item_id.
    results.sort(key=lambda d: d.item_id)
    return tuple(results)


def pick_irt_subset(
    difficulty: Sequence[ItemDifficulty],
    *,
    size: str,
) -> IrtSubsetPlan:
    """Select the high-information subset of items per profile."""
    if not isinstance(difficulty, tuple):
        raise TypeError("difficulty must be a tuple of ItemDifficulty")
    if not difficulty:
        raise ValueError("difficulty must not be empty")
    if size not in IRT_PROFILES:
        raise ValueError(f"size must be one of {sorted(IRT_PROFILES)}")
    keep_fraction = IRT_PROFILES[size]
    total = len(difficulty)
    keep_n = max(1, int(round(total * keep_fraction)))
    # Sort by info descending; tie-break by item_id for determinism.
    ranked = sorted(
        difficulty,
        key=lambda d: (-d.info, d.item_id),
    )
    selected = ranked[:keep_n]
    item_ids = tuple(d.item_id for d in sorted(selected, key=lambda d: d.item_id))
    cost_ratio = keep_n / total if total else 1.0
    return IrtSubsetPlan(
        size=size, item_ids=item_ids,
        total_items=total, cost_ratio=cost_ratio,
    )


def load_response_rows(path: object) -> tuple[dict, ...]:
    """Load per-prompt response rows from a JSONL file under cwd.

    Each row should carry at least ``{item_id, correct}``. Malformed rows
    are silently skipped (matches v0.55.0 / v0.56.0 / v0.61.0 evidence
    loader policy). Uses the shared
    :func:`enforce_under_cwd_and_no_symlink` helper (review H3 fix —
    matches v0.55+ TOCTOU policy used by every release surface), then
    opens with ``O_NOFOLLOW`` (POSIX) and uses ``os.fstat`` on the SAME
    descriptor for size enforcement (review H-NEW-1 fix — double-lstat
    is a TOCTOU race). Streams line-by-line so a 256 MiB file does not
    materialise as a single string. Total iteration is bounded by
    ``_MAX_ROWS`` (review M-NEW-2 — also counts skipped lines toward
    the cap so a 1M-line malformed file does not run unbounded).
    Skipped-row count is logged at WARNING when non-zero.
    """
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    if not isinstance(path, str):
        raise TypeError("path must be str")
    enforce_under_cwd_and_no_symlink(path, "responses_path")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        raise
    except OSError as exc:
        if isinstance(exc, FileNotFoundError):
            raise
        raise ValueError(f"cannot open path: {type(exc).__name__}") from exc
    rows: list[dict] = []
    skipped = 0
    try:
        st = os.fstat(fd)
        if stat.S_ISLNK(st.st_mode):  # impossible under O_NOFOLLOW
            raise ValueError("path must not be a symlink")
        if st.st_size > _MAX_FILE_BYTES:
            raise ValueError(
                f"responses file too large ({st.st_size} > {_MAX_FILE_BYTES})"
            )
        with os.fdopen(fd, "r", encoding="utf-8", closefd=True) as fh:
            fd = -1  # ownership transferred to fdopen
            seen = 0
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                seen += 1
                # Bound TOTAL iteration (kept + skipped) so a 1M-line
                # malformed file can't run unbounded — review M-NEW-2.
                if seen > _MAX_ROWS:
                    raise ValueError(f"too many rows (cap {_MAX_ROWS})")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                if isinstance(row, dict):
                    rows.append(row)
                else:
                    skipped += 1
    finally:
        if fd != -1:
            try:
                os.close(fd)
            except OSError:
                pass
    if skipped:
        _LOG.warning("load_response_rows: skipped %d malformed rows", skipped)
    return tuple(rows)
