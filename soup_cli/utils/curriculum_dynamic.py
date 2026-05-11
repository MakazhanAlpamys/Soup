"""Curriculum-Aware dynamic re-weighting (v0.48.0 Part A — BETA).

Online uncertainty estimation: every N steps, aggregate per-sample loss and
gradient-norm fingerprints into bucket-level weights and surface a recommended
sampler weight per bucket. Up-weight high-uncertainty / under-fit buckets;
down-weight already-mastered ones.

DDP / grad-accum safety: all-reduce of per-sample stats across ranks is the
well-known footgun for dynamic curriculum learning. We document the contract
here and surface a cross-validator that rejects ``curriculum_dynamic=true``
combined with launches that have not declared rank coordination.

This module ships BETA-flagged: the math is pure-Python + numpy-free; the live
HF Trainer callback wiring is deferred to v0.48.1 once external benchmarks have
landed (mirrors the v0.27.0 MII / v0.37.0 multipack / v0.41.0 LLaMA Pro
stub-then-live pattern).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

# Bounds — match project conventions (e.g. v0.32.0 GradAccumMonitor).
_MIN_BUCKETS = 1
_MAX_BUCKETS = 20
_MIN_RECOMPUTE_STEPS = 1
_MAX_RECOMPUTE_STEPS = 100_000
_MAX_BUCKET_SAMPLES = 1_000_000  # DoS cap on stats accumulation
_MAX_HISTORY_ROWS = 100_000  # DoS cap on curriculum-history JSONL parsing
_DEFAULT_FLOOR = 0.05  # min per-bucket weight after normalisation
_DEFAULT_TEMP = 1.0

__all__ = [
    "DynamicCurriculumPolicy",
    "BucketStats",
    "compute_bucket_weights",
    "validate_distributed_curriculum",
]


def _reject_bool_int(name: str, value) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be int, not bool")
    if not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    return value


def _reject_bool_float(name: str, value) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be float, not bool")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be float, got {type(value).__name__}")
    fv = float(value)
    if not math.isfinite(fv):
        raise ValueError(f"{name} must be finite (got {value!r})")
    return fv


@dataclass(frozen=True)
class DynamicCurriculumPolicy:
    """Frozen config for the dynamic re-weighting policy.

    Attributes:
        num_buckets: Number of difficulty buckets (must match
            ``training.curriculum_buckets`` schema field).
        recompute_every_n_steps: Refresh sampler weights every N global steps.
        floor: Minimum normalised per-bucket weight (defends against
            "starve a bucket" pathology). In ``(0, 1/num_buckets]``.
        temperature: Softmax temperature applied to uncertainty signal. Higher
            values flatten the distribution toward uniform; lower values
            concentrate weight on the hardest buckets.
    """

    num_buckets: int
    recompute_every_n_steps: int = 50
    floor: float = _DEFAULT_FLOOR
    temperature: float = _DEFAULT_TEMP

    def __post_init__(self) -> None:
        nb = _reject_bool_int("num_buckets", self.num_buckets)
        if nb < _MIN_BUCKETS or nb > _MAX_BUCKETS:
            raise ValueError(
                f"num_buckets must be in [{_MIN_BUCKETS}, {_MAX_BUCKETS}], got {nb}"
            )
        rs = _reject_bool_int(
            "recompute_every_n_steps", self.recompute_every_n_steps
        )
        if rs < _MIN_RECOMPUTE_STEPS or rs > _MAX_RECOMPUTE_STEPS:
            raise ValueError(
                f"recompute_every_n_steps must be in "
                f"[{_MIN_RECOMPUTE_STEPS}, {_MAX_RECOMPUTE_STEPS}], got {rs}"
            )
        fv = _reject_bool_float("floor", self.floor)
        # floor must leave at least equal-share room; uniform = 1/nb.
        ceiling = 1.0 / nb
        if fv <= 0.0 or fv > ceiling:
            raise ValueError(
                f"floor must be in (0.0, {ceiling}] for num_buckets={nb}, got {fv}"
            )
        tv = _reject_bool_float("temperature", self.temperature)
        if tv <= 0.0:
            raise ValueError(f"temperature must be > 0, got {tv}")

    def should_recompute(self, global_step: int) -> bool:
        """True when the current global step is a recompute boundary."""
        gs = _reject_bool_int("global_step", global_step)
        if gs < 0:
            raise ValueError(f"global_step must be >= 0, got {gs}")
        if gs == 0:
            return False
        return gs % self.recompute_every_n_steps == 0


@dataclass(frozen=True)
class BucketStats:
    """Aggregated per-bucket statistics.

    Attributes:
        bucket_id: 0-indexed bucket position (0 = easiest).
        num_samples: How many samples contributed to mean_loss / mean_grad_norm.
        mean_loss: Average loss across the bucket's recent samples.
        mean_grad_norm: Average parameter-grad-norm fingerprint.
    """

    bucket_id: int
    num_samples: int
    mean_loss: float
    mean_grad_norm: float


def _coerce_stats(raw: Mapping[int, Mapping[str, float]]) -> List[BucketStats]:
    out: List[BucketStats] = []
    for bucket_id, payload in raw.items():
        if isinstance(bucket_id, bool) or not isinstance(bucket_id, int):
            raise TypeError(
                f"bucket id must be int, got {type(bucket_id).__name__}"
            )
        if bucket_id < 0:
            raise ValueError(f"bucket id must be >= 0, got {bucket_id}")
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"bucket payload must be Mapping, got {type(payload).__name__}"
            )
        num_samples = payload.get("num_samples", 0)
        ns = _reject_bool_int("num_samples", num_samples)
        if ns < 0 or ns > _MAX_BUCKET_SAMPLES:
            raise ValueError(
                f"num_samples must be in [0, {_MAX_BUCKET_SAMPLES}], got {ns}"
            )
        ml = _reject_bool_float("mean_loss", payload.get("mean_loss", 0.0))
        mg = _reject_bool_float(
            "mean_grad_norm", payload.get("mean_grad_norm", 0.0)
        )
        if ml < 0.0 or mg < 0.0:
            raise ValueError(
                "mean_loss / mean_grad_norm must be >= 0 "
                f"(got loss={ml}, grad={mg})"
            )
        out.append(BucketStats(bucket_id, ns, ml, mg))
    return out


def _softmax(values: Sequence[float], temperature: float) -> List[float]:
    """Numerically stable softmax."""
    if not values:
        return []
    inv_t = 1.0 / temperature
    scaled = [v * inv_t for v in values]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    total = sum(exps)
    if total <= 0.0 or not math.isfinite(total):
        # Degenerate input → uniform fallback.
        n = len(values)
        return [1.0 / n] * n
    return [e / total for e in exps]


def compute_bucket_weights(
    stats: Mapping[int, Mapping[str, float]],
    policy: DynamicCurriculumPolicy,
) -> Tuple[float, ...]:
    """Return normalised sampler weights per bucket.

    Buckets with no recorded samples fall back to the uniform prior. Buckets
    with higher mean loss + grad norm receive more weight; the floor parameter
    prevents the easiest bucket from ever dropping below ``policy.floor``.

    Args:
        stats: Mapping from ``bucket_id`` to ``{num_samples, mean_loss,
            mean_grad_norm}`` payload.
        policy: A frozen :class:`DynamicCurriculumPolicy`.

    Returns:
        Tuple of ``policy.num_buckets`` floats that sum to 1.0 ± 1e-6.
    """
    if not isinstance(policy, DynamicCurriculumPolicy):
        raise TypeError(
            f"policy must be DynamicCurriculumPolicy, "
            f"got {type(policy).__name__}"
        )
    if not isinstance(stats, Mapping):
        raise TypeError(f"stats must be Mapping, got {type(stats).__name__}")

    coerced = _coerce_stats(stats)
    by_id: Dict[int, BucketStats] = {b.bucket_id: b for b in coerced}

    nb = policy.num_buckets
    # Build per-bucket scalar = mean_loss + mean_grad_norm.
    # Empty buckets get neutral score (median of populated buckets, else 0).
    populated = [
        by_id[i].mean_loss + by_id[i].mean_grad_norm
        for i in range(nb)
        if i in by_id and by_id[i].num_samples > 0
    ]
    if populated:
        # Median is robust to outliers; matches Axolotl curriculum policy.
        srt = sorted(populated)
        mid = len(srt) // 2
        neutral = (
            srt[mid] if len(srt) % 2 == 1 else (srt[mid - 1] + srt[mid]) / 2
        )
    else:
        # No data — uniform fallback.
        return (1.0 / nb,) * nb

    scores: List[float] = []
    for i in range(nb):
        b = by_id.get(i)
        if b is None or b.num_samples == 0:
            scores.append(neutral)
        else:
            scores.append(b.mean_loss + b.mean_grad_norm)

    weights = _softmax(scores, policy.temperature)
    # Water-fill: every bucket gets at least `floor`; remaining
    # (1 - nb*floor) is distributed proportionally to the softmax mass.
    # The softmax already sums to 1.0 so the water-fill output sums to
    # exactly 1.0 (modulo float drift bounded by nb * eps). A subsequent
    # renorm `w / sum(w)` is harmful: it can push elements sitting at
    # `floor` below the floor when the sum is slightly > 1.0. See
    # v0.48.0 Part A code review HIGH #2.
    reserved = policy.floor * nb
    free_mass = 1.0 - reserved
    if free_mass <= 0.0:
        return (1.0 / nb,) * nb
    total = sum(weights)
    if total <= 0.0:
        return (1.0 / nb,) * nb
    return tuple(policy.floor + free_mass * (w / total) for w in weights)


def validate_distributed_curriculum(
    enabled: bool,
    *,
    world_size: int,
    rank_coordinated: bool,
) -> None:
    """Cross-validator for the distributed footgun.

    When ``curriculum_dynamic=true`` and the launch is multi-rank, the caller
    MUST attest that an ``all_reduce`` of per-sample stats is wired (otherwise
    each rank computes a divergent weight and the sampler desynchronises).

    Args:
        enabled: Resolved value of ``training.curriculum_dynamic``.
        world_size: Detected distributed world size (1 for single-process).
        rank_coordinated: Caller confirms the all-reduce hook is registered.

    Raises:
        ValueError: When multi-rank but no coordination is wired.
    """
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be bool")
    if not enabled:
        return
    ws = _reject_bool_int("world_size", world_size)
    if ws < 1:
        raise ValueError(f"world_size must be >= 1, got {ws}")
    if not isinstance(rank_coordinated, bool):
        raise TypeError("rank_coordinated must be bool")
    if ws > 1 and not rank_coordinated:
        raise ValueError(
            f"curriculum_dynamic=true with world_size={ws} requires an "
            "all_reduce hook on per-bucket stats (otherwise each rank "
            "diverges). Register the coordination callback before training."
        )


def render_curve(
    history: Sequence[Mapping[str, float]],
    *,
    num_buckets: int,
    width: int = 60,
) -> str:
    """Render a plain-text time-series of bucket weights over training.

    Each row is one recompute step; each column is one bucket. Output uses
    ASCII glyphs only (matches the v0.24.3 Windows-Unicode policy).

    Args:
        history: Sequence of mappings ``{"step": int, "weights":
            [w0, w1, ...]}`` from :func:`compute_bucket_weights`.
        num_buckets: Expected bucket arity (validates row shape).
        width: Output column width per bucket cell (>= 4).

    Returns:
        Multi-line ASCII table suitable for terminal display.
    """
    nb = _reject_bool_int("num_buckets", num_buckets)
    if nb < 1 or nb > _MAX_BUCKETS:
        raise ValueError(
            f"num_buckets must be in [1, {_MAX_BUCKETS}], got {nb}"
        )
    w = _reject_bool_int("width", width)
    if w < 4 or w > 200:
        raise ValueError(f"width must be in [4, 200], got {w}")
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        raise TypeError("history must be a non-string Sequence")
    if len(history) > _MAX_HISTORY_ROWS:
        raise ValueError(
            f"history has {len(history)} rows; cap is {_MAX_HISTORY_ROWS}"
        )

    if not history:
        return "(no curriculum history recorded yet)"

    header = "step".ljust(8) + "".join(
        f"B{i}".rjust(w) for i in range(nb)
    )
    lines = [header]
    for entry in history:
        if not isinstance(entry, Mapping):
            raise TypeError(
                f"history entry must be Mapping, got {type(entry).__name__}"
            )
        step = _reject_bool_int("step", entry.get("step", 0))
        weights = entry.get("weights", ())
        if not isinstance(weights, Sequence) or isinstance(
            weights, (str, bytes)
        ):
            raise TypeError("weights must be a non-string Sequence")
        if len(weights) != nb:
            raise ValueError(
                f"weights length {len(weights)} != num_buckets {nb} at "
                f"step={step}"
            )
        cells = "".join(f"{float(v):>{w}.4f}" for v in weights)
        lines.append(str(step).ljust(8) + cells)
    return "\n".join(lines)


def parse_history_jsonl(rows: Sequence[Mapping]) -> List[Dict[str, object]]:
    """Validate and normalise a sequence of curriculum-history rows.

    Used by ``soup runs curriculum-curve <run_id>`` to load the JSONL written
    by the dynamic callback.

    Args:
        rows: Sequence of mappings with ``step`` int and ``weights`` list.

    Returns:
        List of normalised dicts with keys ``step`` (int) and
        ``weights`` (tuple of floats summing to 1.0 ± 1e-3).
    """
    out: List[Dict[str, object]] = []
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("rows must be a non-string Sequence")
    if len(rows) > _MAX_HISTORY_ROWS:
        raise ValueError(
            f"history has {len(rows)} rows; cap is {_MAX_HISTORY_ROWS}"
        )
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("history row must be Mapping")
        step = _reject_bool_int("step", row.get("step", 0))
        weights = row.get("weights")
        if not isinstance(weights, Sequence) or isinstance(
            weights, (str, bytes)
        ):
            raise TypeError("weights must be a non-string Sequence")
        floats = []
        for v in weights:
            floats.append(_reject_bool_float("weight", v))
        s = sum(floats)
        if s <= 0 or abs(s - 1.0) > 1e-3:
            raise ValueError(
                f"weights at step={step} must sum to 1.0 ± 1e-3, got {s}"
            )
        out.append({"step": step, "weights": tuple(floats)})
    return out
