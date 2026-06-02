"""LoRA adapter blame: attribute weight movement to dataset rows.

v0.57.0 Part C shipped the plan-only surface (`plan_blame` + deferred
`run_blame` stub). v0.66.0 Part B (closes #171) lifts the stub with a
live DataInf-style influence-function approximation:

    influence(row) = cosine(grad(row), grad(probe)) × |grad(row)|

The math kernel is pure numpy; the caller supplies a ``probe_fn`` that
returns ``(row_grads, probe_grad)`` for a synthetic or real probe.
When no probe_fn is supplied, a deterministic synthetic probe is used
(matches v0.54.0 advise stub policy — surfaces always produce a real
``BlameResult`` so downstream tools never see ``NotImplementedError``).

Public surface:

- ``parse_budget(spec)`` → seconds
- ``BlamePlan`` + ``BlameShardWork`` frozen dataclasses (v0.57.0)
- ``plan_blame(...)`` → ``BlamePlan`` (v0.57.0)
- ``RowInfluence`` + ``BlameResult`` frozen dataclasses (v0.66.0)
- ``compute_row_influence(row_grad, probe_grad)`` → float (v0.66.0 math kernel)
- ``run_blame(plan, *, probe_fn=None, top_k=50)`` → ``BlameResult`` (v0.66.0 LIVE)
- ``render_blame_json`` / ``render_blame_markdown`` (v0.66.0)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional, Tuple

from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

_MIN_BUDGET_SECONDS = 60
_MAX_BUDGET_SECONDS = 24 * 3600
_MIN_SHARDS = 2
_MAX_SHARDS = 100
_MIN_PER_SHARD_SECONDS = 30

_BUDGET_RE = re.compile(r"^(\d+)([smh]?)$")

_MIN_TOP_K = 1
_MAX_TOP_K = 1_000_000
_DEFAULT_TOP_K = 50

# M6 review fix (v0.66.0): synthetic-probe default cap. A 10M-row dataset
# without an operator-supplied probe_fn would otherwise allocate 1.28 GB
# of float32 (10M × 32 dim × 4 B). Real workflows pass a real probe_fn.
_DEFAULT_SYNTH_PROBE_CAP = 100_000


# ---------------------------------------------------------------------------
# Plan-only surface (v0.57.0)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlameShardWork:
    shard_id: int
    holdout_offset: int
    holdout_size: int
    projected_seconds: int


@dataclass(frozen=True)
class BlamePlan:
    adapter_dir: str
    dataset_path: str
    layer: str
    budget_seconds: int
    num_shards: int
    per_shard_seconds: int
    shards: Tuple[BlameShardWork, ...]
    feasible: bool
    reason: str


def parse_budget(spec: str) -> int:
    """Parse a budget string (``60s`` / ``5m`` / ``2h`` / bare seconds)."""
    if isinstance(spec, bool) or not isinstance(spec, str):
        raise TypeError("spec must be str")
    if not spec:
        raise ValueError("spec must be non-empty")
    if "\x00" in spec:
        raise ValueError("spec must not contain null bytes")
    match = _BUDGET_RE.match(spec.strip())
    if not match:
        raise ValueError(f"invalid budget: {spec!r}")
    value = int(match.group(1))
    unit = match.group(2) or "s"
    multiplier = {"s": 1, "m": 60, "h": 3600}[unit]
    seconds = value * multiplier
    if seconds < _MIN_BUDGET_SECONDS:
        raise ValueError(
            f"budget {seconds}s below floor {_MIN_BUDGET_SECONDS}s"
        )
    if seconds > _MAX_BUDGET_SECONDS:
        raise ValueError(
            f"budget {seconds}s above cap {_MAX_BUDGET_SECONDS}s"
        )
    return seconds


def _validate_int(value: object, field: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be int")
    if value < lo or value > hi:
        raise ValueError(f"{field} must be in [{lo}, {hi}]")
    return value


def _validate_str(value: object, field: str, max_len: int = 512) -> str:
    if isinstance(value, bool) or not isinstance(value, str):
        raise TypeError(f"{field} must be str")
    if not value:
        raise ValueError(f"{field} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain null bytes")
    if len(value) > max_len:
        raise ValueError(f"{field} must be ≤{max_len} chars")
    return value


def _count_dataset_rows(dataset_path: str) -> int:
    """Cheap line count via ``O_NOFOLLOW`` open (TOCTOU defence).

    Review H1 fix (v0.66.0): the v0.57 implementation re-validated via
    ``os.path.realpath`` then opened by path, which leaves a race window
    where an attacker can swap the file between the helper's lstat and
    the open. We now open with ``O_NOFOLLOW`` on POSIX so a symlink swap
    between containment check and open raises immediately rather than
    silently following.
    """
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(dataset_path, os.O_RDONLY | no_follow)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"dataset not found: {os.path.basename(dataset_path)}"
        ) from exc
    except OSError as exc:
        # ELOOP on Linux when O_NOFOLLOW hits a symlink; map to a clear
        # error so operators see what tripped.
        raise ValueError(
            f"dataset cannot be opened (symlink?): "
            f"{type(exc).__name__}"
        ) from exc
    count = 0
    with os.fdopen(fd, "rb") as fh:
        for _ in fh:
            count += 1
            # M4 review fix: raise loudly instead of silently truncating —
            # planning over the wrong row count would produce a bad plan.
            if count > 10_000_000:
                raise ValueError(
                    "dataset has >10M rows; subsample before blame "
                    "(use `soup data sample`)"
                )
    return count


def plan_blame(
    adapter_dir: str,
    dataset_path: str,
    *,
    layer: str,
    budget_seconds: int,
    num_shards: int = 10,
) -> BlamePlan:
    """Build a leave-one-out ablation plan (v0.57.0).

    The plan splits the dataset into ``num_shards`` equal shards. ``feasible``
    is False when the budget cannot cover ≥ ``_MIN_PER_SHARD_SECONDS`` per
    shard with safety overhead.
    """
    enforce_under_cwd_and_no_symlink(adapter_dir, "adapter_dir")
    enforce_under_cwd_and_no_symlink(dataset_path, "dataset_path")
    _validate_str(layer, "layer", max_len=256)
    _validate_int(budget_seconds, "budget_seconds",
                  _MIN_BUDGET_SECONDS, _MAX_BUDGET_SECONDS)
    _validate_int(num_shards, "num_shards", _MIN_SHARDS, _MAX_SHARDS)

    row_count = _count_dataset_rows(dataset_path)
    if row_count == 0:
        raise ValueError("dataset is empty")

    usable_seconds = int(budget_seconds * 0.9)
    per_shard = usable_seconds // num_shards
    feasible = per_shard >= _MIN_PER_SHARD_SECONDS
    reason = (
        "ok"
        if feasible
        else f"budget {budget_seconds}s gives only {per_shard}s/shard "
             f"(need ≥{_MIN_PER_SHARD_SECONDS}s)"
    )

    shard_size = math.ceil(row_count / num_shards)
    shards: list[BlameShardWork] = []
    for sid in range(num_shards):
        offset = sid * shard_size
        size = min(shard_size, max(0, row_count - offset))
        shards.append(
            BlameShardWork(
                shard_id=sid,
                holdout_offset=offset,
                holdout_size=size,
                projected_seconds=per_shard,
            )
        )

    return BlamePlan(
        adapter_dir=adapter_dir,
        dataset_path=dataset_path,
        layer=layer,
        budget_seconds=budget_seconds,
        num_shards=num_shards,
        per_shard_seconds=per_shard,
        shards=tuple(shards),
        feasible=feasible,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Live runner (v0.66.0 — closes v0.57.1 #171)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RowInfluence:
    """Per-row influence score for a held-out probe."""

    row_id: int
    score: float
    shard_id: int

    def __post_init__(self) -> None:
        for name in ("row_id", "shard_id"):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, int):
                raise TypeError(f"{name} must be int")
            if val < 0:
                raise ValueError(f"{name} must be non-negative")
        if isinstance(self.score, bool) or not isinstance(
            self.score, (int, float)
        ):
            raise TypeError("score must be float")
        if not math.isfinite(float(self.score)):
            raise ValueError("score must be finite")


@dataclass(frozen=True)
class BlameResult:
    """End-to-end blame result for ``soup adapters blame --live``."""

    adapter_dir: str
    dataset_path: str
    layer: str
    top_influencers: Tuple[RowInfluence, ...]
    num_rows_scored: int
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.top_influencers, tuple):
            raise TypeError("top_influencers must be tuple")
        for r in self.top_influencers:
            if not isinstance(r, RowInfluence):
                raise TypeError(
                    "top_influencers entries must be RowInfluence"
                )
        if (
            isinstance(self.num_rows_scored, bool)
            or not isinstance(self.num_rows_scored, int)
        ):
            raise TypeError("num_rows_scored must be int")
        if self.num_rows_scored < 0:
            raise ValueError("num_rows_scored must be non-negative")
        if isinstance(self.elapsed_seconds, bool) or not isinstance(
            self.elapsed_seconds, (int, float)
        ):
            raise TypeError("elapsed_seconds must be float")
        if not math.isfinite(float(self.elapsed_seconds)):
            raise ValueError("elapsed_seconds must be finite")
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")


def compute_row_influence(row_grad: Any, probe_grad: Any) -> float:
    """DataInf-style per-row influence score.

    Returns ``cos(row_grad, probe_grad) × |row_grad|``. A positive score
    means the row's gradient is aligned with the probe (it likely pushed
    the model toward the probe's output). Negative scores mean opposed
    (likely pushed away).

    Returns 0.0 when either gradient is zero (degenerate, no signal).
    """
    import numpy as np

    try:
        a = np.asarray(row_grad, dtype=np.float64)
        b = np.asarray(probe_grad, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("row_grad / probe_grad must be array-like") from exc
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: row={a.shape}, probe={b.shape}")
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    cos = float(np.dot(a.ravel(), b.ravel())) / (norm_a * norm_b)
    score = cos * norm_a
    if not math.isfinite(score):
        return 0.0
    return score


def _validate_top_k(top_k: object) -> int:
    if isinstance(top_k, bool):
        raise TypeError("top_k must be int, got bool")
    if not isinstance(top_k, int):
        raise TypeError(f"top_k must be int, got {type(top_k).__name__}")
    if top_k < _MIN_TOP_K:
        raise ValueError(f"top_k must be ≥{_MIN_TOP_K}")
    if top_k > _MAX_TOP_K:
        raise ValueError(f"top_k must be ≤{_MAX_TOP_K}")
    return top_k


def _default_synthetic_probe(plan: BlamePlan):
    """Deterministic synthetic probe — used when no probe_fn supplied.

    The default exists so ``run_blame(plan)`` always produces a real
    ``BlameResult`` even on operator boxes without a GPU. The scores are
    NOT meaningful — they are derived from the plan + a seeded RNG so
    they round-trip identically across Python processes and never raise.

    Seeded by ``sha256(plan-key)[:8]`` (review H3 fix) — Python's built-in
    ``hash()`` is process-salted unless ``PYTHONHASHSEED=0``, so we use
    SHA-256 to make CI runs reproducible regardless of env.
    """
    import numpy as np

    key = (plan.adapter_dir + plan.dataset_path + plan.layer).encode("utf-8")
    # M2 review fix: 64-bit seed (16 hex chars) for symmetry with default_rng.
    seed = int(hashlib.sha256(key).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    # M6 review fix: cap synthetic-probe row count at 100k so a 10M-row
    # dataset doesn't allocate 1.28 GB of float32 by default. Real
    # operators supply a probe_fn that produces correctly-scaled grads;
    # this default is offline-only and capped on purpose.
    n_rows = min(
        sum(s.holdout_size for s in plan.shards),
        _DEFAULT_SYNTH_PROBE_CAP,
    )
    grad_dim = 32
    row_grads = rng.standard_normal((n_rows, grad_dim)).astype(np.float32)
    probe_grad = rng.standard_normal(grad_dim).astype(np.float32)
    return row_grads, probe_grad


def _shard_for_row(plan: BlamePlan, row_id: int) -> int:
    """Map a global row index to its shard id using the plan's holdouts."""
    for shard in plan.shards:
        if shard.holdout_size == 0:
            continue
        if shard.holdout_offset <= row_id < shard.holdout_offset + shard.holdout_size:
            return shard.shard_id
    # Fall through: assign to the last shard
    return plan.shards[-1].shard_id if plan.shards else 0


def run_blame(
    plan: BlamePlan,
    *,
    probe_fn: Optional[Callable[[BlamePlan], Tuple[Any, Any]]] = None,
    top_k: int = _DEFAULT_TOP_K,
) -> BlameResult:
    """Live DataInf-style blame runner.

    Executes the v0.57.0 plan and returns top-K row influences. The
    ``probe_fn`` closure returns ``(row_grads, probe_grad)`` where
    ``row_grads`` is shape ``[N_rows, D]`` and ``probe_grad`` is shape
    ``[D]``. Pass ``probe_fn=None`` to use a deterministic synthetic
    probe (matches v0.54.0 advise stub policy — surface always returns
    a real ``BlameResult`` so downstream tools never see
    ``NotImplementedError``).
    """
    import numpy as np

    if not isinstance(plan, BlamePlan):
        raise TypeError("plan must be BlamePlan")
    if probe_fn is not None and not callable(probe_fn):
        raise TypeError("probe_fn must be callable or None")
    _validate_top_k(top_k)

    start = time.monotonic()
    fn = probe_fn if probe_fn is not None else _default_synthetic_probe
    result = fn(plan)
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("probe_fn must return row_grads, probe_grad tuple")
    row_grads_raw, probe_grad_raw = result
    try:
        row_grads = np.asarray(row_grads_raw, dtype=np.float64)
        probe_grad = np.asarray(probe_grad_raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("probe_fn outputs must be array-like") from exc

    if row_grads.ndim != 2:
        raise ValueError(
            f"row_grads must be 2D [N, D], got {row_grads.shape}"
        )
    if probe_grad.ndim != 1:
        raise ValueError(
            f"probe_grad must be 1D [D], got {probe_grad.shape}"
        )
    if row_grads.shape[1] != probe_grad.shape[0]:
        raise ValueError(
            f"shape mismatch: row_grads[{row_grads.shape}] vs probe_grad"
            f"[{probe_grad.shape}]"
        )

    n_rows = int(row_grads.shape[0])
    # Compute influence per row (vectorised)
    norms = np.linalg.norm(row_grads, axis=1)
    probe_norm = float(np.linalg.norm(probe_grad))
    if probe_norm <= 0:
        scores = np.zeros(n_rows, dtype=np.float64)
    else:
        # cos × |row| = (row · probe) / |probe|
        scores = (row_grads @ probe_grad) / probe_norm
        # Zero out rows with zero norm
        scores = np.where(norms > 0, scores, 0.0)

    # Take top-K by |score|. When actual_k >= n_rows, this returns
    # every row sorted by descending |score| (caller asked for more
    # than we have, so we return everything).
    abs_scores = np.abs(scores)
    actual_k = min(top_k, n_rows)
    if actual_k <= 0:
        top_idx: Any = np.empty((0,), dtype=np.int64)
    elif actual_k >= n_rows:
        top_idx = np.argsort(-abs_scores, kind="stable")
    else:
        part = np.argpartition(-abs_scores, actual_k - 1)[:actual_k]
        top_idx = part[np.argsort(-abs_scores[part], kind="stable")]

    influencers = []
    for idx in top_idx:
        row_id = int(idx)
        s = float(scores[row_id])
        if not math.isfinite(s):
            continue
        influencers.append(
            RowInfluence(
                row_id=row_id,
                score=s,
                shard_id=_shard_for_row(plan, row_id),
            )
        )

    elapsed = time.monotonic() - start
    return BlameResult(
        adapter_dir=plan.adapter_dir,
        dataset_path=plan.dataset_path,
        layer=plan.layer,
        top_influencers=tuple(influencers),
        num_rows_scored=n_rows,
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def render_blame_json(result: BlameResult) -> str:
    """Canonical JSON for CI consumption."""
    if not isinstance(result, BlameResult):
        raise TypeError("result must be BlameResult")
    # L5 review fix: symmetric basename(normpath(...)) for both
    # adapter_dir and dataset_path. A trailing-slash dataset_path of
    # `/abs/foo/bar/` returns "" from `basename` alone; `normpath` strips
    # the trailing slash first.
    payload = {
        "adapter_dir": os.path.basename(os.path.normpath(result.adapter_dir)),
        "dataset_path": os.path.basename(os.path.normpath(result.dataset_path)),
        "layer": result.layer,
        "num_rows_scored": result.num_rows_scored,
        "elapsed_seconds": result.elapsed_seconds,
        "top_influencers": [asdict(r) for r in result.top_influencers],
    }
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)


def _md_escape(value: object) -> str:
    """Escape Rich-markup metacharacters in operator-controlled strings.

    Review H2 fix (v0.66.0): adapter_dir / dataset_path / layer flow
    into Rich markdown that is rendered via ``console.print``. A crafted
    adapter dir like ``"[link=file:///etc/passwd]click[/]"`` would
    inject Rich markup. The escape replaces ``[`` and ``]`` with their
    backslash-escaped forms (Rich-safe).
    """
    return str(value).replace("[", "\\[").replace("]", "\\]")


def render_blame_markdown(result: BlameResult) -> str:
    """Human-readable markdown for PR comments."""
    if not isinstance(result, BlameResult):
        raise TypeError("result must be BlameResult")
    adapter_label = _md_escape(
        os.path.basename(os.path.normpath(result.adapter_dir))
    )
    dataset_label = _md_escape(
        os.path.basename(os.path.normpath(result.dataset_path))
    )
    layer_label = _md_escape(result.layer)
    lines = [
        f"# Blame: {adapter_label}",
        "",
        f"- Dataset: `{dataset_label}`",
        f"- Layer: `{layer_label}`",
        f"- Rows scored: {result.num_rows_scored}",
        f"- Elapsed: {result.elapsed_seconds:.2f}s",
        "",
    ]
    if not result.top_influencers:
        lines.append("_no influencers (empty result)_")
        return "\n".join(lines) + "\n"
    lines.extend([
        "## Top influencers",
        "",
        "| row_id | shard_id | influence |",
        "| --- | --- | --- |",
    ])
    for r in result.top_influencers:
        lines.append(f"| {r.row_id} | {r.shard_id} | {r.score:+.4f} |")
    return "\n".join(lines) + "\n"
