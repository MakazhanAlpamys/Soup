"""DynamicCurriculumCallback — live HF Trainer wiring for dynamic curriculum (v0.53.5 #114).

Lifts the v0.48.0 Part A `compute_bucket_weights` + `DynamicCurriculumPolicy`
schema into a real HF `TrainerCallback` that:

1. Accumulates per-step loss + grad_norm fingerprints into N difficulty buckets.
2. Every ``policy.recompute_every_n_steps`` global steps, calls
   :func:`soup_cli.utils.curriculum_dynamic.compute_bucket_weights` to refresh
   the sampler weights.
3. Coordinates per-bucket stats across DDP ranks via ``all_reduce(SUM)`` BEFORE
   computing weights — defends against the per-rank divergence footgun
   documented in v0.48.0 ``validate_distributed_curriculum``.
4. Appends one JSONL row per recompute to ``<output_dir>/curriculum_history.jsonl``
   on rank-0 only (atomic via tempfile + os.replace).

Stub-then-live pattern: the v0.48.0 release shipped the math + schema; this
v0.53.5 release ships the live callback (mirrors v0.27.0 MII / v0.37.0
multipack / v0.41.0 LLaMA Pro). BETA: real-world tuning is still
deferred — flag stays under the ``BETA:`` schema description.

Security:
- ``output_dir`` containment via ``utils.paths.is_under_cwd`` + null-byte
  rejection + 4096-char cap.
- File writes are rank-0 only; the JSONL append uses tempfile staging so a
  crash mid-write cannot leave a truncated file.
- ``torch.distributed`` is imported lazily inside the callback so importing
  this module is cheap and never opens a CUDA context.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import tempfile
from typing import Any, Dict, Optional, Tuple

from soup_cli.utils.curriculum_dynamic import (
    DynamicCurriculumPolicy,
    compute_bucket_weights,
)
from soup_cli.utils.paths import is_under_cwd

logger = logging.getLogger(__name__)

_MAX_PATH_LEN = 4096
_HISTORY_FILENAME = "curriculum_history.jsonl"

__all__ = [
    "DynamicCurriculumCallback",
    "_is_rank_zero",
    "_pick_bucket",
]


def _is_rank_zero() -> bool:
    """Return True when the current process is rank-0 (or single-process).

    Lazy-imports torch so importing this module never opens a CUDA context.
    Any error path returns True (single-process / non-distributed fallback).
    """
    try:
        import torch.distributed as dist  # noqa: PLC0415

        if not dist.is_available() or not dist.is_initialized():
            return True
        return dist.get_rank() == 0
    except Exception:  # noqa: BLE001 — torch missing or pre-init.
        return True


def _pick_bucket(global_step: int, num_buckets: int) -> int:
    """Map a step → bucket id via simple modulo round-robin.

    Higher-quality curricula (e.g. by sample loss percentile) can be wired
    later; this BETA pass uses step-mod so the callback exercises every
    bucket evenly during the warm-up phase.
    """
    if isinstance(global_step, bool) or not isinstance(global_step, int):
        raise TypeError(
            f"global_step must be int, got {type(global_step).__name__}"
        )
    if isinstance(num_buckets, bool) or not isinstance(num_buckets, int):
        raise TypeError(
            f"num_buckets must be int, got {type(num_buckets).__name__}"
        )
    if global_step < 0:
        raise ValueError(f"global_step must be >= 0, got {global_step}")
    if num_buckets < 1:
        raise ValueError(f"num_buckets must be >= 1, got {num_buckets}")
    return global_step % num_buckets


def _validate_output_dir(output_dir: object) -> str:
    if not isinstance(output_dir, str):
        raise TypeError(
            f"output_dir must be str, got {type(output_dir).__name__}"
        )
    if not output_dir:
        raise ValueError("output_dir must be non-empty")
    if "\x00" in output_dir:
        raise ValueError("output_dir must not contain null bytes")
    if len(output_dir) > _MAX_PATH_LEN:
        raise ValueError(
            f"output_dir length {len(output_dir)} exceeds cap {_MAX_PATH_LEN}"
        )
    real = os.path.realpath(output_dir)
    if not is_under_cwd(real):
        raise ValueError(
            f"output_dir is outside cwd: {os.path.basename(real)!r}"
        )
    return real


def _try_import_callback_base():
    """Return the HF ``TrainerCallback`` class or a thin stand-in.

    The HF base supplies ``on_step_end`` / ``on_log`` no-op defaults; when
    ``transformers`` is missing (e.g. on a slim CI runner) we substitute an
    object base so unit tests can still construct the callback without the
    full HF stack.
    """
    try:
        from transformers import TrainerCallback  # noqa: PLC0415

        return TrainerCallback
    except Exception:  # noqa: BLE001 — transformers optional in some tests.
        return object


class DynamicCurriculumCallback(_try_import_callback_base()):  # type: ignore[misc]
    """HF TrainerCallback emitting dynamic curriculum bucket-weight history.

    BETA: the callback is live in v0.53.5 but the sampler-side consumer
    (live re-weighting of an HF Dataset sampler) is still wired through the
    existing v0.48.0 schema gate. This callback's primary deliverable is the
    ``curriculum_history.jsonl`` record so ``soup runs curriculum-curve``
    can render the BO trajectory.

    Args:
        policy: A frozen :class:`DynamicCurriculumPolicy`.
        output_dir: Directory (under cwd) to write
            ``curriculum_history.jsonl`` to. Must be a real path under
            ``os.getcwd()`` — null bytes / oversize / outside-cwd rejected.
    """

    def __init__(
        self,
        policy: DynamicCurriculumPolicy,
        output_dir: str,
    ) -> None:
        if not isinstance(policy, DynamicCurriculumPolicy):
            raise TypeError(
                "policy must be DynamicCurriculumPolicy, got "
                f"{type(policy).__name__}"
            )
        self._policy = policy
        self._output_dir = _validate_output_dir(output_dir)
        # Per-bucket accumulator: bucket_id -> {num_samples, loss_sum, grad_norm_sum}
        self._stats: Dict[int, Dict[str, float]] = {}
        # Most recently computed weights (read by external sampler hook).
        self._current_weights: Tuple[float, ...] = tuple(
            [1.0 / policy.num_buckets] * policy.num_buckets
        )
        self._history_path = os.path.join(self._output_dir, _HISTORY_FILENAME)

    # ------------------------------------------------------------------
    # Public accessors (sampler hook + tests)
    # ------------------------------------------------------------------

    @property
    def policy(self) -> DynamicCurriculumPolicy:
        return self._policy

    @property
    def output_dir(self) -> str:
        return self._output_dir

    @property
    def current_weights(self) -> Tuple[float, ...]:
        """Defensive copy of the latest computed weights."""
        return tuple(self._current_weights)

    @property
    def history_path(self) -> str:
        return self._history_path

    def reset_stats(self) -> None:
        """Clear the in-memory accumulator (called after each recompute)."""
        self._stats = {}

    # ------------------------------------------------------------------
    # HF Trainer hooks (signature compat — ``*args, **kwargs`` for HF ≥4.41)
    # ------------------------------------------------------------------

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: Optional[Dict[str, Any]] = None,
        *_args: Any,
        **_kwargs: Any,
    ) -> None:
        """Capture latest loss + grad_norm from ``state.log_history``."""
        if logs is None:
            return
        try:
            global_step = int(getattr(state, "global_step", 0) or 0)
        except (TypeError, ValueError):
            return
        if global_step < 0:
            return
        # HF emits "loss" + (optionally) "grad_norm" in `logs`.
        loss = logs.get("loss")
        grad_norm = logs.get("grad_norm")
        # Bucket id derived from the global step (round-robin BETA strategy).
        try:
            bucket_id = _pick_bucket(global_step, self._policy.num_buckets)
        except (TypeError, ValueError):
            return
        self._record_sample(bucket_id, loss, grad_norm)

    def on_step_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        *_args: Any,
        **_kwargs: Any,
    ) -> None:
        """Recompute bucket weights when the policy says it's time."""
        try:
            global_step = int(getattr(state, "global_step", 0) or 0)
        except (TypeError, ValueError):
            return
        try:
            if not self._policy.should_recompute(global_step):
                return
        except (TypeError, ValueError):
            return
        self._recompute_and_record(global_step)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_sample(
        self,
        bucket_id: int,
        loss: object,
        grad_norm: object,
    ) -> None:
        try:
            loss_f = float(loss) if loss is not None else 0.0
        except (TypeError, ValueError):
            loss_f = 0.0
        try:
            grad_norm_f = float(grad_norm) if grad_norm is not None else 0.0
        except (TypeError, ValueError):
            grad_norm_f = 0.0
        # Reject NaN/Inf silently — defensive (we don't want a stray spike to
        # crash training; the recompute step will see a representative mean).
        import math  # noqa: PLC0415

        if not math.isfinite(loss_f):
            loss_f = 0.0
        if not math.isfinite(grad_norm_f):
            grad_norm_f = 0.0
        if loss_f < 0.0:
            loss_f = 0.0
        if grad_norm_f < 0.0:
            grad_norm_f = 0.0
        slot = self._stats.setdefault(
            bucket_id,
            {"num_samples": 0.0, "loss_sum": 0.0, "grad_norm_sum": 0.0},
        )
        slot["num_samples"] += 1.0
        slot["loss_sum"] += loss_f
        slot["grad_norm_sum"] += grad_norm_f

    def _recompute_and_record(self, global_step: int) -> None:
        coordinated = self._coordinate_distributed()
        # Build the v0.48.0 BucketStats-shaped mapping (mean_loss/mean_grad_norm).
        stats_mapping: Dict[int, Dict[str, float]] = {}
        for bucket_id, payload in coordinated.items():
            n = max(0, int(payload.get("num_samples", 0.0)))
            if n <= 0:
                continue
            stats_mapping[bucket_id] = {
                "num_samples": n,
                "mean_loss": payload.get("loss_sum", 0.0) / n,
                "mean_grad_norm": payload.get("grad_norm_sum", 0.0) / n,
            }
        try:
            weights = compute_bucket_weights(stats_mapping, self._policy)
        except (TypeError, ValueError) as exc:
            logger.debug(
                "compute_bucket_weights skipped at step=%d: %s",
                global_step,
                exc,
            )
            self.reset_stats()
            return
        self._current_weights = tuple(weights)
        if _is_rank_zero():
            self._append_history_row(global_step, weights)
        # Always reset after recompute — staleness defends against
        # bucket drift dominated by very early steps.
        self.reset_stats()

    def _coordinate_distributed(self) -> Dict[int, Dict[str, float]]:
        """All-reduce per-bucket stats across ranks (SUM); single-process passthrough.

        On any error path (torch missing, group not initialised, mismatched
        bucket sets across ranks) we fall back to the local snapshot — never
        crash training. Documented hazard: silently divergent ranks will
        produce slightly different weights; defenders should ensure the
        ``num_buckets`` matches the policy on every rank (validated by the
        schema at config-load).
        """
        local_snapshot: Dict[int, Dict[str, float]] = {
            bucket_id: {
                "num_samples": float(slot.get("num_samples", 0.0)),
                "loss_sum": float(slot.get("loss_sum", 0.0)),
                "grad_norm_sum": float(slot.get("grad_norm_sum", 0.0)),
            }
            for bucket_id, slot in self._stats.items()
        }
        try:
            import torch  # noqa: PLC0415
            import torch.distributed as dist  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            return local_snapshot
        if not dist.is_available() or not dist.is_initialized():
            return local_snapshot
        try:
            world_size = dist.get_world_size()
        except Exception:  # noqa: BLE001
            return local_snapshot
        if world_size <= 1:
            return local_snapshot
        # Construct a per-bucket tensor (every rank must have the same shape).
        nb = self._policy.num_buckets
        flat = torch.zeros(nb * 3, dtype=torch.float64)
        for bucket_id, payload in local_snapshot.items():
            if not isinstance(bucket_id, int) or bucket_id < 0 or bucket_id >= nb:
                continue
            base = bucket_id * 3
            flat[base + 0] = payload["num_samples"]
            flat[base + 1] = payload["loss_sum"]
            flat[base + 2] = payload["grad_norm_sum"]
        try:
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        except Exception as exc:  # noqa: BLE001
            logger.debug("all_reduce failed in curriculum callback: %s", exc)
            return local_snapshot
        result: Dict[int, Dict[str, float]] = {}
        for bucket_id in range(nb):
            base = bucket_id * 3
            n = float(flat[base + 0].item())
            if n <= 0.0:
                continue
            result[bucket_id] = {
                "num_samples": n,
                "loss_sum": float(flat[base + 1].item()),
                "grad_norm_sum": float(flat[base + 2].item()),
            }
        return result

    def _append_history_row(
        self,
        global_step: int,
        weights: Tuple[float, ...],
    ) -> None:
        """Append one JSONL row to the history file.

        Tempfile-staged then ``os.replace`` to keep the prior file intact on
        partial writes. We re-read the existing file (when present) and
        re-write it plus the new row — keeps the operation atomic at the
        cost of one extra fsync per recompute. For the BETA cadence
        (recompute_every_n_steps >= 1), this is a non-issue.
        """
        try:
            os.makedirs(self._output_dir, exist_ok=True)
        except OSError as exc:
            logger.debug("curriculum_history mkdir failed: %s", exc)
            return
        # Symlink rejection at the target path (TOCTOU defence — mirrors
        # v0.33.0 #22 / v0.43.0 Part C / v0.44.0 Part B / v0.45.0 Part E).
        try:
            st = os.lstat(self._history_path)
            if stat.S_ISLNK(st.st_mode):
                logger.debug(
                    "curriculum_history target is symlink; refusing to write"
                )
                return
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.debug("curriculum_history lstat failed: %s", exc)
            return

        existing = ""
        try:
            with open(self._history_path, "r", encoding="utf-8") as fh:
                existing = fh.read()
        except FileNotFoundError:
            existing = ""
        except OSError as exc:
            logger.debug("curriculum_history read failed: %s", exc)
            return

        row = {
            "step": int(global_step),
            "weights": [float(w) for w in weights],
        }
        try:
            row_json = json.dumps(row, ensure_ascii=True)
        except (TypeError, ValueError) as exc:
            logger.debug("curriculum_history json dump failed: %s", exc)
            return

        fd, tmp_path = tempfile.mkstemp(
            prefix=".curriculum_history.",
            dir=self._output_dir,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                if existing and not existing.endswith("\n"):
                    fh.write(existing)
                    fh.write("\n")
                else:
                    fh.write(existing)
                fh.write(row_json)
                fh.write("\n")
            os.replace(tmp_path, self._history_path)
        except OSError as exc:
            logger.debug("curriculum_history write failed: %s", exc)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
