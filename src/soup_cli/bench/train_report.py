"""The pure half of ``bench train`` (#836): timings, tokens, and three checks.

A throughput number is only evidence if the model was training while it was
measured. Nothing here imports torch: the callback collects, this builds. That
split is what makes every case CPU-testable, including the failure cases, which
are the ones a benchmark harness never exercises in practice.

Why three checks rather than one:

``grad_norm == 0`` is the failure this issue cites, but it can only fire when
the backend reports a norm. MLX deliberately does not, and Soup's own monitoring
callback initialises ``_last_grad_norm = 0.0``
(``monitoring/callback.py:115``), so on that path "absent" and "genuinely zero"
are indistinguishable -- which is exactly why ``None`` here means *not reported*
and is never folded into zero.

The parameter fingerprint is the backend-independent one: bit-identical
trainable parameters between the first and last step mean nothing trained,
whatever any logger said. It is what still works under MLX and DeepSpeed.

The trainable-parameter count catches the case where neither of the others says
anything useful, because there was never anything to move. It counts parameters
with real storage: under layer streaming, adapters can be stranded on ``meta``,
which is why ``utils/layer_stream_runtime.py:2161``
(``assert_trainable_adapters_materialized``) exists at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence


@dataclass
class StepRecord:
    """One optimizer step, as the callback saw it.

    ``grad_norm`` is ``None`` when the backend reported none. That is a
    different fact from ``0.0`` and the two are never merged.
    """

    index: int
    wall_seconds: float
    grad_norm: Optional[float]
    useful_tokens: int
    total_tokens: int


@dataclass
class ParamSnapshot:
    """Trainable-parameter evidence, taken around the measured steps.

    ``count`` counts tensors with ``requires_grad`` AND real storage. The
    fingerprints are content hashes of those tensors at the first and last step;
    the builder only compares them, so how they are produced stays on the
    collecting side.
    """

    count: int
    fingerprint_first: str
    fingerprint_last: str
    meta_excluded: int = 0
    notes: list = field(default_factory=list)


def summarize_step_times(step_times: Sequence[float], warmup_steps: int) -> dict:
    """Median and p95 over the post-warm-up steps, saying how many it dropped.

    p95 is nearest-rank: the reported figure is a step that was actually
    measured, not an interpolation between two of them.

    Raises:
        ValueError: warm-up consumed every step, so there is nothing to
            summarise. Returning an empty median would publish a number derived
            from no measurement.
    """
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must not be negative (got {warmup_steps})")
    counted = list(step_times)[warmup_steps:]
    if not counted:
        raise ValueError(
            f"warmup discarded every step: {warmup_steps} warm-up steps for "
            f"{len(step_times)} measured step(s). Raise --steps or lower --warmup."
        )

    ordered = sorted(counted)
    middle, odd = divmod(len(ordered), 2)
    median = ordered[middle] if odd else (ordered[middle - 1] + ordered[middle]) / 2
    rank = max(1, math.ceil(0.95 * len(ordered)))
    return {
        "median_seconds": median,
        "p95_seconds": ordered[rank - 1],
        "counted_steps": len(counted),
        "warmup_steps_discarded": warmup_steps,
        "total_seconds": sum(counted),
    }


def token_utilisation(useful: int, total: int) -> Optional[float]:
    """``useful / total``, or ``None`` when nothing was counted.

    Supervised tokens are ``labels != -100`` -- what the loss actually sees.
    Reporting tok/s from padded tokens turns a longer pad into a better number.
    """
    if not total:
        return None
    return useful / total


def check_trainable(snapshot: ParamSnapshot) -> Optional[dict]:
    if snapshot.count > 0:
        return None
    detail = ""
    if snapshot.meta_excluded:
        detail = (
            f" {snapshot.meta_excluded} parameter tensor(s) had requires_grad but "
            f"no storage (meta), so they cannot train; under layer streaming this "
            f"is the stranded-adapter case."
        )
    return {
        "check": "trainable_parameters",
        "message": f"0 trainable parameter tensors with real storage.{detail}",
    }


def _check_grad_norms(steps: Sequence[StepRecord]) -> tuple[Optional[dict], str]:
    reported = [s for s in steps if s.grad_norm is not None]
    if not reported:
        # MLX, and any backend that logs no norm. Not a failure on its own; the
        # fingerprint check is what covers these runs.
        return None, "not reported by this backend"
    for step in reported:
        if step.grad_norm == 0.0:
            return {
                "check": "grad_norm",
                "message": (
                    f"step {step.index} logged grad_norm == 0.0: the model was "
                    f"not training while this was measured."
                ),
            }, "failed"
        if not math.isfinite(step.grad_norm):
            return {
                "check": "grad_norm",
                "message": (
                    f"step {step.index} logged a non-finite grad_norm "
                    f"({step.grad_norm!r}); the run diverged."
                ),
            }, "failed"
    return None, "all reported steps finite and non-zero"


def _check_parameters_changed(snapshot: ParamSnapshot) -> Optional[dict]:
    if snapshot.fingerprint_first != snapshot.fingerprint_last:
        return None
    return {
        "check": "parameters_changed",
        "message": (
            "trainable parameters are bit-identical between the first and last "
            "measured step: nothing trained, whatever the logs said."
        ),
    }


def build_train_report(
    *,
    steps: Sequence[StepRecord],
    warmup_steps: int,
    trainable: ParamSnapshot,
    provenance: dict,
    memory: dict,
    config_hash: Optional[str] = None,
    steps_requested: Optional[int] = None,
    extra: Optional[dict] = None,
) -> dict[str, Any]:
    """Assemble the report and run the three hard checks.

    ``valid`` is false when any check fails, and the caller exits non-zero. The
    report is still written, with the failures in it: a run that did not train
    is a result, and hiding it loses the evidence.
    """
    timing = summarize_step_times([s.wall_seconds for s in steps], warmup_steps)
    counted = list(steps)[warmup_steps:]
    useful = sum(s.useful_tokens for s in counted)
    total = sum(s.total_tokens for s in counted)
    seconds = timing["total_seconds"]

    grad_failure, grad_state = _check_grad_norms(counted)
    failures = [
        failure
        for failure in (
            check_trainable(trainable),
            grad_failure,
            _check_parameters_changed(trainable),
        )
        if failure is not None
    ]
    if steps_requested is not None and len(steps) != steps_requested:
        # Matched work is the premise of comparing two runs at all.
        failures.append({
            "check": "step_count",
            "message": (
                f"{len(steps)} optimizer step(s) measured, {steps_requested} "
                f"requested: this run did not do the work it is labelled with."
            ),
        })

    return {
        "valid": not failures,
        "failures": failures,
        "checks": {
            "trainable_parameters": trainable.count,
            "grad_norm": grad_state,
            "parameters_changed": trainable.fingerprint_first != trainable.fingerprint_last,
        },
        "timing": timing,
        "tokens": {
            "useful": useful,
            "total": total,
            "utilisation": token_utilisation(useful, total),
        },
        "throughput": {
            "useful_tokens_per_second": (useful / seconds) if seconds else None,
            "total_tokens_per_second": (total / seconds) if seconds else None,
        },
        "memory": dict(memory),
        "provenance": dict(provenance),
        "config_hash": config_hash,
        "steps_requested": steps_requested,
        "steps_measured": len(steps),
        **(extra or {}),
    }
