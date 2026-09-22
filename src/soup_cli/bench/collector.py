"""The collecting half of ``bench train`` (#836): it measures, it never judges.

Every verdict lives in :mod:`soup_cli.bench.train_report`, so there is one set
of rules rather than one here and another there. This module only records what
happened: step wall times, whatever ``grad_norm`` the backend logged (or
``None`` when it logged none), supervised token counts, and a fingerprint of the
trainable parameters before the first step and after the last.

``transformers`` is not imported at module scope. The callback class is built on
demand through the same PEP 562 factory ``monitoring/callback.py`` uses, so the
light CLI stays torch-free (``tests/test_cli_startup_is_light.py``).
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Optional

from soup_cli.bench.train_report import ParamSnapshot, StepRecord, build_train_report


def count_supervised_tokens(batch: Any) -> tuple[int, int]:
    """``(useful, total)`` tokens in a collated batch.

    Useful means ``labels != -100`` -- exactly what the loss sees. A batch with
    no ``labels`` counts zero useful tokens rather than guessing: some collators
    build them later, and inventing a number here is how a padded batch starts
    looking like more work than it was.
    """
    labels = None
    if hasattr(batch, "get"):
        labels = batch.get("labels")
    elif hasattr(batch, "labels"):
        labels = batch.labels

    if labels is not None:
        total = int(labels.numel())
        return int((labels != -100).sum().item()), total

    for key in ("input_ids", "tokens"):
        ids = batch.get(key) if hasattr(batch, "get") else None
        if ids is not None:
            return 0, int(ids.numel())
    return 0, 0


def _trainable_with_storage(model: Any) -> list[tuple[str, Any]]:
    """Trainable parameters that actually have storage.

    A parameter on ``meta`` has ``requires_grad`` and no data. Under layer
    streaming adapters can be stranded there, which is why
    ``utils/layer_stream_runtime.py:2161`` exists; counting them would let the
    zero-trainable check pass on a run that trains nothing.
    """
    kept = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if getattr(getattr(param, "device", None), "type", None) == "meta":
            continue
        kept.append((name, param))
    return kept


def summarize_trainable(model: Any) -> ParamSnapshot:
    """Count trainable tensors, keeping ``meta`` ones apart, and fingerprint."""
    kept = _trainable_with_storage(model)
    meta = sum(
        1
        for _, param in model.named_parameters()
        if param.requires_grad
        and getattr(getattr(param, "device", None), "type", None) == "meta"
    )
    digest = _digest(kept)
    return ParamSnapshot(
        count=len(kept),
        fingerprint_first=digest,
        fingerprint_last=digest,
        meta_excluded=meta,
    )


def _digest(named_params) -> str:
    hasher = hashlib.sha256()
    for name, param in named_params:
        hasher.update(name.encode("utf-8"))
        detached = param.detach()
        hasher.update(detached.cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def fingerprint_trainable(model: Any) -> str:
    """A content hash of the trainable parameters only.

    Frozen tensors are deliberately excluded: the check asks whether what was
    meant to train moved, and a buffer or a frozen weight changing is not that.
    """
    return _digest(_trainable_with_storage(model))


class _BenchCollector_body:  # noqa: N801
    """Records one benchmark run. Attached to the trainer as a callback."""

    def __init__(self) -> None:
        self.steps: list[StepRecord] = []
        self.meta_excluded = 0
        self.trainable_count = 0
        self._fingerprint_first: Optional[str] = None
        self._fingerprint_last: Optional[str] = None
        self._step_started: Optional[float] = None
        self._pending_useful = 0
        self._pending_total = 0
        self._now = time.perf_counter
        # Set to ``torch.cuda.synchronize`` on CUDA, so a step boundary is read
        # after the kernels it launched have finished, not when they were queued.
        self._sync = lambda: None

    # -- collection ------------------------------------------------------
    def observe_batch(self, batch: Any) -> None:
        """Called by the wrapped collator, once per collated batch."""
        useful, total = count_supervised_tokens(batch)
        self._pending_useful += useful
        self._pending_total += total

    def on_train_begin(self, args=None, state=None, control=None, **kwargs):
        model = kwargs.get("model")
        if model is not None:
            snapshot = summarize_trainable(model)
            self.trainable_count = snapshot.count
            self.meta_excluded = snapshot.meta_excluded
            self._fingerprint_first = snapshot.fingerprint_first
        return control

    def on_step_begin(self, args=None, state=None, control=None, **kwargs):
        # Only the first step starts here. Every later one starts where the
        # previous one ended: the Trainer fetches and collates a step's batches
        # BEFORE it calls on_step_begin, so timing begin-to-end would leave data
        # loading out of the step and flatter tok/s.
        if self._step_started is None:
            self._sync()
            self._step_started = self._now()
        return control

    def on_step_end(self, args=None, state=None, control=None, **kwargs):
        self._sync()
        ended = self._now()
        started = self._step_started if self._step_started is not None else ended
        self.steps.append(
            StepRecord(
                index=len(self.steps),
                wall_seconds=ended - started,
                grad_norm=None,
                useful_tokens=self._pending_useful,
                total_tokens=self._pending_total,
            )
        )
        self._pending_useful = 0
        self._pending_total = 0
        self._step_started = ended
        return control

    def on_log(self, args=None, state=None, control=None, logs=None, **kwargs):
        # The Trainer logs a step AFTER its on_step_end (transformers
        # trainer.py: on_step_end, then _maybe_log_save_evaluate), so the norm
        # belongs to the step just recorded. Only a norm the backend actually
        # reported: MLX reports none, and defaulting to 0.0 is the bug this
        # contract exists to catch.
        if logs and logs.get("grad_norm") is not None and self.steps:
            if self.steps[-1].grad_norm is None:
                self.steps[-1].grad_norm = float(logs["grad_norm"])
        return control

    def on_train_end(self, args=None, state=None, control=None, **kwargs):
        model = kwargs.get("model")
        if model is not None:
            self._fingerprint_last = fingerprint_trainable(model)
        return control

    # -- handover --------------------------------------------------------
    def param_snapshot(self) -> ParamSnapshot:
        return ParamSnapshot(
            count=self.trainable_count,
            fingerprint_first=self._fingerprint_first or "",
            fingerprint_last=self._fingerprint_last or "",
            meta_excluded=self.meta_excluded,
        )

    def build_report(self, *, warmup_steps: int, provenance: dict, memory: dict, **kwargs):
        """Hand the records to the pure builder, which owns every verdict."""
        return build_train_report(
            steps=self.steps,
            warmup_steps=warmup_steps,
            trainable=self.param_snapshot(),
            provenance=provenance,
            memory=memory,
            **kwargs,
        )


def _get_trainer_callback_base():
    """Lazy-resolve ``transformers.TrainerCallback`` (see monitoring/callback.py)."""
    try:
        from transformers import TrainerCallback

        return TrainerCallback
    except ImportError:
        return object


_LAZY_CALLBACKS = {"BenchCollector": _BenchCollector_body}
_BODY_SKIP = frozenset(("__dict__", "__weakref__"))


def __getattr__(name: str):  # PEP 562
    body = _LAZY_CALLBACKS.get(name)
    if body is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    cls = type(name, (_get_trainer_callback_base(),), dict(vars(body)))
    cls.__module__ = __name__
    cls.__qualname__ = name
    globals()[name] = cls
    return cls
