"""KL-controlled DPO variants (v0.40.0 Part C).


Two opt-in controls for DPO-family preference training:

1. **β schedule** — anneal the DPO ``beta`` coefficient over training.
   Three shapes: linear, cosine (1/2 (1 + cos(pi t)) ramp), exponential
   (geometric decay between ``beta_start`` and ``beta_end``).

2. **Reference-model regeneration** — every ``every_n_epochs``, replace
   the frozen ref-model weights with a deep-copy of the current student.
   Useful for self-improving loops where the policy quickly outpaces
   the original reference.

Both helpers are duck-typed callbacks (no ``transformers`` import at
module scope) so they cost nothing on a torch-less interpreter and stay
unit-testable on CI without GPUs.
"""

from __future__ import annotations

import math
from typing import Optional

# Allowed schedule shapes, kept as a frozenset for runtime immutability.
SUPPORTED_SCHEDULES: frozenset[str] = frozenset({"linear", "cosine", "exponential"})


def _validate_finite_positive(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number; got {type(value).__name__}")
    fvalue = float(value)
    if math.isnan(fvalue) or math.isinf(fvalue):
        raise ValueError(f"{name}={value!r} must be finite")
    if fvalue <= 0:
        raise ValueError(f"{name}={value!r} must be > 0")


def compute_beta_at_step(
    beta_start: float,
    beta_end: float,
    step: int,
    total_steps: int,
    schedule: str,
) -> float:
    """Return β at ``step`` for the chosen schedule.

    Endpoint contract (matches HF lr_scheduler convention):
      step ≤ 0           → ``beta_start``
      step ≥ total_steps → ``beta_end``
      total_steps == 0   → ``beta_end`` (degenerate case; nothing to anneal)
    """
    _validate_finite_positive("beta_start", beta_start)
    _validate_finite_positive("beta_end", beta_end)
    if isinstance(total_steps, bool) or not isinstance(total_steps, int):
        raise ValueError(
            f"total_steps must be int; got {type(total_steps).__name__}"
        )
    if isinstance(step, bool) or not isinstance(step, int):
        # Defence-in-depth: bool is a subclass of int. Project policy
        # (v0.30.0 Candidate) rejects bool for int fields.
        raise ValueError(f"step must be int; got {type(step).__name__}")
    if total_steps < 0:
        raise ValueError(f"total_steps={total_steps!r} must be >= 0")
    if schedule not in SUPPORTED_SCHEDULES:
        raise ValueError(
            f"Unknown schedule={schedule!r}; supported: {sorted(SUPPORTED_SCHEDULES)}"
        )
    if total_steps == 0:
        return float(beta_end)
    if step <= 0:
        return float(beta_start)
    if step >= total_steps:
        return float(beta_end)
    progress = step / total_steps  # 0..1
    if schedule == "linear":
        return float(beta_start + (beta_end - beta_start) * progress)
    if schedule == "cosine":
        # Cosine ramp: 1/2 (1 + cos(pi * progress)) goes 1→0 over [0,1].
        weight = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(beta_end + (beta_start - beta_end) * weight)
    # exponential — geometric interpolation; both endpoints are > 0 by guard.
    log_start = math.log(beta_start)
    log_end = math.log(beta_end)
    return float(math.exp(log_start + (log_end - log_start) * progress))


class BetaScheduleCallback:
    """Duck-typed ``TrainerCallback``: updates ``trainer.beta`` per step.

    The HF ``TrainerCallback`` signature is matched without importing
    ``transformers`` at module scope so this object is constructible in
    a torch-less environment for unit testing.

    Use ``attach(trainer)`` once, then add the callback to the trainer.
    Without ``attach``, ``on_step_begin`` is a no-op (defence-in-depth).
    """

    def __init__(
        self,
        beta_start: float,
        beta_end: float,
        total_steps: int,
        schedule: str,
    ) -> None:
        # Validate immediately so callers don't see a deferred crash mid-train.
        _validate_finite_positive("beta_start", beta_start)
        _validate_finite_positive("beta_end", beta_end)
        if schedule not in SUPPORTED_SCHEDULES:
            raise ValueError(
                f"Unknown schedule={schedule!r}; supported: "
                f"{sorted(SUPPORTED_SCHEDULES)}"
            )
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.total_steps = int(total_steps)
        self.schedule = schedule
        self._trainer = None

    def attach(self, trainer) -> None:
        self._trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs) -> None:
        # HF Trainer populates state.max_steps inside _inner_training_loop,
        # which runs before on_train_begin. Resolve total_steps lazily from
        # the live trainer state so the wrappers don't need to compute it
        # ahead of time (the wrappers' pre-train computation is racy and
        # was returning 0 in v0.40.0 first cut).
        if self.total_steps <= 0:
            live_max = int(getattr(state, "max_steps", 0) or 0)
            if live_max > 0:
                self.total_steps = live_max

    def on_step_begin(self, args, state, control, **kwargs) -> None:
        if self._trainer is None:
            return
        if self.total_steps <= 0:
            # Couldn't resolve a meaningful horizon; skip the schedule
            # rather than emit beta_end for every step.
            return
        new_beta = compute_beta_at_step(
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            step=int(getattr(state, "global_step", 0)),
            total_steps=self.total_steps,
            schedule=self.schedule,
        )
        # TRL DPOTrainer exposes .beta directly. Narrow the swallow to
        # AttributeError only — a TypeError here would indicate a real bug
        # in compute_beta_at_step that should surface in tests, not be hidden.
        try:
            self._trainer.beta = new_beta
        except AttributeError:
            return


class RefModelRegenCallback:
    """Duck-typed callback: deep-copy student weights into ref_model on epoch.

    On every Nth epoch (1-indexed; epoch 0 is skipped to avoid copying
    untrained weights), copies the current ``trainer.model`` state_dict
    into ``trainer.ref_model``. Falls back to a no-op when ``ref_model``
    is missing (some preference trainers, e.g. ORPO, are reference-free).
    """

    def __init__(self, every_n_epochs: int) -> None:
        if isinstance(every_n_epochs, bool) or not isinstance(every_n_epochs, int):
            raise TypeError(
                f"every_n_epochs must be int; got {type(every_n_epochs).__name__}"
            )
        if every_n_epochs < 1:
            raise ValueError(
                f"every_n_epochs={every_n_epochs!r} must be >= 1"
            )
        self.every_n_epochs = every_n_epochs
        self._trainer = None
        self.regen_count = 0

    def attach(self, trainer) -> None:
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs) -> None:
        if self._trainer is None:
            return
        epoch = float(getattr(state, "epoch", 0.0))
        # Skip epoch 0 entirely — copying untrained student is a footgun.
        # round() handles HF Trainer's float epoch counters (e.g. 1.999...).
        epoch_int = int(round(epoch))
        if epoch_int < 1:
            return
        if epoch_int % self.every_n_epochs != 0:
            return
        self._regenerate()

    def _regenerate(self) -> None:
        trainer = self._trainer
        if trainer is None:
            return
        ref_model = getattr(trainer, "ref_model", None)
        student = getattr(trainer, "model", None)
        if ref_model is None or student is None:
            return
        try:
            state_dict = student.state_dict()
        except AttributeError:
            return
        # strict=True so a key/shape mismatch (e.g. PEFT-wrapped student vs
        # bare ref) surfaces loudly instead of silently producing a
        # half-copied reference. The except below logs at WARNING and the
        # callback continues — this is an optimisation, not a safety gate,
        # but operators should know when it failed.
        try:
            ref_model.load_state_dict(state_dict, strict=True)
            self.regen_count += 1
        except (RuntimeError, TypeError) as exc:
            import logging

            logging.getLogger(__name__).warning(
                "RefModelRegenCallback: load_state_dict failed (%s); ref "
                "model not updated this epoch.", type(exc).__name__,
            )
            return


def build_dpo_variant_callbacks(
    *,
    beta_start: float,
    beta_end: Optional[float],
    schedule: Optional[str],
    total_steps: int,
    ref_regen_epochs: Optional[int],
) -> list:
    """Build the DPO-variant callback list for a given config slice.

    Returns an empty list when no variants are enabled. Callers should
    ``attach(trainer)`` and ``trainer.add_callback(cb)`` for each entry.
    """
    callbacks: list = []
    if schedule is not None and beta_end is not None:
        callbacks.append(
            BetaScheduleCallback(
                beta_start=beta_start,
                beta_end=beta_end,
                total_steps=total_steps,
                schedule=schedule,
            )
        )
    if ref_regen_epochs is not None:
        callbacks.append(RefModelRegenCallback(every_n_epochs=ref_regen_epochs))
    return callbacks
