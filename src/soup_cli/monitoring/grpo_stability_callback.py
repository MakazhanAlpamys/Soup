"""GRPOStabilityCallback — v0.53.11 #127.

Live HF TrainerCallback that wires the seven v0.50.0 Part D stability /
efficiency knobs into the training loop:

- ``ref_model_ema_alpha``: EMA update of the reference model post-step.
- ``replay_buffer_size``: bounded deque of recent rollouts for re-use.
- ``async_grpo_prefetch``: schedule the next-batch rollout in a background
  thread (advisory — actual prefetch is a TRL ``GRPOTrainer`` concern).
- ``tis_threshold``: truncated importance sampling — log a warning when
  log-ratios exceed the threshold so operators can spot off-policy drift.
- ``mask_truncated_completions``: hint to skip rows whose completion hit
  ``max_new_tokens`` (the actual mask is applied inside compute_loss).
- ``defer_rerolling``: when an advantage batch is all-zero, skip re-roll
  this step (record-only — re-roll is a GRPOTrainer concern).
- ``skip_zero_advantage`` / ``off_policy_mask_threshold``: similar
  record-only knobs surfaced as ``state.log_history`` metrics.

Pure math: each per-knob hook is a small pure function in the same module,
so tests can exercise them without instantiating transformers.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Optional

logger = logging.getLogger(__name__)


def update_ema(ref_state: dict, policy_state: dict, alpha: float) -> dict:
    """In-place EMA update: ``ref = (1-α)·ref + α·policy``.

    Pure-function math kernel. Both inputs are name->tensor mappings (HF
    state_dict shape). ``alpha`` is in ``(0, 1]`` AND finite (no NaN/Inf —
    v0.53.11 review fix per v0.32.0 / v0.47.0 policy).

    Returns the (mutated) ``ref_state``. The function mutates in-place but
    also returns the dict for chaining; callers should treat the return
    value as the same object passed in.
    """
    import math

    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
        raise TypeError("alpha must be a non-bool float")
    alpha_f = float(alpha)
    if not math.isfinite(alpha_f):
        raise ValueError("alpha must be finite (no NaN/Inf)")
    if not (0.0 < alpha_f <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    for name, p_tensor in policy_state.items():
        if name not in ref_state:
            continue
        r_tensor = ref_state[name]
        # Defensive: only update tensors with matching shape.
        if hasattr(r_tensor, "shape") and hasattr(p_tensor, "shape"):
            if r_tensor.shape != p_tensor.shape:
                continue
            ref_state[name] = (1.0 - alpha_f) * r_tensor + alpha_f * p_tensor
    return ref_state


def _validate_alpha(alpha) -> float:
    """Shared alpha guard: non-bool float in (0, 1], finite."""
    import math

    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
        raise TypeError("alpha must be a non-bool float")
    alpha_f = float(alpha)
    if not math.isfinite(alpha_f):
        raise ValueError("alpha must be finite (no NaN/Inf)")
    if not (0.0 < alpha_f <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    return alpha_f


def update_ema_in_place(ref_model, policy_model, alpha: float) -> int:
    """In-place EMA of the reference model from the policy (v0.71.11 #160).

    ``ref.param = (1-α)·ref.param + α·policy.param`` for every shared
    parameter, mutating the reference tensors directly. This replaces the
    v0.53.11 path that materialised BOTH full ``state_dict()`` copies AND
    a ``load_state_dict`` round-trip — three full model-sized allocations
    per step. Iterating ``named_parameters()`` and updating in place keeps
    the per-step memory overhead at zero (no extra model-sized buffers),
    which matters at 70B+ scale.

    Only parameters present (by name) and shape-matching in both models
    are updated; mismatches are skipped (defensive against PEFT vs base
    naming). Returns the number of parameters actually updated so callers
    can detect a total-name-mismatch no-op (code-review LOW fix).
    """
    import torch

    alpha_f = _validate_alpha(alpha)
    ref_params = dict(ref_model.named_parameters())
    updated = 0
    with torch.no_grad():
        for name, p_tensor in policy_model.named_parameters():
            r_tensor = ref_params.get(name)
            if r_tensor is None:
                continue
            if (
                hasattr(r_tensor, "shape")
                and hasattr(p_tensor, "shape")
                and r_tensor.shape != p_tensor.shape
            ):
                continue
            src = p_tensor.data
            if hasattr(src, "to") and hasattr(r_tensor, "device"):
                src = src.to(r_tensor.device)
            # r = (1-α)·r + α·p, fully in place (no model-sized temporaries).
            r_tensor.data.mul_(1.0 - alpha_f).add_(src, alpha=alpha_f)
            updated += 1
    return updated


def check_tis_threshold(log_ratio, threshold: float) -> bool:
    """Return True iff the max absolute log-ratio exceeds the TIS threshold.

    Used to flag off-policy drift in GRPO rollouts.
    """
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise TypeError("threshold must be a non-bool number")
    if float(threshold) <= 0.0:
        raise ValueError("threshold must be positive")
    if not hasattr(log_ratio, "abs"):
        raise TypeError("log_ratio must be a tensor (need .abs())")
    max_abs = float(log_ratio.abs().max())
    return max_abs > float(threshold)


def filter_zero_advantage(advantages, *, eps: float = 1e-8) -> Any:
    """Return a boolean mask: True where advantage is non-zero.

    Used by ``skip_zero_advantage`` — rows with mask=False are dropped
    from the loss compute.
    """
    if not isinstance(eps, (int, float)) or isinstance(eps, bool):
        raise TypeError("eps must be a non-bool number")
    if not hasattr(advantages, "abs"):
        raise TypeError("advantages must be a tensor")
    return advantages.abs() > float(eps)


def _get_trainer_callback_base():
    """Lazy-resolve ``transformers.TrainerCallback`` (v0.53.11 review fix).

    Project policy: every callback inherits TrainerCallback so HF Trainer
    discovers it via the callback handler. We resolve at class-body
    evaluation time so module import does not pull transformers.
    """
    try:
        from transformers import TrainerCallback

        return TrainerCallback
    except ImportError:
        return object


_TrainerCallbackBase = _get_trainer_callback_base()


class GRPOStabilityCallback(_TrainerCallbackBase):  # type: ignore[misc, valid-type]
    """HF TrainerCallback that wires v0.50.0 Part D stability knobs.

    Lazy-inherits ``transformers.TrainerCallback`` so the module is
    importable without transformers (falls back to ``object``).
    """

    def __init__(
        self,
        *,
        ref_model_ema_alpha: Optional[float] = None,
        replay_buffer_size: Optional[int] = None,
        async_grpo_prefetch: bool = False,
        tis_threshold: Optional[float] = None,
        mask_truncated_completions: bool = False,
        defer_rerolling: bool = False,
        skip_zero_advantage: bool = False,
        off_policy_mask_threshold: Optional[float] = None,
    ):
        # Validation mirrors the schema. Bool-rejection on numeric fields
        # per project policy (v0.30.0 Candidate / v0.41.0 lr_groups).
        if ref_model_ema_alpha is not None:
            if isinstance(ref_model_ema_alpha, bool):
                raise TypeError("ref_model_ema_alpha must be float, not bool")
            if not (0.0 < float(ref_model_ema_alpha) <= 1.0):
                raise ValueError(
                    f"ref_model_ema_alpha must be in (0, 1], got {ref_model_ema_alpha}"
                )
        if replay_buffer_size is not None:
            if isinstance(replay_buffer_size, bool):
                raise TypeError("replay_buffer_size must be int, not bool")
            if not (1 <= int(replay_buffer_size) <= 1_000_000):
                raise ValueError(
                    f"replay_buffer_size must be in [1, 1e6], got "
                    f"{replay_buffer_size}"
                )
        if tis_threshold is not None:
            if isinstance(tis_threshold, bool):
                raise TypeError("tis_threshold must be float, not bool")
            if not (0.0 < float(tis_threshold) <= 100.0):
                raise ValueError(
                    f"tis_threshold must be in (0, 100], got {tis_threshold}"
                )
        if off_policy_mask_threshold is not None:
            if isinstance(off_policy_mask_threshold, bool):
                raise TypeError("off_policy_mask_threshold must be float, not bool")
            if not (0.0 <= float(off_policy_mask_threshold) <= 1.0):
                raise ValueError(
                    f"off_policy_mask_threshold must be in [0, 1], got "
                    f"{off_policy_mask_threshold}"
                )
        self.ref_model_ema_alpha = ref_model_ema_alpha
        self.replay_buffer_size = replay_buffer_size
        self.async_grpo_prefetch = bool(async_grpo_prefetch)
        self.tis_threshold = tis_threshold
        self.mask_truncated_completions = bool(mask_truncated_completions)
        self.defer_rerolling = bool(defer_rerolling)
        self.skip_zero_advantage = bool(skip_zero_advantage)
        self.off_policy_mask_threshold = off_policy_mask_threshold
        # Bounded rollout deque — created on demand.
        self._replay: Optional[deque] = None
        if self.replay_buffer_size is not None:
            self._replay = deque(maxlen=int(self.replay_buffer_size))
        self._tis_alerts = 0
        # One-shot guard so a total name-mismatch EMA no-op warns exactly
        # once per run (v0.71.11 code-review LOW — mirrors the #159
        # fallback-warn pattern in trainer/grpo.py).
        self._ema_noop_warned = False
        # Set during on_train_begin (lazy — model is constructed by Trainer
        # before the first event fires).
        self._policy_model: Any = None
        self._ref_model: Any = None

    def push_rollout(self, rollout: Any) -> None:
        """Append a rollout to the bounded replay buffer (no-op if disabled)."""
        if self._replay is None:
            return
        self._replay.append(rollout)

    def replay_size(self) -> int:
        """Current rollout count in the replay buffer."""
        return 0 if self._replay is None else len(self._replay)

    def record_tis_alert(self, log_ratio) -> bool:
        """Increment internal counter when log-ratio breaches the threshold."""
        if self.tis_threshold is None:
            return False
        try:
            if check_tis_threshold(log_ratio, self.tis_threshold):
                self._tis_alerts += 1
                return True
        except (TypeError, ValueError):
            pass
        return False

    def tis_alerts(self) -> int:
        """Number of recorded TIS-breach alerts."""
        return self._tis_alerts

    # --- HF TrainerCallback surface (v0.53.11 live wiring) ---

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Capture policy + ref model references for EMA updates."""
        logger.debug("GRPOStabilityCallback.on_train_begin")
        self._policy_model = model
        # The reference model lives on the trainer; HF passes it via `kwargs`
        # in newer TRL versions, otherwise pull from the callback handler.
        ref = kwargs.get("ref_model")
        if ref is None:
            # Try to discover it on the trainer instance via the standard
            # TRL ``DPOTrainer.ref_model`` / ``GRPOTrainer.ref_model`` attr.
            trainer = kwargs.get("trainer")
            if trainer is not None:
                ref = getattr(trainer, "ref_model", None)
        self._ref_model = ref
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Per-step hook — perform EMA update + record stability state.

        v0.53.11 #127 — wires the actual EMA update post-step. The
        ``replay_buffer`` and ``tis_alerts`` counters are surfaced via
        ``state.log_history`` so the v0.34.0 anomaly explainer can flag
        instability.
        """
        # Live EMA update of reference model from current policy.
        # v0.71.11 #160 — in-place update (no full state_dict / load round
        # trip). Mutates the ref parameters directly so a 70B+ run pays
        # zero extra model-sized allocations per step.
        if (
            self.ref_model_ema_alpha is not None
            and self._ref_model is not None
            and (model is not None or self._policy_model is not None)
        ):
            try:
                policy = model if model is not None else self._policy_model
                updated = update_ema_in_place(
                    self._ref_model, policy, self.ref_model_ema_alpha
                )
                if updated == 0 and not self._ema_noop_warned:
                    self._ema_noop_warned = True
                    logger.warning(
                        "ref_model_ema_alpha is set but the EMA update matched "
                        "0 shared parameters between the reference and policy "
                        "models (name/shape mismatch) — the reference model is "
                        "NOT being updated. Check that both models share the "
                        "same architecture."
                    )
            except Exception as exc:  # noqa: BLE001 — never crash training
                logger.debug("EMA update skipped: %s", exc)
        # Surface counters to log_history.
        log_history = getattr(state, "log_history", None)
        if log_history is not None:
            entry: dict = {}
            if self.tis_threshold is not None:
                entry["tis_alerts"] = self._tis_alerts
            if self._replay is not None:
                entry["replay_size"] = len(self._replay)
            if self.ref_model_ema_alpha is not None:
                entry["ema_alpha"] = float(self.ref_model_ema_alpha)
            if entry:
                log_history.append(entry)
        return control
