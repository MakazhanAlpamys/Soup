"""Unified preference loss dispatcher (v0.40.0 Part B).

Provides a single entry-point for preference-style training that routes to
the per-loss wrappers (DPO / SimPO / ORPO / IPO / BCO). The legacy
``task: dpo``, ``task: simpo``, ... config forms remain first-class —
this module is purely additive.

Usage:

  task: preference
  training:
    preference_loss: dpo  # or simpo, orpo, ipo, bco

The dispatcher constructs a temporary :class:`SoupConfig` view with the
matching legacy task so the underlying TRL trainer is happy, then forwards
``setup`` / ``train`` to the right wrapper.
"""

from __future__ import annotations

from typing import Optional

from soup_cli.config.schema import SoupConfig

_SUPPORTED_LOSSES: frozenset[str] = frozenset({"dpo", "simpo", "orpo", "ipo", "bco"})


def is_multi_objective_preference(cfg: SoupConfig) -> bool:
    """True when the config asks for a weighted blend of preference losses.

    v0.40.0 Part D — schema-level surface only; live runtime weighted
    combination is deferred to v0.40.1 (mirrors the project's
    stub-then-live pattern from v0.27.0 MII / v0.37.0 multipack /
    v0.38.0 quant menu / v0.39.0 ReLoRA).
    """
    weights = cfg.training.preference_loss_weights
    return weights is not None and len(weights) >= 1


def get_loss_weights(cfg: SoupConfig) -> Optional[dict]:
    """Return a defensive copy of ``training.preference_loss_weights``."""
    weights = cfg.training.preference_loss_weights
    if weights is None:
        return None
    return dict(weights)


def resolve_preference_loss(cfg: SoupConfig) -> Optional[str]:
    """Return the preference loss name for a config, or ``None`` if N/A.

    - ``task='preference'`` → ``training.preference_loss``.
    - Legacy ``task in {dpo, simpo, orpo, ipo, bco}`` → that name (identity).
    - Anything else → ``None``.
    """
    if cfg.task == "preference":
        return cfg.training.preference_loss
    if cfg.task in _SUPPORTED_LOSSES:
        return cfg.task
    return None


def _make_inner_cfg(cfg: SoupConfig, loss: str) -> SoupConfig:
    """Build a SoupConfig view with task=<loss> so the inner wrapper can run.

    Uses ``model_copy`` to round-trip without re-running validators on an
    intermediate inconsistent state. The caller's cfg is never mutated.
    """
    inner_training = cfg.training.model_copy(
        update={"preference_loss": None, "preference_loss_weights": None}
    )
    return cfg.model_copy(update={"task": loss, "training": inner_training})


class PreferenceTrainerWrapper:
    """Dispatcher wrapper for ``task: preference`` configs.

    Forwards to DPO / SimPO / ORPO / IPO / BCO wrappers based on
    ``training.preference_loss``.
    """

    def __init__(
        self,
        config: SoupConfig,
        device: str = "cuda",
        report_to: str = "none",
        deepspeed_config: Optional[str] = None,
        fsdp_config: Optional[dict] = None,
    ):
        self.config = config
        self.device = device
        self.report_to = report_to
        self.deepspeed_config = deepspeed_config
        self.fsdp_config = fsdp_config
        self._inner = None

    def _build_inner(self):
        loss = self.config.training.preference_loss
        if loss not in _SUPPORTED_LOSSES:
            raise ValueError(
                f"Unknown preference_loss={loss!r}. "
                f"Supported: {sorted(_SUPPORTED_LOSSES)}"
            )
        inner_cfg = _make_inner_cfg(self.config, loss)
        kwargs = {
            "device": self.device,
            "report_to": self.report_to,
            "deepspeed_config": self.deepspeed_config,
            "fsdp_config": self.fsdp_config,
        }
        if loss == "dpo":
            from soup_cli.trainer.dpo import DPOTrainerWrapper
            return DPOTrainerWrapper(inner_cfg, **kwargs)
        if loss == "simpo":
            from soup_cli.trainer.simpo import SimPOTrainerWrapper
            return SimPOTrainerWrapper(inner_cfg, **kwargs)
        if loss == "orpo":
            from soup_cli.trainer.orpo import ORPOTrainerWrapper
            return ORPOTrainerWrapper(inner_cfg, **kwargs)
        if loss == "ipo":
            from soup_cli.trainer.ipo import IPOTrainerWrapper
            return IPOTrainerWrapper(inner_cfg, **kwargs)
        # BCO — last branch by allowlist exhaustion.
        from soup_cli.trainer.bco import BCOTrainerWrapper
        return BCOTrainerWrapper(inner_cfg, **kwargs)

    def setup(self, dataset: dict) -> None:
        # v0.40.0 Part D — schema-level multi-objective shipped; live
        # weighted-loss combination deferred to v0.40.1 (TRL preference
        # trainers do not expose a clean compute_loss override hook;
        # subclassing each one is tracked separately).
        if is_multi_objective_preference(self.config):
            raise NotImplementedError(
                "preference_loss_weights (multi-objective preference loss) "
                "is config-level only in v0.40.0. Live runtime weighted "
                "combination is deferred to v0.40.1 (subclassing TRL "
                "preference trainers to override compute_loss). For now, "
                "use the scalar 'preference_loss' field instead."
            )
        if self._inner is None:
            self._inner = self._build_inner()
        self._inner.setup(dataset)

    def train(self, **kwargs) -> dict:
        if self._inner is None:
            raise RuntimeError(
                "PreferenceTrainerWrapper.train() called before setup(). "
                "Call setup(dataset) first."
            )
        return self._inner.train(**kwargs)

    @property
    def trainer(self):
        """Expose the underlying HF Trainer for callbacks (HF push, eval-gate)."""
        return getattr(self._inner, "trainer", None)
