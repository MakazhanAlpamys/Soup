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
        trust_remote_code: bool = False,
    ):
        self.config = config
        self.device = device
        self.report_to = report_to
        self.deepspeed_config = deepspeed_config
        self.fsdp_config = fsdp_config
        self.trust_remote_code = trust_remote_code
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
            "trust_remote_code": self.trust_remote_code,
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

    def _build_multi_objective(self):
        """v0.53.11 #68 — build the multi-objective primary trainer + combine hook.

        Lifts the v0.40.1 "primary-loss approximation": the highest-weighted
        loss still becomes the primary inner trainer (because each TRL
        preference trainer owns its data collation), but a
        :func:`combine_losses` wrapper now runs against the policy log-probs
        in the inner trainer's ``compute_loss`` so each named loss
        contributes to the backward pass per its weight.

        BCO mixed with paired losses remains rejected at runtime (data-format
        incompatible — paired needs ``prompt+chosen+rejected``, BCO needs
        ``prompt+completion+label``).

        The math kernel ships in :mod:`soup_cli.utils.preference_combine`
        and is exercised by ``test_v05311.py``. Live wrapper installation
        on the inner Trainer's ``compute_loss`` is attached via
        :func:`attach_weighted_preference_combine` after the inner setup
        completes (called from :meth:`setup`).
        """
        from rich.console import Console as _Console

        from soup_cli.utils.preference_combine import (
            describe_blend,
            validate_weight_compat,
        )

        weights = get_loss_weights(self.config) or {}
        validate_weight_compat(weights)
        primary = max(weights, key=weights.get)
        # Build the primary inner wrapper using the same dispatch table.
        inner_cfg = _make_inner_cfg(self.config, primary)
        kwargs = {
            "device": self.device,
            "report_to": self.report_to,
            "deepspeed_config": self.deepspeed_config,
            "fsdp_config": self.fsdp_config,
            "trust_remote_code": self.trust_remote_code,
        }
        if primary == "dpo":
            from soup_cli.trainer.dpo import DPOTrainerWrapper as PrimaryWrapper
        elif primary == "simpo":
            from soup_cli.trainer.simpo import SimPOTrainerWrapper as PrimaryWrapper
        elif primary == "orpo":
            from soup_cli.trainer.orpo import ORPOTrainerWrapper as PrimaryWrapper
        elif primary == "ipo":
            from soup_cli.trainer.ipo import IPOTrainerWrapper as PrimaryWrapper
        elif primary == "bco":
            from soup_cli.trainer.bco import BCOTrainerWrapper as PrimaryWrapper
        else:  # defensive — schema enforces the allowlist
            raise ValueError(f"unknown primary preference loss: {primary!r}")
        _Console().print(
            f"[cyan]Multi-objective preference loss:[/] {describe_blend(weights)} "
            f"(primary: {primary})"
        )
        self._active_weights = dict(weights)
        return PrimaryWrapper(inner_cfg, **kwargs)

    def setup(self, dataset: dict) -> None:
        # v0.40.1 Part B — multi-objective live runtime (replaces v0.40.0
        # Part D NotImplementedError stub). The combiner shares a single
        # forward pass across the active losses; BCO mixed with paired
        # losses is rejected at runtime (data-format incompatible).
        if is_multi_objective_preference(self.config):
            # v0.40.1 review fix — fail-fast compatibility check (BCO mixed
            # with paired losses) BEFORE building the heavy primary trainer.
            # Advisory print + build are owned by ``_build_multi_objective``
            # so successive setup() calls (e.g. resume) emit one consistent
            # advisory line, not two.
            from soup_cli.utils.preference_combine import validate_weight_compat

            weights = get_loss_weights(self.config) or {}
            validate_weight_compat(weights)
            if self._inner is None:
                self._inner = self._build_multi_objective()
            if self._inner is not None:
                self._inner.setup(dataset)
                # v0.53.11 #68 — attach the weighted-combine compute_loss
                # wrapper after the inner trainer's setup completes so the
                # TRL trainer instance is available.
                from soup_cli.utils.preference_combine import (
                    attach_weighted_preference_combine,
                )

                inner_trainer = getattr(self._inner, "trainer", None)
                if inner_trainer is not None:
                    try:
                        attach_weighted_preference_combine(inner_trainer, weights)
                    except (TypeError, ValueError) as exc:
                        # Schema validates weights at config load; runtime
                        # rejection should be loud — but fall through to the
                        # primary-loss-only path rather than crashing the run.
                        import logging

                        logging.getLogger(__name__).warning(
                            "attach_weighted_preference_combine skipped: %s", exc
                        )
            return
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
