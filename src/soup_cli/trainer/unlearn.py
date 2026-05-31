"""v0.61.0 Part A — Unlearning trainer wrapper (stub).

The wrapper validates the config, captures the configured method, and
exposes the standard ``setup()`` / ``train()`` surface. Both methods
raise ``NotImplementedError`` with an explicit ``v0.61.1`` marker —
mirrors the v0.50.0 GRPO Plus / v0.52.0 Modality II / v0.53.0 Quant
Menu II stub-then-live pattern.

Once live wiring lands in v0.61.1 the wrapper will compose three
backends:

* ``npo`` — load reference model + compute DPO-shaped negative-only
  loss over the forget set with a weighted retain-set CE term.
* ``simnpo`` — drop the reference model + length-normalise the log
  ratios per SimPO.
* ``rmu`` — install forward hooks on the residual stream and inject a
  noise vector for forget-set inputs while preserving retain-set
  activations.

Schema-only this release: ``UnlearnTrainerWrapper(cfg).setup()`` raises
loudly so misconfigured ``task='unlearn'`` runs fail fast instead of
silently producing a no-op checkpoint.
"""

from __future__ import annotations

from typing import Any


class UnlearnTrainerWrapper:
    """Stub trainer for ``task='unlearn'`` — live wiring in v0.61.1.

    Validates the config + captures the method name so the schema is
    locked in this release. Callers should NOT rely on ``setup()`` /
    ``train()`` returning normally — both raise
    ``NotImplementedError`` until v0.61.1 lands the NPO / SimNPO / RMU
    backends.
    """

    def __init__(self, config: Any, **kwargs: Any) -> None:
        # Defer the schema-import to construction-time so callers
        # building the wrapper from a SoupConfig instance (rather than
        # a raw dict) get a meaningful AttributeError before the
        # deferred-live raise.
        try:
            self.config = config
            self.method = config.training.unlearn_method
        except AttributeError as exc:
            raise AttributeError(
                "UnlearnTrainerWrapper requires a SoupConfig with "
                f"training.unlearn_method set; got {exc}"
            ) from exc

        # Capture forward-compat kwargs (device / trust_remote_code /
        # report_to / etc.) without erroring — the live wrapper will
        # consume them. This matches the v0.50.0 ``launch_rollout`` +
        # v0.53.2 ``build_classifier_trainer`` policy of accepting
        # forward-compat kwargs so v0.61.1 wiring is purely additive.
        self._kwargs = dict(kwargs)
        self._setup_called = False
        self._trainer: Any = None

    def setup(self) -> None:
        """Build the inner HF Trainer + frozen ref model + dataset.

        Deferred to v0.61.1. The error message names the configured
        method so operators see WHICH backend is gated.
        """
        raise NotImplementedError(
            f"UnlearnTrainerWrapper.setup() for method={self.method!r} "
            f"is deferred to v0.61.1. Schema accepts the method now so "
            f"YAML written today will work the moment v0.61.1 ships."
        )

    def train(self) -> Any:
        """Run the unlearn training loop. Requires ``setup()`` first.

        The legacy contract (matches every other trainer wrapper) is to
        raise ``RuntimeError`` if the trainer was not set up — even if
        the eventual ``setup()`` is itself deferred. This lets callers
        distinguish between "you forgot to call setup" and "method is
        not yet wired".
        """
        if not self._setup_called:
            raise RuntimeError(
                "UnlearnTrainerWrapper.train() called before setup(); "
                "call setup() first (note: v0.61.0 setup raises "
                "NotImplementedError — live wiring lands in v0.61.1)."
            )
        return self._trainer
