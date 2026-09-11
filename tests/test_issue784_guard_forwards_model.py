"""Issue #784 — the empty-param-group guard must forward ``create_optimizer``'s argument.

``transformers.Trainer.create_optimizer(self, model=None)`` is called both with no
argument and, on the delayed-optimizer-creation path, positionally as
``self.create_optimizer(model)``. The guard used to replace it with a function that
took no arguments, so the positional call raised ``TypeError``.

Not reachable today (the guard is attached only for DeepSpeed, and delayed creation
is set for FSDP / SageMaker MP), but the reachability argument lives outside this
function, so the guard has to accept every calling form on its own.
"""

from __future__ import annotations

import inspect

from soup_cli.utils.deepspeed import attach_empty_param_group_guard


class _FakeOptimizer:
    def __init__(self, param_groups):
        self.param_groups = param_groups


def _lora_like_groups():
    """Decay group populated, no-decay group empty, as LoRA leaves them."""
    return [
        {"params": ["lora_A", "lora_B"], "weight_decay": 0.01},
        {"params": [], "weight_decay": 0.0},
    ]


class _HFStyleTrainer:
    """Mirrors ``transformers.Trainer.create_optimizer(self, model=None)``."""

    def __init__(self):
        self.received = []

    def create_optimizer(self, model=None):
        self.received.append(model)
        return _FakeOptimizer(_lora_like_groups())


class _NoArgTrainer:
    """A trainer whose ``create_optimizer`` takes no argument at all."""

    def __init__(self):
        self.calls = 0

    def create_optimizer(self):
        self.calls += 1
        return _FakeOptimizer(_lora_like_groups())


def test_positional_call_forwards_the_model_and_still_prunes():
    trainer = _HFStyleTrainer()
    model = object()
    attach_empty_param_group_guard(trainer)

    optimizer = trainer.create_optimizer(model)

    # the model reaches the original, rather than being dropped on the delayed path
    assert trainer.received == [model]
    assert len(optimizer.param_groups) == 1


def test_keyword_call_forwards_the_model_and_still_prunes():
    trainer = _HFStyleTrainer()
    model = object()
    attach_empty_param_group_guard(trainer)

    optimizer = trainer.create_optimizer(model=model)

    assert trainer.received == [model]
    assert len(optimizer.param_groups) == 1


def test_no_argument_call_is_passed_through_unchanged():
    """The guard adds no argument of its own, so a zero-arg original still works."""
    trainer = _NoArgTrainer()
    attach_empty_param_group_guard(trainer)

    optimizer = trainer.create_optimizer()

    assert trainer.calls == 1
    assert len(optimizer.param_groups) == 1


def test_hf_no_argument_call_keeps_its_default():
    trainer = _HFStyleTrainer()
    attach_empty_param_group_guard(trainer)

    trainer.create_optimizer()

    assert trainer.received == [None]


def test_wrapper_signature_accepts_the_positional_form():
    """Inspects the wrapper itself, so a rewrite back to ``def create_optimizer():``
    goes red even before anything calls it."""
    trainer = _HFStyleTrainer()
    attach_empty_param_group_guard(trainer)

    inspect.signature(trainer.create_optimizer).bind(object())
    inspect.signature(trainer.create_optimizer).bind(model=object())
