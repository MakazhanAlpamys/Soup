"""Issue #745 — PPO wires LoRA+ so the LR SCHEDULER binds to the LoRA+ optimizer.

Ten of the eleven trainers #745 wires are plain ``transformers.Trainer``
subclasses: their optimizer is created lazily at ``train()``, so a
post-construction ``attach_loraplus_optimizer`` lands before the scheduler is
built and the scheduler is built around the LoRA+ optimizer for free.

PPO is the exception. ``trl.experimental``'s ``PPOTrainer`` builds its optimizer
AND scheduler eagerly inside ``__init__`` (``create_optimizer_and_scheduler``)
and then calls ``accelerator.prepare(model, optimizer, dataloader)`` without the
scheduler. A LoRA+ optimizer assigned AFTER that is still stepped — the B group
does train at ``lr * ratio`` — but ``self.lr_scheduler`` was built against the
original optimizer and is never rebound, so warmup and decay never reach the B
group and ``get_last_lr()`` reports the wrong optimizer. That failure is
invisible to an optimizer-identity assertion, which is the whole reason this
file exists.

The fix (``trainer/ppo.py``) builds the LoRA+ optimizer up front with
``build_loraplus_optimizer`` and hands it to the constructor as
``optimizers=(opt, None)`` so ``create_optimizer_and_scheduler`` binds the
scheduler to it.

These tests exercise that mechanism at the ``transformers.Trainer`` level
PPOTrainer inherits it from — a real tiny PEFT model and a real ``Trainer``, no
mocks — rather than standing up PPO's reward and value models and a GPU rollout.
The eager order PPOTrainer performs in ``__init__`` is reproduced explicitly by
calling ``create_optimizer_and_scheduler`` before the attach in the negative
control, which is the exact ordering that makes the attach wrong here.
"""

import pytest

from soup_cli.utils.peft_wiring import (
    attach_loraplus_optimizer,
    build_loraplus_optimizer,
)

pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

BASE_LR = 2e-5
RATIO = 16.0
WARMUP_STEPS = 5
TRAIN_STEPS = 20

B_GROUP_LR = round(BASE_LR * RATIO, 12)


def _tiny_peft_model(seed: int = 0):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(seed)
    cfg = AutoConfig.for_model(
        "llama", hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, vocab_size=128,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    return get_peft_model(
        model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
    )


def _args(tmp_path):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(tmp_path), learning_rate=BASE_LR, optim="adamw_torch",
        warmup_steps=WARMUP_STEPS, lr_scheduler_type="linear",
        max_steps=TRAIN_STEPS, per_device_train_batch_size=4,
        report_to=[], disable_tqdm=True, logging_steps=1000,
    )


class _TCfg:
    """A real config-shaped object (not a mock): missing attributes raise."""

    def __init__(self, loraplus_lr_ratio=None, use_galore=False):
        self.loraplus_lr_ratio = loraplus_lr_ratio
        self.use_galore = use_galore


def _trainer_with_injected_optimizer(model, args, tcfg):
    """Reproduce PPO's constructor injection at the Trainer level.

    ``optimizers=(opt, None)`` is exactly what ``trainer/ppo.py`` passes to
    PPOTrainer; here it goes to the base ``Trainer`` PPOTrainer derives from,
    then the scheduler is built the way PPOTrainer's ``__init__`` builds it.
    """
    from transformers import Trainer

    optimizer = build_loraplus_optimizer(model, args, tcfg)
    trainer = Trainer(model=model, args=args, optimizers=(optimizer, None))
    trainer.create_optimizer_and_scheduler(num_training_steps=TRAIN_STEPS)
    return trainer, optimizer


def _group_lrs(scheduler):
    return [round(lr, 12) for lr in scheduler.get_last_lr()]


def test_injection_binds_the_scheduler_to_the_loraplus_optimizer(tmp_path):
    trainer, optimizer = _trainer_with_injected_optimizer(
        _tiny_peft_model(), _args(tmp_path), _TCfg(loraplus_lr_ratio=RATIO)
    )

    # The optimizer that actually runs is the LoRA+ one.
    assert trainer.optimizer is optimizer

    # And, the point of #745: the scheduler is bound to THAT optimizer, not a
    # discarded default. This is what a post-construction attach cannot give on
    # PPO's eager-construction shape.
    assert trainer.lr_scheduler.optimizer is optimizer

    # The scheduler captured the LoRA+ split as its per-group base LRs (the live
    # param_groups["lr"] are already 0 here — warmup step 0 — which is exactly
    # why the binding, not the live lr, is what has to be asserted).
    base_lrs = {round(lr, 12) for lr in trainer.lr_scheduler.base_lrs}
    assert BASE_LR in base_lrs
    assert B_GROUP_LR in base_lrs


@pytest.mark.filterwarnings(
    "ignore:Detected call of `lr_scheduler.step\\(\\)`:UserWarning"
)
def test_warmup_reaches_the_loraplus_b_group(tmp_path):
    """Optimizer identity is not enough — a scheduler bound to the right
    optimizer must actually drive the B group's LR. Linear warmup starts every
    group at zero and reaches each group's base LR at the end of warmup, so the
    B group climbs from 0 to lr*ratio through the bound scheduler."""
    trainer, _ = _trainer_with_injected_optimizer(
        _tiny_peft_model(), _args(tmp_path), _TCfg(loraplus_lr_ratio=RATIO)
    )
    scheduler = trainer.lr_scheduler

    # Start of warmup: every group is at (or effectively at) zero.
    assert max(_group_lrs(scheduler)) == pytest.approx(0.0, abs=1e-12)

    for _ in range(WARMUP_STEPS):
        scheduler.step()

    warmed = _group_lrs(scheduler)
    # Warmup has reached both LoRA+ rates, so the B group is being scheduled at
    # lr*ratio rather than left flat.
    assert B_GROUP_LR in warmed
    assert round(BASE_LR, 12) in warmed
    assert max(warmed) == pytest.approx(B_GROUP_LR, rel=1e-9)


def test_post_construction_attach_on_the_eager_shape_leaves_the_scheduler_detached(
    tmp_path,
):
    """The negative control, i.e. the bug the injection avoids and the mutation
    the review named: on a trainer that built its scheduler eagerly, replacing
    ``trainer.optimizer`` after the fact (what attach_loraplus_optimizer does)
    trains the B group but leaves the scheduler bound to the discarded default
    optimizer. If constructor injection were swapped back to a post-attach here,
    the two ``is`` assertions below would flip and this test would fail."""
    from transformers import Trainer

    model = _tiny_peft_model()
    args = _args(tmp_path)

    # Eager order, as PPOTrainer.__init__ does it: optimizer + scheduler first.
    trainer = Trainer(model=model, args=args)
    trainer.create_optimizer_and_scheduler(num_training_steps=TRAIN_STEPS)
    eager_default_optimizer = trainer.optimizer

    # Then the post-construction attach — correct for the lazy trainers, wrong
    # here because the scheduler already exists.
    attached = attach_loraplus_optimizer(trainer, _TCfg(loraplus_lr_ratio=RATIO))
    assert attached is True

    # The LoRA+ optimizer is now the one that would be stepped...
    assert trainer.optimizer is not eager_default_optimizer
    b_group = {round(g["lr"], 12) for g in trainer.optimizer.param_groups}
    assert B_GROUP_LR in b_group  # B really would train at lr*ratio

    # ...but the scheduler is still driving the discarded default optimizer,
    # which has none of the LoRA+ split. That desync is exactly what #745 fixes.
    assert trainer.lr_scheduler.optimizer is eager_default_optimizer
    assert trainer.lr_scheduler.optimizer is not trainer.optimizer
    scheduled_lrs = {round(lr, 12) for lr in trainer.lr_scheduler.get_last_lr()}
    assert B_GROUP_LR not in scheduled_lrs


def test_build_loraplus_optimizer_returns_none_without_ratio(tmp_path):
    """The shared core is a no-op when LoRA+ is off, so PPO passes optimizers
    only when a ratio is set and otherwise leaves trl to build its own."""
    optimizer = build_loraplus_optimizer(
        _tiny_peft_model(), _args(tmp_path), _TCfg(loraplus_lr_ratio=None)
    )
    assert optimizer is None
