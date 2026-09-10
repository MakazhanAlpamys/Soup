"""Regression tests for issue #724.

`training.loraplus_lr_ratio` was inserted into `training_kwargs` and forwarded to
`TrainingArguments(**training_kwargs)` by the SFT, pretrain and embedding wrappers.
`loraplus_lr_ratio` is not a `TrainingArguments` field, so enabling the advertised,
schema-accepted option raised `TypeError` before the first training step.

The fix routes it through PEFT's optimizer construction instead:
`attach_loraplus_optimizer` builds a `create_loraplus_optimizer` optimizer (B
matrices at `lr * ratio`, A at `lr`) and assigns it to `trainer.optimizer` after
the trainer exists. These tests use a real PEFT model and a real
`transformers.Trainer` — no mocks — because a mock would auto-create the LoRA
parameter groups the production path depends on and hide the very defect this fixes.
"""

import pytest

from soup_cli.utils.peft_wiring import attach_loraplus_optimizer

pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

import torch  # noqa: E402 — after importorskip, for the version guard below

BASE_LR = 2e-5
RATIO = 16.0

# transformers refuses torch.load for optimizer/scheduler checkpoints below
# torch 2.6 (CVE-2025-32434) — the same floor gap as #651. This project
# declares torch>=2.5.0, so an install at that floor must skip the resume
# test below rather than fail red on a version restriction it can't control.
_TORCH_VERSION = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2])


def _tiny_peft_model(seed: int = 0):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    # Seeded so two separate calls can build bit-identical base + adapter
    # weights - needed to compare a resumed run against an uninterrupted one.
    torch.manual_seed(seed)
    cfg = AutoConfig.for_model(
        "llama", hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, vocab_size=128,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    return get_peft_model(
        model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
    )


def _tiny_dataset(n=16, seq_len=6, vocab_size=128, seed=123):
    import torch

    g = torch.Generator().manual_seed(seed)
    return [
        {
            "input_ids": torch.randint(0, vocab_size, (seq_len,), generator=g),
            "attention_mask": torch.ones(seq_len, dtype=torch.long),
            "labels": torch.randint(0, vocab_size, (seq_len,), generator=g),
        }
        for _ in range(n)
    ]


def _trainer_with_data(model, tmp_path, dataset, *, max_steps, save_steps):
    from transformers import Trainer, TrainingArguments

    args = TrainingArguments(
        output_dir=str(tmp_path), learning_rate=BASE_LR, optim="adamw_torch",
        max_steps=max_steps, save_steps=save_steps, save_strategy="steps",
        per_device_train_batch_size=4, report_to=[], logging_steps=1000,
        disable_tqdm=True, seed=42, data_seed=42,
    )
    return Trainer(model=model, args=args, train_dataset=dataset)


def _exp_avg_by_name(trainer):
    return {
        name: trainer.optimizer.state[p]["exp_avg"].clone()
        for name, p in trainer.model.named_parameters()
        if p in trainer.optimizer.state and "exp_avg" in trainer.optimizer.state[p]
    }


def _trainer(model, tmp_path, *, weight_decay=0.01, optim="adamw_torch"):
    from transformers import Trainer, TrainingArguments

    args = TrainingArguments(
        output_dir=str(tmp_path), learning_rate=BASE_LR,
        weight_decay=weight_decay, optim=optim, report_to=[],
    )
    return Trainer(model=model, args=args)


class _TCfg:
    """A real config-shaped object (not a mock): missing attributes raise."""
    def __init__(self, loraplus_lr_ratio=None, use_galore=False):
        self.loraplus_lr_ratio = loraplus_lr_ratio
        self.use_galore = use_galore


def test_loraplus_optimizer_is_attached_with_split_learning_rates(tmp_path):
    trainer = _trainer(_tiny_peft_model(), tmp_path)
    attached = attach_loraplus_optimizer(trainer, _TCfg(loraplus_lr_ratio=RATIO))

    assert attached is True
    assert trainer.optimizer is not None
    lrs = {round(g["lr"], 12) for g in trainer.optimizer.param_groups}
    # A/base group at lr, B group at lr * ratio — the whole point of LoRA+.
    assert BASE_LR in lrs
    assert round(BASE_LR * RATIO, 12) in lrs


def test_uses_the_configured_optimizer_class(tmp_path):
    trainer = _trainer(_tiny_peft_model(), tmp_path, optim="adamw_torch")
    attach_loraplus_optimizer(trainer, _TCfg(loraplus_lr_ratio=RATIO))
    assert type(trainer.optimizer).__name__ == "AdamW"


def test_weight_decay_is_applied_through_loraplus(tmp_path):
    # PEFT applies wd via `loraplus_weight_decay`, not the plain kwarg; the helper
    # must pass the configured value so decay groups actually receive it.
    trainer = _trainer(_tiny_peft_model(), tmp_path, weight_decay=0.07)
    attach_loraplus_optimizer(trainer, _TCfg(loraplus_lr_ratio=RATIO))
    assert any(g["weight_decay"] == 0.07 for g in trainer.optimizer.param_groups)


def test_no_ratio_is_a_noop(tmp_path):
    trainer = _trainer(_tiny_peft_model(), tmp_path)
    attached = attach_loraplus_optimizer(trainer, _TCfg(loraplus_lr_ratio=None))
    assert attached is False
    # Untouched: Trainer builds its own optimizer lazily at train() time.
    assert trainer.optimizer is None


def test_galore_conflict_raises(tmp_path):
    trainer = _trainer(_tiny_peft_model(), tmp_path)
    with pytest.raises(ValueError, match="use_galore"):
        attach_loraplus_optimizer(trainer, _TCfg(loraplus_lr_ratio=RATIO, use_galore=True))


def test_non_peft_model_raises(tmp_path):
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.for_model(
        "llama", hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, vocab_size=128,
    )
    plain = AutoModelForCausalLM.from_config(cfg)
    assert not isinstance(plain, PeftModel)
    trainer = _trainer(plain, tmp_path)
    with pytest.raises(ValueError, match="LoRA"):
        attach_loraplus_optimizer(trainer, _TCfg(loraplus_lr_ratio=RATIO))


def test_training_arguments_still_rejects_the_kwarg():
    # The contract behind the fix: loraplus_lr_ratio is NOT a TrainingArguments
    # field. If any wrapper re-adds the old forward, the run crashes here — this
    # is what made the option unusable before #724.
    from transformers import TrainingArguments

    with pytest.raises(TypeError):
        TrainingArguments(output_dir="x", loraplus_lr_ratio=RATIO)


@pytest.mark.skipif(
    _TORCH_VERSION < (2, 6),
    reason=(
        f"torch {torch.__version__} predates 2.6; transformers refuses "
        "torch.load for optimizer/scheduler checkpoints below that version "
        "(CVE-2025-32434), so a real resume cycle cannot run here at all. "
        "Resume is untested at this floor, not proven broken."
    ),
)
class TestResumePreservesOptimizerAndSchedulerState:
    """#724's last acceptance criterion, never previously exercised: does a
    LoRA+ optimizer built by attach_loraplus_optimizer actually survive HF's
    save/resume cycle, rather than just getting attached once and never
    proven to reload correctly?

    Runs a real 4-step training loop twice against the same seeded base
    model and dataset, both targeting the same `max_steps` (matching how
    `soup train --resume` is actually used - same config, continuing after
    an interruption): once straight through, once stopped at the real
    checkpoint saved at step 2 and resumed from it. `attach_loraplus_optimizer`
    runs before `trainer.train()` in every wrapper (#724), which is what lets
    `Trainer.create_optimizer` skip building a default optimizer and instead
    build the scheduler around the LoRA+ one - but that ordering alone
    doesn't prove `Trainer._load_optimizer_and_scheduler` correctly
    restores ITS optimizer momentum and the scheduler's decayed LR onto a
    freshly-built LoRA+ optimizer with matching param groups. If it silently
    dropped or misaligned that state, the resumed run would diverge from the
    uninterrupted one - exactly the kind of silent retuning #724 already
    fixed for the initial construction, just at the resume boundary instead.

    A run resumed with a DIFFERENT `max_steps` than it was checkpointed
    under legitimately produces a different LR curve after resume (the
    schedule's own decay depends on the target step count) - that is a
    general `transformers.Trainer` characteristic, not something specific to
    LoRA+, so it is deliberately not what this test compares.
    """

    def _train_to_step_4(self, out_dir, dataset, *, resume_from=None):
        model = _tiny_peft_model(seed=7)
        trainer = _trainer_with_data(model, out_dir, dataset, max_steps=4, save_steps=2)
        attach_loraplus_optimizer(trainer, _TCfg(loraplus_lr_ratio=RATIO))
        trainer.train(resume_from_checkpoint=resume_from)
        return trainer

    def test_resumed_run_matches_the_uninterrupted_run(self, tmp_path):
        import torch

        dataset = _tiny_dataset()
        out_dir = tmp_path / "run"

        reference = self._train_to_step_4(out_dir, dataset)
        checkpoint = out_dir / "checkpoint-2"
        assert checkpoint.is_dir()

        resumed = self._train_to_step_4(out_dir, dataset, resume_from=str(checkpoint))

        ref_state = _exp_avg_by_name(reference)
        resumed_state = _exp_avg_by_name(resumed)
        assert ref_state and set(ref_state) == set(resumed_state)
        for name in ref_state:
            assert torch.allclose(ref_state[name], resumed_state[name], atol=1e-6), (
                f"optimizer momentum for {name} did not survive resume"
            )

        assert reference.lr_scheduler.get_last_lr() == resumed.lr_scheduler.get_last_lr()

        compared_any = False
        for (name, p_ref), (_, p_resumed) in zip(
            reference.model.named_parameters(), resumed.model.named_parameters()
        ):
            if "lora_" not in name:
                continue
            compared_any = True
            assert torch.allclose(p_ref, p_resumed, atol=1e-6), (
                f"final weight for {name} diverged after resume"
            )
        assert compared_any
