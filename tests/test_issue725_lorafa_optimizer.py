"""Regression tests for issue #725.

PEFT exposes LoRA-FA (Frozen-A LoRA, arXiv:2308.03303), an optimizer that
freezes LoRA A matrices and trains only B matrices. Freezing A avoids retaining
the input activations needed for backpropagating through A, reducing
adapter-rank activation memory retention substantially.

The fix routes it through PEFT''s optimizer construction:
`attach_lorafa_optimizer` builds a `create_lorafa_optimizer` optimizer and assigns
it to `trainer.optimizer` after the trainer exists.

These tests use a real PEFT model and a real `transformers.Trainer` -- no mocks --
because a mock would auto-create parameter groups and hide wiring defects.
"""

import pytest

from soup_cli.utils.peft_wiring import attach_lorafa_optimizer

pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

import torch  # noqa: E402 -- after importorskip

BASE_LR = 2e-5

_TORCH_VERSION = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2])


def _tiny_peft_model(seed: int = 0):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(seed)
    cfg = AutoConfig.for_model(
        "llama",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=128,
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
        output_dir=str(tmp_path),
        learning_rate=BASE_LR,
        optim="adamw_torch",
        max_steps=max_steps,
        save_steps=save_steps,
        save_strategy="steps",
        per_device_train_batch_size=4,
        report_to=[],
        logging_steps=1000,
        disable_tqdm=True,
        seed=42,
        data_seed=42,
    )
    return Trainer(model=model, args=args, train_dataset=dataset)


def _trainer(model, tmp_path, *, weight_decay=0.01, optim="adamw_torch"):
    from transformers import Trainer, TrainingArguments

    args = TrainingArguments(
        output_dir=str(tmp_path),
        learning_rate=BASE_LR,
        weight_decay=weight_decay,
        optim=optim,
        report_to=[],
    )
    return Trainer(model=model, args=args)


class _TCfg:
    """A real config-shaped object (not a mock): missing attributes raise."""

    def __init__(self, use_lorafa=False, loraplus_lr_ratio=None, use_galore=False):
        self.use_lorafa = use_lorafa
        self.loraplus_lr_ratio = loraplus_lr_ratio
        self.use_galore = use_galore


def test_lorafa_optimizer_is_attached_and_freezes_a_matrices(tmp_path):
    model = _tiny_peft_model()
    # Before optimizer attachment, LoRA A and B both have requires_grad=True
    a_params_before = [p for n, p in model.named_parameters() if "lora_A" in n]
    b_params_before = [p for n, p in model.named_parameters() if "lora_B" in n]
    assert a_params_before and all(p.requires_grad for p in a_params_before)
    assert b_params_before and all(p.requires_grad for p in b_params_before)

    trainer = _trainer(model, tmp_path)
    attached = attach_lorafa_optimizer(trainer, _TCfg(use_lorafa=True))

    assert attached is True
    assert trainer.optimizer is not None
    assert type(trainer.optimizer).__name__ == "LoraFAOptimizer"

    # Criterion 2: Assertion that A is actually frozen and B actually trains
    a_params = [p for n, p in model.named_parameters() if "lora_A" in n]
    b_params = [p for n, p in model.named_parameters() if "lora_B" in n]
    assert a_params and all(not p.requires_grad for p in a_params), (
        "LoRA-FA failed to freeze lora_A parameters"
    )
    assert b_params and all(p.requires_grad for p in b_params), (
        "LoRA-FA unexpectedly froze lora_B parameters"
    )


def test_no_lorafa_is_a_noop(tmp_path):
    trainer = _trainer(_tiny_peft_model(), tmp_path)
    attached = attach_lorafa_optimizer(trainer, _TCfg(use_lorafa=False))
    assert attached is False
    assert trainer.optimizer is None


def test_weight_decay_and_learning_rate_applied(tmp_path):
    trainer = _trainer(_tiny_peft_model(), tmp_path, weight_decay=0.05)
    attach_lorafa_optimizer(trainer, _TCfg(use_lorafa=True))
    assert any(g["weight_decay"] == 0.05 for g in trainer.optimizer.param_groups)
    assert any(g["lr"] == BASE_LR for g in trainer.optimizer.param_groups)


def test_galore_conflict_raises(tmp_path):
    trainer = _trainer(_tiny_peft_model(), tmp_path)
    with pytest.raises(ValueError, match="use_galore"):
        attach_lorafa_optimizer(trainer, _TCfg(use_lorafa=True, use_galore=True))


def test_loraplus_conflict_raises(tmp_path):
    trainer = _trainer(_tiny_peft_model(), tmp_path)
    with pytest.raises(ValueError, match="loraplus_lr_ratio"):
        attach_lorafa_optimizer(trainer, _TCfg(use_lorafa=True, loraplus_lr_ratio=16.0))


def test_non_peft_model_raises(tmp_path):
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.for_model(
        "llama",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=128,
    )
    plain = AutoModelForCausalLM.from_config(cfg)
    assert not isinstance(plain, PeftModel)
    trainer = _trainer(plain, tmp_path)
    with pytest.raises(ValueError, match="LoRA"):
        attach_lorafa_optimizer(trainer, _TCfg(use_lorafa=True))


def test_training_config_schema_mutual_exclusion():
    from pydantic import ValidationError

    from soup_cli.config.schema import TrainingConfig

    # Parsing both use_lorafa and loraplus_lr_ratio must fail at parse time
    with pytest.raises(ValidationError, match="mutually exclusive"):
        TrainingConfig(use_lorafa=True, loraplus_lr_ratio=16.0)

    with pytest.raises(ValidationError, match="mutually exclusive"):
        TrainingConfig(use_lorafa=True, use_galore=True)


@pytest.mark.skipif(
    _TORCH_VERSION < (2, 6),
    reason="torch predates 2.6; transformers refuses torch.load below 2.6 (CVE-2025-32434)",
)
class TestResumePreservesOptimizerAndSchedulerState:
    def _train_to_step_4(self, out_dir, dataset, *, resume_from=None):
        model = _tiny_peft_model(seed=7)
        trainer = _trainer_with_data(model, out_dir, dataset, max_steps=4, save_steps=2)
        attach_lorafa_optimizer(trainer, _TCfg(use_lorafa=True))
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

        assert reference.lr_scheduler.get_last_lr() == resumed.lr_scheduler.get_last_lr()

        compared_any = False
        for (name, p_ref), (_, p_resumed) in zip(
            reference.model.named_parameters(), resumed.model.named_parameters()
        ):
            if "lora_B" not in name:
                continue
            compared_any = True
            assert torch.allclose(p_ref, p_resumed, atol=1e-6), (
                f"final weight for {name} diverged after resume"
            )
        assert compared_any
