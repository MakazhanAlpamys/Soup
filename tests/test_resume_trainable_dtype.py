"""A checkpoint resume must not change the dtype of the trainable adapter.

TRL casts a quantized base's LoRA adapter to bf16 at init; the resume path reloads
it through ``load_adapter`` (``autocast_adapter_dtype=True``), which upcasts it to
fp32. ``keep_trainable_dtype_on_resume`` restores the pre-resume dtypes.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from soup_cli.utils.mixed_precision import keep_trainable_dtype_on_resume


def _model():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    model[0].weight.requires_grad_(False)
    model[0].bias.requires_grad_(False)
    model[1].to(torch.bfloat16)
    return model


def _trainer(model, calls):
    def load(resume_from_checkpoint, model=None):
        calls.append(resume_from_checkpoint)
        target = model if model is not None else trainer.model
        for param in target.parameters():
            param.data = param.data.to(torch.float32)   # what load_adapter's autocast does

    trainer = SimpleNamespace(model=model, _load_from_checkpoint=load)
    return trainer


def test_trainable_dtype_restored_after_load():
    calls = []
    trainer = _trainer(_model(), calls)
    keep_trainable_dtype_on_resume(trainer)
    trainer._load_from_checkpoint("ckpt")
    assert calls == ["ckpt"]
    assert trainer.model[1].weight.dtype == torch.bfloat16
    assert trainer.model[1].bias.dtype == torch.bfloat16


def test_frozen_params_left_to_the_loader():
    trainer = _trainer(_model(), [])
    keep_trainable_dtype_on_resume(trainer)
    trainer._load_from_checkpoint("ckpt")
    assert trainer.model[0].weight.dtype == torch.float32


def test_explicit_model_argument_is_restored():
    trainer = _trainer(_model(), [])
    keep_trainable_dtype_on_resume(trainer)
    other = trainer.model
    trainer._load_from_checkpoint("ckpt", other)
    assert other[1].weight.dtype == torch.bfloat16


def test_no_model_is_a_no_op():
    trainer = SimpleNamespace(model=None, _load_from_checkpoint=lambda *a, **k: None)
    keep_trainable_dtype_on_resume(trainer)
    trainer._load_from_checkpoint("ckpt")


@pytest.mark.gpu
def test_qlora_sft_resume_keeps_bf16_adapter(tmp_path):
    """End to end on TRL: 2 steps, resume from checkpoint-2, 2 more steps."""
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    class Stop(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step >= 2:
                control.should_training_stop = True

    rows = [
        {"prompt": f"{a} + {b} =", "completion": f" {a + b}"} for a in range(4) for b in range(4)
    ]

    def build(callbacks):
        model = AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            ),
            dtype=torch.bfloat16,
        )
        args = SFTConfig(
            output_dir=str(tmp_path), max_steps=4, per_device_train_batch_size=4,
            save_steps=2, bf16=True, report_to=[], seed=0,
        )
        return SFTTrainer(
            model=model, args=args, train_dataset=Dataset.from_list(rows),
            peft_config=LoraConfig(r=4, target_modules=["q_proj", "v_proj"]),
            callbacks=callbacks,
        )

    build([Stop()]).train()
    resumed = build([])
    keep_trainable_dtype_on_resume(resumed)
    resumed.train(resume_from_checkpoint=str(tmp_path / "checkpoint-2"))
    dtypes = {p.dtype for p in resumed.model.parameters() if p.requires_grad}
    assert dtypes == {torch.bfloat16}
