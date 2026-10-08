"""A checkpoint resume must not change the dtype of the trainable adapter.

TRL casts a quantized base's LoRA adapter to bf16 at init; the resume path reloads
it through ``load_adapter`` (``autocast_adapter_dtype=True``), which upcasts it to
fp32. ``keep_trainable_dtype_on_resume`` restores the pre-resume dtypes.
"""
from __future__ import annotations

import contextlib
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


class _FakeTrainer:
    """Mirrors transformers.Trainer: train() reloads through _load_from_checkpoint."""

    def __init__(self, model):
        self.model = model
        self.args = SimpleNamespace(fp16=False, bf16=True, should_save=True)
        self.state = SimpleNamespace(log_history=[], global_step=4)
        self.dtypes_in_train = None

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        for param in self.model.parameters():  # what load_adapter's autocast does
            param.data = param.data.to(torch.float32)

    def train(self, *, resume_from_checkpoint):
        if resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint)
        self.dtypes_in_train = {p.dtype for p in self.model.parameters() if p.requires_grad}

    def save_model(self, output):
        pass

    def save_state(self):
        pass


def _wrapper(monkeypatch, tmp_path, trainer):
    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils import ebft_gdpo, peft_wiring

    for name in (
        "attach_loraplus_optimizer",
        "attach_lorafa_optimizer",
        "attach_relora_callback",
        "attach_lisa_callback",
        "attach_curriculum_callback",
        "attach_plugin_callback",
    ):
        monkeypatch.setattr(peft_wiring, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(ebft_gdpo, "attach_ebft_compute_loss", lambda *args: None)

    wrapper = object.__new__(SFTTrainerWrapper)
    wrapper.config = SoupConfig(
        base="org/model", task="sft", data={"train": "data.jsonl"}, training={}
    )
    wrapper._quest_metadata = None
    wrapper._output_dir = str(tmp_path)
    wrapper.trainer = trainer
    wrapper.tokenizer = SimpleNamespace(save_pretrained=lambda output: None)
    wrapper._training_context = lambda context: contextlib.ExitStack()
    wrapper._report_rewind = lambda: None
    return wrapper


def test_sft_train_keeps_the_adapter_dtype_across_a_resume(monkeypatch, tmp_path):
    trainer = _FakeTrainer(_model())
    _wrapper(monkeypatch, tmp_path, trainer).train(resume_from_checkpoint="ckpt")
    assert trainer.dtypes_in_train == {torch.bfloat16}


def test_sft_train_without_resume_leaves_the_loader_alone(monkeypatch, tmp_path):
    trainer = _FakeTrainer(_model())
    _wrapper(monkeypatch, tmp_path, trainer).train(resume_from_checkpoint=None)
    assert "_load_from_checkpoint" not in vars(trainer)
    assert trainer.dtypes_in_train == {torch.bfloat16}


def _dpo_wrapper(tmp_path, trainer):
    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.dpo import DPOTrainerWrapper

    wrapper = object.__new__(DPOTrainerWrapper)
    wrapper.config = SoupConfig(
        base="org/model",
        task="preference",
        data={"train": "data.jsonl", "format": "dpo"},
        training={"preference_loss": "dpo"},
    )
    wrapper._output_dir = str(tmp_path)
    wrapper.trainer = trainer
    wrapper.tokenizer = SimpleNamespace(save_pretrained=lambda output: None)
    wrapper._attach_streamed_save_guard = lambda: None
    wrapper._training_context = lambda context: contextlib.ExitStack()
    wrapper._assert_streamed_adapter_saved = lambda output_dir: None
    return wrapper


def _ipo_wrapper(tmp_path, trainer):
    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.ipo import IPOTrainerWrapper

    wrapper = object.__new__(IPOTrainerWrapper)
    wrapper.config = SoupConfig(
        base="org/model",
        task="preference",
        data={"train": "data.jsonl", "format": "dpo"},
        training={"preference_loss": "ipo"},
    )
    wrapper._output_dir = str(tmp_path)
    wrapper.trainer = trainer
    wrapper.tokenizer = SimpleNamespace(save_pretrained=lambda output: None)
    return wrapper


def _grpo_wrapper(tmp_path, trainer):
    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.grpo import GRPOTrainerWrapper

    wrapper = object.__new__(GRPOTrainerWrapper)
    wrapper.config = SoupConfig(
        base="org/model", task="grpo", data={"train": "data.jsonl"}, training={}
    )
    wrapper._output_dir = str(tmp_path)
    wrapper.trainer = trainer
    wrapper.tokenizer = SimpleNamespace(save_pretrained=lambda output: None)
    return wrapper


def _reward_wrapper(tmp_path, trainer):
    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.reward_model import RewardModelTrainerWrapper

    wrapper = object.__new__(RewardModelTrainerWrapper)
    wrapper.config = SoupConfig(
        base="org/model",
        task="reward_model",
        data={"train": "data.jsonl"},
        training={},
    )
    wrapper._output_dir = str(tmp_path)
    wrapper.trainer = trainer
    wrapper.tokenizer = SimpleNamespace(save_pretrained=lambda output: None)
    return wrapper


@pytest.mark.parametrize(
    "builder",
    [_dpo_wrapper, _ipo_wrapper, _grpo_wrapper, _reward_wrapper],
    ids=["dpo", "ipo", "grpo", "reward_model"],
)
def test_other_trl_trainers_keep_the_adapter_dtype_across_a_resume(builder, tmp_path):
    trainer = _FakeTrainer(_model())
    builder(tmp_path, trainer).train(resume_from_checkpoint="ckpt")
    assert trainer.dtypes_in_train == {torch.bfloat16}


@pytest.mark.parametrize(
    "builder",
    [_dpo_wrapper, _ipo_wrapper, _grpo_wrapper, _reward_wrapper],
    ids=["dpo", "ipo", "grpo", "reward_model"],
)
def test_other_trl_trainers_without_resume_leave_the_loader_alone(builder, tmp_path):
    trainer = _FakeTrainer(_model())
    builder(tmp_path, trainer).train(resume_from_checkpoint=None)
    assert "_load_from_checkpoint" not in vars(trainer)
    assert trainer.dtypes_in_train == {torch.bfloat16}


def test_trainable_dtype_restored_after_load():
    calls = []
    trainer = _trainer(_model(), calls)
    keep_trainable_dtype_on_resume(trainer)
    trainer._load_from_checkpoint("ckpt")
    assert calls == ["ckpt"]
    assert trainer.model[1].weight.dtype == torch.bfloat16
    assert trainer.model[1].bias.dtype == torch.bfloat16


def test_frozen_params_really_keep_the_loader_dtype():
    model = _model()
    model[0].to(torch.bfloat16)  # frozen starts bf16, the loader makes it fp32
    trainer = _FakeTrainer(model)
    keep_trainable_dtype_on_resume(trainer)
    trainer._load_from_checkpoint("ckpt")
    assert model[0].weight.dtype == torch.float32
    assert model[1].weight.dtype == torch.bfloat16


def test_explicit_model_argument_is_the_one_restored():
    trainer = _FakeTrainer(_model())
    other = _model()  # a different module with the same parameter names
    original = trainer._load_from_checkpoint

    def load(resume_from_checkpoint, model=None):
        original(resume_from_checkpoint)
        for param in model.parameters():
            param.data = param.data.to(torch.float32)

    trainer._load_from_checkpoint = load
    keep_trainable_dtype_on_resume(trainer)
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
