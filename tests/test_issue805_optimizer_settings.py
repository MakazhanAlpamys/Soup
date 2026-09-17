"""Exercise #805's configuration values at real trainer/optimizer boundaries."""

from __future__ import annotations

import json
import math
from unittest.mock import patch

import pytest


@pytest.fixture
def tiny_model(tmp_path, monkeypatch):
    """An offline CPU model and tokenizer, including loadable task adapters."""
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    monkeypatch.chdir(tmp_path)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    # Transformers caches MPS availability across tests. Override the Trainer's
    # lookup too, so a prior MPS probe cannot move this CPU fixture's model.
    monkeypatch.setattr("transformers.training_args.is_torch_mps_available", lambda: False)
    torch.manual_seed(42)
    tokenizer = Tokenizer(WordLevel(
        {"<pad>": 0, "<unk>": 1, "<eos>": 2, "one": 3, "two": 4, "three": 5},
        unk_token="<unk>",
    ))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="<pad>", unk_token="<unk>", eos_token="<eos>"
    )
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=6, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
        pad_token_id=0, eos_token_id=2,
    ))
    path = tmp_path / "tiny-model"
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    yield path
    torch.set_num_threads(previous_threads)


def _config(tiny_model, task, **training):
    from soup_cli.config.schema import SoupConfig

    return SoupConfig(
        base=str(tiny_model), task=task, output="output-" + task,
        data={"train": "train.jsonl", "forget_set": "forget.jsonl", "max_length": 64},
        training={
            "epochs": 2, "batch_size": 2, "gradient_accumulation_steps": 2,
            "scheduler": "cosine", "warmup_ratio": 0.25, "weight_decay": 0.1,
            "max_grad_norm": 0.3, "optimizer": "adafactor", "lr": 0.001,
            "quantization": "none", "gradient_checkpointing": False,
            "logging_steps": 1, "save_steps": 100,
            "lora": {"r": 2, "alpha": 4, "dropout": 0.0, "target_modules": ["q_proj"]},
            **training,
        },
    )


@pytest.mark.parametrize("kind", ["prm", "moe_lora_routing"])
def test_prm_and_mole_train_with_configured_optimizer(tiny_model, monkeypatch, kind):
    from peft import LoraConfig, get_peft_model
    from transformers import Adafactor, AutoModelForCausalLM

    from soup_cli.trainer import mole_routing, prm

    training = {"batch_size": "auto"}
    if kind == "moe_lora_routing":
        adapters = []
        for index in range(2):
            model = get_peft_model(
                AutoModelForCausalLM.from_pretrained(tiny_model),
                LoraConfig(r=2, target_modules=["q_proj"], task_type="CAUSAL_LM"),
            )
            path = tiny_model.parent / f"adapter-{index}"
            model.save_pretrained(path)
            adapters.append(str(path))
        training["mole_task_adapters"] = adapters
        module = mole_routing
        wrapper_class = mole_routing.MoleRoutingTrainerWrapper
        rows = [{"text": "one two three"}] * 8
    else:
        module = prm
        wrapper_class = prm.PRMTrainerWrapper
        rows = [{"prompt": "one", "completions": [" two", " three"], "labels": [0.0, 1.0]}] * 8
    monkeypatch.setattr(module, "estimate_batch_size", lambda **kwargs: 2)
    config = _config(tiny_model, kind, **training)
    wrapper = wrapper_class(config, device="cpu")
    wrapper.setup({"train": rows})
    result = wrapper.train()
    args = wrapper.trainer.args
    assert args.per_device_train_batch_size == 2
    assert args.warmup_steps == 1
    assert args.weight_decay == 0.1
    assert args.max_grad_norm == 0.3
    assert args.optim.value == "adafactor"
    assert args.lr_scheduler_type.value == "cosine"
    assert not args.gradient_checkpointing
    optimizer = wrapper.trainer.optimizer
    optimizer = getattr(optimizer, "optimizer", optimizer)
    assert isinstance(optimizer, Adafactor)
    assert any(group["weight_decay"] == 0.1 for group in optimizer.param_groups)
    assert result["total_steps"] == 4


def test_ppo_constructs_real_config_with_nondefault_training_values(tiny_model):
    from trl.experimental.ppo import PPOConfig

    from soup_cli.trainer.ppo import PPOTrainerWrapper

    class ConfigCapture:
        def __init__(self, *, args, **kwargs):
            self.args = args

        def add_callback(self, callback):
            pass

    config = _config(tiny_model, "ppo", reward_fn="accuracy")
    wrapper = PPOTrainerWrapper(config, device="cpu")
    # Preserve actual model loading, dataset preparation and PPOConfig creation.
    # Only stop before PPO's separate rollout-training loop, which #805 does not alter.
    with (
        patch(
            "soup_cli.trainer.ppo._import_ppo_classes",
            return_value=(ConfigCapture, PPOConfig, True),
        ),
        patch.object(wrapper, "_get_or_create_reward_model", return_value=None),
        patch.object(wrapper, "_create_value_model", return_value=None),
    ):
        wrapper.setup({"train": [{"prompt": "one two", "answer": "three"}] * 8})
    args = wrapper.trainer.args
    assert isinstance(args, PPOConfig)
    assert args.warmup_steps == 1
    assert args.weight_decay == 0.1
    assert args.max_grad_norm == 0.3
    assert args.optim.value == "adafactor"
    assert args.lr_scheduler_type.value == "cosine"
    assert args.num_train_epochs == 2
    assert args.logging_steps == 1
    assert args.save_steps == 100
    assert not args.gradient_checkpointing


def _unlearn_wrapper(tiny_model, **training):
    from soup_cli.trainer.unlearn import UnlearnTrainerWrapper

    (tiny_model.parent / "forget.jsonl").write_text(
        (json.dumps({"prompt": "one ", "completion": "two three"}) + "\n") * 5
    )
    config = _config(tiny_model, "unlearn", epochs=1, unlearn_method="simnpo", **training)
    wrapper = UnlearnTrainerWrapper(config, device="cpu")
    wrapper.setup()
    return wrapper


def test_unlearn_default_auto_reaches_real_training(tiny_model, monkeypatch):
    from soup_cli.utils import gpu

    monkeypatch.setattr(gpu, "estimate_batch_size", lambda **kwargs: 2)
    wrapper = _unlearn_wrapper(tiny_model, batch_size="auto")
    assert wrapper._batch_size == 2
    assert wrapper.train()["total_steps"] == 5


@pytest.mark.parametrize(("batch_size", "accumulation"), [(1, 1), (1, 4), (2, 4)])
def test_unlearn_reports_unscaled_loss_and_example_count(tiny_model, batch_size, accumulation):
    import torch
    from transformers import Adafactor

    from soup_cli.utils.live_eval import _tokenize_pair

    wrapper = _unlearn_wrapper(
        tiny_model, batch_size=batch_size, gradient_accumulation_steps=accumulation
    )
    expected = wrapper._step_preference(
        _tokenize_pair, "one ", "two three", "simnpo", "cpu"
    ).item()
    assert isinstance(wrapper._optimizer, Adafactor)
    assert wrapper._optimizer.param_groups[0]["weight_decay"] == 0.1
    updates = math.ceil(5 / (batch_size * accumulation))
    with patch("torch.nn.utils.clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_) as clip:
        result = wrapper.train()
    assert result["initial_loss"] == pytest.approx(expected)
    assert result["total_steps"] == 5
    assert clip.call_count == updates
    assert all(call.args[1] == 0.3 for call in clip.call_args_list)
    assert wrapper._scheduler.last_epoch == updates
    warmup = int(updates * wrapper.config.training.warmup_ratio)
    probe_step = warmup + 0.25 * (updates - warmup)
    assert wrapper._scheduler.lr_lambdas[0](probe_step) == pytest.approx(
        0.5 * (1.0 + math.cos(math.pi * 0.25))
    )


def test_unlearn_uses_configured_sequence_length(tiny_model):
    from soup_cli.utils.live_eval import _tokenize_pair

    wrapper = _unlearn_wrapper(tiny_model)
    lengths = []

    def capture(*args, **kwargs):
        lengths.append(kwargs["max_length"])
        return _tokenize_pair(*args, **kwargs)

    wrapper._step_preference(capture, "one ", "two three " * 20, "simnpo", "cpu")
    assert lengths == [64]


def test_unlearn_partial_group_averages_actual_gradients(tiny_model):
    import torch

    wrapper = _unlearn_wrapper(tiny_model, batch_size=2, gradient_accumulation_steps=2)
    parameter = next(p for p in wrapper.model.parameters() if p.requires_grad)
    gradients = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def capture_clip(parameters, max_norm):
        gradients.append(parameter.grad.detach().clone())
        return real_clip(parameters, max_norm)

    with (
        patch.object(wrapper, "_step_preference", side_effect=lambda *args: 2 * parameter.sum()),
        patch("torch.nn.utils.clip_grad_norm_", side_effect=capture_clip),
    ):
        result = wrapper.train()
    assert result["total_steps"] == 5
    assert len(gradients) == 2
    for gradient in gradients:
        torch.testing.assert_close(gradient, torch.full_like(gradient, 2.0))


def test_unlearn_cap_still_counts_examples(tiny_model, monkeypatch):
    from soup_cli.trainer import unlearn

    monkeypatch.setattr(unlearn, "_MAX_STEPS_CAP", 3)
    wrapper = _unlearn_wrapper(tiny_model, batch_size=2, gradient_accumulation_steps=4)
    assert wrapper.train()["total_steps"] == 3
