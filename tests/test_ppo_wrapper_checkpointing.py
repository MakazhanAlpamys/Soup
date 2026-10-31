"""#2311: a ``task: ppo`` policy with gradient checkpointing on survives the rollout.

trl's ``PolicyAndValueWrapper`` copies ``is_gradient_checkpointing`` from the
policy but has no ``gradient_checkpointing_disable`` / ``_enable``, and the
rollout's ``unwrap_model_for_generation`` calls both on it. Today the policy
has checkpointing on when ``quantization: 4bit`` / ``8bit`` hands it to
``prepare_model_for_kbit_training``. These tests run the real trl trainer on
CPU with the tiny on-disk Llama of the PPO step tests; kbit prep is replaced
by a stand-in, since there is no bitsandbytes on CPU.
"""

from __future__ import annotations

import functools
import math

import pytest

for _mod in ("torch", "transformers", "peft", "trl", "datasets"):
    pytest.importorskip(_mod)

from tests.test_issue1391_ppo_real_step import _PROMPTS, _tiny_models  # noqa: E402


def _config(tmp_path, monkeypatch, *, checkpointing: bool, quantization: str = "none"):
    from soup_cli.config.loader import load_config_from_string

    monkeypatch.setenv("WANDB_DISABLED", "true")
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.chdir(tmp_path)
    base, rm = _tiny_models(tmp_path)
    return load_config_from_string(
        f"base: {base}\ntask: ppo\nbackend: transformers\n"
        "data:\n  train: train.jsonl\n  max_length: 64\n"
        "training:\n  epochs: 1\n  batch_size: 2\n  gradient_accumulation_steps: 1\n"
        f"  quantization: {quantization}\n  ppo_epochs: 1\n  reward_model: {rm}\n"
        f"  gradient_checkpointing: {str(checkpointing).lower()}\n"
        "  lora:\n    r: 4\n    alpha: 8\n    target_modules: [q_proj, v_proj]\n"
        f"output: {(tmp_path / 'out').as_posix()}\n"
    )


def _kbit_stand_in(monkeypatch) -> None:
    """No bitsandbytes on CPU: load unquantized, and let kbit prep do what peft
    0.21 does for a quantized model when ``use_gradient_checkpointing`` is true."""
    import peft

    def kbit_prep(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}
        if use_gradient_checkpointing:
            if gradient_checkpointing_kwargs.get("use_reentrant", True):
                model.enable_input_require_grads()
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        return model

    monkeypatch.setattr(peft, "prepare_model_for_kbit_training", kbit_prep)
    monkeypatch.setattr(
        "soup_cli.utils.quant_menu.build_quantization_config_for_loader",
        lambda **kwargs: None,
    )


def _setup(tmp_path, monkeypatch, **config):
    from soup_cli.trainer.ppo import PPOTrainerWrapper

    cfg = _config(tmp_path, monkeypatch, **config)
    wrapper = PPOTrainerWrapper(cfg, device="cpu")
    wrapper.setup({"train": [{"messages": [{"role": "user", "content": p}]} for p in _PROMPTS]})
    return wrapper


def _checkpoint_kwargs(policy) -> dict:
    """The kwargs ``gradient_checkpointing_enable`` bound into the layers' partial."""
    funcs = [
        module._gradient_checkpointing_func
        for module in policy.modules()
        if getattr(module, "gradient_checkpointing", False)
    ]
    assert funcs, "no checkpointed module"
    assert all(isinstance(func, functools.partial) for func in funcs), funcs
    return dict(funcs[0].keywords)


def _ppo_wrapper(wrapper):
    trainer = wrapper.trainer
    return trainer.accelerator.unwrap_model(trainer.model)


def test_a_4bit_step_runs_with_checkpointing_on(tmp_path, monkeypatch):
    import torch

    _kbit_stand_in(monkeypatch)
    wrapper = _setup(tmp_path, monkeypatch, checkpointing=True, quantization="4bit")
    assert wrapper.model.is_gradient_checkpointing is True
    before = _checkpoint_kwargs(wrapper.model)
    trainable = {
        name: param.detach().clone()
        for name, param in wrapper.model.named_parameters()
        if param.requires_grad
    }
    assert trainable

    checkpointed_while_generating = []
    generate = wrapper.model.generate

    def recording_generate(*args, **kwargs):
        checkpointed_while_generating.append(wrapper.model.is_gradient_checkpointing)
        return generate(*args, **kwargs)

    monkeypatch.setattr(wrapper.model, "generate", recording_generate, raising=False)

    result = wrapper.train()

    assert result["total_steps"] >= 1, result
    losses = [e["loss/policy_avg"] for e in wrapper.trainer.state.log_history
              if "loss/policy_avg" in e]
    assert losses and all(math.isfinite(loss) for loss in losses), losses
    # The rollout generates with checkpointing off, then it is back on with the same kwargs.
    assert checkpointed_while_generating, "the policy never generated"
    assert not any(checkpointed_while_generating), checkpointed_while_generating
    assert wrapper.model.is_gradient_checkpointing is True
    assert _checkpoint_kwargs(wrapper.model) == before
    unchanged = [
        name
        for name, param in wrapper.model.named_parameters()
        if name in trainable and torch.equal(param.detach(), trainable[name])
    ]
    assert not unchanged, unchanged


def test_checkpointing_off_adds_nothing_to_the_wrapper(tmp_path, monkeypatch):
    wrapper = _setup(tmp_path, monkeypatch, checkpointing=False)
    ppo_wrapper = _ppo_wrapper(wrapper)
    assert type(ppo_wrapper).__name__ == "PolicyAndValueWrapper"
    assert ppo_wrapper.is_gradient_checkpointing is False
    assert not hasattr(ppo_wrapper, "gradient_checkpointing_disable")
    assert not hasattr(ppo_wrapper, "gradient_checkpointing_enable")

    result = wrapper.train()
    assert result["total_steps"] >= 1, result
    assert wrapper.model.is_gradient_checkpointing is False


def test_a_wrapper_with_its_own_disable_keeps_it(tmp_path, monkeypatch):
    """A trl whose wrapper has the methods itself is left alone."""
    from trl.experimental.ppo import ppo_trainer

    def own_disable(self):
        self.policy.gradient_checkpointing_disable()

    def own_enable(self):
        self.policy.gradient_checkpointing_enable()

    wrapper_cls = ppo_trainer.PolicyAndValueWrapper
    monkeypatch.setattr(wrapper_cls, "gradient_checkpointing_disable", own_disable, raising=False)
    monkeypatch.setattr(wrapper_cls, "gradient_checkpointing_enable", own_enable, raising=False)
    _kbit_stand_in(monkeypatch)
    wrapper = _setup(tmp_path, monkeypatch, checkpointing=True, quantization="4bit")

    ppo_wrapper = _ppo_wrapper(wrapper)
    assert ppo_wrapper.is_gradient_checkpointing is True
    assert "gradient_checkpointing_disable" not in vars(ppo_wrapper)
    assert "gradient_checkpointing_enable" not in vars(ppo_wrapper)
    assert ppo_wrapper.gradient_checkpointing_disable.__func__ is own_disable
