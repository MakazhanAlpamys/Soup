"""#2197: ``task: ppo`` honours ``training.gradient_checkpointing`` on the policy.

TRL's PPO trainer runs its own loop rather than ``Trainer.train()`` and never calls
``gradient_checkpointing_enable()``, so outside ``quantization: 4bit`` / ``8bit``
(where ``prepare_model_for_kbit_training`` switches it on) the setting reached
``PPOConfig`` and nothing else. These tests build the real ``_setup_transformers``
on CPU with the tiny on-disk Llama the PPO step tests use, and run one real PPO
step with checkpointing on, so rollout generation is shown to still work.
"""

from __future__ import annotations

import math

import pytest

for _mod in ("torch", "transformers", "peft", "trl", "datasets"):
    pytest.importorskip(_mod)

from tests.test_ppo_wrapper_checkpointing import (  # noqa: E402
    _checkpoint_kwargs,
    _config,
    _kbit_stand_in,
    _setup,
)


def _policy(tmp_path, monkeypatch, **config):
    from soup_cli.trainer.ppo import PPOTrainerWrapper

    cfg = _config(tmp_path, monkeypatch, **config)
    wrapper = PPOTrainerWrapper(cfg, device="cpu")
    wrapper._setup_transformers(cfg, cfg.training)
    return wrapper.model


def test_checkpointing_true_turns_it_on_for_the_policy(tmp_path, monkeypatch):
    model = _policy(tmp_path, monkeypatch, checkpointing=True)
    assert model.is_gradient_checkpointing is True


def test_checkpointing_false_leaves_the_policy_without_it(tmp_path, monkeypatch):
    model = _policy(tmp_path, monkeypatch, checkpointing=False)
    assert model.is_gradient_checkpointing is False


@pytest.mark.parametrize("quantization", ["4bit", "8bit"])
def test_a_kbit_run_is_switched_on_by_kbit_prep_alone(tmp_path, monkeypatch, quantization):
    """kbit prep keeps doing it, with its own kwargs; the new site stays out."""
    import peft
    from transformers import PreTrainedModel

    _kbit_stand_in(monkeypatch)
    stand_in = peft.prepare_model_for_kbit_training
    prep_kwargs: dict = {}
    caller = ["setup"]
    enables: list[tuple[str, dict]] = []
    real_enable = PreTrainedModel.gradient_checkpointing_enable

    def spy_enable(self, *args, **kwargs):
        enables.append((caller[0], kwargs))
        return real_enable(self, *args, **kwargs)

    def recording_prep(model, **kwargs):
        prep_kwargs.update(kwargs)
        caller[0] = "kbit prep"
        try:
            return stand_in(model, **kwargs)
        finally:
            caller[0] = "setup"

    monkeypatch.setattr(PreTrainedModel, "gradient_checkpointing_enable", spy_enable)
    monkeypatch.setattr(peft, "prepare_model_for_kbit_training", recording_prep)
    model = _policy(tmp_path, monkeypatch, checkpointing=True, quantization=quantization)

    assert prep_kwargs.get("use_gradient_checkpointing") is True
    assert enables == [("kbit prep", {"gradient_checkpointing_kwargs": {}})]
    assert model.is_gradient_checkpointing is True


def test_one_ppo_step_runs_with_checkpointing_on(tmp_path, monkeypatch):
    """The rollout generates with checkpointing off; the step then trains the LoRA."""
    import torch

    wrapper = _setup(tmp_path, monkeypatch, checkpointing=True)
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
    assert (tmp_path / "out" / "adapter_config.json").is_file()
    losses = [e["loss/policy_avg"] for e in wrapper.trainer.state.log_history
              if "loss/policy_avg" in e]
    assert losses and all(math.isfinite(loss) for loss in losses), losses
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
