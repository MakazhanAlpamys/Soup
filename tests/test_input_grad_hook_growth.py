"""#2365: the input-gradient hook does not pile up across PPO, GRPO and online DPO steps.

With gradient checkpointing on, the trl loops switch it off and back on around
generation. Every ``gradient_checkpointing_enable`` calls
``enable_input_require_grads()``, which registers a new forward hook on the input
embedding and forgets the old one, so the hook count grew with the step count.
``keep_one_input_grad_hook`` makes each enable remove the previous hook first.
The hook itself must stay: the embedding output still has to require grad.

These run the real trl trainers on CPU with the tiny Llama models of
``tests/test_ppo_wrapper_checkpointing.py`` (PPO, with its kbit stand-in for the
4-bit path) and ``tests/test_issue797_trainer_checkpointing.py`` (GRPO, online DPO).
"""

from __future__ import annotations

import pytest

for _mod in ("torch", "transformers", "peft", "trl", "datasets"):
    pytest.importorskip(_mod)

from tests.test_issue797_trainer_checkpointing import (  # noqa: E402, F401
    _build,
    _LocalJudge,
    train_stack,
)
from tests.test_issue1391_ppo_real_step import _PROMPTS  # noqa: E402
from tests.test_ppo_wrapper_checkpointing import _config, _kbit_stand_in  # noqa: E402

_STEPS = 3


def _hook_count(model) -> int:
    return len(model.get_input_embeddings()._forward_hooks)


def _embedding_output_requires_grad(model) -> bool:
    import torch

    model.train()
    embedding = model.get_input_embeddings()
    # The embedding is frozen under LoRA, so only the hook can make its output require grad.
    assert embedding.weight.requires_grad is False
    return embedding(torch.tensor([[1, 2, 3]])).requires_grad


def _held_hook_is_registered(model) -> bool:
    """The hook the model holds is live on the embedding. On the PPO 4-bit path the
    hooks kbit prep registered first stay, so requires-grad alone would not show the
    model's own hook being dropped."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    registered = model.get_input_embeddings()._forward_hooks
    held = getattr(base, "_require_grads_hooks", None) or []
    return bool(held) and all(handle.id in registered for handle in held)


def _trainable(model) -> dict:
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def _unchanged(model, before: dict) -> list[str]:
    import torch

    return [
        name
        for name, param in model.named_parameters()
        if name in before and torch.equal(param.detach(), before[name])
    ]


def _count_hooks_per_step(trainer, model) -> list[int]:
    from transformers import TrainerCallback

    counts: list[int] = []

    class _Counter(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            counts.append(_hook_count(model))

    trainer.add_callback(_Counter())
    return counts


def _record_helper_calls(monkeypatch) -> list:
    import soup_cli.utils.gradient_ckpt as gradient_ckpt

    calls: list = []
    real = gradient_ckpt.keep_one_input_grad_hook

    def recording(model):
        calls.append(model)
        return real(model)

    monkeypatch.setattr(gradient_ckpt, "keep_one_input_grad_hook", recording)
    return calls


def _ppo(tmp_path, monkeypatch, *, checkpointing: bool, quantization: str = "none"):
    from soup_cli.trainer.ppo import PPOTrainerWrapper

    cfg = _config(tmp_path, monkeypatch, checkpointing=checkpointing, quantization=quantization)
    # Four prompts at batch 2 are two PPO steps an epoch.
    cfg.training.epochs = 2
    wrapper = PPOTrainerWrapper(cfg, device="cpu")
    wrapper.setup({"train": [{"messages": [{"role": "user", "content": p}]} for p in _PROMPTS]})
    return wrapper


def _run_ppo(wrapper) -> tuple[int, list[int]]:
    before = _hook_count(wrapper.model)
    counts = _count_hooks_per_step(wrapper.trainer, wrapper.model)
    result = wrapper.train()
    assert result["total_steps"] >= _STEPS, result
    assert len(counts) >= _STEPS, counts
    return before, counts


def _run_trl(trainer, steps: int = _STEPS) -> tuple[int, list[int]]:
    before = _hook_count(trainer.model)
    counts = _count_hooks_per_step(trainer, trainer.model)
    trainer.args.max_steps = steps
    trainer.args.save_strategy = "no"
    trainer.args.disable_tqdm = True
    trainer.train()
    assert trainer.state.global_step == steps
    assert len(counts) == steps, counts
    return before, counts


def test_the_helper_keeps_one_hook_and_a_second_call_does_nothing():
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    from soup_cli.utils.gradient_ckpt import keep_one_input_grad_hook

    config = LlamaConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
    )
    model = get_peft_model(
        LlamaForCausalLM(config), LoraConfig(r=4, target_modules=["q_proj", "v_proj"])
    )
    base = model.get_base_model()
    assert keep_one_input_grad_hook(model) is True
    wrapped = base.enable_input_require_grads
    assert keep_one_input_grad_hook(model) is False
    assert base.enable_input_require_grads is wrapped

    model.gradient_checkpointing_enable()
    once = _hook_count(model)
    for _ in range(3):
        model.gradient_checkpointing_disable()
        model.gradient_checkpointing_enable()
    assert _hook_count(model) == once
    assert _embedding_output_requires_grad(model) is True
    assert _held_hook_is_registered(model) is True


def test_ppo_4bit_hook_count_stays_flat_and_the_hook_still_works(tmp_path, monkeypatch):
    _kbit_stand_in(monkeypatch)
    wrapper = _ppo(tmp_path, monkeypatch, checkpointing=True, quantization="4bit")
    assert wrapper.model.is_gradient_checkpointing is True
    trainable = _trainable(wrapper.model)
    assert trainable

    _, counts = _run_ppo(wrapper)

    assert counts[-1] == counts[0], counts
    assert _embedding_output_requires_grad(wrapper.model) is True
    assert _held_hook_is_registered(wrapper.model) is True
    assert not _unchanged(wrapper.model, trainable)


def test_ppo_checkpointing_off_changes_nothing(tmp_path, monkeypatch):
    calls = _record_helper_calls(monkeypatch)
    wrapper = _ppo(tmp_path, monkeypatch, checkpointing=False)

    before, counts = _run_ppo(wrapper)

    assert calls == []
    assert set(counts) == {before}, (before, counts)


def _completion_length(completions, **kwargs) -> list[float]:
    """A reward that differs within a group, so the steps have an advantage to train on."""
    return [float(len(str(completion))) for completion in completions]


def test_grpo_hook_count_stays_flat_and_the_hook_still_works(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "soup_cli.trainer.grpo._select_reward_fn", lambda *args: _completion_length
    )
    _, trainer = _build(tmp_path, monkeypatch, "grpo", True)
    trainable = _trainable(trainer.model)
    assert trainable

    _, counts = _run_trl(trainer)

    assert counts[-1] == counts[0], counts
    assert _embedding_output_requires_grad(trainer.model) is True
    assert _held_hook_is_registered(trainer.model) is True
    assert not _unchanged(trainer.model, trainable)


def test_grpo_checkpointing_off_changes_nothing(tmp_path, monkeypatch):
    calls = _record_helper_calls(monkeypatch)
    _, trainer = _build(tmp_path, monkeypatch, "grpo", False)

    before, counts = _run_trl(trainer)

    assert calls == []
    assert set(counts) == {before}, (before, counts)


def _prefer_the_longer_completion(self, prompt: str, resp_a: str, resp_b: str) -> int:
    """A verdict that survives the swap check, so the steps have pairs to train on."""
    if (len(resp_a), resp_a) == (len(resp_b), resp_b):
        return -1
    return 0 if (len(resp_a), resp_a) > (len(resp_b), resp_b) else 1


def test_online_dpo_hook_count_stays_flat_and_the_hook_still_works(tmp_path, monkeypatch):
    monkeypatch.setattr(_LocalJudge, "compare_pair", _prefer_the_longer_completion)
    _, trainer = _build(tmp_path, monkeypatch, "online_dpo", True)
    trainable = _trainable(trainer.model)
    assert trainable

    _, counts = _run_trl(trainer)

    assert counts[-1] == counts[0], counts
    assert _embedding_output_requires_grad(trainer.model) is True
    assert _held_hook_is_registered(trainer.model) is True
    assert not _unchanged(trainer.model, trainable)


def test_online_dpo_checkpointing_off_changes_nothing(tmp_path, monkeypatch):
    calls = _record_helper_calls(monkeypatch)
    _, trainer = _build(tmp_path, monkeypatch, "online_dpo", False)

    before, counts = _run_trl(trainer)

    assert calls == []
    assert set(counts) == {before}, (before, counts)
