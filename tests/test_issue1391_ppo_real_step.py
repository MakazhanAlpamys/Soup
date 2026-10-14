"""#1391 — ``task: ppo`` never completed a step on trl 0.29.

``setup()`` handed trl's ``PPOTrainer`` a dataset that still carried the
``prompt_text`` strings, and trl's ``DataCollatorWithPadding`` tried to turn
them into a tensor before the first rollout. A train set smaller than one
rollout batch hung instead, because trl's DataLoader drops the ragged batch and
repeats itself forever.

These tests drive the real trl trainer on CPU with a tiny on-disk Llama policy
and a tiny reward model. Nothing about the trainer is mocked.
"""

from __future__ import annotations

import json

import pytest

for _mod in ("torch", "transformers", "peft", "trl", "datasets"):
    pytest.importorskip(_mod)


def _tiny_models(root):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        LlamaForSequenceClassification,
        PreTrainedTokenizerFast,
    )

    def config(**extra):
        return LlamaConfig(
            vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True,
            max_position_embeddings=128, pad_token_id=3, bos_token_id=1, eos_token_id=2,
            **extra,
        )

    torch.manual_seed(7)
    names = ["<unk>", "<s>", "</s>", "<pad>", "say", "hi", "hello", "world"]
    tok = Tokenizer(models.WordLevel(vocab={w: i for i, w in enumerate(names)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    base, rm = root / "base", root / "rm"
    LlamaForCausalLM(config()).save_pretrained(base)
    PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", bos_token="<s>",
        eos_token="</s>", pad_token="<pad>",
    ).save_pretrained(base)
    LlamaForSequenceClassification(config(num_labels=1)).save_pretrained(rm)
    return base.as_posix(), rm.as_posix()


_PROMPTS = ("say hi", "say hello world", "hi", "hello")


def _wrapper(tmp_path, monkeypatch, grad_accum=1):
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.ppo import PPOTrainerWrapper

    monkeypatch.setenv("WANDB_DISABLED", "true")
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.chdir(tmp_path)
    base, rm = _tiny_models(tmp_path)
    cfg = load_config_from_string(
        f"base: {base}\ntask: ppo\nbackend: transformers\n"
        "data:\n  train: train.jsonl\n  max_length: 64\n"
        # One rollout batch is batch_size * gradient_accumulation_steps rows, so
        # these 4 rows fill two batches at batch_size 2, accumulation 1.
        f"training:\n  epochs: 1\n  batch_size: 2\n  gradient_accumulation_steps: {grad_accum}\n"
        f"  quantization: none\n  ppo_epochs: 1\n  reward_model: {rm}\n"
        "  lora:\n    r: 4\n    alpha: 8\n    target_modules: [q_proj, v_proj]\n"
        f"output: {(tmp_path / 'out').as_posix()}\n"
    )
    wrapper = PPOTrainerWrapper(cfg, device="cpu")
    wrapper.setup({"train": [{"messages": [{"role": "user", "content": p}]} for p in _PROMPTS]})
    return wrapper


def test_ppo_hands_trl_only_token_columns(tmp_path, monkeypatch):
    wrapper = _wrapper(tmp_path, monkeypatch)
    columns = set(wrapper.trainer.train_dataset.column_names)
    assert columns <= {"input_ids", "attention_mask"}, columns
    # the manual loop's copy keeps the prompt text
    assert "prompt_text" in wrapper._train_ds.column_names


def test_ppo_runs_a_real_step(tmp_path, monkeypatch):
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from transformers import AutoModelForCausalLM

    wrapper = _wrapper(tmp_path, monkeypatch)
    wrapper.trainer.args.save_steps = 1
    composite_model = wrapper.trainer.model
    result = wrapper.train()
    assert result["total_steps"] == 2, result
    assert wrapper.trainer.model is composite_model
    assert any("loss/policy_avg" in e for e in wrapper.trainer.state.log_history)
    output = tmp_path / "out"
    for step in (1, 2):
        checkpoint = output / f"checkpoint-{step}"
        for filename in (
            "adapter_config.json", "adapter_model.safetensors", "optimizer.pt",
            "scheduler.pt", "rng_state.pth", "training_args.bin", "tokenizer.json",
        ):
            assert (checkpoint / filename).is_file(), (step, filename)
        state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
        assert state["global_step"] == step

    reloaded = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(wrapper.config.base), output,
    )
    trained = get_peft_model_state_dict(wrapper.model)
    saved = get_peft_model_state_dict(reloaded)
    assert saved.keys() == trained.keys()
    for name, value in trained.items():
        torch.testing.assert_close(saved[name], value, rtol=0, atol=0)


def test_a_train_set_smaller_than_one_rollout_batch_is_refused(tmp_path, monkeypatch):
    # 4 rows, batch_size 2 x accumulation 4 = 8: trl would spin without a step
    with pytest.raises(ValueError) as exc:
        _wrapper(tmp_path, monkeypatch, grad_accum=4)
    msg = str(exc.value)
    assert "4 rows" in msg
    assert "batch_size=2" in msg
    assert "gradient_accumulation_steps=4" in msg


def test_the_rollout_batch_counts_every_process(monkeypatch):
    from soup_cli.trainer.ppo import _check_one_rollout_batch

    monkeypatch.setenv("WORLD_SIZE", "2")
    _check_one_rollout_batch(8, 2, 2)
    with pytest.raises(ValueError, match=r"x 2 processes = 8"):
        _check_one_rollout_batch(7, 2, 2)
    monkeypatch.setenv("WORLD_SIZE", "nonsense")
    _check_one_rollout_batch(4, 2, 2)


def test_the_refusal_comes_before_any_model_is_loaded(tmp_path, monkeypatch):
    """The refusal needs only the row count and two settings, so it must not
    wait for the reward model and the policy to load."""
    from soup_cli.trainer import ppo

    loaded = []
    real_reward = ppo.PPOTrainerWrapper._setup_reward
    real_policy = ppo.PPOTrainerWrapper._setup_transformers

    def reward(self, *args):
        loaded.append("reward model")
        return real_reward(self, *args)

    def policy(self, *args):
        loaded.append("policy")
        return real_policy(self, *args)

    monkeypatch.setattr(ppo.PPOTrainerWrapper, "_setup_reward", reward)
    monkeypatch.setattr(ppo.PPOTrainerWrapper, "_setup_transformers", policy)
    with pytest.raises(ValueError, match="4 rows"):
        _wrapper(tmp_path, monkeypatch, grad_accum=4)
    assert loaded == [], f"refused only after loading: {loaded}"
