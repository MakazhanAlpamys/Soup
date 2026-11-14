"""122B GRPO configuration and native tiny-model consumers.

Public config verified separately at dc4d348443bc740c68e2d77492492c11606384d5:
qwen3_5_moe, 48 text layers, 256 experts, 8 active. Tests stay offline.
Tiny random weights qualify neither full weights nor a GRPO training run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.config.loader import load_config_from_string
from soup_cli.data.loader import load_dataset
from soup_cli.recipes.catalog import RECIPES, get_recipe, search_recipes
from tests.conftest import strip_ansi

NAME = "qwen3.5-122b-a10b-grpo"
BASE = "Qwen/Qwen3.5-122B-A10B"


def test_metadata_names_the_122b_reasoning_model_independently() -> None:
    recipe = get_recipe(NAME)
    assert recipe is not None
    assert recipe.model == BASE
    assert recipe.task == "grpo"
    assert recipe.size == "122B"
    assert {"qwen", "qwen3.5", "grpo", "reasoning", "moe", "large", "multi-gpu"} <= set(recipe.tags)
    assert "Apache-2.0" in recipe.description
    assert recipe in search_recipes(query="reasoning", task="grpo", size="122b")


def test_yaml_selects_the_122b_text_model_and_grpo_starting_values() -> None:
    raw = yaml.safe_load(RECIPES[NAME].yaml_str)
    assert raw["base"] == BASE
    assert raw["task"] == "grpo"
    assert raw["modality"] == "text"
    assert raw["training"]["gradient_checkpointing"] is True
    assert raw["training"]["moe_lora"] is True
    assert raw["training"]["lora"]["dropout"] == 0.0

    cfg = load_config_from_string(RECIPES[NAME].yaml_str)
    assert cfg.base == BASE
    assert cfg.task == "grpo"
    assert cfg.modality == "text"
    assert cfg.data.train == "./data/reasoning_train.jsonl"
    assert cfg.data.format == "auto"
    assert cfg.data.max_length == 4096
    assert cfg.training.lr == 1e-5
    assert cfg.training.epochs == 1
    assert cfg.training.batch_size == 1
    assert cfg.training.gradient_accumulation_steps == 16
    assert cfg.training.lora.r == 32
    assert cfg.training.lora.alpha == 64
    assert cfg.training.lora.target_modules == "auto"
    assert cfg.training.quantization == "4bit"
    assert cfg.training.gradient_checkpointing is True
    assert cfg.training.moe_lora is True
    assert cfg.training.lora.dropout == 0.0
    assert "moe_aux_loss_coeff" not in raw["training"]
    assert cfg.training.grpo_beta == 0.1
    assert cfg.training.num_generations == 4
    assert cfg.training.reward_fn == "accuracy"
    assert cfg.output == "./output"


def test_reasoning_rows_reach_the_accuracy_reward_with_a_gold_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from soup_cli.trainer.grpo import _prepare_grpo_dataset, _validate_grpo_reward_metadata
    from soup_cli.trainer.rewards import load_reward_fn

    monkeypatch.chdir(tmp_path)
    cfg = load_config_from_string(RECIPES[NAME].yaml_str)
    cfg.data.val_split = 0
    data = Path("data")
    data.mkdir()
    row = {"messages": [
        {"role": "user", "content": "What is two plus two?"},
        {"role": "assistant", "content": "4"},
    ]}
    (data / "reasoning_train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    prepared = _prepare_grpo_dataset(load_dataset(cfg.data)["train"])
    assert prepared == [{
        "prompt": [{"role": "user", "content": "What is two plus two?"}], "answer": "4",
    }]
    _validate_grpo_reward_metadata(prepared, cfg.training, split="train")
    reward = load_reward_fn(cfg.training.reward_fn)
    completions = [[{"role": "assistant", "content": text}] for text in ("4", "5")]
    assert reward(completions, answer=["4", "4"]) == [1.0, 0.0]

    without_gold = _prepare_grpo_dataset([{"messages": row["messages"][:1]}])
    with pytest.raises(ValueError, match="missing or empty 'answer'.*reward 'accuracy'"):
        _validate_grpo_reward_metadata(without_gold, cfg.training, split="train")


def test_show_and_use_write_the_same_loadable_text_grpo_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    shown = runner.invoke(app, ["recipes", "show", NAME])
    assert shown.exit_code == 0, (strip_ansi(shown.output), repr(shown.exception))
    assert BASE in strip_ansi(shown.output)
    used = runner.invoke(app, ["recipes", "use", NAME, "--yes"])
    assert used.exit_code == 0, (strip_ansi(used.output), repr(used.exception))
    text = (tmp_path / "soup.yaml").read_text(encoding="utf-8")
    assert yaml.safe_load(text) == yaml.safe_load(RECIPES[NAME].yaml_str)
    cfg = load_config_from_string(text)
    assert cfg.base == BASE
    assert cfg.task == "grpo"
    assert cfg.modality == "text"
    assert cfg.training.reward_fn == "accuracy"


def test_native_text_decoder_attaches_recipe_lora_and_backpropagates_on_cpu() -> None:
    import torch
    from peft import get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    from soup_cli.utils.attach_preflight import loader_for, trainer_lora_config

    cfg = load_config_from_string(RECIPES[NAME].yaml_str)
    assert loader_for(cfg) == ("AutoModelForCausalLM",)
    # The public 122B-A10B decoder mixes linear and full attention. Keep both native
    # block kinds, with tiny random weights and the pure-PyTorch CPU fallback.
    # This qualifies the text/LoRA consumer, not public 122B-A10B weights or NF4,
    # and uses the model's causal-LM loss rather than a GRPO trainer rollout.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        config = AutoConfig.for_model("qwen3_5_moe", text_config={
            "vocab_size": 32, "hidden_size": 32, "intermediate_size": 64,
            "moe_intermediate_size": 16, "shared_expert_intermediate_size": 16,
            "num_experts": 4, "num_experts_per_tok": 2,
            "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 2,
            "head_dim": 16, "max_position_embeddings": 64, "full_attention_interval": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_conv_kernel_dim": 4, "linear_key_head_dim": 8,
            "linear_num_key_heads": 2, "linear_num_value_heads": 2,
            "linear_value_head_dim": 8, "tie_word_embeddings": False,
        })
        model = AutoModelForCausalLM.from_config(config).to("cpu")
        assert type(model).__name__ == "Qwen3_5MoeForCausalLM"
        lora_config, _targets = trainer_lora_config(model, cfg, None)
        model = get_peft_model(model, lora_config)
        assert not any("visual" in name or "vision" in name for name, _ in model.named_parameters())
        expert_grads = [
            param for name, param in model.named_parameters()
            if "experts" in name and "lora_B" in name
        ]
        assert len(expert_grads) == 4  # gate/up and down in each native MoE layer
        tokens = torch.tensor([[1, 2, 3, 4]])
        result = model(input_ids=tokens, labels=tokens)
        assert result.logits.shape == (1, 4, 32)
        assert torch.isfinite(result.loss)
        result.loss.backward()
        for projection in ("q_proj", "v_proj", "in_proj_qkv", "out_proj"):
            grads = [
                param.grad for name, param in model.named_parameters()
                if f".{projection}.lora_B." in name
            ]
            assert len(grads) == 1, projection
            assert grads[0] is not None, projection
            assert torch.isfinite(grads[0]).all() and grads[0].abs().sum() > 0, projection

        for param in expert_grads:
            assert param.grad is not None
            assert torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
