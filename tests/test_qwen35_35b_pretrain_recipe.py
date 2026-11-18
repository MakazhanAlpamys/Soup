"""The 35B-A3B Base pretraining recipe (#275), checked without public weights.

Public config at 0f0813072d2358973511097385626f21fcb6d422 declares qwen3_5_moe.
The tiny CPU model checks native hybrid attention and fused expert LoRA attachment.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.config.loader import load_config_from_string
from soup_cli.recipes.catalog import RECIPES, get_recipe, search_recipes
from tests.conftest import strip_ansi

NAME = "qwen3.5-35b-a3b-pretrain"
BASE = "Qwen/Qwen3.5-35B-A3B-Base"
TEMPLATE = "qwen3.5-4b-pretrain"


def test_metadata_names_the_base_checkpoint_independently() -> None:
    recipe = get_recipe(NAME)
    assert recipe is not None
    assert recipe.model == BASE
    assert recipe.task == "pretrain"
    assert recipe.size == "35B"
    assert {"qwen", "qwen3.5", "pretrain", "continued", "moe"} <= set(recipe.tags)
    assert "Apache-2.0" in recipe.description
    assert recipe in search_recipes(query="qwen3.5", task="pretrain", size="35b")


def test_yaml_selects_the_base_text_decoder_and_fused_expert_lora() -> None:
    raw = yaml.safe_load(RECIPES[NAME].yaml_str)
    assert raw["base"] == BASE
    assert raw["task"] == "pretrain"
    assert raw["modality"] == "text"
    assert raw["training"]["moe_lora"] is True
    assert raw["training"]["lora"]["dropout"] == 0.0
    assert "moe_aux_loss_coeff" not in raw["training"]

    config = load_config_from_string(RECIPES[NAME].yaml_str)
    assert config.base == BASE
    assert config.backend == "transformers"
    assert config.modality == "text"
    assert config.training.moe_lora is True
    assert config.training.lora.dropout == 0.0
    sft = load_config_from_string(RECIPES["qwen3.5-35b-a3b-sft"].yaml_str)
    assert config.training.moe_aux_loss_coeff == sft.training.moe_aux_loss_coeff == 0.01


def test_only_the_base_and_two_moe_settings_differ_from_the_4b_template() -> None:
    mine = yaml.safe_load(RECIPES[NAME].yaml_str)
    template = yaml.safe_load(RECIPES[TEMPLATE].yaml_str)
    assert mine.pop("base") == BASE
    assert template.pop("base") == "Qwen/Qwen3.5-4B-Base"
    assert mine["training"].pop("moe_lora") is True
    assert mine["training"]["lora"].pop("dropout") == 0.0
    assert mine == template


def test_show_and_use_write_the_same_loadable_config(
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
    config = load_config_from_string(text)
    assert (config.base, config.task, config.modality) == (BASE, "pretrain", "text")


def test_plaintext_corpus_rows_reach_the_data_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from soup_cli.data.loader import load_dataset

    monkeypatch.chdir(tmp_path)
    config = load_config_from_string(RECIPES[NAME].yaml_str)
    config.data.val_split = 0
    (tmp_path / "data").mkdir()
    rows = [{"text": "Soup streams model layers."}, {"text": "A second document."}]
    (tmp_path / "data" / "corpus.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )
    loaded = list(load_dataset(config.data)["train"])
    assert sorted(row["text"] for row in loaded) == sorted(row["text"] for row in rows)


def test_native_moe_decoder_attaches_both_expert_projections_and_backpropagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    from peft import TaskType
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from soup_cli.trainer.pretrain import PretrainTrainerWrapper

    config = load_config_from_string(RECIPES[NAME].yaml_str)
    # Keep the recipe's LoRA settings; this CPU control uses tiny floating-point
    # weights in place of the published NF4 checkpoint.
    config.training.quantization = "none"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model_config = AutoConfig.for_model("qwen3_5_moe", text_config={
            "vocab_size": 32, "hidden_size": 32,
            "moe_intermediate_size": 16, "shared_expert_intermediate_size": 16,
            "num_experts": 4, "num_experts_per_tok": 2,
            "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 2,
            "head_dim": 16, "max_position_embeddings": 64, "full_attention_interval": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_conv_kernel_dim": 4, "linear_key_head_dim": 8,
            "linear_num_key_heads": 2, "linear_num_value_heads": 2,
            "linear_value_head_dim": 8, "tie_word_embeddings": False,
        })
        model = AutoModelForCausalLM.from_config(model_config).to("cpu")
        assert type(model).__name__ == "Qwen3_5MoeForCausalLM"
        loader_calls = []

        def load_tokenizer(base: str, **kwargs: Any) -> SimpleNamespace:
            assert base == BASE
            loader_calls.append("tokenizer")
            return SimpleNamespace(pad_token=None, eos_token="</s>", pad_token_id=0)

        def load_model(base: str, **kwargs: Any) -> torch.nn.Module:
            assert base == BASE and kwargs["device_map"] == "cpu"
            assert "quantization_config" not in kwargs
            loader_calls.append("model")
            return model

        monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)
        monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load_model)
        wrapper = object.__new__(PretrainTrainerWrapper)
        wrapper.config, wrapper.device, wrapper._trust_remote_code = config, "cpu", False
        wrapper._setup_transformers(config, config.training)
        model = wrapper.model
        assert loader_calls == ["tokenizer", "model"]
        adapter = model.peft_config["default"]
        assert adapter.task_type == TaskType.CAUSAL_LM
        assert adapter.r == config.training.lora.r
        assert adapter.target_parameters is not None
        assert {name.rsplit(".", 1)[-1] for name in adapter.target_parameters} == {
            "gate_up_proj", "down_proj",
        }
        assert not any("visual" in name or "vision" in name for name, _ in model.named_parameters())
        expert_parameters = [
            parameter for name, parameter in model.named_parameters()
            if ".experts." in name and ".lora_B." in name
        ]
        assert len(expert_parameters) == 4
        tokens = torch.tensor([[1, 2, 3, 4]])
        result = model(input_ids=tokens, labels=tokens)
        assert result.logits.shape == (1, 4, 32)
        assert torch.isfinite(result.loss)
        result.loss.backward()
        for projection in ("q_proj", "v_proj", "in_proj_qkv", "out_proj"):
            gradients = [
                parameter.grad for name, parameter in model.named_parameters()
                if f".{projection}.lora_B." in name
            ]
            assert len(gradients) == 1, projection
            assert gradients[0] is not None, projection
            assert torch.isfinite(gradients[0]).all() and gradients[0].abs().sum() > 0
        for parameter in expert_parameters:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
