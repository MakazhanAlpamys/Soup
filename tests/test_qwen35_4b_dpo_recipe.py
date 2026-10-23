"""Independent recipe metadata, YAML, preference-loader and CLI contracts."""

import json

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.config.loader import load_config_from_string
from soup_cli.data.loader import load_dataset
from soup_cli.recipes.catalog import RECIPES, get_recipe
from tests.conftest import strip_ansi

NAME = "qwen3.5-4b-dpo"
BASE = "Qwen/Qwen3.5-4B"


def test_metadata_names_the_4b_dpo_model_independently():
    recipe = get_recipe(NAME)
    assert recipe is not None
    assert recipe.model == BASE
    assert recipe.task == "dpo"
    assert recipe.size == "4B"
    assert {"qwen", "qwen3.5", "dpo", "preference"} <= set(recipe.tags)
    assert recipe.model == RECIPES["qwen3.5-4b-sft"].model


def test_yaml_names_the_base_and_selects_text_preference_training():
    raw = yaml.safe_load(RECIPES[NAME].yaml_str)
    assert raw["base"] == BASE
    assert raw["task"] == "dpo"
    assert raw["modality"] == "text"
    assert raw["data"]["format"] == "dpo"
    cfg = load_config_from_string(RECIPES[NAME].yaml_str)
    assert cfg.base == BASE and cfg.task == "dpo" and cfg.modality == "text"
    assert cfg.data.train == "./data/preference_train.jsonl"
    assert cfg.data.max_length == 4096
    assert cfg.training.lr == 5e-6
    assert cfg.training.epochs == 3
    assert cfg.training.dpo_beta == 0.1
    assert cfg.training.batch_size == "auto"
    assert cfg.training.quantization == "4bit"
    assert cfg.training.lora.r == 16
    assert cfg.training.lora.alpha == 32


def test_recipe_loader_keeps_chosen_and_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    row = {"prompt": "Q", "chosen": "good", "rejected": "bad"}
    (data_dir / "preference_train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    cfg = load_config_from_string(RECIPES[NAME].yaml_str)
    cfg.data.val_split = 0
    assert load_dataset(cfg.data)["train"] == [row]


def test_show_and_use_write_a_loadable_text_dpo_config(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    shown = runner.invoke(app, ["recipes", "show", NAME])
    assert shown.exit_code == 0, strip_ansi(shown.output)
    assert BASE in strip_ansi(shown.output)
    used = runner.invoke(app, ["recipes", "use", NAME, "--yes"])
    assert used.exit_code == 0, strip_ansi(used.output)
    text = (tmp_path / "soup.yaml").read_text(encoding="utf-8")
    cfg = load_config_from_string(text)
    assert cfg.base == BASE and cfg.task == "dpo" and cfg.modality == "text"
    assert cfg.data.format == "dpo"


def test_native_qwen35_text_decoder_loads_and_backpropagates_on_cpu():
    torch = pytest.importorskip("torch")
    from transformers import AutoConfig, AutoModelForCausalLM

    # Same native model family as the public 4B config, deliberately tiny random
    # weights: this tests the text consumer, not the full 4B checkpoint or NF4.
    config = AutoConfig.for_model("qwen3_5", text_config={
        "vocab_size": 32, "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2,
        "head_dim": 16, "max_position_embeddings": 64, "full_attention_interval": 1,
        "layer_types": ["full_attention"], "tie_word_embeddings": False,
    })
    model = AutoModelForCausalLM.from_config(config).to("cpu")
    assert not any("visual" in name for name, _ in model.named_parameters())
    tokens = torch.tensor([[1, 2, 3, 4]])
    result = model(input_ids=tokens, labels=tokens)
    assert result.logits.shape == (1, 4, 32)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert any(
        param.grad is not None and param.grad.abs().sum() > 0 for param in model.parameters()
    )
