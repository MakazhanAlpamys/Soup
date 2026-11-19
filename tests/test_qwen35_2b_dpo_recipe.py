"""The 2B recipe's independent IDs and real text-preference consumers."""

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

NAME = "qwen3.5-2b-dpo"
BASE = "Qwen/Qwen3.5-2B"


def test_metadata_selects_the_2b_preference_model_independently() -> None:
    recipe = get_recipe(NAME)
    assert recipe is not None
    assert recipe.model == BASE
    assert recipe.task == "dpo"
    assert recipe.size == "2B"
    assert {"qwen", "qwen3.5", "dpo", "preference", "small", "edge"} <= set(recipe.tags)
    assert recipe in search_recipes(query="qwen3.5", task="dpo", size="2b")


def test_yaml_selects_the_2b_text_decoder_and_inherited_dpo_settings() -> None:
    raw = yaml.safe_load(RECIPES[NAME].yaml_str)
    assert raw["base"] == BASE
    assert raw["task"] == "dpo"
    assert raw["modality"] == "text"
    assert raw["data"]["format"] == "dpo"

    cfg = load_config_from_string(RECIPES[NAME].yaml_str)
    assert cfg.base == BASE
    assert cfg.task == "dpo"
    assert cfg.modality == "text"
    assert cfg.data.train == "./data/preference_train.jsonl"
    assert cfg.data.max_length == 4096
    assert cfg.training.lr == 5e-6
    assert cfg.training.epochs == 3
    assert cfg.training.dpo_beta == 0.1
    assert cfg.training.batch_size == "auto"
    assert cfg.training.quantization == "4bit"
    assert cfg.training.lora.r == 16
    assert cfg.training.lora.alpha == 32
    assert cfg.training.lora.target_modules == "auto"
    assert cfg.output == "./output"


def test_preference_loader_preserves_both_completions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = [
        {"prompt": "Name a prime", "chosen": "Three", "rejected": "Four"},
        {"prompt": "Add one and two", "chosen": "Three", "rejected": "Two"},
    ]
    (data_dir / "preference_train.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    cfg = load_config_from_string(RECIPES[NAME].yaml_str)
    cfg.data.val_split = 0
    assert load_dataset(cfg.data)["train"] == rows


def test_show_and_use_write_the_same_loadable_text_dpo_config(
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
    assert cfg.task == "dpo"
    assert cfg.modality == "text"
    assert cfg.data.format == "dpo"


def test_native_text_decoder_attaches_recipe_lora_and_backpropagates_on_cpu() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from peft import get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    from soup_cli.utils.attach_preflight import loader_for, trainer_lora_config

    cfg = load_config_from_string(RECIPES[NAME].yaml_str)
    assert loader_for(cfg) == ("AutoModelForCausalLM",)
    # The public 2B decoder mixes linear and full attention. Keep both native
    # block kinds, with tiny random weights and the pure-PyTorch CPU fallback.
    # This qualifies the text/LoRA consumer, not public 2B weights or NF4.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        config = AutoConfig.for_model("qwen3_5", text_config={
            "vocab_size": 32, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 2,
            "head_dim": 16, "max_position_embeddings": 64, "full_attention_interval": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_conv_kernel_dim": 4, "linear_key_head_dim": 8,
            "linear_num_key_heads": 2, "linear_num_value_heads": 2,
            "linear_value_head_dim": 8, "tie_word_embeddings": False,
        })
        model = AutoModelForCausalLM.from_config(config).to("cpu")
        lora_config, _targets = trainer_lora_config(model, cfg, None)
        model = get_peft_model(model, lora_config)
        assert not any("visual" in name for name, _ in model.named_parameters())
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
            assert grads[0] is not None and grads[0].abs().sum() > 0, projection
