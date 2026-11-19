"""``qwen3.5-2b-pretrain``: continued pre-training on the published 2B Base checkpoint (#275).

Public config verified separately at b1485b2fa6dfa1287294f269f5fb618e03d52d7c:
``Qwen3_5ForConditionalGeneration`` (``qwen3_5``), 24 text layers, hidden size 2048.
Tests stay offline. Tiny random weights qualify neither the public weights nor a
pre-training run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.config.loader import load_config_from_string
from soup_cli.recipes.catalog import RECIPES, get_recipe, search_recipes
from tests.conftest import strip_ansi

NAME = "qwen3.5-2b-pretrain"
BASE = "Qwen/Qwen3.5-2B-Base"
TEMPLATE = "qwen3.5-4b-pretrain"


def test_metadata_names_the_base_checkpoint_independently() -> None:
    recipe = get_recipe(NAME)
    assert recipe is not None
    assert recipe.model == BASE
    assert recipe.model.endswith("-Base")
    assert recipe.task == "pretrain"
    assert recipe.size == "2B"
    assert {"qwen", "qwen3.5", "pretrain", "continued"} <= set(recipe.tags)
    assert "Apache-2.0" in recipe.description
    assert recipe in search_recipes(query="qwen3.5", task="pretrain", size="2b")


def test_yaml_selects_the_base_model_and_continued_pretraining_values() -> None:
    raw = yaml.safe_load(RECIPES[NAME].yaml_str)
    assert raw["base"] == BASE
    assert raw["task"] == "pretrain"
    assert raw["modality"] == "text"

    cfg = load_config_from_string(RECIPES[NAME].yaml_str)
    assert cfg.base == BASE
    assert cfg.task == "pretrain"
    assert cfg.modality == "text"
    assert cfg.data.train == "./data/corpus.jsonl"
    assert cfg.data.format == "plaintext"
    assert cfg.data.max_length == 4096
    assert cfg.training.epochs == 1
    assert cfg.training.lr == 1e-4
    assert cfg.training.batch_size == "auto"
    assert cfg.training.gradient_accumulation_steps == 8
    assert cfg.training.lora.r == 32
    assert cfg.training.lora.alpha == 64
    assert cfg.training.lora.target_modules == "auto"
    assert cfg.training.quantization == "4bit"
    assert cfg.output == "./output"


def test_settings_are_the_shipped_4b_pretrain_values() -> None:
    """Only the base differs from the 4B pretrain row, so the two stay comparable."""
    mine = yaml.safe_load(RECIPES[NAME].yaml_str)
    template = yaml.safe_load(RECIPES[TEMPLATE].yaml_str)
    assert mine.pop("base") == BASE
    assert template.pop("base") == "Qwen/Qwen3.5-4B-Base"
    assert mine == template


def test_plaintext_corpus_rows_reach_the_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from soup_cli.data.loader import load_dataset

    monkeypatch.chdir(tmp_path)
    cfg = load_config_from_string(RECIPES[NAME].yaml_str)
    cfg.data.val_split = 0
    (tmp_path / "data").mkdir()
    rows = [{"text": "Soup streams model layers."}, {"text": "A second document."}]
    (tmp_path / "data" / "corpus.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )
    loaded = list(load_dataset(cfg.data)["train"])
    assert len(loaded) == 2
    assert "Soup streams model layers." in json.dumps(loaded)


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
    cfg = load_config_from_string(text)
    assert cfg.base == BASE
    assert cfg.task == "pretrain"


def test_native_text_decoder_attaches_recipe_lora_and_backpropagates_on_cpu() -> None:
    import torch
    from peft import get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    from soup_cli.utils.attach_preflight import loader_for, trainer_lora_config

    cfg = load_config_from_string(RECIPES[NAME].yaml_str)
    assert loader_for(cfg) == ("AutoModelForCausalLM",)
    # The Base checkpoint is published as the qwen3_5 wrapper, whose text decoder
    # mixes linear and full attention. Keep both native block kinds with tiny
    # random weights and the pure-PyTorch CPU fallback. This qualifies the
    # text/LoRA consumer and the causal-LM loss pre-training uses, not the public
    # weights or NF4.
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
        assert type(model).__name__ == "Qwen3_5ForCausalLM"
        lora_config, _targets = trainer_lora_config(model, cfg, None)
        model = get_peft_model(model, lora_config)
        assert not any("visual" in n or "vision" in n for n, _ in model.named_parameters())
        tokens = torch.tensor([[1, 2, 3, 4]])
        result = model(input_ids=tokens, labels=tokens)
        assert result.logits.shape == (1, 4, 32)
        assert torch.isfinite(result.loss)
        result.loss.backward()
        for projection in ("q_proj", "v_proj", "in_proj_qkv", "out_proj"):
            grads = [
                param.grad for n, param in model.named_parameters()
                if f".{projection}.lora_B." in n
            ]
            assert len(grads) == 1, projection
            assert grads[0] is not None, projection
            assert torch.isfinite(grads[0]).all() and grads[0].abs().sum() > 0, projection
