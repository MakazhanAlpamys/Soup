"""``minimax-m3-dpo``: the #275 preference row that #1145 removed and #1393 unblocked.

#1145 dropped the recipe because DPO always built ``AutoModelForCausalLM`` and the
MiniMax-M3 checkpoint is a vision-language wrapper (``model_type: minimax_m3_vl``).
#1393 (#1707) made DPO load ``AutoModelForImageTextToText`` and the processor for
``modality: vision`` and left this recipe for its own pull request. These tests pin
the row and drive it through the real consumers, offline: a tiny LLaVA built
locally stands in for the MiniMax wrapper, as in ``test_issue1393``. They do not
qualify the 428B weights, its remote-code processor, NF4 or a multi-GPU run.
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

NAME = "minimax-m3-dpo"
BASE = "MiniMaxAI/MiniMax-M3"
SFT_SIBLING = "minimax-m3-sft"


def test_metadata_names_the_minimax_m3_preference_model_independently() -> None:
    recipe = get_recipe(NAME)
    assert recipe is not None
    assert recipe.model == BASE
    assert recipe.task == "dpo"
    assert recipe.size == "428B"
    assert {"minimax", "dpo", "preference", "moe", "large", "multi-gpu"} <= set(recipe.tags)
    description = recipe.description.lower()
    assert "minimax community license" in description
    assert "--trust-remote-code" in description
    assert recipe in search_recipes(query="minimax", task="dpo")


def test_yaml_selects_the_vision_path_and_dpo_starting_values() -> None:
    raw = yaml.safe_load(RECIPES[NAME].yaml_str)
    assert raw["base"] == BASE
    assert raw["task"] == "dpo"
    # Without it DPO builds AutoModelForCausalLM, which cannot load minimax_m3_vl (#1145).
    assert raw["modality"] == "vision"
    # SFT's and DPO's vision paths leave the minimax experts alone, so the flag is not set.
    assert "moe_lora" not in raw["training"]
    assert "moe_aux_loss_coeff" not in raw["training"]

    cfg = load_config_from_string(RECIPES[NAME].yaml_str)
    assert cfg.base == BASE
    assert cfg.task == "dpo"
    assert cfg.modality == "vision"
    assert cfg.backend == "transformers"
    assert cfg.data.train == "./data/preference_train.jsonl"
    assert cfg.data.format == "dpo"
    assert cfg.data.max_length == 4096
    assert cfg.training.epochs == 1
    assert cfg.training.lr == 5e-6
    assert cfg.training.batch_size == 1
    assert cfg.training.gradient_accumulation_steps == 16
    assert cfg.training.dpo_beta == 0.1
    assert cfg.training.quantization == "4bit"
    assert cfg.training.gradient_checkpointing is True
    assert cfg.training.lora.target_modules == "auto"
    assert cfg.output == "./output"


def test_lora_and_size_settings_come_from_the_sft_sibling() -> None:
    """The adapter the DPO row trains is the one the SFT row trains, so a user can
    run SFT then DPO on the same tower with the same rank."""
    dpo = yaml.safe_load(RECIPES[NAME].yaml_str)
    sft = yaml.safe_load(RECIPES[SFT_SIBLING].yaml_str)
    assert dpo["base"] == sft["base"]
    assert dpo["modality"] == sft["modality"]
    assert dpo["data"]["max_length"] == sft["data"]["max_length"]
    for key in (
        "epochs", "batch_size", "gradient_accumulation_steps", "quantization",
        "gradient_checkpointing", "lora",
    ):
        assert dpo["training"][key] == sft["training"][key], key


def test_the_attach_preflight_builds_the_image_text_class() -> None:
    """``soup recipes verify`` builds the class the DPO trainer loads (#1707)."""
    from soup_cli.utils.attach_preflight import loader_for

    cfg = load_config_from_string(RECIPES[NAME].yaml_str)
    assert loader_for(cfg) == ("AutoModelForImageTextToText",)


def test_auto_targets_resolve_to_the_language_tower_of_minimax_m3_vl() -> None:
    """``target_modules: auto`` on this wrapper is the #1102 regex, which never reaches
    the vision tower's own q/k/v projections."""
    import re
    from types import SimpleNamespace

    from soup_cli.utils.peft_wiring import resolve_lora_target_modules

    model = SimpleNamespace(config=SimpleNamespace(
        model_type="minimax_m3_vl", text_config=SimpleNamespace(model_type=None),
    ))
    targets = resolve_lora_target_modules(model, "auto")
    assert isinstance(targets, str), targets
    assert re.fullmatch(targets, "model.language_model.layers.3.self_attn.q_proj")
    assert re.fullmatch(targets, "model.language_model.layers.3.self_attn.o_proj")
    assert not re.fullmatch(targets, "model.vision_tower.encoder.layers.0.self_attn.q_proj")
    assert not re.fullmatch(targets, "model.language_model.layers.3.mlp.experts.0.w1")


def test_preference_rows_reach_the_trainer_with_both_completions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from soup_cli.data.loader import load_dataset

    monkeypatch.chdir(tmp_path)
    cfg = load_config_from_string(RECIPES[NAME].yaml_str)
    cfg.data.val_split = 0
    (tmp_path / "data").mkdir()
    row = {"prompt": "Name a prime number.", "chosen": "7", "rejected": "8"}
    (tmp_path / "data" / "preference_train.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8",
    )
    rows = list(load_dataset(cfg.data)["train"])
    assert len(rows) == 1
    assert rows[0]["chosen"] != rows[0]["rejected"]
    assert "7" in json.dumps(rows[0]["chosen"]) and "8" in json.dumps(rows[0]["rejected"])
    assert "Name a prime number." in json.dumps(rows[0]["prompt"])


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
    assert cfg.task == "dpo"
    assert cfg.modality == "vision"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_the_recipe_trains_one_dpo_step_on_a_vision_language_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recipe's own training block through ``DPOTrainerWrapper`` and trl's real
    ``DPOTrainer``, on a tiny LLaVA (the MiniMax-M3 wrapper shape) with text-only
    rows. Only what a CPU test cannot take is swapped: the base path, 4-bit loading,
    the 4096 limit and the accumulation steps. ``auto`` is replaced by the regex it
    resolves to for ``minimax_m3_vl``, since the stand-in reports ``llava``."""
    pytest.importorskip("torch")
    pytest.importorskip("trl")
    pytest.importorskip("datasets")
    from soup_cli.trainer.dpo import DPOTrainerWrapper
    from soup_cli.utils.peft_wiring import MOE_TEXT_LORA_TARGETS
    from tests.test_issue1393_dpo_vision_loader import _save_tiny_llava

    monkeypatch.setenv("WANDB_DISABLED", "true")
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.chdir(tmp_path)
    base = _save_tiny_llava(tmp_path / "vl")

    raw = yaml.safe_load(RECIPES[NAME].yaml_str)
    raw["base"] = base
    raw["data"]["max_length"] = 64
    raw["training"]["quantization"] = "none"
    raw["training"]["gradient_accumulation_steps"] = 1
    raw["training"]["batch_size"] = 2
    raw["training"]["lora"]["target_modules"] = MOE_TEXT_LORA_TARGETS["minimax_m3_vl"]
    raw["output"] = (tmp_path / "out").as_posix()
    cfg = load_config_from_string(yaml.safe_dump(raw))
    assert cfg.modality == "vision" and cfg.training.dpo_beta == 0.1

    rows = [
        {"prompt": "say hi", "chosen": " hi", "rejected": " no"},
        {"prompt": "say hello", "chosen": " hello world", "rejected": " no"},
        {"prompt": "hello", "chosen": " hi", "rejected": " world"},
        {"prompt": "hi", "chosen": " hello", "rejected": " no no"},
    ]
    wrapper = DPOTrainerWrapper(cfg, device="cpu")
    wrapper.setup({"train": rows})
    adapted = [n for n, m in wrapper.model.named_modules() if hasattr(m, "lora_A")]
    assert adapted and all("language_model" in n for n in adapted), adapted
    assert not any("vision_tower" in n for n in adapted)
    result = wrapper.train()
    assert result["total_steps"] >= 1, result
    assert (tmp_path / "out" / "adapter_config.json").is_file()
