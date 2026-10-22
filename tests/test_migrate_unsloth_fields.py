"""#1807 — the Unsloth migrator takes ``base`` from the model that trains and maps
the trainer fields it used to drop.

Before: every ``from_pretrained`` call overwrote ``base`` in source order, so the
stock notebook's ``if False:`` reload of the local ``lora_model`` directory won
over the training model; and ``gradient_accumulation_steps``, ``warmup_steps``,
``weight_decay`` and ``seed`` were never read from ``TrainingArguments`` or the
TRL ``*Config`` classes (an effective batch of 32 became 8).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soup_cli.config.loader import load_config_from_string
from soup_cli.migrate.common import config_to_yaml
from soup_cli.migrate.unsloth import migrate_unsloth

_LOAD = (
    "from unsloth import FastLanguageModel\n"
    "model, tokenizer = FastLanguageModel.from_pretrained("
    "model_name='unsloth/Meta-Llama-3.1-8B-bnb-4bit', max_seq_length=2048, load_in_4bit=True)\n"
    "model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16)\n"
)
_TRAIN = (
    "from trl import SFTTrainer\nfrom transformers import TrainingArguments\n"
    "trainer = SFTTrainer(model=model, args=TrainingArguments(per_device_train_batch_size=2,"
    " gradient_accumulation_steps=16, warmup_steps=5, num_train_epochs=1, learning_rate=2e-4,"
    " weight_decay=0.1, seed=3407, output_dir='outputs'))\n"
)
_RELOAD_KW = (
    "if False:\n    model, tokenizer = FastLanguageModel.from_pretrained(model_name='lora_model',"
    " max_seq_length=2048, load_in_4bit=True)\n"
)
_RELOAD_POS = (
    "if False:\n    from peft import AutoPeftModelForCausalLM\n"
    "    model = AutoPeftModelForCausalLM.from_pretrained('lora_model')\n"
    "    tokenizer = AutoTokenizer.from_pretrained('lora_model')\n"
)


def _notebook(*cells: str) -> dict:
    return {
        "cells": [{"cell_type": "code", "source": c, "metadata": {}, "outputs": []} for c in cells],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
    }


def _migrate(tmp_path: Path, *cells: str):
    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps(_notebook(*cells)), encoding="utf-8")
    result = migrate_unsloth(path)
    cfg = load_config_from_string(config_to_yaml(result))
    return result, cfg


# ---------------------------------------------------------------------------
# The issue's reproduction
# ---------------------------------------------------------------------------


def test_the_stock_notebook_migrates_to_the_training_model_and_arguments(tmp_path: Path) -> None:
    result, cfg = _migrate(tmp_path, _LOAD, _TRAIN, _RELOAD_KW)
    assert cfg.base == "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
    assert cfg.training.gradient_accumulation_steps == 16
    assert cfg.training.weight_decay == 0.1
    assert cfg.training.seed == 3407
    assert cfg.training.batch_size == 2
    assert any("warmup_steps" in w and "warmup_ratio" in w for w in result["_warnings"])


# ---------------------------------------------------------------------------
# base
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reload", [_RELOAD_KW, _RELOAD_POS], ids=["keyword", "positional"])
def test_a_reload_under_if_false_never_becomes_base(tmp_path: Path, reload) -> None:
    _, cfg = _migrate(tmp_path, _LOAD, _TRAIN, reload)
    assert cfg.base == "unsloth/Meta-Llama-3.1-8B-bnb-4bit"


def test_a_top_level_reload_after_training_does_not_become_base_either(tmp_path: Path) -> None:
    # The first call in source order trains; a later one at top level is the
    # inference reload the stock notebook runs after saving.
    reload = ("model, tokenizer = FastLanguageModel.from_pretrained(model_name='lora_model',"
              " max_seq_length=2048, load_in_4bit=True)\n")
    _, cfg = _migrate(tmp_path, _LOAD, _TRAIN, reload)
    assert cfg.base == "unsloth/Meta-Llama-3.1-8B-bnb-4bit"


def test_a_reload_before_the_training_load_is_skipped_when_dead(tmp_path: Path) -> None:
    _, cfg = _migrate(tmp_path, _RELOAD_KW, _LOAD, _TRAIN)
    assert cfg.base == "unsloth/Meta-Llama-3.1-8B-bnb-4bit"


def test_a_notebook_whose_only_load_is_dead_still_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="from_pretrained"):
        _migrate(tmp_path, _RELOAD_KW, _TRAIN)


def test_the_positional_form_of_the_training_load_is_read(tmp_path: Path) -> None:
    load = ("from unsloth import FastLanguageModel\n"
            "model, tokenizer = FastLanguageModel.from_pretrained('org/base',"
            " max_seq_length=512)\n")
    _, cfg = _migrate(tmp_path, load, _TRAIN, _RELOAD_KW)
    assert cfg.base == "org/base"
    assert cfg.data.max_length == 512


# ---------------------------------------------------------------------------
# Trainer arguments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config_cls", ["TrainingArguments", "SFTConfig", "DPOConfig", "GRPOConfig"]
)
def test_the_four_fields_are_read_from_every_config_class(tmp_path: Path, config_cls) -> None:
    trainer = {"TrainingArguments": "SFTTrainer", "SFTConfig": "SFTTrainer",
               "DPOConfig": "DPOTrainer", "GRPOConfig": "GRPOTrainer"}[config_cls]
    train = (f"trainer = {trainer}(model=model, args={config_cls}(per_device_train_batch_size=2,"
             " gradient_accumulation_steps=8, weight_decay=0.05, seed=7, warmup_ratio=0.1,"
             " output_dir='outputs'))\n")
    _, cfg = _migrate(tmp_path, _LOAD, train)
    assert cfg.training.gradient_accumulation_steps == 8
    assert cfg.training.weight_decay == 0.05
    assert cfg.training.seed == 7
    assert cfg.training.warmup_ratio == 0.1


def test_warmup_steps_is_named_as_having_no_soup_equivalent(tmp_path: Path) -> None:
    result, cfg = _migrate(tmp_path, _LOAD, _TRAIN)
    assert cfg.training.warmup_ratio == 0.03  # Soup's default, and the warning says so
    hits = [w for w in result["_warnings"] if "warmup_steps=5" in w]
    assert len(hits) == 1 and "training.warmup_ratio" in hits[0]


def test_the_fields_absent_from_the_notebook_keep_soup_defaults(tmp_path: Path) -> None:
    train = ("trainer = SFTTrainer(model=model, args=TrainingArguments("
             "per_device_train_batch_size=2, output_dir='outputs'))\n")
    result, cfg = _migrate(tmp_path, _LOAD, train)
    for key in ("gradient_accumulation_steps", "weight_decay", "seed", "warmup_ratio"):
        assert key not in result["training"]
    assert cfg.training.seed is None


def test_only_the_final_stage_supplies_the_fields(tmp_path: Path) -> None:
    warmup = ("trainer = SFTTrainer(model=model, args=SFTConfig(per_device_train_batch_size=1,"
              " gradient_accumulation_steps=2, seed=1, output_dir='warm'))\n")
    rl = ("trainer = GRPOTrainer(model=model, args=GRPOConfig(per_device_train_batch_size=4,"
          " gradient_accumulation_steps=32, seed=9, output_dir='rl'))\n")
    _, cfg = _migrate(tmp_path, _LOAD, warmup, rl)
    assert cfg.task == "grpo"
    assert (cfg.training.gradient_accumulation_steps, cfg.training.seed) == (32, 9)


# ---------------------------------------------------------------------------
# Nothing is dropped silently
# ---------------------------------------------------------------------------


def test_unmapped_trainer_arguments_are_listed_in_one_warning(tmp_path: Path) -> None:
    train = ("trainer = SFTTrainer(model=model, args=TrainingArguments("
             "per_device_train_batch_size=2, logging_steps=1, fp16=False, bf16=True,"
             " report_to='none', output_dir='outputs'))\n")
    result, _ = _migrate(tmp_path, _LOAD, train)
    listing = [w for w in result["_warnings"] if "not migrated" in w and "TrainingArguments" in w]
    assert len(listing) == 1, result["_warnings"]
    for key in ("bf16", "fp16", "logging_steps", "report_to"):
        assert key in listing[0]
    assert "per_device_train_batch_size" not in listing[0]
    assert "output_dir" not in listing[0]


def test_unmapped_peft_arguments_are_listed_too(tmp_path: Path) -> None:
    load = (
        "from unsloth import FastLanguageModel\n"
        "model, tokenizer = FastLanguageModel.from_pretrained(model_name='org/base')\n"
        "model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16, bias='none',"
        " use_gradient_checkpointing='unsloth', random_state=3407)\n"
    )
    result, _ = _migrate(tmp_path, load, _TRAIN)
    listing = [w for w in result["_warnings"] if "not migrated" in w and "get_peft_model" in w]
    assert len(listing) == 1, result["_warnings"]
    for key in ("bias", "random_state", "use_gradient_checkpointing"):
        assert key in listing[0]
    assert " r," not in listing[0] and "lora_alpha" not in listing[0]


def test_a_fully_mapped_notebook_has_no_unmapped_listing(tmp_path: Path) -> None:
    train = ("trainer = SFTTrainer(model=model, args=TrainingArguments("
             "per_device_train_batch_size=2, gradient_accumulation_steps=4, num_train_epochs=1,"
             " learning_rate=2e-4, weight_decay=0.0, seed=1, output_dir='outputs'))\n")
    result, _ = _migrate(tmp_path, _LOAD, train)
    assert not any("not migrated" in w for w in result["_warnings"])
