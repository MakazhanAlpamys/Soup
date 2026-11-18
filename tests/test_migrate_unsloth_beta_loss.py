"""The Unsloth migrator reads ``beta`` and ``loss_type`` off the final trainer's config,
or names them as not migrated.

Before: both keywords were in the "handled" set for every config class, but ``beta``
was mapped only for DPO and KTO and ``loss_type`` only for CPO. ``DPOConfig(loss_type=
"ipo")`` migrated to plain DPO, ``GRPOConfig(beta=0.0)`` trained with Soup's KL of 0.1,
an ``ORPOConfig`` / ``BCOConfig`` ``beta`` was lost, and no warning said so.
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
    "model_name='unsloth/Llama-3.2-1B-Instruct', max_seq_length=2048, load_in_4bit=True)\n"
    "model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16)\n"
)


def _migrate(tmp_path: Path, *cells: str):
    notebook = {
        "cells": [{"cell_type": "code", "source": c, "metadata": {}, "outputs": []}
                  for c in (_LOAD, *cells)],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
    }
    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps(notebook), encoding="utf-8")
    result = migrate_unsloth(path)
    return result, load_config_from_string(config_to_yaml(result))


def _trainer(trainer: str, config: str, kwargs: str) -> str:
    return (f"trainer = {trainer}(model=model, train_dataset=dataset,"
            f" args={config}({kwargs}per_device_train_batch_size=2, output_dir='outputs'))\n")


def _unmapped(result: dict) -> str:
    return " ".join(w for w in result["_warnings"] if "not migrated" in w)


# ---------------------------------------------------------------------------
# DPOConfig loss_type
# ---------------------------------------------------------------------------


def test_an_ipo_loss_migrates_to_task_ipo_with_beta_as_tau(tmp_path: Path) -> None:
    result, cfg = _migrate(tmp_path, _trainer("DPOTrainer", "DPOConfig",
                                              "loss_type='ipo', beta=0.5, "))
    assert cfg.task == "ipo"
    assert cfg.training.ipo_tau == 0.5
    assert cfg.data.format == "dpo"
    assert "dpo_beta" not in result["training"]
    assert "loss_type" not in _unmapped(result)


def test_an_ipo_config_bound_before_the_trainer_counts_too(tmp_path: Path) -> None:
    # The config call is visited before the trainer call here, so the task the
    # trainer sets during the walk must not undo the loss.
    cell = ("cfg = DPOConfig(loss_type='ipo', beta=0.3, output_dir='outputs')\n"
            "trainer = DPOTrainer(model=model, train_dataset=dataset, args=cfg)\n")
    _, cfg = _migrate(tmp_path, cell)
    assert (cfg.task, cfg.training.ipo_tau) == ("ipo", 0.3)


@pytest.mark.parametrize("loss", ["'sigmoid'", "['sigmoid']"])
def test_the_sigmoid_loss_stays_dpo(tmp_path: Path, loss: str) -> None:
    result, cfg = _migrate(tmp_path, _trainer("DPOTrainer", "DPOConfig",
                                              f"loss_type={loss}, beta=0.2, "))
    assert (cfg.task, cfg.training.dpo_beta) == ("dpo", 0.2)
    assert "loss_type" not in _unmapped(result)


@pytest.mark.parametrize("loss", ["'hinge'", "'robust'", "['sigmoid', 'sft']"])
def test_a_dpo_loss_with_no_soup_task_is_refused_by_name(tmp_path: Path, loss: str) -> None:
    with pytest.raises(ValueError, match="DPOTrainer with loss_type=") as info:
        _migrate(tmp_path, _trainer("DPOTrainer", "DPOConfig", f"loss_type={loss}, "))
    assert f"loss_type={loss};" in str(info.value)


# ---------------------------------------------------------------------------
# beta
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trainer,config,field", [
    ("DPOTrainer", "DPOConfig", "dpo_beta"),
    ("KTOTrainer", "KTOConfig", "kto_beta"),
    ("ORPOTrainer", "ORPOConfig", "orpo_beta"),
    ("BCOTrainer", "BCOConfig", "bco_beta"),
    ("GRPOTrainer", "GRPOConfig", "grpo_beta"),
    ("OnlineDPOTrainer", "OnlineDPOConfig", "dpo_beta"),
])
def test_beta_lands_in_the_field_the_task_trains_with(
    tmp_path: Path, trainer: str, config: str, field: str
) -> None:
    result, cfg = _migrate(tmp_path, _trainer(trainer, config, "beta=0.25, "))
    assert getattr(cfg.training, field) == 0.25
    assert "beta" not in _unmapped(result)


def test_a_grpo_beta_of_zero_turns_the_kl_penalty_off(tmp_path: Path) -> None:
    # TRL's GRPOConfig default; Soup's grpo_beta default is 0.1.
    _, cfg = _migrate(tmp_path, _trainer("GRPOTrainer", "GRPOConfig", "beta=0.0, "))
    assert cfg.training.grpo_beta == 0.0


def test_a_beta_bound_to_a_module_level_name_is_read(tmp_path: Path) -> None:
    # The walker resolves a keyword through the notebook's plain assignments.
    _, cfg = _migrate(tmp_path, "BETA = 0.25\n",
                      _trainer("GRPOTrainer", "GRPOConfig", "beta=BETA, "))
    assert cfg.training.grpo_beta == 0.25


def test_a_beta_and_loss_type_that_cannot_be_read_are_named_not_guessed(
    tmp_path: Path,
) -> None:
    # An attribute of an object is not a value the walker can read: both keywords are
    # named in the "not migrated" warning, nothing is guessed, and the migration still
    # completes (no refusal, so the CLI exits 0).
    result, cfg = _migrate(tmp_path, _trainer(
        "GRPOTrainer", "GRPOConfig", "beta=settings.beta, loss_type=settings.loss, "))
    unmapped = _unmapped(result)
    assert "beta" in unmapped
    assert "loss_type" in unmapped
    assert cfg.task == "grpo"
    assert cfg.training.grpo_beta == 0.1  # Soup's default, not a guessed value


def test_a_simpo_beta_has_no_soup_field_and_is_named(tmp_path: Path) -> None:
    result, cfg = _migrate(tmp_path, _trainer("CPOTrainer", "CPOConfig",
                                              "loss_type='simpo', beta=2.0, "))
    assert cfg.task == "simpo"
    unmapped = _unmapped(result)
    assert "beta" in unmapped
    assert "loss_type" not in unmapped


@pytest.mark.parametrize("trainer,config", [
    ("SFTTrainer", "SFTConfig"),
    ("RewardTrainer", "RewardConfig"),
])
def test_a_beta_on_a_task_with_no_beta_field_is_named(
    tmp_path: Path, trainer: str, config: str
) -> None:
    result, _ = _migrate(tmp_path, _trainer(trainer, config, "beta=0.1, "))
    assert "beta" in _unmapped(result)


@pytest.mark.parametrize("trainer,config,loss", [
    ("GRPOTrainer", "GRPOConfig", "dr_grpo"),
    ("KTOTrainer", "KTOConfig", "apo_zero_unpaired"),
    ("SFTTrainer", "SFTConfig", "dft"),
])
def test_a_loss_type_that_no_soup_field_takes_is_named(
    tmp_path: Path, trainer: str, config: str, loss: str
) -> None:
    result, _ = _migrate(tmp_path, _trainer(trainer, config, f"loss_type='{loss}', "))
    assert "loss_type" in _unmapped(result)
