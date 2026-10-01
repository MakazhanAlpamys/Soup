"""#1214, LLaMA-Factory and Unsloth halves: every preference objective the source tool
documents either migrates to its named Soup task or is refused by name.

#1254 settled the axolotl migrator. On ``main`` the other two still took a default:
LLaMA-Factory ``pref_loss: ipo`` loaded as ``task: dpo`` and ``hinge`` / ``kto_pair``
did the same with no warning; the Unsloth migrator kept ``task: sft`` for ``CPOTrainer``,
``RewardTrainer``, ``BCOTrainer`` and ``OnlineDPOTrainer``, and dropped every
hyperparameter passed through ``SFTConfig`` because only ``TrainingArguments`` and five
``*Config`` classes were read.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.loader import load_config_from_string
from soup_cli.migrate.common import config_to_yaml
from soup_cli.migrate.llamafactory import migrate_llamafactory
from soup_cli.migrate.unsloth import migrate_unsloth


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text))


def _loaded_task(result: dict) -> str:
    return load_config_from_string(config_to_yaml(result)).task


# ---------------------------------------------------------------------------
# LLaMA-Factory: stage: dpo + pref_loss
# ---------------------------------------------------------------------------

_LF_BASE = (
    "model_name_or_path: meta-llama/Llama-3.1-8B-Instruct\n"
    "stage: dpo\n"
    "finetuning_type: lora\n"
    "lora_rank: 16\n"
    "dataset: dpo_data\n"
    "cutoff_len: 2048\n"
    "per_device_train_batch_size: 2\n"
    "num_train_epochs: 1\n"
    "learning_rate: 5e-6\n"
    "output_dir: ./output_dpo\n"
)


def _lf(tmp_path: Path, extra: str) -> dict:
    cfg = tmp_path / "lf.yaml"
    cfg.write_text(_LF_BASE + extra, encoding="utf-8")
    return migrate_llamafactory(cfg)


class TestLlamaFactoryPrefLoss:
    @pytest.mark.parametrize(
        "pref_loss,soup_task",
        [("sigmoid", "dpo"), ("ipo", "ipo"), ("orpo", "orpo"), ("simpo", "simpo")],
    )
    def test_every_mapped_pref_loss_loads_as_its_soup_task(
        self, tmp_path: Path, pref_loss: str, soup_task: str
    ) -> None:
        result = _lf(tmp_path, f"pref_loss: {pref_loss}\n")
        assert _loaded_task(result) == soup_task
        assert result["data"]["format"] == "dpo"

    @pytest.mark.parametrize("pref_loss", ["hinge", "kto_pair", "bogus"])
    def test_a_pref_loss_without_a_soup_task_is_refused_by_name(
        self, tmp_path: Path, pref_loss: str
    ) -> None:
        with pytest.raises(ValueError, match=pref_loss) as info:
            _lf(tmp_path, f"pref_loss: {pref_loss}\n")
        assert "ipo" in str(info.value) and "simpo" in str(info.value)  # the supported list

    def test_control_stage_dpo_without_pref_loss_is_plain_dpo(self, tmp_path: Path) -> None:
        assert _loaded_task(_lf(tmp_path, "")) == "dpo"

    def test_pref_loss_outside_stage_dpo_is_ignored_as_llamafactory_does(
        self, tmp_path: Path
    ) -> None:
        # LLaMA-Factory reads pref_loss only in its dpo stage; a stray value under
        # stage: sft must neither change the task nor be refused.
        cfg = tmp_path / "lf.yaml"
        cfg.write_text(
            _LF_BASE.replace("stage: dpo", "stage: sft") + "pref_loss: hinge\n", encoding="utf-8"
        )
        assert _loaded_task(migrate_llamafactory(cfg)) == "sft"

    def test_cli_refuses_hinge_and_names_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "lf.yaml").write_text(_LF_BASE + "pref_loss: hinge\n", encoding="utf-8")
        result = CliRunner().invoke(
            app, ["migrate", "--from", "llamafactory", "lf.yaml", "--dry-run"]
        )
        assert result.exit_code == 1, result.output
        assert "hinge" in _plain(result.output)


# ---------------------------------------------------------------------------
# Unsloth notebooks: TRL trainer classes and *Config hyperparameters
# ---------------------------------------------------------------------------


def _notebook(trainer_cell: str) -> dict:
    return {
        "cells": [
            {
                "cell_type": "code",
                "source": [
                    "from unsloth import FastLanguageModel\n",
                    "model, tokenizer = FastLanguageModel.from_pretrained(\n",
                    "    model_name='meta-llama/Llama-3.1-8B-Instruct',\n",
                    "    max_seq_length=2048,\n",
                    "    load_in_4bit=True,\n",
                    ")\n",
                    "model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32)\n",
                ],
            },
            {"cell_type": "code", "source": [trainer_cell]},
        ]
    }


def _unsloth(tmp_path: Path, trainer_cell: str) -> dict:
    nb = tmp_path / "nb.ipynb"
    nb.write_text(json.dumps(_notebook(trainer_cell)), encoding="utf-8")
    return migrate_unsloth(nb)


_TRAINERS = [
    # (trainer class, config class, loss kwarg on the config, Soup task)
    ("SFTTrainer", "SFTConfig", "", "sft"),
    ("DPOTrainer", "DPOConfig", "", "dpo"),
    ("GRPOTrainer", "GRPOConfig", "", "grpo"),
    ("KTOTrainer", "KTOConfig", "", "kto"),
    ("ORPOTrainer", "ORPOConfig", "", "orpo"),
    ("RewardTrainer", "RewardConfig", "", "reward_model"),
    ("BCOTrainer", "BCOConfig", "", "bco"),
    ("OnlineDPOTrainer", "OnlineDPOConfig", "", "online_dpo"),
    ("CPOTrainer", "CPOConfig", "loss_type='simpo', ", "simpo"),
]


class TestUnslothTrainerClasses:
    @pytest.mark.parametrize("trainer,config,loss_kwarg,soup_task", _TRAINERS)
    def test_every_trainer_class_migrates_to_its_soup_task_and_reads_its_config(
        self, tmp_path: Path, trainer: str, config: str, loss_kwarg: str, soup_task: str
    ) -> None:
        cell = (
            f"from trl import {trainer}, {config}\n"
            f"trainer = {trainer}(model=model, train_dataset=dataset,\n"
            f"    args={config}({loss_kwarg}per_device_train_batch_size=2, num_train_epochs=3,\n"
            f"        learning_rate=2e-4, output_dir='outputs'))\n"
        )
        result = _unsloth(tmp_path, cell)
        assert result["task"] == soup_task
        assert result["training"]["batch_size"] == 2
        assert result["training"]["epochs"] == 3
        assert result["training"]["lr"] == 2e-4
        assert result["output"] == "outputs"

    @pytest.mark.parametrize(
        "trainer,config,loss_kwarg,soup_task",
        [row for row in _TRAINERS if row[3] not in ("online_dpo",)],
    )
    def test_the_migrated_config_loads_with_that_task(
        self, tmp_path: Path, trainer: str, config: str, loss_kwarg: str, soup_task: str
    ) -> None:
        cell = (
            f"from trl import {trainer}, {config}\n"
            f"trainer = {trainer}(model=model, train_dataset=dataset,\n"
            f"    args={config}({loss_kwarg}per_device_train_batch_size=2))\n"
        )
        assert _loaded_task(_unsloth(tmp_path, cell)) == soup_task

    @pytest.mark.parametrize(
        "trainer,soup_task",
        [(t, task) for t, _c, loss, task in _TRAINERS if not loss],
    )
    def test_the_trainer_class_alone_sets_the_task_with_plain_training_arguments(
        self, tmp_path: Path, trainer: str, soup_task: str
    ) -> None:
        # No *Config class in sight: the trainer table must carry the task by itself.
        cell = (
            f"from trl import {trainer}\n"
            "from transformers import TrainingArguments\n"
            f"trainer = {trainer}(model=model, train_dataset=dataset,\n"
            "    args=TrainingArguments(per_device_train_batch_size=2, output_dir='o'))\n"
        )
        result = _unsloth(tmp_path, cell)
        assert result["task"] == soup_task
        assert result["training"]["batch_size"] == 2

    def test_online_dpo_loads_once_a_judge_is_set_and_warns_that_none_was_carried(
        self, tmp_path: Path
    ) -> None:
        cell = (
            "from trl import OnlineDPOTrainer, OnlineDPOConfig\n"
            "trainer = OnlineDPOTrainer(model=model, judge=judge, train_dataset=dataset,\n"
            "    args=OnlineDPOConfig(per_device_train_batch_size=2))\n"
        )
        result = _unsloth(tmp_path, cell)
        assert result["task"] == "online_dpo"
        assert any("judge" in w for w in result["_warnings"])
        result["training"]["online_dpo_judge"] = "ollama://llama3.1"
        assert _loaded_task(result) == "online_dpo"

    @pytest.mark.parametrize("loss_kwarg", ["", "loss_type='sigmoid', ", "loss_type='hinge', "])
    def test_cpo_without_the_simpo_loss_is_refused_by_name(
        self, tmp_path: Path, loss_kwarg: str
    ) -> None:
        cell = (
            "from trl import CPOTrainer, CPOConfig\n"
            "trainer = CPOTrainer(model=model, train_dataset=dataset,\n"
            f"    args=CPOConfig({loss_kwarg}per_device_train_batch_size=2))\n"
        )
        with pytest.raises(ValueError, match="CPOTrainer") as info:
            _unsloth(tmp_path, cell)
        assert "simpo" in str(info.value)

    def test_an_unrelated_cpo_config_elsewhere_does_not_decide_the_trainer(
        self, tmp_path: Path
    ) -> None:
        # CodeRabbit on #1214: a leftover experiment cell with loss_type="simpo" must not
        # make the trainer that actually runs (default CPOConfig) migrate as simpo.
        cell = (
            "from trl import CPOTrainer, CPOConfig\n"
            "unused = CPOConfig(loss_type='simpo', per_device_train_batch_size=8)\n"
            "trainer = CPOTrainer(model=model, train_dataset=dataset, args=CPOConfig())\n"
        )
        with pytest.raises(ValueError, match="CPOTrainer"):
            _unsloth(tmp_path, cell)

    def test_cpo_config_assigned_to_a_name_then_passed_as_args_counts(
        self, tmp_path: Path
    ) -> None:
        cell = (
            "from trl import CPOTrainer, CPOConfig\n"
            "training_args = CPOConfig(loss_type='simpo', per_device_train_batch_size=2)\n"
            "trainer = CPOTrainer(model=model, train_dataset=dataset, args=training_args)\n"
        )
        result = _unsloth(tmp_path, cell)
        assert result["task"] == "simpo" and result["training"]["batch_size"] == 2

    def test_a_config_bound_to_the_name_only_after_the_trainer_does_not_count(
        self, tmp_path: Path
    ) -> None:
        # CodeRabbit on #1214, second round: cells run top to bottom, so the trainer
        # saw the first binding; the simpo one came later.
        cell = (
            "from trl import CPOTrainer, CPOConfig\n"
            "training_args = CPOConfig()\n"
            "trainer = CPOTrainer(model=model, train_dataset=dataset, args=training_args)\n"
            "training_args = CPOConfig(loss_type='simpo')\n"
        )
        with pytest.raises(ValueError, match="CPOTrainer"):
            _unsloth(tmp_path, cell)

    def test_the_latest_binding_before_the_trainer_wins(self, tmp_path: Path) -> None:
        cell = (
            "from trl import CPOTrainer, CPOConfig\n"
            "training_args = CPOConfig()\n"
            "training_args = CPOConfig(loss_type='simpo')\n"
            "trainer = CPOTrainer(model=model, train_dataset=dataset, args=training_args)\n"
            "training_args = CPOConfig(loss_type='hinge')\n"
        )
        assert _unsloth(tmp_path, cell)["task"] == "simpo"

    def test_cpo_loss_type_on_the_trainer_call_counts_too(self, tmp_path: Path) -> None:
        cell = (
            "from trl import CPOTrainer\n"
            "trainer = CPOTrainer(model=model, train_dataset=dataset, loss_type='simpo')\n"
        )
        assert _unsloth(tmp_path, cell)["task"] == "simpo"

    def test_sft_config_hyperparameters_are_read(self, tmp_path: Path) -> None:
        # The issue's third acceptance item, verbatim: the same kwargs under
        # TrainingArguments already survived; under SFTConfig they were dropped.
        cell = (
            "from trl import SFTTrainer, SFTConfig\n"
            "trainer = SFTTrainer(model=model, train_dataset=dataset,\n"
            "    args=SFTConfig(per_device_train_batch_size=2, learning_rate=2e-4,\n"
            "        num_train_epochs=3))\n"
        )
        training = _unsloth(tmp_path, cell)["training"]
        assert (training["batch_size"], training["lr"], training["epochs"]) == (2, 2e-4, 3)

    def test_control_training_arguments_still_read(self, tmp_path: Path) -> None:
        cell = (
            "from trl import SFTTrainer\n"
            "from transformers import TrainingArguments\n"
            "trainer = SFTTrainer(model=model, train_dataset=dataset,\n"
            "    args=TrainingArguments(per_device_train_batch_size=4, output_dir='o'))\n"
        )
        result = _unsloth(tmp_path, cell)
        assert result["task"] == "sft"
        assert result["training"]["batch_size"] == 4 and result["output"] == "o"

    def test_an_unknown_trainer_class_still_defaults_to_sft(self, tmp_path: Path) -> None:
        # Not every call in a notebook is a TRL trainer; a class the table does not
        # list is not evidence of a different objective.
        cell = "trainer = MyTrainer(model=model, train_dataset=dataset)\n"
        assert _unsloth(tmp_path, cell)["task"] == "sft"
