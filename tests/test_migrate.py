"""Tests for soup migrate — config import from LLaMA-Factory, Axolotl, Unsloth."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()

# ---------------------------------------------------------------------------
# Fixtures — sample configs
# ---------------------------------------------------------------------------

LLAMA_FACTORY_SFT = """\
model_name_or_path: meta-llama/Llama-3.1-8B-Instruct
stage: sft
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: all
dataset: alpaca_en
template: llama3
cutoff_len: 2048
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
num_train_epochs: 3
learning_rate: 2e-4
lr_scheduler_type: cosine
warmup_ratio: 0.03
output_dir: ./output
quantization_bit: 4
bf16: true
"""

LLAMA_FACTORY_DPO = """\
model_name_or_path: meta-llama/Llama-3.1-8B-Instruct
stage: dpo
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
dataset: dpo_data
template: llama3
cutoff_len: 2048
per_device_train_batch_size: 2
num_train_epochs: 1
learning_rate: 5e-6
output_dir: ./output_dpo
quantization_bit: 4
pref_beta: 0.1
"""

LLAMA_FACTORY_KTO = """\
model_name_or_path: meta-llama/Llama-3.1-8B-Instruct
stage: kto
finetuning_type: lora
lora_rank: 16
dataset: kto_data
cutoff_len: 2048
per_device_train_batch_size: 2
num_train_epochs: 3
learning_rate: 1e-5
output_dir: ./output_kto
pref_beta: 0.1
"""

LLAMA_FACTORY_PPO = """\
model_name_or_path: meta-llama/Llama-3.1-8B-Instruct
stage: ppo
finetuning_type: lora
lora_rank: 16
dataset: ppo_prompts
cutoff_len: 2048
per_device_train_batch_size: 2
num_train_epochs: 1
learning_rate: 1e-6
output_dir: ./output_ppo
reward_model: ./reward_model
"""

LLAMA_FACTORY_PRETRAIN = """\
model_name_or_path: meta-llama/Llama-3.1-8B
stage: pt
finetuning_type: lora
lora_rank: 32
dataset: corpus
cutoff_len: 4096
per_device_train_batch_size: 2
num_train_epochs: 1
learning_rate: 1e-5
output_dir: ./output_pretrain
"""

LLAMA_FACTORY_RM = """\
model_name_or_path: meta-llama/Llama-3.1-8B-Instruct
stage: rm
finetuning_type: lora
lora_rank: 16
dataset: rm_data
cutoff_len: 2048
per_device_train_batch_size: 2
num_train_epochs: 1
learning_rate: 1e-5
output_dir: ./output_rm
"""

LLAMA_FACTORY_ORPO = """\
model_name_or_path: meta-llama/Llama-3.1-8B-Instruct
stage: dpo
pref_loss: orpo
finetuning_type: lora
lora_rank: 16
dataset: pref_data
cutoff_len: 2048
per_device_train_batch_size: 2
num_train_epochs: 3
learning_rate: 1e-5
output_dir: ./output_orpo
pref_beta: 0.1
"""

LLAMA_FACTORY_SIMPO = """\
model_name_or_path: meta-llama/Llama-3.1-8B-Instruct
stage: dpo
pref_loss: simpo
finetuning_type: lora
lora_rank: 16
dataset: pref_data
cutoff_len: 2048
per_device_train_batch_size: 2
num_train_epochs: 3
learning_rate: 1e-5
output_dir: ./output_simpo
"""

LLAMA_FACTORY_NEFTUNE = """\
model_name_or_path: meta-llama/Llama-3.1-8B-Instruct
stage: sft
finetuning_type: lora
lora_rank: 16
dataset: alpaca_en
cutoff_len: 2048
per_device_train_batch_size: 4
num_train_epochs: 3
learning_rate: 2e-4
output_dir: ./output
neftune_noise_alpha: 5.0
use_dora: true
loraplus_lr_ratio: 16.0
"""

AXOLOTL_SFT = """\
base_model: meta-llama/Llama-3.1-8B-Instruct
datasets:
  - path: ./data/train.jsonl
    type: alpaca
sequence_len: 2048
adapter: lora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - v_proj
micro_batch_size: 4
gradient_accumulation_steps: 4
num_epochs: 3
learning_rate: 2e-4
optimizer: adamw_torch
lr_scheduler: cosine
warmup_ratio: 0.03
output_dir: ./output
"""

AXOLOTL_DPO = """\
base_model: meta-llama/Llama-3.1-8B-Instruct
datasets:
  - path: ./data/dpo_train.jsonl
    type: sharegpt
sequence_len: 2048
adapter: qlora
lora_r: 16
lora_alpha: 32
micro_batch_size: 2
num_epochs: 1
learning_rate: 5e-6
output_dir: ./output_dpo
rl: dpo
"""

AXOLOTL_GRPO = """\
base_model: meta-llama/Llama-3.1-8B-Instruct
datasets:
  - path: ./data/reasoning.jsonl
    type: sharegpt
sequence_len: 4096
adapter: lora
lora_r: 16
lora_alpha: 32
micro_batch_size: 2
num_epochs: 3
learning_rate: 1e-5
output_dir: ./output_grpo
rl: grpo
flash_attention: true
"""

AXOLOTL_MULTI_DATASET = """\
base_model: meta-llama/Llama-3.1-8B-Instruct
datasets:
  - path: ./data/train1.jsonl
    type: alpaca
  - path: ./data/train2.jsonl
    type: sharegpt
sequence_len: 2048
adapter: lora
lora_r: 16
micro_batch_size: 4
num_epochs: 3
learning_rate: 2e-4
output_dir: ./output
"""

AXOLOTL_SAMPLE_PACKING = """\
base_model: meta-llama/Llama-3.1-8B-Instruct
datasets:
  - path: ./data/train.jsonl
    type: alpaca
sequence_len: 2048
adapter: lora
lora_r: 16
micro_batch_size: 4
num_epochs: 3
learning_rate: 2e-4
output_dir: ./output
sample_packing: true
"""

AXOLOTL_LOAD_IN_4BIT = """\
base_model: meta-llama/Llama-3.1-8B-Instruct
datasets:
  - path: ./data/train.jsonl
    type: alpaca
sequence_len: 2048
adapter: lora
lora_r: 16
micro_batch_size: 4
num_epochs: 3
learning_rate: 2e-4
output_dir: ./output
load_in_4bit: true
"""

AXOLOTL_LORA_TARGET_LINEAR = """\
base_model: meta-llama/Llama-3.1-8B-Instruct
datasets:
  - path: ./data/train.jsonl
    type: alpaca
sequence_len: 2048
adapter: lora
lora_r: 16
lora_target_linear: true
micro_batch_size: 4
num_epochs: 3
learning_rate: 2e-4
output_dir: ./output
"""

AXOLOTL_ADAMW_8BIT = """\
base_model: meta-llama/Llama-3.1-8B-Instruct
datasets:
  - path: ./data/train.jsonl
    type: alpaca
sequence_len: 2048
adapter: lora
lora_r: 16
micro_batch_size: 4
num_epochs: 3
learning_rate: 2e-4
optimizer: adamw_8bit
output_dir: ./output
"""

UNSLOTH_SFT_NOTEBOOK = {
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
            ],
        },
        {
            "cell_type": "code",
            "source": [
                "model = FastLanguageModel.get_peft_model(\n",
                "    model,\n",
                "    r=16,\n",
                "    lora_alpha=32,\n",
                "    lora_dropout=0.05,\n",
                "    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],\n",
                "    use_dora=False,\n",
                ")\n",
            ],
        },
        {
            "cell_type": "code",
            "source": [
                "from trl import SFTTrainer\n",
                "from transformers import TrainingArguments\n",
                "trainer = SFTTrainer(\n",
                "    model=model,\n",
                "    train_dataset=dataset,\n",
                "    args=TrainingArguments(\n",
                "        per_device_train_batch_size=4,\n",
                "        num_train_epochs=3,\n",
                "        learning_rate=2e-4,\n",
                "        optim='adamw_torch',\n",
                "        lr_scheduler_type='cosine',\n",
                "        output_dir='./output',\n",
                "    ),\n",
                ")\n",
            ],
        },
    ],
}

UNSLOTH_DPO_NOTEBOOK = {
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
            ],
        },
        {
            "cell_type": "code",
            "source": [
                "model = FastLanguageModel.get_peft_model(\n",
                "    model,\n",
                "    r=16,\n",
                "    lora_alpha=32,\n",
                "    lora_dropout=0.0,\n",
                ")\n",
            ],
        },
        {
            "cell_type": "code",
            "source": [
                "from trl import DPOTrainer, DPOConfig\n",
                "trainer = DPOTrainer(\n",
                "    model=model,\n",
                "    train_dataset=dataset,\n",
                "    args=DPOConfig(\n",
                "        per_device_train_batch_size=2,\n",
                "        num_train_epochs=1,\n",
                "        learning_rate=5e-6,\n",
                "        beta=0.1,\n",
                "        output_dir='./output_dpo',\n",
                "    ),\n",
                ")\n",
            ],
        },
    ],
}

UNSLOTH_GRPO_NOTEBOOK = {
    "cells": [
        {
            "cell_type": "code",
            "source": [
                "from unsloth import FastLanguageModel\n",
                "model, tokenizer = FastLanguageModel.from_pretrained(\n",
                "    model_name='deepseek-ai/DeepSeek-R1-Distill-Llama-8B',\n",
                "    max_seq_length=4096,\n",
                "    load_in_4bit=True,\n",
                ")\n",
            ],
        },
        {
            "cell_type": "code",
            "source": [
                "model = FastLanguageModel.get_peft_model(\n",
                "    model,\n",
                "    r=32,\n",
                "    lora_alpha=64,\n",
                ")\n",
            ],
        },
        {
            "cell_type": "code",
            "source": [
                "from trl import GRPOTrainer, GRPOConfig\n",
                "trainer = GRPOTrainer(\n",
                "    model=model,\n",
                "    train_dataset=dataset,\n",
                "    args=GRPOConfig(\n",
                "        per_device_train_batch_size=2,\n",
                "        num_train_epochs=3,\n",
                "        learning_rate=1e-5,\n",
                "        output_dir='./output_grpo',\n",
                "    ),\n",
                ")\n",
            ],
        },
    ],
}

UNSLOTH_RSLORA_NOTEBOOK = {
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
            ],
        },
        {
            "cell_type": "code",
            "source": [
                "model = FastLanguageModel.get_peft_model(\n",
                "    model,\n",
                "    r=16,\n",
                "    lora_alpha=32,\n",
                "    use_rslora=True,\n",
                ")\n",
            ],
        },
        {
            "cell_type": "code",
            "source": [
                "from trl import SFTTrainer\n",
                "from transformers import TrainingArguments\n",
                "trainer = SFTTrainer(\n",
                "    model=model,\n",
                "    train_dataset=dataset,\n",
                "    args=TrainingArguments(\n",
                "        per_device_train_batch_size=4,\n",
                "        num_train_epochs=3,\n",
                "        learning_rate=2e-4,\n",
                "        output_dir='./output',\n",
                "    ),\n",
                ")\n",
            ],
        },
    ],
}

UNSLOTH_PACKING_NOTEBOOK = {
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
            ],
        },
        {
            "cell_type": "code",
            "source": [
                "model = FastLanguageModel.get_peft_model(\n",
                "    model,\n",
                "    r=16,\n",
                "    lora_alpha=32,\n",
                ")\n",
            ],
        },
        {
            "cell_type": "code",
            "source": [
                "from trl import SFTTrainer\n",
                "from transformers import TrainingArguments\n",
                "trainer = SFTTrainer(\n",
                "    model=model,\n",
                "    train_dataset=dataset,\n",
                "    packing=True,\n",
                "    args=TrainingArguments(\n",
                "        per_device_train_batch_size=4,\n",
                "        num_train_epochs=3,\n",
                "        learning_rate=2e-4,\n",
                "        output_dir='./output',\n",
                "    ),\n",
                ")\n",
            ],
        },
    ],
}


# ---------------------------------------------------------------------------
# LLaMA-Factory migration tests
# ---------------------------------------------------------------------------

class TestLlamaFactoryMigration:
    """LLaMA-Factory → Soup config migration."""

    def test_sft_basic(self, tmp_path):
        """SFT config maps correctly."""
        from soup_cli.migrate.llamafactory import migrate_llamafactory

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(LLAMA_FACTORY_SFT, encoding="utf-8")
        result = migrate_llamafactory(cfg_file)
        assert result["base"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert result["task"] == "sft"
        assert result["training"]["lora"]["r"] == 16
        assert result["training"]["lora"]["alpha"] == 32
        assert result["training"]["lora"]["dropout"] == 0.05
        assert result["training"]["quantization"] == "4bit"
        assert result["training"]["batch_size"] == 4
        assert result["training"]["epochs"] == 3
        assert result["training"]["lr"] == 2e-4
        assert result["training"]["scheduler"] == "cosine"
        assert result["training"]["warmup_ratio"] == 0.03
        assert result["training"]["gradient_accumulation_steps"] == 4
        assert result["data"]["max_length"] == 2048
        assert result["output"] == "./output"

    def test_sft_lora_target_all(self, tmp_path):
        """lora_target: all → auto (Soup auto-detects)."""
        from soup_cli.migrate.llamafactory import migrate_llamafactory

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(LLAMA_FACTORY_SFT, encoding="utf-8")
        result = migrate_llamafactory(cfg_file)
        assert result["training"]["lora"]["target_modules"] == "auto"

    def test_dpo_config(self, tmp_path):
        """DPO stage maps to task: dpo with dpo_beta."""
        from soup_cli.migrate.llamafactory import migrate_llamafactory

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(LLAMA_FACTORY_DPO, encoding="utf-8")
        result = migrate_llamafactory(cfg_file)
        assert result["task"] == "dpo"
        assert result["training"]["dpo_beta"] == 0.1
        assert result["data"]["format"] == "dpo"

    def test_kto_config(self, tmp_path):
        """KTO stage maps to task: kto with kto_beta."""
        from soup_cli.migrate.llamafactory import migrate_llamafactory

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(LLAMA_FACTORY_KTO, encoding="utf-8")
        result = migrate_llamafactory(cfg_file)
        assert result["task"] == "kto"
        assert result["training"]["kto_beta"] == 0.1
        assert result["data"]["format"] == "kto"

    def test_ppo_config(self, tmp_path):
        """PPO stage maps to task: ppo with reward_model."""
        from soup_cli.migrate.llamafactory import migrate_llamafactory

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(LLAMA_FACTORY_PPO, encoding="utf-8")
        result = migrate_llamafactory(cfg_file)
        assert result["task"] == "ppo"
        assert result["training"]["reward_model"] == "./reward_model"

    def test_pretrain_config(self, tmp_path):
        """pt stage maps to task: pretrain, format: plaintext."""
        from soup_cli.migrate.llamafactory import migrate_llamafactory

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(LLAMA_FACTORY_PRETRAIN, encoding="utf-8")
        result = migrate_llamafactory(cfg_file)
        assert result["task"] == "pretrain"
        assert result["data"]["format"] == "plaintext"

    def test_reward_model_config(self, tmp_path):
        """rm stage maps to task: reward_model."""
        from soup_cli.migrate.llamafactory import migrate_llamafactory

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(LLAMA_FACTORY_RM, encoding="utf-8")
        result = migrate_llamafactory(cfg_file)
        assert result["task"] == "reward_model"
        assert result["data"]["format"] == "dpo"

    def test_orpo_via_pref_loss(self, tmp_path):
        """stage: dpo + pref_loss: orpo → task: orpo."""
        from soup_cli.migrate.llamafactory import migrate_llamafactory

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(LLAMA_FACTORY_ORPO, encoding="utf-8")
        result = migrate_llamafactory(cfg_file)
        assert result["task"] == "orpo"
        assert result["training"]["orpo_beta"] == 0.1

    def test_simpo_via_pref_loss(self, tmp_path):
        """stage: dpo + pref_loss: simpo → task: simpo."""
        from soup_cli.migrate.llamafactory import migrate_llamafactory

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(LLAMA_FACTORY_SIMPO, encoding="utf-8")
        result = migrate_llamafactory(cfg_file)
        assert result["task"] == "simpo"

    def test_neftune_and_dora(self, tmp_path):
        """NEFTune, DoRA, LoRA+ are mapped."""
        from soup_cli.migrate.llamafactory import migrate_llamafactory

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(LLAMA_FACTORY_NEFTUNE, encoding="utf-8")
        result = migrate_llamafactory(cfg_file)
        assert result["training"]["lora"]["use_dora"] is True
        assert result["training"]["loraplus_lr_ratio"] == 16.0

    def test_dataset_warning(self, tmp_path):
        """Dataset name (not path) triggers a warning."""
        from soup_cli.migrate.llamafactory import migrate_llamafactory

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(LLAMA_FACTORY_SFT, encoding="utf-8")
        result = migrate_llamafactory(cfg_file)
        assert any("dataset" in w.lower() for w in result.get("_warnings", []))

    def test_empty_config(self, tmp_path):
        """Empty YAML raises ValueError."""
        from soup_cli.migrate.llamafactory import migrate_llamafactory

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            migrate_llamafactory(cfg_file)

    def test_missing_model(self, tmp_path):
        """Config without model_name_or_path raises ValueError."""
        from soup_cli.migrate.llamafactory import migrate_llamafactory

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("stage: sft\n", encoding="utf-8")
        with pytest.raises(ValueError, match="model_name_or_path"):
            migrate_llamafactory(cfg_file)

    def test_full_finetuning_no_lora(self, tmp_path):
        """finetuning_type: full → no lora section."""
        from soup_cli.migrate.llamafactory import migrate_llamafactory

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "model_name_or_path: meta-llama/Llama-3.1-8B\n"
            "stage: sft\n"
            "finetuning_type: full\n"
            "dataset: data\n"
            "num_train_epochs: 1\n"
            "output_dir: ./output\n",
            encoding="utf-8",
        )
        result = migrate_llamafactory(cfg_file)
        assert "lora" not in result.get("training", {})

    def test_quantization_8bit(self, tmp_path):
        """quantization_bit: 8 → quantization: 8bit."""
        from soup_cli.migrate.llamafactory import migrate_llamafactory

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "model_name_or_path: meta-llama/Llama-3.1-8B\n"
            "stage: sft\n"
            "finetuning_type: lora\n"
            "lora_rank: 16\n"
            "dataset: data\n"
            "num_train_epochs: 1\n"
            "output_dir: ./output\n"
            "quantization_bit: 8\n",
            encoding="utf-8",
        )
        result = migrate_llamafactory(cfg_file)
        assert result["training"]["quantization"] == "8bit"

    def test_round_trip_valid_config(self, tmp_path):
        """Migrated config loads as valid SoupConfig."""
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.migrate.common import config_to_yaml
        from soup_cli.migrate.llamafactory import migrate_llamafactory

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(LLAMA_FACTORY_SFT, encoding="utf-8")
        result = migrate_llamafactory(cfg_file)
        yaml_str = config_to_yaml(result)
        soup_config = load_config_from_string(yaml_str)
        assert soup_config.base == "meta-llama/Llama-3.1-8B-Instruct"
        assert soup_config.task == "sft"


# ---------------------------------------------------------------------------
# Axolotl migration tests
# ---------------------------------------------------------------------------

class TestAxolotlMigration:
    """Axolotl → Soup config migration."""

    def test_sft_basic(self, tmp_path):
        """SFT config maps correctly."""
        from soup_cli.migrate.axolotl import migrate_axolotl

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(AXOLOTL_SFT, encoding="utf-8")
        result = migrate_axolotl(cfg_file)
        assert result["base"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert result["task"] == "sft"
        assert result["training"]["lora"]["r"] == 16
        assert result["training"]["lora"]["alpha"] == 32
        assert result["training"]["lora"]["dropout"] == 0.05
        assert result["training"]["lora"]["target_modules"] == ["q_proj", "v_proj"]
        assert result["training"]["batch_size"] == 4
        assert result["training"]["epochs"] == 3
        assert result["training"]["lr"] == 2e-4
        assert result["training"]["optimizer"] == "adamw_torch"
        assert result["training"]["scheduler"] == "cosine"
        assert result["data"]["train"] == "./data/train.jsonl"
        assert result["data"]["format"] == "alpaca"
        assert result["data"]["max_length"] == 2048
        assert result["output"] == "./output"

    def test_dpo_with_qlora(self, tmp_path):
        """rl: dpo → task: dpo, adapter: qlora → quantization: 4bit."""
        from soup_cli.migrate.axolotl import migrate_axolotl

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(AXOLOTL_DPO, encoding="utf-8")
        result = migrate_axolotl(cfg_file)
        assert result["task"] == "dpo"
        assert result["training"]["quantization"] == "4bit"
        assert result["data"]["format"] == "dpo"

    def test_grpo_with_flash_attn(self, tmp_path):
        """rl: grpo → task: grpo, flash_attention → use_flash_attn."""
        from soup_cli.migrate.axolotl import migrate_axolotl

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(AXOLOTL_GRPO, encoding="utf-8")
        result = migrate_axolotl(cfg_file)
        assert result["task"] == "grpo"
        assert result["training"]["use_flash_attn"] is True

    def test_multi_dataset_warning(self, tmp_path):
        """Multiple datasets triggers a warning (Soup uses single dataset)."""
        from soup_cli.migrate.axolotl import migrate_axolotl

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(AXOLOTL_MULTI_DATASET, encoding="utf-8")
        result = migrate_axolotl(cfg_file)
        assert result["data"]["train"] == "./data/train1.jsonl"
        assert any("multiple datasets" in w.lower() for w in result.get("_warnings", []))

    def test_sample_packing_warning(self, tmp_path):
        """sample_packing generates unsupported feature warning."""
        from soup_cli.migrate.axolotl import migrate_axolotl

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(AXOLOTL_SAMPLE_PACKING, encoding="utf-8")
        result = migrate_axolotl(cfg_file)
        assert any("sample_packing" in w.lower() for w in result.get("_warnings", []))

    def test_load_in_4bit(self, tmp_path):
        """load_in_4bit: true → quantization: 4bit."""
        from soup_cli.migrate.axolotl import migrate_axolotl

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(AXOLOTL_LOAD_IN_4BIT, encoding="utf-8")
        result = migrate_axolotl(cfg_file)
        assert result["training"]["quantization"] == "4bit"

    def test_lora_target_linear(self, tmp_path):
        """lora_target_linear: true → target_modules: auto."""
        from soup_cli.migrate.axolotl import migrate_axolotl

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(AXOLOTL_LORA_TARGET_LINEAR, encoding="utf-8")
        result = migrate_axolotl(cfg_file)
        assert result["training"]["lora"]["target_modules"] == "auto"

    def test_adamw_8bit_mapping(self, tmp_path):
        """adamw_8bit → adamw_bnb_8bit."""
        from soup_cli.migrate.axolotl import migrate_axolotl

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(AXOLOTL_ADAMW_8BIT, encoding="utf-8")
        result = migrate_axolotl(cfg_file)
        assert result["training"]["optimizer"] == "adamw_bnb_8bit"

    def test_empty_config(self, tmp_path):
        """Empty YAML raises ValueError."""
        from soup_cli.migrate.axolotl import migrate_axolotl

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            migrate_axolotl(cfg_file)

    def test_missing_base_model(self, tmp_path):
        """Config without base_model raises ValueError."""
        from soup_cli.migrate.axolotl import migrate_axolotl

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("num_epochs: 3\n", encoding="utf-8")
        with pytest.raises(ValueError, match="base_model"):
            migrate_axolotl(cfg_file)

    def test_round_trip_valid_config(self, tmp_path):
        """Migrated config loads as valid SoupConfig."""
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.migrate.axolotl import migrate_axolotl
        from soup_cli.migrate.common import config_to_yaml

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(AXOLOTL_SFT, encoding="utf-8")
        result = migrate_axolotl(cfg_file)
        yaml_str = config_to_yaml(result)
        soup_config = load_config_from_string(yaml_str)
        assert soup_config.base == "meta-llama/Llama-3.1-8B-Instruct"


# ---------------------------------------------------------------------------
# Unsloth migration tests
# ---------------------------------------------------------------------------

class TestUnslothMigration:
    """Unsloth notebook → Soup config migration."""

    def test_sft_notebook(self, tmp_path):
        """SFT notebook extracted correctly."""
        from soup_cli.migrate.unsloth import migrate_unsloth

        nb_file = tmp_path / "finetune.ipynb"
        nb_file.write_text(json.dumps(UNSLOTH_SFT_NOTEBOOK), encoding="utf-8")
        result = migrate_unsloth(nb_file)
        assert result["base"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert result["task"] == "sft"
        assert result["training"]["lora"]["r"] == 16
        assert result["training"]["lora"]["alpha"] == 32
        assert result["training"]["lora"]["dropout"] == 0.05
        assert result["training"]["lora"]["target_modules"] == [
            "q_proj", "v_proj", "k_proj", "o_proj"
        ]
        assert result["training"]["quantization"] == "4bit"
        assert result["training"]["batch_size"] == 4
        assert result["training"]["epochs"] == 3
        assert result["training"]["lr"] == 2e-4
        assert result["training"]["optimizer"] == "adamw_torch"
        assert result["training"]["scheduler"] == "cosine"
        assert result["data"]["max_length"] == 2048
        assert result["output"] == "./output"

    def test_dpo_notebook(self, tmp_path):
        """DPO notebook extracted correctly."""
        from soup_cli.migrate.unsloth import migrate_unsloth

        nb_file = tmp_path / "dpo.ipynb"
        nb_file.write_text(json.dumps(UNSLOTH_DPO_NOTEBOOK), encoding="utf-8")
        result = migrate_unsloth(nb_file)
        assert result["task"] == "dpo"
        assert result["training"]["dpo_beta"] == 0.1
        assert result["training"]["batch_size"] == 2
        assert result["data"]["format"] == "dpo"

    def test_grpo_notebook(self, tmp_path):
        """GRPO notebook extracted correctly."""
        from soup_cli.migrate.unsloth import migrate_unsloth

        nb_file = tmp_path / "grpo.ipynb"
        nb_file.write_text(json.dumps(UNSLOTH_GRPO_NOTEBOOK), encoding="utf-8")
        result = migrate_unsloth(nb_file)
        assert result["task"] == "grpo"
        assert result["base"] == "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        assert result["training"]["lora"]["r"] == 32
        assert result["data"]["max_length"] == 4096

    def test_rslora_propagated(self, tmp_path):
        """use_rslora=True is propagated to lora config."""
        from soup_cli.migrate.unsloth import migrate_unsloth

        nb_file = tmp_path / "rslora.ipynb"
        nb_file.write_text(json.dumps(UNSLOTH_RSLORA_NOTEBOOK), encoding="utf-8")
        result = migrate_unsloth(nb_file)
        assert result["training"]["lora"]["use_rslora"] is True

    def test_packing_warning(self, tmp_path):
        """packing=True generates unsupported warning."""
        from soup_cli.migrate.unsloth import migrate_unsloth

        nb_file = tmp_path / "packing.ipynb"
        nb_file.write_text(json.dumps(UNSLOTH_PACKING_NOTEBOOK), encoding="utf-8")
        result = migrate_unsloth(nb_file)
        assert any("packing" in w.lower() for w in result.get("_warnings", []))

    def test_empty_notebook(self, tmp_path):
        """Notebook with no code cells raises ValueError."""
        from soup_cli.migrate.unsloth import migrate_unsloth

        nb_file = tmp_path / "empty.ipynb"
        nb_file.write_text(json.dumps({"cells": []}), encoding="utf-8")
        with pytest.raises(ValueError, match="(?i)no .* found"):
            migrate_unsloth(nb_file)

    def test_invalid_json(self, tmp_path):
        """Non-JSON file raises ValueError."""
        from soup_cli.migrate.unsloth import migrate_unsloth

        nb_file = tmp_path / "bad.ipynb"
        nb_file.write_text("not json", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON"):
            migrate_unsloth(nb_file)

    def test_round_trip_valid_config(self, tmp_path):
        """Migrated config loads as valid SoupConfig."""
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.migrate.common import config_to_yaml
        from soup_cli.migrate.unsloth import migrate_unsloth

        nb_file = tmp_path / "finetune.ipynb"
        nb_file.write_text(json.dumps(UNSLOTH_SFT_NOTEBOOK), encoding="utf-8")
        result = migrate_unsloth(nb_file)
        yaml_str = config_to_yaml(result)
        soup_config = load_config_from_string(yaml_str)
        assert soup_config.base == "meta-llama/Llama-3.1-8B-Instruct"


# ---------------------------------------------------------------------------
# Common utilities tests
# ---------------------------------------------------------------------------

class TestCommon:
    """Common migration utilities."""

    def test_config_to_yaml_basic(self):
        """config_to_yaml generates valid YAML."""
        from soup_cli.migrate.common import config_to_yaml

        config = {
            "base": "meta-llama/Llama-3.1-8B-Instruct",
            "task": "sft",
            "data": {"train": "./data.jsonl", "format": "auto", "max_length": 2048},
            "training": {
                "epochs": 3,
                "lr": 2e-4,
                "lora": {"r": 16, "alpha": 32},
            },
            "output": "./output",
        }
        yaml_str = config_to_yaml(config)
        assert "base:" in yaml_str
        assert "meta-llama/Llama-3.1-8B-Instruct" in yaml_str
        assert "task:" in yaml_str
        assert "sft" in yaml_str

    def test_config_to_yaml_strips_warnings(self):
        """_warnings key is stripped from YAML output."""
        from soup_cli.migrate.common import config_to_yaml

        config = {
            "base": "model",
            "task": "sft",
            "data": {"train": "./data.jsonl"},
            "output": "./output",
            "_warnings": ["some warning"],
        }
        yaml_str = config_to_yaml(config)
        assert "_warnings" not in yaml_str

    def test_validate_input_path(self, tmp_path, monkeypatch):
        """Input path must exist and be under cwd."""
        from soup_cli.migrate.common import validate_input_path

        monkeypatch.chdir(tmp_path)
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("test", encoding="utf-8")
        # Valid path
        validated = validate_input_path(cfg_file)
        assert validated.exists()

    def test_validate_input_path_traversal(self, tmp_path, monkeypatch):
        """Path traversal outside cwd is blocked."""
        from soup_cli.migrate.common import validate_input_path

        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="outside"):
            validate_input_path(Path("/etc/passwd"))

    def test_validate_input_path_not_exists(self, tmp_path, monkeypatch):
        """Non-existent file raises ValueError."""
        from soup_cli.migrate.common import validate_input_path

        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="not found"):
            validate_input_path(tmp_path / "nonexistent.yaml")

    def test_validate_output_path(self, tmp_path, monkeypatch):
        """Output path must be under cwd."""
        from soup_cli.migrate.common import validate_output_path

        monkeypatch.chdir(tmp_path)
        validated = validate_output_path(Path("soup.yaml"))
        assert str(validated).endswith("soup.yaml")

    def test_validate_output_path_traversal(self, tmp_path, monkeypatch):
        """Output path traversal outside cwd is blocked."""
        from soup_cli.migrate.common import validate_output_path

        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="outside"):
            validate_output_path(Path("/tmp/evil.yaml"))


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------

class TestMigrateCLI:
    """CLI tests for soup migrate command."""

    def test_help(self):
        """soup migrate --help shows usage."""
        result = runner.invoke(app, ["migrate", "--help"])
        assert result.exit_code == 0
        assert "migrate" in result.output.lower()

    def test_llamafactory_sft(self, tmp_path, monkeypatch):
        """Full CLI: soup migrate --from llamafactory config.yaml."""
        monkeypatch.chdir(tmp_path)
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(LLAMA_FACTORY_SFT, encoding="utf-8")
        result = runner.invoke(app, [
            "migrate",
            "--from", "llamafactory",
            str(cfg_file),
            "--output", "soup.yaml",
            "--yes",
        ])
        assert result.exit_code == 0
        out_path = tmp_path / "soup.yaml"
        assert out_path.exists()
        content = out_path.read_text(encoding="utf-8")
        assert "meta-llama/Llama-3.1-8B-Instruct" in content

    def test_axolotl_sft(self, tmp_path, monkeypatch):
        """Full CLI: soup migrate --from axolotl config.yaml."""
        monkeypatch.chdir(tmp_path)
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(AXOLOTL_SFT, encoding="utf-8")
        result = runner.invoke(app, [
            "migrate",
            "--from", "axolotl",
            str(cfg_file),
            "--output", "soup.yaml",
            "--yes",
        ])
        assert result.exit_code == 0
        out_path = tmp_path / "soup.yaml"
        assert out_path.exists()

    def test_unsloth_notebook(self, tmp_path, monkeypatch):
        """Full CLI: soup migrate --from unsloth finetune.ipynb."""
        monkeypatch.chdir(tmp_path)
        nb_file = tmp_path / "finetune.ipynb"
        nb_file.write_text(json.dumps(UNSLOTH_SFT_NOTEBOOK), encoding="utf-8")
        result = runner.invoke(app, [
            "migrate",
            "--from", "unsloth",
            str(nb_file),
            "--output", "soup.yaml",
            "--yes",
        ])
        assert result.exit_code == 0
        out_path = tmp_path / "soup.yaml"
        assert out_path.exists()

    def test_dry_run(self, tmp_path, monkeypatch):
        """--dry-run prints config without writing file."""
        monkeypatch.chdir(tmp_path)
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(LLAMA_FACTORY_SFT, encoding="utf-8")
        result = runner.invoke(app, [
            "migrate",
            "--from", "llamafactory",
            str(cfg_file),
            "--dry-run",
        ])
        assert result.exit_code == 0
        out_path = tmp_path / "soup.yaml"
        assert not out_path.exists()
        assert "meta-llama" in result.output

    def test_invalid_source(self, tmp_path, monkeypatch):
        """Unknown --from value shows error."""
        monkeypatch.chdir(tmp_path)
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("test: true\n", encoding="utf-8")
        result = runner.invoke(app, [
            "migrate",
            "--from", "invalid_tool",
            str(cfg_file),
        ])
        assert result.exit_code != 0

    def test_file_not_found(self, tmp_path, monkeypatch):
        """Non-existent input file shows error."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, [
            "migrate",
            "--from", "llamafactory",
            "nonexistent.yaml",
        ])
        assert result.exit_code != 0

    def test_overwrite_confirmation(self, tmp_path, monkeypatch):
        """Existing soup.yaml asks for confirmation."""
        monkeypatch.chdir(tmp_path)
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(LLAMA_FACTORY_SFT, encoding="utf-8")
        out_file = tmp_path / "soup.yaml"
        out_file.write_text("existing content", encoding="utf-8")
        # Without --yes, deny confirmation
        runner.invoke(app, [
            "migrate",
            "--from", "llamafactory",
            str(cfg_file),
            "--output", "soup.yaml",
        ], input="n\n")
        # File should be unchanged
        assert out_file.read_text(encoding="utf-8") == "existing content"

    def test_warnings_shown(self, tmp_path, monkeypatch):
        """Warnings from migration are displayed."""
        monkeypatch.chdir(tmp_path)
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(LLAMA_FACTORY_SFT, encoding="utf-8")
        result = runner.invoke(app, [
            "migrate",
            "--from", "llamafactory",
            str(cfg_file),
            "--dry-run",
        ])
        assert result.exit_code == 0
        # LF SFT has dataset name warning
        assert "warning" in result.output.lower() or "Warning" in result.output


# ---------------------------------------------------------------------------
# Security tests
# ---------------------------------------------------------------------------

class TestMigrateSecurity:
    """Security tests for soup migrate."""

    def test_input_path_traversal_cli(self, tmp_path, monkeypatch):
        """Path traversal in input file is blocked."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, [
            "migrate",
            "--from", "llamafactory",
            "../../../etc/passwd",
        ])
        assert result.exit_code != 0

    def test_output_path_traversal_cli(self, tmp_path, monkeypatch):
        """Path traversal in output path is blocked."""
        monkeypatch.chdir(tmp_path)
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(LLAMA_FACTORY_SFT, encoding="utf-8")
        result = runner.invoke(app, [
            "migrate",
            "--from", "llamafactory",
            str(cfg_file),
            "--output", "../../../tmp/evil.yaml",
            "--yes",
        ])
        assert result.exit_code != 0

    def test_unsloth_no_exec(self, tmp_path):
        """Unsloth migration uses AST only, never exec/eval."""
        from soup_cli.migrate.unsloth import migrate_unsloth

        # A notebook with dangerous code — should parse but not execute
        dangerous_nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": [
                        "import os; os.system('echo pwned')\n",
                        "from unsloth import FastLanguageModel\n",
                        "model, tokenizer = FastLanguageModel.from_pretrained(\n",
                        "    model_name='meta-llama/Llama-3.1-8B',\n",
                        "    max_seq_length=2048,\n",
                        "    load_in_4bit=True,\n",
                        ")\n",
                    ],
                },
                {
                    "cell_type": "code",
                    "source": [
                        "model = FastLanguageModel.get_peft_model(model, r=16)\n",
                    ],
                },
                {
                    "cell_type": "code",
                    "source": [
                        "from trl import SFTTrainer\n",
                        "from transformers import TrainingArguments\n",
                        "trainer = SFTTrainer(\n",
                        "    model=model,\n",
                        "    args=TrainingArguments(\n",
                        "        per_device_train_batch_size=4,\n",
                        "        num_train_epochs=1,\n",
                        "        output_dir='./output',\n",
                        "    ),\n",
                        ")\n",
                    ],
                },
            ],
        }
        nb_file = tmp_path / "dangerous.ipynb"
        nb_file.write_text(json.dumps(dangerous_nb), encoding="utf-8")
        # Should extract config without executing os.system
        result = migrate_unsloth(nb_file)
        assert result["base"] == "meta-llama/Llama-3.1-8B"


# ---------------------------------------------------------------------------
# Additional coverage tests (TDD review gaps)
# ---------------------------------------------------------------------------

class TestMigrateTDDGaps:
    """Tests for coverage gaps identified in TDD review."""

    def test_oversized_input_file(self, tmp_path, monkeypatch):
        """Input file larger than MAX_CONFIG_FILE_SIZE is rejected."""
        from soup_cli.migrate.common import validate_input_path

        monkeypatch.chdir(tmp_path)
        big_file = tmp_path / "big.yaml"
        big_file.write_text("x" * 100, encoding="utf-8")

        # Temporarily lower the limit to test the check
        import soup_cli.migrate.common as common_mod
        original = common_mod.MAX_CONFIG_FILE_SIZE
        common_mod.MAX_CONFIG_FILE_SIZE = 50  # 50 bytes
        try:
            with pytest.raises(ValueError, match="too large"):
                validate_input_path(big_file)
        finally:
            common_mod.MAX_CONFIG_FILE_SIZE = original

    def test_llamafactory_freeze_warning(self, tmp_path):
        """finetuning_type: freeze → warning + fallback to LoRA."""
        from soup_cli.migrate.llamafactory import migrate_llamafactory

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "model_name_or_path: meta-llama/Llama-3.1-8B\n"
            "stage: sft\n"
            "finetuning_type: freeze\n"
            "lora_rank: 16\n"
            "dataset: data\n"
            "num_train_epochs: 1\n"
            "output_dir: ./output\n",
            encoding="utf-8",
        )
        result = migrate_llamafactory(cfg_file)
        # Should fallback to LoRA and warn
        assert "lora" in result.get("training", {})
        assert any("freeze" in w.lower() for w in result.get("_warnings", []))

    def test_axolotl_load_in_8bit(self, tmp_path):
        """load_in_8bit: true → quantization: 8bit."""
        from soup_cli.migrate.axolotl import migrate_axolotl

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "base_model: meta-llama/Llama-3.1-8B\n"
            "datasets:\n"
            "  - path: ./data/train.jsonl\n"
            "    type: alpaca\n"
            "sequence_len: 2048\n"
            "adapter: lora\n"
            "lora_r: 16\n"
            "micro_batch_size: 4\n"
            "num_epochs: 3\n"
            "learning_rate: 2e-4\n"
            "output_dir: ./output\n"
            "load_in_8bit: true\n",
            encoding="utf-8",
        )
        result = migrate_axolotl(cfg_file)
        assert result["training"]["quantization"] == "8bit"

    def test_axolotl_gdpo_alias(self, tmp_path):
        """rl: gdpo → task: grpo."""
        from soup_cli.migrate.axolotl import migrate_axolotl

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "base_model: meta-llama/Llama-3.1-8B\n"
            "datasets:\n"
            "  - path: ./data/train.jsonl\n"
            "    type: sharegpt\n"
            "sequence_len: 2048\n"
            "adapter: lora\n"
            "lora_r: 16\n"
            "micro_batch_size: 2\n"
            "num_epochs: 3\n"
            "learning_rate: 1e-5\n"
            "output_dir: ./output\n"
            "rl: gdpo\n",
            encoding="utf-8",
        )
        result = migrate_axolotl(cfg_file)
        assert result["task"] == "grpo"

    def test_unsloth_markdown_only_notebook(self, tmp_path):
        """Notebook with only markdown cells (no code) raises ValueError."""
        from soup_cli.migrate.unsloth import migrate_unsloth

        nb = {
            "cells": [
                {"cell_type": "markdown", "source": ["# Title"]},
                {"cell_type": "markdown", "source": ["Some text"]},
            ],
        }
        nb_file = tmp_path / "markdown_only.ipynb"
        nb_file.write_text(json.dumps(nb), encoding="utf-8")
        with pytest.raises(ValueError, match="(?i)no .* found"):
            migrate_unsloth(nb_file)

    def test_config_to_yaml_strips_all_private_keys(self):
        """config_to_yaml strips all underscore-prefixed keys."""
        from soup_cli.migrate.common import config_to_yaml

        config = {
            "base": "model",
            "task": "sft",
            "data": {"train": "./data.jsonl"},
            "output": "./output",
            "_warnings": ["w1"],
            "_internal": "private",
            "_debug": True,
        }
        yaml_str = config_to_yaml(config)
        assert "_warnings" not in yaml_str
        assert "_internal" not in yaml_str
        assert "_debug" not in yaml_str
        assert "base:" in yaml_str

    def test_llamafactory_neftune_warning_only(self, tmp_path):
        """NEFTune in LF config produces warning (not auto-migrated to config)."""
        from soup_cli.migrate.llamafactory import migrate_llamafactory

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(LLAMA_FACTORY_NEFTUNE, encoding="utf-8")
        result = migrate_llamafactory(cfg_file)
        assert any(
            "neftune" in w.lower() for w in result.get("_warnings", [])
        )
