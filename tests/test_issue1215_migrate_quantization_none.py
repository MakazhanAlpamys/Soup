# -*- coding: utf-8 -*-
"""Tests for Issue #1215: soup migrate explicitly writes quantization: none,
supports full fine-tuning (lora.r: 0), handles quantization methods,
and validates output with load_config_from_string.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.loader import load_config_from_string
from soup_cli.migrate.axolotl import migrate_axolotl
from soup_cli.migrate.common import config_to_yaml
from soup_cli.migrate.llamafactory import migrate_llamafactory
from soup_cli.migrate.unsloth import migrate_unsloth
from tests.conftest import strip_ansi

runner = CliRunner()




class TestAxolotlQuantizationAndFullFinetuning:
    def test_axolotl_explicit_load_in_false_gives_none(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "base_model: meta-llama/Llama-3.1-8B-Instruct\n"
            "adapter: lora\n"
            "lora_r: 16\n"
            "load_in_4bit: false\n"
            "load_in_8bit: false\n"
            "datasets:\n"
            "  - path: ./data/train.jsonl\n"
            "    type: alpaca\n",
            encoding="utf-8",
        )
        res = migrate_axolotl(cfg)
        loaded = load_config_from_string(config_to_yaml(res))
        assert loaded.training.quantization == "none"
        assert loaded.training.lora.r == 16

    def test_axolotl_no_load_in_keys_gives_none(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "base_model: meta-llama/Llama-3.1-8B-Instruct\n"
            "adapter: lora\n"
            "lora_r: 32\n"
            "datasets:\n"
            "  - path: ./data/train.jsonl\n"
            "    type: alpaca\n",
            encoding="utf-8",
        )
        res = migrate_axolotl(cfg)
        loaded = load_config_from_string(config_to_yaml(res))
        assert loaded.training.quantization == "none"
        assert loaded.training.lora.r == 32

    def test_axolotl_no_adapter_sft_gives_full_finetuning(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "base_model: meta-llama/Llama-3.1-8B-Instruct\n"
            "datasets:\n"
            "  - path: ./data/train.jsonl\n"
            "    type: alpaca\n",
            encoding="utf-8",
        )
        res = migrate_axolotl(cfg)
        loaded = load_config_from_string(config_to_yaml(res))
        assert loaded.training.quantization == "none"
        assert loaded.training.lora.r == 0

    def test_axolotl_load_in_8bit_and_qlora(self, tmp_path: Path) -> None:
        cfg8 = tmp_path / "cfg8.yaml"
        cfg8.write_text(
            "base_model: meta-llama/Llama-3.1-8B-Instruct\n"
            "adapter: lora\n"
            "lora_r: 16\n"
            "load_in_8bit: true\n"
            "datasets:\n"
            "  - path: ./data/train.jsonl\n"
            "    type: alpaca\n",
            encoding="utf-8",
        )
        res8 = migrate_axolotl(cfg8)
        loaded8 = load_config_from_string(config_to_yaml(res8))
        assert loaded8.training.quantization == "8bit"

        cfg4 = tmp_path / "cfg4.yaml"
        cfg4.write_text(
            "base_model: meta-llama/Llama-3.1-8B-Instruct\n"
            "adapter: qlora\n"
            "lora_r: 16\n"
            "datasets:\n"
            "  - path: ./data/train.jsonl\n"
            "    type: alpaca\n",
            encoding="utf-8",
        )
        res4 = migrate_axolotl(cfg4)
        loaded4 = load_config_from_string(config_to_yaml(res4))
        assert loaded4.training.quantization == "4bit"

    def test_axolotl_no_adapter_dpo_refused(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "base_model: meta-llama/Llama-3.1-8B-Instruct\n"
            "rl: dpo\n"
            "datasets:\n"
            "  - path: ./data/train.jsonl\n"
            "    type: sharegpt\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="task 'dpo'"):
            migrate_axolotl(cfg)


class TestLlamaFactoryQuantizationAndFullFinetuning:
    def test_plain_lora_no_quant_bit_gives_none(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "model_name_or_path: meta-llama/Llama-3.1-8B-Instruct\n"
            "stage: sft\n"
            "finetuning_type: lora\n"
            "lora_rank: 16\n"
            "dataset: alpaca\n",
            encoding="utf-8",
        )
        res = migrate_llamafactory(cfg)
        loaded = load_config_from_string(config_to_yaml(res))
        assert loaded.training.quantization == "none"
        assert loaded.training.lora.r == 16

    @pytest.mark.parametrize(
        "quant_val, expected",
        [(4, "4bit"), (8, "8bit"), ("4", "4bit"), ("8", "8bit")],
    )
    def test_quantization_bit_variants(
        self, tmp_path: Path, quant_val: object, expected: str
    ) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            f"model_name_or_path: meta-llama/Llama-3.1-8B-Instruct\n"
            f"stage: sft\n"
            f"finetuning_type: lora\n"
            f"lora_rank: 16\n"
            f"quantization_bit: {quant_val}\n"
            f"dataset: alpaca\n",
            encoding="utf-8",
        )
        res = migrate_llamafactory(cfg)
        loaded = load_config_from_string(config_to_yaml(res))
        assert loaded.training.quantization == expected

    def test_hqq_and_eetq_quantization_methods(self, tmp_path: Path) -> None:
        # HQQ 2-bit
        cfg_hqq = tmp_path / "cfg_hqq.yaml"
        cfg_hqq.write_text(
            "model_name_or_path: meta-llama/Llama-3.1-8B-Instruct\n"
            "stage: sft\n"
            "finetuning_type: lora\n"
            "quantization_bit: 2\n"
            "quantization_method: hqq\n"
            "dataset: alpaca\n",
            encoding="utf-8",
        )
        res_hqq = migrate_llamafactory(cfg_hqq)
        loaded_hqq = load_config_from_string(config_to_yaml(res_hqq))
        assert loaded_hqq.training.quantization == "hqq:2bit"

        # EETQ 8-bit
        cfg_eetq = tmp_path / "cfg_eetq.yaml"
        cfg_eetq.write_text(
            "model_name_or_path: meta-llama/Llama-3.1-8B-Instruct\n"
            "stage: sft\n"
            "finetuning_type: lora\n"
            "quantization_bit: 8\n"
            "quantization_method: eetq\n"
            "dataset: alpaca\n",
            encoding="utf-8",
        )
        res_eetq = migrate_llamafactory(cfg_eetq)
        loaded_eetq = load_config_from_string(config_to_yaml(res_eetq))
        assert loaded_eetq.training.quantization == "eetq"

        # Unsupported method refused naming it
        cfg_bad = tmp_path / "cfg_bad.yaml"
        cfg_bad.write_text(
            "model_name_or_path: meta-llama/Llama-3.1-8B-Instruct\n"
            "stage: sft\n"
            "finetuning_type: lora\n"
            "quantization_bit: 4\n"
            "quantization_method: awq\n"
            "dataset: alpaca\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="awq"):
            migrate_llamafactory(cfg_bad)

    def test_full_finetuning_sft(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "model_name_or_path: meta-llama/Llama-3.1-8B-Instruct\n"
            "stage: sft\n"
            "finetuning_type: full\n"
            "dataset: alpaca\n",
            encoding="utf-8",
        )
        res = migrate_llamafactory(cfg)
        loaded = load_config_from_string(config_to_yaml(res))
        assert loaded.training.quantization == "none"
        assert loaded.training.lora.r == 0

    @pytest.mark.parametrize("stage", ["dpo", "pt", "kto", "rm", "ppo"])
    def test_full_finetuning_unsupported_stages_refused(self, tmp_path: Path, stage: str) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            f"model_name_or_path: meta-llama/Llama-3.1-8B-Instruct\n"
            f"stage: {stage}\n"
            f"finetuning_type: full\n"
            f"dataset: data\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=f"'{stage}'"):
            migrate_llamafactory(cfg)

    def test_oft_warns_by_name(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "model_name_or_path: meta-llama/Llama-3.1-8B-Instruct\n"
            "stage: sft\n"
            "finetuning_type: oft\n"
            "dataset: alpaca\n",
            encoding="utf-8",
        )
        res = migrate_llamafactory(cfg)
        assert any("oft" in w for w in res["_warnings"])
        loaded = load_config_from_string(config_to_yaml(res))
        assert loaded.training.quantization == "none"
        assert loaded.training.lora.r == 64


class TestUnslothQuantizationAndFullFinetuning:
    def _make_notebook(self, from_pretrained_src: str, extra_code: str = "") -> dict:
        cells = []
        if extra_code:
            cells.append({"cell_type": "code", "source": [extra_code + "\n"]})
        cells.append({
            "cell_type": "code",
            "source": [
                "from unsloth import FastLanguageModel\n",
                from_pretrained_src + "\n",
            ],
        })
        cells.append({
            "cell_type": "code",
            "source": [
                "model = FastLanguageModel.get_peft_model(model, r=16)\n",
            ],
        })
        return {"cells": cells}

    def test_unsloth_load_in_4bit_false_gives_none(self, tmp_path: Path) -> None:
        nb = self._make_notebook(
            "model, tok = FastLanguageModel.from_pretrained(\n"
            "    'meta-llama/Llama-3.1-8B', load_in_4bit=False\n"
            ")"
        )
        nb_file = tmp_path / "test.ipynb"
        nb_file.write_text(json.dumps(nb), encoding="utf-8")
        res = migrate_unsloth(nb_file)
        loaded = load_config_from_string(config_to_yaml(res))
        assert loaded.training.quantization == "none"
        assert loaded.training.lora.r == 16

    def test_unsloth_load_in_16bit_gives_none(self, tmp_path: Path) -> None:
        nb = self._make_notebook(
            "model, tok = FastLanguageModel.from_pretrained(\n"
            "    'meta-llama/Llama-3.1-8B', load_in_16bit=True\n"
            ")"
        )
        nb_file = tmp_path / "test.ipynb"
        nb_file.write_text(json.dumps(nb), encoding="utf-8")
        res = migrate_unsloth(nb_file)
        loaded = load_config_from_string(config_to_yaml(res))
        assert loaded.training.quantization == "none"

    def test_unsloth_load_in_8bit_gives_8bit(self, tmp_path: Path) -> None:
        nb = self._make_notebook(
            "model, tok = FastLanguageModel.from_pretrained(\n"
            "    'meta-llama/Llama-3.1-8B', load_in_8bit=True\n"
            ")"
        )
        nb_file = tmp_path / "test.ipynb"
        nb_file.write_text(json.dumps(nb), encoding="utf-8")
        res = migrate_unsloth(nb_file)
        loaded = load_config_from_string(config_to_yaml(res))
        assert loaded.training.quantization == "8bit"

    def test_unsloth_full_finetuning_gives_none_and_r_zero(self, tmp_path: Path) -> None:
        cells = [{
            "cell_type": "code",
            "source": [
                "from unsloth import FastLanguageModel\n",
                "model, tok = FastLanguageModel.from_pretrained(\n"
                "    'meta-llama/Llama-3.1-8B', full_finetuning=True\n"
                ")\n",
            ],
        }]
        nb_file = tmp_path / "test.ipynb"
        nb_file.write_text(json.dumps({"cells": cells}), encoding="utf-8")
        res = migrate_unsloth(nb_file)
        loaded = load_config_from_string(config_to_yaml(res))
        assert loaded.training.quantization == "none"
        assert loaded.training.lora.r == 0

    def test_unsloth_load_in_4bit_true_or_default(self, tmp_path: Path) -> None:
        nb = self._make_notebook(
            "model, tok = FastLanguageModel.from_pretrained('meta-llama/Llama-3.1-8B')"
        )
        nb_file = tmp_path / "test.ipynb"
        nb_file.write_text(json.dumps(nb), encoding="utf-8")
        res = migrate_unsloth(nb_file)
        loaded = load_config_from_string(config_to_yaml(res))
        assert loaded.training.quantization == "4bit"

    def test_unsloth_load_in_4bit_as_variable_resolved(self, tmp_path: Path) -> None:
        nb = self._make_notebook(
            "model, tok = FastLanguageModel.from_pretrained(\n"
            "    'meta-llama/Llama-3.1-8B', load_in_4bit=load_in_4bit\n"
            ")",
            extra_code="load_in_4bit = False",
        )
        nb_file = tmp_path / "test.ipynb"
        nb_file.write_text(json.dumps(nb), encoding="utf-8")
        res = migrate_unsloth(nb_file)
        loaded = load_config_from_string(config_to_yaml(res))
        assert loaded.training.quantization == "none"

    def test_unsloth_load_in_4bit_unresolved_variable_warns(self, tmp_path: Path) -> None:
        nb = self._make_notebook(
            "model, tok = FastLanguageModel.from_pretrained(\n"
            "    'meta-llama/Llama-3.1-8B', load_in_4bit=load_in_4bit\n"
            ")"
        )
        nb_file = tmp_path / "test.ipynb"
        nb_file.write_text(json.dumps(nb), encoding="utf-8")
        res = migrate_unsloth(nb_file)
        assert any("load_in_4bit" in w for w in res["_warnings"])


class TestMigrateCLIDryRunOutput:
    def test_cli_dry_run_plain_lora_prints_quantization_none(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        cfg_file = tmp_path / "plain_lora.yaml"
        cfg_file.write_text(
            "model_name_or_path: meta-llama/Llama-3.1-8B-Instruct\n"
            "stage: sft\n"
            "finetuning_type: lora\n"
            "lora_rank: 16\n"
            "dataset: alpaca\n",
            encoding="utf-8",
        )
        result = runner.invoke(app, [
            "migrate",
            "--from", "llamafactory",
            str(cfg_file),
            "--dry-run",
        ])
        assert result.exit_code == 0
        clean = strip_ansi(result.output)
        assert "quantization: none" in clean or "quantization=none" in clean

    def test_cli_dry_run_full_finetune_prints_r_zero(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        cfg_file = tmp_path / "full_sft.yaml"
        cfg_file.write_text(
            "model_name_or_path: meta-llama/Llama-3.1-8B-Instruct\n"
            "stage: sft\n"
            "finetuning_type: full\n"
            "dataset: alpaca\n",
            encoding="utf-8",
        )
        result = runner.invoke(app, [
            "migrate",
            "--from", "llamafactory",
            str(cfg_file),
            "--dry-run",
        ])
        assert result.exit_code == 0
        clean = strip_ansi(result.output)
        assert "r: 0" in clean or "lora.r=0" in clean
