"""Regression tests for Issue #1423:
Unified full fine-tuning PEFT classification & profiler budgeting.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.mcp_server.registry import tool_profile
from tests.conftest import strip_ansi

BASE_MODEL = "meta-llama/Llama-3.1-8B"

CONFIG_LORA_CONTROL = """
base: meta-llama/Llama-3.1-8B
data:
  train: dummy.jsonl
  max_length: 512
training:
  quantization: none
  batch_size: 1
  gradient_checkpointing: true
  lora:
    r: 16
    alpha: 16
"""

CONFIG_LORA_ZERO = """
base: meta-llama/Llama-3.1-8B
data:
  train: dummy.jsonl
  max_length: 512
training:
  quantization: none
  batch_size: 1
  gradient_checkpointing: true
  lora:
    r: 0
    alpha: 16
"""

CONFIG_SPECTRUM = """
base: meta-llama/Llama-3.1-8B
data:
  train: dummy.jsonl
  max_length: 512
training:
  quantization: none
  batch_size: 1
  gradient_checkpointing: true
  unfrozen_parameters:
    - "model.layers.0.*"
"""

CONFIG_LISA = """
base: meta-llama/Llama-3.1-8B
data:
  train: dummy.jsonl
  max_length: 512
training:
  quantization: none
  batch_size: 1
  gradient_checkpointing: true
  lisa_enabled: true
"""

CONFIG_PRM = """
base: meta-llama/Llama-3.1-8B
task: prm
data:
  train: dummy.jsonl
  max_length: 512
training:
  quantization: none
  batch_size: 1
  gradient_checkpointing: true
"""

CONFIG_CLASSIFIER = """
base: meta-llama/Llama-3.1-8B
task: classifier
data:
  train: dummy.jsonl
  max_length: 512
training:
  quantization: none
  batch_size: 1
  gradient_checkpointing: true
  num_labels: 2
"""

ALL_CONFIGS = [
    ("lora_control", CONFIG_LORA_CONTROL, "lora", False),
    ("lora_zero", CONFIG_LORA_ZERO, "full", True),
    ("spectrum", CONFIG_SPECTRUM, "full", True),
    ("lisa", CONFIG_LISA, "full", True),
    ("prm", CONFIG_PRM, "full", True),
    ("classifier", CONFIG_CLASSIFIER, "full", True),
]


class TestProfileFullFinetune:
    @pytest.mark.parametrize("name,yaml_text,expected_peft,is_full_ft", ALL_CONFIGS)
    def test_profile_json_trainable_params_and_optimizer(
        self, tmp_path: Path, name: str, yaml_text: str, expected_peft: str, is_full_ft: bool
    ) -> None:
        cfg_file = tmp_path / f"{name}.yaml"
        cfg_file.write_text(yaml_text, encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(app, ["profile", "--config", str(cfg_file), "--json"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)

        assert data.get("peft") == expected_peft

        if is_full_ft:
            # Full fine-tune: trainable params equals model parameter count (8.0B)
            assert data["trainable_params"] == 8_000_000_000
            assert data["optimizer_memory_gb"] > 0
            assert data["total_memory_gb"] > 24.0
        else:
            # LoRA control: exactly unchanged (~29.36M params)
            assert data["trainable_params"] == 29_360_128
            assert data["total_memory_gb"] == 17.74

    @pytest.mark.parametrize("name,yaml_text,expected_peft,is_full_ft", ALL_CONFIGS)
    def test_cli_and_preflight_peft_classification_agree(
        self, tmp_path: Path, name: str, yaml_text: str, expected_peft: str, is_full_ft: bool
    ) -> None:
        from soup_cli.commands.train import _build_hardware_fit_input
        from soup_cli.config.loader import load_config
        from soup_cli.utils.hardware_fit import resolve_peft_mode

        cfg_file = tmp_path / f"{name}.yaml"
        cfg_file.write_text(yaml_text, encoding="utf-8")
        cfg = load_config(str(cfg_file))

        # Check shared resolve_peft_mode
        resolved_peft = resolve_peft_mode(cfg)
        assert resolved_peft == expected_peft

        # Check train preflight builds matching peft
        inp = _build_hardware_fit_input(cfg)
        assert inp is not None
        assert inp.peft == expected_peft

    @pytest.mark.parametrize("name,yaml_text,expected_peft,is_full_ft", ALL_CONFIGS)
    def test_mcp_tool_profile_parity_with_cli(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        name: str,
        yaml_text: str,
        expected_peft: str,
        is_full_ft: bool,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        cfg_file = tmp_path / "soup.yaml"
        cfg_file.write_text(yaml_text, encoding="utf-8")

        # Run CLI --json
        runner = CliRunner()
        cli_res = runner.invoke(app, ["profile", "--config", "soup.yaml", "--json"])
        assert cli_res.exit_code == 0, cli_res.output
        cli_data = json.loads(cli_res.output)

        # Run MCP tool_profile
        mcp_data = tool_profile({"config": "soup.yaml"})

        assert mcp_data["trainable_params"] == cli_data["trainable_params"]
        assert mcp_data["total_memory_gb"] == cli_data["total_memory_gb"]
        assert mcp_data["optimizer_memory_gb"] == cli_data["optimizer_memory_gb"]
        assert mcp_data.get("peft") == cli_data.get("peft")

    def test_cli_rendering_full_finetune_labels(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "lora_zero.yaml"
        cfg_file.write_text(CONFIG_LORA_ZERO, encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(app, ["profile", "--config", str(cfg_file), "--gpu", "rtx4090"])
        assert result.exit_code == 0
        plain = strip_ansi(result.output)
        assert "full fine-tuning" in plain
        assert "Fits in 24 GB VRAM" not in plain

    def test_cli_rendering_lora_control_unchanged(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "lora_control.yaml"
        cfg_file.write_text(CONFIG_LORA_CONTROL, encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(app, ["profile", "--config", str(cfg_file), "--gpu", "rtx4090"])
        assert result.exit_code == 0
        plain = strip_ansi(result.output)
        assert "with LoRA r=16" in plain
        assert "Fits in 24 GB VRAM" in plain

    def test_mutation_check_reverting_peft_to_lora_only_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation check: reverting peft to 'lora' misreports lora.r=0 as 0 params."""
        cfg_file = tmp_path / "lora_zero.yaml"
        cfg_file.write_text(CONFIG_LORA_ZERO, encoding="utf-8")

        # Simulate mutation: resolve_peft_mode returning 'lora' unconditionally
        monkeypatch.setattr(
            "soup_cli.utils.hardware_fit.resolve_peft_mode",
            lambda cfg, **kw: "lora",
        )

        runner = CliRunner()
        result = runner.invoke(app, ["profile", "--config", str(cfg_file), "--json"])
        data = json.loads(result.output)
        # Under mutated code, trainable_params is erroneously 0 instead of 8B
        assert data["trainable_params"] == 0
