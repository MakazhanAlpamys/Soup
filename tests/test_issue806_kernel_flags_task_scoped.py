"""Regression coverage for #806: use_flash_attn/use_liger are validated and
auto-enabled for every task, but only the SFT-family trainers ever read them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig

runner = CliRunner()


def _write_data(path: Path) -> Path:
    path.write_text(
        "\n".join(
            json.dumps({"instruction": f"q{index}", "output": f"a{index}"})
            for index in range(20)
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _config(tmp_path, task, **training_kwargs):
    if task == "tts":
        training_kwargs.setdefault("tts_family", "orpheus")
    return SoupConfig(
        base="sshleifer/tiny-gpt2",
        task=task,
        modality="audio_out" if task == "tts" else "text",
        data=DataConfig(train=str(_write_data(tmp_path / "data.jsonl"))),
        training=TrainingConfig(**training_kwargs),
        output=str(tmp_path / "out"),
    )


class TestKernelFlagsRejectedForNonSftTasks:
    @pytest.mark.parametrize("task", ["dpo", "grpo", "pretrain"])
    def test_use_liger_rejected(self, tmp_path, task):
        with pytest.raises(ValidationError, match="training.use_liger"):
            _config(tmp_path, task, use_liger=True)

    @pytest.mark.parametrize("task", ["dpo", "grpo", "pretrain"])
    def test_use_flash_attn_rejected(self, tmp_path, task):
        with pytest.raises(ValidationError, match="training.use_flash_attn"):
            _config(tmp_path, task, use_flash_attn=True)

    @pytest.mark.parametrize("task", ["sft", "tts"])
    def test_sft_family_still_accepts_both(self, tmp_path, task):
        # Control: the trainer really does read these two for sft/tts
        # (tts.py's TTSTrainerWrapper inherits sft.py's _setup_transformers).
        config = _config(tmp_path, task, use_liger=True, use_flash_attn=True)
        assert config.training.use_liger is True
        assert config.training.use_flash_attn is True


class TestAutopilotScopesKernelFlagsToTask:
    def _build(self, tmp_path, monkeypatch, goal):
        from soup_cli.autopilot import decisions, generate_config
        from soup_cli.autopilot.analyzer import HardwareProfile

        monkeypatch.setattr(
            generate_config,
            "analyze_hardware",
            lambda: HardwareProfile(
                device="cuda",
                gpu_name="a100",
                vram_gb=40.0,
                compute_capability=8.0,
                system_ram_gb=64.0,
            ),
        )
        # #1212 — the decision is install-aware now, so an Ampere test that
        # expects the flags on must also say the packages are present (CI
        # installs neither liger-kernel nor flash-attn).
        monkeypatch.setattr(decisions, "check_liger_available", lambda: True)
        monkeypatch.setattr(
            decisions,
            "check_flash_attn_available",
            lambda: "flash_attention_2",
        )
        return generate_config.build_soup_config(
            model="meta-llama/Llama-3.1-8B-Instruct",
            data_path=str(_write_data(tmp_path / "data.jsonl")),
            goal=goal,
            vram_gb=40.0,
        )

    def test_ampere_gpu_still_enables_kernels_for_sft(self, tmp_path, monkeypatch):
        config = self._build(tmp_path, monkeypatch, goal="chat")
        assert config.task == "sft"
        assert config.training.use_flash_attn is True
        assert config.training.use_liger is True

    def test_ampere_gpu_does_not_enable_kernels_for_dpo(self, tmp_path, monkeypatch):
        # #806's own repro: goal="alignment" -> dpo, compute capability 8.0+.
        config = self._build(tmp_path, monkeypatch, goal="alignment")
        assert config.task == "dpo"
        assert config.training.use_flash_attn is False
        assert config.training.use_liger is False

    def test_ampere_gpu_does_not_enable_kernels_for_grpo(self, tmp_path, monkeypatch):
        # On main this raises ValidationError instead of building at all,
        # since decide_performance_flags would set both True on Ampere+ and
        # nothing downstream used to stop that from reaching a grpo config.
        config = self._build(tmp_path, monkeypatch, goal="reasoning")
        assert config.task == "grpo"
        assert config.training.use_flash_attn is False
        assert config.training.use_liger is False

    def test_ampere_gpu_does_not_enable_kernels_for_pretrain(self, tmp_path, monkeypatch):
        config = self._build(tmp_path, monkeypatch, goal="domain-adapt")
        assert config.task == "pretrain"
        assert config.training.use_flash_attn is False
        assert config.training.use_liger is False


class TestTrainCliSurfacesTaskMismatchNotMissingPackage:
    def test_dry_run_reports_task_gate_not_missing_liger_kernel(self, tmp_path, monkeypatch):
        data_path = _write_data(tmp_path / "data.jsonl")
        config_path = tmp_path / "soup.yaml"
        config_path.write_text(
            "base: sshleifer/tiny-gpt2\n"
            "task: grpo\n"
            f"output: {tmp_path / 'out'}\n"
            "data:\n"
            f"  train: {data_path}\n"
            "training:\n"
            "  use_liger: true\n",
            encoding="utf-8",
        )

        result = runner.invoke(
            app,
            ["train", "--config", str(config_path), "--dry-run", "--yes"],
        )

        assert result.exit_code == 1
        # Before the fix this failed on liger-kernel not being installed,
        # a real message but the wrong one for a task that never reads the
        # flag. It must now fail on the task mismatch instead.
        assert "liger-kernel is not installed" not in result.output
        assert "use_liger" in result.output
        assert "task" in result.output
