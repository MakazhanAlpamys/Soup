"""Tests for Autopilot — zero-config fine-tuning (Part H of v0.25.0)."""

import json
import re

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Dataset analyzer
# ---------------------------------------------------------------------------

class TestAnalyzeDataset:
    def _write_alpaca(self, path, count=100):
        rows = [
            {"instruction": f"q{i} " * 20, "output": f"a{i} " * 10}
            for i in range(count)
        ]
        path.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
        return path

    def test_analyze_alpaca_dataset(self, tmp_path):
        from soup_cli.autopilot.analyzer import analyze_dataset

        data_file = tmp_path / "train.jsonl"
        self._write_alpaca(data_file, count=50)
        profile = analyze_dataset(str(data_file))
        assert profile.samples == 50
        assert profile.format == "alpaca"
        assert profile.avg_tokens > 0
        assert profile.p95_tokens >= profile.avg_tokens

    def test_analyze_empty_raises(self, tmp_path):
        from soup_cli.autopilot.analyzer import analyze_dataset

        data_file = tmp_path / "empty.jsonl"
        data_file.write_text("", encoding="utf-8")
        with pytest.raises(ValueError):
            analyze_dataset(str(data_file))


# ---------------------------------------------------------------------------
# Model analyzer
# ---------------------------------------------------------------------------

class TestAnalyzeModel:
    def test_analyze_llama3_8b(self):
        from soup_cli.autopilot.analyzer import analyze_model

        profile = analyze_model("meta-llama/Llama-3.1-8B-Instruct")
        assert profile.params_b >= 7.0
        assert profile.params_b <= 10.0
        assert profile.context >= 2048

    def test_analyze_tiny_model(self):
        from soup_cli.autopilot.analyzer import analyze_model

        profile = analyze_model("meta-llama/Llama-3.2-1B-Instruct")
        assert profile.params_b <= 2.0


# ---------------------------------------------------------------------------
# Hardware analyzer
# ---------------------------------------------------------------------------

class TestAnalyzeHardware:
    def test_returns_dict_like(self):
        from soup_cli.autopilot.analyzer import analyze_hardware

        profile = analyze_hardware()
        assert hasattr(profile, "vram_gb")
        assert profile.vram_gb >= 0


# ---------------------------------------------------------------------------
# Decision engine
# ---------------------------------------------------------------------------

class TestDecisionEngine:
    def test_decide_task_chat(self):
        from soup_cli.autopilot.decisions import decide_task

        assert decide_task("chat", None) == "sft"

    def test_decide_task_reasoning(self):
        from soup_cli.autopilot.decisions import decide_task

        assert decide_task("reasoning", None) == "grpo"

    def test_decide_task_alignment(self):
        from soup_cli.autopilot.decisions import decide_task

        assert decide_task("alignment", None) == "dpo"

    def test_decide_task_unknown(self):
        from soup_cli.autopilot.decisions import decide_task

        with pytest.raises(ValueError):
            decide_task("evil-goal", None)

    def test_decide_quantization_plenty(self):
        from soup_cli.autopilot.decisions import decide_quantization

        # 80GB VRAM, 7B model ≈ 14GB — plenty for full precision
        assert decide_quantization(model_params_b=7.0, vram_gb=80.0) == "none"

    def test_decide_quantization_4bit(self):
        from soup_cli.autopilot.decisions import decide_quantization

        # 24GB VRAM, 15B model ≈ 30GB full — needs 4bit to fit
        result = decide_quantization(model_params_b=15.0, vram_gb=24.0)
        assert result == "4bit"

    def test_decide_quantization_too_small(self):
        from soup_cli.autopilot.decisions import decide_quantization

        with pytest.raises(ValueError):
            decide_quantization(model_params_b=70.0, vram_gb=4.0)

    def test_decide_quantization_8bit_tier(self):
        from soup_cli.autopilot.decisions import decide_quantization

        # 8B model in fp16 ≈ 16GB, 24GB / 16GB ≈ 1.5× → 8bit tier
        result = decide_quantization(model_params_b=8.0, vram_gb=24.0)
        assert result == "8bit"

    def test_decide_peft_small_data(self):
        from soup_cli.autopilot.decisions import decide_peft

        peft = decide_peft(data_size=500, model_size_b=8.0, vram_gb=24.0)
        assert peft["r"] == 8

    def test_decide_peft_medium_data(self):
        from soup_cli.autopilot.decisions import decide_peft

        peft = decide_peft(data_size=5000, model_size_b=8.0, vram_gb=24.0)
        assert peft["r"] == 16

    def test_decide_peft_large_data(self):
        from soup_cli.autopilot.decisions import decide_peft

        peft = decide_peft(data_size=50_000, model_size_b=8.0, vram_gb=80.0)
        assert peft["r"] == 32

    def test_decide_peft_dora_only_when_headroom(self):
        """DoRA only enabled when data is huge AND VRAM is plentiful."""
        from soup_cli.autopilot.decisions import decide_peft

        # Big data + tight VRAM → LoRA, not DoRA (DoRA doubles the cost)
        tight = decide_peft(data_size=200_000, model_size_b=8.0, vram_gb=12.0)
        assert tight["use_dora"] is False

        # Big data + plenty of VRAM → DoRA
        spacious = decide_peft(data_size=200_000, model_size_b=8.0, vram_gb=80.0)
        assert spacious["use_dora"] is True

    def test_decide_lr_scales_with_rank(self):
        from soup_cli.autopilot.decisions import decide_lr

        assert decide_lr(rank=8, quantization="none") > decide_lr(rank=32, quantization="none")

    def test_decide_epochs_small_data(self):
        from soup_cli.autopilot.decisions import decide_epochs

        assert decide_epochs(200) >= 3
        assert decide_epochs(100_000) == 1

    def test_decide_max_length(self):
        from soup_cli.autopilot.decisions import decide_max_length

        result = decide_max_length(p95_tokens=1800, model_context=8192)
        # Rounded up with 10% margin
        assert result >= 1800
        assert result <= 8192

    def test_decide_max_length_clamp(self):
        from soup_cli.autopilot.decisions import decide_max_length

        # p95 above model context — should clamp
        result = decide_max_length(p95_tokens=20000, model_context=4096)
        assert result == 4096

    def test_decide_performance_flags_ampere(self, monkeypatch):
        from soup_cli.autopilot import decisions

        # #1212 — the flags are install-aware: Ampere+ alone is not enough,
        # the packages must also be present (patched here; CI has neither).
        monkeypatch.setattr(decisions, "check_liger_available", lambda: True)
        monkeypatch.setattr(
            decisions, "check_flash_attn_available", lambda: "flash_attention_2"
        )
        flags = decisions.decide_performance_flags(gpu_name="rtx4090", compute_capability=8.9)
        assert flags["use_flash_attn"] is True
        assert flags["use_liger"] is True

    def test_decide_performance_flags_old_gpu(self):
        from soup_cli.autopilot.decisions import decide_performance_flags

        flags = decide_performance_flags(gpu_name="gtx1080", compute_capability=6.1)
        assert flags["use_flash_attn"] is False

    def test_decide_performance_flags_cpu(self):
        """CPU-only environment (compute_capability=0.0) must disable fast paths."""
        from soup_cli.autopilot.decisions import decide_performance_flags

        flags = decide_performance_flags(gpu_name="none", compute_capability=0.0)
        assert flags["use_flash_attn"] is False
        assert flags["use_liger"] is False

    def test_gradient_checkpointing_long_sequence(self):
        """Long sequences (>8k) enable gradient checkpointing to avoid OOM."""
        from soup_cli.autopilot.decisions import decide_performance_flags

        flags = decide_performance_flags(
            gpu_name="rtx4090",
            compute_capability=8.9,
            max_length=16384,
            vram_headroom_gb=12.0,
        )
        assert flags["gradient_checkpointing"] is True

    def test_gradient_checkpointing_tight_vram(self):
        """Tight VRAM headroom (<4GB) enables gradient checkpointing."""
        from soup_cli.autopilot.decisions import decide_performance_flags

        flags = decide_performance_flags(
            gpu_name="rtx3050",
            compute_capability=8.6,
            max_length=2048,
            vram_headroom_gb=2.0,
        )
        assert flags["gradient_checkpointing"] is True

    def test_gradient_checkpointing_skipped_with_headroom(self):
        from soup_cli.autopilot.decisions import decide_performance_flags

        flags = decide_performance_flags(
            gpu_name="a100",
            compute_capability=8.0,
            max_length=2048,
            vram_headroom_gb=40.0,
        )
        assert flags["gradient_checkpointing"] is False


# ---------------------------------------------------------------------------
# Build config end-to-end
# ---------------------------------------------------------------------------

class TestBuildConfig:
    def _write_data(self, tmp_path):
        rows = [
            {"instruction": f"q{i}", "output": f"a{i}"} for i in range(100)
        ]
        path = tmp_path / "data.jsonl"
        path.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
        return path

    def test_build_soup_config(self, tmp_path):
        from soup_cli.autopilot.generate_config import build_soup_config
        from soup_cli.config.schema import SoupConfig

        data_file = self._write_data(tmp_path)
        cfg = build_soup_config(
            model="meta-llama/Llama-3.1-8B-Instruct",
            data_path=str(data_file),
            goal="chat",
            vram_gb=24.0,
        )
        assert isinstance(cfg, SoupConfig)
        assert cfg.base == "meta-llama/Llama-3.1-8B-Instruct"
        assert cfg.task == "sft"
        assert cfg.training.quantization in ("4bit", "8bit", "none")

    def test_write_yaml(self, tmp_path):
        from soup_cli.autopilot.generate_config import build_soup_config, write_yaml

        data_file = self._write_data(tmp_path)
        cfg = build_soup_config(
            model="meta-llama/Llama-3.1-8B-Instruct",
            data_path=str(data_file),
            goal="chat",
            vram_gb=24.0,
        )
        output_path = tmp_path / "soup.yaml"
        write_yaml(cfg, output_path)
        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "meta-llama/Llama-3.1-8B-Instruct" in content


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------

class TestAutopilotCLI:
    def _write_data(self, tmp_path):
        rows = [
            {"instruction": f"q{i}", "output": f"a{i}"} for i in range(50)
        ]
        path = tmp_path / "data.jsonl"
        path.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
        return path

    def test_help(self):
        result = runner.invoke(app, ["autopilot", "--help"])
        assert result.exit_code == 0
        assert "autopilot" in result.output.lower()

    def test_dry_run(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_file = self._write_data(tmp_path)
        result = runner.invoke(app, [
            "autopilot",
            "--model", "meta-llama/Llama-3.1-8B-Instruct",
            "--data", str(data_file.name),
            "--goal", "chat",
            "--gpu-budget", "24GB",
            "--dry-run",
        ])
        assert result.exit_code == 0

    def test_writes_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_file = self._write_data(tmp_path)
        result = runner.invoke(app, [
            "autopilot",
            "--model", "meta-llama/Llama-3.1-8B-Instruct",
            "--data", str(data_file.name),
            "--goal", "chat",
            "--gpu-budget", "24GB",
            "--output", "soup.yaml",
            "--yes",
        ])
        assert result.exit_code == 0, (
            f"exit={result.exit_code}\n"
            f"output={result.output}\n"
            f"exception={result.exception!r}"
        )
        assert (tmp_path / "soup.yaml").exists()

    def test_rejects_path_traversal_data(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, [
            "autopilot",
            "--model", "meta-llama/Llama-3.1-8B-Instruct",
            "--data", "../../etc/passwd",
            "--goal", "chat",
            "--gpu-budget", "24GB",
        ])
        assert result.exit_code != 0

    def test_rejects_bad_goal(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_file = self._write_data(tmp_path)
        result = runner.invoke(app, [
            "autopilot",
            "--model", "meta-llama/Llama-3.1-8B-Instruct",
            "--data", str(data_file.name),
            "--goal", "evil-goal",
            "--gpu-budget", "24GB",
        ])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# GPU budget parsing
# ---------------------------------------------------------------------------

class TestGPUBudgetParsing:
    def test_parse_gb(self):
        from soup_cli.autopilot.decisions import parse_gpu_budget

        assert parse_gpu_budget("24GB") == 24.0
        assert parse_gpu_budget("80gb") == 80.0

    def test_parse_numeric(self):
        from soup_cli.autopilot.decisions import parse_gpu_budget

        assert parse_gpu_budget("24") == 24.0

    def test_parse_invalid_raises(self):
        from soup_cli.autopilot.decisions import parse_gpu_budget

        with pytest.raises(ValueError):
            parse_gpu_budget("not-a-number")

    def test_parse_out_of_bounds(self):
        from soup_cli.autopilot.decisions import parse_gpu_budget

        with pytest.raises(ValueError):
            parse_gpu_budget("2000GB")  # > 1TB


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# Install-aware kernel flags (#1212)
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output (same as test_v07302)."""
    return " ".join(_ANSI_RE.sub("", text).split())


def _write_ap_data(path, count=20):
    rows = [
        {"instruction": f"q{index} " * 8, "output": f"a{index} " * 4}
        for index in range(count)
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return path


class TestInstallAwareKernelFlags:
    """#1212 — autopilot must not enable flags the environment cannot train.

    Neither ``liger-kernel`` (the separate ``[liger]`` extra) nor
    ``flash-attn`` is part of a standard install, so enabling them blind on
    Ampere+ wrote configs that ``soup train`` then refused.
    """

    def _build(self, tmp_path, monkeypatch, goal, compute_capability, *, liger, flash):
        from soup_cli.autopilot import decisions, generate_config
        from soup_cli.autopilot.analyzer import HardwareProfile

        monkeypatch.setattr(
            generate_config,
            "analyze_hardware",
            lambda: HardwareProfile(
                device="cuda",
                gpu_name="rtx4090",
                vram_gb=24.0,
                compute_capability=compute_capability,
                system_ram_gb=64.0,
            ),
        )
        monkeypatch.setattr(decisions, "check_liger_available", lambda: liger)
        monkeypatch.setattr(
            decisions,
            "check_flash_attn_available",
            lambda: "flash_attention_2" if flash else None,
        )
        return generate_config.build_soup_config(
            model="HuggingFaceTB/SmolLM2-135M-Instruct",
            data_path=str(_write_ap_data(tmp_path / "data.jsonl")),
            goal=goal,
            vram_gb=24.0,
        )

    @pytest.mark.parametrize("goal", ["chat", "code", "classification", "tool-calling"])
    def test_missing_packages_disable_flags_for_every_sft_goal(
        self, tmp_path, monkeypatch, goal
    ):
        config = self._build(tmp_path, monkeypatch, goal, 8.6, liger=False, flash=False)
        assert config.task == "sft"
        assert config.training.use_flash_attn is False
        assert config.training.use_liger is False

    @pytest.mark.parametrize("compute_capability", [8.0, 8.6, 8.9, 9.0, 12.0])
    def test_present_packages_enable_flags_across_ampere_plus(
        self, tmp_path, monkeypatch, compute_capability
    ):
        config = self._build(
            tmp_path, monkeypatch, "chat", compute_capability, liger=True, flash=True
        )
        assert config.training.use_flash_attn is True
        assert config.training.use_liger is True

    def test_pre_ampere_stays_off_even_with_packages(self, tmp_path, monkeypatch):
        config = self._build(tmp_path, monkeypatch, "chat", 7.5, liger=True, flash=True)
        assert config.training.use_flash_attn is False
        assert config.training.use_liger is False

    def test_liger_only_enables_exactly_liger(self, tmp_path, monkeypatch):
        config = self._build(tmp_path, monkeypatch, "chat", 8.6, liger=True, flash=False)
        assert config.training.use_liger is True
        assert config.training.use_flash_attn is False

    def test_flash_only_enables_exactly_flash(self, tmp_path, monkeypatch):
        config = self._build(tmp_path, monkeypatch, "chat", 8.6, liger=False, flash=True)
        assert config.training.use_liger is False
        assert config.training.use_flash_attn is True

    def _patch_cli_hardware(self, monkeypatch, compute_capability=8.6):
        from soup_cli.autopilot import decisions, generate_config
        from soup_cli.autopilot.analyzer import HardwareProfile
        from soup_cli.commands import autopilot as autopilot_cmd

        profile = HardwareProfile(
            device="cuda",
            gpu_name="rtx4090",
            vram_gb=24.0,
            compute_capability=compute_capability,
            system_ram_gb=64.0,
        )
        monkeypatch.setattr(generate_config, "analyze_hardware", lambda: profile)
        monkeypatch.setattr(autopilot_cmd, "analyze_hardware", lambda: profile)
        monkeypatch.setattr(decisions, "check_liger_available", lambda: False)
        monkeypatch.setattr(decisions, "check_flash_attn_available", lambda: None)

    def test_panel_names_missing_packages(self, tmp_path, monkeypatch):
        self._patch_cli_hardware(monkeypatch)
        monkeypatch.chdir(tmp_path)
        _write_ap_data(tmp_path / "ap.jsonl")

        result = runner.invoke(
            app,
            [
                "autopilot",
                "--model", "HuggingFaceTB/SmolLM2-135M-Instruct",
                "--data", "ap.jsonl",
                "--goal", "chat",
                "--output", "ap_chat.yaml",
                "--yes",
            ],
        )
        assert result.exit_code == 0
        plain = _plain(result.output)
        assert "Flash Attention: off (flash-attn not installed" in plain
        assert "Liger Kernel: off (liger-kernel not installed" in plain
        # The full hints wrap inside the Rich panel at unpredictable widths
        # (border glyphs land between the wrapped words), so the panel test
        # asserts the package names; the literal `soup-cli[liger]` hint is
        # asserted in the `soup train` refusal test, which is not a panel.

    def test_autopilot_config_survives_train_dry_run_without_packages(
        self, tmp_path, monkeypatch
    ):
        from soup_cli.commands import train as train_cmd

        self._patch_cli_hardware(monkeypatch)
        # `soup train` on a CPU CI box: the device is CUDA-shaped for the
        # kernel validators, and unknown VRAM (bytes=0) skips the fit gate.
        monkeypatch.setattr(
            train_cmd, "detect_device", lambda backend=None: ("cuda", "CUDA (test)")
        )
        monkeypatch.setattr(
            train_cmd,
            "get_gpu_info",
            lambda backend=None: {
                "memory_total": "24.0 GB",
                "memory_total_bytes": 0,
                "gpu_count": 1,
            },
        )
        monkeypatch.chdir(tmp_path)
        _write_ap_data(tmp_path / "ap.jsonl")

        written = runner.invoke(
            app,
            [
                "autopilot",
                "--model", "HuggingFaceTB/SmolLM2-135M-Instruct",
                "--data", "ap.jsonl",
                "--goal", "chat",
                "--output", "ap_chat.yaml",
                "--yes",
            ],
        )
        assert written.exit_code == 0

        trained = runner.invoke(
            app, ["train", "--config", "ap_chat.yaml", "--dry-run", "--yes"]
        )
        assert trained.exit_code == 0, _plain(trained.output)
        assert "Data OK" in _plain(trained.output)

    def test_liger_refusal_keeps_extra_name_visible(self, tmp_path, monkeypatch):
        from soup_cli.commands import train as train_cmd
        from soup_cli.utils import liger as liger_utils

        monkeypatch.chdir(tmp_path)
        data_path = _write_ap_data(tmp_path / "data.jsonl")
        config_path = tmp_path / "soup.yaml"
        config_path.write_text(
            "base: sshleifer/tiny-gpt2\n"
            "task: sft\n"
            f"output: {tmp_path / 'out'}\n"
            "data:\n"
            f"  train: {data_path}\n"
            "training:\n"
            "  use_liger: true\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            train_cmd, "detect_device", lambda backend=None: ("cuda", "CUDA (test)")
        )
        monkeypatch.setattr(
            train_cmd,
            "get_gpu_info",
            lambda backend=None: {
                "memory_total": "24.0 GB",
                "memory_total_bytes": 0,
                "gpu_count": 1,
            },
        )
        monkeypatch.setattr(liger_utils, "check_liger_available", lambda: False)

        result = runner.invoke(
            app, ["train", "--config", str(config_path), "--dry-run", "--yes"]
        )
        assert result.exit_code == 1
        plain = _plain(result.output)
        assert "liger-kernel is not installed" in plain
        # Before #1212 the hint's [liger] was swallowed as a Rich markup tag,
        # so the one instruction printed told users to reinstall what they
        # already had.
        assert 'pip install "soup-cli[liger]"' in plain
