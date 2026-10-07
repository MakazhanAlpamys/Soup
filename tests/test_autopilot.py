"""Tests for Autopilot — zero-config fine-tuning (Part H of v0.25.0)."""

import json
import re

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output, safe to substring-match."""
    return " ".join(_ANSI_RE.sub("", text).split())


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

    @pytest.mark.parametrize("quant", ["gptq", "awq", "aqlm", "eetq"])
    def test_decide_peft_no_dora_on_prequantized(self, quant):
        """peft cannot apply DoRA to these layers; bnb / none keep it (#1466)."""
        from soup_cli.autopilot.decisions import decide_peft

        kwargs = {"data_size": 200_000, "model_size_b": 8.0, "vram_gb": 80.0}
        assert decide_peft(**kwargs, quantization=quant)["use_dora"] is False
        assert decide_peft(**kwargs, quantization="4bit")["use_dora"] is True
        assert decide_peft(**kwargs, quantization="none")["use_dora"] is True

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

    @pytest.mark.parametrize(
        "model, expected",
        [
            ("TheBloke/Mistral-7B-Instruct-v0.2-GPTQ", "gptq"),
            ("TheBloke/Mistral-7B-Instruct-v0.2-AWQ", "awq"),
        ],
    )
    def test_build_soup_config_no_dora_on_prequantized_base(self, tmp_path, model, expected):
        """peft refuses DoRA on GPTQ / AWQ / AQLM / EETQ layers (#1466)."""
        from soup_cli.autopilot.generate_config import build_soup_config

        data_file = tmp_path / "big.jsonl"
        data_file.write_text(
            "".join(
                json.dumps({"instruction": f"q{i}", "output": f"a{i}"}) + "\n"
                for i in range(100_001)
            ),
            encoding="utf-8",
        )
        cfg = build_soup_config(
            model=model, data_path=str(data_file), goal="chat", vram_gb=24.0,
        )
        assert cfg.training.quantization == expected
        assert cfg.training.lora.use_dora is False

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

    def test_non_terminal_missing_model_exits_2(self):
        result = runner.invoke(app, ["autopilot"])
        assert result.exit_code == 2
        assert "Missing option '--model' / '-m'." in _plain(result.output)

    def test_non_terminal_missing_data_exits_2(self):
        result = runner.invoke(
            app, ["autopilot", "--model", "meta-llama/Llama-3.1-8B-Instruct"]
        )
        assert result.exit_code == 2
        assert "Missing option '--data' / '-d'." in _plain(result.output)

    def test_non_terminal_missing_goal_exits_2(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_file = self._write_data(tmp_path)
        result = runner.invoke(
            app,
            [
                "autopilot",
                "--model", "meta-llama/Llama-3.1-8B-Instruct",
                "--data", str(data_file.name),
            ],
        )
        assert result.exit_code == 2
        assert "Missing option '--goal' / '-g'." in _plain(result.output)

    def test_interactive_prompts_all_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_file = self._write_data(tmp_path)
        monkeypatch.setattr("soup_cli.commands.autopilot._stdin_isatty", lambda: True)
        inputs = [
            "meta-llama/Llama-3.1-8B-Instruct",
            str(data_file.name),
            "reasoning",
        ]
        result = runner.invoke(
            app,
            ["autopilot", "--dry-run", "--gpu-budget", "24GB"],
            input="\n".join(inputs) + "\n",
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        plain_out = _plain(result.output)
        assert "Soup Autopilot" in plain_out
        assert "meta-llama/Llama-3.1-8B-Instruct" in plain_out
        assert "task: grpo" in plain_out

    def test_interactive_prompts_missing_goal_only(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_file = self._write_data(tmp_path)
        monkeypatch.setattr("soup_cli.commands.autopilot._stdin_isatty", lambda: True)
        result = runner.invoke(
            app,
            [
                "autopilot",
                "--model", "meta-llama/Llama-3.1-8B-Instruct",
                "--data", str(data_file.name),
                "--gpu-budget", "24GB",
                "--dry-run",
            ],
            input="reasoning\n",
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        plain_out = _plain(result.output)
        assert "task: grpo" in plain_out

    def test_interactive_mode_does_not_prompt_when_all_flags_given(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        data_file = self._write_data(tmp_path)
        monkeypatch.setattr("soup_cli.commands.autopilot._stdin_isatty", lambda: True)

        def _fail_on_prompt(*args, **kwargs):
            pytest.fail(f"Prompt.ask called unexpectedly: {args}, {kwargs}")

        monkeypatch.setattr("soup_cli.commands.autopilot.Prompt.ask", _fail_on_prompt)

        result = runner.invoke(
            app,
            [
                "autopilot",
                "--model", "meta-llama/Llama-3.1-8B-Instruct",
                "--data", str(data_file.name),
                "--goal", "chat",
                "--gpu-budget", "24GB",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_interactive_empty_model_exits_1(self, monkeypatch):
        monkeypatch.setattr("soup_cli.commands.autopilot._stdin_isatty", lambda: True)
        result = runner.invoke(app, ["autopilot"], input="\n")
        assert result.exit_code == 1
        assert "Base model cannot be empty." in _plain(result.output)

    def test_interactive_empty_data_exits_1(self, monkeypatch):
        monkeypatch.setattr("soup_cli.commands.autopilot._stdin_isatty", lambda: True)
        result = runner.invoke(
            app,
            ["autopilot", "--model", "meta-llama/Llama-3.1-8B-Instruct"],
            input="\n",
        )
        assert result.exit_code == 1
        assert "Dataset path cannot be empty." in _plain(result.output)


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

