"""Tests for soup profile command — memory/speed estimator."""

import json

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()


# --- Memory estimation tests ---


class TestEstimateModelMemory:
    """Test model memory estimation for various sizes and quantizations."""

    def test_7b_4bit(self):
        from soup_cli.utils.profiler import estimate_model_memory

        mem_gb = estimate_model_memory(7.0, "4bit")
        assert 3.0 <= mem_gb <= 5.0  # ~3.5 GB for 7B 4bit

    def test_7b_8bit(self):
        from soup_cli.utils.profiler import estimate_model_memory

        mem_gb = estimate_model_memory(7.0, "8bit")
        assert 6.0 <= mem_gb <= 9.0  # ~7 GB for 7B 8bit

    def test_7b_none(self):
        from soup_cli.utils.profiler import estimate_model_memory

        mem_gb = estimate_model_memory(7.0, "none")
        assert 13.0 <= mem_gb <= 16.0  # ~14 GB for 7B FP16

    def test_1b_4bit(self):
        from soup_cli.utils.profiler import estimate_model_memory

        mem_gb = estimate_model_memory(1.0, "4bit")
        assert 0.3 <= mem_gb <= 1.5

    def test_13b_4bit(self):
        from soup_cli.utils.profiler import estimate_model_memory

        mem_gb = estimate_model_memory(13.0, "4bit")
        assert 5.0 <= mem_gb <= 9.0

    def test_70b_4bit(self):
        from soup_cli.utils.profiler import estimate_model_memory

        mem_gb = estimate_model_memory(70.0, "4bit")
        assert 30.0 <= mem_gb <= 45.0


class TestEstimateLoraMemory:
    """Test LoRA parameter memory estimation."""

    def test_default_lora(self):
        from soup_cli.utils.profiler import estimate_lora_memory

        mem_gb = estimate_lora_memory(7.0, lora_r=64, lora_alpha=16)
        assert 0.01 <= mem_gb <= 1.0

    def test_higher_rank(self):
        from soup_cli.utils.profiler import estimate_lora_memory

        low = estimate_lora_memory(7.0, lora_r=16, lora_alpha=16)
        high = estimate_lora_memory(7.0, lora_r=128, lora_alpha=32)
        assert high > low

    def test_small_model(self):
        from soup_cli.utils.profiler import estimate_lora_memory

        mem_gb = estimate_lora_memory(1.0, lora_r=8, lora_alpha=16)
        assert mem_gb >= 0


class TestEstimateOptimizerMemory:
    """Test optimizer memory estimation."""

    def test_adam_optimizer(self):
        from soup_cli.utils.profiler import estimate_optimizer_memory

        # ~13M trainable params for 7B with LoRA r=64
        mem_gb = estimate_optimizer_memory(13_000_000, "adamw_torch")
        assert 0.01 <= mem_gb <= 0.5

    def test_adam_8bit(self):
        from soup_cli.utils.profiler import estimate_optimizer_memory

        adam = estimate_optimizer_memory(13_000_000, "adamw_torch")
        adam8 = estimate_optimizer_memory(13_000_000, "adamw_bnb_8bit")
        assert adam8 < adam

    def test_sgd_optimizer(self):
        from soup_cli.utils.profiler import estimate_optimizer_memory

        adam = estimate_optimizer_memory(13_000_000, "adamw_torch")
        sgd = estimate_optimizer_memory(13_000_000, "sgd")
        assert sgd < adam


class TestEstimateActivationMemory:
    """Test activation memory estimation."""

    def test_basic(self):
        from soup_cli.utils.profiler import estimate_activation_memory

        mem_gb = estimate_activation_memory(
            batch_size=4, seq_len=2048, hidden_size=4096, num_layers=32
        )
        assert mem_gb > 0

    def test_larger_batch_more_memory(self):
        from soup_cli.utils.profiler import estimate_activation_memory

        small = estimate_activation_memory(
            batch_size=1, seq_len=2048, hidden_size=4096, num_layers=32
        )
        large = estimate_activation_memory(
            batch_size=8, seq_len=2048, hidden_size=4096, num_layers=32
        )
        assert large > small

    def test_longer_seq_more_memory(self):
        from soup_cli.utils.profiler import estimate_activation_memory

        short = estimate_activation_memory(
            batch_size=4, seq_len=512, hidden_size=4096, num_layers=32
        )
        long = estimate_activation_memory(
            batch_size=4, seq_len=4096, hidden_size=4096, num_layers=32
        )
        assert long > short

    def test_gradient_checkpointing_reduces_memory(self):
        from soup_cli.utils.profiler import estimate_activation_memory

        normal = estimate_activation_memory(
            batch_size=4, seq_len=2048, hidden_size=4096, num_layers=32,
            gradient_checkpointing=False,
        )
        checkpointed = estimate_activation_memory(
            batch_size=4, seq_len=2048, hidden_size=4096, num_layers=32,
            gradient_checkpointing=True,
        )
        assert checkpointed < normal


class TestEstimateTotal:
    """Test total profile estimation from SoupConfig."""

    def test_basic_profile(self):
        from soup_cli.utils.profiler import estimate_total

        result = estimate_total(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            model_params_b=8.0,
            quantization="4bit",
            lora_r=64,
            lora_alpha=16,
            batch_size=4,
            seq_len=2048,
            optimizer="adamw_torch",
            gradient_checkpointing=False,
        )
        assert "model_memory_gb" in result
        assert "lora_memory_gb" in result
        assert "optimizer_memory_gb" in result
        assert "activation_memory_gb" in result
        assert "total_memory_gb" in result
        assert result["total_memory_gb"] > 0

    def test_total_is_sum_of_parts(self):
        from soup_cli.utils.profiler import estimate_total

        result = estimate_total(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            model_params_b=8.0,
            quantization="4bit",
            lora_r=64,
            lora_alpha=16,
            batch_size=4,
            seq_len=2048,
            optimizer="adamw_torch",
            gradient_checkpointing=False,
        )
        parts_sum = (
            result["model_memory_gb"]
            + result["lora_memory_gb"]
            + result["optimizer_memory_gb"]
            + result["activation_memory_gb"]
        )
        # Total includes overhead, so should be >= parts_sum
        assert result["total_memory_gb"] >= parts_sum


class TestEstimateSpeed:
    """Test speed estimation."""

    def test_basic_speed(self):
        from soup_cli.utils.profiler import estimate_speed

        tokens_per_sec = estimate_speed(model_params_b=7.0, quantization="4bit", batch_size=4)
        assert tokens_per_sec > 0

    def test_larger_model_slower(self):
        from soup_cli.utils.profiler import estimate_speed

        small = estimate_speed(model_params_b=7.0, quantization="4bit", batch_size=4)
        large = estimate_speed(model_params_b=70.0, quantization="4bit", batch_size=4)
        assert small > large

    def test_no_quantization_slower(self):
        from soup_cli.utils.profiler import estimate_speed

        quantized = estimate_speed(model_params_b=7.0, quantization="4bit", batch_size=4)
        full = estimate_speed(model_params_b=7.0, quantization="none", batch_size=4)
        assert quantized > full


class TestEstimateTrainingTime:
    """Test training time estimation."""

    def test_basic_time(self):
        from soup_cli.utils.profiler import estimate_training_time

        minutes = estimate_training_time(
            dataset_size=10000, epochs=3, samples_per_sec=0.5
        )
        assert minutes > 0

    def test_more_epochs_more_time(self):
        from soup_cli.utils.profiler import estimate_training_time

        time_1 = estimate_training_time(dataset_size=10000, epochs=1, samples_per_sec=0.5)
        time_3 = estimate_training_time(dataset_size=10000, epochs=3, samples_per_sec=0.5)
        assert time_3 == pytest.approx(time_1 * 3, rel=0.01)

    def test_zero_speed_returns_inf(self):
        from soup_cli.utils.profiler import estimate_training_time

        minutes = estimate_training_time(
            dataset_size=10000, epochs=3, samples_per_sec=0.0
        )
        assert minutes == float("inf")


class TestRecommendBatchSize:
    """Test batch size recommendation."""

    def test_basic_recommendation(self):
        from soup_cli.utils.profiler import recommend_batch_size

        bs = recommend_batch_size(total_memory_gb=6.0, gpu_memory_gb=24.0)
        assert bs >= 1

    def test_tight_memory_returns_1(self):
        from soup_cli.utils.profiler import recommend_batch_size

        bs = recommend_batch_size(total_memory_gb=23.0, gpu_memory_gb=24.0)
        assert bs >= 1

    def test_more_gpu_memory_higher_batch(self):
        from soup_cli.utils.profiler import recommend_batch_size

        small_gpu = recommend_batch_size(total_memory_gb=6.0, gpu_memory_gb=8.0)
        big_gpu = recommend_batch_size(total_memory_gb=6.0, gpu_memory_gb=80.0)
        assert big_gpu >= small_gpu


class TestRecommendGpu:
    """Test GPU recommendation logic."""

    def test_small_model(self):
        from soup_cli.utils.profiler import recommend_gpu

        gpus = recommend_gpu(total_memory_gb=5.0)
        assert len(gpus) > 0
        # Should include most GPUs
        assert any("RTX" in g or "A100" in g or "H100" in g for g in gpus)

    def test_large_model(self):
        from soup_cli.utils.profiler import recommend_gpu

        gpus = recommend_gpu(total_memory_gb=50.0)
        # Only large GPUs should be recommended
        assert all("RTX 3060" not in g for g in gpus)

    def test_very_large_model(self):
        from soup_cli.utils.profiler import recommend_gpu

        gpus = recommend_gpu(total_memory_gb=200.0)
        # Should suggest multi-GPU or mention no single GPU fits
        assert isinstance(gpus, list)


class TestModelArchLookup:
    """Test model architecture lookup for hidden_size/num_layers."""

    def test_known_model_8b(self):
        from soup_cli.utils.profiler import get_model_arch

        arch = get_model_arch("meta-llama/Llama-3.1-8B-Instruct", 8.0)
        assert arch["hidden_size"] > 0
        assert arch["num_layers"] > 0

    def test_unknown_model_uses_params(self):
        from soup_cli.utils.profiler import get_model_arch

        arch = get_model_arch("unknown/model-7b", 7.0)
        assert arch["hidden_size"] > 0
        assert arch["num_layers"] > 0

    def test_70b_model(self):
        from soup_cli.utils.profiler import get_model_arch

        arch = get_model_arch("meta-llama/Llama-3.1-70B-Instruct", 70.0)
        assert arch["hidden_size"] >= 8192
        assert arch["num_layers"] >= 80


# --- CLI tests ---


class TestProfileCLI:
    """Test the profile CLI command."""

    def test_profile_with_config_file(self, tmp_path):
        """soup profile --config soup.yaml shows profile panel."""
        config_file = tmp_path / "soup.yaml"
        config_file.write_text(
            "base: meta-llama/Llama-3.1-8B-Instruct\n"
            "task: sft\n"
            "data:\n"
            "  train: ./data/train.jsonl\n"
            "  max_length: 2048\n"
            "training:\n"
            "  epochs: 3\n"
            "  batch_size: 4\n"
            "  quantization: 4bit\n"
            "  lora:\n"
            "    r: 64\n"
            "    alpha: 16\n"
            "output: ./output\n"
        )
        result = runner.invoke(app, ["profile", "--config", str(config_file)])
        assert result.exit_code == 0
        assert "Training Profile" in result.output

    def test_profile_shows_memory(self, tmp_path):
        """Profile output includes memory breakdown."""
        config_file = tmp_path / "soup.yaml"
        config_file.write_text(
            "base: meta-llama/Llama-3.1-8B-Instruct\n"
            "task: sft\n"
            "data:\n"
            "  train: ./data/train.jsonl\n"
            "  max_length: 2048\n"
            "training:\n"
            "  batch_size: 4\n"
            "  quantization: 4bit\n"
            "  lora:\n"
            "    r: 64\n"
            "output: ./output\n"
        )
        result = runner.invoke(app, ["profile", "--config", str(config_file)])
        assert result.exit_code == 0
        assert "Model" in result.output
        assert "LoRA" in result.output
        assert "Total" in result.output

    def test_profile_shows_recommendations(self, tmp_path):
        """Profile output includes recommendations."""
        config_file = tmp_path / "soup.yaml"
        config_file.write_text(
            "base: meta-llama/Llama-3.1-8B-Instruct\n"
            "task: sft\n"
            "data:\n"
            "  train: ./data/train.jsonl\n"
            "  max_length: 2048\n"
            "training:\n"
            "  batch_size: 4\n"
            "  quantization: 4bit\n"
            "  lora:\n"
            "    r: 64\n"
            "output: ./output\n"
        )
        result = runner.invoke(app, ["profile", "--config", str(config_file)])
        assert result.exit_code == 0
        assert "Recommend" in result.output or "GPU" in result.output

    def test_profile_json_output(self, tmp_path):
        """soup profile --json outputs valid JSON."""
        config_file = tmp_path / "soup.yaml"
        config_file.write_text(
            "base: meta-llama/Llama-3.1-8B-Instruct\n"
            "task: sft\n"
            "data:\n"
            "  train: ./data/train.jsonl\n"
            "  max_length: 2048\n"
            "training:\n"
            "  batch_size: 4\n"
            "  quantization: 4bit\n"
            "  lora:\n"
            "    r: 64\n"
            "output: ./output\n"
        )
        result = runner.invoke(app, ["profile", "--config", str(config_file), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "total_memory_gb" in data
        assert "model_memory_gb" in data

    def test_profile_with_gpu_flag(self, tmp_path):
        """soup profile --gpu a100 uses specified GPU memory."""
        config_file = tmp_path / "soup.yaml"
        config_file.write_text(
            "base: meta-llama/Llama-3.1-8B-Instruct\n"
            "task: sft\n"
            "data:\n"
            "  train: ./data/train.jsonl\n"
            "  max_length: 2048\n"
            "training:\n"
            "  batch_size: 4\n"
            "  quantization: 4bit\n"
            "  lora:\n"
            "    r: 64\n"
            "output: ./output\n"
        )
        result = runner.invoke(
            app, ["profile", "--config", str(config_file), "--gpu", "a100"]
        )
        assert result.exit_code == 0
        assert "A100" in result.output or "80" in result.output

    def test_profile_auto_batch_size(self, tmp_path):
        """Profile with batch_size: auto estimates batch size."""
        config_file = tmp_path / "soup.yaml"
        config_file.write_text(
            "base: meta-llama/Llama-3.1-8B-Instruct\n"
            "task: sft\n"
            "data:\n"
            "  train: ./data/train.jsonl\n"
            "  max_length: 2048\n"
            "training:\n"
            "  batch_size: auto\n"
            "  quantization: 4bit\n"
            "  lora:\n"
            "    r: 64\n"
            "output: ./output\n"
        )
        result = runner.invoke(app, ["profile", "--config", str(config_file)])
        assert result.exit_code == 0

    def test_profile_missing_config(self):
        """soup profile --config nonexistent.yaml fails gracefully."""
        result = runner.invoke(app, ["profile", "--config", "nonexistent.yaml"])
        assert result.exit_code != 0

    def test_profile_no_lora(self, tmp_path):
        """Profile works for full-parameter training (no LoRA quantization=none)."""
        config_file = tmp_path / "soup.yaml"
        config_file.write_text(
            "base: meta-llama/Llama-3.1-8B-Instruct\n"
            "task: sft\n"
            "data:\n"
            "  train: ./data/train.jsonl\n"
            "  max_length: 2048\n"
            "training:\n"
            "  batch_size: 4\n"
            "  quantization: none\n"
            "  lora:\n"
            "    r: 64\n"
            "output: ./output\n"
        )
        result = runner.invoke(app, ["profile", "--config", str(config_file)])
        assert result.exit_code == 0

    def test_profile_gradient_checkpointing(self, tmp_path):
        """Profile respects gradient_checkpointing flag."""
        config_file = tmp_path / "soup.yaml"
        config_file.write_text(
            "base: meta-llama/Llama-3.1-8B-Instruct\n"
            "task: sft\n"
            "data:\n"
            "  train: ./data/train.jsonl\n"
            "  max_length: 2048\n"
            "training:\n"
            "  batch_size: 4\n"
            "  quantization: 4bit\n"
            "  gradient_checkpointing: true\n"
            "  lora:\n"
            "    r: 64\n"
            "output: ./output\n"
        )
        result = runner.invoke(app, ["profile", "--config", str(config_file)])
        assert result.exit_code == 0
        assert "gradient_checkpointing" in result.output or "checkpointing" in result.output.lower()


# --- GPU lookup table tests ---


class TestGpuLookup:
    """Test GPU memory lookup table."""

    def test_known_gpus(self):
        from soup_cli.utils.profiler import GPU_MEMORY

        assert "a100" in GPU_MEMORY
        assert "h100" in GPU_MEMORY
        assert "rtx4090" in GPU_MEMORY
        assert GPU_MEMORY["a100"] == 80
        assert GPU_MEMORY["h100"] == 80

    def test_rtx_gpus(self):
        from soup_cli.utils.profiler import GPU_MEMORY

        assert "rtx3090" in GPU_MEMORY
        assert GPU_MEMORY["rtx3090"] == 24

    def test_invalid_gpu_flag(self, tmp_path):
        """Unknown GPU name shows error."""
        config_file = tmp_path / "soup.yaml"
        config_file.write_text(
            "base: meta-llama/Llama-3.1-8B-Instruct\n"
            "task: sft\n"
            "data:\n"
            "  train: ./data/train.jsonl\n"
            "  max_length: 2048\n"
            "training:\n"
            "  batch_size: 4\n"
            "  quantization: 4bit\n"
            "  lora:\n"
            "    r: 64\n"
            "output: ./output\n"
        )
        result = runner.invoke(
            app, ["profile", "--config", str(config_file), "--gpu", "nonexistent_gpu"]
        )
        assert result.exit_code != 0


class TestEstimateModelMemoryEdge:
    """Edge case tests for model memory estimation."""

    def test_unknown_quantization_falls_back_to_fp16(self):
        from soup_cli.utils.profiler import estimate_model_memory

        mem = estimate_model_memory(7.0, "bfloat16")
        fp16_mem = estimate_model_memory(7.0, "none")
        assert mem == fp16_mem  # unknown falls back to 2.0 bytes/param

    def test_negative_samples_per_sec(self):
        from soup_cli.utils.profiler import estimate_training_time

        result = estimate_training_time(1000, 3, -1.0)
        assert result == float("inf")


class TestResolveGpuMemory:
    """Test GPU memory resolution."""

    def test_fallback_to_default_24gb(self, monkeypatch):
        from soup_cli.commands.profile import _resolve_gpu_memory

        # Force GPU detection to fail by patching inside the utils module
        monkeypatch.setattr(
            "soup_cli.utils.gpu.get_gpu_info",
            lambda: (_ for _ in ()).throw(RuntimeError("no GPU")),
        )
        mem = _resolve_gpu_memory(None)
        assert mem == 24.0


class TestTrainableParamsEstimate:
    """Test trainable parameter count estimation."""

    def test_lora_params(self):
        from soup_cli.utils.profiler import estimate_trainable_params

        params = estimate_trainable_params(
            model_params_b=7.0, lora_r=64, hidden_size=4096
        )
        assert params > 0
        # LoRA should be a small fraction of total
        assert params < 7_000_000_000 * 0.05

    def test_higher_rank_more_params(self):
        from soup_cli.utils.profiler import estimate_trainable_params

        low = estimate_trainable_params(model_params_b=7.0, lora_r=16, hidden_size=4096)
        high = estimate_trainable_params(model_params_b=7.0, lora_r=128, hidden_size=4096)
        assert high > low
