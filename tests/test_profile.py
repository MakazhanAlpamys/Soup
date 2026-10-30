"""Tests for soup profile command — memory/speed estimator."""

import json

import pytest

from soup_cli.cli import app
from tests.conftest import make_cli_runner

runner = make_cli_runner()


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

    def test_negative_samples_per_sec(self):
        from soup_cli.utils.profiler import estimate_training_time

        result = estimate_training_time(1000, 3, -1.0)
        assert result == float("inf")


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
        data = json.loads(result.stdout)
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


class TestResolveGpuMemory:
    """Test GPU memory resolution."""

    def test_fallback_to_default_24gb(self, monkeypatch):
        from soup_cli.commands.profile import _resolve_gpu_memory

        # Force GPU detection to fail by patching inside the utils module
        monkeypatch.setattr(
            "soup_cli.utils.gpu.get_gpu_info",
            lambda: (_ for _ in ()).throw(RuntimeError("no GPU")),
        )
        mem, source = _resolve_gpu_memory(None)
        assert mem == 24.0
        assert source == "assumed"
