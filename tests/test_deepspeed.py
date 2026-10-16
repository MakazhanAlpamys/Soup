"""Tests for Multi-GPU / DeepSpeed support."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest


class TestDeepSpeedConfigs:
    """Test DeepSpeed configuration templates."""

    def test_zero2_config_structure(self):
        """ZeRO Stage 2 config should have correct structure."""
        from soup_cli.utils.deepspeed import get_deepspeed_config

        config = get_deepspeed_config("zero2")
        assert config["zero_optimization"]["stage"] == 2
        assert config["bf16"]["enabled"] is True
        assert config["gradient_accumulation_steps"] == "auto"

    def test_zero3_config_structure(self):
        """ZeRO Stage 3 config should have correct structure."""
        from soup_cli.utils.deepspeed import get_deepspeed_config

        config = get_deepspeed_config("zero3")
        assert config["zero_optimization"]["stage"] == 3
        assert config["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] is True

    def test_zero2_offload_config(self):
        """ZeRO Stage 2 with offload should enable CPU offloading."""
        from soup_cli.utils.deepspeed import get_deepspeed_config

        config = get_deepspeed_config("zero2_offload")
        assert config["zero_optimization"]["stage"] == 2
        offload = config["zero_optimization"]["offload_optimizer"]
        assert offload["device"] == "cpu"
        assert offload["pin_memory"] is True

    def test_zero3_offload_config(self):
        """The configuration a VRAM-constrained user actually wants: stage 3
        with the PARAMETERS on the CPU. Until this preset existed the only
        offload config Soup shipped was stage 2, optimizer-only, so the
        H100 comparison in benchmarks/gate-h100-validation.md (STEP 3) had to
        supply hand-written JSON."""
        from soup_cli.utils.deepspeed import get_deepspeed_config

        config = get_deepspeed_config("zero3_offload")
        assert config["zero_optimization"]["stage"] == 3
        offload_param = config["zero_optimization"]["offload_param"]
        assert offload_param["device"] == "cpu"
        assert offload_param["pin_memory"] is True
        assert config["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] is True
        assert config["bf16"]["enabled"] is True

    def test_zero3_offload_does_not_require_optimizer_offload(self):
        """`offload_optimizer: cpu` JIT-builds DeepSpeed's `cpu_adam` op and
        needs a matching CUDA toolkit; on a box without `nvcc` it fails with
        `CUDAMismatchException` and then `'DeepSpeedCPUAdam' object has no
        attribute 'ds_opt_adam'` (measured, STEP 3 of the H100 record). Making
        it mandatory would make the preset unusable on such a box, so it is off
        — which is also the fairer comparison against layer streaming, whose
        optimizer covers only the LoRA parameters and stays on the GPU."""
        from soup_cli.utils.deepspeed import get_deepspeed_config

        config = get_deepspeed_config("zero3_offload")
        assert config["zero_optimization"]["offload_optimizer"]["device"] == "none"

    def test_zero3_offload_is_the_only_new_thing_zero3_is_not_touched(self):
        """A regression guard with a control: the plain `zero3` preset must keep
        offloading nothing, or every existing multi-GPU run silently changes
        behaviour."""
        from soup_cli.utils.deepspeed import get_deepspeed_config

        plain = get_deepspeed_config("zero3")
        assert plain["zero_optimization"]["offload_param"]["device"] == "none"
        assert plain["zero_optimization"]["offload_optimizer"]["device"] == "none"

    def test_invalid_config_name(self):
        """Should raise ValueError for unknown config name."""
        from soup_cli.utils.deepspeed import get_deepspeed_config

        with pytest.raises(ValueError, match="Unknown DeepSpeed config"):
            get_deepspeed_config("zero99")

    def test_get_config_returns_copy(self):
        """Should return a copy, not the original."""
        from soup_cli.utils.deepspeed import get_deepspeed_config

        config1 = get_deepspeed_config("zero2")
        config2 = get_deepspeed_config("zero2")
        config1["bf16"]["enabled"] = False
        assert config2["bf16"]["enabled"] is True

    def test_all_configs_have_auto_fields(self):
        """All configs should have 'auto' for batch sizes."""
        from soup_cli.utils.deepspeed import CONFIGS

        for name, config in CONFIGS.items():
            assert config["train_batch_size"] == "auto", f"{name} missing auto train_batch_size"
            assert config["train_micro_batch_size_per_gpu"] == "auto", (
                f"{name} missing auto micro batch"
            )


class TestWriteDeepSpeedConfig:
    """Test writing DeepSpeed config to temp file."""

    def test_write_creates_file(self):
        """Should create a valid JSON file."""
        from soup_cli.utils.deepspeed import write_deepspeed_config

        path = write_deepspeed_config("zero2")
        assert os.path.exists(path)

        with open(path) as f:
            config = json.load(f)
        assert config["zero_optimization"]["stage"] == 2

        # Cleanup
        os.unlink(path)

    def test_write_file_is_valid_json(self):
        """Written file should be parseable JSON."""
        from soup_cli.utils.deepspeed import write_deepspeed_config

        for stage in ["zero2", "zero3", "zero2_offload", "zero3_offload"]:
            path = write_deepspeed_config(stage)
            with open(path) as f:
                config = json.load(f)
            assert "zero_optimization" in config
            os.unlink(path)


class TestDetectMultiGPU:
    """Test multi-GPU detection."""

    def test_detect_no_gpu(self):
        """Should return 0 GPUs when CUDA not available."""
        from soup_cli.utils.deepspeed import detect_multi_gpu

        with patch("torch.cuda.is_available", return_value=False):
            result = detect_multi_gpu()
            assert result["gpu_count"] == 0
            assert result["gpus"] == []

    def test_detect_single_gpu(self):
        """Should detect a single GPU."""
        from soup_cli.utils.deepspeed import detect_multi_gpu

        mock_props = MagicMock()
        mock_props.name = "NVIDIA RTX 4090"
        mock_props.total_memory = 24 * (1024 ** 3)  # 24GB

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=1), \
             patch("torch.cuda.get_device_properties", return_value=mock_props):
            result = detect_multi_gpu()
            assert result["gpu_count"] == 1
            assert len(result["gpus"]) == 1
            assert result["gpus"][0]["name"] == "NVIDIA RTX 4090"
            assert result["gpus"][0]["memory_gb"] == pytest.approx(24.0)

    def test_detect_multiple_gpus(self):
        """Should detect multiple GPUs."""
        from soup_cli.utils.deepspeed import detect_multi_gpu

        mock_props = MagicMock()
        mock_props.name = "NVIDIA A100"
        mock_props.total_memory = 80 * (1024 ** 3)

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=4), \
             patch("torch.cuda.get_device_properties", return_value=mock_props):
            result = detect_multi_gpu()
            assert result["gpu_count"] == 4
            assert len(result["gpus"]) == 4

    def test_detect_without_torch(self):
        """Should handle missing torch gracefully."""
        from soup_cli.utils.deepspeed import detect_multi_gpu

        with patch.dict("sys.modules", {"torch": None}):
            # Import error should be caught
            result = detect_multi_gpu()
            assert result["gpu_count"] == 0


class TestResolveDeepSpeed:
    """Test DeepSpeed config resolution in train command."""

    def test_resolve_named_preset(self):
        """Should resolve named presets like 'zero2'."""
        from soup_cli.commands.train import _resolve_deepspeed

        path = _resolve_deepspeed("zero2")
        assert os.path.exists(path)

        with open(path) as f:
            config = json.load(f)
        assert config["zero_optimization"]["stage"] == 2
        os.unlink(path)

    def test_resolve_zero3_offload_preset(self):
        """`--deepspeed zero3_offload` must reach the same resolver path every
        other preset does, or the config exists but is unreachable from the CLI.
        """
        from soup_cli.commands.train import _resolve_deepspeed

        path = _resolve_deepspeed("zero3_offload")
        try:
            with open(path) as f:
                config = json.load(f)
            assert config["zero_optimization"]["stage"] == 3
            assert config["zero_optimization"]["offload_param"]["device"] == "cpu"
        finally:
            os.unlink(path)

    def test_resolve_json_file(self, tmp_path):
        """Should resolve path to JSON file."""
        from soup_cli.commands.train import _resolve_deepspeed

        config_file = tmp_path / "ds_config.json"
        config_file.write_text(json.dumps({"zero_optimization": {"stage": 2}}))

        result = _resolve_deepspeed(str(config_file))
        assert result == str(config_file)

    def test_resolve_invalid_name(self):
        """Should raise exit for invalid name."""
        from click.exceptions import Exit

        from soup_cli.commands.train import _resolve_deepspeed

        with pytest.raises(Exit):
            _resolve_deepspeed("invalid_config")


class TestTrainerDeepSpeedParam:
    """Test that trainers accept deepspeed_config parameter."""

    def test_sft_trainer_accepts_deepspeed(self):
        """SFTTrainerWrapper should accept deepspeed_config."""
        from soup_cli.config.schema import SoupConfig
        from soup_cli.trainer.sft import SFTTrainerWrapper

        cfg = SoupConfig(
            base="test-model",
            data={"train": "test.jsonl"},
        )
        wrapper = SFTTrainerWrapper(cfg, device="cpu", deepspeed_config="/tmp/ds.json")
        assert wrapper.deepspeed_config == "/tmp/ds.json"

    def test_dpo_trainer_accepts_deepspeed(self):
        """DPOTrainerWrapper should accept deepspeed_config."""
        from soup_cli.config.schema import SoupConfig
        from soup_cli.trainer.dpo import DPOTrainerWrapper

        cfg = SoupConfig(
            base="test-model",
            task="dpo",
            data={"train": "test.jsonl"},
        )
        wrapper = DPOTrainerWrapper(cfg, device="cpu", deepspeed_config="/tmp/ds.json")
        assert wrapper.deepspeed_config == "/tmp/ds.json"

    def test_sft_trainer_default_no_deepspeed(self):
        """SFTTrainerWrapper should default to no DeepSpeed."""
        from soup_cli.config.schema import SoupConfig
        from soup_cli.trainer.sft import SFTTrainerWrapper

        cfg = SoupConfig(
            base="test-model",
            data={"train": "test.jsonl"},
        )
        wrapper = SFTTrainerWrapper(cfg, device="cpu")
        assert wrapper.deepspeed_config is None


class TestTrainDeepSpeedFlag:
    """Test --deepspeed flag in train command."""

    def test_train_help_shows_deepspeed(self):
        """Train help should mention --deepspeed option."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["train", "--help"])
        assert "deepspeed" in result.output.lower()
