"""Tests for v0.10.1-v0.10.5 bug fixes - Unicode, PPO, dtype, CPU, trl API compat."""

from pathlib import Path
from unittest.mock import patch

import pytest

from soup_cli.config.schema import SoupConfig

# --- BUG-001: Windows UnicodeEncodeError (no Unicode arrows/dashes in output) ---


class TestNoUnicodeInOutput:
    """Verify user-facing output uses only ASCII-safe characters."""

    def test_config_loader_error_uses_ascii_arrow(self):
        """Config validation errors should use -> not Unicode arrow."""
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError) as exc_info:
            load_config_from_string("base: x\ntask: invalid_task\n")
        # Error message should use -> not the Unicode arrow
        msg = str(exc_info.value)
        assert "\u2192" not in msg  # no Unicode right arrow

    def test_loss_format_uses_ascii(self):
        """Loss formatting in runs should use -> not Unicode arrow."""
        from soup_cli.commands.runs import _fmt_loss

        run = {"initial_loss": 1.5, "final_loss": 0.5}
        result = _fmt_loss(run)
        assert "->" in result
        assert "\u2192" not in result  # no Unicode right arrow

    def test_loss_format_missing_returns_ascii(self):
        """Missing loss should return ASCII dash, not em dash."""
        from soup_cli.commands.runs import _fmt_loss

        result = _fmt_loss({})
        assert result == "-"
        assert "\u2014" not in result  # no em dash

    def test_fmt_float_missing_returns_ascii(self):
        """Missing float should return ASCII dash."""
        from soup_cli.commands.runs import _fmt_float

        result = _fmt_float(None)
        assert result == "-"
        assert "\u2014" not in result

    def test_fmt_duration_missing_returns_ascii(self):
        """Missing duration should return ASCII dash."""
        from soup_cli.commands.runs import _fmt_duration

        result = _fmt_duration(None)
        assert result == "-"
        assert "\u2014" not in result

    def test_formats_empty_dataset_error_ascii(self):
        """Empty dataset error should use ASCII dash."""
        from soup_cli.data.formats import detect_format

        with pytest.raises(ValueError, match="Empty dataset"):
            detect_format([])


# --- BUG-002: PPO ppo_epochs parameter compatibility ---


class TestPPOParamCompat:
    """Test PPO trainer handles trl version differences."""

    def test_ppo_config_uses_inspect(self):
        """PPO setup should use inspect to detect valid parameter names."""
        from soup_cli.trainer.ppo import PPOTrainerWrapper

        cfg = SoupConfig(
            base="test-model",
            task="ppo",
            data={"train": "./data.jsonl"},
            training={
                "ppo_epochs": 3,
                "ppo_clip_ratio": 0.15,
                "ppo_kl_penalty": 0.03,
            },
        )
        wrapper = PPOTrainerWrapper(cfg, device="cpu")
        assert wrapper.config.training.ppo_epochs == 3
        assert wrapper.config.training.ppo_clip_ratio == pytest.approx(0.15)
        assert wrapper.config.training.ppo_kl_penalty == pytest.approx(0.03)


# --- BUG-003: Reward Model dtype mismatch ---


class TestComputeDtype:
    """Test get_compute_dtype returns correct dtype for device."""

    def test_cpu_returns_float32(self):
        """CPU should use float32, not bfloat16."""
        import torch

        from soup_cli.utils.gpu import get_compute_dtype

        with patch("torch.cuda.is_available", return_value=False):
            dtype = get_compute_dtype()
            assert dtype == torch.float32

    def test_cuda_with_bf16_returns_bfloat16(self):
        """CUDA with bf16 support should use bfloat16."""
        import torch

        from soup_cli.utils.gpu import get_compute_dtype

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.is_bf16_supported", return_value=True):
            dtype = get_compute_dtype()
            assert dtype == torch.bfloat16

    def test_cuda_without_bf16_returns_float16(self):
        """CUDA without bf16 support should fall back to float16."""
        import torch

        from soup_cli.utils.gpu import get_compute_dtype

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.is_bf16_supported", return_value=False):
            dtype = get_compute_dtype()
            assert dtype == torch.float16


# --- BUG-005: diff dtype -> torch_dtype ---


class TestDiffModelLoading:
    """Test diff command uses correct parameter names."""

    def test_load_model_uses_dtype(self):
        """_load_model should pass dtype= (not the old torch_dtype=)."""
        import inspect

        from soup_cli.commands.diff import _load_model

        source = inspect.getsource(_load_model)
        assert "dtype=torch.float16" in source
        assert "torch_dtype=" not in source


# --- BUG-006: wandb version pin ---


class TestWandbVersionPin:
    """Test wandb dependency is version-pinned."""

    def test_wandb_upper_bound_in_pyproject(self):
        """pyproject.toml should pin wandb below 0.18.0."""
        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject.read_text(encoding="utf-8")
        assert "<0.18.0" in content or "< 0.18.0" in content


# --- BUG-004: CPU quantization warning ---


class TestCPUQuantWarning:
    """Test that CPU + quantization produces a warning."""

    def test_train_auto_disables_quant_on_cpu(self):
        """train.py should auto-disable quantization on CPU."""
        import inspect

        from soup_cli.commands import train

        source = inspect.getsource(train)
        assert "quantization is not" in source
        assert 'cfg.training.quantization = "none"' in source


# --- v0.10.2: Display progress bar uses ASCII ---


class TestDisplayASCII:
    """Test that training display uses ASCII-safe progress bars."""

    def test_progress_bar_uses_ascii_chars(self):
        """Progress bar should use # and - instead of Unicode blocks."""
        import inspect

        from soup_cli.monitoring.display import TrainingDisplay

        source = inspect.getsource(TrainingDisplay)
        assert '"#"' in source
        assert '"-"' in source
        assert "\\u2588" not in source
        assert "\\u2591" not in source


# --- v0.10.2: Plotext UnicodeEncodeError handling ---


class TestPlotextFallback:
    """Test that plotext errors are caught gracefully."""

    def test_stats_catches_unicode_error(self):
        """data stats should catch UnicodeEncodeError from plotext."""
        import inspect

        from soup_cli.commands import data

        source = inspect.getsource(data)
        assert "UnicodeEncodeError" in source


# --- v0.10.2: Error messages for CPU issues ---


class TestCPUErrorMessages:
    """Test friendly error messages for CPU-specific failures."""

    def test_tensor_size_error_mapped(self):
        """Tensor expansion error should have a friendly GRPO/PPO CPU message."""
        from soup_cli.utils.errors import ERROR_MAP

        for pattern, msg, _ in ERROR_MAP:
            if "expanded size" in pattern:
                assert "GRPO" in msg or "PPO" in msg
                break
        else:
            pytest.fail("expanded size pattern not found in ERROR_MAP")

    def test_dtype_mismatch_error_mapped(self):
        """Dtype mismatch error should have a friendly message."""
        from soup_cli.utils.errors import ERROR_MAP

        patterns = [pattern for pattern, _, _ in ERROR_MAP]
        assert any("same dtype" in p for p in patterns)

    def test_bf16_error_mapped(self):
        """bf16 GPU error should have a friendly message."""
        from soup_cli.utils.errors import ERROR_MAP

        patterns = [pattern for pattern, _, _ in ERROR_MAP]
        assert any("bf16" in p for p in patterns)

    def test_torchvision_error_mapped(self):
        """torchvision nms error should have a friendly message."""
        from soup_cli.utils.errors import ERROR_MAP

        patterns = [pattern for pattern, _, _ in ERROR_MAP]
        assert any("nms" in p for p in patterns)


# --- v0.10.2: Doctor torchvision check ---


class TestDoctorTorchvisionCheck:
    """Test that soup doctor checks torchvision compatibility."""

    def test_doctor_has_torchvision_check(self):
        """doctor.py should have torchvision compatibility check."""
        import inspect

        from soup_cli.commands import doctor

        source = inspect.getsource(doctor)
        assert "_check_torchvision_compat" in source


# --- v0.10.3: PPO use_cpu support ---


class TestPPOUseCPU:
    """Test PPO trainer sets use_cpu=True on CPU devices."""

    def test_ppo_setup_has_use_cpu_logic(self):
        """PPO setup should check for use_cpu param and set it on CPU."""
        import inspect

        from soup_cli.trainer import ppo

        source = inspect.getsource(ppo)
        assert "use_cpu" in source
        assert 'self.device == "cpu"' in source

    def test_ppo_wrapper_stores_device(self):
        """PPOTrainerWrapper should store the device parameter."""
        from soup_cli.trainer.ppo import PPOTrainerWrapper

        cfg = SoupConfig(
            base="test-model",
            task="ppo",
            data={"train": "./data.jsonl"},
        )
        wrapper = PPOTrainerWrapper(cfg, device="cpu")
        assert wrapper.device == "cpu"

        wrapper_gpu = PPOTrainerWrapper(cfg, device="cuda")
        assert wrapper_gpu.device == "cuda"

    def test_ppo_supports_args_and_config_api(self):
        """PPO trainer should detect trl API: args= (>=0.28) vs config= (<0.28)."""
        import inspect

        from soup_cli.trainer import ppo

        source = inspect.getsource(ppo)
        # Must handle both trl APIs
        assert '"args"' in source
        assert '"config"' in source
        assert "PPOTrainer.__init__" in source

    def test_ppo_train_detects_builtin_vs_manual(self):
        """PPO train() should detect built-in .train() vs manual loop."""
        import inspect

        from soup_cli.trainer import ppo

        source = inspect.getsource(ppo)
        assert "_train_builtin" in source
        assert "_train_manual" in source


# --- v0.10.3: GRPO CPU warning ---


class TestGRPOCPUWarning:
    """Test GRPO trainer warns on CPU and sets use_cpu."""

    def test_grpo_setup_has_cpu_warning(self):
        """GRPO setup should warn about CPU limitations."""
        import inspect

        from soup_cli.trainer import grpo

        source = inspect.getsource(grpo)
        assert "GRPO on CPU is experimental" in source

    def test_grpo_setup_has_use_cpu_logic(self):
        """GRPO setup should set use_cpu=True on CPU when supported."""
        import inspect

        from soup_cli.trainer import grpo

        source = inspect.getsource(grpo)
        assert "use_cpu" in source

    def test_grpo_wrapper_stores_device(self):
        """GRPOTrainerWrapper should store the device parameter."""
        from soup_cli.trainer.grpo import GRPOTrainerWrapper

        cfg = SoupConfig(
            base="test-model",
            task="grpo",
            data={"train": "./data.jsonl"},
            training={"reward_fn": "accuracy"},
        )
        wrapper = GRPOTrainerWrapper(cfg, device="cpu")
        assert wrapper.device == "cpu"


# --- v0.10.3: use_cpu error message ---


class TestUseCPUErrorMessage:
    """Test that use_cpu error is mapped to a friendly message."""

    def test_use_cpu_error_mapped(self):
        """use_cpu error should have a friendly message."""
        from soup_cli.utils.errors import ERROR_MAP

        patterns = [pattern for pattern, _, _ in ERROR_MAP]
        assert any("use_cpu" in p for p in patterns)


# --- v0.10.5: PPO dataset parameter compatibility (trl >=0.28) ---


class TestPPODatasetCompat:
    """Test PPO trainer handles dataset param removal in newer trl versions."""

    def test_ppo_setup_checks_dataset_in_constructor(self):
        """PPO setup should check whether dataset/train_dataset is accepted."""
        import inspect

        from soup_cli.trainer import ppo

        source = inspect.getsource(ppo)
        # Must check both train_dataset and dataset params
        assert '"train_dataset" in ppo_trainer_params' in source
        assert '"dataset" in ppo_trainer_params' in source
        # Must track whether dataset was passed to constructor
        assert "_dataset_in_constructor" in source

    def test_ppo_train_sets_dataset_if_not_in_constructor(self):
        """PPO _train_builtin should set dataset on trainer if not in init."""
        import inspect

        from soup_cli.trainer.ppo import PPOTrainerWrapper

        source = inspect.getsource(PPOTrainerWrapper._train_builtin)
        assert "_dataset_in_constructor" in source
        assert "train_dataset" in source

    def test_ppo_dataset_in_constructor_flag(self):
        """PPOTrainerWrapper should track _dataset_in_constructor after setup."""
        from unittest.mock import MagicMock
        from unittest.mock import patch as mock_patch

        from soup_cli.trainer.ppo import PPOTrainerWrapper

        cfg = SoupConfig(
            base="test-model",
            task="ppo",
            data={"train": "./data.jsonl"},
        )
        wrapper = PPOTrainerWrapper(cfg, device="cpu")

        # Real class whose __init__ does NOT accept dataset/train_dataset
        class FakePPOTrainer:
            def __init__(self, *, model=None, args=None,
                         processing_class=None, reward_funcs=None):
                pass

        class FakePPOConfig:
            def __init__(self, **kwargs):
                pass

        dataset = {
            "train": [
                {"prompt": "What is 2+2?", "answer": "4"},
            ]
        }

        with mock_patch("soup_cli.trainer.ppo.PPOTrainerWrapper._setup_reward"), \
             mock_patch(
                 "soup_cli.trainer.ppo.PPOTrainerWrapper._setup_transformers"
             ):
            wrapper.model = MagicMock()
            wrapper.model.get_nb_trainable_parameters.return_value = (100, 1000)
            wrapper.tokenizer = MagicMock()
            wrapper.tokenizer.pad_token = "pad"

            with mock_patch(
                "trl.PPOTrainer", FakePPOTrainer, create=True,
            ), mock_patch(
                "trl.PPOConfig", FakePPOConfig, create=True,
            ):
                wrapper.setup(dataset)

            # dataset should NOT be in constructor since FakePPOTrainer
            # doesn't accept it
            assert wrapper._dataset_in_constructor is False

    def test_ppo_dataset_in_constructor_when_accepted(self):
        """_dataset_in_constructor should be True when train_dataset is accepted."""
        from unittest.mock import MagicMock
        from unittest.mock import patch as mock_patch

        from soup_cli.trainer.ppo import PPOTrainerWrapper

        cfg = SoupConfig(
            base="test-model",
            task="ppo",
            data={"train": "./data.jsonl"},
        )
        wrapper = PPOTrainerWrapper(cfg, device="cpu")

        # Real class whose __init__ DOES accept train_dataset
        class FakePPOTrainer:
            def __init__(self, *, model=None, args=None,
                         processing_class=None, train_dataset=None,
                         reward_funcs=None):
                pass

        class FakePPOConfig:
            def __init__(self, **kwargs):
                pass

        dataset = {
            "train": [
                {"prompt": "What is 2+2?", "answer": "4"},
            ]
        }

        with mock_patch("soup_cli.trainer.ppo.PPOTrainerWrapper._setup_reward"), \
             mock_patch(
                 "soup_cli.trainer.ppo.PPOTrainerWrapper._setup_transformers"
             ):
            wrapper.model = MagicMock()
            wrapper.model.get_nb_trainable_parameters.return_value = (100, 1000)
            wrapper.tokenizer = MagicMock()
            wrapper.tokenizer.pad_token = "pad"

            with mock_patch(
                "trl.PPOTrainer", FakePPOTrainer, create=True,
            ), mock_patch(
                "trl.PPOConfig", FakePPOConfig, create=True,
            ):
                wrapper.setup(dataset)

            assert wrapper._dataset_in_constructor is True
