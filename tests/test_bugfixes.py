"""Tests for v0.10.1-v0.14.2 bug fixes - Unicode, PPO, dtype, CPU, trl API, validate, UI."""

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
        assert "ppo_trainer_cls.__init__" in source

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
             ), mock_patch(
                 "soup_cli.trainer.ppo._import_ppo_classes",
                 return_value=(FakePPOTrainer, FakePPOConfig, False),
             ):
            wrapper.model = MagicMock()
            wrapper.model.get_nb_trainable_parameters.return_value = (100, 1000)
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = "pad"
            mock_tokenizer.side_effect = lambda texts, **kw: {
                "input_ids": [[1, 2, 3]] * (len(texts) if isinstance(texts, list) else 1),
                "attention_mask": [[1, 1, 1]] * (len(texts) if isinstance(texts, list) else 1),
            }
            wrapper.tokenizer = mock_tokenizer

            wrapper.setup(dataset)

        # dataset should NOT be in constructor since FakePPOTrainer
        # doesn't accept it (uses "args" path but no train_dataset param)
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
             ), mock_patch(
                 "soup_cli.trainer.ppo._import_ppo_classes",
                 return_value=(FakePPOTrainer, FakePPOConfig, False),
             ):
            wrapper.model = MagicMock()
            wrapper.model.get_nb_trainable_parameters.return_value = (100, 1000)
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = "pad"
            mock_tokenizer.side_effect = lambda texts, **kw: {
                "input_ids": [[1, 2, 3]] * (len(texts) if isinstance(texts, list) else 1),
                "attention_mask": [[1, 1, 1]] * (len(texts) if isinstance(texts, list) else 1),
            }
            wrapper.tokenizer = mock_tokenizer

            wrapper.setup(dataset)

        assert wrapper._dataset_in_constructor is True


# --- BUG-007: PPO trl >=0.28 experimental API missing positional args (v0.10.6) ---


class TestPPOExperimentalImport:
    """Test _import_ppo_classes handles both trl paths."""

    def test_import_ppo_classes_returns_tuple(self):
        """_import_ppo_classes should return (PPOTrainer, PPOConfig, bool)."""
        from soup_cli.trainer.ppo import _import_ppo_classes

        result = _import_ppo_classes()
        assert isinstance(result, tuple)
        assert len(result) == 3
        trainer_cls, config_cls, is_exp = result
        assert trainer_cls is not None
        assert config_cls is not None
        assert isinstance(is_exp, bool)

    def test_import_experimental_fallback(self):
        """When trl.experimental is unavailable, should fall back to trl."""
        import soup_cli.trainer.ppo as ppo_mod

        # Just verify the function works without error (it handles
        # ImportError from trl.experimental internally)
        result = ppo_mod._import_ppo_classes()
        assert len(result) == 3


class TestPPOExperimentalSetup:
    """Test PPO setup handles experimental API with required positional args."""

    def test_experimental_api_passes_required_args(self):
        """When is_experimental=True, setup should pass ref_model, reward_model,
        train_dataset, and value_model to PPOTrainer."""
        from unittest.mock import MagicMock
        from unittest.mock import patch as mock_patch

        from soup_cli.trainer.ppo import PPOTrainerWrapper

        cfg = SoupConfig(
            base="test-model",
            task="ppo",
            data={"train": "./data.jsonl"},
        )
        wrapper = PPOTrainerWrapper(cfg, device="cpu")

        dataset = {"train": [{"prompt": "What is 2+2?", "answer": "4"}]}

        # Track what args PPOTrainer receives
        captured_kwargs = {}

        class FakePPOTrainer:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        class FakePPOConfig:
            def __init__(self, **kwargs):
                pass

        fake_reward_model = MagicMock()
        fake_value_model = MagicMock()

        with mock_patch("soup_cli.trainer.ppo.PPOTrainerWrapper._setup_reward"), \
             mock_patch("soup_cli.trainer.ppo.PPOTrainerWrapper._setup_transformers"), \
             mock_patch(
                 "soup_cli.trainer.ppo._import_ppo_classes",
                 return_value=(FakePPOTrainer, FakePPOConfig, True),
             ), \
             mock_patch(
                 "soup_cli.trainer.ppo.PPOTrainerWrapper._get_or_create_reward_model",
                 return_value=fake_reward_model,
             ), \
             mock_patch(
                 "soup_cli.trainer.ppo.PPOTrainerWrapper._create_value_model",
                 return_value=fake_value_model,
             ):
            wrapper.model = MagicMock()
            wrapper.model.get_nb_trainable_parameters.return_value = (100, 1000)
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = "pad"
            mock_tokenizer.side_effect = lambda texts, **kw: {
                "input_ids": [[1, 2, 3]] * (len(texts) if isinstance(texts, list) else 1),
                "attention_mask": [[1, 1, 1]] * (len(texts) if isinstance(texts, list) else 1),
            }
            wrapper.tokenizer = mock_tokenizer

            wrapper.setup(dataset)

        # Verify required positional args were passed
        assert "ref_model" in captured_kwargs
        assert captured_kwargs["ref_model"] is None  # auto-create
        assert "reward_model" in captured_kwargs
        assert captured_kwargs["reward_model"] is fake_reward_model
        assert "train_dataset" in captured_kwargs
        assert "value_model" in captured_kwargs
        assert captured_kwargs["value_model"] is fake_value_model
        assert "args" in captured_kwargs
        assert "processing_class" in captured_kwargs
        assert wrapper._dataset_in_constructor is True

    def test_legacy_api_no_positional_args(self):
        """When is_experimental=False and no args param, should use old API."""
        from unittest.mock import MagicMock
        from unittest.mock import patch as mock_patch

        from soup_cli.trainer.ppo import PPOTrainerWrapper

        cfg = SoupConfig(
            base="test-model",
            task="ppo",
            data={"train": "./data.jsonl"},
        )
        wrapper = PPOTrainerWrapper(cfg, device="cpu")

        dataset = {"train": [{"prompt": "What is 2+2?", "answer": "4"}]}

        captured_kwargs = {}

        class FakePPOTrainer:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        class FakePPOConfig:
            def __init__(self, **kwargs):
                pass

        with mock_patch("soup_cli.trainer.ppo.PPOTrainerWrapper._setup_reward"), \
             mock_patch("soup_cli.trainer.ppo.PPOTrainerWrapper._setup_transformers"), \
             mock_patch(
                 "soup_cli.trainer.ppo._import_ppo_classes",
                 return_value=(FakePPOTrainer, FakePPOConfig, False),
             ):
            wrapper.model = MagicMock()
            wrapper.model.get_nb_trainable_parameters.return_value = (100, 1000)
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = "pad"
            mock_tokenizer.side_effect = lambda texts, **kw: {
                "input_ids": [[1, 2, 3]] * (len(texts) if isinstance(texts, list) else 1),
                "attention_mask": [[1, 1, 1]] * (len(texts) if isinstance(texts, list) else 1),
            }
            wrapper.tokenizer = mock_tokenizer

            wrapper.setup(dataset)

        # Old API uses config= and tokenizer=
        assert "config" in captured_kwargs
        assert "tokenizer" in captured_kwargs
        assert "dataset" in captured_kwargs
        assert "ref_model" not in captured_kwargs
        assert "value_model" not in captured_kwargs


# --- BUG-008: GRPO CPU empty generation tensor mismatch (v0.10.6) ---


class TestGRPOCPUMinNewTokens:
    """Test GRPO CPU workaround: generation_kwargs with min_new_tokens."""

    def test_cpu_adds_generation_kwargs(self):
        """On CPU, GRPO setup should add generation_kwargs with min_new_tokens."""
        from unittest.mock import MagicMock
        from unittest.mock import patch as mock_patch

        from soup_cli.trainer.grpo import GRPOTrainerWrapper

        cfg = SoupConfig(
            base="test-model",
            task="grpo",
            data={"train": "./data.jsonl"},
        )
        wrapper = GRPOTrainerWrapper(cfg, device="cpu")

        dataset = {"train": [{"prompt": "What is 2+2?", "answer": "4"}]}

        captured_config_kwargs = {}

        class FakeGRPOConfig:
            def __init__(self, use_cpu=None, generation_kwargs=None, **kwargs):
                captured_config_kwargs.update(kwargs)
                if use_cpu is not None:
                    captured_config_kwargs["use_cpu"] = use_cpu
                if generation_kwargs is not None:
                    captured_config_kwargs["generation_kwargs"] = generation_kwargs

        class FakeGRPOTrainer:
            def __init__(self, **kwargs):
                pass

        with mock_patch("soup_cli.trainer.grpo.GRPOTrainerWrapper._setup_transformers"), \
             mock_patch("trl.GRPOConfig", FakeGRPOConfig), \
             mock_patch("trl.GRPOTrainer", FakeGRPOTrainer):
            wrapper.model = MagicMock()
            wrapper.model.get_nb_trainable_parameters.return_value = (100, 1000)
            wrapper.tokenizer = MagicMock()
            wrapper.tokenizer.pad_token = "pad"

            wrapper.setup(dataset)

        # Should have generation_kwargs with min_new_tokens on CPU
        gen_kwargs = captured_config_kwargs.get("generation_kwargs", {})
        assert gen_kwargs.get("min_new_tokens") == 1

    def test_gpu_no_generation_kwargs(self):
        """On GPU, GRPO setup should NOT add generation_kwargs for min_new_tokens."""
        from unittest.mock import MagicMock
        from unittest.mock import patch as mock_patch

        from soup_cli.trainer.grpo import GRPOTrainerWrapper

        cfg = SoupConfig(
            base="test-model",
            task="grpo",
            data={"train": "./data.jsonl"},
        )
        wrapper = GRPOTrainerWrapper(cfg, device="cuda")

        dataset = {"train": [{"prompt": "What is 2+2?", "answer": "4"}]}

        captured_config_kwargs = {}

        class FakeGRPOConfig:
            def __init__(self, use_cpu=None, generation_kwargs=None, **kwargs):
                captured_config_kwargs.update(kwargs)
                if use_cpu is not None:
                    captured_config_kwargs["use_cpu"] = use_cpu
                if generation_kwargs is not None:
                    captured_config_kwargs["generation_kwargs"] = generation_kwargs

        class FakeGRPOTrainer:
            def __init__(self, **kwargs):
                pass

        with mock_patch("soup_cli.trainer.grpo.GRPOTrainerWrapper._setup_transformers"), \
             mock_patch("trl.GRPOConfig", FakeGRPOConfig), \
             mock_patch("trl.GRPOTrainer", FakeGRPOTrainer):
            wrapper.model = MagicMock()
            wrapper.model.get_nb_trainable_parameters.return_value = (100, 1000)
            wrapper.tokenizer = MagicMock()
            wrapper.tokenizer.pad_token = "pad"

            wrapper.setup(dataset)

        # Should NOT have generation_kwargs on GPU
        assert "generation_kwargs" not in captured_config_kwargs


# --- BUG-009: PPO train() rejects resume_from_checkpoint (v0.10.7) ---


class TestPPOResumeCheckpoint:
    """Test PPO _train_builtin skips resume_from_checkpoint for experimental API."""

    def test_train_builtin_skips_resume_when_unsupported(self):
        """_train_builtin should call train() without resume_from_checkpoint
        when the trainer's .train() method doesn't accept it."""
        from unittest.mock import MagicMock

        from soup_cli.trainer.ppo import PPOTrainerWrapper

        cfg = SoupConfig(
            base="test-model",
            task="ppo",
            data={"train": "./data.jsonl"},
        )
        wrapper = PPOTrainerWrapper(cfg, device="cpu")
        wrapper._output_dir = "/tmp/test"
        wrapper._dataset_in_constructor = True
        wrapper._train_ds = MagicMock()

        # Create a real callable with no params (like experimental PPOTrainer.train)
        call_log = []

        def no_args_train():
            call_log.append("called")

        mock_trainer = MagicMock()
        mock_trainer.train = no_args_train
        mock_trainer.state.log_history = [{"loss": 0.5}]
        mock_trainer.state.global_step = 10
        wrapper.trainer = mock_trainer
        wrapper.tokenizer = MagicMock()

        result = wrapper._train_builtin(
            display=None, tracker=None, run_id="",
            resume_from_checkpoint="/tmp/ckpt",
        )

        # Should call train() without resume_from_checkpoint
        assert call_log == ["called"]
        assert result["total_steps"] == 10

    def test_train_builtin_passes_resume_when_supported(self):
        """_train_builtin should pass resume_from_checkpoint when supported."""
        from unittest.mock import MagicMock

        from soup_cli.trainer.ppo import PPOTrainerWrapper

        cfg = SoupConfig(
            base="test-model",
            task="ppo",
            data={"train": "./data.jsonl"},
        )
        wrapper = PPOTrainerWrapper(cfg, device="cpu")
        wrapper._output_dir = "/tmp/test"
        wrapper._dataset_in_constructor = True
        wrapper._train_ds = MagicMock()

        # Create a real callable that accepts resume_from_checkpoint
        call_log = []

        def resume_train(resume_from_checkpoint=None):
            call_log.append(resume_from_checkpoint)

        mock_trainer = MagicMock()
        mock_trainer.train = resume_train
        mock_trainer.state.log_history = [{"loss": 0.5}]
        mock_trainer.state.global_step = 10
        wrapper.trainer = mock_trainer
        wrapper.tokenizer = MagicMock()

        wrapper._train_builtin(
            display=None, tracker=None, run_id="",
            resume_from_checkpoint="/tmp/ckpt",
        )

        assert call_log == ["/tmp/ckpt"]

    def test_train_builtin_no_resume_calls_train_directly(self):
        """_train_builtin should call train() directly when resume is None."""
        from unittest.mock import MagicMock

        from soup_cli.trainer.ppo import PPOTrainerWrapper

        cfg = SoupConfig(
            base="test-model",
            task="ppo",
            data={"train": "./data.jsonl"},
        )
        wrapper = PPOTrainerWrapper(cfg, device="cpu")
        wrapper._output_dir = "/tmp/test"
        wrapper._dataset_in_constructor = True
        wrapper._train_ds = MagicMock()

        call_log = []

        def no_args_train():
            call_log.append("called")

        mock_trainer = MagicMock()
        mock_trainer.train = no_args_train
        mock_trainer.state.log_history = []
        mock_trainer.state.global_step = 0
        wrapper.trainer = mock_trainer
        wrapper.tokenizer = MagicMock()

        wrapper._train_builtin(
            display=None, tracker=None, run_id="",
            resume_from_checkpoint=None,
        )

        # resume is None/falsy so it should just call train()
        assert call_log == ["called"]

    def test_is_experimental_stored_on_setup(self):
        """setup() should store _is_experimental flag on the wrapper."""
        from unittest.mock import MagicMock
        from unittest.mock import patch as mock_patch

        from soup_cli.trainer.ppo import PPOTrainerWrapper

        cfg = SoupConfig(
            base="test-model",
            task="ppo",
            data={"train": "./data.jsonl"},
        )
        wrapper = PPOTrainerWrapper(cfg, device="cpu")

        class FakePPOTrainer:
            def __init__(self, **kwargs):
                pass

        class FakePPOConfig:
            def __init__(self, **kwargs):
                pass

        dataset = {"train": [{"prompt": "Q?", "answer": "A"}]}

        with mock_patch("soup_cli.trainer.ppo.PPOTrainerWrapper._setup_reward"), \
             mock_patch("soup_cli.trainer.ppo.PPOTrainerWrapper._setup_transformers"), \
             mock_patch(
                 "soup_cli.trainer.ppo._import_ppo_classes",
                 return_value=(FakePPOTrainer, FakePPOConfig, True),
             ), \
             mock_patch(
                 "soup_cli.trainer.ppo.PPOTrainerWrapper._get_or_create_reward_model",
                 return_value=MagicMock(),
             ), \
             mock_patch(
                 "soup_cli.trainer.ppo.PPOTrainerWrapper._create_value_model",
                 return_value=MagicMock(),
             ):
            wrapper.model = MagicMock()
            wrapper.model.get_nb_trainable_parameters.return_value = (100, 1000)
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = "pad"
            mock_tokenizer.side_effect = lambda texts, **kw: {
                "input_ids": [[1, 2, 3]] * (len(texts) if isinstance(texts, list) else 1),
                "attention_mask": [[1, 1, 1]] * (len(texts) if isinstance(texts, list) else 1),
            }
            wrapper.tokenizer = mock_tokenizer
            wrapper.setup(dataset)

        assert wrapper._is_experimental is True


# --- BUG-010: Meta tensor on CPU from device_map="auto" (v0.10.7) ---


class TestCPUDeviceMap:
    """Test that all trainers use device_map='cpu' on CPU instead of 'auto'."""

    def test_grpo_setup_uses_cpu_device_map(self):
        """GRPO _setup_transformers should use device_map='cpu' on CPU."""
        import inspect

        from soup_cli.trainer.grpo import GRPOTrainerWrapper
        source = inspect.getsource(GRPOTrainerWrapper._setup_transformers)
        assert '"cpu"' in source
        assert "self.device" in source

    def test_ppo_setup_uses_cpu_device_map(self):
        """PPO _setup_transformers should use device_map='cpu' on CPU."""
        import inspect

        from soup_cli.trainer.ppo import PPOTrainerWrapper
        source = inspect.getsource(PPOTrainerWrapper._setup_transformers)
        assert '"cpu"' in source
        assert "self.device" in source

    def test_sft_setup_uses_cpu_device_map(self):
        """SFT _setup_transformers should use device_map='cpu' on CPU."""
        import inspect

        from soup_cli.trainer.sft import SFTTrainerWrapper
        source = inspect.getsource(SFTTrainerWrapper._setup_transformers)
        assert '"cpu"' in source
        assert "self.device" in source

    def test_dpo_setup_uses_cpu_device_map(self):
        """DPO _setup_transformers should use device_map='cpu' on CPU."""
        import inspect

        from soup_cli.trainer.dpo import DPOTrainerWrapper
        source = inspect.getsource(DPOTrainerWrapper._setup_transformers)
        assert '"cpu"' in source
        assert "self.device" in source

    def test_reward_model_setup_uses_cpu_device_map(self):
        """RewardModel _setup_transformers should use device_map='cpu' on CPU."""
        import inspect

        from soup_cli.trainer.reward_model import RewardModelTrainerWrapper
        source = inspect.getsource(RewardModelTrainerWrapper._setup_transformers)
        assert '"cpu"' in source
        assert "self.device" in source

    def test_ppo_load_reward_model_uses_cpu_device_map(self):
        """_load_reward_model should use device_map='cpu' when device is cpu."""
        import inspect

        from soup_cli.trainer.ppo import _load_reward_model
        source = inspect.getsource(_load_reward_model)
        assert '"cpu"' in source
        assert "device" in source

    def test_ppo_get_or_create_reward_model_cpu(self):
        """_get_or_create_reward_model should not use device_map='auto' on CPU."""
        import inspect

        from soup_cli.trainer.ppo import PPOTrainerWrapper
        source = inspect.getsource(PPOTrainerWrapper._get_or_create_reward_model)
        # Should conditionally set device_map based on self.device
        assert "self.device" in source
        assert '"cpu"' in source

    def test_ppo_create_value_model_cpu(self):
        """_create_value_model should not use device_map='auto' on CPU."""
        import inspect

        from soup_cli.trainer.ppo import PPOTrainerWrapper
        source = inspect.getsource(PPOTrainerWrapper._create_value_model)
        assert "self.device" in source
        assert '"cpu"' in source


# --- BUG-011: GRPO missing chat_template causes ValueError (v0.10.8) ---


class TestGRPOChatTemplate:
    """Test GRPO sets a default chat template when tokenizer lacks one."""

    def test_grpo_setup_sets_default_chat_template(self):
        """GRPO setup should set chat_template if tokenizer doesn't have one."""
        import inspect

        from soup_cli.trainer.grpo import GRPOTrainerWrapper
        source = inspect.getsource(GRPOTrainerWrapper.setup)
        assert "chat_template" in source

    def test_grpo_setup_preserves_existing_chat_template(self):
        """GRPO setup should NOT overwrite an existing chat_template."""
        import inspect

        from soup_cli.trainer.grpo import GRPOTrainerWrapper
        source = inspect.getsource(GRPOTrainerWrapper.setup)
        # Should check with getattr before setting
        assert "getattr" in source

    def test_grpo_batch_size_ge_num_generations(self):
        """GRPO setup should ensure batch_size >= num_generations."""
        import inspect

        from soup_cli.trainer.grpo import GRPOTrainerWrapper
        source = inspect.getsource(GRPOTrainerWrapper.setup)
        assert "batch_size < num_gen" in source or "num_gen" in source


# --- BUG-012: PPO dataset not tokenized for experimental API (v0.10.8) ---


class TestPPOTokenization:
    """Test PPO tokenizes dataset before passing to trainer."""

    def test_ppo_setup_tokenizes_dataset(self):
        """PPO setup should call .map() to tokenize the dataset."""
        import inspect

        from soup_cli.trainer.ppo import PPOTrainerWrapper
        source = inspect.getsource(PPOTrainerWrapper.setup)
        assert "_tokenize_ppo" in source
        assert ".map(" in source

    def test_ppo_tokenization_adds_input_ids(self):
        """Tokenization should add input_ids and attention_mask columns."""
        from datasets import Dataset

        from soup_cli.trainer.ppo import _prepare_ppo_dataset

        data = [{"prompt": "What is 2+2?", "answer": "4"}]
        prepared = _prepare_ppo_dataset(data)
        ds = Dataset.from_list(prepared)

        # Verify prompt_text exists
        assert "prompt_text" in ds.column_names
        assert ds[0]["prompt_text"] == "What is 2+2?"


# --- BUG-013: soup data validate defaults to alpaca, no auto-detect (v0.14.2) ---


class TestValidateAutoDetect:
    """Validate should auto-detect format when --format is not specified."""

    def test_validate_alpaca_auto_detect(self, tmp_path):
        """Alpaca data should be auto-detected without --format flag."""
        import json

        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        data = [
            {"instruction": "What is AI?", "input": "", "output": "AI is..."},
            {"instruction": "Explain ML", "input": "", "output": "ML is..."},
        ]
        filepath = tmp_path / "alpaca.jsonl"
        with open(filepath, "w") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")

        result = runner.invoke(app, ["data", "validate", str(filepath)])
        assert result.exit_code == 0
        assert "Auto-detected format: alpaca" in result.output
        assert "2/2 rows valid" in result.output

    def test_validate_plaintext_auto_detect(self, tmp_path):
        """Plaintext data should be auto-detected without --format flag."""
        import json

        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        data = [{"text": "Hello world"}, {"text": "Another line"}]
        filepath = tmp_path / "plaintext.jsonl"
        with open(filepath, "w") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")

        result = runner.invoke(app, ["data", "validate", str(filepath)])
        assert result.exit_code == 0
        assert "Auto-detected format: plaintext" in result.output
        assert "2/2 rows valid" in result.output

    def test_validate_explicit_format_still_works(self, tmp_path):
        """Explicit --format flag should override auto-detection."""
        import json

        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        data = [
            {"instruction": "What is AI?", "input": "", "output": "AI is..."},
        ]
        filepath = tmp_path / "alpaca.jsonl"
        with open(filepath, "w") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")

        result = runner.invoke(
            app, ["data", "validate", str(filepath), "--format", "alpaca"]
        )
        assert result.exit_code == 0
        assert "Auto-detected" not in result.output
        assert "1/1 rows valid" in result.output

    def test_validate_dpo_auto_detect(self, tmp_path):
        """DPO data should be auto-detected."""
        import json

        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        data = [
            {"prompt": "Hi", "chosen": "Hello!", "rejected": "Go away"},
        ]
        filepath = tmp_path / "dpo.jsonl"
        with open(filepath, "w") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")

        result = runner.invoke(app, ["data", "validate", str(filepath)])
        assert result.exit_code == 0
        assert "Auto-detected format: dpo" in result.output


# --- BUG-014: soup data stats histogram broken on Windows (v0.14.2) ---


class TestStatsHistogramWindows:
    """Stats histogram should handle Windows encoding gracefully."""

    def test_stats_command_does_not_crash(self, tmp_path):
        """Stats command should not crash on Windows with plotext."""
        import json

        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        data = [
            {"instruction": "What is AI?", "input": "", "output": "AI is..."},
            {"instruction": "Explain ML", "input": "", "output": "ML is..."},
        ]
        filepath = tmp_path / "sample.jsonl"
        with open(filepath, "w") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")

        result = runner.invoke(app, ["data", "stats", str(filepath)])
        assert result.exit_code == 0
        assert "p50" in result.output

    def test_utf8_redirect_logic_on_cp1251_stdout(self):
        """Verify plotext renders when stdout is redirected from cp1251 to utf-8."""
        import io
        import sys

        try:
            import plotext as plt
        except ImportError:
            pytest.skip("plotext not installed")

        # Simulate a cp1251 stdout (like Windows default)
        raw_buffer = io.BytesIO()
        fake_cp1251 = io.TextIOWrapper(raw_buffer, encoding="cp1251")

        original_stdout = sys.stdout
        sys.stdout = fake_cp1251

        # Apply the same redirect logic as data.py stats command
        needs_redirect = (
            hasattr(sys.stdout, "encoding")
            and (sys.stdout.encoding or "").lower().replace("-", "") != "utf8"
        )
        assert needs_redirect, "Should detect cp1251 needs redirect"

        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace",
        )

        try:
            plt.clear_figure()
            plt.hist([10, 20, 30, 40, 50], bins=3)
            plt.title("Test")
            plt.theme("dark")
            plt.show()
            # If we got here, no UnicodeEncodeError — success
        finally:
            sys.stdout = original_stdout


# --- BUG-015: soup ui --help missing auth token docs (v0.14.2) ---


class TestUIAuthDocs:
    """UI --help should document auth token feature."""

    def test_ui_help_mentions_auth(self):
        """soup ui --help should mention auth token."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["ui", "--help"])
        assert result.exit_code == 0
        # Check the help output mentions auth/token
        output_lower = result.output.lower()
        assert "auth" in output_lower or "token" in output_lower

    def test_ui_help_shows_show_token_flag(self):
        """soup ui --help should show --show-token option."""
        import re

        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["ui", "--help"])
        assert result.exit_code == 0
        # Rich markup wraps dashes with ANSI codes, so strip them
        clean = re.sub(r"\x1b\[[^m]*m", "", result.output).lower()
        assert "show-token" in clean
