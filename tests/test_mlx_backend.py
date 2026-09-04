"""Tests for Apple Silicon MLX backend — Part E of v0.25.0.

These tests mock MLX entirely so they run on CI (Linux / Windows / macOS).
"""


import pytest

# ---------------------------------------------------------------------------
# MLX detection
# ---------------------------------------------------------------------------

class TestMLXDetection:
    def test_detect_mlx_not_installed(self, monkeypatch):
        """detect_mlx returns False if mlx import fails."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("mlx"):
                raise ImportError("no mlx")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        from soup_cli.utils import mlx as mlx_utils

        # Force re-check via direct call
        assert mlx_utils.detect_mlx() is False

    def test_detect_mlx_installed_mock(self, monkeypatch):
        """detect_mlx returns True when mlx modules are importable (mocked)."""
        import sys
        import types

        fake_mlx = types.ModuleType("mlx")
        fake_mlx.__version__ = "0.20.0"
        fake_core = types.ModuleType("mlx.core")
        fake_core.metal = types.SimpleNamespace(is_available=lambda: True)
        fake_mlx.core = fake_core
        monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
        monkeypatch.setitem(sys.modules, "mlx.core", fake_core)

        from soup_cli.utils import mlx as mlx_utils

        assert mlx_utils.detect_mlx() is True

    def test_get_mlx_info_not_installed(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("mlx"):
                raise ImportError("no mlx")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        from soup_cli.utils import mlx as mlx_utils

        info = mlx_utils.get_mlx_info()
        assert info["available"] is False

    def test_get_mlx_version_reads_core_version(self, monkeypatch):
        """get_mlx_version reads mlx.core.__version__, not mlx.__version__ (#659)."""
        import sys
        import types

        fake_mlx = types.ModuleType("mlx")  # no __version__ on the top-level pkg
        fake_core = types.ModuleType("mlx.core")
        fake_core.__version__ = "0.32.2"
        fake_mlx.core = fake_core
        monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
        monkeypatch.setitem(sys.modules, "mlx.core", fake_core)

        from soup_cli.utils import mlx as mlx_utils

        assert mlx_utils.get_mlx_version() == "0.32.2"

    def test_get_mlx_version_metadata_fallback(self, monkeypatch):
        """get_mlx_version falls back to importlib.metadata when core lacks it."""
        import sys
        import types

        fake_mlx = types.ModuleType("mlx")
        fake_core = types.ModuleType("mlx.core")
        fake_core.metal = types.SimpleNamespace(is_available=lambda: True)
        fake_mlx.core = fake_core
        monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
        monkeypatch.setitem(sys.modules, "mlx.core", fake_core)

        from soup_cli.utils import mlx as mlx_utils

        monkeypatch.setattr(
            "importlib.metadata.version", lambda name: "0.32.2" if name == "mlx" else None
        )
        assert mlx_utils.get_mlx_version() == "0.32.2"

    def test_estimate_mlx_batch_size_small_model(self):
        from soup_cli.utils.mlx import estimate_mlx_batch_size

        # 7B model on 16GB unified memory
        batch = estimate_mlx_batch_size(
            model_params_b=7.0,
            unified_memory_bytes=16 * 1024**3,
            max_length=2048,
            quantization="4bit",
        )
        assert batch >= 1

    def test_estimate_mlx_batch_size_large_model_tiny_mem(self):
        from soup_cli.utils.mlx import estimate_mlx_batch_size

        # 70B on 16GB is not going to fit — should return 1 minimum
        batch = estimate_mlx_batch_size(
            model_params_b=70.0,
            unified_memory_bytes=16 * 1024**3,
            max_length=2048,
            quantization="4bit",
        )
        assert batch >= 1


# ---------------------------------------------------------------------------
# Backend enum
# ---------------------------------------------------------------------------

class TestMLXBackendConfig:
    def test_backend_mlx_accepted(self):
        from soup_cli.config.loader import load_config_from_string

        yaml_str = """
base: mlx-community/Llama-3.1-8B-Instruct-4bit
task: sft
backend: mlx
data:
  train: ./data/train.jsonl
  format: chatml
training:
  epochs: 1
  lr: 1e-4
output: ./output
"""
        cfg = load_config_from_string(yaml_str)
        assert cfg.backend == "mlx"


# ---------------------------------------------------------------------------
# MLX SFT trainer wrapper (mocked)
# ---------------------------------------------------------------------------

class TestMLXSFTTrainer:
    def test_trainer_import(self):
        """Import the MLX SFT trainer."""
        from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

        assert MLXSFTTrainerWrapper is not None

    def test_trainer_setup_mocked(self, tmp_path):
        from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig
        from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

        cfg = SoupConfig(
            base="mlx-community/Llama-3.1-8B-Instruct-4bit",
            task="sft",
            backend="mlx",
            data=DataConfig(train="./data/train.jsonl", format="chatml"),
            training=TrainingConfig(epochs=1),
            output=str(tmp_path),
        )
        wrapper = MLXSFTTrainerWrapper(cfg)
        assert wrapper.config is cfg
        assert wrapper.model is None

    def test_trainer_raises_when_mlx_missing(self, tmp_path, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("mlx"):
                raise ImportError("no mlx")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig
        from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

        cfg = SoupConfig(
            base="mlx-community/Llama-3.1-8B-Instruct-4bit",
            task="sft",
            backend="mlx",
            data=DataConfig(train="./data/train.jsonl", format="chatml"),
            training=TrainingConfig(),
            output=str(tmp_path),
        )
        wrapper = MLXSFTTrainerWrapper(cfg)
        with pytest.raises((ImportError, RuntimeError)):
            wrapper.setup({"train": [], "val": []})


# ---------------------------------------------------------------------------
# MLX DPO + GRPO trainers — smoke import
# ---------------------------------------------------------------------------

class TestMLXOtherTrainers:
    def test_mlx_dpo_import(self):
        from soup_cli.trainer.mlx_dpo import MLXDPOTrainerWrapper

        assert MLXDPOTrainerWrapper is not None

    def test_mlx_grpo_import(self):
        from soup_cli.trainer.mlx_grpo import MLXGRPOTrainerWrapper

        assert MLXGRPOTrainerWrapper is not None


# ---------------------------------------------------------------------------
# train command routing
# ---------------------------------------------------------------------------

class TestMLXRouting:
    def test_mlx_routing_map(self):
        """Routing dict should map backend=mlx tasks to MLX trainers."""
        from soup_cli.trainer import mlx_routing

        assert mlx_routing.MLX_TRAINER_REGISTRY["sft"].__name__ == "MLXSFTTrainerWrapper"
        assert mlx_routing.MLX_TRAINER_REGISTRY["dpo"].__name__ == "MLXDPOTrainerWrapper"
        assert mlx_routing.MLX_TRAINER_REGISTRY["grpo"].__name__ == "MLXGRPOTrainerWrapper"

    def test_mlx_unsupported_task_rejected(self):
        from soup_cli.trainer import mlx_routing

        assert "ppo" not in mlx_routing.MLX_TRAINER_REGISTRY
        assert "pretrain" not in mlx_routing.MLX_TRAINER_REGISTRY
        assert "embedding" not in mlx_routing.MLX_TRAINER_REGISTRY


# ---------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------

class TestMLXRecipes:
    def test_llama3_1_8b_sft_mlx(self):
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.recipes.catalog import get_recipe

        recipe = get_recipe("llama3.1-8b-sft-mlx")
        assert recipe is not None
        cfg = load_config_from_string(recipe.yaml_str)
        assert cfg.backend == "mlx"
        assert cfg.task == "sft"

    def test_qwen3_8b_sft_mlx(self):
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.recipes.catalog import get_recipe

        recipe = get_recipe("qwen3-8b-sft-mlx")
        assert recipe is not None
        cfg = load_config_from_string(recipe.yaml_str)
        assert cfg.backend == "mlx"

    def test_gemma3_9b_sft_mlx(self):
        from soup_cli.recipes.catalog import get_recipe

        recipe = get_recipe("gemma3-9b-sft-mlx")
        assert recipe is not None

    def test_mlx_dpo_config_rejected_at_load(self):
        """backend=mlx + task=dpo is rejected by the SoupConfig validator."""
        from soup_cli.config.loader import load_config_from_string

        yaml_str = """
base: mlx-community/Llama-3.1-8B-Instruct-4bit
task: dpo
backend: mlx
data:
  train: ./x.jsonl
  format: dpo
training:
  epochs: 1
  lr: 1e-6
output: ./output
"""
        with pytest.raises(ValueError, match="MLX backend only ships SFT"):
            load_config_from_string(yaml_str)

    def test_mlx_grpo_config_rejected_at_load(self):
        from soup_cli.config.loader import load_config_from_string

        yaml_str = """
base: mlx-community/Llama-3.1-8B-Instruct-4bit
task: grpo
backend: mlx
data:
  train: ./x.jsonl
  format: chatml
training:
  epochs: 1
  lr: 1e-6
output: ./output
"""
        with pytest.raises(ValueError, match="MLX backend only ships SFT"):
            load_config_from_string(yaml_str)


# ---------------------------------------------------------------------------
# doctor command reports MLX
# ---------------------------------------------------------------------------


class TestMLXDoctor:
    def test_doctor_has_mlx_info(self):
        """`soup doctor` helpers surface MLX info (no crash on non-Apple)."""
        from soup_cli.commands.doctor import _get_mlx_info

        info = _get_mlx_info()
        assert isinstance(info, dict)
        assert "available" in info

    def test_doctor_command_renders_mlx_panel(self, monkeypatch):
        """`soup doctor` renders the MLX panel when MLX is available."""
        import sys
        import types
        from importlib.machinery import ModuleSpec

        from typer.testing import CliRunner

        from soup_cli.cli import app

        fake_mlx = types.ModuleType("mlx")
        fake_mlx.__spec__ = ModuleSpec("mlx", loader=None)
        fake_core = types.ModuleType("mlx.core")
        fake_core.__spec__ = ModuleSpec("mlx.core", loader=None)
        fake_core.__version__ = "0.32.2"
        fake_core.metal = types.SimpleNamespace(is_available=lambda: True)
        fake_mlx.core = fake_core
        monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
        monkeypatch.setitem(sys.modules, "mlx.core", fake_core)

        result = CliRunner().invoke(app, ["doctor"])

        assert result.exit_code == 0, result.output
        assert "MLX" in result.output
        assert "0.32.2" in result.output

    def test_doctor_command_mlx_absent_control(self, monkeypatch):
        """`soup doctor` says MLX is not installed when it cannot be imported."""
        import builtins

        from typer.testing import CliRunner

        from soup_cli.cli import app

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "mlx" or name == "mlx.core":
                raise ImportError("no mlx")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        result = CliRunner().invoke(app, ["doctor"])

        assert result.exit_code == 0
        assert "MLX" in result.output
        assert "not installed" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
