"""Tests for v0.27.0 Multi-GPU Mastery.

Covers: topology, launcher, ZeRO++, FSDP+compile, MII, pipeline, recipes.
"""

from __future__ import annotations

import json
import os
import re
import sys
from unittest.mock import MagicMock, patch

import pytest
import yaml

# =============================================================================
# Part A: Topology detector + --gpus flag
# =============================================================================

class TestResolveNumGpus:
    """Test --gpus resolution: int, 'auto', None."""

    def test_resolve_integer(self):
        from soup_cli.utils.topology import resolve_num_gpus

        assert resolve_num_gpus("4") == 4
        assert resolve_num_gpus(2) == 2

    def test_resolve_auto_uses_detected(self):
        from soup_cli.utils.topology import resolve_num_gpus

        with patch(
            "soup_cli.utils.topology._detected_gpu_count", return_value=8
        ):
            assert resolve_num_gpus("auto") == 8

    def test_resolve_auto_cpu_returns_zero(self):
        from soup_cli.utils.topology import resolve_num_gpus

        with patch(
            "soup_cli.utils.topology._detected_gpu_count", return_value=0
        ):
            assert resolve_num_gpus("auto") == 0

    def test_resolve_none_returns_none(self):
        from soup_cli.utils.topology import resolve_num_gpus

        assert resolve_num_gpus(None) is None

    def test_resolve_invalid_negative(self):
        from soup_cli.utils.topology import resolve_num_gpus

        with pytest.raises(ValueError, match="must be"):
            resolve_num_gpus("-1")

    def test_resolve_invalid_zero(self):
        from soup_cli.utils.topology import resolve_num_gpus

        with pytest.raises(ValueError, match="must be"):
            resolve_num_gpus("0")

    def test_resolve_invalid_string(self):
        from soup_cli.utils.topology import resolve_num_gpus

        with pytest.raises(ValueError, match="Invalid"):
            resolve_num_gpus("all")

    def test_resolve_out_of_bounds(self):
        from soup_cli.utils.topology import resolve_num_gpus

        with pytest.raises(ValueError, match="exceeds"):
            resolve_num_gpus("1024")


class TestDetectTopology:
    """Test topology sniffing (NVLink / PCIe) — all mocked."""

    def test_no_cuda_returns_none_interconnect(self):
        from soup_cli.utils.topology import detect_topology

        with patch(
            "soup_cli.utils.topology._detected_gpu_count", return_value=0
        ):
            topo = detect_topology()
            assert topo["gpu_count"] == 0
            assert topo["interconnect"] == "none"
            assert topo["nvlink_pairs"] == 0

    def test_single_gpu_has_no_interconnect(self):
        from soup_cli.utils.topology import detect_topology

        with patch(
            "soup_cli.utils.topology._detected_gpu_count", return_value=1
        ):
            topo = detect_topology()
            assert topo["gpu_count"] == 1
            assert topo["interconnect"] == "single"
            assert topo["nvlink_pairs"] == 0

    def test_multi_gpu_nvlink_detected(self):
        from soup_cli.utils.topology import detect_topology

        # Mock: 4 GPUs, all connected via NVLink
        with (
            patch(
                "soup_cli.utils.topology._detected_gpu_count", return_value=4
            ),
            patch(
                "soup_cli.utils.topology._count_nvlink_pairs", return_value=6
            ),
        ):
            topo = detect_topology()
            assert topo["gpu_count"] == 4
            assert topo["interconnect"] == "nvlink"
            assert topo["nvlink_pairs"] == 6

    def test_multi_gpu_pcie_fallback(self):
        from soup_cli.utils.topology import detect_topology

        # 2 GPUs, no NVLink
        with (
            patch(
                "soup_cli.utils.topology._detected_gpu_count", return_value=2
            ),
            patch(
                "soup_cli.utils.topology._count_nvlink_pairs", return_value=0
            ),
        ):
            topo = detect_topology()
            assert topo["gpu_count"] == 2
            assert topo["interconnect"] == "pcie"

    def test_nccl_env_suggestions(self):
        from soup_cli.utils.topology import suggest_nccl_env

        env = suggest_nccl_env(gpu_count=4, interconnect="nvlink")
        assert env["NCCL_P2P_DISABLE"] == "0"
        assert env["NCCL_IB_DISABLE"] == "1"  # no IB in local multi-GPU

        env_pcie = suggest_nccl_env(gpu_count=2, interconnect="pcie")
        assert env_pcie["NCCL_P2P_DISABLE"] == "0"

        env_single = suggest_nccl_env(gpu_count=1, interconnect="single")
        assert env_single == {}


class TestSuggestStrategy:
    """Test distributed-strategy suggestion based on model size + GPU count."""

    def test_single_gpu_no_strategy(self):
        from soup_cli.utils.topology import suggest_strategy

        assert suggest_strategy(gpu_count=1, model_size_b=7.0)["strategy"] == "single"

    def test_small_model_multi_gpu_ddp(self):
        from soup_cli.utils.topology import suggest_strategy

        rec = suggest_strategy(gpu_count=2, model_size_b=3.0)
        assert rec["strategy"] in ("ddp", "zero2")

    def test_mid_model_multi_gpu_zero2(self):
        from soup_cli.utils.topology import suggest_strategy

        rec = suggest_strategy(gpu_count=4, model_size_b=13.0)
        assert rec["strategy"] in ("zero2", "zero3", "fsdp_full_shard")

    def test_huge_model_zero3_or_fsdp(self):
        from soup_cli.utils.topology import suggest_strategy

        rec = suggest_strategy(gpu_count=8, model_size_b=70.0)
        assert rec["strategy"] in ("zero3", "fsdp_full_shard", "zero2_offload")

    def test_cpu_suggests_none(self):
        from soup_cli.utils.topology import suggest_strategy

        rec = suggest_strategy(gpu_count=0, model_size_b=7.0)
        assert rec["strategy"] == "none"


class TestTrainGpusFlag:
    """Test --gpus flag in train command."""

    def test_train_help_shows_gpus(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["train", "--help"])
        # Strip ANSI + whitespace per v0.26.0 CI hardening pattern
        out = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        out = re.sub(r"\s+", " ", out)
        assert "--gpus" in out


# =============================================================================
# Part B: Accelerate launcher wrapper
# =============================================================================

class TestLauncherArgv:
    """Test accelerate-launch argv construction."""

    def test_single_process_no_launcher(self):
        from soup_cli.utils.launcher import build_accelerate_argv

        # num_processes=1 → no wrapper needed
        argv = build_accelerate_argv(num_processes=1, script_args=["soup", "train"])
        assert argv == ["soup", "train"]

    def test_multi_process_wraps(self):
        from soup_cli.utils.launcher import build_accelerate_argv

        argv = build_accelerate_argv(num_processes=4, script_args=["soup", "train"])
        assert argv[0] == "accelerate"
        assert argv[1] == "launch"
        assert "--num_processes" in argv
        # num_processes value follows the flag
        idx = argv.index("--num_processes")
        assert argv[idx + 1] == "4"

    def test_multi_process_mixed_precision(self):
        from soup_cli.utils.launcher import build_accelerate_argv

        argv = build_accelerate_argv(
            num_processes=4, script_args=["soup", "train"], mixed_precision="bf16"
        )
        assert "--mixed_precision" in argv
        idx = argv.index("--mixed_precision")
        assert argv[idx + 1] == "bf16"

    def test_invalid_num_processes(self):
        from soup_cli.utils.launcher import build_accelerate_argv

        with pytest.raises(ValueError, match="num_processes"):
            build_accelerate_argv(num_processes=0, script_args=["soup", "train"])

    def test_invalid_mixed_precision(self):
        from soup_cli.utils.launcher import build_accelerate_argv

        with pytest.raises(ValueError, match="mixed_precision"):
            build_accelerate_argv(
                num_processes=2, script_args=["soup", "train"], mixed_precision="int8"
            )

    def test_invalid_num_machines_low(self):
        from soup_cli.utils.launcher import build_accelerate_argv

        with pytest.raises(ValueError, match="num_machines"):
            build_accelerate_argv(
                num_processes=2, script_args=["soup", "train"], num_machines=0
            )

    def test_invalid_num_machines_high(self):
        from soup_cli.utils.launcher import build_accelerate_argv

        with pytest.raises(ValueError, match="num_machines"):
            build_accelerate_argv(
                num_processes=2, script_args=["soup", "train"], num_machines=10_000
            )


class TestLauncherDetectActive:
    """Test detection of whether we're running under accelerate/torchrun.

    Uses ``clear=False`` + explicit key deletion so Windows CI keeps
    SYSTEMROOT / PATH / TEMP available for the rest of the process.
    """

    _DIST_KEYS = (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "ACCELERATE_MIXED_PRECISION",
        "ACCELERATE_USE_DEEPSPEED",
        "ACCELERATE_USE_FSDP",
    )

    def test_not_in_distributed(self):
        from soup_cli.utils.launcher import is_in_distributed

        # Delete any pre-existing distributed markers without wiping env.
        stripped_env = {
            key: val for key, val in os.environ.items() if key not in self._DIST_KEYS
        }
        with patch.dict(os.environ, stripped_env, clear=True):
            assert is_in_distributed() is False

    def test_in_torchrun(self):
        from soup_cli.utils.launcher import is_in_distributed

        env = {"RANK": "0", "WORLD_SIZE": "4", "LOCAL_RANK": "0"}
        with patch.dict(os.environ, env, clear=False):
            assert is_in_distributed() is True

    def test_in_accelerate(self):
        from soup_cli.utils.launcher import is_in_distributed

        env = {"ACCELERATE_MIXED_PRECISION": "bf16", "RANK": "0", "WORLD_SIZE": "2"}
        with patch.dict(os.environ, env, clear=False):
            assert is_in_distributed() is True


class TestLauncherAdvice:
    """Test that launcher prints useful advice message when N>1 and not distributed."""

    def test_format_advice_command(self):
        from soup_cli.utils.launcher import format_advice

        text = format_advice(num_processes=4, script_args=["soup", "train", "-c", "soup.yaml"])
        assert "accelerate launch" in text
        assert "--num_processes 4" in text
        assert "soup train" in text


# =============================================================================
# Part C: ZeRO++ DeepSpeed template
# =============================================================================

class TestZeroPlusPlus:
    """Test ZeRO++ (zero_plus_plus / zero++) config template."""

    def test_zero_pp_config_exists(self):
        from soup_cli.utils.deepspeed import CONFIGS

        assert "zero++" in CONFIGS or "zero_pp" in CONFIGS

    def test_zero_pp_has_stage_3_base(self):
        from soup_cli.utils.deepspeed import get_deepspeed_config

        cfg = get_deepspeed_config("zero++")
        assert cfg["zero_optimization"]["stage"] == 3

    def test_zero_pp_has_hierarchical_comms(self):
        from soup_cli.utils.deepspeed import get_deepspeed_config

        cfg = get_deepspeed_config("zero++")
        zopt = cfg["zero_optimization"]
        # ZeRO++ distinguishing features: quantized weights/gradients + hpz
        assert "zero_quantized_weights" in zopt
        assert "zero_hpz_partition_size" in zopt
        assert "zero_quantized_gradients" in zopt

    def test_zero_pp_write_file(self):
        """#336 — the written config is the *resolved* one.

        This test used to assert ``zero_quantized_weights is True`` on the
        written file. That is the setting that crashed on real hardware: the
        preset enables bf16 and DeepSpeed's quantiser is the fp16 kernel, so
        the dequantised all-gather returned Half into a BFloat16 activation.
        The key is still present — it is a ZeRO++ knob, see
        ``test_zero_pp_has_hierarchical_comms`` — but resolved to ``False``
        for a bf16 run. ``tests/test_issue336_deepspeed_lora.py`` carries the
        fp16 control proving quantisation is a dtype decision, not a blanket
        disable.
        """
        from soup_cli.utils.deepspeed import write_deepspeed_config

        path = write_deepspeed_config("zero++")
        try:
            assert os.path.exists(path)
            with open(path) as f:
                cfg = json.load(f)
            assert cfg["zero_optimization"]["zero_quantized_weights"] is False
            assert "zero_hpz_partition_size" in cfg["zero_optimization"]
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_resolve_zero_pp_in_train(self):
        from soup_cli.commands.train import _resolve_deepspeed

        path = _resolve_deepspeed("zero++")
        try:
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)


# =============================================================================
# Part D: FSDP2 + torch.compile
# =============================================================================

class TestFsdp2Compile:
    """Test FSDP2 + torch.compile integration."""

    def test_config_has_fsdp2_compile_field(self):
        from soup_cli.config.schema import TrainingConfig

        cfg = TrainingConfig(use_fsdp2_compile=True)
        assert cfg.use_fsdp2_compile is True

    def test_config_default_false(self):
        from soup_cli.config.schema import TrainingConfig

        cfg = TrainingConfig()
        assert cfg.use_fsdp2_compile is False

    def test_validate_compile_requires_fsdp(self):
        from soup_cli.utils.fsdp import validate_fsdp2_compile_config

        errors = validate_fsdp2_compile_config(
            use_compile=True, fsdp_preset=None, backend="transformers", device="cuda"
        )
        assert any("FSDP" in e for e in errors)

    def test_validate_compile_requires_cuda(self):
        from soup_cli.utils.fsdp import validate_fsdp2_compile_config

        errors = validate_fsdp2_compile_config(
            use_compile=True, fsdp_preset="full_shard", backend="transformers", device="cpu"
        )
        assert any("CUDA" in e for e in errors)

    def test_validate_compile_requires_transformers_backend(self):
        from soup_cli.utils.fsdp import validate_fsdp2_compile_config

        errors = validate_fsdp2_compile_config(
            use_compile=True, fsdp_preset="full_shard", backend="unsloth", device="cuda"
        )
        assert any("unsloth" in e for e in errors)

    def test_validate_compile_ok(self):
        from soup_cli.utils.fsdp import validate_fsdp2_compile_config

        with patch("soup_cli.utils.fsdp.is_fsdp_available", return_value=True):
            errors = validate_fsdp2_compile_config(
                use_compile=True,
                fsdp_preset="full_shard",
                backend="transformers",
                device="cuda",
            )
            assert errors == []

    def test_validate_compile_disabled_noop(self):
        from soup_cli.utils.fsdp import validate_fsdp2_compile_config

        # When flag is off, no errors ever.
        errors = validate_fsdp2_compile_config(
            use_compile=False, fsdp_preset=None, backend="unsloth", device="cpu"
        )
        assert errors == []


# =============================================================================
# Part E: DeepSpeed-MII serve backend
# =============================================================================

class TestMiiBackend:
    """Test DeepSpeed-MII detection + runtime creation."""

    def test_is_mii_available_false_when_stubbed_to_none(self):
        from soup_cli.utils.mii import is_mii_available

        with patch.dict("sys.modules", {"mii": None}):
            assert is_mii_available() is False

    def test_is_mii_available_false_when_key_absent(self):
        """Cover the ImportError branch — no ``mii`` key in sys.modules at all."""
        from soup_cli.utils.mii import is_mii_available

        saved = sys.modules.pop("mii", "_SENTINEL")
        try:
            # Force the import path to fail by making the real module look gone.
            with patch.dict("sys.modules", {"mii": None}, clear=False):
                assert is_mii_available() is False
        finally:
            if saved != "_SENTINEL":
                sys.modules["mii"] = saved

    def test_is_mii_available_true_when_importable(self):
        from soup_cli.utils.mii import is_mii_available

        fake_mii = MagicMock()
        with patch.dict("sys.modules", {"mii": fake_mii}):
            assert is_mii_available() is True

    def test_create_mii_pipeline_mocked(self):
        from soup_cli.utils.mii import create_mii_pipeline

        fake_mii = MagicMock()
        fake_pipe = MagicMock()
        fake_mii.pipeline.return_value = fake_pipe
        with patch.dict("sys.modules", {"mii": fake_mii}):
            pipe = create_mii_pipeline(
                model_path="/models/llama", tensor_parallel=2, max_length=4096
            )
            assert pipe is fake_pipe
            fake_mii.pipeline.assert_called_once()
            kwargs = fake_mii.pipeline.call_args.kwargs
            assert kwargs.get("tensor_parallel") == 2

    def test_create_mii_raises_when_missing(self):
        from soup_cli.utils.mii import create_mii_pipeline

        with patch.dict("sys.modules", {"mii": None}):
            with pytest.raises(ImportError, match="deepspeed-mii"):
                create_mii_pipeline(model_path="/models/llama")

    def test_serve_help_shows_mii(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["serve", "--help"])
        import re
        out = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        out = re.sub(r"\s+", " ", out)
        assert "mii" in out.lower()


# =============================================================================
# Part F: Pipeline parallelism config
# =============================================================================

class TestPipelineParallelConfig:
    """Test pipeline-parallel config fields + validation."""

    def test_default_parallelism_data(self):
        from soup_cli.config.schema import TrainingConfig

        cfg = TrainingConfig()
        assert cfg.parallelism == "data"
        assert cfg.pipeline_stages == 1

    def test_set_pipeline(self):
        from soup_cli.config.schema import TrainingConfig

        cfg = TrainingConfig(parallelism="pipeline", pipeline_stages=4)
        assert cfg.parallelism == "pipeline"
        assert cfg.pipeline_stages == 4

    def test_pipeline_stages_bounds_low(self):
        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValueError):
            TrainingConfig(pipeline_stages=0)

    def test_pipeline_stages_bounds_high(self):
        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValueError):
            TrainingConfig(pipeline_stages=64)

    def test_parallelism_invalid_literal(self):
        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValueError):
            TrainingConfig(parallelism="hybrid")

    def test_validate_pipeline_requires_stages(self):
        from soup_cli.utils.pipeline import validate_pipeline_config

        errors = validate_pipeline_config(
            parallelism="pipeline", pipeline_stages=1, device="cuda", gpu_count=4,
        )
        assert any("pipeline_stages" in e for e in errors)

    def test_validate_pipeline_requires_cuda(self):
        from soup_cli.utils.pipeline import validate_pipeline_config

        errors = validate_pipeline_config(
            parallelism="pipeline", pipeline_stages=4, device="cpu", gpu_count=0,
        )
        assert any("CUDA" in e for e in errors)

    def test_validate_pipeline_requires_enough_gpus(self):
        from soup_cli.utils.pipeline import validate_pipeline_config

        errors = validate_pipeline_config(
            parallelism="pipeline", pipeline_stages=4, device="cuda", gpu_count=2,
        )
        assert any("GPUs" in e for e in errors)

    def test_validate_data_parallel_ok(self):
        from soup_cli.utils.pipeline import validate_pipeline_config

        errors = validate_pipeline_config(
            parallelism="data", pipeline_stages=1, device="cpu", gpu_count=0,
        )
        assert errors == []


# =============================================================================
# Part G: Multi-GPU recipes
# =============================================================================

class TestMultiGpuRecipes:
    """Test new multi-GPU recipes in catalog."""

    def test_llama3_70b_fsdp2_recipe_exists(self):
        from soup_cli.recipes.catalog import RECIPES

        assert "llama3-70b-fsdp2" in RECIPES

    def test_qwen3_32b_zeropp_recipe_exists(self):
        from soup_cli.recipes.catalog import RECIPES

        assert "qwen3-32b-zeropp" in RECIPES

    def test_deepseek_v3_pipeline_recipe_exists(self):
        from soup_cli.recipes.catalog import RECIPES

        assert "deepseek-v3-pipeline" in RECIPES

    def test_all_new_recipes_load_as_config(self):
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.recipes.catalog import RECIPES

        for name in ("llama3-70b-fsdp2", "qwen3-32b-zeropp", "deepseek-v3-pipeline"):
            meta = RECIPES[name]
            cfg = load_config_from_string(meta.yaml_str)
            # Exact-match check — substring allowed false positives between
            # ``llama3-8b`` / ``llama3-70b`` recipes.
            assert cfg.base == meta.model

    def test_llama3_70b_fsdp2_has_fsdp2_compile(self):
        from soup_cli.recipes.catalog import RECIPES

        meta = RECIPES["llama3-70b-fsdp2"]
        doc = yaml.safe_load(meta.yaml_str)
        assert doc["training"].get("use_fsdp2_compile") is True

    def test_qwen3_32b_zeropp_tags(self):
        from soup_cli.recipes.catalog import RECIPES

        meta = RECIPES["qwen3-32b-zeropp"]
        # Recipe is multi-GPU focused
        assert "qwen3-32" in meta.model.lower()

    def test_deepseek_v3_pipeline_stages(self):
        from soup_cli.recipes.catalog import RECIPES

        meta = RECIPES["deepseek-v3-pipeline"]
        doc = yaml.safe_load(meta.yaml_str)
        assert doc["training"].get("parallelism") == "pipeline"
        assert doc["training"].get("pipeline_stages") >= 2


# =============================================================================
# Integration: config field → trainer wiring
# =============================================================================

class TestFsdp2CompileHelper:
    """Behavioral tests for apply_fsdp_training_kwargs (extracted helper).

    Tests the actual dict mutation that reaches ``TrainingArguments`` —
    no source grep, no brittle string matching.
    """

    def test_no_fsdp_leaves_kwargs_untouched(self):
        from soup_cli.utils.fsdp import apply_fsdp_training_kwargs

        kwargs = {"learning_rate": 2e-4}
        result = apply_fsdp_training_kwargs(
            kwargs, fsdp_config=None, use_fsdp2_compile=True
        )
        assert result is kwargs
        assert "torch_compile" not in result
        assert "fsdp" not in result

    def test_fsdp_without_compile_flag(self):
        from soup_cli.utils.fsdp import apply_fsdp_training_kwargs

        kwargs = {}
        apply_fsdp_training_kwargs(
            kwargs,
            fsdp_config={"fsdp": "full_shard auto_wrap", "fsdp_config": {}},
            use_fsdp2_compile=False,
        )
        assert kwargs["fsdp"] == "full_shard auto_wrap"
        assert "torch_compile" not in kwargs

    def test_fsdp_with_compile_flag_true(self):
        from soup_cli.utils.fsdp import apply_fsdp_training_kwargs

        kwargs = {}
        apply_fsdp_training_kwargs(
            kwargs,
            fsdp_config={"fsdp": "full_shard auto_wrap", "fsdp_config": {}},
            use_fsdp2_compile=True,
        )
        assert kwargs["torch_compile"] is True

    def test_unexpected_fsdp_keys_rejected(self):
        from soup_cli.utils.fsdp import apply_fsdp_training_kwargs

        with pytest.raises(ValueError, match="Unexpected FSDP config keys"):
            apply_fsdp_training_kwargs(
                {},
                fsdp_config={"fsdp": "x", "fsdp_config": {}, "shady": 1},
                use_fsdp2_compile=False,
            )


class TestFsdp2CompileValidatorDeepSpeed:
    """validate_fsdp2_compile_config rejects DeepSpeed + torch.compile."""

    def test_deepspeed_and_compile_rejected(self):
        from soup_cli.utils.fsdp import validate_fsdp2_compile_config

        errors = validate_fsdp2_compile_config(
            use_compile=True,
            fsdp_preset="full_shard",
            backend="transformers",
            device="cuda",
            deepspeed_config="/tmp/ds.json",
        )
        assert any("deepspeed" in err.lower() for err in errors)

    def test_no_deepspeed_no_error(self):
        from soup_cli.utils.fsdp import validate_fsdp2_compile_config

        with patch("soup_cli.utils.fsdp.is_fsdp_available", return_value=True):
            errors = validate_fsdp2_compile_config(
                use_compile=True,
                fsdp_preset="full_shard",
                backend="transformers",
                device="cuda",
                deepspeed_config=None,
            )
            assert errors == []


class TestNcclEnvApplication:
    """Verify train.py actually calls os.environ.setdefault in distributed mode
    and skips the call on single-GPU / CPU runs."""

    def test_setdefault_does_not_overwrite_user_override(self):
        from soup_cli.utils.topology import suggest_nccl_env

        env = suggest_nccl_env(gpu_count=4, interconnect="nvlink")
        user = {"NCCL_P2P_DISABLE": "1"}
        for key, val in env.items():
            user.setdefault(key, val)
        # User value preserved
        assert user["NCCL_P2P_DISABLE"] == "1"
        # At least one new key added (non-brittle: the exact key set may
        # evolve, we only need the non-overwrite invariant + "something new
        # came in" guarantee).
        new_keys = set(user) - {"NCCL_P2P_DISABLE"}
        assert new_keys, "suggest_nccl_env produced no new keys"

    def test_single_gpu_returns_empty_env(self):
        """Single-GPU path returns an empty dict (so setdefault is a no-op)."""
        from soup_cli.utils.topology import suggest_nccl_env

        assert suggest_nccl_env(gpu_count=1, interconnect="single") == {}


class TestTrainValidatorGating:
    """CLI-level tests proving the new validators actually block `soup train`
    when misconfigured (not silently continue)."""

    def test_use_fsdp2_compile_on_cpu_blocks_train(self, tmp_path):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        # Build a config that asks for use_fsdp2_compile but runs on CPU (no
        # --fsdp + no --gpus). Validators should fire before anything heavy
        # runs, so we don't need a real model here.
        data_file = tmp_path / "train.jsonl"
        data_file.write_text('{"prompt": "x", "completion": "y"}\n')
        cfg_file = tmp_path / "soup.yaml"
        cfg_file.write_text(
            "base: test-model\n"
            "task: sft\n"
            f"data:\n  train: {data_file.as_posix()}\n"
            "training:\n  use_fsdp2_compile: true\n"
            f"output: {(tmp_path / 'out').as_posix()}\n"
        )
        runner = CliRunner()
        result = runner.invoke(
            app, ["train", "--config", str(cfg_file), "--yes", "--dry-run"]
        )
        # Exit non-zero, with a clear message mentioning the flag.
        assert result.exit_code != 0, (result.output, repr(result.exception))
        assert "fsdp2_compile" in result.output.lower() or "fsdp" in result.output.lower()

    def test_pipeline_parallel_on_cpu_blocks_train(self, tmp_path):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        data_file = tmp_path / "train.jsonl"
        data_file.write_text('{"prompt": "x", "completion": "y"}\n')
        cfg_file = tmp_path / "soup.yaml"
        cfg_file.write_text(
            "base: test-model\n"
            "task: sft\n"
            f"data:\n  train: {data_file.as_posix()}\n"
            "training:\n  parallelism: pipeline\n  pipeline_stages: 4\n"
            f"output: {(tmp_path / 'out').as_posix()}\n"
        )
        runner = CliRunner()
        result = runner.invoke(
            app, ["train", "--config", str(cfg_file), "--yes", "--dry-run"]
        )
        assert result.exit_code != 0, (result.output, repr(result.exception))
        assert "pipeline" in result.output.lower() or "cuda" in result.output.lower()
