"""Tests for v0.28.0 — Training Speed & Memory features.

Covers:
- Part A: Cut Cross-Entropy (CCE)
- Part B: FP8 training (quantization_aware='fp8')
- Part C: Gradient checkpointing tiers (selective/medium/full/auto)
- Part D: Kernel auto-composition (utils/kernel_picker.py)
- Part E: Cross-document attention masking for sample packing
- Part F: Activation offloading to CPU/disk
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import SoupConfig, TrainingConfig

# ─── Part A: Cut Cross-Entropy (CCE) ───────────────────────────────────────


class TestCutCEConfig:
    """TrainingConfig.use_cut_ce boolean field."""

    def test_use_cut_ce_default_false(self):
        cfg = SoupConfig(base="test/model", data={"train": "./data.jsonl"})
        assert cfg.training.use_cut_ce is False

    def test_use_cut_ce_enabled(self):
        cfg = SoupConfig(
            base="test/model",
            data={"train": "./data.jsonl"},
            training={"use_cut_ce": True},
        )
        assert cfg.training.use_cut_ce is True

    def test_use_cut_ce_type_bool_coerce(self):
        """Pydantic coerces bool-like strings; verify type is bool after."""
        cfg = TrainingConfig(use_cut_ce=True)
        assert cfg.use_cut_ce is True
        assert isinstance(cfg.use_cut_ce, bool)


class TestCutCEAvailability:
    """Cut Cross-Entropy availability + detection."""

    def test_check_cut_ce_available_returns_bool(self):
        from soup_cli.utils.cut_ce import check_cut_ce_available

        result = check_cut_ce_available()
        assert isinstance(result, bool)

    def test_check_cut_ce_not_installed(self):
        from soup_cli.utils.cut_ce import check_cut_ce_available

        # sys.modules[name]=None makes ``import name`` raise ImportError
        with patch.dict("sys.modules", {"cut_cross_entropy": None}):
            assert check_cut_ce_available() is False

    def test_get_cut_ce_version_not_installed(self):
        from soup_cli.utils.cut_ce import get_cut_ce_version

        with patch(
            "soup_cli.utils.cut_ce.check_cut_ce_available", return_value=False
        ):
            result = get_cut_ce_version()
            assert result is None


class TestCutCEApplication:
    """Applying Cut Cross-Entropy to a model."""

    def test_apply_cut_ce_not_installed(self):
        from soup_cli.utils.cut_ce import apply_cut_ce

        with patch(
            "soup_cli.utils.cut_ce.check_cut_ce_available", return_value=False
        ):
            result = apply_cut_ce("meta-llama/Llama-3.1-8B")
            assert result is False

    def test_apply_cut_ce_available_tries_patching(self):
        from soup_cli.utils.cut_ce import apply_cut_ce

        with patch(
            "soup_cli.utils.cut_ce.check_cut_ce_available", return_value=True
        ):
            # With cut_cross_entropy not actually installed, patch should return
            # False (can't import real module). Just verifying it doesn't crash.
            result = apply_cut_ce("meta-llama/Llama-3.1-8B")
            assert isinstance(result, bool)

    def test_apply_cut_ce_calls_llama_patch_for_llama_model(self):
        """Verify the llama detector routes to cce_patch('llama')."""
        from soup_cli.utils.cut_ce import apply_cut_ce

        fake_cce = MagicMock()
        fake_transformers = MagicMock(cce_patch=fake_cce)
        fake_module = MagicMock(transformers=fake_transformers)
        with patch(
            "soup_cli.utils.cut_ce.check_cut_ce_available", return_value=True
        ), patch.dict(
            "sys.modules",
            {
                "cut_cross_entropy": fake_module,
                "cut_cross_entropy.transformers": fake_transformers,
            },
        ):
            assert apply_cut_ce("meta-llama/Llama-3.1-8B") is True
            fake_cce.assert_called_once_with("llama")

    def test_apply_cut_ce_deepseek_phi_does_not_use_phi(self):
        """Regression: org-prefix like 'deepseek-ai/...' must not trigger phi."""
        from soup_cli.utils.cut_ce import apply_cut_ce

        fake_cce = MagicMock()
        fake_transformers = MagicMock(cce_patch=fake_cce)
        fake_module = MagicMock(transformers=fake_transformers)
        with patch(
            "soup_cli.utils.cut_ce.check_cut_ce_available", return_value=True
        ), patch.dict(
            "sys.modules",
            {
                "cut_cross_entropy": fake_module,
                "cut_cross_entropy.transformers": fake_transformers,
            },
        ):
            # Llama-distilled model name contains no "phi" substring
            # anymore thanks to the last-path-component detector.
            assert apply_cut_ce("deepseek-ai/deepseek-coder-7b-instruct") is False
            fake_cce.assert_not_called()


class TestCutCEValidation:
    """Cut Cross-Entropy config validation."""

    def test_validate_cut_ce_disabled_returns_empty(self):
        from soup_cli.utils.cut_ce import validate_cut_ce_config

        errors = validate_cut_ce_config(False, "transformers", "cuda")
        assert errors == []

    def test_validate_cut_ce_not_installed(self):
        from soup_cli.utils.cut_ce import validate_cut_ce_config

        with patch(
            "soup_cli.utils.cut_ce.check_cut_ce_available", return_value=False
        ):
            errors = validate_cut_ce_config(True, "transformers", "cuda")
            assert any("not installed" in err for err in errors)

    def test_validate_cut_ce_requires_cuda(self):
        from soup_cli.utils.cut_ce import validate_cut_ce_config

        with patch(
            "soup_cli.utils.cut_ce.check_cut_ce_available", return_value=True
        ):
            errors = validate_cut_ce_config(True, "transformers", "cpu")
            assert any("CUDA" in err for err in errors)

    def test_validate_cut_ce_unsloth_incompatible(self):
        from soup_cli.utils.cut_ce import validate_cut_ce_config

        with patch(
            "soup_cli.utils.cut_ce.check_cut_ce_available", return_value=True
        ):
            errors = validate_cut_ce_config(True, "unsloth", "cuda")
            assert any("unsloth" in err.lower() for err in errors)

    def test_validate_cut_ce_valid(self):
        from soup_cli.utils.cut_ce import validate_cut_ce_config

        with patch(
            "soup_cli.utils.cut_ce.check_cut_ce_available", return_value=True
        ):
            errors = validate_cut_ce_config(True, "transformers", "cuda")
            assert errors == []

    def test_validate_cut_ce_mlx_incompatible(self):
        from soup_cli.utils.cut_ce import validate_cut_ce_config

        with patch(
            "soup_cli.utils.cut_ce.check_cut_ce_available", return_value=True
        ):
            errors = validate_cut_ce_config(True, "mlx", "mps")
            assert any("mlx" in err.lower() for err in errors)


# ─── Part B: FP8 training ─────────────────────────────────────────────────


class TestFP8Config:
    """quantization_aware now accepts bool or literal 'fp8'."""

    def test_quantization_aware_default_false(self):
        cfg = SoupConfig(base="test/model", data={"train": "./data.jsonl"})
        assert cfg.training.quantization_aware is False

    def test_quantization_aware_bool_true(self):
        cfg = SoupConfig(
            base="test/model",
            data={"train": "./data.jsonl"},
            training={"quantization_aware": True},
        )
        assert cfg.training.quantization_aware is True

    def test_quantization_aware_fp8(self):
        cfg = SoupConfig(
            base="test/model",
            data={"train": "./data.jsonl"},
            training={"quantization_aware": "fp8"},
        )
        assert cfg.training.quantization_aware == "fp8"

    def test_quantization_aware_invalid_string_rejected(self):
        """Only 'fp8' literal is accepted, other strings rejected."""
        with pytest.raises(ValidationError) as exc:
            SoupConfig(
                base="test/model",
                data={"train": "./data.jsonl"},
                training={"quantization_aware": "fp16"},
            )
        assert "fp8" in str(exc.value) or "quantization_aware" in str(exc.value)

    def test_quantization_aware_bool_still_works(self):
        cfg = SoupConfig(
            base="test/model",
            data={"train": "./data.jsonl"},
            training={"quantization_aware": False},
        )
        assert cfg.training.quantization_aware is False


class TestFP8Availability:
    """FP8 training dependency checks."""

    def test_is_fp8_available_returns_bool(self):
        from soup_cli.utils.fp8 import is_fp8_available

        result = is_fp8_available()
        assert isinstance(result, bool)

    def test_fp8_requires_hopper_gpu_info(self):
        """is_fp8_gpu_supported should check GPU compute capability."""
        from soup_cli.utils.fp8 import is_fp8_gpu_supported

        # Shouldn't crash even without CUDA
        result = is_fp8_gpu_supported()
        assert isinstance(result, bool)

    def test_is_fp8_available_false_when_deps_missing(self):
        """Explicit false branch — both torchao.float8 and transformer_engine absent."""
        with patch.dict(
            "sys.modules",
            {"torchao.float8": None, "transformer_engine": None},
        ):
            from soup_cli.utils.fp8 import is_fp8_available

            assert is_fp8_available() is False

    def test_is_fp8_gpu_supported_pre_hopper_false(self):
        """Pre-Hopper (SM 8.x, e.g. A100) is not supported."""
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.get_device_capability.return_value = (8, 0)
        with patch.dict("sys.modules", {"torch": fake_torch}):
            from soup_cli.utils.fp8 import is_fp8_gpu_supported

            assert is_fp8_gpu_supported() is False

    def test_is_fp8_gpu_supported_hopper_true(self):
        """Hopper (SM 9.x, H100) is supported."""
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.get_device_capability.return_value = (9, 0)
        with patch.dict("sys.modules", {"torch": fake_torch}):
            from soup_cli.utils.fp8 import is_fp8_gpu_supported

            assert is_fp8_gpu_supported() is True


class TestFP8Validation:
    """FP8 training config validation."""

    def test_validate_fp8_not_requested_returns_empty(self):
        from soup_cli.utils.fp8 import validate_fp8_config

        errors = validate_fp8_config(False, "transformers", "cuda")
        assert errors == []

    def test_validate_fp8_bool_returns_empty(self):
        """Bool True means int8 QAT (existing path), not FP8."""
        from soup_cli.utils.fp8 import validate_fp8_config

        errors = validate_fp8_config(True, "transformers", "cuda")
        # Bool True is int8 QAT, handled by qat.py, not fp8
        assert errors == []

    def test_validate_fp8_cpu_rejected(self):
        from soup_cli.utils.fp8 import validate_fp8_config

        errors = validate_fp8_config("fp8", "transformers", "cpu")
        assert any("CUDA" in err for err in errors)

    def test_validate_fp8_unsloth_rejected(self):
        from soup_cli.utils.fp8 import validate_fp8_config

        errors = validate_fp8_config("fp8", "unsloth", "cuda")
        assert any("unsloth" in err.lower() for err in errors)

    def test_validate_fp8_mlx_rejected(self):
        from soup_cli.utils.fp8 import validate_fp8_config

        errors = validate_fp8_config("fp8", "mlx", "mps")
        assert any("mlx" in err.lower() or "CUDA" in err for err in errors)


# ─── Part C: Gradient checkpointing tiers ─────────────────────────────────


class TestGradientCheckpointingTiers:
    """gradient_checkpointing accepts bool or tier literal."""

    def test_default_false(self):
        cfg = SoupConfig(base="test/model", data={"train": "./data.jsonl"})
        assert cfg.training.gradient_checkpointing is False

    def test_bool_true(self):
        cfg = SoupConfig(
            base="test/model",
            data={"train": "./data.jsonl"},
            training={"gradient_checkpointing": True},
        )
        assert cfg.training.gradient_checkpointing is True

    def test_tier_selective(self):
        cfg = SoupConfig(
            base="test/model",
            data={"train": "./data.jsonl"},
            training={"gradient_checkpointing": "selective"},
        )
        assert cfg.training.gradient_checkpointing == "selective"

    def test_tier_medium(self):
        cfg = SoupConfig(
            base="test/model",
            data={"train": "./data.jsonl"},
            training={"gradient_checkpointing": "medium"},
        )
        assert cfg.training.gradient_checkpointing == "medium"

    def test_tier_full(self):
        cfg = SoupConfig(
            base="test/model",
            data={"train": "./data.jsonl"},
            training={"gradient_checkpointing": "full"},
        )
        assert cfg.training.gradient_checkpointing == "full"

    def test_tier_auto(self):
        cfg = SoupConfig(
            base="test/model",
            data={"train": "./data.jsonl"},
            training={"gradient_checkpointing": "auto"},
        )
        assert cfg.training.gradient_checkpointing == "auto"

    def test_invalid_tier_rejected(self):
        with pytest.raises(ValidationError) as exc:
            SoupConfig(
                base="test/model",
                data={"train": "./data.jsonl"},
                training={"gradient_checkpointing": "partial"},
            )
        assert "gradient_checkpointing" in str(exc.value)


class TestGradientCheckpointingResolver:
    """Resolve config value + GPU info → kwargs for TrainingArguments."""

    def test_resolve_disabled_returns_empty(self):
        from soup_cli.utils.gradient_ckpt import resolve_gradient_checkpointing

        kwargs = resolve_gradient_checkpointing(False, gpu_memory_gb=80)
        assert kwargs == {}

    def test_resolve_bool_true_returns_full_ckpt(self):
        from soup_cli.utils.gradient_ckpt import resolve_gradient_checkpointing

        kwargs = resolve_gradient_checkpointing(True, gpu_memory_gb=80)
        assert kwargs["gradient_checkpointing"] is True

    def test_resolve_full_tier(self):
        from soup_cli.utils.gradient_ckpt import resolve_gradient_checkpointing

        kwargs = resolve_gradient_checkpointing("full", gpu_memory_gb=80)
        assert kwargs["gradient_checkpointing"] is True

    def test_resolve_selective_tier(self):
        from soup_cli.utils.gradient_ckpt import (
            resolve_gradient_checkpointing,
            resolve_granularity,
        )

        kwargs = resolve_gradient_checkpointing("selective", gpu_memory_gb=80)
        assert kwargs["gradient_checkpointing"] is True
        # No private markers leak into HF TrainingArguments kwargs.
        assert kwargs["gradient_checkpointing_kwargs"] == {"use_reentrant": False}
        # Granularity is exposed via a separate helper for the wrapper.
        assert resolve_granularity("selective", gpu_memory_gb=80) == "selective"

    def test_resolve_medium_tier(self):
        from soup_cli.utils.gradient_ckpt import resolve_gradient_checkpointing

        kwargs = resolve_gradient_checkpointing("medium", gpu_memory_gb=80)
        assert kwargs["gradient_checkpointing"] is True

    def test_resolve_auto_low_memory_selects_full(self):
        from soup_cli.utils.gradient_ckpt import resolve_gradient_checkpointing

        kwargs = resolve_gradient_checkpointing("auto", gpu_memory_gb=16)
        # Low VRAM → full checkpointing
        assert kwargs["gradient_checkpointing"] is True

    def test_resolve_auto_high_memory_selects_selective(self):
        from soup_cli.utils.gradient_ckpt import resolve_gradient_checkpointing

        # 80GB+ → selective only (attention), saving speed
        kwargs = resolve_gradient_checkpointing("auto", gpu_memory_gb=80)
        assert kwargs["gradient_checkpointing"] is True

    def test_resolve_auto_very_high_memory_selects_selective(self):
        from soup_cli.utils.gradient_ckpt import (
            resolve_granularity,
        )

        # 192GB (H200): selective (attention-only) tier, minimize slowdown
        assert resolve_granularity("auto", gpu_memory_gb=192) == "selective"

    def test_resolve_auto_medium_memory_selects_medium(self):
        from soup_cli.utils.gradient_ckpt import resolve_granularity

        # 40GB (A100 40GB): medium (every other block)
        assert resolve_granularity("auto", gpu_memory_gb=40) == "medium"

    def test_resolve_auto_no_gpu_info_full(self):
        from soup_cli.utils.gradient_ckpt import resolve_granularity

        assert resolve_granularity("auto", gpu_memory_gb=None) == "full"

    def test_describe_tier_off(self):
        from soup_cli.utils.gradient_ckpt import describe_tier

        assert describe_tier(False) == "off"

    def test_describe_tier_full(self):
        from soup_cli.utils.gradient_ckpt import describe_tier

        assert "full" in describe_tier(True)
        assert "full" in describe_tier("full")

    def test_describe_tier_auto(self):
        from soup_cli.utils.gradient_ckpt import describe_tier

        assert "auto" in describe_tier("auto")
        assert "auto" in describe_tier("auto", gpu_memory_gb=16)
        assert "selective" in describe_tier("auto", gpu_memory_gb=192)


# ─── Part D: Kernel auto-composition ─────────────────────────────────────


class TestKernelPickerConfig:
    """kernel_auto_compose config flag."""

    def test_kernel_auto_compose_default_false(self):
        cfg = SoupConfig(base="test/model", data={"train": "./data.jsonl"})
        assert cfg.training.kernel_auto_compose is False

    def test_kernel_auto_compose_enabled(self):
        cfg = SoupConfig(
            base="test/model",
            data={"train": "./data.jsonl"},
            training={"kernel_auto_compose": True},
        )
        assert cfg.training.kernel_auto_compose is True


class TestKernelPickerEnumerate:
    """Kernel picker enumerates available kernel combinations."""

    def test_enumerate_returns_list(self):
        from soup_cli.utils.kernel_picker import enumerate_kernel_combos

        combos = enumerate_kernel_combos(backend="transformers", device="cuda")
        assert isinstance(combos, list)

    def test_enumerate_baseline_always_present(self):
        from soup_cli.utils.kernel_picker import enumerate_kernel_combos

        combos = enumerate_kernel_combos(backend="transformers", device="cuda")
        # Baseline (no special kernels) must always be an option
        assert any(c.get("name") == "baseline" for c in combos)

    def test_enumerate_cpu_only_baseline(self):
        from soup_cli.utils.kernel_picker import enumerate_kernel_combos

        combos = enumerate_kernel_combos(backend="transformers", device="cpu")
        # On CPU, only baseline should be available
        assert len(combos) == 1
        assert combos[0]["name"] == "baseline"

    def test_enumerate_unsloth_skips_liger(self):
        from soup_cli.utils.kernel_picker import enumerate_kernel_combos

        combos = enumerate_kernel_combos(backend="unsloth", device="cuda")
        # Unsloth has its own fused kernels - no Liger combos
        for combo in combos:
            assert "liger" not in combo.get("name", "").lower()


class TestKernelPickerDecision:
    """Kernel picker decision logic (mocked benchmarks)."""

    def test_pick_best_returns_dict(self):
        from soup_cli.utils.kernel_picker import pick_best_kernel

        # With fake timing results, picks fastest
        candidates = [
            {"name": "baseline", "time_ms": 100.0},
            {"name": "liger", "time_ms": 70.0},
            {"name": "liger+flash", "time_ms": 50.0},
        ]
        best = pick_best_kernel(candidates)
        assert best["name"] == "liger+flash"

    def test_pick_best_baseline_if_only_one(self):
        from soup_cli.utils.kernel_picker import pick_best_kernel

        candidates = [{"name": "baseline", "time_ms": 100.0}]
        best = pick_best_kernel(candidates)
        assert best["name"] == "baseline"

    def test_pick_best_empty_raises(self):
        from soup_cli.utils.kernel_picker import pick_best_kernel

        with pytest.raises(ValueError, match="at least one"):
            pick_best_kernel([])

    def test_pick_best_tie_returns_first(self):
        """Ties broken by list order (first-wins) — preferred default first."""
        from soup_cli.utils.kernel_picker import pick_best_kernel

        candidates = [
            {"name": "baseline", "time_ms": 50.0},
            {"name": "liger", "time_ms": 50.0},
        ]
        best = pick_best_kernel(candidates)
        assert best["name"] == "baseline"

    def test_pick_best_all_missing_time_raises(self):
        """All-untimed candidates means benchmarking failed — must not promote silently."""
        from soup_cli.utils.kernel_picker import pick_best_kernel

        candidates = [
            {"name": "baseline"},
            {"name": "liger"},
            {"name": "flash", "time_ms": None},
        ]
        with pytest.raises(ValueError, match="finite time_ms"):
            pick_best_kernel(candidates)

    def test_pick_best_nan_time_treated_as_missing(self):
        from soup_cli.utils.kernel_picker import pick_best_kernel

        candidates = [
            {"name": "baseline", "time_ms": float("nan")},
            {"name": "liger", "time_ms": 50.0},
        ]
        best = pick_best_kernel(candidates)
        assert best["name"] == "liger"


# ─── Part E: Cross-document attention masking ────────────────────────────


class TestCrossDocAttnMaskConfig:
    """packing_cross_doc_attn_mask config."""

    def test_default_false(self):
        cfg = SoupConfig(base="test/model", data={"train": "./data.jsonl"})
        assert cfg.training.packing_cross_doc_attn_mask is False

    def test_enabled(self):
        cfg = SoupConfig(
            base="test/model",
            data={"train": "./data.jsonl"},
            training={"packing_cross_doc_attn_mask": True, "packing": True},
        )
        assert cfg.training.packing_cross_doc_attn_mask is True

    def test_requires_packing(self):
        """Enabling cross-doc mask without packing should error."""
        with pytest.raises(ValidationError) as exc:
            SoupConfig(
                base="test/model",
                data={"train": "./data.jsonl"},
                training={
                    "packing_cross_doc_attn_mask": True,
                    "packing": False,
                },
            )
        assert "packing" in str(exc.value).lower()


class TestCrossDocAttnMaskBuild:
    """Build cross-doc attention mask from document boundaries."""

    def test_build_mask_single_doc(self):
        from soup_cli.utils.cross_doc_attn import build_cross_doc_mask

        # Single doc spans the whole sequence — no masking needed
        boundaries = [0, 10]  # doc0 occupies positions 0..9
        mask = build_cross_doc_mask(boundaries, seq_length=10)
        # Shape (10, 10), lower-triangular within the doc
        assert mask.shape == (10, 10)

    def test_build_mask_two_docs(self):
        from soup_cli.utils.cross_doc_attn import build_cross_doc_mask

        # doc0: 0..4, doc1: 5..9
        boundaries = [0, 5, 10]
        mask = build_cross_doc_mask(boundaries, seq_length=10)
        # Position 5 (doc1 start) should NOT attend to position 0 (doc0)
        assert mask[5, 0] == 0
        # Position 5 attending to itself should be 1
        assert mask[5, 5] == 1
        # Position 0 attending to itself should be 1
        assert mask[0, 0] == 1
        # Position 1 attending to 0 should be 1 (same doc, causal)
        assert mask[1, 0] == 1
        # Position 0 attending to 1 should be 0 (causal — future)
        assert mask[0, 1] == 0

    def test_build_mask_boundaries_validation(self):
        from soup_cli.utils.cross_doc_attn import build_cross_doc_mask

        # Boundaries must start at 0 and end at seq_length
        with pytest.raises(ValueError):
            build_cross_doc_mask([1, 10], seq_length=10)

    def test_build_mask_boundaries_monotonic(self):
        from soup_cli.utils.cross_doc_attn import build_cross_doc_mask

        with pytest.raises(ValueError, match="increasing"):
            build_cross_doc_mask([0, 5, 3, 10], seq_length=10)

    def test_build_mask_empty_boundaries_raises(self):
        from soup_cli.utils.cross_doc_attn import build_cross_doc_mask

        with pytest.raises(ValueError, match="non-empty"):
            build_cross_doc_mask([], seq_length=10)

    def test_build_mask_wrong_end_raises(self):
        from soup_cli.utils.cross_doc_attn import build_cross_doc_mask

        with pytest.raises(ValueError, match="seq_length"):
            build_cross_doc_mask([0, 8], seq_length=10)

    def test_compute_doc_boundaries_empty(self):
        from soup_cli.utils.cross_doc_attn import compute_doc_boundaries

        with pytest.raises(ValueError):
            compute_doc_boundaries([])

    def test_compute_doc_boundaries_non_positive(self):
        from soup_cli.utils.cross_doc_attn import compute_doc_boundaries

        with pytest.raises(ValueError):
            compute_doc_boundaries([3, 0, 4])

    def test_compute_doc_boundaries_valid(self):
        from soup_cli.utils.cross_doc_attn import compute_doc_boundaries

        assert compute_doc_boundaries([3, 2, 4]) == [0, 3, 5, 9]


# ─── Part F: Activation offloading ────────────────────────────────────────


class TestActivationOffloadingConfig:
    """activation_offloading config."""

    def test_default_none(self):
        cfg = SoupConfig(base="test/model", data={"train": "./data.jsonl"})
        assert cfg.training.activation_offloading is None

    def test_cpu(self):
        cfg = SoupConfig(
            base="test/model",
            data={"train": "./data.jsonl"},
            training={"activation_offloading": "cpu"},
        )
        assert cfg.training.activation_offloading == "cpu"

    def test_disk(self):
        cfg = SoupConfig(
            base="test/model",
            data={"train": "./data.jsonl"},
            training={"activation_offloading": "disk"},
        )
        assert cfg.training.activation_offloading == "disk"

    def test_invalid_target(self):
        with pytest.raises(ValidationError) as exc:
            SoupConfig(
                base="test/model",
                data={"train": "./data.jsonl"},
                training={"activation_offloading": "gpu"},
            )
        assert "activation_offloading" in str(exc.value)


class TestActivationOffloadingValidation:
    """Offloading config validation."""

    def test_validate_none_returns_empty(self):
        from soup_cli.utils.activation_offload import validate_offload_config

        errors = validate_offload_config(None, "transformers", "cuda")
        assert errors == []

    def test_validate_cpu_requires_cuda(self):
        from soup_cli.utils.activation_offload import validate_offload_config

        errors = validate_offload_config("cpu", "transformers", "cpu")
        assert any("CUDA" in err for err in errors)

    def test_validate_cpu_unsloth_incompatible(self):
        from soup_cli.utils.activation_offload import validate_offload_config

        errors = validate_offload_config("cpu", "unsloth", "cuda")
        assert any("unsloth" in err.lower() for err in errors)

    def test_validate_disk_valid(self):
        from soup_cli.utils.activation_offload import validate_offload_config

        errors = validate_offload_config(
            "disk", "transformers", "cuda", save_dir="./scratch"
        )
        assert errors == []

    def test_validate_disk_requires_save_dir(self):
        """Disk mode must reject calls without save_dir — fail-fast at validate()."""
        from soup_cli.utils.activation_offload import validate_offload_config

        errors = validate_offload_config(
            "disk", "transformers", "cuda", save_dir=None
        )
        assert any("save_dir" in err for err in errors)

    def test_validate_disk_on_cpu_rejected(self):
        from soup_cli.utils.activation_offload import validate_offload_config

        errors = validate_offload_config(
            "disk", "transformers", "cpu", save_dir="./scratch"
        )
        assert any("CUDA" in err for err in errors)

    def test_validate_mlx_rejected(self):
        from soup_cli.utils.activation_offload import validate_offload_config

        errors = validate_offload_config("cpu", "mlx", "cuda")
        assert any("mlx" in err.lower() for err in errors)


class TestActivationOffloadingHooks:
    """Install / uninstall hooks for offloading."""

    def test_context_manager_noop_when_none(self):
        from soup_cli.utils.activation_offload import offload_context

        # Should be a no-op when target is None
        with offload_context(None, save_dir=None):
            pass  # nothing to assert - just shouldn't crash

    def test_context_manager_cpu(self):
        from soup_cli.utils.activation_offload import offload_context

        # Should not crash even without torch
        with offload_context("cpu", save_dir=None):
            pass

    def test_offload_context_disk_requires_dir(self, tmp_path):
        from soup_cli.utils.activation_offload import offload_context

        # Disk mode should accept a save_dir
        with offload_context("disk", save_dir=str(tmp_path)):
            pass

    def test_offload_context_unknown_target_raises(self):
        """Defense-in-depth: unknown target raises even if torch is present."""
        from soup_cli.utils.activation_offload import offload_context

        # Only reaches the ValueError branch if torch imports successfully
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not installed; ValueError branch unreachable")

        with pytest.raises(ValueError, match="Unknown activation_offloading"):
            with offload_context("invalid", save_dir=None):
                pass

    def test_offload_context_disk_creates_save_dir(self, tmp_path):
        """Disk mode should create the scratch directory on context entry."""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not installed; disk hooks unreachable")

        from soup_cli.utils.activation_offload import offload_context

        scratch = tmp_path / "offload_scratch"
        with offload_context("disk", save_dir=str(scratch)):
            assert scratch.exists()


# ─── Integration: multiple features composed ──────────────────────────────


class TestV028SFTOnlyValidator:
    """v0.28.0 features are wired only in SFTTrainerWrapper.

    The SoupConfig validator rejects non-SFT tasks when speed/memory flags
    are set — prevents silent no-ops and the known fp8 crash path in the
    legacy int8 QAT wrapper.
    """

    def test_use_cut_ce_now_accepted_on_dpo(self):
        # v0.33.0 #43 — DPO is now in the supported task set.
        cfg = SoupConfig(
            base="m",
            task="dpo",
            data={"train": "./d.jsonl", "format": "dpo"},
            training={"use_cut_ce": True},
        )
        assert cfg.training.use_cut_ce is True

    def test_fp8_rejected_on_grpo(self):
        with pytest.raises(ValidationError) as exc:
            SoupConfig(
                base="m",
                task="grpo",
                data={"train": "./d.jsonl"},
                training={"quantization_aware": "fp8"},
            )
        assert "fp8" in str(exc.value)

    def test_activation_offloading_rejected_on_ppo(self):
        with pytest.raises(ValidationError) as exc:
            SoupConfig(
                base="m",
                task="ppo",
                data={"train": "./d.jsonl"},
                training={"activation_offloading": "cpu"},
            )
        assert "activation_offloading" in str(exc.value)

    def test_kernel_auto_compose_rejected_on_kto(self):
        with pytest.raises(ValidationError) as exc:
            SoupConfig(
                base="m",
                task="kto",
                data={"train": "./d.jsonl", "format": "kto"},
                training={"kernel_auto_compose": True},
            )
        assert "kernel_auto_compose" in str(exc.value)

    def test_sft_accepts_all_features(self):
        """SFT task should accept every v0.28.0 flag (happy path)."""
        cfg = SoupConfig(
            base="m",
            task="sft",
            data={"train": "./d.jsonl"},
            training={
                "use_cut_ce": True,
                "quantization_aware": "fp8",
                "activation_offloading": "cpu",
                "kernel_auto_compose": True,
            },
        )
        assert cfg.training.use_cut_ce is True

    def test_non_sft_unaffected_when_flags_default(self):
        """DPO/GRPO with default v0.28.0 flags still validate."""
        cfg = SoupConfig(
            base="m",
            task="dpo",
            data={"train": "./d.jsonl", "format": "dpo"},
        )
        assert cfg.task == "dpo"

    def test_quantization_aware_bool_true_allowed_on_dpo(self):
        """Int8 QAT (bool True) still works on non-SFT — only fp8 is restricted."""
        cfg = SoupConfig(
            base="m",
            task="dpo",
            data={"train": "./d.jsonl", "format": "dpo"},
            training={"quantization_aware": True},
        )
        assert cfg.training.quantization_aware is True

    def test_gradient_checkpointing_tier_allowed_on_dpo(self):
        """Tier strings fall back to truthy (bool True) in non-SFT wrappers — no crash."""
        cfg = SoupConfig(
            base="m",
            task="dpo",
            data={"train": "./d.jsonl", "format": "dpo"},
            training={"gradient_checkpointing": "auto"},
        )
        assert cfg.training.gradient_checkpointing == "auto"


class TestV028Integration:
    """Multiple v0.28.0 features composed in one config."""

    def test_all_features_compose(self):
        cfg = SoupConfig(
            base="test/model",
            data={"train": "./data.jsonl"},
            training={
                "use_cut_ce": True,
                "quantization_aware": "fp8",
                "gradient_checkpointing": "auto",
                "kernel_auto_compose": True,
                "packing": True,
                "packing_cross_doc_attn_mask": True,
                "activation_offloading": "cpu",
            },
        )
        tcfg = cfg.training
        assert tcfg.use_cut_ce is True
        assert tcfg.quantization_aware == "fp8"
        assert tcfg.gradient_checkpointing == "auto"
        assert tcfg.kernel_auto_compose is True
        assert tcfg.packing is True
        assert tcfg.packing_cross_doc_attn_mask is True
        assert tcfg.activation_offloading == "cpu"
