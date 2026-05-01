"""Tests for v0.39.0 Part D — surgical PEFT patches.

Covers detection helpers + gated patch entry points for:
- Gemma4 ``ClippableLinear`` (PEFT's LoRA layer registry doesn't know about it)
- Fused-MoE 3-D expert weights (PEFT's ParamWrapper crashes on dropout)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# --- Gemma4 ClippableLinear detection ---------------------------------------


class TestGemma4Detection:
    def test_is_gemma4_positive_lower(self):
        from soup_cli.utils.peft_patches import is_gemma4_model
        assert is_gemma4_model("google/gemma-4-9b") is True
        assert is_gemma4_model("google/gemma-4-it") is True
        assert is_gemma4_model("Gemma-4-2B") is True

    def test_is_gemma4_negative(self):
        from soup_cli.utils.peft_patches import is_gemma4_model
        assert is_gemma4_model("google/gemma-2-9b") is False
        assert is_gemma4_model("meta-llama/Meta-Llama-3.1-8B") is False
        assert is_gemma4_model("") is False
        assert is_gemma4_model(None) is False  # type: ignore

    def test_is_gemma4_word_boundary(self):
        """v0.39.0 security fix — substring match would over-match."""
        from soup_cli.utils.peft_patches import is_gemma4_model
        # NOT Gemma 4
        assert is_gemma4_model("ungemma4ed") is False
        assert is_gemma4_model("megagemma40-experiment") is False
        # IS Gemma 4 — word-boundary cases
        assert is_gemma4_model("my-org/finetuned-gemma-4-style") is True
        assert is_gemma4_model("google/gemma4_instruct") is True

    def test_is_gemma4_rejects_null_byte(self):
        from soup_cli.utils.peft_patches import is_gemma4_model
        # crafted name with null byte should not match
        assert is_gemma4_model("gemma-4\x00malicious") is False


class TestClippableLinearPatch:
    def test_no_clippable_linear_returns_zero(self):
        try:
            import torch.nn as nn
        except ImportError:
            pytest.skip("torch not available")
        from soup_cli.utils.peft_patches import apply_gemma4_clippable_patch

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4)

        model = Model()
        count = apply_gemma4_clippable_patch(model)
        assert count == 0

    def test_clippable_linear_replaced(self):
        try:
            import torch.nn as nn
        except ImportError:
            pytest.skip("torch not available")
        from soup_cli.utils.peft_patches import apply_gemma4_clippable_patch

        # Simulate Gemma4's ClippableLinear by name
        class ClippableLinear(nn.Linear):
            pass

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = ClippableLinear(4, 4)
                self.fc2 = nn.Linear(4, 4)

        model = Model()
        count = apply_gemma4_clippable_patch(model)
        assert count == 1
        # After patch: fc1 is plain nn.Linear (or PEFT-recognised)
        assert type(model.fc1) is nn.Linear


# --- MoE 3D expert dropout strip --------------------------------------------


class TestMoE3DDropoutStrip:
    def test_strip_when_no_3d_experts(self):
        from soup_cli.utils.peft_patches import strip_lora_dropout_for_3d_experts
        # peft model with only 2-D weights — strip is no-op
        peft_model = MagicMock()
        peft_model.named_modules.return_value = [
            ("base.layer1", MagicMock(weight=MagicMock(ndim=2))),
        ]
        count = strip_lora_dropout_for_3d_experts(peft_model)
        assert count == 0

    def test_strip_zeroes_dropout_on_3d_module(self):
        from soup_cli.utils.peft_patches import strip_lora_dropout_for_3d_experts
        # Build a fake module tree: experts.0.gate_proj has 3-D weight + lora_dropout
        expert = MagicMock()
        expert.weight = MagicMock(ndim=3)
        expert.lora_dropout = MagicMock()
        expert.lora_dropout.p = 0.1

        peft_model = MagicMock()
        peft_model.named_modules.return_value = [
            ("base.experts.0.gate_proj", expert),
        ]
        count = strip_lora_dropout_for_3d_experts(peft_model)
        assert count == 1
        assert expert.lora_dropout.p == 0.0

    def test_strip_handles_module_dict_dropout(self):
        """PEFT >=0.10 wraps lora_dropout in a ModuleDict — exercise the values() branch."""
        from soup_cli.utils.peft_patches import strip_lora_dropout_for_3d_experts

        sub_a = MagicMock(spec=["p"])
        sub_a.p = 0.1
        sub_b = MagicMock(spec=["p"])
        sub_b.p = 0.2

        class FakeModuleDict:
            def values(self):
                return [sub_a, sub_b]

        expert = MagicMock(spec=["weight", "lora_dropout"])
        expert.weight = MagicMock(ndim=3)
        # FakeModuleDict has no `p` attribute → hasattr() is False → elif fires.
        expert.lora_dropout = FakeModuleDict()

        peft_model = MagicMock()
        peft_model.named_modules.return_value = [("base.experts.0", expert)]
        count = strip_lora_dropout_for_3d_experts(peft_model)
        assert count == 2
        assert sub_a.p == 0.0
        assert sub_b.p == 0.0


class TestApplySurgicalPatches:
    def test_returns_dict_with_counts(self):
        try:
            import torch.nn as nn
        except ImportError:
            pytest.skip("torch not available")
        from soup_cli.utils.peft_patches import apply_surgical_patches

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4)

        result = apply_surgical_patches(Model(), model_name="meta-llama/Meta-Llama-3.1-8B")
        assert isinstance(result, dict)
        assert "gemma4_clippable" in result
        assert "moe_3d_dropout" in result
        assert result["gemma4_clippable"] == 0

    def test_gemma4_only_runs_for_gemma4_models(self):
        try:
            import torch.nn as nn
        except ImportError:
            pytest.skip("torch not available")
        from soup_cli.utils.peft_patches import apply_surgical_patches

        class ClippableLinear(nn.Linear):
            pass

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = ClippableLinear(4, 4)

        # non-Gemma4: skip
        m = Model()
        result = apply_surgical_patches(m, model_name="meta-llama/Llama-3-8B")
        assert result["gemma4_clippable"] == 0
        assert type(m.fc1) is ClippableLinear

        # Gemma4: apply
        m2 = Model()
        result2 = apply_surgical_patches(m2, model_name="google/gemma-4-9b")
        assert result2["gemma4_clippable"] == 1

    def test_rejects_empty_model_name(self):
        from soup_cli.utils.peft_patches import apply_surgical_patches
        with pytest.raises(ValueError):
            apply_surgical_patches(MagicMock(), model_name="")

    def test_rejects_null_byte_model_name(self):
        from soup_cli.utils.peft_patches import apply_surgical_patches
        with pytest.raises(ValueError):
            apply_surgical_patches(MagicMock(), model_name="gemma-4\x00x")
