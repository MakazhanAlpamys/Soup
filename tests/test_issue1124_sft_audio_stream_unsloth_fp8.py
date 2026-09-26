"""Tests for issue #1124 вЂ” SFT audio FP8 wiring and stream_layers/unsloth gating.

Validates that:
1. SFTTrainerWrapper._setup_audio_transformers calls _apply_quantization_aware(tcfg)
   post-LoRA, bringing audio SFT to parity with text SFT for FP8/NVFP4.
2. stream_layers: true is mutually exclusive with quantization_aware, fp8_attention,
   and nvfp4 at schema validation time.
3. backend: unsloth rejects quantization_aware: fp8, fp8_attention, and nvfp4
   at schema validation time with actionable diagnostics.
4. validate_fp8_attention_compat and validate_nvfp4_compat reject unsloth.
5. backend_support declares ("sft", "unsloth", "text") with REJECTED status for precision knobs.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from soup_cli.config.backend_support import (
    REGISTRY,
    TRAINER_MODULES,
    check_config,
    unsupported_for,
)
from soup_cli.config.schema import SoupConfig
from soup_cli.trainer.sft import SFTTrainerWrapper
from soup_cli.utils.advanced_precision import (
    validate_fp8_attention_compat,
    validate_nvfp4_compat,
)


class TestSFTAudioFP8Wiring:
    """Test that SFTTrainerWrapper applies quantization_aware post-LoRA on audio."""

    def test_audio_sft_stops_at_setup_on_unsupported_card(self, monkeypatch) -> None:
        """Driving the real setup (Issue #1124 AC2): on an sm86 card, SFT audio setup
        with quantization_aware='fp8' must stop at setup with FP8HardwareUnsupportedError.
        On main, this silently ran without error because _apply_quantization_aware was never called.
        """
        import torch
        import torch.nn as nn

        from soup_cli.utils.fp8 import FP8HardwareUnsupportedError

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_a, **_k: (8, 6))

        cfg = SoupConfig(
            base="test/audio-model",
            task="sft",
            modality="audio",
            data={"train": "tests/fixtures/sample_train.jsonl", "format": "audio"},
            training={"quantization_aware": "fp8"},
        )

        class _MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 4)

        fake_model = _MockModel()
        fake_processor = MagicMock()

        fake_peft = types.SimpleNamespace(
            LoraConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
            get_peft_model=lambda m, *a, **kw: m,
            prepare_model_for_kbit_training=lambda m, **kw: m,
        )
        fake_transformers = types.SimpleNamespace(
            AutoModel=types.SimpleNamespace(from_pretrained=MagicMock(return_value=fake_model)),
            AutoProcessor=types.SimpleNamespace(
                from_pretrained=MagicMock(return_value=fake_processor)
            ),
            TrainingArguments=MagicMock(),
        )

        monkeypatch.setitem(sys.modules, "peft", fake_peft)
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        monkeypatch.setattr(
            "soup_cli.utils.quant_menu.build_quantization_config_for_loader",
            lambda **kw: None,
        )
        monkeypatch.setattr(
            "soup_cli.utils.data_pipeline.apply_vocab_expansion",
            lambda *a, **kw: None,
        )
        monkeypatch.setattr(
            "soup_cli.trainer.sft.resolve_device_map",
            lambda d: None,
        )

        wrapper = object.__new__(SFTTrainerWrapper)
        wrapper.config = cfg
        wrapper.device = "cuda:0"
        wrapper._trust_remote_code = False
        wrapper.model = None
        wrapper.processor = None
        wrapper.tokenizer = None

        with pytest.raises(FP8HardwareUnsupportedError, match="8.9"):
            wrapper._setup_audio_transformers(cfg, cfg.training)

    def test_audio_setup_converts_the_lora_wrapped_model(self, monkeypatch) -> None:
        """The converter must see the post-LoRA wrapper, not the bare base model."""
        import torch.nn as nn

        cfg = SoupConfig(
            base="test/audio-model",
            task="sft",
            modality="audio",
            data={"train": "tests/fixtures/sample_train.jsonl", "format": "audio"},
            training={"quantization_aware": "fp8"},
        )

        class _MockBaseModel(nn.Module):
            pass

        class _MockLoraWrapper(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base

        base_model = _MockBaseModel()
        lora_wrapped = _MockLoraWrapper(base_model)

        fake_peft = types.SimpleNamespace(
            LoraConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
            get_peft_model=lambda m, *a, **kw: lora_wrapped,
            prepare_model_for_kbit_training=lambda m, **kw: m,
        )
        fake_transformers = types.SimpleNamespace(
            AutoModel=types.SimpleNamespace(from_pretrained=MagicMock(return_value=base_model)),
            AutoProcessor=types.SimpleNamespace(from_pretrained=MagicMock()),
            TrainingArguments=MagicMock(),
        )

        monkeypatch.setitem(sys.modules, "peft", fake_peft)
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        monkeypatch.setattr(
            "soup_cli.utils.quant_menu.build_quantization_config_for_loader",
            lambda **kw: None,
        )
        monkeypatch.setattr(
            "soup_cli.utils.data_pipeline.apply_vocab_expansion",
            lambda *a, **kw: None,
        )
        monkeypatch.setattr(
            "soup_cli.trainer.sft.resolve_device_map",
            lambda d: None,
        )

        model_seen_by_converter = None

        def fake_apply_qat(tcfg):
            nonlocal model_seen_by_converter
            model_seen_by_converter = wrapper.model

        wrapper = object.__new__(SFTTrainerWrapper)
        wrapper.config = cfg
        wrapper.device = "cpu"
        wrapper._trust_remote_code = False
        wrapper.model = None
        wrapper.processor = None
        wrapper.tokenizer = None
        wrapper._apply_quantization_aware = fake_apply_qat

        wrapper._setup_audio_transformers(cfg, cfg.training)

        assert model_seen_by_converter is lora_wrapped, (
            "_apply_quantization_aware was called on base model instead of the "
            "LoRA-wrapped model (must run after get_peft_model)"
        )
        assert model_seen_by_converter is not base_model

    def test_audio_setup_source_structure(self) -> None:
        """Kill mutation: ensure _setup_audio_transformers calls
        _apply_quantization_aware post-LoRA.
        """
        sft_path = Path(__file__).resolve().parents[1] / "src/soup_cli/trainer/sft.py"
        tree = ast.parse(sft_path.read_text(encoding="utf-8"))

        setup_audio_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_setup_audio_transformers":
                setup_audio_func = node
                break

        assert setup_audio_func is not None, "_setup_audio_transformers not found in sft.py"

        peft_idx = None
        qat_idx = None
        for i, node in enumerate(setup_audio_func.body):
            if isinstance(node, ast.Assign):
                if any(isinstance(t, ast.Attribute) and t.attr == "model" for t in node.targets):
                    if (
                        isinstance(node.value, ast.Call)
                        and getattr(node.value.func, "id", None) == "get_peft_model"
                    ):
                        peft_idx = i
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                if (
                    isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "_apply_quantization_aware"
                ):
                    qat_idx = i

        assert peft_idx is not None, "get_peft_model assignment not found"
        assert qat_idx is not None, "_apply_quantization_aware call not found"
        assert peft_idx < qat_idx, (
            "Mutation kill: _apply_quantization_aware must be called after "
            "get_peft_model in _setup_audio_transformers"
        )


class TestStreamLayersPrecisionConflicts:
    """Test schema mutual exclusivity for stream_layers with FP8/NVFP4."""

    def test_stream_layers_rejects_quantization_aware_fp8(self) -> None:
        pattern = r"stream_layers is mutually exclusive with quantization_aware"
        with pytest.raises(ValueError, match=pattern):
            SoupConfig(
                base="test/model",
                task="sft",
                data={"train": "tests/fixtures/sample_train.jsonl"},
                training={
                    "stream_layers": True,
                    "batch_size": 1,
                    "quantization_aware": "fp8",
                },
            )

    def test_stream_layers_rejects_quantization_aware_bool(self) -> None:
        pattern = r"stream_layers is mutually exclusive with quantization_aware"
        with pytest.raises(ValueError, match=pattern):
            SoupConfig(
                base="test/model",
                task="sft",
                data={"train": "tests/fixtures/sample_train.jsonl"},
                training={
                    "stream_layers": True,
                    "batch_size": 1,
                    "quantization_aware": True,
                },
            )

    def test_stream_layers_rejects_fp8_attention(self) -> None:
        pattern = r"stream_layers is mutually exclusive with fp8_attention"
        with pytest.raises(ValueError, match=pattern):
            SoupConfig(
                base="test/model",
                task="sft",
                data={"train": "tests/fixtures/sample_train.jsonl"},
                training={
                    "stream_layers": True,
                    "batch_size": 1,
                    "fp8_attention": True,
                },
            )

    def test_stream_layers_rejects_nvfp4(self) -> None:
        pattern = r"stream_layers is mutually exclusive with nvfp4"
        with pytest.raises(ValueError, match=pattern):
            SoupConfig(
                base="test/model",
                task="sft",
                data={"train": "tests/fixtures/sample_train.jsonl"},
                training={
                    "stream_layers": True,
                    "batch_size": 1,
                    "nvfp4": True,
                },
            )


class TestUnslothPrecisionRejections:
    """Test unsloth backend rejection of quantization_aware: fp8, fp8_attention, nvfp4."""

    def test_unsloth_rejects_quantization_aware_fp8(self) -> None:
        pattern = r"quantization_aware.*not supported on the unsloth backend"
        with pytest.raises(ValueError, match=pattern):
            SoupConfig(
                base="test/model",
                task="sft",
                backend="unsloth",
                data={"train": "tests/fixtures/sample_train.jsonl"},
                training={"quantization_aware": "fp8"},
            )

    def test_unsloth_rejects_fp8_attention(self) -> None:
        pattern = r"quantization_aware.*not supported on the unsloth backend"
        with pytest.raises(ValueError, match=pattern):
            SoupConfig(
                base="test/model",
                task="sft",
                backend="unsloth",
                data={"train": "tests/fixtures/sample_train.jsonl"},
                training={"quantization_aware": "fp8", "fp8_attention": True},
            )

    def test_unsloth_rejects_fp8_attention_without_qa_reports_qa_prereq_first(
        self,
    ) -> None:
        """Actionable order: prerequisite quantization_aware='fp8' fires first."""
        with pytest.raises(ValueError, match=r"requires training\.quantization_aware='fp8'"):
            SoupConfig(
                base="test/model",
                task="sft",
                backend="unsloth",
                data={"train": "tests/fixtures/sample_train.jsonl"},
                training={"fp8_attention": True},
            )

    def test_unsloth_rejects_nvfp4(self) -> None:
        with pytest.raises(ValueError, match=r"nvfp4=true is not supported on backend=unsloth"):
            SoupConfig(
                base="test/model",
                task="sft",
                backend="unsloth",
                data={"train": "tests/fixtures/sample_train.jsonl"},
                training={"nvfp4": True},
            )

    def test_validate_fp8_attention_compat_rejects_unsloth(self) -> None:
        pattern = r"backend=unsloth.*unsloth uses its own fused kernels"
        with pytest.raises(ValueError, match=pattern):
            validate_fp8_attention_compat(
                fp8_attention=True,
                quantization_aware="fp8",
                backend="unsloth",
            )

    def test_validate_nvfp4_compat_rejects_unsloth(self) -> None:
        pattern = r"backend=unsloth.*unsloth uses its own fused kernels"
        with pytest.raises(ValueError, match=pattern):
            validate_nvfp4_compat(
                nvfp4=True,
                backend="unsloth",
                modality="text",
            )

    def test_unsloth_permits_cut_ce_and_activation_offloading(self) -> None:
        """Control: unsloth validation only refuses FP8/NVFP4, NOT cut_ce
        or activation offloading.
        """
        cfg_cut_ce = SoupConfig(
            base="test/model",
            backend="unsloth",
            task="sft",
            data={"train": "tests/fixtures/sample_train.jsonl"},
            training={"use_cut_ce": True},
        )
        assert cfg_cut_ce.training.use_cut_ce is True

        cfg_offload = SoupConfig(
            base="test/model",
            backend="unsloth",
            task="sft",
            data={"train": "tests/fixtures/sample_train.jsonl"},
            training={"activation_offloading": "cpu"},
        )
        assert cfg_offload.training.activation_offloading == "cpu"


class TestBackendSupportRegistryUnsloth:
    """Test that backend_support registers unsloth SFT rejected entries."""

    def test_unsloth_sft_registered(self) -> None:
        assert ("sft", "unsloth", "text") in REGISTRY
        entries = unsupported_for("sft", "unsloth")
        fields = {e.field: e.status for e in entries}
        assert fields.get("training.quantization_aware") == "rejected"
        assert fields.get("training.fp8_attention") == "rejected"
        assert fields.get("training.nvfp4") == "rejected"
        assert all(e.issue == 1124 for e in entries)

        entries_vision = unsupported_for("sft", "unsloth", "vision")
        assert len(entries_vision) == len(entries)
        fields_vision = {e.field: e.status for e in entries_vision}
        assert fields_vision.get("training.quantization_aware") == "rejected"

    def test_unsloth_sft_in_trainer_modules(self) -> None:
        assert ("sft", "unsloth", "text") in TRAINER_MODULES
        modules = TRAINER_MODULES[("sft", "unsloth", "text")]
        assert "soup_cli/utils/unsloth.py" in modules

    def test_check_config_reports_unsloth_gaps(self) -> None:
        cfg_text = SoupConfig(
            base="test/model",
            task="sft",
            backend="unsloth",
            modality="text",
            data={"train": "tests/fixtures/sample_train.jsonl"},
            training={"quantization_aware": True},
        )
        gaps_text = check_config(cfg_text)
        assert any(g.field == "training.quantization_aware" for g in gaps_text)

        cfg_vision = SoupConfig(
            base="test/model",
            task="sft",
            backend="unsloth",
            modality="vision",
            data={"train": "tests/fixtures/sample_train.jsonl", "format": "multimodal"},
            training={"quantization_aware": True},
        )
        gaps_vision = check_config(cfg_vision)
        assert any(g.field == "training.quantization_aware" for g in gaps_vision)

    def test_setup_unsloth_does_not_wire_fp8_silently(self) -> None:
        """AST guard: SFTTrainerWrapper._setup_unsloth must not wire precision knobs
        while ("sft", "unsloth", "text") declares them REJECTED in backend_support.
        """
        sft_path = Path(__file__).resolve().parents[1] / "src/soup_cli/trainer/sft.py"
        tree = ast.parse(sft_path.read_text(encoding="utf-8"))

        setup_unsloth = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_setup_unsloth":
                setup_unsloth = node
                break

        assert setup_unsloth is not None, "_setup_unsloth not found in sft.py"
        unsloth_source = ast.unparse(setup_unsloth)
        for forbidden in ("quantization_aware", "fp8_attention", "nvfp4", "apply_fp8"):
            assert forbidden not in unsloth_source, (
                f"Precision knob {forbidden} found in _setup_unsloth; update backend_support!"
            )
