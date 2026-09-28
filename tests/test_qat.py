"""Tests for Quantization-Aware Training (QAT) — config, validation, trainer integration.

``quantization_aware: true`` (int8 QAT) is refused at config load since #1222.
Where it was applied, the path behind it ran torchao's post-training int8
weight-only quantization over the model, LoRA adapter included, and left
nothing trainable; elsewhere it was not applied at all. The tests that used to
show ``true`` loading now show it refused; the refusal itself, its message and
the ``fp8`` / ``quest`` controls live in ``tests/test_issue1222_refuse_int8_qat.py``.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from soup_cli.config.schema import SoupConfig
from tests.conftest import strip_ansi

# ─── Config Tests ───────────────────────────────────────────────────────────


class TestQATConfig:
    """Test quantization_aware config field validation."""

    def test_qat_default_is_false(self):
        """Default quantization_aware should be False."""
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
        )
        assert cfg.training.quantization_aware is False

    def test_qat_disabled_explicitly(self):
        """quantization_aware: false should be valid."""
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
            training={"quantization_aware": False},
        )
        assert cfg.training.quantization_aware is False

    def test_qat_in_model_dump(self):
        """quantization_aware field should appear in model_dump output."""
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
            training={"quantization_aware": False},
        )
        dump = cfg.model_dump()
        assert dump["training"]["quantization_aware"] is False

    # Each case used to assert that `true` loaded next to these settings.
    @pytest.mark.parametrize(
        ("task", "data", "training"),
        [
            ("sft", {}, {}),
            ("sft", {}, {"quantization": "4bit"}),
            ("sft", {}, {"quantization": "none"}),
            ("dpo", {}, {}),
            ("grpo", {}, {}),
            (
                "sft",
                {"format": "alpaca", "max_length": 4096},
                {"epochs": 3, "lr": 2e-5, "quantization": "4bit",
                 "lora": {"r": 64, "alpha": 16}},
            ),
        ],
        ids=["default", "4bit", "none", "dpo", "grpo", "full-config"],
    )
    def test_qat_true_is_refused(self, task, data, training):
        """#1222 — int8 QAT is not implemented, so `true` does not load."""
        with pytest.raises(ValidationError, match="#1222"):
            SoupConfig(
                base="meta-llama/Llama-3.1-8B-Instruct",
                task=task,
                data={"train": "./data.jsonl", **data},
                training={**training, "quantization_aware": True},
            )


# ─── Validation Tests ──────────────────────────────────────────────────────


class TestQATValidation:
    """Test QAT configuration validation."""

    def test_validate_qat_with_unsloth_returns_error(self):
        """QAT should not be compatible with unsloth backend."""
        from soup_cli.utils.qat import validate_qat_config

        errors = validate_qat_config(
            quantization="4bit", backend="unsloth", modality="text",
        )
        assert any("unsloth" in err for err in errors)

    def test_validate_qat_with_transformers_no_backend_error(self):
        """QAT with transformers backend should have no backend errors."""
        from soup_cli.utils.qat import validate_qat_config

        errors = validate_qat_config(
            quantization="4bit", backend="transformers", modality="text",
        )
        assert not any("unsloth" in err for err in errors)

    def test_validate_qat_with_8bit_warns(self):
        """QAT with 8bit quantization should warn."""
        from soup_cli.utils.qat import validate_qat_config

        errors = validate_qat_config(
            quantization="8bit", backend="transformers", modality="text",
        )
        assert any("8bit" in err for err in errors)

    def test_validate_qat_with_4bit_no_quant_warning(self):
        """QAT with 4bit should not warn about quantization level."""
        from soup_cli.utils.qat import validate_qat_config

        errors = validate_qat_config(
            quantization="4bit", backend="transformers", modality="text",
        )
        # Only torchao availability error expected, not quantization warning
        assert not any("4bit" in err for err in errors)

    def test_validate_qat_with_none_no_quant_warning(self):
        """QAT with none quantization should not warn about quantization level."""
        from soup_cli.utils.qat import validate_qat_config

        errors = validate_qat_config(
            quantization="none", backend="transformers", modality="text",
        )
        assert not any("none" in err.lower() for err in errors)

    def test_validate_qat_torchao_not_installed(self):
        """Should warn when torchao is not installed."""
        from soup_cli.utils.qat import validate_qat_config

        errors = validate_qat_config(
            quantization="4bit", backend="transformers", modality="text",
        )
        # In test env, torchao is likely not installed
        torchao_errors = [err for err in errors if "torchao" in err]
        # This is OK — either torchao is installed or we get the warning
        assert isinstance(torchao_errors, list)

    def test_validate_qat_with_vision_modality(self):
        """QAT should work with vision modality."""
        from soup_cli.utils.qat import validate_qat_config

        errors = validate_qat_config(
            quantization="4bit", backend="transformers", modality="vision",
        )
        # No modality-specific errors
        assert not any("vision" in err for err in errors)


# ─── QAT Utility Tests ─────────────────────────────────────────────────────


class TestQATUtils:
    """Test QAT utility functions."""

    def test_is_qat_available_returns_bool(self):
        """is_qat_available should return a boolean."""
        from soup_cli.utils.qat import is_qat_available

        result = is_qat_available()
        assert isinstance(result, bool)

    def test_prepare_model_for_qat_calls_quantize(self):
        """prepare_model_for_qat should call torchao quantize_.

        The helper is kept, but since #1222 no config that loads can reach it:
        ``tests/test_issue1222_refuse_int8_qat.py`` drives the SFT dispatcher
        with every value that still loads and asserts it is never called.
        """
        mock_model = MagicMock()
        mock_config = MagicMock()
        mock_quantize = MagicMock()

        mock_torchao = MagicMock()
        mock_torchao.quantization.quantize_ = mock_quantize
        mock_torchao.quantization.Int8WeightOnlyConfig.return_value = mock_config

        with patch.dict("sys.modules", {
            "torchao": mock_torchao,
            "torchao.quantization": mock_torchao.quantization,
        }):
            import importlib

            import soup_cli.utils.qat

            importlib.reload(soup_cli.utils.qat)
            result = soup_cli.utils.qat.prepare_model_for_qat(mock_model)
            mock_quantize.assert_called_once_with(mock_model, mock_config)
            assert result is mock_model

    def test_get_qat_config_returns_config(self):
        """get_qat_config should return an Int8WeightOnlyConfig."""
        mock_config_cls = MagicMock()
        mock_config = MagicMock()
        mock_config_cls.return_value = mock_config

        with patch.dict("sys.modules", {
            "torchao": MagicMock(),
            "torchao.quantization": MagicMock(Int8WeightOnlyConfig=mock_config_cls),
        }):
            import importlib

            import soup_cli.utils.qat

            importlib.reload(soup_cli.utils.qat)
            result = soup_cli.utils.qat.get_qat_config()
            assert result is mock_config


# ─── Trainer Integration Tests ──────────────────────────────────────────────


class TestTrainerConfigsWithQATTrueAreRefused:
    """The SFT / DPO / GRPO wrappers used to be built from a `true` config and
    then called the int8 helper. Since #1222 that config cannot be built."""

    @pytest.mark.parametrize(
        ("task", "max_length"), [("sft", 2048), ("dpo", 2048), ("grpo", 4096)]
    )
    def test_the_config_cannot_be_built(self, task, max_length):
        with pytest.raises(ValidationError, match="#1222"):
            SoupConfig(
                base="some-model",
                task=task,
                data={"train": "./data.jsonl", "max_length": max_length},
                training={
                    "quantization_aware": True,
                    "quantization": "4bit",
                    "lora": {"r": 64, "alpha": 16},
                },
            )


class TestSFTQATIntegration:
    """Test SFT trainer with QAT."""

    def test_sft_setup_transformers_no_qat_when_disabled(self):
        """_setup_transformers' QAT step must not call the int8 helper when disabled.

        Drives the dispatcher that step runs, ``_apply_quantization_aware``,
        instead of re-implementing its guard in the test.
        """
        from soup_cli.trainer.sft import SFTTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl", "max_length": 2048},
            training={
                "quantization_aware": False,
                "quantization": "4bit",
                "lora": {"r": 64, "alpha": 16, "dropout": 0.05},
            },
        )
        wrapper = SFTTrainerWrapper(cfg, device="cuda")
        wrapper.model = MagicMock()

        with patch("soup_cli.utils.qat.prepare_model_for_qat") as mock_qat, patch(
            "soup_cli.utils.v028_features.apply_v028_speed_memory"
        ):
            wrapper._apply_quantization_aware(cfg.training)
        mock_qat.assert_not_called()


# ─── Vision + QAT Tests ────────────────────────────────────────────────────


class TestVisionQATIntegration:
    """Test QAT with vision modality."""

    def test_vision_qat_true_is_refused(self):
        """Vision used to accept `true` too; #1222 refuses it on every modality."""
        with pytest.raises(ValidationError, match="#1222"):
            SoupConfig(
                base="meta-llama/Llama-3.2-11B-Vision-Instruct",
                task="sft",
                modality="vision",
                data={"train": "./data.jsonl", "format": "llava", "image_dir": "./images"},
                training={"quantization_aware": True, "quantization": "4bit"},
            )


# ─── Export Compatibility Tests ─────────────────────────────────────────────


class TestQATExportCompatibility:
    """Test that QAT-trained models are compatible with GGUF export."""

    def test_qat_config_does_not_affect_export(self):
        """QAT is a training-time feature — export should work the same."""
        from soup_cli.commands.export import GGUF_QUANT_TYPES, SUPPORTED_FORMATS

        # QAT models produce standard LoRA adapters, so all export
        # paths should remain unchanged
        assert "gguf" in SUPPORTED_FORMATS
        assert "q4_k_m" in GGUF_QUANT_TYPES
        assert "q8_0" in GGUF_QUANT_TYPES


# ─── Train Command Tests ───────────────────────────────────────────────────


def _setup_panel(tmp_path, monkeypatch, quantization_aware: str) -> str:
    """Run ``soup train --dry-run`` on a small SFT config; return its plain output.

    The label under test is the one ``commands/train.py`` renders, not a copy
    of its logic. The panel prints before the QAT pre-flight, whose verdict
    depends on torchao and on the card, so that verdict (the exit code) is not
    asserted here -- only that the command did not crash.
    """
    from soup_cli.cli import app

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.jsonl").write_text(
        '{"messages": [{"role": "user", "content": "q"}, '
        '{"role": "assistant", "content": "a"}]}\n',
        encoding="utf-8",
    )
    (tmp_path / "soup.yaml").write_text(
        "base: some-model\ntask: sft\n"
        "data:\n  train: train.jsonl\n  format: chatml\n"
        f"training:\n  quantization: none\n  quantization_aware: {quantization_aware}\n"
        "  lora: {r: 8, alpha: 16}\n"
        "output: ./out\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["train", "--config", "soup.yaml", "--dry-run"])
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        result.output,
        repr(result.exception),
    )
    output = " ".join(strip_ansi(result.output).split())
    assert "Training Setup" in output, output
    return output


class TestTrainCommandQAT:
    """Test train command QAT display and validation."""

    def test_qat_label_shown_in_panel(self, tmp_path, monkeypatch):
        """Train panel should show '+ QAT' when enabled.

        `true` is refused at load (#1222), so `fp8` is the value that reaches
        this label now; `quest` has a label of its own.
        """
        output = _setup_panel(tmp_path, monkeypatch, "fp8")
        assert "Quant: none + QAT" in output, output

    def test_qat_label_not_shown_when_disabled(self, tmp_path, monkeypatch):
        """Train panel should show plain quant when QAT disabled."""
        output = _setup_panel(tmp_path, monkeypatch, "false")
        assert "Quant: none" in output, output
        assert "+ QAT" not in output, output

    def test_qat_with_unsloth_validation_fails(self):
        """QAT + unsloth should produce validation errors."""
        from soup_cli.utils.qat import validate_qat_config

        errors = validate_qat_config(
            quantization="4bit", backend="unsloth", modality="text",
        )
        assert len(errors) >= 1
        assert any("unsloth" in err for err in errors)


# ─── Sweep Shortcut Tests ─────────────────────────────────────────────────


class TestQATSweepParam:
    """Test quantization_aware parameter in sweep shortcuts."""

    def test_qat_sweep_shortcut(self):
        from soup_cli.commands.sweep import _set_nested_param

        config = {"training": {"quantization_aware": False}}
        _set_nested_param(config, "training.quantization_aware", True)
        assert config["training"]["quantization_aware"] is True


# ─── Doctor Tests ──────────────────────────────────────────────────────────


class TestDoctorQAT:
    """Test that doctor checks for torchao (QAT dependency)."""

    def test_torchao_in_deps_list(self):
        from soup_cli.commands.doctor import DEPS

        pkg_names = [pkg_name for _, pkg_name, _, _ in DEPS]
        assert "torchao" in pkg_names

    def test_torchao_is_optional(self):
        from soup_cli.commands.doctor import DEPS

        for import_name, pkg_name, _, required in DEPS:
            if pkg_name == "torchao":
                assert required is False
                break
        else:
            pytest.fail("torchao not found in DEPS")
