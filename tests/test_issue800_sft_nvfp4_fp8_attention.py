"""Tests for issue #800 — apply nvfp4 and fp8_attention on task: sft and gate tasks.

Validates that:
1. SFTTrainerWrapper wires apply_v028_speed_memory in _apply_quantization_aware,
   correctly dispatching fp8_attention and nvfp4 post-LoRA.
2. Cut-CE is not double-applied or double-logged on SFT when skip_cut_ce=True.
3. SoupConfig rejects nvfp4 and fp8_attention on unsupported tasks (e.g. distill),
   naming the task.
4. SoupConfig rejects nvfp4 and fp8_attention on backend='mlx', naming MLX.
5. warn_unsupported_features reports fp8_attention and nvfp4 on unsupported tasks.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from soup_cli.config.schema import SoupConfig
from soup_cli.trainer.sft import SFTTrainerWrapper
from soup_cli.utils.v028_features import (
    apply_v028_speed_memory,
    warn_unsupported_features,
)


class TestSFTTrainerV028Wiring:
    """Test that SFTTrainerWrapper applies nvfp4 and fp8_attention post-LoRA."""

    def test_sft_applies_fp8_attention(self) -> None:
        cfg = SoupConfig(
            base="test/model",
            task="sft",
            data={"train": "tests/fixtures/sample_train.jsonl"},
            training={"quantization_aware": "fp8", "fp8_attention": True},
        )
        wrapper = SFTTrainerWrapper(config=cfg, device="cpu")
        wrapper.model = MagicMock()

        with patch(
            "soup_cli.utils.advanced_precision.apply_fp8_attention",
            return_value=4,
        ) as mock_fp8_attn, patch(
            "soup_cli.utils.fp8.apply_fp8_training",
            return_value=True,
        ) as mock_fp8_train:
            wrapper._apply_quantization_aware(cfg.training)
            mock_fp8_train.assert_called_once_with(wrapper.model, recipe="tensorwise")
            mock_fp8_attn.assert_called_once_with(wrapper.model, recipe="tensorwise")

    def test_sft_applies_nvfp4(self) -> None:
        cfg = SoupConfig(
            base="test/model",
            task="sft",
            data={"train": "tests/fixtures/sample_train.jsonl"},
            training={"nvfp4": True},
        )
        wrapper = SFTTrainerWrapper(config=cfg, device="cpu")
        wrapper.model = MagicMock()

        with patch(
            "soup_cli.utils.advanced_precision.apply_nvfp4",
            return_value=2,
        ) as mock_nvfp4:
            wrapper._apply_quantization_aware(cfg.training)
            mock_nvfp4.assert_called_once_with(wrapper.model)

    def test_sft_int8_qat_and_v028_speed_memory(self) -> None:
        cfg = SoupConfig(
            base="test/model",
            task="sft",
            data={"train": "tests/fixtures/sample_train.jsonl"},
            training={"quantization_aware": True},
        )
        wrapper = SFTTrainerWrapper(config=cfg, device="cpu")
        wrapper.model = MagicMock()

        with patch(
            "soup_cli.utils.qat.prepare_model_for_qat",
            return_value=wrapper.model,
        ) as mock_qat, patch(
            "soup_cli.utils.v028_features.apply_v028_speed_memory",
            wraps=apply_v028_speed_memory,
        ) as mock_v028:
            wrapper._apply_quantization_aware(cfg.training)
            mock_qat.assert_called_once_with(wrapper.model)
            mock_v028.assert_called_once()
            assert mock_v028.call_args.kwargs["skip_cut_ce"] is True

    def test_sft_skips_duplicate_cut_ce_patch(self) -> None:
        cfg = SoupConfig(
            base="test/model",
            task="sft",
            data={"train": "tests/fixtures/sample_train.jsonl"},
            training={"use_cut_ce": True},
        )
        wrapper = SFTTrainerWrapper(config=cfg, device="cpu")
        wrapper.model = MagicMock()

        with patch(
            "soup_cli.utils.cut_ce.apply_cut_ce",
            return_value=True,
        ) as mock_cut_ce:
            wrapper._apply_quantization_aware(cfg.training)
            mock_cut_ce.assert_not_called()

    def test_apply_v028_speed_memory_skip_cut_ce_behavior(self) -> None:
        tcfg = SimpleNamespace(
            use_cut_ce=True,
            quantization_aware=False,
            kernel_auto_compose=False,
            fp8_attention=False,
            nvfp4=False,
        )
        with patch("soup_cli.utils.cut_ce.apply_cut_ce", return_value=True) as mock_cut_ce:
            res_skipped = apply_v028_speed_memory(
                model=MagicMock(),
                tcfg=tcfg,
                base_model="test/model",
                skip_cut_ce=True,
            )
            mock_cut_ce.assert_not_called()
            assert res_skipped["cut_ce"] is True

            res_applied = apply_v028_speed_memory(
                model=MagicMock(),
                tcfg=tcfg,
                base_model="test/model",
                skip_cut_ce=False,
            )
            mock_cut_ce.assert_called_once_with("test/model")
            assert res_applied["cut_ce"] is True


class TestSchemaV028Gating:
    """Test schema validation for nvfp4 and fp8_attention task gating."""

    def test_distill_rejects_nvfp4(self) -> None:
        pattern = r"v0\.28\.0 features \['nvfp4'\] are not wired for task='distill'"
        with pytest.raises(ValueError, match=pattern):
            SoupConfig(
                base="test/model",
                task="distill",
                data={"train": "tests/fixtures/sample_train.jsonl"},
                training={"nvfp4": True},
            )

    def test_distill_rejects_fp8_attention(self) -> None:
        with pytest.raises(ValueError, match=r"fp8_attention.*not wired for task='distill'"):
            SoupConfig(
                base="test/model",
                task="distill",
                data={"train": "tests/fixtures/sample_train.jsonl"},
                training={"quantization_aware": "fp8", "fp8_attention": True},
            )

    def test_mlx_backend_rejects_nvfp4(self) -> None:
        with pytest.raises(ValueError, match=r"requires Blackwell"):
            SoupConfig(
                base="test/model",
                task="sft",
                backend="mlx",
                data={"train": "tests/fixtures/sample_train.jsonl"},
                training={"nvfp4": True},
            )

    def test_mlx_backend_rejects_fp8_attention_without_quantization_aware(self) -> None:
        with pytest.raises(ValueError, match=r"requires training\.quantization_aware"):
            SoupConfig(
                base="test/model",
                task="sft",
                backend="mlx",
                data={"train": "tests/fixtures/sample_train.jsonl"},
                training={"fp8_attention": True},
            )

    def test_mlx_backend_rejects_fp8_attention(self) -> None:
        with pytest.raises(ValueError, match=r"Apple Silicon mlx backend"):
            SoupConfig(
                base="test/model",
                task="sft",
                backend="mlx",
                data={"train": "tests/fixtures/sample_train.jsonl"},
                training={"quantization_aware": "fp8", "fp8_attention": True},
            )

    @pytest.mark.parametrize("task", ("sft", "dpo", "pretrain", "grpo"))
    def test_supported_tasks_accept_nvfp4(self, task: str) -> None:
        if task in ("dpo", "grpo"):
            format_name = "dpo"
        elif task == "pretrain":
            format_name = "plaintext"
        else:
            format_name = "alpaca"
        cfg = SoupConfig(
            base="test/model",
            task=task,
            data={"train": "tests/fixtures/sample_train.jsonl", "format": format_name},
            training={"nvfp4": True},
        )
        assert cfg.training.nvfp4 is True


class TestWarnUnsupportedFeatures:
    """Test warn_unsupported_features includes fp8_attention and nvfp4."""

    def test_warn_unsupported_includes_fp8_attention_and_nvfp4(self) -> None:
        tcfg = SimpleNamespace(
            use_cut_ce=False,
            quantization_aware=False,
            kernel_auto_compose=False,
            activation_offloading=None,
            fp8_attention=True,
            nvfp4=True,
        )
        msg = warn_unsupported_features(tcfg, "distill")
        assert msg is not None
        assert "fp8_attention" in msg
        assert "nvfp4" in msg

    def test_warn_unsupported_returns_none_for_sft(self) -> None:
        tcfg = SimpleNamespace(
            use_cut_ce=True,
            quantization_aware="fp8",
            kernel_auto_compose=False,
            activation_offloading=None,
            fp8_attention=True,
            nvfp4=True,
        )
        assert warn_unsupported_features(tcfg, "sft") is None


class TestSFTSetupQuantizationAwareIntegration:
    """Kill mutation M5: ensure SFTTrainerWrapper.setup() calls apply_v028_speed_memory."""

    def test_sft_setup_applies_v028_speed_memory_post_lora(self, tmp_path) -> None:
        for module in ("torch", "transformers", "peft", "trl", "datasets"):
            pytest.importorskip(module, reason=f"{module} is only in the [train] extra")

        from tests.test_issue804_chat_template_trainers import _tiny_llama_dir

        weights = _tiny_llama_dir(tmp_path)

        train_file = tmp_path / "train.jsonl"
        train_file.write_text(
            '{"messages": [{"role": "user", "content": "hi"}, '
            '{"role": "assistant", "content": "hello"}]}\n',
            encoding="utf-8",
        )

        cfg = SoupConfig(
            base=str(weights),
            task="sft",
            data={"train": str(train_file)},
            training={
                "quantization_aware": "fp8",
                "fp8_attention": True,
                "nvfp4": True,
                "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
            },
        )

        class StopAfterSetupError(Exception):
            pass

        wrapper = SFTTrainerWrapper(config=cfg, device="cpu")

        with patch(
            "trl.SFTTrainer.__init__",
            side_effect=StopAfterSetupError,
        ), patch(
            "soup_cli.utils.v028_features.apply_v028_speed_memory",
            wraps=apply_v028_speed_memory,
        ) as mock_v028:
            with pytest.raises(StopAfterSetupError):
                wrapper.setup({
                    "train": [
                        {
                            "messages": [
                                {"role": "user", "content": "hi"},
                                {"role": "assistant", "content": "hello"},
                            ]
                        }
                    ]
                })

            mock_v028.assert_called_once()
            assert mock_v028.call_args.kwargs.get("skip_cut_ce") is True
            passed_model = mock_v028.call_args.kwargs.get("model")
            assert passed_model is not None
            assert passed_model.__class__.__name__ == "PeftModelForCausalLM"
