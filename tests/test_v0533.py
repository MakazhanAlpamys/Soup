"""v0.53.3 — GRPO Plus partial wiring (#128 grpo_fp16, #129 vision-VLM probe).

This release lifts two of the six v0.50.0 deferred-stub items planned for
v0.53.3; the four larger items (#127 stability callback, #123 GRPO variant
losses, #126 PRMTrainerWrapper, #68 multi-objective preference live combine)
are deferred to v0.53.4 (each warrants a focused release).
"""

from __future__ import annotations

import re
import textwrap

import pytest
import yaml
from pydantic import ValidationError

from soup_cli.config.loader import load_config_from_string
from soup_cli.utils.prm import (
    KNOWN_VLM_REGEX,
    is_known_vlm_base,
    validate_vision_grpo_compat,
)

# ---------------------------------------------------------------------------
# #128 grpo_fp16 routing
# ---------------------------------------------------------------------------


def _minimal_grpo_yaml(**overrides: object) -> str:
    base = {
        "base": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "task": "grpo",
        "data": {"train": "data.jsonl", "format": "alpaca"},
        "training": {"reward_fn": "accuracy", "num_generations": 2},
    }
    base.update(overrides)  # type: ignore[arg-type]
    return yaml.safe_dump(base)


class TestGRPOFP16:
    def test_grpo_fp16_default_off(self) -> None:
        cfg = load_config_from_string(_minimal_grpo_yaml())
        assert cfg.training.grpo_fp16 is False

    def test_grpo_fp16_true_accepted_on_grpo(self) -> None:
        cfg = load_config_from_string(_minimal_grpo_yaml(
            training={
                "reward_fn": "accuracy",
                "num_generations": 2,
                "grpo_fp16": True,
            },
        ))
        assert cfg.training.grpo_fp16 is True

    def test_grpo_fp16_with_auto_mixed_precision_rejected(self) -> None:
        """v0.53.3 #128 — cross-validator rejects the combo (pick one)."""
        yml = _minimal_grpo_yaml(
            training={
                "reward_fn": "accuracy",
                "num_generations": 2,
                "grpo_fp16": True,
                "auto_mixed_precision": True,
            },
        )
        with pytest.raises((ValidationError, ValueError)) as exc:
            load_config_from_string(yml)
        msg = str(exc.value)
        assert "grpo_fp16" in msg
        assert "auto_mixed_precision" in msg

    def test_grpo_fp16_alone_with_amp_off_passes(self) -> None:
        cfg = load_config_from_string(_minimal_grpo_yaml(
            training={
                "reward_fn": "accuracy",
                "num_generations": 2,
                "grpo_fp16": True,
                "auto_mixed_precision": False,
            },
        ))
        assert cfg.training.grpo_fp16 is True
        assert cfg.training.auto_mixed_precision is False

    def test_amp_alone_passes(self) -> None:
        cfg = load_config_from_string(_minimal_grpo_yaml(
            training={
                "reward_fn": "accuracy",
                "num_generations": 2,
                "auto_mixed_precision": True,
            },
        ))
        assert cfg.training.grpo_fp16 is False
        assert cfg.training.auto_mixed_precision is True


class TestGRPOFP16Routing:
    def test_grpo_wrapper_sets_fp16_true_bf16_false_when_grpo_fp16(self) -> None:
        """v0.53.3 #128 — wrapper translates grpo_fp16=True into HF kwargs."""
        from soup_cli.trainer.grpo import GRPOTrainerWrapper

        cfg = load_config_from_string(_minimal_grpo_yaml(
            training={
                "reward_fn": "accuracy",
                "num_generations": 2,
                "grpo_fp16": True,
            },
        ))
        wrapper = GRPOTrainerWrapper(cfg, device="cuda")
        kwargs = wrapper._build_precision_kwargs()
        assert kwargs == {"fp16": True, "bf16": False}

    def test_grpo_wrapper_default_bf16_when_no_flag(self) -> None:
        from soup_cli.trainer.grpo import GRPOTrainerWrapper

        cfg = load_config_from_string(_minimal_grpo_yaml())
        wrapper = GRPOTrainerWrapper(cfg, device="cuda")
        kwargs = wrapper._build_precision_kwargs()
        assert kwargs == {"fp16": False, "bf16": True}

    def test_grpo_wrapper_cpu_no_precision(self) -> None:
        from soup_cli.trainer.grpo import GRPOTrainerWrapper

        cfg = load_config_from_string(_minimal_grpo_yaml(
            training={
                "reward_fn": "accuracy",
                "num_generations": 2,
                "grpo_fp16": True,
            },
        ))
        wrapper = GRPOTrainerWrapper(cfg, device="cpu")
        kwargs = wrapper._build_precision_kwargs()
        # CPU path: never enable mixed precision.
        assert kwargs == {"fp16": False, "bf16": False}


# ---------------------------------------------------------------------------
# #129 vision-GRPO base-model probe
# ---------------------------------------------------------------------------


class TestKnownVLMRegex:
    def test_regex_is_compiled(self) -> None:
        assert isinstance(KNOWN_VLM_REGEX, re.Pattern)

    @pytest.mark.parametrize(
        "name",
        [
            "mistralai/Pixtral-12B-2409",
            "Qwen/Qwen2-VL-7B-Instruct",
            "Qwen/Qwen2.5-VL-72B-Instruct",
            "OpenGVLab/InternVL3-8B",
            "OpenGVLab/InternVL2_5-38B",
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
            "llava-hf/llava-1.5-7b-hf",
            "llava-hf/llava-v1.6-mistral-7b-hf",
            "openbmb/MiniCPM-V-2_6",
        ],
    )
    def test_known_vlm_accepted(self, name: str) -> None:
        assert is_known_vlm_base(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "meta-llama/Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "mistralai/Mistral-7B-v0.1",
            "google/gemma-2-9b",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        ],
    )
    def test_non_vlm_rejected(self, name: str) -> None:
        assert is_known_vlm_base(name) is False

    def test_empty_returns_false(self) -> None:
        assert is_known_vlm_base("") is False

    def test_none_returns_false(self) -> None:
        assert is_known_vlm_base(None) is False  # type: ignore[arg-type]

    def test_non_string_returns_false(self) -> None:
        assert is_known_vlm_base(123) is False  # type: ignore[arg-type]
        assert is_known_vlm_base(True) is False  # type: ignore[arg-type]

    def test_null_byte_returns_false(self) -> None:
        assert is_known_vlm_base("Qwen/Qwen2-VL\x00-7B") is False

    def test_oversize_returns_false(self) -> None:
        assert is_known_vlm_base("a" * 600) is False


class TestValidateVisionGRPOCompatWithBase:
    def test_known_vlm_passes(self) -> None:
        # Should NOT raise.
        validate_vision_grpo_compat(
            task="grpo",
            modality="vision",
            backend="transformers",
            base="Qwen/Qwen2-VL-7B-Instruct",
        )

    def test_known_vlm_ppo_passes(self) -> None:
        validate_vision_grpo_compat(
            task="ppo",
            modality="vision",
            backend="transformers",
            base="llava-hf/llava-1.5-7b-hf",
        )

    def test_unknown_base_rejected_with_actionable_message(self) -> None:
        with pytest.raises(ValueError) as exc:
            validate_vision_grpo_compat(
                task="grpo",
                modality="vision",
                backend="transformers",
                base="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            )
        msg = str(exc.value)
        assert "vision_grpo" in msg
        # Names at least one known VLM family so the user has a fix path.
        assert "Qwen2-VL" in msg or "Pixtral" in msg or "LLaVA" in msg

    def test_base_none_skips_probe(self) -> None:
        """Backwards-compatible default — old callers pass no `base`."""
        validate_vision_grpo_compat(
            task="grpo",
            modality="vision",
            backend="transformers",
        )

    def test_base_empty_string_skips_probe(self) -> None:
        validate_vision_grpo_compat(
            task="grpo",
            modality="vision",
            backend="transformers",
            base="",
        )

    def test_pre_existing_rejections_still_fire(self) -> None:
        """Adding `base` must not regress the v0.50.0 Part E rejections."""
        with pytest.raises(ValueError, match="task in"):
            validate_vision_grpo_compat(
                task="sft", modality="vision", backend="transformers",
                base="Qwen/Qwen2-VL-7B-Instruct",
            )
        with pytest.raises(ValueError, match="modality"):
            validate_vision_grpo_compat(
                task="grpo", modality="text", backend="transformers",
                base="Qwen/Qwen2-VL-7B-Instruct",
            )
        with pytest.raises(ValueError, match="mlx"):
            validate_vision_grpo_compat(
                task="grpo", modality="vision", backend="mlx",
                base="Qwen/Qwen2-VL-7B-Instruct",
            )


class TestSchemaIntegrationVisionGRPO:
    def test_vision_grpo_with_non_vlm_base_rejected_at_schema(self) -> None:
        """End-to-end — SoupConfig load surfaces the #129 base-probe rejection."""
        yml = textwrap.dedent("""
            base: TinyLlama/TinyLlama-1.1B-Chat-v1.0
            task: grpo
            modality: vision
            data:
              train: d.jsonl
              format: llava
            training:
              vision_grpo: true
              reward_fn: accuracy
              num_generations: 2
        """).strip()
        with pytest.raises((ValidationError, ValueError)) as exc:
            load_config_from_string(yml)
        assert "vision_grpo" in str(exc.value)

    def test_vision_grpo_with_known_vlm_passes(self) -> None:
        yml = textwrap.dedent("""
            base: Qwen/Qwen2-VL-7B-Instruct
            task: grpo
            modality: vision
            data:
              train: d.jsonl
              format: llava
            training:
              vision_grpo: true
              reward_fn: accuracy
              num_generations: 2
        """).strip()
        cfg = load_config_from_string(yml)
        assert cfg.training.vision_grpo is True


# ---------------------------------------------------------------------------
# v0.53.3 review-fix coverage gaps (tdd-review HIGH/MEDIUM/LOW)
# ---------------------------------------------------------------------------


class TestReviewFixes:
    def test_grpo_fp16_on_non_grpo_rejected_via_task_gate(self) -> None:
        """tdd-review HIGH — `grpo_fp16=True` on `task='sft'` is rejected by
        the v0.50.0 stability task-gate before the v0.53.3 mutex validator
        fires. Confirms the new mutex validator's task short-circuit
        delegates to the task-gate error path."""
        yml = textwrap.dedent("""
            base: TinyLlama/TinyLlama-1.1B-Chat-v1.0
            task: sft
            data:
              train: d.jsonl
              format: alpaca
            training:
              grpo_fp16: true
        """).strip()
        with pytest.raises((ValidationError, ValueError)) as exc:
            load_config_from_string(yml)
        # Task-gate diagnosis takes priority over the mutex one.
        assert "grpo_fp16" in str(exc.value)
        assert "grpo" in str(exc.value).lower()

    def test_qvq_known_vlm(self) -> None:
        """tdd-review MEDIUM — `_VLM_PATTERNS` includes QVQ; ensure it is
        actually matched."""
        assert is_known_vlm_base("Qwen/QVQ-72B-Preview") is True

    def test_llama32_vision_variants(self) -> None:
        """tdd-review MEDIUM — Llama-3.2-Vision regex variant coverage."""
        assert is_known_vlm_base("Llama3.2-Vision-Instruct") is True
        assert is_known_vlm_base("meta-llama/llama-3.2-90b-vision") is True

    def test_is_known_vlm_base_exact_boundary_512(self) -> None:
        """tdd-review LOW — exact upper boundary returns False (no match)."""
        # 512-char string with no VLM substring — should be False (no
        # match), not rejected by oversize cap.
        assert is_known_vlm_base("a" * 512) is False
        # 512-char string ending with a VLM token should match.
        prefix = "a" * (512 - len("-pixtral"))
        assert is_known_vlm_base(prefix + "-pixtral") is True

    def test_precision_kwargs_explicit_false(self) -> None:
        """tdd-review LOW — explicit `grpo_fp16: false` produces bf16 path."""
        from soup_cli.trainer.grpo import GRPOTrainerWrapper

        cfg = load_config_from_string(_minimal_grpo_yaml(
            training={
                "reward_fn": "accuracy",
                "num_generations": 2,
                "grpo_fp16": False,
            },
        ))
        wrapper = GRPOTrainerWrapper(cfg, device="cuda")
        assert wrapper._build_precision_kwargs() == {"fp16": False, "bf16": True}

    def test_precision_kwargs_mps_device(self) -> None:
        """tdd-review HIGH — non-CUDA / non-CPU device (MPS) returns no-MP."""
        from soup_cli.trainer.grpo import GRPOTrainerWrapper

        cfg = load_config_from_string(_minimal_grpo_yaml())
        wrapper = GRPOTrainerWrapper(cfg, device="mps")
        assert wrapper._build_precision_kwargs() == {"fp16": False, "bf16": False}

    def test_error_message_truncates_oversize_base(self) -> None:
        """security-review MEDIUM — long bases truncated in error message."""
        long_base = "a" * 300 + "not-a-vlm"  # 309 chars, no match
        with pytest.raises(ValueError) as exc:
            validate_vision_grpo_compat(
                task="grpo",
                modality="vision",
                backend="transformers",
                base=long_base,
            )
        # Truncated form contains the ellipsis marker; full input does not
        # appear verbatim in the message.
        assert "..." in str(exc.value)
        assert long_base not in str(exc.value)
