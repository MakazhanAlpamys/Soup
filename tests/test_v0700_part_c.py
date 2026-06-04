"""v0.70.0 Part C — MiniLLM reverse-KL on-policy distillation.

MiniLLM (Gu et al. 2024) bundles 3 stability tricks:
1. Teacher-mixed sampling (epsilon-greedy mix of teacher / student rollouts)
2. Length normalisation on rollout completions
3. Pretrain-loss anchor (add a small SFT-on-pretrain term to prevent drift)

Schema-only release; live trainer wiring deferred to v0.70.1.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest


class TestMiniLLMPublicSurface:
    def test_module_imports(self):
        from soup_cli.utils import minillm

        assert hasattr(minillm, "MiniLLMConfig")
        assert hasattr(minillm, "validate_teacher_mix_ratio")
        assert hasattr(minillm, "validate_pretrain_anchor_weight")
        assert hasattr(minillm, "build_minillm_callback")


class TestValidateTeacherMixRatio:
    """Teacher mix ratio in [0, 1]. 0 = student-only; 1 = teacher-only."""

    def test_happy_boundary_zero(self):
        from soup_cli.utils.minillm import validate_teacher_mix_ratio

        assert validate_teacher_mix_ratio(0.0) == 0.0

    def test_happy_boundary_one(self):
        from soup_cli.utils.minillm import validate_teacher_mix_ratio

        assert validate_teacher_mix_ratio(1.0) == 1.0

    def test_happy_mid(self):
        from soup_cli.utils.minillm import validate_teacher_mix_ratio

        assert validate_teacher_mix_ratio(0.3) == 0.3

    def test_above_one_rejected(self):
        from soup_cli.utils.minillm import validate_teacher_mix_ratio

        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            validate_teacher_mix_ratio(1.5)

    def test_negative_rejected(self):
        from soup_cli.utils.minillm import validate_teacher_mix_ratio

        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            validate_teacher_mix_ratio(-0.1)

    def test_nan_rejected(self):
        from soup_cli.utils.minillm import validate_teacher_mix_ratio

        with pytest.raises(ValueError, match="finite"):
            validate_teacher_mix_ratio(float("nan"))

    def test_inf_rejected(self):
        from soup_cli.utils.minillm import validate_teacher_mix_ratio

        with pytest.raises(ValueError, match="finite"):
            validate_teacher_mix_ratio(float("inf"))

    def test_bool_rejected(self):
        from soup_cli.utils.minillm import validate_teacher_mix_ratio

        with pytest.raises(ValueError, match="bool"):
            validate_teacher_mix_ratio(True)

    def test_non_number_rejected(self):
        from soup_cli.utils.minillm import validate_teacher_mix_ratio

        with pytest.raises(ValueError, match="number"):
            validate_teacher_mix_ratio("0.5")


class TestValidatePretrainAnchorWeight:
    """Pretrain anchor weight: small non-negative float, bounded [0, 1]."""

    def test_happy_path(self):
        from soup_cli.utils.minillm import validate_pretrain_anchor_weight

        assert validate_pretrain_anchor_weight(0.1) == 0.1

    def test_zero_allowed(self):
        from soup_cli.utils.minillm import validate_pretrain_anchor_weight

        assert validate_pretrain_anchor_weight(0.0) == 0.0

    def test_one_allowed(self):
        from soup_cli.utils.minillm import validate_pretrain_anchor_weight

        assert validate_pretrain_anchor_weight(1.0) == 1.0

    def test_above_one_rejected(self):
        from soup_cli.utils.minillm import validate_pretrain_anchor_weight

        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            validate_pretrain_anchor_weight(1.1)

    def test_negative_rejected(self):
        from soup_cli.utils.minillm import validate_pretrain_anchor_weight

        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            validate_pretrain_anchor_weight(-0.1)

    def test_non_finite_rejected(self):
        from soup_cli.utils.minillm import validate_pretrain_anchor_weight

        with pytest.raises(ValueError, match="finite"):
            validate_pretrain_anchor_weight(float("inf"))

    def test_bool_rejected(self):
        from soup_cli.utils.minillm import validate_pretrain_anchor_weight

        with pytest.raises(ValueError, match="bool"):
            validate_pretrain_anchor_weight(True)


class TestMiniLLMConfig:
    def test_defaults(self):
        from soup_cli.utils.minillm import MiniLLMConfig

        cfg = MiniLLMConfig()
        assert cfg.teacher_mix_ratio == 0.0
        assert cfg.length_normalize is True
        assert cfg.pretrain_anchor_weight == 0.0
        assert cfg.pretrain_anchor_path is None

    def test_basic_config(self):
        from soup_cli.utils.minillm import MiniLLMConfig

        cfg = MiniLLMConfig(
            teacher_mix_ratio=0.3,
            length_normalize=True,
            pretrain_anchor_weight=0.1,
            pretrain_anchor_path="./pretrain.jsonl",
        )
        assert cfg.teacher_mix_ratio == 0.3
        assert cfg.pretrain_anchor_path == "./pretrain.jsonl"

    def test_frozen(self):
        from soup_cli.utils.minillm import MiniLLMConfig

        cfg = MiniLLMConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.teacher_mix_ratio = 0.5  # type: ignore[misc]

    def test_invalid_mix_ratio_propagates(self):
        from soup_cli.utils.minillm import MiniLLMConfig

        with pytest.raises(ValueError):
            MiniLLMConfig(teacher_mix_ratio=2.0)

    def test_invalid_anchor_weight_propagates(self):
        from soup_cli.utils.minillm import MiniLLMConfig

        with pytest.raises(ValueError):
            MiniLLMConfig(pretrain_anchor_weight=-0.1)

    def test_anchor_weight_without_path_rejected(self):
        """If anchor_weight > 0, pretrain_anchor_path is required."""
        from soup_cli.utils.minillm import MiniLLMConfig

        with pytest.raises(ValueError, match="pretrain_anchor_path"):
            MiniLLMConfig(
                pretrain_anchor_weight=0.1,
                pretrain_anchor_path=None,
            )

    def test_anchor_path_without_weight_rejected(self):
        """If path is set but weight=0, silent no-op — reject."""
        from soup_cli.utils.minillm import MiniLLMConfig

        with pytest.raises(ValueError, match="pretrain_anchor_weight"):
            MiniLLMConfig(
                pretrain_anchor_weight=0.0,
                pretrain_anchor_path="./pretrain.jsonl",
            )

    def test_length_normalize_must_be_bool(self):
        from soup_cli.utils.minillm import MiniLLMConfig

        with pytest.raises(TypeError, match="bool"):
            MiniLLMConfig(length_normalize="yes")  # type: ignore[arg-type]

    def test_anchor_path_null_byte_rejected(self):
        from soup_cli.utils.minillm import MiniLLMConfig

        with pytest.raises(ValueError, match="null byte"):
            MiniLLMConfig(
                pretrain_anchor_weight=0.1,
                pretrain_anchor_path="./bad\x00",
            )

    def test_anchor_path_oversize_rejected(self):
        from soup_cli.utils.minillm import MiniLLMConfig

        with pytest.raises(ValueError, match="exceeds"):
            MiniLLMConfig(
                pretrain_anchor_weight=0.1,
                pretrain_anchor_path="./" + "x" * 5000,
            )


class TestBuildMiniLLMCallback:
    """Live in v0.71.11 #237 — returns a MiniLLMCallback; validates type."""

    def test_non_config_rejected(self):
        from soup_cli.utils.minillm import build_minillm_callback

        with pytest.raises(TypeError, match="MiniLLMConfig"):
            build_minillm_callback({})  # type: ignore[arg-type]

    def test_live_returns_callback(self):
        from soup_cli.utils.minillm import (
            MiniLLMCallback,
            MiniLLMConfig,
            build_minillm_callback,
        )

        assert isinstance(build_minillm_callback(MiniLLMConfig()), MiniLLMCallback)


# ---------------------------------------------------------------------------
# Schema integration — TrainingConfig + SoupConfig
# ---------------------------------------------------------------------------


class TestSchemaTrainingConfig:
    def test_defaults(self):
        from soup_cli.config.schema import TrainingConfig

        tcfg = TrainingConfig()
        assert tcfg.minillm_enabled is False
        assert tcfg.minillm_teacher_mix_ratio == 0.0
        assert tcfg.minillm_length_normalize is True
        assert tcfg.minillm_pretrain_anchor_weight == 0.0
        assert tcfg.minillm_pretrain_anchor_path is None

    def test_enabled_with_all_defaults(self):
        from soup_cli.config.schema import TrainingConfig

        tcfg = TrainingConfig(minillm_enabled=True)
        assert tcfg.minillm_enabled is True

    def test_invalid_mix_ratio_rejected(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError):
            TrainingConfig(minillm_teacher_mix_ratio=2.0)

    def test_invalid_anchor_weight_rejected(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError):
            TrainingConfig(minillm_pretrain_anchor_weight=1.5)


class TestSchemaSoupConfigTaskGate:
    """minillm_enabled only meaningful when task='distill'."""

    def _yaml(self, task: str = "distill", **extras: object) -> str:
        teacher_line = (
            "  teacher_model: meta-llama/Llama-3.1-8B\n"
            if task == "distill" else ""
        )
        extra_lines = "".join(f"  {k}: {v}\n" for k, v in extras.items())
        return f"""
base: meta-llama/Llama-3.1-8B
task: {task}
data:
  train: ./data/train.jsonl
  format: chatml
training:
{teacher_line}{extra_lines}"""

    def test_distill_minillm_accepted(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(
            self._yaml(task="distill", minillm_enabled=True)
        )
        assert cfg.training.minillm_enabled is True

    def test_sft_minillm_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="minillm"):
            load_config_from_string(
                self._yaml(task="sft", minillm_enabled=True)
            )

    def test_mlx_minillm_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError):
            load_config_from_string(
                """
base: mlx-community/Llama-3.1-8B
task: distill
backend: mlx
data:
  train: ./data/train.jsonl
  format: chatml
training:
  teacher_model: meta-llama/Llama-3.1-8B
  minillm_enabled: true
"""
            )

    def test_anchor_weight_without_path_rejected_at_schema(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="minillm_pretrain_anchor_path"):
            load_config_from_string(
                self._yaml(
                    task="distill",
                    minillm_enabled=True,
                    minillm_pretrain_anchor_weight=0.1,
                )
            )

    def test_anchor_path_without_weight_rejected_at_schema(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="minillm_pretrain_anchor_weight"):
            load_config_from_string(
                self._yaml(
                    task="distill",
                    minillm_enabled=True,
                    minillm_pretrain_anchor_path="./pre.jsonl",
                )
            )

    def test_minillm_fields_without_enabled_rejected(self):
        """Setting tunables without minillm_enabled=True is a silent no-op."""
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="minillm_enabled"):
            load_config_from_string(
                self._yaml(
                    task="distill",
                    minillm_teacher_mix_ratio=0.3,
                )
            )


# ---------------------------------------------------------------------------
# Source wiring guards
# ---------------------------------------------------------------------------


class TestSourceWiring:
    def test_module_no_top_level_torch(self):
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent
            / "src" / "soup_cli"
            / "utils"
            / "minillm.py"
        )
        body = src.read_text(encoding="utf-8")
        assert "\nimport torch" not in body
        assert "\nfrom torch" not in body

    def test_math_isfinite_used(self):
        """Anchor-weight + mix-ratio guards must use math.isfinite (not
        the looser ``not nan`` idiom — matches v0.32 / v0.41 / v0.50 / v0.62
        finite-check policy).
        """
        # Importing math at the top of the test triggers the regex.
        _ = math
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent
            / "src" / "soup_cli"
            / "utils"
            / "minillm.py"
        )
        body = src.read_text(encoding="utf-8")
        assert "math.isfinite" in body
