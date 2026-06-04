"""v0.70.0 Part B — Cross-tokenizer distillation (ULD).

Universal Logit Distillation (Boizard et al. 2024) for distilling
across vocab boundaries (e.g. Llama -> Mistral). Schema-only release;
live distill trainer hook deferred to v0.70.1.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest


class TestULDPublicSurface:
    def test_module_imports(self):
        from soup_cli.utils import uld

        assert hasattr(uld, "SUPPORTED_ULD_STRATEGIES")
        assert hasattr(uld, "validate_uld_strategy")
        assert hasattr(uld, "validate_uld_projection_dim")
        assert hasattr(uld, "ULDConfig")
        assert hasattr(uld, "build_uld_projection")

    def test_supported_strategies_frozenset(self):
        from soup_cli.utils.uld import SUPPORTED_ULD_STRATEGIES

        assert isinstance(SUPPORTED_ULD_STRATEGIES, frozenset)
        assert "wasserstein" in SUPPORTED_ULD_STRATEGIES
        assert "topk_align" in SUPPORTED_ULD_STRATEGIES

    def test_supported_strategies_immutable(self):
        from soup_cli.utils.uld import SUPPORTED_ULD_STRATEGIES

        with pytest.raises((AttributeError, TypeError)):
            SUPPORTED_ULD_STRATEGIES.add("evil")


class TestValidateULDStrategy:
    def test_happy_path(self):
        from soup_cli.utils.uld import validate_uld_strategy

        assert validate_uld_strategy("wasserstein") == "wasserstein"
        assert validate_uld_strategy("topk_align") == "topk_align"

    def test_case_insensitive(self):
        from soup_cli.utils.uld import validate_uld_strategy

        assert validate_uld_strategy("WASSERSTEIN") == "wasserstein"
        assert validate_uld_strategy("TopK_Align") == "topk_align"

    def test_bool_rejected(self):
        from soup_cli.utils.uld import validate_uld_strategy

        with pytest.raises(ValueError, match="bool"):
            validate_uld_strategy(True)

    def test_unknown_rejected(self):
        from soup_cli.utils.uld import validate_uld_strategy

        with pytest.raises(ValueError, match="not supported"):
            validate_uld_strategy("evil")

    def test_empty_rejected(self):
        from soup_cli.utils.uld import validate_uld_strategy

        with pytest.raises(ValueError, match="non-empty"):
            validate_uld_strategy("")

    def test_non_string_rejected(self):
        from soup_cli.utils.uld import validate_uld_strategy

        with pytest.raises(ValueError, match="string"):
            validate_uld_strategy(42)

    def test_null_byte_rejected(self):
        from soup_cli.utils.uld import validate_uld_strategy

        with pytest.raises(ValueError, match="null byte"):
            validate_uld_strategy("ws\x00")

    def test_oversize_rejected(self):
        from soup_cli.utils.uld import validate_uld_strategy

        with pytest.raises(ValueError, match="exceeds"):
            validate_uld_strategy("x" * 64)


class TestValidateProjectionDim:
    def test_happy_path(self):
        from soup_cli.utils.uld import validate_uld_projection_dim

        assert validate_uld_projection_dim(128) == 128
        assert validate_uld_projection_dim(32000) == 32000

    def test_minimum_boundary(self):
        from soup_cli.utils.uld import validate_uld_projection_dim

        assert validate_uld_projection_dim(1) == 1

    def test_maximum_boundary(self):
        from soup_cli.utils.uld import validate_uld_projection_dim

        # 262144 — max plausible vocab size (multilingual SentencePiece).
        assert validate_uld_projection_dim(262144) == 262144

    def test_zero_rejected(self):
        from soup_cli.utils.uld import validate_uld_projection_dim

        with pytest.raises(ValueError, match=">= 1"):
            validate_uld_projection_dim(0)

    def test_negative_rejected(self):
        from soup_cli.utils.uld import validate_uld_projection_dim

        with pytest.raises(ValueError, match=">= 1"):
            validate_uld_projection_dim(-5)

    def test_above_cap_rejected(self):
        from soup_cli.utils.uld import validate_uld_projection_dim

        with pytest.raises(ValueError, match="262144"):
            validate_uld_projection_dim(262145)

    def test_bool_rejected(self):
        from soup_cli.utils.uld import validate_uld_projection_dim

        with pytest.raises(ValueError, match="bool"):
            validate_uld_projection_dim(True)

    def test_non_int_rejected(self):
        from soup_cli.utils.uld import validate_uld_projection_dim

        with pytest.raises(ValueError, match="int"):
            validate_uld_projection_dim(128.5)


class TestULDConfig:
    def test_basic(self):
        from soup_cli.utils.uld import ULDConfig

        cfg = ULDConfig(
            strategy="wasserstein",
            student_vocab_size=32000,
            teacher_vocab_size=128256,
        )
        assert cfg.strategy == "wasserstein"
        assert cfg.student_vocab_size == 32000
        assert cfg.teacher_vocab_size == 128256

    def test_frozen(self):
        from soup_cli.utils.uld import ULDConfig

        cfg = ULDConfig(
            strategy="wasserstein",
            student_vocab_size=32000,
            teacher_vocab_size=128256,
        )
        with pytest.raises(FrozenInstanceError):
            cfg.strategy = "topk_align"  # type: ignore[misc]

    def test_invalid_strategy_propagates(self):
        from soup_cli.utils.uld import ULDConfig

        with pytest.raises(ValueError, match="not supported"):
            ULDConfig(
                strategy="evil",
                student_vocab_size=32000,
                teacher_vocab_size=128256,
            )

    def test_invalid_student_vocab(self):
        from soup_cli.utils.uld import ULDConfig

        with pytest.raises(ValueError):
            ULDConfig(
                strategy="wasserstein",
                student_vocab_size=0,
                teacher_vocab_size=128256,
            )

    def test_invalid_teacher_vocab(self):
        from soup_cli.utils.uld import ULDConfig

        with pytest.raises(ValueError):
            ULDConfig(
                strategy="wasserstein",
                student_vocab_size=32000,
                teacher_vocab_size=-1,
            )

    def test_topk_default_optional(self):
        """top_k defaults to None on wasserstein strategy."""
        from soup_cli.utils.uld import ULDConfig

        cfg = ULDConfig(
            strategy="wasserstein",
            student_vocab_size=32000,
            teacher_vocab_size=128256,
        )
        assert cfg.top_k is None

    def test_topk_align_requires_topk(self):
        from soup_cli.utils.uld import ULDConfig

        with pytest.raises(ValueError, match="top_k"):
            ULDConfig(
                strategy="topk_align",
                student_vocab_size=32000,
                teacher_vocab_size=128256,
            )

    def test_topk_align_accepts_topk(self):
        from soup_cli.utils.uld import ULDConfig

        cfg = ULDConfig(
            strategy="topk_align",
            student_vocab_size=32000,
            teacher_vocab_size=128256,
            top_k=128,
        )
        assert cfg.top_k == 128

    def test_topk_on_wasserstein_rejected(self):
        """top_k only makes sense on topk_align."""
        from soup_cli.utils.uld import ULDConfig

        with pytest.raises(ValueError, match="top_k"):
            ULDConfig(
                strategy="wasserstein",
                student_vocab_size=32000,
                teacher_vocab_size=128256,
                top_k=128,
            )


class TestBuildULDProjection:
    """Live in v0.71.11 #236 — returns a ULDProjection; validates type."""

    def test_non_config_rejected(self):
        from soup_cli.utils.uld import build_uld_projection

        with pytest.raises(TypeError, match="ULDConfig"):
            build_uld_projection({"strategy": "wasserstein"})  # type: ignore[arg-type]

    def test_live_returns_projection(self):
        from soup_cli.utils.uld import ULDConfig, ULDProjection, build_uld_projection

        cfg = ULDConfig(
            strategy="wasserstein",
            student_vocab_size=32000,
            teacher_vocab_size=128256,
        )
        assert isinstance(build_uld_projection(cfg), ULDProjection)


# ---------------------------------------------------------------------------
# Schema integration — TrainingConfig + SoupConfig
# ---------------------------------------------------------------------------


class TestSchemaTrainingConfig:
    def test_default_none(self):
        from soup_cli.config.schema import TrainingConfig

        tcfg = TrainingConfig()
        assert tcfg.uld_strategy is None
        assert tcfg.uld_top_k is None

    def test_accept_strategy(self):
        from soup_cli.config.schema import TrainingConfig

        tcfg = TrainingConfig(uld_strategy="wasserstein")
        assert tcfg.uld_strategy == "wasserstein"

    def test_unknown_strategy_rejected(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError):
            TrainingConfig(uld_strategy="evil")

    def test_top_k_bounds(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import TrainingConfig

        # Must be positive
        with pytest.raises(ValidationError):
            TrainingConfig(uld_top_k=0)
        # Allowed.
        tcfg = TrainingConfig(uld_top_k=128)
        assert tcfg.uld_top_k == 128


class TestSchemaSoupConfigTaskGate:
    """uld_strategy only meaningful when task='distill'."""

    def _yaml(
        self,
        task: str = "distill",
        strategy: str = "wasserstein",
        teacher: str = "meta-llama/Llama-3.1-8B",
        top_k: int | None = None,
    ) -> str:
        topk_line = f"  uld_top_k: {top_k}\n" if top_k is not None else ""
        teacher_line = f"  teacher_model: {teacher}\n" if task == "distill" else ""
        return f"""
base: meta-llama/Llama-3.1-8B
task: {task}
data:
  train: ./data/train.jsonl
  format: chatml
training:
{teacher_line}  uld_strategy: {strategy}
{topk_line}"""

    def test_distill_accepted(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(self._yaml(task="distill"))
        assert cfg.training.uld_strategy == "wasserstein"

    def test_sft_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="uld_strategy"):
            load_config_from_string(self._yaml(task="sft"))

    def test_topk_align_requires_topk_at_schema(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="uld_top_k"):
            load_config_from_string(
                self._yaml(strategy="topk_align")
            )

    def test_topk_with_wasserstein_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="uld_top_k"):
            load_config_from_string(self._yaml(top_k=128))

    def test_topk_align_with_topk_accepted(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(
            self._yaml(strategy="topk_align", top_k=128)
        )
        assert cfg.training.uld_strategy == "topk_align"
        assert cfg.training.uld_top_k == 128


class TestSourceWiring:
    def test_module_no_top_level_torch(self):
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent
            / "src" / "soup_cli"
            / "utils"
            / "uld.py"
        )
        body = src.read_text(encoding="utf-8")
        assert "\nimport torch" not in body
        assert "\nfrom torch" not in body
