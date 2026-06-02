"""Tests for v0.62.0 Part E — GRACE codebook (int-running edit).

Discrete latent-space codebook for thousands of sequential edits that
survives lifelong deployments without norm-blowup. Extends the v0.61.0
edit method allowlist with ``grace`` and ships the codebook config
schema + validators.

Schema-only release: live codebook lookup + write kernel land in v0.62.1.
"""

from __future__ import annotations

import dataclasses

import pytest

# ---------- Module surface ----------


class TestModuleSurface:
    def test_imports(self):
        from soup_cli.utils.grace_codebook import (
            MAX_CODEBOOK_DIM,
            MAX_CODEBOOK_SIZE,
            GraceCodebookConfig,
            apply_grace_codebook,
            build_grace_codebook_config,
            validate_grace_codebook_dim,
            validate_grace_codebook_size,
        )
        assert callable(validate_grace_codebook_size)
        assert callable(validate_grace_codebook_dim)
        assert callable(apply_grace_codebook)
        assert callable(build_grace_codebook_config)
        assert dataclasses.is_dataclass(GraceCodebookConfig)
        assert isinstance(MAX_CODEBOOK_SIZE, int)
        assert isinstance(MAX_CODEBOOK_DIM, int)


# ---------- validate_grace_codebook_size ----------


class TestValidateSize:
    def test_happy(self):
        from soup_cli.utils.grace_codebook import validate_grace_codebook_size

        for n in (1, 100, 10_000):
            assert validate_grace_codebook_size(n) == n

    def test_bool_rejected(self):
        from soup_cli.utils.grace_codebook import validate_grace_codebook_size

        with pytest.raises(TypeError):
            validate_grace_codebook_size(True)

    def test_non_int_rejected(self):
        from soup_cli.utils.grace_codebook import validate_grace_codebook_size

        with pytest.raises(TypeError):
            validate_grace_codebook_size(3.14)

    def test_zero_rejected(self):
        from soup_cli.utils.grace_codebook import validate_grace_codebook_size

        with pytest.raises(ValueError):
            validate_grace_codebook_size(0)

    def test_negative_rejected(self):
        from soup_cli.utils.grace_codebook import validate_grace_codebook_size

        with pytest.raises(ValueError):
            validate_grace_codebook_size(-1)

    def test_overcap_rejected(self):
        from soup_cli.utils.grace_codebook import (
            MAX_CODEBOOK_SIZE,
            validate_grace_codebook_size,
        )

        with pytest.raises(ValueError):
            validate_grace_codebook_size(MAX_CODEBOOK_SIZE + 1)


# ---------- validate_grace_codebook_dim ----------


class TestValidateDim:
    def test_happy(self):
        from soup_cli.utils.grace_codebook import validate_grace_codebook_dim

        for d in (8, 768, 4096):
            assert validate_grace_codebook_dim(d) == d

    def test_bool_rejected(self):
        from soup_cli.utils.grace_codebook import validate_grace_codebook_dim

        with pytest.raises(TypeError):
            validate_grace_codebook_dim(True)

    def test_zero_rejected(self):
        from soup_cli.utils.grace_codebook import validate_grace_codebook_dim

        with pytest.raises(ValueError):
            validate_grace_codebook_dim(0)

    def test_overcap_rejected(self):
        from soup_cli.utils.grace_codebook import (
            MAX_CODEBOOK_DIM,
            validate_grace_codebook_dim,
        )

        with pytest.raises(ValueError):
            validate_grace_codebook_dim(MAX_CODEBOOK_DIM + 1)


# ---------- GraceCodebookConfig ----------


class TestCodebookConfig:
    def test_happy(self):
        from soup_cli.utils.grace_codebook import build_grace_codebook_config

        cfg = build_grace_codebook_config(size=128, dim=768)
        assert cfg.size == 128
        assert cfg.dim == 768

    def test_frozen(self):
        from soup_cli.utils.grace_codebook import build_grace_codebook_config

        cfg = build_grace_codebook_config(size=128, dim=768)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.size = 256  # type: ignore[misc]

    def test_invalid_size_propagates(self):
        from soup_cli.utils.grace_codebook import build_grace_codebook_config

        with pytest.raises(ValueError):
            build_grace_codebook_config(size=0, dim=768)

    def test_invalid_dim_propagates(self):
        from soup_cli.utils.grace_codebook import build_grace_codebook_config

        with pytest.raises(ValueError):
            build_grace_codebook_config(size=128, dim=0)


# ---------- apply_grace_codebook ----------


class TestApplyDeferred:
    def test_deferred(self):
        from soup_cli.utils.grace_codebook import (
            apply_grace_codebook,
            build_grace_codebook_config,
        )

        cfg = build_grace_codebook_config(size=128, dim=768)
        with pytest.raises(NotImplementedError, match="v0.62.1"):
            apply_grace_codebook(cfg)

    def test_apply_validates_config_type(self):
        from soup_cli.utils.grace_codebook import apply_grace_codebook

        with pytest.raises(TypeError):
            apply_grace_codebook("not-a-config")


# ---------- knowledge_edit allowlist extension ----------


class TestEditMethodAllowlist:
    def test_grace_in_supported_methods(self):
        from soup_cli.utils.knowledge_edit import SUPPORTED_EDIT_METHODS

        assert "grace" in SUPPORTED_EDIT_METHODS

    def test_grace_method_validates(self):
        from soup_cli.utils.knowledge_edit import validate_edit_method

        assert validate_edit_method("grace") == "grace"
        assert validate_edit_method("GRACE") == "grace"

    def test_grace_spec_present(self):
        from soup_cli.utils.knowledge_edit import get_edit_method_spec

        spec = get_edit_method_spec("grace")
        assert spec.name == "grace"
        assert spec.multi_edit_capable is True
        assert spec.live_wired is False

    def test_grace_edit_plan_happy(self):
        from soup_cli.utils.knowledge_edit import build_edit_plan

        plan = build_edit_plan(
            base="meta-llama/Llama-3.1-8B-Instruct",
            method="grace",
            subject="The capital of France is",
            target="Lyon",
        )
        assert plan.method == "grace"
        assert plan.layer >= 0

    def test_apply_grace_via_edit_path_deferred(self):
        from soup_cli.utils.knowledge_edit import apply_edit, build_edit_plan

        plan = build_edit_plan(
            base="meta-llama/Llama-3.1-8B-Instruct",
            method="grace",
            subject="The capital of France is",
            target="Lyon",
        )
        with pytest.raises(NotImplementedError, match="v0.62.1"):
            apply_edit(plan)


# ---------- Schema integration ----------


class TestSchemaIntegration:
    def test_default_none(self):
        from soup_cli.config.schema import TrainingConfig

        cfg = TrainingConfig()
        assert cfg.grace_codebook is False
        assert cfg.grace_codebook_size is None
        assert cfg.grace_codebook_dim is None

    def test_opt_in_accepts(self):
        from soup_cli.config.schema import TrainingConfig

        cfg = TrainingConfig(
            grace_codebook=True,
            grace_codebook_size=128,
            grace_codebook_dim=768,
        )
        assert cfg.grace_codebook is True
        assert cfg.grace_codebook_size == 128
        assert cfg.grace_codebook_dim == 768

    def test_invalid_size_rejected(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError):
            TrainingConfig(
                grace_codebook=True,
                grace_codebook_size=0,
                grace_codebook_dim=768,
            )

    def test_invalid_dim_rejected(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError):
            TrainingConfig(
                grace_codebook=True,
                grace_codebook_size=128,
                grace_codebook_dim=-1,
            )


# ---------- SoupConfig cross-validator ----------


class TestSoupConfigCrossValidator:
    def test_grace_codebook_size_without_flag_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        yaml_text = """\
base: meta-llama/Llama-3.1-8B-Instruct
task: sft

data:
  train: ./data/train.jsonl

training:
  epochs: 1
  lr: 2e-4
  batch_size: auto
  grace_codebook_size: 128

output: ./output
"""
        with pytest.raises(Exception, match="grace_codebook"):
            load_config_from_string(yaml_text)

    def test_grace_codebook_requires_both_size_and_dim(self):
        from soup_cli.config.loader import load_config_from_string

        # codebook flag but missing size — rejected.
        yaml_text = """\
base: meta-llama/Llama-3.1-8B-Instruct
task: sft

data:
  train: ./data/train.jsonl

training:
  epochs: 1
  lr: 2e-4
  batch_size: auto
  grace_codebook: true
  grace_codebook_dim: 768

output: ./output
"""
        with pytest.raises(Exception, match="grace_codebook"):
            load_config_from_string(yaml_text)

    def test_grace_codebook_happy(self):
        from soup_cli.config.loader import load_config_from_string

        yaml_text = """\
base: meta-llama/Llama-3.1-8B-Instruct
task: sft

data:
  train: ./data/train.jsonl

training:
  epochs: 1
  lr: 2e-4
  batch_size: auto
  grace_codebook: true
  grace_codebook_size: 128
  grace_codebook_dim: 768

output: ./output
"""
        cfg = load_config_from_string(yaml_text)
        assert cfg.training.grace_codebook is True
