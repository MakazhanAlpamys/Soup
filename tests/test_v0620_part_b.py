"""Tests for v0.62.0 Part B — RA-DIT (Retrieval-Augmented Dual Instruction Tuning).

Two-stage Meta 2023 recipe: train the retriever first (contrastive), then
the generator (RAFT-style). v0.62.0 ships the schema + recipe entries;
live runtime composes existing v0.16 embedding trainer + Part A RAFT.
"""

from __future__ import annotations

import dataclasses

import pytest
import yaml

# ---------- Module surface ----------


class TestModuleSurface:
    def test_imports(self):
        from soup_cli.utils.ra_dit import (
            SUPPORTED_RA_DIT_STAGES,
            RaDitStageSpec,
            get_ra_dit_stage_spec,
            validate_ra_dit_stage,
        )
        assert callable(validate_ra_dit_stage)
        assert callable(get_ra_dit_stage_spec)
        assert dataclasses.is_dataclass(RaDitStageSpec)
        assert isinstance(SUPPORTED_RA_DIT_STAGES, frozenset)

    def test_stages_exact(self):
        from soup_cli.utils.ra_dit import SUPPORTED_RA_DIT_STAGES

        assert SUPPORTED_RA_DIT_STAGES == frozenset({"retriever", "generator"})

    def test_metadata_mapping_proxy(self):
        from types import MappingProxyType

        from soup_cli.utils.ra_dit import _RA_DIT_STAGE_METADATA  # type: ignore

        assert isinstance(_RA_DIT_STAGE_METADATA, MappingProxyType)


# ---------- validate_ra_dit_stage ----------


class TestValidateStage:
    def test_happy(self):
        from soup_cli.utils.ra_dit import validate_ra_dit_stage

        assert validate_ra_dit_stage("retriever") == "retriever"
        assert validate_ra_dit_stage("generator") == "generator"

    def test_case_insensitive(self):
        from soup_cli.utils.ra_dit import validate_ra_dit_stage

        assert validate_ra_dit_stage("Retriever") == "retriever"
        assert validate_ra_dit_stage("GENERATOR") == "generator"

    def test_bool_rejected(self):
        from soup_cli.utils.ra_dit import validate_ra_dit_stage

        with pytest.raises(TypeError):
            validate_ra_dit_stage(True)

    def test_non_string_rejected(self):
        from soup_cli.utils.ra_dit import validate_ra_dit_stage

        with pytest.raises(TypeError):
            validate_ra_dit_stage(1)

    def test_empty_rejected(self):
        from soup_cli.utils.ra_dit import validate_ra_dit_stage

        with pytest.raises(ValueError):
            validate_ra_dit_stage("")

    def test_null_byte_rejected(self):
        from soup_cli.utils.ra_dit import validate_ra_dit_stage

        with pytest.raises(ValueError):
            validate_ra_dit_stage("retriever\x00")

    def test_oversize_rejected(self):
        from soup_cli.utils.ra_dit import validate_ra_dit_stage

        with pytest.raises(ValueError):
            validate_ra_dit_stage("retriever" * 100)

    def test_unknown_rejected(self):
        from soup_cli.utils.ra_dit import validate_ra_dit_stage

        with pytest.raises(ValueError, match="ra_dit_stage"):
            validate_ra_dit_stage("decoder")


# ---------- get_ra_dit_stage_spec ----------


class TestStageSpec:
    def test_retriever_spec(self):
        from soup_cli.utils.ra_dit import get_ra_dit_stage_spec

        spec = get_ra_dit_stage_spec("retriever")
        assert spec.name == "retriever"
        assert spec.base_task == "embedding"
        assert spec.live_wired is False

    def test_generator_spec(self):
        from soup_cli.utils.ra_dit import get_ra_dit_stage_spec

        spec = get_ra_dit_stage_spec("generator")
        assert spec.name == "generator"
        assert spec.base_task == "sft"
        assert spec.live_wired is False

    def test_unknown_raises(self):
        from soup_cli.utils.ra_dit import get_ra_dit_stage_spec

        with pytest.raises(ValueError):
            get_ra_dit_stage_spec("nonsense")

    def test_spec_frozen(self):
        from soup_cli.utils.ra_dit import get_ra_dit_stage_spec

        spec = get_ra_dit_stage_spec("retriever")
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.name = "mutated"  # type: ignore[misc]


# ---------- validate_ra_dit_retriever_model ----------


class TestRetrieverModel:
    def test_happy(self):
        from soup_cli.utils.ra_dit import validate_ra_dit_retriever_model

        assert validate_ra_dit_retriever_model("sentence-transformers/all-mpnet-base-v2") == \
            "sentence-transformers/all-mpnet-base-v2"

    def test_none_passthrough(self):
        from soup_cli.utils.ra_dit import validate_ra_dit_retriever_model

        assert validate_ra_dit_retriever_model(None) is None

    def test_empty_rejected(self):
        from soup_cli.utils.ra_dit import validate_ra_dit_retriever_model

        with pytest.raises(ValueError):
            validate_ra_dit_retriever_model("")

    def test_null_byte_rejected(self):
        from soup_cli.utils.ra_dit import validate_ra_dit_retriever_model

        with pytest.raises(ValueError):
            validate_ra_dit_retriever_model("foo\x00bar")

    def test_bool_rejected(self):
        from soup_cli.utils.ra_dit import validate_ra_dit_retriever_model

        with pytest.raises(TypeError):
            validate_ra_dit_retriever_model(True)

    def test_oversize_rejected(self):
        from soup_cli.utils.ra_dit import validate_ra_dit_retriever_model

        with pytest.raises(ValueError):
            validate_ra_dit_retriever_model("a" * 1024)


# ---------- Schema integration ----------


class TestSchemaIntegration:
    def test_default_none(self):
        from soup_cli.config.schema import TrainingConfig

        cfg = TrainingConfig()
        assert cfg.ra_dit_stage is None
        assert cfg.ra_dit_retriever_model is None

    def test_retriever_accepted(self):
        from soup_cli.config.schema import TrainingConfig

        cfg = TrainingConfig(ra_dit_stage="retriever")
        assert cfg.ra_dit_stage == "retriever"

    def test_generator_accepted(self):
        from soup_cli.config.schema import TrainingConfig

        cfg = TrainingConfig(ra_dit_stage="generator")
        assert cfg.ra_dit_stage == "generator"

    def test_case_insensitive_at_schema(self):
        from soup_cli.config.schema import TrainingConfig

        cfg = TrainingConfig(ra_dit_stage="Generator")
        assert cfg.ra_dit_stage == "generator"

    def test_invalid_stage_rejected(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError):
            TrainingConfig(ra_dit_stage="decoder")

    def test_retriever_model_accepted(self):
        from soup_cli.config.schema import TrainingConfig

        cfg = TrainingConfig(
            ra_dit_stage="retriever",
            ra_dit_retriever_model="sentence-transformers/all-mpnet-base-v2",
        )
        assert cfg.ra_dit_retriever_model == "sentence-transformers/all-mpnet-base-v2"

    def test_retriever_model_null_byte_rejected(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError):
            TrainingConfig(ra_dit_retriever_model="foo\x00")


# ---------- Cross-validator (SoupConfig) ----------


class TestSoupConfigGate:
    def test_retriever_stage_on_embedding_task(self):
        from soup_cli.config.loader import load_config_from_string

        yaml_text = """\
base: sentence-transformers/all-mpnet-base-v2
task: embedding

data:
  train: ./data/anchor.jsonl
  format: embedding

training:
  epochs: 1
  lr: 2e-5
  batch_size: auto
  ra_dit_stage: retriever

output: ./output
"""
        cfg = load_config_from_string(yaml_text)
        assert cfg.training.ra_dit_stage == "retriever"

    def test_generator_stage_on_sft_task(self):
        from soup_cli.config.loader import load_config_from_string

        yaml_text = """\
base: meta-llama/Llama-3.1-8B-Instruct
task: sft

data:
  train: ./data/raft.jsonl
  format: raft

training:
  epochs: 1
  lr: 2e-4
  batch_size: auto
  ra_dit_stage: generator

output: ./output
"""
        cfg = load_config_from_string(yaml_text)
        assert cfg.training.ra_dit_stage == "generator"

    def test_retriever_on_sft_task_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        # retriever stage requires embedding-family task
        yaml_text = """\
base: meta-llama/Llama-3.1-8B-Instruct
task: sft

data:
  train: ./data/train.jsonl

training:
  epochs: 1
  lr: 2e-4
  batch_size: auto
  ra_dit_stage: retriever

output: ./output
"""
        with pytest.raises(Exception, match="ra_dit_stage"):
            load_config_from_string(yaml_text)

    def test_generator_on_grpo_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        yaml_text = """\
base: meta-llama/Llama-3.1-8B-Instruct
task: grpo

data:
  train: ./data/prompts.jsonl

training:
  epochs: 1
  lr: 2e-4
  batch_size: auto
  ra_dit_stage: generator
  reward_fn: accuracy

output: ./output
"""
        with pytest.raises(Exception, match="ra_dit_stage"):
            load_config_from_string(yaml_text)

    def test_retriever_model_without_stage_rejected(self):
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
  ra_dit_retriever_model: sentence-transformers/all-mpnet-base-v2

output: ./output
"""
        with pytest.raises(Exception, match="ra_dit"):
            load_config_from_string(yaml_text)


# ---------- Recipes ----------


class TestRaDitRecipes:
    def test_retriever_recipe_present(self):
        from soup_cli.recipes.catalog import get_recipe

        recipe = get_recipe("ra-dit-retriever")
        assert recipe is not None
        assert recipe.task == "embedding"

    def test_generator_recipe_present(self):
        from soup_cli.recipes.catalog import get_recipe

        recipe = get_recipe("ra-dit-llama3-8b")
        assert recipe is not None
        assert recipe.task == "sft"

    def test_retriever_recipe_yaml_loads(self):
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.recipes.catalog import get_recipe

        recipe = get_recipe("ra-dit-retriever")
        assert recipe is not None
        cfg = load_config_from_string(recipe.yaml_str)
        assert cfg.training.ra_dit_stage == "retriever"

    def test_generator_recipe_yaml_loads(self):
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.recipes.catalog import get_recipe

        recipe = get_recipe("ra-dit-llama3-8b")
        assert recipe is not None
        cfg = load_config_from_string(recipe.yaml_str)
        assert cfg.training.ra_dit_stage == "generator"
        assert cfg.data.format == "raft"

    def test_recipe_yaml_parses_as_dict(self):
        from soup_cli.recipes.catalog import get_recipe

        for name in ("ra-dit-retriever", "ra-dit-llama3-8b"):
            recipe = get_recipe(name)
            assert recipe is not None
            parsed = yaml.safe_load(recipe.yaml_str)
            assert isinstance(parsed, dict)
