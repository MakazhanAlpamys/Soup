"""Tests for v0.62.0 Part D — Citation-faithful FT.

Schema-only release: `training.citation_faithful: bool` opt-in trains the
model to cite document IDs verbatim from the training corpus. Composes
with v0.62.0 Part A (RAFT) — the RAFT data already contains the doc
references; this flag adds a citation-precision / recall scorer to the
eval suite + a loss-mask rule that emphasises citation spans.

Live citation-span loss-mask + eval scorer land in v0.62.1.
"""

from __future__ import annotations

import dataclasses

import pytest

# ---------- Module surface ----------


class TestModuleSurface:
    def test_imports(self):
        from soup_cli.utils.citation_faithful import (
            SUPPORTED_CITATION_STYLES,
            CitationScore,
            score_citations,
            validate_citation_style,
            validate_citation_threshold,
        )
        assert callable(validate_citation_style)
        assert callable(validate_citation_threshold)
        assert callable(score_citations)
        assert dataclasses.is_dataclass(CitationScore)
        assert isinstance(SUPPORTED_CITATION_STYLES, frozenset)

    def test_styles_exact(self):
        from soup_cli.utils.citation_faithful import SUPPORTED_CITATION_STYLES

        assert SUPPORTED_CITATION_STYLES == frozenset(
            {"bracket", "inline", "footnote"}
        )


# ---------- validate_citation_style ----------


class TestValidateStyle:
    def test_happy(self):
        from soup_cli.utils.citation_faithful import validate_citation_style

        for name in ("bracket", "inline", "footnote"):
            assert validate_citation_style(name) == name

    def test_case_insensitive(self):
        from soup_cli.utils.citation_faithful import validate_citation_style

        assert validate_citation_style("BRACKET") == "bracket"

    def test_bool_rejected(self):
        from soup_cli.utils.citation_faithful import validate_citation_style

        with pytest.raises(TypeError):
            validate_citation_style(True)

    def test_non_string_rejected(self):
        from soup_cli.utils.citation_faithful import validate_citation_style

        with pytest.raises(TypeError):
            validate_citation_style(1)

    def test_empty_rejected(self):
        from soup_cli.utils.citation_faithful import validate_citation_style

        with pytest.raises(ValueError):
            validate_citation_style("")

    def test_null_byte_rejected(self):
        from soup_cli.utils.citation_faithful import validate_citation_style

        with pytest.raises(ValueError):
            validate_citation_style("bracket\x00")

    def test_oversize_rejected(self):
        from soup_cli.utils.citation_faithful import validate_citation_style

        with pytest.raises(ValueError):
            validate_citation_style("x" * 64)

    def test_unknown_rejected(self):
        from soup_cli.utils.citation_faithful import validate_citation_style

        with pytest.raises(ValueError, match="citation"):
            validate_citation_style("apa-7th")


# ---------- validate_citation_threshold ----------


class TestValidateThreshold:
    def test_happy(self):
        from soup_cli.utils.citation_faithful import validate_citation_threshold

        assert validate_citation_threshold(0.5) == 0.5
        assert validate_citation_threshold(0.0) == 0.0
        assert validate_citation_threshold(1.0) == 1.0

    def test_bool_rejected(self):
        from soup_cli.utils.citation_faithful import validate_citation_threshold

        with pytest.raises(TypeError):
            validate_citation_threshold(True)

    def test_non_finite_rejected(self):
        import math

        from soup_cli.utils.citation_faithful import validate_citation_threshold

        with pytest.raises(ValueError):
            validate_citation_threshold(math.nan)
        with pytest.raises(ValueError):
            validate_citation_threshold(math.inf)

    def test_out_of_range_rejected(self):
        from soup_cli.utils.citation_faithful import validate_citation_threshold

        with pytest.raises(ValueError):
            validate_citation_threshold(-0.1)
        with pytest.raises(ValueError):
            validate_citation_threshold(1.1)


# ---------- score_citations ----------


class TestScoreCitations:
    def test_perfect_match(self):
        from soup_cli.utils.citation_faithful import score_citations

        score = score_citations(
            predicted="The capital is Paris [doc-1].",
            expected_ids=("doc-1",),
        )
        assert score.precision == 1.0
        assert score.recall == 1.0
        assert score.f1 == 1.0

    def test_no_citations(self):
        from soup_cli.utils.citation_faithful import score_citations

        score = score_citations(
            predicted="The capital is Paris.",
            expected_ids=("doc-1",),
        )
        # No predicted citations: precision is undefined (set to 0.0 by
        # convention), recall is 0.0.
        assert score.precision == 0.0
        assert score.recall == 0.0
        assert score.f1 == 0.0

    def test_extra_citation_lowers_precision(self):
        from soup_cli.utils.citation_faithful import score_citations

        score = score_citations(
            predicted="See [doc-1] and [doc-2].",
            expected_ids=("doc-1",),
        )
        # Recall = 1/1 = 1.0, Precision = 1/2 = 0.5.
        assert score.recall == 1.0
        assert score.precision == 0.5
        assert 0.5 < score.f1 < 1.0

    def test_missing_citation_lowers_recall(self):
        from soup_cli.utils.citation_faithful import score_citations

        score = score_citations(
            predicted="See [doc-1].",
            expected_ids=("doc-1", "doc-2"),
        )
        assert score.precision == 1.0
        assert score.recall == 0.5

    def test_empty_expected_ids(self):
        from soup_cli.utils.citation_faithful import score_citations

        # No expected citations: recall is undefined but score returns 0
        # by convention (avoids div-by-zero). Precision still defined.
        score = score_citations(
            predicted="The capital is Paris [doc-1].",
            expected_ids=(),
        )
        assert score.recall == 0.0
        # Predicted IDs are not in the empty expected set, so precision
        # is 0.0 too.
        assert score.precision == 0.0

    def test_bool_predicted_rejected(self):
        from soup_cli.utils.citation_faithful import score_citations

        with pytest.raises(TypeError):
            score_citations(predicted=True, expected_ids=("doc-1",))

    def test_non_string_predicted_rejected(self):
        from soup_cli.utils.citation_faithful import score_citations

        with pytest.raises(TypeError):
            score_citations(predicted=123, expected_ids=("doc-1",))

    def test_oversize_predicted_rejected(self):
        from soup_cli.utils.citation_faithful import score_citations

        with pytest.raises(ValueError):
            score_citations(
                predicted="x" * 2_000_001,
                expected_ids=("doc-1",),
            )

    def test_non_iterable_expected_rejected(self):
        from soup_cli.utils.citation_faithful import score_citations

        with pytest.raises(TypeError):
            score_citations(predicted="text", expected_ids=42)

    def test_too_many_expected_rejected(self):
        from soup_cli.utils.citation_faithful import score_citations

        with pytest.raises(ValueError):
            score_citations(
                predicted="text",
                expected_ids=tuple(f"doc-{i}" for i in range(10_001)),
            )

    def test_score_frozen(self):
        from soup_cli.utils.citation_faithful import score_citations

        score = score_citations(predicted="[doc-1]", expected_ids=("doc-1",))
        with pytest.raises(dataclasses.FrozenInstanceError):
            score.f1 = 0.0  # type: ignore[misc]


# ---------- Schema integration ----------


class TestSchemaIntegration:
    def test_default_off(self):
        from soup_cli.config.schema import TrainingConfig

        cfg = TrainingConfig()
        assert cfg.citation_faithful is False
        assert cfg.citation_style is None
        assert cfg.citation_recall_threshold is None

    def test_opt_in_accepts(self):
        from soup_cli.config.schema import TrainingConfig

        cfg = TrainingConfig(
            citation_faithful=True,
            citation_style="bracket",
            citation_recall_threshold=0.8,
        )
        assert cfg.citation_faithful is True
        assert cfg.citation_style == "bracket"
        assert cfg.citation_recall_threshold == 0.8

    def test_style_case_insensitive(self):
        from soup_cli.config.schema import TrainingConfig

        cfg = TrainingConfig(citation_style="INLINE")
        assert cfg.citation_style == "inline"

    def test_unknown_style_rejected(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError):
            TrainingConfig(citation_style="apa")

    def test_threshold_bounds(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError):
            TrainingConfig(citation_recall_threshold=-0.1)
        with pytest.raises(ValidationError):
            TrainingConfig(citation_recall_threshold=1.5)


# ---------- Cross-validator ----------


class TestSoupConfigCrossValidator:
    def test_citation_faithful_with_raft_format(self):
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
  citation_faithful: true
  citation_style: bracket

output: ./output
"""
        cfg = load_config_from_string(yaml_text)
        assert cfg.training.citation_faithful is True

    def test_citation_faithful_without_raft_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        yaml_text = """\
base: meta-llama/Llama-3.1-8B-Instruct
task: sft

data:
  train: ./data/train.jsonl
  format: alpaca

training:
  epochs: 1
  lr: 2e-4
  batch_size: auto
  citation_faithful: true

output: ./output
"""
        with pytest.raises(Exception, match="citation_faithful"):
            load_config_from_string(yaml_text)

    def test_citation_style_without_faithful_rejected(self):
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
  citation_style: bracket

output: ./output
"""
        with pytest.raises(Exception, match="citation"):
            load_config_from_string(yaml_text)

    def test_threshold_without_faithful_rejected(self):
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
  citation_recall_threshold: 0.8

output: ./output
"""
        with pytest.raises(Exception, match="citation"):
            load_config_from_string(yaml_text)
