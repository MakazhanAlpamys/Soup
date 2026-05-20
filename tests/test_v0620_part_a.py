"""Tests for v0.62.0 Part A — RAFT (Retrieval-Augmented Fine-Tuning) recipe.

Adds ``data.format='raft'`` to the schema + new ``_convert_raft`` validator
+ a ``raft-llama3-8b`` recipe entry. Schema validates RAFT row shape:
``{"query", "golden_doc", "distractor_docs", "answer"}``.
"""

from __future__ import annotations

import pytest
import yaml

# ---------- _convert_raft ----------


class TestConvertRaft:
    def test_happy_path(self):
        from soup_cli.data.formats import _convert_raft

        row = {
            "query": "What year was Python released?",
            "golden_doc": "Python was released in 1991 by Guido van Rossum.",
            "distractor_docs": [
                "Ruby was released in 1995.",
                "Java was released in 1995.",
            ],
            "answer": "1991",
        }
        out = _convert_raft(row)
        assert out["query"] == row["query"]
        assert out["golden_doc"] == row["golden_doc"]
        assert out["distractor_docs"] == row["distractor_docs"]
        assert out["answer"] == row["answer"]

    def test_empty_distractors_ok(self):
        from soup_cli.data.formats import _convert_raft

        out = _convert_raft({
            "query": "q",
            "golden_doc": "g",
            "distractor_docs": [],
            "answer": "a",
        })
        assert out["distractor_docs"] == []

    def test_missing_query_rejected(self):
        from soup_cli.data.formats import _convert_raft

        with pytest.raises((KeyError, ValueError)):
            _convert_raft({
                "golden_doc": "g",
                "distractor_docs": [],
                "answer": "a",
            })

    def test_missing_golden_doc_rejected(self):
        from soup_cli.data.formats import _convert_raft

        with pytest.raises((KeyError, ValueError)):
            _convert_raft({
                "query": "q",
                "distractor_docs": [],
                "answer": "a",
            })

    def test_missing_answer_rejected(self):
        from soup_cli.data.formats import _convert_raft

        with pytest.raises((KeyError, ValueError)):
            _convert_raft({
                "query": "q",
                "golden_doc": "g",
                "distractor_docs": [],
            })

    def test_empty_query_rejected(self):
        from soup_cli.data.formats import _convert_raft

        with pytest.raises(ValueError, match="query"):
            _convert_raft({
                "query": "",
                "golden_doc": "g",
                "distractor_docs": [],
                "answer": "a",
            })

    def test_empty_golden_doc_rejected(self):
        from soup_cli.data.formats import _convert_raft

        with pytest.raises(ValueError, match="golden_doc"):
            _convert_raft({
                "query": "q",
                "golden_doc": "",
                "distractor_docs": [],
                "answer": "a",
            })

    def test_empty_answer_rejected(self):
        from soup_cli.data.formats import _convert_raft

        with pytest.raises(ValueError, match="answer"):
            _convert_raft({
                "query": "q",
                "golden_doc": "g",
                "distractor_docs": [],
                "answer": "",
            })

    def test_non_string_query_rejected(self):
        from soup_cli.data.formats import _convert_raft

        with pytest.raises(ValueError, match="query"):
            _convert_raft({
                "query": 123,
                "golden_doc": "g",
                "distractor_docs": [],
                "answer": "a",
            })

    def test_non_list_distractors_rejected(self):
        from soup_cli.data.formats import _convert_raft

        with pytest.raises(ValueError, match="distractor_docs"):
            _convert_raft({
                "query": "q",
                "golden_doc": "g",
                "distractor_docs": "not-a-list",
                "answer": "a",
            })

    def test_non_string_distractor_rejected(self):
        from soup_cli.data.formats import _convert_raft

        with pytest.raises(ValueError, match="distractor"):
            _convert_raft({
                "query": "q",
                "golden_doc": "g",
                "distractor_docs": [123],
                "answer": "a",
            })

    def test_null_byte_query_rejected(self):
        from soup_cli.data.formats import _convert_raft

        with pytest.raises(ValueError, match="null"):
            _convert_raft({
                "query": "bad\x00",
                "golden_doc": "g",
                "distractor_docs": [],
                "answer": "a",
            })

    def test_distractor_cap(self):
        from soup_cli.data.formats import _MAX_RAFT_DISTRACTORS, _convert_raft

        # exactly cap accepted
        out = _convert_raft({
            "query": "q",
            "golden_doc": "g",
            "distractor_docs": ["d"] * _MAX_RAFT_DISTRACTORS,
            "answer": "a",
        })
        assert len(out["distractor_docs"]) == _MAX_RAFT_DISTRACTORS

    def test_distractor_overcap_rejected(self):
        from soup_cli.data.formats import _MAX_RAFT_DISTRACTORS, _convert_raft

        with pytest.raises(ValueError, match="distractor"):
            _convert_raft({
                "query": "q",
                "golden_doc": "g",
                "distractor_docs": ["d"] * (_MAX_RAFT_DISTRACTORS + 1),
                "answer": "a",
            })


# ---------- format dispatcher ----------


class TestFormatDispatcher:
    def test_format_to_messages_dispatches_raft(self):
        from soup_cli.data.formats import format_to_messages

        row = {
            "query": "q",
            "golden_doc": "g",
            "distractor_docs": [],
            "answer": "a",
        }
        out = format_to_messages(row, "raft")
        assert out is not None
        assert out["query"] == "q"
        assert out["answer"] == "a"

    def test_format_to_messages_invalid_raft_returns_none(self):
        from soup_cli.data.formats import format_to_messages

        # missing key — wrapper catches and returns None
        assert format_to_messages({"query": "q"}, "raft") is None


# ---------- schema integration ----------


class TestSchemaIntegration:
    def test_data_format_raft_accepted(self):
        from soup_cli.config.schema import DataConfig

        cfg = DataConfig(train="data.jsonl", format="raft")
        assert cfg.format == "raft"

    def test_soup_config_raft_roundtrip(self):
        from soup_cli.config.loader import load_config_from_string

        yaml_text = """\
base: meta-llama/Llama-3.1-8B-Instruct
task: sft

data:
  train: ./data/raft.jsonl
  format: raft
  max_length: 4096

training:
  epochs: 1
  lr: 2e-4
  batch_size: auto

output: ./output
"""
        cfg = load_config_from_string(yaml_text)
        assert cfg.data.format == "raft"
        assert cfg.task == "sft"


# ---------- recipe catalog ----------


class TestRaftRecipe:
    def test_recipe_present(self):
        from soup_cli.recipes.catalog import get_recipe

        recipe = get_recipe("raft-llama3-8b")
        assert recipe is not None
        assert recipe.task == "sft"
        assert "raft" in recipe.tags or "rag" in recipe.tags

    def test_recipe_yaml_loads(self):
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.recipes.catalog import get_recipe

        recipe = get_recipe("raft-llama3-8b")
        assert recipe is not None
        cfg = load_config_from_string(recipe.yaml_str)
        assert cfg.data.format == "raft"

    def test_recipe_yaml_parses_as_dict(self):
        from soup_cli.recipes.catalog import get_recipe

        recipe = get_recipe("raft-llama3-8b")
        assert recipe is not None
        parsed = yaml.safe_load(recipe.yaml_str)
        assert isinstance(parsed, dict)
        assert parsed["data"]["format"] == "raft"

    def test_recipe_search_finds_raft(self):
        from soup_cli.recipes.catalog import search_recipes

        results = search_recipes(query="raft")
        assert any(r.task == "sft" and "raft" in r.yaml_str for r in results)
