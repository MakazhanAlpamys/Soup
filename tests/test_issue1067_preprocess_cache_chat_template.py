"""Tests for Issue #1067: data.chat_template must participate in preprocess cache key,
and pre-existing caches without recorded chat template are refused at load time.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from soup_cli.config.schema import DataConfig
from soup_cli.trainer.sft import _maybe_load_pretokenized
from soup_cli.utils.data_pipeline import make_preprocess_cache_key


class TestCacheKeyParticipation:
    """Verify that data.chat_template is bound to the preprocess cache key."""

    def test_two_configs_differing_only_in_chat_template_resolve_to_different_keys(self) -> None:
        """AC1: two configs differing only in chat_template resolve to different entries."""
        key_default = make_preprocess_cache_key(
            dataset_path="data/train.jsonl",
            tokenizer_name="meta-llama/Llama-3.1-8B",
            max_length=2048,
            format_name="chatml",
            chat_template=None,
        )
        key_chatml = make_preprocess_cache_key(
            dataset_path="data/train.jsonl",
            tokenizer_name="meta-llama/Llama-3.1-8B",
            max_length=2048,
            format_name="chatml",
            chat_template="chatml",
        )
        key_llama3 = make_preprocess_cache_key(
            dataset_path="data/train.jsonl",
            tokenizer_name="meta-llama/Llama-3.1-8B",
            max_length=2048,
            format_name="chatml",
            chat_template="llama3",
        )
        key_custom = make_preprocess_cache_key(
            dataset_path="data/train.jsonl",
            tokenizer_name="meta-llama/Llama-3.1-8B",
            max_length=2048,
            format_name="chatml",
            chat_template="{% for message in messages %}{{ message.content }}{% endfor %}",
        )

        keys = {key_default, key_chatml, key_llama3, key_custom}
        assert len(keys) == 4, f"All four distinct chat templates must produce unique keys: {keys}"

    def test_mutation_collision_when_chat_template_is_ignored(self) -> None:
        """AC2: A test that fails if the field is removed from key derivation —
        specifically asserting that two configs with different templates do not collide.
        """
        def _simulate_old_key_derivation(template: str | None) -> str:
            # Simulates what happens if chat_template is dropped from derivation
            return make_preprocess_cache_key(
                dataset_path="data/train.jsonl",
                tokenizer_name="meta-llama/Llama-3.1-8B",
                max_length=2048,
                format_name="chatml",
                chat_template=None,  # simulated omission
            )

        # In old implementation, these two collided:
        colliding_a = _simulate_old_key_derivation("chatml")
        colliding_b = _simulate_old_key_derivation("llama3")
        assert colliding_a == colliding_b, "Simulated bug must produce a collision"

        # In actual implementation, they MUST NOT collide:
        actual_a = make_preprocess_cache_key(
            dataset_path="data/train.jsonl",
            tokenizer_name="meta-llama/Llama-3.1-8B",
            max_length=2048,
            format_name="chatml",
            chat_template="chatml",
        )
        actual_b = make_preprocess_cache_key(
            dataset_path="data/train.jsonl",
            tokenizer_name="meta-llama/Llama-3.1-8B",
            max_length=2048,
            format_name="chatml",
            chat_template="llama3",
        )
        assert actual_a != actual_b, "Real implementation must never collide on differing templates"

    def test_cache_key_input_validation(self) -> None:
        """Type safety and null-byte defenses on chat_template."""
        with pytest.raises(ValueError, match="chat_template must be a string or None"):
            make_preprocess_cache_key(
                dataset_path="data/train.jsonl",
                tokenizer_name="x",
                max_length=512,
                format_name="chatml",
                chat_template=True,  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="chat_template must be a string or None"):
            make_preprocess_cache_key(
                dataset_path="data/train.jsonl",
                tokenizer_name="x",
                max_length=512,
                format_name="chatml",
                chat_template=123,  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="null bytes"):
            make_preprocess_cache_key(
                dataset_path="data/train.jsonl",
                tokenizer_name="x",
                max_length=512,
                format_name="chatml",
                chat_template="chatml\x00extra",
            )


class TestPreExistingCacheRefusal:
    """AC4: Verify that legacy caches (written before chat_template tracking)
    are refused with an explicit message telling the user to re-run preprocess.
    """

    def test_pre_existing_cache_missing_chat_template_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A legacy cache directory with metadata.json lacking 'chat_template' must raise."""
        monkeypatch.chdir(tmp_path)
        cache_dir = tmp_path / "legacy_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        legacy_metadata = {
            "cache_key": "abc123def4567890",
            "row_count": 100,
            "tokenizer_name": "meta-llama/Llama-3.1-8B",
            "max_length": 2048,
            "format": "chatml",
            "task": "sft",
            "soup_version": "0.74.0",
        }
        (cache_dir / "metadata.json").write_text(json.dumps(legacy_metadata), encoding="utf-8")

        dcfg = DataConfig(
            train="data/train.jsonl",
            format="pre_tokenized",
            tokenized_path="./legacy_cache",
            max_length=2048,
        )

        with pytest.raises(ValueError) as excinfo:
            _maybe_load_pretokenized(dcfg, "meta-llama/Llama-3.1-8B", MagicMock())

        err = str(excinfo.value)
        assert "predates chat_template tracking" in err
        assert "re-run `soup data preprocess`" in err

    def test_template_mismatch_between_cache_and_config_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cache recorded with template A must fail when consumed with config asking for B."""
        monkeypatch.chdir(tmp_path)
        cache_dir = tmp_path / "template_a_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        key_a = make_preprocess_cache_key(
            dataset_path="data/train.jsonl",
            tokenizer_name="meta-llama/Llama-3.1-8B",
            max_length=2048,
            format_name="chatml",
            chat_template="chatml",
        )
        metadata = {
            "cache_key": key_a,
            "row_count": 50,
            "tokenizer_name": "meta-llama/Llama-3.1-8B",
            "max_length": 2048,
            "format": "chatml",
            "chat_template": "chatml",
            "task": "sft",
            "soup_version": "0.75.0",
        }
        (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

        # Caller config requests llama3 template instead
        dcfg = DataConfig(
            train="data/train.jsonl",
            format="pre_tokenized",
            tokenized_path="./template_a_cache",
            max_length=2048,
            chat_template="llama3",
        )

        with pytest.raises(ValueError) as excinfo:
            _maybe_load_pretokenized(dcfg, "meta-llama/Llama-3.1-8B", MagicMock())

        err = str(excinfo.value)
        assert "pre_tokenized cache hash mismatch" in err
        assert "re-run `soup data preprocess`" in err

    def test_matching_template_reaches_dataset_loader(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When cache metadata and config agree on chat_template, the gate passes."""
        monkeypatch.chdir(tmp_path)
        cache_dir = tmp_path / "valid_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = make_preprocess_cache_key(
            dataset_path="data/train.jsonl",
            tokenizer_name="meta-llama/Llama-3.1-8B",
            max_length=2048,
            format_name="chatml",
            chat_template="llama3",
        )
        metadata = {
            "cache_key": key,
            "row_count": 25,
            "tokenizer_name": "meta-llama/Llama-3.1-8B",
            "max_length": 2048,
            "format": "chatml",
            "chat_template": "llama3",
            "task": "sft",
            "soup_version": "0.75.0",
        }
        (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

        dcfg = DataConfig(
            train="data/train.jsonl",
            format="pre_tokenized",
            tokenized_path="./valid_cache",
            max_length=2048,
            chat_template="llama3",
        )

        fake_ds = {"train": [{"input_ids": [1, 2, 3]}]}
        monkeypatch.setattr(
            "soup_cli.utils.data_pipeline.load_pretokenized_dataset",
            lambda path: fake_ds,
        )

        res = _maybe_load_pretokenized(dcfg, "meta-llama/Llama-3.1-8B", MagicMock())
        assert res is not None
        train_ds, _ = res
        assert train_ds == [{"input_ids": [1, 2, 3]}]
