"""Tests for unknown config key aliases and suggestion ranking (#2099)."""

from __future__ import annotations

import pytest

from soup_cli.config.aliases import CONFIG_ALIASES
from soup_cli.config.schema import SoupConfig, TrainingConfig
from soup_cli.config.unknown_keys import find_unknown_config_keys, format_unknown_keys
from tests.conftest import strip_ansi

_BASE_CONFIG = """
base: meta-llama/Llama-3.1-8B-Instruct
task: sft
data:
  train: ./train.jsonl
  format: auto
output: ./out
"""


def _raw_with_training(**training_kwargs) -> dict:
    import yaml

    doc = yaml.safe_load(_BASE_CONFIG)
    doc["training"] = dict(training_kwargs)
    return doc


def _raw_with_root(**root_kwargs) -> dict:
    import yaml

    doc = yaml.safe_load(_BASE_CONFIG)
    doc.update(root_kwargs)
    return doc


class TestAliasSchemaIntegrity:
    """Every alias target in CONFIG_ALIASES must be a real declared field in the schema."""

    def test_every_alias_target_exists_in_schema(self) -> None:
        model_map = {
            "training": TrainingConfig,
        }
        for alias_path, target_path in CONFIG_ALIASES.items():
            parts = target_path.split(".")
            assert len(parts) == 2, f"Target '{target_path}' must have format 'section.field'"
            section, field = parts
            assert section in SoupConfig.model_fields, (
                f"Alias target section '{section}' (from '{alias_path}' -> '{target_path}') "
                "is not a declared field of SoupConfig"
            )
            model_cls = model_map.get(section)
            assert model_cls is not None, f"No model class mapped for section '{section}'"
            assert field in model_cls.model_fields, (
                f"Alias target field '{field}' (from '{alias_path}' -> '{target_path}') "
                f"is not a declared field of {model_cls.__name__}"
            )


class TestAliasSuggestions:
    """Each alias in CONFIG_ALIASES must suggest the target field and outrank difflib."""

    @pytest.mark.parametrize("alias_path,target_path", list(CONFIG_ALIASES.items()))
    def test_alias_suggests_canonical_field(self, alias_path: str, target_path: str) -> None:
        target_field = target_path.split(".")[-1]
        if "." in alias_path:
            section, key = alias_path.split(".", 1)
            assert section == "training"
            raw = _raw_with_training(**{key: 123})
        else:
            raw = _raw_with_root(**{alias_path: 123})

        unknown = find_unknown_config_keys(raw)
        assert len(unknown) == 1
        assert unknown[0].path == alias_path
        assert target_field in unknown[0].suggestions
        assert unknown[0].alias == target_path

        msg = strip_ansi(format_unknown_keys(unknown))
        assert f"use {target_path}" in msg
        assert target_field in msg

    def test_typo_lerning_rate_suggests_lr_and_not_use_ring_attention(self) -> None:
        """training.lerning_rate must suggest lr ('use training.lr'), not ring_attention."""
        raw = _raw_with_training(lerning_rate=2e-5)
        unknown = find_unknown_config_keys(raw)
        assert len(unknown) == 1
        assert unknown[0].path == "training.lerning_rate"
        assert "lr" in unknown[0].suggestions
        assert unknown[0].alias == "training.lr"

        msg = strip_ansi(format_unknown_keys(unknown))
        assert "use training.lr" in msg
        assert "use_ring_attention" not in msg

    @pytest.mark.parametrize(
        "typo, canonical",
        [
            ("warmup_ato", "warmup_auto"),
            ("grad_accumauto_tune", "grad_accum_auto_tune"),
        ],
    )
    def test_closer_declared_field_outranks_fuzzy_alias(self, typo: str, canonical: str) -> None:
        """An alias typo must not hide a closer match to an actual Soup field."""
        unknown = find_unknown_config_keys(_raw_with_training(**{typo: True}))
        assert len(unknown) == 1
        assert unknown[0].alias is None
        assert unknown[0].suggestions[0] == canonical
        message = format_unknown_keys(unknown, include_deadline=False)
        assert f"did you mean '{canonical}'" in message
        assert "use training.warmup_ratio" not in message
        assert "use training.gradient_accumulation_steps" not in message

    def test_unrecognized_key_has_no_suggestion_and_no_use_hint(self) -> None:
        """A key with no near match gets no suggestion and no 'use' hint."""
        raw = _raw_with_training(zzzzzzzz=1)
        unknown = find_unknown_config_keys(raw)
        assert len(unknown) == 1
        assert unknown[0].suggestions == ()
        assert unknown[0].alias is None

        msg = strip_ansi(format_unknown_keys(unknown))
        assert "did you mean" not in msg.lower()
        assert "use " not in msg.lower()

    def test_valid_config_has_no_unknown_keys(self) -> None:
        """Control test: a clean valid config produces no unknown keys."""
        import yaml

        doc = yaml.safe_load(_BASE_CONFIG)
        doc["training"] = {"epochs": 2, "lr": 1e-4, "batch_size": 4}
        unknown = find_unknown_config_keys(doc)
        assert unknown == []

    def test_mutation_removing_alias_lookup_fails_alias_suggestion(self, monkeypatch) -> None:
        """Mutating CONFIG_ALIASES to empty causes alias detection to fail."""
        monkeypatch.setattr("soup_cli.config.unknown_keys.CONFIG_ALIASES", {})
        raw = _raw_with_training(learning_rate=2e-5)
        unknown = find_unknown_config_keys(raw)
        assert len(unknown) == 1
        assert unknown[0].alias is None
        assert "lr" not in unknown[0].suggestions

    def test_declared_cutoff_rejects_lerning_rate_when_no_alias_applies(
        self, monkeypatch
    ) -> None:
        """Declared cutoff 0.65 rejects distant suggestions (e.g. use_ring_attention)."""
        monkeypatch.setattr("soup_cli.config.unknown_keys.CONFIG_ALIASES", {})
        unknown = find_unknown_config_keys({"training": {"lerning_rate": 2e-5}})
        assert len(unknown) == 1
        assert unknown[0].suggestions == ()
        message = format_unknown_keys(unknown)
        assert "use_ring_attention" not in message
        assert "did you mean" not in message

