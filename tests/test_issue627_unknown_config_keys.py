"""Unknown config keys must not be silently dropped (#627).

None of the 9 config models set ``extra="forbid"``, so Pydantic's default
``extra="ignore"`` applied everywhere: a misspelled or not-yet-released key
validated clean, printed "Config valid. Ready to train!" under
``soup train --dry-run``, and was discarded.

The keys that reach users are not exotic. ``quantizaton``,
``gradient_checkpoint``, ``lr_scheduler`` and ``max_len`` are each one edit from
a real field, so the run trains in full precision / without checkpointing / on
the default schedule while the operator believes otherwise. Exit 0, plausible
logs, and the requested thing silently not done.

#623 is the live example: ``training.stream_pin`` landed on main two days after
0.73.3 shipped, a user on the released wheel wrote the documented escape hatch,
``--dry-run`` called it valid, the key was dropped, and the resulting OOM was
diagnosed as a layer-streaming bug across two long comments.

Every test here is CPU-only: no GPU, no downloads, no model load.
"""

from __future__ import annotations

import pytest

from soup_cli.config.unknown_keys import (
    find_unknown_config_keys,
    format_unknown_keys,
)

_VALID = """
base: hf/model
task: sft
data:
  train: ./t.jsonl
  format: auto
output: ./o
"""


def _raw(extra_data: str = "", extra_training: str = "") -> dict:
    import yaml

    doc = yaml.safe_load(_VALID)
    if extra_data:
        doc["data"].update(yaml.safe_load(extra_data))
    if extra_training:
        doc["training"] = yaml.safe_load(extra_training)
    return doc


class TestTheFiveReportedCases:
    """Each of the silently-accepted keys from the issue, pinned by name."""

    @pytest.mark.parametrize(
        "key,value,expected_suggestion",
        [
            ("quantizaton", "4bit", "quantization"),
            ("gradient_checkpoint", True, "gradient_checkpointing"),
            ("lr_scheduler", "cosine", "scheduler"),
            ("stream_pin_typo", False, "stream_pin"),
        ],
    )
    def test_training_key_is_reported_with_a_suggestion(
        self, key, value, expected_suggestion
    ) -> None:
        import yaml

        doc = yaml.safe_load(_VALID)
        doc["training"] = {"epochs": 1, key: value}
        unknown = find_unknown_config_keys(doc)

        assert [u.key for u in unknown] == [key]
        assert unknown[0].path == f"training.{key}"
        # The suggestion is the point -- "unknown field 'quantizaton'" alone is
        # much less useful than naming the field the user meant.
        assert expected_suggestion in unknown[0].suggestions

    def test_data_key_is_reported(self) -> None:
        """A different model: a fix guarding training only must fail here."""
        unknown = find_unknown_config_keys(_raw(extra_data="{max_len: 512}"))
        assert [u.path for u in unknown] == ["data.max_len"]
        assert "max_length" in unknown[0].suggestions


class TestNesting:
    """Each model in the tree is walked, not just the top level."""

    def test_unknown_key_inside_lora_is_found(self) -> None:
        unknown = find_unknown_config_keys(
            _raw(extra_training="{epochs: 1, lora: {r: 8, alfa: 16}}")
        )
        assert [u.path for u in unknown] == ["training.lora.alfa"]
        assert "alpha" in unknown[0].suggestions

    def test_unknown_key_at_the_top_level_is_found(self) -> None:
        raw = _raw()
        raw["taks"] = "sft"
        assert "taks" in [u.key for u in find_unknown_config_keys(raw)]

    def test_several_unknowns_are_all_reported(self) -> None:
        raw = _raw(extra_data="{max_len: 512}", extra_training="{epochs: 1, quantizaton: 4bit}")
        paths = sorted(u.path for u in find_unknown_config_keys(raw))
        assert paths == ["data.max_len", "training.quantizaton"]


class TestTheControl:
    """The guard must not be satisfiable by rejecting everything."""

    def test_a_valid_config_reports_nothing(self) -> None:
        assert find_unknown_config_keys(_raw()) == []

    def test_a_config_using_many_real_keys_reports_nothing(self) -> None:
        import yaml

        doc = yaml.safe_load(_VALID)
        doc["data"].update({"max_length": 2048, "val_split": 0.1})
        doc["training"] = {
            "epochs": 3,
            "lr": 5e-6,
            "batch_size": "auto",
            "gradient_accumulation_steps": 8,
            "quantization": "4bit",
            "gradient_checkpointing": True,
            "dpo_beta": 0.1,
            "moe_lora": True,
            "lora": {"r": 16, "alpha": 32, "target_modules": "auto"},
        }
        assert find_unknown_config_keys(doc) == []

    def test_every_real_recipe_in_the_catalog_is_clean(self) -> None:
        """The strongest control available: 160 shipped configs, none flagged."""
        import yaml

        from soup_cli.recipes.catalog import RECIPES

        offenders = {}
        for name, recipe in RECIPES.items():
            unknown = find_unknown_config_keys(yaml.safe_load(recipe.yaml_str))
            if unknown:
                offenders[name] = [u.path for u in unknown]
        assert offenders == {}

    def test_non_mapping_values_do_not_crash_the_walk(self) -> None:
        raw = _raw()
        raw["training"] = "not-a-mapping"
        find_unknown_config_keys(raw)  # must not raise

    def test_empty_and_none_sections_are_tolerated(self) -> None:
        raw = _raw()
        raw["training"] = None
        raw["eval"] = {}
        find_unknown_config_keys(raw)  # must not raise


class TestTheMessage:
    def test_message_names_the_path_and_the_suggestion(self) -> None:
        unknown = find_unknown_config_keys(
            _raw(extra_training="{epochs: 1, quantizaton: 4bit}")
        )
        msg = format_unknown_keys(unknown)
        assert "training.quantizaton" in msg
        assert "quantization" in msg
        assert "did you mean" in msg.lower()

    def test_a_key_with_no_close_match_still_reports_cleanly(self) -> None:
        unknown = find_unknown_config_keys(
            _raw(extra_training="{epochs: 1, zzzzzzzz: 1}")
        )
        assert unknown[0].suggestions == ()
        msg = format_unknown_keys(unknown)
        assert "training.zzzzzzzz" in msg
        assert "did you mean" not in msg.lower()


class TestEveryConstructionSiteIsGuarded:
    """A fourth ``SoupConfig(**...)`` must not be able to appear unguarded.

    The point of #627 is that nothing *reminds* a caller to check. A scanner is
    the only guard that survives someone adding a new entry point next year --
    the same shape as ``test_no_second_hand_rolled_prompt_remains_in_the_serve_backends``.
    """

    def test_all_soupconfig_construction_sites_check_unknown_keys(self) -> None:
        import re
        from pathlib import Path

        src = Path(__file__).parents[1] / "src" / "soup_cli"
        offenders = []
        for path in src.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if not re.search(r"SoupConfig\(\*\*", text):
                continue
            if "find_unknown_config_keys" not in text:
                offenders.append(str(path.relative_to(src)))
        assert offenders == [], (
            f"these build a SoupConfig without checking for unknown keys: {offenders}"
        )
