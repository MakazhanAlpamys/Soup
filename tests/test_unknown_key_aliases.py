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



def _hint(raw: dict, *, path: str) -> str:
    """The report line for ``path``, with the path itself and the verdict word removed."""
    unknown = [item for item in find_unknown_config_keys(raw) if item.path == path]
    assert len(unknown) == 1, (path, find_unknown_config_keys(raw))
    line = strip_ansi(format_unknown_keys(unknown, include_deadline=False))
    return " ".join(line.replace(f"'{path}'", "<key>").split())


def _both_levels(key: str) -> tuple[str, str]:
    """The hint for ``key`` under ``training:`` and at the top level of the file."""
    return (
        _hint(_raw_with_training(**{key: 4}), path=f"training.{key}"),
        _hint(_raw_with_root(**{key: 4}), path=key),
    )


class TestNearIdenticalTyposOnly:
    """"use X" is for the alias table's own spellings and one-edit typos of them (#2099)."""

    @pytest.mark.parametrize(
        ("typo", "target"),
        [
            ("learning_rat", "training.lr"),
            ("lerning_rate", "training.lr"),
            ("num_train_epoch", "training.epochs"),
            ("per_device_train_batch_siz", "training.batch_size"),
            ("train_batch_siz", "training.batch_size"),
        ],
    )
    def test_a_one_edit_typo_of_an_alias_still_says_use(self, typo, target):
        for raw, path in (
            (_raw_with_training(**{typo: 1}), f"training.{typo}"),
            (_raw_with_root(**{typo: 1}), typo),
        ):
            unknown = [item for item in find_unknown_config_keys(raw) if item.path == path]
            assert len(unknown) == 1
            assert unknown[0].alias == target, path
            assert f"use {target}" in format_unknown_keys(unknown)

    def test_a_substituted_word_is_not_a_typo(self):
        # 0.92 similar as a string, but "trial" is not a spelling of "train".
        key = "per_device_trial_batch_size"
        from difflib import SequenceMatcher

        assert SequenceMatcher(None, key, "per_device_train_batch_size").ratio() >= 0.9
        for hint in _both_levels(key):
            assert "use training.batch_size" not in hint
            assert "training.batch_size" not in hint

    def test_a_fuzzy_match_to_an_alias_never_says_use(self):
        unknown = find_unknown_config_keys(_raw_with_training(train_epochs=3))
        assert len(unknown) == 1
        assert unknown[0].alias is None
        assert "use training" not in format_unknown_keys(unknown)

    def test_the_alias_cutoff_is_not_looser_than_a_typo(self):
        # Mutation guard: with a 0.7 cutoff and no token check, a fuzzy neighbour of
        # the alias would say "use". Pin the keys that used to.
        for key in ("learning_rate_decay", "per_device_eval_batch_size", "warmup_steps"):
            for raw, path in (
                (_raw_with_training(**{key: 1}), f"training.{key}"),
                (_raw_with_root(**{key: 1}), key),
            ):
                unknown = [i for i in find_unknown_config_keys(raw) if i.path == path]
                assert len(unknown) == 1
                assert unknown[0].alias is None, path

    @pytest.mark.parametrize(
        ("one_edit", "two_edits", "alias", "target"),
        [
            (
                "per_devce_trai_batc_sze",
                "pr_devce_trai_batc_sze",
                "per_device_train_batch_size",
                "training.batch_size",
            ),
            ("num_trai_epoch", "num_trin_epoc", "num_train_epochs", "training.epochs"),
        ],
    )
    def test_a_key_just_below_the_cutoff_does_not_say_use(
        self, one_edit, two_edits, alias, target
    ):
        # Same words, spelled alike in both keys: only the whole-string cutoff (0.9) tells
        # the key that says "use X" from the one that does not.
        from difflib import SequenceMatcher

        def ratio(left: str, right: str) -> float:
            return SequenceMatcher(None, left, right).ratio()

        assert ratio(one_edit, alias) >= 0.9
        assert 0.85 <= ratio(two_edits, alias) < 0.9
        for word, other in zip(two_edits.split("_"), alias.split("_")):
            assert ratio(word, other) >= 0.8, (word, other)
        assert len(two_edits.split("_")) == len(alias.split("_"))

        for key, says_use in ((one_edit, True), (two_edits, False)):
            for raw, path in (
                (_raw_with_training(**{key: 1}), f"training.{key}"),
                (_raw_with_root(**{key: 1}), key),
            ):
                unknown = [item for item in find_unknown_config_keys(raw) if item.path == path]
                assert len(unknown) == 1, path
                message = format_unknown_keys(unknown, include_deadline=False)
                if says_use:
                    assert unknown[0].alias == target, path
                    assert f"use {target}" in message, path
                else:
                    assert unknown[0].alias is None, path
                    assert "use " not in message, path
                    assert target not in message, path


    def test_a_one_letter_substitution_in_a_short_alias_never_says_use(self):
        # "epochz" is 0.83 similar to the alias "epochs": under the 0.9 string cutoff.
        from difflib import SequenceMatcher

        assert 0.7 <= SequenceMatcher(None, "epochz", "epochs").ratio() < 0.9
        under_training = find_unknown_config_keys(_raw_with_training(epochz=1))
        assert len(under_training) == 1
        assert under_training[0].alias is None
        message = format_unknown_keys(under_training, include_deadline=False)
        assert "did you mean 'epochs'" in message
        assert "use " not in message

        at_root = find_unknown_config_keys(_raw_with_root(epochz=1))
        assert len(at_root) == 1
        assert at_root[0].alias is None
        assert "use " not in format_unknown_keys(at_root, include_deadline=False)


class TestDottedKeysAtTheRoot:
    """A literal ``training.<key>`` at the top level matches the alias table's exact path."""

    @pytest.mark.parametrize(
        ("key", "target"),
        [
            ("training.grad_accum_steps", "training.gradient_accumulation_steps"),
            ("training.warmup_steps_ratio", "training.warmup_ratio"),
        ],
    )
    def test_the_exact_path_says_use_the_aliased_field(self, key, target):
        raw = _raw_with_root(**{key: 1})
        unknown = find_unknown_config_keys(raw)
        assert len(unknown) == 1, unknown
        assert unknown[0].path == key
        assert unknown[0].alias == target
        assert unknown[0].suggestions == (target.split(".")[-1],)
        assert _hint(raw, path=key) == f"unknown config key <key> - use {target}. Refused."


class TestNamesThatAreNotTheTrainingSettings:
    """HF names that look like an alias but mean something else get their own answer."""

    @pytest.mark.parametrize(
        "key", ["per_device_eval_batch_size", "per_gpu_eval_batch_size", "eval_batch_size"]
    )
    def test_an_evaluation_batch_size_is_not_the_training_batch_size(self, key):
        training, root = _both_levels(key)
        assert training == root
        assert "no separate evaluation batch size" in training
        assert "training.batch_size" not in training
        assert "use " not in training

    @pytest.mark.parametrize("key", ["warmup_steps", "warmup_step"])
    def test_warmup_in_steps_is_not_silently_a_fraction(self, key):
        training, root = _both_levels(key)
        assert training == root
        assert "use training.warmup_ratio" not in training
        assert "fraction" in training
        assert "no step-count" in training
        assert "training.warmup_ratio" in training

    def test_learning_rate_decay_is_not_the_learning_rate(self):
        training, root = _both_levels("learning_rate_decay")
        assert training == root
        assert "training.lr" not in training
        assert "use " not in training

    @pytest.mark.parametrize(
        "key",
        [
            "per_device_eval_batch_size",
            "per_gpu_eval_batch_size",
            "eval_batch_size",
            "warmup_steps",
            "warmup_step",
            "learning_rate_decay",
        ],
    )
    def test_the_answer_is_the_same_at_both_levels_and_marks_the_key_refused(self, key):
        training, root = _both_levels(key)
        assert training == root
        refused = format_unknown_keys(
            find_unknown_config_keys(_raw_with_training(**{key: 1})), include_deadline=False
        )
        assert refused.lower().endswith("refused.")
        assert f"'training.{key}'" in refused

    def test_a_note_is_a_fact_not_a_replacement(self):
        # The notes only name settings that exist, and none of them is offered as "use X".
        import re

        from soup_cli.config.aliases import CONFIG_NOTES

        assert CONFIG_NOTES
        for path, note in CONFIG_NOTES.items():
            assert path not in CONFIG_ALIASES, path
            assert note.endswith("."), path
            for field in re.findall(r"training\.(\w+)", note):
                assert field in TrainingConfig.model_fields, (path, field)

    def test_the_loader_refuses_with_the_note(self):
        from soup_cli.config.loader import load_config_from_string

        text = _BASE_CONFIG + "training:\n  per_device_eval_batch_size: 4\n"
        with pytest.raises((SystemExit, ValueError)) as caught:
            load_config_from_string(text)
        message = str(caught.value)
        assert "no separate evaluation batch size" in message
        assert "training.batch_size" not in message
