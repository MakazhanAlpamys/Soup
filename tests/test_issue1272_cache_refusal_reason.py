"""#1272: a refused ``pre_tokenized`` cache says why.

Since #1127 ten ``data`` fields fold into one hash, so a bare hash mismatch no
longer tells a user what changed. ``soup data preprocess`` now records the key
generation and the row-set inputs in ``metadata.json``, and the gate names either
the gap (a cache written before #1127) or the ``data`` fields that differ.
"""

from __future__ import annotations

import json

import pytest

from soup_cli.config.schema import DataConfig
from soup_cli.utils import data_pipeline as dp
from tests.test_issue1054_pretokenized_labels import _ROWS, _run_preprocess, _tokenizer


def _cache(tmp_path, monkeypatch, data_extra=""):
    return _run_preprocess(
        tmp_path,
        monkeypatch,
        _tokenizer(),
        rows=[{"messages": _ROWS[0]}],
        data_extra=data_extra,
    )


def _refusal(cache_dir, tmp_path, **data) -> str:
    from rich.console import Console

    from soup_cli.trainer.sft import _maybe_load_pretokenized

    dcfg = DataConfig(
        train="./d.jsonl",
        format="pre_tokenized",
        tokenized_path=str(cache_dir.relative_to(tmp_path)),
        max_length=128,
        **data,
    )
    with pytest.raises(ValueError, match="cache hash mismatch") as exc:
        _maybe_load_pretokenized(dcfg, "x/y", Console(), None, task="sft")
    return str(exc.value)


class TestMetadataRecordsTheKey:
    def test_key_schema_and_dataset_key_are_written(self, tmp_path, monkeypatch):
        cache_dir = _cache(tmp_path, monkeypatch, data_extra="  val_split: 0.2\n")
        meta = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))

        assert meta["key_schema"] == dp._PREPROCESS_TOKENIZE_SCHEMA
        assert meta["dataset_key"]["val_split"] == 0.2
        assert set(meta["dataset_key"]) == set(dp._DATASET_KEY_FIELDS)


class TestTheRefusalSaysWhy:
    def test_a_v6_cache_names_the_row_set_keying(self, tmp_path, monkeypatch):
        """A v6 cache has ``chat_template`` and ``mask_mode`` but no ``key_schema``."""
        cache_dir = _cache(tmp_path, monkeypatch)
        meta_path = cache_dir / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.pop("key_schema")
        meta.pop("dataset_key")
        meta["cache_key"] = "0" * 16  # v6 hashed fewer inputs
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        message = _refusal(cache_dir, tmp_path)

        assert "predates row-set keying (#1127)" in message
        assert "data.val_split" not in message

    def test_a_different_val_split_is_named(self, tmp_path, monkeypatch):
        cache_dir = _cache(tmp_path, monkeypatch)

        message = _refusal(cache_dir, tmp_path, val_split=0.2)

        assert "the cache was built with a different data.val_split;" in message
        assert "predates" not in message

    def test_every_differing_field_is_named_in_table_order(self, tmp_path, monkeypatch):
        cache_dir = _cache(tmp_path, monkeypatch)

        message = _refusal(cache_dir, tmp_path, audio_dir="./audio", val_split=0.2)

        assert "a different data.val_split, data.audio_dir;" in message

    def test_a_mismatch_outside_the_row_set_names_no_field(self, tmp_path, monkeypatch):
        """``max_length`` is keyed but is not a row-set input, so the message adds
        nothing it cannot back up."""
        cache_dir = _cache(tmp_path, monkeypatch)
        from rich.console import Console

        from soup_cli.trainer.sft import _maybe_load_pretokenized

        dcfg = DataConfig(
            train="./d.jsonl",
            format="pre_tokenized",
            tokenized_path=str(cache_dir.relative_to(tmp_path)),
            max_length=256,
        )
        with pytest.raises(ValueError, match="cache hash mismatch") as exc:
            _maybe_load_pretokenized(dcfg, "x/y", Console(), None, task="sft")

        assert "predates" not in str(exc.value)
        assert "built with a different" not in str(exc.value)


class TestDatasetKeyDiff:
    def test_same_inputs_differ_in_nothing(self):
        current = dp.preprocess_dataset_key_input(DataConfig(train="./d.jsonl"))
        assert dp.preprocess_dataset_key_diff(json.loads(current), current) == []

    @pytest.mark.parametrize("stored", ["edited", None])
    def test_a_stored_value_that_is_not_a_dict_names_nothing(self, stored):
        """A hand-edited ``dataset_key`` cannot say what changed, so no field is named."""
        current = dp.preprocess_dataset_key_input(DataConfig(train="./d.jsonl"))
        assert dp.preprocess_dataset_key_diff(stored, current) == []


class TestTheGateNeedsATask:
    def test_task_has_no_default(self):
        """A forgotten caller is a TypeError, not a silently wrong key."""
        import inspect

        from soup_cli.trainer.sft import _maybe_load_pretokenized

        param = inspect.signature(_maybe_load_pretokenized).parameters["task"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is inspect.Parameter.empty


class TestExclusionReasonsDescribeTheCode:
    @pytest.mark.parametrize(
        "field",
        [
            "eval_on_each_dataset",
            "image_min_pixels",
            "image_max_pixels",
            "image_resize_algorithm",
            "video_fps",
            "video_maxlen",
        ],
    )
    def test_staged_fields_say_nothing_reads_them(self, field):
        from soup_cli.config.staged_fields import STAGED_FIELDS

        assert ("data", field) in STAGED_FIELDS, field
        assert "no runtime code reads it" in dp.NOT_PREPROCESS_KEY_FIELDS[field]

    def test_remove_unused_columns_says_no_trainer_reads_it(self):
        assert "no trainer reads it" in dp.NOT_PREPROCESS_KEY_FIELDS["remove_unused_columns"]

    def test_remove_unused_columns_is_still_unconsumed(self):
        from tests.test_issue748_config_fields_reach_a_consumer import KNOWN_UNCONSUMED

        assert "data.remove_unused_columns" in KNOWN_UNCONSUMED


class TestTheOldestGapStillWins:
    @pytest.mark.parametrize(
        ("dropped", "expected"),
        [
            (("mask_mode",), "predates loss-mask keying (#1054)"),
            (("mask_mode", "chat_template"), "predates chat_template keying (#1067)"),
        ],
    )
    def test_an_older_cache_names_its_own_gap(
        self, tmp_path, monkeypatch, dropped, expected
    ):
        """A real v5 or v4 cache has no ``key_schema`` or ``dataset_key`` either,
        so it predates #1127 too; the oldest gap is the one named."""
        cache_dir = _cache(tmp_path, monkeypatch)
        meta_path = cache_dir / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for field in ("key_schema", "dataset_key", *dropped):
            meta.pop(field)
        meta["cache_key"] = "0" * 16
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        message = _refusal(cache_dir, tmp_path)

        assert expected in message
        assert "#1127" not in message
