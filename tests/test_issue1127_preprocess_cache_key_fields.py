"""#1127: the preprocess cache key covers every input that changes the cached rows.

``data.val_split``, the ``data.replay*`` fields and the top-level ``task`` all
change which rows ``soup data preprocess`` writes, and none of them was part of
the key, so a ``pre_tokenized`` run under a different setting reused the cache
without a mismatch. Every ``DataConfig`` field is now declared either in the key
or out of it with a reason, and a field in neither list fails here.
"""

from __future__ import annotations

import inspect

import pytest

from soup_cli.config.schema import DataConfig
from soup_cli.utils import data_pipeline as dp
from tests.preprocess_cache_blob import CURRENT_SCHEMA, SEGMENTS_BY_SCHEMA, blob_key
from tests.test_issue1054_pretokenized_labels import _ROWS, _run_preprocess, _tokenizer

_KEY_INPUTS = set(inspect.signature(dp.make_preprocess_cache_key).parameters)


class TestDeclaredTable:
    def test_every_data_config_field_is_classified(self):
        classified = set(dp.PREPROCESS_KEY_FIELDS) | set(dp.NOT_PREPROCESS_KEY_FIELDS)
        missing = set(DataConfig.model_fields) - classified
        assert not missing, (
            "classify in PREPROCESS_KEY_FIELDS or NOT_PREPROCESS_KEY_FIELDS: "
            f"{sorted(missing)}"
        )

    def test_classified_fields_exist_on_the_schema(self):
        classified = set(dp.PREPROCESS_KEY_FIELDS) | set(dp.NOT_PREPROCESS_KEY_FIELDS)
        stale = classified - set(DataConfig.model_fields)
        assert not stale, f"not DataConfig fields any more: {sorted(stale)}"

    def test_no_field_is_in_both_tables(self):
        assert not set(dp.PREPROCESS_KEY_FIELDS) & set(dp.NOT_PREPROCESS_KEY_FIELDS)

    def test_excluded_fields_carry_a_reason(self):
        for name, reason in dp.NOT_PREPROCESS_KEY_FIELDS.items():
            assert isinstance(reason, str) and reason.strip(), name

    def test_key_fields_name_a_real_key_input(self):
        for name, carrier in dp.PREPROCESS_KEY_FIELDS.items():
            assert carrier in _KEY_INPUTS, f"{name} -> {carrier!r}"


class TestSharedBlobHelper:
    def test_current_generation_matches_the_production_inputs(self):
        """A new ``make_preprocess_cache_key`` parameter fails here until
        tests/preprocess_cache_blob.py gains a generation that writes it."""
        assert CURRENT_SCHEMA == dp._PREPROCESS_TOKENIZE_SCHEMA
        assert set(SEGMENTS_BY_SCHEMA[CURRENT_SCHEMA]) == _KEY_INPUTS

    def test_helper_reproduces_the_production_key(self):
        args = dict(
            dataset_path="data/train.jsonl",
            tokenizer_name="x/y",
            max_length=128,
            format_name="chatml",
            chat_template="{{ messages }}",
            mask_mode="responses_only+eot",
            task="pretrain",
        )
        assert dp.make_preprocess_cache_key(**args) == blob_key(CURRENT_SCHEMA, **args)


def _key(dcfg: DataConfig, task: str = "sft") -> str:
    """The key the cache writer and the training gate both derive from a config."""
    from soup_cli.data.chat_templates import resolve_chat_template

    return dp.make_preprocess_cache_key(
        dataset_path=dp.preprocess_dataset_key_input(dcfg),
        tokenizer_name="x/y",
        max_length=dcfg.max_length,
        format_name=dcfg.format,
        chat_template=resolve_chat_template(dcfg.chat_template),
        mask_mode=dp.preprocess_mask_mode(dcfg),
        task=task,
    )


_PAIR = ["./a.jsonl", "./b.jsonl"]
_NO_RESPONSES_ONLY = {"train_on_responses_only": False}

# Two configs per key field that differ in that field only. Listed by hand rather
# than derived from PREPROCESS_KEY_FIELDS, so moving a field out of the key still
# runs its case and fails it.
_ONE_FIELD_APART = {
    "train": ({}, {"train": "./other.jsonl"}),
    "interleave": (
        {"train": _PAIR, "interleave": "concat"},
        {"train": _PAIR, "interleave": "under"},
    ),
    "val_split": ({}, {"val_split": 0.2}),
    "replay": ({}, {"replay": "./old.jsonl"}),
    "replay_ratio": (
        {"replay": "./old.jsonl"},
        {"replay": "./old.jsonl", "replay_ratio": 0.3},
    ),
    "replay_seed": (
        {"replay": "./old.jsonl"},
        {"replay": "./old.jsonl", "replay_seed": 7},
    ),
    "streaming": ({}, {"streaming": True}),
    "buffer_size": ({"streaming": True}, {"streaming": True, "buffer_size": 100}),
    "image_dir": ({}, {"image_dir": "./images"}),
    "audio_dir": ({}, {"audio_dir": "./audio"}),
    "format": ({}, {"format": "alpaca"}),
    "max_length": ({}, {"max_length": 256}),
    "chat_template": ({}, {"chat_template": "chatml"}),
    "train_on_responses_only": ({}, _NO_RESPONSES_ONLY),
    "train_on_messages_with_train_field": (
        _NO_RESPONSES_ONLY,
        {**_NO_RESPONSES_ONLY, "train_on_messages_with_train_field": True},
    ),
    "mask_history": ({}, {"mask_history": True}),
}


def _data_config(overrides: dict) -> DataConfig:
    return DataConfig(**{"train": "./d.jsonl", **overrides})


class TestEachKeyFieldChangesTheKey:
    def test_every_key_field_has_a_case(self):
        assert set(_ONE_FIELD_APART) == set(dp.PREPROCESS_KEY_FIELDS)

    @pytest.mark.parametrize("field", sorted(_ONE_FIELD_APART))
    def test_changing_only_this_field_changes_the_key(self, field):
        before, after = _ONE_FIELD_APART[field]
        base, changed = _data_config(before), _data_config(after)
        differing = {
            name
            for name in DataConfig.model_fields
            if getattr(base, name) != getattr(changed, name)
        }
        assert differing == {field}, differing
        assert _key(base) != _key(changed), f"{field} is not part of the cache key"

    def test_task_changes_the_key(self):
        dcfg = _data_config({})
        assert _key(dcfg, task="sft") != _key(dcfg, task="pretrain")

    @pytest.mark.parametrize("bad", ["", "sft\x00", None])
    def test_bad_task_is_rejected(self, bad):
        with pytest.raises(ValueError, match="task"):
            dp.make_preprocess_cache_key(
                dataset_path="./d.jsonl",
                tokenizer_name="x/y",
                max_length=128,
                format_name="chatml",
                mask_mode="responses_only",
                task=bad,
            )


class TestGateRefusesACacheBuiltUnderAnotherSetting:
    """Acceptance criterion 2, end to end: a real ``soup data preprocess`` cache,
    then the ``pre_tokenized`` gate under a config that differs in one field.
    Each case loads on ``main``."""

    def _gate(self, cache_dir, tmp_path, *, task="sft", **data):
        from rich.console import Console

        from soup_cli.trainer.sft import _maybe_load_pretokenized

        dcfg = DataConfig(
            train="./d.jsonl",
            format="pre_tokenized",
            tokenized_path=str(cache_dir.relative_to(tmp_path)),
            max_length=128,
            **data,
        )
        return _maybe_load_pretokenized(dcfg, "x/y", Console(), None, task=task)

    def _cache(self, tmp_path, monkeypatch, *, task="sft", data_extra=""):
        (tmp_path / "old.jsonl").write_text("{}\n", encoding="utf-8")
        return _run_preprocess(
            tmp_path,
            monkeypatch,
            _tokenizer(),
            task=task,
            rows=[{"messages": _ROWS[0]}],
            data_extra=data_extra,
        )

    def test_the_same_config_still_loads(self, tmp_path, monkeypatch):
        cache_dir = self._cache(tmp_path, monkeypatch, data_extra="  val_split: 0.2\n")
        loaded = self._gate(cache_dir, tmp_path, val_split=0.2)
        assert loaded is not None
        assert len(loaded[0]) == 1

    def test_val_split(self, tmp_path, monkeypatch):
        cache_dir = self._cache(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="cache hash mismatch"):
            self._gate(cache_dir, tmp_path, val_split=0.2)

    def test_replay(self, tmp_path, monkeypatch):
        cache_dir = self._cache(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="cache hash mismatch"):
            self._gate(cache_dir, tmp_path, replay="./old.jsonl")

    def test_replay_ratio(self, tmp_path, monkeypatch):
        cache_dir = self._cache(tmp_path, monkeypatch, data_extra="  replay: ./old.jsonl\n")
        with pytest.raises(ValueError, match="cache hash mismatch"):
            self._gate(cache_dir, tmp_path, replay="./old.jsonl", replay_ratio=0.3)

    def test_replay_seed(self, tmp_path, monkeypatch):
        cache_dir = self._cache(tmp_path, monkeypatch, data_extra="  replay: ./old.jsonl\n")
        with pytest.raises(ValueError, match="cache hash mismatch"):
            self._gate(cache_dir, tmp_path, replay="./old.jsonl", replay_seed=7)

    def test_task(self, tmp_path, monkeypatch):
        cache_dir = self._cache(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="cache hash mismatch"):
            self._gate(cache_dir, tmp_path, task="pretrain")


class TestEveryGateCallerForwardsTheTask:
    def test_each_call_site_passes_task(self):
        """A caller has to name ``task`` (keyword-only, no default since #1272),
        and it has to be the run's own: keying a pretrain run as sft refuses every
        cache ``soup data preprocess`` wrote for it."""
        import ast
        from pathlib import Path

        import soup_cli

        root = Path(soup_cli.__file__).parent
        calls = []
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "_maybe_load_pretokenized"
                ):
                    calls.append((path.name, node))
        assert {"sft.py", "pretrain.py"} <= {name for name, _ in calls}
        for name, node in calls:
            keywords = {kw.arg: kw.value for kw in node.keywords}
            assert "task" in keywords, f"{name}:{node.lineno} does not pass task"
            assert ast.unparse(keywords["task"]) == "cfg.task", (
                f"{name}:{node.lineno} passes {ast.unparse(keywords['task'])!r}"
            )
