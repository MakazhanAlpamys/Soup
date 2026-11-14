"""Native cache consumer: real producer round trips and exact stored targets."""


import pytest
from rich.console import Console

from soup_cli.config.schema import SoupConfig
from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper
from tests.test_issue634_mlx_resume import _FakeMlxModel, _install_fake_mlx
from tests.test_issue1038_preprocess_to_pretokenized import _BASE, _preprocess, _tokenizer


def cache_config(cache, **data):
    return SoupConfig(
        base=_BASE, backend="mlx", output="adapter",
        data={"train": "train.jsonl", "format": "pre_tokenized", "tokenized_path": str(cache),
              "max_length": 128, "val_split": 0, **data},
        training={"quantization": "none", "batch_size": 1},
    )


def test_public_setup_consumes_real_preprocessing_cache(tmp_path, monkeypatch):
    from datasets import load_from_disk

    cache = _preprocess(tmp_path, monkeypatch, "train.jsonl")
    expected = list(load_from_disk(cache))
    assert expected and any(label == -100 for label in expected[0]["labels"])
    _install_fake_mlx(monkeypatch)
    monkeypatch.setattr("soup_cli.utils.mlx.load_mlx_model", lambda *a, **k: (
        _FakeMlxModel(), _tokenizer(),
    ))
    wrapper = MLXSFTTrainerWrapper(cache_config(cache))
    wrapper.setup({"train": [], "val": []})
    assert list(wrapper._dataset["train"]) == expected
    assert not wrapper._dataset["val"]


@pytest.mark.parametrize("change", [
    {"max_length": 127}, {"train": "other.jsonl"}, {"mask_history": True},
])
def test_public_setup_refuses_changed_fingerprint(tmp_path, monkeypatch, change):
    cache = _preprocess(tmp_path, monkeypatch, "train.jsonl")
    _install_fake_mlx(monkeypatch)
    monkeypatch.setattr("soup_cli.utils.mlx.load_mlx_model", lambda *a, **k: (
        _FakeMlxModel(), _tokenizer(),
    ))
    with pytest.raises(ValueError, match="cache hash mismatch"):
        MLXSFTTrainerWrapper(cache_config(cache, **change)).setup({"train": []})


def test_cache_splits_and_legacy_advisory(tmp_path, monkeypatch, capsys):
    from datasets import Dataset, DatasetDict

    from soup_cli.utils.pretokenized import maybe_load_pretokenized

    monkeypatch.chdir(tmp_path)
    rows = {"input_ids": [[1, 3, 2], [1, 4, 2]], "labels": [[-100, 3, 2], [-100, 4, 2]]}
    DatasetDict(train=Dataset.from_dict(rows), validation=Dataset.from_dict({
        "input_ids": [[1, 5, 2]], "labels": [[-100, 5, 2]],
    })).save_to_disk("cache")
    cfg = cache_config("cache")
    train, val = maybe_load_pretokenized(cfg.data, cfg.base, Console(), cfg.training, task="sft")
    assert list(train)[0]["input_ids"] == [1, 3, 2]
    assert list(train)[1]["input_ids"] == [1, 4, 2]
    assert list(val)[0]["labels"] == [-100, 5, 2]
    assert "no metadata.json" in capsys.readouterr().out


@pytest.mark.parametrize("metadata", ["{", "[]", "null", "42"])
def test_corrupt_metadata_refused_without_raw_fallback(tmp_path, monkeypatch, metadata):
    from soup_cli.utils.pretokenized import maybe_load_pretokenized

    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "metadata.json").write_text(metadata)
    cfg = cache_config(cache)
    with pytest.raises(ValueError, match="metadata.json"):
        maybe_load_pretokenized(cfg.data, cfg.base, Console(), cfg.training, task="sft")


def test_missing_shard_refused(tmp_path, monkeypatch):
    from soup_cli.utils.pretokenized import maybe_load_pretokenized

    cache = _preprocess(tmp_path, monkeypatch, "train.jsonl")
    next(cache.glob("*.arrow")).unlink()
    cfg = cache_config(cache)
    with pytest.raises((OSError, ValueError)):
        maybe_load_pretokenized(cfg.data, cfg.base, Console(), cfg.training, task="sft")


def valid_row():
    return {"input_ids": [1, 3, 4, 2], "labels": [-100, -100, 4, 2],
            "attention_mask": [1, 1, 1, 1]}


def test_stored_labels_preserved_including_distinct_targets_and_pad_eos():
    from soup_cli.trainer.mlx_pretokenized import PretokenizedDataset

    row = valid_row()
    row["labels"][2] = 5  # A valid explicit target can differ from its input token.
    row["input_ids"] += [2]
    row["labels"] += [-100]
    row["attention_mask"] += [0]
    dataset = PretokenizedDataset([row], vocab_size=8, max_length=128)
    assert dataset.process(dataset[0]) == ([1, 3, 4, 2, 2], [-100, -100, 5, 2, -100])
    assert row["labels"] == [-100, -100, 5, 2, -100]


@pytest.mark.parametrize("field,value,reason", [
    ("input_ids", [], "nonempty"), ("input_ids", [True, 3, 4, 2], "integer"),
    ("input_ids", [1, 3.0, 4, 2], "integer"), ("input_ids", [1, -1, 4, 2], "vocabulary"),
    ("input_ids", [1, 8, 4, 2], "vocabulary"),
    ("labels", None, "labels"), ("labels", [-100, 4], "length"),
    ("labels", [-100, False, 4, 2], "integer"),
    ("labels", [-100, -2, 4, 2], "vocabulary"), ("labels", [-100, 8, 4, 2], "vocabulary"),
    ("labels", [-100]*4, "causal-loss target"),
    ("attention_mask", [1, 1], "length"), ("attention_mask", [1, 2, 1, 1], "binary"),
    ("attention_mask", [1, 0, 1, 1], "right padding"),
    ("attention_mask", [1, 1, 1, 0], "padding.*labels"),
])
def test_invalid_rows_refused(field, value, reason):
    from soup_cli.trainer.mlx_pretokenized import PretokenizedDataset

    row = valid_row()
    row[field] = value
    with pytest.raises(ValueError, match=reason):
        PretokenizedDataset([row], vocab_size=8, max_length=128)


def test_every_row_checked_and_truncated_targets_refused():
    from soup_cli.trainer.mlx_pretokenized import PretokenizedDataset

    row = valid_row()
    PretokenizedDataset([row], vocab_size=8, max_length=3)
    with pytest.raises(ValueError, match="row 2.*causal-loss target"):
        PretokenizedDataset([row, {**row, "labels": [1, -100, -100, -100]}],
                            vocab_size=8, max_length=128)
    with pytest.raises(ValueError, match="causal-loss target.*truncation"):
        PretokenizedDataset([row], vocab_size=8, max_length=2)


@pytest.mark.parametrize("change", ["tokenizer", "task", "template"])
def test_shared_cache_gate_checks_other_fingerprint_inputs(tmp_path, monkeypatch, change):
    from soup_cli.utils.pretokenized import maybe_load_pretokenized

    cache = _preprocess(tmp_path, monkeypatch, "train.jsonl")
    payload = cache_config(cache).model_dump()
    payload["backend"] = "transformers"
    if change == "tokenizer":
        payload["base"] = "another/tokenizer"
    elif change == "task":
        payload["task"] = "pretrain"
    else:
        payload["data"]["chat_template"] = "chatml"
    cfg = SoupConfig.model_validate(payload)
    with pytest.raises(ValueError, match="cache hash mismatch"):
        maybe_load_pretokenized(cfg.data, cfg.base, Console(), cfg.training, task=cfg.task)


def test_native_custom_template_refusal_remains_explicit(tmp_path):
    with pytest.raises(ValueError, match="chat_template is not applied by backend: mlx"):
        cache_config("cache", chat_template="chatml")


def test_shared_consumer_does_not_import_the_hf_training_stack(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys

    cache = _preprocess(tmp_path, monkeypatch, "train.jsonl")
    script = '''
import builtins, json, sys
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'peft', 'trl'}:
        raise AssertionError('forbidden training-stack import: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from rich.console import Console
from soup_cli.config.schema import SoupConfig
from soup_cli.utils.pretokenized import maybe_load_pretokenized
cfg = SoupConfig.model_validate_json(sys.argv[1])
train, val = maybe_load_pretokenized(cfg.data, cfg.base, Console(), cfg.training, task='sft')
assert len(train) == 4 and train[0]['labels'][0] == -100
assert not any(name in sys.modules for name in ('torch', 'peft', 'trl'))
'''
    result = subprocess.run(
        [sys.executable, "-c", script, cache_config(cache).model_dump_json()],
        env={**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
        text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("missing_train", [False, True])
def test_native_missing_or_empty_train_split_refused_before_model_loading(
    tmp_path, monkeypatch, missing_train,
):
    from datasets import Dataset, DatasetDict

    monkeypatch.chdir(tmp_path)
    rows = Dataset.from_dict({"input_ids": [[1, 3, 2]], "labels": [[-100, 3, 2]]})
    if missing_train:
        DatasetDict(validation=rows).save_to_disk("cache")
    else:
        rows.save_to_disk("cache")
        # datasets cannot round-trip its own zero-shard empty dataset. Exercise
        # the empty-split guard at the external reader boundary with a real
        # empty Dataset, rather than passing malformed shards to another guard.
        monkeypatch.setattr("soup_cli.utils.data_pipeline.load_pretokenized_dataset",
                            lambda path: rows.select([]))
    _install_fake_mlx(monkeypatch)

    def must_not_load(*args, **kwargs):
        raise AssertionError("invalid cache must fail before loading a model")

    monkeypatch.setattr("soup_cli.utils.mlx.load_mlx_model", must_not_load)
    reason = "requires a train split" if missing_train else "split is empty"
    with pytest.raises(ValueError, match=reason):
        MLXSFTTrainerWrapper(cache_config("cache")).setup({"train": []})
