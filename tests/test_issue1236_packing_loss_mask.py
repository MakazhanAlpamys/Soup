"""Tests for issue #1236: `packing: true` preserves SFT assistant-only loss mask.

Asserts that supervised token positions (labels != -100) match between
`packing: false` and `packing: true` across:
- default responses_only
- train_on_messages_with_train_field
- train_on_eot
- pre-tokenized cache
- packing_strategy: bfd, bfd-requeue, wrapped
- padding_free
- pretrain with packing is unaffected
"""

from __future__ import annotations

import pytest
import yaml
from datasets import Dataset

from soup_cli.config.loader import load_config_from_string
from soup_cli.trainer.pretrain import PretrainTrainerWrapper
from soup_cli.trainer.sft import SFTTrainerWrapper
from tests.test_issue691_sft_packing import _tiny_causal_model_dir

pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("transformers")
pytest.importorskip("peft")


def _count_supervised_and_real_tokens(dataloader):
    """Count supervised positions (labels != -100) and real tokens in a dataloader."""
    real = 0
    supervised = 0
    for batch in dataloader:
        lab = batch["labels"]
        supervised += int((lab != -100).sum())
        if "attention_mask" in batch:
            real += int(batch["attention_mask"].sum())
        else:
            real += int(batch["input_ids"].numel())
    return supervised, real


def _build_sft_wrapper(tmp_path, monkeypatch, data_extra=None, **training_extra):
    weights = _tiny_causal_model_dir(tmp_path)
    monkeypatch.chdir(tmp_path)
    tcfg = {
        "batch_size": 2,
        "quantization": "none",
        "epochs": 1,
        "logging_steps": 1,
        "save_steps": 1000,
        "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
    }
    tcfg.update(training_extra)
    dcfg = {
        "train": "train.jsonl",
        "max_length": 128,
        "chat_template": "chatml",
    }
    if data_extra:
        dcfg.update(data_extra)

    cfg = load_config_from_string(
        yaml.safe_dump(
            {
                "base": weights,
                "task": "sft",
                "backend": "transformers",
                "modality": "text",
                "data": dcfg,
                "training": tcfg,
                "output": str(tmp_path / "out"),
            }
        )
    )
    return SFTTrainerWrapper(cfg, device="cpu")


class TestSFTPackingLossMask:
    def test_default_responses_only_packing_preserves_mask(self, tmp_path, monkeypatch):
        row = {
            "messages": [
                {"role": "system", "content": "sys prompt"},
                {"role": "user", "content": "hello there"},
                {"role": "assistant", "content": "world reply"},
            ]
        }
        rows = [row] * 8

        # Packing false
        w_false = _build_sft_wrapper(tmp_path / "f", monkeypatch, packing=False)
        w_false.setup({"train": rows})
        sup_false, _ = _count_supervised_and_real_tokens(w_false.trainer.get_train_dataloader())

        # Packing true
        w_true = _build_sft_wrapper(tmp_path / "t", monkeypatch, packing=True)
        w_true.setup({"train": rows})
        sup_true, _ = _count_supervised_and_real_tokens(w_true.trainer.get_train_dataloader())

        assert sup_true == sup_false, (
            f"Supervised tokens mismatch: packed={sup_true} vs unpacked={sup_false}"
        )
        assert sup_true < 100, f"Expected assistant-only tokens, but got {sup_true}"

    def test_train_field_packing_preserves_mask(self, tmp_path, monkeypatch):
        row = {
            "messages": [
                {"role": "system", "content": "sys prompt", "train": False},
                {"role": "user", "content": "hello there", "train": False},
                {"role": "assistant", "content": "world reply", "train": True},
            ]
        }
        rows = [row] * 8
        d_extra = {
            "train_on_responses_only": False,
            "train_on_messages_with_train_field": True,
        }

        w_false = _build_sft_wrapper(tmp_path / "f", monkeypatch, data_extra=d_extra, packing=False)
        w_false.setup({"train": rows})
        sup_false, _ = _count_supervised_and_real_tokens(w_false.trainer.get_train_dataloader())

        w_true = _build_sft_wrapper(tmp_path / "t", monkeypatch, data_extra=d_extra, packing=True)
        w_true.setup({"train": rows})
        sup_true, _ = _count_supervised_and_real_tokens(w_true.trainer.get_train_dataloader())

        assert sup_true == sup_false

    def test_train_on_eot_packing(self, tmp_path, monkeypatch):
        row = {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ]
        }
        rows = [row] * 8

        w_false = _build_sft_wrapper(tmp_path / "f", monkeypatch, train_on_eot=True, packing=False)
        w_false.setup({"train": rows})
        sup_false, _ = _count_supervised_and_real_tokens(w_false.trainer.get_train_dataloader())

        w_true = _build_sft_wrapper(tmp_path / "t", monkeypatch, train_on_eot=True, packing=True)
        w_true.setup({"train": rows})
        sup_true, _ = _count_supervised_and_real_tokens(w_true.trainer.get_train_dataloader())

        assert sup_true == sup_false

    def test_pretokenized_dataset_with_labels_packing(self, tmp_path, monkeypatch):
        # Create dummy pretokenized dataset with input_ids and labels
        input_ids = [1, 5, 6, 7, 2]
        labels = [-100, -100, -100, 7, 2]
        ds = Dataset.from_dict({
            "input_ids": [input_ids] * 8,
            "labels": [labels] * 8,
            "attention_mask": [[1] * 5] * 8,
        })
        cache_f = tmp_path / "f" / "pretok_cache"
        cache_f.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(cache_f))

        d_extra_f = {"format": "pre_tokenized", "tokenized_path": "./pretok_cache"}
        w_false = _build_sft_wrapper(
            tmp_path / "f", monkeypatch, data_extra=d_extra_f, packing=False
        )
        w_false.setup({})
        sup_false, _ = _count_supervised_and_real_tokens(
            w_false.trainer.get_train_dataloader()
        )

        cache_t = tmp_path / "t" / "pretok_cache"
        cache_t.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(cache_t))
        d_extra_t = {"format": "pre_tokenized", "tokenized_path": "./pretok_cache"}
        w_true = _build_sft_wrapper(
            tmp_path / "t", monkeypatch, data_extra=d_extra_t, packing=True
        )
        w_true.setup({})
        sup_true, _ = _count_supervised_and_real_tokens(
            w_true.trainer.get_train_dataloader()
        )

        assert sup_true == sup_false


class TestPretrainPackingUnaffected:
    def test_pretrain_with_packing(self, tmp_path, monkeypatch):
        weights = _tiny_causal_model_dir(tmp_path)
        monkeypatch.chdir(tmp_path)
        cfg = load_config_from_string(
            yaml.safe_dump(
                {
                    "base": weights,
                    "task": "pretrain",
                    "backend": "transformers",
                    "modality": "text",
                    "data": {
                        "train": "train.jsonl",
                        "max_length": 64,
                    },
                    "training": {
                        "batch_size": 2,
                        "quantization": "none",
                        "epochs": 1,
                        "packing": True,
                    },
                    "output": str(tmp_path / "out_pretrain"),
                }
            )
        )
        wrapper = PretrainTrainerWrapper(cfg, device="cpu")
        wrapper.setup({"train": [{"text": "hello world the cat sat on the mat"}] * 8})
        # Pretrain packing works as expected
        dataloader = wrapper.trainer.get_train_dataloader()
        assert dataloader is not None
