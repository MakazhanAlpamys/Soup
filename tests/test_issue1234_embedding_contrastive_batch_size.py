"""Tests for #1234: embedding contrastive in-batch negatives require batch_size >= 2."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string
from soup_cli.utils.gpu import model_size_from_name


def _tiny_bert_dir(tmp_path):
    """Build a tiny offline BERT model and tokenizer on CPU without downloading."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    model_dir = tmp_path / "tiny_bert"
    model_dir.mkdir(parents=True, exist_ok=True)

    config = BertConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
    )
    model = BertModel(config).eval()
    model.save_pretrained(str(model_dir))

    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "hello": 4, "world": 5, "python": 6}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )
    fast.save_pretrained(str(model_dir))
    return str(model_dir)


class TestParseTimeRefusal:
    """Explicit batch_size: 1 with embedding_loss: contrastive must be refused at parse time."""

    def test_refuse_explicit_batch_size_1_contrastive(self):
        cfg_str = """
base: test-model
task: embedding
data:
  train: dummy.jsonl
  max_length: 64
training:
  batch_size: 1
  embedding_loss: contrastive
"""
        with pytest.raises(ValueError) as excinfo:
            load_config_from_string(cfg_str)
        err = str(excinfo.value)
        assert "contrastive" in err
        assert "batch_size" in err
        assert "batch_size >= 2" in err

    def test_refuse_explicit_batch_size_1_with_gradient_accumulation(self):
        cfg_str = """
base: test-model
task: embedding
data:
  train: dummy.jsonl
  max_length: 64
training:
  batch_size: 1
  gradient_accumulation_steps: 4
  embedding_loss: contrastive
"""
        with pytest.raises(ValueError) as excinfo:
            load_config_from_string(cfg_str)
        err = str(excinfo.value)
        assert "contrastive" in err
        assert "batch_size" in err

    def test_allow_batch_size_1_cosine(self):
        cfg_str = """
base: test-model
task: embedding
data:
  train: dummy.jsonl
  max_length: 64
training:
  batch_size: 1
  embedding_loss: cosine
"""
        cfg = load_config_from_string(cfg_str)
        assert cfg.training.batch_size == 1
        assert cfg.training.embedding_loss == "cosine"

    def test_allow_batch_size_1_triplet(self):
        cfg_str = """
base: test-model
task: embedding
data:
  train: dummy.jsonl
  max_length: 64
training:
  batch_size: 1
  embedding_loss: triplet
"""
        cfg = load_config_from_string(cfg_str)
        assert cfg.training.batch_size == 1
        assert cfg.training.embedding_loss == "triplet"

    def test_allow_batch_size_2_contrastive(self):
        cfg_str = """
base: test-model
task: embedding
data:
  train: dummy.jsonl
  max_length: 64
training:
  batch_size: 2
  embedding_loss: contrastive
"""
        cfg = load_config_from_string(cfg_str)
        assert cfg.training.batch_size == 2
        assert cfg.training.embedding_loss == "contrastive"


class TestAutoBatchSizeFlooring:
    """batch_size: auto resolves to at least 2 when effective loss is contrastive."""

    @pytest.mark.parametrize("gib", [0, 16, 24, 40])
    def test_auto_batch_size_floored_at_2_contrastive(self, tmp_path, monkeypatch, gib):
        from soup_cli.trainer.embedding import EmbeddingTrainerWrapper

        model_path = _tiny_bert_dir(tmp_path)
        monkeypatch.chdir(tmp_path)

        cfg_dict = {
            "base": model_path,
            "task": "embedding",
            "data": {"train": "train.jsonl", "max_length": 64},
            "training": {
                "batch_size": "auto",
                "embedding_loss": "contrastive",
                "quantization": "none",
                "lora": {"r": 8, "alpha": 16, "target_modules": ["query", "value"]},
            },
            "output": str(tmp_path / "out"),
        }
        cfg = load_config_from_string(yaml.safe_dump(cfg_dict))

        mock_gpu = {"memory_total_bytes": int(gib * 1024**3)}
        with patch("soup_cli.utils.gpu.get_gpu_info", return_value=mock_gpu):
            wrapper = EmbeddingTrainerWrapper(cfg, device="cpu")
            dataset = {
                "train": [
                    {"anchor": "hello", "positive": "world"},
                    {"anchor": "hi", "positive": "there"},
                    {"anchor": "good", "positive": "day"},
                    {"anchor": "great", "positive": "job"},
                ]
            }
            wrapper.setup(dataset)
            assert wrapper.args.per_device_train_batch_size >= 2
            assert wrapper.args.dataloader_drop_last is True

    def test_triplet_fallback_to_contrastive_with_batch_size_1_rejected_at_setup(
        self, tmp_path, monkeypatch
    ):
        from soup_cli.trainer.embedding import EmbeddingTrainerWrapper

        model_path = _tiny_bert_dir(tmp_path)
        monkeypatch.chdir(tmp_path)

        cfg_dict = {
            "base": model_path,
            "task": "embedding",
            "data": {"train": "train.jsonl", "max_length": 64},
            "training": {
                "batch_size": 1,
                "embedding_loss": "triplet",
                "quantization": "none",
                "lora": {"r": 8, "alpha": 16, "target_modules": ["query", "value"]},
            },
            "output": str(tmp_path / "out"),
        }
        cfg = load_config_from_string(yaml.safe_dump(cfg_dict))
        wrapper = EmbeddingTrainerWrapper(cfg, device="cpu")
        # Dataset without 'negative' column triggers fallback to contrastive
        dataset = {
            "train": [
                {"anchor": "hello", "positive": "world"},
                {"anchor": "hi", "positive": "there"},
            ]
        }
        with pytest.raises(ValueError) as excinfo:
            wrapper.setup(dataset)
        err = str(excinfo.value)
        assert "contrastive in-batch negatives need batch_size >= 2" in err

    def test_triplet_with_negatives_at_batch_size_1_allowed(self, tmp_path, monkeypatch):
        from soup_cli.trainer.embedding import EmbeddingTrainerWrapper

        model_path = _tiny_bert_dir(tmp_path)
        monkeypatch.chdir(tmp_path)

        cfg_dict = {
            "base": model_path,
            "task": "embedding",
            "data": {"train": "train.jsonl", "max_length": 64},
            "training": {
                "batch_size": 1,
                "embedding_loss": "triplet",
                "quantization": "none",
                "lora": {"r": 8, "alpha": 16, "target_modules": ["query", "value"]},
            },
            "output": str(tmp_path / "out"),
        }
        cfg = load_config_from_string(yaml.safe_dump(cfg_dict))
        wrapper = EmbeddingTrainerWrapper(cfg, device="cpu")
        dataset = {
            "train": [
                {"anchor": "hello", "positive": "world", "negative": "bad"},
                {"anchor": "hi", "positive": "there", "negative": "wrong"},
            ]
        }
        wrapper.setup(dataset)
        assert wrapper.args.per_device_train_batch_size == 1
        assert wrapper.args.dataloader_drop_last is False


class TestRealCpuEmbeddingTraining:
    """Verify that template training block on CPU updates weights and has non-zero loss."""

    def test_cpu_template_training_trains_and_updates_lora(self, tmp_path, monkeypatch):
        from soup_cli.trainer.embedding import EmbeddingTrainerWrapper

        model_path = _tiny_bert_dir(tmp_path)
        monkeypatch.chdir(tmp_path)

        # Match the shipped template configuration
        cfg_dict = {
            "base": model_path,
            "task": "embedding",
            "data": {"train": "train.jsonl", "max_length": 64},
            "training": {
                "epochs": 1,
                "lr": 1e-4,
                "batch_size": "auto",
                "gradient_accumulation_steps": 1,
                "lora": {"r": 8, "alpha": 16, "target_modules": ["query", "value"]},
                "quantization": "none",
                "embedding_loss": "contrastive",
                "embedding_margin": 0.5,
                "embedding_pooling": "mean",
            },
            "output": str(tmp_path / "out"),
        }
        cfg = load_config_from_string(yaml.safe_dump(cfg_dict))
        wrapper = EmbeddingTrainerWrapper(cfg, device="cpu")

        dataset = {
            "train": [
                {"anchor": "hello world", "positive": "hello world"},
                {"anchor": "python programming", "positive": "python language"},
                {"anchor": "data science", "positive": "machine learning"},
                {"anchor": "good morning", "positive": "good day"},
                {"anchor": "computer vision", "positive": "image processing"},
            ]
        }
        wrapper.setup(dataset)
        assert wrapper.args.per_device_train_batch_size >= 2
        assert wrapper.args.dataloader_drop_last is True

        res = wrapper.train()
        assert res["total_steps"] > 0
        assert res["final_loss"] > 0.0

        # Verify LoRA B weights moved (not all zeros)
        moved = 0
        for name, param in wrapper.model.named_parameters():
            if "lora_B" in name:
                if (param != 0).any():
                    moved += 1
        assert moved > 0, "LoRA B weights should have been updated by training"


class TestRuntimeGuard:
    """Runtime guard in _compute_loss raises if similarity.size(0) < 2."""

    def test_compute_loss_raises_on_single_sample_batch(self, tmp_path, monkeypatch):
        import torch

        from soup_cli.trainer.embedding import EmbeddingTrainerWrapper

        model_path = _tiny_bert_dir(tmp_path)
        monkeypatch.chdir(tmp_path)

        cfg_dict = {
            "base": model_path,
            "task": "embedding",
            "data": {"train": "train.jsonl", "max_length": 64},
            "training": {
                "batch_size": 2,
                "embedding_loss": "contrastive",
                "quantization": "none",
                "lora": {"r": 8, "alpha": 16, "target_modules": ["query", "value"]},
            },
            "output": str(tmp_path / "out"),
        }
        cfg = load_config_from_string(yaml.safe_dump(cfg_dict))
        wrapper = EmbeddingTrainerWrapper(cfg, device="cpu")
        dataset = {
            "train": [
                {"anchor": "hello", "positive": "world"},
                {"anchor": "hi", "positive": "there"},
            ]
        }
        wrapper.setup(dataset)

        # Force a single-item input batch to compute_loss
        single_inputs = {
            "anchor_input_ids": torch.ones((1, 4), dtype=torch.long),
            "anchor_attention_mask": torch.ones((1, 4), dtype=torch.long),
            "positive_input_ids": torch.ones((1, 4), dtype=torch.long),
            "positive_attention_mask": torch.ones((1, 4), dtype=torch.long),
        }
        with pytest.raises(ValueError) as excinfo:
            wrapper.trainer.compute_loss(wrapper.model, single_inputs)
        assert "contrastive in-batch negatives need batch_size >= 2" in str(excinfo.value)


class TestModelSizeFromNameEncoder:
    """Encoder models (BGE, E5, GTE, etc.) resolve to accurate parameter sizes."""

    def test_bge_base_size(self):
        assert model_size_from_name("BAAI/bge-base-en-v1.5") == 0.11

    def test_bge_large_size(self):
        assert model_size_from_name("BAAI/bge-large-en-v1.5") == 0.335

    def test_bge_small_size(self):
        assert model_size_from_name("BAAI/bge-small-en-v1.5") == 0.033
