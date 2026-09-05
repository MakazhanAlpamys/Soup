"""Tests for issue #700: task=embedding honours lora.r: 0 (full fine-tuning).

Verifies that:
1. lora.r: 0 on task: embedding trains without an adapter (no peft LoraConfig call).
2. No raw peft ValueError ('r' should be a positive integer value) escapes.
3. Real torch.nn.Module (without MagicMock) computes trainable parameters cleanly
   and does not attempt to call get_nb_trainable_parameters() on bare base models.
4. Console label displays "Full fine-tuning" instead of "LoRA applied".
5. Schema validation rejects invalid combinations (quantization, backend, conflicting LoRA flags).
6. _EmbeddingTrainer exposes model and args attributes.
"""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError

from soup_cli.config.schema import SoupConfig
from soup_cli.trainer.embedding import EmbeddingTrainerWrapper, _EmbeddingTrainer


class DummyEmbeddingModel(nn.Module):
    """Real torch.nn.Module imitating an embedding backbone without PEFT."""

    def __init__(self, vocab_size=100, hidden_size=32):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.embeddings(input_ids)
        x = self.encoder(x)
        from types import SimpleNamespace
        return SimpleNamespace(last_hidden_state=x)


class DummyTokenizer:
    """Minimal tokenizer for embedding tests."""

    def __init__(self):
        self.pad_token = "[PAD]"
        self.eos_token = "[EOS]"

    def __call__(self, texts, padding=True, truncation=True, max_length=128, return_tensors="pt"):
        n = len(texts)
        return {
            "input_ids": torch.ones((n, 4), dtype=torch.long),
            "attention_mask": torch.ones((n, 4), dtype=torch.long),
        }

    def save_pretrained(self, path):
        pass


class TestEmbeddingLoraZero:
    """Test suite for issue #700."""

    def test_schema_allows_embedding_with_lora_r_zero(self):
        """lora.r: 0 with task=embedding and quantization='none' should validate clean."""
        cfg = SoupConfig(
            base="BAAI/bge-base-en-v1.5",
            task="embedding",
            data={"train": "./data.jsonl", "format": "embedding"},
            training={"lora": {"r": 0}, "quantization": "none"},
        )
        assert cfg.training.lora.r == 0
        assert cfg.task == "embedding"

    def test_schema_rejects_quantization_with_lora_r_zero(self):
        """lora.r: 0 requires quantization='none' (default '4bit' must be rejected)."""
        with pytest.raises(ValidationError) as exc_info:
            SoupConfig(
                base="BAAI/bge-base-en-v1.5",
                task="embedding",
                data={"train": "./data.jsonl", "format": "embedding"},
                training={"lora": {"r": 0}},  # quantization defaults to '4bit'
            )
        err = str(exc_info.value)
        assert "lora.r=0" in err
        assert "quantization='none'" in err

    def test_schema_rejects_explicit_quantization_with_lora_r_zero(self):
        """lora.r: 0 explicitly paired with quantization='8bit' must be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SoupConfig(
                base="BAAI/bge-base-en-v1.5",
                task="embedding",
                data={"train": "./data.jsonl", "format": "embedding"},
                training={"lora": {"r": 0}, "quantization": "8bit"},
            )
        err = str(exc_info.value)
        assert "lora.r=0" in err
        assert "quantization='none'" in err

    def test_schema_rejects_unsloth_with_lora_r_zero(self):
        """lora.r: 0 requires backend='transformers'."""
        with pytest.raises(ValidationError) as exc_info:
            SoupConfig(
                base="BAAI/bge-base-en-v1.5",
                task="embedding",
                backend="unsloth",
                data={"train": "./data.jsonl", "format": "embedding"},
                training={"lora": {"r": 0}, "quantization": "none"},
            )
        err = str(exc_info.value)
        assert "lora.r=0" in err
        assert "backend='transformers'" in err

    def test_schema_rejects_lora_feature_flags_with_lora_r_zero(self):
        """lora.r: 0 is mutually exclusive with adapter-specific features like use_dora."""
        with pytest.raises(ValidationError) as exc_info:
            SoupConfig(
                base="BAAI/bge-base-en-v1.5",
                task="embedding",
                data={"train": "./data.jsonl", "format": "embedding"},
                training={"lora": {"r": 0, "use_dora": True}, "quantization": "none"},
            )
        err = str(exc_info.value)
        assert "lora.r=0 means full fine-tuning" in err
        assert "lora.use_dora" in err

    def test_setup_transformers_skips_peft_when_r_is_zero(self):
        """When lora.r is 0, peft.get_peft_model must NOT be called."""
        cfg = SoupConfig(
            base="BAAI/bge-base-en-v1.5",
            task="embedding",
            data={"train": "./data.jsonl", "format": "embedding"},
            training={"lora": {"r": 0}, "quantization": "none"},
        )
        real_model = DummyEmbeddingModel()

        with patch("transformers.AutoModel.from_pretrained", return_value=real_model), \
             patch("transformers.AutoTokenizer.from_pretrained", return_value=DummyTokenizer()), \
             patch("peft.get_peft_model") as mock_get_peft, \
             patch("peft.LoraConfig") as mock_lora_cfg:

            wrapper = EmbeddingTrainerWrapper(cfg, device="cpu")
            wrapper._setup_transformers(cfg, cfg.training)

            # PEFT must NOT have been called
            mock_lora_cfg.assert_not_called()
            mock_get_peft.assert_not_called()

            # The model is the original real PyTorch model (not a PeftModel)
            assert wrapper.model is real_model
            assert not hasattr(wrapper.model, "peft_config")

    def test_setup_computes_trainable_params_on_real_module_without_peft_helper(self, capsys):
        """Full fine-tuning prints 'Full fine-tuning' and computes params without PeftModel."""
        cfg = SoupConfig(
            base="BAAI/bge-base-en-v1.5",
            task="embedding",
            data={"train": "./data.jsonl", "format": "embedding"},
            training={"lora": {"r": 0}, "quantization": "none", "batch_size": 2},
        )
        real_model = DummyEmbeddingModel()
        real_tokenizer = DummyTokenizer()

        # Notice: real_model does NOT have get_nb_trainable_parameters
        assert not hasattr(real_model, "get_nb_trainable_parameters")

        with patch("transformers.AutoModel.from_pretrained", return_value=real_model), \
             patch("transformers.AutoTokenizer.from_pretrained", return_value=real_tokenizer):

            wrapper = EmbeddingTrainerWrapper(cfg, device="cpu")
            dataset = {
                "train": [
                    {"anchor": "query", "positive": "document"},
                    {"anchor": "what is soup", "positive": "soup is a library"},
                ]
            }
            # setup should complete without AttributeError
            wrapper.setup(dataset)

            assert wrapper.model is real_model
            # Model and args should be accessible on wrapper.trainer
            assert wrapper.trainer.model is real_model
            assert wrapper.trainer.args is not None

            # Verify console output says 'Full fine-tuning' and not 'LoRA applied'
            out = capsys.readouterr().out
            assert "Full fine-tuning" in out
            assert "LoRA applied" not in out

    def test_embedding_trainer_delegates_model_and_args(self):
        """_EmbeddingTrainer must expose model and args properties."""
        model = DummyEmbeddingModel()
        from transformers import TrainingArguments
        args = TrainingArguments(output_dir="./tmp_test_output", report_to="none")

        trainer = _EmbeddingTrainer(
            model=model,
            args=args,
            train_dataset=[],
            eval_dataset=None,
            processing_class=DummyTokenizer(),
            loss_type="contrastive",
            margin=0.5,
            pooling="mean",
            temperature=0.05,
            max_length=128,
        )
        # Must delegate model and args to the inner Trainer
        assert trainer.model is model
        assert trainer.args is args
