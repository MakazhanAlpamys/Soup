"""#690 - `_EmbeddingTrainer` does not expose `model` or `args`.

`EmbeddingTrainerWrapper.train()` reads `self.trainer.model` and
`self.trainer.args` before calling `train()`. The wrapper stored the HF
trainer in `_trainer` and delegated `train` / `save_model` / `add_callback` /
`state`, but not those two attributes.

`tests/test_embedding.py` stayed green because it assigns
`wrapper.trainer = MagicMock()`, which auto-creates both. These tests use
the real wrapper from `setup()`.
"""

from __future__ import annotations

import importlib.machinery
import json
import os
import sys
import types

try:
    import torchaudio  # noqa: F401
except ImportError:
    pass
except Exception:
    _torchaudio = types.ModuleType("torchaudio")
    _torchaudio.__version__ = "0.0.0"
    _torchaudio.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
    sys.modules["torchaudio"] = _torchaudio

import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("datasets")


def _write_tiny_tokenizer(directory: str) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    for word in ("hello", "world", "hi", "yo", "the", "cat", "sat", "on", "mat"):
        vocab[word] = len(vocab)
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(os.path.join(directory, "tokenizer.json"))
    with open(os.path.join(directory, "tokenizer_config.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "<unk>",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
                "model_max_length": 128,
                "clean_up_tokenization_spaces": False,
            },
            fh,
        )


def _tiny_embedding_model_dir(tmp_path) -> str:
    """Llama encoder weights for `AutoModel.from_pretrained` (not CausalLM)."""
    import torch
    from transformers import LlamaConfig, LlamaModel

    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = LlamaModel(config).to(torch.float32).eval()
    weights = tmp_path / "model"
    model.save_pretrained(weights)
    _write_tiny_tokenizer(str(weights))
    return str(weights)


def _embedding_wrapper(tmp_path, monkeypatch, *, precision=None):
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.embedding import EmbeddingTrainerWrapper

    if precision is not None:
        monkeypatch.setattr(
            "soup_cli.trainer.embedding.bf16_fp16_flags",
            lambda device, **kwargs: precision,
        )

    weights = _tiny_embedding_model_dir(tmp_path)
    monkeypatch.chdir(tmp_path)
    cfg = load_config_from_string(
        yaml.safe_dump(
            {
                "base": weights,
                "task": "embedding",
                "backend": "transformers",
                "modality": "text",
                "data": {
                    "train": "train.jsonl",
                    "format": "embedding",
                    "max_length": 64,
                },
                "training": {
                    "batch_size": 1,
                    "quantization": "none",
                    "epochs": 1,
                    "logging_steps": 1,
                    "save_steps": 1000,
                    "lora": {
                        "r": 4,
                        "alpha": 8,
                        "target_modules": ["q_proj", "v_proj"],
                    },
                },
                "output": str(tmp_path / "out"),
            }
        )
    )
    wrapper = EmbeddingTrainerWrapper(cfg, device="cpu")
    wrapper.setup(
        {
            "train": [
                {"anchor": "hello", "positive": "world"},
                {"anchor": "hi", "positive": "yo"},
            ]
        }
    )
    return wrapper


def _stub_inner_train(wrapper):
    inner = wrapper.trainer._trainer
    reached = []

    def train(*args, **kwargs):
        reached.append(("train", kwargs.get("resume_from_checkpoint")))
        return None

    def save_model(*args, **kwargs):
        reached.append("save")
        return None

    inner.train = train
    inner.save_model = save_model
    return reached


def test_train_after_setup_reaches_inner_trainer_precision_off(tmp_path, monkeypatch):
    from soup_cli.trainer.embedding import _EmbeddingTrainer

    wrapper = _embedding_wrapper(tmp_path, monkeypatch)
    assert isinstance(wrapper.trainer, _EmbeddingTrainer)
    assert wrapper.trainer.model is wrapper.model
    assert wrapper.trainer.args.fp16 is False
    assert wrapper.trainer.args.bf16 is False

    reached = _stub_inner_train(wrapper)
    wrapper.train()
    assert reached[0] == ("train", None)


def test_train_after_setup_reaches_inner_trainer_fp16(tmp_path, monkeypatch):
    from soup_cli.trainer.embedding import _EmbeddingTrainer

    wrapper = _embedding_wrapper(tmp_path, monkeypatch, precision=(False, True))
    assert isinstance(wrapper.trainer, _EmbeddingTrainer)
    assert wrapper.trainer.model is wrapper.model
    assert wrapper.trainer.args.fp16 is True
    assert wrapper.trainer.args.bf16 is False

    reached = _stub_inner_train(wrapper)
    wrapper.train()
    assert reached[0] == ("train", None)
