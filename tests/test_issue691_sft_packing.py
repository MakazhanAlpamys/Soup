"""#691 - SFT packing is passed to SFTTrainer instead of SFTConfig.

TRL 0.29 moved `packing` onto `SFTConfig`. Soup still does
`trainer_kwargs["packing"] = True`, which is a TypeError on every supported
trl. `packing_strategy="attention_free"` is also invalid (allowlist is
bfd / bfd-requeue / wrapped).

tests/test_packing.py stayed green by copying those kwargs into a MagicMock.
These tests use the real SFTConfig / SFTTrainer from setup().
"""

from __future__ import annotations

import json
import os
import sys
import types

try:
    import torchaudio  # noqa: F401
except ImportError:
    pass
except Exception:
    import importlib.machinery
    _torchaudio = types.ModuleType("torchaudio")
    _torchaudio.__version__ = "0.0.0"
    _torchaudio.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
    sys.modules["torchaudio"] = _torchaudio

import pytest
import yaml
from pydantic import ValidationError

from soup_cli.config.schema import SoupConfig

pytest.importorskip("torch")


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


def _tiny_causal_model_dir(tmp_path) -> str:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).to(torch.float32).eval()
    weights = tmp_path / "model"
    weights.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(weights)
    _write_tiny_tokenizer(str(weights))
    return str(weights)


def _sft_wrapper(tmp_path, monkeypatch, **training):
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.sft import SFTTrainerWrapper

    pytest.importorskip("trl")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")

    max_length = training.pop("max_length", 64)
    n_rows = training.pop("n_rows", 4)
    weights = _tiny_causal_model_dir(tmp_path)
    monkeypatch.chdir(tmp_path)
    tcfg = {
        "batch_size": 1,
        "quantization": "none",
        "epochs": 1,
        "logging_steps": 1,
        "save_steps": 1000,
        "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
    }
    tcfg.update(training)
    cfg = load_config_from_string(
        yaml.safe_dump(
            {
                "base": weights,
                "task": "sft",
                "backend": "transformers",
                "modality": "text",
                "data": {
                    "train": "train.jsonl",
                    "max_length": max_length,
                    "chat_template": "chatml",
                },
                "training": tcfg,
                "output": str(tmp_path / "out"),
            }
        )
    )
    wrapper = SFTTrainerWrapper(cfg, device="cpu")
    row = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ]
    }
    wrapper.setup({"train": [row] * n_rows})
    return wrapper, n_rows


def test_packing_cross_doc_rejected_names_trl_allowlist():
    with pytest.raises(ValidationError, match="bfd") as exc:
        SoupConfig(
            base="test-model",
            task="sft",
            data={"train": "data.jsonl"},
            training={"packing": True, "packing_cross_doc_attn_mask": True},
        )
    msg = str(exc.value)
    assert "bfd-requeue" in msg
    assert "wrapped" in msg


def test_setup_packing_true_lands_on_sft_config(tmp_path, monkeypatch):
    from trl import SFTConfig

    wrapper, _ = _sft_wrapper(tmp_path, monkeypatch, packing=True)
    args = wrapper.trainer.args
    assert isinstance(args, SFTConfig)
    assert args.packing is True
    assert args.packing_strategy in {"bfd", "bfd-requeue", "wrapped"}


def test_setup_packing_false_unchanged(tmp_path, monkeypatch):
    from trl import SFTConfig

    wrapper, _ = _sft_wrapper(tmp_path, monkeypatch, packing=False)
    args = wrapper.trainer.args
    assert isinstance(args, SFTConfig)
    assert args.packing is not True


def test_setup_packing_true_actually_packs(tmp_path, monkeypatch):
    wrapper, n_rows = _sft_wrapper(
        tmp_path, monkeypatch, packing=True, max_length=64, n_rows=8,
    )
    packed = wrapper.trainer.train_dataset
    if packed is not None and hasattr(packed, "__len__"):
        assert len(packed) < n_rows
        return
    batch = next(iter(wrapper.trainer.get_train_dataloader()))
    ids = batch["input_ids"]
    assert ids.shape[-1] == wrapper.trainer.args.max_length
