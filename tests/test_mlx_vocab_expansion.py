"""Native vocabulary expansion and trainable new tokens for MLX text SFT (#1994).

Tests deterministic token registration, additive parameter slices with bitwise
frozen base embeddings, end-to-end tiny Llama SFT training, artifact serialization
and loading, and incompatible consumer refusals.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.models.llama import Model, ModelArgs  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from soup_cli.config.schema import SoupConfig  # noqa: E402
from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper  # noqa: E402
from soup_cli.utils.mlx_vocab import (  # noqa: E402
    ExtendedEmbedding,
    ExtendedLinear,
    attach_mlx_vocab_extension,
    load_expanded_vocab_artifact,
    register_mlx_tokens,
)


def _make_tiny_llama(path: Path, *, tied: bool = True, vocab_size: int = 16) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    mx.random.seed(42)
    config = dict(
        model_type="llama",
        hidden_size=32,
        num_hidden_layers=1,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=vocab_size,
        tie_word_embeddings=tied,
    )
    args = ModelArgs(**config)
    model = Model(args)
    mx.save_safetensors(str(path / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    vocab = {"<unk>": 0, "<eos>": 1, "<pad>": 2}
    for i in range(3, vocab_size):
        vocab[f"word_{i}"] = i
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", eos_token="<eos>", pad_token="<pad>"
    )
    fast_tok.chat_template = (
        "{% for message in messages %}{{ message['role'] }} {{ message['content'] }} "
        "<eos> {% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    )
    fast_tok.save_pretrained(str(path))
    return path


class TestTokenRegistration:
    def test_empty_tokens(self, tmp_path):
        _make_tiny_llama(tmp_path)
        from soup_cli.utils.mlx import load_mlx_model

        _, tok = load_mlx_model(tmp_path, quantization="none")
        cfg = SimpleNamespace(add_new_tokens=None, new_special_tokens=None)
        delta_v, added = register_mlx_tokens(tok, cfg)
        assert delta_v == 0
        assert added == []

    def test_deduplication_and_precedence(self, tmp_path):
        _make_tiny_llama(tmp_path)
        from soup_cli.utils.mlx import load_mlx_model

        _, tok = load_mlx_model(tmp_path, quantization="none")
        cfg = SimpleNamespace(
            add_new_tokens=["<foo>", "<bar>", "<foo>"],
            new_special_tokens=["<spec1>", "<foo>", "<spec1>"],
        )
        delta_v, added = register_mlx_tokens(tok, cfg)
        assert delta_v == 3
        # <foo> appears in new_special_tokens, so special precedence applies
        assert added == ["<spec1>", "<foo>", "<bar>"]

    def test_existing_tokens_skipped(self, tmp_path):
        _make_tiny_llama(tmp_path)
        from soup_cli.utils.mlx import load_mlx_model

        _, tok = load_mlx_model(tmp_path, quantization="none")
        cfg = SimpleNamespace(
            add_new_tokens=["word_3", "word_4", "<custom_tok>"],
            new_special_tokens=[],
        )
        delta_v, added = register_mlx_tokens(tok, cfg)
        assert delta_v == 1
        assert added == ["<custom_tok>"]


class TestAdditiveParameterSlices:
    def test_extended_embedding_indexing_and_gradients(self):
        base_embed = nn.Embedding(8, 16)
        orig_base_w = mx.array(base_embed.weight)
        new_w = mx.random.normal((3, 16))
        orig_new_w = mx.array(new_w)
        ext = ExtendedEmbedding(base_embed, new_w)
        base_embed.freeze()

        # Check forward indexing
        x = mx.array([0, 7, 8, 10])
        out = ext(x)
        assert out.shape == (4, 16)

        # Autograd & optimizer update
        opt = optim.AdamW(learning_rate=0.05)

        def loss_fn(m, inp):
            return mx.sum(m(inp))

        loss_and_grad = nn.value_and_grad(ext, loss_fn)
        _, grads = loss_and_grad(ext, x)
        opt.update(ext, grads)

        # Base weights MUST stay 100% bitwise invariant
        assert mx.array_equal(base_embed.weight, orig_base_w).item()
        # New weights MUST update
        assert not mx.array_equal(ext.new_weights, orig_new_w).item()

    def test_extended_embedding_as_linear(self):
        base_embed = nn.Embedding(8, 16)
        orig_base_w = mx.array(base_embed.weight)
        new_w = mx.random.normal((3, 16))
        orig_new_w = mx.array(new_w)
        ext = ExtendedEmbedding(base_embed, new_w)
        base_embed.freeze()

        x = mx.random.normal((2, 4, 16))
        logits = ext.as_linear(x)
        assert logits.shape == (2, 4, 11)

        opt = optim.AdamW(learning_rate=0.05)

        def loss_fn(m, inp):
            return mx.sum(m.as_linear(inp))

        loss_and_grad = nn.value_and_grad(ext, loss_fn)
        _, grads = loss_and_grad(ext, x)
        opt.update(ext, grads)

        assert mx.array_equal(base_embed.weight, orig_base_w).item()
        assert not mx.array_equal(ext.new_weights, orig_new_w).item()

    def test_extended_linear_gradients(self):
        base_head = nn.Linear(16, 8, bias=False)
        orig_base_w = mx.array(base_head.weight)
        new_w = mx.random.normal((3, 16))
        orig_new_w = mx.array(new_w)
        ext = ExtendedLinear(base_head, new_w)
        base_head.freeze()

        x = mx.random.normal((2, 4, 16))
        logits = ext(x)
        assert logits.shape == (2, 4, 11)

        opt = optim.AdamW(learning_rate=0.05)

        def loss_fn(m, inp):
            return mx.sum(m(inp))

        loss_and_grad = nn.value_and_grad(ext, loss_fn)
        _, grads = loss_and_grad(ext, x)
        opt.update(ext, grads)

        assert mx.array_equal(base_head.weight, orig_base_w).item()
        assert not mx.array_equal(ext.new_weights, orig_new_w).item()


class TestEndToEndMLXSFTVocabExpansion:
    def test_tied_tiny_llama_training(self, tmp_path):
        base_path = tmp_path / "base"
        out_path = tmp_path / "adapter_out"
        _make_tiny_llama(base_path, tied=True, vocab_size=16)

        new_tokens = ["<special_new>", "<custom_new>"]
        cfg = SoupConfig(
            base=str(base_path),
            backend="mlx",
            output=str(out_path),
            data={
                "train": "unused.jsonl",
                "train_on_responses_only": False,
                "max_length": 64,
                "add_new_tokens": new_tokens,
            },
            training={
                "quantization": "none",
                "epochs": 1,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "lr": 0.05,
                "logging_steps": 1,
                "save_steps": 1,
                "scheduler": "constant",
                "lora": {"r": 4, "alpha": 8, "dropout": 0, "target_modules": "auto"},
            },
        )

        rows = {
            "train": [
                {"text": "word_3 <special_new> word_4 <custom_new>"},
                {"text": "<special_new> word_5 <custom_new> word_6"},
            ]
        }

        trainer = MLXSFTTrainerWrapper(cfg)
        trainer.setup(rows)

        base_embed_weight_before = mx.array(trainer.model.model.embed_tokens.weight)

        res = trainer.train()
        assert res["total_steps"] > 0
        assert out_path.exists()

        # 1. Base weights remained 100% invariant
        base_embed_weight_after = trainer.model.model.embed_tokens.base_embed.weight
        assert mx.array_equal(base_embed_weight_before, base_embed_weight_after).item()

        # 2. Check files created
        assert (out_path / "adapters.safetensors").is_file()
        assert (out_path / "added_tokens.safetensors").is_file()
        assert (out_path / "added_tokens_info.json").is_file()
        assert (out_path / "adapter_config.json").is_file()

        # 3. adapters.safetensors must contain ONLY LoRA keys
        adapter_weights = mx.load(str(out_path / "adapters.safetensors"))
        for k in adapter_weights:
            assert "lora" in k
            assert "new_weights" not in k
            assert "new_embed_tokens" not in k

        # 4. added_tokens.safetensors contains new_embed_tokens
        added_weights = mx.load(str(out_path / "added_tokens.safetensors"))
        assert "new_embed_tokens" in added_weights
        assert added_weights["new_embed_tokens"].shape == (2, 32)

        # 5. Metadata contents
        info = json.loads((out_path / "added_tokens_info.json").read_text(encoding="utf-8"))
        assert info["base_vocab_size"] == 16
        assert info["added_vocab_size"] == 2
        assert info["is_tied"] is True
        assert info["added_tokens"] == new_tokens

        adapter_cfg = json.loads((out_path / "adapter_config.json").read_text(encoding="utf-8"))
        assert adapter_cfg["vocab_expansion"] is True
        assert adapter_cfg["added_vocab_size"] == 2

        # 6. Verify reload via load_expanded_vocab_artifact
        from soup_cli.utils.mlx import load_mlx_model

        fresh_model, fresh_tok = load_mlx_model(base_path, quantization="none")
        is_tied, ext_embed, ext_head = load_expanded_vocab_artifact(
            out_path, fresh_model, fresh_tok
        )
        assert is_tied is True
        assert ext_head is None
        assert mx.array_equal(ext_embed.new_weights, added_weights["new_embed_tokens"]).item()

        # Model forward pass generates expanded vocabulary dimension logits
        sample_input = mx.array([[0, 16, 17]])
        logits = fresh_model(sample_input)
        assert logits.shape[-1] == 18

    def test_untied_tiny_llama_training(self, tmp_path):
        base_path = tmp_path / "base_untied"
        out_path = tmp_path / "adapter_out_untied"
        _make_tiny_llama(base_path, tied=False, vocab_size=16)

        new_tokens = ["<tok_alpha>", "<tok_beta>"]
        cfg = SoupConfig(
            base=str(base_path),
            backend="mlx",
            output=str(out_path),
            data={
                "train": "unused.jsonl",
                "train_on_responses_only": False,
                "max_length": 64,
                "add_new_tokens": new_tokens,
            },
            training={
                "quantization": "none",
                "epochs": 1,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "lr": 0.05,
                "logging_steps": 1,
                "save_steps": 1,
                "scheduler": "constant",
                "lora": {"r": 4, "alpha": 8, "dropout": 0, "target_modules": "auto"},
            },
        )

        rows = {
            "train": [
                {"text": "word_3 <tok_alpha> word_4 <tok_beta>"},
                {"text": "<tok_alpha> word_5 <tok_beta> word_6"},
            ]
        }

        trainer = MLXSFTTrainerWrapper(cfg)
        trainer.setup(rows)

        base_embed_weight_before = mx.array(trainer.model.model.embed_tokens.weight)
        base_head_weight_before = mx.array(trainer.model.lm_head.weight)

        trainer.train()

        # Base weights unchanged
        base_embed_after = trainer.model.model.embed_tokens.base_embed.weight
        base_head_after = trainer.model.lm_head.base_head.weight
        assert mx.array_equal(base_embed_weight_before, base_embed_after).item()
        assert mx.array_equal(base_head_weight_before, base_head_after).item()

        added_weights = mx.load(str(out_path / "added_tokens.safetensors"))
        assert "new_embed_tokens" in added_weights
        assert "new_lm_head" in added_weights
        assert added_weights["new_embed_tokens"].shape == (2, 32)
        assert added_weights["new_lm_head"].shape == (2, 32)

        info = json.loads((out_path / "added_tokens_info.json").read_text(encoding="utf-8"))
        assert info["is_tied"] is False

    def test_negative_control_frozen_slice_fails_update(self, tmp_path):
        """Negative control: if new slice is intentionally frozen, it does NOT update."""
        base_path = tmp_path / "base_neg"
        _make_tiny_llama(base_path, tied=True, vocab_size=16)

        from soup_cli.utils.mlx import load_mlx_model

        model, tok = load_mlx_model(base_path, quantization="none")
        is_tied, ext_embed, ext_head = attach_mlx_vocab_extension(model, delta_v=2)

        orig_new_w = mx.array(ext_embed.new_weights)
        # Freeze the new slice explicitly
        ext_embed.freeze()

        class Container(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.dummy = nn.Linear(32, 1)

            def __call__(self, x):
                return mx.sum(self.m(x)) + mx.sum(self.dummy(mx.zeros((1, 32))))

        container = Container(model)
        opt = optim.AdamW(learning_rate=0.05)
        loss_and_grad = nn.value_and_grad(container, lambda m, x: m(x))
        _, grads = loss_and_grad(container, mx.array([[0, 16, 17]]))
        opt.update(container, grads)

        # Confirm new weights did NOT update because they were frozen
        assert mx.array_equal(ext_embed.new_weights, orig_new_w).item()

    def test_quantized_model_refused_in_setup(self, tmp_path):
        base_path = tmp_path / "base_quantized"
        _make_tiny_llama(base_path, tied=True, vocab_size=16)

        cfg = SoupConfig(
            base=str(base_path),
            backend="mlx",
            output=str(tmp_path / "out"),
            data={
                "train": "unused.jsonl",
                "train_on_responses_only": False,
                "max_length": 64,
                "add_new_tokens": ["<new_tok>"],
            },
            training={
                "quantization": "none",
                "epochs": 1,
                "batch_size": 1,
                "lora": {"r": 4, "alpha": 8, "dropout": 0, "target_modules": "auto"},
            },
        )
        trainer = MLXSFTTrainerWrapper(cfg)
        # Manually quantize a layer on model to simulate a quantized base model
        nn.quantize(trainer.model, group_size=32, bits=4) if trainer.model else None

        # When setup runs on an already-quantized model, it raises ValueError
        from soup_cli.utils.mlx import load_mlx_model

        model, tok = load_mlx_model(base_path, quantization="none")
        nn.quantize(model, group_size=32, bits=4)

        # Monkeypatch load_mlx_model to return the quantized model
        import soup_cli.utils.mlx as mlx_mod

        original_load = mlx_mod.load_mlx_model
        try:
            mlx_mod.load_mlx_model = lambda *a, **kw: (model, tok)
            with pytest.raises(ValueError, match="quantized MLX model"):
                trainer.setup({"train": [{"text": "word_3"}]})
        finally:
            mlx_mod.load_mlx_model = original_load


class TestConsumerRefusals:
    def test_generation_read_native_adapter_refuses_expanded_vocab(self, tmp_path):
        from soup_cli.utils.mlx_generation import read_native_adapter

        # Case A: has added_tokens.safetensors
        (tmp_path / "added_tokens.safetensors").write_text("stub", encoding="utf-8")
        with pytest.raises(ValueError, match="Native MLX generation does not currently support"):
            read_native_adapter(tmp_path)

        (tmp_path / "added_tokens.safetensors").unlink()

        # Case B: has vocab_expansion: true in adapter_config.json
        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"model": "org/base", "vocab_expansion": True}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="Native MLX generation does not currently support"):
            read_native_adapter(tmp_path)

    def test_soup_merge_refuses_expanded_vocab(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from typer.testing import CliRunner

        from soup_cli.cli import app

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        out_dir = tmp_path / "out"

        # Case A: added_tokens.safetensors exists
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "org/base"}), encoding="utf-8"
        )
        (adapter_dir / "added_tokens.safetensors").write_text("stub", encoding="utf-8")

        runner = CliRunner()
        res = runner.invoke(
            app, ["merge", "--adapter", str(adapter_dir), "--output", str(out_dir)]
        )
        assert res.exit_code != 0
        refusal_msg = "soup merge does not support adapters with expanded vocabularies (#1988)"
        assert refusal_msg in res.output

        (adapter_dir / "added_tokens.safetensors").unlink()

        # Case B: vocab_expansion is true in adapter_config.json
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "org/base", "vocab_expansion": True}),
            encoding="utf-8",
        )
        res = runner.invoke(
            app, ["merge", "--adapter", str(adapter_dir), "--output", str(out_dir)]
        )
        assert res.exit_code != 0
        assert refusal_msg in res.output

    def test_padded_embeddings_unaligned_with_tokenizer_refused(self, tmp_path):
        base_path = tmp_path / "base_padded"
        _make_tiny_llama(base_path, tied=True, vocab_size=16)

        cfg = SoupConfig(
            base=str(base_path),
            backend="mlx",
            output=str(tmp_path / "out"),
            data={
                "train": "unused.jsonl",
                "train_on_responses_only": False,
                "max_length": 64,
                "add_new_tokens": ["<new_tok>"],
            },
            training={
                "quantization": "none",
                "epochs": 1,
                "batch_size": 1,
                "lora": {"r": 4, "alpha": 8, "dropout": 0, "target_modules": "auto"},
            },
        )
        trainer = MLXSFTTrainerWrapper(cfg)

        from soup_cli.utils.mlx import load_mlx_model

        model, tok = load_mlx_model(base_path, quantization="none")
        # Artificially expand model embed_tokens to 32 rows to simulate padded model
        model.model.embed_tokens = nn.Embedding(32, 32)

        import soup_cli.utils.mlx as mlx_mod

        orig_load = mlx_mod.load_mlx_model
        try:
            mlx_mod.load_mlx_model = lambda *a, **kw: (model, tok)
            with pytest.raises(ValueError, match="does not match base embedding table rows"):
                trainer.setup({"train": [{"text": "word_3"}]})
        finally:
            mlx_mod.load_mlx_model = orig_load
