"""Tests for issue #1448: soup merge --save-format 4bit_forced.

Acceptance criteria:
- 4bit_forced sets llm_int8_skip_modules=[] in BitsAndBytesConfig kwargs.
- merge_4bit passes llm_int8_skip_modules=[] to BitsAndBytesConfig when forced=True.
- When forced=False, llm_int8_skip_modules is not set in kwargs.
"""

from unittest.mock import MagicMock, patch

import pytest

from soup_cli.utils.save_formats import _build_merge_4bit_bnb_kwargs, merge_4bit


def test_build_merge_4bit_bnb_kwargs_forced_sets_llm_int8_skip_modules():
    """forced=True must set llm_int8_skip_modules=[] so transformers quantizes all modules."""
    kw_forced = _build_merge_4bit_bnb_kwargs(
        compute_dtype="bfloat16", forced=True, double_quant=True
    )
    assert "llm_int8_skip_modules" in kw_forced
    assert kw_forced["llm_int8_skip_modules"] == []

    kw_unforced = _build_merge_4bit_bnb_kwargs(
        compute_dtype="bfloat16", forced=False, double_quant=True
    )
    assert "llm_int8_skip_modules" not in kw_unforced


def test_merge_4bit_forced_passes_llm_int8_skip_modules_to_transformers(tmp_path, monkeypatch):
    """merge_4bit(forced=True) must pass llm_int8_skip_modules=[] to BitsAndBytesConfig."""
    import sys
    import types

    monkeypatch.chdir(tmp_path)
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"

    fake_model = MagicMock()
    fake_tokenizer = MagicMock()
    mock_bnb_cls = MagicMock(return_value=MagicMock())
    fake_config = types.SimpleNamespace(tie_word_embeddings=False)

    fake_torch = types.SimpleNamespace(
        float16="float16",
        bfloat16="bfloat16",
        float32="float32",
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_transformers = types.SimpleNamespace(
        AutoConfig=MagicMock(from_pretrained=MagicMock(return_value=fake_config)),
        AutoModelForCausalLM=MagicMock(from_pretrained=MagicMock(return_value=fake_model)),
        AutoTokenizer=MagicMock(from_pretrained=MagicMock(return_value=fake_tokenizer)),
        BitsAndBytesConfig=mock_bnb_cls,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    merge_4bit(
        merged_dir=str(src),
        output_dir=str(out),
        forced=True,
        dtype="bfloat16",
        double_quant=True,
    )

    mock_bnb_cls.assert_called_once()
    _, kwargs = mock_bnb_cls.call_args
    assert kwargs.get("llm_int8_skip_modules") == []
    fake_transformers.AutoConfig.from_pretrained.assert_called_once_with(
        str(src), trust_remote_code=False
    )


def test_merge_4bit_forced_refuses_tied_embeddings(tmp_path, monkeypatch):
    """A tied lm_head shares the input-embedding weight, which a 4-bit Linear cannot hold:
    the checkpoint would load and then fail on its first forward pass. Refuse by name,
    before merge_4bit loads any weights."""
    pytest.importorskip("transformers")
    from transformers import LlamaConfig

    monkeypatch.chdir(tmp_path)
    LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                num_attention_heads=2, num_key_value_heads=2,
                tie_word_embeddings=True).save_pretrained("merged")
    with patch("transformers.AutoModelForCausalLM.from_pretrained") as load, \
            patch("transformers.AutoTokenizer.from_pretrained"):
        with pytest.raises(ValueError, match="4bit_forced") as err:
            merge_4bit(merged_dir="merged", output_dir="out", forced=True)
    assert "tie_word_embeddings" in str(err.value)
    load.assert_not_called()


def test_merge_4bit_plain_still_merges_tied_embeddings(tmp_path, monkeypatch):
    """The refusal belongs to 4bit_forced only: plain 4bit keeps BNB's default
    lm_head skip, so a tied model must still reach the weight load."""
    pytest.importorskip("transformers")
    from transformers import LlamaConfig

    monkeypatch.chdir(tmp_path)
    LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=True,
    ).save_pretrained("merged")
    with patch("transformers.AutoModelForCausalLM.from_pretrained") as load, \
            patch("transformers.AutoTokenizer.from_pretrained"), \
            patch("transformers.BitsAndBytesConfig"):
        merge_4bit(merged_dir="merged", output_dir="out", forced=False)
    load.assert_called_once()

