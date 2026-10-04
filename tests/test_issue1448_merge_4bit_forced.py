"""Tests for issue #1448: soup merge --save-format 4bit_forced.

Acceptance criteria:
- 4bit_forced sets llm_int8_skip_modules=[] in BitsAndBytesConfig kwargs.
- merge_4bit passes llm_int8_skip_modules=[] to BitsAndBytesConfig when forced=True.
- When forced=False, llm_int8_skip_modules is not set in kwargs.
"""

from unittest.mock import MagicMock

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

    fake_torch = types.SimpleNamespace(
        float16="float16",
        bfloat16="bfloat16",
        float32="float32",
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_transformers = types.SimpleNamespace(
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
