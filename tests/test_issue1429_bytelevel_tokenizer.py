"""Issue #1429: `soup tokenizer train` must save a complete byte-level BPE.

A ByteLevel pre-tokenizer without a matching decoder decodes spaces as
"Ġ"; training without the 256-byte initial alphabet turns every character
absent from the corpus into `<unk>`. These tests pin the round-trip for
ASCII and unseen characters, both raw `tokenizers` and
`PreTrainedTokenizerFast`, and with a custom `--special-token` list.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi

_CORPUS = [
    "hello world number 1",
    "hello world number 2",
    "the quick brown fox jumps",
    "number 3 the quick fox",
] * 50

# Absent from the corpus: accented, emoji, CJK.
_UNSEEN = ["héllo", "\U0001F600", "世界"]


def _write_corpus(tmp_path: Path) -> str:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(json.dumps({"text": t}) for t in _CORPUS) + "\n",
        encoding="utf-8",
    )
    return "corpus.jsonl"


def _train(tmp_path, monkeypatch, *extra: str) -> Path:
    monkeypatch.chdir(tmp_path)
    _write_corpus(tmp_path)
    result = CliRunner().invoke(
        app,
        ["tokenizer", "train", "-i", "corpus.jsonl", "-v", "500",
         "-o", "tok", *extra],
    )
    assert result.exit_code == 0, result.output
    return tmp_path / "tok" / "tokenizer.json"


def test_round_trip_default_special_tokens(tmp_path, monkeypatch):
    tok_path = _train(tmp_path, monkeypatch)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tok_path))
    for text in ["hello world", "the quick brown fox", *_UNSEEN]:
        assert tok.decode(tok.encode(text).ids) == text


def test_round_trip_pretrained_tokenizer_fast(tmp_path, monkeypatch):
    from transformers import PreTrainedTokenizerFast

    monkeypatch.chdir(tmp_path)
    _write_corpus(tmp_path)
    result = CliRunner().invoke(
        app, ["tokenizer", "train", "-i", "corpus.jsonl", "-v", "500",
              "-o", "tok2"],
    )
    assert result.exit_code == 0, result.output
    fast = PreTrainedTokenizerFast(
        tokenizer_file=str(tmp_path / "tok2" / "tokenizer.json")
    )
    for text in ["hello world", *_UNSEEN]:
        assert fast.decode(fast(text)["input_ids"]) == text


def test_custom_special_tokens_no_unk_still_encodes(tmp_path, monkeypatch):
    tok_path = _train(tmp_path, monkeypatch, "--special-token", "<|endoftext|>")
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tok_path))
    assert "<unk>" not in tok.get_vocab()
    for text in _UNSEEN:
        assert "<unk>" not in tok.encode(text).tokens


def test_no_unk_token_in_any_encoded_sequence(tmp_path, monkeypatch):
    tok_path = _train(tmp_path, monkeypatch)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tok_path))
    for text in ["hello world", *_UNSEEN]:
        assert "<unk>" not in tok.encode(text).tokens


def test_saved_model_declares_no_unk_token(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_corpus(tmp_path)
    result = CliRunner().invoke(
        app,
        ["tokenizer", "train", "-i", "corpus.jsonl", "-v", "400", "-o", "tok",
         "--special-token", "<|endoftext|>"],
    )
    assert result.exit_code == 0, result.output
    model = json.loads((tmp_path / "tok" / "tokenizer.json").read_text(encoding="utf-8"))["model"]
    assert model["unk_token"] is None


@pytest.mark.parametrize(
    "vocab, extra_specials, ok",
    [
        (258, ("a", "b"), False),  # 256 + 2 specials = 258: zero merges
        (259, ("a", "b"), True),   # one merge fits
        (260, (), False),          # default four specials: 260 = zero merges
        (261, (), True),
    ],
)
def test_vocab_floor_is_alphabet_plus_specials_plus_one_merge(
    tmp_path, monkeypatch, vocab, extra_specials, ok
):
    monkeypatch.chdir(tmp_path)
    _write_corpus(tmp_path)
    args = ["tokenizer", "train", "-i", "corpus.jsonl", "-v", str(vocab), "-o", "tok"]
    for tok in extra_specials:
        args += ["--special-token", tok]
    result = CliRunner().invoke(app, args)
    text = " ".join(strip_ansi(result.output).split())
    if ok:
        assert result.exit_code == 0, text
    else:
        assert result.exit_code == 2, text
        assert f"--vocab-size must be >= {vocab + 1}" in text
