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

from typer.testing import CliRunner

from soup_cli.cli import app

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


def test_vocab_size_below_alphabet_plus_specials_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_corpus(tmp_path)
    result = CliRunner().invoke(
        app,
        ["tokenizer", "train", "-i", "corpus.jsonl", "-v", "257",
         "-o", "tok3", "--special-token", "<|endoftext|>", "--special-token",
         "<|endoftext2|>"],
    )
    # 256 bytes + 2 specials + merges do not fit in 257.
    assert result.exit_code != 0
