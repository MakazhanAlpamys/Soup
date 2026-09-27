"""#1038: ``format: pre_tokenized`` rejected every cache ``soup data preprocess`` wrote.

Preprocess hashes the SOURCE format (``chatml``, ...) into the cache key
(``commands/data.py``); the training gate (``trainer/sft.py``
``_maybe_load_pretokenized``) recomputed it with ``dcfg.format``, which must be
``pre_tokenized`` for the gate to run at all. The keys never matched, so the
documented flow ended in "cache hash mismatch; re-run `soup data preprocess`",
and re-running wrote the same key again. A list ``data.train`` did not even get
that far: the gate passed the raw list where preprocess passes a JSON blob, and
``make_preprocess_cache_key`` raised ``ValueError``.

The existing gate tests built ``metadata.json`` by hand with the gate's own
``format_name="pre_tokenized"``, so they agreed with the gate by construction.
These tests run the real ``soup data preprocess`` and hand its output to the gate.
"""

import json

import pytest

_SPECIALS = ["<unk>", "<s>", "</s>", "<|user|>", "<|assistant|>", "<|end|>"]
_WORDS = ["What", "is", "the", "capital", "of", "France", "?", "Paris", "."]
_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} <|end|> {% endfor %}"
)
_ROW = {
    "messages": [
        {"role": "user", "content": "What is the capital of France ?"},
        {"role": "assistant", "content": "Paris ."},
    ]
}
_BASE = "x/tiny"


def _tokenizer():
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import models, pre_tokenizers

    vocab = {token: index for index, token in enumerate(_SPECIALS + _WORDS)}
    backend = tokenizers.Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.add_special_tokens(_SPECIALS)
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", bos_token="<s>", eos_token="</s>"
    )
    tok.chat_template = _TEMPLATE
    return tok


def _train_yaml(train, *, max_length=128, base=_BASE, interleave="concat"):
    if isinstance(train, list):
        train_block = (
            "  train:\n" + "".join(f"    - {p}\n" for p in train)
            + f"  interleave: {interleave}\n"
        )
    else:
        train_block = f"  train: {train}\n"
    return train_block, max_length, base


def _preprocess(tmp_path, monkeypatch, train, *, rows_per_file=4, max_length=128):
    """Run the real ``soup data preprocess``; return the cache directory it wrote."""
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("datasets")
    from typer.testing import CliRunner

    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    for path in train if isinstance(train, list) else [train]:
        (tmp_path / path).write_text(
            "".join(json.dumps(_ROW) + "\n" for _ in range(rows_per_file)),
            encoding="utf-8",
        )
    tok = _tokenizer()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tok)
    train_block, _, _ = _train_yaml(train)
    (tmp_path / "soup.yaml").write_text(
        f"base: {_BASE}\ntask: sft\ndata:\n{train_block}"
        f"  format: chatml\n  max_length: {max_length}\n  val_split: 0.0\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["data", "preprocess", "soup.yaml", "--yes"])
    assert result.exit_code == 0, result.output
    (cache_dir,) = [p for p in (tmp_path / ".soup-tokenized").iterdir() if p.is_dir()]
    assert (cache_dir / "metadata.json").is_file(), "preprocess must write metadata.json"
    return cache_dir


def _gate(tmp_path, train, cache_dir, *, max_length=128, base=_BASE, interleave="concat"):
    """Load a ``format: pre_tokenized`` config and run the training gate on it."""
    from unittest.mock import MagicMock

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.sft import _maybe_load_pretokenized

    train_block, _, _ = _train_yaml(train, interleave=interleave)
    cfg = load_config_from_string(
        f"base: {base}\ntask: sft\ndata:\n{train_block}"
        "  format: pre_tokenized\n"
        f"  tokenized_path: {cache_dir.relative_to(tmp_path).as_posix()}\n"
        f"  max_length: {max_length}\n  val_split: 0.0\n"
        "output: ./out\n"
    )
    return _maybe_load_pretokenized(cfg.data, cfg.base, MagicMock(), task="sft")


class TestTheRoundTrip:
    def test_a_single_file_cache_reaches_training(self, tmp_path, monkeypatch):
        """The documented flow, end to end. Fails on ``main`` with
        ``pre_tokenized cache hash mismatch``."""
        cache_dir = _preprocess(tmp_path, monkeypatch, "d.jsonl")
        result = _gate(tmp_path, "d.jsonl", cache_dir)

        assert result is not None
        train_ds, _ = result
        assert len(train_ds) == 4
        assert len(train_ds[0]["input_ids"]) > 0

    def test_a_list_train_cache_reaches_training(self, tmp_path, monkeypatch):
        """#443's list ``data.train``. On ``main`` the gate passed the raw list to
        ``make_preprocess_cache_key`` and raised ``ValueError: dataset_path must be
        a non-empty string`` before it could even compare keys."""
        train = ["a.jsonl", "b.jsonl"]
        cache_dir = _preprocess(tmp_path, monkeypatch, train)
        result = _gate(tmp_path, train, cache_dir)

        assert result is not None
        train_ds, _ = result
        assert len(train_ds) > 0


class TestTheGateStillRefuses:
    """The fix must make the gate agree with preprocess, not stop it firing."""

    def test_a_different_max_length(self, tmp_path, monkeypatch):
        cache_dir = _preprocess(tmp_path, monkeypatch, "d.jsonl", max_length=128)
        with pytest.raises(ValueError, match="cache hash mismatch"):
            _gate(tmp_path, "d.jsonl", cache_dir, max_length=256)

    def test_a_different_tokenizer(self, tmp_path, monkeypatch):
        cache_dir = _preprocess(tmp_path, monkeypatch, "d.jsonl")
        with pytest.raises(ValueError, match="cache hash mismatch"):
            _gate(tmp_path, "d.jsonl", cache_dir, base="x/other")

    def test_a_different_dataset(self, tmp_path, monkeypatch):
        cache_dir = _preprocess(tmp_path, monkeypatch, "d.jsonl")
        (tmp_path / "other.jsonl").write_text(json.dumps(_ROW) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="cache hash mismatch"):
            _gate(tmp_path, "other.jsonl", cache_dir)

    def test_a_different_list_order(self, tmp_path, monkeypatch):
        """The list key is order-sensitive in preprocess (#443: interleave mixes
        in list order), so the gate must refuse a reordered list."""
        cache_dir = _preprocess(tmp_path, monkeypatch, ["a.jsonl", "b.jsonl"])
        with pytest.raises(ValueError, match="cache hash mismatch"):
            _gate(tmp_path, ["b.jsonl", "a.jsonl"], cache_dir)

    def test_a_different_interleave_strategy(self, tmp_path, monkeypatch):
        """#443: the same files under a different mixture are a different dataset.
        No test covered this before -- dropping ``interleave`` from the list key
        left the whole suite green, on ``main`` and here."""
        train = ["a.jsonl", "b.jsonl"]
        cache_dir = _preprocess(tmp_path, monkeypatch, train)
        with pytest.raises(ValueError, match="cache hash mismatch"):
            _gate(tmp_path, train, cache_dir, interleave="over")

    def test_a_tampered_source_format(self, tmp_path, monkeypatch):
        """The stored ``format`` is an input to the recomputed key, not a value the
        gate trusts on its own: editing it without the key must refuse."""
        cache_dir = _preprocess(tmp_path, monkeypatch, "d.jsonl")
        meta_path = cache_dir / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["format"] == "chatml", "preprocess records the source format"
        meta["format"] = "alpaca"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="cache hash mismatch"):
            _gate(tmp_path, "d.jsonl", cache_dir)


class TestMetadata:
    def test_soup_version_is_the_installed_one(self, tmp_path, monkeypatch):
        """``metadata.json`` hardcoded ``"soup_version": "0.53.7"`` on every
        install. It is a record of what wrote the cache, so it must be real."""
        import soup_cli

        cache_dir = _preprocess(tmp_path, monkeypatch, "d.jsonl")
        meta = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
        assert meta["soup_version"] == soup_cli.__version__
