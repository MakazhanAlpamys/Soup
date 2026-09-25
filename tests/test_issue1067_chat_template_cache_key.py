"""#1067: ``data.chat_template`` was not part of the ``soup data preprocess`` cache.

Two configs differing only in the template resolved to the same cache key. And the
key was not the whole story: preprocess rendered every row with the tokenizer's
shipped template, whatever ``data.chat_template`` said, so the override was lost
even with no second config around. These run the real ``soup data preprocess``.
"""

import hashlib
import json

import pytest

from tests.test_issue1038_preprocess_to_pretokenized import _BASE, _ROW, _tokenizer

# Renders content only, so its tokens are distinguishable from the tokenizer's
# shipped template (which adds <|user|> / <|assistant|> role tags).
_OVERRIDE = "{% for m in messages %}{{ m['content'] }} <|end|> {% endfor %}"


def _config(train_format, chat_template, extra=""):
    template_line = (
        f"  chat_template: {json.dumps(chat_template)}\n" if chat_template else ""
    )
    return (
        f"base: {_BASE}\ntask: sft\ndata:\n  train: d.jsonl\n"
        f"  format: {train_format}\n{template_line}{extra}"
        "  max_length: 128\n  val_split: 0.0\n"
    )


def _preprocess(tmp_path, monkeypatch, chat_template=None):
    """Run the real ``soup data preprocess``; return the cache directory it wrote."""
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("datasets")
    from typer.testing import CliRunner

    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    (tmp_path / "d.jsonl").write_text(json.dumps(_ROW) + "\n", encoding="utf-8")
    tok = _tokenizer()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tok)
    (tmp_path / "soup.yaml").write_text(_config("chatml", chat_template), encoding="utf-8")
    result = CliRunner().invoke(app, ["data", "preprocess", "soup.yaml", "--yes"])
    assert result.exit_code == 0, result.output
    # the directory this run wrote, even when an earlier run in this cwd left one
    return max(
        (p for p in (tmp_path / ".soup-tokenized").iterdir() if p.is_dir()),
        key=lambda p: (p / "metadata.json").stat().st_mtime_ns,
    )


def _gate(tmp_path, cache_dir, chat_template=None):
    """Run the ``format: pre_tokenized`` training gate against ``cache_dir``."""
    from unittest.mock import MagicMock

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.sft import _maybe_load_pretokenized

    path = cache_dir.relative_to(tmp_path).as_posix()
    cfg = load_config_from_string(
        _config("pre_tokenized", chat_template, f"  tokenized_path: {path}\n")
        + "output: ./out\n"
    )
    return _maybe_load_pretokenized(cfg.data, cfg.base, MagicMock())


def _first_row(cache_dir):
    from datasets import load_from_disk

    return _tokenizer().decode(load_from_disk(str(cache_dir))[0]["input_ids"])


class TestTheTemplateIsInTheKey:
    def test_two_templates_do_not_share_a_cache(self, tmp_path, monkeypatch):
        """The collision itself. Fails if the template is dropped from the key."""
        default = _preprocess(tmp_path, monkeypatch)
        override = _preprocess(tmp_path, monkeypatch, _OVERRIDE)

        assert default.name != override.name
        assert sorted(p.name for p in (tmp_path / ".soup-tokenized").iterdir()) == sorted(
            [default.name, override.name]
        )


class TestPreprocessRendersTheTemplate:
    def test_cached_tokens_follow_data_chat_template(self, tmp_path, monkeypatch):
        """On ``main`` this cache holds ``<|user|> ... <|assistant|> ...``: the
        tokenizer's shipped template, with the override ignored."""
        text = _first_row(_preprocess(tmp_path, monkeypatch, _OVERRIDE))

        assert "What is the capital of France ?" in text
        assert "<|user|>" not in text
        assert "<|assistant|>" not in text

    def test_no_override_still_uses_the_shipped_template(self, tmp_path, monkeypatch):
        text = _first_row(_preprocess(tmp_path, monkeypatch))

        assert "<|user|>" in text


class TestTheGate:
    def test_the_same_template_reaches_training(self, tmp_path, monkeypatch):
        cache_dir = _preprocess(tmp_path, monkeypatch, _OVERRIDE)
        result = _gate(tmp_path, cache_dir, _OVERRIDE)

        assert result is not None
        train_ds, _ = result
        assert len(train_ds) == 1

    def test_a_different_template_is_refused(self, tmp_path, monkeypatch):
        """Training saves the tokenizer with the config's template, so a cache
        rendered with another one would train on one format and ship another."""
        cache_dir = _preprocess(tmp_path, monkeypatch, _OVERRIDE)
        with pytest.raises(ValueError, match="cache hash mismatch"):
            _gate(tmp_path, cache_dir)

    def test_a_cache_written_before_the_fix_is_refused(self, tmp_path, monkeypatch):
        """A v3 cache recorded no template, so the gate refuses it and says to
        re-run ``soup data preprocess`` rather than guessing what rendered it."""
        cache_dir = _preprocess(tmp_path, monkeypatch)
        metadata_path = cache_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        v3_blob = f"v3\x1fd.jsonl\x1f{_BASE}\x1f128\x1fchatml"
        metadata["cache_key"] = hashlib.sha256(v3_blob.encode("utf-8")).hexdigest()[:16]
        del metadata["chat_template"]
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        with pytest.raises(ValueError, match="re-run `soup data preprocess`") as exc:
            _gate(tmp_path, cache_dir)
        assert "predates chat_template keying" in str(exc.value)

    def test_a_current_cache_mismatch_does_not_claim_to_predate(
        self, tmp_path, monkeypatch
    ):
        cache_dir = _preprocess(tmp_path, monkeypatch, _OVERRIDE)
        with pytest.raises(ValueError, match="cache hash mismatch") as exc:
            _gate(tmp_path, cache_dir)
        assert "predates" not in str(exc.value)
