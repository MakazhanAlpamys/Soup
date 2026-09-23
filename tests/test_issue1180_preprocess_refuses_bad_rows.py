"""#1180: ``soup data preprocess`` skipped every chat row it could not tokenize.

A row whose template raised was dropped with a bare ``continue`` and the command
printed only the count it kept, exiting 0. The live ``soup train`` path stops on
the same row (measured on ``hf-internal-testing/tiny-random-LlamaForCausalLM``:
``TemplateError: Conversation roles must alternate ...``), so a cached run
trained on fewer rows than the live run of the same ``soup.yaml``, silently.

These run the real ``soup data preprocess``. The tokenizer is local, with a
strict template built the way Llama-2 / Gemma templates are: it calls
``raise_exception`` when the roles do not alternate.
"""

from __future__ import annotations

import json

import pytest

from tests.conftest import strip_ansi
from tests.test_issue1038_preprocess_to_pretokenized import _tokenizer

_STRICT = (
    "{% for m in messages %}"
    "{% if (m['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{% endif %}"
    "{% if m['role'] == 'user' %}<s> {{ m['content'] }}"
    "{% else %} {{ m['content'] }} </s>{% endif %}"
    "{% endfor %}"
)


def _ok(i):
    return {"messages": [
        {"role": "user", "content": f"What is {i} ?"},
        {"role": "assistant", "content": "Paris ."},
    ]}


_TWO_USERS = {"messages": [
    {"role": "user", "content": "first"},
    {"role": "user", "content": "second"},
    {"role": "assistant", "content": "Paris ."},
]}


def _preprocess(tmp_path, monkeypatch, rows, *, template=_STRICT):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("datasets")
    from typer.testing import CliRunner

    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    (tmp_path / "d.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    tok = _tokenizer()
    tok.chat_template = template
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tok)
    (tmp_path / "soup.yaml").write_text(
        "base: x/tiny\ntask: sft\n"
        "data:\n  train: d.jsonl\n  format: chatml\n  max_length: 64\n  val_split: 0\n"
        "training:\n  epochs: 1\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["data", "preprocess", "soup.yaml", "--yes"])
    return result, " ".join(strip_ansi(result.output).split())


class TestABadRowStopsTheCommand:
    def test_a_non_alternating_row_is_refused_and_named(self, tmp_path, monkeypatch):
        result, out = _preprocess(tmp_path, monkeypatch, [_ok(0), _TWO_USERS, _ok(2)])

        assert result.exit_code == 1, out
        assert "Train row 1 cannot be tokenized" in out
        assert "Conversation roles must alternate" in out
        assert "first message: user: 'first'" in out

    def test_nothing_is_written(self, tmp_path, monkeypatch):
        """A partial cache is exactly what the old behaviour produced."""
        _preprocess(tmp_path, monkeypatch, [_ok(0), _TWO_USERS])

        assert not (tmp_path / ".soup-tokenized").exists()

    def test_a_tokenizer_without_a_chat_template_is_named(self, tmp_path, monkeypatch):
        result, out = _preprocess(tmp_path, monkeypatch, [_ok(0)], template=None)

        assert result.exit_code == 1, out
        assert "has no chat_template" in out

    def test_an_empty_conversation_is_refused(self, tmp_path, monkeypatch):
        """The live path stops on it too (``ValueError: messages list is empty``,
        measured with ``soup train`` on tiny-random-Llama)."""
        result, out = _preprocess(tmp_path, monkeypatch, [_ok(0), {"messages": []}])

        assert result.exit_code == 1, out
        assert "Train row 1 cannot be tokenized: it has no messages" in out

    def test_a_row_the_template_renders_empty_is_refused(self, tmp_path, monkeypatch):
        blank_for_one = (
            "{% if messages[0]['content'] != 'blank' %}" + _STRICT + "{% endif %}"
        )
        blank = {"messages": [
            {"role": "user", "content": "blank"},
            {"role": "assistant", "content": "Paris ."},
        ]}
        result, out = _preprocess(
            tmp_path, monkeypatch, [_ok(0), blank], template=blank_for_one
        )

        assert result.exit_code == 1, out
        assert "Train row 1 cannot be tokenized: the chat template rendered it as empty" in out

    def test_a_tokenizer_error_is_refused_too(self, tmp_path, monkeypatch):
        """The same shape one call later: the tokenizer itself raising."""
        transformers = pytest.importorskip("transformers")
        real = transformers.PreTrainedTokenizerFast.__call__

        def _boom(self, text, *a, **k):
            if "second" in str(text):
                raise ValueError("tokenizer refused this text")
            return real(self, text, *a, **k)

        monkeypatch.setattr(transformers.PreTrainedTokenizerFast, "__call__", _boom)
        ok_second = {"messages": [
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "Paris ."},
        ]}
        result, out = _preprocess(tmp_path, monkeypatch, [_ok(0), ok_second])

        assert result.exit_code == 1, out
        assert "Train row 1 cannot be tokenized: ValueError: tokenizer refused this text" in out


class TestTheControls:
    def test_every_row_valid_writes_every_row(self, tmp_path, monkeypatch):
        result, out = _preprocess(tmp_path, monkeypatch, [_ok(i) for i in range(4)])

        assert result.exit_code == 0, out
        (cache,) = (p for p in (tmp_path / ".soup-tokenized").iterdir() if p.is_dir())
        metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["row_count"] == 4
        assert "cannot be tokenized" not in out
