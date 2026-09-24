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
        # Numbered from 1, as trainer/sft.py numbers the same row.
        assert "Train row 2 cannot be tokenized" in out
        assert "Conversation roles must alternate" in out
        assert "first message: 'user': 'first'" in out

    def test_nothing_is_written(self, tmp_path, monkeypatch):
        """A partial cache is exactly what the old behaviour produced."""
        _preprocess(tmp_path, monkeypatch, [_ok(0), _TWO_USERS])

        assert not (tmp_path / ".soup-tokenized").exists()

    def test_a_tokenizer_without_a_chat_template_is_named(self, tmp_path, monkeypatch):
        """Reported once, as a dataset-wide problem: removing a row cannot fix it."""
        result, out = _preprocess(tmp_path, monkeypatch, [_ok(0), _ok(1)], template=None)

        assert result.exit_code == 1, out
        assert "has no chat_template" in out
        assert "Set data.chat_template" in out
        assert "Train row" not in out
        assert not (tmp_path / ".soup-tokenized").exists()

    def test_an_empty_conversation_is_refused(self, tmp_path, monkeypatch):
        """The live path stops on it too (``ValueError: messages list is empty``,
        measured with ``soup train`` on tiny-random-Llama)."""
        result, out = _preprocess(tmp_path, monkeypatch, [_ok(0), {"messages": []}])

        assert result.exit_code == 1, out
        assert "Train row 2 cannot be tokenized: it has no messages" in out

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
        assert "Train row 2 cannot be tokenized: the chat template rendered it as empty" in out

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
        assert "Train row 2 cannot be tokenized: ValueError: tokenizer refused this text" in out


@pytest.fixture
def plain_console(monkeypatch):
    """The command's console with no colour system, so Rich emits no escape code of
    its own and ANY ESC byte in the output came from the dataset. Without this the
    assertion depends on the runner: under FORCE_COLOR Rich's highlighter splits an
    injected ``\\x1b[2J`` into ``\\x1b\\x1b[1m[`` -- still a raw ESC, but no longer the
    sequence -- so a sequence check passes on the leaking code."""
    from rich.console import Console

    import soup_cli.commands.data as data_cmd

    monkeypatch.setattr(
        data_cmd, "console", Console(color_system=None, highlight=False, width=400)
    )


@pytest.mark.usefixtures("plain_console")
class TestTheMessageIsSafeForATerminal:
    """#1182 review: every part of the refusal comes from the dataset. A role of
    ``\\x1b[2J\\x1b[31mEVIL`` reached stdout raw (clear screen, then red), because
    ``rich.markup.escape`` neutralises ``[...]`` and nothing else."""

    def test_control_bytes_in_the_role_do_not_reach_the_terminal(self, tmp_path, monkeypatch):
        evil = {"messages": [
            {"role": "\x1b[2J\x1b[31mEVIL", "content": "x"},
            {"role": "assistant", "content": "Paris ."},
        ]}
        result, _ = _preprocess(tmp_path, monkeypatch, [_ok(0), evil])

        assert result.exit_code == 1, result.output
        # Valid only because plain_console pins colour off (see the fixture).
        assert "\x1b" not in result.output
        assert "EVIL" in result.output, "the row is still named"

    def test_control_bytes_in_the_reason_are_stripped_too(self, tmp_path, monkeypatch):
        transformers = pytest.importorskip("transformers")
        real = transformers.PreTrainedTokenizerFast.__call__

        def _boom(self, text, *a, **k):
            if "second" in str(text):
                raise ValueError("bad \x1b]0;TITLE\x07 text")
            return real(self, text, *a, **k)

        monkeypatch.setattr(transformers.PreTrainedTokenizerFast, "__call__", _boom)
        row = {"messages": [
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "Paris ."},
        ]}
        result, _ = _preprocess(tmp_path, monkeypatch, [_ok(0), row])

        assert result.exit_code == 1
        assert "\x1b" not in result.output
        assert "TITLE" in result.output

    def test_a_markup_role_prints_literally(self, tmp_path, monkeypatch):
        styled = {"messages": [
            {"role": "[bold red]x[/]", "content": "y"},
            {"role": "assistant", "content": "Paris ."},
        ]}
        result, out = _preprocess(tmp_path, monkeypatch, [_ok(0), styled])

        assert result.exit_code == 1
        assert "[bold red]x[/]" in out


class TestPretrainIsUnchanged:
    def test_a_pretrain_row_the_tokenizer_rejects_is_still_skipped(self, tmp_path, monkeypatch):
        """The deliberate carve-out: a raw-text pretrain row the tokenizer rejects
        is skipped, as before, and the rest are written."""
        transformers = pytest.importorskip("transformers")
        pytest.importorskip("datasets")
        from typer.testing import CliRunner

        from soup_cli.cli import app

        real = transformers.PreTrainedTokenizerFast.__call__

        def _boom(self, text, *a, **k):
            if "reject" in str(text):
                raise ValueError("no")
            return real(self, text, *a, **k)

        monkeypatch.setattr(transformers.PreTrainedTokenizerFast, "__call__", _boom)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "d.jsonl").write_text(
            "".join(json.dumps({"text": t}) + "\n" for t in ("Paris .", "reject me", "What ?")),
            encoding="utf-8",
        )
        tok = _tokenizer()
        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tok)
        (tmp_path / "soup.yaml").write_text(
            "base: x/tiny\ntask: pretrain\n"
            "data:\n  train: d.jsonl\n  format: plaintext\n  max_length: 64\n  val_split: 0\n"
            "training:\n  epochs: 1\n",
            encoding="utf-8",
        )
        result = CliRunner().invoke(app, ["data", "preprocess", "soup.yaml", "--yes"])

        assert result.exit_code == 0, result.output
        (cache,) = (p for p in (tmp_path / ".soup-tokenized").iterdir() if p.is_dir())
        metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["row_count"] == 2


class TestTheControls:
    def test_every_row_valid_writes_every_row(self, tmp_path, monkeypatch):
        result, out = _preprocess(tmp_path, monkeypatch, [_ok(i) for i in range(4)])

        assert result.exit_code == 0, out
        (cache,) = (p for p in (tmp_path / ".soup-tokenized").iterdir() if p.is_dir())
        metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["row_count"] == 4
        assert "cannot be tokenized" not in out
