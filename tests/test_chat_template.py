"""Tests for chat-template hardening (v0.36.0 Part C).

Replaces sft.py's silent ``f"{role}: {content}"`` fallback (which produced
garbage training data on tokenizers without ``chat_template``) with a hard
error and an explicit ``DataConfig.chat_template`` field for overrides.
"""

from __future__ import annotations

import pytest

from tests.conftest import strip_ansi

# ---------------------------------------------------------------------------
# Schema field
# ---------------------------------------------------------------------------


class TestSchemaField:
    def test_chat_template_default_none(self):
        from soup_cli.config.schema import DataConfig

        cfg = DataConfig(train="data.jsonl")
        assert cfg.chat_template is None

    def test_chat_template_accepts_registered_name(self):
        from soup_cli.config.schema import DataConfig

        cfg = DataConfig(train="data.jsonl", chat_template="chatml")
        assert cfg.chat_template == "chatml"

    def test_chat_template_accepts_jinja_string(self):
        from soup_cli.config.schema import DataConfig

        jinja = "{% for m in messages %}{{ m.role }}: {{ m.content }}{% endfor %}"
        cfg = DataConfig(train="data.jsonl", chat_template=jinja)
        assert cfg.chat_template == jinja

    def test_chat_template_rejects_null_byte(self):
        from soup_cli.config.schema import DataConfig

        with pytest.raises(ValueError):
            DataConfig(train="data.jsonl", chat_template="bad\x00template")

    def test_chat_template_rejects_oversize(self):
        from soup_cli.config.schema import DataConfig

        # Cap at 64KB to prevent template-injection DoS payloads.
        with pytest.raises(ValueError):
            DataConfig(train="data.jsonl", chat_template="x" * 100_000)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_lists_known_templates(self):
        from soup_cli.data.chat_templates import list_template_names

        names = list_template_names()
        # At minimum: the 7 declared in v0.36.0 Part C.
        for required in (
            "chatml", "llama3", "qwen2.5", "gemma3", "phi4", "deepseek-r1", "mistral",
        ):
            assert required in names

    def test_chatml_is_jinja_string(self):
        from soup_cli.data.chat_templates import get_template

        tmpl = get_template("chatml")
        assert isinstance(tmpl, str)
        assert "{%" in tmpl
        # ChatML signature markers.
        assert "<|im_start|>" in tmpl
        assert "<|im_end|>" in tmpl

    def test_unknown_name_raises(self):
        from soup_cli.data.chat_templates import get_template

        with pytest.raises(KeyError, match="not registered"):
            get_template("not-a-real-template")

    def test_resolve_returns_jinja_for_known_name(self):
        from soup_cli.data.chat_templates import resolve_chat_template

        out = resolve_chat_template("chatml")
        assert "<|im_start|>" in out

    def test_resolve_returns_passthrough_for_jinja(self):
        from soup_cli.data.chat_templates import resolve_chat_template

        jinja = "{% for m in messages %}<x>{{ m.content }}</x>{% endfor %}"
        out = resolve_chat_template(jinja)
        assert out == jinja

    def test_resolve_none_returns_none(self):
        from soup_cli.data.chat_templates import resolve_chat_template

        assert resolve_chat_template(None) is None

    def test_resolve_empty_returns_none(self):
        from soup_cli.data.chat_templates import resolve_chat_template

        assert resolve_chat_template("") is None

    def test_resolve_unknown_name_raises(self):
        """Public surface: bad name through resolve_chat_template."""
        from soup_cli.data.chat_templates import resolve_chat_template

        with pytest.raises(KeyError, match="not registered"):
            resolve_chat_template("not-a-real-template")


# ---------------------------------------------------------------------------
# Schema rejects Jinja directives that touch the filesystem
# ---------------------------------------------------------------------------


class TestJinjaDirectiveBlocking:
    @pytest.mark.parametrize(
        "bad",
        [
            "{% include 'config.yaml' %}",
            "{%- include 'config.yaml' %}",
            "{% import 'os' as os %}",
            "{% from 'os' import system %}",
            "{% macro evil() %}{% endmacro %}",
            "{% extends 'base.j2' %}",
        ],
    )
    def test_schema_rejects_filesystem_directives(self, bad):
        from soup_cli.config.schema import DataConfig

        with pytest.raises(ValueError, match="directive"):
            DataConfig(train="data.jsonl", chat_template=bad)

    def test_schema_accepts_for_loop(self):
        """Standard control flow must still work."""
        from soup_cli.config.schema import DataConfig

        ok = "{% for m in messages %}{{ m.content }}{% endfor %}"
        cfg = DataConfig(train="data.jsonl", chat_template=ok)
        assert cfg.chat_template == ok

    def test_schema_accepts_if_else(self):
        from soup_cli.config.schema import DataConfig

        ok = "{% if true %}x{% else %}y{% endif %}"
        cfg = DataConfig(train="data.jsonl", chat_template=ok)
        assert cfg.chat_template == ok

    def test_schema_empty_string_normalised_to_none(self):
        from soup_cli.config.schema import DataConfig

        cfg = DataConfig(train="data.jsonl", chat_template="")
        assert cfg.chat_template is None


# ---------------------------------------------------------------------------
# apply_chat_template_override
# ---------------------------------------------------------------------------


class TestApplyOverride:
    def test_sets_tokenizer_chat_template(self):
        from soup_cli.data.chat_templates import apply_chat_template_override

        class _T:
            chat_template = None

        tok = _T()
        apply_chat_template_override(tok, "chatml")
        assert tok.chat_template is not None
        assert "<|im_start|>" in tok.chat_template

    def test_none_leaves_tokenizer_alone(self):
        from soup_cli.data.chat_templates import apply_chat_template_override

        class _T:
            chat_template = "existing"

        tok = _T()
        apply_chat_template_override(tok, None)
        assert tok.chat_template == "existing"

    def test_empty_leaves_tokenizer_alone(self):
        from soup_cli.data.chat_templates import apply_chat_template_override

        class _T:
            chat_template = "existing"

        tok = _T()
        apply_chat_template_override(tok, "")
        assert tok.chat_template == "existing"

    def test_override_emits_save_pretrained_warning(self):
        """v0.36.0 review fix: warn when overriding so users know push will
        persist the new template into tokenizer_config.json."""
        from io import StringIO

        from rich.console import Console

        from soup_cli.data.chat_templates import apply_chat_template_override

        class _T:
            chat_template = None

        buf = StringIO()
        console = Console(file=buf, force_terminal=False)
        applied = apply_chat_template_override(_T(), "chatml", console=console)
        assert applied is True
        assert "save_pretrained" in buf.getvalue() or "soup push" in buf.getvalue()

    def test_override_returns_false_when_noop(self):
        from soup_cli.data.chat_templates import apply_chat_template_override

        class _T:
            chat_template = "existing"

        applied = apply_chat_template_override(_T(), None)
        assert applied is False


# ---------------------------------------------------------------------------
# Hard error in sft_format when no chat_template AND no override
# ---------------------------------------------------------------------------


class TestHardError:
    def test_no_template_no_override_raises(self):
        """The legacy `f"{role}: {content}"` silent fallback is now an error."""
        from soup_cli.config.schema import DataConfig
        from soup_cli.data.sft_format import build_format_row

        class _NoTemplate:
            chat_template = None

            def apply_chat_template(self, *args, **kwargs):
                raise AssertionError("must not be reached")

        cfg = DataConfig(
            train="data.jsonl",
            train_on_responses_only=False,  # legacy path
            train_on_messages_with_train_field=False,
            chat_template=None,
        )
        fn = build_format_row(_NoTemplate(), cfg, console=None)
        with pytest.raises(ValueError, match="chat_template"):
            fn({"messages": [{"role": "user", "content": "hi"}]})

    def test_override_applies_to_legacy_path(self):
        """When user passes chat_template override, build_format_row works."""
        from soup_cli.config.schema import DataConfig
        from soup_cli.data.sft_format import build_format_row

        class _T:
            chat_template = None
            applied = []

            def apply_chat_template(
                self, messages, tokenize=False, add_generation_prompt=False, **kwargs
            ):
                self.applied.append(messages)
                # #785: the legacy path now pre-tokenizes (tokenize=True); it
                # renders to text only for display/other callers.
                return [1, 2, 3] if tokenize else "RENDERED"

        tok = _T()
        cfg = DataConfig(
            train="data.jsonl",
            train_on_responses_only=False,
            train_on_messages_with_train_field=False,
            chat_template="chatml",
        )
        fn = build_format_row(tok, cfg, console=None)
        out = fn({"messages": [{"role": "user", "content": "hi"}]})
        assert out["input_ids"] == [1, 2, 3]  # override made the legacy path usable
        assert tok.applied  # apply_chat_template was invoked
        assert tok.chat_template is not None  # override was applied


# ---------------------------------------------------------------------------
# An unregistered name exits with a message, not a KeyError traceback
# ---------------------------------------------------------------------------


_CONFIG_TEMPLATE = (
    "base: some-org/some-model\ntask: sft\ndata:\n  train: d.jsonl\n"
    "  format: chatml\n  chat_template: {name}\n  max_length: 128\n"
)
_UNREGISTERED_CONFIG = _CONFIG_TEMPLATE.format(name="not_a_real_name")
# A double-quoted YAML string can carry ESC and BEL: clear-screen plus a window title.
_CONTROL_BYTES_CONFIG = _CONFIG_TEMPLATE.format(name='"bad\\e[2J\\e]0;PWNED\\aname"')
# Rich markup in the name: an unbalanced closing tag would raise MarkupError.
_MARKUP_CONFIG = _CONFIG_TEMPLATE.format(name="'x[/]'")

_COMMANDS = pytest.mark.parametrize(
    "args",
    [["data", "preprocess", "soup.yaml", "--yes"], ["train", "--config", "soup.yaml"]],
    ids=["preprocess", "train"],
)


class TestUnregisteredNameExitsCleanly:
    def _invoke(self, tmp_path, monkeypatch, args, config=_UNREGISTERED_CONFIG):
        transformers = pytest.importorskip("transformers")
        from typer.testing import CliRunner

        from soup_cli.cli import app

        def _no_download(*_args, **_kwargs):
            raise AssertionError("nothing should load for an invalid config")

        for name in ("AutoTokenizer", "AutoModelForCausalLM"):
            monkeypatch.setattr(
                getattr(transformers, name), "from_pretrained", _no_download
            )
        monkeypatch.chdir(tmp_path)
        (tmp_path / "d.jsonl").write_text("{}\n", encoding="utf-8")
        (tmp_path / "soup.yaml").write_text(config, encoding="utf-8")
        return CliRunner().invoke(app, args)

    @_COMMANDS
    def test_exits_1_naming_the_template(self, tmp_path, monkeypatch, args):
        result = self._invoke(tmp_path, monkeypatch, args)
        output = strip_ansi(result.output)

        assert result.exit_code == 1, result.output
        assert isinstance(result.exception, SystemExit), result.exception
        assert "Invalid data.chat_template" in output
        assert "not_a_real_name" in output
        assert "chatml" in output, "the message still lists the known names"

    @_COMMANDS
    def test_control_bytes_in_the_name_do_not_reach_the_terminal(self, tmp_path, monkeypatch, args):
        result = self._invoke(tmp_path, monkeypatch, args, config=_CONTROL_BYTES_CONFIG)
        # strip_ansi removes only Rich's own SGR colour codes, not the injected sequences.
        output = strip_ansi(result.output)

        assert result.exit_code == 1, result.output
        assert "Invalid data.chat_template" in output, repr(output)
        assert "\x1b" not in output and "\x07" not in output, repr(output)

    @_COMMANDS
    def test_markup_in_the_name_is_printed_literally(self, tmp_path, monkeypatch, args):
        result = self._invoke(tmp_path, monkeypatch, args, config=_MARKUP_CONFIG)
        output = strip_ansi(result.output)

        assert result.exit_code == 1, result.output
        assert isinstance(result.exception, SystemExit), result.exception
        assert "'x[/]'" in output, repr(output)
