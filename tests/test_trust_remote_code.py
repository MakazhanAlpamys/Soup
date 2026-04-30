"""Tests for ``--trust-remote-code`` opt-in (v0.36.0 Part B).

Replaces the previous unconditional ``trust_remote_code=True`` smell across
sft.py / chat.py / serve.py with an explicit, auditable opt-in flag plus a
trusted-org allowlist that suppresses noise on first-party models.
"""

from __future__ import annotations

import re
from io import StringIO

import pytest
from rich.console import Console

# Rich help renderer can split a flag like --trust-remote-code with ANSI
# colour escapes between `-`, `-trust`, `-remote-code` when the terminal
# is narrow (macOS CI runners hit this; Windows local does not). Strip
# ANSI so substring assertions are robust. Mirrors the helper in
# tests/test_log_level.py.
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[mK]")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)

# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------


class TestKnownSafePrefixes:
    def test_meta_llama_is_safe(self):
        from soup_cli.utils.trust_remote import is_known_safe

        assert is_known_safe("meta-llama/Llama-3.2-1B")

    def test_qwen_is_safe(self):
        from soup_cli.utils.trust_remote import is_known_safe

        assert is_known_safe("Qwen/Qwen2.5-7B")

    def test_mistral_is_safe(self):
        from soup_cli.utils.trust_remote import is_known_safe

        assert is_known_safe("mistralai/Mistral-7B-Instruct-v0.3")

    def test_random_org_not_safe(self):
        from soup_cli.utils.trust_remote import is_known_safe

        assert not is_known_safe("randomuser/SomeModel")

    def test_local_path_not_safe(self):
        from soup_cli.utils.trust_remote import is_known_safe

        assert not is_known_safe("./local-checkpoint")

    def test_partial_prefix_does_not_match(self):
        """`meta-llama-evil/...` must NOT match the `meta-llama/` prefix."""
        from soup_cli.utils.trust_remote import is_known_safe

        assert not is_known_safe("meta-llama-evil/SomeModel")

    def test_empty_string_not_safe(self):
        from soup_cli.utils.trust_remote import is_known_safe

        assert not is_known_safe("")

    def test_non_string_not_safe(self):
        from soup_cli.utils.trust_remote import is_known_safe

        assert not is_known_safe(None)
        assert not is_known_safe(123)


# ---------------------------------------------------------------------------
# resolve_trust_remote_code — main entry
# ---------------------------------------------------------------------------


class TestResolve:
    def test_default_off_for_safe_prefix_passes_silently(self):
        """Trusted org + flag off → returns False, no warning."""
        from soup_cli.utils.trust_remote import resolve_trust_remote_code

        buf = StringIO()
        console = Console(file=buf, force_terminal=False)
        out = resolve_trust_remote_code(
            "meta-llama/Llama-3.2-1B",
            requested=False,
            console=console,
            requires_remote_code=False,
        )
        assert out is False
        assert buf.getvalue() == ""

    def test_flag_enabled_warns_once(self):
        """User opted in → return True + warning panel."""
        from soup_cli.utils.trust_remote import resolve_trust_remote_code

        buf = StringIO()
        console = Console(file=buf, force_terminal=False)
        out = resolve_trust_remote_code(
            "shady-org/SomeModel",
            requested=True,
            console=console,
            requires_remote_code=True,
        )
        assert out is True
        output = buf.getvalue()
        assert "trust_remote_code" in output.lower() or "remote code" in output.lower()
        assert "shady-org/SomeModel" in output

    def test_flag_enabled_safe_prefix_suppresses_warning(self):
        """Trusted org doesn't ship custom code — suppress warning even when flag set."""
        from soup_cli.utils.trust_remote import resolve_trust_remote_code

        buf = StringIO()
        console = Console(file=buf, force_terminal=False)
        out = resolve_trust_remote_code(
            "meta-llama/Llama-3.2-1B",
            requested=True,
            console=console,
            requires_remote_code=False,
        )
        assert out is True
        # No noisy panel when the model is from a trusted prefix.
        assert "WARNING" not in buf.getvalue().upper()

    def test_default_off_for_unknown_with_remote_code_raises(self):
        """Model needs custom code + flag off → fail fast with actionable error."""
        from soup_cli.utils.trust_remote import resolve_trust_remote_code

        buf = StringIO()
        console = Console(file=buf, force_terminal=False)
        with pytest.raises(ValueError) as exc_info:
            resolve_trust_remote_code(
                "shady-org/CustomModel",
                requested=False,
                console=console,
                requires_remote_code=True,
            )
        msg = str(exc_info.value)
        assert "shady-org/CustomModel" in msg
        assert "--trust-remote-code" in msg

    def test_default_off_for_unknown_without_remote_code_passes(self):
        """Standard model + flag off → returns False, no error."""
        from soup_cli.utils.trust_remote import resolve_trust_remote_code

        buf = StringIO()
        console = Console(file=buf, force_terminal=False)
        out = resolve_trust_remote_code(
            "shady-org/StandardLlama",
            requested=False,
            console=console,
            requires_remote_code=False,
        )
        assert out is False

    def test_console_optional(self):
        """resolve_trust_remote_code must work when console is None."""
        from soup_cli.utils.trust_remote import resolve_trust_remote_code

        out = resolve_trust_remote_code(
            "meta-llama/Llama-3.2-1B",
            requested=True,
            console=None,
            requires_remote_code=False,
        )
        assert out is True

    def test_invalid_model_name_rejected(self):
        from soup_cli.utils.trust_remote import resolve_trust_remote_code

        with pytest.raises(ValueError, match="model_name"):
            resolve_trust_remote_code(
                "",
                requested=False,
                console=None,
                requires_remote_code=False,
            )


# ---------------------------------------------------------------------------
# model_requires_trust_remote_code (probe HF config for auto_map)
# ---------------------------------------------------------------------------


class TestRequiresProbe:
    def test_local_path_no_auto_map_returns_false(self, tmp_path, monkeypatch):
        """Local path with config.json lacking auto_map → False."""
        from soup_cli.utils.trust_remote import model_requires_trust_remote_code

        config = tmp_path / "config.json"
        config.write_text('{"model_type": "llama"}', encoding="utf-8")
        assert model_requires_trust_remote_code(str(tmp_path)) is False

    def test_local_path_with_auto_map_returns_true(self, tmp_path):
        from soup_cli.utils.trust_remote import model_requires_trust_remote_code

        config = tmp_path / "config.json"
        config.write_text(
            '{"model_type": "custom", "auto_map": '
            '{"AutoModelForCausalLM": "modeling.Custom"}}',
            encoding="utf-8",
        )
        assert model_requires_trust_remote_code(str(tmp_path)) is True

    def test_missing_config_returns_none(self, tmp_path):
        """Missing config.json → None (unknown — caller decides)."""
        from soup_cli.utils.trust_remote import model_requires_trust_remote_code

        out = model_requires_trust_remote_code(str(tmp_path))
        assert out is None

    def test_malformed_config_returns_none(self, tmp_path):
        from soup_cli.utils.trust_remote import model_requires_trust_remote_code

        config = tmp_path / "config.json"
        config.write_text("{this is not json", encoding="utf-8")
        out = model_requires_trust_remote_code(str(tmp_path))
        assert out is None

    def test_non_dict_root_returns_none(self, tmp_path):
        """Config with non-dict root (e.g. JSON array) → None."""
        from soup_cli.utils.trust_remote import model_requires_trust_remote_code

        config = tmp_path / "config.json"
        config.write_text("[1, 2, 3]", encoding="utf-8")
        out = model_requires_trust_remote_code(str(tmp_path))
        assert out is None

    def test_non_directory_path_returns_none(self):
        """Bare HF repo id (not a local dir) → None (unknown)."""
        from soup_cli.utils.trust_remote import model_requires_trust_remote_code

        out = model_requires_trust_remote_code("meta-llama/Llama-3.2-1B")
        assert out is None


# ---------------------------------------------------------------------------
# CLI flag plumbing
# ---------------------------------------------------------------------------


class TestCLIPlumbing:
    """Smoke check that --trust-remote-code is a registered Typer option."""

    def test_train_help_lists_flag(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "--trust-remote-code" in _strip_ansi(result.output), result.output

    def test_chat_help_lists_flag(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0
        assert "--trust-remote-code" in _strip_ansi(result.output), result.output

    def test_serve_help_lists_flag(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--trust-remote-code" in _strip_ansi(result.output), result.output
