"""``soup serve --tool-auth-token`` follows the token rule ``soup ui`` and ``soup mcp serve`` use.

The rule is ``utils.qr_url.validate_token``: 16 to 128 characters from ``A-Z``, ``a-z``,
``0-9``, ``_`` and ``-``, the shape ``secrets.token_urlsafe`` produces. ``soup serve``
checks the value from the option or from ``SOUP_TOOL_AUTH_TOKEN`` first, on every bind,
and exits 2 before anything else runs when it does not match. The message names the
option, the environment variable and the rule, and never repeats the value. An empty
value still counts as no token (``tests/test_serve_code_tool_token.py``).

The rule matches the whole value: a valid token followed by a newline is refused too.
"""

from __future__ import annotations

import secrets
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.qr_url import validate_token
from tests.conftest import strip_ansi

runner = CliRunner()

RULE = "16-128 urlsafe-base64 chars"

VALID = [
    "a" * 16,
    "Z" * 128,
    "s3cret-token_value",
    secrets.token_urlsafe(32),
]

# Each way a value can miss the rule: length, a character outside the set, and
# whitespace before, inside or after an otherwise valid token.
INVALID = {
    "space-only": " ",
    "short": "abc",
    "one-short": "a" * 15,
    "one-long": "a" * 129,
    "space-inside": "valid-token value1",
    "dot": "valid.token.value1",
    "plus-slash": "valid+token/value1",
    "padding": "validtokenvalue1==",
    "non-ascii": "validtokenvalué12",
    "trailing-newline": "a" * 16 + "\n",
    "trailing-crlf": "a" * 16 + "\r\n",
    "leading-newline": "\n" + "a" * 16,
    "trailing-space": "a" * 16 + " ",
    "inner-newline": "a" * 8 + "\n" + "a" * 8,
}


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output, safe to substring-match."""
    return " ".join(strip_ansi(text).split())


class TestValidateToken:
    # Stable ids: VALID holds a freshly generated token, and xdist workers must collect
    # the same test ids.
    @pytest.mark.parametrize(
        "token", VALID, ids=["16-letters", "128-letters", "mixed-set", "token-urlsafe-32"]
    )
    def test_a_token_inside_the_rule_is_returned(self, token):
        assert validate_token(token) == token

    @pytest.mark.parametrize("token", list(INVALID.values()), ids=list(INVALID))
    def test_a_token_outside_the_rule_is_refused(self, token):
        with pytest.raises(ValueError, match=RULE):
            validate_token(token)


class TestServeChecksTheToken:
    """The check runs first: the model loader is never reached."""

    @pytest.fixture
    def model_dir(self, tmp_path):
        directory = tmp_path / "model"
        directory.mkdir()
        (directory / "config.json").write_text("{}", encoding="utf-8")
        return directory

    def _invoke(self, model_dir, *args, env_token=None, monkeypatch):
        monkeypatch.setenv("COLUMNS", "400")
        if env_token is not None:
            monkeypatch.setenv("SOUP_TOOL_AUTH_TOKEN", env_token)
        with patch(
            "soup_cli.commands.serve._load_model",
            side_effect=AssertionError("the model loader ran before the token was checked"),
        ) as loader:
            result = runner.invoke(app, ["serve", "--model", str(model_dir), *args])
        return result, loader

    @pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0"])
    @pytest.mark.parametrize(
        "token", ["abc", "a" * 15, "a" * 129, "valid.token.value1", "a" * 16 + "\n"],
        ids=["short", "one-short", "one-long", "dot", "trailing-newline"],
    )
    def test_the_option_outside_the_rule_exits_2(self, model_dir, host, token, monkeypatch):
        result, loader = self._invoke(
            model_dir, "--host", host, "--tool-auth-token", token, monkeypatch=monkeypatch
        )

        assert result.exit_code == 2, (result.output, repr(result.exception))
        out = _plain(result.output)
        assert "--tool-auth-token (or SOUP_TOOL_AUTH_TOKEN)" in out, out
        assert RULE in out, out
        assert "secrets.token_urlsafe(32)" in out, out
        loader.assert_not_called()

    @pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0"])
    def test_the_env_var_outside_the_rule_exits_2(self, model_dir, host, monkeypatch):
        result, loader = self._invoke(
            model_dir, "--host", host, env_token="short-token", monkeypatch=monkeypatch
        )

        assert result.exit_code == 2, (result.output, repr(result.exception))
        out = _plain(result.output)
        assert "--tool-auth-token (or SOUP_TOOL_AUTH_TOKEN)" in out, out
        assert RULE in out, out
        loader.assert_not_called()

    def test_the_refusal_does_not_repeat_the_value(self, model_dir, monkeypatch):
        token = "Distinct-Value-"  # 15 characters: one short of the rule
        result, _ = self._invoke(model_dir, "--tool-auth-token", token, monkeypatch=monkeypatch)

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert token not in result.output

    def test_the_refusal_comes_before_the_bind_check(self, model_dir, monkeypatch):
        """A non-loopback bind with a malformed token is told about the token, not the bind."""
        result, _ = self._invoke(
            model_dir, "--host", "0.0.0.0", "--tool-auth-token", "abc", monkeypatch=monkeypatch
        )

        out = _plain(result.output)
        assert RULE in out, out
        assert "requires --tool-auth-token <secret>" not in out, out
        assert "Note: bound to" not in out, out

    def test_a_token_inside_the_rule_passes_the_check(self, tmp_path, monkeypatch):
        """The model path does not exist, so the command stops at that check instead."""
        pytest.importorskip("fastapi")
        pytest.importorskip("uvicorn")
        monkeypatch.setenv("COLUMNS", "400")

        result = runner.invoke(
            app,
            [
                "serve",
                "--model",
                str(tmp_path / "no-such-model-dir"),
                "--tool-auth-token",
                secrets.token_urlsafe(32),
            ],
        )

        out = _plain(result.output)
        assert RULE not in out, out
        assert "Model path not found" in out, out
