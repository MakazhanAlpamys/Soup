"""Repo-wide ratchet: literal bracketed text in CLI help must not be swallowed by Rich.

Rich interprets `[tag]` where tag starts with `[a-z#/@]` as markup. If the tag is
not a recognised style (like `[deprecated]`, `[doc-N]`, `[sigstore]`), Rich drops it
without a trace unless escaped as `\\[tag]`.

This ratchet walks the live Click command tree derived from Typer and ensures:
1. No command or parameter help string has unescaped non-style markup tags dropped.
2. The 7 fixed help strings retain their literal bracketed text in rendered `--help`.
"""

from __future__ import annotations

import inspect
import re
from typing import Iterator

import click
import pytest
from rich.errors import StyleSyntaxError
from rich.markup import _parse
from rich.style import Style
from rich.text import Text
from typer.testing import CliRunner

from soup_cli.cli import app

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_BOX_RE = re.compile(r"[─-╿│╭╮╰╯]")


def _plain(text: str) -> str:
    """ANSI- and border-stripped, whitespace-collapsed CLI output."""
    return " ".join(_BOX_RE.sub(" ", _ANSI_RE.sub("", text)).split())


def _is_valid_rich_style(tag_name: str) -> bool:
    """True if ``tag_name`` is a valid Rich style or control tag."""
    if (
        tag_name == "/"
        or tag_name.startswith("/")
        or tag_name.startswith("link=")
        or tag_name.startswith("@")
    ):
        return True
    try:
        Style.parse(tag_name)
        return True
    except (StyleSyntaxError, Exception):
        return False


def find_dropped_markup_tags(text: str) -> list[str]:
    """Return all unescaped tag names in ``text`` that are dropped by Rich markup."""
    if not text:
        return []
    dropped: list[str] = []
    try:
        for _pos, _text_content, tag in _parse(text):
            if tag is not None and not _is_valid_rich_style(tag.name):
                # An unrecognised tag name that Rich drops when rendering markup
                dropped.append(tag.name)
    except Exception as exc:
        dropped.append(f"parse_error:{exc}")
    return dropped


def _iter_cli_help_strings(
    cmd: click.Command, path: str = "soup"
) -> Iterator[tuple[str, str]]:
    """Yield ``(location, help_string)`` for every command and parameter."""
    for attr in ("help", "short_help", "description", "epilog"):
        val = getattr(cmd, attr, None)
        if isinstance(val, str) and val.strip():
            yield f"{path} ({attr})", val

    # Click parameter help
    for param in getattr(cmd, "params", []):
        val = getattr(param, "help", None)
        if isinstance(val, str) and val.strip():
            yield f"{path} param {param.name} (help)", val

    # Typer parameter declarations (preserves Argument help Click 8.5 drops, #1790)
    callback = getattr(cmd, "callback", None)
    if callback:
        try:
            sig = inspect.signature(callback)
            for param_name, param in sig.parameters.items():
                default = param.default
                help_val = getattr(default, "help", None)
                if isinstance(help_val, str) and help_val.strip():
                    yield f"{path} typer {param_name} (help)", help_val
        except (ValueError, TypeError):
            pass

    if isinstance(cmd, click.Group):
        for sub_name, sub_cmd in cmd.commands.items():
            yield from _iter_cli_help_strings(sub_cmd, f"{path} {sub_name}")


class TestCliHelpRichMarkupRatchet:
    def test_no_unescaped_non_style_tags_in_cli_help(self):
        """Walk the live Click tree: assert zero dropped bracketed tags across all help."""
        from typer.main import get_command

        root_cmd = get_command(app)
        offenders: list[str] = []
        scanned = 0

        for location, help_text in _iter_cli_help_strings(root_cmd):
            scanned += 1
            dropped = find_dropped_markup_tags(help_text)
            if dropped:
                offenders.append(
                    f"{location}: dropped tags {dropped!r} in {help_text[:80]!r}"
                )

        assert scanned > 50, f"scanned only {scanned} help strings — walker broken?"
        assert not offenders, (
            "Found help strings with unescaped bracketed text swallowed by Rich markup.\n"
            "Escape with `\\[tag]` so Rich renders literal brackets (issue #1792):\n  "
            + "\n  ".join(offenders)
        )

    def test_plain_text_survives_from_markup(self):
        """Ensure Text.from_markup preserves bracketed content for all scanned help."""
        from typer.main import get_command

        root_cmd = get_command(app)
        checked = 0
        for _location, help_text in _iter_cli_help_strings(root_cmd):
            checked += 1
            # Should not raise MarkupError
            rendered = Text.from_markup(help_text).plain
            assert isinstance(rendered, str)
        assert checked > 50


class TestMarkupScannerDetection:
    """The scanner must be shown capable of detecting unescaped tags."""

    SAMPLE_BROKEN_STRINGS = [
        ("Upload token. For --hub hf: [deprecated] use HF_TOKEN", "deprecated"),
        ("Citation style: ([doc-id]) / inline ((doc-id))", "doc-id"),
        ("RAFT document-shuffle seed: the [doc-N] ids", "doc-N"),
        ("install soup-cli[sigstore] and run", "sigstore"),
        ("routes through the langdetect package ([data-pro] extras)", "data-pro"),
    ]

    @pytest.mark.parametrize(("raw_text", "expected_tag"), SAMPLE_BROKEN_STRINGS)
    def test_scanner_flags_unescaped_non_style_tags(self, raw_text: str, expected_tag: str):
        dropped = find_dropped_markup_tags(raw_text)
        assert expected_tag in dropped

    @pytest.mark.parametrize(("raw_text", "expected_tag"), SAMPLE_BROKEN_STRINGS)
    def test_escaped_version_is_accepted_and_preserves_text(self, raw_text: str, expected_tag: str):
        escaped = raw_text.replace(f"[{expected_tag}]", f"\\[{expected_tag}]")
        dropped = find_dropped_markup_tags(escaped)
        assert dropped == []
        plain = Text.from_markup(escaped).plain
        assert f"[{expected_tag}]" in plain


class TestEscapedHelpStringsReachTerminal:
    """The 7 fixed sites must actually display their literal bracketed text in --help."""

    @pytest.mark.parametrize(
        ("argv", "expected_literal"),
        [
            (["push", "--help"], "[deprecated] use HF_TOKEN"),
            (["data", "generate", "--help"], "[deprecated] Use OPENAI_API_KEY"),
            (["diagnose", "--help"], "([doc-id])"),
            (["diagnose", "--help"], "[doc-N]"),
            (["eval", "citation", "--help"], "[doc-N]"),
            (["adapters", "sign", "--help"], "soup-cli[sigstore]"),
            (["data", "brain-rot", "--help"], "[data-pro]"),
        ],
    )
    def test_help_contains_literal_bracketed_text(self, argv: list[str], expected_literal: str):
        result = CliRunner().invoke(app, argv)
        assert result.exit_code == 0, (result.output, repr(result.exception))
        plain_output = _plain(result.output)
        assert expected_literal in plain_output, (
            f"`soup {' '.join(argv)}` --help output does not contain {expected_literal!r}.\n"
            f"Output was: {plain_output[:300]}..."
        )
