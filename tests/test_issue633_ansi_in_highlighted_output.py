"""Syntax-highlighted CLI output must be normalised before it is read (#633).

``soup recipes show`` and ``soup data mix --apply`` render YAML through Rich
with Pygments highlighting. With colour enabled, escape sequences land *between
the tokens of one logical line* -- ``modality`` / ``:`` / ``text`` are three
tokens, so ``"modality: text"`` is not a substring of the rendered output, and
``yaml.safe_load`` rejects ``\\x1b`` outright.

CI never sees this: GitHub runners have no TTY and do not set ``FORCE_COLOR``,
so Rich disables colour. It appears only for a contributor whose terminal forces
colour, as three red tests on a clean checkout with no local changes.

This is the failure mode ``test_cli_help_assertions_are_ansi_safe.py`` was
written for after it "turned CI red everywhere else (it has, four times)". That
guard only ever looks at functions containing ``--help``, so no other subcommand
was covered.

Note what does *not* break: a bare model id such as ``Qwen/Qwen3.5-9B`` is a
single Pygments token, so no escape lands inside it. The distinction is between
single-token and multi-token assertions, which is why this went unnoticed while
neighbouring assertions in the same files kept passing.
"""

from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

from .conftest import strip_ansi

_ESC = re.compile(r"\x1b\[[0-9;]*m")


class TestTheHelperItself:
    def test_strip_ansi_removes_escapes(self) -> None:
        assert strip_ansi("\x1b[1;32mmodality\x1b[0m: text") == "modality: text"

    def test_strip_ansi_leaves_clean_text_alone(self) -> None:
        assert strip_ansi("modality: text") == "modality: text"

    def test_strip_ansi_tolerates_none(self) -> None:
        assert strip_ansi(None) == ""


class TestHighlightedOutputIsReadable:
    """These fail without the fix whenever colour is forced."""

    def _show(self, name: str) -> str:
        runner = CliRunner(env={"FORCE_COLOR": "1", "TERM": "xterm-256color", "COLUMNS": "300"})
        result = runner.invoke(app, ["recipes", "show", name])
        assert result.exit_code == 0, result.output
        return result.output

    def test_the_hazard_is_real_and_this_test_reproduces_it(self) -> None:
        """A control: without stripping, the multi-token assertion fails.

        If Rich ever stops highlighting here the rest of this file would pass
        vacuously, so the hazard itself is asserted rather than assumed.

        The skip is tied to the hazard, not to the mere presence of escapes:
        Rich colours the panel border even when it does not syntax-highlight the
        body, so "some escape somewhere" is not evidence that the YAML is
        tokenised. Environments that honour NO_COLOR land here legitimately.
        """
        raw = self._show("qwen3.8-27b-sft")
        if "modality: text" in raw:
            pytest.skip("YAML is not syntax-highlighted here; nothing to normalise")
        assert _ESC.search(raw), "expected escapes if the literal is absent"
        assert "modality: text" in strip_ansi(raw)

    def test_multi_token_assertion_holds_after_stripping(self) -> None:
        assert "modality: text" in strip_ansi(self._show("qwen3.8-27b-sft"))

    def test_single_token_assertion_holds_either_way(self) -> None:
        """Why the neighbouring assertions in the same file never broke."""
        raw = self._show("qwen3.8-27b-sft")
        assert "Qwen/Qwen3.8-27B" in raw
        assert "Qwen/Qwen3.8-27B" in strip_ansi(raw)

    def test_structural_yaml_substrings_hold_after_stripping(self) -> None:
        """``recipes show`` renders inside a Panel, so stripping ANSI is
        necessary but not sufficient to *parse* it -- the box borders remain.
        Assert on structure rather than pretending the panel is a document;
        ``data mix --apply`` is the unpanelled command whose output is parsed,
        and that path is covered in test_issue330."""
        plain = strip_ansi(self._show("qwen3.8-27b-sft"))
        assert "base: Qwen/Qwen3.8-27B" in plain
        assert "task: sft" in plain


class TestTheGuardCoversHighlightedCommands:
    """The scanner must reach subcommands other than ``--help``."""

    def test_scanner_flags_a_parsed_raw_output_assertion(self) -> None:
        from tests.test_cli_help_assertions_are_ansi_safe import (
            find_unsafe_highlighted_assertions,
        )

        source = (
            "def test_x():\n"
            "    result = runner.invoke(app, ['recipes', 'show', 'r'])\n"
            "    loaded = yaml.safe_load(result.output)\n"
        )
        assert find_unsafe_highlighted_assertions(source)

    def test_scanner_flags_a_multi_token_assertion(self) -> None:
        from tests.test_cli_help_assertions_are_ansi_safe import (
            find_unsafe_highlighted_assertions,
        )

        source = (
            "def test_x():\n"
            "    result = runner.invoke(app, ['recipes', 'show', 'r'])\n"
            "    assert 'modality: text' in result.output\n"
        )
        assert find_unsafe_highlighted_assertions(source)

    def test_scanner_accepts_a_normalised_assertion(self) -> None:
        from tests.test_cli_help_assertions_are_ansi_safe import (
            find_unsafe_highlighted_assertions,
        )

        source = (
            "def test_x():\n"
            "    result = runner.invoke(app, ['recipes', 'show', 'r'])\n"
            "    assert 'modality: text' in strip_ansi(result.output)\n"
        )
        assert not find_unsafe_highlighted_assertions(source)

    def test_scanner_ignores_single_token_assertions(self) -> None:
        """Single tokens are genuinely safe; flagging them would be noise."""
        from tests.test_cli_help_assertions_are_ansi_safe import (
            find_unsafe_highlighted_assertions,
        )

        source = (
            "def test_x():\n"
            "    result = runner.invoke(app, ['recipes', 'show', 'r'])\n"
            "    assert 'Qwen/Qwen3.8-27B' in result.output\n"
        )
        assert not find_unsafe_highlighted_assertions(source)

    def test_the_suite_has_no_unsafe_highlighted_assertions(self) -> None:
        from pathlib import Path

        from tests.test_cli_help_assertions_are_ansi_safe import (
            find_unsafe_highlighted_assertions,
        )

        offenders: list[str] = []
        tests_dir = Path(__file__).parent
        for path in sorted(tests_dir.glob("test_*.py")):
            source = path.read_text(encoding="utf-8", errors="replace")
            for lineno, text in find_unsafe_highlighted_assertions(source):
                offenders.append(f"{path.name}:{lineno}: {text}")
        assert not offenders, (
            "These read syntax-highlighted CLI output without stripping ANSI. "
            "Rich splits one logical line across Pygments tokens, so a "
            "multi-token substring is absent and yaml.safe_load rejects \\x1b. "
            "Route it through strip_ansi() from tests/conftest.py:\n"
            + "\n".join(offenders)
        )
