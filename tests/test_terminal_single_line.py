"""``single_line=True`` shows tab, line feed and carriage return as text.

A value that belongs on one display line (an output path, a model name, a data path) must not
start a second line in a panel; every other caller keeps the behaviour it always had.
"""

from __future__ import annotations

import pytest

from soup_cli.utils.terminal import for_terminal, strip_control


class TestTheDefaultIsUnchanged:
    def test_tab_line_feed_and_carriage_return_are_kept(self):
        assert strip_control("a\tb\nc\rd") == "a\tb\nc\rd"
        assert for_terminal("a\tb\nc\rd") == "a\tb\nc\rd"

    def test_other_controls_are_stripped(self):
        assert strip_control("a\x1bb\x07c\x7fd\x9be") == "abcde"

    def test_markup_is_escaped(self):
        assert for_terminal("[bold]x") == "\\[bold]x"


class TestSingleLine:
    @pytest.mark.parametrize(
        ("raw", "shown"),
        [
            ("a\tb", "a\\tb"),
            ("a\nb", "a\\nb"),
            ("a\rb", "a\\rb"),
            ("a\r\nb", "a\\r\\nb"),
        ],
    )
    def test_tab_line_feed_and_carriage_return_are_shown_as_text(self, raw, shown):
        assert strip_control(raw, single_line=True) == shown
        assert for_terminal(raw, single_line=True) == shown

    def test_the_result_holds_no_character_that_moves_the_cursor(self):
        result = strip_control("a\tb\nc\rd\x0be\x0cf\x85g h i", single_line=True)

        assert not set(result) & set("\t\n\r\x0b\x0c\x85  ")

    def test_other_controls_are_stripped_as_before(self):
        assert strip_control("a\x1b[2J\tb", single_line=True) == "a[2J\\tb"

    def test_markup_is_still_escaped(self):
        assert for_terminal("[bold]x\ny", single_line=True) == "\\[bold]x\\ny"

    def test_the_option_is_keyword_only(self):
        with pytest.raises(TypeError):
            strip_control("x", True)  # type: ignore[misc]
