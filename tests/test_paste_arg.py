"""``paste_arg``: a configured path inside a command a person pastes.

Soup prints follow-up commands (the cloud download hint, the "Quick test / Push /
Export" lines after training) whose argument is the config's ``output``. The shell the
line is pasted into is unknown, so the spelling has to read as the same name in POSIX
shells, PowerShell and cmd.exe -- or the command has to leave the value out.

The oracle below is written independently of the implementation: per shell it lists the
characters that act OUTSIDE quotes and the ones that still act INSIDE double quotes, and
the property test asserts that no result exposes such a character to a shell.
"""

from __future__ import annotations

import string
import unicodedata

import pytest

from soup_cli.utils.paths import paste_arg

# Characters a shell acts on when they are not quoted (POSIX sh / bash, PowerShell, cmd).
_ACTS_UNQUOTED = set(" \t\r\n\"'`$\\%!^&|;<>()[]{}*?~#=,@") | {"\x00", "\x1b", "\x7f"}
# Characters that still act inside double quotes in at least one of those shells.
_ACTS_IN_DOUBLE_QUOTES = set('"$`\\%!')


def _non_ascii_is_not_a_letter_or_digit(text: str) -> bool:
    """Non-ASCII punctuation, spaces and format characters are quotes/dashes/spaces to
    some shell (typographic quotes, en dash, no-break space) or hide text (bidi marks)."""
    return any(
        not char.isascii() and unicodedata.category(char)[0] not in "LNM" for char in text
    )


def _exposes_a_shell_character(result: str) -> bool:
    """True when a shell would act on part of ``result`` (quoted or not)."""
    if _non_ascii_is_not_a_letter_or_digit(result):
        return True
    if result.startswith('"') and result.endswith('"') and len(result) >= 2:
        inner = result[1:-1]
        if any(char in _ACTS_IN_DOUBLE_QUOTES for char in inner):
            return True
        return any(ord(char) < 0x20 or 0x7F <= ord(char) <= 0x9F for char in inner)
    return any(char in _ACTS_UNQUOTED for char in result)


class TestPlainValues:
    @pytest.mark.parametrize(
        "value",
        [
            "./output",
            "output",
            "outputs/run-1",
            "out_dir.v2",
            "C:/work/out",
            "a+b/c",
            "результаты/модель",
            "résultats/été",
            "出力/モデル",
        ],
    )
    def test_plain_names_are_returned_unchanged(self, value):
        assert paste_arg(value) == value

    def test_a_path_object_is_accepted(self):
        from pathlib import PurePosixPath

        assert paste_arg(PurePosixPath("out/run")) == "out/run"


class TestValuesThatNeedQuotes:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("my outputs", '"my outputs"'),
            ("out (v2)", '"out (v2)"'),
            ("Bob's outputs", '"Bob\'s outputs"'),
            ("out;run", '"out;run"'),
            ("out&run", '"out&run"'),
            ("out|run", '"out|run"'),
            ("out<in>", '"out<in>"'),
            ("out^run", '"out^run"'),
            ("~/out", '"~/out"'),
            ("#out", '"#out"'),
            ("a=b,c@d", '"a=b,c@d"'),
            ("out[1]{2}*?", '"out[1]{2}*?"'),
            (" lead", '" lead"'),
            ("trail ", '"trail "'),
        ],
    )
    def test_double_quotes_keep_these_literal_in_every_shell(self, value, expected):
        assert paste_arg(value) == expected


class TestValuesThatAreNotRepeated:
    @pytest.mark.parametrize(
        "value",
        [
            "out$(cmd)",
            "out$VAR",
            "out${VAR}",
            "out`cmd`",
            'out"x',
            "out\\dir",
            "C:\\work\\out",
            "out%COMSPEC%",
            "out!x",
            "out\nx",
            "out\rx",
            "out\tx",
            "out\x00x",
            "out\x1b[2Jx",
            "out\x7fx",
            "out\x85x",
            "out\u201cx\u201d",  # typographic double quotes: quotes to PowerShell
            "out\u2013force",  # en dash: a dash to PowerShell
            "out\u00a0x",  # no-break space
            "out\u202ex",  # bidirectional override
            "out\u200bx",  # zero-width space
        ],
    )
    def test_value_is_left_out(self, value):
        assert paste_arg(value) is None

    def test_empty_value_is_left_out(self):
        assert paste_arg("") is None


class TestLeadingDash:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("-rf", "./-rf"),
            ("--force", "./--force"),
            ("-", "./-"),
            ("-my out", '"./-my out"'),
        ],
    )
    def test_a_leading_dash_cannot_become_an_option(self, value, expected):
        assert paste_arg(value) == expected

    def test_a_dash_inside_the_value_is_untouched(self):
        assert paste_arg("run-1/--x") == "run-1/--x"


class TestShellSyntaxIsNeverRepeatedAsSyntax:
    @pytest.mark.parametrize("windows_separators", [False, True])
    def test_every_single_character_value(self, windows_separators):
        # Every character of ASCII, Latin-1 and the typographic/format blocks, alone and
        # embedded between plain text.
        ordinals = [*range(0x00, 0x180), *range(0x2000, 0x2070), 0xFEFF, 0x3000]
        exposed = []
        for ordinal in ordinals:
            char = chr(ordinal)
            for value in (char, f"out{char}x", f"{char}out"):
                result = paste_arg(value, windows_separators=windows_separators)
                if result is not None and _exposes_a_shell_character(result):
                    exposed.append((value, result))
        assert exposed == []

    def test_every_pair_of_printable_ascii(self):
        printable = string.printable[:-5]  # without \t\n\r\x0b\x0c
        exposed = []
        for first in printable:
            for second in printable:
                value = f"a{first}{second}b"
                result = paste_arg(value)
                if result is not None and _exposes_a_shell_character(result):
                    exposed.append((value, result))
        assert exposed == []

    @pytest.mark.parametrize(
        "value",
        [
            "out$(cmd)",
            "out`cmd`",
            "out;cmd",
            "out&&cmd",
            "out||cmd",
            "out|cmd",
            "out>X",
            "out<X",
            "out&next",
            "out%COMSPEC%",
            "out\"; cmd; echo \"",
            "out'; cmd; echo '",
            "out\ncmd",
            "$(cmd)",
            "-$(cmd)",
            "${IFS}x",
            "out\\",
            "out^&next",
        ],
    )
    def test_names_with_shell_syntax(self, value):
        result = paste_arg(value)
        assert result is None or not _exposes_a_shell_character(result)

    def test_the_result_is_one_token_in_a_posix_split(self):
        import shlex

        for value in ("my outputs", "out (v2)", "out;run", "Bob's x", "out&run"):
            result = paste_arg(value)
            assert result is not None
            assert shlex.split(f"cmd {result}") == ["cmd", value]


class TestWindowsSeparators:
    """``str(Path('output/sub'))`` is ``output\\sub`` on Windows: a nested path is repeated
    with forward slashes there, and stays a reason for the placeholder elsewhere."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("output\\sub", "output/sub"),
            ("output\\sub\\run 1", '"output/sub/run 1"'),
            ("C:\\work\\out", "C:/work/out"),
            ("out\\", "out/"),
            ("out/sub\\run", "out/sub/run"),
        ],
    )
    def test_backslashes_become_forward_slashes(self, value, expected):
        assert paste_arg(value, windows_separators=True) == expected

    @pytest.mark.parametrize("value", ["output\\sub", "C:\\work\\out", "out\\"])
    def test_a_backslash_is_a_placeholder_when_the_host_does_not_use_it_as_a_separator(
        self, value
    ):
        assert paste_arg(value) is None
        assert paste_arg(value, windows_separators=False) is None

    @pytest.mark.parametrize(
        "value", ["out\\$x", 'out\\"x', "out\\%x", "out\\!x", "out\\`x", "out\\\nx"]
    )
    def test_other_syntax_is_still_left_out(self, value):
        assert paste_arg(value, windows_separators=True) is None

    def test_a_leading_dash_still_gets_a_prefix(self):
        assert paste_arg("-run\\x", windows_separators=True) == "./-run/x"


class TestCombiningMarks:
    @pytest.mark.parametrize(
        "value",
        [
            "cafe\u0301/data",  # decomposed accent
            "\u0939\u093f\u0928\u094d\u0926\u0940/out",  # Devanagari with vowel signs
            "\u0e44\u0e17\u0e22/out",  # Thai
            "\u0646\u062a\u0627\u0626\u062c\u064e/out",  # Arabic with a short-vowel mark
        ],
    )
    def test_marks_of_a_script_are_repeated_as_part_of_the_name(self, value):
        assert paste_arg(value) == value
