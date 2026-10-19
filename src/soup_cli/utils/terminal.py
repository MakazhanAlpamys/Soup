"""Terminal hygiene for text that came from outside the program.

Two things go wrong when a string from outside the program -- a config key
from a YAML file, a benchmark name from an evidence file, a dataset row -- is
printed through a Rich console:

* ``rich.markup.escape`` neutralises ``[...]`` tags and NOTHING else, so a
  key spelled ``[bold red]x[/]`` would restyle the terminal.
* Rich passes control characters through, so a raw ESC (0x1B) in the text --
  or a C1 control such as U+009B, which some terminals treat in UTF-8 output
  as a one-character CSI -- starts a live escape sequence (a window-title
  change, an OSC-8 link) that ``escape()`` never sees.
* Unicode format characters reorder or hide what is shown (the Trojan Source
  class, CVE-2021-42574): in a terminal with bidi support U+202E displays the
  text after it reversed, and a zero-width space hides a break between words.

:func:`for_terminal` does all three. Some call sites only need the strip half --
e.g. text handed to ``Text.append(..., style=...)``, which Rich never parses
as markup, a plain log line, or a string that a caller escapes itself further
downstream -- so :func:`strip_control` is exported too. Both draw on the same
table so there is exactly one definition of "which characters never reach the
terminal" in the tree.
"""

from __future__ import annotations

from rich.markup import escape as _escape_markup

#: Every C0 control character except TAB / LF / CR, plus DEL, plus the C1
#: range U+0080-U+009F.
_CONTROL_STRIP_TABLE = {i: None for i in range(0x20) if i not in (0x09, 0x0A, 0x0D)}
_CONTROL_STRIP_TABLE[0x7F] = None
_CONTROL_STRIP_TABLE.update(dict.fromkeys(range(0x80, 0xA0)))

#: Bidi controls: embeddings and overrides U+202A-U+202E, isolates
#: U+2066-U+2069, and the marks ALM U+061C, LRM U+200E and RLM U+200F.
#: A preview may lose a bidi mark; it must not reorder what it shows.
_BIDI_CONTROLS = (*range(0x202A, 0x202F), *range(0x2066, 0x206A), 0x061C, 0x200E, 0x200F)
#: Invisible separators: ZWSP U+200B, LINE / PARAGRAPH SEPARATOR U+2028 /
#: U+2029, and ZWNBSP / BOM U+FEFF. ZWNJ U+200C and ZWJ U+200D stay: Arabic,
#: Persian, Indic and emoji text need them to render correctly.
_INVISIBLE_SEPARATORS = (0x200B, 0x2028, 0x2029, 0xFEFF)
_CONTROL_STRIP_TABLE.update(dict.fromkeys(_BIDI_CONTROLS + _INVISIBLE_SEPARATORS))


def strip_control(text: object) -> str:
    """Strip C0 (except TAB/LF/CR), DEL, C1, bidi controls and invisible separators.

    ZWJ and ZWNJ are kept; Rich markup is left alone.
    """
    return str(text).translate(_CONTROL_STRIP_TABLE)


def for_terminal(text: object) -> str:
    """Strip control characters, then escape Rich markup, in that order."""
    return _escape_markup(strip_control(text))
