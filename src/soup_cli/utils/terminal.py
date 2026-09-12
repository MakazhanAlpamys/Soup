"""Terminal hygiene for text that came from outside the program.

Two things bite when an untrusted string -- a config key from a YAML file, a
benchmark name from an evidence file, a dataset row -- is printed through a
Rich console:

* ``rich.markup.escape`` neutralises ``[...]`` tags and NOTHING else, so a
  key spelled ``[bold red]x[/]`` would restyle the terminal.
* Rich passes C0 control bytes through, so a raw ESC (0x1B) in the text is a
  live escape sequence -- window-title spoofing, OSC-8 link spoofing, and on
  some emulators worse -- that ``escape()`` never sees.

:func:`for_terminal` does both. Some call sites only need the strip half —
e.g. text handed to ``Text.append(..., style=...)``, which Rich never parses
as markup, or a string that a caller escapes itself further downstream — so
:func:`strip_control` is exported too. Both draw on the same table so there
is exactly one definition of "which bytes are dangerous" in the tree.
"""

from __future__ import annotations

from rich.markup import escape as _escape_markup

#: Every C0 control byte except TAB / LF / CR, plus DEL.
_CONTROL_STRIP_TABLE = {i: None for i in range(0x20) if i not in (0x09, 0x0A, 0x0D)}
_CONTROL_STRIP_TABLE[0x7F] = None


def strip_control(text: object) -> str:
    """Strip C0/DEL control bytes without touching Rich markup."""
    return str(text).translate(_CONTROL_STRIP_TABLE)


def for_terminal(text: object) -> str:
    """Strip control bytes, then escape Rich markup, in that order."""
    return _escape_markup(strip_control(text))
