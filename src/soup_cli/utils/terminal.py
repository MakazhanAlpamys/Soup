"""Terminal hygiene for text that came from outside the program.

Two things bite when an untrusted string -- a config key from a YAML file, a
benchmark name from an evidence file, a dataset row -- is printed through a
Rich console:

* ``rich.markup.escape`` neutralises ``[...]`` tags and NOTHING else, so a
  key spelled ``[bold red]x[/]`` would restyle the terminal.
* Rich passes C0 control bytes through, so a raw ESC (0x1B) in the text is a
  live escape sequence -- window-title spoofing, OSC-8 link spoofing, and on
  some emulators worse -- that ``escape()`` never sees.

:func:`for_terminal` does both. Six command modules carry a private copy of
this helper already (``commands/data_doctor.py`` was the first, v0.71.27);
new call sites import this one.
"""

from __future__ import annotations

from rich.markup import escape as _escape_markup

#: Every C0 control byte except TAB / LF / CR, plus DEL.
_CONTROL_STRIP_TABLE = {i: None for i in range(0x20) if i not in (0x09, 0x0A, 0x0D)}
_CONTROL_STRIP_TABLE[0x7F] = None


def for_terminal(text: object) -> str:
    """Strip control bytes, then escape Rich markup, in that order."""
    return _escape_markup(str(text).translate(_CONTROL_STRIP_TABLE))
