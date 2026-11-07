"""Spreadsheet-safe text for CSV cells.

A cell whose text starts with ``=``, ``+``, ``-``, ``@``, TAB, CR or LF is read as a
formula by Excel, LibreOffice and Sheets. :func:`spreadsheet_safe` prefixes such a
cell with a single quote, the convention spreadsheets honour.
"""

from __future__ import annotations

_FORMULA_LEADERS = ("=", "+", "-", "@", "\t", "\r", "\n")


def spreadsheet_safe(value: object) -> str:
    """Return ``str(value)``, prefixed with ``'`` if a spreadsheet would read it as a formula.

    Apply this to free-text cells only (model, benchmark and similar names). Numeric
    columns must be formatted separately and left alone, so a negative score such as
    -0.5 stays -0.5.
    """
    text = str(value)
    if text.startswith(_FORMULA_LEADERS):
        return "'" + text
    return text
