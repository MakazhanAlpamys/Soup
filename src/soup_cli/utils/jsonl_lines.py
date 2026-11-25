"""Split decoded JSONL text into rows at the file's own line ends (#3066).

``str.splitlines()`` also splits at U+0085, U+2028 and U+2029. Those three
characters are legal inside a JSON string, and ``json.dumps(...,
ensure_ascii=False)`` writes them as they are, so ``splitlines()`` cuts such a
row into two broken lines. A row ends here at a line feed, a carriage return
plus line feed, or a lone carriage return, and at nothing else: none of the
three can occur raw inside a valid JSON string, so splitting at them never cuts
a row.

The other characters ``splitlines()`` splits at (vertical tab, form feed and
U+001C to U+001E) are no longer row ends either. They cannot occur raw in a
valid JSON row, so a row that holds one is still refused or skipped by its
reader, as before.
"""

from __future__ import annotations

import re

_LINE_END = re.compile(r"\r\n|\r|\n")


def jsonl_lines(text: str) -> list[str]:
    """Return the rows of ``text``, numbered as ``str.splitlines()`` numbers them.

    A final line end does not open an empty last row, so ``enumerate(...,
    1)`` gives the same line numbers as before for any text in which
    ``splitlines()`` finds no line end other than LF, CRLF and CR, and blank
    rows in the middle are kept for the caller to skip or refuse.
    """
    lines = _LINE_END.split(text)
    if lines[-1] == "":
        lines.pop()
    return lines
