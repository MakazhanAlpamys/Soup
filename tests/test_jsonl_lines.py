"""``jsonl_lines`` ends a JSONL row at the file's line ends only (#3066).

``str.splitlines()`` also splits at U+0085, U+2028 and U+2029, which are legal
inside a JSON string and are written raw by ``json.dumps(..., ensure_ascii=False)``.
The seven readers that used it now share ``soup_cli.utils.jsonl_lines``; their own
tests sit next to each reader's other tests.
"""

from __future__ import annotations

import json

import pytest

from soup_cli.utils.jsonl_lines import jsonl_lines

# The three characters ``str.splitlines()`` splits at that may occur raw in a JSON string.
SEPARATORS = {"U+0085": chr(0x85), "U+2028": chr(0x2028), "U+2029": chr(0x2029)}

# The other characters ``str.splitlines()`` splits at; none can occur raw in valid JSON.
NOT_IN_JSON = {
    "VT": "\v",
    "FF": "\f",
    "U+001C": chr(0x1C),
    "U+001D": chr(0x1D),
    "U+001E": chr(0x1E),
}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("a\nb\n", ["a", "b"], id="lf"),
        pytest.param("a\r\nb\r\n", ["a", "b"], id="crlf"),
        pytest.param("a\rb\r", ["a", "b"], id="lone-cr"),
        pytest.param("a\nb\r\nc\rd", ["a", "b", "c", "d"], id="mixed"),
    ],
)
def test_line_ends_still_end_a_row(text, expected):
    assert jsonl_lines(text) == expected


@pytest.mark.parametrize(
    "text",
    ["", "a", "a\n", "a\n\nb", "\n", "\n\n", "a\r\n\r\nb\r\n", "a\n "],
    ids=["empty", "no-end", "one-end", "blank-middle", "lone-lf", "two-lf", "crlf-blank", "tail"],
)
def test_numbering_matches_splitlines_without_the_three_characters(text):
    """Control: callers keep their line numbers and blank-row handling."""
    assert jsonl_lines(text) == text.splitlines()


@pytest.mark.parametrize("char", SEPARATORS.values(), ids=SEPARATORS.keys())
def test_character_inside_a_json_string_stays_in_the_row(char):
    first = json.dumps({"t": f"a{char}b"}, ensure_ascii=False)
    assert char in first
    text = f"{first}\n" + json.dumps({"t": "next"}) + "\n"

    rows = [json.loads(line) for line in jsonl_lines(text)]

    assert rows == [{"t": f"a{char}b"}, {"t": "next"}]


@pytest.mark.parametrize("char", NOT_IN_JSON.values(), ids=NOT_IN_JSON.keys())
def test_character_that_cannot_be_raw_in_json_is_still_refused(char):
    """Control: no longer a row end, and still not a valid row."""
    row = '{"t": "a' + char + 'b"}'
    lines = jsonl_lines(row + "\n")

    assert lines == [row]
    with pytest.raises(json.JSONDecodeError):
        json.loads(lines[0])
