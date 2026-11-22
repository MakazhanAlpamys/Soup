"""``utils.stop_strings``: the OpenAI ``stop`` / Anthropic ``stop_sequences`` rules (#2549).

Every spelling the parser has to take is listed first and tested one by one:
a string, a list of strings, ``null`` / absent, an empty list, an empty string
(refused), a list holding an empty string (refused), more than four entries
(refused), an entry over the length bound (refused), and every non-string
spelling (refused). The matcher and the streaming filter are pure functions of
text, so the same replies are run through them whole, character by character and
in uneven chunks, and all three must give the same cut.
"""

from __future__ import annotations

import pytest

from soup_cli.utils.stop_strings import (
    MAX_STOP_LENGTH,
    MAX_STOP_STRINGS,
    StopFilter,
    StopStringError,
    cut_at_stop,
    find_stop,
    parse_stop,
)

# --------------------------------------------------------------------------- #
# parse_stop: the accepted spellings
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ()),
        ([], ()),
        ((), ()),
        ("END", ("END",)),
        ("\n", ("\n",)),
        (" ", (" ",)),
        ("\nUser:", ("\nUser:",)),
        (["END"], ("END",)),
        (["a", "b"], ("a", "b")),
        (["a", "b", "c", "d"], ("a", "b", "c", "d")),
        (("a", "b"), ("a", "b")),
        (["same", "same"], ("same",)),
        (["x" * MAX_STOP_LENGTH], ("x" * MAX_STOP_LENGTH,)),
        ("日本語", ("日本語",)),
    ],
)
def test_parse_stop_accepts(value, expected):
    assert parse_stop(value) == expected


def test_the_bounds_are_the_openai_four_and_a_fixed_length():
    assert MAX_STOP_STRINGS == 4
    assert MAX_STOP_LENGTH == 256


# --------------------------------------------------------------------------- #
# parse_stop: the refused spellings, each naming the field
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", "stop must not be an empty string"),
        ([""], "stop[0] must not be an empty string"),
        (["a", ""], "stop[1] must not be an empty string"),
        (["a", "b", "c", "d", "e"], "stop accepts at most 4 stop strings, got 5"),
        (["a"] * 5, "stop accepts at most 4 stop strings, got 5"),
        ("x" * (MAX_STOP_LENGTH + 1), "stop is longer than 256 characters"),
        (["ok", "x" * (MAX_STOP_LENGTH + 1)], "stop[1] is longer than 256 characters"),
        ([1], "stop[0] must be a string"),
        (["a", None], "stop[1] must be a string"),
        ([["nested"]], "stop[0] must be a string"),
        ([True], "stop[0] must be a string"),
        (7, "stop must be a string or a list of strings"),
        (0, "stop must be a string or a list of strings"),
        (True, "stop must be a string or a list of strings"),
        (False, "stop must be a string or a list of strings"),
        (1.5, "stop must be a string or a list of strings"),
        ({"a": "b"}, "stop must be a string or a list of strings"),
        (b"END", "stop must be a string or a list of strings"),
    ],
)
def test_parse_stop_refuses(value, message):
    with pytest.raises(StopStringError) as caught:
        parse_stop(value)
    assert str(caught.value) == message


@pytest.mark.parametrize("value", ["END", ""])
def test_anthropic_stop_sequences_is_a_list_not_a_string(value):
    with pytest.raises(StopStringError) as caught:
        parse_stop(value, field="stop_sequences", allow_string=False)
    assert str(caught.value) == "stop_sequences must be a list of strings"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, ()), ([], ()), (["a"], ("a",)), (["a", "b"], ("a", "b"))],
)
def test_anthropic_stop_sequences_lists_are_accepted(value, expected):
    assert parse_stop(value, field="stop_sequences", allow_string=False) == expected


def test_anthropic_stop_sequences_other_types_say_list():
    with pytest.raises(StopStringError) as caught:
        parse_stop(7, field="stop_sequences", allow_string=False)
    assert str(caught.value) == "stop_sequences must be a list of strings"


def test_the_error_names_the_field_it_was_given():
    with pytest.raises(StopStringError) as caught:
        parse_stop(["a", ""], field="stop_sequences")
    assert str(caught.value) == "stop_sequences[1] must not be an empty string"
    with pytest.raises(StopStringError) as caught:
        parse_stop(["a"] * 5, field="stop_sequences")
    assert str(caught.value) == "stop_sequences accepts at most 4 stop strings, got 5"


def test_the_error_does_not_echo_the_value():
    secret = "sk-secret-" + "z" * MAX_STOP_LENGTH
    with pytest.raises(StopStringError) as caught:
        parse_stop(secret)
    assert "sk-secret" not in str(caught.value)


def test_stop_string_error_is_a_value_error():
    assert issubclass(StopStringError, ValueError)


def test_a_huge_list_is_refused_by_its_length_alone():
    class Huge(list):
        def __iter__(self):  # pragma: no cover - must never be reached
            raise AssertionError("the entries were walked before the count check")

    with pytest.raises(StopStringError, match="at most 4"):
        parse_stop(Huge(["a"] * 1_000_000))


# --------------------------------------------------------------------------- #
# find_stop / cut_at_stop
# --------------------------------------------------------------------------- #


def test_find_stop_returns_none_without_a_match_or_without_stops():
    assert find_stop("Hello world", ("END",)) is None
    assert find_stop("Hello world", ()) is None
    assert find_stop("", ("END",)) is None


def test_find_stop_match_at_position_zero():
    assert find_stop("END of it", ("END",)) == (0, "END")


def test_find_stop_picks_the_first_occurrence():
    assert find_stop("a END b END", ("END",)) == (2, "END")


def test_find_stop_picks_the_stop_string_that_completes_first():
    # "ab" completes at index 3, "bc" at 4: generation stops at "ab".
    assert find_stop("xabcx", ("bc", "ab")) == (1, "ab")


def test_find_stop_for_nested_strings_follows_which_one_completes_first():
    # "bc" is inside "abcd" and completes first, so it is the one that ends the reply.
    assert find_stop("xabcd", ("abcd", "bc")) == (2, "bc")
    assert find_stop("xabcd", ("bc", "abcd")) == (2, "bc")


def test_find_stop_tie_on_the_end_takes_the_longer_string():
    # Both end at index 4; the longer one starts first, so the cut excludes both.
    assert find_stop("xabc", ("c", "abc")) == (1, "abc")
    assert find_stop("xabc", ("abc", "c")) == (1, "abc")


def test_find_stop_overlapping_strings():
    assert find_stop("aaa", ("aa", "aaa")) == (0, "aa")


def test_cut_at_stop_without_a_match_changes_nothing():
    assert cut_at_stop("Hello world", ("END",)) == ("Hello world", None)
    assert cut_at_stop("Hello world", ()) == ("Hello world", None)


def test_cut_at_stop_drops_the_stop_string_and_everything_after():
    assert cut_at_stop("Hello\nUser: hi\nAssistant: yo", ("\nUser:",)) == ("Hello", "\nUser:")


def test_cut_at_stop_at_position_zero_is_empty():
    assert cut_at_stop("END and more", ("END",)) == ("", "END")


def test_cut_at_stop_drops_the_whitespace_before_the_stop_like_strip_does():
    assert cut_at_stop("Hello world \n\nUser: hi", ("User:",)) == ("Hello world", "User:")


def test_cut_at_stop_keeps_interior_whitespace():
    assert cut_at_stop("a  b\nc END d", ("END",)) == ("a  b\nc", "END")


# --------------------------------------------------------------------------- #
# StopFilter: the incremental form
# --------------------------------------------------------------------------- #


def _run_filter(chunks, stops):
    """Feed ``chunks``; return ``(sent, filter)`` where ``sent`` is what left the filter."""
    stopper = StopFilter(stops)
    sent = ""
    for chunk in chunks:
        sent += stopper.feed(chunk)
    return sent, stopper


def _split(text, size):
    return [text[index:index + size] for index in range(0, len(text), size)]


def test_filter_without_stops_passes_everything_through():
    sent, stopper = _run_filter(["Hello", " ", "world "], ())
    assert sent == "Hello world "
    assert stopper.matched is None


def test_filter_holds_back_at_most_the_longest_stop_minus_one_character():
    stopper = StopFilter(("STOP",))
    assert stopper.feed("abcdef") == "abc"  # "def" (3 = 4 - 1 characters) is held
    assert stopper.feed("g") == "d"
    assert stopper.matched is None


def test_filter_holds_nothing_for_a_one_character_stop():
    stopper = StopFilter(("\n",))
    assert stopper.feed("abc") == "abc"
    assert stopper.feed("d") == "d"


def test_filter_never_sends_the_start_of_a_stop_string_that_completes_later():
    stopper = StopFilter(("\nUser:",))
    sent = stopper.feed("Hello\nUs")
    assert "\n" not in sent and "Us" not in sent
    sent += stopper.feed("er: hi")
    assert sent == "Hello"
    assert stopper.matched == "\nUser:"


def test_filter_match_at_position_zero_sends_nothing():
    sent, stopper = _run_filter(["EN", "D and more"], ("END",))
    assert sent == ""
    assert stopper.matched == "END"


def test_filter_sends_nothing_after_a_match():
    stopper = StopFilter(("END",))
    assert stopper.feed("abc END def") == "abc"
    assert stopper.matched == "END"
    assert stopper.feed("more text") == ""
    assert stopper.feed("") == ""
    assert stopper.matched == "END"


def test_filter_holds_trailing_whitespace_until_text_follows_it():
    stopper = StopFilter(("X",))
    assert stopper.feed("Hello ") == "Hello"
    assert stopper.feed("world") == " world"


def test_filter_drops_whitespace_that_precedes_the_stop_string():
    sent, stopper = _run_filter(["Hello world ", "\n", "User: hi"], ("User:",))
    assert sent == "Hello world"
    assert stopper.matched == "User:"


def test_filter_with_overlapping_strings_stops_at_the_one_that_completes_first():
    sent, stopper = _run_filter(list("xabcd"), ("abcd", "bc"))
    assert sent == "xa"
    assert stopper.matched == "bc"


_REPLIES = [
    "Hello world",
    "Hello\nUser: how are you?\nAssistant: fine",
    "END at the start",
    "no stop in this reply at all",
    "ends with the stop END",
    "ends with a partial EN",
    "a stop inside a wordEND and more",
    "two END stops END here",
    "xabcd",
    "xabcx",
    "日本語のEND続き",
    "tab\tseparated\t\tEND",
    "double  space  END  tail",
    "",
]
_STOPS = [
    ("END",),
    ("\nUser:",),
    ("END", "\nUser:"),
    ("abcd", "bc"),
    ("ab", "bc"),
    ("EN",),
    ("e",),
    ("x" * 40,),
    ("END", "END!"),
]


def _final_text(chunks, stops):
    """What ``_stream_response`` ends up sending: the filter's output plus the rest of the cut."""
    stopper = StopFilter(stops)
    sent = ""
    for chunk in chunks:
        sent += stopper.feed(chunk)
    whole = "".join(chunks)
    cut, _ = cut_at_stop(whole, stops)
    assert cut.startswith(sent), (sent, cut)
    return sent + cut[len(sent):], stopper.matched


@pytest.mark.parametrize("stops", _STOPS)
@pytest.mark.parametrize("reply", _REPLIES)
@pytest.mark.parametrize("size", [1, 2, 3, 5, 1000])
def test_every_chunking_gives_the_same_text_as_the_whole_reply(reply, stops, size):
    text = reply.strip()
    expected, expected_match = cut_at_stop(text, stops)

    final, matched = _final_text(_split(text, size), stops)

    assert final == expected
    # The filter sees the stop string itself, whatever the chunking, and the same one.
    assert matched == expected_match


@pytest.mark.parametrize("stops", _STOPS)
@pytest.mark.parametrize("reply", _REPLIES)
def test_text_after_a_stop_string_never_leaves_the_filter(reply, stops):
    text = reply.strip()
    expected, _ = cut_at_stop(text, stops)
    for size in (1, 2, 3, 7):
        sent, _ = _run_filter(_split(text, size), stops)
        assert expected.startswith(sent), (size, sent, expected)
