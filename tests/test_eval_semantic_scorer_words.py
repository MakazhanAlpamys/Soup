"""How the ``semantic`` scorer of ``eval.custom`` reads a word (#3013).

``score_semantic`` is Jaccard overlap of the two sides' lowercased words. A word
used to keep the punctuation around it, so ``h2o.`` was not ``h2o``. A word is
now read with its leading and trailing punctuation and symbols removed, a word
that is only punctuation is dropped, and punctuation inside a word stays part of
it. A combining mark (a vowel sign, a niqqud point, an accent written as its own
character) is part of the word, also at its edge.
"""

from __future__ import annotations

import pytest

from soup_cli.eval.custom import score_semantic


def test_the_report_example_matches_the_answer_word_and_scores_one_seventh() -> None:
    # h2o now matches; the six other words of the sentence stay in the union.
    assert score_semantic("The chemical symbol for water is H2O.", "H2O") == pytest.approx(1 / 7)


def test_a_trailing_full_stop_does_not_change_the_word() -> None:
    assert score_semantic("H2O.", "H2O") == 1.0


@pytest.mark.parametrize("output", ["(h2o)", '"h2o",', "¿h2o?", "...H2O!!", "'h2o'"])
def test_leading_and_trailing_punctuation_is_removed(output: str) -> None:
    assert score_semantic(output, "H2O") == 1.0


@pytest.mark.parametrize("output", ["h2o —", "h2o ...", "— h2o ?", "h2o - !"])
def test_a_word_that_is_only_punctuation_is_dropped(output: str) -> None:
    assert score_semantic(output, "H2O") == 1.0


@pytest.mark.parametrize(
    ("output", "part"),
    [("3.14", "14"), ("3.14", "3"), ("don't", "don"), ("e-mail", "mail"), ("e-mail", "e")],
)
def test_punctuation_inside_a_word_keeps_it_one_word(output: str, part: str) -> None:
    assert score_semantic(output, part) == 0.0
    assert score_semantic(f"{output}.", output) == 1.0


def test_a_non_ascii_word_keeps_its_letters() -> None:
    assert score_semantic("café.", "Café") == 1.0
    assert score_semantic("café", "caf") == 0.0


# Each pair differs only in a final combining mark, so the two are different words. The
# strings are built from escapes so that this file stays ASCII.
COMBINING_MARK_PAIRS = [
    # Hindi: the last vowel sign is U+0940 in one word and U+0942 in the other.
    ("\u0939\u093f\u0928\u094d\u0926\u0940", "\u0939\u093f\u0928\u094d\u0926\u0942"),
    # cafe with a combining acute accent (U+0301) against cafe.
    ("cafe\u0301", "cafe"),
    # Thai: the vowel sign U+0E34 against U+0E35 on the same consonant.
    ("\u0e01\u0e34", "\u0e01\u0e35"),
    # Hebrew: a final kaf with a qamats (U+05B8) against the bare letters.
    ("\u05dc\u05b0\u05da\u05b8", "\u05dc\u05b0\u05da"),
]


@pytest.mark.parametrize(("word", "other"), COMBINING_MARK_PAIRS)
def test_a_final_combining_mark_is_part_of_the_word(word: str, other: str) -> None:
    assert score_semantic(word, other) == 0.0
    assert score_semantic(other, word) == 0.0


@pytest.mark.parametrize("word", [w for pair in COMBINING_MARK_PAIRS for w in pair])
def test_a_full_stop_after_a_combining_mark_is_still_removed(word: str) -> None:
    assert score_semantic(f"{word}.", word) == 1.0
    assert score_semantic(f"({word})", word) == 1.0


@pytest.mark.parametrize("output", ["", "   ", "...", "— ?! ,", "()"])
def test_an_answer_with_no_word_scores_zero(output: str) -> None:
    assert score_semantic(output, "H2O") == 0.0
    assert score_semantic("H2O", output) == 0.0


def test_a_score_can_also_fall_when_two_words_become_one() -> None:
    # Before: {a., a, x} and {a., a, y} share two of four words, 0.5, a match.
    # After: a. is a on both sides, so {a, x} and {a, y} share one of three.
    assert score_semantic("a. a x", "a. a y") == pytest.approx(1 / 3)


@pytest.mark.parametrize(("output", "want"), [("The answer is 4.", 0.25), ("14", 0.0)])
def test_the_docs_table_cells(output: str, want: float) -> None:
    assert score_semantic(output, "4") == pytest.approx(want)


# A minus sign is the ASCII hyphen-minus or U+2212 MINUS SIGN.
@pytest.mark.parametrize(("output", "expected"), [("-4", "4"), ("\u22124.", "4")])
def test_a_minus_sign_before_a_digit_stays_part_of_the_number(output: str, expected: str) -> None:
    assert score_semantic(output, expected) == 0.0
    assert score_semantic(expected, output) == 0.0


def test_the_edges_of_a_signed_number_are_still_removed() -> None:
    assert score_semantic("(-4).", "-4") == 1.0
    assert score_semantic("\u2212(\u22124)", "\u22124") == 1.0


# A leading dot before a digit is part of the number, after a kept sign as well.
@pytest.mark.parametrize(
    ("output", "expected"),
    [(".5", "5"), ("-.5", "5"), ("-.5", ".5"), ("\u2212.5", ".5"), (".5", "0.5")],
)
def test_a_dot_before_a_digit_stays_part_of_the_number(output: str, expected: str) -> None:
    # ``.5`` against ``0.5`` stays 0.0 as on main: two spellings are two words.
    assert score_semantic(output, expected) == 0.0
    assert score_semantic(expected, output) == 0.0


def test_the_edges_of_a_number_with_a_leading_dot_are_still_removed() -> None:
    assert score_semantic("(.5).", ".5") == 1.0
    assert score_semantic("(-.5).", "-.5") == 1.0
    assert score_semantic("h2o.", "h2o") == 1.0


def test_a_hyphen_before_a_letter_is_still_removed() -> None:
    assert score_semantic("-h2o", "h2o") == 1.0


def test_a_symbol_with_a_variation_selector_is_not_a_word() -> None:
    # Sun and snowflake, each followed by U+FE0F VARIATION SELECTOR-16 (category Mn).
    sun, snowflake = "\u2600\ufe0f", "\u2744\ufe0f"
    assert score_semantic(sun, snowflake) == 0.0
    assert score_semantic(f"{sun} h2o", f"{snowflake} h2o") == 1.0
