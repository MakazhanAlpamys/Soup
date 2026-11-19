"""How the ``semantic`` scorer of ``eval.custom`` reads a word (#3013).

``score_semantic`` is Jaccard overlap of the two sides' lowercased words. A word
used to keep the punctuation around it, so ``h2o.`` was not ``h2o``. A word is
now read with its leading and trailing non-word characters removed, a word that
is only punctuation is dropped, and punctuation inside a word stays part of it.
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
