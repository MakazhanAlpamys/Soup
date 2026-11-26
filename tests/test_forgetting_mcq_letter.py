"""MCQ letter extraction: an echo of the option list is not an answer; letter-first answers are.

A model that repeats "(A) x (B) y (C) z" instead of answering used to be credited with the
last marker, so every item keyed on the last option passed. "A is the correct answer." read
as no answer at all when it was not the very first thing in the output.
"""

from __future__ import annotations

import re

import pytest

from soup_cli.eval import forgetting, gate_suites
from soup_cli.eval.forgetting import extract_mcq_letter, score_answer

BOX = "\\boxed"


@pytest.mark.parametrize(
    ("output", "letter"),
    [
        ("A is the correct answer.", "A"),
        (" A is the correct answer.", "A"),
        ("\nA is the correct answer.", "A"),
        ("Well.\n\nC is right.", "C"),
        ("(B) is correct", "B"),
        ("D is the right option", "D"),
        ("A.", "A"),
        ("A)", "A"),
        ("(A)", "A"),
        ("**A**", "A"),
        ("Answer: A", "A"),
        ("The answer is A", "A"),
        ("answer: a", "A"),
        ("(a)", "A"),
        (" A\nbecause it is first", "A"),
        ("\nB\n\nBecause 4 is right", "B"),
        (BOX + "{C}", "C"),
        (BOX + " {C}", "C"),
        ("(A) x (B) y. Answer: " + BOX + "{A}", "A"),
    ],
)
def test_every_spelling_of_a_committed_letter_is_read(output, letter):
    assert extract_mcq_letter(output) == letter
    assert score_answer(output, letter)


@pytest.mark.parametrize(
    "output",
    [
        "A) foo B) bar C) baz D) qux",
        "(A) London (B) Berlin (C) Paris",
        "(A) hands \n(B) head \n(C) feet",
        "(A) 3 (B) 4 (C) 5 (D) 6",
        "(a) one (b) two",
    ],
)
def test_a_bare_echo_of_the_options_is_not_a_choice(output):
    assert extract_mcq_letter(output) is None
    assert not score_answer(output, "A")
    assert not score_answer(output, "C")
    assert not score_answer(output, "D")


@pytest.mark.parametrize(
    ("output", "letter"),
    [
        ("(A) x (B) y (C) z\nB", "B"),  # a bare letter after the echo still counts
        ("(A) 3 (B) 4 (C) 5. Answer: (B)", "B"),  # a cue after the echo wins
        ("(A) x (B) y (C) z\n**B**", "B"),  # a wrapped letter after the echo wins
        ("(A) x (B) y (C) z\nC is the correct answer.", "C"),
        ("(B) 4", "B"),  # one marker is a choice, not an echo
        ("(B) 4 (C) 5", "C"),  # not a list from (A): unchanged behaviour
        ("C. Au", "C"),
    ],
)
def test_choices_around_or_without_an_echo_are_kept(output, letter):
    assert extract_mcq_letter(output) == letter


@pytest.mark.parametrize(
    "output",
    [
        "A dog is correct to bark",  # prose opener
        "a is the correct answer",  # lower-case article
        "cA is correct",  # a letter inside a word
        "Paris is the correct answer",  # a word, not a letter
        "a",  # lone lower-case "a" is the article
        "Berlin",
    ],
)
def test_prose_lower_case_and_letters_inside_words_do_not_match(output):
    assert extract_mcq_letter(output) is None


def test_a_model_that_only_echoes_scores_zero_on_the_bundled_mcq_suites():
    for name in ("MINI_MMLU", "MINI_COMMON_SENSE"):
        items = getattr(forgetting, name)
        echoes = [item["question"][item["question"].find("(A)") :] for item in items]
        assert sum(score_answer(echo, item["answer"]) for echo, item in zip(echoes, items)) == 0
        keyed_last = [
            item
            for item in items
            if re.findall(r"\(([A-J])\)", item["question"])[-1] == item["answer"]
        ]
        assert keyed_last, "the regression needs items keyed on the last option"


def test_scorer_revision_moved_and_fingerprint_detects_a_revert(monkeypatch):
    assert gate_suites.BUNDLED_SCORER_REVISION >= 9
    fixed = gate_suites.bundled_scorer_fingerprint()
    assert fixed == gate_suites.BUNDLED_SCORER_FINGERPRINT

    def last_marker_wins(output):  # the previous reading of an echo
        markers = re.findall(r"\(([A-J])\)", output)
        return markers[-1] if markers else None

    monkeypatch.setattr(gate_suites, "_fingerprint_responses", None)
    monkeypatch.setattr(forgetting, "extract_mcq_letter", last_marker_wins)
    assert gate_suites.bundled_scorer_fingerprint() != fixed
