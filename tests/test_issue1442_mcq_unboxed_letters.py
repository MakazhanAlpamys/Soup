"""#1442: `extract_mcq_letter` rejected common UNBOXED answer spellings.

#1365 fixed the boxed spellings (`\\boxed{B.}`, `\\boxed{**B**}`), but the
unboxed set still read as no answer: markdown emphasis (`**B**`,
`Answer: **B**`), a letter followed by a line break and the explanation
(`B\\n\\nBecause ...`), and `B is the correct answer.`. On `MINI_MMLU` a
model that bolds every letter scored 0.000 while the same model writing
plain letters scored 1.000.

Two additions close it: a wrapped lone letter (`**B**`, `*B*`, `__B__`,
`` `B` ``, `"B"`, `'B'`, `[B]`) counts as an explicit commitment — same
last-wins tier as boxed/cue/paren — and a standalone uppercase letter that
LEADS the output counts when a line break or " is"/" -" follows.
"""

from __future__ import annotations

import pytest

from soup_cli.eval.forgetting import MINI_MMLU, extract_mcq_letter, score_answer


class TestIssue1442SpellingsTable:
    """Every spelling in the issue's acceptance table, plus the extras it
    measured as None on main (`98905096`)."""

    @pytest.mark.parametrize(
        "completion",
        [
            "**B**",
            "The answer is **B**.",
            "Answer: **B**",
            "B\n\nExplanation: 2 + 2 = 4.",
            "B is the correct answer.",
            "Option **B**",
            "B\nBecause 2 + 2 = 4.",
            "B\r\nBecause 2 + 2 = 4.",
            "B - because 2 + 2 = 4.",
            "*B*",
            "__B__",
            "`B`",
            "[B]",
            '"B"',
            "'B'",
            "**B** (4)",
            "The correct choice is **B**, since 2 + 2 = 4.",
        ],
    )
    def test_unboxed_spelling_scores(self, completion):
        assert extract_mcq_letter(completion) == "B"
        assert score_answer(completion, "B") is True


class TestIssue1442Controls:
    @pytest.mark.parametrize(
        "completion",
        [
            "B",
            "(B)",
            "The answer is B.",
            r"\boxed{B}",
            "**B.**",
            "**Answer:** B",
        ],
    )
    def test_existing_spellings_still_score(self, completion):
        assert score_answer(completion, "B") is True

    @pytest.mark.parametrize(
        "completion",
        [
            "A cat is an animal.",  # the article "a" is not option A
            "the word 'a' is an article",  # lowercase quoted prose is not option A
            "Berlin",
            r"\boxed{4}",  # a boxed VALUE is not an option letter
            "I choose C",  # a real but wrong letter earns no credit
        ],
    )
    def test_negatives_earn_no_credit(self, completion):
        assert score_answer(completion, "B") is False

    def test_lowercase_wrapped_letter_is_not_an_option(self):
        assert extract_mcq_letter("the word 'a' is an article") is None


class TestIssue1442SuiteLevel:
    def test_mini_mmlu_all_bolded_scores_1_0(self):
        """The issue's headline metric: an always-correct model that bolds
        its letter scored 0.000 before this fix."""
        right = [item["answer"] for item in MINI_MMLU]
        score = sum(score_answer(f"**{a}**", a) for a in right) / len(right)
        assert score == pytest.approx(1.0)

    def test_wrapped_loses_to_a_later_explicit_commitment(self):
        """Last explicit commitment wins: a scratch ``**A**`` followed by a
        self-corrected ``\\boxed{B}`` reads B, not A."""
        assert extract_mcq_letter("first guess **A** ... on reflection \\boxed{B}") == "B"


class TestIssue1442OrderingAndAnchor:
    """Pin the two behaviours the comments promise but the table above only
    exercises in one direction: a wrapped letter joins the last-wins tier
    (so it can override an earlier cue or box, and the later of two wrapped
    letters wins), and the leading form is anchored to the start (so a stray
    ``B is`` mid-prose is not read as an answer)."""

    def test_wrapped_letter_after_an_earlier_cue_wins(self):
        assert extract_mcq_letter("Answer: A. On reflection, **B**") == "B"
        assert score_answer("Answer: A. On reflection, **B**", "B") is True

    def test_wrapped_letter_after_an_earlier_boxed_wins(self):
        assert extract_mcq_letter(r"\boxed{A} ... on reflection **B**") == "B"

    def test_the_later_of_two_wrapped_letters_wins(self):
        assert extract_mcq_letter("first **A**, final **B**") == "B"

    def test_leading_form_is_anchored_to_the_start(self):
        assert extract_mcq_letter("Statement B is false") is None
        assert extract_mcq_letter("Statement B - false") is None
