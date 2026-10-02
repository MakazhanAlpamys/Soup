r"""Edge cases of the #1226 final-answer parser, found while reviewing the fix.

1. **A time or a ratio is one value.** "3:45" and "1:1,000" are one answer, like "4/5", so an
   answer phrase that states one is compared as text instead of being read as a hedge between
   its numbers (or, for "3:45pm", as the value 3).
2. **Parenthetical justifications hedge the answer clause, while connectives end it.** Parenthetical
   justifications belong to the clause and make it a hedge if numbers conflict. The justification
   connectives "because" or "since" (case-insensitive, requiring whitespace on both sides), or
   punctuation, end the clause before the justification.
3. **A later answer phrase supersedes a "####" line**, chatter included ("I hope this answer is
   helpful!"): it cannot be told apart from a heading followed by its answer.
4. **A "####" heading is not a gold.** A reference whose "####" line is not a number and has more
   text after it ("#### Solution") may be a markdown heading over an unmarked body. It is
   refused before generation instead of becoming the text gold "Solution", which every
   completion would miss. A completion's "####" line stays its answer.
5. **A LaTeX line break keeps both backslashes.** ``\\ -14`` is a row break followed by a space,
   not the control space ``\ ``, so it must not normalise to ``\-14``. Every MATH-500 gold with a
   space after a row break is a matrix (7 of 500), and each one matched only a completion that
   spaced its rows identically. The spacing commands themselves are still dropped. After an
   answer phrase a matrix is one answer: a LaTeX environment is a bracket to the value scan, so
   entries that stand apart are not a hedge between several values.

The linear-time check of the clause scanner is in ``test_issue1226_clause_scanner_is_linear.py``.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from soup_cli.config.schema import TrainingConfig
from soup_cli.trainer.grpo import _validate_grpo_reward_metadata
from soup_cli.trainer.rewards import accuracy_reward, math_verify_reward
from soup_cli.utils.final_answer import (
    FinalAnswer,
    normalize_answer,
    parse_completion,
    parse_number,
    parse_reference,
)

_REWARDS = [("accuracy", None), ("verifiable", "math")]


def _msg(text: str) -> list[list[dict]]:
    return [[{"role": "assistant", "content": text}]]


def _scores(completion: str, gold: str) -> tuple[float, float]:
    return (
        accuracy_reward(_msg(completion), answer=[gold])[0],
        math_verify_reward(_msg(completion), answer=[gold])[0],
    )


def _validate(golds: list[str], reward_fn: str, domain: str | None) -> None:
    _validate_grpo_reward_metadata(
        [{"prompt": f"q{index}", "answer": gold} for index, gold in enumerate(golds)],
        TrainingConfig(reward_fn=reward_fn, verifiable_domain=domain),
        split="train",
    )


# ===========================================================================
# 1. a time or a ratio is one value
# ===========================================================================

# (completion, the gold it states). Each is ONE answer, like "4/5", compared as text.
TIMES_AND_RATIOS = [
    ("The answer is 3:45.", "3:45"),
    ("The answer is 3:45 PM.", "3:45 PM"),
    ("The answer is 3:45pm.", "3:45pm"),
    ("Answer: 12:30", "12:30"),
    ("The answer is 12:30:15.", "12:30:15"),
    ("The answer is 3:4.", "3:4"),
    ("The final answer is 1:1,000.", "1:1,000"),
]
# (completion, numbers it contains): none of them is paid as the answer.
TIME_DIGITS = [
    ("The answer is 3:45.", ["3", "45"]),
    ("The answer is 3:45pm.", ["3", "45"]),
    ("The answer is 12:30:15.", ["12", "30", "15"]),
    ("The final answer is 1:10,500.", ["1", "10", "500", "10500"]),
]


class TestATimeOrARatioIsOneValue:
    @pytest.mark.parametrize(("completion", "gold"), TIMES_AND_RATIOS)
    def test_it_is_read_whole_and_compared_as_text(self, completion, gold):
        assert parse_completion(completion) == FinalAnswer(gold, None)
        assert _scores(completion, gold) == (1.0, 1.0)

    @pytest.mark.parametrize(("completion", "numbers"), TIME_DIGITS)
    def test_none_of_its_numbers_is_paid_as_the_answer(self, completion, numbers):
        for number in numbers:
            assert _scores(completion, number) == (0.0, 0.0), number

    @pytest.mark.parametrize(("reward_fn", "domain"), _REWARDS)
    def test_a_gold_that_states_one_is_read_not_refused(self, reward_fn, domain):
        assert parse_reference("The answer is 3:45.") == FinalAnswer("3:45", None)
        _validate(
            ["The answer is 3:45.", "Answer: 12:30 PM", "The answer is 1:1,000."],
            reward_fn,
            domain,
        )

    def test_a_spaced_colon_is_two_values_as_a_spaced_slash_is(self):
        # Only a colon between two digits joins them, as only "4/5" (not "4 / 5") is a fraction.
        for completion in ("The answer is 3 : 4.", "The answer is 4 / 5."):
            assert parse_completion(completion) is None, completion

    def test_a_clause_naming_two_times_pays_neither(self):
        completion = "The answer is 3:45 or 4:15."
        assert _scores(completion, "3:45") == (0.0, 0.0)
        assert _scores(completion, "4:15") == (0.0, 0.0)


# ===========================================================================
# 2. parenthetical justifications hedge, while connectives end the clause
# ===========================================================================

# (completion, every number in its clause). Parenthetical justifications belong to the clause
# and make it a hedge, scoring 0.0 against every one of them.
JUSTIFIED_IN_A_PARENTHESIS = [
    ("The answer is 42 (6*7=42).", ["42", "6", "7"]),
    ("The answer is 42 (option 3).", ["42", "3"]),
]
# A connective ("because", "since"), a comma, a period or a line end before the justification ends
# the clause, so the number before it is the answer.
JUSTIFIED_AFTER_THE_CLAUSE = [
    ("The answer is 42 because 6*7=42.", Decimal(42), "42"),
    ("The answer is 42 because 6 times 7 is 42.", Decimal(42), "42"),
    ("The answer is 42 since 6*7=42.", Decimal(42), "42"),
    ("So the answer is 18 dollars since 9 * 2 = 18.", Decimal(18), "18"),
    ("The answer is 42, because 6*7=42.", Decimal(42), "42"),
    ("The answer is 42. Because 6*7=42.", Decimal(42), "42"),
    ("The answer is 42\n6*7=42", Decimal(42), "42"),
]
# A hedge before the connective remains a hedge; a wrong answer before the connective is not paid.
HEDGED_OR_WRONG_WITH_A_CONNECTIVE = [
    ("The answer is 41 or 42 because I am not sure.", ["41", "42"]),
]


class TestAJustificationInsideTheClause:
    @pytest.mark.parametrize(("completion", "numbers"), JUSTIFIED_IN_A_PARENTHESIS)
    def test_parenthetical_justification_is_a_hedge_and_pays_none_of_its_numbers(
        self, completion, numbers
    ):
        assert parse_completion(completion) is None
        assert parse_reference(completion) is None  # as a gold, it is refused
        for number in numbers:
            assert _scores(completion, number) == (0.0, 0.0), number

    @pytest.mark.parametrize(("completion", "expected_number", "gold"), JUSTIFIED_AFTER_THE_CLAUSE)
    def test_after_the_clause_it_is_not_part_of_the_answer(self, completion, expected_number, gold):
        assert parse_completion(completion).number == expected_number
        ref = parse_reference(completion)
        assert ref is not None
        if ref.number is not None:
            assert ref.number == expected_number
        assert _scores(completion, gold) == (1.0, 1.0)

    @pytest.mark.parametrize(("completion", "numbers"), HEDGED_OR_WRONG_WITH_A_CONNECTIVE)
    def test_hedge_before_connective_remains_a_hedge(self, completion, numbers):
        assert parse_completion(completion) is None
        assert parse_reference(completion) is None
        for number in numbers:
            assert _scores(completion, number) == (0.0, 0.0), number

    def test_wrong_answer_before_connective_scores_zero(self):
        completion = "The answer is 41 because 6*7=42."
        assert parse_completion(completion).number == Decimal(41)
        assert _scores(completion, "42") == (0.0, 0.0)


class TestTheConnectiveBoundaryItself:
    @pytest.mark.parametrize(
        "connective", ["because", "Because", "BECAUSE", "bEcAuSe", "since", "Since", "SINCE"]
    )
    def test_any_case_ends_the_clause(self, connective):
        completion = f"The answer is 42 {connective} 6*7=42."
        assert parse_completion(completion).number == Decimal(42)
        assert _scores(completion, "42") == (1.0, 1.0)

    @pytest.mark.parametrize("separator", ["\t", "\u00a0", "  ", "\n"])
    def test_any_whitespace_on_both_sides(self, separator):
        completion = f"The answer is 42{separator}because{separator}6*7=42."
        assert parse_completion(completion).number == Decimal(42)
        assert _scores(completion, "42") == (1.0, 1.0)

    @pytest.mark.parametrize(
        "completion",
        [
            "The answer is 42 because-6*7=42.",  # glued to what follows
            "The answer is 42 because6*7=42.",
            "The answer is 42-because 6*7=42.",  # glued to what came before
            "The answer is 42 sincerely 6*7=42.",  # a longer word
            "The answer is 42 (which is right since 6*7=42).",  # inside a bracket: an aside
        ],
    )
    def test_only_a_stand_alone_connective_outside_brackets_ends_the_clause(self, completion):
        # Without a cut the clause keeps 6 and 7 next to 42: still a hedge, as before the rule.
        assert parse_completion(completion) is None
        assert _scores(completion, "42") == (0.0, 0.0)

    def test_the_connective_is_not_part_of_a_text_answer(self):
        completion = "The answer is Paris because it is the capital."
        assert parse_completion(completion).text == "Paris"
        assert _scores(completion, "Paris") == (1.0, 1.0)


class TestAnAnswerThatBeginsWithAConnective:
    """The connective ends a clause that has an answer in it; it never empties one."""

    @pytest.mark.parametrize(
        "gold", ["Answer: Because it rains.", "Answer: since 1990", "The answer is Since then."]
    )
    def test_a_text_gold_that_starts_with_one_is_still_a_gold(self, gold):
        assert parse_reference(gold) is not None


# ===========================================================================
# 3. a later answer phrase supersedes a '####' line, chatter included
# ===========================================================================


class TestALaterAnswerPhraseAfterAMarker:
    def test_a_justification_with_connective_in_the_later_phrase_reads_the_answer(self):
        completion = "6*7=42\n#### 42\nThe answer is 42 because 6 x 7 = 42."
        assert parse_completion(completion).number == Decimal(42)
        assert _scores(completion, "42") == (1.0, 1.0)

    def test_a_hedge_in_the_later_phrase_makes_it_a_hedge(self):
        completion = "6*7=42\n#### 42\nThe answer is 41 or 42 because I am not sure."
        assert parse_completion(completion) is None
        assert _scores(completion, "42") == (0.0, 0.0)

    def test_a_later_phrase_with_no_digits_leaves_the_number_to_the_last_number(self):
        completion = "#### 42\nI hope this answer is helpful!"
        assert parse_completion(completion) == FinalAnswer("helpful", Decimal(42))
        assert _scores(completion, "42") == (1.0, 1.0)

    def test_chatter_after_a_text_marker_supersedes_it_as_an_answer_under_a_heading_does(self):
        # Both texts are "#### <text>" followed by a later answer phrase; the parser cannot tell
        # the heading from the answer, so the later phrase wins in both.
        assert _scores("#### Solution\nThe answer is Paris.", "Paris") == (1.0, 1.0)
        assert _scores("#### Paris\nI hope this answer is helpful!", "Paris") == (0.0, 0.0)
        # With no answer phrase after it, the '####' line stays the answer.
        assert _scores("#### Paris\nHope this helps!", "Paris") == (1.0, 1.0)


# ===========================================================================
# 4. a '####' heading is not a gold
# ===========================================================================

# A '####' line that is not a number and has more text after it may be a markdown heading over
# an unmarked body, so a gold cannot rely on it.
HEADING_GOLDS = [
    "#### Solution\nSix times seven is 42.",
    "#### Step 1\n6*7 = 42",
    "#### 1. Multiply\n6*7 = 42",
    "#### Worked solution\n\nMultiply six by seven to get 42.",
    "#### Paris\n(the capital of France)",  # an answer and a note, or a heading: refused
]
# (gold, what it reads as): a '####' line that is a number (never a heading), the last line,
# or superseded by a later box or answer phrase is still read.
MARKER_GOLDS = [
    ("#### 42\nFrom GSM8K.", FinalAnswer("42", Decimal(42))),
    ("#### 1,000\nThat is the total.", FinalAnswer("1,000", Decimal(1000))),
    ("6*7=42\n#### 42\n\n", FinalAnswer("42", Decimal(42))),
    ("#### Paris", FinalAnswer("Paris", None)),
    ("#### Solution\nThe answer is 42.", FinalAnswer("42", Decimal(42))),
    ("#### Solution\n\\boxed{42}", FinalAnswer("42", Decimal(42))),
    ("#### Final Answer\n42", FinalAnswer("42", Decimal(42))),
]


class TestAMarkerHeadingIsNotAGold:
    @pytest.mark.parametrize("gold", HEADING_GOLDS)
    @pytest.mark.parametrize(("reward_fn", "domain"), _REWARDS)
    def test_a_heading_gold_states_no_answer_and_is_refused(self, gold, reward_fn, domain):
        assert parse_reference(gold) is None
        with pytest.raises(ValueError) as excinfo:
            _validate([gold], reward_fn, domain)
        message = str(excinfo.value)
        assert "GRPO train row 0 'answer' states no single final answer" in message
        assert "markdown heading" in message

    def test_every_heading_row_is_counted(self):
        golds = ["42", HEADING_GOLDS[0], "Paris", HEADING_GOLDS[1], HEADING_GOLDS[2]]
        with pytest.raises(ValueError) as excinfo:
            _validate(golds, "verifiable", "math")
        message = str(excinfo.value)
        assert "GRPO train row 1 'answer'" in message
        assert "2 more of the 5 rows have the same problem" in message

    @pytest.mark.parametrize(("gold", "reading"), MARKER_GOLDS)
    @pytest.mark.parametrize(("reward_fn", "domain"), _REWARDS)
    def test_a_marker_that_is_an_answer_is_still_read(self, gold, reading, reward_fn, domain):
        assert parse_reference(gold) == reading
        _validate([gold], reward_fn, domain)

    def test_a_completion_keeps_its_marker_line_as_its_answer(self):
        # A completion is scored, never refused: its '####' line is its answer.
        assert parse_completion("#### Paris\nHope this helps!") == FinalAnswer("Paris", None)
        assert _scores("#### Paris\nHope this helps!", "Paris") == (1.0, 1.0)
        assert _scores("#### 42\nHope this helps!", "42") == (1.0, 1.0)
        assert _scores("#### Solution\nSix times seven is 42.", "42") == (0.0, 0.0)


# ===========================================================================
# 5. a LaTeX line break keeps both backslashes
# ===========================================================================

LINE_BREAK = "\\" * 2  # LaTeX's row break: two backslashes
# The seven MATH-500 golds (HuggingFaceH4/MATH-500, test split) with a space after a row break.
# Every one is a matrix.
MATH500_MATRIX_GOLDS = [
    r"\begin{pmatrix} -1/3 \\ 2/3 \\ 5/3 \end{pmatrix}",
    r"\begin{pmatrix} -2 \\ -14 \\ -7 \end{pmatrix}",
    r"\begin{pmatrix} -1 & 0 \\ 0 & -1 \end{pmatrix}",
    r"\begin{pmatrix} -18 \\ -49 \\ 96 \end{pmatrix}",
    r"\begin{pmatrix} 1/5 \\ -18/5 \end{pmatrix}",
    r"\begin{pmatrix} -7 \\ 16 \\ 5 \end{pmatrix}",
    r"\begin{pmatrix} 16/49 \\ 48/49 \\ 24/49 \end{pmatrix}",
]


def _row_spacings(gold: str) -> list[str]:
    """The same matrix, with its rows spaced the ways a model writes them."""
    no_space_after = gold.replace(LINE_BREAK + " ", LINE_BREAK)
    return [
        gold,
        no_space_after,
        no_space_after.replace(" " + LINE_BREAK, LINE_BREAK),
        "".join(gold.split()),
    ]


MATRIX_SPELLINGS = [
    (gold, spelling) for gold in MATH500_MATRIX_GOLDS for spelling in _row_spacings(gold)
]


class TestALatexLineBreakKeepsBothBackslashes:
    @pytest.mark.parametrize(("gold", "spelling"), MATRIX_SPELLINGS)
    def test_a_matrix_matches_however_its_rows_are_spaced(self, gold, spelling):
        # Either side can carry either spacing, boxed or on a '####' line.
        assert _scores(r"\boxed{" + spelling + "}", gold) == (1.0, 1.0)
        assert _scores(r"\boxed{" + gold + "}", spelling) == (1.0, 1.0)
        assert _scores("#### " + spelling, gold) == (1.0, 1.0)

    def test_normalisation_keeps_a_row_break_whole(self):
        assert normalize_answer(r"-2 \\ -14 \\ -7") == r"-2 \\ -14 \\ -7"

    def test_a_different_matrix_still_scores_zero(self):
        gold = MATH500_MATRIX_GOLDS[1]
        assert _scores(r"\boxed{\begin{pmatrix}-2\\-14\\7\end{pmatrix}}", gold) == (0.0, 0.0)

    def test_a_row_break_is_kept_not_dropped(self):
        # The two entries 1 and 2 are not the one entry 12.
        column = r"\begin{pmatrix} 1 \\ 2 \end{pmatrix}"
        assert _scores(r"\boxed{\begin{pmatrix}1\\2\end{pmatrix}}", column) == (1.0, 1.0)
        assert _scores(r"\boxed{\begin{pmatrix}12\end{pmatrix}}", column) == (0.0, 0.0)

    @pytest.mark.parametrize(("gold", "spelling"), MATRIX_SPELLINGS)
    def test_a_matrix_after_an_answer_phrase_is_one_answer(self, gold, spelling):
        # A matrix's entries are one answer, not several values: no hedge, whatever the spacing.
        assert _scores("The answer is $" + spelling + "$.", gold) == (1.0, 1.0)
        assert _scores("Answer: $" + spelling + "$", gold) == (1.0, 1.0)

    def test_a_phrased_matrix_is_one_text_answer_not_a_number(self):
        column = r"\begin{pmatrix} 3 \\ 4 \end{pmatrix}"
        phrased = "The answer is $" + column + "$."
        assert _scores(phrased, column) == (1.0, 1.0)
        assert _scores(phrased, "3") == (0.0, 0.0)  # not its first entry
        assert parse_reference(phrased) is not None  # a gold written that way is readable
        # The environment closes: a value on each side of the matrix is still a hedge.
        assert _scores("The answer is 5 or $" + column + "$ or 6.", "5") == (0.0, 0.0)

    @pytest.mark.parametrize("command", [r"\,", r"\!", r"\;", r"\:", "\\ "])
    def test_the_spacing_commands_are_still_dropped(self, command):
        # "\\ " here is one backslash and a space: the control space, which is still dropped.
        assert normalize_answer("1" + command + "000") == "1000"
        assert parse_number("1" + command + "000") == Decimal(1000)
