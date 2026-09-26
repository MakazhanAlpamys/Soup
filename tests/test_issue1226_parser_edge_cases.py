r"""Edge cases of the #1226 final-answer parser, found while reviewing the fix.

1. **A time or a ratio is one value.** "3:45" and "1:1,000" are one answer, like "4/5", so an
   answer phrase that states one is compared as text instead of being read as a hedge between
   its numbers (or, for "3:45pm", as the value 3).
2. **A justification inside the answer clause is a hedge.** This pins the ruling in
   ``test_issue1226_reward_gold_parsing.py``: every stand-alone number in the clause counts. A
   comma or a period before the justification ends the clause.
3. **A later answer phrase supersedes a "####" line**, chatter included ("I hope this answer is
   helpful!"): it cannot be told apart from a heading followed by its answer.
4. **A "####" heading is not a gold.** A reference whose "####" line is not a number and has more
   text after it ("#### Solution") may be a markdown heading over an unmarked body. It is
   refused before generation instead of becoming the text gold "Solution", which every
   completion would miss. A completion's "####" line stays its answer.

The linear-time check of the clause scanner is in ``test_issue1226_clause_scanner_is_linear.py``.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from soup_cli.config.schema import TrainingConfig
from soup_cli.trainer.grpo import _validate_grpo_reward_metadata
from soup_cli.trainer.rewards import accuracy_reward, math_verify_reward
from soup_cli.utils.final_answer import FinalAnswer, parse_completion, parse_reference

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
# 2. a justification inside the answer clause is a hedge (the ruling)
# ===========================================================================

# (completion, every number in its clause). Each scores 0.0 against every one of them.
JUSTIFIED_IN_THE_CLAUSE = [
    ("The answer is 42 because 6*7=42.", ["42", "6", "7"]),
    ("The answer is 42 because 6 times 7 is 42.", ["42", "6", "7"]),
    ("The answer is 42 (6*7=42).", ["42", "6", "7"]),
    ("The answer is 42 (option 3).", ["42", "3"]),
]
# A comma, a period or a line end before the justification ends the clause.
JUSTIFIED_AFTER_THE_CLAUSE = [
    "The answer is 42, because 6*7=42.",
    "The answer is 42. Because 6*7=42.",
    "The answer is 42\n6*7=42",
]


class TestAJustificationInsideTheClause:
    @pytest.mark.parametrize(("completion", "numbers"), JUSTIFIED_IN_THE_CLAUSE)
    def test_is_a_hedge_and_pays_none_of_its_numbers(self, completion, numbers):
        assert parse_completion(completion) is None
        assert parse_reference(completion) is None  # as a gold, it is refused
        for number in numbers:
            assert _scores(completion, number) == (0.0, 0.0), number

    @pytest.mark.parametrize("completion", JUSTIFIED_AFTER_THE_CLAUSE)
    def test_after_the_clause_it_is_not_part_of_the_answer(self, completion):
        assert parse_completion(completion).number == Decimal(42)
        assert parse_reference(completion).number == Decimal(42)
        assert _scores(completion, "42") == (1.0, 1.0)


# ===========================================================================
# 3. a later answer phrase supersedes a '####' line, chatter included
# ===========================================================================


class TestALaterAnswerPhraseAfterAMarker:
    def test_a_justification_in_the_later_phrase_makes_it_a_hedge(self):
        completion = "6*7=42\n#### 42\nThe answer is 42 because 6 x 7 = 42."
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
