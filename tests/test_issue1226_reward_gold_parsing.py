"""Regression tests for #1226: GRPO rewards parse the gold answer the way they parse completions.

``accuracy`` and ``verifiable/math`` extracted the final answer from the completion but compared
it with the RAW gold. With an Alpaca ``output`` or a ShareGPT reference turn such as
``6*7=42\\n#### 42`` exposed as ``answer``, every correct completion scored 0.0 and GRPO got no
signal; meanwhile the 0.5 substring credit paid wrong answers such as ``#### 420``. Both sides
now go through the one parser in ``soup_cli.utils.final_answer``.

The variant list below was written BEFORE the implementation, and the tests iterate all of it.
The boxed-answer fix for #357 matched the ticket's exact string and missed ``\\boxed {C}``, one
space to the left (#396), so no spelling here is checked in isolation.

The GRPO data side (loader -> validation -> reward, refusals, the validation summary, shipped
data, answer spray) is in ``test_issue1226_grpo_gold_validation.py``.
"""

from __future__ import annotations

import time
from collections import Counter
from decimal import Decimal

import pytest

from soup_cli.config.schema import TrainingConfig
from soup_cli.trainer.grpo import _validate_grpo_reward_metadata
from soup_cli.trainer.rewards import _extract_answer, accuracy_reward, math_verify_reward

MINUS = "\N{MINUS SIGN}"  # U+2212, which models emit for a negative sign

# ===========================================================================
# THE VARIANT LIST - enumerated before any code was written. G/C/W 1-17 came first; the entries
# marked "review" were added before the second round of code (the review of this fix).
# ===========================================================================

# Gold spellings whose final answer is 42. G1-G9 are the table in #1226; G10+ are neighbours.
GOLDS_42 = {
    "G1 bare": "42",
    "G2 bare + newline": "42\n",
    "G3 decimal": "42.0",
    "G4 marker": "#### 42",
    "G5 marker without space": "####42",
    "G6 gsm8k reference": "6*7=42\n#### 42",
    "G7 boxed": r"\boxed{42}",
    "G8 boxed reference": r"6*7 = \boxed{42}",
    "G9 prose reference": "The answer is 42.",
    "G10 answer colon": "Answer: 42",
    "G11 boxed, space before brace": r"\boxed {42}",
    "G12 leading dollar": "$42",
    "G13 inline math": "The final answer is $42$.",
    "G14 multi-line reference ending in Answer:": "Step 1: 6*7 = 42\n\nAnswer: 42",
    "G15 surrounding whitespace": "  42  ",
    "G16 marker + period": "6*7=42\n#### 42.",
    "G17 explicit plus sign": "+42",
    "G18 markdown bold": "**Answer:** 42",
    # review: siblings of G13 / G18, the newline form, and a markdown heading before the answer
    "G19 bold label, colon outside": "**Answer**: 42",
    "G20 bold final label": "**Final Answer**: 42",
    "G21 inline math \\(...\\)": r"\(42\)",
    "G22 display math \\[...\\]": r"\[42\]",
    "G23 phrase + inline math": r"The answer is \(42\).",
    "G24 answer on the next line": "Answer:\n42",
    "G25 bold label, answer below": "**Answer:**\n\n42",
    "G26 heading before a box": "#### Final Answer\n" + r"\boxed{42}",
    "G27 heading before a phrase": "#### Step 1: multiply\n6*7 = 42\nThe answer is 42.",
    "G28 answer-label heading, answer below": "#### Final Answer\n42",
    "G29 heading with an inline label": "#### Answer: 42",
    "G30 marker, then a box": r"#### \boxed{42}",
    "G31 phrase, answer on the next line": "The answer is\n42",
    "G32 dollars around the bare answer": "$42$",
}

# Completions that state the correct answer (42). C1-C10 are the table in #1226.
CORRECT_42 = {
    "C1 marker": "Six times seven is 42.\n#### 42",
    "C2 marker without space": "Six times seven is 42.\n####42",
    "C3 boxed": r"Six times seven is \boxed{42}.",
    "C4 boxed, space before brace": r"Six times seven is \boxed {42}.",
    "C5 boxed, inner spaces": r"Six times seven is \boxed{ 42 }.",
    "C6 prose only": "Six times seven is 42.",
    "C7 marker + trailing line": "#### 42\nHope this helps!",
    "C8 marker + period": "Six times seven.\n#### 42.",
    "C9 marker + dollar": "It costs\n#### $42",
    "C10 decimal": "#### 42.0",
    "C11 answer is": "Six times seven, so the answer is 42.",
    "C12 answer colon": "6 * 7\nAnswer: 42",
    "C13 minerva style": "Final Answer: The final answer is $42$. I hope it is correct.",
    "C14 markdown bold": "**Answer:** 42",
    "C15 boxed inside inline math": r"So $\boxed{42}$",
    "C16 boxed escaped dollar": r"\boxed{\$42}",
    "C17 marker + trailing spaces": "#### 42   ",
    "C18 lowercase answer": "answer: 42",
    "C19 answer, then a clause": "The answer is 42, which is 6 times 7.",
    "C20 five hashes": "##### 42",
    "C21 marker + exclamation": "#### 42!",
    "C22 the last box is the final answer": r"First guess \boxed{41}; corrected: \boxed{42}",
    "C23 answer phrase + unit": "The answer is 42 apples.",
    # review
    "C24 bold label, colon outside": "6 * 7\n**Answer**: 42",
    "C25 bold final label": "**Final Answer**: 42",
    "C26 marker + inline math": r"#### \(42\)",
    "C27 phrase + display math": r"The answer is \[42\]",
    "C28 answer on the next line": "6 * 7\nAnswer:\n42",
    "C29 bold label, answer below": "**Answer:**\n\n42",
    "C30 heading before a box": "#### Step 1\n6 * 7 = 42\n" + r"\boxed{42}",
    "C31 heading before a phrase": "#### Step 1: multiply\n6*7 = 42\nThe answer is 42.",
    "C32 answer-label heading, answer below": "#### Final Answer\n42",
    "C33 heading with an inline label": "#### Answer: 42",
    "C34 marker, then a box": r"#### \boxed{42}",
    "C35 phrase, answer on the next line": "The answer is\n42",
    "C36 the clause names the right answer first": "The answer is 42 apples, not 41.",
    "C37 worded clause, digits in an aside": "The answer is: forty-two (42).",
    "C38 dollars around the bare answer": "$42$",
    "C39 italic label": "*Answer*: 42",
    "C40 a later box supersedes a marker": "#### 41\n" + r"\boxed{42}",
}

# Completions that state a WRONG answer (gold 42). W1-W2 are from #1226. None may score above 0.
WRONG_42 = {
    "W1 mentions the gold, boxes 41": r"It is not 42, so the answer is \boxed{41}.",
    "W2 420": "#### 420",
    "W3 gold in the reasoning, marker 41": "Is it 42?\n#### 41",
    "W4 boxed 420": r"\boxed{420}",
    "W5 answer is 41": "42 is tempting, but the answer is 41.",
    "W6 answer colon 41": "Answer: 41",
    "W7 self-correction": "The answer is 41, not 42.",
    "W8 near decimal": "#### 42.5",
    "W9 sign flipped": "#### -42",
    "W10 scaled": r"\boxed{4.2}",
    "W11 thousands": "#### 4,200",
    "W12 answer spray": "The candidate options considered were 42, 52, 62, and 44.",
    "W13 hedge inside the marker": "#### 42 or 43",
    "W14 the last box is wrong": r"\boxed{42} at first, then \boxed{41}",
    "W15 attached variable": r"\boxed{42x}",
    "W16 power": r"\boxed{42^2}",
    "W17 empty": "",
    # review: the answer clause is read first, so a gold named AFTER it is not the answer
    "W18 clause names the gold after the answer": "The answer is 41 apples, not 42.",
    "W19 colon form of W18": "Answer: 41 apples, not 42",
    "W20 gold in a parenthetical aside": "The answer is 41 apples (not 42).",
    "W21 gold in the next sentence": "The answer is 41 apples. 42 was my first guess.",
    "W22 gold after a semicolon": "The final answer is 41 apples; 42 is wrong.",
    "W23 a later box supersedes a right marker": "#### 42\n" + r"\boxed{41}",
    "W24 self-correction after a marker": "#### 42\nWait, the answer is 41.",
    "W25 bold label, wrong": "**Answer**: 41",
    "W26 answer-label heading, wrong answer below": "#### Final Answer\n41",
}

# Other values: thousands separators, negatives, decimals. (gold spellings, correct, wrong)
NUMERIC_GROUPS = {
    "1000": (
        ["1000", "1,000", "#### 1,000", r"\boxed{1{,}000}", "The answer is 1,000."],
        [
            "#### 1,000",
            r"\boxed{1,000}",
            "#### 1000",
            "The answer is 1,000.",
            r"\boxed{1{,}000}",
            r"\boxed{1,\!000}",
            r"\boxed{1\;000}",
            "#### $1,000",
            "The total is 1,000.",
        ],
        ["#### 1,001", "#### 100", "#### 10,000", r"\boxed{1,00}"],
    ),
    "1000000": (
        ["1000000", "1,000,000"],
        ["#### 1,000,000", r"\boxed{1,000,000}", "#### 1000000"],
        ["#### 1,000,001", "#### 100,000"],
    ),
    "-42": (
        ["-42", "#### -42", r"\boxed{-42}", "The answer is -42.", MINUS + "42", r"\(-42\)"],
        [
            "#### -42",
            r"\boxed{-42}",
            "The answer is -42.",
            "#### " + MINUS + "42",
            "It drops to -42.",
            r"\boxed{ -42 }",
            "#### -$42",
        ],
        ["#### 42", "#### -41", r"\boxed{-4.2}"],
    ),
    "0.5": (
        ["0.5", ".5", "#### 0.5", r"\boxed{0.50}"],
        ["#### 0.5", "#### .5", r"\boxed{0.50}", "The answer is 0.5.", "It is 0.5."],
        ["#### 5", "#### 0.05", "#### 50"],
    ),
    "-3.25": (
        ["-3.25", "#### -3.25"],
        ["#### -3.25", r"\boxed{-3.25}", "Answer: -3.25", r"\[-3.25\]"],
        ["#### 3.25", "#### -3.2"],
    ),
}

# Non-numeric golds: (gold, completion, expected). Both rewards compare them as normalised text.
STRING_GOLD_CASES = [
    ("Paris", "#### Paris", 1.0),
    ("Paris", r"\boxed{Paris}", 1.0),
    ("Paris", "The answer is Paris.", 1.0),
    ("Paris", "Paris", 1.0),
    ("Paris", "paris", 1.0),
    ("Paris", "Answer: PARIS", 1.0),
    ("Paris", "Let me think.\nParis", 1.0),
    ("Paris", "**Answer**: Paris", 1.0),
    ("Paris", "#### London", 0.0),
    ("Paris", "The answer is London, not Paris.", 0.0),
    ("Paris", "Not Paris.", 0.0),
    ("Paris", "Paris or London", 0.0),
    ("76 years", "76 years", 1.0),
    ("76 years", "The answer is 76 years.", 1.0),
    ("76 years", "#### 76 years", 1.0),
    ("76 years", "80 years", 0.0),
    ("76 years", "It is not 76 years.", 0.0),
    (r"\frac{1}{2}", r"\boxed{\frac{1}{2}}", 1.0),
    (r"\boxed{\frac{1}{2}}", r"so x = \boxed{\frac{1}{2}}", 1.0),
    (r"\frac{1}{2}", r"$\frac{1}{2}$", 1.0),
    (r"\frac{1}{2}", r"\boxed{\frac{1}{3}}", 0.0),
    # review (maintainer ruling): LaTeX golds, as in MATH-500
    (r"\frac{14}{3}", r"\boxed{\frac{14}{3}}", 1.0),
    (r"\frac{14}{3}", r"\boxed{\dfrac{14}{3}}", 1.0),
    (r"\dfrac{14}{3}", r"\boxed{\tfrac{14}{3}}", 1.0),
    (r"\frac{14}{3}", r"\boxed{\frac {14} {3}}", 1.0),
    (r"\frac{14}{3}", r"The answer is $\frac{14}{3}$.", 1.0),
    (r"\frac{14}{3}", r"The answer is \(\frac{14}{3}\).", 1.0),
    (r"\frac{14}{3}", r"\boxed{\frac{3}{14}}", 0.0),
    (r"\left( 3, \frac{\pi}{2} \right)", r"\boxed{(3,\frac{\pi}{2})}", 1.0),
    (r"\left( 3, \frac{\pi}{2} \right)", r"The answer is \left(3, \frac{\pi}{2}\right).", 1.0),
    (r"\left( 3, \frac{\pi}{2} \right)", r"\boxed{(3,\frac{\pi}{4})}", 0.0),
    ("p - q", r"\boxed{p-q}", 1.0),
    (r"\(p - q\)", r"\boxed{p - q}", 1.0),
    ("p - q", r"\boxed{q-p}", 0.0),
    (r"90^\circ", r"\boxed{90^\circ}", 1.0),
    (r"90^\circ", r"\boxed{90}", 0.0),  # no unit or degree stripping
    (r"x \leftarrow y", r"\boxed{x \leftarrow y}", 1.0),  # \left is not stripped from \leftarrow
    (r"x \leftarrow y", r"\boxed{x arrow y}", 0.0),
]

# Hedges (ruling): an answer clause that names more than one DISTINCT value states no answer.
# (completion, every value it hedges between). None of them may be paid for any of its values.
HEDGES = [
    ("The answer is either 41 or 42.", ["41", "42"]),
    ("Answer: 41 or 42", ["41", "42"]),
    ("the answer is 41, 42 or 43", ["41", "42", "43"]),
    ("The answer is 42 or 43.", ["42", "43"]),
    ("The answer is 42 (or 43).", ["42", "43"]),  # an aside is part of the clause
    ("**Answer**: 41 or 42", ["41", "42"]),
    ("Answer:\n41 or 42", ["41", "42"]),
    ("#### Final Answer\n41 or 42", ["41", "42"]),
    ("#### 42\nThe answer is 41 or 42.", ["41", "42"]),  # a later phrase supersedes the marker
    ("The final answer is 42; 43 is also possible.", ["42", "43"]),  # "; " + a number goes on
    ("The answer is 41 apples (not 42).", ["41", "42"]),
    ("The answer is 41 apples or 42.", ["41", "42"]),
    ("The answer is 1,000 or 2,000.", ["1000", "2000"]),
    ("The answer is -3 or 3.", ["-3", "3"]),
    ("The answer is $41$ or $42$.", ["41", "42"]),
    ("The answer is 6 times 7 which is 42.", ["6", "7", "42"]),  # boundary: every number counts
    ("The answer is 42 (6 times 7).", ["42", "6", "7"]),  # boundary: so does an aside's
    ("The answer is 3, 4.", ["3", "4"]),  # boundary: a bare list in a phrase is a hedge
]

# Not hedges: ONE value, maybe repeated or respelled; a tuple or an expression is one answer.
# (completion, gold, expected score in both rewards)
SINGLE_VALUE_CLAUSES = [
    ("The answer is 42 (i.e. 42.0).", "42", 1.0),
    ("The answer is 42 or 42.0.", "42", 1.0),
    ("The answer is 42, i.e. 42.0.", "42", 1.0),
    ("The answer is -42 (that is, " + MINUS + "42).", "-42", 1.0),
    ("The answer is 1,000 (one thousand).", "1000", 1.0),
    ("The answer is 41 apples, not 42.", "41", 1.0),  # the comma ends the clause: only 41
    ("The answer is 41 apples, not 42.", "42", 0.0),
    ("The answer is 42 apples, not 41.", "42", 1.0),
    ("The answer is 42. 41 was my first guess.", "42", 1.0),  # ". " ends the clause
    ("The answer is (3, 4).", "(3, 4)", 1.0),  # a tuple is one answer
    (r"The answer is \left(3, \frac{\pi}{2}\right).", r"\left( 3, \frac{\pi}{2} \right)", 1.0),
    (r"The answer is \frac{14}{3}.", r"\frac{14}{3}", 1.0),  # so is an expression...
    (r"The answer is \frac{14}{3}.", "3", 0.0),  # ...and its digits are not its value
    (r"The answer is 2^{10}.", "10", 0.0),
    ("The answer is 2x.", "2", 0.0),
    ("41 or 42", "42", 1.0),  # no answer phrase: an unmarked completion reads its last number
    (r"\boxed{42}. The answer is 41 or 42.", "42", 1.0),  # a box outranks the phrase
    ("#### 41 or 42", "42", 0.0),  # a delimited answer is compared whole: never a number
]

# Hedged golds are a data problem: refused, never compared.
HEDGED_GOLDS = [
    "The answer is either 41 or 42.",
    "Answer: 41 or 42",
    "the answer is 41, 42 or 43",
    "The answer is 42 (or 43).",
    "6*7 is 42.\nThe answer is 42 or 43.",
    "#### Final Answer\n41 or 42",
]

# The partial-credit policy, pinned: (completion, gold, accuracy before #1226, accuracy now).
# The 0.5 substring credit is REMOVED. It paid a wrong answer that mentions the gold exactly as
# much as a right answer in a spelling the parser missed, which is a reward-hacking surface.
POLICY_CHANGES = [
    ("The answer is 42 degrees", "42", 0.5, 1.0),
    ("Six times seven is 42.", "42", 0.5, 1.0),
    ("Not 42.", "42", 0.5, 1.0),  # ruling: an unmarked completion is read by its last number
    (r"Not 42, so \boxed{41}", "42", 0.5, 0.0),
    ("#### 420", "42", 0.5, 0.0),
    ("The answer is 41 apples, not 42.", "42", 0.5, 0.0),
    ("The capital of France is Paris.", "Paris", 0.5, 0.0),
    # A '####' or \boxed{} answer must BE the number: a unit after it is not stripped.
    ("#### 42 apples", "42", 0.5, 0.0),
]

# Control: plain numeric golds that already worked. Identical on origin/main and after the fix.
# (completion, gold, accuracy, math_verify)
CONTROL_BOTH = [
    ("The answer is #### 42", "42", 1.0, 1.0),
    (r"So \boxed{42} is the result", "42", 1.0, 1.0),
    (r"\boxed{3.14}", "3.14", 1.0, 1.0),
    ("#### 3.14159", "3.14160", 0.0, 1.0),
    ("#### 3.141", "3.142", 0.0, 0.6),
    ("#### 3.0", "3.14159", 0.0, 0.0),
    ("#### 100", "42", 0.0, 0.0),
    ("no answer here", "42", 0.0, 0.0),
    ("I don't know", "42", 0.0, 0.0),
    ("#### __import__('os').system('rm')", "42", 0.0, 0.0),
    ("#### -17", "-17", 1.0, 1.0),
    ("#### 42.0", "42.0", 1.0, 1.0),
    ("Wrong answer", "42", 0.0, 0.0),
    ("", "42", 0.0, 0.0),
]
# math_verify only (accuracy's value for these moved to numeric equality by design).
CONTROL_MATH = [
    ("#### 100000", "1e5", 1.0),
    ("#### 42", "042", 1.0),
    ("Six times seven is 42.", "42", 1.0),
    ("The answer is 42", "42", 1.0),
    ("The answer is 42 apples.", "42", 1.0),
    ("#### 5", "+5", 1.0),
    ("The product is 1000", "1000", 1.0),
    ("#### 3.14", " 3.14 \n", 1.0),
]

# ===========================================================================
# helpers
# ===========================================================================


def _msg(text: str) -> list[list[dict]]:
    return [[{"role": "assistant", "content": text}]]


def _scores(completion: str, gold: str) -> tuple[float, float]:
    return (
        accuracy_reward(_msg(completion), answer=[gold])[0],
        math_verify_reward(_msg(completion), answer=[gold])[0],
    )


def _short(name: str) -> str:
    return name.split(" ", 1)[0]


def _pairs(completions: dict[str, str], golds: dict[str, str]) -> list:
    return [
        pytest.param(c_text, g_text, id=f"{_short(c_name)}-{_short(g_name)}")
        for c_name, c_text in completions.items()
        for g_name, g_text in golds.items()
    ]


def _group_params(kind: str) -> list:
    params = []
    for value, (golds, correct, wrong) in NUMERIC_GROUPS.items():
        completions = correct if kind == "correct" else wrong
        for gi, gold in enumerate(golds):
            for ci, completion in enumerate(completions):
                params.append(pytest.param(completion, gold, id=f"{value}-g{gi}-c{ci}"))
    return params


# ===========================================================================
# the issue's own reproduction
# ===========================================================================


class TestIssueRepro:
    def test_gsm8k_reference_gold_pays_correct_completions(self):
        gold = "6*7=42\n#### 42"
        for text in ("Six times seven is 42.\n#### 42", r"6*7 = \boxed{42}"):
            assert accuracy_reward(_msg(text), answer=[gold]) == [1.0], text
            assert math_verify_reward(_msg(text), answer=[gold]) == [1.0], text

    def test_bare_gold_boxed_spellings_and_wrong_answers(self):
        assert accuracy_reward(_msg(r"6*7 = \boxed{42}"), answer=["42"]) == [1.0]
        assert accuracy_reward(_msg(r"6*7 = \boxed {42}"), answer=["42"]) == [1.0]
        assert accuracy_reward(_msg(r"Not 42, so \boxed{41}"), answer=["42"]) == [0.0]
        assert accuracy_reward(_msg("#### 420"), answer=["42"]) == [0.0]

    def test_nested_braces_extract_the_whole_fraction(self):
        text = r"so x = \boxed{\frac{1}{2}}"
        assert _extract_answer(text) == r"\frac{1}{2}"
        assert accuracy_reward(_msg(text), answer=[r"\frac{1}{2}"]) == [1.0]


# ===========================================================================
# the spelling matrix: every gold x every completion, both rewards
# ===========================================================================


class TestSpellingMatrix:
    @pytest.mark.parametrize(("completion", "gold"), _pairs(CORRECT_42, GOLDS_42))
    def test_every_correct_completion_scores_one_against_every_gold(self, completion, gold):
        assert _scores(completion, gold) == (1.0, 1.0)

    @pytest.mark.parametrize(("completion", "gold"), _pairs(WRONG_42, GOLDS_42))
    def test_every_wrong_completion_scores_zero_against_every_gold(self, completion, gold):
        assert _scores(completion, gold) == (0.0, 0.0)

    @pytest.mark.parametrize(("completion", "gold"), _group_params("correct"))
    def test_thousands_negative_and_decimal_spellings_match(self, completion, gold):
        assert _scores(completion, gold) == (1.0, 1.0)

    @pytest.mark.parametrize(("completion", "gold"), _group_params("wrong"))
    def test_thousands_negative_and_decimal_wrong_answers_score_zero(self, completion, gold):
        assert _scores(completion, gold) == (0.0, 0.0)

    @pytest.mark.parametrize(("gold", "completion", "expected"), STRING_GOLD_CASES)
    def test_non_numeric_golds_compare_as_normalised_text_in_both_rewards(
        self, gold, completion, expected
    ):
        assert _scores(completion, gold) == (expected, expected)


# ===========================================================================
# hedges: an answer clause naming more than one distinct value (ruling)
# ===========================================================================


class TestHedges:
    @pytest.mark.parametrize(("completion", "values"), HEDGES)
    def test_a_hedged_answer_extracts_nothing_and_is_paid_for_none_of_its_values(
        self, completion, values
    ):
        from soup_cli.utils.final_answer import extract_final_answer, parse_completion

        assert extract_final_answer(completion) is None
        assert parse_completion(completion) is None
        for value in values:
            assert _scores(completion, value) == (0.0, 0.0), value

    @pytest.mark.parametrize(("completion", "gold", "expected"), SINGLE_VALUE_CLAUSES)
    def test_one_value_repeated_or_respelled_is_not_a_hedge(self, completion, gold, expected):
        assert _scores(completion, gold) == (expected, expected)

    @pytest.mark.parametrize("gold", HEDGED_GOLDS)
    def test_a_hedged_gold_is_not_a_reference(self, gold):
        from soup_cli.utils.final_answer import parse_reference

        assert parse_reference(gold) is None
        for completion in ("#### 41", "#### 42", "#### 43", "41 or 42"):
            assert _scores(completion, gold) == (0.0, 0.0), completion

    def test_the_clause_rule_decides_what_the_hedge_rule_sees(self):
        from soup_cli.utils.final_answer import parse_completion, parse_reference

        # The comma ends the clause, so only 41 is in it: one value, read as 41 (not a hedge).
        assert parse_completion("The answer is 41 apples, not 42.").number == Decimal(41)
        assert parse_reference("The answer is 41 apples, not 42.") is not None
        # The same two numbers inside ONE clause are a hedge.
        assert parse_completion("The answer is 41 apples or 42.") is None
        assert parse_completion("The answer is 41 apples (not 42).") is None
        # A comma that a number follows continues a list rather than ending the clause.
        assert parse_completion("The answer is 41, 42 or 43.") is None
        assert parse_completion("The answer is 41, then 42.").number == Decimal(41)


# ===========================================================================
# the partial-credit policy
# ===========================================================================


class TestPartialCreditPolicy:
    @pytest.mark.parametrize(("completion", "gold", "before", "now"), POLICY_CHANGES)
    def test_policy_is_pinned(self, completion, gold, before, now):
        assert before != now  # every row documents a real change of policy
        assert accuracy_reward(_msg(completion), answer=[gold]) == [now]

    def test_accuracy_pays_only_zero_or_one_across_the_whole_variant_list(self):
        golds = list(GOLDS_42.values()) + ["Paris", "76 years", r"\frac{1}{2}"]
        completions = (
            list(CORRECT_42.values())
            + list(WRONG_42.values())
            + [case[1] for case in STRING_GOLD_CASES]
            + [case[0] for case in POLICY_CHANGES]
        )
        seen = {
            accuracy_reward(_msg(completion), answer=[gold])[0]
            for gold in golds
            for completion in completions
        }
        assert seen == {0.0, 1.0}


# ===========================================================================
# one parser: the gold and the completion go through the same function
# ===========================================================================

REFLEXIVE_SPELLINGS = list(GOLDS_42.values()) + [
    "#### 1,000",
    r"\boxed{1{,}000}",
    "The answer is -3.25.",
    r"\boxed{\frac{1}{2}}",
    r"\boxed{\left( 3, \frac{\pi}{2} \right)}",
    "Answer: Paris",
    "#### 76 years",
    MINUS + "42",
]


class TestOneParser:
    @pytest.mark.parametrize("spelling", REFLEXIVE_SPELLINGS)
    def test_a_spelling_used_as_both_gold_and_completion_matches_itself(self, spelling):
        from soup_cli.utils import final_answer

        reference = final_answer.parse_reference(spelling)
        completion = final_answer.parse_completion(spelling)
        assert reference is not None and completion is not None
        assert completion.text.casefold() == reference.text.casefold()
        assert accuracy_reward(_msg(spelling), answer=[spelling]) == [1.0]
        assert math_verify_reward(_msg(spelling), answer=[spelling]) == [1.0]
        if reference.number is not None:
            assert completion.number == reference.number

    def test_rewards_and_validation_parse_through_the_shared_helper(self, monkeypatch):
        from soup_cli.utils import final_answer

        calls: Counter = Counter()
        real_reference = final_answer.parse_reference
        real_completion = final_answer.parse_completion

        def spy_reference(text):
            calls["reference", text] += 1
            return real_reference(text)

        def spy_completion(text):
            calls["completion", text] += 1
            return real_completion(text)

        monkeypatch.setattr(final_answer, "parse_reference", spy_reference)
        monkeypatch.setattr(final_answer, "parse_completion", spy_completion)

        gold = "6*7=42\n#### 42"
        assert accuracy_reward(_msg("#### 42"), answer=[gold]) == [1.0]
        assert math_verify_reward(_msg("#### 42"), answer=[gold]) == [1.0]
        _validate_grpo_reward_metadata(
            [{"prompt": "q", "answer": gold}], TrainingConfig(reward_fn="accuracy"), split="train"
        )
        assert calls == Counter({("reference", gold): 3, ("completion", "#### 42"): 2})


# ===========================================================================
# the shared helper, directly
# ===========================================================================

EXPLICIT_ANSWERS = [
    ("Some work\n#### 42", "42"),
    ("####42", "42"),
    ("#### 42\nHope this helps!", "42"),
    ("#### 42.", "42"),
    ("#### $42", "42"),
    ("#### 42   ", "42"),
    ("##### 42", "42"),
    ("#### step\n#### 42", "42"),
    ("#### 42 #### 43", "43"),
    ("#### 1,000", "1,000"),
    (r"\boxed{42}", "42"),
    (r"\boxed {42}", "42"),
    (r"\boxed{ 42 }", "42"),
    (r"so x = \boxed{\frac{1}{2}}", r"\frac{1}{2}"),
    (r"\boxed{\sqrt{\frac{1}{2}}}", r"\sqrt{\frac{1}{2}}"),
    (r"\boxed{41} then \boxed{42}", "42"),
    (r"$\boxed{42}$", "42"),
    (r"\boxed{\$42}", "42"),
    (r"\boxed{1{,}000}", "1,000"),
    (r"\boxed{1,\!000}", "1,000"),
    ("#### 41\n" + r"\boxed{42}", "42"),  # a box AFTER a marker supersedes it
    (r"\boxed{41}" + "\n#### 42", "42"),  # a marker after a box still wins
    ("The answer is 42.", "42"),
    ("Answer: 42", "42"),
    ("answer: 42", "42"),
    ("ANSWER IS 42", "42"),
    ("Final Answer: The final answer is $42$. I hope it is correct.", "42"),
    ("**Answer:** 42", "42"),
    ("The answer is 41, not 42.", "41"),
    ("The answer is 42 (six times seven).", "42"),
    ("The answer is -42.", "-42"),
    ("The answer is " + MINUS + "42.", "-42"),
    ("The answer is Paris.", "Paris"),
    ("The answer is New York.", "New York"),
    # review
    ("**Answer**: 42", "42"),
    ("**Final Answer**: 42", "42"),
    ("*Answer*: 42", "42"),
    ("__Answer__: 42", "42"),
    (r"#### \(42\)", "42"),
    (r"The answer is \[42\].", "42"),
    ("Answer:\n42", "42"),
    ("**Answer:**\n\n42", "42"),
    ("The answer is\n42", "42"),
    ("#### Final Answer\n" + r"\boxed{42}", "42"),
    ("#### Step 1: multiply\n6*7 = 42\nThe answer is 42.", "42"),
    ("#### Final Answer\n42", "42"),
    ("#### Final Answer:\n\n42", "42"),
    ("#### Answer: 42", "42"),
    (r"#### \boxed{42}", "42"),
    ("#### 42\nWait, the answer is 41.", "41"),
    ("The answer is 41 apples, not 42.", "41 apples"),
    (r"The answer is \left( 3, \frac{\pi}{2} \right).", r"( 3, \frac{\pi}{2} )"),
    ("The answer is (3, 4).", "(3, 4)"),
    (r"\boxed{\dfrac{14}{3}}", r"\frac{14}{3}"),
    (r"\boxed{\tfrac{1}{2}}", r"\frac{1}{2}"),
    (r"The answer is x \leftarrow y.", r"x \leftarrow y"),
]

NO_EXPLICIT_ANSWER = [
    "Just plain text", "Six times seven is 42.", "", "#### ", r"\boxed{}",
    r"\boxed{\frac{1}{2}", "the answer isn't 41; it's 42", "What is the answer?",
    "The answers: 41 and 42",
]

NUMBERS = [
    ("42", "42"), ("42.0", "42"), ("-42", "-42"), ("+42", "42"), (MINUS + "42", "-42"),
    ("1,000", "1000"), ("1,000,000", "1000000"), ("1,000.5", "1000.5"), (".5", "0.5"),
    ("-.5", "-0.5"), ("1e5", "100000"), ("1E-3", "0.001"), ("042", "42"), ("42.", "42"),
    ("$42", "42"), (r"\$42", "42"), ("  42  ", "42"), ("1{,}000", "1000"),
    (r"1\;000", "1000"), (r"\(42\)", "42"), (r"\[-3.5\]", "-3.5"),
]

NOT_NUMBERS = [
    "1,00", "12,34", "42 apples", "4 2", "42x", "x", "", "1/2", r"\frac{1}{2}", "nan",
    "inf", "1_000", "--42", "4.2.1", "0x1A", "42 or 43",
]


class TestSharedHelper:
    @pytest.mark.parametrize(("text", "expected"), EXPLICIT_ANSWERS)
    def test_extract_final_answer(self, text, expected):
        from soup_cli.utils.final_answer import extract_final_answer

        assert extract_final_answer(text) == expected
        # The reward module's private name stays a thin alias of the shared helper.
        assert _extract_answer(text) == expected

    @pytest.mark.parametrize("text", NO_EXPLICIT_ANSWER)
    def test_text_without_an_explicit_answer(self, text):
        from soup_cli.utils.final_answer import extract_final_answer

        assert extract_final_answer(text) is None

    @pytest.mark.parametrize(("text", "expected"), NUMBERS)
    def test_parse_number(self, text, expected):
        from soup_cli.utils.final_answer import parse_number

        assert parse_number(text) == Decimal(expected)

    @pytest.mark.parametrize("text", NOT_NUMBERS)
    def test_parse_number_refuses_anything_that_is_not_exactly_one_number(self, text):
        from soup_cli.utils.final_answer import parse_number

        assert parse_number(text) is None

    def test_reference_readings(self):
        from soup_cli.utils.final_answer import parse_reference

        gsm8k = parse_reference("6*7=42\n#### 42")
        assert (gsm8k.text, gsm8k.number) == ("42", Decimal(42))
        paris = parse_reference("Paris")
        assert (paris.text, paris.number) == ("Paris", None)
        worded = parse_reference("The answer is forty-two.")
        assert (worded.text, worded.number) == ("forty-two", None)
        # A reference is never read as free text: its answer must be stated.
        unit = parse_reference("The answer is 42 apples.")
        assert (unit.text, unit.number) == ("42 apples", None)
        latex = parse_reference(r"\left( 3, \dfrac{\pi}{2} \right)")
        assert (latex.text, latex.number) == (r"( 3, \frac{\pi}{2} )", None)

    @pytest.mark.parametrize(
        "text",
        ["", "   ", "Step 1: multiply.\nStep 2: report.", "6*7=42\n####", r"\boxed{}", "Answer:"],
    )
    def test_a_reference_with_no_statable_answer_is_none(self, text):
        from soup_cli.utils.final_answer import parse_reference

        assert parse_reference(text) is None

    def test_completion_readings(self):
        from soup_cli.utils.final_answer import parse_completion

        prose = parse_completion("Six times seven is 42.")
        assert (prose.text, prose.number) == ("Six times seven is 42", Decimal(42))
        unit = parse_completion("The answer is 42 apples.")
        assert (unit.text, unit.number) == ("42 apples", Decimal(42))
        # The answer clause is read first: a number AFTER the clause is not the answer.
        after = parse_completion("The answer is 41 apples, not 42.")
        assert (after.text, after.number) == ("41 apples", Decimal(41))
        # With no digits in the clause, the completion's last number is read instead.
        worded = parse_completion("The answer is: forty-two (42).")
        assert (worded.text, worded.number) == ("forty-two", Decimal(42))
        # A delimited answer is authoritative: no fallback to other numbers in the text.
        marker = parse_completion("6*7 = 42\n#### 42 apples")
        assert (marker.text, marker.number) == ("42 apples", None)
        last_line = parse_completion("Let me think.\nParis")
        assert (last_line.text, last_line.number) == ("Paris", None)
        assert parse_completion("") is None
        assert parse_completion("  \n ") is None

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            (r"\frac{14}{3}", r"\frac {14} {3}"),
            (r"\dfrac{14}{3}", r"\tfrac{14}{3}"),
            (r"\left( 3, \frac{\pi}{2} \right)", r"(3,\frac{\pi}{2})"),
            (r"\(p - q\)", "p-q"),
            ("New York", "new york"),
        ],
    )
    def test_text_answers_compare_ignoring_whitespace_case_and_latex_wrappers(self, left, right):
        from soup_cli.utils.final_answer import answers_match, parse_reference

        assert answers_match(parse_reference(right), parse_reference(left))
        assert answers_match(parse_reference(left), parse_reference(right))


# ===========================================================================
# exact numbers: Decimal, not float
# ===========================================================================


class TestExactNumbers:
    def test_integers_beyond_two_to_the_53_do_not_collide(self):
        big, neighbour = "12345678901234567890", "12345678901234567891"
        assert float(big) == float(neighbour)  # what a float comparison would have seen
        assert _scores(r"\boxed{" + big + "}", neighbour) == (0.0, 0.0)
        assert _scores(r"\boxed{" + big + "}", big) == (1.0, 1.0)

    def test_an_exponent_beyond_decimal_range_scores_zero_instead_of_raising(self):
        assert math_verify_reward(_msg("#### 1e999999999"), answer=["1"]) == [0.0]
        assert math_verify_reward(_msg("#### 1e999999999"), answer=["1e999999999"]) == [1.0]


# ===========================================================================
# control: data that already worked does not change
# ===========================================================================


class TestControlUnchanged:
    @pytest.mark.parametrize(("completion", "gold", "accuracy", "math"), CONTROL_BOTH)
    def test_plain_numeric_gold_scores_are_unchanged(self, completion, gold, accuracy, math):
        assert _scores(completion, gold) == (accuracy, math)

    @pytest.mark.parametrize(("completion", "gold", "math"), CONTROL_MATH)
    def test_plain_numeric_gold_math_verify_is_unchanged(self, completion, gold, math):
        assert math_verify_reward(_msg(completion), answer=[gold]) == [math]


# ===========================================================================
# untrusted model output cannot make the parser quadratic
# ===========================================================================


class TestAdversarialInputsStayLinear:
    @pytest.mark.parametrize(
        "text",
        [
            "\\boxed{" * 20_000,
            "\\boxed{" * 10_000 + "}" + "\\boxed{" * 10_000,
            "the answer is " * 20_000,
            "#### " * 20_000,
            "1," * 50_000,
            " " * 100_000 + "x",
            "{" * 100_000,
            "answer" + " " * 200_000 + "x",
            "**answer" * 50_000,
            "The answer is " + "(" * 200_000,
            "#### Final Answer\n" * 20_000,
            "The answer is 42," + " " * 20_000 + "x",  # see test_issue1226_clause_scanner_*
        ],
        ids=["open-boxes", "open-boxes-around-a-close", "phrases", "markers", "commas",
             "whitespace", "braces", "phrase-whitespace", "bold-labels", "open-parens",
             "answer-headings", "clause-whitespace"],
    )
    def test_parsing_is_fast_on_adversarial_output(self, text):
        start = time.perf_counter()
        accuracy_reward(_msg(text), answer=["42"])
        math_verify_reward(_msg(text), answer=["42"])
        assert time.perf_counter() - start < 2.0
