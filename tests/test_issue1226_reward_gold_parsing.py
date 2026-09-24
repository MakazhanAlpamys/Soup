"""Regression tests for #1226: GRPO rewards parse the gold answer the way they parse completions.

``accuracy`` and ``verifiable/math`` extracted the final answer from the completion but compared
it with the RAW gold. With an Alpaca ``output`` or a ShareGPT reference turn such as
``6*7=42\\n#### 42`` exposed as ``answer``, every correct completion scored 0.0 and GRPO got no
signal; meanwhile the 0.5 substring credit paid wrong answers such as ``#### 420``. Both sides
now go through the one parser in ``soup_cli.utils.final_answer``.

The variant list below was written BEFORE the implementation, and the tests iterate all of it.
The boxed-answer fix for #357 matched the ticket's exact string and missed ``\\boxed {C}``, one
space to the left (#396), so no spelling here is checked in isolation.
"""

from __future__ import annotations

import importlib
import json
import re
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pytest

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import DataConfig, TrainingConfig
from soup_cli.data.loader import load_dataset
from soup_cli.trainer.grpo import _prepare_grpo_dataset, _validate_grpo_reward_metadata
from soup_cli.trainer.rewards import (
    _extract_answer,
    accuracy_reward,
    load_reward_fn,
    load_reward_fns,
    math_verify_reward,
    validate_reward_funcs,
)

_REPO = Path(__file__).resolve().parents[1]

# ===========================================================================
# THE VARIANT LIST - enumerated before any code was written.
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
        ["-42", "#### -42", r"\boxed{-42}", "The answer is -42.", "\u221242"],
        [
            "#### -42",
            r"\boxed{-42}",
            "The answer is -42.",
            "#### \u221242",
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
        ["#### -3.25", r"\boxed{-3.25}", "Answer: -3.25"],
        ["#### 3.25", "#### -3.2"],
    ),
}

# Non-numeric golds (accuracy only): (gold, completion, expected accuracy).
STRING_GOLD_CASES = [
    ("Paris", "#### Paris", 1.0),
    ("Paris", r"\boxed{Paris}", 1.0),
    ("Paris", "The answer is Paris.", 1.0),
    ("Paris", "Paris", 1.0),
    ("Paris", "paris", 1.0),
    ("Paris", "Answer: PARIS", 1.0),
    ("Paris", "Let me think.\nParis", 1.0),
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
]

# The partial-credit policy, pinned: (completion, gold, accuracy before #1226, accuracy now).
# The 0.5 substring credit is REMOVED. It paid a wrong answer that mentions the gold exactly as
# much as a right answer in a spelling the parser missed, which is a reward-hacking surface.
POLICY_CHANGES = [
    ("The answer is 42 degrees", "42", 0.5, 1.0),
    ("Six times seven is 42.", "42", 0.5, 1.0),
    (r"Not 42, so \boxed{41}", "42", 0.5, 0.0),
    ("#### 420", "42", 0.5, 0.0),
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

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output, safe to substring-match."""
    return " ".join(_ANSI_RE.sub("", text).split())


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
        assert _extract_answer(r"so x = \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
        assert accuracy_reward(
            _msg(r"so x = \boxed{\frac{1}{2}}"), answer=[r"\frac{1}{2}"]
        ) == [1.0]


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
    def test_non_numeric_golds_compare_as_text(self, gold, completion, expected):
        assert accuracy_reward(_msg(completion), answer=[gold]) == [expected]
        # A gold that is not a number can never be paid by the numeric reward.
        assert math_verify_reward(_msg(completion), answer=[gold]) == [0.0]


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
    "Answer: Paris",
    "#### 76 years",
    "\u221242",
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
        if reference.number is not None:
            assert completion.number == reference.number
            assert math_verify_reward(_msg(spelling), answer=[spelling]) == [1.0]

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
    ("#### 41\n" + r"\boxed{42}", "41"),  # a marker still outranks a box, as before
    ("The answer is 42.", "42"),
    ("Answer: 42", "42"),
    ("answer: 42", "42"),
    ("ANSWER IS 42", "42"),
    ("Final Answer: The final answer is $42$. I hope it is correct.", "42"),
    ("**Answer:** 42", "42"),
    ("The answer is 41, not 42.", "41"),
    ("The answer is 42 (six times seven).", "42"),
    ("The answer is -42.", "-42"),
    ("The answer is \u221242.", "-42"),
    ("The answer is Paris.", "Paris"),
    ("The answer is New York.", "New York"),
]

NO_EXPLICIT_ANSWER = [
    "Just plain text", "Six times seven is 42.", "", "#### ", r"\boxed{}",
    r"\boxed{\frac{1}{2}", "the answer isn't 41; it's 42", "What is the answer?",
]

NUMBERS = [
    ("42", "42"), ("42.0", "42"), ("-42", "-42"), ("+42", "42"), ("\u221242", "-42"),
    ("1,000", "1000"), ("1,000,000", "1000000"), ("1,000.5", "1000.5"), (".5", "0.5"),
    ("-.5", "-0.5"), ("1e5", "100000"), ("1E-3", "0.001"), ("042", "42"), ("42.", "42"),
    ("$42", "42"), (r"\$42", "42"), ("  42  ", "42"), ("1{,}000", "1000"),
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

    @pytest.mark.parametrize(
        "text",
        ["", "   ", "Step 1: multiply.\nStep 2: report.", "6*7=42\n####", r"\boxed{}"],
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
        # A delimited answer is authoritative: no fallback to other numbers in the text.
        marker = parse_completion("6*7 = 42\n#### 42 apples")
        assert (marker.text, marker.number) == ("42 apples", None)
        last_line = parse_completion("Let me think.\nParis")
        assert (last_line.text, last_line.number) == ("Paris", None)
        assert parse_completion("") is None
        assert parse_completion("  \n ") is None


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
# the real data path: loader -> _prepare_grpo_dataset -> validation -> reward
# ===========================================================================

_GROUP = [
    "Six times seven is 42.\n#### 42",
    r"6*7 = \boxed{42}",
    "The product is 42.\n#### 42",
    "I think it is 41.\n#### 41",
]


def _alpaca_rows() -> list[dict]:
    return [
        {"instruction": f"What is 6*7? (v{i})", "input": "", "output": "6*7=42\n#### 42"}
        for i in range(4)
    ]


def _sharegpt_rows() -> list[dict]:
    return [
        {
            "conversations": [
                {"from": "human", "value": f"What is 6*7? (v{i})"},
                {"from": "gpt", "value": r"6 times 7 is \boxed{42}"},
            ]
        }
        for i in range(4)
    ]


def _chatml_rows() -> list[dict]:
    return [
        {
            "messages": [
                {"role": "user", "content": f"What is 6*7? (v{i})"},
                {"role": "assistant", "content": "The answer is 42."},
            ]
        }
        for i in range(4)
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


class TestGrpoDataPath:
    @pytest.mark.parametrize(
        ("fmt", "make_rows"),
        [("alpaca", _alpaca_rows), ("sharegpt", _sharegpt_rows), ("chatml", _chatml_rows)],
    )
    @pytest.mark.parametrize(("reward_fn", "domain"), [("accuracy", None), ("verifiable", "math")])
    def test_three_correct_one_wrong_group_has_reward_variance(
        self, tmp_path, fmt, make_rows, reward_fn, domain
    ):
        path = _write_jsonl(tmp_path / "train.jsonl", make_rows())
        loaded = load_dataset(
            DataConfig(train=str(path), format=fmt, val_split=0), preserve_source_columns=True
        )
        prepared = _prepare_grpo_dataset(loaded["train"])
        tcfg = TrainingConfig(reward_fn=reward_fn, verifiable_domain=domain)
        _validate_grpo_reward_metadata(prepared, tcfg, split="train")

        reward = validate_reward_funcs(
            load_reward_fns(tcfg.reward_fn, verifiable_domain=tcfg.verifiable_domain)
        )[0]
        gold = prepared[0]["answer"]
        rewards = reward(
            completions=[_msg(text)[0] for text in _GROUP], answer=[gold] * len(_GROUP)
        )
        assert rewards == [1.0, 1.0, 1.0, 0.0]
        assert len(set(rewards)) > 1  # non-zero group variance: GRPO gets an advantage signal


# ===========================================================================
# a gold that cannot be parsed is refused before generation
# ===========================================================================


class TestUnparseableGoldRefused:
    @pytest.mark.parametrize(
        ("reward_fn", "domain", "reward_name", "gold"),
        [
            ("verifiable", "math", "verifiable/math", "Paris"),
            ("verifiable", "math", "verifiable/math", "The answer is forty-two."),
            ("verifiable", "math", "verifiable/math", r"\frac{1}{2}"),
            ("verifiable", "math", "verifiable/math", "6*7 is\nforty-two"),
            ("verifiable", "math", "verifiable/math", "6*7=42\n####"),
            ("accuracy", None, "accuracy", "Step 1: multiply six by seven.\nStep 2: report it."),
            ("accuracy", None, "accuracy", "6*7=42\n####"),
            ("accuracy", None, "accuracy", r"\boxed{}"),
            ("accuracy,format", None, "accuracy", "Line one\nLine two"),
        ],
    )
    def test_refusal_names_the_row_and_the_field(self, reward_fn, domain, reward_name, gold):
        tcfg = TrainingConfig(reward_fn=reward_fn, verifiable_domain=domain)
        rows = [
            {"prompt": "q0", "answer": "#### 42"},
            {"prompt": "q1", "answer": "42"},
            {"prompt": "q2", "answer": gold},
        ]
        with pytest.raises(ValueError) as excinfo:
            _validate_grpo_reward_metadata(rows, tcfg, split="train")
        message = str(excinfo.value)
        assert "GRPO train row 2 'answer'" in message
        assert f"reward {reward_name!r}" in message
        assert "####" in message and r"\boxed{}" in message  # says how to fix it

    def test_refusal_names_the_split(self):
        with pytest.raises(ValueError, match=r"GRPO validation row 0 'answer'"):
            _validate_grpo_reward_metadata(
                [{"prompt": "q", "answer": "Paris"}],
                TrainingConfig(reward_fn="verifiable", verifiable_domain="math"),
                split="validation",
            )

    @pytest.mark.parametrize(
        ("reward_fn", "domain", "gold"),
        [
            ("accuracy", None, "Paris"),
            ("accuracy", None, "The answer is forty-two."),
            ("accuracy", None, r"\frac{1}{2}"),
            ("accuracy", None, 42),
            ("verifiable", "math", 42),
            ("verifiable", "math", 3.5),
            ("verifiable", "math", "1,000"),
            ("verifiable", "code", "multi\nline stdout"),
        ]
        + [("accuracy", None, gold) for gold in GOLDS_42.values()]
        + [("verifiable", "math", gold) for gold in GOLDS_42.values()],
    )
    def test_parseable_golds_pass(self, reward_fn, domain, gold):
        _validate_grpo_reward_metadata(
            [{"prompt": "q", "answer": gold}],
            TrainingConfig(reward_fn=reward_fn, verifiable_domain=domain),
            split="train",
        )


# ===========================================================================
# the repo's own GRPO data still validates, and now gets a signal
# ===========================================================================


class TestShippedDataStillValidates:
    def test_the_grpo_reasoning_example_validates_with_its_own_reward(self):
        from soup_cli.utils.final_answer import parse_reference

        example = _REPO / "examples" / "configs" / "grpo_reasoning.yaml"
        cfg = load_config_from_string(example.read_text(encoding="utf-8"))
        assert cfg.training.reward_fn == "accuracy"
        data = DataConfig(
            train=str(_REPO / "examples" / "data" / "reasoning_math.jsonl"),
            format="alpaca",
            val_split=0,
        )
        prepared = _prepare_grpo_dataset(load_dataset(data, preserve_source_columns=True)["train"])
        _validate_grpo_reward_metadata(prepared, cfg.training, split="train")

        parsed = [parse_reference(row["answer"]) for row in prepared]
        assert [p.text for p in parsed] == ["17", "c = 5", "6", "x = 7", "30"]
        assert [p.number for p in parsed] == [Decimal(17), None, Decimal(6), None, Decimal(30)]

    @pytest.mark.parametrize(
        ("module", "reward_fn", "domain"),
        [
            ("calculator", "verifiable", "math"),
            ("guess_number", "verifiable", "math"),
            ("retrieval_qa", "accuracy", None),
        ],
    )
    def test_bundled_rollout_envs_validate_and_pay_a_just_the_value_reply(
        self, module, reward_fn, domain
    ):
        rows = importlib.import_module(f"soup_cli.envs.{module}").rollout([])
        prepared = _prepare_grpo_dataset([dict(row) for row in rows])
        tcfg = TrainingConfig(reward_fn=reward_fn, verifiable_domain=domain)
        _validate_grpo_reward_metadata(prepared, tcfg, split="rollout")

        # Every env prompt says "Reply with just the value/number".
        reward = load_reward_fns(reward_fn, verifiable_domain=domain)[0]
        golds = [row["answer"] for row in prepared]
        assert reward([_msg(gold)[0] for gold in golds], answer=golds) == [1.0] * len(golds)


# ===========================================================================
# the reward-hacking surface: answer spray no longer pays
# ===========================================================================


class TestAnswerSprayNoLongerPays:
    def test_builtin_accuracy_rejects_every_answer_spray_variant(self):
        import soup_cli.utils.reward_stress as rst

        report = rst.run_stress(load_reward_fn("accuracy"), ["42", "17", "1000", "-3"])
        per_attack = {attack.kind: attack for attack in report.attacks}
        assert per_attack["answer_spray"].accepted == 0
        assert report.gameable is False
        assert report.reference_accept == 1.0

    def test_reward_stress_cli_reports_builtin_accuracy_robust(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app as soup_app

        monkeypatch.chdir(tmp_path)
        refs = _write_jsonl(tmp_path / "refs.jsonl", [{"answer": "42"}, {"answer": "17"}])
        report_path = tmp_path / "report.json"
        result = CliRunner().invoke(
            soup_app,
            [
                "reward", "stress", "accuracy",
                "--references", str(refs),
                "--output-report", str(report_path),
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "robust (not gameable)" in _plain(result.output).lower()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        per_attack = {attack["kind"]: attack for attack in report["attacks"]}
        assert per_attack["answer_spray"]["accepted"] == 0


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
        ],
        ids=["open-boxes", "open-boxes-around-a-close", "phrases", "markers", "commas",
             "whitespace", "braces"],
    )
    def test_parsing_is_fast_on_adversarial_output(self, text):
        start = time.perf_counter()
        accuracy_reward(_msg(text), answer=["42"])
        math_verify_reward(_msg(text), answer=["42"])
        assert time.perf_counter() - start < 2.0
