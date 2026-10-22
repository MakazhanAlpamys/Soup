"""#1192 — bundled mini_arithmetic must discriminate away from the 1.000 rail."""

from __future__ import annotations

import math
from fractions import Fraction

from soup_cli.eval.forgetting import MINI_ARITHMETIC, _standalone_match, score_answer

# The multi-step rows that took the strong reference off the ceiling. Dropping
# any of them (or restoring the v0.73.2 single-step set) fails this file.
_HARD_ANSWERS = {
    "1786", "69104", "16", "75", "32768", "26", "259200", "998001",
    "31375", "120", "265", "90", "13717421", "360", "19/24", "20",
}


class TestArithmeticFixtureHasHeadroom:
    def test_fixture_keeps_40_rows(self):
        assert len(MINI_ARITHMETIC) == 40

    def test_the_last_16_rows_are_the_hard_set(self):
        assert {item["answer"] for item in MINI_ARITHMETIC[-16:]} == _HARD_ANSWERS

    def test_questions_are_unique(self):
        questions = [item["question"] for item in MINI_ARITHMETIC]
        assert len(questions) == len(set(questions))

    def test_no_hard_answer_is_echoed_by_its_question(self):
        # score_answer is a standalone-token match anywhere in the output, so a
        # model that repeats the question would score any answer it contains.
        for item in MINI_ARITHMETIC[-16:]:
            assert not _standalone_match(item["question"], item["answer"]), item

    def test_every_expected_answer_scores(self):
        for item in MINI_ARITHMETIC:
            assert score_answer(f"The answer is {item['answer']}.", item["answer"]), item


def test_fixture_change_bumps_baseline_provenance_revision():
    from soup_cli.eval.gate_suites import BUNDLED_SCORER_REVISION

    assert BUNDLED_SCORER_REVISION >= 4


# Every hard row's key, recomputed by Python from the numbers in its question.
_HARD_TRUTH = {
    "What is 47 times 38?": 47 * 38,
    "What is 1234 times 56?": 1234 * 56,
    "What is 7 + 3 times 4 - 6 divided by 2?": 7 + 3 * 4 - 6 // 2,
    "A shirt costs 80 dollars. It is discounted by 25 percent, then the "
    "discounted price is raised by 25 percent. What is the final price "
    "in dollars?": int(80 * Fraction(3, 4) * Fraction(5, 4)),
    "What is 2 to the power of 15?": 2**15,
    "What is the remainder when 2024 is divided by 37?": 2024 % 37,
    "How many seconds are in 3 days?": 3 * 24 * 60 * 60,
    "What is 999 times 999?": 999 * 999,
    "What is the sum of all the whole numbers from 1 to 250?": sum(range(1, 251)),
    "What is 17 squared minus 13 squared?": 17**2 - 13**2,
    "A train leaves at 9:47 and arrives at 14:12 the same day. How many "
    "minutes long is the trip?": (14 * 60 + 12) - (9 * 60 + 47),
    "What is 15 percent of 15 percent of 4000?": int(4000 * Fraction(15, 100) ** 2),
    "What is 123456789 divided by 9?": 123456789 // 9,
    "What is the least common multiple of 18, 24 and 40?": math.lcm(18, 24, 40),
    "What is 3/8 plus 5/12, as a fraction in lowest terms?": Fraction(3, 8) + Fraction(5, 12),
    "How many times is the digit 7 written when you write out every whole "
    "number from 1 to 100?": sum(str(n).count("7") for n in range(1, 101)),
}


def test_every_hard_row_key_is_what_python_computes():
    rows = {item["question"]: item["answer"] for item in MINI_ARITHMETIC[-16:]}
    assert set(rows) == set(_HARD_TRUTH), set(rows) ^ set(_HARD_TRUTH)
    for question, value in _HARD_TRUTH.items():
        assert rows[question] == str(value), question
