"""#1192 — bundled mini_mmlu must discriminate away from the 1.000 rail."""

from __future__ import annotations

import math
import re
from collections import Counter

import pytest

from soup_cli.eval.forgetting import MINI_MMLU

# Opening words of the 14 selected candidate rows, in fixture order. Dropping
# any of them (or restoring the v0.73.2 three-option set) fails this file.
_HARD_OPENINGS = (
    "Which planet has the shortest day",
    "The interior angles of a hexagon",
    "Which country has the longest coastline?",
    "How many hearts does an octopus have?",
    "Measured from base to peak",
    "In computing, a nibble",
    "In which phase of mitosis",
    "Which planet has the highest average surface temperature?",
    "Which element has the chemical symbol K?",
    "Which is the smallest of the world's five oceans?",
    "The largest moon of Saturn",
    "The hardest substance in the human body",
    "The Rock of Gibraltar",
    "The capital of Turkey",
)


class TestMmluFixtureHasHeadroom:
    def test_fixture_has_40_rows(self):
        assert len(MINI_MMLU) == 40

    def test_the_last_14_rows_are_the_selected_hard_set(self):
        for item, opening in zip(MINI_MMLU[-14:], _HARD_OPENINGS, strict=True):
            assert item["question"].startswith(opening), item

    def test_hard_rows_offer_four_options(self):
        for item in MINI_MMLU[-14:]:
            assert "(D)" in item["question"], item

    def test_questions_are_unique(self):
        questions = [item["question"] for item in MINI_MMLU]
        assert len(questions) == len(set(questions))


class TestAnswerKeyIsNotLopsided:
    def test_no_letter_answers_more_than_30_percent(self):
        counts = Counter(item["answer"] for item in MINI_MMLU)
        assert max(counts.values()) / len(MINI_MMLU) <= 0.30, counts

    @pytest.mark.parametrize("letter", "ABCD")
    def test_a_constant_letter_model_cannot_score_above_30_percent(self, letter):
        from soup_cli.eval.gate_suites import score_bundled_suite

        assert score_bundled_suite("mini_mmlu", lambda _p: letter) <= 0.30


def test_fixture_change_bumps_baseline_provenance_revision():
    from soup_cli.eval.gate_suites import BUNDLED_SCORER_REVISION

    assert BUNDLED_SCORER_REVISION >= 7


_OPTION_RE = re.compile(r"\(([A-D])\) (.*?)(?= \([A-D]\) |$)")

# The TEXT of the correct option for every row, in fixture order. The key is a
# letter, so the test reads the option that letter points at: a key moved to the
# wrong option, or options re-lettered without their key, fail here. The
# arithmetic rows are computed.
_GOLD_TEXT = (
    str(2 + 2), "Paris", "0C", "carbon dioxide", "1", "Jupiter", "Au", "7",
    "mitochondrion", "sound", str(math.isqrt(81)), "oxygen", "Shakespeare", "32",
    "Pacific", "12", "yen", "mercury", "300,000 km/s", "3", "Sun",
    "genetic information", "Everest", "13", "nitrogen", "vacuum",
    "Jupiter", f"{(6 - 2) * 180} degrees", "Canada", "3", "Mauna Kea", str(8 // 2),
    "anaphase", "Venus", "potassium", "the Arctic", "Titan", "tooth enamel",
    "the United Kingdom", "Ankara",
)

# What each of the 14 new rows asks, up to its option list. Opening words alone
# let the question change under its key.
_HARD_STEMS = (
    "Which planet has the shortest day (fastest rotation)?",
    "The interior angles of a hexagon add up to:",
    "Which country has the longest coastline?",
    "How many hearts does an octopus have?",
    "Measured from base to peak, the tallest mountain on Earth is:",
    "In computing, a nibble is how many bits?",
    "In which phase of mitosis do sister chromatids separate?",
    "Which planet has the highest average surface temperature?",
    "Which element has the chemical symbol K?",
    "Which is the smallest of the world's five oceans?",
    "The largest moon of Saturn is:",
    "The hardest substance in the human body is:",
    "The Rock of Gibraltar is a territory of:",
    "The capital of Turkey is:",
)


def test_every_key_points_at_the_correct_option_text():
    for item, gold in zip(MINI_MMLU, _GOLD_TEXT, strict=True):
        options = dict(_OPTION_RE.findall(item["question"]))
        assert len(set(options.values())) == len(options) >= 3, item
        assert options.get(item["answer"]) == gold, item


def test_every_hard_row_asks_the_question_its_key_answers():
    for item, stem in zip(MINI_MMLU[-14:], _HARD_STEMS, strict=True):
        assert item["question"].startswith(stem + " (A) "), item
