"""#1192 — bundled mini_common_sense must discriminate away from the 1.000 rail."""

from __future__ import annotations

import re
from collections import Counter

import pytest

from soup_cli.eval.forgetting import MINI_COMMON_SENSE

# Opening words of the 16 selected candidate rows, in fixture order. Dropping
# any of them (or restoring the v0.73.2 three-option set) fails this file.
_HARD_OPENINGS = (
    "You put a completely full, tightly sealed glass bottle",
    "A candle is burning inside a glass jar",
    "Your flight leaves at 7:00",
    "Which would NOT help a phone battery",
    "A heavy box and a light box",
    "A metal spoon and a wooden spoon",
    "Maria is taller than Ben, and Ben is taller than Chen",
    "Maria is taller than Ben, and Chen is taller than Ben",
    "You wash a wool sweater",
    "Ignoring air resistance",
    "Where would you see the stars",
    "It is 3 p.m. where you take off",
    "It is raining and the wind",
    "A fish tank has no lid",
    "You borrowed a friend's book",
    "A clock's hour hand points at 3",
)


class TestCommonSenseFixtureHasHeadroom:
    def test_fixture_keeps_40_rows(self):
        assert len(MINI_COMMON_SENSE) == 40

    def test_the_last_16_rows_are_the_selected_hard_set(self):
        hard = MINI_COMMON_SENSE[-16:]
        for item, opening in zip(hard, _HARD_OPENINGS, strict=True):
            assert item["question"].startswith(opening), item

    def test_hard_rows_offer_four_options(self):
        for item in MINI_COMMON_SENSE[-16:]:
            assert "(D)" in item["question"], item

    def test_questions_are_unique(self):
        questions = [item["question"] for item in MINI_COMMON_SENSE]
        assert len(questions) == len(set(questions))


class TestAnswerKeyIsNotLopsided:
    def test_no_letter_answers_more_than_30_percent(self):
        counts = Counter(item["answer"] for item in MINI_COMMON_SENSE)
        assert max(counts.values()) / len(MINI_COMMON_SENSE) <= 0.30, counts

    @pytest.mark.parametrize("letter", "ABCD")
    def test_a_constant_letter_model_cannot_score_above_30_percent(self, letter):
        from soup_cli.eval.gate_suites import score_bundled_suite

        assert score_bundled_suite("mini_common_sense", lambda _p: letter) <= 0.30


def test_fixture_change_bumps_baseline_provenance_revision():
    from soup_cli.eval.gate_suites import BUNDLED_SCORER_REVISION

    assert BUNDLED_SCORER_REVISION >= 5


_OPTION_RE = re.compile(r"\(([A-D])\) (.*?)(?= \([A-D]\) |$)")

# The TEXT of the correct option for every row, in fixture order. The key is a
# letter, so the test reads the option that letter points at: a key moved to the
# wrong option, or options re-lettered without their key, fail here. The three
# time rows are computed.
_GOLD_TEXT = (
    "umbrella", "morning", "water", "east", "heated", "key", "look both ways",
    "break", "pen", "hanging it in the sun", "coat", "sunlight", "phone", "washed",
    "light", "grandparent", "scissors", "moldy", "water", "fuel", "feet",
    "refrigerator", "sleep", "clouds",
    "cracked", "go out soon", f"{7 - 2 - 1}:00", "turning the volume up to maximum",
    "move faster", "the metal one", "Chen", "it cannot be told", "shrink",
    "both at the same time", "a countryside field on a clear, moonless night",
    f"{3 + 2 + 1} p.m.", "to the left", "fall", "tell them and offer to replace it",
    f"{(3 - 5) % 12} o'clock",
)

# What each of the 16 new rows asks, up to its option list. Opening words alone
# let the question change under its key ("Who is the shortest?" -> "tallest?").
_HARD_STEMS = (
    "You put a completely full, tightly sealed glass bottle of water in the freezer "
    "overnight. In the morning the bottle is most likely:",
    "A candle is burning inside a glass jar and you screw the lid on. The flame will:",
    "Your flight leaves at 7:00, the airport is 1 hour away, and you must arrive 2 hours "
    "before the flight. The latest time you can leave home is:",
    "Which would NOT help a phone battery last longer?",
    "A heavy box and a light box of the same size are pushed across the same floor with "
    "the same force. Compared with the heavy box, the light box will:",
    "A metal spoon and a wooden spoon are left standing in a pot of boiling soup for a "
    "few minutes. Which handle is hotter to touch?",
    "Maria is taller than Ben, and Ben is taller than Chen. Who is the shortest?",
    "Maria is taller than Ben, and Chen is taller than Ben. Who is the tallest?",
    "You wash a wool sweater in very hot water and then tumble-dry it on high heat. It "
    "will most likely:",
    "Ignoring air resistance, you drop a rubber ball and a stone of the same size from "
    "the same window at the same moment. Which lands first?",
    "Where would you see the stars most clearly?",
    "It is 3 p.m. where you take off. You fly for 2 hours and land in a time zone 1 hour "
    "ahead. What is the local time when you land?",
    "It is raining and the wind is blowing hard from your left. To stay as dry as "
    "possible you tilt your umbrella:",
    "A fish tank has no lid and nobody adds water to it for a month. The water level will:",
    "You borrowed a friend's book and spilled coffee on it. The most considerate thing to "
    "do is:",
    "A clock's hour hand points at 3 and its minute hand points at 12. What time did the "
    "clock show five hours earlier?",
)


def test_every_key_points_at_the_correct_option_text():
    for item, gold in zip(MINI_COMMON_SENSE, _GOLD_TEXT, strict=True):
        options = dict(_OPTION_RE.findall(item["question"]))
        assert len(set(options.values())) == len(options) >= 3, item
        assert options.get(item["answer"]) == gold, item


def test_every_hard_row_asks_the_question_its_key_answers():
    for item, stem in zip(MINI_COMMON_SENSE[-16:], _HARD_STEMS, strict=True):
        assert item["question"].startswith(stem + " (A) "), item
