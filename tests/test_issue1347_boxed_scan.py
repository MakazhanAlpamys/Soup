"""#1347: the shared ``\\boxed{...}`` scan behind both answer readers.

`extract_mcq_letter` (`eval/forgetting.py`) and the reward parser
(`utils/final_answer.py`) each read the same boxed answer. #1316 replaced the
parser's boxed regex with a normalising, brace-balancing read, but the letter
tier kept the old regex whose comment still claimed the two mirrored each other.
They diverged on ``\\boxed{B.}`` and ``\\boxed{**B**}`` (B there, C here).

`iter_boxed_answers` is that read, exposed: one pass over the braces, yielding
every box that closes with its position and normalised content. The parser takes
the last box, and the letter tier takes the last box whose content is one A–J
letter. Neither owns a regex any more.
"""

from __future__ import annotations

from soup_cli.utils.final_answer import BoxedAnswer, iter_boxed_answers, parse_completion
from tests._growth_timing import assert_grows_linearly, linux_only

# The size the growth checks start from. ``tests/_growth_timing.py`` doubles it until one scan
# lasts long enough to time, then compares it with a scan of 8x the input. It replaced a
# 20,000-box input timed once per size and an absolute 30 s ceiling (3.0 s had flaked): the
# ceiling cannot trip when the reference is calibrated, and the ratio is what pins linearity.
# The start is small on purpose: a super-linear scan then costs about 64x the calibrated
# reference to catch, not 64x a large start.
_GROWTH_INPUT = 100


def _scan(text: str) -> int:
    return sum(1 for _ in iter_boxed_answers(text))


def _unclosed(count: int) -> str:
    return r"\boxed{" * count


def _closed(count: int) -> str:
    return r"\boxed{B} " * count


def _nested(depth: int) -> str:
    return r"\boxed{" * depth + "B" + "}" * depth


class TestIssue1347TheScanReadsEveryClosedBox:
    def test_innermost_boxes_are_yielded_once(self):
        text = r"a \boxed{1} b \boxed{\frac{2}{3}} c \boxed{} d \boxed{4"
        assert [(box.start, box.answer) for box in iter_boxed_answers(text)] == [
            (2, "1"),
            (14, r"\frac{2}{3}"),
            (36, ""),
        ]

    def test_the_positions_bracket_the_box(self):
        text = r"so \boxed{ 42 }."
        (box,) = iter_boxed_answers(text)
        assert isinstance(box, BoxedAnswer)
        assert (box.start, box.end) == (3, len(text) - 1)
        assert (box.answer, text[box.start : box.end]) == ("42", r"\boxed{ 42 }")

    def test_the_last_box_that_closes_is_the_answer(self):
        """Two boxes in one completion: the LAST one is the commitment, as on main."""
        parsed = parse_completion(r"first \boxed{1}, so \boxed{42}")
        assert parsed is not None and parsed.text == "42"

    def test_a_box_that_never_closes_is_skipped(self):
        assert list(iter_boxed_answers(r"\boxed{B")) == []
        assert list(iter_boxed_answers(r"\boxed{B \boxed{C}"))[0].answer == "C"

    def test_the_box_scan_and_the_reward_parser_read_the_same_content(self):
        """The parser takes the scan's last box, so the two agree by construction."""
        for completion in (r"\boxed{B.}", r"\boxed{**B**}", r"\boxed{42}", r"\boxed{ 42 }"):
            (box,) = iter_boxed_answers(completion)
            parsed = parse_completion(completion)
            assert parsed is not None and parsed.text == box.answer


@linux_only
class TestIssue1347TheScanGrowsLinearly:
    def test_unclosed_boxes_cost_linear_time(self):
        """A run of ``\\boxed{`` that never closes is the scan's worst case: every brace is
        walked and nothing is yielded. Quadratic growth would be ~64x for 8x the input, so
        the bound sits far above the ~8x a linear scan measures."""
        assert _scan(_unclosed(8 * _GROWTH_INPUT)) == 0

        assert_grows_linearly(_scan, _unclosed, _GROWTH_INPUT)

    def test_closed_boxes_cost_linear_time_too(self):
        assert _scan(_closed(8 * _GROWTH_INPUT)) == _GROWTH_INPUT * 8

        assert_grows_linearly(_scan, _closed, _GROWTH_INPUT)


class TestIssue1347NestedBoxes:
    @linux_only
    def test_nested_closed_boxes_cost_linear_time(self):
        """Every box closes and holds the next, so the boxes' contents overlap. Skipping the
        boxes that hold a box is what keeps this linear: normalising them all would be quadratic
        in the nesting depth."""
        assert_grows_linearly(_scan, _nested, 100)

    def test_a_box_inside_a_box_reads_the_inner_answer(self):
        parsed = parse_completion(r"\boxed{\boxed{42}}")
        assert parsed is not None and parsed.text == "42"
