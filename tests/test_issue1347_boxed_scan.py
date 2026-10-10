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

import time

from soup_cli.utils.final_answer import BoxedAnswer, iter_boxed_answers, parse_completion

# Long enough that the scan's own cost dominates the interpreter's.
_GROWTH_INPUT = 20_000
# An absolute ceiling only a scan that is not linear can reach. The growth ratio next to it
# is what pins linearity; this catches a scan that is slow at every size. It was 3.0, and a
# linear scan measured 3.01 on a four-core runner with coverage on and every core busy with
# another test file, so it leaves ten times that.
_CEILING_SECONDS = 30.0


def _best_seconds(text: str, repeats: int = 3) -> float:
    """The fastest of ``repeats`` full scans, to keep a loaded runner out of the way."""
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        sum(1 for _ in iter_boxed_answers(text))
        best = min(best, time.perf_counter() - started)
    return best


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


class TestIssue1347TheScanGrowsLinearly:
    def test_unclosed_boxes_cost_linear_time(self):
        """A run of ``\boxed{`` that never closes is the scan's worst case: every brace is
        walked and nothing is yielded. Quadratic growth would be ~64x for 8x the input, so
        the bound sits far above the ~8x a linear scan measures."""
        small = r"\boxed{" * _GROWTH_INPUT
        large = r"\boxed{" * (_GROWTH_INPUT * 8)

        assert sum(1 for _ in iter_boxed_answers(small)) == 0

        small_seconds = _best_seconds(small)
        large_seconds = _best_seconds(large)
        assert large_seconds < small_seconds * 20, (small_seconds, large_seconds)
        assert large_seconds < _CEILING_SECONDS, large_seconds

    def test_closed_boxes_cost_linear_time_too(self):
        small = r"\boxed{B} " * _GROWTH_INPUT
        large = r"\boxed{B} " * (_GROWTH_INPUT * 8)

        assert sum(1 for _ in iter_boxed_answers(large)) == _GROWTH_INPUT * 8

        small_seconds = _best_seconds(small)
        large_seconds = _best_seconds(large)
        assert large_seconds < small_seconds * 20, (small_seconds, large_seconds)
        assert large_seconds < _CEILING_SECONDS, large_seconds


class TestIssue1347NestedBoxes:
    def test_nested_closed_boxes_cost_linear_time(self):
        """Every box closes and holds the next, so the boxes' contents overlap. Skipping the
        boxes that hold a box is what keeps this linear: normalising them all would be quadratic
        in the nesting depth."""
        small = r"\boxed{" * 1_000 + "B" + "}" * 1_000
        large = r"\boxed{" * 8_000 + "B" + "}" * 8_000

        small_seconds = _best_seconds(small)
        large_seconds = _best_seconds(large)
        assert large_seconds < small_seconds * 20, (small_seconds, large_seconds)

    def test_a_box_inside_a_box_reads_the_inner_answer(self):
        parsed = parse_completion(r"\boxed{\boxed{42}}")
        assert parsed is not None and parsed.text == "42"
