"""#1347: `extract_mcq_letter` read `\\boxed{B.}` and `\\boxed{**B**}` as C.

The letter tier kept its own boxed regex, `\\boxed\\s*\\{\\s*([A-Ja-j])\\s*\\}`, whose
comment claimed it mirrored the reward parser. It did not. The reward parser
normalises the box content, so a trailing `.` or markdown `*` is dropped and the
braces are balanced, and `utils/final_answer` replaced that regex in #1316.

The two readers disagreed on a completion that echoes the option list. When the
box tier found no letter, the parenthesised tier took the last option of the
echo, so `\\boxed{B.}` after `(A) 3 (B) 4 (C) 5` read C: the right answer scored
wrong and the wrong option scored right.

The tier now reads the boxes through the shared scan in `utils.final_answer` (see
`test_issue1347_boxed_scan.py`), so the two readers cannot diverge again.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from soup_cli.eval import forgetting
from soup_cli.eval.forgetting import extract_mcq_letter, score_answer
from soup_cli.utils.final_answer import parse_completion

# The echo is what made the defect visible: without it, the parenthesised tier
# has nothing to fall back to and the box tier's miss looked like "no answer".
_ECHOED_OPTIONS = "What is 2 + 2? (A) 3 (B) 4 (C) 5\n"


class TestIssue1347ABoxedLetterIsReadAfterAnEchoedOptionList:
    @pytest.mark.parametrize(
        "completion",
        [
            r"\boxed{B}",
            r"\boxed {B}",  # whitespace between the command and its brace
            r"\boxed{ B }",  # whitespace inside the box
            r"\boxed{B.}",  # trailing full stop
            r"\boxed{**B**}",  # markdown emphasis
            r"$\boxed{B}$",  # math delimiters around the box
        ],
    )
    def test_both_readers_read_the_boxed_letter(self, completion):
        output = _ECHOED_OPTIONS + completion
        assert extract_mcq_letter(output) == "B"
        assert parse_completion(output).text == "B"

    def test_the_right_answer_scores_and_the_echoed_option_does_not(self):
        output = _ECHOED_OPTIONS + r"\boxed{B.}"
        assert score_answer(output, "B") is True
        assert score_answer(output, "C") is False


class TestIssue1347TheTierReadsBoxesThroughTheSharedScan:
    def test_the_tier_scans_the_boxes_with_the_shared_scan(self, monkeypatch):
        """A local regex is what let the two readers drift apart: this pins the call."""
        seen: list[str] = []
        output = r"(C) x \boxed{A}"

        def fake_scan(text: str):
            seen.append(text)
            return iter([SimpleNamespace(answer="B", end=len(text))])

        monkeypatch.setattr(forgetting, "iter_boxed_answers", fake_scan)
        assert forgetting.extract_mcq_letter(output) == "B"
        assert seen == [output]

    def test_a_box_that_never_closes_does_not_hide_the_one_before_it(self):
        """The last CLOSED box is the commitment, so a half-written box is not read."""
        assert extract_mcq_letter(r"first \boxed{A}, then \boxed{B") == "A"

    def test_a_boxed_value_is_still_not_an_option_letter(self):
        """``\\boxed{4}`` answers with a value; only A–J in a box is a choice."""
        assert extract_mcq_letter(r"\boxed{4}") is None
        assert extract_mcq_letter(r"\boxed{4}") != "D"
