"""A SHIP under ``--gate loop-v2`` never reads as "the held-out edits passed" when they were not.

The held-out check is ``not_measured`` below 5 edits and ``weak`` below 10, and only a handful
of discordant edits can make it fail, so at the default trigger it almost never does. The
verdict says so in words: the printed gate lines and the reason, and the ``--output`` JSON
(``held_out_note``, ``None`` when the check ran on enough edits).
"""

from __future__ import annotations

import json
import re
from io import StringIO

import pytest
from rich.console import Console

import soup_cli.utils.ship_loop_gate as gate
from soup_cli.utils.ship_loop_gate import corrections_check, held_out_check
from soup_cli.utils.ship_verdict import (
    DECISION_SHIP,
    ShipVerdict,
    TaskWin,
    format_ship_rubric,
    render_ship_panel,
    verdict_to_dict,
)

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _flat(text: str) -> str:
    return " ".join(_ANSI.sub("", text).split())


def _verdict(better: int, worse: int, ties: int) -> ShipVerdict:
    return ShipVerdict(
        decision=DECISION_SHIP,
        task_win=TaskWin(mode="metric", base=0.1, tuned=0.9, won=True),
        benchmark_deltas=(),
        failed_rule=None,
        forgetting_threshold=0.05,
        soup_version="0",
        loop_gate=gate.LoopGate(
            corrections=corrections_check(6, 0, 0, 12, rows=6),
            held_out=held_out_check(better, worse, ties),
            canary_decides=False,
        ),
    )


def _panel_text(verdict: ShipVerdict) -> str:
    buffer = StringIO()
    Console(file=buffer, width=140, force_terminal=True, color_system="standard").print(
        render_ship_panel(verdict)
    )
    return _flat(buffer.getvalue())


class TestPrintedText:
    def test_the_panel_says_the_held_out_edits_were_not_checked_and_how_many_it_had(self):
        text = _panel_text(_verdict(0, 0, 2))

        assert "Held-out edits: better 0, worse 0, ties 2 - not_measured" in text
        assert (
            "held-out edits not checked: 2 of 5 needed; this verdict says nothing about them"
            in text
        )

    def test_the_reason_of_a_ship_does_not_claim_the_held_out_edits_held(self):
        reason = _flat(format_ship_rubric(_verdict(0, 0, 2)))

        assert "no held-out edit significantly worse" not in reason
        assert "held-out edits not checked: 2 of 5 needed" in reason

    def test_a_weak_check_says_so(self):
        text = _panel_text(_verdict(2, 1, 4))

        assert "held-out edits checked on few edits (7; 10 for a firm check): weak evidence" in text

    def test_a_firm_check_has_no_note_and_keeps_the_old_sentence(self):
        text = _panel_text(_verdict(5, 0, 7))
        rubric = _flat(format_ship_rubric(_verdict(5, 0, 7)))

        assert "not checked" not in text and "few edits" not in text
        assert "no held-out edit significantly worse" in rubric


class TestJsonEvidence:
    @pytest.mark.parametrize(
        ("counts", "note"),
        [
            ((0, 0, 0), "held-out edits not checked: 0 of 5 needed; this verdict says nothing"),
            ((0, 0, 4), "held-out edits not checked: 4 of 5 needed"),
            ((1, 1, 5), "held-out edits checked on few edits (7; 10 for a firm check)"),
        ],
    )
    def test_the_output_carries_the_note(self, counts, note):
        data = json.loads(json.dumps(verdict_to_dict(_verdict(*counts))))

        assert data["held_out_note"].startswith(note)
        assert data["held_out"]["outcome"] in ("not_measured", "weak")

    def test_the_note_is_null_when_the_check_ran_on_enough_edits(self):
        data = verdict_to_dict(_verdict(5, 0, 7))

        assert data["held_out_note"] is None and data["held_out"]["outcome"] == "pass"

    def test_without_the_gate_there_is_no_note_key(self):
        verdict = ShipVerdict(
            decision=DECISION_SHIP,
            task_win=TaskWin(mode="metric", base=0.1, tuned=0.9, won=True),
            benchmark_deltas=(), failed_rule=None, forgetting_threshold=0.05, soup_version="0",
        )

        assert "held_out_note" not in verdict_to_dict(verdict)
