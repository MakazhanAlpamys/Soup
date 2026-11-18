"""soup ship --gate loop-v2: did the new version learn the corrections, and keep the held-out edits?

Three layers, none of which loads a model or touches a GPU:

* the scoring and statistics as pure functions (``added_recall_hit``, ``sign_test``, the two
  checks, the gate truth table), pinned with the values of the reference
  implementation of the consuming application (its corrections module and tests);
* the two JSONL inputs, refused at the boundary with exit 3 and a message that names the file,
  the line and the problem;
* the CLI, with ``live_eval.make_generator`` replaced by a recorder so the real
  ``_verdict_live`` wiring runs and the test sees what each side answered.
"""

from __future__ import annotations

import json
import re
from io import StringIO
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pytest
from rich.console import Console

from soup_cli.cli import app
from soup_cli.commands import ship as ship_cmd
from soup_cli.utils import live_eval
from soup_cli.utils import ship_loop_gate as gate
from soup_cli.utils.ship_loop_gate import (
    CorrectionEdit,
    GateInputError,
    HeldOutEdit,
    added_recall_hit,
    added_words,
    check_corrections,
    check_held_out,
    content_words,
    corrections_check,
    decide_loop_gate,
    held_out_check,
    parse_corrections,
    parse_held_out,
    scoring,
    sign_test,
    strip_think,
    style_from_win_rate,
    words_present,
)
from soup_cli.utils.ship_verdict import (
    DECISION_DONT_SHIP,
    DECISION_SHIP,
    BenchmarkDelta,
    ShipVerdict,
    TaskWin,
    apply_loop_gate,
    render_ship_panel,
    verdict_to_dict,
)
from tests.cli_runner import CliRunner
from tests.conftest import strip_ansi

runner = CliRunner()

SCORING = {"method": "added_recall", "threshold": 0.5, "template": None}


# ---------------------------------------------------------------------------
# added_recall: content words, added words, the hit
# ---------------------------------------------------------------------------

# Pinned from the reference implementation: one small fixed English list. Negations and numbers
# are NOT on it.
PINNED_STOP_WORDS = """
a an the and or but if so as of at by for in on to up with from into onto than then that
this these those is are was were be been being am do does did doing have has had having will
would can could shall should may might must it its it's i me my we our you your he him his she
her they them their what which who whom whose when where why how there here also just very too
about any some such own same each all both
"""


def test_the_stop_list_is_zeros_list_exactly():
    assert gate.STOP_WORDS == frozenset(PINNED_STOP_WORDS.split())
    assert isinstance(gate.STOP_WORDS, frozenset)
    assert not {"not", "no", "never"} & gate.STOP_WORDS  # a negation can be the whole correction
    assert not any(word.isdigit() for word in gate.STOP_WORDS)  # so can a number


def test_content_words_are_lowercased_not_casefolded_and_stripped_of_punctuation():
    assert content_words("The Router, is... RESET!") == {"router", "reset"}
    assert content_words("Hold reset for 10 s; wait (then) reconnect.") == {
        "hold", "reset", "10", "s", "wait", "reconnect",
    }
    # letters of any script stay; "it's" is a stop word, a curly apostrophe is part of the word
    assert content_words("Zürich — Tōkyō: it's \"fine\"") == {"zürich", "tōkyō", "fine"}
    assert content_words("it’s") == {"it’s"}
    assert content_words("") == frozenset()
    assert content_words("the of and") == frozenset()
    assert content_words(None) == frozenset()  # type: ignore[arg-type]
    # lower(), not casefold(): the German sharp s is not folded to "ss"
    assert content_words("Straße") == {"straße"}
    # an underscore separates words, a digit is a word
    assert content_words("snake_case 2fa") == {"snake", "case", "2fa"}


def test_added_words_are_the_expected_content_words_the_original_lacked():
    expected = "Hold reset for ten seconds, then reconnect to the default network."
    original = "Press the reset button and wait."
    assert added_words(expected, original) == {
        "hold", "ten", "seconds", "reconnect", "default", "network",
    }


def test_an_edit_that_adds_fewer_than_two_words_uses_all_its_content_words():
    original = "Paris is the capital of France."
    # one added word: all of the content words
    assert added_words("Paris is the capital of France, on the Seine.", original) == {
        "paris", "capital", "france", "seine",
    }
    # none added: all of them
    assert added_words("The capital of France is Paris.", original) == {
        "capital", "france", "paris",
    }
    # exactly two added: those two (the boundary of the fallback)
    assert added_words("Paris is the capital of France, on the Seine river.", original) == {
        "seine", "river",
    }
    assert added_words("", original) == frozenset()
    assert added_words("the of", original) == frozenset()
    assert gate.MIN_ADDED == 2


def test_an_answer_is_a_hit_when_at_least_half_of_the_added_words_are_in_it():
    words = frozenset({"hold", "reset", "ten", "seconds"})
    assert words_present("Hold the RESET button.", words) == 2
    expected, original = "Hold reset ten seconds.", "Nothing."
    assert added_words(expected, original) == words
    # 2 of 4: exactly half
    assert added_recall_hit(expected, original, "Hold the reset button.") is True
    assert added_recall_hit(expected, original, "Just reset it.") is False  # 1 of 4
    assert added_recall_hit(expected, original, "Hold reset for ten seconds") is True
    assert added_recall_hit(expected, original, "hold reset ten") is True  # 3 of 4
    assert gate.THRESHOLD == 0.5


def test_with_no_words_to_find_there_is_nothing_to_hit():
    assert added_recall_hit("the of", "anything", "the of anything at all") is False
    assert added_recall_hit("", "", "") is False


def test_thinking_is_removed_before_scoring():
    expected, original = "Paris, on the Seine.", "A big city."
    thought = "<think>The answer is Paris, on the Seine.</think>It is a large city."
    assert strip_think(thought) == "It is a large city."
    # the words were only in the thinking
    assert added_recall_hit(expected, original, thought) is False
    answer = "<think>hmm</think>\n\nParis, on the Seine."
    assert added_recall_hit(expected, original, answer) is True


def test_strip_think_closed_lone_closing_and_unclosed():
    assert strip_think("<think>a</think>x<think>b</think>y") == "xy"  # every closed block goes
    assert strip_think("<think>\n\n</think>\n\nanswer") == "answer"
    assert strip_think("<THINK>a</THINK>kept") == "kept"  # case-insensitive
    # the opening was in the chat template
    assert strip_think("still thinking</think> The answer.") == "The answer."
    assert strip_think("a</think>b</think> last") == "last"  # after the LAST lone closing tag
    # unclosed: what is before it
    assert strip_think("The answer. <think>cut off mid-thought") == "The answer."
    assert strip_think("plain") == "plain"
    assert strip_think("") == ""
    assert strip_think(None) == ""  # type: ignore[arg-type]


# The reference implementation's own fixtures (from its review): (expected, original,
# answer) -> the added set, the words present, hit or miss.
REFUND_ADDED = {
    "14", "account", "ask", "billing", "days", "purchase", "request", "settings", "within",
}
FIXTURE_REFUND = (
    "Yes, if you ask within 14 days of purchase: Settings, Account, Billing, Request refund.",
    "Yes, you can get a refund for the yearly plan.",
)


@pytest.mark.parametrize(
    ("expected", "original", "answer", "added", "present", "hit"),
    [
        pytest.param(
            *FIXTURE_REFUND,
            "You can request a refund within 14 days in Settings > Billing.",
            REFUND_ADDED, 6, True, id="A-refund-hit",
        ),
        pytest.param(
            *FIXTURE_REFUND,
            "Yes, refunds are possible for the yearly plan.",
            REFUND_ADDED, 0, False, id="B-refund-miss",
        ),
        pytest.param(
            "No, CSV import is not supported; use the JSON importer.",
            "Yes, you can import tasks from a CSV file.",
            "No. CSV is not supported, please use JSON.",
            {"importer", "json", "no", "not", "supported", "use"},
            5, True, id="C-negation-is-content",
        ),
        pytest.param(
            "Да, в течение 14 дней: Настройки, Оплата, Возврат.",
            "Да, можно вернуть деньги.",
            "Возврат в течение 14 дней через Настройки.",
            # Known limitation: there are no Russian stop words, so "в" counts as a content word.
            {"14", "в", "возврат", "дней", "настройки", "оплата", "течение"},
            6, True, id="D-russian",
        ),
    ],
)
def test_zeros_fixtures(expected, original, answer, added, present, hit):
    assert added_words(expected, original) == added
    assert words_present(answer, added_words(expected, original)) == present
    assert added_recall_hit(expected, original, answer) is hit


def test_scoring_object_is_the_vendored_one_and_a_fresh_copy():
    assert scoring() == SCORING
    assert scoring() is not scoring()


# ---------------------------------------------------------------------------
# The one-sided exact sign test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hits", "total", "expected"),
    [
        (5, 5, 0.03125),  # 1/32: the smallest sample that can pass
        (4, 4, 0.0625),  # 1/16: four of four is NOT significant
        (4, 5, 6 / 32),
        (3, 5, 0.5),
        (0, 5, 1.0),
        (1, 1, 0.5),
        (0, 1, 1.0),
        (2, 2, 0.25),
        (7, 8, 9 / 256),  # C(8,7) + C(8,8) over 2**8
        (9, 10, 11 / 1024),
        (6, 10, 386 / 1024),
        (6, 6, 1 / 64),
    ],
)
def test_the_sign_test_is_the_upper_tail_of_binomial_one_half(hits, total, expected):
    assert sign_test(hits, total) == expected  # dyadic rationals: exact, no approx


def test_the_sign_test_edges():
    assert sign_test(0, 0) is None
    assert sign_test(3, 0) is None
    assert sign_test(6, 5) == 0.0  # more than all of them: impossible
    assert 0.0 <= sign_test(30, 60) <= 1.0
    # exact integer arithmetic: no overflow or float drift at a big n
    assert sign_test(500, 1000) == pytest.approx(0.5, abs=0.03)
    assert sign_test(1000, 1000) == 2.0 ** -1000


# ---------------------------------------------------------------------------
# Check 1: corrections learned
# ---------------------------------------------------------------------------


def correction(number: int, expected: str, original: str = "I do not know.", paraphrases=None):
    return CorrectionEdit(
        edit_id=f"e{number}",
        expected=expected,
        original=original,
        paraphrases=tuple(
            paraphrases if paraphrases is not None else (f"q{number} a", f"q{number} b")
        ),
    )


def asker(answers: Dict[str, str]) -> Callable[[str], str]:
    """An ``ask(question)`` from a dict question -> answer (an unknown question gets no answer)."""
    return lambda question: answers.get(question, "")


def test_each_edit_is_fixed_broken_or_a_tie_by_the_number_of_its_paraphrase_hits():
    edits = [
        correction(1, "Hold reset ten seconds."),
        correction(2, "Press the blue button twice."),
        correction(3, "Restart the gateway device."),
    ]
    old = asker({
        "q1 a": "no idea", "q1 b": "no idea",  # edit 1: old 0 hits, new 2 -> fixed
        # edit 2: old 2 hits, new 0 -> broken
        "q2 a": "Press the blue button twice.", "q2 b": "Press the blue button twice.",
        "q3 a": "restart the gateway", "q3 b": "nothing",  # old 1, new 1 -> tie
    })
    new = asker({
        "q1 a": "Hold reset ten seconds.", "q1 b": "hold the reset for ten seconds",
        "q2 a": "dunno", "q2 b": "dunno",
        "q3 a": "nothing", "q3 b": "Restart the gateway device",
    })
    result = check_corrections(old, new, edits)
    assert (result.fixed, result.broken, result.ties) == (1, 1, 1)
    assert (result.n_edits, result.n_paraphrases) == (3, 6)
    assert result.p_value == 0.75  # P(X >= 1) for X ~ Binomial(2, 1/2)
    assert result.outcome == "insufficient"  # fixed + broken = 2 < 5


def test_the_hits_of_an_edit_are_compared_as_counts_not_as_booleans():
    """Old 1 hit vs new 2 is fixed; old 2 vs new 1 is broken. An any()/all() ties both."""
    edits = [
        correction(1, "Hold reset ten seconds."),
        correction(2, "Press the blue button twice."),
    ]
    hit1, hit2 = "hold reset ten seconds", "press blue button twice"
    old = asker({"q1 a": hit1, "q1 b": "no idea", "q2 a": hit2, "q2 b": hit2})
    new = asker({"q1 a": hit1, "q1 b": hit1, "q2 a": hit2, "q2 b": "no idea"})
    result = check_corrections(old, new, edits)
    assert (result.fixed, result.broken, result.ties) == (1, 1, 0)


def test_an_edit_with_three_paraphrases_counts_each_of_them():
    edits = [correction(1, "Hold reset ten seconds.", paraphrases=["a", "b", "c"])]
    hit = "hold reset ten seconds"
    old = asker({"a": hit})  # 1 hit
    new = asker({"a": hit, "b": hit, "c": ""})  # 2 hits
    result = check_corrections(old, new, edits)
    assert (result.fixed, result.broken, result.ties, result.n_paraphrases) == (1, 0, 0, 3)


def test_only_the_added_words_count_for_a_correction_and_thinking_is_ignored():
    # The original answer already said "reset"; only "hold", "ten" and "seconds" were added.
    edit = correction(1, "Hold the reset button ten seconds.", original="Use the reset button.",
                      paraphrases=["p"])
    assert added_words(edit.expected, edit.original) == {"hold", "ten", "seconds"}
    old = asker({"p": "Use the reset button."})
    # the added words are only in the thinking
    new = asker({"p": "<think>hold ten seconds</think>Use the reset button."})
    result = check_corrections(old, new, [edit])
    assert (result.fixed, result.broken, result.ties) == (0, 0, 1)
    new = asker({"p": "<think>hmm</think>Hold it ten seconds."})
    assert check_corrections(old, new, [edit]).fixed == 1


def test_the_old_version_answers_everything_first_and_then_the_new_one():
    edits = [
        correction(1, "Hold reset ten seconds."),
        correction(2, "Press the blue button twice."),
    ]
    seen: List[tuple] = []
    check_corrections(
        lambda q: seen.append(("old", q)) or "", lambda q: seen.append(("new", q)) or "", edits,
    )
    assert [who for who, _q in seen] == ["old"] * 4 + ["new"] * 4
    assert [q for who, q in seen if who == "old"] == [q for who, q in seen if who == "new"]


@pytest.mark.parametrize(
    ("fixed", "broken", "ties", "outcome", "p_value"),
    [
        (5, 0, 0, "pass", 1 / 32),  # exactly the five discordant edits that are required
        (4, 0, 3, "insufficient", 1 / 16),  # four discordant: not enough, however lopsided
        (7, 1, 2, "pass", 9 / 256),
        (4, 1, 0, "fail", 6 / 32),  # five discordant, p = 0.1875
        (1, 4, 5, "fail", 31 / 32),
        (3, 2, 0, "fail", 0.5),
        (0, 5, 0, "fail", 1.0),
        (0, 0, 6, "insufficient", None),  # nothing changed either way: no p-value
        (6, 4, 0, "fail", 386 / 1024),
        (9, 1, 0, "pass", 11 / 1024),
        (3, 1, 0, "insufficient", 5 / 16),  # four discordant
        (2, 3, 4, "fail", 26 / 32),  # five discordant, ties do not count towards the five
    ],
)
def test_corrections_outcomes(fixed, broken, ties, outcome, p_value):
    rows = fixed + broken + ties
    result = corrections_check(fixed, broken, ties, 2 * rows, rows=rows)
    assert result.outcome == outcome
    assert result.p_value == p_value
    assert result.n_edits == rows
    assert result.to_dict() == {
        "fixed": fixed, "broken": broken, "ties": ties, "n_edits": rows, "n_paraphrases": 2 * rows,
        "p_value": p_value, "scoring": SCORING, "outcome": outcome,
    }


def test_no_corrections_is_not_applicable_and_unmeasured_corrections_are_insufficient():
    assert corrections_check(0, 0, 0, 0, rows=0).outcome == "not_applicable"
    assert check_corrections(asker({}), asker({}), []).outcome == "not_applicable"
    # rows that exist but could not be asked (nothing measured) are not "no corrections"
    unmeasured = corrections_check(0, 0, 0, 0, rows=3)
    assert unmeasured.outcome == "insufficient"
    assert unmeasured.p_value is None
    assert unmeasured.n_edits == 0


# ---------------------------------------------------------------------------
# Check 2: held-out edits (non-inferiority)
# ---------------------------------------------------------------------------


def held(number: int, expected: str, original: str = "I do not know.") -> HeldOutEdit:
    return HeldOutEdit(edit_id=f"h{number}", prompt=f"question {number}?", expected=expected,
                       original=original)


def test_held_out_edits_are_asked_the_prompt_and_scored_by_how_many_added_words_each_answer_holds():
    edits = [
        held(1, "Hold reset ten seconds then reconnect."),
        held(2, "Press the blue button twice."),
        held(3, "Restart the gateway device."),
    ]
    # added words: 1 = hold, reset, ten, seconds, reconnect (5); 2 = press, blue, button, twice (4);
    # 3 = restart, gateway, device (3)
    old = asker({"question 1?": "hold reset", "question 2?": "press blue button twice",
                 "question 3?": "restart gateway"})
    new = asker({"question 1?": "hold reset ten", "question 2?": "press blue",
                 "question 3?": "gateway restart"})
    result = check_held_out(old, new, edits)
    assert (result.better, result.worse, result.ties, result.n) == (1, 1, 1, 3)
    assert result.p_value == 0.75
    assert result.outcome == "not_measured"  # fewer than five edits
    assert result.style is None
    assert "style" not in result.to_dict()


def test_held_out_compares_the_recall_not_the_hit():
    """Both answers are under half of the added words, so neither is a hit, but one holds more."""
    edits = [held(1, "Hold reset ten seconds then reconnect to the default network.")]
    result = check_held_out(
        asker({"question 1?": "hold"}), asker({"question 1?": "hold reset"}), edits,
    )
    assert (result.better, result.worse, result.ties) == (1, 0, 0)
    # and two answers that are both hits are still ordered by how much they hold
    edits = [held(2, "Hold reset ten seconds.")]
    result = check_held_out(
        asker({"question 2?": "hold reset"}), asker({"question 2?": "hold reset ten"}), edits,
    )
    assert (result.better, result.worse, result.ties) == (1, 0, 0)


@pytest.mark.parametrize(
    ("better", "worse", "ties", "outcome", "p_value"),
    [
        (0, 0, 0, "not_measured", None),
        (3, 0, 1, "not_measured", 1.0),  # n = 4: below the minimum, even with no worse at all
        (0, 4, 0, "not_measured", 1 / 16),  # n = 4 and all worse: still not measured
        (0, 5, 0, "fail", 1 / 32),  # n = 5: five worse, p = 0.03125 < 0.05
        (0, 4, 1, "weak", 1 / 16),  # n = 5, not significant, fewer than 10
        (3, 2, 0, "weak", 26 / 32),
        (0, 3, 6, "weak", 1 / 8),  # n = 9
        (1, 2, 8, "pass", 0.5),  # n = 11, mostly tied
        (2, 7, 1, "pass", 46 / 512),  # n = 10 and p = 0.09: not significantly worse
        (10, 0, 0, "pass", 1.0),
        (0, 9, 0, "fail", 1 / 512),
        (1, 9, 0, "fail", 11 / 1024),
        (4, 0, 6, "pass", 1.0),  # n = 10: the weak/pass boundary
        (4, 0, 5, "weak", 1.0),  # n = 9
    ],
)
def test_held_out_outcomes(better, worse, ties, outcome, p_value):
    result = held_out_check(better, worse, ties)
    assert result.outcome == outcome
    assert result.p_value == p_value
    assert result.n == better + worse + ties
    assert result.to_dict() == {
        "better": better, "worse": worse, "ties": ties, "n": better + worse + ties,
        "p_value": p_value, "scoring": SCORING, "outcome": outcome,
    }


def test_the_held_out_bounds_are_five_and_ten():
    assert (gate.MIN_HELD_OUT, gate.WEAK_BELOW, gate.SIGNIFICANT) == (5, 10, 0.05)
    assert gate.MIN_DISCORDANT == 5


def test_the_style_note_is_information_only():
    plain = held_out_check(0, 5, 0)
    styled = held_out_check(0, 5, 0, style=style_from_win_rate(0.75, 4))
    assert styled.to_dict()["style"] == {
        "preferred_new": 3, "preferred_old": 1, "n": 4, "p_value": 0.625,
    }
    without = styled.to_dict()
    del without["style"]
    assert without == plain.to_dict()  # it never decides the outcome


def test_style_from_a_pairwise_win_rate_matches_the_reference():
    won = style_from_win_rate(0.75, 4)
    assert (won.preferred_new, won.preferred_old, won.n) == (3, 1, 4)
    assert won.p_value == 0.625  # the two-sided sign test of 3:1
    assert style_from_win_rate(0.25, 8).preferred_new == 2
    assert style_from_win_rate(0.5, 3).preferred_new == 2  # a tie counts half: 1.5 rounds up
    assert style_from_win_rate(0.0, 5).to_dict() == {
        "preferred_new": 0, "preferred_old": 5, "n": 5, "p_value": 0.0625,
    }
    for not_a_rate in (-0.1, 1.5, "x", None, True, float("nan")):
        assert style_from_win_rate(not_a_rate, 4) is None
    assert style_from_win_rate(0.7, 0) is None  # nothing was compared


# ---------------------------------------------------------------------------
# The gate truth table
# ---------------------------------------------------------------------------


def _decide(c1: str, c2: str, suites_rule=None):
    """Build the two checks that have the named outcomes and decide."""
    corrections = {
        "pass": corrections_check(5, 0, 1, 12, rows=6),
        "fail": corrections_check(4, 1, 0, 10, rows=5),
        "insufficient": corrections_check(3, 1, 0, 8, rows=4),
        "not_applicable": corrections_check(0, 0, 0, 0, rows=0),
    }[c1]
    held_out = {
        "pass": held_out_check(1, 0, 10),
        "weak": held_out_check(1, 0, 5),
        "not_measured": held_out_check(0, 0, 2),
        "fail": held_out_check(0, 6, 0),
    }[c2]
    assert (corrections.outcome, held_out.outcome) == (c1, c2)
    return decide_loop_gate(corrections, held_out, suites_failed_rule=suites_rule)


@pytest.mark.parametrize(
    ("c1", "c2", "suites_rule", "ship", "failed_rule", "canary"),
    [
        # SHIP iff check 1 passes AND check 2 is not a fail AND the suites hold
        ("pass", "pass", None, True, None, False),
        ("pass", "weak", None, True, None, False),
        ("pass", "not_measured", None, True, None, False),
        ("pass", "fail", None, False, "held_out", False),
        ("fail", "pass", None, False, "corrections", False),
        ("fail", "weak", None, False, "corrections", False),
        ("fail", "not_measured", None, False, "corrections", False),
        ("insufficient", "pass", None, False, "corrections_insufficient", False),
        ("insufficient", "not_measured", None, False, "corrections_insufficient", False),
        ("not_applicable", "pass", None, False, "no_corrections", True),
        ("not_applicable", "not_measured", None, False, "no_corrections", True),
        ("not_applicable", "weak", None, False, "no_corrections", True),
        # the suites decide on their own, whatever the two checks say
        ("pass", "pass", "regression", False, "regression", False),
        ("pass", "not_measured", "missing_baseline", False, "missing_baseline", False),
        ("not_applicable", "pass", "regression", False, "regression", False),
        ("not_applicable", "pass", "missing_baseline", False, "missing_baseline", False),
        ("insufficient", "pass", "regression", False, "regression", False),
        # more than one thing wrong: the suites first, then the held-out edits, then the corrections
        ("fail", "fail", None, False, "held_out", False),
        ("fail", "fail", "regression", False, "regression", False),
        ("insufficient", "fail", None, False, "held_out", False),
        ("not_applicable", "fail", None, False, "held_out", False),
    ],
)
def test_the_gate_truth_table(c1, c2, suites_rule, ship, failed_rule, canary):
    decision = _decide(c1, c2, suites_rule)
    assert decision.ship is ship
    assert decision.failed_rule == failed_rule
    assert decision.canary_decides is canary


def test_canary_decides_only_when_there_are_no_corrections_and_nothing_else_is_wrong():
    assert _decide("not_applicable", "pass").canary_decides is True
    for c1 in ("pass", "fail", "insufficient"):
        assert _decide(c1, "pass").canary_decides is False


# -- apply_loop_gate: ship's own suites rules, with the task win demoted to information ----------


def _suite_verdict(deltas, *, won=True) -> ShipVerdict:
    """What ``decide_ship`` hands the gate: the plain verdict of the same run."""
    return ShipVerdict(
        decision=DECISION_SHIP if won and deltas else DECISION_DONT_SHIP,
        task_win=TaskWin(mode="metric", base=0.2, tuned=0.8 if won else 0.2, won=won),
        benchmark_deltas=tuple(deltas),
        failed_rule=None if won and deltas else "task_win",
        forgetting_threshold=0.05,
        soup_version="0",
    )


def _delta(name: str, base: float, tuned: float, regressed: bool) -> BenchmarkDelta:
    return BenchmarkDelta(name=name, base=base, tuned=tuned, delta=round(tuned - base, 6),
                          regressed=regressed)


PASSING = corrections_check(6, 0, 0, 12, rows=6)
HELD_FINE = held_out_check(0, 0, 10)


def test_no_measured_suite_is_a_missing_baseline_never_a_ship():
    verdict = apply_loop_gate(_suite_verdict([]), PASSING, HELD_FINE)
    assert verdict.decision == DECISION_DONT_SHIP
    assert verdict.failed_rule == "missing_baseline"
    assert verdict.loop_gate.canary_decides is False
    # also with nothing to check: the canary cannot rescue a run whose suites were never measured
    none = corrections_check(0, 0, 0, 0, rows=0)
    verdict = apply_loop_gate(_suite_verdict([]), none, HELD_FINE)
    assert (verdict.failed_rule, verdict.loop_gate.canary_decides) == ("missing_baseline", False)
    assert "No general-suite benchmarks were measured" in verdict_to_dict(verdict)["reason"]


def test_a_regressed_benchmark_is_a_regression_whatever_the_checks_say():
    deltas = [_delta("mini_mmlu", 0.5, 0.2, True), _delta("mini_safety", 0.5, 0.5, False)]
    verdict = apply_loop_gate(_suite_verdict(deltas), PASSING, HELD_FINE)
    assert verdict.decision == DECISION_DONT_SHIP
    assert verdict.failed_rule == "regression"
    assert "mini_mmlu" in verdict_to_dict(verdict)["reason"]
    assert "mini_safety" not in verdict_to_dict(verdict)["reason"]


def test_a_lost_task_win_does_not_stop_a_ship_but_stays_on_the_verdict():
    verdict = apply_loop_gate(_suite_verdict([_delta("mini_mmlu", 0.5, 0.5, False)], won=False),
                              PASSING, HELD_FINE)
    assert verdict.decision == DECISION_SHIP
    assert verdict.failed_rule is None
    assert verdict.task_win.won is False  # information only
    assert verdict.task_win.tuned == 0.2
    assert verdict_to_dict(verdict)["reason"] is None


def test_apply_loop_gate_only_takes_a_ship_verdict():
    with pytest.raises(TypeError):
        apply_loop_gate({"decision": "SHIP"}, PASSING, HELD_FINE)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The two files, refused at the boundary
# ---------------------------------------------------------------------------

GOOD_CORRECTION = {
    "edit_id": "e1", "expected": "Hold reset ten seconds.", "original": "Press it.",
    "paraphrases": ["How long do I hold reset?"],
}
GOOD_HELD_OUT = {
    "edit_id": "h1", "prompt": "How do I reset?", "expected": "Hold reset ten seconds.",
    "original": "Press it.",
}


def jsonl(*rows) -> str:
    return "\n".join(row if isinstance(row, str) else json.dumps(row) for row in rows) + "\n"


def without(row: dict, key: str) -> dict:
    return {name: value for name, value in row.items() if name != key}


def test_good_rows_parse_into_frozen_records():
    edits = parse_corrections(jsonl(GOOD_CORRECTION), "--corrections c.jsonl")
    assert edits == (CorrectionEdit("e1", "Hold reset ten seconds.", "Press it.",
                                    ("How long do I hold reset?",)),)
    rows = parse_held_out(jsonl(GOOD_HELD_OUT), "--held-out h.jsonl")
    assert rows == (HeldOutEdit("h1", "How do I reset?", "Hold reset ten seconds.", "Press it."),)


def test_blank_lines_are_skipped_but_still_counted_for_line_numbers():
    bad = json.dumps({**GOOD_CORRECTION, "edit_id": 7})
    text = "\n" + json.dumps(GOOD_CORRECTION) + "\n\n" + bad + "\n"
    with pytest.raises(GateInputError) as raised:
        parse_corrections(text, "--corrections c.jsonl")
    assert str(raised.value) == '--corrections c.jsonl line 4: "edit_id" must be a non-empty string'


def test_an_empty_file_and_a_file_of_blank_lines_have_no_rows():
    assert parse_corrections("", "--corrections c.jsonl") == ()
    assert parse_corrections("\n\n  \n", "--corrections c.jsonl") == ()
    assert parse_held_out("", "--held-out h.jsonl") == ()


def test_a_byte_order_mark_and_crlf_line_ends_are_accepted():
    text = (
        "\ufeff" + json.dumps(GOOD_CORRECTION) + "\r\n"
        + json.dumps({**GOOD_CORRECTION, "edit_id": "e2"}) + "\r\n"
    )
    assert [edit.edit_id for edit in parse_corrections(text, "x")] == ["e1", "e2"]


def test_a_json_string_may_hold_the_unicode_line_separators():
    row = {**GOOD_CORRECTION, "expected": "one\u2028two\u0085three"}
    (edit,) = parse_corrections(json.dumps(row, ensure_ascii=False) + "\n", "x")
    assert edit.expected == "one\u2028two\u0085three"


CORRECTION_REFUSALS = {
    "not_json": ("this is not json", 'invalid JSON: Expecting value'),
    "not_an_object": ("[1, 2]", "expected a JSON object, got array"),
    "null_row": ("null", "expected a JSON object, got null"),
    "number_row": ("12", "expected a JSON object, got number"),
    "bool_row": ("true", "expected a JSON object, got boolean"),
    "no_edit_id": (without(GOOD_CORRECTION, "edit_id"), 'missing "edit_id"'),
    "edit_id_int": ({**GOOD_CORRECTION, "edit_id": 3}, '"edit_id" must be a non-empty string'),
    "edit_id_blank": ({**GOOD_CORRECTION, "edit_id": "  "}, '"edit_id" must be a non-empty string'),
    "edit_id_long": (
        {**GOOD_CORRECTION, "edit_id": "x" * 257}, '"edit_id" is longer than 256 characters',
    ),
    "no_expected": (without(GOOD_CORRECTION, "expected"), 'missing "expected"'),
    "expected_empty": (
        {**GOOD_CORRECTION, "expected": ""}, '"expected" must be a non-empty string',
    ),
    "expected_list": (
        {**GOOD_CORRECTION, "expected": ["a"]}, '"expected" must be a non-empty string',
    ),
    "expected_long": (
        {**GOOD_CORRECTION, "expected": "x" * 20001}, '"expected" is longer than 20000 characters',
    ),
    "no_original": (without(GOOD_CORRECTION, "original"), 'missing "original"'),
    "original_int": ({**GOOD_CORRECTION, "original": 5}, '"original" must be a string'),
    "original_long": (
        {**GOOD_CORRECTION, "original": "x" * 20001}, '"original" is longer than 20000 characters',
    ),
    "no_paraphrases": (without(GOOD_CORRECTION, "paraphrases"), 'missing "paraphrases"'),
    "paraphrases_empty": (
        {**GOOD_CORRECTION, "paraphrases": []},
        '"paraphrases" must be a list of 1 to 20 non-empty strings',
    ),
    "paraphrases_string": (
        {**GOOD_CORRECTION, "paraphrases": "How?"},
        '"paraphrases" must be a list of 1 to 20 non-empty strings',
    ),
    "paraphrases_too_many": (
        {**GOOD_CORRECTION, "paraphrases": [f"q{i}" for i in range(21)]},
        '"paraphrases" must be a list of 1 to 20 non-empty strings',
    ),
    "paraphrase_int": (
        {**GOOD_CORRECTION, "paraphrases": ["ok", 3]},
        '"paraphrases"[1] must be a non-empty string',
    ),
    "paraphrase_blank": (
        {**GOOD_CORRECTION, "paraphrases": [" "]}, '"paraphrases"[0] must be a non-empty string',
    ),
    "paraphrase_long": (
        {**GOOD_CORRECTION, "paraphrases": ["x" * 20001]},
        '"paraphrases"[0] is longer than 20000 characters',
    ),
}


@pytest.mark.parametrize(
    ("row", "problem"), list(CORRECTION_REFUSALS.values()), ids=list(CORRECTION_REFUSALS),
)
def test_a_bad_corrections_row_names_the_file_the_line_and_the_problem(row, problem):
    text = jsonl(GOOD_CORRECTION, row)  # the bad row is on line 2
    with pytest.raises(GateInputError) as raised:
        parse_corrections(text.replace('"e1"', '"e0"', 1), "--corrections c.jsonl")
    assert str(raised.value) == f"--corrections c.jsonl line 2: {problem}"


HELD_OUT_REFUSALS = {
    "not_json": ("{oops", "invalid JSON: Expecting property name enclosed in double quotes"),
    "not_an_object": ('"text"', "expected a JSON object, got string"),
    "no_edit_id": (without(GOOD_HELD_OUT, "edit_id"), 'missing "edit_id"'),
    "edit_id_blank": ({**GOOD_HELD_OUT, "edit_id": ""}, '"edit_id" must be a non-empty string'),
    "no_prompt": (without(GOOD_HELD_OUT, "prompt"), 'missing "prompt"'),
    "prompt_int": ({**GOOD_HELD_OUT, "prompt": 1}, '"prompt" must be a non-empty string'),
    "prompt_long": (
        {**GOOD_HELD_OUT, "prompt": "x" * 20001}, '"prompt" is longer than 20000 characters',
    ),
    "no_expected": (without(GOOD_HELD_OUT, "expected"), 'missing "expected"'),
    "expected_empty": ({**GOOD_HELD_OUT, "expected": ""}, '"expected" must be a non-empty string'),
    "no_original": (without(GOOD_HELD_OUT, "original"), 'missing "original"'),
    "original_none": ({**GOOD_HELD_OUT, "original": None}, '"original" must be a string'),
}


@pytest.mark.parametrize(
    ("row", "problem"), list(HELD_OUT_REFUSALS.values()), ids=list(HELD_OUT_REFUSALS),
)
def test_a_bad_held_out_row_names_the_file_the_line_and_the_problem(row, problem):
    text = jsonl(GOOD_HELD_OUT, row)
    with pytest.raises(GateInputError) as raised:
        parse_held_out(text.replace('"h1"', '"h0"', 1), "--held-out h.jsonl")
    assert str(raised.value) == f"--held-out h.jsonl line 2: {problem}"


def test_a_duplicate_edit_id_names_both_lines():
    text = jsonl(GOOD_CORRECTION, {**GOOD_CORRECTION, "expected": "Something else entirely."})
    with pytest.raises(GateInputError) as raised:
        parse_corrections(text, "--corrections c.jsonl")
    assert str(raised.value) == (
        '--corrections c.jsonl line 2: duplicate edit_id "e1" (first on line 1)'
    )
    with pytest.raises(GateInputError) as raised:
        parse_held_out(jsonl(GOOD_HELD_OUT, GOOD_HELD_OUT), "--held-out h.jsonl")
    assert str(raised.value) == (
        '--held-out h.jsonl line 2: duplicate edit_id "h1" (first on line 1)'
    )


def test_a_text_field_exactly_at_its_limit_is_accepted():
    at_limit = "x" * gate.MAX_TEXT_CHARS
    row = {"edit_id": "i" * gate.MAX_ID_CHARS, "expected": at_limit, "original": at_limit,
           "paraphrases": [at_limit] * gate.MAX_PARAPHRASES}
    (edit,) = parse_corrections(jsonl(row), "--corrections c.jsonl")
    assert (len(edit.edit_id), len(edit.expected), len(edit.original)) == (256, 20000, 20000)
    assert (len(edit.paraphrases), len(edit.paraphrases[0])) == (20, 20000)
    held_row = {"edit_id": "i" * gate.MAX_ID_CHARS, "prompt": at_limit, "expected": at_limit,
                "original": at_limit}
    (held_edit,) = parse_held_out(jsonl(held_row), "--held-out h.jsonl")
    assert (len(held_edit.edit_id), len(held_edit.prompt)) == (256, 20000)


def test_too_many_rows_is_refused_at_the_row_over_the_limit(monkeypatch):
    monkeypatch.setattr(gate, "MAX_ROWS", 3)
    rows = [{**GOOD_CORRECTION, "edit_id": f"e{i}"} for i in range(4)]
    assert len(parse_corrections(jsonl(*rows[:3]), "x")) == 3  # exactly the limit is fine
    with pytest.raises(GateInputError) as raised:
        parse_corrections(jsonl(*rows), "--corrections c.jsonl")
    assert str(raised.value) == "--corrections c.jsonl line 4: more than 3 rows"
    held_rows = [{**GOOD_HELD_OUT, "edit_id": f"h{i}"} for i in range(4)]
    with pytest.raises(GateInputError) as raised:
        parse_held_out(jsonl(*held_rows), "--held-out h.jsonl")
    assert str(raised.value) == "--held-out h.jsonl line 4: more than 3 rows"


def test_the_bounds_are_the_documented_ones():
    assert gate.MAX_ROWS == 10_000
    assert gate.MAX_TEXT_CHARS == 20_000
    assert gate.MAX_PARAPHRASES == 20
    assert gate.MAX_ID_CHARS == 256


def test_a_very_deeply_nested_row_is_a_bad_row_not_a_crash():
    deep = "[" * 100000 + "]" * 100000
    with pytest.raises(GateInputError) as raised:
        parse_corrections(deep + "\n", "--corrections c.jsonl")
    assert str(raised.value).startswith("--corrections c.jsonl line 1: ")


# ---------------------------------------------------------------------------
# The verdict JSON validates against the loop formats' iteration schema
# ---------------------------------------------------------------------------

def _shipped_iteration_schema() -> dict:
    """The iteration schema the package ships (``soup_cli/schemas/loop``): the authority, read
    from the installed package, never a copy."""
    from importlib import resources

    text = (
        resources.files("soup_cli.schemas.loop")
        .joinpath("iteration.schema.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(text)




def _validator(name: str):
    jsonschema = pytest.importorskip("jsonschema")
    shipped = _shipped_iteration_schema()
    verdict = shipped["properties"]["verdict"]["oneOf"][0]["properties"]
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": shipped["$defs"],
        **verdict[name],
    }
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def test_the_shipped_schema_rejects_what_it_should():
    """The shipped schema itself must be strict, or the tests below prove nothing."""
    corrections = _validator("corrections")
    good = corrections_check(5, 0, 0, 10, rows=5).to_dict()
    assert list(corrections.iter_errors(good)) == []
    assert list(corrections.iter_errors({**good, "outcome": "weak"}))  # an enum of its own
    assert list(corrections.iter_errors({**good, "extra": 1}))  # additionalProperties: false
    assert list(corrections.iter_errors({k: v for k, v in good.items() if k != "p_value"}))
    held_out = _validator("held_out")
    wrong = {**held_out_check(0, 0, 0).to_dict(), "outcome": "insufficient"}
    assert list(held_out.iter_errors(wrong))


@pytest.mark.parametrize(
    "counts", [(5, 0, 0), (4, 0, 3), (0, 0, 0), (3, 4, 5), (0, 5, 0), (12, 1, 0)],
)
def test_every_corrections_object_validates(counts):
    fixed, broken, ties = counts
    rows = fixed + broken + ties
    result = corrections_check(fixed, broken, ties, 2 * rows, rows=rows)
    assert list(_validator("corrections").iter_errors(result.to_dict())) == []


@pytest.mark.parametrize(
    "counts", [(0, 0, 0), (1, 0, 4), (0, 5, 0), (3, 2, 0), (2, 7, 1), (10, 0, 0), (0, 0, 20)],
)
def test_every_held_out_object_validates_with_and_without_a_style_note(counts):
    better, worse, ties = counts
    validator = _validator("held_out")
    assert list(validator.iter_errors(held_out_check(better, worse, ties).to_dict())) == []
    styled = held_out_check(better, worse, ties, style=style_from_win_rate(0.6, 7))
    assert "style" in styled.to_dict()
    assert list(validator.iter_errors(styled.to_dict())) == []


# ---------------------------------------------------------------------------
# The CLI, with the model loading replaced
# ---------------------------------------------------------------------------

Q_A = "base_model.model.layers.0.self_attn.q_proj.lora_A.weight"
Q_B = "base_model.model.layers.0.self_attn.q_proj.lora_B.weight"
TASK_PROMPT = "task"
HIT = "alpha bravo"  # holds two of the three added words of ``expected`` below
MISS = "I do not know."
VERBOSE = "Well, to answer that properly I would first have to explain how the whole thing works,"


def _plain(text: str) -> str:
    plain = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", strip_ansi(text))
    return " ".join(re.sub(r"[^\x20-\x7e]+", " ", plain).split())


def _write_adapter(path: Path, b_value: float) -> None:
    pytest.importorskip("safetensors")
    from safetensors.numpy import save_file

    path.mkdir()
    save_file(
        {
            Q_A: np.ones((2, 4), dtype=np.float32),
            Q_B: np.full((4, 2), b_value, dtype=np.float32),
        },
        str(path / "adapter_model.safetensors"),
    )
    (path / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 2, "lora_alpha": 4}), encoding="utf-8"
    )


def _stub_pairwise_leg(monkeypatch, win_rate: float = 0.6) -> None:
    """Leg 1 under ``--task-mode pairwise`` without a judge: a fixed win rate."""

    def build(task_eval, judge_model):
        return lambda old_gen, new_gen: win_rate

    monkeypatch.setattr(ship_cmd, "_build_pairwise_scorer", build)


class Lab:
    """Fake models plus the two input files, built edit by edit.

    A generator is labelled by the adapter it was built with (``bare`` for none). What a label
    answers to a corrections or held-out prompt comes from ``table``; every other prompt (the task
    eval and the bundled suites) gets ``B``, except from a label in ``regressed``, which answers
    nothing and so drops on the suites. ``cand`` solves the task prompt, everything else does not.

    A generator built with a ``max_new_tokens`` under the gate's answer budget cuts a corrections or
    held-out answer short (``VERBOSE``), as a model that talks for 300 tokens before the fact does.
    A generator built on an already loaded model (``loaded=``) is recorded in ``rebuilt``, not in
    ``calls``: ``calls`` is the model loads.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.table: Dict[str, Dict[str, str]] = {"bare": {}, "prod": {}, "cand": {}}
        self.regressed: set = set()
        self.corrections: List[dict] = []
        self.held_out: List[dict] = []
        self.calls: List[tuple] = []
        self.rebuilt: List[tuple] = []
        self.asked: Dict[str, List[str]] = {"bare": [], "prod": [], "cand": []}
        self.budgets: Dict[str, set] = {}  # prompt -> the token budgets that answered it
        self.task_wins = True

    # -- the input files ------------------------------------------------------------------
    def add_correction(
        self, number: int, old_hits: int, new_hits: int, old="bare", new="cand",
    ) -> None:
        expected = f"alpha{number} bravo{number} charlie{number}"
        hit = f"alpha{number} bravo{number}"
        paraphrases = [f"c{number} first", f"c{number} second"]
        for index, paraphrase in enumerate(paraphrases):
            self.table[old][paraphrase] = hit if index < old_hits else MISS
            self.table[new][paraphrase] = hit if index < new_hits else MISS
        self.corrections.append({
            "edit_id": f"e{number}", "expected": expected, "original": MISS,
            "paraphrases": paraphrases,
        })

    def add_held_out(
        self, number: int, old_words: int, new_words: int, old="bare", new="cand",
    ) -> None:
        words = [f"delta{number}", f"echo{number}", f"foxtrot{number}", f"golf{number}"]
        prompt = f"h{number} question"
        self.table[old][prompt] = " ".join(words[:old_words]) or MISS
        self.table[new][prompt] = " ".join(words[:new_words]) or MISS
        self.held_out.append({
            "edit_id": f"h{number}", "prompt": prompt, "expected": " ".join(words),
            "original": MISS,
        })

    def corrections_pattern(self, fixed: int = 0, broken: int = 0, ties: int = 0, **sides) -> None:
        number = len(self.corrections)
        for _ in range(fixed):
            number += 1
            self.add_correction(number, 0, 2, **sides)
        for _ in range(broken):
            number += 1
            self.add_correction(number, 2, 0, **sides)
        for _ in range(ties):
            number += 1
            self.add_correction(number, 1, 1, **sides)

    def held_out_pattern(self, better: int = 0, worse: int = 0, ties: int = 0, **sides) -> None:
        number = len(self.held_out)
        for _ in range(better):
            number += 1
            self.add_held_out(number, 1, 3, **sides)
        for _ in range(worse):
            number += 1
            self.add_held_out(number, 3, 1, **sides)
        for _ in range(ties):
            number += 1
            self.add_held_out(number, 2, 2, **sides)

    def write(self) -> None:
        (self.root / "corrections.jsonl").write_text(jsonl(*self.corrections), encoding="utf-8")
        (self.root / "held.jsonl").write_text(jsonl(*self.held_out), encoding="utf-8")

    # -- the fake model loader ------------------------------------------------------------
    def make_generator(self, model_id, **kwargs):
        loaded = kwargs.get("loaded")
        if loaded is not None:
            label = loaded[0]
            self.rebuilt.append((model_id, kwargs))
        else:
            label = kwargs.get("adapter") or "bare"
            self.calls.append((model_id, kwargs))
        budget = kwargs.get("max_new_tokens")

        def generate(prompt: str) -> str:
            self.asked.setdefault(label, []).append(prompt)
            self.budgets.setdefault(prompt, set()).add(budget)
            if prompt in self.table.get(label, {}):
                if budget is not None and budget < gate.ANSWER_MAX_NEW_TOKENS:
                    return VERBOSE
                return self.table[label][prompt]
            if prompt == TASK_PROMPT:
                return "answer" if (label == "cand" and self.task_wins) else "wrong"
            return "" if label in self.regressed else "B"

        # what the real ``live_eval.make_generator`` exposes: the loaded model, for a rebuild
        generate.loaded = (label, f"tokenizer of {label}", "cpu")
        return generate

    def adapters(self) -> list:
        return [kwargs.get("adapter") for _model, kwargs in self.calls]

    def suite_calls(self, label: str) -> int:
        known = set(self.table[label]) | {TASK_PROMPT}
        return sum(1 for prompt in self.asked[label] if prompt not in known)


@pytest.fixture
def lab(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_adapter(tmp_path / "prod", 0.5)
    _write_adapter(tmp_path / "cand", 0.7)
    _write_adapter(tmp_path / "zero", 0.0)
    (tmp_path / "task.jsonl").write_text(
        json.dumps({"prompt": TASK_PROMPT, "expected": "answer"}) + "\n", encoding="utf-8"
    )
    fake = Lab(tmp_path)
    monkeypatch.setattr(live_eval, "make_generator", fake.make_generator)
    return fake


def _ship(*args: str):
    return runner.invoke(app, ["ship", *args])


LIVE = ["--base", "m", "--adapter", "cand", "--task-eval", "task.jsonl", "--device", "cpu"]


def _gated(lab: Lab, *extra: str, corrections=True, held_out=True):
    """Write the files and run the gate: bare vs cand unless --base-adapter is passed."""
    lab.write()
    args = [*LIVE, "--gate", "loop-v2", "--output", "out.json"]
    if corrections:
        args += ["--corrections", "corrections.jsonl"]
    if held_out:
        args += ["--held-out", "held.jsonl"]
    return _ship(*args, *extra)


def _out(lab: Lab) -> dict:
    return json.loads((lab.root / "out.json").read_text(encoding="utf-8"))


def _ok(result, code: int):
    assert result.exit_code == code, (result.output, repr(result.exception))


def _healthy(lab: Lab) -> None:
    """Six fixed corrections and ten tied held-out edits: every check passes."""
    lab.corrections_pattern(fixed=6)
    lab.held_out_pattern(ties=10)


# -- SHIP ---------------------------------------------------------------------------------


def test_everything_holds_so_it_ships_and_the_output_carries_the_gate_objects(lab):
    _healthy(lab)
    result = _gated(lab)
    _ok(result, 0)
    out = _out(lab)
    assert out["decision"] == DECISION_SHIP
    assert out["gate"] == "loop-v2"
    assert out["failed_rule"] is None
    assert out["reason"] is None
    assert out["canary_decides"] is False
    assert out["corrections"] == {
        "fixed": 6, "broken": 0, "ties": 0, "n_edits": 6, "n_paraphrases": 12, "p_value": 1 / 64,
        "scoring": SCORING, "outcome": "pass",
    }
    assert out["held_out"] == {
        "better": 0, "worse": 0, "ties": 10, "n": 10, "p_value": None, "scoring": SCORING,
        "outcome": "pass",
    }
    assert out["benchmark_deltas"]  # the suites ran


def test_the_output_still_has_every_key_it_had(lab):
    _healthy(lab)
    _ok(_gated(lab), 0)
    existing = {
        "decision", "task_win", "benchmark_deltas", "failed_rule", "forgetting_threshold",
        "soup_version", "noise_floor", "numerics", "base_adapter",
    }
    assert existing <= set(_out(lab))


def test_the_output_validates_against_the_iteration_schema(lab):
    _healthy(lab)
    _ok(_gated(lab), 0)
    out = _out(lab)
    assert list(_validator("corrections").iter_errors(out["corrections"])) == []
    assert list(_validator("held_out").iter_errors(out["held_out"])) == []


def test_every_outcome_the_cli_can_write_validates(lab, monkeypatch):
    errors = []
    outcomes = set()
    for fixed, broken, better, worse in [(3, 0, 0, 0), (0, 0, 0, 0), (4, 1, 0, 6), (5, 0, 2, 3)]:
        fresh = Lab(lab.root)
        monkeypatch.setattr(live_eval, "make_generator", fresh.make_generator)
        fresh.corrections_pattern(fixed=fixed, broken=broken)
        fresh.held_out_pattern(better=better, worse=worse)
        result = _gated(fresh, corrections=bool(fixed + broken), held_out=bool(better + worse))
        assert result.exit_code in (0, 2), (result.output, repr(result.exception))
        out = _out(fresh)
        outcomes.add((out["corrections"]["outcome"], out["held_out"]["outcome"]))
        errors += list(_validator("corrections").iter_errors(out["corrections"]))
        errors += list(_validator("held_out").iter_errors(out["held_out"]))
    assert errors == []
    assert len(outcomes) == 4  # the four runs really covered four different pairs


def test_the_old_version_is_the_bare_base_and_the_new_one_the_candidate(lab):
    _healthy(lab)
    _ok(_gated(lab), 0)
    assert lab.adapters() == [None, "cand"]
    assert lab.asked["bare"].count("c1 first") == 1
    assert lab.asked["cand"].count("c1 first") == 1


def test_with_a_base_adapter_the_old_version_is_the_production_adapter(lab):
    # prod already knows two edits and cand knows all six: against prod, only four are fixed.
    lab.corrections_pattern(fixed=4, old="prod")
    lab.add_correction(5, 2, 2, old="prod")
    lab.add_correction(6, 2, 2, old="prod")
    lab.held_out_pattern(ties=10, old="prod")
    result = _gated(lab, "--base-adapter", "prod")
    _ok(result, 2)
    out = _out(lab)
    assert lab.adapters() == ["prod", "cand"]
    assert (out["corrections"]["fixed"], out["corrections"]["ties"]) == (4, 2)
    assert out["corrections"]["outcome"] == "insufficient"
    assert out["base_adapter"] == "prod"
    assert lab.asked["bare"] == []  # nothing was asked of the bare base


def test_a_separate_tuned_model_is_the_new_version(lab):
    lab.corrections_pattern(fixed=5)
    lab.held_out_pattern(ties=5)
    lab.write()
    result = _ship("--base", "m", "--tuned", "cand", "--task-eval", "task.jsonl", "--device", "cpu",
                   "--gate", "loop-v2", "--corrections", "corrections.jsonl",
                   "--held-out", "held.jsonl", "--output", "out.json")
    # a tuned model id is not an adapter, so this fake labels both sides "bare" and cannot tell
    # them apart (every correction ties); what matters here is which model each side loaded.
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert [model for model, _kwargs in lab.calls] == ["m", "cand"]
    assert _out(lab)["gate"] == "loop-v2"


# -- DON'T SHIP: check 1 -------------------------------------------------------------------


def test_corrections_that_were_not_learned_fail(lab):
    lab.corrections_pattern(fixed=4, broken=1)
    lab.held_out_pattern(ties=10)
    _ok(_gated(lab), 2)
    out = _out(lab)
    assert out["decision"] == DECISION_DONT_SHIP
    assert out["failed_rule"] == "corrections"
    assert out["corrections"]["outcome"] == "fail"
    assert out["corrections"]["p_value"] == 6 / 32
    assert out["canary_decides"] is False
    assert "did not learn" in out["reason"]


def test_not_enough_corrections_names_how_many_there_are(lab):
    lab.corrections_pattern(fixed=3, broken=1, ties=2)
    lab.held_out_pattern(ties=10)
    result = _gated(lab)
    _ok(result, 2)
    out = _out(lab)
    assert out["failed_rule"] == "corrections_insufficient"
    assert out["corrections"]["outcome"] == "insufficient"
    assert out["reason"] == "not enough corrections yet (4/5)"
    assert out["canary_decides"] is False
    assert "not enough corrections yet (4/5)" in _plain(result.output)


def test_exactly_five_discordant_corrections_are_enough(lab):
    lab.corrections_pattern(fixed=5, ties=3)
    lab.held_out_pattern(ties=10)
    _ok(_gated(lab), 0)
    assert _out(lab)["corrections"]["outcome"] == "pass"


def test_four_fixed_corrections_are_not_enough_however_clean(lab):
    lab.corrections_pattern(fixed=4)
    lab.held_out_pattern(ties=10)
    _ok(_gated(lab), 2)
    out = _out(lab)
    assert out["corrections"]["p_value"] == 1 / 16
    assert out["reason"] == "not enough corrections yet (4/5)"


def test_without_a_corrections_file_the_canary_decides(lab):
    lab.held_out_pattern(ties=10)
    result = _gated(lab, corrections=False)
    _ok(result, 2)
    out = _out(lab)
    assert out["decision"] == DECISION_DONT_SHIP
    assert out["failed_rule"] == "no_corrections"
    assert out["reason"] == "no corrections to check - the canary's ratings decide"
    assert out["canary_decides"] is True
    assert out["corrections"] == {
        "fixed": 0, "broken": 0, "ties": 0, "n_edits": 0, "n_paraphrases": 0, "p_value": None,
        "scoring": SCORING, "outcome": "not_applicable",
    }
    assert "no corrections to check - the canary's ratings decide" in _plain(result.output)


def test_an_empty_corrections_file_is_the_same_as_none(lab):
    lab.held_out_pattern(ties=10)
    _ok(_gated(lab), 2)  # corrections.jsonl is written empty
    out = _out(lab)
    assert out["corrections"]["outcome"] == "not_applicable"
    assert out["canary_decides"] is True
    assert out["reason"] == "no corrections to check - the canary's ratings decide"


def test_no_corrections_and_regressed_suites_do_not_leave_it_to_the_canary(lab):
    lab.held_out_pattern(ties=10)
    lab.regressed.add("cand")
    _ok(_gated(lab, corrections=False), 2)
    out = _out(lab)
    assert out["corrections"]["outcome"] == "not_applicable"
    assert out["canary_decides"] is False
    assert out["failed_rule"] == "regression"
    assert "no corrections to check" not in out["reason"]


# -- DON'T SHIP: check 2 and the suites ------------------------------------------------------


def test_significantly_worse_held_out_edits_fail_even_when_the_corrections_pass(lab):
    lab.corrections_pattern(fixed=6)
    lab.held_out_pattern(worse=5, ties=2)
    _ok(_gated(lab), 2)
    out = _out(lab)
    assert out["failed_rule"] == "held_out"
    assert out["held_out"]["outcome"] == "fail"
    held_out = out["held_out"]
    assert (held_out["worse"], held_out["better"], held_out["ties"]) == (5, 0, 2)
    assert out["held_out"]["p_value"] == 1 / 32
    assert out["corrections"]["outcome"] == "pass"
    assert "worse" in out["reason"]


def test_weak_and_unmeasured_held_out_edits_do_not_block_a_ship(lab):
    lab.corrections_pattern(fixed=6)
    lab.held_out_pattern(worse=4)  # four edits: not measured, although every one is worse
    _ok(_gated(lab), 0)
    assert _out(lab)["held_out"]["outcome"] == "not_measured"


def test_weak_held_out_edits_ship(lab):
    lab.corrections_pattern(fixed=6)
    lab.held_out_pattern(better=2, worse=1, ties=2)  # n = 5
    _ok(_gated(lab), 0)
    assert _out(lab)["held_out"]["outcome"] == "weak"


def test_held_out_boundaries_at_the_cli(lab):
    lab.corrections_pattern(fixed=6)
    lab.held_out_pattern(ties=9)
    _ok(_gated(lab), 0)
    assert (_out(lab)["held_out"]["n"], _out(lab)["held_out"]["outcome"]) == (9, "weak")
    lab.held_out_pattern(ties=1)
    _ok(_gated(lab), 0)
    assert (_out(lab)["held_out"]["n"], _out(lab)["held_out"]["outcome"]) == (10, "pass")


def test_a_regressed_suite_blocks_a_ship_that_everything_else_passes(lab):
    _healthy(lab)
    lab.regressed.add("cand")
    result = _gated(lab)
    _ok(result, 2)
    out = _out(lab)
    assert out["failed_rule"] == "regression"
    assert out["corrections"]["outcome"] == "pass"
    assert out["held_out"]["outcome"] == "pass"
    assert any(item["regressed"] for item in out["benchmark_deltas"])
    assert "regressed" in out["reason"]
    assert "even though the task improved" not in out["reason"]


def test_the_task_eval_win_is_information_only(lab):
    _healthy(lab)
    lab.task_wins = False
    _ok(_gated(lab), 0)
    out = _out(lab)
    assert out["decision"] == DECISION_SHIP
    assert out["task_win"]["won"] is False
    assert out["failed_rule"] is None


# -- the suites always run ------------------------------------------------------------------


@pytest.mark.parametrize("held_file", ["empty", "absent"])
def test_the_suites_and_the_task_eval_run_without_any_held_out_edit(lab, held_file):
    lab.corrections_pattern(fixed=6)
    lab.write()  # held.jsonl is written empty
    args = [
        *LIVE, "--gate", "loop-v2", "--corrections", "corrections.jsonl", "--output", "out.json",
    ]
    if held_file == "empty":
        args += ["--held-out", "held.jsonl"]
    result = _ship(*args)
    _ok(result, 0)
    out = _out(lab)
    assert out["decision"] == DECISION_SHIP
    assert out["held_out"] == {
        "better": 0, "worse": 0, "ties": 0, "n": 0, "p_value": None, "scoring": SCORING,
        "outcome": "not_measured",
    }
    assert out["benchmark_deltas"], "the suites must be measured even with no held-out edit"
    assert lab.suite_calls("cand") > 0 and lab.suite_calls("bare") > 0
    assert TASK_PROMPT in lab.asked["cand"] and TASK_PROMPT in lab.asked["bare"]
    assert out["task_win"]["tuned"] == 1.0


def test_the_suites_run_when_neither_file_is_given(lab):
    lab.write()
    result = _ship(*LIVE, "--gate", "loop-v2", "--output", "out.json")
    _ok(result, 2)
    out = _out(lab)
    assert out["benchmark_deltas"]
    assert out["corrections"]["outcome"] == "not_applicable"
    assert out["held_out"]["outcome"] == "not_measured"
    assert out["canary_decides"] is True


# -- how many tokens the gate's questions get ------------------------------------------------


def test_the_answer_budget_is_zeros_512_and_the_live_eval_budget_is_still_256():
    from soup_cli.eval.gate_suites import BEHAVIOURAL_MAX_NEW_TOKENS

    assert gate.ANSWER_MAX_NEW_TOKENS == 512
    assert BEHAVIOURAL_MAX_NEW_TOKENS == 256  # the task eval and the suites are not touched


def test_the_gate_asks_its_questions_with_512_new_tokens_and_the_rest_keeps_256(lab):
    _healthy(lab)
    _ok(_gated(lab), 0)
    # the two model loads are the same two as without the gate, at the live eval's budget
    assert [kwargs["max_new_tokens"] for _model, kwargs in lab.calls] == [256, 256]
    # and the gate's generators are rebuilt on them at the answer budget
    assert [kwargs["max_new_tokens"] for _model, kwargs in lab.rebuilt] == [512, 512]
    gate_prompts = {row["prompt"] for row in lab.held_out}
    gate_prompts |= {text for row in lab.corrections for text in row["paraphrases"]}
    assert gate_prompts and all(lab.budgets[prompt] == {512} for prompt in gate_prompts)
    assert lab.budgets[TASK_PROMPT] == {256}
    suite_prompts = set(lab.budgets) - gate_prompts - {TASK_PROMPT}
    assert suite_prompts and all(lab.budgets[prompt] == {256} for prompt in suite_prompts)


def test_a_verbose_answer_is_not_a_false_miss(lab):
    """The new version states the corrected fact only after a long preamble: at 256 tokens that
    answer is cut and scored as a miss. With the answer budget the gate sees the fact."""
    lab.corrections_pattern(fixed=5)
    lab.held_out_pattern(better=5)
    _ok(_gated(lab), 0)
    out = _out(lab)
    assert (out["corrections"]["fixed"], out["corrections"]["outcome"]) == (5, "pass")
    assert out["held_out"]["better"] == 5


def test_the_gate_reuses_the_models_that_are_already_loaded(lab):
    _healthy(lab)
    _ok(_gated(lab, "--base-adapter", "prod"), 0)
    assert lab.adapters() == ["prod", "cand"]  # two loads, no third and no fourth
    assert [kwargs["loaded"][0] for _model, kwargs in lab.rebuilt] == ["prod", "cand"]
    assert [model for model, _kwargs in lab.rebuilt] == ["m", "m"]


def test_a_separate_tuned_model_is_rebuilt_under_its_own_id(lab):
    lab.corrections_pattern(fixed=5)
    lab.write()
    _ship("--base", "m", "--tuned", "t", "--task-eval", "task.jsonl", "--device", "cpu",
          "--gate", "loop-v2", "--corrections", "corrections.jsonl", "--output", "out.json")
    assert [model for model, _kwargs in lab.calls] == ["m", "t"]
    assert [model for model, _kwargs in lab.rebuilt] == ["m", "t"]


def test_the_chat_template_options_reach_the_gate_generators_too(lab):
    _healthy(lab)
    _ok(_gated(lab, "--chat-template-kwargs", '{"enable_thinking": false}'), 0)
    options = {"enable_thinking": False}
    assert [kwargs["chat_template_kwargs"] for _model, kwargs in lab.calls] == [options, options]
    assert [kwargs["chat_template_kwargs"] for _model, kwargs in lab.rebuilt] == [options, options]


def test_without_chat_template_options_the_rebuild_passes_none(lab):
    _healthy(lab)
    _ok(_gated(lab), 0)
    assert all("chat_template_kwargs" not in kwargs for _model, kwargs in lab.rebuilt)


def test_a_generator_that_hides_its_loaded_model_is_a_clear_error():
    with pytest.raises(ValueError, match="does not expose its loaded model"):
        ship_cmd._answer_generators("m", "m", lambda prompt: "", lambda prompt: "")


def test_the_gate_runs_no_wider_generation_without_the_flag(lab):
    result = _ship(*LIVE, "--output", "out.json")
    _ok(result, 0)
    assert lab.rebuilt == []
    assert [kwargs["max_new_tokens"] for _model, kwargs in lab.calls] == [256, 256]


def _tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocab = {word: index for index, word in enumerate(
        ["<unk>", "<bos>", "<eos>", "hello", "world"]
    )}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", 1)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<bos>", eos_token="<eos>", pad_token="<eos>",
        unk_token="<unk>",
    )


def test_a_live_generator_exposes_its_loaded_model_and_a_rebuild_has_its_own_budget(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    class Recorder:
        budgets: list = []

        def generate(self, input_ids, **kwargs):
            self.budgets.append(kwargs["max_new_tokens"])
            return torch.cat([input_ids, input_ids[:, -1:]], dim=1)

    model, tokenizer = Recorder(), _tiny_tokenizer()

    def never(*args, **kwargs):
        pytest.fail("the weights were loaded a second time")

    monkeypatch.setattr(live_eval, "load_model_and_tokenizer", never)
    loaded = (model, tokenizer, "cpu")
    narrow = live_eval.make_generator("unused", loaded=loaded, max_new_tokens=256)
    assert narrow.loaded == loaded
    wide = live_eval.make_generator("unused", loaded=narrow.loaded, max_new_tokens=512)
    narrow("hello")
    wide("hello")
    assert model.budgets == [256, 512]
    assert wide.loaded == narrow.loaded


# -- an all-zero adapter ---------------------------------------------------------------------


def test_an_all_zero_adapter_is_still_the_base_and_loads_no_model(lab):
    lab.corrections_pattern(fixed=6)
    lab.held_out_pattern(ties=10)
    lab.write()
    result = _ship(
        "--base", "m", "--adapter", "zero", "--task-eval", "task.jsonl", "--device", "cpu",
        "--gate", "loop-v2", "--corrections", "corrections.jsonl",
        "--held-out", "held.jsonl", "--output", "out.json",
    )
    _ok(result, 2)
    out = _out(lab)
    assert lab.calls == []
    assert out["failed_rule"] == "adapter_noop"
    assert out["gate"] == "loop-v2"
    assert out["canary_decides"] is False
    # rows exist but nothing could be asked: insufficient, not "no corrections"
    assert out["corrections"]["outcome"] == "insufficient"
    assert out["corrections"]["n_edits"] == 0
    assert out["held_out"]["outcome"] == "not_measured"


# -- the style note ---------------------------------------------------------------------------


def test_a_pairwise_judge_adds_a_style_note_that_does_not_decide(lab, monkeypatch):
    _healthy(lab)
    seen = {}

    def fake_rate(judge_model, pairs):
        seen["judge"] = judge_model
        seen["pairs"] = list(pairs)
        return 0.7

    monkeypatch.setattr(ship_cmd, "_held_out_win_rate", fake_rate)
    _stub_pairwise_leg(monkeypatch)
    result = _gated(lab, "--task-mode", "pairwise", "--judge-model", "ollama://judge")
    _ok(result, 0)
    out = _out(lab)
    assert seen["judge"] == "ollama://judge"
    # (prompt, old answer, new answer): what the judge compares, in file order
    assert [pair[0] for pair in seen["pairs"]] == [row["prompt"] for row in lab.held_out]
    assert seen["pairs"][0] == ("h1 question", "delta1 echo1", "delta1 echo1")
    assert out["held_out"]["style"] == {
        "preferred_new": 7, "preferred_old": 3, "n": 10, "p_value": 0.34375,
    }
    assert out["held_out"]["outcome"] == "pass"
    assert list(_validator("held_out").iter_errors(out["held_out"])) == []


def test_a_style_note_that_prefers_the_old_version_does_not_block(lab, monkeypatch):
    _healthy(lab)
    monkeypatch.setattr(ship_cmd, "_held_out_win_rate", lambda judge, pairs: 0.0)
    _stub_pairwise_leg(monkeypatch)
    _ok(_gated(lab, "--task-mode", "pairwise", "--judge-model", "ollama://judge"), 0)
    assert _out(lab)["held_out"]["style"]["preferred_new"] == 0


def test_there_is_no_style_note_without_a_pairwise_judge_or_held_out_edits(lab, monkeypatch):
    _healthy(lab)
    called = []
    monkeypatch.setattr(ship_cmd, "_held_out_win_rate", lambda *a: called.append(a) or 0.5)
    _ok(_gated(lab), 0)
    assert "style" not in _out(lab)["held_out"]
    assert called == []

    empty = Lab(lab.root)
    empty.corrections_pattern(fixed=6)
    monkeypatch.setattr(live_eval, "make_generator", empty.make_generator)
    _stub_pairwise_leg(monkeypatch)
    _ok(_gated(empty, "--task-mode", "pairwise", "--judge-model", "ollama://judge"), 0)
    assert "style" not in _out(empty)["held_out"]
    assert called == []


# -- the panel and the plain-text rubric --------------------------------------------------------


def test_the_panel_states_the_gate_and_both_checks(lab):
    _healthy(lab)
    result = _gated(lab)
    _ok(result, 0)
    text = _plain(result.output)
    assert "Gate loop-v2" in text
    assert "Corrections learned: fixed 6, broken 0, ties 0 (p=0.0156) - pass" in text
    assert "Held-out edits: better 0, worse 0, ties 10 - pass" in text


def test_the_rubric_and_the_pr_comment_state_the_gate_too(lab):
    from soup_cli.utils.ship_verdict import format_ship_rubric, render_ship_pr_markdown

    verdict = ShipVerdict(
        decision=DECISION_DONT_SHIP,
        task_win=TaskWin(mode="metric", base=0.1, tuned=0.9, won=True),
        benchmark_deltas=(),
        failed_rule="corrections_insufficient",
        forgetting_threshold=0.05,
        soup_version="0",
        loop_gate=gate.LoopGate(
            corrections=corrections_check(3, 1, 0, 8, rows=4),
            held_out=held_out_check(0, 0, 0),
            canary_decides=False,
        ),
    )
    rubric = format_ship_rubric(verdict)
    assert "Gate:        loop-v2" in rubric
    assert "Reason:      not enough corrections yet (4/5)" in rubric
    assert "not enough corrections yet (4/5)" in render_ship_pr_markdown(verdict)


def test_the_panel_escapes_a_hostile_reason():
    verdict = ShipVerdict(
        decision=DECISION_DONT_SHIP,
        task_win=TaskWin(mode="metric", base=0.1, tuned=0.9, won=True),
        benchmark_deltas=(),
        failed_rule="corrections_insufficient",
        forgetting_threshold=0.05,
        soup_version="0",
        loop_gate=gate.LoopGate(
            corrections=corrections_check(3, 1, 0, 8, rows=4),
            held_out=held_out_check(0, 0, 0),
            canary_decides=False,
        ),
    )
    buffer = StringIO()
    Console(file=buffer, width=120, force_terminal=False).print(render_ship_panel(verdict))
    assert "\x1b" not in buffer.getvalue()
    assert "not enough corrections yet (4/5)" in buffer.getvalue()


def test_a_verdict_defaults_to_no_gate():
    verdict = ShipVerdict(
        decision=DECISION_SHIP,
        task_win=TaskWin(mode="metric", base=0.1, tuned=0.9, won=True),
        benchmark_deltas=(),
        failed_rule=None,
        forgetting_threshold=0.05,
        soup_version="0",
    )
    assert verdict.loop_gate is None
    plain = verdict_to_dict(verdict)
    assert not {"gate", "corrections", "held_out", "canary_decides", "reason"} & set(plain)


# -- nothing changes without the gate ---------------------------------------------------------


def test_without_the_gate_the_output_has_no_gate_keys_and_the_files_are_never_read(lab):
    result = _ship(*LIVE, "--output", "out.json")
    _ok(result, 0)
    out = _out(lab)
    assert not {"gate", "corrections", "held_out", "canary_decides", "reason"} & set(out)
    assert out["decision"] == DECISION_SHIP
    assert "Gate loop-v2" not in _plain(result.output)


# -- refusals -----------------------------------------------------------------------------------

REFUSALS = {
    "unknown_gate": (
        ["--gate", "loop-v3"], "--gate must be loop-v2; got 'loop-v3'",
    ),
    "empty_gate": (["--gate", ""], "--gate must be loop-v2; got ''"),
    "corrections_without_gate": (
        ["--corrections", "corrections.jsonl"], "--corrections needs --gate loop-v2",
    ),
    "held_out_without_gate": (
        ["--held-out", "held.jsonl"], "--held-out needs --gate loop-v2",
    ),
    "gate_with_evidence": (
        ["--gate", "loop-v2", "--evidence", "ev.json"],
        "--gate loop-v2 cannot be used with --evidence: offline evidence has no models to ask the "
        "corrections and held-out questions (live run only)",
    ),
    "gate_with_emit_evidence": (
        ["--gate", "loop-v2", "--emit-evidence", "emitted.json"],
        "--gate loop-v2 cannot be used with --emit-evidence: evidence holds the scores only and "
        "replays as the plain verdict, not the gate's (live run only)",
    ),
    "gate_with_require_provenance": (
        ["--gate", "loop-v2", "--require-provenance"],
        "--gate loop-v2 cannot be used with --require-provenance: it checks offline evidence "
        "(live run only)",
    ),
    "corrections_missing_file": (
        ["--gate", "loop-v2", "--corrections", "nope.jsonl"],
        "--corrections 'nope.jsonl' is not a file",
    ),
    "held_out_directory": (
        ["--gate", "loop-v2", "--held-out", "prod"], "--held-out 'prod' is not a file",
    ),
    "corrections_outside_cwd": (
        ["--gate", "loop-v2", "--corrections", "../corrections.jsonl"],
        "--corrections path '../corrections.jsonl' must stay under cwd",
    ),
    "held_out_outside_cwd": (
        ["--gate", "loop-v2", "--held-out", "../held.jsonl"],
        "--held-out path '../held.jsonl' must stay under cwd",
    ),
}


@pytest.mark.parametrize("argv, message", list(REFUSALS.values()), ids=list(REFUSALS))
def test_a_refusal_is_a_usage_error_that_loads_no_model(lab, argv, message):
    result = _ship(*LIVE, *argv)
    assert result.exit_code == 3, (result.output, repr(result.exception))
    assert message in _plain(result.output), result.output
    assert lab.calls == []
    assert not (lab.root / "emitted.json").exists()


def test_the_gate_composes_with_a_noise_floor(lab):
    _healthy(lab)
    result = _gated(lab, "--noise-floor", "2")
    _ok(result, 0)
    out = _out(lab)
    assert out["noise_floor"]["runs"] == 2
    assert out["gate"] == "loop-v2"
    assert out["corrections"]["outcome"] == "pass"


def test_the_gate_cannot_be_used_without_a_live_run(lab):
    result = _ship("--gate", "loop-v2")
    assert result.exit_code == 3, (result.output, repr(result.exception))
    assert "provide --evidence" in _plain(result.output) or "live run" in _plain(result.output)
    assert lab.calls == []


def test_a_bad_row_in_a_file_is_exit_three_naming_the_file_and_the_line_before_any_model_loads(lab):
    (lab.root / "corrections.jsonl").write_text(
        jsonl(GOOD_CORRECTION, {**GOOD_CORRECTION, "edit_id": "e2", "paraphrases": []}),
        encoding="utf-8",
    )
    result = _ship(*LIVE, "--gate", "loop-v2", "--corrections", "corrections.jsonl")
    assert result.exit_code == 3, (result.output, repr(result.exception))
    assert (
        "--corrections 'corrections.jsonl' line 2: "
        '"paraphrases" must be a list of 1 to 20 non-empty strings'
    ) in _plain(result.output)
    assert lab.calls == []

    (lab.root / "held.jsonl").write_text(
        jsonl(GOOD_HELD_OUT, {**GOOD_HELD_OUT, "edit_id": "h2", "prompt": ""}), encoding="utf-8",
    )
    result = _ship(*LIVE, "--gate", "loop-v2", "--held-out", "held.jsonl")
    assert result.exit_code == 3, (result.output, repr(result.exception))
    assert (
        "--held-out 'held.jsonl' line 2: " '"prompt" must be a non-empty string'
    ) in _plain(result.output)
    assert lab.calls == []


def test_a_file_that_is_not_utf8_is_refused_by_name(lab):
    (lab.root / "corrections.jsonl").write_bytes(b'{"edit_id": "\xff\xfe"}\n')
    result = _ship(*LIVE, "--gate", "loop-v2", "--corrections", "corrections.jsonl")
    assert result.exit_code == 3, (result.output, repr(result.exception))
    assert "--corrections 'corrections.jsonl' is not valid UTF-8 text" in _plain(result.output)
    assert lab.calls == []


def test_a_file_over_the_size_cap_is_refused(lab, monkeypatch):
    monkeypatch.setattr(ship_cmd, "_MAX_GATE_FILE_BYTES", 50)
    (lab.root / "corrections.jsonl").write_text(jsonl(GOOD_CORRECTION), encoding="utf-8")
    result = _ship(*LIVE, "--gate", "loop-v2", "--corrections", "corrections.jsonl")
    assert result.exit_code == 3, (result.output, repr(result.exception))
    assert "--corrections path exceeds 50 bytes" in _plain(result.output)
    assert lab.calls == []


def test_a_hostile_file_name_does_not_reach_the_terminal_raw(lab):
    result = _ship(*LIVE, "--gate", "loop-v2", "--corrections", "bad\x1b[2Jname.jsonl")
    assert result.exit_code == 3, (result.output, repr(result.exception))
    assert "\x1b[2J" not in result.output
    assert lab.calls == []


@pytest.mark.requires_symlink
def test_a_symlinked_input_file_is_refused(lab):
    lab.write()
    (lab.root / "link.jsonl").symlink_to(lab.root / "corrections.jsonl")
    result = _ship(*LIVE, "--gate", "loop-v2", "--corrections", "link.jsonl")
    assert result.exit_code == 3, (result.output, repr(result.exception))
    assert "must not be a symlink" in _plain(result.output)
    assert lab.calls == []


# -- --help --------------------------------------------------------------------------------------


def test_help_lists_the_three_options():
    result = _ship("--help")
    assert result.exit_code == 0, (result.output, repr(result.exception))
    flat = _plain(result.output)
    for flag in ("--gate", "--corrections", "--held-out"):
        assert flag in flat
    assert "loop-v2" in flat
