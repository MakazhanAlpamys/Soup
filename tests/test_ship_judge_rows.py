"""``soup ship --task-mode pairwise`` keeps the judge's per-row verdicts beside ``--output``.

Without a model, a GPU or a network: ``live_eval.make_generator`` is a recorder and the judge is
the real ``JudgeEvaluator`` with only its HTTP call scripted (so the real prompt, the real reply
parser and the real two-order combination all run). Four layers:

* ``pairwise_compare_detailed`` / ``pairwise_winrate(on_row=...)`` on a fake ``compare_pair``;
* the row ids, the prompt hash, the JSONL text, the sidecar path and the write guards, as pure
  functions;
* the CLI: the sidecar's rows (schema 2: the prompt and both answers, with the hash of the prompt),
  both orders, the combined result, why a tie is a tie (``tie_reason``), whether an answer hit the
  generation cap, the judge's model name, the held-out rows under ``--gate loop-v2``, no file
  without ``--output``, no noise-floor rows;
* the verdict JSON: byte for byte what the command wrote before the rows existed, plus the one
  ``judge_rows`` pointer when the rows were written.

The schema and the example file are tested in ``test_ship_judge_row_schema.py``.

The new names are imported inside the tests that use them (``GOLDEN_VERDICT`` was captured on a
tree that did not have the rows).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import pytest

from soup_cli import __version__
from soup_cli.cli import app
from soup_cli.eval.judge import JudgeEvaluator, JudgeUnavailableError
from soup_cli.utils import live_eval
from soup_cli.utils.exit_codes import EXIT_GATE_FAILED, EXIT_RUNTIME_ERROR
from soup_cli.utils.live_eval import GeneratedText
from tests.cli_runner import CliRunner
from tests.conftest import strip_ansi

# ---------------------------------------------------------------------------
# The scenario: five task prompts and three held-out edits, a judge with a script
# ---------------------------------------------------------------------------

TASKS = [
    {"prompt": "capital of France?", "expected": "Paris", "id": "q-fr"},
    {"prompt": "2+2?", "expected": "4", "edit_id": "e-two"},
    {"prompt": "unlabelled question", "expected": "x"},
    {"prompt": "order sensitive", "expected": "y", "id": "q-order"},
    {"prompt": "garbled judge", "expected": "z", "id": "q-garbled"},
]
HELD_OUT = [
    {"edit_id": "h-1", "prompt": "held one", "expected": "alpha bravo", "original": "no idea"},
    {"edit_id": "h-2", "prompt": "held two", "expected": "alpha bravo", "original": "no idea"},
    {"edit_id": "h-3", "prompt": "held three", "expected": "alpha bravo", "original": "no idea"},
]

# What the judge answers per question, for each order, in terms of the candidate it picks:
# "base" / "tuned", "tie" (a reply that says so: ``{"winner": "tie"}``) or "garbage" (a reply that
# is not an answer). ``base_first`` is the order in which the base answer is shown as A.
SCRIPT: Dict[str, Dict[str, str]] = {
    "capital of France?": {"base_first": "tuned", "tuned_first": "tuned"},
    "2+2?": {"base_first": "base", "tuned_first": "base"},
    "unlabelled question": {"base_first": "tuned", "tuned_first": "tuned"},
    # Picks position A in both orders: base when base is first, tuned when tuned is first.
    "order sensitive": {"base_first": "base", "tuned_first": "tuned"},
    "garbled judge": {"base_first": "tuned", "tuned_first": "garbage"},
    "held one": {"base_first": "tuned", "tuned_first": "tuned"},
    "held two": {"base_first": "base", "tuned_first": "base"},
    "held three": {"base_first": "tie", "tuned_first": "tie"},
}
# The answers (label, prompt) that stopped at the generation cap, as the generator reports it.
CAPPED = {
    ("base", "capital of France?"),
    ("tuned", "unlabelled question"),
    ("base", "held one"),
    ("tuned", "held three"),
}
ANSWERED = {row["prompt"] for row in TASKS + HELD_OUT}

_JUDGE_PROMPT = re.compile(
    r"## Prompt\n(.*)\n\n## Response A\n(.*)\n\n## Response B\n(.*)\n\n## Task", re.DOTALL
)


class ScriptedJudge(JudgeEvaluator):
    """The real evaluator, its prompt and its reply parser; only the HTTP call is scripted."""

    def _call_llm(self, prompt: str, max_tokens: int = 1024) -> str:
        match = _JUDGE_PROMPT.search(prompt)
        assert match is not None, prompt
        question, resp_a, _resp_b = match.groups()
        first = "base" if resp_a.startswith("base: ") else "tuned"
        picked = SCRIPT[question][f"{first}_first"]
        if picked == "garbage":
            return "I cannot decide between them."
        if picked == "tie":
            return json.dumps({"winner": "tie"})
        return json.dumps({"winner": "A" if picked == first else "B"})


def fake_make_generator(model_id, **kwargs):
    loaded = kwargs.get("loaded")
    label = loaded[0] if loaded is not None else model_id

    def generate(prompt: str) -> str:
        text = f"{label}: {prompt}" if prompt in ANSWERED else "B"
        return GeneratedText(text, truncated=(label, prompt) in CAPPED)

    generate.loaded = (label, "tokenizer", "cpu")  # what the real generator exposes
    return generate


def jsonl(rows) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows)


def run_ship(tmp_path: Path, monkeypatch, *args: str, tasks=TASKS):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(live_eval, "make_generator", fake_make_generator)
    monkeypatch.setattr("soup_cli.eval.judge.JudgeEvaluator", ScriptedJudge)
    (tmp_path / "tasks.jsonl").write_text(jsonl(tasks), encoding="utf-8")
    (tmp_path / "held.jsonl").write_text(jsonl(HELD_OUT), encoding="utf-8")
    return CliRunner().invoke(
        app,
        [
            "ship",
            "--base",
            "base",
            "--tuned",
            "tuned",
            "--task-eval",
            "tasks.jsonl",
            "--device",
            "cpu",
            *args,
        ],
    )


PAIRWISE = ["--task-mode", "pairwise", "--judge-model", "ollama://judge"]


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def read_rows(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def expected_row(**fields):
    """One row as it is written, with every field spelled out by the caller (nothing derived)."""
    return {"schema_version": 2, "judge_model": "judge", **fields}


EXPECTED_TASK_ROWS = [
    expected_row(
        leg="task_eval",
        index=0,
        id="q-fr",
        id_source="id",
        prompt_hash="7b9322536ee4",
        prompt="capital of France?",
        base_answer="base: capital of France?",
        tuned_answer="tuned: capital of France?",
        base_truncated=True,
        tuned_truncated=False,
        base_first="tuned",
        tuned_first="tuned",
        result="tuned",
        tie_reason=None,
        score=1.0,
    ),
    expected_row(
        leg="task_eval",
        index=1,
        id="e-two",
        id_source="edit_id",
        prompt_hash="30ad205245cd",
        prompt="2+2?",
        base_answer="base: 2+2?",
        tuned_answer="tuned: 2+2?",
        base_truncated=False,
        tuned_truncated=False,
        base_first="base",
        tuned_first="base",
        result="base",
        tie_reason=None,
        score=0.0,
    ),
    expected_row(
        leg="task_eval",
        index=2,
        id=digest("unlabelled question"),
        id_source="prompt_sha256",
        prompt_hash="6f6c187f40b6",
        prompt="unlabelled question",
        base_answer="base: unlabelled question",
        tuned_answer="tuned: unlabelled question",
        base_truncated=False,
        tuned_truncated=True,
        base_first="tuned",
        tuned_first="tuned",
        result="tuned",
        tie_reason=None,
        score=1.0,
    ),
    # A judge that always says "the first one": the two orders disagree, so a tie.
    expected_row(
        leg="task_eval",
        index=3,
        id="q-order",
        id_source="id",
        prompt_hash="c34a5ace8ac9",
        prompt="order sensitive",
        base_answer="base: order sensitive",
        tuned_answer="tuned: order sensitive",
        base_truncated=False,
        tuned_truncated=False,
        base_first="base",
        tuned_first="tuned",
        result="tie",
        tie_reason="order_disagreement",
        score=0.5,
    ),
    # A reply that is not an answer in one order: no pick there, so a tie nobody chose.
    expected_row(
        leg="task_eval",
        index=4,
        id="q-garbled",
        id_source="id",
        prompt_hash="86d45f7aa214",
        prompt="garbled judge",
        base_answer="base: garbled judge",
        tuned_answer="tuned: garbled judge",
        base_truncated=False,
        tuned_truncated=False,
        base_first="tuned",
        tuned_first="none",
        result="tie",
        tie_reason="unreadable_reply",
        score=0.5,
    ),
]
EXPECTED_HELD_OUT_ROWS = [
    expected_row(
        leg="held_out",
        index=0,
        id="h-1",
        id_source="edit_id",
        prompt_hash="be9c01a32835",
        prompt="held one",
        base_answer="base: held one",
        tuned_answer="tuned: held one",
        base_truncated=True,
        tuned_truncated=False,
        base_first="tuned",
        tuned_first="tuned",
        result="tuned",
        tie_reason=None,
        score=1.0,
    ),
    expected_row(
        leg="held_out",
        index=1,
        id="h-2",
        id_source="edit_id",
        prompt_hash="149371b6e6fd",
        prompt="held two",
        base_answer="base: held two",
        tuned_answer="tuned: held two",
        base_truncated=False,
        tuned_truncated=False,
        base_first="base",
        tuned_first="base",
        result="base",
        tie_reason=None,
        score=0.0,
    ),
    # The judge said "tie" in both orders: a tie it chose.
    expected_row(
        leg="held_out",
        index=2,
        id="h-3",
        id_source="edit_id",
        prompt_hash="8eb462209977",
        prompt="held three",
        base_answer="base: held three",
        tuned_answer="tuned: held three",
        base_truncated=False,
        tuned_truncated=True,
        base_first="none",
        tuned_first="none",
        result="tie",
        tie_reason="real_tie",
        score=0.5,
    ),
]


@pytest.fixture
def judged(tmp_path, monkeypatch):
    """One pairwise run with ``--output out.json``: ``(result, tmp_path)``."""
    result = run_ship(tmp_path, monkeypatch, *PAIRWISE, "--output", "out.json")
    assert result.exit_code == 0, (result.output, repr(result.exception))
    return result, tmp_path


# ---------------------------------------------------------------------------
# The judge: both presentation orders, kept
# ---------------------------------------------------------------------------


class ScriptedPairs:
    """A ``compare_pair`` that answers by (prompt, shown first) from a table."""

    def __init__(self, table):
        self.table = table
        self.calls: list = []

    def compare_pair(self, prompt, resp_a, resp_b):
        self.calls.append((prompt, resp_a, resp_b))
        return self.table[(prompt, resp_a)]


# (verdict with A,B shown; verdict with B,A shown) -> (translated second, combined)
ORDER_CASES = [
    (0, 1, 0, 0),  # A in both orders: A wins
    (1, 0, 1, 1),  # B in both orders: B wins
    (0, 0, 1, -1),  # always the first one shown: position bias, no winner
    (1, 1, 0, -1),  # always the second one shown
    (-1, 0, 1, -1),  # no pick in the first order
    (0, -1, -1, -1),  # no pick in the second order
    (-1, -1, -1, -1),
]


@pytest.mark.parametrize("first,swapped,second,combined", ORDER_CASES)
def test_detailed_outcome_keeps_both_orders_and_the_combined_verdict(
    first, swapped, second, combined
):
    from soup_cli.eval.judge import PairwiseOutcome, pairwise_compare, pairwise_compare_detailed

    judge = ScriptedPairs({("p", "aa"): first, ("p", "bb"): swapped})
    outcome = pairwise_compare_detailed("p", "aa", "bb", judge)
    assert outcome == PairwiseOutcome(
        first=first, swapped=swapped, second=second, combined=combined
    )
    assert judge.calls == [("p", "aa", "bb"), ("p", "bb", "aa")]
    # the original entry point is the same verdict
    assert pairwise_compare("p", "aa", "bb", ScriptedPairs(judge.table)) == combined


def test_a_single_order_outcome_has_no_second_order():
    from soup_cli.eval.judge import PairwiseOutcome, pairwise_compare_detailed

    judge = ScriptedPairs({("p", "aa"): 1})
    assert pairwise_compare_detailed("p", "aa", "bb", judge, swap=False) == PairwiseOutcome(
        first=1, swapped=None, second=None, combined=1
    )
    assert judge.calls == [("p", "aa", "bb")]


def test_on_row_sees_every_pair_in_order_and_does_not_change_the_win_rate():
    from soup_cli.eval.judge import PairwiseOutcome, pairwise_winrate

    table = {
        ("p0", "b0"): 1,
        ("p0", "t0"): 0,  # tuned both orders: win
        ("p1", "b1"): 0,
        ("p1", "t1"): 1,  # base both orders: loss
        ("p2", "b2"): 0,
        ("p2", "t2"): 0,  # first one shown both orders: tie
    }
    pairs = [("p0", "b0", "t0"), ("p1", "b1", "t1"), ("p2", "b2", "t2")]
    seen: list = []
    with_hook = pairwise_winrate(
        pairs, ScriptedPairs(table), on_row=lambda index, outcome: seen.append((index, outcome))
    )
    assert with_hook == pairwise_winrate(pairs, ScriptedPairs(table)) == 0.5
    assert seen == [
        (0, PairwiseOutcome(first=1, swapped=0, second=1, combined=1)),
        (1, PairwiseOutcome(first=0, swapped=1, second=0, combined=0)),
        (2, PairwiseOutcome(first=0, swapped=0, second=1, combined=-1)),
    ]


def test_on_row_is_not_called_for_a_pair_the_judge_could_not_judge():
    from soup_cli.eval.judge import pairwise_winrate

    class Dies(ScriptedPairs):
        def compare_pair(self, prompt, resp_a, resp_b):
            if prompt == "p1":
                raise JudgeUnavailableError("gone", url="http://judge")
            return super().compare_pair(prompt, resp_a, resp_b)

    table = {("p0", "b0"): 1, ("p0", "t0"): 0}
    seen: list = []
    with pytest.raises(JudgeUnavailableError):
        pairwise_winrate(
            [("p0", "b0", "t0"), ("p1", "b1", "t1")],
            Dies(table),
            on_row=lambda index, outcome: seen.append(index),
        )
    assert seen == [0]


# ---------------------------------------------------------------------------
# Why a tie is a tie
# ---------------------------------------------------------------------------

# (verdict, shown A,B; verdict, shown B,A; the first reply was unreadable; the second was)
# -> tie_reason. A verdict of -1 is "no pick"; "unreadable" is a reply the parser could not read.
TIE_REASON_CASES = [
    (0, 1, False, False, None),  # A in both orders: a winner, not a tie
    (1, 0, False, False, None),  # B in both orders
    (0, 0, False, False, "order_disagreement"),  # always the first one shown
    (1, 1, False, False, "order_disagreement"),  # always the second one shown
    (-1, -1, False, False, "real_tie"),  # no pick in either order, and both replies were read
    (-1, 0, False, False, "order_disagreement"),  # tie, then a pick: the orders disagree
    (0, -1, False, False, "order_disagreement"),
    (-1, 1, False, False, "order_disagreement"),
    (1, -1, False, False, "order_disagreement"),
    (-1, -1, True, False, "unreadable_reply"),  # the first reply could not be read
    (-1, -1, False, True, "unreadable_reply"),  # the second could not
    (-1, -1, True, True, "unreadable_reply"),
    (-1, 0, True, False, "unreadable_reply"),  # unreadable beats a disagreement
    (0, -1, False, True, "unreadable_reply"),
    (-1, 1, True, False, "unreadable_reply"),
    (1, -1, False, True, "unreadable_reply"),
]


def _scripted(first, swapped, first_unreadable, swapped_unreadable):
    from soup_cli.eval.judge import UnreadableReply

    return ScriptedPairs(
        {
            ("p", "aa"): UnreadableReply() if first_unreadable else first,
            ("p", "bb"): UnreadableReply() if swapped_unreadable else swapped,
        }
    )


@pytest.mark.parametrize("first,swapped,first_bad,swapped_bad,reason", TIE_REASON_CASES)
def test_an_outcome_says_why_it_is_a_tie(first, swapped, first_bad, swapped_bad, reason):
    from soup_cli.eval.judge import pairwise_compare, pairwise_compare_detailed

    outcome = pairwise_compare_detailed(
        "p", "aa", "bb", _scripted(first, swapped, first_bad, swapped_bad)
    )
    assert outcome.tie_reason == reason
    assert (outcome.first_unreadable, outcome.swapped_unreadable) == (first_bad, swapped_bad)
    # what a reply could not be read as changes nothing about who wins
    plain = ScriptedPairs({("p", "aa"): first, ("p", "bb"): swapped})
    winner = pairwise_compare("p", "aa", "bb", _scripted(first, swapped, first_bad, swapped_bad))
    assert winner == pairwise_compare("p", "aa", "bb", plain)
    assert type(winner) is int  # a plain verdict, whatever the judge returned
    assert (outcome.combined == -1) == (reason is not None)


@pytest.mark.parametrize("unreadable,reason", [(False, "real_tie"), (True, "unreadable_reply")])
def test_a_pair_judged_in_one_order_with_no_pick_is_a_tie_for_the_reason_of_its_reply(
    unreadable, reason
):
    from soup_cli.eval.judge import UnreadableReply, pairwise_compare_detailed

    judge = ScriptedPairs({("p", "aa"): UnreadableReply() if unreadable else -1})
    outcome = pairwise_compare_detailed("p", "aa", "bb", judge, swap=False)
    assert (outcome.combined, outcome.second, outcome.tie_reason) == (-1, None, reason)
    assert type(outcome.first) is int
    one_pick = ScriptedPairs({("p", "aa"): 1})
    decided = pairwise_compare_detailed("p", "aa", "bb", one_pick, swap=False)
    assert decided.tie_reason is None


def test_an_unreadable_reply_is_a_minus_one_everywhere_a_verdict_is_compared():
    from soup_cli.eval.judge import UnreadableReply

    unreadable = UnreadableReply()
    assert unreadable == -1 and isinstance(unreadable, int)
    assert unreadable not in (0, 1) and unreadable in (-1,)
    assert {0: "a", 1: "b"}.get(unreadable, "tie") == "tie"
    assert json.dumps([unreadable]) == "[-1]"


class ReplyJudge(JudgeEvaluator):
    """The real evaluator and its reply parser, answering every call with one fixed reply."""

    reply = ""

    def _call_llm(self, prompt: str, max_tokens: int = 1024) -> str:
        return self.reply


def _read(reply: str):
    from soup_cli.eval.judge import UnreadableReply

    judge = ReplyJudge(provider="ollama", model="judge")
    judge.reply = reply
    verdict = judge.compare_pair("p", "aa", "bb")
    return int(verdict), isinstance(verdict, UnreadableReply)


@pytest.mark.parametrize(
    "reply,verdict,unreadable",
    [
        ('{"winner": "A"}', 0, False),
        ('{"winner": "B"}', 1, False),
        ('the answer is {"winner": "b"}', 1, False),
        ("A is clearly better", 0, False),
        ("B wins here", 1, False),
        # a judge that says it is a tie: read, and no pick
        ('{"winner": "tie"}', -1, False),
        ('{"winner": "Tie."}', -1, False),
        ('{"winner": " NEITHER "}', -1, False),
        ('{"winner": "equal"}', -1, False),
        ('{"winner": "draw"}', -1, False),
        ('{"winner": "tied"}', -1, False),
        ('{"winner": "both"}', -1, False),
        ('{"winner": "none"}', -1, False),
        ("tie", -1, False),
        (" Tie.\n", -1, False),
        ('"draw"', -1, False),
        # anything else that is not an answer: unreadable
        ("I cannot decide between them.", -1, True),
        ("It is a tie between the two.", -1, True),
        ("", -1, True),
        ("   ", -1, True),
        ("{}", -1, True),
        ('{"winner": ""}', -1, True),
        ('{"winner": "C"}', -1, True),
        ('{"winner": null}', -1, True),
        ('{"winner": ["tie"]}', -1, True),
        ('{"winner": 7}', -1, True),
        ('{"verdict": "A"}', -1, True),
        ('{"winner": "tie"', -1, True),  # cut off: no closing brace
        ("[1, 2]", -1, True),
    ],
)
def test_the_real_judge_tells_a_tie_it_was_given_from_a_reply_it_could_not_read(
    reply, verdict, unreadable
):
    from soup_cli.eval.judge import _parse_pairwise

    assert _read(reply) == (verdict, unreadable)
    # and the verdict is the one the parser has always returned
    assert _parse_pairwise(reply) == verdict


def test_the_judge_is_asked_exactly_what_it_was_asked_before():
    # nothing the judge sees moved: the same calls, in the same two orders, with the same prompt
    from soup_cli.eval.judge import _PAIRWISE_INSTRUCTIONS, pairwise_compare_detailed

    seen: list = []

    class Recording(ReplyJudge):
        def _call_llm(self, prompt: str, max_tokens: int = 1024) -> str:
            seen.append((prompt, max_tokens))
            return '{"winner": "A"}'

    pairwise_compare_detailed("the prompt", "one", "two", Recording(provider="ollama", model="m"))
    assert seen == [
        (_PAIRWISE_INSTRUCTIONS.format(prompt="the prompt", resp_a="one", resp_b="two"), 1024),
        (_PAIRWISE_INSTRUCTIONS.format(prompt="the prompt", resp_a="two", resp_b="one"), 1024),
    ]


# ---------------------------------------------------------------------------
# Ids, the JSONL text, the path, the write guards
# ---------------------------------------------------------------------------


def test_a_row_is_named_by_its_edit_id_then_its_id_then_the_hash_of_its_prompt():
    from soup_cli.utils.ship_judge_rows import RowId, row_id_of

    prompt = "what is 2+2?"
    assert row_id_of(prompt, {"edit_id": "e1", "id": "i1"}) == RowId("e1", "edit_id")
    assert row_id_of(prompt, {"id": "i1"}) == RowId("i1", "id")
    assert row_id_of(prompt, {"id": 17}) == RowId("17", "id")  # a numeric id is still an id
    assert row_id_of(prompt, {}) == RowId(digest(prompt), "prompt_sha256")
    assert row_id_of(prompt, None) == RowId(digest(prompt), "prompt_sha256")


@pytest.mark.parametrize(
    "bad",
    [True, False, "", "   ", "x" * 257, None, 1.5, ["a"], {"a": 1}],
    ids=["true", "false", "empty", "blank", "too-long", "none", "float", "list", "dict"],
)
def test_an_id_that_is_not_usable_falls_back_to_the_hash_instead_of_failing_the_run(bad):
    from soup_cli.utils.ship_judge_rows import RowId, row_id_of

    assert row_id_of("q", {"id": bad}) == RowId(digest("q"), "prompt_sha256")
    # an unusable edit_id does not hide a usable id behind it
    assert row_id_of("q", {"edit_id": bad, "id": "ok"}) == RowId("ok", "id")


def test_the_longest_allowed_id_is_kept():
    from soup_cli.utils.ship_judge_rows import RowId, row_id_of

    assert row_id_of("q", {"id": "x" * 256}) == RowId("x" * 256, "id")


def test_the_hash_of_a_prompt_with_a_lone_surrogate_does_not_stop_the_run():
    from soup_cli.utils.ship_judge_rows import row_id_of

    found = row_id_of("bad \ud800 prompt")
    assert found.source == "prompt_sha256"
    assert re.fullmatch(r"[0-9a-f]{64}", found.value)


def test_every_row_is_one_json_line_whatever_the_id_and_the_texts_hold():
    from soup_cli.eval.judge import PairwiseOutcome
    from soup_cli.utils.ship_judge_rows import JudgeRowRecorder, RowId

    recorder = JudgeRowRecorder()
    outcome = PairwiseOutcome(first=1, swapped=0, second=1, combined=1)
    nasty = ["line\nbreak", "tab\there", "naïve ✓", "lone \ud800 surrogate", '"quoted"']
    for index, name in enumerate(nasty):
        # the same nasty text as the id, the prompt and both answers
        recorder.add(
            "task_eval",
            index,
            RowId(name, "id"),
            outcome,
            (name, name + "<", name + ">"),
            judge_model=name,
        )
    text = recorder.to_jsonl()
    lines = text.split("\n")
    assert lines[-1] == "" and len(lines) == len(nasty) + 1
    parsed = [json.loads(line) for line in lines[:-1]]
    assert [row["id"] for row in parsed] == nasty
    assert [row["prompt"] for row in parsed] == nasty
    assert [row["base_answer"] for row in parsed] == [name + "<" for name in nasty]
    assert [row["tuned_answer"] for row in parsed] == [name + ">" for name in nasty]
    assert [row["judge_model"] for row in parsed] == nasty
    assert text.isascii()


def test_a_tuned_win_scores_one_a_tie_a_half_and_a_base_win_zero():
    from soup_cli.eval.judge import PairwiseOutcome
    from soup_cli.utils.ship_judge_rows import JudgeRowRecorder, RowId

    recorder = JudgeRowRecorder()
    for index, combined in enumerate([1, -1, 0]):
        recorder.add(
            "held_out",
            index,
            RowId("r", "id"),
            PairwiseOutcome(first=combined, swapped=None, second=None, combined=combined),
            ("p", "b", "t"),
            judge_model="judge",
        )
    assert [(row["result"], row["score"]) for row in recorder.rows] == [
        ("tuned", 1.0),
        ("tie", 0.5),
        ("base", 0.0),
    ]
    # judged in one order only: the second order is null, not "none"
    assert [row["tuned_first"] for row in recorder.rows] == [None, None, None]


@pytest.mark.parametrize("first,swapped,first_bad,swapped_bad,reason", TIE_REASON_CASES)
def test_a_tie_row_carries_its_reason_and_every_other_row_a_null(
    first, swapped, first_bad, swapped_bad, reason
):
    from soup_cli.eval.judge import pairwise_compare_detailed
    from soup_cli.utils.ship_judge_rows import JudgeRowRecorder, RowId

    outcome = pairwise_compare_detailed(
        "p", "aa", "bb", _scripted(first, swapped, first_bad, swapped_bad)
    )
    recorder = JudgeRowRecorder()
    recorder.add("task_eval", 0, RowId("r", "id"), outcome, ("p", "aa", "bb"), judge_model="judge")
    (row,) = recorder.rows
    assert "tie_reason" in row  # the key is always there: null on a row that is not a tie
    assert row["tie_reason"] == reason
    assert (row["result"] == "tie") == (reason is not None)


def test_a_row_says_whether_each_answer_hit_the_generation_cap():
    from soup_cli.eval.judge import PairwiseOutcome
    from soup_cli.utils.ship_judge_rows import JudgeRowRecorder, RowId

    outcome = PairwiseOutcome(first=1, swapped=0, second=1, combined=1)
    recorder = JudgeRowRecorder()
    for base_capped, tuned_capped in [(True, False), (False, True), (True, True), (False, False)]:
        pair = (
            "p",
            GeneratedText("short", truncated=base_capped),
            GeneratedText("a much much longer answer than the other one", truncated=tuned_capped),
        )
        recorder.add("task_eval", 0, RowId("r", "id"), outcome, pair, judge_model="judge")
    assert [(row["base_truncated"], row["tuned_truncated"]) for row in recorder.rows] == [
        (True, False),
        (False, True),
        (True, True),
        (False, False),
    ]
    # the flag is the generator's word, not a guess from the text
    assert all(type(row["base_truncated"]) is bool for row in recorder.rows)


def test_an_answer_from_a_generator_that_does_not_say_is_null_not_false():
    from soup_cli.eval.judge import PairwiseOutcome
    from soup_cli.utils.ship_judge_rows import JudgeRowRecorder, RowId

    recorder = JudgeRowRecorder()
    outcome = PairwiseOutcome(first=1, swapped=0, second=1, combined=1)
    pair = ("p", "a plain string", GeneratedText("capped", truncated=True))
    recorder.add("task_eval", 0, RowId("r", "id"), outcome, pair, judge_model="judge")
    (row,) = recorder.rows
    assert (row["base_truncated"], row["tuned_truncated"]) == (None, True)


def test_every_row_names_the_judge_model_it_was_given():
    from soup_cli.eval.judge import PairwiseOutcome
    from soup_cli.utils.ship_judge_rows import JudgeRowRecorder, RowId, recording, row_sink

    outcome = PairwiseOutcome(first=1, swapped=0, second=1, combined=1)
    recorder = JudgeRowRecorder()
    recorder.add("held_out", 0, RowId("r", "id"), outcome, ("p", "b", "t"), judge_model="m-1")
    with recording(recorder, "task_eval"):
        row_sink([("p", "b", "t")], judge_model="m-2")(0, outcome)
    assert [row["judge_model"] for row in recorder.rows] == ["m-1", "m-2"]


@pytest.mark.parametrize(
    "output,expected",
    [
        ("verdict.json", "verdict.judge-rows.jsonl"),
        ("out/run.v2.json", "out/run.v2.judge-rows.jsonl"),
        ("verdict", "verdict.judge-rows.jsonl"),
        ("dir.d/verdict", "dir.d/verdict.judge-rows.jsonl"),
        ("verdict.judge-rows.jsonl", "verdict.judge-rows.judge-rows.jsonl"),
    ],
)
def test_the_sidecar_sits_beside_the_output_under_its_stem(output, expected):
    from soup_cli.utils.ship_judge_rows import judge_rows_path

    assert judge_rows_path(output) == expected
    assert judge_rows_path(output) != output


def test_the_sidecar_is_refused_outside_the_working_directory(tmp_path, monkeypatch):
    from soup_cli.utils.ship_judge_rows import JudgeRowRecorder, write_judge_rows

    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    for outside in ("../escape.json", str(tmp_path / "escape.json")):
        with pytest.raises(ValueError, match="must stay under cwd"):
            write_judge_rows(JudgeRowRecorder(), outside)
    assert not list(tmp_path.glob("escape*"))


@pytest.mark.requires_symlink
def test_the_sidecar_is_never_written_through_a_symlink(tmp_path, monkeypatch):
    from soup_cli.utils.ship_judge_rows import JudgeRowRecorder, write_judge_rows

    monkeypatch.chdir(tmp_path)
    target = tmp_path / "victim.txt"
    target.write_text("keep me", encoding="utf-8")
    os.symlink(target, tmp_path / "verdict.judge-rows.jsonl")
    with pytest.raises(ValueError, match="must not be a symlink"):
        write_judge_rows(JudgeRowRecorder(), "verdict.json")
    assert target.read_text(encoding="utf-8") == "keep me"


# ---------------------------------------------------------------------------
# The scope: rows are recorded only inside the passes ship asks to keep
# ---------------------------------------------------------------------------


def test_there_is_no_sink_outside_a_recording_scope_and_none_is_left_behind():
    from soup_cli.utils.ship_judge_rows import JudgeRowRecorder, recording, row_sink

    pairs = [("p", "a", "b")]
    assert row_sink(pairs, judge_model="judge") is None
    with recording(JudgeRowRecorder(), "task_eval"):
        assert row_sink(pairs, judge_model="judge") is not None
    assert row_sink(pairs, judge_model="judge") is None
    with pytest.raises(RuntimeError):
        with recording(JudgeRowRecorder(), "task_eval"):
            raise RuntimeError("boom")
    assert row_sink(pairs, judge_model="judge") is None
    with recording(None, "task_eval"):
        assert row_sink(pairs, judge_model="judge") is None


def test_a_scope_names_its_rows_by_position_and_a_sink_records_into_its_recorder():
    from soup_cli.eval.judge import PairwiseOutcome
    from soup_cli.utils.ship_judge_rows import JudgeRowRecorder, RowId, recording, row_sink

    recorder = JudgeRowRecorder()
    ids = [RowId("first", "edit_id"), RowId("second", "edit_id")]
    pairs = [("p0", "a0", "b0"), ("p1", "a1", "b1"), ("p2", "a2", "b2")]
    with recording(recorder, "held_out", ids):
        sink = row_sink(pairs, judge_model="judge")
        for index in range(3):
            sink(index, PairwiseOutcome(first=1, swapped=0, second=1, combined=1))
    assert [(row["leg"], row["id"], row["id_source"]) for row in recorder.rows] == [
        ("held_out", "first", "edit_id"),
        ("held_out", "second", "edit_id"),
        ("held_out", digest("p2"), "prompt_sha256"),  # past the named ids: the prompt's hash
    ]


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


def test_the_sidecar_has_each_rows_id_both_orders_and_the_combined_result(judged):
    result, root = judged
    assert read_rows(root / "out.judge-rows.jsonl") == EXPECTED_TASK_ROWS
    text = _plain(result.output)
    assert "Wrote judge rows" in text
    assert "(5 rows)" in text


def test_the_scores_of_the_rows_are_the_win_rate_in_the_verdict(judged):
    _result, root = judged
    rows = read_rows(root / "out.judge-rows.jsonl")
    verdict = json.loads((root / "out.json").read_text(encoding="utf-8"))
    assert verdict["task_win"] == {"mode": "pairwise", "base": 0.5, "tuned": 0.6, "won": True}
    assert sum(row["score"] for row in rows) / len(rows) == verdict["task_win"]["tuned"]


def test_the_sidecar_carries_the_prompt_and_both_answers_the_judge_saw(judged):
    _result, root = judged
    rows = read_rows(root / "out.judge-rows.jsonl")
    # the generator answers "<model>: <prompt>", so a swapped pair would read the other way round
    for row, task in zip(rows, TASKS):
        assert row["prompt"] == task["prompt"]
        assert row["base_answer"] == f"base: {task['prompt']}"
        assert row["tuned_answer"] == f"tuned: {task['prompt']}"
        assert row["base_answer"] != row["tuned_answer"]
    text = (root / "out.judge-rows.jsonl").read_text(encoding="utf-8")
    assert "Paris" not in text  # an expected answer is not what the judge saw: it stays out


def test_the_texts_are_the_base_answer_and_the_tuned_answer_whatever_order_the_judge_picked(
    judged,
):
    _result, root = judged
    rows = {row["id"]: row for row in read_rows(root / "out.judge-rows.jsonl")}
    # the judge picks the FIRST answer shown in both orders: base_first=base, tuned_first=tuned.
    # The two orders differ in who was shown first; the texts do not move.
    order_sensitive = rows["q-order"]
    assert (order_sensitive["base_first"], order_sensitive["tuned_first"]) == ("base", "tuned")
    assert order_sensitive["base_answer"] == "base: order sensitive"
    assert order_sensitive["tuned_answer"] == "tuned: order sensitive"
    # and where both orders pick the same side, the texts are still not swapped with it
    assert rows["e-two"]["result"] == "base"
    assert rows["e-two"]["base_answer"] == "base: 2+2?"
    assert rows["e-two"]["tuned_answer"] == "tuned: 2+2?"


@pytest.mark.parametrize("first,swapped,second,combined", ORDER_CASES)
def test_a_sink_records_the_pair_as_base_then_tuned_for_every_outcome(
    first, swapped, second, combined
):
    from soup_cli.eval.judge import PairwiseOutcome
    from soup_cli.utils.ship_judge_rows import JudgeRowRecorder, recording, row_sink

    recorder = JudgeRowRecorder()
    pairs = [("the prompt", "BASE TEXT", "TUNED TEXT")]
    with recording(recorder, "task_eval"):
        row_sink(pairs, judge_model="judge")(
            0, PairwiseOutcome(first=first, swapped=swapped, second=second, combined=combined)
        )
    (row,) = recorder.rows
    assert (row["prompt"], row["base_answer"], row["tuned_answer"]) == (
        "the prompt",
        "BASE TEXT",
        "TUNED TEXT",
    )


def test_the_rows_are_schema_two(judged):
    from soup_cli.utils import ship_judge_rows

    _result, root = judged
    assert ship_judge_rows.SCHEMA_VERSION == 2
    rows = read_rows(root / "out.judge-rows.jsonl")
    assert len(rows) == 5
    assert {row["schema_version"] for row in rows} == {2}


def test_the_prompt_hash_is_the_first_twelve_hex_of_sha256_of_the_stripped_casefolded_prompt():
    from soup_cli.utils.ship_judge_rows import prompt_hash

    prompt = "  What is 2+2?  "
    # the literal Soup Zero's held_out_audit joins on: a change of the rule must show up here
    assert prompt_hash(prompt) == "61a3385003d9"
    # and the same rule written out independently of the implementation
    expected = hashlib.sha256(prompt.strip().casefold().encode("utf-8", "surrogatepass"))
    assert prompt_hash(prompt) == expected.hexdigest()[:12]
    assert re.fullmatch(r"[0-9a-f]{12}", prompt_hash(prompt))


@pytest.mark.parametrize(
    "prompt,same_as",
    [
        ("WHAT IS 2+2?", "What is 2+2?"),  # case
        ("\t What is 2+2?\n\n", "What is 2+2?"),  # outer whitespace only
        ("STRASSE", "Straße"),  # casefold, which lower() is not: ß folds to ss
    ],
)
def test_the_prompt_hash_ignores_case_and_outer_whitespace(prompt, same_as):
    from soup_cli.utils.ship_judge_rows import prompt_hash

    assert prompt_hash(prompt) == prompt_hash(same_as)


def test_the_prompt_hash_keeps_inner_whitespace_and_tells_prompts_apart():
    from soup_cli.utils.ship_judge_rows import prompt_hash

    assert prompt_hash("What is  2+2?") != prompt_hash("What is 2+2?")
    assert prompt_hash("What is 2+3?") != prompt_hash("What is 2+2?")
    assert prompt_hash("Straße") == "16d969520877"  # the casefolded key is 'strasse'


def test_the_prompt_hash_of_a_prompt_with_a_lone_surrogate_does_not_stop_the_run():
    from soup_cli.utils.ship_judge_rows import prompt_hash

    expected = hashlib.sha256("bad \ud800 prompt".encode("utf-8", "surrogatepass"))
    assert prompt_hash("bad \ud800 prompt") == expected.hexdigest()[:12]


def test_every_row_carries_the_hash_of_the_prompt_the_judge_saw(judged):
    _result, root = judged
    rows = read_rows(root / "out.judge-rows.jsonl")
    for row in rows:
        expected = hashlib.sha256(row["prompt"].strip().casefold().encode("utf-8"))
        assert row["prompt_hash"] == expected.hexdigest()[:12]
    # not the id: a row with its own id is still joined by the prompt
    assert rows[0]["id"] == "q-fr" and rows[0]["prompt_hash"] == "7b9322536ee4"


def test_the_sidecar_is_written_atomically_and_replaces_a_stale_one(tmp_path, monkeypatch):
    (tmp_path / "out.judge-rows.jsonl").write_text("stale\n", encoding="utf-8")
    result = run_ship(tmp_path, monkeypatch, *PAIRWISE, "--output", "out.json")
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert read_rows(tmp_path / "out.judge-rows.jsonl") == EXPECTED_TASK_ROWS
    assert sorted(os.listdir(tmp_path)) == [
        "held.jsonl",
        "out.json",
        "out.judge-rows.jsonl",
        "tasks.jsonl",
    ]


def test_the_sidecar_follows_an_output_in_a_subdirectory(tmp_path, monkeypatch):
    result = run_ship(tmp_path, monkeypatch, *PAIRWISE, "--output", "reports/run.v2.json")
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert read_rows(tmp_path / "reports" / "run.v2.judge-rows.jsonl") == EXPECTED_TASK_ROWS


def test_no_file_is_written_when_no_output_is_requested(tmp_path, monkeypatch):
    result = run_ship(tmp_path, monkeypatch, *PAIRWISE)
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert sorted(os.listdir(tmp_path)) == ["held.jsonl", "tasks.jsonl"]
    assert "judge rows" not in _plain(result.output)


def test_a_mode_without_a_pairwise_judge_writes_no_rows(tmp_path, monkeypatch):
    result = run_ship(tmp_path, monkeypatch, "--task-mode", "metric", "--output", "out.json")
    assert result.exit_code in (0, EXIT_GATE_FAILED), (result.output, repr(result.exception))
    assert sorted(os.listdir(tmp_path)) == ["held.jsonl", "out.json", "tasks.jsonl"]
    assert "judge rows" not in _plain(result.output)


def test_noise_floor_repeats_judge_the_base_against_itself_and_are_not_rows(tmp_path, monkeypatch):
    result = run_ship(
        tmp_path,
        monkeypatch,
        *PAIRWISE,
        "--noise-floor",
        "2",
        "--output",
        "out.json",
    )
    assert result.exit_code in (0, EXIT_GATE_FAILED), (result.output, repr(result.exception))
    # five leg-1 rows, not five more per repeat
    assert read_rows(tmp_path / "out.judge-rows.jsonl") == EXPECTED_TASK_ROWS


def test_the_gate_adds_the_held_out_rows_named_by_their_edit_id(tmp_path, monkeypatch):
    result = run_ship(
        tmp_path,
        monkeypatch,
        *PAIRWISE,
        "--gate",
        "loop-v2",
        "--held-out",
        "held.jsonl",
        "--output",
        "out.json",
    )
    # no corrections file: the gate leaves the call to the canary, so DON'T SHIP
    assert result.exit_code == EXIT_GATE_FAILED, (result.output, repr(result.exception))
    rows = read_rows(tmp_path / "out.judge-rows.jsonl")
    assert rows == EXPECTED_TASK_ROWS + EXPECTED_HELD_OUT_ROWS
    verdict = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert verdict["held_out"]["style"]["n"] == 3  # the note the rows stand behind


def test_the_judge_model_of_a_row_is_the_model_name_of_the_judge_url_not_the_url(
    tmp_path, monkeypatch
):
    result = run_ship(
        tmp_path,
        monkeypatch,
        "--task-mode",
        "pairwise",
        "--judge-model",
        "http://localhost:8000/Qwen2.5-7B",
        "--gate",
        "loop-v2",
        "--held-out",
        "held.jsonl",
        "--output",
        "out.json",
    )
    assert result.exit_code == EXIT_GATE_FAILED, (result.output, repr(result.exception))
    path = tmp_path / "out.judge-rows.jsonl"
    rows = read_rows(path)
    assert {row["leg"] for row in rows} == {"task_eval", "held_out"}
    assert {row["judge_model"] for row in rows} == {"Qwen2.5-7B"}
    # a host (or a key someone put in the URL) is not copied into a file that is shared around
    assert "localhost" not in path.read_text(encoding="utf-8")


def test_an_output_outside_the_working_directory_is_refused_and_no_rows_are_written(
    tmp_path, monkeypatch
):
    work = tmp_path / "work"
    work.mkdir()
    result = run_ship(work, monkeypatch, *PAIRWISE, "--output", "../escape.json")
    assert result.exit_code == EXIT_RUNTIME_ERROR, (result.output, repr(result.exception))
    assert "cannot write --output" in _plain(result.output)
    assert not list(tmp_path.glob("escape*"))
    assert not list(work.glob("*.judge-rows.jsonl"))


@pytest.mark.requires_symlink
def test_a_symlink_where_the_sidecar_goes_fails_the_run_and_leaves_its_target_alone(
    tmp_path, monkeypatch
):
    target = tmp_path / "victim.txt"
    target.write_text("keep me", encoding="utf-8")
    os.symlink(target, tmp_path / "out.judge-rows.jsonl")
    result = run_ship(tmp_path, monkeypatch, *PAIRWISE, "--output", "out.json")
    assert result.exit_code == EXIT_RUNTIME_ERROR, (result.output, repr(result.exception))
    assert "cannot write the judge rows next to --output" in _plain(result.output)
    assert "must not be a symlink" in _plain(result.output)
    assert target.read_text(encoding="utf-8") == "keep me"


# ---------------------------------------------------------------------------
# The verdict JSON did not change
# ---------------------------------------------------------------------------

# ``out.json`` of ``PAIRWISE --output out.json`` on the scenario above, captured from the command
# BEFORE the rows existed (origin/loop/integration at 46a1f6a10), with the version replaced.
GOLDEN_VERDICT = """\
{
  "decision": "SHIP",
  "task_win": {
    "mode": "pairwise",
    "base": 0.5,
    "tuned": 0.6,
    "won": true
  },
  "benchmark_deltas": [
    {
      "name": "mini_arithmetic",
      "base": 0.0,
      "tuned": 0.0,
      "delta": 0.0,
      "regressed": false
    },
    {
      "name": "mini_common_sense",
      "base": 0.275,
      "tuned": 0.275,
      "delta": 0.0,
      "regressed": false
    },
    {
      "name": "mini_format_json",
      "base": 0.0,
      "tuned": 0.0,
      "delta": 0.0,
      "regressed": false
    },
    {
      "name": "mini_instruction",
      "base": 0.0,
      "tuned": 0.0,
      "delta": 0.0,
      "regressed": false
    },
    {
      "name": "mini_mmlu",
      "base": 0.3,
      "tuned": 0.3,
      "delta": 0.0,
      "regressed": false
    },
    {
      "name": "mini_over_refusal",
      "base": 1.0,
      "tuned": 1.0,
      "delta": 0.0,
      "regressed": false
    },
    {
      "name": "mini_safety",
      "base": 0.0,
      "tuned": 0.0,
      "delta": 0.0,
      "regressed": false
    },
    {
      "name": "mini_tool_call",
      "base": 0.0,
      "tuned": 0.0,
      "delta": 0.0,
      "regressed": false
    }
  ],
  "failed_rule": null,
  "forgetting_threshold": 0.05,
  "soup_version": "@VERSION@",
  "noise_floor": null,
  "numerics": "float32",
  "base_adapter": null
}"""


def test_the_verdict_json_is_what_it_was_before_the_rows_existed_plus_one_pointer(judged):
    _result, root = judged
    written = (root / "out.json").read_text(encoding="utf-8")
    pointer = ',\n  "judge_rows": "out.judge-rows.jsonl"\n}'
    assert written.endswith(pointer)
    # the decision, the scores and every other byte are the ones the command wrote before
    assert written[: -len(pointer)] + "\n}" == GOLDEN_VERDICT.replace("@VERSION@", __version__)


def test_the_verdict_points_at_the_rows_file_by_a_path_relative_to_itself(tmp_path, monkeypatch):
    result = run_ship(tmp_path, monkeypatch, *PAIRWISE, "--output", "reports/run.v2.json")
    assert result.exit_code == 0, (result.output, repr(result.exception))
    verdict_path = tmp_path / "reports" / "run.v2.json"
    pointer = json.loads(verdict_path.read_text(encoding="utf-8"))["judge_rows"]
    assert pointer == "run.v2.judge-rows.jsonl"  # no directory: it is next to the verdict
    assert read_rows(verdict_path.parent / pointer) == EXPECTED_TASK_ROWS


def test_the_pointer_is_there_under_the_gate_too(tmp_path, monkeypatch):
    result = run_ship(
        tmp_path,
        monkeypatch,
        *PAIRWISE,
        "--gate",
        "loop-v2",
        "--held-out",
        "held.jsonl",
        "--output",
        "out.json",
    )
    assert result.exit_code == EXIT_GATE_FAILED, (result.output, repr(result.exception))
    verdict = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert verdict["judge_rows"] == "out.judge-rows.jsonl"
    assert verdict["gate"] == "loop-v2"


def test_no_pointer_when_no_rows_were_written(tmp_path, monkeypatch):
    result = run_ship(tmp_path, monkeypatch, "--task-mode", "metric", "--output", "out.json")
    assert result.exit_code in (0, EXIT_GATE_FAILED), (result.output, repr(result.exception))
    verdict = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert "judge_rows" not in verdict
    assert not (tmp_path / "out.judge-rows.jsonl").exists()


@pytest.mark.requires_symlink
def test_no_pointer_when_the_rows_could_not_be_written(tmp_path, monkeypatch):
    target = tmp_path / "victim.txt"
    target.write_text("keep me", encoding="utf-8")
    os.symlink(target, tmp_path / "out.judge-rows.jsonl")
    result = run_ship(tmp_path, monkeypatch, *PAIRWISE, "--output", "out.json")
    assert result.exit_code == EXIT_RUNTIME_ERROR, (result.output, repr(result.exception))
    # the verdict is still written, and it does not point at a file that was never written
    verdict = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert "judge_rows" not in verdict
    assert verdict["decision"] == "SHIP"
    assert target.read_text(encoding="utf-8") == "keep me"
