"""``soup ship --task-mode pairwise`` keeps the judge's per-row verdicts beside ``--output``.

Without a model, a GPU or a network: ``live_eval.make_generator`` is a recorder and the judge is
the real ``JudgeEvaluator`` with only its HTTP call scripted (so the real prompt, the real reply
parser and the real two-order combination all run). Four layers:

* ``pairwise_compare_detailed`` / ``pairwise_winrate(on_row=...)`` on a fake ``compare_pair``;
* the row ids, the JSONL text, the sidecar path and the write guards, as pure functions;
* the CLI: the sidecar's rows, both orders, the combined result, a tie and an unreadable reply,
  the held-out rows under ``--gate loop-v2``, no file without ``--output``, no noise-floor rows;
* the verdict JSON, byte for byte against what the command wrote before the rows existed.

The new names are imported inside the tests that use them, so the scenario helpers at the top can
also be run against a tree that does not have them (that is how ``GOLDEN_VERDICT`` was captured).
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
# "base" / "tuned", or "garbage" (a reply that is not an answer). ``base_first`` is the order in
# which the base answer is shown as A.
SCRIPT: Dict[str, Dict[str, str]] = {
    "capital of France?": {"base_first": "tuned", "tuned_first": "tuned"},
    "2+2?": {"base_first": "base", "tuned_first": "base"},
    "unlabelled question": {"base_first": "tuned", "tuned_first": "tuned"},
    # Picks position A in both orders: base when base is first, tuned when tuned is first.
    "order sensitive": {"base_first": "base", "tuned_first": "tuned"},
    "garbled judge": {"base_first": "tuned", "tuned_first": "garbage"},
    "held one": {"base_first": "tuned", "tuned_first": "tuned"},
    "held two": {"base_first": "base", "tuned_first": "base"},
    "held three": {"base_first": "garbage", "tuned_first": "tuned"},
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
        return json.dumps({"winner": "A" if picked == first else "B"})


def fake_make_generator(model_id, **kwargs):
    loaded = kwargs.get("loaded")
    label = loaded[0] if loaded is not None else model_id

    def generate(prompt: str) -> str:
        return f"{label}: {prompt}" if prompt in ANSWERED else "B"

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


def plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def read_rows(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


EXPECTED_TASK_ROWS = [
    {
        "schema_version": 1,
        "leg": "task_eval",
        "index": 0,
        "id": "q-fr",
        "id_source": "id",
        "base_first": "tuned",
        "tuned_first": "tuned",
        "result": "tuned",
        "score": 1.0,
    },
    {
        "schema_version": 1,
        "leg": "task_eval",
        "index": 1,
        "id": "e-two",
        "id_source": "edit_id",
        "base_first": "base",
        "tuned_first": "base",
        "result": "base",
        "score": 0.0,
    },
    {
        "schema_version": 1,
        "leg": "task_eval",
        "index": 2,
        "id": digest("unlabelled question"),
        "id_source": "prompt_sha256",
        "base_first": "tuned",
        "tuned_first": "tuned",
        "result": "tuned",
        "score": 1.0,
    },
    # A judge that always says "the first one": the two orders disagree, so a tie.
    {
        "schema_version": 1,
        "leg": "task_eval",
        "index": 3,
        "id": "q-order",
        "id_source": "id",
        "base_first": "base",
        "tuned_first": "tuned",
        "result": "tie",
        "score": 0.5,
    },
    # A reply that is not an answer in one order: no pick there, so a tie.
    {
        "schema_version": 1,
        "leg": "task_eval",
        "index": 4,
        "id": "q-garbled",
        "id_source": "id",
        "base_first": "tuned",
        "tuned_first": "none",
        "result": "tie",
        "score": 0.5,
    },
]
EXPECTED_HELD_OUT_ROWS = [
    {
        "schema_version": 1,
        "leg": "held_out",
        "index": 0,
        "id": "h-1",
        "id_source": "edit_id",
        "base_first": "tuned",
        "tuned_first": "tuned",
        "result": "tuned",
        "score": 1.0,
    },
    {
        "schema_version": 1,
        "leg": "held_out",
        "index": 1,
        "id": "h-2",
        "id_source": "edit_id",
        "base_first": "base",
        "tuned_first": "base",
        "result": "base",
        "score": 0.0,
    },
    {
        "schema_version": 1,
        "leg": "held_out",
        "index": 2,
        "id": "h-3",
        "id_source": "edit_id",
        "base_first": "none",
        "tuned_first": "tuned",
        "result": "tie",
        "score": 0.5,
    },
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


def test_every_row_is_one_json_line_whatever_the_id_holds():
    from soup_cli.eval.judge import PairwiseOutcome
    from soup_cli.utils.ship_judge_rows import JudgeRowRecorder, RowId

    recorder = JudgeRowRecorder()
    outcome = PairwiseOutcome(first=1, swapped=0, second=1, combined=1)
    nasty = ["line\nbreak", "tab\there", "naïve ✓", "lone \ud800 surrogate", '"quoted"']
    for index, name in enumerate(nasty):
        recorder.add("task_eval", index, RowId(name, "id"), outcome)
    text = recorder.to_jsonl()
    lines = text.split("\n")
    assert lines[-1] == "" and len(lines) == len(nasty) + 1
    assert [json.loads(line)["id"] for line in lines[:-1]] == nasty
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
        )
    assert [(row["result"], row["score"]) for row in recorder.rows] == [
        ("tuned", 1.0),
        ("tie", 0.5),
        ("base", 0.0),
    ]
    # judged in one order only: the second order is null, not "none"
    assert [row["tuned_first"] for row in recorder.rows] == [None, None, None]


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
    assert row_sink(pairs) is None
    with recording(JudgeRowRecorder(), "task_eval"):
        assert row_sink(pairs) is not None
    assert row_sink(pairs) is None
    with pytest.raises(RuntimeError):
        with recording(JudgeRowRecorder(), "task_eval"):
            raise RuntimeError("boom")
    assert row_sink(pairs) is None
    with recording(None, "task_eval"):
        assert row_sink(pairs) is None


def test_a_scope_names_its_rows_by_position_and_a_sink_records_into_its_recorder():
    from soup_cli.eval.judge import PairwiseOutcome
    from soup_cli.utils.ship_judge_rows import JudgeRowRecorder, RowId, recording, row_sink

    recorder = JudgeRowRecorder()
    ids = [RowId("first", "edit_id"), RowId("second", "edit_id")]
    pairs = [("p0", "a", "b"), ("p1", "a", "b"), ("p2", "a", "b")]
    with recording(recorder, "held_out", ids):
        sink = row_sink(pairs)
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
    text = plain(result.output)
    assert "Wrote judge rows" in text
    assert "(5 rows)" in text


def test_the_scores_of_the_rows_are_the_win_rate_in_the_verdict(judged):
    _result, root = judged
    rows = read_rows(root / "out.judge-rows.jsonl")
    verdict = json.loads((root / "out.json").read_text(encoding="utf-8"))
    assert verdict["task_win"] == {"mode": "pairwise", "base": 0.5, "tuned": 0.6, "won": True}
    assert sum(row["score"] for row in rows) / len(rows) == verdict["task_win"]["tuned"]


def test_the_sidecar_carries_no_prompt_or_answer_text(judged):
    _result, root = judged
    text = (root / "out.judge-rows.jsonl").read_text(encoding="utf-8")
    for row in TASKS:
        assert row["prompt"] not in text
    assert "Paris" not in text  # an expected answer
    assert "base: " not in text and "tuned: " not in text


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
    assert "judge rows" not in plain(result.output)


def test_a_mode_without_a_pairwise_judge_writes_no_rows(tmp_path, monkeypatch):
    result = run_ship(tmp_path, monkeypatch, "--task-mode", "metric", "--output", "out.json")
    assert result.exit_code in (0, EXIT_GATE_FAILED), (result.output, repr(result.exception))
    assert sorted(os.listdir(tmp_path)) == ["held.jsonl", "out.json", "tasks.jsonl"]
    assert "judge rows" not in plain(result.output)


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


def test_an_output_outside_the_working_directory_is_refused_and_no_rows_are_written(
    tmp_path, monkeypatch
):
    work = tmp_path / "work"
    work.mkdir()
    result = run_ship(work, monkeypatch, *PAIRWISE, "--output", "../escape.json")
    assert result.exit_code == EXIT_RUNTIME_ERROR, (result.output, repr(result.exception))
    assert "cannot write --output" in plain(result.output)
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
    assert "cannot write the judge rows next to --output" in plain(result.output)
    assert "must not be a symlink" in plain(result.output)
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


def test_the_verdict_json_is_byte_for_byte_what_it_was_before_the_rows_existed(judged):
    _result, root = judged
    written = (root / "out.json").read_text(encoding="utf-8")
    assert written == GOLDEN_VERDICT.replace("@VERSION@", __version__)
