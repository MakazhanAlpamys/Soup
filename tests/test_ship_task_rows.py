"""`soup ship` writes one line per judged task row, so a verdict can be audited row by row.

With ``--output verdict.json`` (and only then: ``--emit-evidence`` alone writes nothing new) a
live ``--task-mode metric`` or ``judge_score`` run also writes ``verdict.task-rows.jsonl`` next to
it: one JSON object per task with both answers, both scores, whether each answer was cut off at
the token cap, whether a generation failed, and which of better / worse / same the row is. The
mean of each score column, computed as ``row_mean`` says, is the reported ``task_win.base`` /
``task_win.tuned``: EXACTLY (``==``) for ``metric``, within ``JUDGE_MEAN_TOLERANCE`` (1e-12) for
``judge_score``, whose reported number stays the evaluator's own and is never rebuilt from the
rows. The file is checked against the verdict, never the other way round.

Everything runs on CPU with no model: the pure functions take ``LegScore`` objects, the CLI tests
patch ``ship_cmd._resolve_generators`` with scripted generators of both shapes (a scalar callable,
and one with ``.batch`` returning ``GeneratedText``).
"""

from __future__ import annotations

import json
import math
import os
import re
from fractions import Fraction
from pathlib import Path
from typing import Optional

import pytest

from soup_cli.commands import ship as ship_cmd
from soup_cli.eval.custom import EvalTask
from soup_cli.utils import ship_task_rows
from soup_cli.utils.live_eval import GeneratedText
from soup_cli.utils.ship_task_rows import (
    JUDGE_MEAN_TOLERANCE,
    MAX_FILE_BYTES,
    MAX_LINE_BYTES,
    MAX_TEXT_CHARS,
    ROW_KEYS,
    TaskRow,
    build_task_rows,
    load_task_rows,
    load_task_rows_for,
    prompt_hash,
    row_mean,
    rows_to_jsonl,
    task_rows_path,
    write_task_rows,
)
from soup_cli.utils.ship_verdict import (
    EVIDENCE_SCHEMA_FIELDS,
    MAX_TASK_ROWS,
    TASK_ROWS_SUFFIX,
    LegScore,
    compare_rows,
    row_change,
    task_rows_pointer_error,
    verdict_from_evidence,
    verdict_to_evidence,
)
from tests.cli_runner import CliRunner
from tests.conftest import strip_ansi

runner = CliRunner()


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


# -- the mean: one definition, plain left-to-right ------------------------------------------------


def test_row_mean_is_a_plain_left_to_right_sum_divided_by_n():
    # Python 3.12 made ``sum()`` of floats compensated (it gives 1.0 here, 3.11 gives
    # 0.9999999999999999) and ``math.fsum`` is exactly rounded (1.0): a consumer that adds the
    # column up the obvious way in any language gets the plain left-to-right result
    assert row_mean([0.1] * 10) == 0.09999999999999999
    assert math.fsum([0.1] * 10) / 10 == 0.1  # the two definitions really differ
    assert row_mean([0.25, 0.5]) == 0.375


@pytest.mark.parametrize("bad", [[], [1.0, float("nan")], [float("inf")], [True], ["1"]])
def test_row_mean_refuses_what_is_not_a_mean_of_finite_numbers(bad):
    with pytest.raises((ValueError, TypeError)):
        row_mean(bad)


def test_row_mean_of_zero_one_scores_is_exactly_the_accuracy_division():
    # the metric leg reports ``correct / total`` (int / int); the loop mean of 1.0 / 0.0 is the
    # same correctly rounded division, for every count the task bound allows to matter
    for total in (1, 3, 7, 10, 64, 999, 10_000):
        for correct in {0, 1, total // 3, total // 2, total - 1, total}:
            scores = [1.0] * correct + [0.0] * (total - correct)
            assert row_mean(scores) == correct / total


# -- prompt_hash: the loop's prompt key -----------------------------------------------------------


def test_prompt_hash_is_the_pinned_twelve_hex_key_of_the_loop_holdout_split():
    assert prompt_hash("  What is 2+2?  ") == "61a3385003d9"


def test_prompt_hash_trims_and_case_folds_and_survives_a_lone_surrogate():
    assert prompt_hash("What is 2+2?") == prompt_hash("\n  WHAT IS 2+2?\t")
    assert prompt_hash("STRASSE") == prompt_hash("straße")  # casefold, not lower
    assert re.fullmatch(r"[0-9a-f]{12}", prompt_hash("a\ud800b"))


# -- the file name -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("anchor", "expected"),
    [
        ("verdict.json", "verdict.task-rows.jsonl"),
        ("out/verdict.json", "out/verdict.task-rows.jsonl"),
        ("a.b.json", "a.b.task-rows.jsonl"),
        ("verdict", "verdict.task-rows.jsonl"),
        (".json", ".json.task-rows.jsonl"),
        ("my dir/out file.json", "my dir/out file.task-rows.jsonl"),
        ("out[1].json", "out[1].task-rows.jsonl"),
        ("dir [x]/o.json", "dir [x]/o.task-rows.jsonl"),
        ("a/../o.json", "a/../o.task-rows.jsonl"),
    ],
)
def test_the_rows_file_name_is_the_stem_of_the_anchor_plus_the_suffix(anchor, expected):
    path = task_rows_path(anchor)
    assert path.replace("\\", "/") == expected
    assert TASK_ROWS_SUFFIX == ".task-rows.jsonl"
    # in the very directory the user named: the same text up to the last component
    assert os.path.dirname(path) == os.path.dirname(anchor)
    assert os.path.dirname(os.path.realpath(path)) == os.path.dirname(os.path.realpath(anchor))


@pytest.mark.parametrize(
    ("anchor", "message"),
    [
        ("", "must be a non-empty path"),
        ("out\x00.json", "must not contain null bytes"),
        ("o\n.json", "has a control character"),
        ("o\x1b[2J.json", "has a control character"),
        ("x" * 300 + ".json", "is too long"),
        ("\\\\host\\share\\out.json", "network share or a device"),
        ("//host/share/out.json", "network share or a device"),
    ],
)
def test_an_anchor_that_cannot_carry_the_rows_file_is_refused_with_a_named_reason(
    anchor, message
):
    with pytest.raises(ValueError, match=re.escape(message)):
        task_rows_path(anchor)


@pytest.mark.parametrize(("control", "escaped"), [("\x7f", "\\x7f"), ("\x85", "\\x85")])
def test_del_and_c1_in_the_derived_file_name_are_named_safely(control, escaped):
    with pytest.raises(ValueError) as exc_info:
        task_rows_path(f"verdict{control}.json")

    message = _plain(str(exc_info.value))
    assert message == (
        f"the task-rows file name for 'verdict{escaped}.json' has a control character; "
        "use a plain --output name"
    )
    assert control not in message


@pytest.mark.skipif(os.name == "nt", reason="a backslash is a path separator on Windows")
def test_on_posix_a_backslash_in_the_file_name_cannot_be_named_by_the_pointer():
    with pytest.raises(ValueError, match="path separator.*use a plain --output name"):
        task_rows_path("sub\\out.json")


@pytest.mark.parametrize(
    ("value", "ok"),
    [
        ("verdict.task-rows.jsonl", True),
        ("out file[1].task-rows.jsonl", True),
        ("x" * 239 + ".task-rows.jsonl", True),  # 255 characters
        ("x" * 240 + ".task-rows.jsonl", False),  # 256
        (".task-rows.jsonl", False),  # no stem
        ("verdict.jsonl", False),  # not the suffix
        ("../verdict.task-rows.jsonl", False),
        ("sub/verdict.task-rows.jsonl", False),
        ("sub\\verdict.task-rows.jsonl", False),
        ("/abs/verdict.task-rows.jsonl", False),
        ("C:\\x\\verdict.task-rows.jsonl", False),
        ("C:/x/verdict.task-rows.jsonl", False),
        ("v\x00.task-rows.jsonl", False),
        ("v\x1b.task-rows.jsonl", False),
        ("v\x85.task-rows.jsonl", False),
        ("", False),
        (None, False),
        (True, False),
        (3, False),
        (["verdict.task-rows.jsonl"], False),
    ],
)
def test_the_pointer_is_a_bare_file_name_and_nothing_else(value, ok):
    error = task_rows_pointer_error(value)
    assert (error is None) == ok, error


# -- one row, built from the legs ---------------------------------------------------------------


def _task(index: int, **extra) -> EvalTask:
    return EvalTask(
        prompt=f"q{index}", expected=f"a{index}", category="default", scoring="exact", **extra
    )


def _leg(
    scores,
    *,
    prefix="b",
    truncated=None,
    failed=None,
    value=None,
    tasks=None,
    answers=None,
):
    count = len(scores)
    return LegScore(
        row_mean(scores) if value is None else value,
        row_scores=scores,
        truncated=[False] * count if truncated is None else truncated,
        failures=sum(failed or ()),
        answers=[f"{prefix}{index}" for index in range(count)] if answers is None else answers,
        failed_rows=[False] * count if failed is None else failed,
        tasks=[_task(index) for index in range(count)] if tasks is None else tasks,
    )


def test_rows_carry_the_inputs_both_answers_both_scores_flags_and_the_class():
    base = _leg([1.0, 0.0, 1.0, 0.0], prefix="b", truncated=[False, True, None, False])
    tuned = _leg([1.0, 1.0, 0.0, 0.0], prefix="t", truncated=[False, False, True, None])
    rows = build_task_rows(base, tuned, scoring_applies=True)
    assert [row.index for row in rows] == [0, 1, 2, 3]
    assert [row.change for row in rows] == ["same", "better", "worse", "same"]
    first = rows[1]
    assert (first.prompt, first.expected, first.category, first.scoring) == (
        "q1", "a1", "default", "exact",
    )
    assert (first.base_answer, first.tuned_answer) == ("b1", "t1")
    assert (first.base_score, first.tuned_score) == (0.0, 1.0)
    assert (first.base_truncated, first.tuned_truncated) == (True, False)
    assert (rows[2].base_truncated, rows[2].tuned_truncated) == (None, True)  # null is not False
    assert first.prompt_hash == prompt_hash("q1")


def test_the_classes_of_the_rows_are_the_counts_of_the_verdict():
    base_scores = [1.0, 1.0, 0.5, 0.25, 0.0, 0.7]
    tuned_scores = [0.0, 1.0, 0.5, 0.75, 0.0, 0.7 + 1e-12]
    rows = build_task_rows(_leg(base_scores), _leg(tuned_scores, prefix="t"), scoring_applies=True)
    counts = compare_rows(base_scores, tuned_scores)
    got = [row.change for row in rows]
    assert (got.count("better"), got.count("worse"), got.count("same")) == (
        counts.better, counts.worse, counts.same,
    )
    assert counts.same == 4  # includes the 1e-12 float noise
    assert row_change(0.5, 0.5 + 1e-8) == "better" and row_change(0.5, 0.5 - 1e-8) == "worse"
    assert row_change(0.5, 0.5 + 5e-10) == "same"


def test_a_judge_row_has_no_scoring_rule():
    base, tuned = _leg([0.5, 0.25]), _leg([0.75, 0.25], prefix="t")
    rows = build_task_rows(base, tuned, scoring_applies=False)
    assert [row.scoring for row in rows] == [None, None]
    assert [row.expected for row in rows] == ["a0", "a1"]


def test_a_failed_generation_is_flagged_and_has_an_empty_answer_and_unknown_truncation():
    base = _leg([0.0, 1.0], failed=[True, False], truncated=[None, False], answers=["", "b1"])
    tuned = _leg([1.0, 1.0], prefix="t")
    rows = build_task_rows(base, tuned, scoring_applies=True)
    assert (rows[0].base_failed, rows[0].tuned_failed) == (True, False)
    assert (rows[1].base_failed, rows[1].tuned_failed) == (False, False)
    assert rows[0].base_answer == "" and rows[0].base_truncated is None


def test_the_row_mean_must_be_the_reported_number_exactly():
    base = _leg([0.1, 0.2, 0.3], value=0.2)  # 0.1 + 0.2 + 0.3 averaged is not 0.2 in binary
    with pytest.raises(ValueError, match="the base row scores average"):
        build_task_rows(base, _leg([0.1, 0.2, 0.3], prefix="t"), scoring_applies=True)
    tuned = _leg([0.1, 0.2, 0.3], prefix="t", value=0.25)
    with pytest.raises(ValueError, match="the tuned row scores average"):
        build_task_rows(_leg([0.1, 0.2, 0.3]), tuned, scoring_applies=True)


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: (0.5, _leg([0.5])), "both legs need LegScore"),
        (lambda: (LegScore(0.5, row_scores=[0.5]), _leg([0.5])), "carries no answers"),
        (lambda: (_leg([0.5, 0.5]), _leg([0.5])), "same number of rows"),
        (
            lambda: (_leg([0.5], tasks=[_task(0)]), _leg([0.5], tasks=[_task(1)])),
            "different tasks",
        ),
        (lambda: (_leg([float("nan")], value=0.5), _leg([0.5])), "must be finite"),
        (lambda: (_leg([True], value=1.0), _leg([1.0])), "must not be bool"),
        (lambda: (_leg([1.5], value=1.5), _leg([0.5])), "must be in [0, 1]"),
        (lambda: (_leg([-0.1], value=-0.1), _leg([0.5])), "must be in [0, 1]"),
    ],
)
def test_legs_that_cannot_be_audited_are_refused_not_half_written(make, message):
    base, tuned = make()
    with pytest.raises((ValueError, TypeError), match=re.escape(message)):
        build_task_rows(base, tuned, scoring_applies=True)


def test_there_are_at_most_as_many_rows_as_a_task_file_can_hold():
    from soup_cli.eval.custom import MAX_EVAL_TASKS

    assert MAX_TASK_ROWS == MAX_EVAL_TASKS == 10_000
    too_many = MAX_TASK_ROWS + 1
    base = _leg([0.0] * too_many)
    with pytest.raises(ValueError, match="at most 10000 rows"):
        build_task_rows(base, _leg([0.0] * too_many, prefix="t"), scoring_applies=True)


# -- the line -----------------------------------------------------------------------------------


def _row(**changes) -> TaskRow:
    base = dict(
        index=0, prompt_hash=prompt_hash("q0"), prompt="q0", expected="a0", category="default",
        scoring="exact", base_answer="b0", tuned_answer="t0", base_score=0.0, tuned_score=1.0,
        change="better", base_truncated=False, tuned_truncated=None, base_failed=False,
        tuned_failed=False, text_cut=False,
    )
    base.update(changes)
    return TaskRow(**base)


def test_a_line_has_exactly_these_keys_in_this_order():
    assert ROW_KEYS == (
        "schema_version", "leg", "index", "prompt_hash", "prompt", "expected", "category",
        "scoring", "base_answer", "tuned_answer", "base_score", "tuned_score", "change",
        "base_truncated", "tuned_truncated", "base_failed", "tuned_failed", "text_cut",
    )
    text = rows_to_jsonl([_row(), _row(index=1)])
    lines = text.split("\n")
    assert lines[-1] == "" and len(lines) == 3
    parsed = json.loads(lines[0])
    assert tuple(parsed) == ROW_KEYS
    assert parsed["schema_version"] == 1 and parsed["leg"] == "task"
    assert parsed["tuned_truncated"] is None and parsed["base_truncated"] is False


def test_one_example_line_is_compact_json_ascii_and_ends_with_lf():
    text = rows_to_jsonl([_row()])
    assert text == (
        '{"schema_version":1,"leg":"task","index":0,"prompt_hash":"%s","prompt":"q0",'
        '"expected":"a0","category":"default","scoring":"exact","base_answer":"b0",'
        '"tuned_answer":"t0","base_score":0.0,"tuned_score":1.0,"change":"better",'
        '"base_truncated":false,"tuned_truncated":null,"base_failed":false,'
        '"tuned_failed":false,"text_cut":false}\n' % prompt_hash("q0")
    )
    assert "\r" not in text


def test_text_is_ascii_escaped_so_one_row_is_always_one_line():
    hostile = "a\u2028b\u2029c\x1b[31m\x00\r\nd\u00e9\U0001f600\ud800"
    text = rows_to_jsonl([_row(prompt=hostile, base_answer=hostile)])
    text.encode("ascii")  # no byte above 0x7f, so no encoding error and no U+2028 splitter
    assert text.count("\n") == 1 and "\r" not in text and "\x1b" not in text and "\x00" not in text
    back = json.loads(text)
    assert back["prompt"] == hostile and back["base_answer"] == hostile


def test_scores_are_written_with_their_full_repr_and_read_back_to_the_same_float():
    tricky = [0.1 + 0.2, 1 / 3, 2 / 3, 0.175, 0.49687500000000007, 1e-7, 0.30000000000000004]
    for value in tricky:
        assert Fraction(value) != Fraction(str(value))  # not exactly the decimal it prints as
        text = rows_to_jsonl([_row(base_score=value, tuned_score=value, change="same")])
        parsed = json.loads(text)
        assert parsed["base_score"] == value and parsed["tuned_score"] == value
        assert repr(value) in text  # the shortest round-trip repr, never rounded


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_a_non_finite_score_is_never_written(bad):
    with pytest.raises(ValueError):
        rows_to_jsonl([_row(base_score=bad, change="same")])


def test_a_long_text_is_cut_and_says_so_while_the_hash_is_that_of_the_whole_prompt():
    prompt = "x" * (MAX_TEXT_CHARS + 5000)
    base = _leg([1.0], tasks=[EvalTask(prompt=prompt, expected="e" * 30_000, scoring="exact")])
    tuned = _leg([1.0], prefix="t", tasks=base.tasks, answers=["y" * 25_000])
    (row,) = build_task_rows(base, tuned, scoring_applies=True)
    assert len(row.prompt) == MAX_TEXT_CHARS and len(row.expected) == MAX_TEXT_CHARS
    assert len(row.tuned_answer) == MAX_TEXT_CHARS and row.text_cut is True
    assert row.prompt_hash == prompt_hash(prompt) != prompt_hash(row.prompt)
    short = build_task_rows(_leg([1.0]), _leg([1.0], prefix="t"), scoring_applies=True)[0]
    assert short.text_cut is False


def test_the_worst_case_line_still_fits_the_line_bound():
    astral = "\U0001f600" * MAX_TEXT_CHARS  # 12 bytes each once escaped
    row = _row(prompt=astral, expected=astral, base_answer=astral, tuned_answer=astral,
               text_cut=True)
    (line,) = rows_to_jsonl([row]).splitlines()
    assert len(line.encode("ascii")) < MAX_LINE_BYTES


def test_a_row_longer_than_the_line_bound_is_refused_by_the_writer():
    with pytest.raises(ValueError, match="longer than 1048576 bytes"):
        rows_to_jsonl([_row(prompt="x" * (MAX_LINE_BYTES + 1))])


def test_a_category_longer_than_the_cap_is_cut_and_flagged():
    long_category = "c" * 1000
    base = _leg([1.0], tasks=[EvalTask(prompt="q", expected="a", category=long_category)])
    tuned = _leg([1.0], prefix="t", tasks=base.tasks)
    (row,) = build_task_rows(base, tuned, scoring_applies=True)
    assert len(row.category) == 256 and row.text_cut is True


# -- the exact invariant, through the file, with numbers that are not binary fractions ----------


def test_the_mean_read_back_from_the_file_is_the_reported_number_exactly(tmp_path):
    base_scores = [0.1, 0.2, 0.3, 0.7, 1 / 3, 2 / 3, 0.175, 0.49687500000000007]
    tuned_scores = [0.9, 0.1 + 0.2, 0.123456789, 0.0, 1.0, 0.5, 0.25, 0.3]
    base = _leg(base_scores)
    tuned = _leg(tuned_scores, prefix="t")
    rows = build_task_rows(base, tuned, scoring_applies=True)
    path = tmp_path / "v.task-rows.jsonl"
    path.write_text(rows_to_jsonl(rows), encoding="ascii", newline="\n")
    loaded = load_task_rows(str(path), n=8, base=float(base), tuned=float(tuned))
    assert loaded.base_mean == float(base) and loaded.tuned_mean == float(tuned)
    # and by hand, the way a consumer adds a column up
    parsed = [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]
    total = 0.0
    for entry in parsed:
        total += entry["base_score"]
    assert total / len(parsed) == float(base)


# -- writing: where, atomically, and what it never overwrites -----------------------------------


def test_the_rows_are_written_next_to_the_anchor_and_replace_an_earlier_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = [_row(), _row(index=1, prompt="q1", prompt_hash=prompt_hash("q1"))]
    path = write_task_rows(rows, "sub/verdict.json")
    assert Path(path).as_posix() == "sub/verdict.task-rows.jsonl"
    first = Path(path).read_bytes()
    assert first == rows_to_jsonl(rows).encode("ascii")
    # an earlier file of this command is replaced by this command's own run
    write_task_rows([_row(index=0, base_answer="changed")], "sub/verdict.json")
    assert Path(path).read_bytes() != first
    assert Path(path).read_bytes().count(b"\n") == 1


def test_the_file_is_the_same_bytes_on_every_platform_and_every_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_task_rows([_row()], "v.json")
    first = Path("v.task-rows.jsonl").read_bytes()
    write_task_rows([_row()], "v.json")
    assert Path("v.task-rows.jsonl").read_bytes() == first
    assert b"\r" not in first  # LF, never CRLF (a text-mode write on Windows would add it)


def test_a_failed_replace_leaves_the_earlier_file_whole_and_no_temp_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_task_rows([_row(base_answer="first run")], "v.json")
    before = Path("v.task-rows.jsonl").read_bytes()

    def broken_replace(source, target):
        raise OSError("disk went away")

    with monkeypatch.context() as patch:
        patch.setattr("soup_cli.utils.paths.os.replace", broken_replace)
        with pytest.raises(OSError, match="disk went away"):
            write_task_rows([_row(base_answer="second run")], "v.json")
    assert Path("v.task-rows.jsonl").read_bytes() == before  # never half a file
    assert sorted(p.name for p in tmp_path.iterdir()) == ["v.task-rows.jsonl"]


def test_a_file_that_is_not_ours_is_left_alone_and_the_refusal_says_why(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    foreign = b'{"prompt": "my own eval set", "expected": "kept"}\n'
    Path("v.task-rows.jsonl").write_bytes(foreign)
    with pytest.raises(ValueError, match="exists and is not a task-rows file written by soup ship"):
        write_task_rows([_row()], "v.json")
    assert Path("v.task-rows.jsonl").read_bytes() == foreign


@pytest.mark.parametrize(
    "content",
    [b"not json at all\n", b'["a list"]\n', b'{"leg": "judge"}\n', b'{"leg": "task"}\n',
     b'{"leg": "task", "schema_version": true}\n', b"\xff\xfe\n"],
    ids=["not-json", "list", "judge-leg", "task-leg-only", "bool-version", "not-utf8"],
)
def test_only_a_line_that_looks_like_ours_is_replaced(tmp_path, monkeypatch, content):
    monkeypatch.chdir(tmp_path)
    Path("v.task-rows.jsonl").write_bytes(content)
    with pytest.raises(ValueError, match="not a task-rows file written by soup ship"):
        write_task_rows([_row()], "v.json")
    assert Path("v.task-rows.jsonl").read_bytes() == content


def test_an_empty_file_or_our_own_file_is_replaced(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("v.task-rows.jsonl").write_bytes(b"")
    write_task_rows([_row()], "v.json")
    assert Path("v.task-rows.jsonl").read_bytes().startswith(b'{"schema_version":1,"leg":"task"')
    write_task_rows([_row(index=0, base_answer="again")], "v.json")
    assert b"again" in Path("v.task-rows.jsonl").read_bytes()


def test_the_rows_never_leave_the_working_directory(tmp_path, monkeypatch):
    inside = tmp_path / "work"
    inside.mkdir()
    monkeypatch.chdir(inside)
    with pytest.raises(ValueError, match="must stay under cwd"):
        write_task_rows([_row()], "../escaped.json")
    assert not (tmp_path / "escaped.task-rows.jsonl").exists()
    with pytest.raises(ValueError, match="must stay under cwd"):
        write_task_rows([_row()], str(tmp_path / "abs" / "escaped.json"))
    assert not (tmp_path / "abs").exists()


@pytest.mark.requires_symlink
def test_the_rows_are_not_written_through_a_symlink(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "precious.txt"
    target.write_text("keep", encoding="utf-8")
    os.symlink(target, tmp_path / "v.task-rows.jsonl")
    with pytest.raises(ValueError, match="symlink"):
        write_task_rows([_row()], "v.json")
    assert target.read_text(encoding="utf-8") == "keep"


def test_the_writer_refuses_too_many_rows_and_a_file_over_the_byte_bound(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="at most 10000 rows"):
        rows_to_jsonl([_row(index=i) for i in range(MAX_TASK_ROWS + 1)])
    monkeypatch.setattr(ship_task_rows, "MAX_FILE_BYTES", 100)
    with pytest.raises(ValueError, match="more than 100 bytes"):
        rows_to_jsonl([_row(), _row(index=1)])
    assert not list(tmp_path.iterdir())
    assert MAX_FILE_BYTES == 256 * 1024 * 1024


# -- reading it back, strictly ------------------------------------------------------------------


def _good_lines(count: int = 3, **changes):
    rows = []
    for index in range(count):
        rows.append(_row(
            index=index, prompt=f"q{index}", prompt_hash=prompt_hash(f"q{index}"),
            expected=f"a{index}", **changes,
        ))
    return [json.loads(line) for line in rows_to_jsonl(rows).splitlines()]


def _write_lines(tmp_path, entries, *, raw: Optional[str] = None) -> str:
    path = tmp_path / "v.task-rows.jsonl"
    if raw is None:
        raw = "".join(json.dumps(entry, separators=(",", ":")) + "\n" for entry in entries)
    path.write_text(raw, encoding="utf-8", newline="\n")
    return str(path)


def test_a_good_file_loads_with_its_means_and_counts(tmp_path):
    entries = _good_lines(3)
    entries[1]["base_score"] = 1.0
    entries[1]["tuned_score"] = 0.0
    entries[1]["change"] = "worse"
    loaded = load_task_rows(
        _write_lines(tmp_path, entries), n=3, base=1 / 3, tuned=2 / 3,
        counts={"better": 2, "worse": 1, "same": 0},
    )
    assert [row["index"] for row in loaded.rows] == [0, 1, 2]
    assert (loaded.better, loaded.worse, loaded.same) == (2, 1, 0)
    assert loaded.base_mean == 1 / 3 and loaded.tuned_mean == 2 / 3


def test_the_file_may_list_its_rows_in_any_order_and_the_mean_is_taken_in_index_order(tmp_path):
    entries = _good_lines(4)
    for entry, score in zip(entries, (0.1, 0.2, 0.3, 0.7)):
        entry["base_score"] = score
        entry["tuned_score"] = score
        entry["change"] = "same"
    expected = row_mean([0.1, 0.2, 0.3, 0.7])
    shuffled = [entries[2], entries[0], entries[3], entries[1]]
    loaded = load_task_rows(_write_lines(tmp_path, shuffled), n=4, base=expected, tuned=expected)
    assert [row["index"] for row in loaded.rows] == [0, 1, 2, 3]


def _mutate(entries, line, key, value):
    entries = [dict(entry) for entry in entries]
    if value is _DELETE:
        del entries[line][key]
    else:
        entries[line][key] = value
    return entries


_DELETE = object()

REFUSALS = [
    ("not json", "task-rows line 2 is not valid JSON"),
    ("a list", "task-rows line 2 must be a JSON object"),
    ("blank", "task-rows line 2 is blank"),
    ("long", "task-rows line 2 is longer than"),
    ("nested", "task-rows line 2 is not valid JSON"),
    ("dup key", "task-rows line 2 repeats a key"),
    ("NaN token", "task-rows line 2 has a non-finite number (NaN or Infinity)"),
    ("Infinity token", "task-rows line 2 has a non-finite number (NaN or Infinity)"),
    ("1e999", "task-rows line 2 base_score must be finite"),
    ("unknown key", "task-rows line 2 has unsupported field(s): 'extra'"),
    ("missing key", "task-rows line 2 needs base_score"),
    ("version 2", "task-rows line 2 schema_version must be 1"),
    ("version true", "task-rows line 2 schema_version must be 1"),
    ("leg", 'task-rows line 2 leg must be "task"'),
    ("bool score", "task-rows line 2 base_score must be a number, not a boolean"),
    ("string score", "task-rows line 2 tuned_score must be a number"),
    ("score above 1", "task-rows line 2 base_score must be in [0, 1]"),
    ("score below 0", "task-rows line 2 tuned_score must be in [0, 1]"),
    ("bool index", "task-rows line 2 index must be an integer"),
    ("negative index", "task-rows line 2 index must be an integer in [0, 10000)"),
    ("float index", "task-rows line 2 index must be an integer"),
    ("duplicate id", "task-rows line 3 repeats row id 0"),
    ("index beyond count", "task-rows line 3 index 7 is outside [0, 3)"),
    ("bad change", "task-rows line 2 change must be one of better, worse, same"),
    ("wrong change", "task-rows line 2 change says worse but the scores say better"),
    ("bad flag", "task-rows line 2 base_truncated must be true, false or null"),
    ("null failed", "task-rows line 2 base_failed must be true or false"),
    ("non-string prompt", "task-rows line 2 prompt must be a string"),
    ("scoring number", "task-rows line 2 scoring must be a string or null"),
    ("bad hash", "task-rows line 2 prompt_hash must be 12 lowercase hex digits"),
    ("wrong hash", "task-rows line 2 prompt_hash does not match the prompt"),
    ("nested value", "task-rows line 2 prompt must be a string"),
]


def _refusal_file(tmp_path, name):
    entries = _good_lines(3)
    raw: Optional[str] = None
    if name == "not json":
        raw = json.dumps(entries[0]) + "\n{not json}\n" + json.dumps(entries[2]) + "\n"
    elif name == "a list":
        raw = json.dumps(entries[0]) + "\n[1, 2]\n" + json.dumps(entries[2]) + "\n"
    elif name == "blank":
        raw = json.dumps(entries[0]) + "\n\n" + json.dumps(entries[2]) + "\n"
    elif name == "long":
        big = _mutate(entries, 1, "prompt", "x" * (MAX_LINE_BYTES + 10))
        raw = "".join(json.dumps(entry) + "\n" for entry in big)
    elif name == "nested":
        raw = json.dumps(entries[0]) + "\n" + "[" * 200_000 + "\n" + json.dumps(entries[2]) + "\n"
    elif name == "dup key":
        line = json.dumps(entries[1]).replace('"index": 1', '"index": 1, "index": 1')
        raw = json.dumps(entries[0]) + "\n" + line + "\n" + json.dumps(entries[2]) + "\n"
    elif name in ("NaN token", "Infinity token", "1e999"):
        token = {"NaN token": "NaN", "Infinity token": "Infinity", "1e999": "1e999"}[name]
        line = json.dumps(entries[1]).replace('"base_score": 0.0', f'"base_score": {token}')
        assert token in line
        raw = json.dumps(entries[0]) + "\n" + line + "\n" + json.dumps(entries[2]) + "\n"
    elif name == "unknown key":
        entries = _mutate(entries, 1, "extra", 1)
    elif name == "missing key":
        entries = _mutate(entries, 1, "base_score", _DELETE)
    elif name == "version 2":
        entries = _mutate(entries, 1, "schema_version", 2)
    elif name == "version true":
        entries = _mutate(entries, 1, "schema_version", True)
    elif name == "leg":
        entries = _mutate(entries, 1, "leg", "judge")
    elif name == "bool score":
        entries = _mutate(entries, 1, "base_score", True)
    elif name == "string score":
        entries = _mutate(entries, 1, "tuned_score", "1.0")
    elif name == "score above 1":
        entries = _mutate(entries, 1, "base_score", 1.5)
    elif name == "score below 0":
        entries = _mutate(entries, 1, "tuned_score", -0.5)
    elif name == "bool index":
        entries = _mutate(entries, 1, "index", True)
    elif name == "negative index":
        entries = _mutate(entries, 1, "index", -1)
    elif name == "float index":
        entries = _mutate(entries, 1, "index", 1.0)
    elif name == "duplicate id":
        entries = _mutate(entries, 2, "index", 0)
    elif name == "index beyond count":
        entries = _mutate(entries, 2, "index", 7)
    elif name == "bad change":
        entries = _mutate(entries, 1, "change", "improved")
    elif name == "wrong change":
        entries = _mutate(entries, 1, "change", "worse")
    elif name == "bad flag":
        entries = _mutate(entries, 1, "base_truncated", "yes")
    elif name == "null failed":
        entries = _mutate(entries, 1, "base_failed", None)
    elif name == "non-string prompt":
        entries = _mutate(entries, 1, "prompt", 5)
    elif name == "scoring number":
        entries = _mutate(entries, 1, "scoring", 3)
    elif name == "bad hash":
        entries = _mutate(entries, 1, "prompt_hash", "ABC")
    elif name == "wrong hash":
        entries = _mutate(entries, 1, "prompt_hash", "0" * 12)
    elif name == "nested value":
        entries = _mutate(entries, 1, "prompt", {"a": ["b"]})
    return _write_lines(tmp_path, entries, raw=raw)


@pytest.mark.parametrize(("name", "message"), REFUSALS, ids=[name for name, _ in REFUSALS])
def test_the_loader_refuses_each_defect_with_a_message_that_names_it(tmp_path, name, message):
    path = _refusal_file(tmp_path, name)
    with pytest.raises(ValueError) as caught:
        load_task_rows(path)
    assert message in str(caught.value), str(caught.value)
    # a refusal never echoes what the file said
    assert "x" * 50 not in str(caught.value)


def test_the_count_must_be_the_evidence_n(tmp_path):
    path = _write_lines(tmp_path, _good_lines(3))
    with pytest.raises(ValueError, match="has 3 rows but the evidence says n = 4"):
        load_task_rows(path, n=4)
    load_task_rows(path, n=3)  # the same file, the right n


def test_the_mean_must_be_the_reported_number_exactly_not_nearly(tmp_path):
    entries = _good_lines(3)
    for entry, score in zip(entries, (0.1, 0.2, 0.3)):
        entry["base_score"] = score
        entry["tuned_score"] = score
        entry["change"] = "same"
    path = _write_lines(tmp_path, entries)
    exact = row_mean([0.1, 0.2, 0.3])
    assert exact == 0.20000000000000004 != 0.2
    load_task_rows(path, base=exact, tuned=exact)
    with pytest.raises(ValueError, match="the base_score mean .* is not the reported base"):
        load_task_rows(path, base=0.2)  # one rounding away is still not equal
    with pytest.raises(ValueError, match="the tuned_score mean .* is not the reported tuned"):
        load_task_rows(path, tuned=0.2)


def test_the_change_counts_must_be_the_evidence_rows_block(tmp_path):
    path = _write_lines(tmp_path, _good_lines(3))
    with pytest.raises(ValueError, match="better 3, worse 0, same 0, which is not the evidence"):
        load_task_rows(path, counts={"better": 1, "worse": 1, "same": 1})


def test_an_empty_file_and_a_missing_file_and_a_directory_are_refused(tmp_path):
    empty = tmp_path / "e.task-rows.jsonl"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="task-rows file is empty"):
        load_task_rows(str(empty))
    with pytest.raises(ValueError, match="task-rows file not found"):
        load_task_rows(str(tmp_path / "missing.task-rows.jsonl"))
    with pytest.raises(ValueError, match="is not a regular file"):
        load_task_rows(str(tmp_path))


def test_a_file_over_the_byte_bound_is_refused_before_it_is_read(tmp_path, monkeypatch):
    path = _write_lines(tmp_path, _good_lines(3))
    monkeypatch.setattr(ship_task_rows, "MAX_FILE_BYTES", 50)
    with pytest.raises(ValueError, match="task-rows file is larger than 50 bytes"):
        load_task_rows(path)


def test_a_file_with_more_rows_than_a_task_file_can_hold_is_refused(tmp_path, monkeypatch):
    path = _write_lines(tmp_path, _good_lines(3))
    monkeypatch.setattr(ship_task_rows, "MAX_TASK_ROWS", 2)
    with pytest.raises(ValueError, match="more than 2 rows"):
        load_task_rows(path)


def test_the_text_cut_flag_lets_a_cut_prompt_keep_the_hash_of_the_whole_one(tmp_path):
    entries = _good_lines(1)
    entries[0]["prompt"] = "short version"
    entries[0]["text_cut"] = True
    load_task_rows(_write_lines(tmp_path, entries))  # no hash check on a cut prompt
    entries[0]["text_cut"] = False
    with pytest.raises(ValueError, match="prompt_hash does not match the prompt"):
        load_task_rows(_write_lines(tmp_path, entries))


# -- finding the file from the verdict or the evidence ---------------------------------------


def _write_pair(tmp_path, *, pointer="v.task-rows.jsonl", key="task_win", n=3, base=None,
                tuned=None, rows_block=True, directory=None, mode="metric"):
    directory = directory or tmp_path
    entries = _good_lines(n)
    loaded = load_task_rows(_write_lines(tmp_path, entries))
    if Path(directory) != tmp_path:
        (tmp_path / "v.task-rows.jsonl").replace(Path(directory) / "v.task-rows.jsonl")
    leg = {"mode": mode, "base": loaded.base_mean if base is None else base,
           "tuned": loaded.tuned_mean if tuned is None else tuned}
    if rows_block:
        leg["rows"] = {"better": loaded.better, "worse": loaded.worse, "same": loaded.same,
                       "n": n}
    payload = {key: leg}
    if pointer is not None:
        payload["task_rows"] = pointer
    target = Path(directory) / "verdict.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    return str(target)


@pytest.mark.parametrize("key", ["task_win", "task"])
def test_the_file_is_found_through_the_pointer_of_a_verdict_or_of_evidence(tmp_path, key):
    loaded = load_task_rows_for(_write_pair(tmp_path, key=key))
    assert len(loaded.rows) == 3


def test_a_pointer_that_names_nothing_or_nothing_in_this_directory_is_refused(tmp_path):
    with pytest.raises(ValueError, match="names no task-rows file"):
        load_task_rows_for(_write_pair(tmp_path, pointer=None))
    sub = tmp_path / "sub"
    sub.mkdir()
    path = _write_pair(tmp_path, pointer="../v.task-rows.jsonl", directory=sub)
    with pytest.raises(ValueError, match="task_rows must be a bare file name"):
        load_task_rows_for(path)


@pytest.mark.requires_symlink
def test_a_pointer_to_a_link_that_leaves_the_directory_is_refused(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    inside = tmp_path / "inside"
    inside.mkdir()
    path = _write_pair(tmp_path, directory=outside)
    os.symlink(outside / "v.task-rows.jsonl", inside / "v.task-rows.jsonl")
    (inside / "verdict.json").write_bytes((outside / "verdict.json").read_bytes())
    with pytest.raises(ValueError, match="resolves outside the directory"):
        load_task_rows_for(str(inside / "verdict.json"))
    assert len(load_task_rows_for(path).rows) == 3  # the same file in its own directory is fine


def test_a_pointer_to_a_file_that_is_not_there_says_so(tmp_path):
    path = _write_pair(tmp_path, pointer="gone.task-rows.jsonl")
    with pytest.raises(ValueError, match="task-rows file not found next to the verdict"):
        load_task_rows_for(path)


def test_the_file_is_checked_against_the_numbers_of_the_json_that_names_it(tmp_path):
    with pytest.raises(ValueError, match="is not the reported base"):
        load_task_rows_for(_write_pair(tmp_path, base=0.5))
    with pytest.raises(ValueError, match="is not the reported tuned"):
        load_task_rows_for(_write_pair(tmp_path, tuned=0.5))


def test_the_row_count_is_checked_against_the_rows_block_when_the_json_has_one(tmp_path):
    path = _write_pair(tmp_path)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["task_win"]["rows"]["n"] = 4
    Path(path).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="has 3 rows but the evidence says n = 4"):
        load_task_rows_for(path)


def test_a_json_that_is_too_big_or_too_deep_or_not_an_object_is_refused(tmp_path):
    big = tmp_path / "big.json"
    big.write_bytes(b" " * (16 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="is larger than"):
        load_task_rows_for(str(big))
    deep = tmp_path / "deep.json"
    deep.write_text("[" * 100_000, encoding="utf-8")
    with pytest.raises(ValueError, match="is not valid JSON"):
        load_task_rows_for(str(deep))
    listy = tmp_path / "list.json"
    listy.write_text("[1]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_task_rows_for(str(listy))


def test_the_json_bound_is_the_evidence_bound():
    assert ship_task_rows.MAX_POINTER_FILE_BYTES == ship_cmd._MAX_EVIDENCE_BYTES


# -- the pointer in the evidence schema -----------------------------------------------------------


def _evidence(**extra):
    payload = {
        "schema_version": 2,
        "task": {
            "mode": "metric", "base": 0.5, "tuned": 0.75,
            "rows": {"better": 1, "worse": 0, "same": 3, "n": 4},
        },
        "benchmarks": {"mini_mmlu": {"base": 0.7, "tuned": 0.72}},
    }
    payload.update(extra)
    return payload


def test_the_pointer_is_a_registered_version_2_root_field():
    assert "task_rows" in EVIDENCE_SCHEMA_FIELDS["root"]


def test_evidence_with_a_pointer_reads_and_round_trips_through_the_parameter():
    payload = _evidence(task_rows="verdict.task-rows.jsonl")
    verdict = verdict_from_evidence(payload)
    assert verdict_to_evidence(verdict, task_rows="verdict.task-rows.jsonl") == payload
    # the verdict itself does not carry the pointer: it is metadata of the file, like provenance
    assert "task_rows" not in verdict_to_evidence(verdict)


def test_evidence_without_a_pointer_still_reads_and_a_replay_has_none():
    verdict = verdict_from_evidence(_evidence())
    assert "task_rows" not in verdict_to_evidence(verdict)


@pytest.mark.parametrize(
    "value",
    ["/abs/v.task-rows.jsonl", "../v.task-rows.jsonl", "a/v.task-rows.jsonl",
     "C:\\v.task-rows.jsonl", "v.jsonl", "", None, True, 3, ["v.task-rows.jsonl"],
     "v\x00.task-rows.jsonl"],
)
def test_the_reader_refuses_a_pointer_that_is_not_a_bare_file_name(value):
    with pytest.raises(ValueError, match=r"evidence\.task_rows must be a bare file name"):
        verdict_from_evidence(_evidence(task_rows=value))


def test_a_pointer_under_schema_version_1_is_refused_as_a_version_2_field():
    payload = _evidence(task_rows="v.task-rows.jsonl")
    del payload["schema_version"]
    del payload["task"]["rows"]
    with pytest.raises(ValueError, match=r"evidence\.task_rows is a schema_version 2 field"):
        verdict_from_evidence(payload)


def test_a_pointer_is_stamped_as_version_2_even_when_nothing_else_is(tmp_path):
    from soup_cli.utils.ship_verdict import TaskWin, decide_ship

    verdict = decide_ship(TaskWin("metric", 0.5, 0.75, True), [])
    payload = verdict_to_evidence(verdict, task_rows="v.task-rows.jsonl")
    assert payload["schema_version"] == 2 and payload["task_rows"] == "v.task-rows.jsonl"


# -- the records ride on the detail and reach no serialiser ---------------------------------------


def _detail_with_records():
    from soup_cli.utils.ship_verdict import TaskDetail, TaskRows

    rows = (
        _row(
            prompt="SECRET-PROMPT-2208", base_answer="SECRET-ANSWER-4417",
            expected="SECRET-EXPECTED-9031",
        ),
    )
    return TaskDetail(rows=TaskRows(better=1, worse=0, same=0), records=rows), rows


def test_records_survive_decide_ship_when_the_floor_flips_won():
    from soup_cli.utils.ship_verdict import (
        TASK_AXIS,
        BenchmarkDelta,
        NoiseFloor,
        TaskWin,
        decide_ship,
    )

    detail, rows = _detail_with_records()
    floor = NoiseFloor(runs=2, floors=((TASK_AXIS, 0.5),))
    win = TaskWin("metric", 0.2, 0.4, True, detail=detail)  # won without the floor, not with it
    verdict = decide_ship(win, [BenchmarkDelta("b", 0.7, 0.7, 0.0, False)], noise_floor=floor)
    assert verdict.task_win.won is False
    assert verdict.task_win.detail.records == rows


def test_the_records_are_not_compared_or_shown_with_the_detail():
    from soup_cli.utils.ship_verdict import TaskDetail

    detail, _rows = _detail_with_records()
    assert detail == TaskDetail(rows=detail.rows)  # records are not part of equality
    assert "SECRET" not in repr(detail)


def test_no_serialiser_of_a_verdict_carries_a_row_text():
    from soup_cli.utils.ship_verdict import (
        BenchmarkDelta,
        TaskWin,
        decide_ship,
        format_ship_rubric,
        render_ship_pr_markdown,
        verdict_to_dict,
    )

    detail, _rows = _detail_with_records()
    verdict = decide_ship(
        TaskWin("metric", 0.0, 1.0, True, detail=detail),
        [BenchmarkDelta("b", 0.7, 0.7, 0.0, False)],
    )
    for text in (
        json.dumps(verdict_to_dict(verdict, task_rows="v.task-rows.jsonl")),
        json.dumps(verdict_to_evidence(verdict, task_rows="v.task-rows.jsonl")),
        format_ship_rubric(verdict),
        format_ship_rubric(verdict, public=True),
        render_ship_pr_markdown(verdict),
    ):
        assert "SECRET" not in text


def test_the_verdict_dict_always_has_the_key_and_refuses_a_pointer_that_is_not_a_name():
    from soup_cli.utils.ship_verdict import TaskWin, decide_ship, verdict_to_dict

    verdict = decide_ship(TaskWin("metric", 0.0, 1.0, True), [])
    assert verdict_to_dict(verdict)["task_rows"] is None
    assert verdict_to_dict(verdict, task_rows="v.task-rows.jsonl")["task_rows"] == (
        "v.task-rows.jsonl"
    )
    with pytest.raises(ValueError, match="task_rows must be a bare file name"):
        verdict_to_dict(verdict, task_rows="../v.task-rows.jsonl")
    with pytest.raises(ValueError, match="task_rows must be a bare file name"):
        verdict_to_evidence(verdict, task_rows="/abs/v.task-rows.jsonl")


# -- the CLI, with scripted generators of both shapes ---------------------------------------------

_PLAIN_STR = object()


class Run:
    """The result of one ``soup ship`` invocation and every file it left in its directory."""

    def __init__(self, result, files):
        self.result = result
        self.files = files

    def json(self, name):
        return json.loads(self.files[name].decode("utf-8"))

    def rows(self, name="out.task-rows.jsonl"):
        text = self.files[name].decode("ascii")
        assert text.endswith("\n")
        return [json.loads(line) for line in text.split("\n")[:-1]]

    @property
    def plain(self):
        return _plain(self.result.output)


def _tasks(count: int = 8, **extra):
    return [{"prompt": f"q{i}", "expected": f"a{i}", **extra} for i in range(count)]


def _scripted(answers: dict, *, fail=(), default="B"):
    def generate(prompt):
        if prompt in fail:
            raise RuntimeError("generation exploded")
        if prompt not in answers:
            return default
        entry = answers[prompt]
        text, flag = entry if isinstance(entry, tuple) else (entry, False)
        return text if flag is _PLAIN_STR else GeneratedText(text, truncated=flag)

    return generate


def _batched(answers, *, fail=()):
    """The shape ``live_eval.make_generator`` returns: a scalar callable that also has ``.batch``,
    and ``.batch`` returns ``GeneratedText`` rows."""
    scalar = _scripted(answers, fail=fail)

    def generate(prompt):
        return scalar(prompt)

    def batch(prompts):
        return [scalar(prompt) for prompt in prompts]

    generate.batch = batch
    generate.batch_size = 3
    return generate


def _answers(correct, *, count: int = 8, truncated=()):
    out = {}
    for index in range(count):
        text = f"a{index}" if index in correct else "nope"
        out[f"q{index}"] = (text, index in truncated)
    return out


def _snapshot():
    root = Path.cwd()
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }


def _run(monkeypatch, tasks, base_gen, tuned_gen, *extra, output="out.json", evidence="ev.json",
         adapter="out/checkpoint-12", before=None, task_file="task.jsonl", args_before=()):
    def _fake_resolve(base, tuned, adapter, device, quantization=None, **kwargs):
        return base_gen, tuned_gen

    monkeypatch.setattr(ship_cmd, "_resolve_generators", _fake_resolve)
    with runner.isolated_filesystem():
        Path(task_file).parent.mkdir(parents=True, exist_ok=True)
        Path(task_file).write_text(
            "\n".join(json.dumps(task) for task in tasks) + "\n", encoding="utf-8"
        )
        if before is not None:
            before()
        flags = ["--tuned", "tuned-model"] if adapter is None else ["--adapter", adapter]
        args = ["--base", "base-model", *flags, "--task-eval", task_file,
                "--general-suite", "mini_arithmetic", "--device", "cpu"]
        output = output() if callable(output) else output
        evidence = evidence() if callable(evidence) else evidence
        if output is not None:
            args += ["--output", output]
        if evidence is not None:
            args += ["--emit-evidence", evidence]
        result = runner.invoke(ship_cmd.app, [*args_before, *args, *extra])
        files = _snapshot()
    return Run(result, files)


SHAPES = {
    "scalar": _scripted,
    "batch": _batched,
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_cli_metric_run_writes_one_row_per_task_next_to_the_output(monkeypatch, shape):
    make = SHAPES[shape]
    base = make(_answers(correct={0, 1, 2, 3, 4, 5}, truncated={2}))
    tuned = make(_answers(correct={5, 6}, truncated={0, 6}))
    run = _run(monkeypatch, _tasks(), base, tuned)
    assert run.result.exit_code == 2, (run.result.output, repr(run.result.exception))
    rows = run.rows()
    assert len(rows) == 8
    assert [row["index"] for row in rows] == list(range(8))
    for row in rows:
        assert tuple(row) == ROW_KEYS
    first = rows[0]
    assert first["prompt"] == "q0" and first["expected"] == "a0"
    assert first["category"] == "default" and first["scoring"] == "exact"
    assert first["prompt_hash"] == prompt_hash("q0")
    assert (first["base_answer"], first["tuned_answer"]) == ("a0", "nope")
    assert (first["base_score"], first["tuned_score"], first["change"]) == (1.0, 0.0, "worse")
    assert (first["base_truncated"], first["tuned_truncated"]) == (False, True)
    assert (rows[2]["base_truncated"], rows[6]["tuned_truncated"]) == (True, True)
    assert (rows[6]["base_score"], rows[6]["tuned_score"], rows[6]["change"]) == (
        0.0, 1.0, "better",
    )
    assert "Wrote task rows out.task-rows.jsonl (8 rows;" in run.plain


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_cli_the_reported_numbers_are_the_means_of_the_file_exactly(monkeypatch, shape):
    make = SHAPES[shape]
    base = make(_answers(correct={0, 1, 2, 3, 4, 5, 6}))  # 7 / 8, but also test a 1/3-ish set
    tuned = make(_answers(correct={1, 2, 3, 7}))
    run = _run(monkeypatch, _tasks(), base, tuned)
    rows = run.rows()
    win = run.json("out.json")["task_win"]
    assert row_mean([row["base_score"] for row in rows]) == win["base"] == 7 / 8
    assert row_mean([row["tuned_score"] for row in rows]) == win["tuned"] == 4 / 8
    third = _run(
        monkeypatch, _tasks(3), make(_answers(correct={0}, count=3)),
        make(_answers(correct={0, 1}, count=3)),
    )
    win = third.json("out.json")["task_win"]
    assert win["base"] == 1 / 3 and win["tuned"] == 2 / 3  # not binary fractions
    assert row_mean([r["base_score"] for r in third.rows()]) == win["base"]
    assert row_mean([r["tuned_score"] for r in third.rows()]) == win["tuned"]


def test_cli_the_loader_accepts_what_the_command_wrote_and_checks_it_against_the_json(
    monkeypatch, tmp_path
):
    base = _scripted(_answers(correct={0, 1, 2}))
    tuned = _scripted(_answers(correct={1, 2, 3, 4}))
    run = _run(monkeypatch, _tasks(), base, tuned)
    for name in ("out.json", "ev.json"):
        (tmp_path / name).write_bytes(run.files[name])
    (tmp_path / "out.task-rows.jsonl").write_bytes(run.files["out.task-rows.jsonl"])
    for name in ("out.json", "ev.json"):
        loaded = load_task_rows_for(str(tmp_path / name))
        assert len(loaded.rows) == 8
        assert (loaded.better, loaded.worse, loaded.same) == (2, 1, 5)


def test_cli_the_classes_of_the_rows_are_the_rows_block_of_the_verdict(monkeypatch):
    base = _scripted(_answers(correct={0, 1, 2, 3}))
    tuned = _scripted(_answers(correct={2, 3, 4, 5, 6}))
    run = _run(monkeypatch, _tasks(), base, tuned)
    block = run.json("out.json")["task_win"]["rows"]
    changes = [row["change"] for row in run.rows()]
    assert {"better": changes.count("better"), "worse": changes.count("worse"),
            "same": changes.count("same"), "n": len(changes)} == block
    assert block == {"better": 3, "worse": 2, "same": 3, "n": 8}


def test_cli_truncation_flags_of_the_rows_are_the_counts_of_the_verdict(monkeypatch):
    base = _batched(_answers(correct={0}, truncated={1}))
    tuned = _batched(_answers(correct={0, 1}, truncated={1, 2, 3, 4}))
    run = _run(monkeypatch, _tasks(), base, tuned)
    block = run.json("out.json")["task_win"]["truncated"]
    rows = run.rows()
    assert sum(row["base_truncated"] is True for row in rows) == block["base"] == 1
    assert sum(row["tuned_truncated"] is True for row in rows) == block["tuned"] == 4


def test_cli_a_plain_str_generator_gives_null_flags_never_false(monkeypatch):
    def plain(correct):
        return _scripted({k: (v[0], _PLAIN_STR) for k, v in _answers(correct=correct).items()})

    run = _run(monkeypatch, _tasks(), plain({0, 1}), plain({0}))
    for row in run.rows():
        assert row["base_truncated"] is None and row["tuned_truncated"] is None


def test_cli_a_failed_generation_is_flagged_in_its_row(monkeypatch):
    base = _scripted(_answers(correct={0}))
    tuned = _scripted(_answers(correct={0, 1, 2}), fail={"q3", "q4"})
    run = _run(monkeypatch, _tasks(), base, tuned)
    assert run.result.exit_code == 0, (run.result.output, repr(run.result.exception))
    rows = run.rows()
    assert [row["tuned_failed"] for row in rows] == [False] * 3 + [True, True] + [False] * 3
    assert not any(row["base_failed"] for row in rows)
    failed = rows[3]
    assert failed["tuned_answer"] == "" and failed["tuned_truncated"] is None
    assert failed["tuned_score"] == 0.0
    assert sum(row["tuned_failed"] for row in rows) == run.json("out.json")["task_win"][
        "generation_failures"]["tuned"]


def test_cli_a_semantic_row_is_scored_by_matched_not_by_similarity(monkeypatch):
    tasks = [{"prompt": f"r{i}", "expected": "alpha beta gamma delta", "scoring": "semantic"}
             for i in range(3)]
    base = _scripted({"r0": "alpha beta gamma", "r1": "alpha", "r2": "alpha beta gamma delta"})
    tuned = _scripted({"r0": "alpha beta gamma delta", "r1": "alpha beta",
                       "r2": "alpha beta gamma"})
    run = _run(monkeypatch, tasks, base, tuned)
    rows = run.rows()
    assert [row["scoring"] for row in rows] == ["semantic"] * 3
    assert [(row["base_score"], row["tuned_score"]) for row in rows] == [
        (1.0, 1.0), (0.0, 1.0), (1.0, 1.0),
    ]
    win = run.json("out.json")["task_win"]
    assert row_mean([row["base_score"] for row in rows]) == win["base"]


# a judge whose per-row scores are not binary fractions once rescaled to [0, 1]
BASE_WEIGHTS = [1.7, 2.9, 4.3, 3.1, 2.2, 4.9, 1.1, 3.7]
TUNED_WEIGHTS = [2.3, 2.9, 3.9, 3.3, 4.4, 1.9, 1.3, 3.7]


def _evaluator_number(weights):
    """What the judge leg reports for these per-row weighted scores: the evaluator's overall
    score on the 1-5 rubric, rescaled to [0, 1]."""
    from soup_cli.eval.judge import JudgeResults, JudgeScore

    results = JudgeResults(scores=[JudgeScore("p", "r", weighted_score=w) for w in weights])
    results.compute()
    return max(0.0, min(1.0, (results.overall_score - 1.0) / 4.0))


def _patch_judge(monkeypatch):
    from soup_cli.eval.judge import JudgeEvaluator, JudgeScore

    table = {f"b{i}": w for i, w in enumerate(BASE_WEIGHTS)}
    table.update({f"t{i}": w for i, w in enumerate(TUNED_WEIGHTS)})
    monkeypatch.setattr("soup_cli.eval.gate._parse_judge_url", lambda url: ("ollama", "m", None))

    def evaluate(self, prompt, response, category="default"):
        return JudgeScore(prompt, str(response), weighted_score=table[str(response)])

    monkeypatch.setattr(JudgeEvaluator, "evaluate", evaluate)


def _judge_gens(shape):
    make = SHAPES[shape]
    base = make({f"q{i}": (f"b{i}", i % 3 == 0) for i in range(8)})
    tuned = make({f"q{i}": (f"t{i}", False) for i in range(8)})
    return base, tuned


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_cli_judge_score_rows_average_to_the_reported_number_within_1e_12(monkeypatch, shape):
    _patch_judge(monkeypatch)
    base, tuned = _judge_gens(shape)
    run = _run(monkeypatch, _tasks(), base, tuned,
               "--task-mode", "judge_score", "--judge-model", "ollama://judge")
    assert run.result.exit_code in (0, 2), (run.result.output, repr(run.result.exception))
    rows = run.rows()
    win = run.json("out.json")["task_win"]
    base_scores = [row["base_score"] for row in rows]
    tuned_scores = [row["tuned_score"] for row in rows]
    # the scores are the judge's, rescaled to [0, 1] on its 1-5 rubric
    assert base_scores == [(w - 1.0) / 4.0 for w in BASE_WEIGHTS]
    assert any(Fraction(score) != Fraction(str(score)) for score in base_scores)  # not binary
    # the reported number is the evaluator's own, byte for byte (what 0.77.x reported); the rows
    # add up to it within 1e-12, which is all a judge leg claims
    assert win["base"] == _evaluator_number(BASE_WEIGHTS)
    assert win["tuned"] == _evaluator_number(TUNED_WEIGHTS)
    assert abs(row_mean(base_scores) - win["base"]) <= JUDGE_MEAN_TOLERANCE
    assert abs(row_mean(tuned_scores) - win["tuned"]) <= JUDGE_MEAN_TOLERANCE
    assert [row["scoring"] for row in rows] == [None] * 8
    assert [row["base_failed"] for row in rows] == [False] * 8
    assert [row["base_truncated"] for row in rows][0:3] == [True, False, False]
    assert rows[0]["base_answer"] == "b0" and rows[0]["tuned_answer"] == "t0"
    assert win["rows"]["n"] == 8
    changes = [row["change"] for row in rows]
    assert changes.count("better") == win["rows"]["better"]
    assert changes.count("worse") == win["rows"]["worse"]
    assert changes.count("same") == win["rows"]["same"]


def test_cli_the_judge_number_of_a_row_file_that_cannot_be_audited_stays_the_judges(monkeypatch):
    # an evaluator whose overall score is not the mean of its rows: the file is left out and the
    # reported number is the evaluator's own, exactly as before
    from types import SimpleNamespace

    from soup_cli.eval.judge import JudgeEvaluator

    monkeypatch.setattr("soup_cli.eval.gate._parse_judge_url", lambda url: ("ollama", "m", None))
    monkeypatch.setattr(
        JudgeEvaluator, "evaluate_batch",
        lambda self, items: SimpleNamespace(
            overall_score=3.0,
            scores=[SimpleNamespace(weighted_score=1.0)] * len(items),
        ),
    )
    base, tuned = _judge_gens("scalar")
    run = _run(monkeypatch, _tasks(), base, tuned,
               "--task-mode", "judge_score", "--judge-model", "ollama://judge")
    assert run.result.exit_code == 2, (run.result.output, repr(run.result.exception))
    win = run.json("out.json")["task_win"]
    assert win["base"] == win["tuned"] == 0.5  # (3.0 - 1) / 4: the evaluator's overall score
    assert "out.task-rows.jsonl" not in run.files
    assert run.json("out.json")["task_rows"] is None
    assert "Task rows file (per-row results) could not be built and is left out: ValueError" in (
        run.plain
    )


def test_cli_pairwise_has_no_row_file_and_a_null_pointer(monkeypatch):
    from soup_cli.eval.judge import JudgeEvaluator

    monkeypatch.setattr("soup_cli.eval.gate._parse_judge_url", lambda url: ("ollama", "m", None))
    monkeypatch.setattr(JudgeEvaluator, "compare_pair", lambda *args, **kwargs: 1)
    base = _scripted(_answers(correct={0}))
    tuned = _scripted(_answers(correct={0, 1}))
    run = _run(monkeypatch, _tasks(), base, tuned,
               "--task-mode", "pairwise", "--judge-model", "ollama://judge")
    assert run.result.exit_code in (0, 2), (run.result.output, repr(run.result.exception))
    assert run.json("out.json")["task_rows"] is None
    assert "task_rows" not in run.json("ev.json")
    assert not [name for name in run.files if name.endswith(".task-rows.jsonl")]


def test_cli_the_pointer_is_the_bare_name_in_the_verdict_and_in_the_evidence(monkeypatch):
    base = _scripted(_answers(correct={0}))
    tuned = _scripted(_answers(correct={0, 1}))
    run = _run(monkeypatch, _tasks(), base, tuned)
    assert run.json("out.json")["task_rows"] == "out.task-rows.jsonl"
    assert run.json("ev.json")["task_rows"] == "out.task-rows.jsonl"
    assert run.json("ev.json")["schema_version"] == 2
    verdict_from_evidence(run.json("ev.json"))  # and the strict reader takes what was written


def test_cli_a_pointer_is_a_name_even_when_the_output_is_in_a_subdirectory(monkeypatch):
    base = _scripted(_answers(correct={0}))
    tuned = _scripted(_answers(correct={0, 1}))
    run = _run(monkeypatch, _tasks(), base, tuned, output="deep/er/out.json",
               evidence="deep/er/ev.json")
    assert run.json("deep/er/out.json")["task_rows"] == "out.task-rows.jsonl"
    assert "deep/er/out.task-rows.jsonl" in run.files
    assert run.json("deep/er/ev.json")["task_rows"] == "out.task-rows.jsonl"
    assert "Wrote task rows deep/er/out.task-rows.jsonl (8 rows;" in run.plain.replace("\\", "/")


def test_cli_the_evidence_names_the_file_only_when_it_sits_in_the_same_directory(monkeypatch):
    base = _scripted(_answers(correct={0}))
    tuned = _scripted(_answers(correct={0, 1}))
    run = _run(monkeypatch, _tasks(), base, tuned, output="a/out.json", evidence="b/ev.json")
    assert run.json("a/out.json")["task_rows"] == "out.task-rows.jsonl"
    assert "a/out.task-rows.jsonl" in run.files
    assert "task_rows" not in run.json("b/ev.json")  # a name that would not resolve next to it
    assert not any(name.startswith("b/") and name.endswith(".jsonl") for name in run.files)


def test_cli_without_output_or_evidence_there_is_nowhere_to_put_the_file(monkeypatch):
    base = _scripted(_answers(correct={0}))
    tuned = _scripted(_answers(correct={0, 1}))
    run = _run(monkeypatch, _tasks(), base, tuned, output=None, evidence=None)
    assert run.result.exit_code == 0, (run.result.output, repr(run.result.exception))
    assert not [name for name in run.files if name != "task.jsonl"]
    assert "task rows" not in run.plain.lower()


def test_cli_adapter_noop_writes_no_rows(monkeypatch):
    monkeypatch.setattr(ship_cmd, "_adapter_is_noop", lambda adapter: True)
    base = _scripted(_answers(correct={0}))
    run = _run(monkeypatch, _tasks(), base, base)
    assert run.result.exit_code == 2, (run.result.output, repr(run.result.exception))
    assert run.json("out.json")["task_rows"] is None
    assert not [name for name in run.files if name.endswith(".task-rows.jsonl")]


def test_cli_a_replay_writes_no_rows_and_leaves_an_earlier_file_untouched(monkeypatch):
    base = _scripted(_answers(correct={0}))
    tuned = _scripted(_answers(correct={0, 1}))
    live = _run(monkeypatch, _tasks(), base, tuned)
    with runner.isolated_filesystem():
        Path("ev.json").write_bytes(live.files["ev.json"])
        Path("ev.task-rows.jsonl").write_bytes(b"whatever an earlier run left\n")
        result = runner.invoke(
            ship_cmd.app, ["--evidence", "ev.json", "--output", "replay.json"],
        )
        files = _snapshot()
    assert result.exit_code in (0, 2), (result.output, repr(result.exception))
    assert files["ev.task-rows.jsonl"] == b"whatever an earlier run left\n"
    assert "replay.task-rows.jsonl" not in files
    assert json.loads(files["replay.json"])["task_rows"] is None


def test_cli_evidence_whose_row_file_is_missing_still_replays(monkeypatch):
    base = _scripted(_answers(correct={0}))
    tuned = _scripted(_answers(correct={0, 1}))
    live = _run(monkeypatch, _tasks(), base, tuned)
    with runner.isolated_filesystem():
        Path("ev.json").write_bytes(live.files["ev.json"])  # the row file is not next to it
        result = runner.invoke(ship_cmd.app, ["--evidence", "ev.json"])
    assert result.exit_code == 0, (result.output, repr(result.exception))


def test_cli_two_identical_runs_write_the_same_bytes(monkeypatch):
    base = _scripted(_answers(correct={0, 1}))
    tuned = _scripted(_answers(correct={1, 2, 3}))
    first = _run(monkeypatch, _tasks(), base, tuned)
    second = _run(monkeypatch, _tasks(), base, tuned)
    assert first.files["out.task-rows.jsonl"] == second.files["out.task-rows.jsonl"]


# paths with a space, with brackets, and the Windows forms


@pytest.mark.parametrize(
    "output",
    ["my out dir/out file.json", "out[1].json", "dir [x]/o [bold].json", "sub/out.json"],
)
def test_cli_paths_with_a_space_or_a_bracket_work_and_print_literally(monkeypatch, output):
    base = _scripted(_answers(correct={0}))
    tuned = _scripted(_answers(correct={0, 1}))
    run = _run(monkeypatch, _tasks(), base, tuned, output=output, evidence=None)
    assert run.result.exit_code == 0, (run.result.output, repr(run.result.exception))
    stem = output[: -len(".json")]
    assert f"{stem}.task-rows.jsonl" in run.files
    assert run.json(output)["task_rows"] == os.path.basename(stem) + ".task-rows.jsonl"
    assert f"Wrote task rows {stem}.task-rows.jsonl (8 rows;" in run.plain


def test_cli_an_absolute_path_in_the_platform_form_works(monkeypatch):
    base = _scripted(_answers(correct={0}))
    tuned = _scripted(_answers(correct={0, 1}))
    run = _run(monkeypatch, _tasks(), base, tuned,
               output=lambda: str(Path.cwd() / "abs dir" / "out.json"), evidence=None)
    assert run.result.exit_code == 0, (run.result.output, repr(run.result.exception))
    assert "abs dir/out.task-rows.jsonl" in run.files
    assert run.json("abs dir/out.json")["task_rows"] == "out.task-rows.jsonl"


@pytest.mark.skipif(os.name != "nt", reason="a backslash separates directories only on Windows")
def test_cli_a_backslash_relative_path_works_on_windows(monkeypatch):
    base = _scripted(_answers(correct={0}))
    tuned = _scripted(_answers(correct={0, 1}))
    run = _run(monkeypatch, _tasks(), base, tuned, output="sub\\out.json", evidence=None)
    assert run.result.exit_code == 0, (run.result.output, repr(run.result.exception))
    assert "sub/out.task-rows.jsonl" in run.files
    assert run.json("sub/out.json")["task_rows"] == "out.task-rows.jsonl"


@pytest.mark.skipif(os.name == "nt", reason="a backslash is an ordinary file-name character")
def test_cli_on_posix_a_backslash_in_the_output_name_is_refused_before_any_model_loads(
    monkeypatch,
):
    called = []

    def generate(prompt):
        called.append(prompt)
        return "x"

    run = _run(monkeypatch, _tasks(), generate, generate, output="sub\\out.json", evidence=None)
    assert run.result.exit_code == 3, (run.result.output, repr(run.result.exception))
    assert "path separator" in run.plain
    assert called == []


@pytest.mark.parametrize("output", ["\\\\host\\share\\out.json", "//host/share/out.json"])
def test_cli_a_network_share_output_is_refused_before_any_model_loads(monkeypatch, output):
    called = []

    def generate(prompt):
        called.append(prompt)
        return "x"

    run = _run(monkeypatch, _tasks(), generate, generate, output=output, evidence=None)
    assert run.result.exit_code == 3, (run.result.output, repr(run.result.exception))
    assert "network share or a device" in run.plain
    assert called == []


# collisions and the replace rule, before any model loads


def test_cli_an_evidence_path_that_is_the_rows_file_is_refused_before_the_run(monkeypatch):
    called = []

    def generate(prompt):
        called.append(prompt)
        return "x"

    run = _run(monkeypatch, _tasks(), generate, generate, output="out.json",
               evidence="out.task-rows.jsonl")
    assert run.result.exit_code == 3, (run.result.output, repr(run.result.exception))
    assert "is the same file as --emit-evidence" in run.plain
    assert called == []


def test_cli_a_task_file_that_is_the_rows_file_is_refused_and_untouched(monkeypatch):
    base = _scripted(_answers(correct={0}))
    run = _run(monkeypatch, _tasks(), base, base, output="eval.json", evidence=None,
               task_file="eval.task-rows.jsonl")
    assert run.result.exit_code == 3, (run.result.output, repr(run.result.exception))
    assert "is the same file as --task-eval" in run.plain
    assert json.loads(run.files["eval.task-rows.jsonl"].split(b"\n")[0])["prompt"] == "q0"


def test_cli_a_foreign_file_at_the_rows_path_is_refused_before_the_run_and_kept(monkeypatch):
    def put_foreign():
        Path("out.task-rows.jsonl").write_bytes(b'{"mine": 1}\n')

    base = _scripted(_answers(correct={0}))
    run = _run(monkeypatch, _tasks(), base, base, before=put_foreign)
    assert run.result.exit_code == 3, (run.result.output, repr(run.result.exception))
    assert "exists and is not a task-rows file written by soup ship" in run.plain
    assert run.files["out.task-rows.jsonl"] == b'{"mine": 1}\n'
    assert "out.json" not in run.files  # refused before the run, nothing was written


def test_cli_a_rows_file_of_an_earlier_run_is_replaced_by_this_run(monkeypatch):
    base = _scripted(_answers(correct={0}))
    tuned = _scripted(_answers(correct={0, 1}))
    first = _run(monkeypatch, _tasks(), base, tuned)

    def put_earlier():
        Path("out.task-rows.jsonl").write_bytes(first.files["out.task-rows.jsonl"])

    other = _run(monkeypatch, _tasks(), _scripted(_answers(correct={3})), tuned,
                 before=put_earlier)
    assert other.result.exit_code in (0, 2), (other.result.output, repr(other.result.exception))
    assert other.files["out.task-rows.jsonl"] != first.files["out.task-rows.jsonl"]


def test_cli_a_stale_rows_file_is_left_alone_by_a_run_that_writes_none(monkeypatch):
    from soup_cli.eval.judge import JudgeEvaluator

    monkeypatch.setattr("soup_cli.eval.gate._parse_judge_url", lambda url: ("ollama", "m", None))
    monkeypatch.setattr(JudgeEvaluator, "compare_pair", lambda *args, **kwargs: 1)
    base = _scripted(_answers(correct={0}))
    tuned = _scripted(_answers(correct={0, 1}))
    earlier = _run(monkeypatch, _tasks(), base, tuned)

    def put_earlier():
        Path("out.task-rows.jsonl").write_bytes(earlier.files["out.task-rows.jsonl"])

    run = _run(monkeypatch, _tasks(), base, tuned, "--task-mode", "pairwise",
               "--judge-model", "ollama://judge", before=put_earlier)
    assert run.files["out.task-rows.jsonl"] == earlier.files["out.task-rows.jsonl"]
    assert run.json("out.json")["task_rows"] is None  # and the verdict says there is none


def test_cli_a_run_that_writes_no_rows_is_not_asked_about_the_rows_path(monkeypatch):
    # pairwise writes no per-row file, so a file of the user's own at that name is none of its
    # business: it neither refuses the run nor is touched
    from soup_cli.eval.judge import JudgeEvaluator

    monkeypatch.setattr("soup_cli.eval.gate._parse_judge_url", lambda url: ("ollama", "m", None))
    monkeypatch.setattr(JudgeEvaluator, "compare_pair", lambda *args, **kwargs: 1)

    def put_foreign():
        Path("out.task-rows.jsonl").write_bytes(b'{"mine": 1}\n')

    base = _scripted(_answers(correct={0}))
    tuned = _scripted(_answers(correct={0, 1}))
    run = _run(monkeypatch, _tasks(), base, tuned, "--task-mode", "pairwise",
               "--judge-model", "ollama://judge", before=put_foreign)
    assert run.result.exit_code in (0, 2), (run.result.output, repr(run.result.exception))
    assert run.files["out.task-rows.jsonl"] == b'{"mine": 1}\n'
    assert run.json("out.json")["task_rows"] is None


# the write fails: the verdict stands, a SHIP becomes exit 1 (the spec: the file is part of the
# evidence, as the judge rows are), a DON'T SHIP keeps its exit 2, and nothing is claimed that was
# not written


@pytest.mark.parametrize(("correct", "code"), [({0, 1, 2}, 1), (set(), 2)])
def test_cli_a_failed_write_is_reported_after_the_verdict_and_the_evidence_are_written(
    monkeypatch, correct, code
):
    def broken(rows, anchor):
        raise OSError("no space left on device")

    monkeypatch.setattr(ship_cmd, "write_task_rows", broken)
    base = _scripted(_answers(correct={0}))
    tuned = _scripted(_answers(correct=correct))
    run = _run(monkeypatch, _tasks(), base, tuned)
    assert run.result.exit_code == code, (run.result.output, repr(run.result.exception))
    assert "cannot write the task rows next to --output: OSError: no space left on device" in (
        run.plain
    )
    verdict = run.json("out.json")  # the verdict was written first and is complete
    assert verdict["decision"] in ("SHIP", "DON'T SHIP") and verdict["task_rows"] is None
    assert "task_rows" not in run.json("ev.json")  # evidence never names a file that is not there
    assert "out.task-rows.jsonl" not in run.files


def test_cli_a_failed_write_changes_neither_the_decision_nor_the_numbers(monkeypatch):
    base = _scripted(_answers(correct={0, 1}))
    tuned = _scripted(_answers(correct={0, 1, 2}))
    good = _run(monkeypatch, _tasks(), base, tuned)

    def broken(rows, anchor):
        raise ValueError("refused")

    monkeypatch.setattr(ship_cmd, "write_task_rows", broken)
    bad = _run(monkeypatch, _tasks(), base, tuned)
    for key in ("decision", "failed_rule", "task_win", "benchmark_deltas", "judged"):
        assert bad.json("out.json")[key] == good.json("out.json")[key], key


def test_cli_a_failed_write_still_posts_the_pr_comment(monkeypatch):
    from soup_cli.utils import adapter_pr

    posted = []
    monkeypatch.setattr(
        adapter_pr, "post_pr_comment", lambda target, body: posted.append(body) or "https://x/1"
    )

    def broken(rows, anchor):
        raise OSError("nope")

    monkeypatch.setattr(ship_cmd, "write_task_rows", broken)
    base = _scripted(_answers(correct={0}))
    tuned = _scripted(_answers(correct={0, 1}))
    run = _run(monkeypatch, _tasks(), base, tuned, "--push", "owner/repo#42")
    assert run.result.exit_code == 1
    assert len(posted) == 1  # the verdict is announced before the exit


def test_cli_a_failure_to_build_the_rows_is_display_only(monkeypatch):
    def boom(*args, **kwargs):
        raise ValueError("secret \x1b[31mrow failure text")

    monkeypatch.setattr(ship_cmd, "build_task_rows", boom)
    base = _scripted(_answers(correct={0}))
    tuned = _scripted(_answers(correct={0, 1}))
    run = _run(monkeypatch, _tasks(), base, tuned)
    assert run.result.exit_code == 0, (run.result.output, repr(run.result.exception))
    assert run.json("out.json")["task_win"]["rows"] is not None  # PR-A's detail is kept
    assert run.json("out.json")["task_rows"] is None
    assert "out.task-rows.jsonl" not in run.files
    assert "Task rows file (per-row results) could not be built and is left out: ValueError" in (
        run.plain
    )
    assert "secret" not in run.plain and "\x1b" not in run.result.output


def test_cli_the_path_it_prints_is_made_safe_for_the_terminal(monkeypatch):
    def fake_write(rows, anchor):
        return "o\x1b[2Jut\x07.task-rows.jsonl"

    monkeypatch.setattr(ship_cmd, "write_task_rows", fake_write)
    base = _scripted(_answers(correct={0}))
    tuned = _scripted(_answers(correct={0, 1}))
    run = _run(monkeypatch, _tasks(), base, tuned, evidence=None)
    assert "\x1b[2J" not in run.result.output and "\x07" not in run.result.output
    assert "Wrote task rows" in run.plain


def test_cli_a_failed_write_message_is_made_safe_for_the_terminal(monkeypatch):
    def broken(rows, anchor):
        raise OSError("bad \x1b[31mred [bold]name")

    monkeypatch.setattr(ship_cmd, "write_task_rows", broken)
    base = _scripted(_answers(correct={0}))
    run = _run(monkeypatch, _tasks(), base, base)  # a tie: DON'T SHIP keeps its exit 2
    assert run.result.exit_code == 2
    assert "\x1b[31mred" not in run.result.output
    assert "[bold]name" in run.plain  # markup is printed, not interpreted


# privacy: the texts are in the local file and nowhere that leaves the machine


SECRET = "PRIVATE-PROMPT-7731"
SECRET_ANSWER = "PRIVATE-ANSWER-5519"


def test_cli_the_texts_are_in_the_row_file_and_in_nothing_that_is_uploaded(monkeypatch):
    from soup_cli.utils import adapter_pr

    posted = []
    monkeypatch.setattr(
        adapter_pr, "post_pr_comment", lambda target, body: posted.append(body) or "https://x/1"
    )
    tasks = [{"prompt": f"{SECRET} {i}", "expected": f"{SECRET_ANSWER} {i}"} for i in range(4)]
    base = _scripted({f"{SECRET} {i}": "wrong" for i in range(4)})
    tuned = _scripted({f"{SECRET} {i}": f"{SECRET_ANSWER} {i}" for i in range(4)})
    run = _run(monkeypatch, tasks, base, tuned, "--push", "owner/repo#42")
    assert run.result.exit_code == 0, (run.result.output, repr(run.result.exception))
    local = run.files["out.task-rows.jsonl"].decode("ascii")
    assert SECRET in local and SECRET_ANSWER in local  # the audit needs them, locally
    assert len(posted) == 1
    for name in ("out.json", "ev.json"):
        assert SECRET not in run.files[name].decode("utf-8")
        assert SECRET_ANSWER not in run.files[name].decode("utf-8")
    assert SECRET not in posted[0] and SECRET_ANSWER not in posted[0]
    assert SECRET not in run.result.output and SECRET_ANSWER not in run.result.output


def test_cli_a_hostile_row_text_is_exact_in_the_file_and_never_printed(monkeypatch):
    hostile = "q\x1b]0;owned\x07 [bold]x[/] \u202e end"
    tasks = [{"prompt": hostile, "expected": "a"}]
    base = _scripted({hostile: "no"})
    tuned = _scripted({hostile: "a"})
    run = _run(monkeypatch, tasks, base, tuned)
    assert run.result.exit_code == 0, (run.result.output, repr(run.result.exception))
    assert run.rows()[0]["prompt"] == hostile  # the file is exact
    assert "\x1b" not in run.result.output and "owned" not in run.result.output


# -- second read: the judge number, the anchor, the refusals, the exit code ------------------------


def _patch_judge_weighted(monkeypatch, table):
    """A judge whose weighted score for an answer is ``table[answer]`` (any number, in or out of
    the rubric's scale)."""
    from soup_cli.eval.judge import JudgeEvaluator, JudgeScore

    monkeypatch.setattr("soup_cli.eval.gate._parse_judge_url", lambda url: ("ollama", "m", None))

    def evaluate(self, prompt, response, category="default"):
        return JudgeScore(prompt, str(response), weighted_score=table[str(response)])

    monkeypatch.setattr(JudgeEvaluator, "evaluate", evaluate)


def _patch_judge_criteria(monkeypatch, table):
    """A judge whose criteria scores for an answer are ``table[answer]`` on the built-in rubric
    (3 criteria, weight 1, scale 1-5), weighted by the real ``_compute_weighted_score``."""
    from soup_cli.eval.judge import (
        DEFAULT_RUBRIC,
        JudgeEvaluator,
        JudgeScore,
        _compute_weighted_score,
    )

    names = [criterion["name"] for criterion in DEFAULT_RUBRIC["criteria"]]
    monkeypatch.setattr("soup_cli.eval.gate._parse_judge_url", lambda url: ("ollama", "m", None))

    def evaluate(self, prompt, response, category="default"):
        scores = dict(zip(names, table[str(response)]))
        return JudgeScore(
            prompt, str(response), scores=scores,
            weighted_score=_compute_weighted_score(scores, self.rubric), category=category,
        )

    monkeypatch.setattr(JudgeEvaluator, "evaluate", evaluate)


def test_cli_a_mathematical_tie_of_a_judge_stays_a_tie_and_never_wins_by_one_ulp(monkeypatch):
    # Both sides earn 12 criteria points over two rows on the built-in rubric: 0.25 against
    # 0.25, a tie, DON'T SHIP. Reporting the mean of the rescaled rows made it 0.24999999999999997
    # against 0.25 = a win. The evaluator's own number stays the reported number, byte for byte.
    _patch_judge_criteria(
        monkeypatch,
        {"b0": (1, 1, 2), "b1": (1, 2, 5), "t0": (1, 1, 1), "t1": (1, 3, 5)},
    )
    base = _scripted({"q0": ("b0", False), "q1": ("b1", False)})
    tuned = _scripted({"q0": ("t0", False), "q1": ("t1", False)})
    run = _run(monkeypatch, _tasks(2), base, tuned,
               "--task-mode", "judge_score", "--judge-model", "ollama://judge")
    assert run.result.exit_code == 2, (run.result.output, repr(run.result.exception))
    verdict = run.json("out.json")
    assert verdict["task_win"]["base"] == 0.25
    assert verdict["task_win"]["tuned"] == 0.25
    assert verdict["task_win"]["won"] is False
    assert verdict["decision"] == "DON'T SHIP" and verdict["failed_rule"] == "task_win"
    # the file is there and adds up to the number within 1e-12, which is all it claims
    rows = run.rows()
    assert abs(row_mean([row["base_score"] for row in rows]) - 0.25) <= JUDGE_MEAN_TOLERANCE
    assert abs(row_mean([row["tuned_score"] for row in rows]) - 0.25) <= JUDGE_MEAN_TOLERANCE
    assert verdict["task_win"]["rows"] == {"better": 1, "worse": 1, "same": 0, "n": 2}
    assert [row["change"] for row in rows] == ["worse", "better"]


def test_a_judge_leg_may_differ_from_its_row_mean_by_1e_12_and_no_more():
    assert JUDGE_MEAN_TOLERANCE == 1e-12
    scores = [0.1, 0.2, 0.3, 0.7]
    mean = row_mean(scores)
    tuned = _leg(scores, prefix="t")
    for offset in (0.0, 9e-13, -9e-13):  # inside
        build_task_rows(_leg(scores, value=mean + offset), tuned, scoring_applies=False)
    for offset in (1.1e-12, -1.1e-12):  # outside
        with pytest.raises(ValueError, match="the base row scores average"):
            build_task_rows(_leg(scores, value=mean + offset), tuned, scoring_applies=False)
    with pytest.raises(ValueError, match="the tuned row scores average"):
        build_task_rows(_leg(scores), _leg(scores, prefix="t", value=mean + 1.1e-12),
                        scoring_applies=False)


def test_a_metric_leg_still_has_to_equal_its_row_mean_exactly():
    scores = [0.0, 1.0, 1.0]
    mean = row_mean(scores)
    with pytest.raises(ValueError, match="the base row scores average"):
        build_task_rows(_leg(scores, value=mean + 1e-15), _leg(scores, prefix="t"),
                        scoring_applies=True)


@pytest.mark.parametrize("scoring_applies", [True, False], ids=["metric", "judge"])
def test_a_nan_reported_number_is_never_within_a_tolerance(scoring_applies):
    scores = [0.25, 0.75]
    with pytest.raises(ValueError, match="the base row scores average"):
        build_task_rows(_leg(scores, value=float("nan")), _leg(scores, prefix="t"),
                        scoring_applies=scoring_applies)


def test_the_loader_takes_the_judge_tolerance_only_when_asked_to(tmp_path):
    entries = _good_lines(3)
    for entry, score in zip(entries, (0.1, 0.2, 0.3)):
        entry["base_score"] = score
        entry["tuned_score"] = score
        entry["change"] = "same"
    path = _write_lines(tmp_path, entries)
    mean = row_mean([0.1, 0.2, 0.3])
    load_task_rows(path, base=mean + 9e-13, mean_tolerance=JUDGE_MEAN_TOLERANCE)
    with pytest.raises(ValueError, match="is not the reported base"):
        load_task_rows(path, base=mean + 1.1e-12, mean_tolerance=JUDGE_MEAN_TOLERANCE)
    with pytest.raises(ValueError, match="is not the reported tuned"):
        load_task_rows(path, tuned=mean - 1.1e-12, mean_tolerance=JUDGE_MEAN_TOLERANCE)
    with pytest.raises(ValueError, match="is not the reported base"):
        load_task_rows(path, base=mean + 9e-13)  # the default is exact


def test_the_verdict_names_its_mode_and_that_picks_the_tolerance(tmp_path):
    entries = _good_lines(3)
    for entry, score in zip(entries, (0.1, 0.2, 0.3)):
        entry["base_score"] = score
        entry["tuned_score"] = score
        entry["change"] = "same"
    mean = row_mean([0.1, 0.2, 0.3])
    rows_path = _write_lines(tmp_path, entries)
    assert Path(rows_path).name == "v.task-rows.jsonl"
    for mode, loads in (("judge_score", True), ("metric", False), ("pairwise", False)):
        payload = {
            "task_win": {"mode": mode, "base": mean + 9e-13, "tuned": mean,
                         "rows": {"better": 0, "worse": 0, "same": 3, "n": 3}},
            "task_rows": "v.task-rows.jsonl",
        }
        (tmp_path / "verdict.json").write_text(json.dumps(payload), encoding="utf-8")
        if loads:
            assert len(load_task_rows_for(str(tmp_path / "verdict.json")).rows) == 3
        else:
            with pytest.raises(ValueError, match="is not the reported base"):
                load_task_rows_for(str(tmp_path / "verdict.json"))


def test_cli_judge_rows_are_rescaled_without_clamping_and_the_counts_are_the_files(monkeypatch):
    # scores outside the rubric's scale (a judge with a negative criterion weight can give them)
    # are not clamped together: 6 and 7 on the 1-5 scale stay two different rows, better by one.
    # The row file cannot hold a score above 1, so it is left out, and says so.
    _patch_judge_weighted(monkeypatch, {"b0": 6.0, "b1": 3.0, "t0": 7.0, "t1": 3.0})
    base = _scripted({"q0": ("b0", False), "q1": ("b1", False)})
    tuned = _scripted({"q0": ("t0", False), "q1": ("t1", False)})
    run = _run(monkeypatch, _tasks(2), base, tuned,
               "--task-mode", "judge_score", "--judge-model", "ollama://judge")
    assert run.result.exit_code == 0, (run.result.output, repr(run.result.exception))
    win = run.json("out.json")["task_win"]
    assert win["rows"] == {"better": 1, "worse": 0, "same": 1, "n": 2}
    assert win["base"] == (4.5 - 1.0) / 4.0 and win["tuned"] == 1.0
    assert run.json("out.json")["task_rows"] is None
    assert not [name for name in run.files if name.endswith(".task-rows.jsonl")]
    assert "Task rows file (per-row results) could not be built and is left out: ValueError" in (
        run.plain
    )


def test_cli_the_counts_of_a_judge_run_are_the_classes_of_its_file(monkeypatch):
    _patch_judge(monkeypatch)
    base, tuned = _judge_gens("scalar")
    run = _run(monkeypatch, _tasks(), base, tuned,
               "--task-mode", "judge_score", "--judge-model", "ollama://judge")
    block = run.json("out.json")["task_win"]["rows"]
    changes = [row["change"] for row in run.rows()]
    assert block == {"better": changes.count("better"), "worse": changes.count("worse"),
                     "same": changes.count("same"), "n": len(changes)}


# the row file belongs to --output only: with --emit-evidence alone nothing new appears


def test_cli_with_only_emit_evidence_no_row_file_is_written_and_the_evidence_has_no_pointer(
    monkeypatch,
):
    base = _scripted(_answers(correct={0}))
    tuned = _scripted(_answers(correct={0, 1}))
    run = _run(monkeypatch, _tasks(), base, tuned, output=None, evidence="sub/ev.json")
    assert run.result.exit_code == 0, (run.result.output, repr(run.result.exception))
    assert not [name for name in run.files if name.endswith(".task-rows.jsonl")]
    assert "task_rows" not in run.json("sub/ev.json")
    assert "task rows" not in run.plain.lower()


def test_cli_the_terminal_line_says_the_file_holds_the_prompts_and_the_answers(monkeypatch):
    base = _scripted(_answers(correct={0}))
    tuned = _scripted(_answers(correct={0, 1}))
    run = _run(monkeypatch, _tasks(), base, tuned)
    assert (
        "Wrote task rows out.task-rows.jsonl (8 rows; it holds your prompts and the answers)"
        in run.plain
    )


# a failed write keeps the verdict's own exit code unless that is 0


def test_cli_a_failed_write_keeps_dont_ship_at_exit_2_and_prints_the_error(monkeypatch):
    def broken(rows, anchor):
        raise OSError("no space left on device")

    monkeypatch.setattr(ship_cmd, "write_task_rows", broken)
    base = _scripted(_answers(correct={0, 1, 2}))
    tuned = _scripted(_answers(correct={0}))  # tuned is worse: DON'T SHIP
    run = _run(monkeypatch, _tasks(), base, tuned)
    assert run.result.exit_code == 2, (run.result.output, repr(run.result.exception))
    assert "cannot write the task rows next to --output: OSError: no space left on device" in (
        run.plain
    )
    assert run.json("out.json")["decision"] == "DON'T SHIP"


def test_cli_a_failed_write_turns_a_ship_into_exit_1(monkeypatch):
    def broken(rows, anchor):
        raise OSError("no space left on device")

    monkeypatch.setattr(ship_cmd, "write_task_rows", broken)
    base = _scripted(_answers(correct={0}))
    tuned = _scripted(_answers(correct={0, 1, 2}))
    run = _run(monkeypatch, _tasks(), base, tuned)
    assert run.result.exit_code == 1, (run.result.output, repr(run.result.exception))
    assert run.json("out.json")["decision"] == "SHIP"


# early checks: an adapter that changes nothing writes no file, so it is not asked about one


def test_cli_an_adapter_that_changes_nothing_is_not_refused_over_a_foreign_rows_file(
    monkeypatch,
):
    monkeypatch.setattr(ship_cmd, "_adapter_is_noop", lambda adapter: True)

    def put_foreign():
        Path("out.task-rows.jsonl").write_bytes(b'{"mine": 1}\n')

    base = _scripted(_answers(correct={0}))
    run = _run(monkeypatch, _tasks(), base, base, before=put_foreign)
    assert run.result.exit_code == 2, (run.result.output, repr(run.result.exception))
    assert run.json("out.json")["failed_rule"] == "adapter_noop"
    assert run.files["out.task-rows.jsonl"] == b'{"mine": 1}\n'


def test_cli_a_config_that_is_the_rows_file_is_refused_before_the_run(monkeypatch):
    called = []

    def generate(prompt):
        called.append(prompt)
        return "x"

    def put_config():
        Path("out.task-rows.jsonl").write_text(
            "base: owner/base\ndata: {train: train.jsonl}\n", encoding="utf-8"
        )

    run = _run(monkeypatch, _tasks(), generate, generate, "--config", "out.task-rows.jsonl",
               before=put_config)
    assert run.result.exit_code == 3, (run.result.output, repr(run.result.exception))
    assert "is the same file as --config" in run.plain
    assert called == []
    assert run.files["out.task-rows.jsonl"].startswith(b"base: owner/base")


# a path that cannot be checked is a named refusal, not a traceback


@pytest.mark.parametrize("error", [NotADirectoryError, PermissionError])
def test_a_path_that_cannot_be_checked_is_a_named_refusal_not_a_traceback(
    tmp_path, monkeypatch, error
):
    monkeypatch.chdir(tmp_path)
    real = os.lstat

    def broken(path, *args, **kwargs):
        if str(path).endswith(".task-rows.jsonl"):
            raise error("cannot look")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", broken)
    with pytest.raises(ValueError, match=rf"cannot be checked \({error.__name__}\)"):
        write_task_rows([_row()], "v.json")
    assert not list(tmp_path.glob("*.task-rows.jsonl"))


@pytest.mark.skipif(os.name == "nt", reason="Windows reports a path under a file as not found")
def test_a_path_under_a_regular_file_is_a_named_refusal_on_posix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("blocker").write_text("a file, not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match=r"cannot be checked \(NotADirectoryError\)"):
        write_task_rows([_row()], "blocker/out.json")


@pytest.mark.parametrize("error", [NotADirectoryError, PermissionError])
def test_cli_a_path_that_cannot_be_checked_is_exit_3_before_any_model_loads(monkeypatch, error):
    real = os.lstat

    def broken(path, *args, **kwargs):
        if str(path).endswith(".task-rows.jsonl"):
            raise error("cannot look")
        return real(path, *args, **kwargs)

    called = []

    def generate(prompt):
        called.append(prompt)
        return "x"

    monkeypatch.setattr(os, "lstat", broken)
    run = _run(monkeypatch, _tasks(), generate, generate)
    assert run.result.exit_code == 3, (run.result.output, repr(run.result.exception))
    assert f"cannot be checked ({error.__name__})" in run.plain
    assert called == []


# a file is ours only if its first line has exactly our keys


def _own_first_line(**changes) -> bytes:
    line = json.loads(rows_to_jsonl([_row()]))
    line.update(changes)
    return (json.dumps(line, separators=(",", ":")) + "\n").encode("ascii")


@pytest.mark.parametrize(
    "content",
    [
        _own_first_line(schema_version=99),  # a row of a version this one does not write
        _own_first_line(extra_key="mine"),  # our keys plus one of the user's own
        b'{"leg": "task", "schema_version": 1, "prompt": "mine", "expected": "mine"}\n',
        b'{"leg":"task","schema_version":99}\n',
    ],
    ids=["version-99", "extra-key", "user-jsonl-with-our-two-keys", "two-keys-version-99"],
)
def test_a_file_whose_first_line_is_not_exactly_ours_is_left_alone(
    tmp_path, monkeypatch, content
):
    monkeypatch.chdir(tmp_path)
    Path("v.task-rows.jsonl").write_bytes(content)
    with pytest.raises(ValueError, match="not a task-rows file written by soup ship"):
        write_task_rows([_row()], "v.json")
    assert Path("v.task-rows.jsonl").read_bytes() == content


@pytest.mark.parametrize("variant", ["bom", "crlf", "bom-and-crlf"])
def test_a_copy_of_our_own_file_with_a_bom_or_crlf_is_still_ours(tmp_path, monkeypatch, variant):
    monkeypatch.chdir(tmp_path)
    own = rows_to_jsonl([_row()]).encode("ascii")
    content = {
        "bom": b"\xef\xbb\xbf" + own,
        "crlf": own.replace(b"\n", b"\r\n"),
        "bom-and-crlf": b"\xef\xbb\xbf" + own.replace(b"\n", b"\r\n"),
    }[variant]
    Path("v.task-rows.jsonl").write_bytes(content)
    write_task_rows([_row(index=0, base_answer="again")], "v.json")
    assert b"again" in Path("v.task-rows.jsonl").read_bytes()


# the loader: huge numbers, long messages, open failures


HUGE = 10 ** 400


@pytest.mark.parametrize("value", [HUGE, -HUGE], ids=["huge", "huge-negative"])
def test_a_huge_integer_score_is_a_named_refusal_not_an_overflow(tmp_path, value):
    entries = _mutate(_good_lines(3), 1, "base_score", value)
    with pytest.raises(ValueError, match=r"task-rows line 2 base_score must be in \[0, 1\]"):
        load_task_rows(_write_lines(tmp_path, entries))


def test_a_huge_integer_is_not_a_row_score_or_a_mean_of_row_scores():
    with pytest.raises(ValueError, match="must be finite"):
        row_mean([HUGE])
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        build_task_rows(_leg([HUGE], value=0.5), _leg([0.5], prefix="t"), scoring_applies=True)


@pytest.mark.parametrize("key", ["base", "tuned"])
def test_a_huge_reported_number_is_a_named_refusal_not_an_overflow(tmp_path, key):
    path = _write_pair(tmp_path, **{key: HUGE})
    with pytest.raises(ValueError, match=f"the reported {key} must be a finite number"):
        load_task_rows_for(path)


def test_a_reported_number_that_is_not_finite_is_refused_by_name(tmp_path):
    path = _write_pair(tmp_path)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["task_win"]["base"] = float("inf")
    # json writes it as Infinity; the literal 1e999 is what parses to infinity without a constant
    Path(path).write_text(json.dumps(data).replace("Infinity", "1e999"), encoding="utf-8")
    with pytest.raises(ValueError, match="the reported base must be a finite number"):
        load_task_rows_for(path)


@pytest.mark.parametrize("block", [
    {"better": 1, "worse": 1, "same": 1, "n": HUGE},
    {"better": HUGE, "worse": 1, "same": 1, "n": 3},
    {"better": 1, "worse": 1, "same": 1, "n": True},
    {"better": 1, "worse": 1, "same": "1", "n": 3},
    {"better": 1, "worse": 1, "same": 1, "n": -3},
])
def test_a_bad_rows_block_is_refused_without_echoing_its_numbers(tmp_path, block):
    path = _write_pair(tmp_path)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["task_win"]["rows"] = block
    Path(path).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        load_task_rows_for(path)
    assert "the rows block of the verdict or evidence file" in str(caught.value)
    assert "0" * 50 not in str(caught.value) and len(str(caught.value)) < 200


def test_the_unknown_field_message_names_at_most_five_keys(tmp_path):
    entry = dict(_good_lines(1)[0])
    entry.update({f"k{i}": 1 for i in range(5000)})
    path = _write_lines(tmp_path, [entry])
    with pytest.raises(ValueError) as caught:
        load_task_rows(path)
    message = str(caught.value)
    assert message.startswith("task-rows line 1 has unsupported field(s): ")
    assert message.count("'k") == 5 and "and 4995 more" in message
    assert len(message) < 300


def test_an_open_failure_after_the_stat_is_a_named_refusal(tmp_path, monkeypatch):
    path = _write_lines(tmp_path, _good_lines(3))

    def broken(*args, **kwargs):
        raise PermissionError("no")

    monkeypatch.setattr(ship_task_rows, "open", broken, raising=False)
    with pytest.raises(ValueError, match="task-rows file cannot be read: PermissionError"):
        load_task_rows(path)


# the file name is limited in what the file system counts, not in characters


@pytest.mark.parametrize(
    ("anchor", "ok"),
    [
        ("x" * 230 + ".json", True),  # derived name 246: a name that worked before still works
        ("x" * 239 + ".json", True),  # derived name 255: exactly the limit
        ("x" * 240 + ".json", False),  # derived name 256: cannot exist
        ("x" * 245 + ".json", False),  # an --output name that worked, whose derived name cannot
    ],
)
def test_the_derived_name_is_refused_only_when_it_cannot_exist(anchor, ok):
    if ok:
        assert task_rows_path(anchor).endswith(TASK_ROWS_SUFFIX)
    else:
        with pytest.raises(ValueError, match="is too long"):
            task_rows_path(anchor)


@pytest.mark.skipif(os.name == "nt", reason="bytes on POSIX, UTF-16 units on Windows")
def test_on_posix_the_name_limit_counts_encoded_bytes_not_characters():
    assert task_rows_path("中" * 79 + ".json")  # 79 * 3 + 16 = 253 bytes
    with pytest.raises(ValueError, match="is too long"):
        task_rows_path("中" * 80 + ".json")  # 256 bytes, though only 96 characters


@pytest.mark.skipif(os.name != "nt", reason="bytes on POSIX, UTF-16 units on Windows")
def test_on_windows_the_name_limit_counts_utf16_units_not_characters():
    assert task_rows_path("中" * 100 + ".json")  # 100 + 16 units: far more bytes, still fine
    assert task_rows_path("\U0001f600" * 119 + ".json")  # 2 * 119 + 16 = 254 units
    with pytest.raises(ValueError, match="is too long"):
        task_rows_path("\U0001f600" * 120 + ".json")  # 256 units, though only 136 characters


# -- delta read: a score beyond float range, and what a refused name tells you to do ------------


def test_cli_a_judge_row_score_beyond_float_range_leaves_the_file_out_and_the_verdict_alone(
    monkeypatch,
):
    # an integer of 400 digits is not a float: ``math.isfinite`` overflows on it. A judge whose
    # rows carry such scores (a fake one: the real evaluator always returns a float) must end
    # exactly as it does without the file: base == tuned here, so DON'T SHIP, exit 2, no file.
    from types import SimpleNamespace

    from soup_cli.eval.judge import JudgeEvaluator

    monkeypatch.setattr("soup_cli.eval.gate._parse_judge_url", lambda url: ("ollama", "m", None))
    monkeypatch.setattr(
        JudgeEvaluator, "evaluate_batch",
        lambda self, items: SimpleNamespace(
            overall_score=3.0,
            scores=[
                SimpleNamespace(weighted_score=10 ** 400),
                SimpleNamespace(weighted_score=-(10 ** 400)),
            ][: len(items)],
        ),
    )
    base = _scripted({"q0": ("b0", False), "q1": ("b1", False)})
    tuned = _scripted({"q0": ("t0", False), "q1": ("t1", False)})
    run = _run(monkeypatch, _tasks(2), base, tuned,
               "--task-mode", "judge_score", "--judge-model", "ollama://judge")
    assert run.result.exit_code == 2, (run.result.output, repr(run.result.exception))
    assert run.result.exception is None or isinstance(run.result.exception, SystemExit)
    verdict = run.json("out.json")
    assert verdict["task_win"]["base"] == verdict["task_win"]["tuned"] == 0.5
    assert verdict["failed_rule"] == "task_win" and verdict["task_rows"] is None
    assert not [name for name in run.files if name.endswith(".task-rows.jsonl")]


def test_a_refused_name_says_what_to_do():
    with pytest.raises(ValueError, match="is too long.*use a shorter --output name"):
        task_rows_path("x" * 245 + ".json")
    with pytest.raises(ValueError, match="has a control character.*use a plain --output name"):
        task_rows_path("o\n.json")
