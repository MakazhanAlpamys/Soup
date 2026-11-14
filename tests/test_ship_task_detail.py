"""`soup ship` says what is inside the task leg: rows better / worse / same, answers cut off at
the token cap, generations that failed, and which adapter or checkpoint was judged.

The verdict of a run is the same with and without this detail (the one exception is a side
whose every generation failed, which is refused). Everything here runs on CPU with no model:
the pure functions take tuples of per-row scores and flags, and the CLI tests patch
``ship_cmd._resolve_generators`` with scripted generators.

Pinned fixture (the consumer's live counts): of 8 rows, 0 better, 5 worse, 3 the same, and one
of 8 answers cut off at 256 tokens.
"""

from __future__ import annotations

import json
import math
from io import StringIO
from pathlib import Path
from typing import Optional

import pytest
from rich.console import Console

from soup_cli.commands import ship as ship_cmd
from soup_cli.eval.gate_suites import BEHAVIOURAL_MAX_NEW_TOKENS
from soup_cli.utils.live_eval import GeneratedText
from soup_cli.utils.ship_verdict import (
    DECISION_DONT_SHIP,
    DECISION_SHIP,
    FAILED_TASK_WIN,
    TASK_AXIS,
    TRUNCATION_GROWTH_MIN_ROWS,
    TRUNCATION_GROWTH_MIN_SHARE,
    WARNING_TRUNCATION_GROWTH,
    GenerationFailures,
    Judged,
    LegScore,
    NoiseFloor,
    ShipVerdict,
    TaskDetail,
    TaskRows,
    TaskWin,
    Truncation,
    build_judged,
    build_task_detail,
    build_task_win,
    build_truncation,
    checkpoint_step_of,
    compare_rows,
    count_truncated,
    decide_ship,
    format_ship_rubric,
    render_ship_panel,
    truncation_growth_threshold,
    verdict_to_dict,
)
from tests.cli_runner import CliRunner
from tests.conftest import strip_ansi

runner = CliRunner()


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def _panel_text(verdict: ShipVerdict) -> str:
    buffer = StringIO()
    Console(file=buffer, width=100, force_terminal=False, color_system=None).print(
        render_ship_panel(verdict)
    )
    return buffer.getvalue()


# -- rows: better / worse / same -----------------------------------------------------------


def test_rows_say_how_many_rows_got_better_worse_and_stayed_the_same():
    rows = compare_rows((1, 0, 1, 0), (1, 1, 0, 0))
    assert rows == TaskRows(better=1, worse=1, same=2)
    assert rows.n == 4


def test_swapping_the_two_legs_swaps_better_and_worse():
    rows = compare_rows((1, 1, 1, 0), (0, 0, 0, 1))  # tuned worse on three, better on one
    assert (rows.better, rows.worse, rows.same) == (1, 3, 0)
    swapped = compare_rows((0, 0, 0, 1), (1, 1, 1, 0))
    assert (swapped.better, swapped.worse, swapped.same) == (3, 1, 0)


def test_the_consumers_counts_are_better_0_worse_5_same_3():
    base = (1, 1, 1, 1, 1, 1, 0, 0)
    tuned = (0, 0, 0, 0, 0, 1, 0, 0)
    rows = compare_rows(base, tuned)
    assert (rows.better, rows.worse, rows.same, rows.n) == (0, 5, 3, 8)


@pytest.mark.parametrize(
    ("tuned_delta", "expected"),
    [
        (1e-12, "same"),  # float noise is not a change
        (-1e-12, "same"),
        (5e-10, "same"),
        (1e-8, "better"),  # a real, tiny change is
        (-1e-8, "worse"),
    ],
)
def test_rows_ignore_float_noise_below_1e_9(tuned_delta, expected):
    rows = compare_rows((0.5,), (0.5 + tuned_delta,))
    assert getattr(rows, expected) == 1
    assert rows.n == 1


def test_better_plus_worse_plus_same_is_always_n():
    rows = compare_rows((0.0, 0.25, 0.5, 0.75, 1.0, 1.0), (1.0, 0.25, 0.4, 0.8, 1.0, 0.0))
    assert rows.better + rows.worse + rows.same == rows.n == 6


@pytest.mark.parametrize(
    "base,tuned,message",
    [
        ((1, 0), (1,), "same number of rows"),
        ((True, 0), (1, 0), "must not be bool"),
        ((float("nan"), 0), (1, 0), "must be finite"),
        (("a", 0), (1, 0), "must be a number"),
    ],
)
def test_rows_refuse_scores_that_are_not_comparable(base, tuned, message):
    with pytest.raises((TypeError, ValueError), match=message):
        compare_rows(base, tuned)


# -- truncation counts ----------------------------------------------------------------------


def test_a_count_is_the_number_of_answers_that_hit_the_cap():
    assert count_truncated((True, False, True, False)) == 2
    assert count_truncated((False, False)) == 0
    assert count_truncated(()) == 0


def test_one_unmeasured_answer_makes_the_count_unmeasured_and_never_zero():
    assert count_truncated((None, None, None)) is None
    assert count_truncated((False, None, False)) is None  # not 0: one answer is unknown
    assert count_truncated((True, None)) is None  # and not 1 either


def test_build_truncation_keeps_the_two_sides_apart():
    truncation = build_truncation((True,) * 5 + (False,) * 59, (True,) * 14 + (False,) * 50,
                                  max_new_tokens=256)
    assert truncation == Truncation(base=5, tuned=14, n=64, max_new_tokens=256)


def test_build_truncation_refuses_two_sides_of_different_length():
    with pytest.raises(ValueError, match="same number of rows"):
        build_truncation((True,), (True, False), max_new_tokens=256)


@pytest.mark.parametrize(
    ("n", "base", "tuned", "fires"),
    [
        # the competitor's run: 5 -> 14 of 64, threshold ceil(6.4) = 7
        (64, 5, 11, False),  # rise 6, one below
        (64, 5, 12, True),  # rise 7, exactly at
        (64, 5, 14, True),
        (64, 0, 6, False),
        (64, 0, 7, True),
        # small n: the floor of 3 rows
        (8, 0, 2, False),
        (8, 0, 3, True),
        (8, 0, 1, False),  # without the floor a 1-row rise would fire at n=8 (ceil(0.8) = 1)
        (10, 0, 3, True),
        (10, 0, 2, False),
        (11, 0, 2, False),  # ceil(1.1) = 2, but the floor is 3
        (11, 0, 3, True),
        # n = 30 is exactly 10 %: the threshold is 3, and n = 31 is the first that needs 4
        (30, 0, 2, False),
        (30, 0, 3, True),
        (31, 0, 3, False),  # ceil(3.1) = 4
        (31, 0, 4, True),
        (31, 1, 4, False),
        (31, 1, 5, True),
        # direction: a fall and a tie never fire
        (64, 14, 5, False),
        (64, 14, 14, False),
        (8, 5, 5, False),
    ],
)
def test_the_warning_fires_when_the_rise_reaches_max_3_or_a_tenth_of_n(n, base, tuned, fires):
    truncation = Truncation(base=base, tuned=tuned, n=n, max_new_tokens=256)
    assert truncation.warning == (WARNING_TRUNCATION_GROWTH if fires else None)


@pytest.mark.parametrize(
    ("n", "threshold"),
    [(1, 3), (8, 3), (10, 3), (11, 3), (30, 3), (31, 4), (40, 4), (41, 5), (64, 7), (100, 10)],
)
def test_the_threshold_table(n, threshold):
    assert truncation_growth_threshold(n) == threshold


def test_the_two_constants_are_the_documented_ones():
    assert TRUNCATION_GROWTH_MIN_ROWS == 3
    assert TRUNCATION_GROWTH_MIN_SHARE == 0.10
    assert WARNING_TRUNCATION_GROWTH == "truncation_growth"


def test_the_warning_reads_base_and_tuned_the_right_way_round():
    rose = Truncation(base=5, tuned=14, n=64, max_new_tokens=256)
    fell = Truncation(base=14, tuned=5, n=64, max_new_tokens=256)
    assert rose.warning == WARNING_TRUNCATION_GROWTH
    assert fell.warning is None


@pytest.mark.parametrize("base,tuned", [(None, 14), (5, None), (None, None)])
def test_one_side_not_measured_means_no_warning_and_the_counts_stay(base, tuned):
    truncation = Truncation(base=base, tuned=tuned, n=64, max_new_tokens=256)
    assert truncation.warning is None
    assert (truncation.base, truncation.tuned) == (base, tuned)


# -- judged: which adapter / checkpoint -----------------------------------------------------


@pytest.mark.parametrize(
    ("adapter", "step"),
    [
        ("checkpoint-12", 12),
        ("checkpoint-0", 0),
        ("output/run1/checkpoint-300", 300),
        ("output/run1/checkpoint-300/", 300),
        ("output\\run1\\checkpoint-300", 300),
        ("output\\run1\\checkpoint-300\\", 300),
        ("/abs/checkpoint-7", 7),
        # never guessed: anything that is not exactly checkpoint-<digits>
        ("checkpoint-12-old", None),
        ("checkpoint-12\n", None),  # `$` would accept a trailing newline
        ("checkpoint-12 ", None),
        ("ckpt-12", None),
        ("checkpoint-", None),
        ("checkpoint", None),
        ("Checkpoint-12", None),
        ("my-checkpoint-12", None),
        ("checkpoint-1_2", None),
        ("checkpoint--12", None),
        ("checkpoint-١٢", None),  # Arabic-Indic digits are not [0-9]
        ("checkpoint-" + "9" * 5000, None),  # no int() on an unbounded digit string
        ("checkpoint-12/final", None),  # the LAST component decides
        ("", None),
        (None, None),
    ],
)
def test_checkpoint_step_is_read_from_the_last_path_component_only(adapter, step):
    assert checkpoint_step_of(adapter) == step


def test_judged_for_an_adapter_run_carries_the_strings_as_given():
    judged = build_judged(base="Qwen/Qwen3-0.6B", adapter="out/checkpoint-12")
    assert judged == Judged(
        base="Qwen/Qwen3-0.6B", adapter="out/checkpoint-12", tuned=None, base_adapter=None,
        checkpoint_step=12,
    )


def test_judged_for_a_tuned_model_run_has_no_checkpoint_step():
    judged = build_judged(base="m", tuned="checkpoint-12")  # a model id, not --adapter
    assert judged.tuned == "checkpoint-12"
    assert judged.adapter is None
    assert judged.checkpoint_step is None


def test_judged_for_a_base_adapter_run_names_both_adapters():
    judged = build_judged(base="m", adapter="new/checkpoint-5", base_adapter="prod")
    assert (judged.adapter, judged.base_adapter, judged.checkpoint_step) == (
        "new/checkpoint-5", "prod", 5,
    )


# -- the detail survives decide_ship, rendering, and the dict --------------------------------


def _detail(*, rows=None, truncated=None, failures=None) -> TaskDetail:
    return TaskDetail(rows=rows, truncated=truncated, generation_failures=failures)


def test_detail_survives_decide_ship_when_the_floor_flips_won():
    detail = _detail(
        rows=TaskRows(better=1, worse=0, same=7),
        truncated=Truncation(base=0, tuned=1, n=8, max_new_tokens=256),
    )
    win = TaskWin(mode="metric", base=0.50, tuned=0.54, won=True, detail=detail)
    floor = NoiseFloor(runs=2, floors=((TASK_AXIS, 0.10),))
    verdict = decide_ship(win, [], noise_floor=floor)
    assert verdict.task_win.won is False  # the floor flipped it: the canonical copy was built
    assert verdict.task_win.detail is detail  # and it kept the detail
    assert verdict.task_win.mode == "metric"


def test_detail_survives_decide_ship_when_nothing_flips():
    detail = _detail(rows=TaskRows(better=2, worse=0, same=0))
    win = build_task_win("metric", 0.5, 1.0, detail=detail)
    verdict = decide_ship(win, [])
    assert verdict.task_win.detail is detail


def test_a_taskwin_without_detail_renders_and_serialises_as_before():
    win = build_task_win("metric", 0.5, 0.75)
    assert win.detail is None
    verdict = decide_ship(win, [])
    data = verdict_to_dict(verdict)
    assert data["task_win"]["rows"] is None
    assert data["task_win"]["truncated"] is None
    assert data["task_win"]["generation_failures"] is None
    assert data["judged"] is None
    text = format_ship_rubric(verdict)
    for word in ("Rows", "Truncated", "Generation failures", "Judged"):
        assert word not in text


def _verdict_with(detail: Optional[TaskDetail], judged: Optional[Judged] = None) -> ShipVerdict:
    win = build_task_win("metric", 0.75, 0.125, detail=detail)
    from dataclasses import replace

    return replace(decide_ship(win, []), judged=judged)


CONSUMER_DETAIL = TaskDetail(
    rows=TaskRows(better=0, worse=5, same=3),
    truncated=Truncation(base=0, tuned=1, n=8, max_new_tokens=256),
    generation_failures=GenerationFailures(base=0, tuned=0, n=8),
)


def test_the_dict_carries_rows_truncated_and_failures_with_every_old_key():
    data = verdict_to_dict(_verdict_with(CONSUMER_DETAIL, build_judged(base="m", adapter="a")))
    assert data["task_win"]["rows"] == {"better": 0, "worse": 5, "same": 3, "n": 8}
    assert data["task_win"]["truncated"] == {
        "base": 0, "tuned": 1, "n": 8, "max_new_tokens": 256, "warning": None,
    }
    assert data["task_win"]["generation_failures"] == {"base": 0, "tuned": 0, "n": 8}
    assert {"mode", "base", "tuned", "won"} <= set(data["task_win"])
    assert data["schema_version"] == 2
    assert data["judged"] == {
        "base": "m", "adapter": "a", "tuned": None, "base_adapter": None, "checkpoint_step": None,
    }
    assert json.loads(json.dumps(data)) == data  # plain JSON


def test_the_panel_and_the_rubric_print_the_counts_the_consumer_asked_for():
    verdict = _verdict_with(CONSUMER_DETAIL, build_judged(base="m", adapter="out/checkpoint-12"))
    panel = _plain(_panel_text(verdict))
    rubric = _plain(format_ship_rubric(verdict))
    for text in (panel, rubric):
        assert "Rows (tuned vs base): better 0, worse 5, same 3 of 8" in text
        assert "Truncated answers: base 0, tuned 1 of 8 (cap 256 new tokens)" in text
        assert "Generation failures: base 0, tuned 0 of 8" in text
        assert "Judged: base m, adapter out/checkpoint-12 (checkpoint step 12)" in text
        assert WARNING_TRUNCATION_GROWTH not in text


def test_the_panel_prints_the_five_to_fourteen_of_sixty_four_warning():
    detail = _detail(truncated=Truncation(base=5, tuned=14, n=64, max_new_tokens=256))
    verdict = _verdict_with(detail)
    for text in (_plain(_panel_text(verdict)), _plain(format_ship_rubric(verdict))):
        assert "Truncated answers: base 5, tuned 14 of 64 (cap 256 new tokens)" in text
        assert "Warning truncation_growth: truncated 5 -> 14 of 64 (rise 9, threshold 7)" in text
    assert verdict_to_dict(verdict)["task_win"]["truncated"]["warning"] == "truncation_growth"


def test_a_count_that_was_not_measured_prints_as_not_measured_not_as_zero():
    detail = _detail(truncated=Truncation(base=None, tuned=None, n=8, max_new_tokens=256))
    text = _plain(_panel_text(_verdict_with(detail)))
    assert "Truncated answers: base not measured, tuned not measured of 8" in text
    assert "Truncated answers: base 0" not in text
    data = verdict_to_dict(_verdict_with(detail))
    assert data["task_win"]["truncated"]["base"] is None
    assert data["task_win"]["truncated"]["tuned"] is None


def test_failed_generations_are_shown_as_a_warning_line_when_there_are_any():
    detail = _detail(failures=GenerationFailures(base=0, tuned=2, n=8))
    text = _plain(_panel_text(_verdict_with(detail)))
    assert "Generation failures: base 0, tuned 2 of 8 (scored as wrong)" in text


def test_a_hostile_adapter_path_is_printed_through_for_terminal():
    hostile = "out/\x1b[2Jevil/\x07bell/checkpoint-3"
    verdict = _verdict_with(None, build_judged(base="m", adapter=hostile))
    for text in (_panel_text(verdict), format_ship_rubric(verdict)):
        assert "\x1b" not in text
        assert "\x07" not in text
        assert "checkpoint-3" in text
    # the JSON keeps the string exactly as given
    assert verdict_to_dict(verdict)["judged"]["adapter"] == hostile


# -- LegScore: a float that remembers where it came from --------------------------------------


def test_a_leg_score_is_a_float_in_every_respect():
    score = LegScore(0.75, row_scores=(1.0, 0.5), truncated=(False, None), failures=0)
    assert score == 0.75 and isinstance(score, float)
    assert json.dumps({"s": score}) == '{"s": 0.75}'
    assert build_task_win("metric", score, 0.5).base == 0.75
    assert type(float(score)) is float


def test_build_task_detail_joins_the_two_legs():
    base = LegScore(0.75, row_scores=(1, 1, 0), truncated=(False, False, False), failures=0)
    tuned = LegScore(0.25, row_scores=(1, 0, 0), truncated=(False, True, True), failures=1)
    detail = build_task_detail(base, tuned, max_new_tokens=256)
    assert detail == TaskDetail(
        rows=TaskRows(better=0, worse=1, same=2),
        truncated=Truncation(base=0, tuned=2, n=3, max_new_tokens=256),
        generation_failures=GenerationFailures(base=0, tuned=1, n=3),
    )


def test_build_task_detail_is_none_for_plain_floats_and_partial_legs_stay_partial():
    assert build_task_detail(0.5, 0.75, max_new_tokens=256) is None
    only_flags = LegScore(0.5, truncated=(False, False))
    detail = build_task_detail(only_flags, LegScore(0.7, truncated=(True, True)),
                               max_new_tokens=256)
    assert detail.rows is None
    assert detail.truncated == Truncation(base=0, tuned=2, n=2, max_new_tokens=256)
    assert detail.generation_failures is None


# -- the CLI, with scripted generators -------------------------------------------------------

_PLAIN_STR = object()  # a generator that returns a bare str: truncation not measured


def _tasks(count: int = 8, **extra) -> list:
    return [{"prompt": f"q{i}", "expected": f"a{i}", **extra} for i in range(count)]


def _scripted(answers: dict, *, fail=(), default="B"):
    """``answers``: prompt -> text, or (text, truncated) with truncated True / False / _PLAIN_STR.
    A prompt in ``fail`` raises; every other prompt (the bundled suites) answers ``default``."""

    def generate(prompt):
        if prompt in fail:
            raise RuntimeError("generation exploded")
        if prompt not in answers:
            return default
        entry = answers[prompt]
        text, flag = entry if isinstance(entry, tuple) else (entry, False)
        return text if flag is _PLAIN_STR else GeneratedText(text, truncated=flag)

    return generate


def _answers(correct, *, count: int = 8, truncated=()):
    """Row i answers a<i> when i is in ``correct`` and 'nope' otherwise; rows in ``truncated``
    are flagged cut off."""
    out = {}
    for index in range(count):
        text = f"a{index}" if index in correct else "nope"
        out[f"q{index}"] = (text, index in truncated)
    return out


def _ship(monkeypatch, tasks, base_gen, tuned_gen, *extra, adapter="out/checkpoint-12"):
    def _fake_resolve(base, tuned, adapter, device, quantization=None, **kwargs):
        return base_gen, tuned_gen

    monkeypatch.setattr(ship_cmd, "_resolve_generators", _fake_resolve)
    with runner.isolated_filesystem():
        Path("task.jsonl").write_text(
            "\n".join(json.dumps(task) for task in tasks) + "\n", encoding="utf-8"
        )
        flags = ["--tuned", "tuned-model"] if adapter is None else ["--adapter", adapter]
        result = runner.invoke(
            ship_cmd.app,
            ["--base", "base-model", *flags, "--task-eval", "task.jsonl",
             "--general-suite", "mini_arithmetic", "--device", "cpu", "--output", "out.json",
             "--emit-evidence", "ev.json", *extra],
        )
        out = Path("out.json")
        ev = Path("ev.json")
        verdict = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
        evidence = json.loads(ev.read_text(encoding="utf-8")) if ev.exists() else None
    return result, verdict, evidence


def test_cli_pins_the_consumers_live_counts(monkeypatch):
    # of 8 rows: base right on 0-5, tuned right on 5 only -> 0 better, 5 worse, 3 same;
    # one of the 8 tuned answers (row 6) hit the 256-token cap
    base = _scripted(_answers(correct={0, 1, 2, 3, 4, 5}))
    tuned = _scripted(_answers(correct={5}, truncated={6}))
    result, verdict, _ = _ship(monkeypatch, _tasks(), base, tuned)
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert verdict["decision"] == DECISION_DONT_SHIP
    assert verdict["failed_rule"] == FAILED_TASK_WIN
    assert verdict["task_win"]["base"] == 0.75
    assert verdict["task_win"]["tuned"] == 0.125
    assert verdict["task_win"]["rows"] == {"better": 0, "worse": 5, "same": 3, "n": 8}
    assert verdict["task_win"]["truncated"] == {
        "base": 0, "tuned": 1, "n": 8, "max_new_tokens": BEHAVIOURAL_MAX_NEW_TOKENS,
        "warning": None,
    }
    assert BEHAVIOURAL_MAX_NEW_TOKENS == 256
    assert verdict["task_win"]["generation_failures"] == {"base": 0, "tuned": 0, "n": 8}
    plain = _plain(result.output)
    assert "Rows (tuned vs base): better 0, worse 5, same 3 of 8" in plain
    assert "Truncated answers: base 0, tuned 1 of 8 (cap 256 new tokens)" in plain
    assert "Generation failures: base 0, tuned 0 of 8" in plain


def test_cli_counts_truncation_from_the_generated_text_flag_only(monkeypatch):
    long_but_finished = "a0 " + "word " * 400  # 2000 characters, but the model stopped on its own
    base = _scripted({"q0": ("a0", False), "q1": ("a1", False)})
    tuned = _scripted({"q0": (long_but_finished, False), "q1": ("a1", True)})
    tasks = [{"prompt": "q0", "expected": "a0", "scoring": "contains"},
             {"prompt": "q1", "expected": "a1"}]
    result, verdict, _ = _ship(monkeypatch, tasks, base, tuned)
    assert result.exit_code in (0, 2), (result.output, repr(result.exception))
    assert verdict["task_win"]["truncated"]["tuned"] == 1  # the flag, not the length
    assert verdict["task_win"]["truncated"]["base"] == 0


def test_cli_a_plain_str_generator_leaves_both_counts_null_not_zero(monkeypatch):
    plain_base = _scripted({k: (v[0], _PLAIN_STR) for k, v in _answers(correct={0, 1}).items()})
    plain_tuned = _scripted({k: (v[0], _PLAIN_STR) for k, v in _answers(correct={0}).items()})
    result, verdict, evidence = _ship(monkeypatch, _tasks(), plain_base, plain_tuned)
    assert result.exit_code == 2, (result.output, repr(result.exception))
    block = verdict["task_win"]["truncated"]
    assert block["base"] is None and block["tuned"] is None  # not measured is never 0
    assert block["n"] == 8
    assert block["warning"] is None
    assert verdict["task_win"]["rows"] == {"better": 0, "worse": 1, "same": 7, "n": 8}
    assert evidence["task"]["truncated"]["base"] is None
    assert "Truncated answers: base not measured, tuned not measured of 8" in _plain(
        result.output
    )


def test_cli_one_plain_str_answer_makes_that_side_not_measured(monkeypatch):
    answers = _answers(correct=set(range(8)))
    answers["q3"] = ("a3", _PLAIN_STR)
    base = _scripted(answers)
    tuned = _scripted(_answers(correct=set(range(8)), truncated={0}))
    result, verdict, _ = _ship(monkeypatch, _tasks(), base, tuned)
    block = verdict["task_win"]["truncated"]
    assert block["base"] is None
    assert block["tuned"] == 1
    assert block["warning"] is None  # one side unmeasured: counts kept, no warning


@pytest.mark.parametrize(
    ("truncated_rows", "warns"),
    [({0, 1}, False), ({0, 1, 2}, True), (set(range(8)), True)],
)
def test_cli_the_warning_never_changes_a_ship_or_the_exit_code(
    monkeypatch, truncated_rows, warns
):
    # the tuned model is right on every row: SHIP, with or without the warning
    base = _scripted(_answers(correct=set()))
    tuned = _scripted(_answers(correct=set(range(8)), truncated=truncated_rows))
    result, verdict, _ = _ship(monkeypatch, _tasks(), base, tuned)
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert verdict["decision"] == DECISION_SHIP
    assert verdict["failed_rule"] is None
    expected = WARNING_TRUNCATION_GROWTH if warns else None
    assert verdict["task_win"]["truncated"]["warning"] == expected
    assert (WARNING_TRUNCATION_GROWTH in _plain(result.output)) == warns


@pytest.mark.parametrize("truncated_rows", [{0, 1}, {0, 1, 2}, set(range(8))])
def test_cli_a_dont_ship_with_the_warning_stays_exit_2(monkeypatch, truncated_rows):
    base = _scripted(_answers(correct=set(range(8))))
    tuned = _scripted(_answers(correct=set(), truncated=truncated_rows))
    result, verdict, _ = _ship(monkeypatch, _tasks(), base, tuned)
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert verdict["decision"] == DECISION_DONT_SHIP
    assert verdict["failed_rule"] == FAILED_TASK_WIN


def test_cli_the_warning_is_computed_not_stored_so_the_verdict_never_depends_on_it(monkeypatch):
    # the same scores with and without a rise in truncation give the same decision
    base = _scripted(_answers(correct={0, 1, 2}))
    quiet = _scripted(_answers(correct={0, 1, 2, 3}))
    loud = _scripted(_answers(correct={0, 1, 2, 3}, truncated={4, 5, 6, 7}))
    _, quiet_verdict, _ = _ship(monkeypatch, _tasks(), base, quiet)
    _, loud_verdict, _ = _ship(monkeypatch, _tasks(), base, loud)
    for key in ("decision", "failed_rule", "benchmark_deltas"):
        assert quiet_verdict[key] == loud_verdict[key]
    assert quiet_verdict["task_win"]["truncated"]["warning"] is None
    assert loud_verdict["task_win"]["truncated"]["warning"] == WARNING_TRUNCATION_GROWTH


def test_cli_metric_rows_use_matched_not_the_fractional_semantic_score(monkeypatch):
    # semantic scoring: matched is similarity >= 0.5, score is the Jaccard similarity itself
    tasks = [
        {"prompt": "r0", "expected": "alpha beta gamma delta", "scoring": "semantic"},
        {"prompt": "r1", "expected": "alpha beta gamma delta", "scoring": "semantic"},
        {"prompt": "r2", "expected": "alpha beta gamma delta", "scoring": "semantic"},
    ]
    base = _scripted({"r0": "alpha beta gamma", "r1": "alpha", "r2": "alpha beta gamma delta"})
    tuned = _scripted({"r0": "alpha beta gamma delta", "r1": "alpha beta",
                       "r2": "alpha beta gamma"})
    result, verdict, _ = _ship(monkeypatch, tasks, base, tuned)
    assert result.exit_code in (0, 2), (result.output, repr(result.exception))
    # by matched: r0 both match (same), r1 base no / tuned yes (better), r2 both match (same);
    # by the raw score r0 would be better and r2 worse
    assert verdict["task_win"]["rows"] == {"better": 1, "worse": 0, "same": 2, "n": 3}
    assert verdict["task_win"]["base"] == pytest.approx(2 / 3)
    assert verdict["task_win"]["tuned"] == pytest.approx(1.0)


def test_cli_judged_carries_the_command_line_strings_and_the_checkpoint_step(monkeypatch):
    base = _scripted(_answers(correct=set()))
    tuned = _scripted(_answers(correct={0}))
    result, verdict, evidence = _ship(
        monkeypatch, _tasks(), base, tuned, adapter="runs/exp 1/checkpoint-12",
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    expected = {
        "base": "base-model", "adapter": "runs/exp 1/checkpoint-12", "tuned": None,
        "base_adapter": None, "checkpoint_step": 12,
    }
    assert verdict["judged"] == expected
    assert evidence["judged"] == expected
    assert "Judged: base base-model, adapter runs/exp 1/checkpoint-12" in _plain(result.output)


@pytest.mark.parametrize(
    "adapter", ["checkpoint-12-old", "ckpt-12", "checkpoint-", "out/final", "checkpoint-12/x"],
)
def test_cli_judged_checkpoint_step_is_null_unless_the_last_component_says_so(
    monkeypatch, adapter
):
    base = _scripted(_answers(correct=set()))
    tuned = _scripted(_answers(correct={0}))
    result, verdict, _ = _ship(monkeypatch, _tasks(), base, tuned, adapter=adapter)
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert verdict["judged"]["adapter"] == adapter
    assert verdict["judged"]["checkpoint_step"] is None


def test_cli_judged_for_a_tuned_model_run_has_no_adapter_and_no_step(monkeypatch):
    base = _scripted(_answers(correct=set()))
    tuned = _scripted(_answers(correct={0}))
    result, verdict, _ = _ship(monkeypatch, _tasks(), base, tuned, adapter=None)
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert verdict["judged"] == {
        "base": "base-model", "adapter": None, "tuned": "tuned-model", "base_adapter": None,
        "checkpoint_step": None,
    }


def test_cli_a_hostile_adapter_path_never_reaches_the_terminal_raw(monkeypatch):
    hostile = "out/\x1b[31mred\x07/checkpoint-4"
    base = _scripted(_answers(correct=set()))
    tuned = _scripted(_answers(correct={0}))
    result, verdict, _ = _ship(monkeypatch, _tasks(), base, tuned, adapter=hostile)
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "\x1b[31mred" not in result.output
    assert "\x07" not in result.output
    assert verdict["judged"]["adapter"] == hostile  # the file is exact
    assert verdict["judged"]["checkpoint_step"] == 4


def test_cli_the_output_keeps_every_key_it_had_and_gains_the_new_ones(monkeypatch):
    base = _scripted(_answers(correct=set()))
    tuned = _scripted(_answers(correct={0}))
    result, verdict, _ = _ship(monkeypatch, _tasks(), base, tuned)
    assert result.exit_code == 0, (result.output, repr(result.exception))
    old_keys = {
        "decision", "task_win", "benchmark_deltas", "failed_rule", "forgetting_threshold",
        "soup_version", "noise_floor", "numerics", "adapter_change",
    }
    assert old_keys <= set(verdict)
    assert {"schema_version", "judged"} <= set(verdict)
    assert {"mode", "base", "tuned", "won"} <= set(verdict["task_win"])
    assert {"rows", "truncated", "generation_failures"} <= set(verdict["task_win"])
    assert verdict["schema_version"] == 2


# -- generation failures ----------------------------------------------------------------------


def test_cli_failed_generations_are_counted_and_shown(monkeypatch):
    base = _scripted(_answers(correct=set()))
    tuned = _scripted(_answers(correct={0, 1, 2}), fail={"q3", "q4"})
    result, verdict, evidence = _ship(monkeypatch, _tasks(), base, tuned)
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert verdict["task_win"]["generation_failures"] == {"base": 0, "tuned": 2, "n": 8}
    assert evidence["task"]["generation_failures"] == {"base": 0, "tuned": 2, "n": 8}
    # a failed row is no answer: its truncation is unknown, so that count is not measured
    assert verdict["task_win"]["truncated"]["tuned"] is None
    assert verdict["task_win"]["truncated"]["base"] == 0
    assert "Generation failures: base 0, tuned 2 of 8 (scored as wrong)" in _plain(result.output)


def test_cli_an_empty_answer_is_an_answer_not_a_failure(monkeypatch):
    base = _scripted(_answers(correct={0, 1}))
    tuned = _scripted({f"q{i}": ("", False) for i in range(8)})
    result, verdict, _ = _ship(monkeypatch, _tasks(), base, tuned)
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert verdict["task_win"]["generation_failures"] == {"base": 0, "tuned": 0, "n": 8}
    assert verdict["task_win"]["truncated"]["tuned"] == 0


@pytest.mark.parametrize("side", ["base", "tuned"])
def test_cli_a_side_whose_every_generation_failed_is_refused_not_scored_zero(
    monkeypatch, side
):
    every_prompt = {f"q{i}" for i in range(8)}
    good = _scripted(_answers(correct={0, 1, 2}))
    dead = _scripted(_answers(correct=set()), fail=every_prompt)
    base, tuned = (dead, good) if side == "base" else (good, dead)
    result, verdict, evidence = _ship(monkeypatch, _tasks(), base, tuned)
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert verdict is None and evidence is None  # no verdict is written
    plain = _plain(result.output)
    assert f"every task generation failed on the {side} side (8 of 8)" in plain
    assert "not a score of 0" in plain


def test_cli_a_dead_base_cannot_make_any_tuned_model_ship(monkeypatch):
    # before this was refused, base accuracy read 0.0 and any tuned model beat it
    every_prompt = {f"q{i}" for i in range(8)}
    dead = _scripted(_answers(correct=set()), fail=every_prompt)
    tuned = _scripted(_answers(correct={0}))
    result, verdict, _ = _ship(monkeypatch, _tasks(), dead, tuned)
    assert result.exit_code != 0
    assert result.exit_code == 1
    assert verdict is None


# -- the adapter that changes nothing ---------------------------------------------------------


def test_adapter_noop_has_no_detail_but_says_what_was_judged(monkeypatch):
    from tests.test_ship_adapter_noop import _lora, _run

    res, loads, out, evidence_written = _run(
        monkeypatch, _lora(0.0), "--emit-evidence", "ev.json"
    )
    assert res.exit_code == 2, (res.output, repr(res.exception))
    assert loads == []
    assert out["failed_rule"] == "adapter_noop"
    assert out["task_win"]["rows"] is None
    assert out["task_win"]["truncated"] is None
    assert out["task_win"]["generation_failures"] is None
    assert out["schema_version"] == 2
    assert out["judged"] == {
        "base": "m", "adapter": "lora", "tuned": None, "base_adapter": None,
        "checkpoint_step": None,
    }
    assert not evidence_written  # nothing was measured, so no evidence either


# -- the leg builders: judge_score and pairwise ------------------------------------------------


class _Task:
    def __init__(self, prompt):
        self.prompt = prompt
        self.category = "default"


def _patch_judge(monkeypatch, tasks, scorer):
    from soup_cli.eval.judge import JudgeEvaluator, JudgeScore

    monkeypatch.setattr("soup_cli.eval.custom.load_eval_tasks", lambda path: tasks)
    monkeypatch.setattr("soup_cli.eval.gate._parse_judge_url", lambda url: ("ollama", "m", None))

    def evaluate(self, prompt, response, category="default"):
        return JudgeScore(prompt, response, weighted_score=scorer(prompt, response))

    monkeypatch.setattr(JudgeEvaluator, "evaluate", evaluate)


def test_judge_score_rows_compare_the_per_row_judge_scores(monkeypatch):
    prompts = ["p0", "p1", "p2", "p3"]
    _patch_judge(
        monkeypatch, [_Task(prompt) for prompt in prompts],
        lambda prompt, response: {"bad": 1.0, "ok": 3.0, "good": 5.0}[str(response)],
    )
    answers_base = dict(zip(prompts, ["ok", "ok", "good", "bad"]))
    answers_tuned = dict(zip(prompts, ["good", "ok", "bad", "bad"]))

    def gen(table, flags):
        return lambda prompt: GeneratedText(table[prompt], truncated=flags[prompt])

    base_gen = gen(answers_base, {p: False for p in prompts})
    tuned_gen = gen(answers_tuned, {"p0": False, "p1": True, "p2": True, "p3": True})
    win = ship_cmd._leg1_judge(base_gen, tuned_gen, "t.jsonl", "ollama://j")
    assert win.detail.rows == TaskRows(better=1, worse=1, same=2)
    assert win.detail.truncated == Truncation(
        base=0, tuned=3, n=4, max_new_tokens=BEHAVIOURAL_MAX_NEW_TOKENS,
    )
    assert win.detail.truncated.warning == WARNING_TRUNCATION_GROWTH  # 3 rows reach the floor
    assert win.detail.generation_failures is None  # a generation error raises here, not swallowed
    # the verdict numbers are the ones the leg always gave: the mean judge score on the 1-5
    # rubric, rescaled to [0, 1]
    assert win.base == pytest.approx((3.0 - 1) / 4)  # mean(3, 3, 5, 1) = 3.0
    assert win.tuned == pytest.approx((2.5 - 1) / 4)  # mean(5, 3, 1, 1) = 2.5


def test_judge_score_with_plain_str_generators_has_rows_but_unmeasured_truncation(monkeypatch):
    prompts = ["p0", "p1"]
    _patch_judge(monkeypatch, [_Task(prompt) for prompt in prompts],
                 lambda prompt, response: 5.0 if response == "good" else 1.0)
    win = ship_cmd._leg1_judge(
        lambda prompt: "bad", lambda prompt: "good", "t.jsonl", "ollama://j",
    )
    assert win.detail.rows == TaskRows(better=2, worse=0, same=0)
    assert win.detail.truncated.base is None and win.detail.truncated.tuned is None


def test_pairwise_has_no_rows_but_counts_truncation_from_both_answers(monkeypatch):
    from soup_cli.eval.judge import JudgeEvaluator

    monkeypatch.setattr("soup_cli.eval.custom.load_eval_tasks",
                        lambda path: [_Task(f"p{i}") for i in range(4)])
    monkeypatch.setattr("soup_cli.eval.gate._parse_judge_url", lambda url: ("ollama", "m", None))
    monkeypatch.setattr(JudgeEvaluator, "compare_pair", lambda *args, **kwargs: 1)
    win = ship_cmd._leg1_pairwise(
        lambda prompt: GeneratedText("short", truncated=False),
        lambda prompt: GeneratedText("long", truncated=prompt in ("p0", "p1", "p2")),
        "t.jsonl", "ollama://j",
    )
    assert win.mode == "pairwise"
    assert win.detail.rows is None  # no per-row scores until the judge rows are recorded
    assert win.detail.truncated == Truncation(
        base=0, tuned=3, n=4, max_new_tokens=BEHAVIOURAL_MAX_NEW_TOKENS,
    )


def test_a_fake_scorer_that_returns_a_plain_float_still_works(monkeypatch):
    # tests (and callers) that stub the scorers return a bare float: no detail, no crash
    monkeypatch.setattr(ship_cmd, "_build_pairwise_scorer",
                        lambda task_eval, judge_model: (lambda gen_a, gen_b: 0.7))
    win = ship_cmd._leg1_pairwise(lambda p: "a", lambda p: "b", "t.jsonl", "ollama://j")
    assert win.tuned == 0.7 and win.detail is None
    monkeypatch.setattr(ship_cmd, "_build_judge_scorer",
                        lambda task_eval, judge_model: (lambda gen: 0.4))
    win = ship_cmd._leg1_judge(lambda p: "a", lambda p: "b", "t.jsonl", "ollama://j")
    assert win.base == 0.4 and win.detail is None


def test_the_threshold_is_exact_for_every_n_up_to_a_thousand():
    for n in range(1, 1001):
        assert truncation_growth_threshold(n) == max(3, -(-n // 10)), n


def test_the_threshold_uses_exact_arithmetic_so_a_changed_share_cannot_slip_a_boundary(
    monkeypatch,
):
    # 0.07 * 100 is 7.000000000000001 as a float, whose ceiling is 8; 7 % of 100 rows is 7
    from soup_cli.utils import ship_verdict

    monkeypatch.setattr(ship_verdict, "TRUNCATION_GROWTH_MIN_SHARE", 0.07)
    assert math.ceil(0.07 * 100) == 8  # the float trap is real for this share
    assert truncation_growth_threshold(100) == 7
    assert truncation_growth_threshold(200) == 14


def test_judge_score_without_per_row_scores_has_flags_but_no_rows(monkeypatch):
    # an evaluator whose batch result carries no per-row scores (only an overall score)
    from soup_cli.eval.judge import JudgeEvaluator

    monkeypatch.setattr("soup_cli.eval.custom.load_eval_tasks",
                        lambda path: [_Task("p0"), _Task("p1")])
    monkeypatch.setattr("soup_cli.eval.gate._parse_judge_url", lambda url: ("ollama", "m", None))

    class _Overall:
        overall_score = 3.0

    monkeypatch.setattr(JudgeEvaluator, "evaluate_batch", lambda self, items: _Overall())
    win = ship_cmd._leg1_judge(
        lambda prompt: GeneratedText("x", truncated=False),
        lambda prompt: GeneratedText("y", truncated=True),
        "t.jsonl", "ollama://j",
    )
    assert win.detail.rows is None
    assert win.detail.truncated == Truncation(
        base=0, tuned=2, n=2, max_new_tokens=BEHAVIOURAL_MAX_NEW_TOKENS,
    )


def test_run_eval_marks_a_generation_that_raised_as_failed_and_an_empty_answer_as_not():
    from soup_cli.eval.custom import EvalTask, run_eval

    def generate(prompt):
        if prompt == "boom":
            raise RuntimeError("no")
        return ""  # an answer, however empty

    results = run_eval(
        "m", [EvalTask(prompt="boom", expected="x"), EvalTask(prompt="quiet", expected="x")],
        generate_fn=generate,
    ).results
    assert [row.failed for row in results] == [True, False]
    assert [row.matched for row in results] == [False, False]
    assert [row.output for row in results] == ["", ""]
