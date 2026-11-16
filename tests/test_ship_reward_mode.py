"""``soup ship --task-mode reward``: leg 1 scored by a reward function (float 0..1 per row).

The generated text of both models is scored row by row by ``--reward-fn`` (``accuracy``,
``format``, ``verifiable:math``, ``verifiable:json_schema`` or the path of a ``.py`` file), and the
verdict is decided on the MEAN of the per-row floats with the same strict rule as every other mode
(tuned > base + noise floor). What this file pins:

* what the function is handed and what it may return (one finite number in [0, 1] per row; anything
  else is a named refusal that names the row, never a verdict);
* ties are ties: the mean is the exact mean of the decimal numbers in the per-row file, so the same
  numbers in another order, or numbers that add up to the same decimal, never become a win by one
  rounding;
* a ``.py`` reward file is CODE: it runs only when ``--reward-fn`` (or ``eval.ship.reward_fn``)
  names it, only from under the working directory, never through a link, never on an offline
  replay, and the evidence records its name and the sha256 of the bytes that ran;
* the paired summary (mean difference, bootstrap interval with its recorded seed, better / worse /
  same) and the "underpowered" line are descriptive: they decide nothing.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from fractions import Fraction
from pathlib import Path

import pytest

from soup_cli.commands import ship as ship_cmd
from soup_cli.utils.live_eval import GeneratedText
from soup_cli.utils.ship_task_rows import (
    ROW_KEYS,
    build_task_rows,
    load_task_rows,
    load_task_rows_for,
    row_mean,
)
from soup_cli.utils.ship_verdict import (
    DECISION_DONT_SHIP,
    DECISION_SHIP,
    EVIDENCE_SCHEMA_FIELDS,
    PAIRED_BOOTSTRAP_RESAMPLES,
    PAIRED_BOOTSTRAP_SEED,
    PAIRED_CONFIDENCE,
    SUPPORTED_TASK_MODES,
    TASK_MODES,
    V2_ONLY_FIELDS,
    LegScore,
    RewardSpec,
    TaskDetail,
    TaskWin,
    build_paired_summary,
    build_task_detail,
    build_task_win,
    compare_rows,
    compute_benchmark_deltas,
    decide_ship,
    decimal_mean,
    detail_lines,
    paired_min_rows,
    reward_line,
    verdict_from_evidence,
    verdict_to_dict,
    verdict_to_evidence,
)
from tests.cli_runner import CliRunner
from tests.conftest import strip_ansi

runner = CliRunner()


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


# -- the mode is registered everywhere it has to be ------------------------------------------------


def test_reward_is_a_task_mode_of_the_verdict_the_reader_and_the_config():
    from soup_cli.config.schema import ShipConfig

    assert "reward" in TASK_MODES and "reward" in SUPPORTED_TASK_MODES
    assert ShipConfig(task_mode="reward").task_mode == "reward"
    assert ("task", "reward") in V2_ONLY_FIELDS and ("task", "paired") in V2_ONLY_FIELDS
    assert {"reward", "paired"} <= EVIDENCE_SCHEMA_FIELDS["task"]
    assert EVIDENCE_SCHEMA_FIELDS["reward"] == frozenset({"fn", "sha256"})
    assert EVIDENCE_SCHEMA_FIELDS["paired"] == frozenset({
        "n", "better", "worse", "same", "mean_difference", "ci_low", "ci_high", "confidence",
        "resamples", "seed", "underpowered",
    })


# -- the mean: the exact mean of the decimal numbers, so a tie stays a tie -------------------------


def test_the_same_numbers_in_another_order_are_the_same_mean():
    # a plain left-to-right sum is order dependent (it is why this is not the definition here): the
    # two orders differ by one rounding, and a tuned model with the SAME rewards in another order
    # would "win" by 4e-17
    assert row_mean([0.1, 0.2, 0.3]) != row_mean([0.3, 0.2, 0.1])
    assert decimal_mean([0.1, 0.2, 0.3]) == decimal_mean([0.3, 0.2, 0.1]) == 0.2


def test_numbers_that_add_up_to_the_same_decimal_are_the_same_mean():
    # 0.3 + 0.0 and 0.1 + 0.2 are both 0.3 in decimal; in binary floats the second sum is one
    # rounding above the first
    assert row_mean([0.3, 0.0]) != row_mean([0.1, 0.2])
    assert decimal_mean([0.3, 0.0]) == decimal_mean([0.1, 0.2]) == 0.15


def test_the_mean_is_the_exact_decimal_mean_rounded_once():
    assert decimal_mean([1.0, 0.0, 0.0]) == 1 / 3
    assert decimal_mean([0.25, 0.5]) == 0.375
    for scores in ([0.1] * 7, [0.7, 0.9, 0.3, 0.3], [1e-05, 0.123456789, 1.0]):
        exact = sum(Fraction(repr(score)) for score in scores) / len(scores)
        assert decimal_mean(scores) == float(exact)


@pytest.mark.parametrize("bad", [True, "0.5", None, math.nan, math.inf, 10**400])
def test_the_mean_refuses_what_is_not_a_finite_number(bad):
    with pytest.raises((TypeError, ValueError)):
        decimal_mean([0.5, bad])


def test_a_mean_needs_at_least_one_score():
    with pytest.raises(ValueError, match="at least one"):
        decimal_mean([])


def test_equal_means_are_never_a_win_whatever_the_order_of_the_rows():
    base = LegScore(decimal_mean([0.3, 0.2, 0.1]), row_scores=[0.3, 0.2, 0.1])
    tuned = LegScore(decimal_mean([0.1, 0.2, 0.3]), row_scores=[0.1, 0.2, 0.3])
    win = build_task_win("reward", base, tuned, reward=RewardSpec("accuracy", None))
    assert win.base == win.tuned and win.won is False


# -- the TaskWin of the mode names what scored it --------------------------------------------------


def test_a_reward_win_carries_its_reward_and_every_other_mode_refuses_one():
    spec = RewardSpec("rewards/score.py", "a" * 64)
    win = build_task_win("reward", 0.5, 0.6, reward=spec)
    assert win.reward == spec and win.won is True
    with pytest.raises(ValueError, match="reward"):
        build_task_win("reward", 0.5, 0.6)  # a reward verdict says what scored it
    for mode in ("metric", "judge_score", "pairwise"):
        with pytest.raises(ValueError, match="reward"):
            build_task_win(mode, 0.5, 0.6, reward=spec)


def test_the_reward_survives_decide_ship_when_the_floor_flips_won():
    from soup_cli.utils.ship_verdict import TASK_AXIS, NoiseFloor

    spec = RewardSpec("accuracy", None)
    win = build_task_win("reward", 0.5, 0.52, reward=spec)
    assert win.won is True
    floor = NoiseFloor(runs=2, floors=((TASK_AXIS, 0.1),))
    verdict = decide_ship(win, [], noise_floor=floor)
    assert verdict.task_win.won is False and verdict.task_win.reward == spec


@pytest.mark.parametrize("fn, sha", [("", None), ("a\x1bb", None), ("x" * 513, None),
                                     ("accuracy", "xyz"), ("accuracy", "A" * 64),
                                     ("accuracy", "a" * 63)])
def test_a_reward_spec_with_a_bad_name_or_hash_is_refused(fn, sha):
    with pytest.raises(ValueError):
        RewardSpec(fn, sha)


# -- the reward line -------------------------------------------------------------------------------


def test_the_reward_line_names_the_function_and_the_start_of_the_hash():
    win = build_task_win("reward", 0.5, 0.6, reward=RewardSpec("rewards/score.py", "ab" * 32))
    assert reward_line(win) == "Reward function: rewards/score.py (sha256 abababababab)"
    builtin = build_task_win("reward", 0.5, 0.6, reward=RewardSpec("accuracy", None))
    assert reward_line(builtin) == "Reward function: accuracy"
    assert reward_line(build_task_win("metric", 0.5, 0.6)) is None


def test_the_reward_line_is_made_safe_for_the_terminal_even_for_a_spec_built_in_code():
    # the reader refuses a control character; this is the second layer
    spec = object.__new__(RewardSpec)
    object.__setattr__(spec, "fn", "evil\x1b[2J\x07.py")
    object.__setattr__(spec, "sha256", None)
    win = TaskWin(mode="reward", base=0.5, tuned=0.6, won=True, reward=spec)
    line = reward_line(win)
    assert "\x1b" not in line and "\x07" not in line and "evil" in line


# -- the paired summary: descriptive, deterministic, honest about its seed -------------------------


def _reference_interval(base, tuned):
    """An independent re-derivation of the interval (the documented algorithm, written out)."""
    diffs = [t - b for b, t in zip(base, tuned)]
    size = len(diffs)
    rng = random.Random(PAIRED_BOOTSTRAP_SEED)
    means = sorted(
        sum(diffs[int(rng.random() * size)] for _ in range(size)) / size
        for _ in range(PAIRED_BOOTSTRAP_RESAMPLES)
    )
    tail = 50  # 2.5 % of 2000 resamples
    return means[tail], means[PAIRED_BOOTSTRAP_RESAMPLES - 1 - tail]


BASE = [0.1, 0.4, 0.4, 0.7, 0.2, 0.9, 0.5, 0.3, 0.6, 0.8, 0.0, 0.5]
TUNED = [0.3, 0.4, 0.5, 0.6, 0.5, 1.0, 0.5, 0.2, 0.9, 0.8, 0.1, 0.7]


def test_the_paired_summary_is_the_documented_bootstrap_with_its_seed_recorded():
    summary = build_paired_summary(BASE, TUNED)
    low, high = _reference_interval(BASE, TUNED)
    assert (summary.ci_low, summary.ci_high) == (low, high)
    assert summary.seed == PAIRED_BOOTSTRAP_SEED and summary.resamples == 2000
    assert summary.confidence == PAIRED_CONFIDENCE == 0.95
    assert summary.n == 12
    exact = (
        sum(Fraction(repr(t)) for t in TUNED) - sum(Fraction(repr(b)) for b in BASE)
    ) / 12
    assert summary.mean_difference == float(exact)
    assert isinstance(summary.seed, int) and not isinstance(summary.seed, bool)


def test_the_same_rows_give_the_same_summary_every_time():
    assert build_paired_summary(BASE, TUNED) == build_paired_summary(BASE, TUNED)
    other = build_paired_summary(BASE, [*TUNED[:-1], 0.0])
    assert other != build_paired_summary(BASE, TUNED)


def test_the_counts_are_the_rows_block_counts():
    summary = build_paired_summary(BASE, TUNED)
    rows = compare_rows(BASE, TUNED)
    assert (summary.better, summary.worse, summary.same) == (rows.better, rows.worse, rows.same)
    assert summary.better + summary.worse + summary.same == summary.n == rows.n


def test_the_difference_is_tuned_minus_base_per_row():
    gain = build_paired_summary([0.2, 0.3, 0.4, 0.5, 0.6, 0.7], [0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    assert gain.mean_difference == pytest.approx(0.1, abs=1e-12) and gain.better == 6
    assert gain.ci_low == pytest.approx(0.1, abs=1e-12)  # every row moved by the same amount
    assert gain.ci_high == pytest.approx(0.1, abs=1e-12)
    loss = build_paired_summary([0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert loss.mean_difference == pytest.approx(-0.1, abs=1e-12) and loss.worse == 6


def test_a_tie_has_a_mean_difference_of_exactly_zero():
    summary = build_paired_summary([0.3, 0.2, 0.1], [0.1, 0.2, 0.3])
    assert summary.mean_difference == 0.0
    assert summary.better == 1 and summary.worse == 1 and summary.same == 1


def test_the_interval_orders_its_ends_and_stays_in_the_range_of_a_difference():
    summary = build_paired_summary(BASE, TUNED)
    assert -1.0 <= summary.ci_low <= summary.ci_high <= 1.0
    assert summary.ci_low <= summary.mean_difference <= summary.ci_high


@pytest.mark.parametrize("base, tuned", [
    ([], []), ([0.5], [0.5, 0.6]), ([0.5, True], [0.5, 0.5]), ([0.5, math.nan], [0.5, 0.5]),
])
def test_the_paired_summary_refuses_rows_that_cannot_be_paired(base, tuned):
    with pytest.raises((TypeError, ValueError)):
        build_paired_summary(base, tuned)


# -- underpowered: derived from the confidence of the interval, not a constant ---------------------


def test_the_least_rows_are_those_where_every_row_better_could_reach_the_confidence():
    # a two-sided sign test on n rows that ALL improved: p = 2 / 2**n; it can reach 1 - confidence
    # only from the first n where that is at most 0.05 (n = 6: 3.1 %; n = 5 is 6.25 %)
    assert paired_min_rows(0.95) == 6
    assert paired_min_rows(0.90) == 5  # n = 5 is 6.25 %, n = 4 is 12.5 %
    assert paired_min_rows(0.99) == 8  # n = 8 is 0.78 %, n = 7 is 1.56 %
    for confidence in (0.5, 0.8, 0.95, 0.999):
        smallest = paired_min_rows(confidence)
        assert Fraction(2, 2 ** smallest) <= 1 - Fraction(str(confidence))
        assert Fraction(2, 2 ** (smallest - 1)) > 1 - Fraction(str(confidence))


@pytest.mark.parametrize(
    "rows, expected", [(1, True), (4, True), (5, True), (6, False), (12, False)]
)
def test_few_rows_are_flagged_underpowered_and_enough_are_not(rows, expected):
    summary = build_paired_summary([0.2] * rows, [0.4] * rows)
    assert summary.underpowered is expected


def test_the_underpowered_line_is_yellow_and_says_it_decides_nothing():
    summary = build_paired_summary([0.2] * 4, [0.4] * 4)
    detail = TaskDetail(paired=summary)
    win = build_task_win("reward", 0.2, 0.4, detail=detail, reward=RewardSpec("accuracy", None))
    lines = dict(detail_lines(win))
    text = next(line for line in lines if line.startswith("Underpowered:"))
    assert lines[text] is True
    assert "4 rows" in text and "at least 6" in text and "does not change the verdict" in text
    enough = build_paired_summary([0.2] * 8, [0.4] * 8)
    win8 = build_task_win(
        "reward", 0.2, 0.4, detail=TaskDetail(paired=enough),
        reward=RewardSpec("accuracy", None),
    )
    assert not any(line.startswith("Underpowered") for line, _ in detail_lines(win8))


def test_the_paired_line_says_what_it_is_and_that_it_is_descriptive():
    summary = build_paired_summary(BASE, TUNED)
    win = build_task_win(
        "reward", 0.5, 0.55, detail=TaskDetail(paired=summary),
        reward=RewardSpec("accuracy", None),
    )
    text = next(line for line, _ in detail_lines(win) if line.startswith("Paired difference"))
    assert "tuned - base" in text and "95% interval" in text
    assert f"{summary.mean_difference:+.4f}" in text
    assert "2000 resamples" in text and f"seed {PAIRED_BOOTSTRAP_SEED}" in text
    assert "descriptive" in text and "decides nothing" in text


def test_build_task_detail_adds_the_summary_only_when_asked_and_only_with_rows():
    base = LegScore(0.5, row_scores=[0.5, 0.5], truncated=[None, None], failures=0)
    tuned = LegScore(0.6, row_scores=[0.6, 0.6], truncated=[None, None], failures=0)
    assert build_task_detail(base, tuned).paired is None
    detail = build_task_detail(base, tuned, paired=True)
    assert detail.paired is not None and detail.paired.n == 2
    no_rows = LegScore(0.5, truncated=[None, None], failures=0)
    assert build_task_detail(no_rows, no_rows, paired=True).paired is None


# -- the evidence: written, read back strictly -----------------------------------------------------


def _reward_verdict(base_scores, tuned_scores, *, spec=None):
    spec = spec or RewardSpec("rewards/score.py", hashlib.sha256(b"reward").hexdigest())
    base = LegScore(decimal_mean(base_scores), row_scores=base_scores, failures=0,
                    truncated=[None] * len(base_scores))
    tuned = LegScore(decimal_mean(tuned_scores), row_scores=tuned_scores, failures=0,
                     truncated=[None] * len(tuned_scores))
    detail = build_task_detail(base, tuned, paired=True)
    win = build_task_win("reward", base, tuned, detail=detail, reward=spec)
    deltas = compute_benchmark_deltas({"mini_mmlu": 0.7}, {"mini_mmlu": 0.72})
    return decide_ship(win, deltas), spec


def test_the_evidence_of_a_reward_run_names_the_function_and_carries_the_summary():
    verdict, spec = _reward_verdict(BASE, TUNED)
    evidence = verdict_to_evidence(verdict)
    task = evidence["task"]
    assert evidence["schema_version"] == 2
    assert task["mode"] == "reward"
    assert task["reward"] == {"fn": spec.fn, "sha256": spec.sha256}
    assert set(task["paired"]) == EVIDENCE_SCHEMA_FIELDS["paired"]
    assert task["paired"]["seed"] == PAIRED_BOOTSTRAP_SEED
    assert task["paired"]["resamples"] == PAIRED_BOOTSTRAP_RESAMPLES
    assert task["paired"]["underpowered"] is False
    assert task["rows"]["n"] == task["paired"]["n"] == 12


def test_a_reward_verdict_round_trips_through_its_evidence():
    verdict, _ = _reward_verdict(BASE, TUNED)
    back = verdict_from_evidence(json.loads(json.dumps(verdict_to_evidence(verdict))))
    assert back.task_win.reward == verdict.task_win.reward
    assert back.task_win.detail.paired == verdict.task_win.detail.paired
    assert back.task_win.detail.rows == verdict.task_win.detail.rows
    assert (back.decision, back.failed_rule) == (verdict.decision, verdict.failed_rule)
    assert [line for line, _ in detail_lines(back.task_win)] == [
        line for line, _ in detail_lines(verdict.task_win)
    ]
    assert reward_line(back.task_win) == reward_line(verdict.task_win)


def test_the_verdict_dict_has_the_reward_and_the_summary_and_null_for_other_modes():
    verdict, spec = _reward_verdict(BASE, TUNED)
    block = verdict_to_dict(verdict)["task_win"]
    assert block["reward"] == {"fn": spec.fn, "sha256": spec.sha256}
    assert block["paired"]["underpowered"] is False and block["paired"]["seed"] == (
        PAIRED_BOOTSTRAP_SEED
    )
    metric = decide_ship(build_task_win("metric", 0.5, 0.6), [])
    other = verdict_to_dict(metric)["task_win"]
    assert other["reward"] is None and other["paired"] is None


def _evidence_of(mutate=None):
    verdict, _ = _reward_verdict(BASE, TUNED)
    evidence = json.loads(json.dumps(verdict_to_evidence(verdict)))
    if mutate is not None:
        mutate(evidence)
    return evidence


def _set(path, value):
    def apply(evidence):
        node = evidence
        for key in path[:-1]:
            node = node[key]
        if value is ...:
            del node[path[-1]]
        else:
            node[path[-1]] = value

    return apply


def _as_version_1_keeping(evidence, key):
    """Stamp version 1 and drop every version-2 task block but ``key``: it alone is refused."""
    evidence["schema_version"] = 1
    for name in ("rows", "truncated", "generation_failures", "reward", "paired"):
        if name != key:
            evidence["task"].pop(name, None)


READER_REFUSALS = {
    "reward under another mode": (
        _set(("task", "mode"), "metric"), "evidence.task.reward"),
    "reward missing for the reward mode": (
        _set(("task", "reward"), ...), "evidence.task.reward"),
    "reward not an object": (_set(("task", "reward"), "accuracy"), "evidence.task.reward"),
    "reward unknown key": (
        lambda e: e["task"]["reward"].update(extra=1), "unsupported field"),
    "reward without fn": (_set(("task", "reward", "fn"), ...), "fn"),
    "reward fn empty": (_set(("task", "reward", "fn"), ""), "evidence.task.reward.fn"),
    "reward fn control character": (
        _set(("task", "reward", "fn"), "a\x1b[2Jb.py"), "evidence.task.reward.fn"),
    "reward fn not a string": (_set(("task", "reward", "fn"), 5), "evidence.task.reward.fn"),
    "reward fn too long": (
        _set(("task", "reward", "fn"), "x" * 513), "evidence.task.reward.fn"),
    "reward sha not hex": (
        _set(("task", "reward", "sha256"), "z" * 64), "evidence.task.reward.sha256"),
    "reward sha upper case": (
        _set(("task", "reward", "sha256"), "A" * 64), "evidence.task.reward.sha256"),
    "reward sha short": (
        _set(("task", "reward", "sha256"), "a" * 63), "evidence.task.reward.sha256"),
    "reward sha a number": (_set(("task", "reward", "sha256"), 5), "evidence.task.reward.sha256"),
    "reward under schema_version 1": (
        lambda e: _as_version_1_keeping(e, "reward"), "evidence.task.reward is a schema_version 2"),
    "paired unknown key": (
        lambda e: e["task"]["paired"].update(extra=1), "unsupported field"),
    "paired missing key": (_set(("task", "paired", "seed"), ...), "seed"),
    "paired n differs from rows": (_set(("task", "paired", "n"), 13), "evidence.task"),
    "paired counts differ from rows": (
        lambda e: e["task"]["paired"].update(better=0, worse=0, same=12),
        "evidence.task.paired"),
    "paired counts do not add up": (
        _set(("task", "paired", "same"), 99), "evidence.task.paired"),
    "paired mean out of range": (
        _set(("task", "paired", "mean_difference"), 1.5), "evidence.task.paired.mean_difference"),
    "paired mean NaN": (
        _set(("task", "paired", "mean_difference"), math.nan),
        "evidence.task.paired.mean_difference"),
    "paired mean a bool": (
        _set(("task", "paired", "mean_difference"), True),
        "evidence.task.paired.mean_difference"),
    "paired interval reversed": (
        lambda e: e["task"]["paired"].update(ci_low=0.5, ci_high=-0.5),
        "evidence.task.paired"),
    "paired ci_high huge": (
        _set(("task", "paired", "ci_high"), 10 ** 400), "evidence.task.paired.ci_high"),
    "paired confidence 1": (
        _set(("task", "paired", "confidence"), 1.0), "evidence.task.paired.confidence"),
    "paired confidence 0": (
        _set(("task", "paired", "confidence"), 0), "evidence.task.paired.confidence"),
    "paired resamples 0": (
        _set(("task", "paired", "resamples"), 0), "evidence.task.paired.resamples"),
    "paired seed a bool": (_set(("task", "paired", "seed"), True), "evidence.task.paired.seed"),
    "paired seed negative": (_set(("task", "paired", "seed"), -1), "evidence.task.paired.seed"),
    "paired underpowered contradicts n": (
        _set(("task", "paired", "underpowered"), True), "underpowered"),
    "paired underpowered not a bool": (
        _set(("task", "paired", "underpowered"), 0), "evidence.task.paired.underpowered"),
    "paired under schema_version 1": (
        lambda e: _as_version_1_keeping(e, "paired"), "evidence.task.paired is a schema_version 2"),
}


@pytest.mark.parametrize("name", sorted(READER_REFUSALS))
def test_the_reader_refuses_each_defect_and_names_it(name):
    mutate, message = READER_REFUSALS[name]
    evidence = _evidence_of(mutate)
    with pytest.raises(ValueError) as caught:
        verdict_from_evidence(evidence)
    assert message in str(caught.value), str(caught.value)


def test_the_reader_does_not_echo_a_hostile_reward_name():
    evidence = _evidence_of(_set(("task", "reward", "fn"), "a\x1b]0;pwn\x07b.py"))
    with pytest.raises(ValueError) as caught:
        verdict_from_evidence(evidence)
    assert "\x1b" not in str(caught.value) and "pwn" not in str(caught.value)


def test_evidence_of_a_reward_mode_decides_on_its_two_numbers_only():
    evidence = _evidence_of(lambda e: e["task"].update(base=0.7, tuned=0.7))
    assert verdict_from_evidence(evidence).decision == DECISION_DONT_SHIP
    # a summary that says "the interval excludes zero" does not turn a tie into a win
    evidence = _evidence_of(lambda e: (
        e["task"].update(base=0.5, tuned=0.5),
        e["task"]["paired"].update(mean_difference=0.9, ci_low=0.8, ci_high=0.95),
    ))
    assert verdict_from_evidence(evidence).decision == DECISION_DONT_SHIP


def test_an_evidence_of_a_reward_mode_without_a_summary_still_reads():
    evidence = _evidence_of(_set(("task", "paired"), ...))
    verdict = verdict_from_evidence(evidence)
    assert verdict.task_win.detail.paired is None and verdict.task_win.reward is not None
    assert not any(line.startswith("Paired") for line, _ in detail_lines(verdict.task_win))


# -- the command: scripted generators, a real ``soup ship`` invocation -----------------------------

LEN_REWARD = (
    "def reward_fn(completions, **kwargs):\n"
    "    return [len(c[-1][\"content\"]) / 10 for c in completions]\n"
)

CALLS_REWARD = (
    "import json\n"
    "def reward_fn(completions, **kwargs):\n"
    "    with open('calls.jsonl', 'a', encoding='utf-8') as handle:\n"
    "        handle.write(json.dumps({'completions': completions, **kwargs}) + chr(10))\n"
    "    return [len(c[-1]['content']) / 10 for c in completions]\n"
)


BASE_LENGTHS = [1, 2, 3, 7, 7, 9, 4, 6]  # rewards 0.1 0.2 0.3 0.7 0.7 0.9 0.4 0.6
TUNED_LENGTHS = [3, 2, 1, 9, 2, 9, 5, 9]  # one character more in total: a small gain


def _tasks(count=8, **extra):
    return [{"prompt": f"q{i}", "expected": f"a{i}", **extra} for i in range(count)]


def _gen(lengths, *, fail=(), plain=False):
    """A generator whose answer to ``q<i>`` is ``lengths[i]`` characters (``B`` to any other)."""
    table = {f"q{i}": length for i, length in enumerate(lengths)}

    def generate(prompt):
        if prompt in fail:
            raise RuntimeError("generation exploded")
        if prompt not in table:
            return "B"
        text = "x" * table[prompt]
        return text if plain else GeneratedText(text, truncated=False)

    return generate


class Run:
    def __init__(self, result, files, marker_seen):
        self.result = result
        self.files = files
        self.marker_seen = marker_seen

    def json(self, name):
        return json.loads(self.files[name].decode("utf-8"))

    def rows(self, name="out.task-rows.jsonl"):
        return [json.loads(line) for line in self.files[name].decode("ascii").splitlines()]

    @property
    def plain(self):
        return _plain(self.result.output)


def _snapshot():
    root = Path.cwd()
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }


def _work(tmp_path, monkeypatch, tasks, files=None):
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    monkeypatch.chdir(work)
    (work / "task.jsonl").write_text(
        "\n".join(json.dumps(task) for task in tasks) + "\n", encoding="utf-8"
    )
    for name, text in (files or {}).items():
        target = work / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text.encode("utf-8"))  # the bytes as written: no newline translation
    return work


LIVE = ["--base", "base-model", "--adapter", "out/checkpoint-12", "--task-eval", "task.jsonl",
        "--general-suite", "mini_arithmetic", "--device", "cpu"]


def _run(tmp_path, monkeypatch, tasks, base_gen, tuned_gen, *extra, files=None,
         output="out.json", evidence="ev.json", fn="reward.py", mode_flags=True):
    _work(tmp_path, monkeypatch, tasks, files)

    def _fake_resolve(base, tuned, adapter, device, quantization=None, **kwargs):
        return base_gen, tuned_gen

    monkeypatch.setattr(ship_cmd, "_resolve_generators", _fake_resolve)
    args = list(LIVE)
    if mode_flags:
        args += ["--task-mode", "reward"]
        if fn is not None:
            args += ["--reward-fn", fn]
    if output is not None:
        args += ["--output", output]
    if evidence is not None:
        args += ["--emit-evidence", evidence]
    result = runner.invoke(ship_cmd.app, [*args, *extra])
    return Run(result, _snapshot(), None)


def _refuse(tmp_path, monkeypatch, *flags, files=None, tasks=None, code=3):
    """Invoke with a generator resolver that fails the test if a model is ever asked for."""
    _work(tmp_path, monkeypatch, tasks or _tasks(3), files)

    def _no_model(*args, **kwargs):
        raise AssertionError("a model was loaded")

    monkeypatch.setattr(ship_cmd, "_resolve_generators", _no_model)
    result = runner.invoke(ship_cmd.app, [*LIVE, *flags])
    assert result.exit_code == code, (result.output, repr(result.exception))
    return Run(result, _snapshot(), None)


def test_accuracy_reward_is_the_same_verdict_as_the_answer_scoring_of_the_metric_mode(
    tmp_path, monkeypatch
):
    # both read the answer with the same parser: the reward is the binary accuracy
    def answers(correct):
        return {f"q{i}": (f"The answer is {i}" if i in correct else "nonsense") for i in range(8)}

    def gen(table):
        return lambda prompt: table.get(prompt, "B")

    tasks = [{"prompt": f"q{i}", "expected": str(i), "scoring": "answer"} for i in range(8)]
    base, tuned = gen(answers({0, 1, 2})), gen(answers({0, 1, 2, 3, 4}))
    metric = _run(tmp_path, monkeypatch, tasks, base, tuned, mode_flags=False)
    assert metric.result.exit_code == 0, (metric.result.output, repr(metric.result.exception))
    reward = _run(tmp_path, monkeypatch, tasks, base, tuned, fn="accuracy")
    assert reward.result.exit_code == 0, (reward.result.output, repr(reward.result.exception))
    a, b = metric.json("out.json")["task_win"], reward.json("out.json")["task_win"]
    assert (a["base"], a["tuned"], a["won"]) == (b["base"], b["tuned"], b["won"]) == (
        3 / 8, 5 / 8, True,
    )
    assert reward.json("out.json")["task_win"]["mode"] == "reward"


def test_the_function_is_called_once_per_row_with_the_documented_arguments(
    tmp_path, monkeypatch
):
    tasks = [{"prompt": "q0", "expected": "a0", "schema": "S0", "level": 3},
             {"prompt": "q1", "expected": "a1", "schema": "S1", "level": 4}]
    run = _run(tmp_path, monkeypatch, tasks, _gen([1, 2]), _gen([3, 4]),
               files={"reward.py": CALLS_REWARD})
    assert run.result.exit_code in (0, 2), (run.result.output, repr(run.result.exception))
    calls = [json.loads(line) for line in run.files["calls.jsonl"].decode().splitlines()]
    assert len(calls) == 4  # two rows, base and tuned (no noise floor, so no repeats)
    first = calls[0]
    assert first["completions"] == [[{"role": "assistant", "content": "x"}]]
    assert first["answer"] == ["a0"] and first["prompt"] == ["q0"]
    assert first["schema"] == ["S0"] and first["level"] == [3]
    assert [c["prompt"][0] for c in calls] == ["q0", "q1", "q0", "q1"]


def test_a_column_named_like_an_argument_does_not_replace_the_argument(tmp_path, monkeypatch):
    tasks = [{"prompt": "q0", "expected": "a0", "answer": "FROM-THE-COLUMN",
              "completions": "nope"},
             {"prompt": "q1", "expected": "a1"}]
    run = _run(tmp_path, monkeypatch, tasks, _gen([1, 2]), _gen([3, 4]),
               files={"reward.py": CALLS_REWARD})
    assert run.result.exit_code in (0, 2), (run.result.output, repr(run.result.exception))
    first = json.loads(run.files["calls.jsonl"].decode().splitlines()[0])
    assert first["answer"] == ["a0"]  # the task's expected text, as documented
    assert first["completions"] == [[{"role": "assistant", "content": "x"}]]


def test_the_mean_decides_the_verdict_with_the_strict_rule(tmp_path, monkeypatch):
    # base 0.3 mean, tuned 0.4 mean: SHIP; equal: DON'T SHIP; a loss: DON'T SHIP
    files = {"reward.py": LEN_REWARD}
    win = _run(tmp_path, monkeypatch, _tasks(4), _gen([2, 3, 3, 4]), _gen([3, 4, 3, 4]),
               files=files)
    assert win.result.exit_code == 0, (win.result.output, repr(win.result.exception))
    data = win.json("out.json")
    assert data["decision"] == DECISION_SHIP and data["task_win"]["won"] is True
    assert (data["task_win"]["base"], data["task_win"]["tuned"]) == (0.3, 0.35)
    tie = _run(tmp_path, monkeypatch, _tasks(4), _gen([2, 3, 3, 4]), _gen([4, 3, 3, 2]),
               files=files)
    assert tie.result.exit_code == 2, (tie.result.output, repr(tie.result.exception))
    assert tie.json("out.json")["failed_rule"] == "task_win"
    assert tie.json("out.json")["task_win"]["won"] is False
    loss = _run(tmp_path, monkeypatch, _tasks(4), _gen([3, 4, 3, 4]), _gen([2, 3, 3, 4]),
                files=files)
    assert loss.result.exit_code == 2


def test_the_same_rewards_in_another_order_is_a_tie_not_a_one_rounding_win(
    tmp_path, monkeypatch
):
    # rewards 0.3 0.2 0.1 against 0.1 0.2 0.3: a plain float sum calls the second mean larger
    run = _run(tmp_path, monkeypatch, _tasks(3), _gen([3, 2, 1]), _gen([1, 2, 3]),
               files={"reward.py": LEN_REWARD})
    assert run.result.exit_code == 2, (run.result.output, repr(run.result.exception))
    win = run.json("out.json")["task_win"]
    assert win["base"] == win["tuned"] == 0.2 and win["won"] is False
    assert run.json("out.json")["failed_rule"] == "task_win"
    swapped = _run(tmp_path, monkeypatch, _tasks(3), _gen([1, 2, 3]), _gen([3, 2, 1]),
                   files={"reward.py": LEN_REWARD})
    assert swapped.result.exit_code == 2 and swapped.json("out.json")["task_win"]["won"] is False


def test_rewards_that_add_up_to_the_same_decimal_are_a_tie(tmp_path, monkeypatch):
    run = _run(tmp_path, monkeypatch, _tasks(2), _gen([3, 0]), _gen([1, 2]),
               files={"reward.py": LEN_REWARD})
    assert run.result.exit_code == 2, (run.result.output, repr(run.result.exception))
    win = run.json("out.json")["task_win"]
    assert win["base"] == win["tuned"] == 0.15 and win["won"] is False


def test_the_summary_decides_nothing_a_gain_the_interval_cannot_separate_still_ships(
    tmp_path, monkeypatch
):
    # one row gained 0.1, one lost 0.1 plus a tiny net gain: the mean rule SHIPs and the interval
    # contains zero; the summary only says so
    run = _run(tmp_path, monkeypatch, _tasks(8),
               _gen([1, 9, 3, 3, 3, 3, 3, 3]), _gen([4, 5, 3, 3, 3, 3, 3, 5]),
               files={"reward.py": LEN_REWARD})
    assert run.result.exit_code == 0, (run.result.output, repr(run.result.exception))
    paired = run.json("out.json")["task_win"]["paired"]
    assert paired["ci_low"] < 0.0 < paired["ci_high"] and paired["mean_difference"] > 0.0
    assert run.json("out.json")["decision"] == DECISION_SHIP


def test_a_user_file_is_executed_from_under_the_working_directory_and_its_hash_is_recorded(
    tmp_path, monkeypatch
):
    run = _run(tmp_path, monkeypatch, _tasks(6), _gen([1, 2, 3, 4, 5, 6]),
               _gen([2, 3, 4, 5, 6, 7]), files={"rewards/score.py": LEN_REWARD},
               fn="rewards/score.py")
    assert run.result.exit_code == 0, (run.result.output, repr(run.result.exception))
    digest = hashlib.sha256(LEN_REWARD.encode("utf-8")).hexdigest()
    block = run.json("out.json")["task_win"]["reward"]
    assert block == {"fn": "rewards/score.py", "sha256": digest}
    assert run.json("ev.json")["task"]["reward"] == {"fn": "rewards/score.py", "sha256": digest}
    assert run.json("ev.json")["schema_version"] == 2
    assert f"Reward function: rewards/score.py (sha256 {digest[:12]})" in run.plain


def test_the_hash_is_that_of_the_bytes_even_with_crlf_and_non_ascii(tmp_path, monkeypatch):
    work = _work(tmp_path, monkeypatch, _tasks(6))
    source = LEN_REWARD.replace("\n", "\r\n") + "# café ☃\r\n"
    (work / "reward.py").write_bytes(source.encode("utf-8"))
    run = _run(tmp_path, monkeypatch, _tasks(6), _gen([1] * 6), _gen([2] * 6),
               files={"x.txt": "keep"}, fn="reward.py")
    # _run rewrote task.jsonl only; the CRLF file is still the one on disk
    assert run.json("out.json")["task_win"]["reward"]["sha256"] == hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()


def test_a_builtin_names_itself_and_has_no_hash(tmp_path, monkeypatch):
    run = _run(tmp_path, monkeypatch, _tasks(6), _gen([1] * 6), _gen([2] * 6), fn="format")
    assert run.result.exit_code in (0, 2), (run.result.output, repr(run.result.exception))
    assert run.json("out.json")["task_win"]["reward"] == {"fn": "format", "sha256": None}
    assert "EXECUTES" not in run.result.output  # no file runs


def test_the_terminal_says_the_file_is_executed_before_it_runs(tmp_path, monkeypatch):
    run = _run(tmp_path, monkeypatch, _tasks(6), _gen([1] * 6), _gen([2] * 6),
               files={"reward.py": LEN_REWARD})
    digest = hashlib.sha256(LEN_REWARD.encode("utf-8")).hexdigest()
    assert "--reward-fn reward.py is a Python file and soup ship EXECUTES it " in run.plain
    assert f"sha256 {digest[:12]}" in run.plain
    assert "Only use a reward file you trust" in run.plain


def test_the_help_says_a_python_reward_file_is_executed():
    result = runner.invoke(ship_cmd.app, ["--help"])
    text = _plain(result.output)
    assert "--reward-fn" in text and "EXECUTED" in text
    assert "--task-mode" in text and "reward" in text


@pytest.mark.parametrize("name, expected_fn", [
    ("format", "format"), ("verifiable:math", "verifiable:math"),
])
def test_the_builtin_rewards_run(tmp_path, monkeypatch, name, expected_fn):
    run = _run(tmp_path, monkeypatch, _tasks(6), _gen([1] * 6), _gen([2] * 6), fn=name)
    assert run.result.exit_code in (0, 2), (run.result.output, repr(run.result.exception))
    assert run.json("out.json")["task_win"]["reward"]["fn"] == expected_fn


def test_a_json_schema_reward_reads_the_schema_column_of_the_task_file(tmp_path, monkeypatch):
    schema = json.dumps(
        {"type": "object", "required": ["a"], "properties": {"a": {"type": "number"}}}
    )
    tasks = [{"prompt": f"q{i}", "expected": "-", "schema": schema} for i in range(6)]
    good = {f"q{i}": '{"a": 1}' for i in range(6)}
    bad = {f"q{i}": "no json" for i in range(6)}
    run = _run(tmp_path, monkeypatch, tasks, lambda p: bad.get(p, "B"), lambda p: good.get(p, "B"),
               fn="verifiable:json_schema")
    assert run.result.exit_code == 0, (run.result.output, repr(run.result.exception))
    win = run.json("out.json")["task_win"]
    assert (win["base"], win["tuned"]) == (0.0, 1.0)
    assert win["reward"]["fn"] == "verifiable:json_schema"


# -- what the function may return ------------------------------------------------------------------


def _bad_return(expression):
    """A reward file whose function misbehaves on the row ``q3`` only (a 0.5 for every other)."""
    return (
        "def reward_fn(completions, **kwargs):\n"
        "    if kwargs['prompt'][0] == 'q3':\n"
        f"        return {expression}\n"
        "    return [0.5]\n"
    )


BAD_RETURNS = {
    "above one": ("[1.5]", "returned 1.5, outside [0, 1]"),
    "below zero": ("[-0.1]", "returned -0.1, outside [0, 1]"),
    "NaN": ("[float('nan')]", "returned a score that is not finite"),
    "infinity": ("[float('inf')]", "returned a score that is not finite"),
    "negative infinity": ("[float('-inf')]", "returned a score that is not finite"),
    "a boolean": ("[True]", "returned a boolean"),
    "a string": ("['0.5']", "returned a str, not a number"),
    "None": ("[None]", "returned a NoneType, not a number"),
    "a huge integer": ("[10 ** 400]", "outside [0, 1]"),
    "an empty list (one score short)": ("[]", "returned 0 scores"),
    "two scores (one too many)": ("[0.5, 0.5]", "returned 2 scores"),
    "a bare number": ("0.5", "returned a float, not a list"),
    "a string, not a list": ("'0.5'", "returned a str, not a list"),
    "a dict": ("{'a': 1}", "returned a dict, not a list"),
}


@pytest.mark.parametrize("name", sorted(BAD_RETURNS))
def test_a_return_that_is_not_one_finite_number_in_range_names_the_row_and_gives_no_verdict(
    tmp_path, monkeypatch, name
):
    expression, message = BAD_RETURNS[name]
    run = _run(tmp_path, monkeypatch, _tasks(6), _gen([1] * 6), _gen([2] * 6),
               files={"reward.py": _bad_return(expression)})
    assert run.result.exit_code == 1, (run.result.output, repr(run.result.exception))
    assert message in run.plain, run.plain
    assert "for row 3 (base)" in run.plain
    assert "out.json" not in run.files and "ev.json" not in run.files  # no verdict, no files
    assert "SHIP" not in run.plain.replace("soup ship", "")


def test_the_tuned_side_is_named_when_the_tuned_side_is_the_one_that_fails(tmp_path, monkeypatch):
    source = (
        "def reward_fn(completions, **kwargs):\n"
        "    content = completions[0][-1]['content']\n"
        "    return [2.0] if content == 'xx' else [0.5]\n"
    )
    run = _run(tmp_path, monkeypatch, _tasks(4), _gen([1] * 4), _gen([2] * 4),
               files={"reward.py": source})
    assert run.result.exit_code == 1
    assert "for row 0 (tuned)" in run.plain and "returned 2.0, outside [0, 1]" in run.plain


@pytest.mark.parametrize("zero_or_one", ["[0]", "[1]", "[0.0]", "[1.0]"])
def test_the_ends_of_the_range_and_a_whole_number_are_accepted(tmp_path, monkeypatch, zero_or_one):
    source = f"def reward_fn(completions, **kwargs):\n    return {zero_or_one}\n"
    run = _run(tmp_path, monkeypatch, _tasks(6), _gen([1] * 6), _gen([2] * 6),
               files={"reward.py": source})
    assert run.result.exit_code == 2, (run.result.output, repr(run.result.exception))
    assert isinstance(run.json("out.json")["task_win"]["base"], float)


def test_a_function_that_raises_is_a_named_runtime_error(tmp_path, monkeypatch):
    source = (
        "def reward_fn(completions, **kwargs):\n"
        "    return [kwargs['missing_column'][0]]\n"
    )
    run = _run(tmp_path, monkeypatch, _tasks(4), _gen([1] * 4), _gen([2] * 4),
               files={"reward.py": source})
    assert run.result.exit_code == 1
    assert "reward function raised KeyError for row 0 (base)" in run.plain
    assert "out.json" not in run.files


def test_a_function_that_exits_the_process_is_a_runtime_error_never_a_zero(
    tmp_path, monkeypatch
):
    source = "def reward_fn(completions, **kwargs):\n    raise SystemExit(0)\n"
    run = _run(tmp_path, monkeypatch, _tasks(4), _gen([1] * 4), _gen([2] * 4),
               files={"reward.py": source})
    assert run.result.exit_code == 1, (run.result.output, repr(run.result.exception))
    assert "reward function called sys.exit for row 0 (base)" in run.plain


def test_a_file_that_exits_while_loading_is_a_usage_error_never_a_zero(tmp_path, monkeypatch):
    run = _refuse(tmp_path, monkeypatch, "--task-mode", "reward", "--reward-fn", "reward.py",
                  files={"reward.py": "import sys\nsys.exit(0)\n"})
    assert "--reward-fn file called sys.exit while it was loaded" in run.plain


def test_a_file_that_cannot_be_loaded_is_a_usage_error_before_any_model_loads(
    tmp_path, monkeypatch
):
    run = _refuse(tmp_path, monkeypatch, "--task-mode", "reward", "--reward-fn", "reward.py",
                  files={"reward.py": "def reward_fn(:\n"})
    assert "could not load --reward-fn reward.py: SyntaxError" in run.plain


def test_a_file_without_a_reward_fn_is_a_usage_error(tmp_path, monkeypatch):
    run = _refuse(tmp_path, monkeypatch, "--task-mode", "reward", "--reward-fn", "reward.py",
                  files={"reward.py": "x = 1\n"})
    assert "could not load --reward-fn reward.py" in run.plain
    assert "reward_fn" in run.plain


def test_a_reward_fn_that_is_not_callable_is_a_usage_error(tmp_path, monkeypatch):
    run = _refuse(tmp_path, monkeypatch, "--task-mode", "reward", "--reward-fn", "reward.py",
                  files={"reward.py": "reward_fn = 5\n"})
    assert "--reward-fn reward.py: reward_fn is not callable" in run.plain


def test_a_file_that_changes_while_it_is_loaded_is_refused(tmp_path, monkeypatch):
    import soup_cli.trainer.rewards as rewards

    original = rewards.load_reward_fn

    def swap_then_load(spec, *args, **kwargs):
        loaded = original(spec, *args, **kwargs)
        Path(spec).write_text("def reward_fn(completions, **kwargs):\n    return [0.9]\n",
                              encoding="utf-8")
        return loaded

    monkeypatch.setattr(rewards, "load_reward_fn", swap_then_load)
    run = _refuse(tmp_path, monkeypatch, "--task-mode", "reward", "--reward-fn", "reward.py",
                  files={"reward.py": LEN_REWARD})
    assert "--reward-fn reward.py changed while it was loaded" in run.plain


# -- failed generations ----------------------------------------------------------------------------


def test_a_failed_generation_is_a_zero_row_the_function_never_sees(tmp_path, monkeypatch):
    run = _run(tmp_path, monkeypatch, _tasks(4), _gen([2, 2, 2, 2]),
               _gen([3, 3, 3, 3], fail=("q1",)), files={"reward.py": CALLS_REWARD})
    assert run.result.exit_code == 0, (run.result.output, repr(run.result.exception))
    calls = [json.loads(line) for line in run.files["calls.jsonl"].decode().splitlines()]
    assert len(calls) == 4 + 3  # the failed tuned row was not handed to the function
    rows = run.rows()
    assert rows[1]["tuned_failed"] is True and rows[1]["tuned_score"] == 0.0
    assert rows[1]["tuned_answer"] == "" and rows[0]["tuned_failed"] is False
    failures = run.json("out.json")["task_win"]["generation_failures"]
    assert failures == {"base": 0, "tuned": 1, "n": 4}


def test_a_side_whose_every_generation_failed_gives_no_verdict(tmp_path, monkeypatch):
    run = _run(tmp_path, monkeypatch, _tasks(4), _gen([2] * 4),
               _gen([3] * 4, fail=("q0", "q1", "q2", "q3")), files={"reward.py": LEN_REWARD})
    assert run.result.exit_code == 1
    assert "every task generation failed on the tuned side" in run.plain
    assert "out.json" not in run.files


# -- the per-row file and the evidence of a run ----------------------------------------------------


def test_the_row_file_of_a_reward_run_adds_up_to_the_reported_numbers_exactly(
    tmp_path, monkeypatch
):
    base_lengths, tuned_lengths = BASE_LENGTHS, TUNED_LENGTHS
    run = _run(tmp_path, monkeypatch, _tasks(8), _gen(base_lengths), _gen(tuned_lengths),
               files={"reward.py": LEN_REWARD})
    assert run.result.exit_code == 0, (run.result.output, repr(run.result.exception))
    rows = run.rows()
    assert len(rows) == 8
    for row in rows:
        assert tuple(row) == ROW_KEYS and row["scoring"] is None
    assert [row["base_score"] for row in rows] == [n / 10 for n in base_lengths]
    assert [row["tuned_score"] for row in rows] == [n / 10 for n in tuned_lengths]
    assert [row["base_answer"] for row in rows] == ["x" * n for n in base_lengths]
    win = run.json("out.json")["task_win"]
    expected_base = float(sum(Fraction(repr(n / 10)) for n in base_lengths) / 8)
    assert win["base"] == expected_base == decimal_mean([r["base_score"] for r in rows])
    assert abs(row_mean([r["base_score"] for r in rows]) - win["base"]) <= 1e-12
    assert abs(row_mean([r["tuned_score"] for r in rows]) - win["tuned"]) <= 1e-12
    changes = [row["change"] for row in rows]
    block = win["rows"]
    assert (changes.count("better"), changes.count("worse"), changes.count("same")) == (
        block["better"], block["worse"], block["same"],
    )
    paired = win["paired"]
    assert (paired["better"], paired["worse"], paired["same"], paired["n"]) == (
        block["better"], block["worse"], block["same"], block["n"],
    )


def test_the_loader_accepts_what_the_command_wrote_and_checks_it_exactly(tmp_path, monkeypatch):
    run = _run(tmp_path, monkeypatch, _tasks(8), _gen(BASE_LENGTHS), _gen(TUNED_LENGTHS),
               files={"reward.py": LEN_REWARD})
    work = Path.cwd()
    loaded = load_task_rows_for(str(work / "out.json"))
    win = run.json("out.json")["task_win"]
    assert loaded.base_mean == win["base"] and loaded.tuned_mean == win["tuned"]
    assert (loaded.better, loaded.worse, loaded.same) == (
        win["rows"]["better"], win["rows"]["worse"], win["rows"]["same"],
    )
    assert run.json("ev.json")["task_rows"] == "out.task-rows.jsonl"
    load_task_rows_for(str(work / "ev.json"))  # the evidence names the file too
    # a number that is the plain loop mean and not the exact one is refused for a reward verdict
    rows_path = str(work / "out.task-rows.jsonl")
    with pytest.raises(ValueError, match="is not the reported"):
        load_task_rows(rows_path, base=win["base"] + 1e-13, decimal=True)
    assert load_task_rows(rows_path, base=win["base"], decimal=True).base_mean == win["base"]


def test_the_builder_checks_a_reward_leg_against_its_exact_mean(tmp_path):
    from soup_cli.eval.custom import EvalTask

    tasks = [EvalTask(prompt=f"q{i}", expected="e") for i in range(3)]

    def leg(scores, reported):
        return LegScore(reported, row_scores=scores, truncated=[None] * 3, failures=0,
                        answers=["a"] * 3, failed_rows=[False] * 3, tasks=tasks)

    base = leg([0.3, 0.2, 0.1], decimal_mean([0.3, 0.2, 0.1]))
    tuned = leg([0.1, 0.2, 0.3], decimal_mean([0.1, 0.2, 0.3]))
    rows = build_task_rows(base, tuned, scoring_applies=False, decimal=True)
    assert [row.scoring for row in rows] == [None] * 3
    off_by_a_rounding = leg([0.3, 0.2, 0.1], decimal_mean([0.3, 0.2, 0.1]) + 1e-13)
    with pytest.raises(ValueError, match="average"):
        build_task_rows(off_by_a_rounding, tuned, scoring_applies=False, decimal=True)


def test_the_evidence_of_a_run_replays_to_the_same_verdict_and_the_same_lines(
    tmp_path, monkeypatch
):
    run = _run(tmp_path, monkeypatch, _tasks(8), _gen(BASE_LENGTHS), _gen(TUNED_LENGTHS),
               files={"reward.py": LEN_REWARD})
    assert run.result.exit_code == 0, (run.result.output, repr(run.result.exception))
    replay = runner.invoke(ship_cmd.app, ["--evidence", "ev.json", "--output", "replay.json"])
    assert replay.exit_code == 0, (replay.output, repr(replay.exception))
    live = run.json("out.json")
    again = json.loads(Path("replay.json").read_text(encoding="utf-8"))
    for key in ("decision", "failed_rule", "benchmark_deltas", "judged"):
        assert again[key] == live[key], key
    assert again["task_win"] == live["task_win"]
    for line in ("Reward function: reward.py (sha256", "Paired difference (tuned - base per row)"):
        assert line in run.plain and line in _plain(replay.output)


def test_the_replay_of_a_reward_evidence_never_loads_the_file(tmp_path, monkeypatch):
    marker = "marker.txt"
    evil = (
        f"open('{marker}', 'w').write('ran')\n"
        "def reward_fn(completions, **kwargs):\n    return [0.5]\n"
    )
    work = _work(tmp_path, monkeypatch, _tasks(3), {"reward.py": evil})
    verdict, _ = _reward_verdict(BASE, TUNED, spec=RewardSpec("reward.py", "b" * 64))
    (work / "ev.json").write_text(json.dumps(verdict_to_evidence(verdict)), encoding="utf-8")
    result = runner.invoke(ship_cmd.app, ["--evidence", "ev.json"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert not (work / marker).exists()
    with_config = runner.invoke(
        ship_cmd.app, ["--evidence", "ev.json", "--task-mode", "reward"],
    )
    assert with_config.exit_code == 0 and not (work / marker).exists()


# -- noise floor -----------------------------------------------------------------------------------


def test_the_noise_floor_of_a_reward_run_is_measured_decode_only(tmp_path, monkeypatch):
    run = _run(tmp_path, monkeypatch, _tasks(6), _gen([1, 2, 3, 4, 5, 6]),
               _gen([2, 3, 4, 5, 6, 7]), "--noise-floor", "2",
               files={"reward.py": LEN_REWARD})
    assert run.result.exit_code == 0, (run.result.output, repr(run.result.exception))
    floor = run.json("out.json")["noise_floor"]
    assert floor["judge_inclusive"] is False and "__task__" in floor["floors"]
    assert floor["floors"]["__task__"] == 0.0  # a scripted generator repeats exactly


def test_a_gain_inside_the_measured_floor_is_not_a_win(tmp_path, monkeypatch):
    # the base answers to q0 change from call to call: its leg-1 mean spreads by 0.2 over the
    # repeats, so a gain of 0.05 is the instrument and the verdict is DON'T SHIP
    calls = {"count": 0}

    def wobbling_base(prompt):
        if prompt == "q0":
            calls["count"] += 1
            return "x" * (5 if calls["count"] % 2 == 1 else 1)
        return "x" * 3 if prompt == "q1" else "B"

    run = _run(tmp_path, monkeypatch, _tasks(2), wobbling_base, _gen([6, 3]),
               "--noise-floor", "2", files={"reward.py": LEN_REWARD})
    data = run.json("out.json")
    assert data["noise_floor"]["floors"]["__task__"] == pytest.approx(0.2)
    assert data["task_win"]["tuned"] > data["task_win"]["base"]  # the mean did go up ...
    assert data["task_win"]["won"] is False and data["failed_rule"] == "task_win"
    assert run.result.exit_code == 2  # ... by less than what the instrument can resolve


def test_the_floor_scorer_of_the_mode_scores_the_base_with_the_function(tmp_path, monkeypatch):
    _work(tmp_path, monkeypatch, _tasks(3))
    scorer = ship_cmd._build_task_floor_scorer(
        "reward", _gen([3, 2, 1]), base_id="b", task_eval="task.jsonl", judge_model=None,
        reward_fn=lambda completions, **kwargs: [len(completions[0][-1]["content"]) / 10],
    )
    assert scorer() == 0.2
    with pytest.raises(ValueError, match="reward"):
        ship_cmd._build_task_floor_scorer(
            "reward", _gen([3, 2, 1]), base_id="b", task_eval="task.jsonl", judge_model=None,
        )


# -- the refusals: exit 3 before any model loads ---------------------------------------------------

REFUSALS = {
    "--reward-fn without the reward mode": (
        ["--reward-fn", "accuracy"],
        "--reward-fn applies only with --task-mode reward"),
    "--reward-fn beside another mode": (
        ["--task-mode", "judge_score", "--judge-model", "ollama://j", "--reward-fn", "accuracy"],
        "--reward-fn applies only with --task-mode reward"),
    "the reward mode without --reward-fn": (
        ["--task-mode", "reward"], "--task-mode reward needs --reward-fn"),
    "verifiable code": (
        ["--task-mode", "reward", "--reward-fn", "verifiable:code"],
        "--reward-fn verifiable:code is not supported"),
    "verifiable without a domain": (
        ["--task-mode", "reward", "--reward-fn", "verifiable"],
        "--reward-fn verifiable needs a domain"),
    "verifiable with an unknown domain": (
        ["--task-mode", "reward", "--reward-fn", "verifiable:poetry"],
        "--reward-fn verifiable needs a domain"),
    "judge_rank": (
        ["--task-mode", "reward", "--reward-fn", "judge_rank"],
        "--reward-fn judge_rank is built from a GRPO config"),
    "a list of functions": (
        ["--task-mode", "reward", "--reward-fn", "accuracy,format"],
        "--reward-fn takes one reward function"),
    "an unknown name": (
        ["--task-mode", "reward", "--reward-fn", "bleu"],
        "--reward-fn must be accuracy, format, verifiable:math"),
    "a file that is not python": (
        ["--task-mode", "reward", "--reward-fn", "reward.txt"],
        "--reward-fn must be accuracy, format, verifiable:math"),
    "an empty name": (
        ["--task-mode", "reward", "--reward-fn", ""], "--reward-fn must not be empty"),
    "a padded name": (
        ["--task-mode", "reward", "--reward-fn", " accuracy"], "--reward-fn must not start or end"),
    "a control character": (
        ["--task-mode", "reward", "--reward-fn", "r\x1b[2J.py"],
        "--reward-fn must not contain a control character"),
    "a name that is too long": (
        ["--task-mode", "reward", "--reward-fn", "r" * 600 + ".py"],
        "--reward-fn must be at most 512 characters"),
    "a missing file": (
        ["--task-mode", "reward", "--reward-fn", "missing.py"], "is not a file"),
    "a directory named like a file": (
        ["--task-mode", "reward", "--reward-fn", "dir.py"], "is not a file"),
    "a path outside the working directory": (
        ["--task-mode", "reward", "--reward-fn", "../outside.py"], "must stay under cwd"),
    "a network share": (
        ["--task-mode", "reward", "--reward-fn", "\\\\host\\share\\r.py"],
        "names a network share or a device"),
    "a network share with slashes": (
        ["--task-mode", "reward", "--reward-fn", "//host/share/r.py"],
        "names a network share or a device"),
}


@pytest.mark.parametrize("name", sorted(REFUSALS))
def test_a_bad_reward_flag_is_refused_before_any_model_loads(tmp_path, monkeypatch, name):
    flags, message = REFUSALS[name]
    (tmp_path / "outside.py").write_text(LEN_REWARD, encoding="utf-8")
    run = _refuse(tmp_path, monkeypatch, *flags,
                  files={"reward.txt": LEN_REWARD, "dir.py/keep": "x"})
    assert message in run.plain, run.plain
    assert "SHIP" not in run.plain.replace("soup ship", "")


def test_an_oversized_file_is_refused_before_it_is_read_into_memory(tmp_path, monkeypatch):
    run = _refuse(tmp_path, monkeypatch, "--task-mode", "reward", "--reward-fn", "big.py",
                  files={"big.py": "#" * (1024 * 1024 + 1)})
    assert "is larger than 1048576 bytes" in run.plain


def test_a_symlinked_reward_file_is_refused_and_not_executed(tmp_path, monkeypatch):
    work = _work(tmp_path, monkeypatch, _tasks(3), {"real.py": "open('ran.txt','w').write('x')\n"})
    try:
        (work / "link.py").symlink_to(work / "real.py")
    except (OSError, NotImplementedError):
        pytest.skip("this platform cannot create a symlink here")
    result = runner.invoke(ship_cmd.app, [*LIVE, "--task-mode", "reward", "--reward-fn", "link.py"])
    assert result.exit_code == 3, (result.output, repr(result.exception))
    assert "must not be a symlink" in _plain(result.output)
    assert not (work / "ran.txt").exists()


def test_a_reward_file_is_never_found_by_itself(tmp_path, monkeypatch):
    # a reward.py lying in the working directory is not picked up: the mode needs the flag
    marker = "ran.txt"
    run = _refuse(tmp_path, monkeypatch, "--task-mode", "reward",
                  files={"reward.py": f"open('{marker}', 'w').write('x')\n"})
    assert "--task-mode reward needs --reward-fn" in run.plain
    assert marker not in run.files


def test_a_refused_flag_does_not_execute_the_file_it_names(tmp_path, monkeypatch):
    marker = "ran.txt"
    run = _refuse(tmp_path, monkeypatch, "--reward-fn", "reward.py",
                  files={"reward.py": f"open('{marker}', 'w').write('x')\n"})
    assert "applies only with --task-mode reward" in run.plain
    assert marker not in run.files


def test_an_offline_replay_refuses_the_flag_beside_another_mode_too(tmp_path, monkeypatch):
    marker = "ran.txt"
    work = _work(tmp_path, monkeypatch, _tasks(3),
                 {"reward.py": f"open('{marker}', 'w').write('x')\n"})
    verdict, _ = _reward_verdict(BASE, TUNED)
    (work / "ev.json").write_text(json.dumps(verdict_to_evidence(verdict)), encoding="utf-8")
    result = runner.invoke(ship_cmd.app, ["--evidence", "ev.json", "--reward-fn", "reward.py"])
    assert result.exit_code == 3, (result.output, repr(result.exception))
    assert "--reward-fn applies only with --task-mode reward" in _plain(result.output)
    assert not (work / marker).exists()


def test_an_offline_replay_refuses_the_flag_and_never_runs_a_file(tmp_path, monkeypatch):
    marker = "ran.txt"
    work = _work(tmp_path, monkeypatch, _tasks(3),
                 {"reward.py": f"open('{marker}', 'w').write('x')\n"})
    verdict, _ = _reward_verdict(BASE, TUNED)
    (work / "ev.json").write_text(json.dumps(verdict_to_evidence(verdict)), encoding="utf-8")
    result = runner.invoke(
        ship_cmd.app, ["--evidence", "ev.json", "--task-mode", "reward", "--reward-fn", "reward.py"]
    )
    assert result.exit_code == 3, (result.output, repr(result.exception))
    assert "--reward-fn applies only to a live run" in _plain(result.output)
    assert not (work / marker).exists()


def test_the_adapter_that_changes_nothing_runs_no_reward_code(tmp_path, monkeypatch):
    marker = "ran.txt"
    work = _work(tmp_path, monkeypatch, _tasks(3),
                 {"reward.py": (f"open('{marker}', 'w').write('x')\n"
                                "def reward_fn(c, **k):\n    return [0.5]\n")})
    monkeypatch.setattr(ship_cmd, "_adapter_is_noop", lambda adapter: True)

    def _no_model(*args, **kwargs):
        raise AssertionError("a model was loaded")

    monkeypatch.setattr(ship_cmd, "_resolve_generators", _no_model)
    result = runner.invoke(
        ship_cmd.app,
        [*LIVE, "--task-mode", "reward", "--reward-fn", "reward.py", "--output", "out.json"],
    )
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert not (work / marker).exists()
    data = json.loads((work / "out.json").read_text(encoding="utf-8"))
    assert data["failed_rule"] == "adapter_noop" and data["task_win"]["mode"] == "reward"
    assert data["task_win"]["reward"]["fn"] == "reward.py"
    assert data["task_win"]["reward"]["sha256"] == hashlib.sha256(
        (work / "reward.py").read_bytes()
    ).hexdigest()


# -- the config: eval.ship.reward_fn ---------------------------------------------------------------

CONFIG = (
    "base: sshleifer/tiny-gpt2\n"
    "data:\n"
    "  train: x.jsonl\n"
    "eval:\n"
    "  ship:\n"
    "    task_mode: {mode}\n"
    "{extra}"
)


def _config(mode="reward", reward_fn="reward.py"):
    extra = f"    reward_fn: {reward_fn}\n" if reward_fn is not None else ""
    return CONFIG.format(mode=mode, extra=extra)


def test_the_config_key_is_refused_while_the_mode_is_not_reward_naming_both():
    import pydantic

    from soup_cli.config.schema import ShipConfig

    assert ShipConfig(task_mode="reward", reward_fn="accuracy").reward_fn == "accuracy"
    assert ShipConfig().reward_fn is None
    for mode in ("metric", "judge_score", "pairwise"):
        with pytest.raises(pydantic.ValidationError) as caught:
            ShipConfig(task_mode=mode, reward_fn="accuracy")
        message = str(caught.value)
        assert "eval.ship.reward_fn" in message and "eval.ship.task_mode" in message


@pytest.mark.parametrize("bad", [True, 5, "", "   ", "a\x00b", "x" * 513, "a\x1b[2Jb"])
def test_the_config_key_is_a_plain_name_or_path(bad):
    import pydantic

    from soup_cli.config.schema import ShipConfig

    with pytest.raises(pydantic.ValidationError):
        ShipConfig(task_mode="reward", reward_fn=bad)


def test_an_unknown_key_beside_it_still_refuses_the_load():
    from soup_cli.config.loader import load_config_from_string

    text = _config("reward", "accuracy") + "    reward_fns: accuracy\n"
    with pytest.raises(ValueError):
        load_config_from_string(text)


def test_the_config_supplies_the_mode_and_the_function(tmp_path, monkeypatch):
    run = _run(tmp_path, monkeypatch, _tasks(6), _gen([1] * 6), _gen([2] * 6),
               "--config", "soup.yaml", files={"soup.yaml": _config("reward", "reward.py"),
                                              "reward.py": LEN_REWARD},
               mode_flags=False, evidence=None)
    assert run.result.exit_code == 0, (run.result.output, repr(run.result.exception))
    assert run.json("out.json")["task_win"]["mode"] == "reward"
    assert run.json("out.json")["task_win"]["reward"]["fn"] == "reward.py"


def test_the_flag_beats_the_config(tmp_path, monkeypatch):
    other = LEN_REWARD.replace("/ 10", "/ 20")
    run = _run(tmp_path, monkeypatch, _tasks(6), _gen([2] * 6), _gen([4] * 6),
               "--config", "soup.yaml",
               files={"soup.yaml": _config("reward", "reward.py"), "reward.py": LEN_REWARD,
                      "other.py": other},
               fn="other.py", evidence=None)
    assert run.result.exit_code == 0, (run.result.output, repr(run.result.exception))
    win = run.json("out.json")["task_win"]
    assert win["reward"]["fn"] == "other.py" and win["base"] == 0.1  # the other file's /20


def test_the_mode_flag_beats_the_config_and_the_dormant_function_is_never_run(
    tmp_path, monkeypatch
):
    marker = "ran.txt"
    run = _run(tmp_path, monkeypatch, [{"prompt": f"q{i}", "expected": "x"} for i in range(6)],
               _gen([1] * 6), _gen([2] * 6),
               "--config", "soup.yaml", "--task-mode", "metric",
               files={"soup.yaml": _config("reward", "reward.py"),
                      "reward.py": f"open('{marker}', 'w').write('x')\n"},
               mode_flags=False, evidence=None)
    assert run.result.exit_code in (0, 2), (run.result.output, repr(run.result.exception))
    assert run.json("out.json")["task_win"]["mode"] == "metric"
    assert marker not in run.files


def test_a_config_that_names_the_mode_without_a_function_needs_the_flag(tmp_path, monkeypatch):
    run = _refuse(tmp_path, monkeypatch, "--config", "soup.yaml",
                  files={"soup.yaml": _config("reward", None)})
    assert "--task-mode reward needs --reward-fn" in run.plain


def test_a_config_function_is_contained_like_the_flag(tmp_path, monkeypatch):
    run = _refuse(tmp_path, monkeypatch, "--config", "soup.yaml",
                  files={"soup.yaml": _config("reward", "../outside.py")})
    assert "must stay under cwd" in run.plain


def test_re_stamping_evidence_beside_a_config_that_names_a_reward_file_runs_nothing(
    tmp_path, monkeypatch
):
    marker = "ran.txt"
    work = _work(tmp_path, monkeypatch, _tasks(3), {
        "soup.yaml": _config("reward", "reward.py"),
        "reward.py": f"open('{marker}', 'w').write('x')\n",
    })
    verdict, spec = _reward_verdict(BASE, TUNED)
    (work / "ev.json").write_text(json.dumps(verdict_to_evidence(verdict)), encoding="utf-8")
    result = runner.invoke(
        ship_cmd.app,
        ["--evidence", "ev.json", "--config", "soup.yaml", "--emit-evidence", "stamped.json"],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert not (work / marker).exists()
    stamped = json.loads((work / "stamped.json").read_text(encoding="utf-8"))
    assert stamped["task"]["reward"] == {"fn": spec.fn, "sha256": spec.sha256}
    # the binding it stamped names the function the evidence names, not the config's file
    policy = stamped["provenance"]["binding"]["policy"]
    assert policy["reward"] == {"fn": spec.fn, "sha256": spec.sha256}


def test_the_binding_policy_names_the_reward_function_for_the_reward_mode_only():
    spec = RewardSpec("rewards/score.py", "c" * 64)

    def binding(mode, reward):
        return ship_cmd._compute_ship_binding(
            None, adapter=None, task_eval=None, suite_names=["mini_arithmetic"],
            task_mode=mode, judge_model=None, threshold=0.05, noise_floor_runs=None,
            data_sha256=None, reward=reward,
        )

    reward_policy = binding("reward", spec)["policy"]
    assert reward_policy["reward"] == {"fn": "rewards/score.py", "sha256": "c" * 64}
    assert "reward" not in binding("metric", None)["policy"]
    other = binding("reward", RewardSpec("rewards/score.py", "d" * 64))
    assert other["policy_sha256"] != binding("reward", spec)["policy_sha256"]


# -- the terminal ----------------------------------------------------------------------------------


def test_a_hostile_function_name_in_evidence_never_reaches_the_terminal_raw(
    tmp_path, monkeypatch
):
    # the reader refuses a control character, so a file that carries one is refused as a whole,
    # with the name left out of the message
    work = _work(tmp_path, monkeypatch, _tasks(3))
    verdict, _ = _reward_verdict(BASE, TUNED)
    evidence = verdict_to_evidence(verdict)
    evidence["task"]["reward"]["fn"] = "\x1b[2Jevil.py"
    (work / "ev.json").write_text(json.dumps(evidence), encoding="utf-8")
    result = runner.invoke(ship_cmd.app, ["--evidence", "ev.json"])
    assert result.exit_code == 3
    assert "\x1b" not in result.output and "evil" not in result.output


@pytest.mark.parametrize("name", ["r[bold red]x.py", "r[/bold]x.py"])
def test_a_markup_looking_path_is_printed_as_text(tmp_path, monkeypatch, name):
    run = _run(tmp_path, monkeypatch, _tasks(6), _gen([1] * 6), _gen([2] * 6),
               files={name: LEN_REWARD}, fn=name)
    assert run.result.exit_code == 0, (run.result.output, repr(run.result.exception))
    assert f"Reward function: {name} (sha256" in run.plain
    assert f"--reward-fn {name} is a Python file and soup ship EXECUTES it" in run.plain


# -- the mode touches no other mode ----------------------------------------------------------------


def test_the_other_modes_have_no_reward_and_no_summary_in_their_output(tmp_path, monkeypatch):
    tasks = [{"prompt": f"q{i}", "expected": "x"} for i in range(6)]
    run = _run(tmp_path, monkeypatch, tasks, _gen([1] * 6), _gen([2] * 6), mode_flags=False)
    block = run.json("out.json")["task_win"]
    assert block["reward"] is None and block["paired"] is None
    assert "reward" not in run.json("ev.json")["task"]
    assert "paired" not in run.json("ev.json")["task"]


# -- --require-provenance binds the reward file without running it --------------------------------

PROVENANCE_CONFIG = (
    "base: owner/base\n"
    "data: {train: train.jsonl}\n"
    "training: {quantization: none}\n"
    "eval:\n"
    "  ship:\n"
    "    task_eval: task.jsonl\n"
    "    task_mode: reward\n"
    "{reward}"
    "    general_suite: mini_mmlu\n"
    "    forgetting_threshold: 0.05\n"
)

MARKED_REWARD = "open('ran.txt', 'w').write('x')\n" + LEN_REWARD


def _provenance_setup(tmp_path, monkeypatch, *, reward="    reward_fn: reward.py\n"):
    work = _work(tmp_path, monkeypatch, _tasks(2), {
        "reward.py": MARKED_REWARD,
        "adapter/adapter_model.safetensors": "local fixture weights",
        "train.jsonl": '{"text":"train"}\n',
        "soup.yaml": PROVENANCE_CONFIG.replace("{reward}", reward),
    })
    digest = hashlib.sha256(MARKED_REWARD.encode("utf-8")).hexdigest()
    raw = {
        "schema_version": 2,
        "task": {"mode": "reward", "base": 0.5, "tuned": 0.7,
                 "reward": {"fn": "reward.py", "sha256": digest}},
        "benchmarks": {"mini_mmlu": {"base": 0.6, "tuned": 0.6}},
        "numerics": "float32",
    }
    (work / "raw.json").write_text(json.dumps(raw), encoding="utf-8")
    produced = runner.invoke(ship_cmd.app, [
        "--evidence", "raw.json", "--config", "soup.yaml", "--adapter", "adapter",
        "--emit-evidence", "bound.json",
    ])
    assert produced.exit_code == 0, (produced.output, repr(produced.exception))
    return work, digest


def _gate(*extra):
    return runner.invoke(ship_cmd.app, [
        "--evidence", "bound.json", "--config", "soup.yaml", "--adapter", "adapter",
        "--require-provenance", *extra,
    ])


def test_the_binding_names_the_reward_file_and_the_gate_compares_it_without_running_it(
    tmp_path, monkeypatch
):
    work, digest = _provenance_setup(tmp_path, monkeypatch)
    bound = json.loads((work / "bound.json").read_text(encoding="utf-8"))
    assert bound["provenance"]["binding"]["policy"]["reward"] == {
        "fn": "reward.py", "sha256": digest,
    }
    ok = _gate()
    assert ok.exit_code == 0, (ok.output, repr(ok.exception))
    assert not (work / "ran.txt").exists()  # hashed, never run
    (work / "reward.py").write_bytes((MARKED_REWARD + "# edited\n").encode("utf-8"))
    stale = _gate()
    assert stale.exit_code == 3, (stale.output, repr(stale.exception))
    assert "provenance" in _plain(stale.output).lower()
    assert not (work / "ran.txt").exists()


def test_the_gate_in_reward_mode_needs_the_function_it_compares(tmp_path, monkeypatch):
    work, _ = _provenance_setup(tmp_path, monkeypatch)
    (work / "soup.yaml").write_text(PROVENANCE_CONFIG.replace("{reward}", ""), encoding="utf-8")
    result = _gate()
    assert result.exit_code == 3, (result.output, repr(result.exception))
    assert "--require-provenance with --task-mode reward needs --reward-fn" in _plain(
        result.output
    )
    assert not (work / "ran.txt").exists()


def test_a_foreign_file_at_the_rows_path_is_refused_before_the_reward_file_runs(
    tmp_path, monkeypatch
):
    # the early check of the per-row file covers the reward mode, and it runs before the reward
    # file is executed and before any model loads
    marker = "ran.txt"
    run = _refuse(
        tmp_path, monkeypatch, "--task-mode", "reward", "--reward-fn", "reward.py",
        "--output", "out.json",
        files={
            "reward.py": f"open('{marker}', 'w').write('x')\n" + LEN_REWARD,
            "out.task-rows.jsonl": "not a file of ours\n",
        },
    )
    assert "is not a task-rows file written by soup ship" in run.plain
    assert marker not in run.files
    assert run.files["out.task-rows.jsonl"] == b"not a file of ours\n"
    assert "out.json" not in run.files


def _rows_file_and_verdict(work, mode):
    """Three rows of 0.1 0.2 0.3 (base) against 0.3 0.2 0.1 (tuned) and the verdict that names
    them: the exact decimal mean of each column is 0.2, a plain left-to-right sum is not."""
    from soup_cli.utils.ship_task_rows import TaskRow, prompt_hash, rows_to_jsonl
    from soup_cli.utils.ship_verdict import row_change

    base, tuned = [0.1, 0.2, 0.3], [0.3, 0.2, 0.1]
    rows = [
        TaskRow(
            index=i, prompt_hash=prompt_hash(f"q{i}"), prompt=f"q{i}", expected="e",
            category="default", scoring=None, base_answer="a", tuned_answer="b",
            base_score=b, tuned_score=t, change=row_change(b, t), base_truncated=None,
            tuned_truncated=None, base_failed=False, tuned_failed=False,
        )
        for i, (b, t) in enumerate(zip(base, tuned))
    ]
    (work / "v.task-rows.jsonl").write_text(rows_to_jsonl(rows), encoding="ascii", newline="")
    verdict = {
        "task_win": {
            "mode": mode, "base": 0.2, "tuned": 0.2,
            "rows": {"better": 1, "worse": 1, "same": 1, "n": 3},
        },
        "task_rows": "v.task-rows.jsonl",
    }
    (work / "v.json").write_text(json.dumps(verdict), encoding="utf-8")
    return str(work / "v.json")


def test_the_loader_takes_the_exact_decimal_mean_for_a_reward_verdict_and_no_other(
    tmp_path, monkeypatch
):
    work = _work(tmp_path, monkeypatch, _tasks(3))
    loaded = load_task_rows_for(_rows_file_and_verdict(work, "reward"))
    assert loaded.base_mean == loaded.tuned_mean == 0.2
    # the same numbers under another mode are checked with the plain loop (whose sum is one
    # rounding off), exactly for a metric verdict
    with pytest.raises(ValueError, match="is not the reported"):
        load_task_rows_for(_rows_file_and_verdict(work, "metric"))
