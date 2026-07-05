"""v0.71.31 — Judge-in-the-loop suite.

Covers the shared pairwise-judge layer (``eval/judge.pairwise_compare`` /
``pairwise_winrate`` / ``make_soup_pairwise_judge``), ``soup ship --task-mode
pairwise`` (#284), ``task='online_dpo'`` (schema + trainer + routing),
``soup data best-of-n``, and ``soup data evolve``.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Shared test doubles
# ---------------------------------------------------------------------------


class _FakeJudge:
    """Deterministic pairwise judge: prefers the LONGER response (A vs B)."""

    def __init__(self, rubric=None):
        self.rubric = rubric or {"scale": {"min": 1, "max": 5}, "criteria": []}

    def compare_pair(self, prompt, resp_a, resp_b):
        if len(resp_a) == len(resp_b):
            return -1
        return 0 if len(resp_a) > len(resp_b) else 1


class _PosBias:
    """A biased judge that ALWAYS says the first response is best."""

    def compare_pair(self, prompt, resp_a, resp_b):
        return 0


# ---------------------------------------------------------------------------
# Task 1 — pairwise_compare / pairwise_winrate
# ---------------------------------------------------------------------------


class TestPairwiseCompare:
    def test_a_preferred(self):
        from soup_cli.eval.judge import pairwise_compare

        assert pairwise_compare("p", "longer response", "short", _FakeJudge(), swap=True) == 0

    def test_b_preferred(self):
        from soup_cli.eval.judge import pairwise_compare

        assert pairwise_compare("p", "short", "longer response", _FakeJudge(), swap=True) == 1

    def test_tie_on_equal(self):
        from soup_cli.eval.judge import pairwise_compare

        assert pairwise_compare("p", "aaaa", "bbbb", _FakeJudge(), swap=True) == -1

    def test_swap_debias_disagreement_is_tie(self):
        from soup_cli.eval.judge import pairwise_compare

        # A judge that ALWAYS says "first is best" disagrees under swap -> tie.
        assert pairwise_compare("p", "x", "y", _PosBias(), swap=True) == -1

    def test_no_swap_uses_single_call(self):
        from soup_cli.eval.judge import pairwise_compare

        assert pairwise_compare("p", "x", "y", _PosBias(), swap=False) == 0


class TestPairwiseWinrate:
    def test_tuned_always_wins(self):
        from soup_cli.eval.judge import pairwise_winrate

        # base short, tuned long -> _FakeJudge prefers tuned every time -> 1.0
        pairs = [("p", "s", "longer"), ("q", "s", "longer")]
        assert pairwise_winrate(pairs, _FakeJudge()) == 1.0

    def test_all_ties_is_half(self):
        from soup_cli.eval.judge import pairwise_winrate

        pairs = [("p", "aaa", "bbb")]  # equal length -> tie -> 0.5
        assert pairwise_winrate(pairs, _FakeJudge()) == 0.5

    def test_empty_pairs_is_half(self):
        from soup_cli.eval.judge import pairwise_winrate

        assert pairwise_winrate([], _FakeJudge()) == 0.5

    def test_mixed_winrate(self):
        from soup_cli.eval.judge import pairwise_winrate

        # tuned wins (long), tuned loses (short), tie (equal) -> (1 + 0 + 0.5)/3
        pairs = [("a", "s", "longer"), ("b", "longer", "s"), ("c", "xx", "yy")]
        assert pairwise_winrate(pairs, _FakeJudge()) == (1.0 + 0.0 + 0.5) / 3


# ---------------------------------------------------------------------------
# Task 2 — make_soup_pairwise_judge (TRL BasePairwiseJudge adapter)
# ---------------------------------------------------------------------------


class TestSoupPairwiseJudge:
    def test_judge_returns_best_index(self):
        from soup_cli.eval.judge import make_soup_pairwise_judge

        j = make_soup_pairwise_judge(_FakeJudge())  # prefers longer
        # prompt0: [short, long] -> B(1); prompt1: [long, short] -> A(0)
        out = j.judge(["p0", "p1"], [["s", "longer"], ["longer", "s"]])
        assert out == [1, 0]

    def test_judge_tie_returns_minus_one(self):
        from soup_cli.eval.judge import make_soup_pairwise_judge

        j = make_soup_pairwise_judge(_FakeJudge())
        assert j.judge(["p"], [["aaaa", "bbbb"]]) == [-1]

    def test_shuffle_order_false_no_swap(self):
        from soup_cli.eval.judge import make_soup_pairwise_judge

        j = make_soup_pairwise_judge(_PosBias())
        assert j.judge(["p"], [["x", "y"]], shuffle_order=False) == [0]

    def test_malformed_pair_returns_minus_one(self):
        from soup_cli.eval.judge import make_soup_pairwise_judge

        j = make_soup_pairwise_judge(_FakeJudge())
        assert j.judge(["p"], [["only-one"]]) == [-1]

    def test_is_trl_base_pairwise_judge(self):
        from trl import BasePairwiseJudge

        from soup_cli.eval.judge import make_soup_pairwise_judge

        assert isinstance(make_soup_pairwise_judge(_FakeJudge()), BasePairwiseJudge)


# ---------------------------------------------------------------------------
# Task 3 — soup ship --task-mode pairwise (#284)
# ---------------------------------------------------------------------------


class _Task:
    def __init__(self, prompt):
        self.prompt = prompt
        self.category = "default"


class TestShipPairwise:
    def test_pairwise_in_supported(self):
        from soup_cli.utils.ship_verdict import SUPPORTED_TASK_MODES

        assert "pairwise" in SUPPORTED_TASK_MODES

    def test_leg1_pairwise_builds_taskwin(self, monkeypatch):
        from soup_cli.commands import ship as ship_cmd

        monkeypatch.setattr(
            "soup_cli.eval.custom.load_eval_tasks",
            lambda path: [_Task("q1"), _Task("q2")],
        )
        monkeypatch.setattr(
            "soup_cli.eval.gate._parse_judge_url",
            lambda url: ("ollama", "m", None),
        )
        monkeypatch.setattr(
            "soup_cli.eval.judge.JudgeEvaluator", lambda **kw: _FakeJudge()
        )
        # base_gen short, tuned_gen long; _FakeJudge prefers long -> winrate 1.0
        tw = ship_cmd._leg1_pairwise(
            lambda p: "s", lambda p: "longer", "x.jsonl", "ollama://m"
        )
        assert tw.mode == "pairwise"
        assert tw.base == 0.5
        assert tw.tuned == 1.0
        assert tw.won is True

    def test_evidence_pairwise_accepted(self):
        import json
        import os

        from typer.testing import CliRunner

        from soup_cli.commands.ship import app

        path = os.path.join(os.getcwd(), "_ev_pairwise_v07131.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "task": {"mode": "pairwise", "base": 0.5, "tuned": 0.7},
                    "benchmarks": {"mini_mmlu": {"base": 0.8, "tuned": 0.8}},
                },
                fh,
            )
        try:
            result = CliRunner().invoke(app, ["--evidence", path])
            assert result.exit_code == 0, (result.output, repr(result.exception))
            assert "SHIP" in result.output
        finally:
            os.remove(path)

    def test_evidence_pairwise_tie_is_dont_ship(self):
        import json
        import os

        from typer.testing import CliRunner

        from soup_cli.commands.ship import app

        path = os.path.join(os.getcwd(), "_ev_pairwise_tie_v07131.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "task": {"mode": "pairwise", "base": 0.5, "tuned": 0.5},
                    "benchmarks": {"mini_mmlu": {"base": 0.8, "tuned": 0.8}},
                },
                fh,
            )
        try:
            result = CliRunner().invoke(app, ["--evidence", path])
            assert result.exit_code == 2, (result.output, repr(result.exception))
            assert "DON'T SHIP" in result.output
        finally:
            os.remove(path)
