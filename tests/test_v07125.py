"""v0.71.25 "soup ship — SHIP / DON'T-SHIP verdict engine" tests.

`soup ship` fuses two questions into one binary decision after fine-tuning:

  SHIP  <=>  (leg 1: task_tuned  >  task_base   — STRICT inequality)
        AND  (leg 2: for every benchmark:  base - tuned  <=  forgetting_threshold)
  else DON'T SHIP — even if the task metric looks great.

The moat is leg 2 (catastrophic-forgetting / regression gate) as a first-class
co-equal of leg 1, fused into ONE verdict. The regression delta math is reused
from ``eval/gate.py`` / ``eval/leaderboard.py`` semantics
(``delta = tuned - base``; regress when the drop exceeds the threshold).

Engine (``utils/ship_verdict.py``) is pure-python (NO top-level torch). The CLI
(``commands/ship.py``) orchestrates base+tuned eval and exits 0 = SHIP /
2 = DON'T SHIP / 1 = runtime error, mirroring ``soup diagnose``.
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli import __version__
from soup_cli.utils.ship_verdict import (
    DECISION_DONT_SHIP,
    DECISION_SHIP,
    FAILED_MISSING_BASELINE,
    FAILED_REGRESSION,
    FAILED_TASK_WIN,
    SUPPORTED_TASK_MODES,
    TASK_MODES,
    ShipVerdict,
    TaskWin,
    build_task_win,
    compute_benchmark_deltas,
    decide_ship,
    format_ship_rubric,
    render_ship_panel,
    verdict_to_dict,
)

_SRC = Path(__file__).resolve().parent.parent / "src" / "soup_cli"

runner = CliRunner()


def _module_head(rel_path: str) -> str:
    src = (_SRC / rel_path).read_text(encoding="utf-8").replace("\r\n", "\n")
    for marker in ("\ndef ", "\nclass "):
        src = src.split(marker, 1)[0]
    return src


# ---------------------------------------------------------------------------
# build_task_win
# ---------------------------------------------------------------------------

class TestBuildTaskWin:
    def test_strict_win(self):
        win = build_task_win("metric", 0.40, 0.55)
        assert isinstance(win, TaskWin)
        assert win.mode == "metric"
        assert win.base == 0.40
        assert win.tuned == 0.55
        assert win.won is True

    def test_tie_is_not_a_win(self):
        # STRICT inequality — equal scores never SHIP.
        assert build_task_win("metric", 0.50, 0.50).won is False

    def test_worse_is_not_a_win(self):
        assert build_task_win("metric", 0.60, 0.50).won is False

    def test_judge_score_mode(self):
        win = build_task_win("judge_score", 6.0, 7.5)
        assert win.mode == "judge_score"
        assert win.won is True

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="mode"):
            build_task_win("bogus", 0.1, 0.2)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite(self, bad):
        with pytest.raises(ValueError, match="finite"):
            build_task_win("metric", 0.1, bad)

    def test_rejects_bool(self):
        with pytest.raises(TypeError):
            build_task_win("metric", True, 0.2)

    def test_frozen(self):
        import dataclasses

        win = build_task_win("metric", 0.1, 0.2)
        with pytest.raises(dataclasses.FrozenInstanceError):
            win.won = False  # type: ignore[misc]

    def test_supported_modes_subset_of_all(self):
        assert set(SUPPORTED_TASK_MODES).issubset(set(TASK_MODES))
        assert "metric" in SUPPORTED_TASK_MODES
        assert "judge_score" in SUPPORTED_TASK_MODES
        # pairwise became supported in v0.71.31 (#284).
        assert "pairwise" in TASK_MODES
        assert "pairwise" in SUPPORTED_TASK_MODES


# ---------------------------------------------------------------------------
# compute_benchmark_deltas
# ---------------------------------------------------------------------------

class TestComputeBenchmarkDeltas:
    def test_basic_delta(self):
        deltas = compute_benchmark_deltas(
            {"mini_mmlu": 0.80}, {"mini_mmlu": 0.78}, forgetting_threshold=0.05
        )
        assert len(deltas) == 1
        delta = deltas[0]
        assert delta.name == "mini_mmlu"
        assert delta.base == 0.80
        assert delta.tuned == 0.78
        # delta is signed tuned - base (leaderboard semantics).
        assert delta.delta == pytest.approx(-0.02)
        assert delta.regressed is False

    def test_regression_flagged(self):
        deltas = compute_benchmark_deltas(
            {"b": 0.80}, {"b": 0.70}, forgetting_threshold=0.05
        )
        assert deltas[0].regressed is True

    def test_improvement_never_regresses(self):
        deltas = compute_benchmark_deltas(
            {"b": 0.50}, {"b": 0.90}, forgetting_threshold=0.05
        )
        assert deltas[0].delta == pytest.approx(0.40)
        assert deltas[0].regressed is False

    def test_boundary_exactly_threshold_is_ok(self):
        # -5.00% drop is OK (<= threshold). 0.80 - 0.75 is 0.05000...4 in float,
        # which WOULD trip a bare `> 0.05` — this pins that _REGRESSION_EPS
        # absorbs the noise.
        assert (0.80 - 0.75) > 0.05  # the float noise is real
        deltas = compute_benchmark_deltas(
            {"b": 0.80}, {"b": 0.75}, forgetting_threshold=0.05
        )
        assert deltas[0].regressed is False

    def test_non_string_keys_coerced(self):
        deltas = compute_benchmark_deltas({1: 0.5}, {1: 0.4}, forgetting_threshold=0.05)
        assert deltas[0].name == "1"

    def test_boundary_just_past_threshold_regresses(self):
        # -5.01% drop regresses.
        deltas = compute_benchmark_deltas(
            {"b": 0.80}, {"b": 0.7499}, forgetting_threshold=0.05
        )
        assert deltas[0].regressed is True

    def test_only_common_benchmarks(self):
        # A benchmark present in only one map cannot be compared — skip it.
        deltas = compute_benchmark_deltas(
            {"a": 0.5, "b": 0.5}, {"a": 0.5, "c": 0.5}, forgetting_threshold=0.05
        )
        names = {d.name for d in deltas}
        assert names == {"a"}

    def test_deterministic_order(self):
        deltas = compute_benchmark_deltas(
            {"z": 0.5, "a": 0.5, "m": 0.5},
            {"z": 0.5, "a": 0.5, "m": 0.5},
            forgetting_threshold=0.05,
        )
        assert [d.name for d in deltas] == ["a", "m", "z"]

    def test_rejects_bad_threshold(self):
        with pytest.raises(ValueError):
            compute_benchmark_deltas({"b": 0.5}, {"b": 0.5}, forgetting_threshold=-0.1)
        with pytest.raises(ValueError):
            compute_benchmark_deltas({"b": 0.5}, {"b": 0.5}, forgetting_threshold=2.0)


# ---------------------------------------------------------------------------
# decide_ship — the moat truth table
# ---------------------------------------------------------------------------

def _deltas(pairs, threshold=0.05):
    base = {n: b for n, b, _ in pairs}
    tuned = {n: t for n, _, t in pairs}
    return compute_benchmark_deltas(base, tuned, forgetting_threshold=threshold)


class TestDecideShip:
    def test_both_legs_pass_ships(self):
        win = build_task_win("metric", 0.40, 0.55)
        deltas = _deltas([("mini_mmlu", 0.80, 0.79), ("mini_cs", 0.60, 0.61)])
        verdict = decide_ship(win, deltas, forgetting_threshold=0.05)
        assert verdict.decision == DECISION_SHIP
        assert verdict.failed_rule is None
        assert verdict.soup_version == __version__

    def test_leg1_tie_dont_ship(self):
        win = build_task_win("metric", 0.50, 0.50)
        deltas = _deltas([("mini_mmlu", 0.80, 0.81)])
        verdict = decide_ship(win, deltas)
        assert verdict.decision == DECISION_DONT_SHIP
        assert verdict.failed_rule == FAILED_TASK_WIN

    def test_leg1_worse_dont_ship(self):
        win = build_task_win("metric", 0.60, 0.50)
        deltas = _deltas([("mini_mmlu", 0.80, 0.81)])
        verdict = decide_ship(win, deltas)
        assert verdict.decision == DECISION_DONT_SHIP
        assert verdict.failed_rule == FAILED_TASK_WIN

    def test_leg2_regression_dont_ship_even_if_task_wins(self):
        # The canonical moat case — task metric looks great, but a general
        # benchmark regressed: DON'T SHIP.
        win = build_task_win("metric", 0.40, 0.95)
        deltas = _deltas([("mini_mmlu", 0.80, 0.60)])
        verdict = decide_ship(win, deltas)
        assert verdict.decision == DECISION_DONT_SHIP
        assert verdict.failed_rule == FAILED_REGRESSION

    def test_multi_benchmark_one_regresses_names_it(self):
        win = build_task_win("metric", 0.40, 0.55)
        deltas = _deltas(
            [("mini_mmlu", 0.80, 0.80), ("mini_cs", 0.60, 0.40), ("mini_in", 0.5, 0.5)]
        )
        verdict = decide_ship(win, deltas)
        assert verdict.decision == DECISION_DONT_SHIP
        assert verdict.failed_rule == FAILED_REGRESSION
        rubric = format_ship_rubric(verdict)
        assert "mini_cs" in rubric  # the reason names which benchmark broke
        assert "REGRESSED" in rubric  # ...and flags it as regressed

    def test_missing_baseline_refuses(self):
        # No benchmarks measured -> cannot verify leg 2 -> refuse (NOT SHIP).
        win = build_task_win("metric", 0.40, 0.99)
        verdict = decide_ship(win, [])
        assert verdict.decision == DECISION_DONT_SHIP
        assert verdict.failed_rule == FAILED_MISSING_BASELINE

    def test_missing_baseline_precedence_over_leg1(self):
        win = build_task_win("metric", 0.50, 0.50)  # leg1 also fails
        verdict = decide_ship(win, [])
        assert verdict.failed_rule == FAILED_MISSING_BASELINE

    def test_leg1_precedence_over_leg2(self):
        # Both legs fail — leg 1 (did it improve?) is the headline.
        win = build_task_win("metric", 0.60, 0.40)
        deltas = _deltas([("mini_mmlu", 0.80, 0.50)])
        verdict = decide_ship(win, deltas)
        assert verdict.failed_rule == FAILED_TASK_WIN

    def test_leg1_tie_with_regression_reports_task_win(self):
        # A leg-1 TIE (not just "worse") still takes precedence over a leg-2
        # regression in the failed-rule report.
        win = build_task_win("metric", 0.50, 0.50)
        deltas = _deltas([("b", 0.80, 0.60)])
        verdict = decide_ship(win, deltas)
        assert verdict.decision == DECISION_DONT_SHIP
        assert verdict.failed_rule == FAILED_TASK_WIN

    def test_boundary_exact_threshold_ships(self):
        win = build_task_win("metric", 0.40, 0.55)
        deltas = _deltas([("b", 0.80, 0.75)], threshold=0.05)  # -5.00% drop
        verdict = decide_ship(win, deltas, forgetting_threshold=0.05)
        assert verdict.decision == DECISION_SHIP

    def test_boundary_just_past_threshold_dont_ship(self):
        win = build_task_win("metric", 0.40, 0.55)
        deltas = _deltas([("b", 0.80, 0.7499)], threshold=0.05)  # -5.01% drop
        verdict = decide_ship(win, deltas, forgetting_threshold=0.05)
        assert verdict.decision == DECISION_DONT_SHIP

    def test_decide_ship_is_single_source_of_truth_for_threshold(self):
        # Deltas built with a LENIENT threshold (regressed=False) must still be
        # caught when decide_ship is told to use a STRICTER threshold.
        win = build_task_win("metric", 0.40, 0.55)
        lenient = compute_benchmark_deltas(
            {"b": 0.80}, {"b": 0.70}, forgetting_threshold=0.50
        )
        assert lenient[0].regressed is False  # lenient build said OK
        verdict = decide_ship(win, lenient, forgetting_threshold=0.05)
        assert verdict.decision == DECISION_DONT_SHIP
        assert verdict.failed_rule == FAILED_REGRESSION
        # The verdict carries CANONICAL deltas recomputed at the strict threshold.
        assert verdict.benchmark_deltas[0].regressed is True

    def test_records_threshold(self):
        win = build_task_win("metric", 0.4, 0.5)
        verdict = decide_ship(win, _deltas([("b", 0.5, 0.5)]), forgetting_threshold=0.1)
        assert verdict.forgetting_threshold == 0.1

    def test_rejects_bad_threshold(self):
        win = build_task_win("metric", 0.4, 0.5)
        for bad in (-0.01, 1.5, float("nan")):
            with pytest.raises(ValueError):
                decide_ship(win, _deltas([("b", 0.5, 0.5)]), forgetting_threshold=bad)

    def test_rejects_bad_types(self):
        with pytest.raises(TypeError):
            decide_ship("not-a-taskwin", [])  # type: ignore[arg-type]
        win = build_task_win("metric", 0.4, 0.5)
        with pytest.raises(TypeError):
            decide_ship(win, [{"name": "b"}])  # type: ignore[list-item]

    def test_verdict_frozen(self):
        win = build_task_win("metric", 0.4, 0.5)
        verdict = decide_ship(win, _deltas([("b", 0.5, 0.5)]))
        import dataclasses

        with pytest.raises(dataclasses.FrozenInstanceError):
            verdict.decision = "X"  # type: ignore[misc]
        assert isinstance(verdict, ShipVerdict)


# ---------------------------------------------------------------------------
# render + rubric + serialization
# ---------------------------------------------------------------------------

class TestRenderAndSerialize:
    def _ship(self):
        win = build_task_win("metric", 0.40, 0.55)
        return decide_ship(win, _deltas([("mini_mmlu", 0.80, 0.81)]))

    def _dont(self):
        win = build_task_win("metric", 0.40, 0.99)
        return decide_ship(win, _deltas([("mini_mmlu", 0.80, 0.50)]))

    def test_rubric_names_decision_and_legs(self):
        rubric = format_ship_rubric(self._ship())
        assert "SHIP" in rubric
        assert "metric" in rubric
        assert "mini_mmlu" in rubric

    def test_rubric_names_failed_rule(self):
        rubric = format_ship_rubric(self._dont())
        assert "DON'T SHIP" in rubric
        assert FAILED_REGRESSION in rubric  # the rule code appears verbatim
        assert "regressed" in rubric.lower()  # and the human state is named

    def test_rubric_rejects_non_verdict(self):
        with pytest.raises(TypeError):
            format_ship_rubric("nope")  # type: ignore[arg-type]

    def test_panel_renders_to_text(self):
        panel = render_ship_panel(self._dont())
        buf = StringIO()
        Console(file=buf, width=100).print(panel)
        out = buf.getvalue()
        assert "DON'T SHIP" in out

    def test_panel_rejects_non_verdict(self):
        with pytest.raises(TypeError):
            render_ship_panel(42)  # type: ignore[arg-type]

    def test_to_dict_round_trips(self):
        verdict = self._dont()
        payload = verdict_to_dict(verdict)
        # JSON-serializable.
        text = json.dumps(payload)
        loaded = json.loads(text)
        assert loaded["decision"] == DECISION_DONT_SHIP
        assert loaded["failed_rule"] == FAILED_REGRESSION
        assert loaded["task_win"]["mode"] == "metric"
        assert loaded["soup_version"] == __version__
        assert isinstance(loaded["benchmark_deltas"], list)
        assert loaded["benchmark_deltas"][0]["name"] == "mini_mmlu"

    def test_to_dict_ship_failed_rule_is_json_null(self):
        loaded = json.loads(json.dumps(verdict_to_dict(self._ship())))
        assert loaded["decision"] == DECISION_SHIP
        assert loaded["failed_rule"] is None  # JSON null, NOT the string "None"
        assert loaded["task_win"]["won"] is True

    def test_panel_ship_has_no_regression(self):
        panel = render_ship_panel(self._ship())
        buf = StringIO()
        Console(file=buf, width=100).print(panel)
        out = buf.getvalue()
        assert "SHIP" in out
        assert "DON'T" not in out
        assert "REGRESSED" not in out

    def test_panel_names_regressed_benchmark(self):
        panel = render_ship_panel(self._dont())
        buf = StringIO()
        Console(file=buf, width=100).print(panel)
        out = buf.getvalue()
        assert "mini_mmlu" in out  # footer reason names the broken benchmark


# ---------------------------------------------------------------------------
# engine purity
# ---------------------------------------------------------------------------

class TestEnginePurity:
    def test_no_top_level_torch(self):
        head = _module_head("utils/ship_verdict.py")
        for mod in ("torch", "transformers", "peft", "trl"):
            assert f"\nimport {mod}" not in head, f"top-level import {mod}"
            assert f"\nfrom {mod} " not in head, f"top-level from {mod}"


# ---------------------------------------------------------------------------
# CLI — offline --evidence path (primary tested contract, no GPU)
# ---------------------------------------------------------------------------

def _write_evidence(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


_EVIDENCE_SHIP = {
    "task": {"mode": "metric", "base": 0.40, "tuned": 0.55},
    "benchmarks": {
        "mini_mmlu": {"base": 0.80, "tuned": 0.79},
        "mini_common_sense": {"base": 0.60, "tuned": 0.62},
    },
}

_EVIDENCE_DONT = {
    "task": {"mode": "metric", "base": 0.40, "tuned": 0.95},
    "benchmarks": {"mini_mmlu": {"base": 0.80, "tuned": 0.50}},
}


class TestShipCliEvidence:
    def test_help(self):
        from soup_cli.commands import ship as ship_cmd

        res = runner.invoke(ship_cmd.app, ["--help"])
        assert res.exit_code == 0, (res.output, repr(res.exception))

    def test_registered_in_main_app(self):
        from soup_cli.cli import app as main_app

        res = runner.invoke(main_app, ["ship", "--help"])
        assert res.exit_code == 0, (res.output, repr(res.exception))
        assert "SHIP" in res.output or "ship" in res.output.lower()

    def test_evidence_ship_exit_0(self):
        from soup_cli.commands import ship as ship_cmd

        with runner.isolated_filesystem():
            _write_evidence(Path("ev.json"), _EVIDENCE_SHIP)
            res = runner.invoke(ship_cmd.app, ["--evidence", "ev.json"])
            assert res.exit_code == 0, (res.output, repr(res.exception))
            assert "SHIP" in res.output

    def test_evidence_dont_ship_exit_2_names_benchmark(self):
        from soup_cli.commands import ship as ship_cmd

        with runner.isolated_filesystem():
            _write_evidence(Path("ev.json"), _EVIDENCE_DONT)
            res = runner.invoke(ship_cmd.app, ["--evidence", "ev.json"])
            assert res.exit_code == 2, (res.output, repr(res.exception))
            assert "DON'T SHIP" in res.output
            assert "mini_mmlu" in res.output

    def test_evidence_leg1_tie_exit_2(self):
        from soup_cli.commands import ship as ship_cmd

        payload = {
            "task": {"mode": "metric", "base": 0.5, "tuned": 0.5},
            "benchmarks": {"mini_mmlu": {"base": 0.8, "tuned": 0.81}},
        }
        with runner.isolated_filesystem():
            _write_evidence(Path("ev.json"), payload)
            res = runner.invoke(ship_cmd.app, ["--evidence", "ev.json"])
            assert res.exit_code == 2, (res.output, repr(res.exception))

    def test_evidence_missing_baseline_refuses_exit_2(self):
        from soup_cli.commands import ship as ship_cmd

        payload = {"task": {"mode": "metric", "base": 0.4, "tuned": 0.9}, "benchmarks": {}}
        with runner.isolated_filesystem():
            _write_evidence(Path("ev.json"), payload)
            res = runner.invoke(ship_cmd.app, ["--evidence", "ev.json"])
            assert res.exit_code == 2, (res.output, repr(res.exception))
            assert "baseline" in res.output.lower()

    @pytest.mark.parametrize(
        "payload, keyword",
        [
            # task missing entirely
            ({"benchmarks": {"mini_mmlu": {"base": 0.8, "tuned": 0.7}}}, "task"),
            # benchmarks not an object
            (
                {"task": {"mode": "metric", "base": 0.4, "tuned": 0.5}, "benchmarks": 42},
                "benchmarks",
            ),
            # benchmark entry missing 'tuned'
            (
                {"task": {"mode": "metric", "base": 0.4, "tuned": 0.5},
                 "benchmarks": {"b": {"base": 0.8}}},
                "tuned",
            ),
            # non-numeric benchmark score
            (
                {"task": {"mode": "metric", "base": 0.4, "tuned": 0.5},
                 "benchmarks": {"b": {"base": 0.8, "tuned": "great"}}},
                "number",
            ),
            # boolean task score
            (
                {"task": {"mode": "metric", "base": True, "tuned": 0.5},
                 "benchmarks": {"b": {"base": 0.8, "tuned": 0.7}}},
                "bool",
            ),
            # unsupported task mode
            (
                {"task": {"mode": "bogus", "base": 0.4, "tuned": 0.5},
                 "benchmarks": {"b": {"base": 0.8, "tuned": 0.7}}},
                "mode",
            ),
        ],
    )
    def test_malformed_evidence_exit_1(self, payload, keyword):
        from soup_cli.commands import ship as ship_cmd

        with runner.isolated_filesystem():
            _write_evidence(Path("ev.json"), payload)
            res = runner.invoke(ship_cmd.app, ["--evidence", "ev.json"])
            assert res.exit_code == 1, (res.output, repr(res.exception))
            assert keyword in res.output.lower()

    def test_evidence_writes_output_json(self):
        from soup_cli.commands import ship as ship_cmd

        with runner.isolated_filesystem():
            _write_evidence(Path("ev.json"), _EVIDENCE_SHIP)
            res = runner.invoke(
                ship_cmd.app, ["--evidence", "ev.json", "--output", "verdict.json"]
            )
            assert res.exit_code == 0, (res.output, repr(res.exception))
            data = json.loads(Path("verdict.json").read_text(encoding="utf-8"))
            assert data["decision"] == DECISION_SHIP
            assert data["soup_version"] == __version__

    def test_custom_forgetting_threshold_flips_decision(self):
        from soup_cli.commands import ship as ship_cmd

        # -10% drop: regresses at 0.05 (DON'T), OK at 0.20 (SHIP, since task won).
        payload = {
            "task": {"mode": "metric", "base": 0.40, "tuned": 0.55},
            "benchmarks": {"mini_mmlu": {"base": 0.80, "tuned": 0.70}},
        }
        with runner.isolated_filesystem():
            _write_evidence(Path("ev.json"), payload)
            strict = runner.invoke(ship_cmd.app, ["--evidence", "ev.json"])
            assert strict.exit_code == 2, (strict.output, repr(strict.exception))
            lenient = runner.invoke(
                ship_cmd.app,
                ["--evidence", "ev.json", "--forgetting-threshold", "0.20"],
            )
            assert lenient.exit_code == 0, (lenient.output, repr(lenient.exception))

    def test_bad_threshold_rejected_usage_exit_3(self):
        # v0.71.38: usage/validation errors exit 3 (was 2), so CI can tell a
        # config typo from a DON'T-SHIP verdict.
        from soup_cli.commands import ship as ship_cmd

        with runner.isolated_filesystem():
            _write_evidence(Path("ev.json"), _EVIDENCE_SHIP)
            res = runner.invoke(
                ship_cmd.app,
                ["--evidence", "ev.json", "--forgetting-threshold", "2.0"],
            )
            assert res.exit_code == 3, (res.output, repr(res.exception))
            assert "threshold" in res.output.lower()

    def test_pairwise_mode_now_supported(self):
        # v0.71.31 (#284): --task-mode pairwise is no longer rejected. The
        # evidence path reads mode from the file (metric here -> SHIP), so the
        # flag is simply accepted (not an exit-2 "later release" refusal).
        from soup_cli.commands import ship as ship_cmd

        with runner.isolated_filesystem():
            _write_evidence(Path("ev.json"), _EVIDENCE_SHIP)
            res = runner.invoke(
                ship_cmd.app,
                ["--evidence", "ev.json", "--task-mode", "pairwise"],
            )
            assert res.exit_code == 0, (res.output, repr(res.exception))
            assert "later release" not in res.output.lower()

    def test_evidence_outside_cwd_rejected(self):
        from soup_cli.commands import ship as ship_cmd

        res = runner.invoke(ship_cmd.app, ["--evidence", "../escape.json"])
        # Evidence read/parse problems are coded exit 1 (mirrors `soup diagnose`).
        assert res.exit_code == 1, (res.output, repr(res.exception))
        assert "cwd" in res.output.lower() or "outside" in res.output.lower()

    def test_evidence_not_a_dict_rejected(self):
        from soup_cli.commands import ship as ship_cmd

        with runner.isolated_filesystem():
            Path("ev.json").write_text("[1, 2, 3]", encoding="utf-8")
            res = runner.invoke(ship_cmd.app, ["--evidence", "ev.json"])
            assert res.exit_code == 1, (res.output, repr(res.exception))
            assert "json object" in res.output.lower() or "evidence" in res.output.lower()

    def test_no_args_errors(self):
        # Neither --evidence nor a live (--base + tuned) combo: refuse clearly.
        from soup_cli.commands import ship as ship_cmd

        res = runner.invoke(ship_cmd.app, [])
        assert res.exit_code == 3, (res.output, repr(res.exception))  # usage (v0.71.38)
        assert "evidence" in res.output.lower()


# ---------------------------------------------------------------------------
# CLI — live path with injected fake generators (no GPU / no model load)
# ---------------------------------------------------------------------------

from soup_cli.eval.forgetting import MINI_BENCHMARKS  # noqa: E402


def _gold_answer(prompt: str) -> str:
    for bench in MINI_BENCHMARKS.values():
        for item in bench:
            if item["question"] == prompt:
                return item["answer"]
    return ""


def _make_fake_factory(base_cfg: dict, tuned_cfg: dict):
    """Return a stand-in for live_eval.make_generator.

    Branches base vs tuned on the adapter (or a "tuned" model id). Each side
    answers task prompts (marked "TASKMARK") and mini-benchmark prompts
    independently, controlled by ``task_ok`` / ``bench_ok``.
    """

    def factory(model_id, *, adapter=None, device=None, max_new_tokens=64, **kwargs):
        is_tuned = adapter is not None or "tuned" in str(model_id)
        cfg = tuned_cfg if is_tuned else base_cfg

        def gen(prompt: str) -> str:
            if "TASKMARK" in prompt:
                return "the magic widget" if cfg["task_ok"] else "nope"
            return _gold_answer(prompt) if cfg["bench_ok"] else "zzz"

        return gen

    return factory


def _write_task_eval(path: Path) -> None:
    rows = [
        {"prompt": "TASKMARK one", "expected": "widget", "scoring": "contains"},
        {"prompt": "TASKMARK two", "expected": "widget", "scoring": "contains"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


class TestShipCliLive:
    def test_live_ship(self, monkeypatch):
        from soup_cli.commands import ship as ship_cmd
        from soup_cli.utils import live_eval

        monkeypatch.setattr(
            live_eval,
            "make_generator",
            _make_fake_factory(
                base_cfg={"task_ok": False, "bench_ok": False},
                tuned_cfg={"task_ok": True, "bench_ok": True},
            ),
        )
        with runner.isolated_filesystem():
            _write_task_eval(Path("tasks.jsonl"))
            res = runner.invoke(
                ship_cmd.app,
                [
                    "--base", "fake-base",
                    "--adapter", "fake-adapter",
                    "--task-eval", "tasks.jsonl",
                    "--general-suite", "mini_mmlu",
                ],
            )
            assert res.exit_code == 0, (res.output, repr(res.exception))
            assert "SHIP" in res.output

    def test_live_dont_ship_on_regression(self, monkeypatch):
        from soup_cli.commands import ship as ship_cmd
        from soup_cli.utils import live_eval

        # tuned wins the task but breaks the benchmark.
        monkeypatch.setattr(
            live_eval,
            "make_generator",
            _make_fake_factory(
                base_cfg={"task_ok": False, "bench_ok": True},
                tuned_cfg={"task_ok": True, "bench_ok": False},
            ),
        )
        with runner.isolated_filesystem():
            _write_task_eval(Path("tasks.jsonl"))
            res = runner.invoke(
                ship_cmd.app,
                [
                    "--base", "fake-base",
                    "--adapter", "fake-adapter",
                    "--task-eval", "tasks.jsonl",
                    "--general-suite", "mini_mmlu",
                ],
            )
            assert res.exit_code == 2, (res.output, repr(res.exception))
            assert "DON'T SHIP" in res.output
            assert "mini_mmlu" in res.output

    def test_live_requires_task_eval(self, monkeypatch):
        from soup_cli.commands import ship as ship_cmd
        from soup_cli.utils import live_eval

        monkeypatch.setattr(
            live_eval,
            "make_generator",
            _make_fake_factory({"task_ok": True, "bench_ok": True},
                               {"task_ok": True, "bench_ok": True}),
        )
        res = runner.invoke(
            ship_cmd.app, ["--base", "fake-base", "--adapter", "fake-adapter"]
        )
        assert res.exit_code == 3, (res.output, repr(res.exception))
        assert "task-eval" in res.output.lower() or "task_eval" in res.output.lower()

    def test_live_baseline_override_supplies_base_scores(self, monkeypatch):
        from soup_cli.commands import ship as ship_cmd
        from soup_cli.utils import live_eval

        # tuned is perfect on the benchmark; baseline file supplies the base
        # score directly (no base model run needed for leg 2).
        monkeypatch.setattr(
            live_eval,
            "make_generator",
            _make_fake_factory(
                base_cfg={"task_ok": False, "bench_ok": False},
                tuned_cfg={"task_ok": True, "bench_ok": True},
            ),
        )
        with runner.isolated_filesystem():
            _write_task_eval(Path("tasks.jsonl"))
            Path("baseline.json").write_text(
                json.dumps({"mini_mmlu": 0.2}), encoding="utf-8"
            )
            res = runner.invoke(
                ship_cmd.app,
                [
                    "--base", "fake-base",
                    "--adapter", "fake-adapter",
                    "--task-eval", "tasks.jsonl",
                    "--general-suite", "mini_mmlu",
                    "--baseline", "baseline.json",
                ],
            )
            assert res.exit_code == 0, (res.output, repr(res.exception))
            assert "SHIP" in res.output

    def test_live_failure_exit_1(self, monkeypatch):
        from soup_cli.commands import ship as ship_cmd
        from soup_cli.utils import live_eval

        def boom(*args, **kwargs):
            raise RuntimeError("no model here")

        monkeypatch.setattr(live_eval, "make_generator", boom)
        with runner.isolated_filesystem():
            _write_task_eval(Path("tasks.jsonl"))
            res = runner.invoke(
                ship_cmd.app,
                [
                    "--base", "fake-base",
                    "--adapter", "fake-adapter",
                    "--task-eval", "tasks.jsonl",
                ],
            )
            assert res.exit_code == 1, (res.output, repr(res.exception))
            assert "failed" in res.output.lower()

    def test_live_bad_baseline_is_usage_error(self):
        # A bad --baseline (outside cwd) is a USAGE error (exit 2), resolved
        # BEFORE any model load — not a runtime error (exit 1).
        from soup_cli.commands import ship as ship_cmd

        with runner.isolated_filesystem():
            _write_task_eval(Path("tasks.jsonl"))
            res = runner.invoke(
                ship_cmd.app,
                [
                    "--base", "fake-base",
                    "--adapter", "fake-adapter",
                    "--task-eval", "tasks.jsonl",
                    "--general-suite", "mini_mmlu",
                    "--baseline", "../escape.json",
                ],
            )
            assert res.exit_code == 3, (res.output, repr(res.exception))
            assert "baseline" in res.output.lower()


class TestShipCliLmEvalRouting:
    def test_non_mini_suite_routes_through_lm_eval(self, monkeypatch):
        """A --general-suite of non-mini names uses _run_lm_eval (override)."""
        from soup_cli.commands import eval as eval_cmd
        from soup_cli.commands import ship as ship_cmd
        from soup_cli.utils import live_eval

        monkeypatch.setattr(
            live_eval,
            "make_generator",
            _make_fake_factory({"task_ok": False, "bench_ok": False},
                               {"task_ok": True, "bench_ok": True}),
        )

        calls = {"n": 0}

        def fake_lm_eval(model_arg, tasks, num_fewshot, batch_size, device):
            calls["n"] += 1
            # Tuned (adapter in model_arg) scores higher than base.
            score = 0.9 if "peft" in model_arg or "adapter" in model_arg else 0.85
            return {"results": {t: {"acc,none": score} for t in tasks}}

        monkeypatch.setattr(eval_cmd, "_run_lm_eval", fake_lm_eval)
        with runner.isolated_filesystem():
            _write_task_eval(Path("tasks.jsonl"))
            res = runner.invoke(
                ship_cmd.app,
                [
                    "--base", "fake-base",
                    "--adapter", "fake-adapter",
                    "--task-eval", "tasks.jsonl",
                    "--general-suite", "hellaswag",
                ],
            )
            assert res.exit_code == 0, (res.output, repr(res.exception))
            assert calls["n"] >= 1  # the lm-eval override actually ran

    def test_lm_eval_adapter_injection_rejected(self, monkeypatch):
        """An --adapter with ',' / '=' must not smuggle lm-eval model_args."""
        from soup_cli.commands import ship as ship_cmd
        from soup_cli.utils import live_eval

        monkeypatch.setattr(
            live_eval,
            "make_generator",
            _make_fake_factory({"task_ok": False, "bench_ok": False},
                               {"task_ok": True, "bench_ok": True}),
        )
        with runner.isolated_filesystem():
            _write_task_eval(Path("tasks.jsonl"))
            res = runner.invoke(
                ship_cmd.app,
                [
                    "--base", "fake-base",
                    "--adapter", "lora,trust_remote_code=True",
                    "--task-eval", "tasks.jsonl",
                    "--general-suite", "hellaswag",
                ],
            )
            assert res.exit_code == 1, (res.output, repr(res.exception))
            assert "must not contain" in res.output.lower()

    def test_unscored_lm_eval_benchmark_refuses(self, monkeypatch):
        """A requested benchmark that lm-eval can't score must REFUSE, not vanish."""
        from soup_cli.commands import eval as eval_cmd
        from soup_cli.commands import ship as ship_cmd
        from soup_cli.utils import live_eval

        monkeypatch.setattr(
            live_eval,
            "make_generator",
            _make_fake_factory({"task_ok": False, "bench_ok": False},
                               {"task_ok": True, "bench_ok": True}),
        )

        def empty_lm_eval(model_arg, tasks, num_fewshot, batch_size, device):
            return {"results": {}}  # no score for the requested benchmark

        monkeypatch.setattr(eval_cmd, "_run_lm_eval", empty_lm_eval)
        with runner.isolated_filesystem():
            _write_task_eval(Path("tasks.jsonl"))
            res = runner.invoke(
                ship_cmd.app,
                [
                    "--base", "fake-base",
                    "--adapter", "fake-adapter",
                    "--task-eval", "tasks.jsonl",
                    "--general-suite", "hellaswag",
                ],
            )
            # Refuse loudly (runtime error) rather than silently SHIP on missing data.
            assert res.exit_code == 1, (res.output, repr(res.exception))
            assert "could not score" in res.output.lower() or "hellaswag" in res.output.lower()


class TestShipCliSecurity:
    def test_judge_url_ssrf_bypass_rejected(self, monkeypatch):
        """`http://localhost.attacker.com` must NOT pass the judge-URL guard."""
        from soup_cli.commands import ship as ship_cmd
        from soup_cli.utils import live_eval

        monkeypatch.setattr(
            live_eval,
            "make_generator",
            _make_fake_factory({"task_ok": True, "bench_ok": True},
                               {"task_ok": True, "bench_ok": True}),
        )
        with runner.isolated_filesystem():
            _write_task_eval(Path("tasks.jsonl"))
            res = runner.invoke(
                ship_cmd.app,
                [
                    "--base", "fake-base",
                    "--adapter", "fake-adapter",
                    "--task-eval", "tasks.jsonl",
                    "--task-mode", "judge_score",
                    "--judge-model", "http://localhost.attacker.com/model",
                ],
            )
            assert res.exit_code == 3, (res.output, repr(res.exception))
            assert "disallowed" in res.output.lower()

    def test_judge_url_ollama_allowed(self, monkeypatch):
        """An allowlisted ollama:// judge URL passes the guard (reaches eval)."""
        from soup_cli.commands import ship as ship_cmd
        from soup_cli.utils import live_eval

        monkeypatch.setattr(
            live_eval,
            "make_generator",
            _make_fake_factory({"task_ok": True, "bench_ok": True},
                               {"task_ok": True, "bench_ok": True}),
        )

        class _FakeJudge:
            def __init__(self, *a, **k):
                self.rubric = {"scale": {"min": 1, "max": 5}}

            def evaluate_batch(self, items):
                return type("R", (), {"overall_score": 4.0})()

        from soup_cli.eval import judge as judge_mod

        monkeypatch.setattr(judge_mod, "JudgeEvaluator", _FakeJudge)
        with runner.isolated_filesystem():
            _write_task_eval(Path("tasks.jsonl"))
            res = runner.invoke(
                ship_cmd.app,
                [
                    "--base", "fake-base",
                    "--adapter", "fake-adapter",
                    "--task-eval", "tasks.jsonl",
                    "--task-mode", "judge_score",
                    "--judge-model", "ollama://llama3.1",
                    "--general-suite", "mini_mmlu",
                ],
            )
            # base == tuned judge score -> leg-1 tie -> DON'T SHIP (exit 2), but
            # crucially NOT a "disallowed scheme" rejection.
            assert "disallowed" not in res.output.lower()
            assert res.exit_code in (0, 2), (res.output, repr(res.exception))

    def test_task_eval_outside_cwd_rejected(self):
        from soup_cli.commands import ship as ship_cmd

        res = runner.invoke(
            ship_cmd.app,
            [
                "--base", "fake-base",
                "--adapter", "fake-adapter",
                "--task-eval", "../escape.jsonl",
            ],
        )
        assert res.exit_code == 3, (res.output, repr(res.exception))
        assert "cwd" in res.output.lower()

    def test_general_suite_too_large_rejected(self):
        from soup_cli.commands import ship as ship_cmd

        big = ",".join(f"b{i}" for i in range(60))  # > _MAX_SUITE_BENCHMARKS (50)
        with runner.isolated_filesystem():
            _write_task_eval(Path("tasks.jsonl"))
            res = runner.invoke(
                ship_cmd.app,
                [
                    "--base", "fake-base",
                    "--adapter", "fake-adapter",
                    "--task-eval", "tasks.jsonl",
                    "--general-suite", big,
                ],
            )
            assert res.exit_code == 3, (res.output, repr(res.exception))
            assert "too many" in res.output.lower()

    def test_general_suite_long_name_rejected(self):
        from soup_cli.commands import ship as ship_cmd

        with runner.isolated_filesystem():
            _write_task_eval(Path("tasks.jsonl"))
            res = runner.invoke(
                ship_cmd.app,
                [
                    "--base", "fake-base",
                    "--adapter", "fake-adapter",
                    "--task-eval", "tasks.jsonl",
                    "--general-suite", "a" * 300,
                ],
            )
            assert res.exit_code == 3, (res.output, repr(res.exception))
