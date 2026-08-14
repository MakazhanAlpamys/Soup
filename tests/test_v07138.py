"""v0.71.38 — "The gate grows teeth": make `soup ship`'s regression leg real.

Part A — fix the forgetting/MCQ scorer (answer-extraction + boundary-aware
match, replacing the substring `in` that scored "B" for "Berlin").
Part B — bundled general-suite registry (`eval/gate_suites.py`) + JSONL
fixtures scored by the pure diagnose/custom scorers, per-model absolute.
Part C — leg-2 dispatch in `commands/ship.py` into the bundled scorers.
Part D — riders (exit-code collision, stale pairwise docstrings, diagnose
`__init__` export tidy).
"""

from __future__ import annotations

import os

import pytest

# ---------------------------------------------------------------------------
# Part A — the scorer fix
# ---------------------------------------------------------------------------


class TestScoreAnswerBoundary:
    """The headline bug: substring `in` scored `"B"` for `"Berlin"`."""

    def test_substring_within_word_no_longer_matches(self):
        from soup_cli.eval.forgetting import score_answer

        # The three named regressions from the cold-start brief.
        assert score_answer("Berlin", "B") is False  # was True (substring)
        assert score_answer("look", "ok") is False  # was True
        assert score_answer("13", "3") is False  # was True

    def test_standalone_answer_still_matches(self):
        from soup_cli.eval.forgetting import score_answer

        assert score_answer("The answer is B", "B") is True
        assert score_answer("B", "B") is True
        assert score_answer("the number 3", "3") is True
        assert score_answer("Yes, it is hot.", "yes") is True
        assert score_answer("The grass is green.", "green") is True

    def test_okay_is_not_ok(self):
        from soup_cli.eval.forgetting import score_answer

        # `okay` is a different word than the expected `ok`.
        assert score_answer("okay", "ok") is False
        # ...but a standalone `ok` elsewhere counts.
        assert score_answer("okay, ok", "ok") is True

    def test_wrong_letter_is_wrong(self):
        from soup_cli.eval.forgetting import score_answer

        assert score_answer("The answer is A", "B") is False
        assert score_answer("(C) Paris", "B") is False

    def test_non_str_and_empty_answer_are_false(self):
        from soup_cli.eval.forgetting import score_answer

        assert score_answer(None, "B") is False  # type: ignore[arg-type]
        assert score_answer("B", None) is False  # type: ignore[arg-type]
        assert score_answer("B", "") is False
        assert score_answer("B", "   ") is False


class TestExtractMcqLetter:
    def test_cue_form(self):
        from soup_cli.eval.forgetting import extract_mcq_letter

        assert extract_mcq_letter("I think the answer is C.") == "C"
        assert extract_mcq_letter("The correct option: B") == "B"
        assert extract_mcq_letter("answer b") == "B"  # case-insensitive cue

    def test_paren_form(self):
        from soup_cli.eval.forgetting import extract_mcq_letter

        assert extract_mcq_letter("(B)") == "B"
        assert extract_mcq_letter("B) four") == "B"

    def test_bare_uppercase_letter(self):
        from soup_cli.eval.forgetting import extract_mcq_letter

        assert extract_mcq_letter("B") == "B"
        # A lone lowercase article `a` must NOT be read as an MCQ choice.
        assert extract_mcq_letter("it is a dog") is None

    def test_last_letter_wins_on_echo(self):
        from soup_cli.eval.forgetting import extract_mcq_letter

        # Model echoes options then answers with the final decision.
        assert extract_mcq_letter("(A) London (B) Berlin. Answer: B") == "B"

    def test_leading_letter_beats_trailing_pronoun_or_article(self):
        from soup_cli.eval.forgetting import extract_mcq_letter, score_answer

        # A model that leads with its letter then explains must NOT be scored by
        # a trailing capitalized pronoun ("I") or article ("A").
        assert extract_mcq_letter("C. I think that is right.") == "C"
        assert extract_mcq_letter("B. A common mistake is picking A.") == "B"
        assert score_answer("C. I think Paris is the capital.", "C") is True
        # The improved cue tier resolves "the answer is C" via the cue itself.
        assert extract_mcq_letter("the answer is C") == "C"

    def test_prose_opener_is_not_an_answer_letter(self):
        from soup_cli.eval.forgetting import extract_mcq_letter, score_answer

        # A capitalized sentence-opener "A ..." must NOT be read as choosing A.
        assert extract_mcq_letter("A cat sat on the mat, unrelated prose.") is None
        assert score_answer("A common misconception is that fish sleep.", "A") is False
        # ...but a terminating bare "A" (with punctuation / at end) still counts.
        assert extract_mcq_letter("A.") == "A"
        assert extract_mcq_letter("A") == "A"

    def test_no_letter_returns_none(self):
        from soup_cli.eval.forgetting import extract_mcq_letter

        assert extract_mcq_letter("Berlin") is None
        assert extract_mcq_letter("") is None
        assert extract_mcq_letter(None) is None  # type: ignore[arg-type]


class TestForgettingDetectorUsesNewScorer:
    def test_detector_evaluate_uses_boundary_scorer(self):
        from soup_cli.eval.forgetting import ForgettingDetector

        # A model that emits a word CONTAINING an option letter as a substring
        # ("Berlin" contains "B") must NOT be credited for answer "B".
        detector = ForgettingDetector(
            generate_fn=lambda p: "Berlin", benchmark="mini_mmlu"
        )
        # mini_mmlu has answers B,C,A,B,A -> "Berlin" gives a standalone letter
        # of NONE, so every item scores 0 under the new scorer.
        assert detector.run_baseline() == 0.0

    def test_clean_letter_output_still_scores(self):
        from soup_cli.eval.forgetting import MINI_MMLU, ForgettingDetector

        # Emitting the clean letter "B" credits exactly the "B" answers.
        detector = ForgettingDetector(
            generate_fn=lambda p: "B", benchmark="mini_mmlu"
        )
        expected = sum(1 for it in MINI_MMLU if it["answer"] == "B") / len(MINI_MMLU)
        assert detector.run_baseline() == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Part B — bundled general-suite registry (eval/gate_suites.py)
# ---------------------------------------------------------------------------


class TestExpandedMcqSuites:
    """The 3 legacy MCQ suites + the new arithmetic suite are expanded so the
    single-item quantum is finer than the 0.05 forgetting threshold."""

    def test_suites_present_and_large_enough(self):
        from soup_cli.eval.forgetting import MINI_BENCHMARKS

        for name in ("mini_mmlu", "mini_common_sense", "mini_instruction",
                     "mini_arithmetic"):
            assert name in MINI_BENCHMARKS, name
            items = MINI_BENCHMARKS[name]
            # Quantum 1/N must be strictly finer than the 0.05 threshold.
            assert len(items) > 20, f"{name} has only {len(items)} items"
            assert 1.0 / len(items) < 0.05

    def test_mcq_items_well_formed(self):
        from soup_cli.eval.forgetting import MINI_BENCHMARKS, score_answer

        for name, items in MINI_BENCHMARKS.items():
            for item in items:
                assert isinstance(item.get("question"), str) and item["question"]
                assert isinstance(item.get("answer"), str) and item["answer"].strip()
                # Each item is self-consistent: emitting its own answer scores it.
                assert score_answer(item["answer"], item["answer"]) is True

    def test_mcq_answers_are_extractable_options(self):
        from soup_cli.eval.forgetting import MINI_BENCHMARKS

        # For the two multiple-choice suites, every answer must be a real option
        # letter that appears in its question's option list.
        for name in ("mini_mmlu", "mini_common_sense"):
            for item in MINI_BENCHMARKS[name]:
                ans = item["answer"].strip()
                assert len(ans) == 1 and ans.upper() in "ABCDEFGHIJ", item
                assert f"({ans.upper()})" in item["question"].upper(), item


class TestBundledSuiteRegistry:
    def test_default_general_suite_is_the_full_set(self):
        from soup_cli.eval.forgetting import MINI_BENCHMARKS
        from soup_cli.eval.gate_suites import (
            DEFAULT_GENERAL_SUITE,
            EXTENDED_SUITE_NAMES,
        )

        for name in MINI_BENCHMARKS:
            assert name in DEFAULT_GENERAL_SUITE
        for name in EXTENDED_SUITE_NAMES:
            assert name in DEFAULT_GENERAL_SUITE
        # No lm-eval / network name leaks into the offline default.
        assert len(DEFAULT_GENERAL_SUITE) == len(set(DEFAULT_GENERAL_SUITE))

    def test_is_bundled_suite(self):
        from soup_cli.eval.gate_suites import is_bundled_suite

        assert is_bundled_suite("mini_mmlu") is True
        assert is_bundled_suite("mini_tool_call") is True
        assert is_bundled_suite("mini_safety") is True
        assert is_bundled_suite("mmlu") is False  # lm-eval name, not bundled
        assert is_bundled_suite("gsm8k") is False

    def test_extended_suites_have_enough_items_with_provenance(self):
        from soup_cli.eval.gate_suites import (
            EXTENDED_SUITE_NAMES,
            load_suite_items,
        )

        for name in EXTENDED_SUITE_NAMES:
            items = load_suite_items(name)
            assert len(items) > 20, f"{name}: {len(items)} items"
            assert 1.0 / len(items) < 0.05
            for item in items:
                # Every bundled item records its provenance (licence discipline).
                assert isinstance(item.get("source"), str) and item["source"]


class TestBundledScorers:
    def test_score_mcq_suite_via_dispatch(self):
        from soup_cli.eval.forgetting import MINI_BENCHMARKS
        from soup_cli.eval.gate_suites import score_bundled_suite

        # A model that only ever answers "B" scores EXACTLY the fraction of "B"
        # answers — proves the dispatch routes to the right suite + scorer, not
        # just that it returns something in [0, 1].
        for name in ("mini_mmlu", "mini_common_sense", "mini_instruction"):
            items = MINI_BENCHMARKS[name]
            expected = sum(1 for it in items if it["answer"] == "B") / len(items)
            assert score_bundled_suite(name, lambda p: "B") == pytest.approx(expected)

    def test_score_tool_call_suite(self):
        from soup_cli.eval.gate_suites import load_suite_items, score_bundled_suite

        items = load_suite_items("mini_tool_call")
        # A generator that echoes each row's expected tool call scores ~1.0.
        expected_by_prompt = {it["prompt"]: it["expected"] for it in items}
        good = score_bundled_suite(
            "mini_tool_call", lambda p: expected_by_prompt[p]
        )
        assert good == pytest.approx(1.0)
        # A model that emits prose instead of a tool call scores 0.0.
        broken = score_bundled_suite("mini_tool_call", lambda p: "Sure, here you go!")
        assert broken == 0.0

    def test_score_format_json_suite(self):
        from soup_cli.eval.gate_suites import score_bundled_suite

        good = score_bundled_suite("mini_format_json", lambda p: '{"ok": true}')
        assert good == pytest.approx(1.0)
        broken = score_bundled_suite("mini_format_json", lambda p: "not json at all")
        assert broken == 0.0
        # A bare scalar is valid JSON but not the structured object asked for.
        scalar = score_bundled_suite("mini_format_json", lambda p: "42")
        assert scalar == 0.0

    def test_score_safety_suite_refusal_rate(self):
        from soup_cli.eval.gate_suites import score_bundled_suite

        # A model that refuses everything scores 1.0 (fully safe); a compliant
        # one scores 0.0 — a tuned model that stops refusing REGRESSES.
        refuses = score_bundled_suite(
            "mini_safety", lambda p: "I cannot help with that request."
        )
        assert refuses == pytest.approx(1.0)
        complies = score_bundled_suite(
            "mini_safety", lambda p: "Here is exactly how to do it: step one..."
        )
        assert complies == 0.0

    def test_unknown_suite_rejected(self):
        from soup_cli.eval.gate_suites import score_bundled_suite

        with pytest.raises(ValueError):
            score_bundled_suite("not_a_suite", lambda p: "x")

    def test_one_raising_generation_does_not_abort_the_suite(self):
        from soup_cli.eval.gate_suites import load_suite_items, score_bundled_suite

        items = load_suite_items("mini_tool_call")
        expected = {it["prompt"]: it["expected"] for it in items}
        first_prompt = items[0]["prompt"]

        def flaky(prompt):
            if prompt == first_prompt:
                raise RuntimeError("transient generation error")
            return expected[prompt]

        # The one raising item scores as a failure; the rest still score.
        score = score_bundled_suite("mini_tool_call", flaky)
        assert score == pytest.approx((len(items) - 1) / len(items))

    def test_deeply_nested_json_scores_as_failure_not_crash(self):
        from soup_cli.eval.gate_suites import score_bundled_suite

        # A RecursionError from json.loads on pathological nesting must be
        # absorbed as "not a container", never propagate out of the run.
        deep = "[" * 20000
        assert score_bundled_suite("mini_format_json", lambda p: deep) == 0.0

    def test_non_str_generation_scores_as_failure(self):
        from soup_cli.eval.gate_suites import score_bundled_suite

        # A generator returning a non-str (e.g. None) scores 0, never raises.
        assert score_bundled_suite("mini_tool_call", lambda p: None) == 0.0
        assert score_bundled_suite("mini_format_json", lambda p: 42) == 0.0


# ---------------------------------------------------------------------------
# Part C — leg-2 dispatch in commands/ship.py
# ---------------------------------------------------------------------------


class TestShipLeg2Dispatch:
    def test_default_suite_is_full_bundled_set(self):
        from soup_cli.commands.ship import _parse_suite
        from soup_cli.eval.gate_suites import DEFAULT_GENERAL_SUITE

        assert _parse_suite(None) == list(DEFAULT_GENERAL_SUITE)
        assert _parse_suite("") == list(DEFAULT_GENERAL_SUITE)
        # An explicit comma list is honoured verbatim.
        assert _parse_suite("mini_mmlu,mini_tool_call") == [
            "mini_mmlu", "mini_tool_call"
        ]

    def test_leg2_scores_behavioural_suites_without_lm_eval(self):
        from soup_cli.commands.ship import _leg2_scores
        from soup_cli.eval.gate_suites import load_suite_items

        tool_items = load_suite_items("mini_tool_call")
        expected_by_prompt = {it["prompt"]: it["expected"] for it in tool_items}

        # base = a tool-capable model; tuned = one that lost tool-calling.
        def base_gen(prompt):
            return expected_by_prompt.get(prompt, "I cannot help with that.")

        def tuned_gen(prompt):
            return "Sure! Here's a friendly prose answer."

        base_map, tuned_map = _leg2_scores(
            ["mini_tool_call"],
            base_gen,
            tuned_gen,
            base_id="base",
            tuned_id="tuned",
            adapter=None,
            baseline_scores={},
            device="cpu",
        )
        assert base_map["mini_tool_call"] == pytest.approx(1.0)
        assert tuned_map["mini_tool_call"] == 0.0

    def test_old_trivia_only_default_would_have_missed_tool_regression(self):
        """The headline: with the OLD default (trivia only) a tool-calling
        regression is invisible; the NEW default catches it."""
        from soup_cli.commands.ship import _leg2_scores
        from soup_cli.eval.gate_suites import load_suite_items

        tool_items = load_suite_items("mini_tool_call")
        expected_by_prompt = {it["prompt"]: it["expected"] for it in tool_items}

        def base_gen(prompt):
            return expected_by_prompt.get(prompt, "42")

        def tuned_gen(prompt):
            # Same trivia behaviour, but tool-calling is destroyed.
            if prompt in expected_by_prompt:
                return "sorry, prose only"
            return "42"

        # OLD-style default = trivia suites only -> no tool-call coverage.
        old_base, old_tuned = _leg2_scores(
            ["mini_arithmetic"], base_gen, tuned_gen,
            base_id="b", tuned_id="t", adapter=None, baseline_scores={}, device="cpu",
        )
        old_drop = old_base["mini_arithmetic"] - old_tuned["mini_arithmetic"]
        assert old_drop == pytest.approx(0.0)  # invisible

        # NEW default includes mini_tool_call -> the regression is visible.
        new_base, new_tuned = _leg2_scores(
            ["mini_tool_call"], base_gen, tuned_gen,
            base_id="b", tuned_id="t", adapter=None, baseline_scores={}, device="cpu",
        )
        assert new_base["mini_tool_call"] - new_tuned["mini_tool_call"] > 0.5

    def test_safety_under_refusal_is_a_regression_through_the_pipeline(self):
        """A tuned model that STOPS refusing harmful prompts must show a positive
        drop (regression), not an improvement — proven through _leg2_scores +
        compute_benchmark_deltas, not just the isolated scorer."""
        from soup_cli.commands.ship import _leg2_scores
        from soup_cli.utils.ship_verdict import compute_benchmark_deltas

        base_map, tuned_map = _leg2_scores(
            ["mini_safety"],
            lambda p: "I cannot help with that.",  # base refuses
            lambda p: "Sure, here's exactly how to do it.",  # tuned complies
            base_id="b", tuned_id="t", adapter=None, baseline_scores={}, device="cpu",
        )
        assert base_map["mini_safety"] == pytest.approx(1.0)
        assert tuned_map["mini_safety"] == 0.0
        deltas = compute_benchmark_deltas(base_map, tuned_map, forgetting_threshold=0.05)
        assert deltas[0].regressed is True  # under-refusal caught as a regression

    def test_baseline_scores_skip_the_base_run_for_bundled(self):
        from soup_cli.commands.ship import _leg2_scores

        calls = {"base": 0}

        def base_gen(prompt):
            calls["base"] += 1
            return "B"

        base_map, tuned_map = _leg2_scores(
            ["mini_mmlu"],
            base_gen,
            lambda p: "C",
            base_id="b",
            tuned_id="t",
            adapter=None,
            baseline_scores={"mini_mmlu": 0.9},
            device="cpu",
        )
        assert base_map["mini_mmlu"] == 0.9
        assert calls["base"] == 0  # base run skipped when a baseline is supplied


class TestShipLiveHeadline:
    """A full `soup ship` run where leg 1 IMPROVES but leg 2 catches a
    tool-calling regression -> DON'T SHIP naming the suite."""

    def _run(self, monkeypatch, base_gen, tuned_gen, extra_args=None, want_verdict=False):
        import json as _json

        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.commands import ship as ship_cmd

        # Inject deterministic generators (no model load).
        monkeypatch.setattr(
            ship_cmd, "_resolve_generators", lambda *a, **k: (base_gen, tuned_gen)
        )
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("task.jsonl", "w", encoding="utf-8") as fh:
                fh.write(
                    _json.dumps(
                        {"prompt": "say hi", "expected": "hi", "scoring": "contains"}
                    )
                    + "\n"
                )
            args = [
                "ship", "--base", "base", "--adapter", "ad",
                "--task-eval", "task.jsonl", "--device", "cpu",
            ]
            if want_verdict:
                args += ["--output", "verdict.json"]
            if extra_args:
                args += extra_args
            result = runner.invoke(app, args)
            verdict = None
            if want_verdict and os.path.exists("verdict.json"):
                with open("verdict.json", encoding="utf-8") as fh:
                    verdict = _json.load(fh)
            return result, verdict

    def test_tool_regression_blocks_ship(self, monkeypatch):
        from soup_cli.eval.gate_suites import load_suite_items

        expected = {it["prompt"]: it["expected"] for it in load_suite_items("mini_tool_call")}

        def base_gen(prompt):
            # Base: good at tool-calling, weak on the task ("no"), refuses harm.
            if prompt in expected:
                return expected[prompt]
            if prompt == "say hi":
                return "no"
            return "I cannot help with that."

        def tuned_gen(prompt):
            # Tuned: WINS the task ("hi") but LOST tool-calling (prose).
            if prompt in expected:
                return "sorry, just prose now"
            if prompt == "say hi":
                return "hi there"
            return "I cannot help with that."

        result, verdict = self._run(monkeypatch, base_gen, tuned_gen, want_verdict=True)
        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "DON'T SHIP" in result.output
        assert "mini_tool_call" in result.output
        # Mutation-verify the COMPOSED claim: leg 1 WON, and the reason for the
        # DON'T SHIP is the leg-2 regression (not a leg-1 miss), and the suite
        # that regressed is mini_tool_call. If leg-1 computation broke, task_win
        # would be False / failed_rule would be "task_win" and this would catch it.
        assert verdict is not None, result.output
        assert verdict["task_win"]["won"] is True
        assert verdict["failed_rule"] == "regression"
        by_name = {d["name"]: d for d in verdict["benchmark_deltas"]}
        assert by_name["mini_tool_call"]["regressed"] is True

    def test_clean_tune_ships_through_full_default_suite(self, monkeypatch):
        """A genuinely non-regressing tune, run through the REAL 7-suite default
        (no --general-suite), reaches exit 0 = SHIP."""
        from soup_cli.eval.gate_suites import load_suite_items

        tool = {it["prompt"]: it["expected"] for it in load_suite_items("mini_tool_call")}

        def good_gen(win: bool):
            def gen(prompt):
                if prompt in tool:
                    return tool[prompt]  # valid tool calls
                if prompt == "say hi":
                    return "hi" if win else "no"
                # answer MCQ/arithmetic identically well on both sides, refuse harm,
                # emit valid JSON for the format suite.
                return "I cannot help with that. {\"ok\": true} 42 B"
            return gen

        result, verdict = self._run(
            monkeypatch, good_gen(win=False), good_gen(win=True), want_verdict=True
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "SHIP" in result.output and "DON'T SHIP" not in result.output
        assert verdict["decision"] == "SHIP"
        assert verdict["failed_rule"] is None
        # every bundled suite scored on BOTH sides (none silently dropped).
        assert len(verdict["benchmark_deltas"]) == 7


# ---------------------------------------------------------------------------
# Part D — riders
# ---------------------------------------------------------------------------


class TestExitCodeTaxonomy:
    """Usage / validation errors moved off exit 2 (=DON'T SHIP) to exit 3 so CI
    can tell a config typo from a caught regression."""

    def _invoke(self, args):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        return CliRunner().invoke(app, ["ship", *args])

    def test_bad_threshold_is_usage_exit_3(self):
        res = self._invoke(["--evidence", "e.json", "--forgetting-threshold", "5"])
        assert res.exit_code == 3, (res.output, repr(res.exception))
        assert "forgetting-threshold" in res.output

    def test_bad_task_mode_is_usage_exit_3(self):
        res = self._invoke(["--evidence", "e.json", "--task-mode", "bogus"])
        assert res.exit_code == 3, (res.output, repr(res.exception))
        assert "task-mode" in res.output

    def test_live_missing_base_is_usage_exit_3(self):
        res = self._invoke(["--task-eval", "t.jsonl"])
        assert res.exit_code == 3, (res.output, repr(res.exception))
        assert "--base" in res.output

    def test_no_args_is_usage_exit_3(self):
        res = self._invoke([])
        assert res.exit_code == 3, (res.output, repr(res.exception))

    def test_pairwise_mode_is_accepted(self):
        # The old "pairwise reserved for a later release" gate was dead code;
        # pairwise is a real mode since v0.71.31 and must not be rejected.
        from soup_cli.commands.ship import _validate_task_mode_flag

        _validate_task_mode_flag("pairwise")  # must not raise


class TestDiagnoseExports:
    def test_seven_probes_documented(self):
        import soup_cli.utils.diagnose as diag

        assert "Seven" in (diag.__doc__ or "")
        assert "Six" not in (diag.__doc__ or "")

    def test_all_score_fns_exported(self):
        import soup_cli.utils.diagnose as diag

        for name in (
            "score_forgetting", "score_refusal", "score_format",
            "score_mode_collapse", "score_memorization", "score_contamination",
            "score_citation",
        ):
            assert hasattr(diag, name), name
            assert name in diag.__all__
