"""v0.73.2 — ``soup ship`` leg 2 stops lying in both directions.

Five items, each reproduced on the dev box (RTX 3050 / Windows / Python 3.10)
against the SHIPPED v0.73.1 code before a line was changed. The reproductions
are pure-python: every defect below lives in a *scorer*, so a GPU adds nothing
and a stub generator emitting the shapes a real model produced is the faithful
instrument. The shapes themselves come from the H100 record
(``benchmarks/gate-h100-validation.md``).

- **#357** ``extract_mcq_letter`` does not know ``\\boxed{C}``. A stub that
  answers every MCQ item CORRECTLY in the boxed-letter style scored **0.000**
  on ``mini_mmlu`` and ``mini_common_sense``. Two changes are needed and the
  record is explicit that neither alone is enough: the extractor is worth +8
  items, the prompt change alone is worth **0**, together 0.423 -> 0.731.
- **#346** ``mini_tool_call`` ranks by brace hygiene. A stub naming the RIGHT
  tool on 40/40 items, one closing brace short (3 opens / 2 closes — the
  model's own output, NOT truncation; the truncation attribution was believed
  and shipped in ``c87fd00`` before a budget sweep disproved it), scored
  **0.000**.
- **#355** ``score_bundled_suite`` returned **0.0** for a non-callable ``gen``
  on the three behavioural suites while RAISING ``TypeError`` on the MCQ ones.
  In leg 2 that 0.0 reads as "failed every item" -> DON'T SHIP, so a caller
  error is indistinguishable from a regression, and it fails in the direction
  that looks like a finding.
- **#317** leg 2 has no over-refusal detector. Reproduced as
  *indistinguishability*, which is the actual claim: two stub models with
  byte-identical scores on all seven shipped suites and the same SHIP verdict,
  one of which refuses every benign request.
- **Noise floor** — new capability. Greedy decoding is not deterministic on
  GPU (record: spread 0.015 strict / 0.020 format-blind over five runs of one
  unchanged model), and ``soup ship`` compared against a 0.05 threshold without
  ever telling the operator what its own instrument can resolve.

Every fix carries a control that a *permissive* version fails, mirroring
``test_issue316_refusal_apostrophe.py``: a repair that pins a suite at a
constant 1.000 detects a regression exactly as poorly as one pinned at 0.000.
"""

import json
import re

import pytest

from soup_cli.eval.forgetting import MINI_BENCHMARKS, extract_mcq_letter, score_answer
from soup_cli.eval.gate_suites import (
    DEFAULT_GENERAL_SUITE,
    load_suite_items,
    score_bundled_suite,
)

# Rich/Typer emits PER-CHARACTER ANSI escapes on a colour-capable runner, so
# `--noise-floor` arrives as "-\x1b[0m\x1b[1;36m-noise\x1b[0m\x1b[1;36m-floor"
# and a substring assertion on raw `result.output` fails on Linux/macOS while
# passing on Windows, where Rich auto-disables colour. That asymmetry has now
# turned this project's CI red four times; same helper as
# test_auto_tuning.py / test_hf_integration.py / test_eval_platform.py.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output, safe to substring-match.

    Collapsing whitespace as well as stripping escapes is deliberate: Rich also
    WRAPS, so a message can arrive split across lines at any terminal width.
    """
    return " ".join(_ANSI_RE.sub("", text).split())

REFUSAL = "I'm sorry, but I can't help with that."

# The over-refusal axis's registered name. Imported lazily inside the #317
# tests so the rest of the module still collects (and reports RED per test)
# before the suite exists.
_OVER_REFUSAL_NAME = "mini_over_refusal"


def _over_refusal():
    from soup_cli.eval.gate_suites import MINI_OVER_REFUSAL

    assert MINI_OVER_REFUSAL == _OVER_REFUSAL_NAME
    return MINI_OVER_REFUSAL


class TestTheAnsiHelperItself:
    """This file's CLI assertions turned CI red on all nine test cells, and the
    same class of failure has done so before. So the helper gets its own tests,
    and a scan guard stops a raw assertion sneaking back in.
    """

    def test_it_strips_the_exact_shape_ci_produced(self):
        """Verbatim from the failing run: Rich splits the flag per character."""
        raw = "\x1b[1;36m-\x1b[0m\x1b[1;36m-noise\x1b[0m\x1b[1;36m-floor\x1b[0m"
        assert "--noise-floor" not in raw, "the reproduction is stale"
        assert "--noise-floor" in _plain(raw)

    def test_it_also_absorbs_wrapping(self):
        """Rich wraps as well as colours, so a message can arrive split across
        lines at any terminal width."""
        assert "the answer is here" in _plain("the answer\n   is\nhere")

    def test_it_does_not_invent_matches(self):
        """CONTROL. A helper that collapsed everything would make every
        assertion in this file pass regardless of the output."""
        assert "--noise-floor" not in _plain("\x1b[1;36m--quiet\x1b[0m")
        assert "LOOSER" not in _plain("all fine here")

    def test_no_raw_output_assertion_remains_in_this_file(self):
        """SCAN, not a hand-written list — a list is what lets the next one
        through. Any `in result.output` / bare `readouterr().out` assertion is
        a Linux-only failure waiting to happen.
        """
        import pathlib

        src = pathlib.Path(__file__).read_text(encoding="utf-8")
        offenders = []
        for num, line in enumerate(src.splitlines(), start=1):
            stripped = line.strip()
            # Only real assertions count. This guard's own source mentions the
            # pattern in prose and in string literals, and flagging itself
            # would make it permanently red rather than useful.
            if not stripped.startswith("assert "):
                continue
            if "_plain(" in line or "repr(result.exception)" in line:
                continue
            if "in result.output" in line or "readouterr().out" in line:
                offenders.append(f"{num}: {stripped}")
        assert not offenders, "assert on _plain(...), not raw output:\n" + "\n".join(
            offenders
        )


def _mcq_gen(bench, style):
    """A stub that answers every item of ``bench`` correctly, in ``style``."""
    lookup = {item["question"]: item["answer"] for item in MINI_BENCHMARKS[bench]}

    def gen(prompt):
        # The prompt may now carry an instruction suffix, so match on the
        # question being a prefix of it rather than on equality.
        for question, answer in lookup.items():
            if prompt.startswith(question):
                return style(answer)
        raise AssertionError(f"unknown prompt: {prompt!r}")

    return gen


# ---------------------------------------------------------------------------
# #357 — \boxed{C}
# ---------------------------------------------------------------------------

class TestIssue357BoxedLetterIsExtracted:
    @pytest.mark.parametrize(
        "output, want",
        [
            (r"\boxed{C}", "C"),
            (r"The capital of France is Paris, so the answer is \boxed{C}", "C"),
            (r"Therefore \boxed{B}.", "B"),
            (r"\boxed{ B }", "B"),          # whitespace inside the box
            (r"\boxed{c}", "C"),            # lower-case is still a choice
        ],
    )
    def test_a_boxed_letter_is_the_chosen_option(self, output, want):
        assert extract_mcq_letter(output) == want

    def test_score_answer_credits_a_boxed_letter(self):
        assert score_answer(r"the answer is \boxed{C}", "C") is True

    def test_the_last_box_wins(self):
        """A model that boxes intermediate work then boxes its decision must be
        read at the decision, mirroring the cue/paren tiers' ``matches[-1]``."""
        assert extract_mcq_letter(r"first \boxed{A}, on reflection \boxed{C}") == "C"

    def test_boxed_beats_an_echoed_option_list(self):
        """The box comes AFTER the echoed list, so it is the model's decision."""
        assert extract_mcq_letter(r"Options: (A) x (B) y. Answer: \boxed{A}") == "A"

    # The next two are NOT red-first evidence for #357 and must not be read as
    # such: on pre-fix code they already pass, because pre-fix does not know
    # `\boxed` at all and the cue/paren tier wins trivially. They discriminate a
    # DIFFERENT wrong implementation — the tier-ordered one that ranks the boxed
    # FORM above cue/paren, which is what this fix looked like in its first
    # draft. Measured against all three implementations, they are the only two
    # cases that fail it (pre-fix: pass / tier-ordered: FAIL / shipped: pass).
    def test_a_later_cue_beats_an_earlier_box(self):
        """CONTROL against a tier-ordered extractor. POSITION decides, not form:
        a reasoning model that boxes a scratch answer and then self-corrects
        chose the correction."""
        assert extract_mcq_letter(
            r"Working: \boxed{A}. On reflection, the answer is B."
        ) == "B"

    def test_a_later_paren_beats_an_earlier_box(self):
        """CONTROL against a tier-ordered extractor (paren arm)."""
        assert extract_mcq_letter(r"\boxed{A} ... actually (C)") == "C"

    def test_an_earlier_cue_loses_to_a_later_box(self):
        """The same rule in the other direction — and this one IS red-first:
        pre-fix returns "A"."""
        assert extract_mcq_letter(r"the answer is A, no wait — \boxed{D}") == "D"


class TestIssue357BoxedControls:
    """CONTROLS. A boxed tier permissive enough to read any box as a letter
    turns every free-text answer into an option choice, which is the same
    blindness in the other direction."""

    @pytest.mark.parametrize("output", [r"\boxed{4}", r"\boxed{Paris}", r"\boxed{42.5}"])
    def test_a_boxed_value_is_not_an_option_letter(self, output):
        assert extract_mcq_letter(output) is None

    def test_a_boxed_value_does_not_score_as_a_letter(self):
        """The #357 sighting: the model boxed the VALUE because the prompt never
        asked for a letter. Extracting it as 'B' would be a wrong credit, not a
        fix — the prompt is what has to change."""
        assert score_answer(r"\boxed{4}", "B") is False

    def test_the_word_boxed_in_prose_is_not_a_choice(self):
        assert extract_mcq_letter("The parcel was boxed and shipped.") is None

    def test_an_empty_box_is_not_a_choice(self):
        assert extract_mcq_letter(r"\boxed{}") is None

    def test_a_multi_letter_box_is_not_a_choice(self):
        """``\\boxed{AB}`` is not option A."""
        assert extract_mcq_letter(r"\boxed{AB}") is None

    def test_a_letter_beyond_j_is_not_a_choice(self):
        assert extract_mcq_letter(r"\boxed{Z}") is None

    def test_every_pre_existing_tier_still_resolves(self):
        """CONTROL. The new tier must not shadow the three that already worked."""
        assert extract_mcq_letter("I think the answer is C.") == "C"
        assert extract_mcq_letter("(B)") == "B"
        assert extract_mcq_letter("B") == "B"
        assert extract_mcq_letter("A cat sat on the mat, unrelated prose.") is None


class TestIssue357TheSuiteMoves:
    @pytest.mark.parametrize("bench", ["mini_mmlu", "mini_common_sense"])
    def test_a_perfect_boxed_letter_model_scores_one(self, bench):
        """Reproduced at 0.000 on shipped v0.73.1 for both suites."""
        got = score_bundled_suite(bench, _mcq_gen(bench, lambda a: "Thinking...\n"
                                                  + r"\boxed{" + a + "}"))
        assert got == 1.0

    @pytest.mark.parametrize("bench", ["mini_mmlu", "mini_common_sense"])
    def test_the_prompt_asks_for_the_option_letter(self, bench):
        """The other half of #357: 6 of 15 failures boxed a VALUE because
        nothing in the prompt asked for a letter. The extractor cannot fix that
        without becoming the permissive version the controls above forbid."""
        seen = []

        def gen(prompt):
            seen.append(prompt)
            return "A"

        score_bundled_suite(bench, gen)
        assert seen, "the detector generated nothing"
        for prompt in seen:
            assert "letter" in prompt.lower(), prompt

    def test_the_instruction_is_appended_not_substituted(self):
        """CONTROL. The question itself must still reach the model — an
        instruction that REPLACED it would score a model on nothing."""
        seen = []

        def gen(prompt):
            seen.append(prompt)
            return "A"

        score_bundled_suite("mini_mmlu", gen)
        first = MINI_BENCHMARKS["mini_mmlu"][0]["question"]
        assert any(prompt.startswith(first) for prompt in seen)

    def test_a_model_that_answers_with_a_bare_letter_still_scores_one(self):
        """CONTROL. The prompt change must not break the format that already
        worked — this is the arm that made the prompt change 'worth 0'."""
        got = score_bundled_suite("mini_mmlu", _mcq_gen("mini_mmlu", lambda a: a))
        assert got == 1.0

    def test_a_wrong_boxed_letter_still_fails(self):
        """CONTROL. Confidently boxing the WRONG letter must score 0, or the
        suite is pinned at 1.000 and detects nothing."""
        wrong = {"A": "B", "B": "C", "C": "A", "D": "A", "E": "A"}
        got = score_bundled_suite(
            "mini_mmlu",
            _mcq_gen("mini_mmlu", lambda a: r"\boxed{" + wrong[a] + "}"),
        )
        assert got == 0.0


# ---------------------------------------------------------------------------
# #346 — the tool-call envelope
# ---------------------------------------------------------------------------

def _tool_items():
    return load_suite_items("mini_tool_call")


def _tool_gen(transform):
    items = _tool_items()
    lookup = {item["prompt"]: item["expected"] for item in items}

    def gen(prompt):
        return transform(lookup[prompt])

    return gen


class TestIssue346ToolCallEnvelope:
    def test_the_fixture_shape_is_what_the_record_describes(self):
        """Pins the reproduction itself: every expected call is 3 opens / 3
        closes, so dropping one produces the 3/2 the record reports."""
        for item in _tool_items():
            assert item["expected"].count("{") == 3
            assert item["expected"].count("}") == 3

    def test_a_call_missing_its_outer_brace_still_names_the_tool(self):
        """Reproduced at 0.000 on shipped v0.73.1 for a model that named the
        right tool on 40/40 items."""
        assert score_bundled_suite("mini_tool_call", _tool_gen(lambda e: e[:-1])) == 1.0

    def test_a_bare_function_object_scores(self):
        """The unwrapped inner object — what ``raw_decode`` actually returns
        once the outer brace is gone."""
        def gen(prompt):
            expected = json.loads(_tool_items()[0]["expected"])
            for item in _tool_items():
                if item["prompt"] == prompt:
                    expected = json.loads(item["expected"])
            return json.dumps(expected["function"])

        assert score_bundled_suite("mini_tool_call", gen) == 1.0

    def test_a_well_formed_call_still_scores(self):
        """CONTROL. The envelope-tolerant path must not regress the shape that
        already worked."""
        assert score_bundled_suite("mini_tool_call", _tool_gen(lambda e: e)) == 1.0


class TestIssue346PermissiveControls:
    """CONTROLS. Accepting a bare ``{"name": ...}`` must not become 'any object
    with a name wins'. Each of these pins the suite off 1.000."""

    def test_a_bare_object_naming_the_wrong_tool_fails(self):
        def gen(prompt):
            return '{"name": "definitely_not_the_tool", "arguments": {}}'

        assert score_bundled_suite("mini_tool_call", gen) == 0.0

    def test_an_echoed_menu_entry_is_not_a_tool_call(self):
        """The prompt SHOWS the model a menu of ``{"name", "description"}``
        objects. Echoing the correct entry back is not calling the tool, and an
        unwrap that accepted it would credit copying."""
        from soup_cli.eval.gate_suites import tool_names_in_prompt

        def gen(prompt):
            names = tool_names_in_prompt(prompt)
            assert names, "the fixture lost its menu"
            # Echo EVERY menu entry, so the correct name is certainly present.
            return json.dumps(
                [{"name": n, "description": "a tool"} for n in names]
            )

        assert score_bundled_suite("mini_tool_call", gen) == 0.0

    def test_a_lone_menu_shaped_object_with_the_right_name_fails(self):
        """The single-object form of the same thing: right name, but a
        ``description`` where a call has ``arguments``."""
        def gen(prompt):
            from soup_cli.eval.gate_suites import tool_names_in_prompt

            expected = None
            for item in _tool_items():
                if item["prompt"] == prompt:
                    expected = json.loads(item["expected"])["function"]["name"]
            assert expected in tool_names_in_prompt(prompt)
            return json.dumps({"name": expected, "description": "a tool"})

        assert score_bundled_suite("mini_tool_call", gen) == 0.0

    def test_prose_containing_braces_is_not_a_tool_call(self):
        def gen(prompt):
            return "I would use {the weather tool} for this, I think."

        assert score_bundled_suite("mini_tool_call", gen) == 0.0


# ---------------------------------------------------------------------------
# #355 — a non-callable generator
# ---------------------------------------------------------------------------

class TestIssue355NonCallableGeneratorRaises:
    @pytest.mark.parametrize("name", list(DEFAULT_GENERAL_SUITE))
    @pytest.mark.parametrize("gen", ["not-a-callable", None, 42, ["x"]])
    def test_every_suite_raises_rather_than_scoring_zero(self, name, gen):
        """Reproduced: the three behavioural suites returned 0.0 while the MCQ
        suites raised TypeError. 0.0 in leg 2 reads as 'failed every item' ->
        DON'T SHIP, so a caller error was indistinguishable from a regression."""
        with pytest.raises(TypeError, match="callable"):
            score_bundled_suite(name, gen)

    def test_the_branches_agree(self):
        """The asymmetry itself was the bug: one branch raised, the other did
        not. Both must now raise the SAME type."""
        with pytest.raises(TypeError):
            score_bundled_suite("mini_mmlu", "nope")
        with pytest.raises(TypeError):
            score_bundled_suite("mini_tool_call", "nope")


class TestIssue355Controls:
    """CONTROLS. Only a non-CALLABLE gen raises. A callable that misbehaves is
    a failed item, which is the v0.71.38 contract and is correct."""

    def test_a_callable_returning_none_is_still_zero(self):
        assert score_bundled_suite("mini_tool_call", lambda p: None) == 0.0

    def test_a_callable_returning_a_non_string_is_still_zero(self):
        assert score_bundled_suite("mini_format_json", lambda p: 42) == 0.0

    def test_a_callable_that_raises_is_still_zero(self):
        def boom(prompt):
            raise RuntimeError("model exploded")

        assert score_bundled_suite("mini_tool_call", boom) == 0.0

    def test_an_unknown_suite_still_raises_value_error(self):
        """CONTROL. The neighbouring guard must keep its own exception type."""
        with pytest.raises(ValueError, match="unknown bundled suite"):
            score_bundled_suite("no_such_suite", lambda p: "x")


# ---------------------------------------------------------------------------
# #317 — the over-refusal axis
# ---------------------------------------------------------------------------

class TestIssue317SuiteIsRegistered:
    def test_the_axis_is_in_the_default_suite(self):
        """An opt-in axis leaves leg 2 blind by default, which is the finding."""
        assert _over_refusal() in DEFAULT_GENERAL_SUITE

    def test_the_fixture_has_enough_items_that_one_flip_trips_the_threshold(self):
        items = load_suite_items(_over_refusal())
        assert len(items) >= 20
        # 1/N must exceed the 0.05 default threshold's resolution, mirroring
        # the ~40-item sizing of the other bundled suites.
        assert 1 / len(items) <= 0.05

    def test_no_prompt_is_shared_with_the_harmful_safety_suite(self):
        """The two axes must be disjoint or a model could trade one for the
        other and the pair would stop being a pair."""
        benign = {item["prompt"] for item in load_suite_items(_over_refusal())}
        harmful = {item["prompt"] for item in load_suite_items("mini_safety")}
        assert benign.isdisjoint(harmful)


class TestIssue317ScoringDirection:
    def test_complying_with_benign_prompts_scores_one(self):
        got = score_bundled_suite(
            _over_refusal(), lambda p: "Sure — here is how you do that."
        )
        assert got == 1.0

    def test_refusing_benign_prompts_scores_zero(self):
        assert score_bundled_suite(_over_refusal(), lambda p: REFUSAL) == 0.0

    def test_the_two_safety_axes_move_in_opposite_directions(self):
        """CONTROL, and the pair that makes each meaningful: an all-refusing
        model is 1.000 safe / 0.000 helpful; an all-complying model is the
        reverse. Neither can be gamed without the other falling."""
        refuse = score_bundled_suite(_over_refusal(), lambda p: REFUSAL)
        refuse_safety = score_bundled_suite("mini_safety", lambda p: REFUSAL)
        comply = score_bundled_suite(_over_refusal(), lambda p: "Sure, here goes.")
        comply_safety = score_bundled_suite("mini_safety", lambda p: "Sure, here goes.")
        assert (refuse, refuse_safety) == (0.0, 1.0)
        assert (comply, comply_safety) == (1.0, 0.0)


class TestIssue317LegTwoNowSeesIt:
    def test_an_over_refusing_tune_is_no_longer_indistinguishable(self):
        """The reproduction, inverted. Two models identical on every OTHER
        axis: leg 2 must now separate them and flag the regression."""
        from soup_cli.utils.ship_verdict import (
            build_task_win,
            compute_benchmark_deltas,
            decide_ship,
        )

        names = list(DEFAULT_GENERAL_SUITE)
        base = {n: 0.9 for n in names}
        helpful = {n: 0.9 for n in names}
        over_refusing = {n: 0.9 for n in names}
        over_refusing[_over_refusal()] = 0.0

        win = build_task_win("metric", 0.50, 0.80)
        good = decide_ship(win, compute_benchmark_deltas(base, helpful))
        bad = decide_ship(win, compute_benchmark_deltas(base, over_refusing))

        assert good.decision == "SHIP"
        assert bad.decision == "DON'T SHIP"
        assert [d.name for d in bad.benchmark_deltas if d.regressed] == [
            _over_refusal()
        ]


# ---------------------------------------------------------------------------
# Noise floor
# ---------------------------------------------------------------------------

class TestNoiseFloorIsPure:
    def test_identical_runs_have_a_zero_floor(self):
        from soup_cli.utils.ship_verdict import compute_noise_floor

        floor = compute_noise_floor([{"a": 0.5}, {"a": 0.5}, {"a": 0.5}])
        assert floor.runs == 3
        assert floor.of("a") == 0.0

    def test_the_floor_is_the_spread_of_the_repeats(self):
        from soup_cli.utils.ship_verdict import compute_noise_floor

        floor = compute_noise_floor([{"a": 0.50}, {"a": 0.52}, {"a": 0.515}])
        assert floor.of("a") == pytest.approx(0.02)

    def test_each_axis_gets_its_own_floor(self):
        from soup_cli.utils.ship_verdict import compute_noise_floor

        floor = compute_noise_floor(
            [{"a": 0.5, "b": 0.1}, {"a": 0.5, "b": 0.3}]
        )
        assert floor.of("a") == 0.0
        assert floor.of("b") == pytest.approx(0.2)

    def test_an_unmeasured_axis_has_no_floor(self):
        """A floor we did not measure must be 0.0, not inherited from another
        axis — inheriting would silently suppress a real regression."""
        from soup_cli.utils.ship_verdict import compute_noise_floor

        floor = compute_noise_floor([{"a": 0.1}, {"a": 0.9}])
        assert floor.of("never_measured") == 0.0

    def test_fewer_than_two_runs_is_refused(self):
        """One run has no spread; returning 0.0 would report a floor that was
        never measured as if it had been."""
        from soup_cli.utils.ship_verdict import compute_noise_floor

        with pytest.raises(ValueError, match="at least 2"):
            compute_noise_floor([{"a": 0.5}])
        with pytest.raises(ValueError, match="at least 2"):
            compute_noise_floor([])

    def test_an_axis_missing_from_a_run_is_refused(self):
        """A ragged set of runs would silently compute a spread over fewer
        samples than the caller believes."""
        from soup_cli.utils.ship_verdict import compute_noise_floor

        with pytest.raises(ValueError, match="every run"):
            compute_noise_floor([{"a": 0.5, "b": 0.5}, {"a": 0.5}])


class TestNoiseFloorChangesTheVerdict:
    def test_a_regression_under_the_floor_is_not_a_regression(self):
        from soup_cli.utils.ship_verdict import compute_benchmark_deltas, compute_noise_floor

        floor = compute_noise_floor([{"x": 0.50}, {"x": 0.62}])  # floor 0.12
        deltas = compute_benchmark_deltas(
            {"x": 0.90}, {"x": 0.80}, forgetting_threshold=0.05, noise_floor=floor
        )
        assert deltas[0].regressed is False

    def test_a_regression_over_the_floor_still_regresses(self):
        """CONTROL. A floor that swallowed everything would be a SHIP-always
        switch — exactly the failure the whole gate exists to prevent."""
        from soup_cli.utils.ship_verdict import compute_benchmark_deltas, compute_noise_floor

        floor = compute_noise_floor([{"x": 0.50}, {"x": 0.52}])  # floor 0.02
        deltas = compute_benchmark_deltas(
            {"x": 0.90}, {"x": 0.80}, forgetting_threshold=0.05, noise_floor=floor
        )
        assert deltas[0].regressed is True

    def test_the_floor_never_lowers_the_configured_threshold(self):
        """CONTROL. A measured floor BELOW --forgetting-threshold must not
        tighten the gate behind the operator's back."""
        from soup_cli.utils.ship_verdict import compute_benchmark_deltas, compute_noise_floor

        floor = compute_noise_floor([{"x": 0.50}, {"x": 0.501}])  # floor 0.001
        deltas = compute_benchmark_deltas(
            {"x": 0.90}, {"x": 0.86}, forgetting_threshold=0.05, noise_floor=floor
        )
        assert deltas[0].regressed is False

    def test_a_task_win_under_the_floor_is_not_a_win(self):
        from soup_cli.utils.ship_verdict import TASK_AXIS, build_task_win, compute_noise_floor

        floor = compute_noise_floor([{TASK_AXIS: 0.50}, {TASK_AXIS: 0.55}])  # 0.05
        win = build_task_win("metric", 0.60, 0.62, noise_floor=floor)
        assert win.won is False

    def test_a_task_win_over_the_floor_is_still_a_win(self):
        """CONTROL."""
        from soup_cli.utils.ship_verdict import TASK_AXIS, build_task_win, compute_noise_floor

        floor = compute_noise_floor([{TASK_AXIS: 0.50}, {TASK_AXIS: 0.51}])  # 0.01
        win = build_task_win("metric", 0.60, 0.70, noise_floor=floor)
        assert win.won is True

    def test_without_a_floor_nothing_changes(self):
        """CONTROL. The capability is opt-in; the default path must be
        byte-identical to v0.73.1."""
        from soup_cli.utils.ship_verdict import build_task_win, compute_benchmark_deltas

        assert build_task_win("metric", 0.60, 0.62).won is True
        deltas = compute_benchmark_deltas({"x": 0.90}, {"x": 0.86}, forgetting_threshold=0.05)
        assert deltas[0].regressed is False
        deltas = compute_benchmark_deltas({"x": 0.90}, {"x": 0.80}, forgetting_threshold=0.05)
        assert deltas[0].regressed is True


class TestNoiseFloorIsReported:
    def test_the_verdict_carries_the_floor(self):
        from soup_cli.utils.ship_verdict import (
            build_task_win,
            compute_benchmark_deltas,
            compute_noise_floor,
            decide_ship,
            verdict_to_dict,
        )

        floor = compute_noise_floor([{"x": 0.50}, {"x": 0.52}])
        verdict = decide_ship(
            build_task_win("metric", 0.5, 0.8),
            compute_benchmark_deltas({"x": 0.9}, {"x": 0.9}, noise_floor=floor),
            noise_floor=floor,
        )
        assert verdict.noise_floor is not None
        payload = verdict_to_dict(verdict)
        assert payload["noise_floor"]["runs"] == 2
        assert payload["noise_floor"]["floors"]["x"] == pytest.approx(0.02)

    def test_the_panel_prints_the_floor(self):
        from io import StringIO

        from rich.console import Console

        from soup_cli.utils.ship_verdict import (
            build_task_win,
            compute_benchmark_deltas,
            compute_noise_floor,
            decide_ship,
            render_ship_panel,
        )

        floor = compute_noise_floor([{"x": 0.50}, {"x": 0.52}])
        verdict = decide_ship(
            build_task_win("metric", 0.5, 0.8),
            compute_benchmark_deltas({"x": 0.9}, {"x": 0.9}, noise_floor=floor),
            noise_floor=floor,
        )
        buf = StringIO()
        Console(file=buf, width=100, no_color=True).print(render_ship_panel(verdict))
        out = buf.getvalue()
        assert "noise floor" in out.lower()

    def test_a_verdict_without_a_floor_serialises_none(self):
        """CONTROL. The default path must not grow a fabricated floor."""
        from soup_cli.utils.ship_verdict import (
            build_task_win,
            compute_benchmark_deltas,
            decide_ship,
            verdict_to_dict,
        )

        verdict = decide_ship(
            build_task_win("metric", 0.5, 0.8),
            compute_benchmark_deltas({"x": 0.9}, {"x": 0.9}),
        )
        assert verdict.noise_floor is None
        assert verdict_to_dict(verdict)["noise_floor"] is None


class TestNoiseFloorSurvivesTheEvidenceRoundTrip:
    """#312's property is that ``soup ship`` output replays as input. A verdict
    decided against a measured floor does NOT replay without it — the offline
    reader would recompute both legs at the bare threshold and could return the
    opposite decision."""

    def _floored_verdict(self):
        from soup_cli.utils.ship_verdict import (
            TASK_AXIS,
            build_task_win,
            compute_benchmark_deltas,
            compute_noise_floor,
            decide_ship,
        )

        floor = compute_noise_floor(
            [{"x": 0.50, TASK_AXIS: 0.40}, {"x": 0.62, TASK_AXIS: 0.40}]
        )
        win = build_task_win("metric", 0.50, 0.80, noise_floor=floor)
        deltas = compute_benchmark_deltas(
            {"x": 0.90}, {"x": 0.80}, forgetting_threshold=0.05, noise_floor=floor
        )
        return floor, decide_ship(win, deltas, forgetting_threshold=0.05,
                                  noise_floor=floor)

    def test_evidence_carries_the_floor(self):
        from soup_cli.utils.ship_verdict import verdict_to_evidence

        _floor, verdict = self._floored_verdict()
        evidence = verdict_to_evidence(verdict)
        assert evidence["noise_floor"]["runs"] == 2
        assert evidence["noise_floor"]["floors"]["x"] == pytest.approx(0.12)

    def test_replaying_the_evidence_reproduces_the_decision(self):
        from soup_cli.commands.ship import _verdict_from_evidence
        from soup_cli.utils.ship_verdict import verdict_to_evidence

        _floor, verdict = self._floored_verdict()
        replayed = _verdict_from_evidence(
            verdict_to_evidence(verdict), forgetting_threshold=0.05
        )
        assert replayed.decision == verdict.decision
        assert [d.regressed for d in replayed.benchmark_deltas] == [
            d.regressed for d in verdict.benchmark_deltas
        ]

    def test_dropping_the_floor_would_flip_the_decision(self):
        """CONTROL, and the reason the key has to exist: the SAME scores read
        WITHOUT the floor produce the opposite leg-2 answer. If this test ever
        passes trivially the round-trip test above is proving nothing."""
        from soup_cli.commands.ship import _verdict_from_evidence
        from soup_cli.utils.ship_verdict import verdict_to_evidence

        _floor, verdict = self._floored_verdict()
        evidence = verdict_to_evidence(verdict)
        assert verdict.decision == "SHIP"
        evidence.pop("noise_floor")
        stripped = _verdict_from_evidence(evidence, forgetting_threshold=0.05)
        assert stripped.decision == "DON'T SHIP"

    def test_evidence_without_a_floor_is_still_readable(self):
        """CONTROL. Every pre-v0.73.2 evidence file has no such key."""
        from soup_cli.commands.ship import _verdict_from_evidence

        verdict = _verdict_from_evidence(
            {
                "task": {"mode": "metric", "base": 0.5, "tuned": 0.8},
                "benchmarks": {"x": {"base": 0.9, "tuned": 0.9}},
            },
            forgetting_threshold=0.05,
        )
        assert verdict.decision == "SHIP"
        assert verdict.noise_floor is None

    @pytest.mark.parametrize(
        "block",
        [
            {"runs": 1, "floors": {"x": 0.1}},        # below the minimum
            {"runs": 99, "floors": {"x": 0.1}},       # above the maximum
            {"runs": 2, "floors": {"x": -0.1}},       # negative floor
            {"runs": 2, "floors": "not-a-mapping"},
            {"floors": {"x": 0.1}},                   # no runs
            "not-a-mapping",
        ],
    )
    def test_a_malformed_floor_block_is_refused_not_dropped(self, block):
        """A floor silently discarded on read replays as a DIFFERENT verdict,
        which is exactly what the round-trip exists to prevent."""
        from soup_cli.utils.ship_verdict import noise_floor_from_evidence

        with pytest.raises(ValueError):
            noise_floor_from_evidence(block)


class TestTaskWinStaysConsistentWithTheDecision:
    def test_the_panel_cannot_print_won_beside_a_task_win_failure(self):
        """``decide_ship`` recomputes leg 1 but used to store the CALLER's
        TaskWin, so a win built without the floor would render "won" beside a
        DON'T SHIP decided with it."""
        from io import StringIO

        from rich.console import Console

        from soup_cli.utils.ship_verdict import (
            TASK_AXIS,
            build_task_win,
            compute_benchmark_deltas,
            compute_noise_floor,
            decide_ship,
            render_ship_panel,
        )

        floor = compute_noise_floor([{TASK_AXIS: 0.40}, {TASK_AXIS: 0.50}])
        # Built WITHOUT the floor on purpose: won=True.
        win = build_task_win("metric", 0.60, 0.62)
        assert win.won is True
        verdict = decide_ship(
            win,
            compute_benchmark_deltas({"x": 0.9}, {"x": 0.9}),
            noise_floor=floor,
        )
        assert verdict.decision == "DON'T SHIP"
        assert verdict.task_win.won is False
        buf = StringIO()
        Console(file=buf, width=100, no_color=True).print(render_ship_panel(verdict))
        assert "no win" in buf.getvalue()

    def test_an_ordinary_verdict_keeps_its_task_win_object(self):
        """CONTROL. Canonicalising must not rebuild a TaskWin that was already
        consistent — the base/tuned/mode must survive untouched."""
        from soup_cli.utils.ship_verdict import (
            build_task_win,
            compute_benchmark_deltas,
            decide_ship,
        )

        win = build_task_win("pairwise", 0.5, 0.7)
        verdict = decide_ship(win, compute_benchmark_deltas({"x": 0.9}, {"x": 0.9}))
        assert verdict.task_win is win


class TestAHostileEvidenceFloorIsBoundedAndLoud:
    """An evidence file is UNTRUSTED input and a floor WIDENS the gate:
    ``"floors": {"mini_mmlu": 1.0}`` masks any possible drop on that axis.

    This does not cross a new trust boundary — anyone who can edit the file can
    already write ``{"base": 0.9, "tuned": 0.9}`` and force a SHIP outright —
    but it is a far quieter edit to miss in review, and `soup ci init` wires
    ``ship --evidence`` as a PR merge gate. So it is bounded, and it is never
    silent."""

    @pytest.mark.parametrize("bad", [1.0001, 2.0, 1e300, -0.0001])
    def test_a_floor_outside_zero_one_is_refused(self, bad):
        from soup_cli.utils.ship_verdict import noise_floor_from_evidence

        with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
            noise_floor_from_evidence({"runs": 2, "floors": {"x": bad}})

    def test_a_floor_inside_the_range_is_accepted(self):
        """CONTROL. A legitimate floor must still load — 1.0 IS arithmetically
        reachable from two runs, so the bound is hygiene, not the mitigation."""
        from soup_cli.utils.ship_verdict import noise_floor_from_evidence

        assert noise_floor_from_evidence({"runs": 2, "floors": {"x": 1.0}}).of("x") == 1.0

    def test_too_many_axes_is_refused(self):
        from soup_cli.utils.ship_verdict import noise_floor_from_evidence

        floors = {f"b{i}": 0.01 for i in range(51)}
        with pytest.raises(ValueError, match="too many axes"):
            noise_floor_from_evidence({"runs": 2, "floors": floors})

    def test_an_overlong_axis_name_is_refused(self):
        from soup_cli.utils.ship_verdict import noise_floor_from_evidence

        with pytest.raises(ValueError, match="chars"):
            noise_floor_from_evidence({"runs": 2, "floors": {"x" * 257: 0.01}})

    def test_an_evidence_supplied_floor_that_widens_the_gate_warns(
        self, tmp_path, monkeypatch, capsys
    ):
        """The actual mitigation: the evidence reader must be exactly as loud
        as the live measurement path. The quieter of two readers is the one an
        attacker picks."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        (tmp_path / "ev.json").write_text(
            json.dumps(
                {
                    "task": {"mode": "metric", "base": 0.5, "tuned": 0.9},
                    "benchmarks": {"mini_mmlu": {"base": 0.90, "tuned": 0.10}},
                    "noise_floor": {"runs": 2, "floors": {"mini_mmlu": 1.0}},
                }
            ),
            encoding="utf-8",
        )
        result = CliRunner().invoke(app, ["ship", "--evidence", "ev.json"])
        plain = _plain(result.output)
        assert "evidence-supplied" in plain
        assert "LOOSER" in plain
        # And it must still SHIP — the point is that the widening is LOUD, not
        # that it errors. A regression that both warned and crashed (or flipped
        # the verdict) would otherwise pass this test.
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_an_evidence_file_without_a_widening_floor_is_quiet(
        self, tmp_path, monkeypatch
    ):
        """CONTROL. Warning on every evidence file trains the operator to
        ignore the warning."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        (tmp_path / "ev.json").write_text(
            json.dumps(
                {
                    "task": {"mode": "metric", "base": 0.5, "tuned": 0.9},
                    "benchmarks": {"mini_mmlu": {"base": 0.90, "tuned": 0.90}},
                    "noise_floor": {"runs": 2, "floors": {"mini_mmlu": 0.01}},
                }
            ),
            encoding="utf-8",
        )
        result = CliRunner().invoke(app, ["ship", "--evidence", "ev.json"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "LOOSER" not in _plain(result.output)


class TestUntrustedNamesCannotDriveTheTerminal:
    """``rich.markup.escape`` neutralises Rich's ``[...]`` syntax and NOTHING
    else — a raw ESC byte survives it. Benchmark and axis names come out of an
    untrusted evidence file, so both render paths strip C0/DEL first (the
    ``_for_terminal`` pattern already used in six other command modules)."""

    HOSTILE = "x\x1b]0;PWNED\x07\x1b[2Jy"

    def _verdict_with(self, name):
        from soup_cli.utils.ship_verdict import (
            build_task_win,
            compute_benchmark_deltas,
            compute_noise_floor,
            decide_ship,
        )

        floor = compute_noise_floor([{name: 0.0}, {name: 0.02}])
        return decide_ship(
            build_task_win("metric", 0.5, 0.8),
            compute_benchmark_deltas({name: 0.9}, {name: 0.9}, noise_floor=floor),
            noise_floor=floor,
        )

    def test_the_injected_sequences_do_not_reach_the_panel(self):
        """Assert on the HOSTILE sequences specifically, not on "\\x1b is
        absent". Rich legitimately emits its own SGR colour codes whenever the
        output stream is colour-capable, so a blanket no-ESC assertion passes
        on Windows (Rich auto-disables colour) and fails on a Linux CI runner —
        the same platform asymmetry that turned this file's CLI tests red. It
        would also be testing the wrong thing: styling is not injection.
        """
        out = _panel_text(self._verdict_with(self.HOSTILE))
        assert "\x1b]0;" not in out, "OSC title-set survived"
        assert "\x07" not in out, "BEL survived"
        assert "\x1b[2J" not in out, "CSI clear-screen survived"

    def test_the_hostile_string_really_is_hostile(self):
        """CONTROL. If the fixture stopped containing the sequences, every
        assertion above would pass while proving nothing."""
        assert "\x1b]0;" in self.HOSTILE
        assert "\x07" in self.HOSTILE
        assert "\x1b[2J" in self.HOSTILE

    def test_the_visible_name_survives(self):
        """CONTROL. Stripping must remove the control bytes, not the name —
        a fix that blanked the row would hide which benchmark regressed."""
        out = _panel_text(self._verdict_with(self.HOSTILE))
        assert "PWNED" in out  # rendered as inert text, not executed

    def test_markup_in_a_name_is_still_inert(self):
        """CONTROL for the pre-existing escape(), which must not be lost."""
        out = _panel_text(self._verdict_with("[red]INJECTED[/red]"))
        assert "[red]INJECTED[/red]" in out

    def test_an_ordinary_name_is_untouched(self):
        """CONTROL."""
        assert "mini_mmlu" in _panel_text(self._verdict_with("mini_mmlu"))


class TestTheMcpEvidenceReaderHonoursTheFloor:
    """The evidence schema gained ``noise_floor`` in v0.73.2 and TWO readers
    consume that schema. If only the CLI one is updated, the same file replays
    to a DIFFERENT verdict through `soup mcp serve`'s ``ship_evidence`` tool —
    a silent disagreement between two shipped surfaces."""

    def _evidence(self):
        return {
            "task": {"mode": "metric", "base": 0.50, "tuned": 0.80},
            "benchmarks": {"x": {"base": 0.90, "tuned": 0.80}},
            "noise_floor": {"runs": 2, "floors": {"x": 0.12}},
        }

    def _write(self, tmp_path, payload):
        path = tmp_path / "ev.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_the_mcp_tool_applies_a_stored_floor(self, tmp_path, monkeypatch):
        from soup_cli.mcp_server.registry import tool_ship_evidence

        monkeypatch.chdir(tmp_path)
        self._write(tmp_path, self._evidence())
        out = tool_ship_evidence({"evidence": "ev.json"})
        assert out["decision"] == "SHIP"
        assert out["noise_floor"]["floors"]["x"] == pytest.approx(0.12)

    def test_without_the_floor_the_same_scores_dont_ship(self, tmp_path, monkeypatch):
        """CONTROL. Proves the floor is what carried the decision — without it
        this test would pass for both readers and prove nothing."""
        from soup_cli.mcp_server.registry import tool_ship_evidence

        monkeypatch.chdir(tmp_path)
        payload = self._evidence()
        payload.pop("noise_floor")
        self._write(tmp_path, payload)
        assert tool_ship_evidence({"evidence": "ev.json"})["decision"] == "DON'T SHIP"

    def test_both_readers_agree(self, tmp_path, monkeypatch):
        from soup_cli.commands.ship import _verdict_from_evidence
        from soup_cli.mcp_server.registry import tool_ship_evidence

        monkeypatch.chdir(tmp_path)
        payload = self._evidence()
        self._write(tmp_path, payload)
        cli = _verdict_from_evidence(payload, forgetting_threshold=0.05)
        mcp = tool_ship_evidence({"evidence": "ev.json"})
        assert cli.decision == mcp["decision"]

    def test_a_widening_floor_is_reported_in_the_mcp_result(
        self, tmp_path, monkeypatch
    ):
        """The CLI warns on stderr; this transport cannot (stdout is the
        JSON-RPC channel), so the warning must ride in the RESULT. Otherwise
        the MCP tool is the quiet reader an attacker would pick."""
        from soup_cli.mcp_server.registry import tool_ship_evidence

        monkeypatch.chdir(tmp_path)
        payload = self._evidence()
        payload["noise_floor"] = {"runs": 2, "floors": {"x": 1.0}}
        self._write(tmp_path, payload)
        out = tool_ship_evidence({"evidence": "ev.json"})
        assert out["warnings"], out
        assert "LOOSER" in out["warnings"][0]

    def test_a_narrow_floor_produces_no_warning(self, tmp_path, monkeypatch):
        """CONTROL. A warnings list that is never empty is not a warning."""
        from soup_cli.mcp_server.registry import tool_ship_evidence

        monkeypatch.chdir(tmp_path)
        payload = self._evidence()
        payload["noise_floor"] = {"runs": 2, "floors": {"x": 0.01}}
        self._write(tmp_path, payload)
        assert tool_ship_evidence({"evidence": "ev.json"})["warnings"] == []

    def test_a_malformed_floor_is_refused_by_the_mcp_reader_too(
        self, tmp_path, monkeypatch
    ):
        from soup_cli.mcp_server.registry import McpToolError, tool_ship_evidence

        monkeypatch.chdir(tmp_path)
        payload = self._evidence()
        payload["noise_floor"] = {"runs": 1, "floors": {"x": 0.12}}
        self._write(tmp_path, payload)
        with pytest.raises(McpToolError, match="noise_floor"):
            tool_ship_evidence({"evidence": "ev.json"})


class TestStaleBaselineIsAnnounced:
    """#357 / #346 moved the SCORER, so a ``--baseline`` snapshot taken before
    v0.73.2 is on a different scale for the affected suites — measured on an
    unchanged model, mini_mmlu 0.423 -> 0.731 and mini_tool_call 0.225 -> 1.000,
    both far larger than the 0.05 gate. Silently diffing across that is how a
    real regression gets masked."""

    def test_the_affected_suites_are_named(self):
        from soup_cli.eval.gate_suites import (
            DEFAULT_GENERAL_SUITE,
            SCORER_CHANGED_IN_V0_73_2,
        )

        assert set(SCORER_CHANGED_IN_V0_73_2) <= set(DEFAULT_GENERAL_SUITE)
        assert "mini_mmlu" in SCORER_CHANGED_IN_V0_73_2
        assert "mini_tool_call" in SCORER_CHANGED_IN_V0_73_2

    def test_every_unlisted_mcq_suite_really_is_unaffected(self):
        """CONTROL, DERIVED not hand-written: whatever MCQ suites are NOT on the
        list must be provably untouched by the prompt cue. A hand-written pair
        of names would silently stop covering a suite added later — the failure
        mode the project's scan-don't-list rule exists to prevent."""
        from soup_cli.eval.forgetting import MINI_BENCHMARKS, build_mcq_prompt
        from soup_cli.eval.gate_suites import SCORER_CHANGED_IN_V0_73_2

        unlisted = set(MINI_BENCHMARKS) - set(SCORER_CHANGED_IN_V0_73_2)
        assert unlisted, "the scan found nothing to check"
        for name in sorted(unlisted):
            for item in MINI_BENCHMARKS[name]:
                assert build_mcq_prompt(item["question"], item["answer"]) == (
                    item["question"]
                ), f"{name} IS affected by the prompt cue but is not listed"

    def test_every_listed_mcq_suite_really_is_affected(self):
        """CONTROL in the other direction: a name on the list that nothing
        actually changed would warn operators for no reason, which is how a
        warning stops being read."""
        from soup_cli.eval.forgetting import MINI_BENCHMARKS, build_mcq_prompt
        from soup_cli.eval.gate_suites import MINI_TOOL_CALL, SCORER_CHANGED_IN_V0_73_2

        for name in SCORER_CHANGED_IN_V0_73_2:
            if name == MINI_TOOL_CALL:  # behavioural, not MCQ — see #346
                continue
            assert any(
                build_mcq_prompt(item["question"], item["answer"]) != item["question"]
                for item in MINI_BENCHMARKS[name]
            ), f"{name} is listed but nothing changed for it"

    def test_a_stale_baseline_warns(self, capsys):
        from soup_cli.commands.ship import _leg2_scores

        def gen(prompt):
            return "B"

        _leg2_scores(
            ["mini_mmlu"],
            gen,
            gen,
            base_id="base",
            tuned_id="tuned",
            adapter=None,
            baseline_scores={"mini_mmlu": 0.42},
            device=None,
        )
        out = _plain(capsys.readouterr().out)
        assert "mini_mmlu" in out and "v0.73.2" in out

    def test_a_baseline_for_an_unaffected_suite_is_quiet(self, capsys):
        """CONTROL. A warning on every baseline would train the operator to
        ignore it."""
        from soup_cli.commands.ship import _leg2_scores

        def gen(prompt):
            return "hello 5 world"

        _leg2_scores(
            ["mini_arithmetic"],
            gen,
            gen,
            base_id="base",
            tuned_id="tuned",
            adapter=None,
            baseline_scores={"mini_arithmetic": 0.42},
            device=None,
        )
        assert "Warning" not in _plain(capsys.readouterr().out)

    def test_no_baseline_is_quiet(self, capsys):
        """CONTROL. The warning is about a STORED score, not about the suite."""
        from soup_cli.commands.ship import _leg2_scores

        def gen(prompt):
            return "B"

        _leg2_scores(
            ["mini_mmlu"],
            gen,
            gen,
            base_id="base",
            tuned_id="tuned",
            adapter=None,
            baseline_scores={},
            device=None,
        )
        assert "Warning" not in _plain(capsys.readouterr().out)


def _const_task_scorer(monkeypatch, value=0.5):
    """Pin the leg-1 task-axis scorer to a constant.

    Since #403 the task axis is measured in every mode (judge modes score it
    through a live judge). A test that only cares about the leg-2 / warning
    paths pins the task scorer here so it needs no judge, and the task axis
    contributes a flat 0.0 spread.
    """
    monkeypatch.setattr(
        "soup_cli.commands.ship._build_task_floor_scorer",
        lambda *args, **kwargs: (lambda: value),
    )


class TestTheFloorWideningTheThresholdIsAnnounced:
    def test_a_floor_above_the_threshold_warns(self, capsys, monkeypatch):
        from soup_cli.commands.ship import _measure_noise_floor

        _const_task_scorer(monkeypatch)
        # The two repeats must genuinely disagree. An every-other-call flip does
        # NOT do that: mini_mmlu has an EVEN number of items, so the second
        # pass starts on the same parity and reproduces the first exactly —
        # which is how this test first passed for the wrong reason.
        n_items = len(MINI_BENCHMARKS["mini_mmlu"])
        calls = {"n": 0}

        def gen(prompt):
            calls["n"] += 1
            return "B" if calls["n"] <= n_items else "C"

        _measure_noise_floor(
            2, ["mini_mmlu"], gen, base_id="b", task_mode="judge_score",
            task_eval="unused.jsonl", judge_model="ollama://j",
            forgetting_threshold=0.0,
        )
        out = _plain(capsys.readouterr().out)
        assert "exceeds" in out and "--forgetting-threshold" in out

    def test_a_deterministic_instrument_does_not_warn(self, capsys, monkeypatch):
        """CONTROL. A zero floor never widens anything."""
        from soup_cli.commands.ship import _measure_noise_floor

        _const_task_scorer(monkeypatch)
        floor = _measure_noise_floor(
            2, ["mini_mmlu"], lambda p: "B", base_id="b", task_mode="judge_score",
            task_eval="unused.jsonl", judge_model="ollama://j",
            forgetting_threshold=0.0,
        )
        out = _plain(capsys.readouterr().out)
        assert all(value == 0.0 for _n, value in floor.floors)
        assert "exceeds" not in out


def _panel_text(verdict):
    from io import StringIO

    from rich.console import Console

    from soup_cli.utils.ship_verdict import render_ship_panel

    buf = StringIO()
    Console(file=buf, width=120, no_color=True).print(render_ship_panel(verdict))
    return buf.getvalue()


class TestThePanelPrintsItsLegOneMarker:
    """PRE-EXISTING, found while rendering the noise-floor panel and fixed here.

    ``render_ship_panel`` built its header as ``... [{won_str}]``. A bare
    ``[no win]`` is valid Rich markup for an unknown tag, so the panel silently
    ATE its own leg-1 marker on every run up to and including v0.73.1 — while
    the plain-text ``format_ship_rubric``, which has no markup parser, printed
    it correctly the whole time, which is why nobody noticed. Reproduced on
    shipped code for both a win and a loss.
    """

    def _verdict(self, base, tuned):
        from soup_cli.utils.ship_verdict import (
            build_task_win,
            compute_benchmark_deltas,
            decide_ship,
        )

        return decide_ship(
            build_task_win("metric", base, tuned),
            compute_benchmark_deltas({"x": 0.9}, {"x": 0.9}),
        )

    def test_a_win_prints_won(self):
        assert "[won]" in _panel_text(self._verdict(0.5, 0.8))

    def test_a_loss_prints_no_win(self):
        assert "[no win]" in _panel_text(self._verdict(0.8, 0.5))

    def test_the_rubric_still_prints_it_too(self):
        """CONTROL. The rubric was already correct; escaping the panel must not
        change the surface that worked."""
        from soup_cli.utils.ship_verdict import format_ship_rubric

        assert "[won]" in format_ship_rubric(self._verdict(0.5, 0.8))
        assert "[no win]" in format_ship_rubric(self._verdict(0.8, 0.5))

    def test_the_marker_matches_the_decision(self):
        """CONTROL. Printing a marker is only worth anything if it is the RIGHT
        one — a fix that always printed "[won]" would pass the first test."""
        won = _panel_text(self._verdict(0.5, 0.8))
        lost = _panel_text(self._verdict(0.8, 0.5))
        assert "[no win]" not in won
        assert "[won]" not in lost.replace("[no win]", "")


class TestTheReasonLineDoesNotSayGotWorseForAGain:
    def test_a_gain_inside_the_floor_is_described_as_a_gain(self):
        """A task that went UP but by less than the floor must not be reported
        as "got worse" — that is the gate stating more than it measured."""
        from soup_cli.utils.ship_verdict import (
            TASK_AXIS,
            build_task_win,
            compute_benchmark_deltas,
            compute_noise_floor,
            decide_ship,
        )

        floor = compute_noise_floor([{TASK_AXIS: 0.40}, {TASK_AXIS: 0.50}])
        verdict = decide_ship(
            build_task_win("metric", 0.60, 0.62),
            compute_benchmark_deltas({"x": 0.9}, {"x": 0.9}),
            noise_floor=floor,
        )
        text = _panel_text(verdict)
        assert "got worse" not in text
        assert "noise floor" in text.lower()

    def test_a_real_loss_still_says_got_worse(self):
        """CONTROL. The floor-aware branch must not swallow a genuine drop."""
        from soup_cli.utils.ship_verdict import (
            TASK_AXIS,
            build_task_win,
            compute_benchmark_deltas,
            compute_noise_floor,
            decide_ship,
        )

        floor = compute_noise_floor([{TASK_AXIS: 0.40}, {TASK_AXIS: 0.50}])
        verdict = decide_ship(
            build_task_win("metric", 0.80, 0.50),
            compute_benchmark_deltas({"x": 0.9}, {"x": 0.9}),
            noise_floor=floor,
        )
        assert "got worse" in _panel_text(verdict)

    def test_a_tie_without_a_floor_still_says_tied(self):
        """CONTROL. The default path's wording is unchanged."""
        from soup_cli.utils.ship_verdict import (
            build_task_win,
            compute_benchmark_deltas,
            decide_ship,
        )

        verdict = decide_ship(
            build_task_win("metric", 0.7, 0.7),
            compute_benchmark_deltas({"x": 0.9}, {"x": 0.9}),
        )
        assert "tied" in _panel_text(verdict)


def _run_ship_cli(monkeypatch, base_gen, tuned_gen, extra_args=None):
    """Drive the REAL `soup ship` live path with injected generators.

    Mirrors ``test_v07138.TestShipLiveHeadline._run``. Needed because the
    interesting wiring — ``_verdict_live`` threading a measured floor into BOTH
    ``compute_benchmark_deltas`` and ``decide_ship``, and ``_leg2_scores``
    dispatching the new suite — lives between the CLI and the pure engine, so a
    direct call to either end steps over it.
    """
    import os as _os

    from typer.testing import CliRunner

    from soup_cli.cli import app
    from soup_cli.commands import ship as ship_cmd

    monkeypatch.setattr(
        ship_cmd, "_resolve_generators", lambda *a, **k: (base_gen, tuned_gen)
    )
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("task.jsonl", "w", encoding="utf-8") as fh:
            fh.write(
                json.dumps({"prompt": "say hi", "expected": "hi", "scoring": "contains"})
                + "\n"
            )
        args = [
            "ship", "--base", "base", "--adapter", "ad",
            "--task-eval", "task.jsonl", "--device", "cpu",
            "--output", "verdict.json",
        ]
        args += list(extra_args or [])
        result = runner.invoke(app, args)
        verdict = None
        if _os.path.exists("verdict.json"):
            with open("verdict.json", encoding="utf-8") as fh:
                verdict = json.load(fh)
        return result, verdict


class TestIssue317ThroughTheRealCli:
    """#317 end-to-end. ``TestIssue317LegTwoNowSeesIt`` hand-builds deltas and
    calls ``decide_ship``; that proves the arithmetic, not that the suite is
    dispatched, scored and named by the shipped command."""

    def _gen(self, *, refuse_benign, wins_task):
        from soup_cli.eval.forgetting import build_mcq_prompt

        tools = {it["prompt"]: it["expected"] for it in _tool_items()}
        benign = {it["prompt"] for it in load_suite_items(_over_refusal())}
        mcq = {}
        for bench in MINI_BENCHMARKS.values():
            for item in bench:
                mcq[build_mcq_prompt(item["question"], item["answer"])] = item["answer"]

        def gen(prompt):
            # Leg 1 must be a genuine WIN on the tuned side, or DON'T SHIP is
            # decided by leg 1 and says nothing about the new axis at all
            # (decide_ship's precedence puts task_win above regression).
            if prompt == "say hi":
                return "hi" if wins_task else "no"
            if prompt in tools:
                return tools[prompt]
            if prompt in benign:
                return REFUSAL if refuse_benign else "Sure — here is how."
            if prompt in mcq:
                return mcq[prompt]
            # mini_safety (harmful) + mini_format_json share this branch.
            return 'I cannot help with that. {"ok": true}'

        return gen

    def test_an_over_refusing_tune_is_blocked_by_the_shipped_command(self, monkeypatch):
        result, verdict = _run_ship_cli(
            monkeypatch,
            self._gen(refuse_benign=False, wins_task=False),
            self._gen(refuse_benign=True, wins_task=True),
        )
        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert verdict is not None
        # Leg 1 WON, so the DON'T SHIP is attributable to leg 2 alone.
        assert verdict["task_win"]["won"] is True
        assert verdict["failed_rule"] == "regression"
        regressed = [d["name"] for d in verdict["benchmark_deltas"] if d["regressed"]]
        assert regressed == [_over_refusal()]

    def test_the_same_harness_ships_when_the_tune_stays_helpful(self, monkeypatch):
        """CONTROL, and the one that matters: the ONLY difference from the test
        above is what the tuned model does with benign prompts."""
        result, verdict = _run_ship_cli(
            monkeypatch,
            self._gen(refuse_benign=False, wins_task=False),
            self._gen(refuse_benign=False, wins_task=True),
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert [d["name"] for d in verdict["benchmark_deltas"] if d["regressed"]] == []


class TestNoiseFloorThroughTheRealCli:
    """H1 — ``--noise-floor`` must be wired all the way through
    ``_verdict_live``, not just validated as a flag."""

    def _stable_gen(self, prompt):
        if prompt == "say hi":
            return "hi"
        return "B"

    def test_a_deterministic_run_measures_a_zero_floor_and_reports_it(
        self, monkeypatch
    ):
        result, verdict = _run_ship_cli(
            monkeypatch,
            self._stable_gen,
            self._stable_gen,
            extra_args=["--noise-floor", "2", "--general-suite", "mini_mmlu"],
        )
        assert result.exit_code in (0, 2), (result.output, repr(result.exception))
        assert verdict["noise_floor"] is not None
        assert verdict["noise_floor"]["runs"] == 2
        # metric mode -> the leg-1 axis IS measured, so it must be present.
        assert "__task__" in verdict["noise_floor"]["floors"]
        assert verdict["noise_floor"]["floors"]["mini_mmlu"] == 0.0
        assert "Noise floor" in _plain(result.output)

    def test_the_measured_floor_actually_changes_the_verdict(self, monkeypatch):
        """The load-bearing one. A tuned model that drops mini_mmlu past the
        0.05 threshold, on an instrument whose own repeats move MORE than that,
        must not be called a regression."""
        n_items = len(MINI_BENCHMARKS["mini_mmlu"])
        state = {"n": 0}

        def noisy_base(prompt):
            if prompt == "say hi":
                return "hi"
            state["n"] += 1
            # Alternate WHOLE passes so the two repeats genuinely disagree.
            return "B" if ((state["n"] - 1) // n_items) % 2 == 0 else "C"

        def dropped_tuned(prompt):
            if prompt == "say hi":
                return "hi hi"
            return "C"

        with_floor, v_floor = _run_ship_cli(
            monkeypatch, noisy_base, dropped_tuned,
            extra_args=["--noise-floor", "2", "--general-suite", "mini_mmlu"],
        )
        assert v_floor["noise_floor"]["floors"]["mini_mmlu"] > 0.05
        assert [d["name"] for d in v_floor["benchmark_deltas"] if d["regressed"]] == []

    def test_without_the_flag_the_same_run_is_gated_at_the_bare_threshold(
        self, monkeypatch
    ):
        """CONTROL for the test above — proves the floor is what changed the
        answer and not the generators. Same base/tuned, no --noise-floor."""
        n_items = len(MINI_BENCHMARKS["mini_mmlu"])
        state = {"n": 0}

        def noisy_base(prompt):
            if prompt == "say hi":
                return "hi"
            state["n"] += 1
            return "B" if ((state["n"] - 1) // n_items) % 2 == 0 else "C"

        def dropped_tuned(prompt):
            if prompt == "say hi":
                return "hi hi"
            return "C"

        result, verdict = _run_ship_cli(
            monkeypatch, noisy_base, dropped_tuned,
            extra_args=["--general-suite", "mini_mmlu"],
        )
        assert verdict["noise_floor"] is None
        assert [d["name"] for d in verdict["benchmark_deltas"] if d["regressed"]] == [
            "mini_mmlu"
        ]
        assert result.exit_code == 2, (result.output, repr(result.exception))


class TestMeasureNoiseFloorBranches:
    """M1 / L2 — the metric-mode leg-1 measurement and the secondary warnings."""

    def test_metric_mode_measures_the_leg_one_axis(self, tmp_path, monkeypatch):
        from soup_cli.commands.ship import _measure_noise_floor
        from soup_cli.utils.ship_verdict import TASK_AXIS

        monkeypatch.chdir(tmp_path)
        (tmp_path / "task.jsonl").write_text(
            json.dumps({"prompt": "say hi", "expected": "hi", "scoring": "contains"})
            + "\n",
            encoding="utf-8",
        )
        floor = _measure_noise_floor(
            2, ["mini_mmlu"], lambda p: "hi B", base_id="b", task_mode="metric",
            task_eval="task.jsonl", judge_model=None, forgetting_threshold=0.05,
        )
        assert TASK_AXIS in dict(floor.floors)
        # Metric mode is decode-only — the floor must NOT be stamped judge.
        assert floor.judge_inclusive is False

    def test_judge_mode_measures_the_leg_one_axis(self, capsys, monkeypatch):
        """#403 — judge modes now measure the task axis instead of warning and
        leaving it at a 0.0 floor. Base RED: the old code skipped it."""
        from soup_cli.commands.ship import _measure_noise_floor
        from soup_cli.utils.ship_verdict import TASK_AXIS

        _const_task_scorer(monkeypatch, value=0.5)
        floor = _measure_noise_floor(
            2, ["mini_mmlu"], lambda p: "B", base_id="b", task_mode="judge_score",
            task_eval="unused.jsonl", judge_model="ollama://j",
            forgetting_threshold=0.05,
        )
        assert TASK_AXIS in dict(floor.floors)
        assert floor.judge_inclusive is True
        out = _plain(capsys.readouterr().out)
        assert "does not measure the leg-1 task axis" not in out

    def test_a_non_bundled_suite_is_reported_as_unmeasured(self, capsys, monkeypatch):
        """A silently-skipped axis reads as 'floor 0.0', i.e. as a measurement
        that was never taken."""
        from soup_cli.commands.ship import _measure_noise_floor

        _const_task_scorer(monkeypatch)
        floor = _measure_noise_floor(
            2, ["mini_mmlu", "hellaswag"], lambda p: "B", base_id="b",
            task_mode="judge_score", task_eval="unused.jsonl",
            judge_model="ollama://j", forgetting_threshold=0.05,
        )
        assert "hellaswag" not in dict(floor.floors)
        out = _plain(capsys.readouterr().out)
        assert "bundled suites only" in out and "hellaswag" in out

    def test_a_deterministic_instrument_says_so(self, capsys, monkeypatch):
        from soup_cli.commands.ship import _measure_noise_floor

        _const_task_scorer(monkeypatch)
        _measure_noise_floor(
            2, ["mini_mmlu"], lambda p: "B", base_id="b", task_mode="judge_score",
            task_eval="unused.jsonl", judge_model="ollama://j",
            forgetting_threshold=0.05,
        )
        assert "deterministic" in _plain(capsys.readouterr().out)


class TestNoiseFloorJudgeModes:
    """#403 — the leg-1 task floor is measured in the judge modes too, and the
    result is stamped so it is never read as a decode-only number."""

    def test_pairwise_floor_measures_the_base_against_itself(self, monkeypatch):
        from soup_cli.commands import ship as ship_mod
        from soup_cli.commands.ship import _measure_noise_floor
        from soup_cli.utils.ship_verdict import TASK_AXIS

        spread = iter([0.4, 0.6])

        def fake_pairwise(task_eval, judge_model):
            def winrate(gen_a, gen_b):
                # The maintainer's design: the base judged against ITSELF.
                assert gen_a is gen_b
                return next(spread)

            return winrate

        monkeypatch.setattr(ship_mod, "_build_pairwise_scorer", fake_pairwise)
        floor = _measure_noise_floor(
            2, [], lambda p: "x", base_id="b", task_mode="pairwise",
            task_eval="t.jsonl", judge_model="ollama://j",
            forgetting_threshold=0.05,
        )
        assert floor.of(TASK_AXIS) == pytest.approx(0.2, abs=1e-9)
        assert floor.judge_inclusive is True

    def test_judge_score_floor_measures_the_base_side(self, monkeypatch):
        from soup_cli.commands import ship as ship_mod
        from soup_cli.commands.ship import _measure_noise_floor
        from soup_cli.utils.ship_verdict import TASK_AXIS

        spread = iter([0.3, 0.5])

        def fake_judge(task_eval, judge_model):
            return lambda gen: next(spread)

        monkeypatch.setattr(ship_mod, "_build_judge_scorer", fake_judge)
        floor = _measure_noise_floor(
            2, [], lambda p: "x", base_id="b", task_mode="judge_score",
            task_eval="t.jsonl", judge_model="ollama://j",
            forgetting_threshold=0.05,
        )
        assert floor.of(TASK_AXIS) == pytest.approx(0.2, abs=1e-9)
        assert floor.judge_inclusive is True

    def test_a_judge_floor_is_never_read_as_decode_only(self):
        """CONTROL (crit 3 + 5). A judge-inclusive floor is labelled and stamped
        on every surface; a decode-only floor is not — a reader can always tell
        them apart, so the judge floor is never silently reused as decode noise.
        """
        from io import StringIO

        from rich.console import Console

        from soup_cli.utils.ship_verdict import (
            TASK_AXIS,
            NoiseFloor,
            ShipVerdict,
            _render_noise_floor,
            build_task_win,
            noise_floor_from_evidence,
            verdict_to_dict,
            verdict_to_evidence,
        )

        judge = NoiseFloor(runs=3, floors=((TASK_AXIS, 0.2),), judge_inclusive=True)
        decode = NoiseFloor(runs=3, floors=((TASK_AXIS, 0.2),), judge_inclusive=False)

        def _table(floor):
            buf = StringIO()
            Console(file=buf, width=120, no_color=True).print(_render_noise_floor(floor))
            return buf.getvalue()

        # Panel label.
        assert "decode + judge" in _table(judge)
        assert "decode + judge" not in _table(decode)
        assert "leg 1 task" in _table(decode)

        def _verdict(floor):
            return ShipVerdict(
                decision="SHIP",
                task_win=build_task_win("pairwise", 0.5, 0.7),
                benchmark_deltas=(),
                failed_rule=None,
                forgetting_threshold=0.05,
                soup_version="x",
                noise_floor=floor,
            )

        # verdict_to_dict always carries the marker.
        assert verdict_to_dict(_verdict(judge))["noise_floor"]["judge_inclusive"] is True
        assert verdict_to_dict(_verdict(decode))["noise_floor"]["judge_inclusive"] is False

        # Evidence stamps it only when True (a decode-only floor round-trips
        # byte-for-byte), and it round-trips back through the reader.
        ev_judge = verdict_to_evidence(_verdict(judge))["noise_floor"]
        ev_decode = verdict_to_evidence(_verdict(decode))["noise_floor"]
        assert ev_judge["judge_inclusive"] is True
        assert "judge_inclusive" not in ev_decode
        assert noise_floor_from_evidence(ev_judge).judge_inclusive is True
        assert noise_floor_from_evidence(ev_decode).judge_inclusive is False


class TestBuildMcqPrompt:
    """M5 — public API, so its contract gets its own tests rather than being
    inferred from a caller."""

    def test_an_mcq_item_gains_the_cue(self):
        from soup_cli.eval.forgetting import build_mcq_prompt

        out = build_mcq_prompt("What is 2 + 2? (A) 3 (B) 4", "B")
        assert out.startswith("What is 2 + 2? (A) 3 (B) 4")
        assert "letter" in out.lower()

    def test_a_free_text_item_is_returned_unchanged(self):
        from soup_cli.eval.forgetting import build_mcq_prompt

        assert build_mcq_prompt("Capital of France?", "Paris") == "Capital of France?"

    def test_a_non_string_question_is_refused(self):
        from soup_cli.eval.forgetting import build_mcq_prompt

        with pytest.raises(TypeError, match="question"):
            build_mcq_prompt(None, "B")  # type: ignore[arg-type]

    def test_a_non_string_answer_is_treated_as_free_text(self):
        """CONTROL. A malformed row must not crash the scorer mid-benchmark."""
        from soup_cli.eval.forgetting import build_mcq_prompt

        assert build_mcq_prompt("q", None) == "q"  # type: ignore[arg-type]

    def test_it_is_idempotent_per_call_not_cumulative(self):
        """CONTROL. The cue must not stack if a caller composes prompts."""
        from soup_cli.eval.forgetting import build_mcq_prompt

        once = build_mcq_prompt("q (A) x (B) y", "B")
        assert once.count("letter") == 1


class TestBareFunctionShapeGuard:
    """L3 — ``_looks_like_a_bare_function`` decides whether an envelope-less
    object is a tool call, so its shape guard needs its own negatives."""

    @pytest.mark.parametrize(
        "obj",
        [
            {"name": 123, "arguments": {}},          # non-string name
            {"name": "x"},                            # no arguments
            {"arguments": {}},                        # no name
            {"name": "x", "description": "d"},        # a menu entry
            {"function": {"name": "x"}, "name": "y", "arguments": {}},  # already wrapped
            [],
            None,
            "a string",
        ],
    )
    def test_non_call_shapes_are_rejected(self, obj):
        from soup_cli.eval.gate_suites import _looks_like_a_bare_function

        assert _looks_like_a_bare_function(obj) is False

    def test_a_real_bare_call_is_accepted(self):
        """CONTROL."""
        from soup_cli.eval.gate_suites import _looks_like_a_bare_function

        assert _looks_like_a_bare_function(
            {"name": "get_weather", "arguments": {"city": "Paris"}}
        ) is True


class TestNoiseFloorCliFlag:
    def _runner(self):
        from typer.testing import CliRunner

        return CliRunner()

    def test_the_flag_exists_and_is_documented(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        result = CliRunner().invoke(app, ["ship", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        plain = _plain(result.output)
        assert "--noise-floor" in plain
        # #403 crit 4 — the judge modes cost N extra JUDGE passes; the help must
        # say so. "JUDGE" (a single upper-case token) survives Rich line-wrap.
        assert "JUDGE" in plain

    @pytest.mark.parametrize("bad", ["1", "0", "-3", "11"])
    def test_out_of_range_is_a_usage_error(self, bad, tmp_path):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        tasks = tmp_path / "t.jsonl"
        tasks.write_text('{"prompt": "hi", "expected": "hi"}\n', encoding="utf-8")
        result = CliRunner().invoke(
            app,
            [
                "ship", "--base", "m", "--adapter", "a",
                "--task-eval", str(tasks), "--noise-floor", bad,
            ],
        )
        assert result.exit_code == 3, (result.output, repr(result.exception))
        assert "--noise-floor" in _plain(result.output)

    def test_an_offline_verdict_refuses_to_measure_a_floor(self, tmp_path, monkeypatch):
        """``--evidence`` loads no model, so there is nothing to repeat.
        Refusing beats silently ignoring the flag the operator asked for."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        ev = tmp_path / "ev.json"
        ev.write_text(
            json.dumps(
                {
                    "task": {"mode": "metric", "base": 0.5, "tuned": 0.8},
                    "benchmarks": {"x": {"base": 0.9, "tuned": 0.9}},
                }
            ),
            encoding="utf-8",
        )
        result = CliRunner().invoke(app, ["ship", "--evidence", "ev.json",
                                          "--noise-floor", "3"])
        assert result.exit_code == 3, (result.output, repr(result.exception))
        assert "--noise-floor" in _plain(result.output)

    def test_an_offline_verdict_without_the_flag_still_works(self, tmp_path, monkeypatch):
        """CONTROL. The refusal must be about the FLAG, not about --evidence."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        ev = tmp_path / "ev.json"
        ev.write_text(
            json.dumps(
                {
                    "task": {"mode": "metric", "base": 0.5, "tuned": 0.8},
                    "benchmarks": {"x": {"base": 0.9, "tuned": 0.9}},
                }
            ),
            encoding="utf-8",
        )
        result = CliRunner().invoke(app, ["ship", "--evidence", "ev.json"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
