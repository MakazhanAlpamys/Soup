"""Tests for issue #1755 — tool-call scorers must not conflate JSON booleans with numbers.

JSON has one boolean type and one number type. Python has ``bool`` as a subclass of
``int``, so ``True == 1`` and ``False == 0``, and that holds inside dicts and lists too.
Both scorers compared parsed arguments with ``==``, so a model that answered ``1`` where
the task expected ``true`` was given the full credit owed to one that answered ``true`` —
a wrong answer scored as a right one, through ``score_task`` and ``run_eval`` alike.

The policy implemented here is the one the issue's Expected section states:

- ``tool_call_match`` gives ``0.0`` for a boolean/number substitution;
- ``tool_call_args_subset`` keeps its ``0.5`` function-name credit and gives no argument
  credit, so ``0.5``. Its existing ``0.5`` ``matched`` threshold is untouched;
- numbers stay equal across forms, so ``1 == 1.0`` and ``0 == 0.0`` still match. The
  issue calls that a passing control and not a policy change.

Every case here is driven through ``score_task`` rather than the scorer functions alone,
because the scorer's return value is not what the bug was reported against.
"""

from __future__ import annotations

import json

import pytest

from soup_cli.eval.custom import EvalTask, run_eval, score_task

#: The two scorers that compare argument values.
COMPARING_SCORERS = ("tool_call_match", "tool_call_args_subset")

#: What each scorer must answer for a boolean/number substitution. The subset scorer
#: keeps half its credit for the function name, which does match.
EXPECTED_ON_SUBSTITUTION = {"tool_call_match": 0.0, "tool_call_args_subset": 0.5}

#: (gold, generated) pairs that are different JSON values of different JSON types.
#: Both directions, both booleans, per the issue.
SUBSTITUTIONS = [
    (True, 1),
    (False, 0),
    (1, True),
    (0, False),
]

#: (label, gold, generated) pairs that must keep scoring 1.0. `1 == 1.0` is the one that
#: a `type(a) == type(b)` guard would break, which is why it is here.
CONTROLS = [
    ("equal booleans", True, True),
    ("equal false", False, False),
    ("one equals one point zero", 1, 1.0),
    ("zero equals zero point zero", 0, 0.0),
    ("equal strings", "a", "a"),
    ("equal null", None, None),
    ("equal floats", 1.5, 1.5),
]


def call(arguments: object, name: str = "set_flag") -> str:
    """One tool call, in the shape both scorers parse.

    The value goes under an `enabled` key rather than being the whole of `arguments`:
    `_parse_args` returns `None` unless `arguments` parses to a dict, so a bare scalar
    would take both scorers down a different path and quietly score 1.0 whatever it was.
    """
    return json.dumps({"function": {"name": name, "arguments": {"enabled": arguments}}})


def call_args(arguments: dict, name: str = "set_flag") -> str:
    """One tool call whose `arguments` is exactly `arguments`, with no nesting added."""
    return json.dumps({"function": {"name": name, "arguments": arguments}})


def task(scoring: str, expected_arguments: object) -> EvalTask:
    return EvalTask(prompt="set the flag", expected=call(expected_arguments), scoring=scoring)


def scored(scoring: str, gold: object, generated: object) -> float:
    return score_task(task(scoring, gold), call(generated)).score


class TestABooleanIsNotItsNumericTwin:
    """The defect: a wrong answer scored as a right one."""

    @pytest.mark.parametrize("scoring", COMPARING_SCORERS)
    @pytest.mark.parametrize(("gold", "generated"), SUBSTITUTIONS)
    def test_a_substitution_does_not_score(self, scoring: str, gold, generated) -> None:
        assert scored(scoring, gold, generated) == EXPECTED_ON_SUBSTITUTION[scoring], (
            f"{scoring} gave a JSON {type(generated).__name__} where the task expected a "
            f"JSON {type(gold).__name__}. They are different JSON values, so the answer "
            "is wrong and must not earn argument credit."
        )

    def test_the_subset_scorer_halves_only_the_swapped_argument(self) -> None:
        """The partial-credit arithmetic, which the single-argument cases cannot show.

        Two expected arguments, one correct and one swapped: the name credit is the same
        0.5 as `test_the_subset_scorer_keeps_exactly_its_name_credit`, and the argument
        credit is halved rather than withheld, so 0.5 + 0.25. Reverting the matched-count
        to `==` would make all four arguments match and score 1.0.
        """
        result = score_task(
            EvalTask(
                prompt="p",
                expected=call_args({"enabled": True, "level": 2}),
                scoring="tool_call_args_subset",
            ),
            call_args({"enabled": 1, "level": 2}),
        )
        assert result.score == pytest.approx(0.75)

    def test_the_subset_scorer_keeps_exactly_its_name_credit(self) -> None:
        """CONTROL for the 0.5. Half the score is the function name matching, which is
        unaffected; only the argument half is withheld. If this ever fell below 0.5 the
        ``matched >= 0.5`` threshold in score_task would report the task as not matched,
        which is a behaviour change the issue explicitly did not ask for."""
        result = score_task(task("tool_call_args_subset", {"enabled": True}),
                            call({"enabled": 1}))
        assert result.score == 0.5
        assert result.matched is True, "the issue asked for no change to the 0.5 threshold"

    def test_a_wrong_function_name_still_scores_zero_not_half(self) -> None:
        """CONTROL. The retained 0.5 is the NAME credit, so a name mismatch must still be
        0.0 — otherwise the fix would have read as 'arguments are always worth half'."""
        result = score_task(
            task("tool_call_args_subset", {"enabled": True}),
            call({"enabled": 1}, name="other_fn"),
        )
        assert result.score == 0.0
        assert result.matched is False


class TestWhatMustKeepScoring:
    """The controls. A fix that separates `1` from `1.0` is wrong, not stricter."""

    @pytest.mark.parametrize("scoring", COMPARING_SCORERS)
    @pytest.mark.parametrize(("label", "gold", "generated"), CONTROLS, ids=[c[0] for c in CONTROLS])
    def test_equal_values_still_score(self, scoring: str, label: str, gold, generated) -> None:
        assert scored(scoring, {"enabled": gold}, {"enabled": generated}) == 1.0, (
            f"{label}: equal arguments must still earn full credit. Numbers comparing "
            "equal across forms is an existing control, not a policy change."
        )

    @pytest.mark.parametrize("scoring", COMPARING_SCORERS)
    def test_a_mixed_object_of_equal_values_scores(self, scoring: str) -> None:
        """CONTROL. A whole argument object that is genuinely equal still matches, so
        the fix did not turn into 'every object must be compared element-wise strictly'."""
        same = {"flag": True, "count": 1.0, "label": "a", "items": [1, 2, {"deep": None}]}
        assert scored(scoring, same, dict(same)) == 1.0


class TestTheSubstitutionIsCaughtInsideNesting:
    """A substitution buried in a list or a nested object is the same defect."""

    @pytest.mark.parametrize("scoring", COMPARING_SCORERS)
    @pytest.mark.parametrize(
        ("label", "gold", "generated"),
        [
            ("nested dict", {"a": {"b": True}}, {"a": {"b": 1}}),
            # The swapped element must be a bool against its own numeric twin. An
            # earlier cut used [1, 2] vs [1, True], which compares 2 with True --
            # different under plain `==` as well, so the case passed with the bool
            # branch disabled and proved nothing.
            ("inside a list", {"a": [1, 1]}, {"a": [1, True]}),
            ("list of objects", {"a": [{"b": False}]}, {"a": [{"b": 0}]}),
        ],
        ids=["nested dict", "inside a list", "list of objects"],
    )
    def test_a_buried_substitution_does_not_score(
        self, scoring: str, label, gold, generated
    ) -> None:
        assert scored(scoring, gold, generated) == EXPECTED_ON_SUBSTITUTION[scoring], (
            f"{label}: the comparison recurses, so a substitution at any depth is caught"
        )

    @pytest.mark.parametrize("scoring", COMPARING_SCORERS)
    def test_an_equal_structure_still_scores(self, scoring: str) -> None:
        same = {"a": [1, 2, {"b": True, "c": [0.5]}]}
        assert scored(scoring, same, {"a": [1, 2, {"b": True, "c": [0.5]}]}) == 1.0

    def test_a_list_of_a_different_length_does_not_score(self) -> None:
        assert scored("tool_call_match", {"a": [1, 2]}, {"a": [1, 2, 3]}) == 0.0

    def test_a_dict_with_a_different_key_set_does_not_score(self) -> None:
        assert scored("tool_call_match", {"a": 1}, {"a": 1, "b": 2}) == 0.0


class TestAMissingKeyIsStillAMissingKey:
    """CONTROL. The subset scorer counts present keys; the fix must not change that.

    These build their argument objects directly rather than through `call`, which nests
    a single value under `enabled` and would turn a two-key expected object into
    `{"enabled": {"enabled": ..., "level": 2}}`.
    """

    def test_a_missing_argument_is_not_matched(self) -> None:
        result = score_task(
            EvalTask(
                prompt="p",
                expected=call_args({"enabled": True, "level": 2}),
                scoring="tool_call_args_subset",
            ),
            call_args({"enabled": True}),
        )
        # One of two expected arguments present and correct: 0.5 name + 0.25 args.
        assert result.score == pytest.approx(0.75)

    def test_an_extra_argument_does_not_inflate_the_score(self) -> None:
        result = score_task(
            EvalTask(
                prompt="p",
                expected=call_args({"enabled": True}),
                scoring="tool_call_args_subset",
            ),
            call_args({"enabled": True, "extra": 99}),
        )
        assert result.score == 1.0


class TestItReachesThePublicPaths:
    """The issue reports the score through `score_task` and `run_eval`, not a helper.

    Everything above drives `score_task`. These two drive the entry point a caller
    actually uses, so the fix cannot be satisfied by changing a private helper that
    nothing routes through.
    """

    def test_run_eval_scores_a_substitution_as_incorrect(self) -> None:
        results = run_eval(
            model_path="unused",
            tasks=[task("tool_call_match", {"enabled": True})],
            generate_fn=lambda prompt: call({"enabled": 1}),
        )
        assert results.results[0].score == 0.0
        assert results.results[0].matched is False
        assert results.accuracy == 0.0

    def test_run_eval_keeps_scoring_a_correct_call(self) -> None:
        results = run_eval(
            model_path="unused",
            tasks=[task("tool_call_match", {"enabled": True})],
            generate_fn=lambda prompt: call({"enabled": True}),
        )
        assert results.results[0].score == 1.0
        assert results.accuracy == 1.0

    def test_run_eval_halves_a_subset_scorer_substitution(self) -> None:
        results = run_eval(
            model_path="unused",
            tasks=[task("tool_call_args_subset", {"enabled": True})],
            generate_fn=lambda prompt: call({"enabled": 1}),
        )
        assert results.results[0].score == 0.5


class TestStringEncodedArgumentsBehaveTheSame:
    """CONTROL. `_parse_args` accepts `arguments` as a JSON string or an object, and both
    spellings must be judged by the same rule — otherwise the fix would only close half
    the door, since the existing tests all use the string spelling."""

    @pytest.mark.parametrize("scoring", COMPARING_SCORERS)
    @pytest.mark.parametrize(("gold", "generated"), [(True, 1), (1, True)])
    def test_encoded_arguments_are_compared_the_same_way(self, scoring, gold, generated) -> None:
        output = json.dumps(
            {"function": {"name": "set_flag", "arguments": json.dumps({"enabled": generated})}}
        )
        expected = json.dumps(
            {"function": {"name": "set_flag", "arguments": json.dumps({"enabled": gold})}}
        )
        from soup_cli.eval.custom import score_task as direct

        assert direct(EvalTask(prompt="p", expected=expected, scoring=scoring),
                      output).score == EXPECTED_ON_SUBSTITUTION[scoring]

    @pytest.mark.parametrize("scoring", COMPARING_SCORERS)
    def test_encoded_equal_arguments_still_score(self, scoring) -> None:
        output = json.dumps(
            {"function": {"name": "set_flag", "arguments": json.dumps({"enabled": 1.0})}}
        )
        expected = json.dumps(
            {"function": {"name": "set_flag", "arguments": json.dumps({"enabled": 1})}}
        )
        from soup_cli.eval.custom import score_task as direct

        assert direct(EvalTask(prompt="p", expected=expected, scoring=scoring),
                      output).score == 1.0
