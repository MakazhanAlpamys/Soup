"""#355 — ``score_bundled_suite`` returned 0.0 for a non-callable ``gen``.

``score_bundled_suite(name, gen)`` feeds ``soup ship``'s leg 2. Its output is a
per-model absolute score in ``[0, 1]`` where ``0.0`` reads as "the model failed
every item" — a DON'T-SHIP verdict. Before this fix, handing it anything that is
not a callable generator (a list, ``None``, a string) returned ``0.0`` instead of
raising, so a *caller error* was indistinguishable from a real model regression,
and it failed in the direction that looks like a genuine finding.

The docstring already committed to the opposite behaviour for a neighbouring
case ("Raises ``ValueError`` for an unknown suite — never silently 0.0"); the
intent was only half enforced. This locks in:

- a non-callable ``gen`` raises ``TypeError`` (both behavioural and MCQ suites);
- a callable generator that errors on *every* item raises ``RuntimeError`` rather
  than reporting a plausible unanimous ``0.0``;
- a working generator still scores normally, and a generator that legitimately
  returns ``""`` on every item is a real ``0.0`` — NOT an error — so it must not
  raise. That control is what keeps the fix from turning every genuine zero into
  a crash.
"""

import pytest

from soup_cli.eval.gate_suites import (
    MINI_TOOL_CALL,
    load_suite_items,
    score_bundled_suite,
)

_BEHAVIOURAL_SUITE = MINI_TOOL_CALL  # JSONL-backed, routes through _fraction_passing
_MCQ_SUITE = "mini_arithmetic"  # routes through ForgettingDetector
_JSON_SUITE = "mini_format_json"  # scorer = "is this a JSON container?"

_NON_CALLABLES = [["a", "b"], None, "text", 42, {"function": "get_weather"}]


class TestNonCallableGenRaises:
    """Acceptance 1 — a non-callable ``gen`` raises ``TypeError``, never 0.0."""

    @pytest.mark.parametrize("bad_gen", _NON_CALLABLES)
    def test_behavioural_suite_rejects_non_callable(self, bad_gen):
        with pytest.raises(TypeError):
            score_bundled_suite(_BEHAVIOURAL_SUITE, bad_gen)

    @pytest.mark.parametrize("bad_gen", _NON_CALLABLES)
    def test_mcq_suite_rejects_non_callable(self, bad_gen):
        # Acceptance 4 — the guard covers the OTHER entry route too, not just the
        # behavioural scorers where the bug was first observed.
        with pytest.raises(TypeError):
            score_bundled_suite(_MCQ_SUITE, bad_gen)

    def test_exact_issue_examples_no_longer_return_zero(self):
        """The three snippets from the issue body, each of which returned 0.0."""
        for bad in (["a", "b"], None, "text"):
            with pytest.raises(TypeError):
                score_bundled_suite(_BEHAVIOURAL_SUITE, bad)


class TestAllItemsErroredIsDistinguishable:
    """Acceptance 2 — a generator that raises on every item is a broken
    generator, not a model scoring zero, so it raises rather than returning 0.0."""

    def test_generator_raising_on_every_item_raises(self):
        def broken(_prompt):
            raise RuntimeError("generator is broken")

        with pytest.raises(RuntimeError):
            score_bundled_suite(_BEHAVIOURAL_SUITE, broken)

    def test_a_single_raising_item_is_still_isolated_not_fatal(self):
        # The per-item isolation stays a feature: only a TOTAL failure raises.
        # One raising prompt among many still scores, it does not abort the run.
        items = load_suite_items(_JSON_SUITE)
        first_prompt = items[0].get("prompt", "")

        def flaky(prompt):
            if prompt == first_prompt:
                raise RuntimeError("one bad row")
            return '{"answer": 1}'

        score = score_bundled_suite(_JSON_SUITE, flaky)
        assert 0.0 < score < 1.0

    def test_generator_returning_non_str_scores_zero_not_raises(self):
        # A callable that RETURNS a non-str produced output, it did not error, so
        # the documented "non-str generation scores 0, never raises" contract is
        # preserved (a raise is reserved for a generator that never runs at all).
        assert score_bundled_suite(_BEHAVIOURAL_SUITE, lambda _p: None) == 0.0


class TestGenuineZeroStillReturns:
    """Control (acceptance 3) — the fix must not turn a real 0.0 into a crash.

    A generator that legitimately returns ``""`` runs fine on every item; the
    model just never produces a JSON container. That is a true ``0.0`` and must
    stay a returned value, distinguishable from the errored cases above.
    """

    def test_empty_string_generator_scores_zero_without_raising(self):
        assert score_bundled_suite(_JSON_SUITE, lambda _p: "") == 0.0

    def test_working_generator_scores_above_zero(self):
        # Every answer is a valid JSON container -> the format suite passes it.
        score = score_bundled_suite(_JSON_SUITE, lambda _p: '{"answer": 1}')
        assert score > 0.0


class TestUnknownSuiteContractUnchanged:
    """The pre-existing contract (unknown suite -> ValueError) still holds, and a
    callable ``gen`` is validated before the suite name so neither guard hides
    the other."""

    def test_unknown_suite_still_raises_value_error(self):
        with pytest.raises(ValueError):
            score_bundled_suite("no_such_suite", lambda _p: "x")

    def test_non_callable_gen_is_rejected_even_for_unknown_suite(self):
        with pytest.raises(TypeError):
            score_bundled_suite("no_such_suite", ["not", "callable"])
