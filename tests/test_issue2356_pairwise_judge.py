"""Regression tests for the pairwise judge parser (#2356).

Verifies that `_parse_pairwise`:
1. Accepts clear A/B verdicts in various formats (Answer: A, Winner: B, etc.)
2. Rejects ambiguous text that happens to start with A/B (Answer: B, Better: A, Banana)
2. Returns -1 for unparseable replies to trigger the invalid-response policy
3. The swapped-order check still works correctly with the new parsing
"""

from __future__ import annotations

from soup_cli.eval.judge import _parse_pairwise, pairwise_compare


class _FakeJudge:
    """Synthetic judge that always returns a specific verdict (0=A, 1=B, -1=tie)."""

    def __init__(self, verdict: int):
        self.verdict = verdict
        self.calls = 0

    def compare_pair(self, prompt, resp_a, resp_b):
        self.calls += 1
        return self.verdict


class TestParsePairwiseRegression:
    """Tests for the specific regression cases from #2356."""

    def test_answer_b_is_parsed_as_b(self) -> None:
        """`Answer: B` should parse as B (1), not A (0)."""
        assert _parse_pairwise("Answer: B") == 1
        assert _parse_pairwise("Answer: b") == 1
        assert _parse_pairwise("The answer is B") == 1
        assert _parse_pairwise("ANSWER: B") == 1

    def test_better_response_is_a_parsed_as_a(self) -> None:
        """`Better response is A` should parse as A (0), not B (1)."""
        assert _parse_pairwise("Better response is A") == 0
        assert _parse_pairwise("The better one is A") == 0
        assert _parse_pairwise("BETTER: A") == 0

    def test_banana_is_unparseable(self) -> None:
        """`Banana` and similar words should return -1 (unparseable)."""
        assert _parse_pairwise("Banana") == -1
        assert _parse_pairwise("Apple") == -1
        assert _parse_pairwise("Coconut") == -1
        assert _parse_pairwise("Random text") == -1

    def test_explicit_verdict_patterns(self) -> None:
        """Explicit verdict patterns should be recognized."""
        # Winner patterns
        assert _parse_pairwise("Winner: A") == 0
        assert _parse_pairwise("Winner: B") == 1
        assert _parse_pairwise("The winner is A") == 0
        assert _parse_pairwise("The winner is B") == 1
        # Choice patterns
        assert _parse_pairwise("Choice: A") == 0
        assert _parse_pairwise("Choice: B") == 1
        # Vote patterns
        assert _parse_pairwise("Vote: A") == 0
        assert _parse_pairwise("Vote: B") == 1
        # Prefer patterns
        assert _parse_pairwise("Prefer: A") == 0
        assert _parse_pairwise("Prefer: B") == 1
        # Bare tokens
        assert _parse_pairwise("A") == 0
        assert _parse_pairwise("B") == 1
        assert _parse_pairwise("a") == 0
        assert _parse_pairwise("b") == 1

    def test_ambiguous_text_returns_tie(self) -> None:
        """Ambiguous or conflicting text should return -1."""
        # Conflicting signals
        assert _parse_pairwise("A and B are both good") == -1
        assert _parse_pairwise("I prefer A but B is also good") == -1
        # No clear verdict
        assert _parse_pairwise("They are similar") == -1
        assert _parse_pairwise("Cannot decide") == -1
        assert _parse_pairwise("") == -1
        assert _parse_pairwise("   ") == -1


class TestPairwiseCompareRegression:
    """Integration tests for `pairwise_compare` with the fixed parser."""

    def test_swapped_order_check_still_works(self) -> None:
        """The swap-debiasing should still work with the fixed parser."""
        # Judge always says A wins (0)
        judge_a = _FakeJudge(0)
        result = pairwise_compare("prompt", "resp_a", "resp_b", judge_a, swap=True)
        # First call: A vs B -> 0 (A wins)
        # Second call (swapped): B vs A -> 0 (A wins) but swapped means B wins -> disagreement -> -1
        assert result == -1

        # Judge always says first response wins
        class _FirstWins:
            def __init__(self):
                self.call = 0
            def compare_pair(self, prompt, resp_a, resp_b):
                self.call += 1
                return 0  # A wins
        judge_first = _FirstWins()
        result = pairwise_compare("p", "x", "y", judge_first, swap=True)
        # First: (x,y) -> 0 (A wins)
        # Second: (y,x) -> 0 (A wins) but swapped means B wins -> disagreement -> -1
        assert result == -1

    def test_agreement_across_swaps_returns_winner(self) -> None:
        """When both orders agree, winner is returned."""
        # Judge that prefers longer response
        class _LongerWins:
            def compare_pair(self, prompt, resp_a, resp_b):
                if len(resp_a) > len(resp_b):
                    return 0  # A wins
                if len(resp_b) > len(resp_a):
                    return 1  # B wins
                return -1  # Tie
        judge = _LongerWins()
        # A is longer -> first says 0 (A wins), second (swapped) says
        # 1 (B wins) -> but B is now the longer one -> agreement
        result = pairwise_compare("p", "longer", "short", judge, swap=True)
        assert result == 0  # longer (A) wins

        result = pairwise_compare("p", "short", "longer", judge, swap=True)
        assert result == 1  # longer (B) wins


class TestRegressionCase:
    """The specific regression case from the issue:
    Both replies favor base but produce a tuned win.
    """

    def test_both_favor_base_no_false_tuned_win(self) -> None:
        """When both swapped replies favor base, no tuned win should be reported."""
        # Simulate a judge that consistently says base is better
        class _BaseWins:
            def compare_pair(self, prompt, resp_a, resp_b):
                return 0  # Always first response
        _BaseWins()
        # With swap=True, first call (base, tuned) -> 0 (base wins)
        # Second call (tuned, base) -> 0 (tuned wins)
        # Disagreement -> -1 (tie), not a tuned win
        result = pairwise_compare("p", "base_resp", "tuned_resp", _BaseWins(), swap=True)
        assert result == -1  # Disagreement, not a tuned win

