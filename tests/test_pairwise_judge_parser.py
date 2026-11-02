"""Regression tests for the pairwise judge parser (#2356).

A pairwise judge reply is read as a verdict only when the letter sits where a
verdict would be. These tests pin the replies from the report and the spellings
the triage comment listed, and drive the real ``compare_pair`` path so a reply
that names no winner cannot produce a tuned win.
"""

from __future__ import annotations

import pytest

from soup_cli.eval.judge import JudgeEvaluator, _parse_pairwise, pairwise_winrate


@pytest.mark.parametrize(
    ("replies", "rate"),
    [
        # Both orders name the base response: no tuned win.
        (["Better response is A", "Answer: B"], 0.0),
        # The second reply names no winner, so there is no cross-checked verdict.
        (["Response B is better.", "It is a close call."], 0.5),
    ],
)
def test_pairwise_winrate_reads_real_replies(monkeypatch, replies, rate) -> None:
    judge = JudgeEvaluator(provider="ollama", model="m")
    answers = iter(replies)
    monkeypatch.setattr(judge, "_call_llm", lambda prompt: next(answers))
    assert pairwise_winrate([("p", "base", "tuned")], judge) == rate


@pytest.mark.parametrize(
    ("reply", "verdict"),
    [
        # JSON, including the nested-scores case the old object regex truncated.
        ('{"winner": "b"}', 1),
        ('{"winner": "A", "reason": "B misses the question"}', 0),
        ('{"winner": "B", "scores": {"A": 2, "B": 4}}', 1),
        ('```json\n{"winner": "B"}\n```', 1),
        ('{"winner": "tie", "reason": "a draw"}', -1),
        # A whole reply that is just the letter, with or without decoration.
        ("B", 1),
        ("B.", 1),
        ('"A"', 0),
        ("**B**", 1),
        # Labelled verdicts.
        ("Answer: B", 1),
        ("Winner: A", 0),
        ("The winner is B", 1),
        # The leading-letter form already pinned by tests/test_v07131.py.
        ("A is clearly better", 0),
        ("B wins here", 1),
        # Prose that names no winner, including the English article "a".
        ("This is a close call", -1),
        ("Neither response is a good one", -1),
        ("The answer is not A", -1),
        ("Banana", -1),
        ("Both responses are equally good.", -1),
        ("Response B is better, with a clear structure", -1),
        ("Winner: B. It is a clearer answer.", -1),
        ("", -1),
    ],
)
def test_pairwise_reply_spellings(reply, verdict) -> None:
    assert _parse_pairwise(reply) == verdict
