"""Langfuse feedback preserves falsy scores rather than treating them as absent."""

import pytest

from soup_cli.data.traces import build_pairs
from soup_cli.utils.ingest_sources import parse_langfuse


@pytest.mark.parametrize(
    ("feedback", "signal"),
    [
        ({"score": 0}, "thumbs_down"),
        ({"score": 0.0}, "thumbs_down"),
        ({"score": "0"}, "thumbs_down"),
        ({"score": -1}, "thumbs_down"),
        ({"rating": 0}, "thumbs_down"),
        ({"score": False}, "thumbs_down"),
        ({"score": 1}, "thumbs_up"),
        ({"score": True}, "thumbs_up"),
        ({"score": 0, "rating": "up"}, "thumbs_down"),
        ({"score": False, "rating": "up"}, "thumbs_down"),
        ({"score": 1, "rating": "down"}, "thumbs_up"),
        ({"rating": "down"}, "thumbs_down"),
        ({"score": None, "rating": "up"}, "thumbs_up"),
        ({"score": "", "rating": "up"}, "none"),
        ({}, "none"),
    ],
)
def test_feedback_score_takes_precedence_unless_none(feedback, signal):
    records = list(parse_langfuse([{"input": "question", "output": "answer", **feedback}]))
    assert len(records) == 1
    assert records[0].signal == signal


def test_langfuse_zero_score_can_supply_the_rejected_preference_answer():
    records = list(parse_langfuse([
        {"id": "good", "input": "same question", "output": "good answer", "score": 1},
        {"id": "bad", "input": "same question", "output": "bad answer", "score": 0},
    ]))
    pairs = list(build_pairs(records, signal="thumbs_up"))
    assert len(pairs) == 1
    assert pairs[0].chosen == "good answer"
    assert pairs[0].rejected == "bad answer"
