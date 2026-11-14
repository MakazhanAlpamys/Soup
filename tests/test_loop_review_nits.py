"""Three small defects from the python review of the loop branch.

* ``BucketStats.verdict`` compared floats: stable 1.0 against a canary 38 of 40 is exactly five
  points, which is OK at the default threshold, and was MAJOR (``1.0 - 0.95`` is
  ``0.05000000000000004``);
* ``strip_think`` cut the text at an index found in ``text.lower()``: U+0130 lower-cases to two
  code points, so the cut landed one character off (both copies, the gate and the paraphraser);
* the feedback reader skipped a rating of ``4.0`` that ``soup serve`` accepts and stores as 4.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

import soup_cli.utils.loop_feedback as lf
import soup_cli.utils.loop_paraphrase as lp
import soup_cli.utils.ship_loop_gate as gate
from soup_cli.utils.canary_router import BucketStats

T0 = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc).timestamp()


def _stats(stable_ok, stable_major, canary_ok, canary_major) -> BucketStats:
    stats = BucketStats()
    stats.stable_ok, stats.stable_major = stable_ok, stable_major
    stats.canary_ok, stats.canary_major = canary_ok, canary_major
    return stats


class TestVerdictBoundaryIsExact:
    def test_exactly_five_points_below_stable_is_ok(self):
        assert _stats(40, 0, 38, 2).verdict(min_samples=30) == "OK"

    def test_one_answer_more_failed_is_major(self):
        assert _stats(40, 0, 37, 3).verdict(min_samples=30) == "MAJOR"

    @pytest.mark.parametrize(
        ("stable", "canary", "threshold", "expected"),
        [
            ((100, 0), (95, 5), 0.05, "OK"),
            ((100, 0), (94, 6), 0.05, "MAJOR"),
            ((90, 10), (80, 20), 0.1, "OK"),
            ((90, 10), (79, 21), 0.1, "MAJOR"),
            ((0, 0), (30, 0), 0.05, "OK"),
        ],
    )
    def test_decimal_thresholds_are_read_as_written(self, stable, canary, threshold, expected):
        stats = _stats(*stable, *canary)

        assert stats.verdict(min_samples=30, regression_threshold=threshold) == expected


class TestStripThinkOffsets:
    @pytest.mark.parametrize("strip", [gate.strip_think, lp.strip_think])
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("\u0130</think>answer", "answer"),
            ("\u0130\u0130 thought</THINK>The answer", "The answer"),
            ("keep \u0130<think>cut off", "keep \u0130"),
            ("<think>x</think>\u0130 stays", "\u0130 stays"),
            ("a</Think>b</think>final", "final"),
            ("plain", "plain"),
            ("", ""),
        ],
    )
    def test_the_cut_lands_on_the_tag_not_one_character_off(self, strip, text, expected):
        assert strip(text) == expected


def _feedback_file(tmp_path, value, kind="rating") -> str:
    path = tmp_path / "feedback.jsonl"
    line = {
        "schema_version": 1, "id": "00000000-0000-4000-8000-000000000001",
        "response_id": "00000000-0000-4000-8000-000000000002",
        "ts": datetime.fromtimestamp(T0, timezone.utc).isoformat(), "kind": kind,
        "source": "playground", "value": value,
    }
    path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    return str(path)


class TestWholeFloatRatings:
    @pytest.mark.parametrize(("raw", "kept"), [(4.0, 4), (1.0, 1), (5.0, 5), (4, 4)])
    def test_a_whole_number_float_rating_is_read_as_the_integer(self, tmp_path, raw, kept):
        log = lf.read_feedback(_feedback_file(tmp_path, raw))

        assert log.stats.skipped == 0
        assert next(iter(log.by_response.values())).rating == kept

    @pytest.mark.parametrize("raw", [4.5, 0.0, 6.0, float("inf"), True, "4", None.__class__])
    def test_anything_else_is_still_skipped(self, tmp_path, raw):
        value = raw if raw is not None.__class__ else [4]

        log = lf.read_feedback(_feedback_file(tmp_path, value))

        assert log.stats.skipped == 1 and not log.by_response

    def test_a_whole_number_float_thumb_is_read_too(self, tmp_path):
        log = lf.read_feedback(_feedback_file(tmp_path, -1.0, kind="thumb"))

        assert log.stats.skipped == 0
        assert next(iter(log.by_response.values())).thumb == -1
