"""A canary is judged by what people say about its answers, not only by failed requests.

The error verdict (``BucketStats``) counts requests that failed. A canary whose answers
all arrive but are bad (people thumb them down, rate them 1-2, rewrite them) passed it.
The quality verdict reads the feedback log: for the answers of the active rollout (the
conversation record's ``canary_bucket`` is stable or canary, its ``adapter`` is that side's
adapter, and it was given after the rollout started, ``canary_started_ts`` in loop.yaml) it
takes the share of rated answers that got negative feedback (a thumb down, a rating of 1-2
or an edit) per bucket. With 30 rated answers on each side it is MAJOR when the canary's
share exceeds stable's by more than five percentage points AND a one-sided pooled
two-proportion z-test calls the gap significant (z > 1.645), else OK; with fewer it is
UNKNOWN. (How long a canary waits for those ratings: tests/test_loop_canary_max_wait.py.)

The two verdicts combine strictly: MAJOR if either is MAJOR, OK only when both are OK,
UNKNOWN otherwise. The quality verdict takes part only when there is a feedback log to read
(a loop without one judges by errors alone, as before) and the rollout recorded its start.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import replace
from datetime import datetime, timezone

import pytest
from typer.testing import CliRunner

import soup_cli.utils.loop_daemon as daemon
import soup_cli.utils.loop_feedback as lf
import soup_cli.utils.loop_stages as ls
from soup_cli.utils.canary_router import CanaryPolicy, QualityStats, record_bucket_outcome
from soup_cli.utils.loop_daemon import WatchConfig, combine_canary_verdicts, run_once
from soup_cli.utils.loop_state import LoopState, init_state, read_state, write_state

T0 = 1_700_000_000.0
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_LINES = iter(range(5000, 10**6))


def _clean(text: str) -> str:
    return " ".join(_ANSI.sub("", text).split())


def uid(number: int) -> str:
    return f"00000000-0000-4000-8000-{number:012d}"


def iso(offset: float) -> str:
    return datetime.fromtimestamp(T0 + offset, timezone.utc).isoformat()


STABLE = "v1"
CANARY = "v2"


def answer(number, *, ts=10, bucket="canary", adapter=CANARY, response="an answer"):
    """A conversation record (version 1) served in a canary bucket."""
    return {
        "schema_version": 1, "id": uid(number), "conversation_id": f"chat-{number}",
        "ts": iso(ts), "model": "base", "version": 1, "adapter": adapter,
        "canary_bucket": bucket, "messages": [{"role": "user", "content": f"q{number}"}],
        "response": response, "tokens": {"prompt": 1, "completion": 1}, "source": "soup-serve",
    }


def feedback(number, kind, ts=20, **fields):
    return {
        "schema_version": 1, "id": uid(next(_LINES)), "response_id": uid(number),
        "ts": iso(ts), "kind": kind, "source": "playground", **fields,
    }


def down(number, ts=20):
    return feedback(number, "thumb", ts, value=-1)


def up(number, ts=20):
    return feedback(number, "thumb", ts, value=1)


def write_lines(path, lines):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(line) + "\n" for line in lines))
    return path


TRACES = os.path.join(".soup-loops", "traces", "conversations.jsonl")
FEEDBACK = os.path.join(".soup-loops", "feedback.jsonl")


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    for name in (
        "SOUP_LOOP_TRACE_DIR", "SOUP_LOOP_THUMBS_DB", "SOUP_LOOP_SERVE_ENDPOINT",
        "SOUP_LOOP_FEEDBACK_LOG",
    ):
        monkeypatch.delenv(name, raising=False)
    for name in ("_DEPLOY_POSTER", "_LOAD_POSTER", "_TRACE_DIR_RESOLVER"):
        monkeypatch.setattr(ls, name, None)
    (tmp_path / "tasks.jsonl").write_text('{"prompt": "2+2", "expected": "4"}\n', "utf-8")
    return tmp_path


def stats(canary_rated, canary_negative, stable_rated, stable_negative):
    return QualityStats(
        stable_rated=stable_rated, stable_negative=stable_negative,
        canary_rated=canary_rated, canary_negative=canary_negative,
    )


# ---------------------------------------------------------------------------
# The quality verdict
# ---------------------------------------------------------------------------


class TestVerdict:
    def test_fewer_than_thirty_rated_canary_answers_are_unknown(self):
        assert stats(29, 29, 100, 0).verdict() == "UNKNOWN"
        assert stats(0, 0, 100, 0).verdict() == "UNKNOWN"

    def test_thirty_rated_canary_answers_are_enough(self):
        assert stats(30, 30, 100, 0).verdict() == "MAJOR"
        assert stats(30, 0, 100, 0).verdict() == "OK"

    def test_a_comparison_needs_thirty_rated_answers_on_both_sides(self):
        assert stats(10, 10, 500, 0).verdict() == "UNKNOWN"   # canary side short
        assert stats(100, 100, 29, 0).verdict() == "UNKNOWN"  # stable side short
        assert stats(100, 100, 0, 0).verdict() == "UNKNOWN"   # nothing to hold it against
        assert stats(30, 30, 30, 0).verdict() == "MAJOR"      # thirty each: enough
        assert stats(30, 0, 30, 0).verdict() == "OK"

    def test_a_canary_no_worse_than_stable_is_ok(self):
        assert stats(40, 4, 40, 4).verdict() == "OK"
        assert stats(40, 2, 40, 8).verdict() == "OK"  # better than stable

    def test_exactly_five_points_worse_is_still_ok(self):
        # 21% vs 16%: five points, to the digit (floats say 0.05000000000000002).
        assert stats(100, 21, 100, 16).verdict() == "OK"
        assert stats(40, 8, 40, 6).verdict() == "OK"      # 20% vs 15%
        assert stats(30, 6, 40, 6).verdict() == "OK"      # 20% vs 15% with other sizes

    def test_a_clearly_worse_canary_is_major(self):
        assert stats(100, 30, 100, 15).verdict() == "MAJOR"   # 30% vs 15%, z 2.54
        assert stats(100, 25, 100, 15).verdict() == "MAJOR"   # 25% vs 15%, z 1.77
        assert stats(200, 45, 200, 32).verdict() == "MAJOR"   # 22.5% vs 16%, z 1.649

    def test_more_than_five_points_worse_is_not_major_without_significance(self):
        """These were MAJOR before the z-test; a gap this size is common by chance."""
        assert stats(40, 9, 40, 6).verdict() == "OK"      # 22.5% vs 15%, z 0.86
        assert stats(30, 7, 40, 6).verdict() == "OK"      # 23.3% vs 15%, z 0.89
        assert stats(100, 22, 100, 16).verdict() == "OK"  # 22% vs 16%, z 1.08
        assert stats(100, 24, 100, 15).verdict() == "OK"  # 24% vs 15%, z 1.61
        assert stats(200, 44, 200, 32).verdict() == "OK"  # 22% vs 16%, z 1.53

    def test_the_significance_threshold_is_one_point_six_four_five(self):
        from soup_cli.utils.canary_router import QUALITY_Z_THRESHOLD

        assert QUALITY_Z_THRESHOLD == 1.645
        # z 1.645026: over the threshold by 0.00003, so it is MAJOR although z shows as 1.645
        over = stats(106, 7, 39, 0)
        assert over.evidence()["z"] == 1.645
        assert over.verdict() == "MAJOR"
        # z 1.644992: under it, and also shows as 1.645
        under = stats(119, 17, 30, 1)
        assert under.evidence()["z"] == 1.645
        assert under.verdict() == "OK"

    def test_a_significant_gap_of_five_points_or_less_is_still_ok(self):
        """The tolerance stays: a tiny gap measured on many answers is not a regression."""
        assert stats(1000, 140, 1000, 100).verdict() == "OK"   # 14% vs 10%, z 2.75
        # exactly five points on a thousand answers each (z 2.88): not MORE than five
        assert stats(1000, 210, 1000, 160).verdict() == "OK"
        assert stats(1000, 211, 1000, 160).verdict() == "MAJOR"
        # 19.6% vs 14.6% is five points to the digit (z 2.97), but 0.196 - 0.146 is
        # 0.05000000000000002 in floats: the comparison must be exact
        assert stats(1000, 196, 1000, 146).verdict() == "OK"
        assert stats(1000, 197, 1000, 146).verdict() == "MAJOR"

    def test_a_canary_with_no_variance_in_the_ratings_is_never_major(self):
        """Every answer negative, or none: the test statistic is undefined (se is 0)."""
        assert stats(30, 0, 30, 0).verdict() == "OK"
        assert stats(30, 30, 30, 30).verdict() == "OK"
        assert stats(500, 500, 30, 30).verdict() == "OK"
        assert stats(30, 30, 30, 30).verdict(regression_threshold=0.0) == "OK"
        assert stats(30, 0, 30, 0).evidence()["z"] is None
        assert stats(30, 30, 30, 30).evidence()["z"] is None

    def test_the_z_test_needs_a_gap_in_the_bad_direction(self):
        assert stats(100, 5, 100, 30).verdict() == "OK"       # canary much better
        assert stats(100, 5, 100, 30).evidence()["z"] < 0
        assert stats(100, 5, 100, 30).evidence()["p_value"] > 0.99

    def test_the_thresholds_are_parameters_checked_like_the_error_verdict(self):
        assert stats(10, 10, 10, 0).verdict(min_samples=10) == "MAJOR"
        assert stats(30, 12, 30, 0).verdict(regression_threshold=0.5) == "OK"
        for bad in (0, -1, True, 2.5):
            with pytest.raises(ValueError, match="min_samples"):
                stats(30, 0, 30, 0).verdict(min_samples=bad)
        for bad in (-0.1, 1.5, float("nan"), True, "x"):
            with pytest.raises(ValueError, match="regression_threshold"):
                stats(30, 0, 30, 0).verdict(regression_threshold=bad)

    def test_the_counts_are_checked(self):
        with pytest.raises(ValueError, match="negative"):
            QualityStats(canary_rated=3, canary_negative=4)
        with pytest.raises(ValueError, match="canary_rated"):
            QualityStats(canary_rated=-1)
        with pytest.raises(ValueError, match="stable_rated"):
            QualityStats(stable_rated=True)

    def test_the_summary_and_the_progress_are_what_the_iteration_notes_carry(self):
        assert stats(31, 3, 40, 2).summary() == "3/31 negative vs 2/40 stable"
        assert stats(12, 0, 40, 0).progress() == "12/30 canary, 40/30 stable rated"
        assert stats(12, 0, 40, 0).progress(min_samples=10) == "12/10 canary, 40/10 stable rated"


class TestEvidence:
    """The numbers behind a verdict, in the field names of the iteration format."""

    FIELDS = {
        "verdict", "rated", "negative", "share", "difference", "z", "p_value", "min_rated",
        "threshold", "timed_out", "decided_on",
    }

    def test_a_gap_over_five_points_that_is_not_significant(self):
        # (canary 40 rated / 9 negative) vs (stable 40 rated / 6 negative): was MAJOR
        assert stats(40, 9, 40, 6).evidence() == {
            "verdict": "OK",
            "rated": {"stable": 40, "canary": 40},
            "negative": {"stable": 6, "canary": 9},
            "share": {"stable": 0.15, "canary": 0.225},
            "difference": 0.075,
            "z": 0.8593,
            "p_value": 0.1951,
            "min_rated": 30,
            "threshold": 0.05,
            "timed_out": False,
            "decided_on": "errors_and_quality",
        }

    def test_a_significant_regression(self):
        # (canary 100 rated / 30 negative) vs (stable 100 rated / 15 negative)
        assert stats(100, 30, 100, 15).evidence() == {
            "verdict": "MAJOR",
            "rated": {"stable": 100, "canary": 100},
            "negative": {"stable": 15, "canary": 30},
            "share": {"stable": 0.15, "canary": 0.3},
            "difference": 0.15,
            "z": 2.54,
            "p_value": 0.0055,
            "min_rated": 30,
            "threshold": 0.05,
            "timed_out": False,
            "decided_on": "errors_and_quality",
        }

    def test_nothing_rated_yet_has_no_shares_and_no_test(self):
        assert stats(0, 0, 0, 0).evidence() == {
            "verdict": "UNKNOWN",
            "rated": {"stable": 0, "canary": 0},
            "negative": {"stable": 0, "canary": 0},
            "share": {"stable": None, "canary": None},
            "difference": None,
            "z": None,
            "p_value": None,
            "min_rated": 30,
            "threshold": 0.05,
            "timed_out": False,
            "decided_on": "errors_and_quality",
        }

    def test_one_side_without_ratings_has_a_share_for_the_other_only(self):
        found = stats(12, 3, 0, 0).evidence()

        assert found["share"] == {"stable": None, "canary": 0.25}
        assert found["difference"] is None and found["z"] is None and found["p_value"] is None
        assert found["verdict"] == "UNKNOWN"

    def test_the_fields_are_exactly_those_of_the_iteration_format(self):
        for found in (stats(40, 9, 40, 6), stats(0, 0, 0, 0), stats(30, 30, 30, 30)):
            assert set(found.evidence()) == self.FIELDS

    def test_it_is_plain_json_even_where_the_test_is_undefined(self):
        for found in (stats(40, 9, 40, 6), stats(0, 0, 0, 0), stats(30, 30, 30, 30)):
            text = json.dumps(found.evidence(), allow_nan=False)
            assert json.loads(text) == found.evidence()

    def test_the_p_value_is_the_one_sided_tail_of_z(self):
        import math

        for args in ((40, 9, 40, 6), (100, 30, 100, 15), (100, 5, 100, 30), (50, 20, 70, 30)):
            found = stats(*args).evidence()
            assert found["p_value"] == pytest.approx(
                0.5 * math.erfc(found["z"] / math.sqrt(2)), abs=1e-4
            )

    def test_z_is_the_pooled_two_proportion_statistic_positive_when_the_canary_is_worse(self):
        import math

        # canary 18/60, stable 9/40: pooled p = 27/100, se = sqrt(p (1-p) (1/60 + 1/40))
        pooled = 27 / 100
        se = math.sqrt(pooled * (1 - pooled) * (1 / 60 + 1 / 40))
        expected = (18 / 60 - 9 / 40) / se

        assert stats(60, 18, 40, 9).evidence()["z"] == pytest.approx(expected, abs=1e-4)
        assert expected > 0
        assert stats(40, 9, 60, 18).evidence()["z"] == pytest.approx(
            (9 / 40 - 18 / 60) / se, abs=1e-4
        )

    def test_the_difference_is_canary_minus_stable(self):
        assert stats(40, 4, 40, 8).evidence()["difference"] == -0.1
        assert stats(40, 8, 40, 4).evidence()["difference"] == 0.1

    def test_shares_and_statistics_are_rounded_for_display_to_four_places(self):
        found = stats(3, 1, 7, 1).evidence(min_samples=1)

        assert found["share"] == {"stable": 0.1429, "canary": 0.3333}
        assert found["difference"] == 0.1905

    def test_a_negative_zero_does_not_leak_into_the_record(self):
        found = stats(20001, 1, 20000, 1).evidence()  # a gap of -0.0000000025

        assert str(found["difference"]) == "0.0"
        assert str(found["z"]) == "0.0"
        assert "-0.0" not in json.dumps(found)

    def test_it_echoes_the_minimum_and_the_threshold_it_was_asked_with(self):
        found = stats(12, 0, 12, 0).evidence(min_samples=10, regression_threshold=0.1)

        assert found["min_rated"] == 10 and found["threshold"] == 0.1
        assert found["verdict"] == "OK"

    def test_a_verdict_can_be_given_for_a_wait_that_ran_out(self):
        found = stats(7, 0, 40, 0).evidence(
            verdict="INSUFFICIENT", timed_out=True, decided_on="errors_only"
        )

        assert found["verdict"] == "INSUFFICIENT"
        assert found["timed_out"] is True and found["decided_on"] == "errors_only"
        assert found["rated"] == {"stable": 40, "canary": 7}

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"verdict": "GOOD"}, "verdict"),
            ({"verdict": None, "decided_on": "quality_only"}, "decided_on"),
            ({"timed_out": 1}, "timed_out"),
            ({"timed_out": None}, "timed_out"),
            ({"min_samples": 0}, "min_samples"),
            ({"min_samples": True}, "min_samples"),
            ({"regression_threshold": 1.5}, "regression_threshold"),
            ({"regression_threshold": float("nan")}, "regression_threshold"),
        ],
    )
    def test_bad_arguments_are_refused_naming_the_argument(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            stats(40, 9, 40, 6).evidence(**kwargs)

    def test_the_shortfall_names_the_sides_that_are_short(self):
        assert stats(7, 0, 40, 0).shortfall() == "7/30 canary rated"
        assert stats(40, 0, 5, 0).shortfall() == "5/30 stable rated"
        assert stats(7, 0, 5, 0).shortfall() == "7/30 canary, 5/30 stable rated"
        assert stats(7, 0, 5, 0).shortfall(min_samples=10) == "7/10 canary, 5/10 stable rated"
        assert stats(0, 0, 0, 0).shortfall() == "0/30 canary, 0/30 stable rated"

    def test_the_shortfall_of_a_side_that_is_enough_is_not_listed(self):
        assert "stable" not in stats(7, 0, 30, 0).shortfall()
        assert "canary" not in stats(30, 0, 7, 0).shortfall()


class TestCombination:
    @pytest.mark.parametrize(
        ("error", "quality", "expected"),
        [
            ("OK", "INSUFFICIENT", "OK"),            # the ratings never came: errors decide
            ("MAJOR", "INSUFFICIENT", "MAJOR"),
            ("UNKNOWN", "INSUFFICIENT", "UNKNOWN"),
            ("OK", "OK", "OK"),
            ("OK", "MAJOR", "MAJOR"),
            ("MAJOR", "OK", "MAJOR"),
            ("MAJOR", "MAJOR", "MAJOR"),
            ("MAJOR", "UNKNOWN", "MAJOR"),
            ("UNKNOWN", "MAJOR", "MAJOR"),
            ("OK", "UNKNOWN", "UNKNOWN"),
            ("UNKNOWN", "OK", "UNKNOWN"),
            ("UNKNOWN", "UNKNOWN", "UNKNOWN"),
        ],
    )
    def test_major_if_either_ok_only_if_both(self, error, quality, expected):
        assert combine_canary_verdicts(error, quality) == expected


# ---------------------------------------------------------------------------
# Reading the quality stats from the logs
# ---------------------------------------------------------------------------


def read(*, started=0.0, stable=STABLE, canary=CANARY, stable_is_base=False,
         trace_dir=os.path.join(".soup-loops", "traces"), feedback_path=FEEDBACK):
    return lf.read_quality_stats(
        trace_dir, feedback_path, stable=stable, canary=canary,
        started_ts=T0 + started, stable_is_base=stable_is_base,
    )


def counts(found):
    return (
        found.canary_rated, found.canary_negative, found.stable_rated, found.stable_negative
    )


def logs(answers, notes):
    write_lines(TRACES, answers)
    write_lines(FEEDBACK, notes)


class TestReading:
    def test_negative_feedback_is_a_thumb_down_a_low_rating_or_an_edit(self, cwd):
        logs(
            [answer(n) for n in range(1, 9)],
            [
                down(1),
                feedback(2, "rating", value=1),
                feedback(3, "rating", value=2),
                feedback(4, "edit", edited_text="a better answer"),
                up(5),
                feedback(6, "rating", value=3),
                feedback(7, "rating", value=4),
                feedback(8, "rating", value=5),
            ],
        )

        assert counts(read()) == (8, 4, 0, 0)

    def test_an_answer_counts_once_however_much_feedback_it_got(self, cwd):
        logs(
            [answer(1)],
            [down(1), feedback(1, "rating", value=1), feedback(1, "edit", edited_text="x"),
             feedback(1, "comment", comment="bad")],
        )

        assert counts(read()) == (1, 1, 0, 0)

    def test_one_negative_signal_makes_the_answer_negative(self, cwd):
        logs([answer(1)], [up(1), feedback(1, "edit", edited_text="a rewrite")])

        assert counts(read()) == (1, 1, 0, 0)

    def test_the_latest_line_per_kind_decides_and_a_clear_removes_it(self, cwd):
        logs(
            [answer(1), answer(2), answer(3)],
            [
                up(1, 10), down(1, 11),            # ends as a thumb down
                down(2, 10), up(2, 11),            # ends as a thumb up
                down(3, 10), feedback(3, "thumb", 11, value=None),  # cleared: not rated
            ],
        )

        assert counts(read()) == (2, 1, 0, 0)

    def test_an_answer_with_only_a_comment_or_an_unchanged_edit_is_not_rated(self, cwd):
        logs(
            [answer(1), answer(2), answer(3)],
            [
                feedback(1, "comment", comment="hmm"),
                feedback(2, "edit", edited_text="an answer"),    # the answer as given
                feedback(3, "edit", edited_text="  an answer \n"),
            ],
        )

        assert counts(read()) == (0, 0, 0, 0)

    def test_the_two_buckets_are_counted_apart(self, cwd):
        logs(
            [answer(1), answer(2), answer(3, bucket="stable", adapter=STABLE),
             answer(4, bucket="stable", adapter=STABLE), answer(5, bucket="stable",
                                                               adapter=STABLE)],
            [down(1), up(2), down(3), up(4), up(5)],
        )

        assert counts(read()) == (2, 1, 3, 1)

    def test_answers_from_before_the_rollout_started_do_not_count(self, cwd):
        logs(
            [answer(1, ts=5), answer(2, ts=10), answer(3, ts=11),
             answer(4, ts=9, bucket="stable", adapter=STABLE),
             answer(5, ts=12, bucket="stable", adapter=STABLE)],
            [down(1), down(2), down(3), down(4), down(5)],
        )

        # The rollout started at +10: an answer at +10 is in, +9 and +5 are out.
        assert counts(read(started=10)) == (2, 2, 1, 1)

    def test_an_answer_without_a_usable_ts_cannot_be_placed_after_the_start(self, cwd):
        line = answer(1)
        line["ts"] = "yesterday"
        logs([line, answer(2)], [down(1), down(2)])

        assert counts(read()) == (1, 1, 0, 0)

    def test_an_answer_outside_a_canary_split_does_not_count(self, cwd):
        logs([answer(1, bucket=None), answer(2, bucket=None, adapter=None)], [down(1), down(2)])

        assert counts(read()) == (0, 0, 0, 0)

    def test_the_canary_bucket_counts_only_the_active_canary_adapter(self, cwd):
        logs(
            [answer(1, adapter="v9"), answer(2, adapter=None), answer(3, adapter=STABLE),
             answer(4)],
            [down(1), down(2), down(3), down(4)],
        )

        assert counts(read()) == (1, 1, 0, 0)

    def test_the_stable_bucket_counts_only_the_stable_adapter(self, cwd):
        logs(
            [answer(1, bucket="stable", adapter="v9"),
             answer(2, bucket="stable", adapter=None),
             answer(3, bucket="stable", adapter=CANARY),
             answer(4, bucket="stable", adapter=STABLE)],
            [down(1), down(2), down(3), down(4)],
        )

        assert counts(read()) == (0, 0, 1, 1)

    def test_when_the_base_model_is_stable_its_answers_carry_no_adapter(self, cwd):
        logs(
            [answer(1, bucket="stable", adapter=None),
             answer(2, bucket="stable", adapter="./base"),
             answer(3, bucket="stable", adapter="v9")],
            [down(1), down(2), down(3)],
        )

        assert counts(read(stable="./base", stable_is_base=True)) == (0, 0, 2, 2)
        assert counts(read(stable="./base", stable_is_base=False)) == (0, 0, 1, 1)

    def test_feedback_about_unknown_answers_and_version_zero_lines_is_ignored(self, cwd):
        legacy = {"ts": T0 + 10, "prompt": "q", "response": "r"}
        logs([answer(1), legacy], [down(1), down(99)])

        assert counts(read()) == (1, 1, 0, 0)

    def test_a_missing_log_reads_as_nothing(self, cwd):
        found = read(trace_dir=os.path.join("nowhere", "traces"),
                     feedback_path=os.path.join("nowhere", "feedback.jsonl"))

        assert counts(found) == (0, 0, 0, 0)

    def test_malformed_lines_are_skipped(self, cwd):
        write_lines(TRACES, [answer(1), "junk", {"schema_version": 3}])
        write_lines(FEEDBACK, [down(1), "junk", {"kind": "thumb"}])

        assert counts(read()) == (1, 1, 0, 0)


# ---------------------------------------------------------------------------
# The stage: when does the quality verdict take part?
# ---------------------------------------------------------------------------


def _state(**kwargs) -> LoopState:
    kwargs.setdefault("served_model", "./base")
    kwargs.setdefault("base_model", "org/base")
    kwargs.setdefault("eval_suite", "tasks.jsonl")
    kwargs.setdefault("verify", "ship")
    kwargs.setdefault("status", "running")
    return LoopState(baseline="b", **kwargs)


def _canary_state(**kwargs) -> LoopState:
    kwargs.setdefault("canary_started_ts", T0)
    kwargs.setdefault("production_name", STABLE)
    kwargs.setdefault("production_adapter", "adapters/v1")
    return _state(
        canary_active=CANARY, canary_traffic_pct=5.0, canary_rollout_id="r1",
        canary_adapter="adapters/v2", **kwargs,
    )


class TestStage:
    def test_it_reads_the_logs_of_the_rollout(self, cwd):
        logs([answer(1), answer(2, bucket="stable", adapter=STABLE)], [down(1), up(2)])

        found = ls.canary_quality(_canary_state())

        assert counts(found) == (1, 1, 1, 0)

    def test_without_a_feedback_log_the_quality_verdict_does_not_take_part(self, cwd):
        write_lines(TRACES, [answer(1)])

        assert ls.canary_quality(_canary_state()) is None

    def test_a_rollout_that_recorded_no_start_is_judged_by_errors_alone(self, cwd):
        logs([answer(1)], [down(1)])

        assert ls.canary_quality(_canary_state(canary_started_ts=None)) is None

    def test_without_a_canary_there_is_nothing_to_judge(self, cwd):
        logs([answer(1)], [down(1)])

        assert ls.canary_quality(_state()) is None

    def test_it_follows_the_configured_places(self, cwd, monkeypatch):
        write_lines(os.path.join("logs", "c.jsonl"), [answer(1)])
        write_lines("fb.jsonl", [down(1)])

        found = ls.canary_quality(_canary_state(trace_dir="logs", feedback_log="fb.jsonl"))
        assert counts(found) == (1, 1, 0, 0)

        write_lines("env.jsonl", [up(1)])
        monkeypatch.setenv("SOUP_LOOP_FEEDBACK_LOG", "env.jsonl")
        found = ls.canary_quality(_canary_state(trace_dir="logs", feedback_log="fb.jsonl"))
        assert counts(found) == (1, 0, 0, 0)

    def test_the_base_model_is_the_stable_side_until_something_is_promoted(self, cwd):
        logs(
            [answer(1, bucket="stable", adapter=None), answer(2)], [down(1), down(2)]
        )
        state = _canary_state(production_name=None, production_adapter=None)

        assert counts(ls.canary_quality(state)) == (1, 1, 1, 1)

    def test_the_pre_wired_watch_config_judges_quality(self):
        assert ls.build_prewired_watch_config().quality_fn is ls.canary_quality

    def test_a_plain_watch_config_does_not(self):
        assert WatchConfig().quality_fn is None


# ---------------------------------------------------------------------------
# The daemon: both verdicts, one decision
# ---------------------------------------------------------------------------


def _record(state, ok: int, bad: int, stable=(40, 0)):
    policy = CanaryPolicy(stable=state.stable_name(), canary=state.canary_active, traffic_pct=5.0)
    path = os.path.join(".soup", "canary-stats.json")
    for bucket, (good, failed) in {"stable": stable, "canary": (ok, bad)}.items():
        for _ in range(good):
            record_bucket_outcome(policy, bucket, True, rollout_id="r1", path=path)
        for _ in range(failed):
            record_bucket_outcome(policy, bucket, False, rollout_id="r1", path=path)


def _config(quality, **kwargs):
    kwargs.setdefault("harvest_fn", lambda s: {"pairs_harvested": 0})
    kwargs.setdefault("train_fn", lambda s, c: {"run_id": None, "skipped": True})
    kwargs.setdefault("gate_fn", lambda s, c: {"gate_verdict": "SKIPPED"})
    kwargs.setdefault("deploy_fn", lambda s, c: {"deployed": False, "canary_verdict": None})
    kwargs.setdefault("cost_fn", lambda s: 0.0)
    kwargs.setdefault("promote_fn", lambda s: {"promoted": True})
    return WatchConfig(quality_fn=quality, **kwargs)


class TestDaemon:
    @pytest.fixture(autouse=True)
    def _clock(self, monkeypatch):
        """A minute after the rollout started: nobody has waited long enough to give up."""
        monkeypatch.setattr(
            daemon, "_now", lambda: datetime.fromtimestamp(T0 + 60, timezone.utc)
        )

    def test_bad_feedback_withdraws_a_canary_whose_requests_all_succeeded(self, cwd):
        state = _canary_state()
        _record(state, ok=30, bad=0)

        new_state, record, _ = run_once(state, _config(lambda s: stats(31, 12, 40, 2)))

        assert new_state.canary_active is None and new_state.canary_adapter is None
        assert new_state.canary_started_ts is None
        assert new_state.production_name == STABLE  # production untouched
        assert record.rolled_back is True and record.canary_verdict == "MAJOR"
        assert "canary v2 rolled back (MAJOR)" in record.notes
        assert "canary errors: OK" in record.notes
        assert "canary quality: 12/31 negative vs 2/40 stable (MAJOR)" in record.notes

    def test_failed_requests_withdraw_it_even_when_people_like_the_answers(self, cwd):
        state = _canary_state()
        _record(state, ok=20, bad=10)

        new_state, record, _ = run_once(state, _config(lambda s: stats(40, 0, 40, 2)))

        assert new_state.canary_active is None
        assert record.rolled_back is True
        assert "canary errors: MAJOR" in record.notes
        assert "canary quality: 0/40 negative vs 2/40 stable (OK)" in record.notes

    def test_a_canary_that_is_fine_both_ways_is_promoted(self, cwd):
        state = _canary_state()
        _record(state, ok=30, bad=0)
        told = []
        config = _config(
            lambda s: stats(31, 3, 40, 2),
            promote_fn=lambda s: told.append(s.canary_active) or {"promoted": True},
        )

        new_state, record, _ = run_once(state, config)

        assert told == [CANARY]
        assert new_state.production_name == CANARY
        assert new_state.canary_active is None and new_state.canary_started_ts is None
        assert record.canary_verdict == "OK"
        assert "canary v2 promoted to production" in record.notes
        assert "canary quality: 3/31 negative vs 2/40 stable (OK)" in record.notes

    def test_too_little_feedback_keeps_the_canary_running(self, cwd):
        state = _canary_state()
        _record(state, ok=31, bad=0)
        told = []
        config = _config(
            lambda s: stats(12, 0, 5, 0),
            promote_fn=lambda s: told.append(1) or {"promoted": True},
        )

        new_state, record, decision = run_once(state, config)

        assert told == []
        assert decision.proceed is False
        assert new_state.canary_active == CANARY
        assert new_state.canary_started_ts == T0
        assert record.canary_verdict == "UNKNOWN"
        assert record.notes == (
            "canary-wait: v2 has 31 canary answers (verdict at 30) | canary errors: OK | "
            "canary quality: 12/30 canary, 5/30 stable rated (UNKNOWN)"
        )

    def test_a_canary_that_looks_bad_waits_while_the_stable_side_is_unrated(self, cwd):
        """With no stable ratings there is nothing to compare with: no withdrawal on a guess."""
        state = _canary_state()
        _record(state, ok=30, bad=0)

        new_state, record, _ = run_once(state, _config(lambda s: stats(40, 40, 0, 0)))

        assert new_state.canary_active == CANARY
        assert record.rolled_back is False and record.canary_verdict == "UNKNOWN"
        assert "canary quality: 40/30 canary, 0/30 stable rated (UNKNOWN)" in record.notes

    def test_too_few_answers_with_good_feedback_keep_it_running_too(self, cwd):
        state = _canary_state()
        _record(state, ok=10, bad=0)

        new_state, record, _ = run_once(state, _config(lambda s: stats(40, 0, 40, 0)))

        assert new_state.canary_active == CANARY
        assert record.canary_verdict == "UNKNOWN"
        assert "canary errors: UNKNOWN" in record.notes
        assert "canary quality: 0/40 negative vs 0/40 stable (OK)" in record.notes

    def test_a_major_quality_verdict_decides_before_the_error_stats_are_enough(self, cwd):
        state = _canary_state()
        _record(state, ok=5, bad=0)

        new_state, record, _ = run_once(state, _config(lambda s: stats(30, 30, 40, 0)))

        assert new_state.canary_active is None
        assert record.rolled_back is True

    def test_without_the_hook_or_without_data_the_decision_is_the_error_verdict(self, cwd):
        for quality in (None, lambda s: None):
            state = _canary_state()
            _record(state, ok=30, bad=0)

            new_state, record, _ = run_once(state, _config(quality))

            assert new_state.production_name == CANARY
            assert "canary quality" not in record.notes
            assert "canary errors" not in record.notes
            os.remove(os.path.join(".soup", "canary-stats.json"))

    def test_the_wait_note_without_quality_is_unchanged(self, cwd):
        state = _canary_state()
        _record(state, ok=12, bad=0)

        _, record, _ = run_once(state, _config(None))

        assert record.notes == "canary-wait: v2 has 12 canary answers (verdict at 30)"

    def test_a_major_quality_verdict_withdraws_nothing_when_autoroll_is_off(self, cwd):
        state = _canary_state(canary_autoroll_on_regress=False)
        _record(state, ok=30, bad=0)

        new_state, record, decision = run_once(state, _config(lambda s: stats(31, 31, 40, 0)))

        assert new_state.canary_active == CANARY
        assert record.rolled_back is False
        assert decision.proceed is False

    def test_a_failing_quality_read_does_not_stop_the_loop(self, cwd, caplog):
        state = _canary_state()
        _record(state, ok=30, bad=0)

        def broken(_state):
            raise OSError("disk gone")

        with caplog.at_level(logging.WARNING):
            new_state, record, _ = run_once(state, _config(broken))

        assert new_state.production_name == CANARY  # judged by errors alone this cycle
        assert "canary quality" not in record.notes
        assert any("quality" in message for message in caplog.messages)

    def test_a_deployed_canary_records_when_it_started(self, cwd):
        deploy_out = {
            "deployed": True, "canary_verdict": None, "canary_active": CANARY,
            "canary_adapter": "adapters/v2", "canary_traffic_pct": 5.0,
            "canary_rollout_id": "r9", "canary_started_ts": T0 + 123.5,
        }
        config = _config(
            None,
            harvest_fn=lambda s: {"pairs_harvested": 1},
            train_fn=lambda s, c: {"run_id": "r", "skipped": False, "adapter_path": "a"},
            gate_fn=lambda s, c: {"gate_verdict": "OK"},
            deploy_fn=lambda s, c: deploy_out,
        )

        state, record, _ = run_once(_state(), config)

        assert record.shipped is True
        assert state.canary_started_ts == T0 + 123.5

    def test_a_deploy_that_gives_no_start_gets_the_time_of_the_cycle(self, cwd):
        deploy_out = {
            "deployed": True, "canary_verdict": None, "canary_active": CANARY,
            "canary_adapter": "adapters/v2", "canary_traffic_pct": 5.0,
        }
        config = _config(
            None,
            harvest_fn=lambda s: {"pairs_harvested": 1},
            train_fn=lambda s, c: {"run_id": "r", "skipped": False, "adapter_path": "a"},
            gate_fn=lambda s, c: {"gate_verdict": "OK"},
            deploy_fn=lambda s, c: deploy_out,
        )
        before = datetime.now(timezone.utc).timestamp()

        state, _, _ = run_once(_state(), config)

        assert before <= state.canary_started_ts <= datetime.now(timezone.utc).timestamp()

    def test_the_deploy_stage_stamps_the_start(self, cwd, monkeypatch):
        monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", "http://127.0.0.1:8000")
        monkeypatch.setattr(ls, "_LOAD_POSTER", lambda endpoint, name, path: (200, ""))
        os.makedirs("adapters")
        before = time.time()

        out = ls.deploy_to_canary(
            _state(), {"gate_verdict": "OK", "adapter_path": os.path.join("adapters", "loop-x")}
        )

        assert out["deployed"] is True
        assert before <= out["canary_started_ts"] <= time.time()


# ---------------------------------------------------------------------------
# LoopState / CLI
# ---------------------------------------------------------------------------


def _cli(*args):
    from soup_cli.commands.loop import app

    result = CliRunner().invoke(app, list(args))
    return result, _clean(result.output)


_INIT = ("init", "./base", "--eval", "tasks.jsonl", "--baseline", "b", "--base-model", "org/base")


class TestState:
    def test_the_field_follows_train_method_and_defaults_to_none(self):
        """Appended last when added; the fields added after it are in test_loop_canary_max_wait."""
        names = list(LoopState.__dataclass_fields__)

        assert names.index("canary_started_ts") == names.index("train_method") + 1
        assert _state().canary_started_ts is None

    @pytest.mark.parametrize("value", [0, 0.0, 1_700_000_000.25])
    def test_it_is_epoch_seconds(self, value):
        assert _state(canary_started_ts=value).canary_started_ts == value

    @pytest.mark.parametrize(
        "value", [True, "5", [1], -1, float("nan"), float("inf"), 1e300]
    )
    def test_a_bad_value_is_refused_naming_the_field(self, value):
        with pytest.raises(ValueError, match="canary_started_ts"):
            _state(canary_started_ts=value)

    def test_it_round_trips_and_an_old_file_loads(self, cwd):
        state, path = init_state("m", "e", "b", path=str(cwd / "loop.json"))
        assert state.canary_started_ts is None
        write_state(replace(state, canary_started_ts=T0 + 0.5), path)

        assert read_state(path).canary_started_ts == T0 + 0.5

        legacy = {"served_model": "m", "eval_suite": "e", "baseline": "b",
                  "canary_active": "v2", "canary_traffic_pct": 5.0}
        (cwd / "legacy.json").write_text(json.dumps(legacy), "utf-8")
        assert read_state(str(cwd / "legacy.json")).canary_started_ts is None

    def test_a_manual_canary_records_its_start(self, cwd):
        _cli(*_INIT)
        before = time.time()

        result, _ = _cli("canary", "candidate", "--traffic", "10%")
        assert result.exit_code == 0, (result.output, repr(result.exception))

        started = read_state().canary_started_ts
        assert before <= started <= time.time()

    def test_rolling_back_clears_it(self, cwd):
        _cli(*_INIT)
        state = read_state()
        write_state(replace(
            state, production_name="v1", previous_production_name=None,
            canary_active="v2", canary_traffic_pct=5.0, canary_started_ts=T0,
        ))

        result, _ = _cli("rollback")
        assert result.exit_code == 0, (result.output, repr(result.exception))

        assert read_state().canary_started_ts is None
        assert read_state().canary_active is None

    def test_withdrawing_a_first_canary_clears_it(self, cwd):
        # Nothing promoted yet: rollback withdraws the canary that waits for ratings.
        _cli(*_INIT)
        write_state(replace(
            read_state(), canary_active="v1", canary_traffic_pct=5.0, canary_started_ts=T0,
        ))

        result, text = _cli("rollback")
        assert result.exit_code == 0, (result.output, repr(result.exception))

        assert "canary withdrawn: v1" in text
        assert read_state().canary_started_ts is None
        assert read_state().canary_active is None
