"""``--canary-max-wait`` ends a canary whatever the count, with or without a feedback log.

The timeout used to apply only when a feedback log gave a quality verdict AND the request-error
verdict already had its 30 canary answers, so a canary on 5% of keyed traffic could stall the
loop for ever. Now, once the wait is over, ``--canary-on-timeout`` decides:

* ``errors-only``: the request errors alone decide, with however many answers there are (a
  canary that failed more than stable is withdrawn, any other is promoted);
* ``rollback``: the canary is withdrawn;
* ``wait``: no time limit, as before.

The recorded reason says "timeout" and the counts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from soup_cli.utils.canary_router import CanaryPolicy, QualityStats, record_bucket_outcome
from soup_cli.utils.loop_daemon import WatchConfig, run_once
from soup_cli.utils.loop_state import LoopState

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc)
DAY = 86400.0


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _state(
    *, waited: float, on_timeout: str = "errors-only", autoroll: bool = True,
    explicit: bool = False,
) -> LoopState:
    return LoopState(
        served_model="base", eval_suite="e.json", baseline="b", status="running",
        canary_active="cand", canary_traffic_pct=5.0, canary_rollout_id="r1",
        canary_adapter="out/cand", canary_started_ts=NOW.timestamp() - waited,
        canary_autoroll_on_regress=autoroll, canary_on_timeout=on_timeout,
        canary_on_timeout_explicit=explicit,
    )


def _config(root: Path, **kwargs) -> WatchConfig:
    return WatchConfig(
        poll_interval_sec=1.0, max_iterations=1, state_path=str(root / ".soup" / "loop.yaml"),
        canary_stats_path=str(root / ".soup" / "canary-stats.json"),
        iteration_dir=str(root / ".soup-loops"), **kwargs,
    )


def _answers(root: Path, *, canary_ok=0, canary_bad=0, stable_ok=0) -> None:
    policy = CanaryPolicy(stable="base", canary="cand", traffic_pct=5.0)
    path = str(root / ".soup" / "canary-stats.json")
    for bucket, ok, count in (("canary", True, canary_ok), ("canary", False, canary_bad),
                              ("stable", True, stable_ok)):
        for _ in range(count):
            record_bucket_outcome(policy, bucket, ok, rollout_id="r1", path=path)


def _run(root, state, **kwargs):
    new_state, record, _decision = run_once(state, _config(root, **kwargs), now=NOW)
    return new_state, record


class TestWithoutAFeedbackLog:
    def test_errors_only_promotes_a_canary_with_few_answers_after_the_wait(self, cwd):
        _answers(cwd, canary_ok=5, stable_ok=20)

        state, record = _run(cwd, _state(waited=2 * DAY))

        assert state.canary_active is None and state.production_name == "cand"
        assert "timeout" in record.notes and "5 canary answers" in record.notes

    def test_the_default_withdraws_a_canary_that_got_no_answers_at_all(self, cwd):
        state, record = _run(cwd, _state(waited=2 * DAY))

        assert state.canary_active is None and state.production_name is None
        assert record.rolled_back is True
        assert "withdrawn on timeout" in record.notes
        assert "0 canary answers" in record.notes and "timeout_no_answers" in record.notes

    def test_an_explicit_errors_only_still_promotes_a_canary_with_no_answers(self, cwd):
        state, record = _run(cwd, _state(waited=2 * DAY, explicit=True))

        assert state.production_name == "cand"
        assert "timeout" in record.notes and "0 canary answers" in record.notes

    def test_one_answer_is_enough_for_the_default_to_decide_on_errors(self, cwd):
        _answers(cwd, canary_ok=1, stable_ok=20)

        state, record = _run(cwd, _state(waited=2 * DAY))

        assert state.production_name == "cand"
        assert "1 canary answers" in record.notes

    def test_rollback_on_timeout_with_no_answers_keeps_its_own_reason(self, cwd):
        state, record = _run(cwd, _state(waited=2 * DAY, on_timeout="rollback"))

        assert state.canary_active is None and record.rolled_back is True
        assert "timeout_no_answers" not in record.notes
        assert "rolled back on timeout" in record.notes

    def test_wait_with_no_answers_still_waits(self, cwd):
        state, _record = _run(cwd, _state(waited=30 * DAY, on_timeout="wait"))

        assert state.canary_active == "cand"

    def test_errors_only_withdraws_a_canary_that_failed_more_than_stable(self, cwd):
        _answers(cwd, canary_ok=1, canary_bad=4, stable_ok=20)

        state, record = _run(cwd, _state(waited=2 * DAY))

        assert state.canary_active is None and state.production_name is None
        assert record.rolled_back is True

    def test_rollback_on_timeout_withdraws_whatever_the_count(self, cwd):
        _answers(cwd, canary_ok=5, stable_ok=20)

        state, record = _run(cwd, _state(waited=2 * DAY, on_timeout="rollback"))

        assert state.canary_active is None and state.production_name is None
        assert record.rolled_back is True
        assert "timeout" in record.notes and "5 canary answers" in record.notes

    def test_wait_never_decides(self, cwd):
        _answers(cwd, canary_ok=5, stable_ok=20)

        state, _record = _run(cwd, _state(waited=30 * DAY, on_timeout="wait"))

        assert state.canary_active == "cand"

    def test_before_the_wait_is_over_the_canary_keeps_waiting(self, cwd):
        _answers(cwd, canary_ok=5, stable_ok=20)

        state, _record = _run(cwd, _state(waited=DAY / 2))

        assert state.canary_active == "cand"

    def test_a_rollout_with_no_start_time_cannot_time_out(self, cwd):
        base = _state(waited=2 * DAY)
        from dataclasses import replace

        state, _record = _run(cwd, replace(base, canary_started_ts=None))

        assert state.canary_active == "cand"


class TestWithAFeedbackLog:
    def test_few_answers_and_no_ratings_still_end_on_timeout(self, cwd):
        _answers(cwd, canary_ok=5, stable_ok=20)

        state, record = _run(
            cwd, _state(waited=2 * DAY), quality_fn=lambda state: QualityStats()
        )

        assert state.canary_active is None and state.production_name == "cand"
        assert "timeout" in record.notes or "unverified" in record.notes

    def test_rollback_on_timeout_with_a_feedback_log(self, cwd):
        _answers(cwd, canary_ok=5, stable_ok=20)

        state, record = _run(
            cwd, _state(waited=2 * DAY, on_timeout="rollback"),
            quality_fn=lambda state: QualityStats(),
        )

        assert state.canary_active is None and state.production_name is None
        assert record.rolled_back is True

    def test_wait_keeps_the_canary_with_a_feedback_log(self, cwd):
        _answers(cwd, canary_ok=5, stable_ok=20)

        state, _record = _run(
            cwd, _state(waited=30 * DAY, on_timeout="wait"),
            quality_fn=lambda state: QualityStats(),
        )

        assert state.canary_active == "cand"
