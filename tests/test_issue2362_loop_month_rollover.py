"""#2362: `soup loop watch` resets the monthly spend when the UTC month rolls over.

Every test fixes the clock by passing ``now`` into ``run_once``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from soup_cli.utils.loop_daemon import WatchConfig, run_once
from soup_cli.utils.loop_state import LoopState, read_state, write_state


def _at(year: int, month: int, day: int, hour: int = 12) -> datetime:
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


def _state(**overrides) -> LoopState:
    base = dict(
        served_model="m",
        eval_suite="e",
        baseline="b",
        status="running",
        monthly_budget_usd=10.0,
        spent_this_month_usd=10.0,
        last_run_date="2026-09-30",
    )
    base.update(overrides)
    return LoopState(**base)


def _config(cost: float, harvested: list) -> WatchConfig:
    def harvest(state):
        harvested.append(state.last_run_date)
        return {"pairs_harvested": 0}

    return WatchConfig(harvest_fn=harvest, cost_fn=lambda s: cost)


class TestMonthRollover:
    def test_new_month_proceeds_and_resets_spend(self):
        harvested: list = []
        new_state, _, decision = run_once(_state(), _config(0.50, harvested), now=_at(2026, 10, 30))
        assert decision.proceed is True
        assert harvested == ["2026-10-30"]
        assert new_state.spent_this_month_usd == pytest.approx(0.50)
        assert new_state.last_run_date == "2026-10-30"

    def test_same_state_same_month_is_still_a_budget_skip(self):
        harvested: list = []
        new_state, record, decision = run_once(
            _state(), _config(0.50, harvested), now=_at(2026, 9, 30)
        )
        assert decision.proceed is False
        assert "would exceed monthly budget" in decision.reason
        assert record.notes.startswith("budget-skip:")
        assert harvested == []
        assert new_state.spent_this_month_usd == pytest.approx(10.0)

    def test_first_iteration_after_midnight_on_the_first(self):
        new_state, _, decision = run_once(_state(), _config(0.50, []), now=_at(2026, 10, 1, hour=0))
        assert decision.proceed is True
        assert new_state.spent_this_month_usd == pytest.approx(0.50)

    def test_new_year_resets(self):
        state = _state(last_run_date="2026-12-31")
        new_state, _, decision = run_once(state, _config(0.50, []), now=_at(2027, 1, 1))
        assert decision.proceed is True
        assert new_state.spent_this_month_usd == pytest.approx(0.50)
        assert new_state.last_run_date == "2027-01-01"

    def test_same_month_keeps_accumulated_spend(self):
        state = _state(spent_this_month_usd=2.0, last_run_date="2026-10-01")
        cfg = _config(1.0, [])
        state, _, first = run_once(state, cfg, now=_at(2026, 10, 15))
        state, _, second = run_once(state, cfg, now=_at(2026, 10, 31))
        assert first.proceed is True
        assert second.proceed is True
        assert state.spent_this_month_usd == pytest.approx(4.0)

    def test_never_run_state_keeps_spend(self):
        state = _state(spent_this_month_usd=3.0, last_run_date=None)
        new_state, _, decision = run_once(state, _config(1.0, []), now=_at(2026, 10, 30))
        assert decision.proceed is True
        assert new_state.spent_this_month_usd == pytest.approx(4.0)


class TestRestartFromSavedState:
    def test_restart_in_same_month_keeps_spend(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = str(tmp_path / ".soup" / "loop.yaml")
        cfg = _config(1.0, [])
        state = _state(spent_this_month_usd=0.0, last_run_date=None)
        state, _, _ = run_once(state, cfg, now=_at(2026, 10, 2))
        write_state(state, path)
        restarted = read_state(path)
        assert restarted.spent_this_month_usd == pytest.approx(1.0)
        assert restarted.last_run_date == "2026-10-02"
        state, _, decision = run_once(restarted, cfg, now=_at(2026, 10, 20))
        assert decision.proceed is True
        assert state.spent_this_month_usd == pytest.approx(2.0)

    def test_restart_in_next_month_resets_spend(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = str(tmp_path / ".soup" / "loop.yaml")
        write_state(_state(), path)
        restarted = read_state(path)
        assert restarted.spent_this_month_usd == pytest.approx(10.0)
        state, _, decision = run_once(restarted, _config(0.50, []), now=_at(2026, 10, 30))
        assert decision.proceed is True
        assert state.spent_this_month_usd == pytest.approx(0.50)


class TestSkipInNewMonth:
    def test_skip_resets_once_and_next_iteration_keeps_spend(self):
        harvested: list = []
        state, record, decision = run_once(
            _state(), _config(12.0, harvested), now=_at(2026, 10, 30)
        )
        assert decision.proceed is False
        assert "would exceed monthly budget" in decision.reason
        assert record.gate_verdict == "SKIPPED"
        assert harvested == []
        assert state.spent_this_month_usd == 0.0
        assert state.last_run_date == "2026-10-30"

        cfg = _config(0.50, harvested)
        state, _, second = run_once(state, cfg, now=_at(2026, 10, 30, hour=13))
        state, _, third = run_once(state, cfg, now=_at(2026, 10, 31))
        assert second.proceed is True
        assert third.proceed is True
        assert state.spent_this_month_usd == pytest.approx(1.0)


class TestDailyCapAcrossMonthChange:
    def test_cap_resets_on_the_first_and_still_holds_that_day(self):
        state = _state(max_runs_per_day=1, runs_today=1)
        cfg = _config(0.50, [])
        state, _, first = run_once(state, cfg, now=_at(2026, 10, 1))
        assert first.proceed is True
        assert state.runs_today == 1
        assert state.spent_this_month_usd == pytest.approx(0.50)

        state, _, second = run_once(state, cfg, now=_at(2026, 10, 1, hour=18))
        assert second.proceed is False
        assert "daily cap reached" in second.reason
        assert state.spent_this_month_usd == pytest.approx(0.50)
