"""#2390: `soup loop status` shows the spend and run count the next iteration will see.

`run_once` resets both from the saved ``last_run_date``, so between iterations the stored values
can belong to a past month or day. Every test goes through `soup loop status`, with the UTC clock
fixed by replacing ``datetime`` in the command module, except the one that uses the real clock.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

import soup_cli.commands.loop as loop_cmd
from soup_cli.cli import app
from soup_cli.utils.loop_state import LoopState, write_state
from tests.conftest import strip_ansi

runner = CliRunner()


def _at(year: int, month: int, day: int, hour: int = 12) -> datetime:
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


def _state(**overrides) -> LoopState:
    base = dict(
        served_model="m",
        eval_suite="e",
        baseline="b",
        monthly_budget_usd=10.0,
        spent_this_month_usd=4.0,
        max_runs_per_day=5,
        runs_today=5,
        last_run_date="2026-10-31",
    )
    base.update(overrides)
    return LoopState(**base)


def _fix_clock(monkeypatch, now: datetime) -> None:
    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            assert tz is timezone.utc
            return now

    monkeypatch.setattr(loop_cmd, "datetime", _Clock)


def _invoke() -> str:
    result = runner.invoke(app, ["loop", "status"])
    assert result.exit_code == 0, result.output
    return " ".join(strip_ansi(result.output).split())


def _status(tmp_path: Path, monkeypatch, state: LoopState) -> str:
    monkeypatch.chdir(tmp_path)
    write_state(state, str(tmp_path / ".soup" / "loop.yaml"))
    return _invoke()


class TestStatusShowsTheNextIteration:
    def test_new_month_shows_no_spend(self, tmp_path, monkeypatch):
        _fix_clock(monkeypatch, _at(2026, 10, 1, hour=0))
        out = _status(
            tmp_path, monkeypatch, _state(spent_this_month_usd=10.0, last_run_date="2026-09-30")
        )
        assert "budget │ $0.00 / $10.00 │" in out

    def test_new_year_shows_no_spend(self, tmp_path, monkeypatch):
        _fix_clock(monkeypatch, _at(2027, 1, 1))
        out = _status(tmp_path, monkeypatch, _state(last_run_date="2026-12-31"))
        assert "budget │ $0.00 / $10.00 │" in out

    def test_new_day_shows_no_runs_and_keeps_the_month_spend(self, tmp_path, monkeypatch):
        _fix_clock(monkeypatch, _at(2026, 10, 31))
        out = _status(tmp_path, monkeypatch, _state(last_run_date="2026-10-30"))
        assert "runs_today │ 0 / 5 │" in out
        assert "budget │ $4.00 / $10.00 │" in out

    def test_same_day_shows_the_stored_values(self, tmp_path, monkeypatch):
        _fix_clock(monkeypatch, _at(2026, 10, 31, hour=18))
        out = _status(tmp_path, monkeypatch, _state())
        assert "budget │ $4.00 / $10.00 │" in out
        assert "runs_today │ 5 / 5 │" in out

    def test_never_run_state_keeps_its_spend(self, tmp_path, monkeypatch):
        _fix_clock(monkeypatch, _at(2026, 10, 31))
        out = _status(tmp_path, monkeypatch, _state(last_run_date=None, runs_today=0))
        assert "budget │ $4.00 / $10.00 │" in out

    def test_real_clock_after_a_past_month(self, tmp_path, monkeypatch):
        out = _status(
            tmp_path, monkeypatch, _state(spent_this_month_usd=10.0, last_run_date="2000-01-15")
        )
        assert "budget │ $0.00 / $10.00 │" in out
        assert "runs_today │ 0 / 5 │" in out


def test_status_does_not_write_the_state(tmp_path, monkeypatch):
    _fix_clock(monkeypatch, _at(2026, 10, 1))
    path = tmp_path / ".soup" / "loop.yaml"
    _status(tmp_path, monkeypatch, _state(spent_this_month_usd=10.0, last_run_date="2026-09-30"))
    before = path.read_bytes()
    out = _invoke()
    assert "budget │ $0.00 / $10.00 │" in out
    assert path.read_bytes() == before


def test_both_values_read_one_clock(tmp_path, monkeypatch):
    reads = []

    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            reads.append(tz)
            return _at(2026, 10, 31)  # a new object on every read

    monkeypatch.setattr(loop_cmd, "datetime", _Clock)
    seen = []
    real_month = loop_cmd.reset_monthly_spend_if_new_month
    real_day = loop_cmd.reset_daily_counter_if_new_day

    def month(spent, last_run_date, *, now=None):
        seen.append(now)
        return real_month(spent, last_run_date, now=now)

    def day(runs_today, last_run_date, *, now=None):
        seen.append(now)
        return real_day(runs_today, last_run_date, now=now)

    monkeypatch.setattr(loop_cmd, "reset_monthly_spend_if_new_month", month)
    monkeypatch.setattr(loop_cmd, "reset_daily_counter_if_new_day", day)
    _status(tmp_path, monkeypatch, _state(last_run_date="2026-10-30"))
    assert reads == [timezone.utc]
    assert len(seen) == 2
    assert seen[0] is not None
    assert seen[0] is seen[1]


@pytest.mark.parametrize("token", ["Infinity", "NaN"])
def test_non_finite_spend_is_reported_as_invalid_state(tmp_path, monkeypatch, token):
    # write_state refuses these, but read_state's json.loads accepts a hand-edited file with them.
    _fix_clock(monkeypatch, _at(2026, 10, 31))
    monkeypatch.chdir(tmp_path)
    path = tmp_path / ".soup" / "loop.yaml"
    write_state(_state(), str(path))
    text = path.read_text(encoding="utf-8")
    assert text.count('"spent_this_month_usd": 4.0') == 1
    path.write_text(
        text.replace('"spent_this_month_usd": 4.0', f'"spent_this_month_usd": {token}'),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["loop", "status"])
    out = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 2
    assert "loop state invalid" in out
    assert result.exception is None or isinstance(result.exception, SystemExit)
