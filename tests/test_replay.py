"""Tests for v0.34.0 Part E — runs replay."""

from __future__ import annotations

import math

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.replay import (
    MAX_PLOT_POINTS,
    ReplaySummary,
    downsample,
    summarise,
)


def _row(step, loss):
    return {"step": step, "loss": loss}


class TestSummarise:
    def test_empty(self):
        result = summarise([])
        assert result.total_rows == 0
        assert result.initial_loss is None

    def test_basic(self):
        rows = [_row(0, 2.0), _row(10, 1.5), _row(20, 1.0)]
        result = summarise(rows)
        assert result.total_rows == 3
        assert result.initial_loss == 2.0
        assert result.final_loss == 1.0
        assert result.min_loss == 1.0
        assert result.min_loss_step == 20
        assert result.first_step == 0
        assert result.last_step == 20

    def test_min_loss_in_middle(self):
        rows = [_row(0, 2.0), _row(10, 0.5), _row(20, 1.0)]
        result = summarise(rows)
        assert result.min_loss == 0.5
        assert result.min_loss_step == 10

    def test_skips_nan(self):
        rows = [_row(0, 2.0), _row(10, float("nan")), _row(20, 1.0)]
        result = summarise(rows)
        assert result.initial_loss == 2.0
        assert result.final_loss == 1.0
        assert math.isfinite(result.min_loss)

    def test_all_nan(self):
        rows = [_row(0, float("nan")), _row(10, float("nan"))]
        result = summarise(rows)
        assert result.initial_loss is None

    def test_summary_frozen(self):
        result = ReplaySummary(0, None, None, None, None, None, None)
        with pytest.raises(Exception):
            result.total_rows = 5  # type: ignore[misc]


class TestDownsample:
    def test_short_unchanged(self):
        rows = [_row(i, 2.0) for i in range(50)]
        assert downsample(rows, max_points=100) == rows

    def test_long_capped(self):
        rows = [_row(i, 2.0) for i in range(10_000)]
        out = downsample(rows, max_points=200)
        assert len(out) == 200  # hard cap, no endpoint-pin overshoot (#473)
        assert out[0] == rows[0]
        assert out[-1] == rows[-1]

    def test_default_cap(self):
        rows = [_row(i, 2.0) for i in range(MAX_PLOT_POINTS * 4)]
        out = downsample(rows)
        assert len(out) == MAX_PLOT_POINTS

    def test_last_index_lands_on_last_row_no_duplicate(self):
        # 5 rows, max_points=3 -> evenly spaced indices 0, 2, 4, so the final
        # sample IS the last row and appears exactly once. Renamed from
        # test_stride_lands_on_last_row_no_duplicate_pin (#470): the endpoint
        # pin is gone, but this is still the only guard against an off-by-one
        # in the index arithmetic.
        rows = [_row(i, 2.0) for i in range(5)]
        out = downsample(rows, max_points=3)
        assert out == [rows[0], rows[2], rows[4]]
        assert out[-1] is rows[-1]

    @pytest.mark.parametrize("total", [3, 5, 8, 13, 101, 1000])
    @pytest.mark.parametrize("max_points", [1, 2, 3, 7, 50])
    def test_never_returns_more_than_max_points(self, total, max_points):
        # The docstring promises "at most max_points"; before #473 a 5-row
        # series with max_points=2 came back with 3 rows.
        rows = [_row(i, 2.0) for i in range(total)]
        out = downsample(rows, max_points=max_points)
        assert len(out) <= max_points

    def test_one_row_over_the_cap_drops_exactly_one_row(self):
        # A ceil-based stride would halve a series sitting a single row over
        # the cap (101 rows, cap 100 -> 51 points). Even spacing keeps 100.
        rows = [_row(i, 2.0) for i in range(101)]
        out = downsample(rows, max_points=100)
        assert len(out) == 100
        assert out[0] is rows[0]
        assert out[-1] is rows[-1]

    def test_endpoints_survive_every_shape(self):
        for total in range(3, 40):
            for max_points in range(2, total):
                rows = [_row(i, 2.0) for i in range(total)]
                out = downsample(rows, max_points=max_points)
                assert out[0]["step"] == 0, (total, max_points)
                assert out[-1]["step"] == total - 1, (total, max_points)

    def test_order_preserved_without_duplicates(self):
        rows = [_row(i, 2.0) for i in range(97)]
        out = downsample(rows, max_points=13)
        steps = [row["step"] for row in out]
        assert steps == sorted(steps)
        assert len(set(steps)) == len(steps)

    def test_max_points_one_keeps_the_final_row(self):
        # Both endpoints cannot fit in a single slot; the final loss wins,
        # which is exactly what the old endpoint pin existed to guarantee.
        rows = [_row(i, 2.0) for i in range(5)]
        assert downsample(rows, max_points=1) == [rows[-1]]

    def test_zero_max_rejected(self):
        with pytest.raises(ValueError):
            downsample([_row(0, 2.0)], max_points=0)

    def test_negative_max_rejected(self):
        with pytest.raises(ValueError):
            downsample([_row(0, 2.0)], max_points=-5)


class TestCli:
    def test_replay_help(self):
        runner = CliRunner()
        result = runner.invoke(app, ["runs", "replay", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "replay" in result.output.lower()

    def test_replay_unknown_run(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "x.db"))
        runner = CliRunner()
        result = runner.invoke(app, ["runs", "replay", "doesnotexist"])
        assert result.exit_code != 0, (result.output, repr(result.exception))
        assert "not found" in result.output.lower()

    def test_replay_renders(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "y.db"))
        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker()
        run_id = tracker.start_run(
            config_dict={"base": "x", "task": "sft"},
            device="cpu", device_name="cpu", gpu_info={},
        )
        for step in range(0, 50, 10):
            tracker.log_metrics(run_id, step=step, loss=2.0 - step * 0.01)
        tracker.finish_run(
            run_id=run_id, initial_loss=2.0, final_loss=1.5,
            total_steps=50, duration_secs=10.0, output_dir="/tmp/x",
        )
        tracker.close()
        runner = CliRunner()
        result = runner.invoke(app, ["runs", "replay", run_id, "--no-plot"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert run_id in result.output
        # Summary should reference initial / final loss
        assert "2.0" in result.output or "2.00" in result.output
