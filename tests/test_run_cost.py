"""Tests for v0.34.0 Part B — per-run cost tracking."""

from __future__ import annotations

from soup_cli.utils.run_cost import (
    MAX_DURATION_SECS,
    estimate_run_cost_usd,
    format_cost_usd,
    lookup_gpu_rate,
)


class TestLookup:
    def test_h100_match(self):
        result = lookup_gpu_rate("NVIDIA H100 80GB HBM3")
        assert result is not None
        label, rate = result
        assert label == "H100"
        assert rate > 0

    def test_h100_sxm_more_specific_wins(self):
        result = lookup_gpu_rate("NVIDIA H100 SXM 80GB")
        assert result is not None
        label, _rate = result
        assert label == "H100 SXM"

    def test_a100_80gb_more_specific_than_a100(self):
        result = lookup_gpu_rate("NVIDIA A100 80GB")
        assert result is not None
        assert result[0] == "A100 80GB"

    def test_a100_40gb_falls_back(self):
        result = lookup_gpu_rate("NVIDIA A100-PCIE-40GB")
        assert result is not None
        assert result[0] == "A100 40GB"

    def test_unknown_returns_none(self):
        assert lookup_gpu_rate("Some Unknown GPU") is None

    def test_none_returns_none(self):
        assert lookup_gpu_rate(None) is None

    def test_empty_returns_none(self):
        assert lookup_gpu_rate("") is None

    def test_null_byte_returns_none(self):
        assert lookup_gpu_rate("h100\x00rm") is None

    def test_non_string_returns_none(self):
        assert lookup_gpu_rate(123) is None  # type: ignore[arg-type]


class TestEstimate:
    def test_one_hour_h100(self):
        cost = estimate_run_cost_usd("NVIDIA H100", 3600.0)
        assert cost is not None
        assert cost > 1.0  # rough sanity

    def test_zero_duration_none(self):
        assert estimate_run_cost_usd("H100", 0) is None

    def test_negative_duration_none(self):
        assert estimate_run_cost_usd("H100", -1.0) is None

    def test_unknown_gpu_none(self):
        assert estimate_run_cost_usd("Some GPU", 3600) is None

    def test_cpu_none(self):
        assert estimate_run_cost_usd("cpu", 3600) is None

    def test_multi_gpu_scales(self):
        single = estimate_run_cost_usd("A100 80GB", 3600.0, num_gpus=1)
        quad = estimate_run_cost_usd("A100 80GB", 3600.0, num_gpus=4)
        assert single is not None and quad is not None
        assert abs(quad - 4 * single) < 0.01

    def test_duration_clamped(self):
        # Absurd duration is clamped, not rejected.
        cost = estimate_run_cost_usd("T4", MAX_DURATION_SECS * 100)
        assert cost is not None  # finite
        assert cost > 0

    def test_invalid_num_gpus_treated_as_one(self):
        cost_zero = estimate_run_cost_usd("H100", 3600.0, num_gpus=0)
        cost_one = estimate_run_cost_usd("H100", 3600.0, num_gpus=1)
        assert cost_zero == cost_one

    def test_bool_num_gpus_rejected(self):
        # bool is a subclass of int; True must NOT silently scale by 1.
        cost_true = estimate_run_cost_usd("H100", 3600.0, num_gpus=True)  # type: ignore[arg-type]
        cost_one = estimate_run_cost_usd("H100", 3600.0, num_gpus=1)
        assert cost_true == cost_one  # bool falls back to single-GPU rate

    def test_none_duration_returns_none(self):
        assert estimate_run_cost_usd("H100", None) is None  # type: ignore[arg-type]


class TestFormat:
    def test_none_renders_dash(self):
        assert format_cost_usd(None) == "—"

    def test_small_renders_lt_one_cent(self):
        assert format_cost_usd(0.001) == "<$0.01"

    def test_normal_dollars(self):
        assert format_cost_usd(1.234) == "$1.23"

    def test_round_dollars(self):
        assert format_cost_usd(10.0) == "$10.00"


class TestTrackerIntegration:
    def test_finish_run_records_cost(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "x.db"))
        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker()
        run_id = tracker.start_run(
            config_dict={"base": "x", "task": "sft"},
            device="cuda",
            device_name="NVIDIA H100",
            gpu_info={"memory_total": "80GB"},
        )
        tracker.finish_run(
            run_id=run_id,
            initial_loss=2.0,
            final_loss=0.5,
            total_steps=100,
            duration_secs=3600.0,
            output_dir="/tmp/x",
        )
        run = tracker.get_run(run_id)
        assert run is not None
        assert run["cost_usd"] is not None
        assert run["cost_usd"] > 0
        assert run["cost_gpu_label"] == "H100"
        tracker.close()

    def test_finish_run_unknown_gpu_no_cost(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "y.db"))
        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker()
        run_id = tracker.start_run(
            config_dict={"base": "x", "task": "sft"},
            device="cpu",
            device_name="cpu",
            gpu_info={},
        )
        tracker.finish_run(
            run_id=run_id,
            initial_loss=2.0,
            final_loss=0.5,
            total_steps=10,
            duration_secs=60.0,
            output_dir="/tmp/x",
        )
        run = tracker.get_run(run_id)
        assert run is not None
        assert run["cost_usd"] is None
        tracker.close()
