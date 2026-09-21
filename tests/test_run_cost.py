"""Tests for v0.34.0 Part B — per-run cost tracking.

Device-name fixtures are the strings ``torch.cuda.get_device_name()`` returns,
taken from Appendix A "Supported NVIDIA GPU Products" of the NVIDIA Linux
x86_64 driver README (590.48.01,
https://us.download.nvidia.com/XFree86/Linux-x86_64/590.48.01/README/supportedchips.html)
and from the probe list in issue #831. Non-NVIDIA names come from
``detect_device()`` in ``soup_cli/utils/gpu.py``.
"""

from __future__ import annotations

import pytest

from soup_cli.utils import run_cost
from soup_cli.utils.run_cost import (
    MAX_DURATION_SECS,
    estimate_run_cost_usd,
    format_cost_usd,
    lookup_gpu_rate,
)

# (device name as the driver reports it, expected label or None)
REAL_DEVICE_NAMES = [
    ("NVIDIA RTX A4000", "RTX A4000"),
    ("NVIDIA RTX A4500", "RTX A4500"),
    ("NVIDIA A40", "A40"),
    ("NVIDIA RTX A6000", "A6000"),
    ("NVIDIA RTX 6000 Ada Generation", "RTX 6000 Ada"),
    ("NVIDIA H100 80GB HBM3", "H100"),  # SXM5 board; name carries no SXM token
    ("NVIDIA H100 NVL", "H100"),
    ("NVIDIA H200", "H200"),
    ("NVIDIA B200", "B200"),
    ("NVIDIA L4", "L4"),
    ("NVIDIA L40S", "L40S"),
    ("NVIDIA GeForce RTX 5090", "RTX 5090"),
    ("NVIDIA GeForce RTX 5080", "RTX 5080"),
    ("NVIDIA GeForce RTX 5070", "RTX 5070"),
    ("NVIDIA GeForce RTX 3090 Ti", "RTX 3090"),  # no separate rental tier
    ("NVIDIA A100-SXM4-80GB", "A100 80GB"),
    ("NVIDIA A100 80GB PCIe", "A100 80GB"),
    ("NVIDIA A100-SXM4-40GB", "A100 40GB"),
    ("NVIDIA A10G", "A10G"),
    ("Tesla T4", "T4"),
    ("Tesla V100-SXM2-16GB", "V100"),
    # Priced as V100 on main; the anchored pattern must not drop the "S" variant.
    ("Tesla V100S-PCIE-32GB", "V100"),
    ("NVIDIA GeForce RTX 4070 Ti", "RTX 4070 Ti"),
    ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "RTX PRO 6000"),
]


class TestRealDeviceNames:
    @pytest.mark.parametrize(("name", "label"), REAL_DEVICE_NAMES)
    def test_resolves(self, name, label):
        result = lookup_gpu_rate(name)
        assert result is not None
        assert result[0] == label

    def test_rtx_a4000_is_not_priced_as_an_a40(self):
        # "a40" used to match inside "a4000".
        assert lookup_gpu_rate("NVIDIA RTX A4000") != ("A40", 0.55)
        assert lookup_gpu_rate("NVIDIA A40") == ("A40", 0.55)

    def test_a100_sxm4_80gb_gets_the_80gb_rate(self):
        # The SXM part reports "A100-SXM4-80GB"; it used to fall to the 40 GB row.
        assert lookup_gpu_rate("NVIDIA A100-SXM4-80GB") == ("A100 80GB", 2.05)

    def test_h100_80gb_hbm3_stays_on_generic_h100_row(self):
        # The name does not say SXM or PCIe, so we must not claim either.
        assert lookup_gpu_rate("NVIDIA H100 80GB HBM3") == ("H100", 4.11)

    def test_h100_nvl_stays_on_generic_h100_row(self):
        assert lookup_gpu_rate("NVIDIA H100 NVL") == ("H100", 4.11)

    def test_general_row_cannot_shadow_specific_row(self):
        table = run_cost._GPU_RATE_TABLE
        labels = [label for _p, label, _r in table]
        pairs = [
            ("A100 80GB", "A100 40GB", "NVIDIA A100-SXM4-80GB"),
            ("H100 SXM", "H100", "NVIDIA H100-SXM5-80GB"),
            ("RTX 4070 Ti", "RTX 4070", "NVIDIA GeForce RTX 4070 Ti"),
            ("RTX 5070 Ti", "RTX 5070", "NVIDIA GeForce RTX 5070 Ti"),
        ]
        for specific, general, name in pairs:
            general_pattern = table[labels.index(general)][0]
            # The tie is real: the general row also matches this name...
            assert general_pattern.search(name)
            # ...but the specific row is earlier and therefore wins.
            assert labels.index(specific) < labels.index(general)
            assert lookup_gpu_rate(name)[0] == specific

    @pytest.mark.parametrize(
        "name",
        [
            "Some Unknown GPU",
            "CPU (no GPU detected)",
            "Apple Silicon (Apple M2 Max)",
            "NVIDIA T4G",
            "NVIDIA GH200 480GB",
            "NVIDIA GB200",
            "NVIDIA B100",
            # Real Turing card (NVIDIA list). Without the \bada\b discriminator it
            # would be priced as an RTX 6000 Ada.
            "Quadro RTX 6000",
            # Trailing-boundary guards: a row must not match a longer model number.
            "NVIDIA GeForce RTX 40900",
            "NVIDIA RTX A40000",
            # Synthetic near-misses that the old unanchored patterns priced as
            # A100 / A100 / T4. They now (correctly) resolve to unknown.
            "NVIDIA A100X",
            "NVIDIA A1000",
            "NVIDIA RTX A1000 Laptop GPU",
            "NVIDIA T400",
        ],
    )
    def test_unknown_names_stay_none(self, name):
        assert lookup_gpu_rate(name) is None

    def test_80g_without_b_is_not_the_80gb_row(self):
        # "80G" is not a name any driver reports; only "80GB" selects the 80 GB row.
        assert lookup_gpu_rate("NVIDIA A100 80G") == ("A100 40GB", 1.10)

    def test_overlong_name_is_unknown_and_fast(self):
        import time

        pathological = "a100 " * 20_000  # ~100 KB, quadratic on the A100 80GB pattern
        start = time.perf_counter()
        assert lookup_gpu_rate(pathological) is None
        assert time.perf_counter() - start < 1.0

    @pytest.mark.parametrize(
        ("name", "label"),
        [
            # NVIDIA list: laptop dies report the desktop model number plus "Laptop GPU".
            ("NVIDIA GeForce RTX 4090 Laptop GPU", "RTX 4090"),
            ("NVIDIA GeForce RTX 5070 Laptop GPU", "RTX 5070"),
        ],
    )
    def test_laptop_parts_price_as_the_desktop_card(self, name, label):
        # Deliberate and reviewable: there is no laptop rental market, so the desktop
        # rate is a rough stand-in that overstates a laptop run. Pinned so a change
        # to this behaviour is a conscious one (see changelog fragment for #1081).
        assert lookup_gpu_rate(name)[0] == label

    def test_normal_length_name_unaffected_by_cap(self):
        assert lookup_gpu_rate("NVIDIA A100-SXM4-80GB") == ("A100 80GB", 2.05)


class TestTableHygiene:
    def test_every_pattern_is_word_anchored_and_case_insensitive(self):
        import re

        for pattern, label, rate in run_cost._GPU_RATE_TABLE:
            # Leading AND trailing boundary: dropping either lets "4090" match "40900".
            assert pattern.pattern.startswith("\\b"), label
            assert pattern.pattern.endswith("\\b"), label
            assert pattern.flags & re.IGNORECASE, label
            assert rate > 0, label

    def test_labels_unique(self):
        labels = [label for _p, label, _r in run_cost._GPU_RATE_TABLE]
        assert len(labels) == len(set(labels))

    def test_rates_other_suites_depend_on_are_unchanged(self):
        # tests/test_v07115.py pins A100 40GB and T4; a repricing PR must update both.
        assert lookup_gpu_rate("NVIDIA A100")[1] == 1.10
        assert lookup_gpu_rate("T4")[1] == 0.20
        assert lookup_gpu_rate("NVIDIA RTX 4090")[1] == 0.35


class TestLookup:
    def test_h100_match(self):
        result = lookup_gpu_rate("NVIDIA H100 80GB HBM3")
        assert result is not None
        label, rate = result
        assert label == "H100"
        assert rate > 0

    def test_h100_sxm_more_specific_wins(self):
        # Provider label (RunPod pod type "H100 SXM"), not a driver string.
        result = lookup_gpu_rate("NVIDIA H100 SXM 80GB")
        assert result is not None
        label, _rate = result
        assert label == "H100 SXM"

    def test_a100_80gb_more_specific_than_a100(self):
        result = lookup_gpu_rate("NVIDIA A100 80GB PCIe")
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
