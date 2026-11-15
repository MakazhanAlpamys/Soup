from benchmarks.measure_preference_vram_grid import classify_ratio, parse_nvidia_smi_sample


def test_grid_calls_a_probe_under_read_a_refusal():
    assert classify_ratio(999, 1000) == "under-read"


def test_grid_identifies_the_sft_over_read_band():
    assert classify_ratio(1130, 1000) == "over-read-in-SFT-band"


def test_grid_keeps_a_larger_over_read_visible():
    assert classify_ratio(1200, 1000) == "over-read-above-SFT-band"


def test_grid_does_not_draw_a_conclusion_from_a_missing_peak():
    assert classify_ratio(None, 1000) == "incomplete"


def test_nvidia_smi_sample_records_device_memory_temperature_and_utilization():
    sample = parse_nvidia_smi_sample("2026/11/15 17:30:01.250, 6123, 2069, 58, 91")

    assert sample == {
        "timestamp": "2026/11/15 17:30:01.250",
        "memory_used_mib": 6123,
        "memory_free_mib": 2069,
        "temperature_c": 58,
        "utilization_percent": 91,
    }
