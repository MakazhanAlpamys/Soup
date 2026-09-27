from benchmarks.harness.depth_vs_bytes import measure_synthetic_sweep


def test_measurement_sweep_requires_distinct_depths():
    try:
        measure_synthetic_sweep(depths=[2, 2], measure=lambda **_: {})
    except RuntimeError:
        pass
    else:
        raise AssertionError("degenerate depth sweep was accepted")
