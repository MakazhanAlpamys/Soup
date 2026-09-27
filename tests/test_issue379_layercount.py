from benchmarks.harness.layercount import measure_layer_sweep


def test_measurement_api_rejects_degenerate_sweep():
    try:
        measure_layer_sweep(layer_counts=[32, 32], measure=lambda **_: {})
    except RuntimeError:
        pass
    else:
        raise AssertionError("degenerate layer sweep was accepted")
