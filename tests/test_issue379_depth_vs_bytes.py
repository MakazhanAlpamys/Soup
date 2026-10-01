import pytest

from benchmarks.harness.depth_vs_bytes import (
    MeasurementInvalidError,
    depth_vs_bytes_verdict,
    measure_synthetic_sweep,
    run_self_test,
)


def recorded_rows():
    return [
        {
            "depth": 48,
            "quant": "nf4",
            "layer_mib": 163.8,
            "exact_counts": [192] * 3,
            "total": 192,
        },
        {
            "depth": 48,
            "quant": "nf4",
            "layer_mib": 171.5,
            "exact_counts": [192, 8, 8],
            "total": 192,
        },
        {
            "depth": 32,
            "quant": "bf16",
            "layer_mib": 935.0,
            "exact_counts": [128] * 3,
            "total": 128,
        },
    ]


def test_recorded_threshold_pattern_passes():
    assert depth_vs_bytes_verdict(recorded_rows())


def test_every_nf4_point_below_boundary_must_be_exact():
    rows = recorded_rows()
    rows.append(
        {
            "depth": 48,
            "quant": "nf4",
            "layer_mib": 105.0,
            "exact_counts": [192, 8, 8],
            "total": 192,
        }
    )
    assert not depth_vs_bytes_verdict(rows)


def test_every_nf4_point_above_boundary_must_be_wrong_after_first_read():
    rows = recorded_rows()
    rows.append(
        {
            "depth": 48,
            "quant": "nf4",
            "layer_mib": 432.0,
            "exact_counts": [192] * 3,
            "total": 192,
        }
    )
    assert not depth_vs_bytes_verdict(rows)


def test_bf16_control_must_be_exact_and_above_boundary():
    rows = recorded_rows()
    rows[-1]["exact_counts"] = [128, 8, 8]
    assert not depth_vs_bytes_verdict(rows)

    rows = recorded_rows()
    rows[-1]["layer_mib"] = 1.0
    assert not depth_vs_bytes_verdict(rows)


def test_empty_and_nonfinite_measurements_fail():
    assert not depth_vs_bytes_verdict([])
    rows = recorded_rows()
    rows[0]["layer_mib"] = float("inf")
    with pytest.raises(MeasurementInvalidError, match="finite"):
        depth_vs_bytes_verdict(rows)


def test_self_test_pins_recorded_boundaries():
    assert run_self_test() == 0


def test_measurement_sweep_requires_distinct_layer_sizes():
    with pytest.raises(MeasurementInvalidError, match="two layer sizes"):
        measure_synthetic_sweep(intermediate_sizes=[13824, 13824], measure=lambda **_: {})


def test_measurement_sweep_holds_nf4_depth_fixed_and_includes_recorded_bf16_control():
    calls = []

    def measure(**kwargs):
        calls.append(kwargs)
        return {"layer_mib": 1.0, "exact_counts": [192] * 3, "total": 192}

    rows = measure_synthetic_sweep(intermediate_sizes=[13824, 18432], measure=measure)
    assert {row["quant"] for row in rows} == {"nf4", "bf16"}
    assert [call["quant"] for call in calls] == ["nf4", "nf4", "bf16"]
    assert all(call["depth"] == 48 for call in calls[:2])
    assert calls[-1] == {"depth": 32, "intermediate_size": 18432, "quant": "bf16"}
    assert len(rows) == 3


def test_live_sweep_builds_recorded_shapes(tmp_path, monkeypatch):
    from benchmarks.harness import depth_vs_bytes

    generated = []
    measured = []

    def fake_run(command, *, check):
        assert check
        checkpoint = command[command.index("--out") + 1]
        generated.append(command)
        __import__("pathlib").Path(checkpoint).mkdir(parents=True)

    def fake_measure(weights, shards, quant):
        measured.append((weights, shards, quant))
        return {"layer_mib": 200.0, "exact_counts": [1, 0, 0], "total": 1}

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(depth_vs_bytes, "_measure_checkpoint", fake_measure)

    rows = depth_vs_bytes.run_live_sweep(tmp_path)

    assert len(generated) == 8
    assert len(rows) == 8
    assert [row["quant"] for row in rows] == ["nf4"] * 7 + ["bf16"]
    assert all(row["depth"] == 48 for row in rows[:7])
    assert rows[-1]["depth"] == 32
    assert rows[-1]["intermediate_size"] == 27648
    assert any("17408" in command for command in generated)
    assert any("18432" in command for command in generated)
    assert len(measured) == 8
