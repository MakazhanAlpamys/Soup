from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.harness.layercount import (
    MeasurementInvalidError,
    layercount_verdict,
    measure_layer_sweep,
    run_self_test,
)


def recorded_rows():
    return [
        {
            "layers": 48,
            "quant": "nf4",
            "role": "sweep",
            "layer_mib": 0.01,
            "exact_counts": [192] * 3,
            "total": 192,
        },
        {
            "layers": 64,
            "quant": "nf4",
            "role": "sweep",
            "layer_mib": 0.01,
            "exact_counts": [256] * 3,
            "total": 256,
        },
        {
            "layers": 48,
            "quant": "bf16",
            "role": "sweep",
            "layer_mib": 0.05,
            "exact_counts": [192] * 3,
            "total": 192,
        },
        {
            "layers": 64,
            "quant": "bf16",
            "role": "sweep",
            "layer_mib": 0.05,
            "exact_counts": [256] * 3,
            "total": 256,
        },
        {
            "layers": 48,
            "quant": "nf4",
            "role": "control",
            "layer_mib": 187.0,
            "exact_counts": [192, 8, 8],
            "total": 192,
        },
    ]


def test_recorded_layercount_relationship_passes():
    assert layercount_verdict(recorded_rows())


def test_nf4_sweep_must_span_the_48_to_64_boundary():
    rows = recorded_rows()
    rows[1]["layers"] = 32
    rows[1]["total"] = 128
    rows[1]["exact_counts"] = [128] * 3
    assert not layercount_verdict(rows)


def test_small_sweep_rows_must_be_exact_and_constant_size():
    rows = recorded_rows()
    rows[0]["exact_counts"] = [192, 8, 8]
    assert not layercount_verdict(rows)

    rows = recorded_rows()
    rows[1]["layer_mib"] = 0.02
    assert not layercount_verdict(rows)

    rows = recorded_rows()
    rows[0]["layer_mib"] = 187.0
    rows[1]["layer_mib"] = 187.0
    assert not layercount_verdict(rows)


def test_bf16_sweep_is_required_and_must_be_exact():
    rows = [row for row in recorded_rows() if row["quant"] != "bf16"]
    assert not layercount_verdict(rows)

    rows = recorded_rows()
    rows[2]["exact_counts"] = [192, 8, 8]
    assert not layercount_verdict(rows)


def test_positive_control_is_required_and_must_turn_wrong_later():
    rows = [row for row in recorded_rows() if row["role"] != "control"]
    assert not layercount_verdict(rows)

    rows = recorded_rows()
    rows[-1]["exact_counts"] = [192] * 3
    assert not layercount_verdict(rows)

    rows = recorded_rows()
    rows[-1]["layer_mib"] = 1.0
    assert not layercount_verdict(rows)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("total", 128, "four LoRA"),
        ("exact_counts", [300, 300, 300], "invalid exact"),
        ("layer_mib", float("nan"), "finite"),
    ],
)
def test_malformed_measurements_fail_closed(field, value, message):
    rows = deepcopy(recorded_rows())
    rows[0][field] = value
    with pytest.raises(MeasurementInvalidError, match=message):
        layercount_verdict(rows)


def test_nonpositive_layer_count_is_rejected_independently():
    rows = recorded_rows()
    rows[0].update(layers=0, total=0, exact_counts=[0] * 3)
    with pytest.raises(MeasurementInvalidError, match="positive integers"):
        layercount_verdict(rows)


def test_measurement_api_rejects_degenerate_sweep():
    with pytest.raises(MeasurementInvalidError, match="distinct counts"):
        measure_layer_sweep(layer_counts=[32, 32], measure=lambda **_: {})


def test_measurement_api_covers_both_quantizations_and_control():
    calls = []

    def measure(**kwargs):
        calls.append(kwargs)
        return {"layer_mib": 0.01, "exact_counts": [1] * 3, "total": 1}

    rows = measure_layer_sweep(layer_counts=[48, 64], measure=measure)

    assert len(rows) == 5
    assert [call["quant"] for call in calls] == ["nf4", "bf16", "nf4", "bf16", "nf4"]
    assert all(call["hidden"] == 64 for call in calls[:4])
    assert calls[-1] == {
        "layers": 48,
        "hidden": 5120,
        "intermediate": 20480,
        "quant": "nf4",
        "role": "control",
    }


def test_live_sweep_builds_recorded_shapes(tmp_path, monkeypatch):
    from benchmarks.harness import layercount

    generated = []
    measured = []

    def fake_run(command, *, check):
        assert check
        checkpoint = Path(command[command.index("--out") + 1])
        generated.append(command)
        checkpoint.mkdir(parents=True)

    def fake_measure(weights, shards, quant):
        measured.append((weights, shards, quant))
        return {"layer_mib": 0.01, "exact_counts": [1] * 3, "total": 1}

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(layercount, "_measure_checkpoint", fake_measure)

    rows = layercount.run_live_sweep(tmp_path)

    assert len(generated) == 7
    assert len(measured) == 13
    assert len(rows) == 13
    assert [row["layers"] for row in rows if row["quant"] == "nf4"] == [
        32,
        48,
        56,
        60,
        64,
        80,
        48,
    ]
    assert rows[-1]["role"] == "control"
    assert any("20480" in command for command in generated)


def test_self_test_exercises_the_full_verdict():
    assert run_self_test() == 0
