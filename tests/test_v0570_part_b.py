"""v0.57.0 Part B — adapters merge: linear / ties / dare / svd."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from soup_cli.cli import app as soup_app
from soup_cli.utils.adapter_merge import (
    SUPPORTED_STRATEGIES,
    MergeReport,
    merge_adapters,
    merge_dare,
    merge_linear,
    merge_svd,
    merge_ties,
    predict_merged_verdict,
)

runner = CliRunner()


# ---------- merge_linear ----------


def test_merge_linear_average_of_zero_and_one():
    a = {"w": np.zeros((2, 2), dtype=np.float32)}
    b = {"w": np.ones((2, 2), dtype=np.float32)}
    merged, skipped = merge_linear([a, b], [1.0, 1.0])
    assert np.allclose(merged["w"], 0.5)
    assert skipped == ()


def test_merge_linear_weighted():
    a = {"w": np.zeros((2,), dtype=np.float32)}
    b = {"w": np.ones((2,), dtype=np.float32)}
    merged, _ = merge_linear([a, b], [3.0, 1.0])
    assert np.allclose(merged["w"], 0.25)


def test_merge_linear_intersection_only():
    a = {"shared": np.ones((2,), dtype=np.float32), "only_a": np.ones((2,))}
    b = {"shared": np.zeros((2,), dtype=np.float32), "only_b": np.ones((2,))}
    merged, _ = merge_linear([a, b], [1.0, 1.0])
    assert set(merged.keys()) == {"shared"}


def test_merge_linear_shape_mismatch_skipped():
    a = {"w": np.zeros((4,), dtype=np.float32)}
    b = {"w": np.zeros((2,), dtype=np.float32)}
    merged, skipped = merge_linear([a, b], [1.0, 1.0])
    assert merged == {}
    assert skipped == ("w",)


def test_merge_linear_rejects_single_adapter():
    with pytest.raises(ValueError, match="at least 2"):
        merge_linear([{"w": np.zeros(1)}], [1.0])


def test_merge_linear_rejects_too_many():
    with pytest.raises(ValueError, match="at most 16"):
        merge_linear([{}] * 17, [1.0] * 17)


def test_merge_linear_bool_weight_rejected():
    a = {"w": np.zeros(1, dtype=np.float32)}
    b = {"w": np.zeros(1, dtype=np.float32)}
    with pytest.raises(TypeError):
        merge_linear([a, b], [True, 1.0])  # type: ignore[list-item]


def test_merge_linear_negative_weight_rejected():
    a = {"w": np.zeros(1, dtype=np.float32)}
    b = {"w": np.zeros(1, dtype=np.float32)}
    with pytest.raises(ValueError):
        merge_linear([a, b], [-1.0, 1.0])


def test_merge_linear_nan_weight_rejected():
    a = {"w": np.zeros(1, dtype=np.float32)}
    b = {"w": np.zeros(1, dtype=np.float32)}
    with pytest.raises(ValueError):
        merge_linear([a, b], [float("nan"), 1.0])


def test_merge_linear_zero_sum_rejected():
    a = {"w": np.zeros(1, dtype=np.float32)}
    b = {"w": np.zeros(1, dtype=np.float32)}
    with pytest.raises(ValueError, match="positive"):
        merge_linear([a, b], [0.0, 0.0])


def test_merge_linear_wrong_weights_length():
    a = {"w": np.zeros(1, dtype=np.float32)}
    b = {"w": np.zeros(1, dtype=np.float32)}
    with pytest.raises(ValueError, match="length"):
        merge_linear([a, b], [1.0])


# ---------- merge_ties ----------


def test_merge_ties_density_keeps_top():
    # Top-half of [1, 2, 3, 4]: keep 3, 4
    a = {"w": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)}
    b = {"w": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)}
    merged, _ = merge_ties([a, b], [1.0, 1.0], density=0.5)
    # First two slots should be trimmed to zero
    assert merged["w"][0] == 0.0
    assert merged["w"][3] != 0.0


def test_merge_ties_majority_sign_election():
    # Two adapters agree positive, one disagrees → elected sign is positive
    a = {"w": np.array([1.0], dtype=np.float32)}
    b = {"w": np.array([2.0], dtype=np.float32)}
    c = {"w": np.array([-3.0], dtype=np.float32)}
    merged, _ = merge_ties([a, b, c], [1.0, 1.0, 1.0], density=1.0)
    # Elected sign positive; negative entry dropped
    assert merged["w"][0] > 0


def test_merge_ties_invalid_density():
    a = {"w": np.zeros(1, dtype=np.float32)}
    b = {"w": np.zeros(1, dtype=np.float32)}
    with pytest.raises(ValueError):
        merge_ties([a, b], [1.0, 1.0], density=0.0)
    with pytest.raises(ValueError):
        merge_ties([a, b], [1.0, 1.0], density=1.5)


def test_merge_ties_bool_density_rejected():
    a = {"w": np.zeros(1, dtype=np.float32)}
    b = {"w": np.zeros(1, dtype=np.float32)}
    with pytest.raises(TypeError):
        merge_ties([a, b], [1.0, 1.0], density=True)  # type: ignore[arg-type]


# ---------- merge_dare ----------


def test_merge_dare_deterministic_with_seed():
    a = {"w": np.ones((10,), dtype=np.float32)}
    b = {"w": np.ones((10,), dtype=np.float32)}
    m1, _ = merge_dare([a, b], [1.0, 1.0], density=0.5, seed=42)
    m2, _ = merge_dare([a, b], [1.0, 1.0], density=0.5, seed=42)
    assert np.allclose(m1["w"], m2["w"])


def test_merge_dare_different_seeds_diverge():
    a = {"w": np.ones((100,), dtype=np.float32)}
    b = {"w": np.ones((100,), dtype=np.float32)}
    m1, _ = merge_dare([a, b], [1.0, 1.0], density=0.5, seed=1)
    m2, _ = merge_dare([a, b], [1.0, 1.0], density=0.5, seed=2)
    assert not np.allclose(m1["w"], m2["w"])


def test_merge_dare_bool_seed_rejected():
    a = {"w": np.zeros(1, dtype=np.float32)}
    b = {"w": np.zeros(1, dtype=np.float32)}
    with pytest.raises(TypeError):
        merge_dare([a, b], [1.0, 1.0], seed=True)  # type: ignore[arg-type]


def test_merge_dare_negative_seed_rejected():
    a = {"w": np.zeros(1, dtype=np.float32)}
    b = {"w": np.zeros(1, dtype=np.float32)}
    with pytest.raises(ValueError):
        merge_dare([a, b], [1.0, 1.0], seed=-1)


def test_merge_dare_density_1_equals_linear():
    a = {"w": np.array([2.0, 4.0], dtype=np.float32)}
    b = {"w": np.array([6.0, 8.0], dtype=np.float32)}
    merged, _ = merge_dare([a, b], [1.0, 1.0], density=1.0, seed=0)
    # density=1 → no drop, no rescale → identical to linear average
    assert np.allclose(merged["w"], [4.0, 6.0])


# ---------- merge_svd ----------


def test_merge_svd_no_rank_equals_linear():
    a = {"w": np.eye(4, dtype=np.float32)}
    b = {"w": np.eye(4, dtype=np.float32)}
    merged, _ = merge_svd([a, b], [1.0, 1.0])
    assert np.allclose(merged["w"], np.eye(4))


def test_merge_svd_with_rank_reduces_rank():
    # Random matrix → low-rank reconstruction should be lower rank
    rng = np.random.default_rng(0)
    a = {"w": rng.standard_normal((8, 8)).astype(np.float32)}
    b = {"w": rng.standard_normal((8, 8)).astype(np.float32)}
    merged, _ = merge_svd([a, b], [1.0, 1.0], rank=2)
    actual_rank = np.linalg.matrix_rank(merged["w"], tol=1e-5)
    assert actual_rank <= 2


def test_merge_svd_non_2d_passthrough():
    a = {"bias": np.ones((4,), dtype=np.float32)}
    b = {"bias": np.ones((4,), dtype=np.float32)}
    merged, _ = merge_svd([a, b], [1.0, 1.0], rank=1)
    assert np.allclose(merged["bias"], 1.0)


def test_merge_svd_rank_clamp():
    # Rank > min dimension → clamped
    a = {"w": np.eye(4, dtype=np.float32)}
    b = {"w": np.eye(4, dtype=np.float32)}
    merged, _ = merge_svd([a, b], [1.0, 1.0], rank=100)
    assert merged["w"].shape == (4, 4)


def test_merge_svd_invalid_rank():
    a = {"w": np.eye(2, dtype=np.float32)}
    b = {"w": np.eye(2, dtype=np.float32)}
    with pytest.raises(ValueError):
        merge_svd([a, b], [1.0, 1.0], rank=0)
    with pytest.raises(TypeError):
        merge_svd([a, b], [1.0, 1.0], rank=True)  # type: ignore[arg-type]


# ---------- merge_adapters end-to-end + CLI ----------


def _write_adapter(dir_path: Path, weights: dict) -> None:
    pytest.importorskip("safetensors")
    from safetensors.numpy import save_file

    dir_path.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(dir_path / "adapter_model.safetensors"))
    (dir_path / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 8}), encoding="utf-8"
    )


def test_merge_adapters_e2e_linear(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_adapter(tmp_path / "a", {"w": np.ones((2, 2), dtype=np.float32)})
    _write_adapter(tmp_path / "b", {"w": np.zeros((2, 2), dtype=np.float32)})
    report = merge_adapters(["a", "b"], "out", strategy="linear")
    assert isinstance(report, MergeReport)
    assert report.strategy == "linear"
    assert report.merged_layers == 1
    assert (tmp_path / "out" / "adapter_model.safetensors").exists()
    assert (tmp_path / "out" / "adapter_config.json").exists()


def test_merge_adapters_unknown_strategy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    with pytest.raises(ValueError, match="strategy must be"):
        merge_adapters(["a", "b"], "out", strategy="bogus")  # type: ignore[arg-type]


def test_merge_adapters_output_outside_cwd(tmp_path, monkeypatch):
    import os
    monkeypatch.chdir(tmp_path)
    _write_adapter(tmp_path / "a", {"w": np.zeros((2, 2), dtype=np.float32)})
    _write_adapter(tmp_path / "b", {"w": np.zeros((2, 2), dtype=np.float32)})
    outside = os.path.join(os.path.dirname(str(tmp_path)), "outside")
    with pytest.raises(ValueError):
        merge_adapters(["a", "b"], outside, strategy="linear")


def test_supported_strategies_immutable():
    # v0.57.0 review fix: SUPPORTED_STRATEGIES is a frozenset (matches v0.41.0+
    # allowlist policy). STRATEGY_ORDER preserves canonical iteration order.
    from soup_cli.utils.adapter_merge import STRATEGY_ORDER
    assert SUPPORTED_STRATEGIES == frozenset({"linear", "ties", "dare", "svd"})
    assert isinstance(SUPPORTED_STRATEGIES, frozenset)
    assert STRATEGY_ORDER == ("linear", "ties", "dare", "svd")


def test_merge_ties_density_one_keeps_everything():
    """density=1.0 is the inclusive upper bound — must not raise."""
    import numpy as np  # noqa: F811
    a = {"w": np.ones((4,), dtype=np.float32)}
    b = {"w": np.ones((4,), dtype=np.float32)}
    merged, _ = merge_ties([a, b], [1.0, 1.0], density=1.0)
    assert "w" in merged


def test_merge_linear_inf_weight_rejected():
    """math.isfinite must reject +inf as well as NaN."""
    import numpy as np  # noqa: F811
    a = {"w": np.zeros(1, dtype=np.float32)}
    b = {"w": np.zeros(1, dtype=np.float32)}
    with pytest.raises(ValueError, match="finite"):
        merge_linear([a, b], [float("inf"), 1.0])


def test_merge_ties_tied_sign_defaults_positive():
    """Sign-sum == 0 (tied vote) must elect +1, not silently zero parameters."""
    import numpy as np  # noqa: F811
    a = {"w": np.array([2.0], dtype=np.float32)}
    b = {"w": np.array([-2.0], dtype=np.float32)}
    merged, _ = merge_ties([a, b], [1.0, 1.0], density=1.0)
    # Tied sign → elected +1 → positive entry kept, negative dropped → result 2.0
    assert merged["w"][0] > 0


@pytest.mark.skipif(__import__("os").name == "nt",
                    reason="POSIX-only symlink semantics")
def test_merge_adapters_rejects_symlink_at_output_safetensors(tmp_path, monkeypatch):
    """Pre-placed symlink at output safetensors path must be rejected (TOCTOU)."""
    import os
    monkeypatch.chdir(tmp_path)
    _write_adapter(tmp_path / "a", {"w": np.ones((2, 2), dtype=np.float32)})
    _write_adapter(tmp_path / "b", {"w": np.zeros((2, 2), dtype=np.float32)})
    out = tmp_path / "out"
    out.mkdir()
    target = tmp_path / "evil.bin"
    target.write_bytes(b"x")
    os.symlink(str(target), str(out / "adapter_model.safetensors"))
    with pytest.raises(ValueError, match="symlink"):
        merge_adapters(["a", "b"], "out", strategy="linear")
    # Symlink target untouched
    assert target.read_bytes() == b"x"


def test_no_top_level_torch_import_in_merge():
    src = (Path(__file__).parent.parent / "soup_cli" / "utils" / "adapter_merge.py"
           ).read_text(encoding="utf-8")
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("import torch") or stripped.startswith("from torch"):
            indent = len(line) - len(stripped)
            assert indent > 0, f"top-level torch import: {line}"


def test_predict_merged_verdict_stub():
    report = MergeReport(
        strategy="linear", adapters=("a", "b"), weights=(0.5, 0.5),
        merged_layers=1, skipped_layers=(), output_dir="out", verdict="UNKNOWN",
    )
    assert predict_merged_verdict(report) == "UNKNOWN"


def test_predict_merged_verdict_rejects_non_report():
    with pytest.raises(TypeError):
        predict_merged_verdict("not a report")  # type: ignore[arg-type]


def test_predict_merged_verdict_canary_must_be_str():
    report = MergeReport(
        strategy="linear", adapters=("a", "b"), weights=(0.5, 0.5),
        merged_layers=1, skipped_layers=(), output_dir="out", verdict="OK",
    )
    with pytest.raises(TypeError):
        predict_merged_verdict(report, canary_suite=123)  # type: ignore[arg-type]


def test_merge_report_frozen():
    import dataclasses
    report = MergeReport(
        strategy="linear", adapters=("a", "b"), weights=(0.5, 0.5),
        merged_layers=1, skipped_layers=(), output_dir="out", verdict="OK",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.strategy = "ties"  # type: ignore[misc]


def test_adapters_merge_cli_help():
    result = runner.invoke(soup_app, ["adapters", "merge", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "--strategy" in result.output
    assert "--weights" in result.output


def test_adapters_merge_cli_linear(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_adapter(tmp_path / "a", {"w": np.ones((2, 2), dtype=np.float32)})
    _write_adapter(tmp_path / "b", {"w": np.zeros((2, 2), dtype=np.float32)})
    result = runner.invoke(soup_app, [
        "adapters", "merge", "a", "b", "-o", "out", "--strategy", "linear",
    ])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert (tmp_path / "out" / "adapter_model.safetensors").exists()


def test_adapters_merge_cli_unknown_strategy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_adapter(tmp_path / "a", {"w": np.zeros((2, 2), dtype=np.float32)})
    _write_adapter(tmp_path / "b", {"w": np.zeros((2, 2), dtype=np.float32)})
    result = runner.invoke(soup_app, [
        "adapters", "merge", "a", "b", "-o", "out", "--strategy", "bogus",
    ])
    assert result.exit_code == 2
    assert "Unknown --strategy" in result.output


def test_adapters_merge_cli_invalid_weights(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_adapter(tmp_path / "a", {"w": np.zeros((2, 2), dtype=np.float32)})
    _write_adapter(tmp_path / "b", {"w": np.zeros((2, 2), dtype=np.float32)})
    result = runner.invoke(soup_app, [
        "adapters", "merge", "a", "b", "-o", "out",
        "--strategy", "linear", "--weights", "1.0,abc",
    ])
    assert result.exit_code == 2


def test_adapters_merge_cli_single_adapter_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_adapter(tmp_path / "a", {"w": np.zeros((2, 2), dtype=np.float32)})
    result = runner.invoke(soup_app, [
        "adapters", "merge", "a", "-o", "out", "--strategy", "linear",
    ])
    assert result.exit_code == 2
    assert "at least 2" in result.output
