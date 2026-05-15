"""v0.57.0 Part A — adapters diff: math kernel + CLI smoke tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from soup_cli.cli import app as soup_app
from soup_cli.utils.adapter_diff import (
    AdapterDiffReport,
    LayerDiff,
    compute_adapter_diff,
    compute_layer_diffs,
    effective_rank,
    render_report_json,
    render_report_markdown,
)

runner = CliRunner()


# ---------- effective_rank ----------


def test_effective_rank_identity_matrix():
    eye = np.eye(8)
    rank = effective_rank(eye)
    assert abs(rank - 8.0) < 1e-6


def test_effective_rank_concentrated():
    # All energy in one direction → effective rank ≈ 1
    matrix = np.zeros((4, 4))
    matrix[0, 0] = 1.0
    assert effective_rank(matrix) == pytest.approx(1.0, abs=1e-6)


def test_effective_rank_empty():
    assert effective_rank(np.zeros((0, 0))) == 0.0


def test_effective_rank_1d_reshapes():
    assert effective_rank(np.ones(5)) >= 0.0


def test_effective_rank_zero_matrix():
    assert effective_rank(np.zeros((3, 3))) == 0.0


def test_effective_rank_bool_eps_rejected():
    with pytest.raises(TypeError):
        effective_rank(np.eye(2), eps=True)


def test_effective_rank_non_finite_eps_rejected():
    with pytest.raises(ValueError):
        effective_rank(np.eye(2), eps=float("nan"))


def test_effective_rank_negative_eps_rejected():
    with pytest.raises(ValueError):
        effective_rank(np.eye(2), eps=-1.0)


# ---------- compute_layer_diffs ----------


def test_compute_layer_diffs_identical_zero():
    a = {"foo": np.ones((2, 2))}
    b = {"foo": np.ones((2, 2))}
    diffs, only_a, only_b = compute_layer_diffs(a, b)
    assert len(diffs) == 1
    assert diffs[0].frobenius == 0.0
    assert diffs[0].relative == 0.0
    assert only_a == ()
    assert only_b == ()


def test_compute_layer_diffs_known_norm():
    a = {"w": np.zeros((2, 2))}
    b = {"w": np.array([[3.0, 4.0], [0.0, 0.0]])}
    diffs, _, _ = compute_layer_diffs(a, b)
    # Frobenius = sqrt(9+16) = 5
    assert diffs[0].frobenius == pytest.approx(5.0)
    assert diffs[0].relative == pytest.approx(1.0)


def test_compute_layer_diffs_skips_shape_mismatch():
    a = {"w": np.zeros((4, 4))}
    b = {"w": np.zeros((2, 2))}
    diffs, _, _ = compute_layer_diffs(a, b)
    assert diffs == ()  # shape mismatch silently skipped


def test_compute_layer_diffs_partitions():
    a = {"only_a": np.zeros(2), "shared": np.zeros(2)}
    b = {"only_b": np.zeros(2), "shared": np.zeros(2)}
    diffs, only_a, only_b = compute_layer_diffs(a, b)
    assert [d.name for d in diffs] == ["shared"]
    assert only_a == ("only_a",)
    assert only_b == ("only_b",)


def test_compute_layer_diffs_rejects_non_mapping():
    with pytest.raises(TypeError):
        compute_layer_diffs([], {})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        compute_layer_diffs({}, [])  # type: ignore[arg-type]


def test_compute_layer_diffs_too_many_tensors():
    big = {f"t{i}": np.zeros(1) for i in range(10_001)}
    small: dict[str, np.ndarray] = {}
    with pytest.raises(ValueError, match=">10000"):
        compute_layer_diffs(big, small)


# ---------- render report ----------


def _sample_report() -> AdapterDiffReport:
    return AdapterDiffReport(
        adapter_a="A",
        adapter_b="B",
        per_layer=(
            LayerDiff(name="layer1", frobenius=2.0, norm_a=4.0, norm_b=3.0, relative=0.5),
        ),
        top_changed=("layer1",),
        effective_rank_a=8.0,
        effective_rank_b=8.0,
        shared_layers=1,
        only_in_a=(),
        only_in_b=(),
    )


def test_render_report_json_roundtrip():
    report = _sample_report()
    text = render_report_json(report)
    parsed = json.loads(text)
    assert parsed["adapter_a"] == "A"
    assert parsed["adapter_b"] == "B"
    assert parsed["shared_layers"] == 1
    assert parsed["top_changed"] == ["layer1"]
    assert parsed["per_layer"][0]["frobenius"] == 2.0


def test_render_report_json_rejects_non_report():
    with pytest.raises(TypeError):
        render_report_json({"foo": "bar"})  # type: ignore[arg-type]


def test_render_report_markdown_renders():
    text = render_report_markdown(_sample_report())
    assert "# Adapter diff: A vs B" in text
    assert "layer1" in text
    assert text.endswith("\n")


def test_render_report_markdown_only_lists():
    report = AdapterDiffReport(
        adapter_a="x", adapter_b="y",
        per_layer=(), top_changed=(),
        effective_rank_a=None, effective_rank_b=None,
        shared_layers=0,
        only_in_a=("foo",),
        only_in_b=("bar",),
    )
    text = render_report_markdown(report)
    assert "Only in A" in text
    assert "Only in B" in text


def test_render_report_markdown_rejects_non_report():
    with pytest.raises(TypeError):
        render_report_markdown(None)  # type: ignore[arg-type]


def test_frozen_dataclasses():
    import dataclasses
    diff = LayerDiff(name="x", frobenius=1.0, norm_a=1.0, norm_b=1.0, relative=1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        diff.frobenius = 2.0  # type: ignore[misc]


# ---------- compute_adapter_diff (end-to-end with safetensors fixture) ----------


def _write_safetensors(dir_path: Path, weights: dict) -> None:
    """Helper: write adapter_model.safetensors + minimal adapter_config.json."""
    pytest.importorskip("safetensors")
    from safetensors.numpy import save_file

    dir_path.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(dir_path / "adapter_model.safetensors"))
    (dir_path / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 8}), encoding="utf-8"
    )


def test_compute_adapter_diff_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("safetensors")
    monkeypatch.chdir(tmp_path)
    weights_a = {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight":
            np.ones((4, 8), dtype=np.float32),
        "base_model.model.layers.0.self_attn.v_proj.lora_A.weight":
            np.zeros((4, 8), dtype=np.float32),
    }
    weights_b = {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight":
            np.ones((4, 8), dtype=np.float32),  # identical
        "base_model.model.layers.0.self_attn.v_proj.lora_A.weight":
            np.ones((4, 8), dtype=np.float32),  # different
    }
    _write_safetensors(tmp_path / "a", weights_a)
    _write_safetensors(tmp_path / "b", weights_b)

    report = compute_adapter_diff("a", "b", top_k=5)
    assert isinstance(report, AdapterDiffReport)
    assert report.shared_layers == 2
    # v_proj should be the top-changed projection
    assert report.top_changed[0].endswith("v_proj.lora_A.weight")


def test_compute_adapter_diff_outside_cwd_rejected(tmp_path):
    with pytest.raises(ValueError):
        compute_adapter_diff(str(tmp_path), "b")


def test_compute_adapter_diff_bool_top_k_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    with pytest.raises(TypeError):
        compute_adapter_diff("a", "b", top_k=True)  # type: ignore[arg-type]


def test_compute_adapter_diff_top_k_out_of_range(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    with pytest.raises(ValueError):
        compute_adapter_diff("a", "b", top_k=0)
    with pytest.raises(ValueError):
        compute_adapter_diff("a", "b", top_k=201)


def test_compute_adapter_diff_top_k_lower_bound(tmp_path, monkeypatch):
    """top_k=1 should be accepted (exact lower bound)."""
    pytest.importorskip("safetensors")
    monkeypatch.chdir(tmp_path)
    _write_safetensors(tmp_path / "a", {"w": np.ones((2, 2), dtype=np.float32)})
    _write_safetensors(tmp_path / "b", {"w": np.zeros((2, 2), dtype=np.float32)})
    report = compute_adapter_diff("a", "b", top_k=1)
    assert len(report.top_changed) == 1


@pytest.mark.skipif(os.name == "nt", reason="POSIX-only symlink semantics")
def test_compute_adapter_diff_rejects_symlinked_weights(tmp_path, monkeypatch):
    """Symlink at adapter_model.safetensors must be rejected (TOCTOU)."""
    pytest.importorskip("safetensors")
    monkeypatch.chdir(tmp_path)
    _write_safetensors(tmp_path / "real", {"w": np.zeros((2, 2), dtype=np.float32)})
    (tmp_path / "fake").mkdir()
    os.symlink(
        str(tmp_path / "real" / "adapter_model.safetensors"),
        str(tmp_path / "fake" / "adapter_model.safetensors"),
    )
    (tmp_path / "fake" / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA"}), encoding="utf-8"
    )
    _write_safetensors(tmp_path / "b", {"w": np.zeros((2, 2), dtype=np.float32)})
    with pytest.raises(ValueError, match="symlink"):
        compute_adapter_diff("fake", "b")


def test_no_top_level_torch_import():
    """Lazy-import policy: adapter_diff must not import torch at module level."""
    src = (Path(__file__).parent.parent / "soup_cli" / "utils" / "adapter_diff.py"
           ).read_text(encoding="utf-8")
    # Only allowed inside def bodies
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("import torch") or stripped.startswith("from torch"):
            indent = len(line) - len(stripped)
            assert indent > 0, f"top-level torch import found: {line}"


def test_compute_adapter_diff_missing_safetensors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    with pytest.raises(FileNotFoundError):
        compute_adapter_diff("a", "b")


def test_compute_adapter_diff_bin_format_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "adapter_model.bin").write_bytes(b"x")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "adapter_model.bin").write_bytes(b"x")
    with pytest.raises(RuntimeError, match=".bin format not supported"):
        compute_adapter_diff("a", "b")


# ---------- CLI ----------


def test_adapters_diff_help():
    result = runner.invoke(soup_app, ["adapters", "diff", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "--top-k" in result.output
    assert "--format" in result.output


def test_adapters_diff_table_format(tmp_path, monkeypatch):
    pytest.importorskip("safetensors")
    monkeypatch.chdir(tmp_path)
    _write_safetensors(tmp_path / "a", {"w": np.ones((2, 2), dtype=np.float32)})
    _write_safetensors(tmp_path / "b", {"w": np.zeros((2, 2), dtype=np.float32)})
    result = runner.invoke(soup_app, ["adapters", "diff", "a", "b"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "Adapter diff" in result.output


def test_adapters_diff_json_output(tmp_path, monkeypatch):
    pytest.importorskip("safetensors")
    monkeypatch.chdir(tmp_path)
    _write_safetensors(tmp_path / "a", {"w": np.ones((2, 2), dtype=np.float32)})
    _write_safetensors(tmp_path / "b", {"w": np.zeros((2, 2), dtype=np.float32)})
    result = runner.invoke(soup_app, [
        "adapters", "diff", "a", "b",
        "--format", "json",
        "--output", "report.json",
    ])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    parsed = json.loads((tmp_path / "report.json").read_text())
    assert parsed["shared_layers"] == 1


def test_adapters_diff_markdown_output(tmp_path, monkeypatch):
    pytest.importorskip("safetensors")
    monkeypatch.chdir(tmp_path)
    _write_safetensors(tmp_path / "a", {"w": np.ones((2, 2), dtype=np.float32)})
    _write_safetensors(tmp_path / "b", {"w": np.zeros((2, 2), dtype=np.float32)})
    result = runner.invoke(soup_app, [
        "adapters", "diff", "a", "b",
        "--format", "markdown",
        "--output", "report.md",
    ])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = (tmp_path / "report.md").read_text()
    assert "# Adapter diff" in text


def test_adapters_diff_unknown_format(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    result = runner.invoke(soup_app, [
        "adapters", "diff", "a", "b", "--format", "yaml",
    ])
    assert result.exit_code == 2
    assert "Unknown --format" in result.output


def test_adapters_diff_output_requires_non_table(tmp_path, monkeypatch):
    pytest.importorskip("safetensors")
    monkeypatch.chdir(tmp_path)
    _write_safetensors(tmp_path / "a", {"w": np.ones((2, 2), dtype=np.float32)})
    _write_safetensors(tmp_path / "b", {"w": np.zeros((2, 2), dtype=np.float32)})
    result = runner.invoke(soup_app, [
        "adapters", "diff", "a", "b", "--output", "out.txt",
    ])
    assert result.exit_code == 2
    assert "requires --format" in result.output


def test_adapters_diff_output_outside_cwd_rejected(tmp_path, monkeypatch):
    pytest.importorskip("safetensors")
    monkeypatch.chdir(tmp_path)
    _write_safetensors(tmp_path / "a", {"w": np.ones((2, 2), dtype=np.float32)})
    _write_safetensors(tmp_path / "b", {"w": np.zeros((2, 2), dtype=np.float32)})
    outside = os.path.join(os.path.dirname(str(tmp_path)), "outside.json")
    result = runner.invoke(soup_app, [
        "adapters", "diff", "a", "b",
        "--format", "json", "--output", outside,
    ])
    assert result.exit_code != 0
