"""v0.66.0 Part A — SAE feature diff tests (TDD).

Sparse Autoencoder pre/post feature attribution. The SAE itself is loaded
lazily (an encode-only weight matrix w_enc); the diff math is pure-numpy
so tests can exercise everything on CPU without HF Hub access.

Public surface under test:

- ``SaeFeatureChange`` frozen dataclass — per-feature pre/post activation diff
- ``SaeFeatureDiffReport`` frozen dataclass — top-k changed features + summary
- ``encode_activations(activations, w_enc, b_enc=None)`` — sparse-ReLU encode
- ``compute_feature_diff(pre, post, *, top_k)`` — pure math kernel
- ``load_sae_weights(path)`` — safetensors + containment + symlink rejection
- ``HF_HUB_ALLOWLIST`` — closed frozenset of known SAE repos
- ``validate_sae_repo(name)`` — allowlist + null-byte rejection
- ``compute_sae_diff(pre_acts, post_acts, sae_weights, *, top_k)`` — orchestrator
- ``render_report_json`` / ``render_report_markdown``
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _chdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield tmp_path


# ---------------------------------------------------------------------------
# Public surface presence
# ---------------------------------------------------------------------------


def test_module_imports():
    """Module exists and exposes the documented surface."""
    from soup_cli.utils import sae_diff

    for name in (
        "SaeFeatureChange",
        "SaeFeatureDiffReport",
        "encode_activations",
        "compute_feature_diff",
        "load_sae_weights",
        "compute_sae_diff",
        "render_report_json",
        "render_report_markdown",
        "HF_HUB_ALLOWLIST",
        "validate_sae_repo",
    ):
        assert hasattr(sae_diff, name), name


def test_hf_hub_allowlist_is_frozenset():
    from soup_cli.utils.sae_diff import HF_HUB_ALLOWLIST

    assert isinstance(HF_HUB_ALLOWLIST, frozenset)
    assert len(HF_HUB_ALLOWLIST) >= 1
    # Known popular SAE families
    expected_substrings = (
        "gemma-scope",  # DeepMind Gemma 2 SAEs
        "sae",
    )
    joined = " ".join(HF_HUB_ALLOWLIST)
    for sub in expected_substrings:
        assert sub.lower() in joined.lower(), f"missing {sub}"


# ---------------------------------------------------------------------------
# validate_sae_repo
# ---------------------------------------------------------------------------


def test_validate_sae_repo_happy():
    from soup_cli.utils.sae_diff import HF_HUB_ALLOWLIST, validate_sae_repo

    name = next(iter(HF_HUB_ALLOWLIST))
    assert validate_sae_repo(name) == name


def test_validate_sae_repo_case_insensitive():
    from soup_cli.utils.sae_diff import HF_HUB_ALLOWLIST, validate_sae_repo

    name = next(iter(HF_HUB_ALLOWLIST))
    upper = name.upper()
    canon = validate_sae_repo(upper)
    assert canon == name  # canonicalised lower


def test_validate_sae_repo_rejects_unknown():
    from soup_cli.utils.sae_diff import validate_sae_repo

    with pytest.raises(ValueError, match="not in HF_HUB_ALLOWLIST"):
        validate_sae_repo("malicious/sae-x")


def test_validate_sae_repo_rejects_bool():
    from soup_cli.utils.sae_diff import validate_sae_repo

    with pytest.raises(TypeError):
        validate_sae_repo(True)


def test_validate_sae_repo_rejects_non_string():
    from soup_cli.utils.sae_diff import validate_sae_repo

    with pytest.raises(TypeError):
        validate_sae_repo(42)


def test_validate_sae_repo_rejects_empty():
    from soup_cli.utils.sae_diff import validate_sae_repo

    with pytest.raises(ValueError):
        validate_sae_repo("")


def test_validate_sae_repo_rejects_null_byte():
    from soup_cli.utils.sae_diff import HF_HUB_ALLOWLIST, validate_sae_repo

    name = next(iter(HF_HUB_ALLOWLIST))
    with pytest.raises(ValueError, match="null"):
        validate_sae_repo(name + "\x00")


def test_validate_sae_repo_rejects_oversize():
    from soup_cli.utils.sae_diff import validate_sae_repo

    with pytest.raises(ValueError):
        validate_sae_repo("a" * 1000)


# ---------------------------------------------------------------------------
# encode_activations — sparse ReLU encoder
# ---------------------------------------------------------------------------


def test_encode_activations_returns_correct_shape():
    from soup_cli.utils.sae_diff import encode_activations

    # 4 tokens × 8-dim activation -> 4 × 16 (16 SAE features)
    activations = np.random.RandomState(0).randn(4, 8).astype(np.float32)
    w_enc = np.random.RandomState(1).randn(8, 16).astype(np.float32)
    features = encode_activations(activations, w_enc)
    assert features.shape == (4, 16)


def test_encode_activations_is_relu():
    from soup_cli.utils.sae_diff import encode_activations

    # Crafted: negative pre-activation -> zero output
    activations = np.array([[-1.0, -1.0]], dtype=np.float32)
    w_enc = np.array([[1.0], [1.0]], dtype=np.float32)
    features = encode_activations(activations, w_enc)
    assert features.shape == (1, 1)
    assert features[0, 0] == 0.0


def test_encode_activations_with_bias():
    from soup_cli.utils.sae_diff import encode_activations

    activations = np.array([[1.0, 1.0]], dtype=np.float32)
    w_enc = np.array([[1.0], [1.0]], dtype=np.float32)  # 2 -> 1
    b_enc = np.array([10.0], dtype=np.float32)
    features = encode_activations(activations, w_enc, b_enc=b_enc)
    # 1*1 + 1*1 + 10 = 12
    assert features[0, 0] == pytest.approx(12.0)


def test_encode_activations_rejects_shape_mismatch():
    from soup_cli.utils.sae_diff import encode_activations

    activations = np.zeros((4, 8), dtype=np.float32)
    w_enc = np.zeros((10, 16), dtype=np.float32)  # 10 != 8
    with pytest.raises(ValueError, match="shape"):
        encode_activations(activations, w_enc)


def test_encode_activations_rejects_non_2d_activations():
    from soup_cli.utils.sae_diff import encode_activations

    activations = np.zeros((4,), dtype=np.float32)
    w_enc = np.zeros((4, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="2D"):
        encode_activations(activations, w_enc)


def test_encode_activations_rejects_non_2d_w_enc():
    from soup_cli.utils.sae_diff import encode_activations

    activations = np.zeros((4, 8), dtype=np.float32)
    w_enc = np.zeros((8,), dtype=np.float32)
    with pytest.raises(ValueError, match="2D"):
        encode_activations(activations, w_enc)


def test_encode_activations_rejects_non_array():
    from soup_cli.utils.sae_diff import encode_activations

    with pytest.raises(TypeError):
        encode_activations("not an array", np.zeros((1, 1)))


# ---------------------------------------------------------------------------
# compute_feature_diff — pure math kernel
# ---------------------------------------------------------------------------


def test_compute_feature_diff_zero_when_identical():
    from soup_cli.utils.sae_diff import compute_feature_diff

    feats = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    report = compute_feature_diff(feats, feats, top_k=3)
    # All deltas zero
    for change in report.changes:
        assert change.delta == 0.0


def test_compute_feature_diff_finds_largest_change():
    from soup_cli.utils.sae_diff import compute_feature_diff

    pre = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    post = np.array([[1.0, 5.0, 1.0]], dtype=np.float32)
    report = compute_feature_diff(pre, post, top_k=1)
    assert len(report.changes) == 1
    top = report.changes[0]
    assert top.feature_id == 1
    assert top.delta == pytest.approx(4.0)


def test_compute_feature_diff_top_k_caps():
    from soup_cli.utils.sae_diff import compute_feature_diff

    pre = np.zeros((1, 100), dtype=np.float32)
    post = np.arange(100, dtype=np.float32).reshape(1, 100)
    report = compute_feature_diff(pre, post, top_k=5)
    assert len(report.changes) == 5
    # Largest delta = feature 99
    assert report.changes[0].feature_id == 99
    assert report.changes[0].delta == pytest.approx(99.0)


def test_compute_feature_diff_rejects_shape_mismatch():
    from soup_cli.utils.sae_diff import compute_feature_diff

    pre = np.zeros((1, 5), dtype=np.float32)
    post = np.zeros((1, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        compute_feature_diff(pre, post, top_k=1)


def test_compute_feature_diff_rejects_bool_top_k():
    from soup_cli.utils.sae_diff import compute_feature_diff

    pre = np.zeros((1, 5), dtype=np.float32)
    with pytest.raises(TypeError):
        compute_feature_diff(pre, pre, top_k=True)


def test_compute_feature_diff_rejects_zero_top_k():
    from soup_cli.utils.sae_diff import compute_feature_diff

    pre = np.zeros((1, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        compute_feature_diff(pre, pre, top_k=0)


def test_compute_feature_diff_rejects_oversize_top_k():
    from soup_cli.utils.sae_diff import compute_feature_diff

    pre = np.zeros((1, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        compute_feature_diff(pre, pre, top_k=100_000_000)


def test_compute_feature_diff_top_k_larger_than_features_clamps():
    from soup_cli.utils.sae_diff import compute_feature_diff

    pre = np.zeros((1, 3), dtype=np.float32)
    post = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    # top_k=10 but only 3 features -> 3 reports
    report = compute_feature_diff(pre, post, top_k=10)
    assert len(report.changes) == 3


def test_compute_feature_diff_mean_pooled_over_tokens():
    """Per-feature delta is mean over the token dim."""
    from soup_cli.utils.sae_diff import compute_feature_diff

    # 3 tokens, 1 feature. Pre: [0, 0, 0]. Post: [1, 2, 3]. Mean delta = 2.
    pre = np.zeros((3, 1), dtype=np.float32)
    post = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    report = compute_feature_diff(pre, post, top_k=1)
    assert report.changes[0].delta == pytest.approx(2.0)


def test_feature_diff_report_frozen():
    from soup_cli.utils.sae_diff import SaeFeatureDiffReport

    report = SaeFeatureDiffReport(
        num_features=10,
        num_tokens=5,
        changes=tuple(),
        l2_drift=0.0,
    )
    with pytest.raises((AttributeError, Exception)):
        report.num_features = 99  # type: ignore[misc]


def test_feature_change_frozen():
    from soup_cli.utils.sae_diff import SaeFeatureChange

    change = SaeFeatureChange(
        feature_id=0,
        delta=1.0,
        pre_mean=0.0,
        post_mean=1.0,
    )
    with pytest.raises((AttributeError, Exception)):
        change.delta = 2.0  # type: ignore[misc]


def test_feature_change_rejects_negative_feature_id():
    from soup_cli.utils.sae_diff import SaeFeatureChange

    with pytest.raises(ValueError):
        SaeFeatureChange(feature_id=-1, delta=0.0, pre_mean=0.0, post_mean=0.0)


def test_feature_change_rejects_bool_feature_id():
    from soup_cli.utils.sae_diff import SaeFeatureChange

    with pytest.raises(TypeError):
        SaeFeatureChange(feature_id=True, delta=0.0, pre_mean=0.0, post_mean=0.0)


def test_feature_change_rejects_non_finite_delta():
    from soup_cli.utils.sae_diff import SaeFeatureChange

    with pytest.raises(ValueError):
        SaeFeatureChange(
            feature_id=0, delta=float("nan"), pre_mean=0.0, post_mean=0.0
        )


def test_feature_diff_report_l2_drift_computed():
    from soup_cli.utils.sae_diff import compute_feature_diff

    pre = np.zeros((1, 3), dtype=np.float32)
    post = np.array([[3.0, 4.0, 0.0]], dtype=np.float32)
    # L2 = sqrt(9+16+0) = 5
    report = compute_feature_diff(pre, post, top_k=3)
    assert report.l2_drift == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# load_sae_weights — containment + symlink rejection
# ---------------------------------------------------------------------------


def test_load_sae_weights_outside_cwd_rejected(tmp_path):
    from soup_cli.utils.sae_diff import load_sae_weights

    # tmp_path/.. exists but is outside cwd
    outside = str(tmp_path.parent / "evil")
    with pytest.raises(ValueError, match="under cwd"):
        load_sae_weights(outside)


def test_load_sae_weights_missing_file_raises(tmp_path):
    from soup_cli.utils.sae_diff import load_sae_weights

    with pytest.raises(FileNotFoundError):
        load_sae_weights("nonexistent_sae.safetensors")


def test_load_sae_weights_null_byte_rejected():
    from soup_cli.utils.sae_diff import load_sae_weights

    with pytest.raises(ValueError):
        load_sae_weights("bad\x00.safetensors")


def test_load_sae_weights_non_string_rejected():
    from soup_cli.utils.sae_diff import load_sae_weights

    with pytest.raises(TypeError):
        load_sae_weights(42)


def test_load_sae_weights_empty_path_rejected():
    from soup_cli.utils.sae_diff import load_sae_weights

    with pytest.raises(ValueError):
        load_sae_weights("")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink test")
def test_load_sae_weights_symlink_rejected(tmp_path):
    from soup_cli.utils.sae_diff import load_sae_weights

    target = tmp_path / "real.safetensors"
    target.write_bytes(b"x")
    sym = tmp_path / "sae.safetensors"
    os.symlink(str(target), str(sym))
    # symlink is rejected via os.lstat + S_ISLNK
    with pytest.raises(ValueError, match="symlink"):
        load_sae_weights("sae.safetensors")


# ---------------------------------------------------------------------------
# compute_sae_diff — end-to-end orchestrator
# ---------------------------------------------------------------------------


def test_compute_sae_diff_happy(tmp_path):
    from soup_cli.utils.sae_diff import compute_sae_diff

    # Synthetic activations: 4 tokens, 8 model dims
    pre = np.random.RandomState(0).randn(4, 8).astype(np.float32)
    post = pre + 0.5  # uniform shift
    w_enc = np.random.RandomState(1).randn(8, 32).astype(np.float32)
    sae = {"W_enc": w_enc, "b_enc": None}
    report = compute_sae_diff(pre, post, sae, top_k=5)
    assert report.num_tokens == 4
    assert report.num_features == 32
    assert len(report.changes) == 5


def test_compute_sae_diff_rejects_non_dict_sae():
    from soup_cli.utils.sae_diff import compute_sae_diff

    pre = np.zeros((1, 8), dtype=np.float32)
    post = np.zeros((1, 8), dtype=np.float32)
    with pytest.raises(TypeError):
        compute_sae_diff(pre, post, "not a dict", top_k=1)


def test_compute_sae_diff_missing_w_enc_raises():
    from soup_cli.utils.sae_diff import compute_sae_diff

    pre = np.zeros((1, 8), dtype=np.float32)
    post = np.zeros((1, 8), dtype=np.float32)
    with pytest.raises(KeyError):
        compute_sae_diff(pre, post, {}, top_k=1)


def test_compute_sae_diff_rejects_pre_post_mismatch():
    from soup_cli.utils.sae_diff import compute_sae_diff

    pre = np.zeros((4, 8), dtype=np.float32)
    post = np.zeros((4, 10), dtype=np.float32)
    w_enc = np.zeros((8, 16), dtype=np.float32)
    with pytest.raises(ValueError):
        compute_sae_diff(pre, post, {"W_enc": w_enc}, top_k=1)


# ---------------------------------------------------------------------------
# Render — JSON + Markdown
# ---------------------------------------------------------------------------


def test_render_report_json_roundtrip():
    from soup_cli.utils.sae_diff import (
        SaeFeatureChange,
        SaeFeatureDiffReport,
        render_report_json,
    )

    report = SaeFeatureDiffReport(
        num_features=10,
        num_tokens=4,
        l2_drift=2.5,
        changes=(
            SaeFeatureChange(feature_id=3, delta=1.5, pre_mean=0.0, post_mean=1.5),
            SaeFeatureChange(feature_id=7, delta=0.5, pre_mean=0.0, post_mean=0.5),
        ),
    )
    text = render_report_json(report)
    payload = json.loads(text)
    assert payload["num_features"] == 10
    assert payload["num_tokens"] == 4
    assert payload["l2_drift"] == pytest.approx(2.5)
    assert len(payload["changes"]) == 2
    assert payload["changes"][0]["feature_id"] == 3


def test_render_report_json_rejects_non_report():
    from soup_cli.utils.sae_diff import render_report_json

    with pytest.raises(TypeError):
        render_report_json("not a report")


def test_render_report_markdown_has_table():
    from soup_cli.utils.sae_diff import (
        SaeFeatureChange,
        SaeFeatureDiffReport,
        render_report_markdown,
    )

    report = SaeFeatureDiffReport(
        num_features=10,
        num_tokens=4,
        l2_drift=2.5,
        changes=(
            SaeFeatureChange(feature_id=3, delta=1.5, pre_mean=0.0, post_mean=1.5),
        ),
    )
    text = render_report_markdown(report)
    assert "SAE feature diff" in text
    assert "3" in text  # feature id
    assert "1.5" in text  # delta


def test_render_report_markdown_rejects_non_report():
    from soup_cli.utils.sae_diff import render_report_markdown

    with pytest.raises(TypeError):
        render_report_markdown(None)


def test_render_report_markdown_empty_changes():
    from soup_cli.utils.sae_diff import SaeFeatureDiffReport, render_report_markdown

    report = SaeFeatureDiffReport(
        num_features=0,
        num_tokens=0,
        l2_drift=0.0,
        changes=tuple(),
    )
    text = render_report_markdown(report)
    assert "no" in text.lower() or "empty" in text.lower()


# ---------------------------------------------------------------------------
# No heavy top-level imports
# ---------------------------------------------------------------------------


def test_no_heavy_top_level_imports():
    """Source-grep regression: lazy-import torch/transformers/safetensors.

    Top-level = column 0 (no leading whitespace). Imports inside function
    bodies are indented, so they pass this gate.
    """
    import inspect

    from soup_cli.utils import sae_diff

    source = inspect.getsource(sae_diff)
    # Only inspect top-level lines (no leading whitespace).
    top_level_imports = [
        line for line in source.splitlines()
        if (line.startswith("import ") or line.startswith("from "))
    ]
    forbidden = ("torch", "transformers", "peft", "safetensors")
    for line in top_level_imports:
        for bad in forbidden:
            assert bad not in line, f"top-level {bad} import: {line!r}"
