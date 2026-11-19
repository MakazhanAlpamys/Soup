"""``soup adapters diff`` compares LoRA modules on their effective delta (#3103).

The report's "ΔW Frobenius" column must be ``||ΔW_a - ΔW_b||_F`` with
``ΔW = scaling · B @ A`` per module: LoRA factors are not unique, two adapters
of different rank share no factor shape, and the same factors under a different
``lora_alpha`` are a different ``ΔW``.
"""

from __future__ import annotations

import json
import math
import re

import numpy as np
import pytest

from tests.conftest import strip_ansi

pytest.importorskip("safetensors")

_STEM = "base_model.model.model.layers.0.self_attn.q_proj"
_A = f"{_STEM}.lora_A.weight"
_B = f"{_STEM}.lora_B.weight"
_DIM = 6


def _factors(r, value, slot=0):
    """Rank-``r`` factors of ``value · e_slot e_slot^T`` on a ``_DIM`` x ``_DIM`` layer."""
    a = np.zeros((r, _DIM), dtype=np.float32)
    b = np.zeros((_DIM, r), dtype=np.float32)
    a[slot, slot] = 1.0
    b[slot, slot] = value
    return a, b


def _write(path, tensors, config=True, **fields):
    from safetensors.numpy import save_file

    path.mkdir(parents=True)
    save_file(tensors, str(path / "adapter_model.safetensors"))
    if config:
        cfg = {"peft_type": "LORA", "target_modules": ["q_proj"], **fields}
        (path / "adapter_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return path.name


def _diff(tmp_path, monkeypatch, left, right):
    from soup_cli.utils.adapter_diff import compute_adapter_diff

    monkeypatch.chdir(tmp_path)
    return compute_adapter_diff(left, right)


def _f64(array):
    return np.asarray(array, dtype=np.float64)


def test_lora_alpha_change_is_a_change(tmp_path, monkeypatch):
    a, b = _factors(2, 1.0)
    _write(tmp_path / "x", {_A: a, _B: b}, r=2, lora_alpha=2)  # ΔW = 1 · e0 e0^T
    _write(tmp_path / "y", {_A: a, _B: b}, r=2, lora_alpha=4)  # ΔW = 2 · e0 e0^T
    report = _diff(tmp_path, monkeypatch, "x", "y")
    assert [d.name for d in report.per_layer] == [_STEM]
    row = report.per_layer[0]
    assert row.norm_a == pytest.approx(1.0)
    assert row.norm_b == pytest.approx(2.0)
    assert row.frobenius == pytest.approx(1.0)
    assert row.relative == pytest.approx(0.5)


def test_different_ranks_are_compared(tmp_path, monkeypatch):
    a2, b2 = _factors(2, 1.0)
    a4, b4 = _factors(4, 3.0)
    _write(tmp_path / "x", {_A: a2, _B: b2}, r=2, lora_alpha=2)  # ΔW = e0 e0^T
    _write(tmp_path / "y", {_A: a4, _B: b4}, r=4, lora_alpha=4)  # ΔW = 3 e0 e0^T
    report = _diff(tmp_path, monkeypatch, "x", "y")
    assert report.shared_layers == 1
    assert report.per_layer[0].frobenius == pytest.approx(2.0)
    assert report.top_changed == (_STEM,)
    assert report.only_in_a == () and report.only_in_b == ()


def test_reparameterised_factors_are_no_change(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, _DIM)).astype(np.float32)
    b = rng.standard_normal((_DIM, 4)).astype(np.float32)
    # (2B, A/2) is the same ΔW as (B, A); both scale exactly in float32.
    _write(tmp_path / "x", {_A: a, _B: b}, r=4, lora_alpha=8)
    _write(tmp_path / "y", {_A: a / 2, _B: b * 2}, r=4, lora_alpha=8)
    report = _diff(tmp_path, monkeypatch, "x", "y")
    row = report.per_layer[0]
    assert row.frobenius <= 1e-6 * row.norm_a
    expected = 2.0 * np.linalg.norm(_f64(b) @ _f64(a))
    assert row.norm_a == pytest.approx(expected)


def test_rslora_scaling_is_applied(tmp_path, monkeypatch):
    a, b = _factors(4, 1.0)
    _write(tmp_path / "x", {_A: a, _B: b}, r=4, lora_alpha=4)  # 4 / 4 = 1
    _write(tmp_path / "y", {_A: a, _B: b}, r=4, lora_alpha=4, use_rslora=True)  # 4 / 2 = 2
    report = _diff(tmp_path, monkeypatch, "x", "y")
    assert report.per_layer[0].norm_b == pytest.approx(2.0)
    assert report.per_layer[0].frobenius == pytest.approx(1.0)


def test_identical_adapters_diff_to_exact_zero(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    tensors = {
        _A: rng.standard_normal((4, _DIM)).astype(np.float32),
        _B: rng.standard_normal((_DIM, 4)).astype(np.float32),
    }
    _write(tmp_path / "x", tensors, r=4, lora_alpha=8)
    _write(tmp_path / "y", tensors, r=4, lora_alpha=8)
    report = _diff(tmp_path, monkeypatch, "x", "y")
    assert report.per_layer[0].frobenius == 0.0


def test_untrained_adapter_has_zero_effective_rank(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    a = rng.standard_normal((4, _DIM)).astype(np.float32)
    zero_b = np.zeros((_DIM, 4), dtype=np.float32)
    _write(tmp_path / "x", {_A: a, _B: zero_b}, r=4, lora_alpha=8)  # ΔW = 0
    _write(tmp_path / "y", {_A: a, _B: zero_b}, r=4, lora_alpha=8)
    report = _diff(tmp_path, monkeypatch, "x", "y")
    assert report.effective_rank_a == 0.0
    assert report.per_layer[0].norm_a == 0.0


# ---------- dense reference: the rank-sized blocks against the (out, in) product ----------

_OUT, _IN = 7, 5  # neither square nor equal to a rank, so a transposed block cannot fit


@pytest.mark.parametrize(("rank_a", "rank_b"), [(2, 4), (3, 3), (1, 5)])
def test_matches_a_dense_reference(tmp_path, monkeypatch, rank_a, rank_b):
    rng = np.random.default_rng(10 * rank_a + rank_b)
    a_x = rng.standard_normal((rank_a, _IN)).astype(np.float32)
    b_x = rng.standard_normal((_OUT, rank_a)).astype(np.float32)
    a_y = rng.standard_normal((rank_b, _IN)).astype(np.float32)
    b_y = rng.standard_normal((_OUT, rank_b)).astype(np.float32)
    _write(tmp_path / "x", {_A: a_x, _B: b_x}, r=rank_a, lora_alpha=3)
    _write(tmp_path / "y", {_A: a_y, _B: b_y}, r=rank_b, lora_alpha=5, use_rslora=True)
    report = _diff(tmp_path, monkeypatch, "x", "y")

    dense_x = (3.0 / rank_a) * (_f64(b_x) @ _f64(a_x))
    dense_y = (5.0 / math.sqrt(rank_b)) * (_f64(b_y) @ _f64(a_y))
    assert [d.name for d in report.per_layer] == [_STEM]
    row = report.per_layer[0]
    assert row.frobenius == pytest.approx(np.linalg.norm(dense_x - dense_y), rel=1e-9)
    assert row.norm_a == pytest.approx(np.linalg.norm(dense_x), rel=1e-9)
    assert row.norm_b == pytest.approx(np.linalg.norm(dense_y), rel=1e-9)

    def dense_effective_rank(matrix):
        singular = np.linalg.svd(matrix, compute_uv=False)
        probs = singular / singular.sum()
        probs = probs[probs > 1e-12]
        return math.exp(-float(np.sum(probs * np.log(probs))))

    assert report.effective_rank_a == pytest.approx(dense_effective_rank(dense_x), rel=1e-9)
    assert report.effective_rank_b == pytest.approx(dense_effective_rank(dense_y), rel=1e-9)


# ---------- the scaling ----------

_STEM_1 = "base_model.model.model.layers.1.self_attn.q_proj"


def test_rank_and_alpha_pattern_apply_to_their_module(tmp_path, monkeypatch):
    # An anchored key matches only the module name PEFT uses, without the saved
    # "base_model.model." prefix, so it applies only through peft_module_name.
    key = "^model.layers.0.self_attn.q_proj"
    a_0, b_0 = _factors(2, 1.0)  # layer 0 at its pattern rank 2
    a_1, b_1 = _factors(4, 1.0)  # layer 1 at the default rank 4
    _write(
        tmp_path / "x",
        {_A: a_0, _B: b_0, f"{_STEM_1}.lora_A.weight": a_1, f"{_STEM_1}.lora_B.weight": b_1},
        r=4, lora_alpha=8, rank_pattern={key: 2}, alpha_pattern={key: 6},
    )
    _write(
        tmp_path / "y",
        {_A: a_1, _B: b_1, f"{_STEM_1}.lora_A.weight": a_1, f"{_STEM_1}.lora_B.weight": b_1},
        r=4, lora_alpha=4,  # scaling 1 on both modules
    )
    report = _diff(tmp_path, monkeypatch, "x", "y")
    rows = {d.name: d for d in report.per_layer}
    assert sorted(rows) == [_STEM, _STEM_1]
    assert rows[_STEM].norm_a == pytest.approx(3.0)  # alpha_pattern 6 / rank_pattern 2
    assert rows[_STEM].frobenius == pytest.approx(2.0)
    assert rows[_STEM_1].norm_a == pytest.approx(2.0)  # lora_alpha 8 / r 4
    assert rows[_STEM_1].frobenius == pytest.approx(1.0)


@pytest.mark.parametrize(
    "fields",
    [
        None,  # no adapter_config.json
        {},
        {"r": 2},
        {"lora_alpha": 2},
        {"r": 0, "lora_alpha": 2},
        {"r": "2", "lora_alpha": 2},
    ],
    ids=["no-config", "neither", "no-alpha", "no-r", "r-zero", "r-not-a-number"],
)
def test_no_usable_r_or_alpha_scales_by_one_as_check_does(tmp_path, monkeypatch, fields):
    from soup_cli.utils.adapter_check import check_adapter

    a, b = _factors(2, 2.0)  # B @ A = 2 · e0 e0^T
    if fields is None:
        _write(tmp_path / "x", {_A: a, _B: b}, config=False)
    else:
        _write(tmp_path / "x", {_A: a, _B: b}, **fields)
    a_1, b_1 = _factors(2, 1.0)
    _write(tmp_path / "y", {_A: a_1, _B: b_1}, r=2, lora_alpha=2)  # ΔW = e0 e0^T
    report = _diff(tmp_path, monkeypatch, "x", "y")
    row = report.per_layer[0]
    assert row.norm_a == pytest.approx(2.0)  # scaling 1.0
    assert row.frobenius == pytest.approx(1.0)
    assert check_adapter("x").total_frobenius == pytest.approx(row.norm_a)


# ---------- which tensors are paired ----------

_EMB = "base_model.model.model.embed_tokens"
_MAG = f"{_STEM}.lora_magnitude_vector"


def test_linear_and_embedding_pairs_are_modules_and_dora_magnitude_is_by_name(
    tmp_path, monkeypatch,
):
    rng = np.random.default_rng(4)
    vocab = 9

    def tensors(magnitude):
        return {
            _A: rng.standard_normal((2, _DIM)).astype(np.float32),
            _B: rng.standard_normal((_DIM, 2)).astype(np.float32),
            f"{_EMB}.lora_embedding_A": rng.standard_normal((2, vocab)).astype(np.float32),
            f"{_EMB}.lora_embedding_B": rng.standard_normal((_DIM, 2)).astype(np.float32),
            _MAG: np.full(_DIM, magnitude, dtype=np.float32),
        }

    x, y = tensors(1.0), tensors(2.0)
    _write(tmp_path / "x", x, r=2, lora_alpha=4, use_dora=True)  # scaling 2
    _write(tmp_path / "y", y, r=2, lora_alpha=2, use_dora=True)  # scaling 1
    report = _diff(tmp_path, monkeypatch, "x", "y")
    rows = {d.name: d for d in report.per_layer}
    assert sorted(rows) == [_EMB, _STEM, _MAG]
    assert report.shared_layers == 3
    for stem, a_name, b_name in (
        (_STEM, _A, _B),
        (_EMB, f"{_EMB}.lora_embedding_A", f"{_EMB}.lora_embedding_B"),
    ):
        dense_x = 2.0 * (_f64(x[b_name]) @ _f64(x[a_name]))
        dense_y = _f64(y[b_name]) @ _f64(y[a_name])
        assert rows[stem].frobenius == pytest.approx(np.linalg.norm(dense_x - dense_y), rel=1e-9)
    assert rows[_MAG].frobenius == pytest.approx(math.sqrt(_DIM))


def test_tensors_outside_a_2d_pair_are_diffed_by_name(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    conv = "base_model.model.unet.conv_in"
    bias = f"{_STEM}.bias"
    score = "base_model.model.score.weight"  # a modules_to_save weight
    names_shared = (
        _A, bias, _MAG, score, f"{conv}.lora_A.weight", f"{conv}.lora_B.weight",
    )
    shapes = {
        _A: (2, _DIM), _B: (_DIM, 2), bias: (_DIM,), _MAG: (_DIM,), score: (2, _DIM),
        f"{conv}.lora_A.weight": (2, 3, 3, 3), f"{conv}.lora_B.weight": (4, 2, 1, 1),
    }

    def tensors(names):
        return {n: rng.standard_normal(shapes[n]).astype(np.float32) for n in names}

    x = tensors((*names_shared, _B))
    y = tensors(names_shared)  # q_proj has only its lora_A half here
    _write(tmp_path / "x", x, r=2, lora_alpha=2)
    _write(tmp_path / "y", y, r=2, lora_alpha=2)
    report = _diff(tmp_path, monkeypatch, "x", "y")
    assert [d.name for d in report.per_layer] == sorted(names_shared)
    for row in report.per_layer:
        assert row.frobenius == pytest.approx(np.linalg.norm(_f64(x[row.name]) - _f64(y[row.name])))
    assert report.only_in_a == (_B,)


# ---------- the CLI ----------


def test_cli_json_reports_the_module(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from soup_cli.cli import app

    a2, b2 = _factors(2, 1.0)
    a4, b4 = _factors(4, 3.0)
    _write(tmp_path / "x", {_A: a2, _B: b2}, r=2, lora_alpha=2)
    _write(tmp_path / "y", {_A: a4, _B: b4}, r=4, lora_alpha=4)
    monkeypatch.chdir(tmp_path)
    res = CliRunner().invoke(
        app, ["adapters", "diff", "x", "y", "--format", "json", "-o", "d.json"]
    )
    assert res.exit_code == 0, res.output
    payload = json.loads((tmp_path / "d.json").read_text(encoding="utf-8"))
    assert payload["shared_layers"] == 1
    assert payload["top_changed"] == [_STEM]
    assert payload["per_layer"] == [
        {
            "name": _STEM,
            "frobenius": pytest.approx(2.0),
            "norm_a": pytest.approx(1.0),
            "norm_b": pytest.approx(3.0),
            "relative": pytest.approx(2.0 / 3.0),
        }
    ]
    assert payload["effective_rank_a"] == pytest.approx(1.0)
    assert payload["effective_rank_b"] == pytest.approx(1.0)


def _issue_adapters(root):
    """The adapters of #3103: one 6x6 ``q_proj`` module each."""
    a, b = _factors(2, 1.0)
    _write(root / "a", {_A: a, _B: b}, r=2, lora_alpha=2)  # ΔW = e0 e0^T
    _write(root / "a2", {_A: a, _B: b}, r=2, lora_alpha=4)  # ΔW = 2 e0 e0^T
    a_d, b_d = _factors(4, 3.0)
    _write(root / "d", {_A: a_d, _B: b_d}, r=4, lora_alpha=4)  # ΔW = 3 e0 e0^T
    rng = np.random.default_rng(3103)
    a_p = rng.standard_normal((4, _DIM)).astype(np.float32)
    b_p = rng.standard_normal((_DIM, 4)).astype(np.float32)
    _write(root / "p", {_A: a_p, _B: b_p}, r=4, lora_alpha=8)
    _write(root / "p2", {_A: a_p / 2, _B: b_p * 2}, r=4, lora_alpha=8)  # same ΔW as p
    zero_b = np.zeros((_DIM, 4), dtype=np.float32)
    _write(root / "z", {_A: a_p, _B: zero_b}, r=4, lora_alpha=8)  # ΔW = 0


@pytest.mark.parametrize(
    ("left", "right", "frobenius", "relative", "ranks"),
    [
        ("a", "a2", "1.0000", "50.00%", "A=1.00, B=1.00"),
        ("a", "d", "2.0000", "66.67%", "A=1.00, B=1.00"),
        ("p", "p2", "0.0000", "0.00%", None),
        ("z", "z", "0.0000", "0.00%", "A=0.00, B=0.00"),
    ],
)
def test_issue_commands_through_the_cli(
    tmp_path, monkeypatch, left, right, frobenius, relative, ranks,
):
    from typer.testing import CliRunner

    from soup_cli.cli import app

    _issue_adapters(tmp_path)
    monkeypatch.chdir(tmp_path)
    res = CliRunner().invoke(app, ["adapters", "diff", left, right])
    assert res.exit_code == 0, res.output
    out = " ".join(strip_ansi(res.output).split())
    assert _STEM in out
    assert ".lora_A.weight" not in out and ".lora_B.weight" not in out
    # The row's two cells, side by side: "<frobenius> │ <relative>".
    assert re.search(rf"(?<![\d.]){re.escape(frobenius)}\W+{re.escape(relative)}", out), out
    if ranks is not None:
        assert f"Effective rank: {ranks}" in out
    assert "Shared layers: 1 | only-in-A: 0 | only-in-B: 0" in out
