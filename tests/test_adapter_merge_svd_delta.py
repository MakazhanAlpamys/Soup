"""``soup adapters merge --strategy svd --rank k`` truncates each module's delta (#2984).

``merge_svd`` replaced every 2-D tensor of the linear merge by its own rank-k
SVD reconstruction: ``lora_A`` and ``lora_B`` each kept their own top
directions, so ``B @ A`` could come out as 0, depended on how the inputs split
their delta between ``A`` and ``B``, and full tensors saved with the adapter (a
``modules_to_save`` head) were cut to rank k. Now each LoRA module's merged
delta ``B @ A`` is replaced by its best rank-k approximation, written back as
two factors zero-padded to the input rank, and every other tensor is written as
the linear merge gives it.

The adapters here are built by hand so that every delta is known.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from soup_cli.utils.adapter_merge import merge_adapters, merge_linear, merge_svd

from .conftest import strip_ansi

MODULE = "base_model.model.model.layers.0.self_attn.q_proj"
A_KEY = f"{MODULE}.lora_A.weight"
B_KEY = f"{MODULE}.lora_B.weight"
EMBED = "base_model.model.model.embed_tokens"
EMB_A_KEY = f"{EMBED}.lora_embedding_A"
EMB_B_KEY = f"{EMBED}.lora_embedding_B"
HEAD_KEY = "base_model.model.score.weight"
BIAS_KEY = f"{MODULE}.lora_B.bias"
MAGNITUDE_KEY = f"{MODULE}.lora_magnitude_vector"


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def _delta(tensors, a_key: str = A_KEY, b_key: str = B_KEY) -> np.ndarray:
    return np.asarray(tensors[b_key], np.float64) @ np.asarray(tensors[a_key], np.float64)


def _best_rank(matrix: np.ndarray, k: int) -> np.ndarray:
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    return (u[:, :k] * s[:k]) @ vt[:k, :]


def _report_pair(a_key: str = A_KEY, b_key: str = B_KEY) -> dict[str, np.ndarray]:
    """The example of the report: ``A = diag(10, 1)``, ``B = diag(0.01, 1)``.

    The delta is ``diag(0.1, 1)``. Truncating each factor to rank 1 keeps slot
    0 of ``A`` and slot 1 of ``B``, whose product is 0.
    """
    lora_a = np.zeros((2, 4))
    lora_b = np.zeros((4, 2))
    lora_a[0, 0], lora_a[1, 1] = 10.0, 1.0
    lora_b[0, 0], lora_b[1, 1] = 0.01, 1.0
    return {a_key: lora_a.astype(np.float32), b_key: lora_b.astype(np.float32)}


def _random_pair(seed: int, *, rank: int = 8, d_in: int = 24, d_out: int = 20):
    rng = np.random.default_rng(seed)
    lora_a = rng.standard_normal((rank, d_in)) * rng.uniform(0.1, 3.0, (rank, 1))
    lora_b = rng.standard_normal((d_out, rank)) * rng.uniform(0.1, 3.0, (1, rank))
    return {A_KEY: lora_a.astype(np.float32), B_KEY: lora_b.astype(np.float32)}


def _resplit(pair: dict[str, np.ndarray], gauge: np.ndarray) -> dict[str, np.ndarray]:
    """The same delta, split otherwise: ``A -> G A``, ``B -> B G^-1``."""
    lora_a = gauge @ np.asarray(pair[A_KEY], np.float64)
    lora_b = np.asarray(pair[B_KEY], np.float64) @ np.linalg.inv(gauge)
    return {A_KEY: lora_a.astype(np.float32), B_KEY: lora_b.astype(np.float32)}


def _linear_delta(inputs, weights, a_key: str = A_KEY, b_key: str = B_KEY) -> np.ndarray:
    merged, _ = merge_linear(inputs, weights)
    return _delta(merged, a_key, b_key)


# ---------------------------------------------------------------------------
# The merged delta, not each factor
# ---------------------------------------------------------------------------


def test_the_report_example_keeps_the_top_direction_of_the_delta() -> None:
    pair = _report_pair()
    merged, _ = merge_svd([pair, pair], [0.5, 0.5], rank=1)
    # The linear merge of two identical inputs is diag(0.2, 2) (cross terms);
    # its best rank-1 approximation keeps the 2, not 0.
    expected = np.zeros((4, 4))
    expected[1, 1] = 2.0
    np.testing.assert_allclose(_delta(merged), expected, atol=1e-5)


@pytest.mark.parametrize(
    "gauge",
    [
        np.diag([0.05, 1.0]),
        np.diag([3.0, 0.2]),
        np.array([[1.0, 2.0], [0.0, 1.0]]),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
    ],
    ids=["shrink-slot-0", "both-slots", "shear", "swap"],
)
def test_the_result_does_not_depend_on_how_a_and_b_split_the_delta(gauge) -> None:
    first, second = _report_pair(), _report_pair()
    second[A_KEY] = second[A_KEY] * np.float32(2.0)  # a second, different input
    plain, _ = merge_svd([first, second], [0.6, 0.4], rank=1)
    resplit, _ = merge_svd([_resplit(first, gauge), _resplit(second, gauge)], [0.6, 0.4], rank=1)
    np.testing.assert_allclose(_delta(resplit), _delta(plain), atol=1e-5)
    np.testing.assert_allclose(
        _delta(plain), _best_rank(_linear_delta([first, second], [0.6, 0.4]), 1), atol=1e-5
    )


@pytest.mark.parametrize("k", [1, 2, 4, 7])
def test_the_written_delta_is_the_best_rank_k_approximation(k) -> None:
    inputs = [_random_pair(0), _random_pair(1)]
    merged, _ = merge_svd(inputs, [0.5, 0.5], rank=k)
    full = _linear_delta(inputs, [0.5, 0.5])
    np.testing.assert_allclose(_delta(merged), _best_rank(full, k), atol=1e-4 * np.abs(full).max())


@pytest.mark.parametrize("k", [1, 3])
def test_the_factors_keep_the_input_rank_with_zero_padding(k) -> None:
    inputs = [_random_pair(2, rank=4), _random_pair(3, rank=4)]
    merged, _ = merge_svd(inputs, [0.5, 0.5], rank=k)
    assert merged[A_KEY].shape == (4, 24)
    assert merged[B_KEY].shape == (20, 4)
    assert merged[A_KEY].dtype == np.float32
    assert merged[B_KEY].dtype == np.float32
    assert not merged[A_KEY][k:].any()
    assert not merged[B_KEY][:, k:].any()
    assert np.linalg.matrix_rank(_delta(merged), tol=1e-6) == k


def test_the_embedding_pair_is_truncated_like_the_linear_pair() -> None:
    pair = _report_pair(EMB_A_KEY, EMB_B_KEY)
    merged, _ = merge_svd([pair, pair], [0.5, 0.5], rank=1)
    full = _linear_delta([pair, pair], [0.5, 0.5], EMB_A_KEY, EMB_B_KEY)
    np.testing.assert_allclose(_delta(merged, EMB_A_KEY, EMB_B_KEY), _best_rank(full, 1), atol=1e-5)
    assert merged[EMB_A_KEY].shape == (2, 4)
    assert merged[EMB_B_KEY].shape == (4, 2)


@pytest.mark.parametrize(("a_key", "b_key"), [
    (f"{MODULE}.lora_A.{{name}}.weight", f"{MODULE}.lora_B.{{name}}.weight"),
    (f"{EMBED}.lora_embedding_A.{{name}}", f"{EMBED}.lora_embedding_B.{{name}}"),
])
def test_named_factors_are_paired_by_adapter_name(a_key, b_key) -> None:
    """Two adapter names on one module: each ``A`` is truncated with the ``B`` of
    its own name, so each written delta is the best rank-k one of that name."""
    names = ("default", "other")
    inputs = []
    for seed in (0, 1):
        tensors = {}
        for offset, name in enumerate(names):
            pair = _random_pair(seed * 2 + offset)
            tensors[a_key.format(name=name)] = pair[A_KEY]
            tensors[b_key.format(name=name)] = pair[B_KEY]
        inputs.append(tensors)

    merged, _ = merge_svd(inputs, [0.5, 0.5], rank=2)

    assert set(merged) == set(inputs[0])
    for name in names:
        a_name, b_name = a_key.format(name=name), b_key.format(name=name)
        full = _linear_delta(inputs, [0.5, 0.5], a_name, b_name)
        np.testing.assert_allclose(
            _delta(merged, a_name, b_name), _best_rank(full, 2), atol=1e-4 * np.abs(full).max(),
        )
        assert merged[a_name].shape == (8, 24)
        assert merged[b_name].shape == (20, 8)


# ---------------------------------------------------------------------------
# Tensors that are not LoRA factors keep the linear merge
# ---------------------------------------------------------------------------


def test_tensors_that_are_not_lora_factors_are_not_truncated() -> None:
    rng = np.random.default_rng(4)
    inputs = []
    for _ in range(2):
        adapter = _report_pair()
        adapter[HEAD_KEY] = rng.standard_normal((20, 16)).astype(np.float32)
        adapter[BIAS_KEY] = rng.standard_normal(4).astype(np.float32)
        adapter[MAGNITUDE_KEY] = rng.standard_normal(4).astype(np.float32)
        inputs.append(adapter)
    linear, _ = merge_linear(inputs, [0.5, 0.5])
    merged, _ = merge_svd(inputs, [0.5, 0.5], rank=1)
    for key in (HEAD_KEY, BIAS_KEY, MAGNITUDE_KEY):
        assert merged[key].dtype == np.float32
        assert merged[key].tobytes() == linear[key].tobytes(), key


def test_a_factor_whose_pair_is_missing_keeps_the_linear_merge() -> None:
    first, second = _random_pair(5), _random_pair(6)
    only_a = [{A_KEY: first[A_KEY]}, {A_KEY: second[A_KEY]}]
    linear, _ = merge_linear(only_a, [0.5, 0.5])
    merged, _ = merge_svd(only_a, [0.5, 0.5], rank=1)
    assert merged[A_KEY].tobytes() == linear[A_KEY].tobytes()


def test_a_pair_that_is_not_2d_keeps_the_linear_merge() -> None:
    # A conv LoRA: lora_A is (r, in, kh, kw), lora_B is (out, r, 1, 1).
    rng = np.random.default_rng(7)
    inputs = [
        {
            A_KEY: rng.standard_normal((2, 3, 3, 3)).astype(np.float32),
            B_KEY: rng.standard_normal((4, 2, 1, 1)).astype(np.float32),
        }
        for _ in range(2)
    ]
    linear, _ = merge_linear(inputs, [0.5, 0.5])
    merged, _ = merge_svd(inputs, [0.5, 0.5], rank=1)
    for key in (A_KEY, B_KEY):
        assert merged[key].tobytes() == linear[key].tobytes()


def _svd_does_not_converge(_delta_matrix, _rank):
    raise np.linalg.LinAlgError("SVD did not converge")


def test_a_pair_whose_svd_raises_keeps_the_linear_merge(monkeypatch) -> None:
    monkeypatch.setattr(
        "soup_cli.utils.adapter_arithmetic._truncated_factors", _svd_does_not_converge
    )
    inputs = [_random_pair(8), _random_pair(9)]
    linear, _ = merge_linear(inputs, [0.5, 0.5])
    merged, _ = merge_svd(inputs, [0.5, 0.5], rank=1)
    assert sorted(merged) == sorted(linear)
    for key in (A_KEY, B_KEY):
        assert merged[key].tobytes() == linear[key].tobytes(), key


# ---------------------------------------------------------------------------
# Degenerate deltas
# ---------------------------------------------------------------------------


def test_an_all_zero_delta_gives_zero_factors_and_no_nan() -> None:
    zero = {A_KEY: np.zeros((4, 6), np.float32), B_KEY: np.zeros((5, 4), np.float32)}
    merged, _ = merge_svd([zero, zero], [0.5, 0.5], rank=2)
    for key in (A_KEY, B_KEY):
        assert np.isfinite(merged[key]).all()
    assert merged[A_KEY].shape == (4, 6)
    assert merged[B_KEY].shape == (5, 4)
    assert not _delta(merged).any()


def test_a_delta_of_lower_rank_than_k_is_kept_whole_and_finite() -> None:
    # Rank-1 delta through rank 4 slots, truncated to k = 2.
    rng = np.random.default_rng(8)
    column = rng.standard_normal((5, 1))
    row = rng.standard_normal((1, 6))
    lora_a = np.zeros((4, 6))
    lora_b = np.zeros((5, 4))
    lora_a[2], lora_b[:, 2] = row[0], column[:, 0]
    pair = {A_KEY: lora_a.astype(np.float32), B_KEY: lora_b.astype(np.float32)}
    merged, _ = merge_svd([pair, pair], [0.5, 0.5], rank=2)
    for key in (A_KEY, B_KEY):
        assert np.isfinite(merged[key]).all()
    np.testing.assert_allclose(_delta(merged), _linear_delta([pair, pair], [0.5, 0.5]), atol=1e-5)


def test_a_rank_above_the_smaller_dimension_is_clamped() -> None:
    # r = 6 slots on a 3 x 4 module: the delta has at most 3 singular values.
    inputs = [_random_pair(9, rank=6, d_in=4, d_out=3), _random_pair(10, rank=6, d_in=4, d_out=3)]
    merged, _ = merge_svd(inputs, [0.5, 0.5], rank=5)
    full = _linear_delta(inputs, [0.5, 0.5])
    np.testing.assert_allclose(_delta(merged), full, atol=1e-4 * np.abs(full).max())
    assert merged[A_KEY].shape == (6, 4)
    assert merged[B_KEY].shape == (3, 6)


# ---------------------------------------------------------------------------
# Controls: no --rank, and k at or above the LoRA rank, write the linear merge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rank", [None, 8, 9, 64])
def test_without_truncation_the_bytes_are_the_linear_merge(rank) -> None:
    rng = np.random.default_rng(11)
    inputs = []
    for seed in (12, 13):
        adapter = _random_pair(seed)
        adapter[HEAD_KEY] = rng.standard_normal((20, 16)).astype(np.float32)
        inputs.append(adapter)
    linear, _ = merge_linear(inputs, [0.7, 0.3])
    merged, _ = merge_svd(inputs, [0.7, 0.3], rank=rank)
    assert sorted(merged) == sorted(linear)
    for key in linear:
        assert merged[key].tobytes() == linear[key].tobytes(), key


def test_k_one_below_the_rank_truncates() -> None:
    inputs = [_random_pair(14), _random_pair(15)]
    merged, _ = merge_svd(inputs, [0.5, 0.5], rank=7)
    full = _linear_delta(inputs, [0.5, 0.5])
    assert np.linalg.matrix_rank(_delta(merged), tol=1e-6 * np.abs(full).max()) == 7
    np.testing.assert_allclose(_delta(merged), _best_rank(full, 7), atol=1e-4 * np.abs(full).max())


# ---------------------------------------------------------------------------
# The SVD runs on the factors, never on the dense delta
# ---------------------------------------------------------------------------

_OUT, _IN, _LORA_RANK = 40, 90, 6


def _rectangular_pair(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        A_KEY: rng.standard_normal((_LORA_RANK, _IN)).astype(np.float32),
        B_KEY: rng.standard_normal((_OUT, _LORA_RANK)).astype(np.float32),
    }


@pytest.mark.parametrize("k", [1, 3, 5])
def test_rectangular_factors_give_the_best_rank_k_of_the_dense_delta(k) -> None:
    inputs = [_rectangular_pair(20), _rectangular_pair(21)]
    merged, _ = merge_svd(inputs, [0.5, 0.5], rank=k)
    full = _linear_delta(inputs, [0.5, 0.5])
    assert merged[A_KEY].shape == (_LORA_RANK, _IN)
    assert merged[B_KEY].shape == (_OUT, _LORA_RANK)
    np.testing.assert_allclose(_delta(merged), _best_rank(full, k), atol=1e-4 * np.abs(full).max())


def test_no_svd_is_larger_than_the_lora_rank(monkeypatch) -> None:
    shapes: list[tuple[int, ...]] = []
    real_svd = np.linalg.svd

    def recording_svd(matrix, *args, **kwargs):
        shapes.append(np.shape(matrix))
        return real_svd(matrix, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", recording_svd)
    merge_svd([_rectangular_pair(22), _rectangular_pair(23)], [0.5, 0.5], rank=2)

    assert shapes, "the truncation ran no SVD"
    assert all(rows <= _LORA_RANK and cols <= _LORA_RANK for rows, cols in shapes), shapes


# ---------------------------------------------------------------------------
# Through merge_adapters and the command
# ---------------------------------------------------------------------------


def _write(name: str, tensors: dict[str, np.ndarray], *, rank: int = 2, alpha: int = 4) -> str:
    from safetensors.numpy import save_file

    folder = Path(name)
    folder.mkdir()
    save_file(tensors, str(folder / "adapter_model.safetensors"))
    (folder / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": rank, "lora_alpha": alpha}), encoding="utf-8"
    )
    return name


def _load(folder: str) -> dict[str, np.ndarray]:
    from safetensors.numpy import load_file

    return load_file(str(Path(folder) / "adapter_model.safetensors"))


@pytest.fixture
def with_head(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(16)
    heads = []
    for name in ("c1", "c2"):
        adapter = _report_pair()
        adapter[HEAD_KEY] = rng.standard_normal((20, 16)).astype(np.float32)
        heads.append(adapter[HEAD_KEY])
        _write(name, adapter)
    return heads


def test_merge_adapters_writes_the_truncated_delta_and_the_whole_head(with_head) -> None:
    merge_adapters(["c1", "c2"], "out", strategy="svd", rank=1)
    out = _load("out")
    expected = np.zeros((4, 4))
    expected[1, 1] = 2.0
    np.testing.assert_allclose(_delta(out), expected, atol=1e-5)
    want_head = (0.5 * with_head[0].astype(np.float64) + 0.5 * with_head[1]).astype(np.float32)
    assert out[HEAD_KEY].tobytes() == want_head.tobytes()


def test_the_command_keeps_shapes_and_config(with_head) -> None:
    from soup_cli.cli import app

    result = CliRunner().invoke(
        app,
        [
            "adapters",
            "merge",
            "c1",
            "c2",
            "--strategy",
            "svd",
            "--rank",
            "1",
            "-o",
            "out",
            "--allow-unscanned",
        ],
    )
    assert result.exit_code == 0, _plain(result.output)
    out = _load("out")
    assert out[A_KEY].shape == (2, 4)
    assert out[B_KEY].shape == (4, 2)
    assert out[HEAD_KEY].shape == (20, 16)
    assert np.linalg.matrix_rank(out[HEAD_KEY].astype(np.float64)) == 16
    assert np.abs(_delta(out)).max() == pytest.approx(2.0, abs=1e-5)
    config = json.loads(Path("out", "adapter_config.json").read_text(encoding="utf-8"))
    assert (config["r"], config["lora_alpha"]) == (2, 4)


def test_the_command_exits_0_and_writes_the_linear_merge_when_the_svd_raises(
    with_head, monkeypatch
) -> None:
    from soup_cli.cli import app

    monkeypatch.setattr(
        "soup_cli.utils.adapter_arithmetic._truncated_factors", _svd_does_not_converge
    )
    result = CliRunner().invoke(
        app,
        [
            "adapters",
            "merge",
            "c1",
            "c2",
            "--strategy",
            "svd",
            "--rank",
            "1",
            "-o",
            "out",
            "--allow-unscanned",
        ],
    )
    assert result.exit_code == 0, _plain(result.output)
    out = _load("out")
    linear, _ = merge_linear([_load("c1"), _load("c2")], [0.5, 0.5])
    assert sorted(out) == sorted(linear)
    for key in linear:
        assert out[key].tobytes() == linear[key].tobytes(), key
