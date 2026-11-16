"""#2908 — ``soup adapters merge`` scales each input's delta by its weight, not its square.

A LoRA's delta is ``B @ A``. ``merge_linear`` and ``merge_dare`` multiplied
both factors by the normalised weight ``c``, so an input's own term came out as
``c²`` times its delta: ``--weights 0.7,0.3`` gave 0.49 and 0.09, and the
default equal weights kept a quarter of each of two inputs. The factors now
take ``√c`` each, the rule ``soup adapters arithmetic`` already uses.

The adapters here are set by hand so that every delta is known: input ``i``
holds a single 1 at ``ΔW[i, i]``, through rank slot ``slot``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from soup_cli.utils.adapter_merge import (
    merge_adapters,
    merge_dare,
    merge_linear,
    merge_svd,
    merge_ties,
)

from .conftest import strip_ansi

MODULE = "base_model.model.model.layers.0.self_attn.q_proj"
A_KEY = f"{MODULE}.lora_A.weight"
B_KEY = f"{MODULE}.lora_B.weight"


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def _unit(slot: int, index: int, *, rank: int = 2, dim: int = 8,
          a_key: str = A_KEY, b_key: str = B_KEY) -> dict[str, np.ndarray]:
    """An adapter whose delta is a single 1 at ``[index, index]``, via ``slot``."""
    lora_a = np.zeros((rank, dim), dtype=np.float32)
    lora_b = np.zeros((dim, rank), dtype=np.float32)
    lora_a[slot, index] = 1.0
    lora_b[index, slot] = 1.0
    return {a_key: lora_a, b_key: lora_b}


def _delta(merged, a_key: str = A_KEY, b_key: str = B_KEY) -> np.ndarray:
    return np.asarray(merged[b_key], dtype=np.float64) @ np.asarray(merged[a_key], dtype=np.float64)


def _own_terms(merged, count: int = 2) -> list[float]:
    delta = _delta(merged)
    return [float(delta[i, i]) for i in range(count)]


FIRST = _unit(0, 0)       # ΔW = e0·e0ᵀ
SECOND = _unit(1, 1)      # ΔW = e1·e1ᵀ, the other rank slot: no cross terms
SAME_SLOT = _unit(0, 1)   # ΔW = e1·e1ᵀ, in FIRST's rank slot


# ---------------------------------------------------------------------------
# linear
# ---------------------------------------------------------------------------


def test_linear_scales_each_own_term_by_its_weight() -> None:
    merged, skipped = merge_linear([FIRST, SECOND], [0.7, 0.3])
    assert skipped == ()
    assert _own_terms(merged) == pytest.approx([0.7, 0.3], abs=1e-6)
    delta = _delta(merged)
    assert delta[0, 1] == 0.0 and delta[1, 0] == 0.0


def test_linear_equal_weights_keep_half_of_each_of_two_inputs() -> None:
    merged, _ = merge_linear([FIRST, SECOND], [1.0, 1.0])
    assert _own_terms(merged) == pytest.approx([0.5, 0.5], abs=1e-6)


def test_weights_that_are_not_normalised_give_the_same_merge() -> None:
    merged, _ = merge_linear([FIRST, SECOND], [7, 3])
    assert _own_terms(merged) == pytest.approx([0.7, 0.3], abs=1e-6)


@pytest.mark.parametrize("count", [2, 3, 16])
def test_equal_weights_give_one_nth_of_each_input(count: int) -> None:
    # 16 is the most inputs a merge takes (_MAX_ADAPTERS); each has its own slot.
    inputs = [_unit(i, i, rank=16, dim=16) for i in range(count)]
    merged, _ = merge_linear(inputs, [1.0] * count)
    assert _own_terms(merged, count) == pytest.approx([1.0 / count] * count, abs=1e-6)


def test_inputs_in_one_rank_slot_keep_their_cross_terms() -> None:
    # A factor-wise merge of inputs that use the same rank slot also adds
    # √(cᵢcⱼ)·Bᵢ@Aⱼ; `adapters arithmetic` does the same on its same-rank path.
    merged, _ = merge_linear([FIRST, SAME_SLOT], [1.0, 1.0])
    delta = _delta(merged)
    assert [delta[0, 0], delta[1, 1], delta[0, 1], delta[1, 0]] == pytest.approx(
        [0.5, 0.5, 0.5, 0.5], abs=1e-6
    )


def test_lora_embedding_factors_take_the_square_root_too() -> None:
    module = "base_model.model.model.embed_tokens"
    a_key, b_key = f"{module}.lora_embedding_A", f"{module}.lora_embedding_B"
    first = _unit(0, 0, a_key=a_key, b_key=b_key)
    second = _unit(1, 1, a_key=a_key, b_key=b_key)
    merged, _ = merge_linear([first, second], [0.7, 0.3])
    delta = _delta(merged, a_key, b_key)
    assert [delta[0, 0], delta[1, 1]] == pytest.approx([0.7, 0.3], abs=1e-6)


def test_other_tensors_keep_the_plain_weighted_average() -> None:
    # A DoRA magnitude vector is not a factor of a product: it keeps cᵢ.
    magnitude = f"{MODULE}.lora_magnitude_vector"
    first = {**FIRST, magnitude: np.full((8,), 2.0, dtype=np.float32), "bias": np.ones(3)}
    second = {**SECOND, magnitude: np.full((8,), 4.0, dtype=np.float32), "bias": np.zeros(3)}
    merged, _ = merge_linear([first, second], [0.7, 0.3])
    np.testing.assert_allclose(merged[magnitude], np.full((8,), 0.7 * 2.0 + 0.3 * 4.0), rtol=1e-6)
    np.testing.assert_allclose(merged["bias"], np.full((3,), 0.7), rtol=1e-6)
    assert _own_terms(merged) == pytest.approx([0.7, 0.3], abs=1e-6)


# ---------------------------------------------------------------------------
# dare
# ---------------------------------------------------------------------------


def test_dare_at_density_one_scales_each_own_term_by_its_weight() -> None:
    merged, _ = merge_dare([FIRST, SECOND], [0.7, 0.3], density=1.0, seed=0)
    assert _own_terms(merged) == pytest.approx([0.7, 0.3], abs=1e-6)


def test_dare_below_density_one_keeps_its_rescale_on_top() -> None:
    rng = np.random.default_rng(5)
    inputs = [
        {A_KEY: rng.standard_normal((2, 8)).astype(np.float32),
         B_KEY: rng.standard_normal((8, 2)).astype(np.float32)}
        for _ in range(2)
    ]
    density, seed, coeffs = 0.5, 3, (0.7, 0.3)
    merged, _ = merge_dare(inputs, coeffs, density=density, seed=seed)

    # The same draws, in the order merge_dare makes them: names sorted, then inputs.
    draws = np.random.default_rng(seed)
    for name in sorted(inputs[0]):
        expected = np.zeros(inputs[0][name].shape, dtype=np.float64)
        for coeff, adapter in zip(coeffs, inputs):
            tensor = np.asarray(adapter[name], dtype=np.float64)
            mask = draws.random(tensor.shape) < density
            expected += np.sqrt(coeff) * np.where(mask, tensor / density, 0.0)
        np.testing.assert_allclose(merged[name], expected.astype(np.float32), rtol=1e-6)


# ---------------------------------------------------------------------------
# svd
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rank", [None, 2])
def test_svd_follows_the_linear_merge(rank) -> None:
    merged, _ = merge_svd([FIRST, SECOND], [0.7, 0.3], rank=rank)
    assert _own_terms(merged) == pytest.approx([0.7, 0.3], abs=1e-5)


# ---------------------------------------------------------------------------
# ties is unchanged
# ---------------------------------------------------------------------------


def test_ties_is_unchanged() -> None:
    # Its disjoint mean divides by the weights that agree, so on entries only
    # one input holds the weights cancel: each own term stays 1, as on main.
    merged, _ = merge_ties([FIRST, SECOND], [0.7, 0.3], density=1.0)
    assert _own_terms(merged) == pytest.approx([1.0, 1.0], abs=1e-6)


def test_ties_does_not_look_at_tensor_names() -> None:
    rng = np.random.default_rng(9)
    left = rng.standard_normal((4, 6)).astype(np.float32)
    right = rng.standard_normal((4, 6)).astype(np.float32)
    as_factor, _ = merge_ties([{A_KEY: left}, {A_KEY: right}], [0.7, 0.3], density=0.5)
    as_plain, _ = merge_ties([{"w": left}, {"w": right}], [0.7, 0.3], density=0.5)
    np.testing.assert_array_equal(as_factor[A_KEY], as_plain["w"])


# ---------------------------------------------------------------------------
# On disk: merge_adapters, the command, cmaes, and `adapters arithmetic`
# ---------------------------------------------------------------------------


def _write(name: str, tensors: dict[str, np.ndarray]) -> str:
    from safetensors.numpy import save_file

    folder = Path(name)
    folder.mkdir()
    save_file(tensors, str(folder / "adapter_model.safetensors"))
    (folder / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 2, "lora_alpha": 2,
                    "base_model_name_or_path": "tiny-base"}),
        encoding="utf-8",
    )
    return name


def _load(folder: str) -> dict[str, np.ndarray]:
    from safetensors.numpy import load_file

    return load_file(str(Path(folder) / "adapter_model.safetensors"))


@pytest.fixture
def on_disk(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write("a", FIRST)
    _write("b", SECOND)


def _soup(*args: str):
    from soup_cli.cli import app

    return CliRunner().invoke(app, list(args))


@pytest.mark.parametrize("strategy,extra", [
    ("linear", {}),
    ("dare", {"density": 1.0}),
    ("svd", {}),
    ("svd", {"rank": 2}),
])
def test_merge_adapters_writes_the_weighted_deltas(on_disk, strategy, extra) -> None:
    merge_adapters(["a", "b"], "out", strategy=strategy, weights=[0.7, 0.3], **extra)
    assert _own_terms(_load("out")) == pytest.approx([0.7, 0.3], abs=1e-5)


def test_the_command_and_adapters_arithmetic_agree_on_one_pair(on_disk) -> None:
    merged = _soup("adapters", "merge", "a", "b", "-o", "merged", "--weights", "0.7,0.3",
                   "--allow-unscanned")
    assert merged.exit_code == 0, _plain(merged.output)
    arithmetic = _soup("adapters", "arithmetic", "0.7*a + 0.3*b", "--adapter", "a=a",
                       "--adapter", "b=b", "-o", "arith", "--allow-unscanned")
    assert arithmetic.exit_code == 0, _plain(arithmetic.output)

    assert _own_terms(_load("merged")) == pytest.approx([0.7, 0.3], abs=1e-6)
    np.testing.assert_allclose(_delta(_load("merged")), _delta(_load("arith")), atol=1e-6)


def test_the_command_with_default_weights_keeps_half_of_each(on_disk) -> None:
    result = _soup("adapters", "merge", "a", "b", "-o", "merged", "--allow-unscanned")
    assert result.exit_code == 0, _plain(result.output)
    assert _own_terms(_load("merged")) == pytest.approx([0.5, 0.5], abs=1e-6)


def test_cmaes_scores_candidates_merged_with_the_same_rule(on_disk) -> None:
    from soup_cli.utils.cmaes_merge import CmaesPlan, build_cmaes_eval_fn

    seen: list[list[float]] = []

    def scorer(merged_dir: str, eval_suite: str) -> float:
        seen.append(_own_terms(_load(merged_dir)))
        return 0.5

    plan = CmaesPlan(adapters=("a", "b"), eval_suite="suite.jsonl", budget_seconds=600,
                     population_size=4, max_generations=2, seed=0)
    eval_fn = build_cmaes_eval_fn(plan, scorer=scorer)
    eval_fn((0.7, 0.3))
    eval_fn((1.0, 3.0))
    assert seen[0] == pytest.approx([0.7, 0.3], abs=1e-6)
    assert seen[1] == pytest.approx([0.25, 0.75], abs=1e-6)


def test_the_cmaes_command_writes_its_best_weights_with_the_same_rule(
    on_disk, monkeypatch
) -> None:
    import soup_cli.utils.cmaes_merge as cm

    def scorer(merged_dir: str, eval_suite: str) -> float:
        first, second = _own_terms(_load(merged_dir))
        return 1.0 - abs(first - second)  # the balanced candidate wins

    monkeypatch.setattr(cm, "_CMAES_SCORER_OVERRIDE", scorer)
    Path("suite.jsonl").write_text('{"prompt": "hi", "expected": "hi"}\n', encoding="utf-8")
    result = _soup("adapters", "merge", "a", "b", "-o", "best", "--strategy", "cmaes",
                   "--eval", "suite.jsonl", "--budget", "60s", "--population", "4",
                   "--max-generations", "3", "--allow-unscanned")
    assert result.exit_code == 0, _plain(result.output)
    first, second = _own_terms(_load("best"))
    # The normalised weights sum to 1, so the own terms do as well. With each
    # weight squared the sum is below 1 for any merge that mixes the two.
    assert first > 0.05 and second > 0.05
    assert first + second == pytest.approx(1.0, abs=1e-5)
