"""#2907 — ``soup adapters merge`` refuses inputs it cannot merge.

Every strategy merges the tensors all inputs share and skips a shared tensor
whose shapes differ. Two adapters with different LoRA ranks differ in shape on
every ``lora_A`` and ``lora_B``, so the merge was empty: the command wrote an
``adapter_model.safetensors`` with no tensors beside the first input's
``adapter_config.json`` and exited 0. Inputs that share no tensor name at all
(disjoint ``target_modules``) ended the same way.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from soup_cli.utils.adapter_merge import merge_adapters
from soup_cli.utils.cmaes_merge import CmaesPlan, build_cmaes_eval_fn

from .conftest import strip_ansi

Q_PROJ = "base_model.model.model.layers.0.self_attn.q_proj"
V_PROJ = "base_model.model.model.layers.0.self_attn.v_proj"
STRATEGIES = ["linear", "ties", "dare", "svd"]


def check_mergeable(weights_list) -> None:
    # Imported on use: the name does not exist before the fix, and the tests
    # that do not need it should still run there.
    from soup_cli.utils import adapter_merge

    adapter_merge.check_mergeable(weights_list)


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def _factors(module: str, rank: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        f"{module}.lora_A.weight": rng.standard_normal((rank, 8)).astype(np.float32),
        f"{module}.lora_B.weight": rng.standard_normal((8, rank)).astype(np.float32),
    }


def _write(name: str, tensors: dict[str, np.ndarray], rank: int) -> str:
    from safetensors.numpy import save_file

    folder = Path(name)
    folder.mkdir()
    save_file(tensors, str(folder / "adapter_model.safetensors"))
    (folder / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": rank, "lora_alpha": rank}), encoding="utf-8"
    )
    return name


@pytest.fixture
def adapters(tmp_path, monkeypatch) -> dict[str, dict[str, np.ndarray]]:
    """Adapter folders under the working directory, and the tensors in each."""
    monkeypatch.chdir(tmp_path)
    tensors = {
        "r2": _factors(Q_PROJ, 2, 0),
        "r2_other": _factors(Q_PROJ, 2, 1),
        "r4": _factors(Q_PROJ, 4, 2),
        # rank_pattern: the same rank on q_proj, another one on v_proj.
        "pattern_a": {**_factors(Q_PROJ, 2, 3), **_factors(V_PROJ, 2, 4)},
        "pattern_b": {**_factors(Q_PROJ, 2, 5), **_factors(V_PROJ, 4, 6)},
        "v_only": _factors(V_PROJ, 2, 7),
    }
    for name, weights in tensors.items():
        _write(name, weights, rank=2)
    return tensors


def _merge(*paths_then_options: str):
    from soup_cli.cli import app

    args = ["adapters", "merge", *paths_then_options, "-o", "out", "--allow-unscanned"]
    return CliRunner().invoke(app, args)


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_mixed_ranks_are_refused_on_every_strategy(adapters, strategy) -> None:
    extra = ["--strategy", strategy]
    if strategy == "svd":
        extra += ["--rank", "4"]
    result = _merge("r2", "r4", *extra)
    out = _plain(result.output)
    assert result.exit_code == 2, out
    assert Q_PROJ in out
    assert "rank 2 in adapter 1" in out
    assert "rank 4 in adapter 2" in out
    assert "soup adapters arithmetic" in out and "--rank" in out
    assert "Merged layers" not in out
    assert not Path("out").exists()


def test_a_rank_that_differs_on_one_module_only_is_refused(adapters) -> None:
    result = _merge("pattern_a", "pattern_b")
    out = _plain(result.output)
    assert result.exit_code == 2, out
    assert V_PROJ in out  # the module that differs, not the one that matches
    assert Q_PROJ not in out
    assert "rank 2 in adapter 1" in out and "rank 4 in adapter 2" in out
    assert not Path("out").exists()


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_inputs_that_share_no_tensor_are_refused(adapters, strategy) -> None:
    result = _merge("r2", "v_only", "--strategy", strategy)
    out = _plain(result.output)
    assert result.exit_code == 2, out
    assert "share no tensor name" in out
    assert not Path("out").exists()


def test_the_third_input_is_named_when_it_is_the_one_that_differs(adapters) -> None:
    result = _merge("r2", "r2_other", "r4")
    out = _plain(result.output)
    assert result.exit_code == 2, out
    assert "rank 2 in adapter 1, rank 2 in adapter 2, rank 4 in adapter 3" in out


# ---------------------------------------------------------------------------
# merge_adapters and the check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_merge_adapters_raises_before_it_writes(adapters, strategy) -> None:
    with pytest.raises(ValueError, match="rank 2 in adapter 1, rank 4 in adapter 2"):
        merge_adapters(["r2", "r4"], "out", strategy=strategy)
    assert not Path("out").exists()


def test_a_same_rank_shape_mismatch_names_the_tensor_and_the_shapes() -> None:
    name = f"{Q_PROJ}.lora_A.weight"
    first = {name: np.zeros((2, 8), dtype=np.float32)}
    second = {name: np.zeros((2, 16), dtype=np.float32)}
    with pytest.raises(ValueError) as exc:
        check_mergeable([first, second])
    msg = str(exc.value)
    assert name in msg
    assert "(2, 8) in adapter 1" in msg and "(2, 16) in adapter 2" in msg
    assert "arithmetic" not in msg  # another rank is not what is wrong here


def test_a_non_lora_tensor_that_differs_in_shape_is_refused() -> None:
    shared = _factors(Q_PROJ, 2, 0)
    extra = "base_model.model.lm_head.weight"
    first = {**shared, extra: np.zeros((4, 8), dtype=np.float32)}
    second = {**shared, extra: np.zeros((6, 8), dtype=np.float32)}
    with pytest.raises(ValueError, match="lm_head"):
        check_mergeable([first, second])


def test_lora_embedding_factors_report_their_rank() -> None:
    module = "base_model.model.model.embed_tokens"
    first = {
        f"{module}.lora_embedding_A": np.zeros((2, 32), dtype=np.float32),
        f"{module}.lora_embedding_B": np.zeros((8, 2), dtype=np.float32),
    }
    second = {
        f"{module}.lora_embedding_A": np.zeros((4, 32), dtype=np.float32),
        f"{module}.lora_embedding_B": np.zeros((8, 4), dtype=np.float32),
    }
    with pytest.raises(ValueError) as exc:
        check_mergeable([first, second])
    msg = str(exc.value)
    assert repr(module) in msg
    assert "rank 2 in adapter 1, rank 4 in adapter 2" in msg


# ---------------------------------------------------------------------------
# cmaes
# ---------------------------------------------------------------------------


def _plan(first: str, second: str) -> CmaesPlan:
    return CmaesPlan(adapters=(first, second), eval_suite="suite.jsonl", budget_seconds=600,
                     population_size=4, max_generations=2, seed=0)


@pytest.mark.parametrize("second,needle", [("r4", "rank 4 in adapter 2"),
                                           ("v_only", "share no tensor name")])
def test_cmaes_refuses_before_a_candidate_is_scored(adapters, second, needle) -> None:
    scored: list[str] = []

    def scorer(merged_dir: str, eval_suite: str) -> float:
        scored.append(merged_dir)
        return 0.5

    with pytest.raises(ValueError, match=needle):
        build_cmaes_eval_fn(_plan("r2", second), scorer=scorer)
    assert scored == []


def test_cmaes_from_the_command_exits_2_and_writes_nothing(adapters, monkeypatch) -> None:
    import soup_cli.utils.cmaes_merge as cm

    scored: list[str] = []

    def scorer(merged_dir: str, eval_suite: str) -> float:
        scored.append(merged_dir)
        return 0.5

    monkeypatch.setattr(cm, "_CMAES_SCORER_OVERRIDE", scorer)
    Path("suite.jsonl").write_text('{"prompt": "hi", "expected": "hi"}\n', encoding="utf-8")
    result = _merge("r2", "r4", "--strategy", "cmaes", "--eval", "suite.jsonl",
                    "--budget", "60s", "--population", "2", "--max-generations", "1")
    out = _plain(result.output)
    assert result.exit_code == 2, out
    assert "rank 2 in adapter 1, rank 4 in adapter 2" in out
    assert scored == []
    assert not Path("out").exists()


# ---------------------------------------------------------------------------
# Controls: what merges today still merges, to the same tensors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_control_a_same_rank_merge_gives_the_same_tensors(adapters, strategy) -> None:
    from safetensors.numpy import load_file

    from soup_cli.utils import adapter_merge as am

    inputs = [adapters["r2"], adapters["r2_other"]]
    direct = {
        "linear": lambda: am.merge_linear(inputs, [0.5, 0.5]),
        "ties": lambda: am.merge_ties(inputs, [0.5, 0.5], density=0.2),
        "dare": lambda: am.merge_dare(inputs, [0.5, 0.5], density=0.2, seed=0),
        "svd": lambda: am.merge_svd(inputs, [0.5, 0.5], rank=None),
    }[strategy]
    expected, skipped = direct()
    assert skipped == () and len(expected) == 2

    result = _merge("r2", "r2_other", "--strategy", strategy)
    assert result.exit_code == 0, _plain(result.output)
    assert "Merged layers: 2" in _plain(result.output)
    written = load_file(str(Path("out") / "adapter_model.safetensors"))
    assert set(written) == set(expected)
    for name, tensor in expected.items():
        np.testing.assert_array_equal(written[name], tensor)


def test_control_a_tensor_in_one_input_only_still_stays_out_of_the_merge(adapters) -> None:
    # pattern_a has q_proj and v_proj, r2 has q_proj only: q_proj merges, as before.
    result = _merge("pattern_a", "r2")
    assert result.exit_code == 0, _plain(result.output)
    assert "Merged layers: 2" in _plain(result.output)


def test_control_one_input_is_left_to_the_count_check() -> None:
    check_mergeable([_factors(Q_PROJ, 2, 0)])  # no ValueError from this check


# ---------------------------------------------------------------------------
# The hint `adapters arithmetic` gives for its own same-rank path
# ---------------------------------------------------------------------------


def test_arithmetic_no_longer_sends_a_shape_mismatch_to_adapters_merge() -> None:
    from soup_cli.utils.adapter_arithmetic import merge_task_arithmetic

    with pytest.raises(ValueError) as exc:
        merge_task_arithmetic([_factors(Q_PROJ, 2, 0), _factors(Q_PROJ, 4, 1)], [1.0, 1.0])
    msg = str(exc.value)
    assert "soup adapters arithmetic" in msg and "--rank" in msg
    assert "adapters merge" not in msg
