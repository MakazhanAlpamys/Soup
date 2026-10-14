"""Pure counter-based checks for the independent expert-profile holdout record."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).parents[1] / "benchmarks" / "harness" / "moe_expert_holdout.py"


@pytest.fixture(scope="module")
def harness():
    original_path = sys.path[:]
    names = ("_moe_expert_holdout", "moe_expert_coverage")
    original_modules = {name: sys.modules.get(name) for name in names}
    spec = importlib.util.spec_from_file_location(names[0], HARNESS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = original_path
        for name, previous in original_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _shape(counts, *, second_layer=None, top_k=1):
    layers = [{"layer": 0, "router": "model.layers.0.router", "per_step_counts": counts}]
    if second_layer is not None:
        layers.append(
            {"layer": 1, "router": "model.layers.1.router", "per_step_counts": second_layer}
        )
    return {
        "batch": 1,
        "seq": 4,
        "tokens_per_step": 4,
        "steps": 16,
        "chunks_available": 16,
        "layers": layers,
    }


def _direction(gain, loss):
    return {"gain": gain, "transfer_loss": loss}


def test_default_split_is_exact_disjoint_contiguous_eight_steps(harness):
    left, right = harness.contiguous_split(16)
    assert left == list(range(8))
    assert right == list(range(8, 16))
    assert set(left).isdisjoint(right)


@pytest.mark.parametrize("steps", [0, 1, 3, 15, -2, 16.0, True])
def test_default_split_refuses_invalid_or_odd_sizes(harness, steps):
    with pytest.raises(ValueError):
        harness.contiguous_split(steps)


@pytest.mark.parametrize("split", [0, 16, -1, 17, 8.5, True])
def test_split_refuses_empty_halves_and_invalid_indexes(harness, split):
    with pytest.raises(ValueError):
        harness.contiguous_split(16, split)


def test_explicit_custom_split_accepts_odd_total(harness):
    assert harness.contiguous_split(5, 2) == ([0, 1], [2, 3, 4])


def test_opposite_hot_experts_do_not_leak_evaluation_into_training(harness):
    shape = _shape([[4, 0, 0, 0]] * 8 + [[0, 4, 0, 0]] * 8)
    scored = harness.score_shape(shape, n_experts=4, top_k=1)
    assert scored["verdict"] == "NO MATERIAL GAIN"
    assert scored["full_in_sample"]["in_sample_hit"] == 0.5
    assert scored["full_in_sample"]["identity_delta"] == 0.0
    forward, reverse = scored["layers"][0]["directions"]
    assert forward["hot_ids"] == [0]
    assert reverse["hot_ids"] == [1]
    for direction in (forward, reverse):
        assert direction["in_sample_hit"] == 1.0
        assert direction["heldout_hit"] == 0.0
        assert direction["oracle_hit"] == 1.0
        assert direction["gain"] == -0.25
        assert direction["relative_gain"] == -1.0
        assert direction["transfer_loss"] == 1.0
        assert direction["heldout_step_hits"] == [0.0] * 8
    assert forward["train_steps"] == list(range(8))
    assert forward["evaluation_steps"] == list(range(8, 16))
    assert reverse["train_steps"] == list(range(8, 16))
    assert reverse["evaluation_steps"] == list(range(8))


def test_per_step_oracle_bounds_fixed_cache_and_signed_loss_is_preserved(harness):
    shape = _shape([[3, 1, 0, 0]] * 8 + [[1, 3, 0, 0], [3, 1, 0, 0]] * 4)
    scored = harness.score_shape(shape, n_experts=4, top_k=1)
    forward, reverse = scored["layers"][0]["directions"]
    assert forward["heldout_step_hits"] == [0.25, 0.75] * 4
    assert forward["oracle_step_hits"] == [0.75] * 8
    assert forward["oracle_hot_ids"] == [[1], [0]] * 4
    assert forward["heldout_hit"] == 0.5
    assert forward["oracle_hit"] == 0.75
    assert forward["in_sample_hit"] == 0.75
    assert forward["gain"] == 0.25
    assert forward["relative_gain"] == 1.0
    assert forward["transfer_loss"] == 0.25
    assert reverse["heldout_hit"] == 0.75
    assert reverse["in_sample_hit"] == 0.5
    assert reverse["transfer_loss"] == -0.25
    for direction in (forward, reverse):
        assert all(
            fixed <= oracle
            for fixed, oracle in zip(
                direction["heldout_step_hits"], direction["oracle_step_hits"]
            )
        )
    assert scored["verdict"] == "GAIN WITH DRIFT"


def test_uniform_traffic_equals_analytic_random_and_has_zero_gain(harness):
    scored = harness.score_shape(_shape([[1, 1, 1, 1]] * 16), n_experts=4, top_k=1)
    assert scored["random_hit"] == 0.25
    assert scored["full_in_sample"]["top25_share"] == 0.25
    assert scored["verdict"] == "NO MATERIAL GAIN"
    for direction in scored["directions"]:
        assert direction["heldout_hit"] == 0.25
        assert direction["in_sample_hit"] == 0.25
        assert direction["oracle_hit"] == 0.25
        assert direction["gain"] == 0.0
        assert direction["relative_gain"] == 0.0
        assert direction["transfer_loss"] == 0.0


def test_tied_hot_set_uses_lowest_expert_ids(harness):
    shape = _shape([[1] * 8] * 16)
    scored = harness.score_shape(shape, n_experts=8, top_k=2)
    assert scored["verdict"] == "NO MATERIAL GAIN"
    assert scored["layers"][0]["full_in_sample"]["hot_ids"] == [0, 1]
    for direction in scored["layers"][0]["directions"]:
        assert direction["hot_ids"] == [0, 1]
        assert direction["oracle_hot_ids"] == [[0, 1]] * 8


def test_all_layer_means_include_layer_zero_for_every_metric(harness):
    shape = _shape([[4, 0, 0, 0]] * 16, second_layer=[[1, 1, 1, 1]] * 16)
    scored = harness.score_shape(shape, n_experts=4, top_k=1)
    assert scored["layer_count"] == 2
    assert scored["full_in_sample"]["in_sample_hit"] == 0.625
    for direction in scored["directions"]:
        assert direction["heldout_hit"] == 0.625
        assert direction["in_sample_hit"] == 0.625
        assert direction["oracle_hit"] == 0.625
        assert direction["gain"] == 0.375
        assert direction["relative_gain"] == 1.5
        assert direction["transfer_loss"] == 0.0
        assert direction["worst_layer"]["layer"] == 1
        assert direction["worst_layer"]["heldout_hit"] == 0.25
    assert scored["verdict"] == "HOTSET TRANSFERS"


def test_exact_tokens_times_top_k_count_identity_is_accepted(harness):
    scored = harness.score_shape(_shape([[2, 2, 2, 2]] * 16), n_experts=4, top_k=2)
    assert scored["verdict"] == "NO MATERIAL GAIN"
    assert scored["invariants_ok"] is True


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_steps",
        "unequal_steps",
        "negative",
        "float_count",
        "bool_count",
        "wrong_expert_length",
        "wrong_count_total",
        "too_few_chunks",
        "wrong_tokens",
        "no_layers",
        "unknown_layer",
        "duplicate_router",
    ],
)
def test_invalid_capture_is_void_before_scoring(harness, mutation):
    shape = _shape([[1, 1, 1, 1] for _ in range(16)])
    if mutation == "missing_steps":
        del shape["steps"]
    elif mutation == "unequal_steps":
        shape["layers"][0]["per_step_counts"].pop()
    elif mutation == "negative":
        shape["layers"][0]["per_step_counts"][0] = [-1, 3, 1, 1]
    elif mutation == "float_count":
        shape["layers"][0]["per_step_counts"][0] = [1.0, 1, 1, 1]
    elif mutation == "bool_count":
        shape["layers"][0]["per_step_counts"][0] = [True, 1, 1, 1]
    elif mutation == "wrong_expert_length":
        shape["layers"][0]["per_step_counts"][0] = [2, 1, 1]
    elif mutation == "wrong_count_total":
        shape["layers"][0]["per_step_counts"][0] = [2, 1, 1, 1]
    elif mutation == "too_few_chunks":
        shape["chunks_available"] = 15
    elif mutation == "wrong_tokens":
        shape["tokens_per_step"] = 5
    elif mutation == "no_layers":
        shape["layers"] = []
    elif mutation == "unknown_layer":
        shape["layers"][0]["layer"] = -1
    elif mutation == "duplicate_router":
        shape["layers"].append(copy.deepcopy(shape["layers"][0]))
    scored = harness.score_shape(shape, n_experts=4, top_k=1)
    assert scored["verdict"] == "VOID"
    assert scored["invariants_ok"] is False
    assert scored["errors"]


@pytest.mark.parametrize("experts,top_k", [(3, 1), (0, 1), (4, 0), (4, 5)])
def test_invalid_expert_or_top_k_config_is_void(harness, experts, top_k):
    scored = harness.score_shape(_shape([[1, 1, 1, 1]] * 16), experts, top_k)
    assert scored["verdict"] == "VOID"


@pytest.mark.parametrize("steps", [8, 15, 17])
def test_non_sixteen_step_capture_cannot_claim_default_verdict(harness, steps):
    shape = _shape([[1, 1, 1, 1]] * steps)
    shape["steps"] = steps
    shape["chunks_available"] = steps
    assert harness.score_shape(shape, 4, 1)["verdict"] == "VOID"


def test_huge_invalid_step_count_is_void_without_constructing_split_indexes(harness):
    shape = _shape([[1, 1, 1, 1]] * 16)
    shape["steps"] = 10**20
    shape["chunks_available"] = 10**20
    scored = harness.score_shape(shape, 4, 1)
    assert scored["verdict"] == "VOID"
    assert scored["invariants_ok"] is False
    assert scored["errors"]


def test_custom_split_is_recomputed_both_ways_but_diagnostics_only(harness):
    shape = _shape([[4, 0, 0, 0]] * 4 + [[0, 4, 0, 0]] * 12)
    scored = harness.score_shape(shape, 4, 1, split=4)
    assert scored["invariants_ok"] is True
    assert scored["verdict"] is None
    assert scored["rule_applicable"] is False
    assert scored["diagnostics_only"] is True
    forward, reverse = scored["layers"][0]["directions"]
    assert forward["train_steps"] == [0, 1, 2, 3]
    assert forward["evaluation_steps"] == list(range(4, 16))
    assert reverse["train_steps"] == list(range(4, 16))
    assert reverse["evaluation_steps"] == [0, 1, 2, 3]
    assert forward["heldout_hit"] == reverse["heldout_hit"] == 0.0
    assert harness.score_shape(shape, 4, 1, split=8)["rule_applicable"] is True


@pytest.mark.parametrize(
    "directions,valid,want",
    [
        ([_direction(0.4, 0.0), _direction(0.4, 0.0)], False, "VOID"),
        ([_direction(0.10, 0.05), _direction(0.10, -0.50)], True, "HOTSET TRANSFERS"),
        ([_direction(0.10, 0.05000001), _direction(0.10, 0.0)], True, "GAIN WITH DRIFT"),
        ([_direction(0.10, 0.90), _direction(0.09, 0.0)], True, "DIRECTION-DEPENDENT"),
        ([_direction(0.09, 0.0), _direction(0.10, 0.90)], True, "DIRECTION-DEPENDENT"),
        ([_direction(0.09, -0.1), _direction(0.0, -0.1)], True, "NO MATERIAL GAIN"),
        ([_direction(0.10 - 5e-13, 0.05 + 5e-13), _direction(0.1, 0.0)],
         True, "HOTSET TRANSFERS"),
        ([_direction(0.10 - 2e-12, 0.0), _direction(0.1, 0.0)],
         True, "DIRECTION-DEPENDENT"),
        ([_direction(0.1, 0.05 + 2e-12), _direction(0.1, 0.0)], True, "GAIN WITH DRIFT"),
        ([_direction(float("nan"), 0.0), _direction(0.1, 0.0)], True, "VOID"),
        ([_direction(0.1, float("inf")), _direction(0.1, 0.0)], True, "VOID"),
        ([_direction(0.1, 0.0)], True, "VOID"),
    ],
)
def test_rule_precedence_and_inclusive_boundaries(harness, directions, valid, want):
    assert harness.verdict(directions, invariants_ok=valid) == want


def test_replay_recalculates_saved_counts_and_does_not_trust_saved_summary(harness):
    source = {
        "schema_version": 1,
        "meta": {
            "n_experts": 4,
            "top_k": 1,
            "n_routers": 1,
            "routers": [{"layer": 0, "name": "model.layers.0.router"}],
            "rule": {"untrusted": True},
        },
        "corpora": [{"label": "uniform", "shapes": {"1x4": _shape([[1] * 4] * 16)}}],
    }
    source["corpora"][0]["shapes"]["1x4"]["summary"] = {"verdict": "HOTSET TRANSFERS"}
    before = copy.deepcopy(source)
    result = harness.replay_results(source)
    assert result["corpora"][0]["shapes"]["1x4"]["verdict"] == "NO MATERIAL GAIN"
    assert result["meta"]["rule"]["material_gain_min"] == 0.10
    assert "untrusted" not in result["meta"]["rule"]
    assert source == before


@pytest.mark.parametrize("missing_layer", [0, 1])
def test_replay_missing_discovered_layer_is_void_not_a_reduced_population(harness, missing_layer):
    shape = _shape(
        [[4, 0, 0, 0]] * 8 + [[0, 4, 0, 0]] * 8,
        second_layer=[[4, 0, 0, 0]] * 16,
    )
    source = {
        "schema_version": 1,
        "meta": {
            "n_experts": 4,
            "top_k": 1,
            "n_routers": 2,
            "routers": [
                {"layer": 0, "name": "model.layers.0.router"},
                {"layer": 1, "name": "model.layers.1.router"},
            ],
        },
        "corpora": [{"label": "drift", "shapes": {"1x4": shape}}],
    }
    intact = harness.replay_results(source)
    assert intact["corpora"][0]["shapes"]["1x4"]["verdict"] == "GAIN WITH DRIFT"
    shape["layers"].pop(missing_layer)
    result = harness.replay_results(source)["corpora"][0]["shapes"]["1x4"]
    assert result["verdict"] == "VOID"
    assert result["invariants_ok"] is False
    assert result["errors"]


@pytest.mark.parametrize("inventory", [None, [], [{"layer": 1, "name": "wrong.router"}]])
def test_replay_requires_saved_matching_router_inventory(harness, inventory):
    source = {
        "schema_version": 1,
        "meta": {"n_experts": 4, "top_k": 1, "routers": inventory},
        "corpora": [{"label": "uniform", "shapes": {"1x4": _shape([[1] * 4] * 16)}}],
    }
    scored = harness.replay_results(source)["corpora"][0]["shapes"]["1x4"]
    assert scored["verdict"] == "VOID"
    assert scored["invariants_ok"] is False
    assert scored["errors"]


def test_published_audit_recomputes_identity_and_population_gap_without_mutation(
    harness, tmp_path
):
    paths = []
    original = []
    for index in range(4):
        record = {
            "corpora": [{"label": "fixture", "shapes": {"1x4": {
                "traffic_caught_by_corpus_hot25_mean": 0.5,
                "layers": [
                    {"layer": 0, "counts": [3, 1, 0, 0], "top25_share": 0.75,
                     "one_layer_ahead": None},
                    {"layer": 1, "counts": [2, 1, 1, 0], "top25_share": 0.5,
                     "one_layer_ahead": {"traffic_caught_by_corpus_hot25": 0.5}},
                ],
            }}}],
        }
        path = tmp_path / f"published-{index}.json"
        payload = json.dumps(record).encode("utf-8")
        path.write_bytes(payload)
        paths.append(path)
        original.append(payload)
    result = harness.audit_published(paths)
    assert result["scored_layer_rows"] == 4
    assert result["all_layer_rows"] == 8
    assert result["max_abs_identity_difference"] == 0.0
    for file_result in result["files"]:
        assert file_result["scored_layer_rows"] == 1
        assert file_result["all_layer_rows"] == 2
        assert len(file_result["sha256"]) == 64
        assert file_result["sha256_16"] == file_result["sha256"][:16]
        shape = file_result["shapes"][0]
        assert shape["top25_mean_all_layers"] == 0.625
        assert shape["top25_mean_scored_layers"] == 0.5
        assert shape["hot25_mean_scored_layers"] == 0.5
        assert shape["mismatched_layer_summary_gap"] == 0.125
        assert shape["max_abs_count_top25_difference"] == 0.0
    assert [path.read_bytes() for path in paths] == original
