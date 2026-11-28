"""Micro-batch routing under layer-to-layer execution."""

import inspect
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from soup_cli.config.loader import load_config_from_string
from tests.test_l2l_moe import HIDDEN, LEVERS, SEQ, _assert_same, _build, _grads, _micro_batches
from tests.test_l2l_moe import _yaml as training_yaml

MBR = "l2l_route_preserving_mbr"
SIZE = "l2l_mbr_group_size"
SPLIT = "l2l_mbr_split_backward"
ORDERED = "l2l_mbr_ordered_backward"
SWITCHES = (MBR, SPLIT, ORDERED)
OPTIONS = [(name, True) for name in SWITCHES] + [(SIZE, 2)]
PREREQUISITES = {"l2l_two_read_experts": True, "l2l_grouped_expert_routes": True}


def _load(**overrides: Any) -> Any:
    options = {"l2l": True, "stream_expert_layout": "per_expert", "gradient_accumulation_steps": 5}
    options.update(overrides)
    return load_config_from_string(training_yaml(**options)).training


def _assert_config_error(message: str, *, field: str | None = None, **options: Any) -> None:
    with pytest.raises(ValueError) as caught:
        _load(**options)
    location = f"training -> {field}" if field else ""
    prefix = "" if message.startswith("Input should") else "Value error, "
    assert str(caught.value) == f"{location}: {prefix}{message}"


def test_mbr_is_off_by_default() -> None:
    training = _load(l2l=False, stream_expert_layout="fused")
    assert all(getattr(training, name) is False for name in SWITCHES)
    assert training.l2l_mbr_group_size is None


@pytest.mark.parametrize(("name", "value"), OPTIONS)
def test_mbr_options_require_l2l(name: str, value: Any) -> None:
    _assert_config_error(
        f"training.{name} is set but training.l2l is false: it changes how "
        f"L2L runs and nothing else reads it. Set l2l=true or drop {name}.",
        l2l=False,
        **{name: value},
    )


@pytest.mark.parametrize(("name", "value"), OPTIONS)
def test_mbr_options_require_per_expert_layout(name: str, value: Any) -> None:
    _assert_config_error(
        f"training.{name} acts on the routed experts of a per_expert cache and "
        f"a fused layout has none to read: set training.stream_expert_layout='per_expert' "
        f"or drop {name}.",
        stream_expert_layout="fused",
        **{name: value},
    )


@pytest.mark.parametrize("name", SWITCHES)
@pytest.mark.parametrize("value", [1, 0, "true"])
def test_mbr_switches_require_bools(name: str, value: Any) -> None:
    _assert_config_error(
        f"training.{name} must be true or false, got {type(value).__name__} {value!r}",
        field=name,
        **{name: value},
    )


@pytest.mark.parametrize("value", [1.0, "2"])
def test_mbr_group_size_requires_an_integer(value: Any) -> None:
    _assert_config_error(
        f"training.{SIZE} must be an integer or null, got {type(value).__name__} {value!r}",
        field=SIZE,
        **{SIZE: value},
    )


@pytest.mark.parametrize("value", [True, False])
def test_mbr_group_size_refuses_a_bool_with_the_shared_wording(value: Any) -> None:
    _assert_config_error(f"training.{SIZE} must be an int, not a bool", field=SIZE, **{SIZE: value})


@pytest.mark.parametrize("value", [0, -1])
def test_mbr_group_size_has_a_lower_bound(value: int) -> None:
    _assert_config_error("Input should be greater than or equal to 1", field=SIZE, **{SIZE: value})


@pytest.mark.parametrize(("name", "value"), [(SIZE, 2), (SPLIT, True), (ORDERED, True)])
def test_mbr_options_require_mbr_enabled(name: str, value: Any) -> None:
    _assert_config_error(f"training.{name} requires training.{MBR}=true", **{name: value})


@pytest.mark.parametrize("missing", PREREQUISITES)
def test_mbr_requires_both_expert_levers(missing: str) -> None:
    options = {**PREREQUISITES, missing: False, MBR: True}
    _assert_config_error(
        f"training.{MBR} requires both l2l_two_read_experts=true "
        "and l2l_grouped_expert_routes=true",
        **options,
    )


def test_mbr_group_size_cannot_exceed_accumulation_window() -> None:
    _assert_config_error(
        f"training.{SIZE} cannot exceed training.gradient_accumulation_steps",
        **{**PREREQUISITES, MBR: True, SIZE: 6},
    )


def test_mbr_backward_modes_are_mutually_exclusive() -> None:
    _assert_config_error(
        f"training.{ORDERED} and training.{SPLIT} exclude each other",
        **{**PREREQUISITES, MBR: True, ORDERED: True, SPLIT: True},
    )


@pytest.mark.parametrize(
    ("name", "before", "after"),
    [(MBR, False, True), (SIZE, None, 2), (SPLIT, False, True), (ORDERED, False, True)],
)
def test_mbr_recipe_refuses_changed_options_on_resume(
    tmp_path, monkeypatch, name: str, before: Any, after: Any,
) -> None:
    import json

    from soup_cli.utils.l2l_recipe import (
        RECIPE_NAME,
        resolve_l2l_recipe,
        validate_resume,
        write_recipe,
    )

    monkeypatch.chdir(tmp_path)
    options = {**PREREQUISITES, MBR: True}
    saved = resolve_l2l_recipe(_load(**{**options, name: before}))
    write_recipe(tmp_path, saved)
    values = json.loads((tmp_path / RECIPE_NAME).read_text(encoding="utf-8"))
    assert {key.removeprefix("l2l_") for key, _ in OPTIONS} <= values.keys()
    validate_resume(tmp_path, saved)
    recipe = resolve_l2l_recipe(_load(**{**options, name: after}))
    key = name.removeprefix("l2l_")
    assert values[key] == before
    with pytest.raises(ValueError) as caught:
        validate_resume(tmp_path, recipe)
    assert str(caught.value) == (
        f"cannot resume: the checkpoint was trained with {key}={before!r}, but this "
        f"run has {key}={after!r}. Restore the old value in the config."
    )


@pytest.mark.parametrize(
    ("name", "before", "after"),
    [(MBR, False, True), (SIZE, None, 2), (SPLIT, False, True), (ORDERED, False, True)],
)
def test_sft_resume_refuses_changed_mbr_setting_from_source_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, before: Any, after: Any,
) -> None:
    from unittest.mock import Mock

    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils.l2l_recipe import RECIPE_NAME, resolve_l2l_recipe, write_recipe

    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source"
    checkpoint = source / "checkpoint-5"
    checkpoint.mkdir(parents=True)
    destination = tmp_path / "destination"
    destination.mkdir()

    options = {
        **PREREQUISITES,
        "l2l": True,
        "stream_expert_layout": "per_expert",
        "gradient_accumulation_steps": 5,
        MBR: True,
    }
    saved = resolve_l2l_recipe(_load(**{**options, name: before}))
    write_recipe(source, saved)
    original_recipe = (source / RECIPE_NAME).read_bytes()

    config = load_config_from_string(training_yaml(**{**options, name: after}))
    wrapper = SFTTrainerWrapper.__new__(SFTTrainerWrapper)
    wrapper.config = config
    wrapper._quest_metadata = None
    wrapper._output_dir = str(destination)
    wrapper._l2l_recipe = resolve_l2l_recipe(config.training)
    train = Mock(side_effect=AssertionError("Trainer.train reached after a recipe mismatch"))
    wrapper.trainer = SimpleNamespace(model=None, args=SimpleNamespace(), train=train)

    with pytest.raises(ValueError) as caught:
        wrapper.train(resume_from_checkpoint=str(checkpoint))

    key = name.removeprefix("l2l_")
    assert str(caught.value) == (
        f"cannot resume: the checkpoint was trained with {key}={before!r}, but this "
        f"run has {key}={after!r}. Restore the old value in the config."
    )
    train.assert_not_called()
    assert (source / RECIPE_NAME).read_bytes() == original_recipe
    assert not (destination / RECIPE_NAME).exists()


def test_recipe_from_before_mbr_resumes_with_mbr_defaults(tmp_path) -> None:
    from soup_cli.utils.l2l_recipe import RECIPE_NAME, resolve_l2l_recipe, validate_resume

    (tmp_path / RECIPE_NAME).write_text(
        '{"requested_k": 5, "effective_k": 5, "activation_codec": "none", "auto_k": false}',
        encoding="utf-8",
    )
    validate_resume(tmp_path, resolve_l2l_recipe(_load()))


@pytest.mark.parametrize(("mode", "window_size"), [(SPLIT, 5), (ORDERED, 5), (ORDERED, 2)])
def test_l2l_trainer_dispatches_mbr_settings_to_the_step(
    monkeypatch, mode: str, window_size: int,
) -> None:
    pytest.importorskip("torch")
    from soup_cli.trainer import l2l_trainer
    from soup_cli.utils.l2l_recipe import resolve_l2l_recipe
    from tests.test_l2l_sft_dispatch import _builder

    calls = []
    attach = l2l_trainer.attach_l2l_state
    owner, build = _builder(monkeypatch, "none", calls, k=5)
    monkeypatch.setattr(l2l_trainer, "attach_l2l_state", attach)
    monkeypatch.setattr(l2l_trainer, "make_l2l_trainer_class", lambda _base: SimpleNamespace)
    owner.config.training = _load(**{**PREREQUISITES, MBR: True, SIZE: 3, mode: True})
    owner._l2l_recipe = resolve_l2l_recipe(owner.config.training)
    state = build()._soup_l2l

    stepper = SimpleNamespace(run=lambda *args, **kwargs: calls.append(kwargs), counters=None)
    monkeypatch.setattr(l2l_trainer, "_l2l_step_for", lambda *args: (stepper, None))
    monkeypatch.setattr(l2l_trainer, "_store_for", lambda *args: None)
    l2l_trainer.run_window(state, None, [None] * window_size, [None] * window_size, 1, "cpu")
    assert state.mbr_group_size == 3
    assert calls[-1]["route_preserving_mbr"] is True
    assert calls[-1]["mbr_group_size"] == min(3, window_size)
    assert calls[-1]["mbr_split_backward"] is (mode == SPLIT)
    assert calls[-1]["mbr_ordered_backward"] is (mode == ORDERED)


@pytest.fixture
def window(tmp_path, request):
    from soup_cli.utils import l2l_rng
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    options = getattr(request, "param", {})
    model, runtime = _build(tmp_path, **LEVERS["pool-all-levers"], **options)
    try:
        dropout = options.get("dropout", 0.0)
        if dropout:
            model.train()
        lengths = (5,) * 5 if dropout else (3, 6, 4, 5, 3)
        batches = [
            ids[:, :length].contiguous()
            for ids, length in zip(_micro_batches(5), lengths)
        ]
        calls = [capture_layer_call(model, ids) for ids in batches]
        if dropout:
            assert l2l_rng.has_draws(calls[0].draws)
        yield model, L2LStep(model, runtime, calls[0]), batches, calls
    finally:
        runtime.close()


def _run(
    window: Any, bf16: bool = False, *, store_options: dict | None = None, **options: Any,
) -> Any:
    import torch

    from soup_cli.utils.l2l_activations import ActivationStore

    model, stepper, batches, calls = window
    model.zero_grad(set_to_none=True)
    context = torch.autocast("cpu", dtype=torch.bfloat16) if bf16 else nullcontext()
    store = ActivationStore(
        n_layers=len(stepper.layers), k=len(batches), chunk_shape=(1, SEQ, HIDDEN),
        mb_shapes=[(*ids.shape, HIDDEN) for ids in batches],
        dtype=stepper.owner.embed_tokens.weight.dtype, device="cpu", pin=False,
        **(store_options or {}),
    )
    try:
        with context:
            losses = stepper.run(batches, store, calls=calls, **options)
    finally:
        store.close()
    grads = _grads(model)
    assert len(grads) == sum(
        "lora_" in name and parameter.requires_grad for name, parameter in model.named_parameters()
    )
    return losses, grads


@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16-autocast"])
@pytest.mark.parametrize(("mode", "group_size"), [("plain", 1), ("split", 3), ("ordered", 3)])
def test_mbr_matches_every_plain_l2l_loss_and_lora_gradient(
    window, bf16: bool, mode: str, group_size: int,
) -> None:
    expected = _run(window, bf16)
    stepper = window[1]
    baseline_reads = stepper.counters.expert_fetches
    actual = _run(
        window, bf16, route_preserving_mbr=True, mbr_group_size=group_size,
        mbr_split_backward=mode == "split", mbr_ordered_backward=mode == "ordered",
        collect_phase_stats=True,
    )
    _assert_same(actual, expected)
    for layer in (1, 2):
        stats = stepper.route_preserving_mbr_stats["layers"][layer]
        assert stats["init_forward"]["grouped_expert_dispatch_calls"] == (
            5 if mode == "plain" else 2
        )
        assert stats["recompute_forward"]["expert_fetches"] == 0
    if mode != "plain":
        assert 0 < stepper.counters.expert_fetches < baseline_reads


def test_split_backward_refuses_deferred_decoder_hooks(window) -> None:
    from soup_cli.utils.l2l_step import mbr_deferred_forward_hook

    @mbr_deferred_forward_hook
    def hook(_module: Any, _args: Any, output: Any) -> Any:
        return output + 0.01

    layer = window[1].layers[1].inner
    handle = layer.register_forward_hook(hook)
    try:
        with pytest.raises(RuntimeError) as caught:
            _run(window, route_preserving_mbr=True, mbr_split_backward=True)
        assert str(caught.value) == (
            "mbr_split_backward does not support deferred decoder forward hooks: "
            "a hook may read tensors that phase one would sum group-wide"
        )
    finally:
        handle.remove()


def test_deferred_decoder_hook_sees_materialized_experts_in_recompute(window) -> None:
    import torch

    from soup_cli.utils.l2l_step import mbr_deferred_forward_hook

    observed = []

    @mbr_deferred_forward_hook
    def hook(_module: Any, _args: Any, output: Any) -> None:
        observed.append(output.detach().clone())

    layer = window[1].layers[1].inner
    handle = layer.register_forward_hook(hook)
    try:
        # Group size one isolates the hook's recompute semantics from grouped GEMMs.
        _run(window, route_preserving_mbr=True, mbr_group_size=1, collect_phase_stats=True)
        assert len(observed) == 10
        for initial, recomputed in zip(observed[:5], observed[5:]):
            assert torch.equal(initial, recomputed)
        stats = window[1].route_preserving_mbr_stats["layers"][1]
        assert stats["recompute_forward"]["expert_fetches"] > 0
        assert layer.mlp.experts.materialize_recompute is False
    finally:
        handle.remove()


def test_mbr_refuses_changed_routes_during_recompute(window) -> None:
    import torch

    def change_route(module: Any, _args: Any, output: Any) -> Any:
        if not torch.is_grad_enabled():
            return output
        routes = output[-1].clone()
        routes[0, 0] = (routes[0, 0] + 1) % module.num_experts
        return (*output[:-1], routes)

    handle = window[1].layers[1].inner.mlp.gate.register_forward_hook(change_route)
    try:
        with pytest.raises(RuntimeError) as caught:
            _run(window, route_preserving_mbr=True, mbr_group_size=3)
        assert str(caught.value) == (
            "route-preserving MBR recompute changed 1 route slots at layer 1, micro-batch 0"
        )
    finally:
        handle.remove()


def test_mbr_group_size_cannot_exceed_store_capacity(window) -> None:
    with pytest.raises(ValueError) as caught:
        _run(window, route_preserving_mbr=True, mbr_group_size=6)
    assert str(caught.value) == "mbr_group_size=6 exceeds the 5-micro-batch capacity"


def test_mbr_requires_a_pooled_expert_runtime(window, monkeypatch) -> None:
    experts = window[1].layers[1].inner.mlp.experts
    monkeypatch.setattr(experts, "pool", None)
    with pytest.raises(ValueError) as caught:
        _run(window, route_preserving_mbr=True)
    assert str(caught.value) == "route-preserving MBR layer 1 requires an expert buffer pool"


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"mbr_decoupled_subwindow": 1}, "mbr_decoupled_subwindow must be a bool"),
        (
            {"route_preserving_mbr": False, "mbr_decoupled_subwindow": True},
            "mbr_decoupled_subwindow is only valid with route_preserving_mbr=True",
        ),
        (
            {"mbr_recompute_subwindow": 1},
            "mbr_recompute_subwindow is only valid with mbr_decoupled_subwindow=True",
        ),
        (
            {"mbr_decoupled_subwindow": True, "mbr_split_backward": True},
            "mbr_decoupled_subwindow excludes mbr_split_backward and mbr_ordered_backward",
        ),
        (
            {"mbr_decoupled_subwindow": True, "mbr_ordered_backward": True},
            "mbr_decoupled_subwindow excludes mbr_split_backward and mbr_ordered_backward",
        ),
    ],
)
def test_decoupled_backward_refuses_incompatible_options(window, options, error) -> None:
    options = {"route_preserving_mbr": True, **options}
    with pytest.raises(ValueError) as caught:
        _run(window, **options)
    assert str(caught.value) == error


@pytest.mark.parametrize("size", [True, 0, -1, 1.5])
def test_decoupled_backward_requires_a_positive_integer_subwindow(window, size) -> None:
    with pytest.raises(ValueError) as caught:
        _run(
            window, route_preserving_mbr=True, mbr_decoupled_subwindow=True,
            mbr_recompute_subwindow=size,
        )
    assert str(caught.value) == "mbr_recompute_subwindow must be a positive integer or None"


@pytest.mark.parametrize(
    ("setting", "value", "error"),
    [
        ("expert_gemm", "batch_invariant", "requires native expert GEMM"),
        ("group_compute", object(), "does not support grouped expert compute"),
    ],
)
def test_decoupled_backward_refuses_non_native_expert_compute(
    window, monkeypatch, setting, value, error,
) -> None:
    experts = window[1].layers[1].inner.mlp.experts
    owner = experts.pool if setting == "group_compute" else experts
    monkeypatch.setattr(owner, setting, value, raising=False)
    with pytest.raises(ValueError) as caught:
        _run(window, route_preserving_mbr=True, mbr_decoupled_subwindow=True)
    assert str(caught.value) == f"mbr_decoupled_subwindow {error}"


def test_decoupled_backward_requires_per_expert_layout(window, monkeypatch) -> None:
    from torch.nn import Identity

    mlp = window[1].layers[1].inner.mlp
    monkeypatch.setattr(mlp, "experts", Identity())
    with pytest.raises(ValueError) as caught:
        _run(window, route_preserving_mbr=True, mbr_decoupled_subwindow=True)
    assert str(caught.value) == (
        "route-preserving MBR layer 1 requires a streamed "
        "DeepseekV3DecoderLayer with per-expert routed weights"
    )


@pytest.mark.parametrize("method", ["forward", "backward"])
def test_decoupled_backward_refuses_an_overridden_expert_node(window, monkeypatch, method) -> None:
    experts = window[1].layers[1].inner.mlp.experts
    node = inspect.getclosurevars(type(experts)._forward_pooled).nonlocals["PooledWeightedExpert"]
    original = getattr(node, method)

    def override(*args: Any) -> Any:
        return original(*args)

    monkeypatch.setattr(node, method, staticmethod(override))
    with pytest.raises(ValueError) as caught:
        _run(window, route_preserving_mbr=True, mbr_decoupled_subwindow=True)
    assert str(caught.value) == (
        f"mbr_decoupled_subwindow requires the native pooled expert {method}"
    )


def test_decoupled_backward_refuses_deferred_decoder_hooks(window) -> None:
    from soup_cli.utils.l2l_step import mbr_deferred_forward_hook

    @mbr_deferred_forward_hook
    def observe(_module: Any, _args: Any, _output: Any) -> None:
        pytest.fail("the hook ran before the mode was refused")

    layer = window[1].layers[1].inner
    handle = layer.register_forward_hook(observe)
    try:
        with pytest.raises(ValueError) as caught:
            _run(window, route_preserving_mbr=True, mbr_decoupled_subwindow=True)
        assert str(caught.value) == (
            "mbr_decoupled_subwindow does not support deferred decoder forward hooks"
        )
    finally:
        handle.remove()


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("ids", "route-preserving MBR recompute changed 1 route slots at layer 2, micro-batch 0"),
        ("weights", "decoupled MBR gate weights changed between replays at layer 2, micro-batch 0"),
    ],
)
def test_decoupled_backward_checks_routes_again_when_building_the_graph(
    window, change, error,
) -> None:
    import torch

    def change_routes(module: Any, _args: Any, output: Any) -> Any:
        if not torch.is_grad_enabled():
            return output
        if change == "weights":
            return (*output[:-2], output[-2] + 0.25, output[-1])
        ids = output[-1].clone()
        ids[0, 0] = (ids[0, 0] + 1) % module.num_experts
        return (*output[:-1], ids)

    handle = window[1].layers[2].inner.mlp.gate.register_forward_hook(change_routes)
    try:
        with pytest.raises(RuntimeError) as caught:
            _run(window, route_preserving_mbr=True, mbr_decoupled_subwindow=True)
        assert str(caught.value) == error
    finally:
        handle.remove()


@pytest.mark.parametrize(
    ("window", "group_size", "subwindow", "autocast"),
    [
        ({}, 1, None, False),
        ({"shared_adapters": True}, 3, 2, False),
        ({}, 3, 1, True),
        ({"dtype": "bfloat16", "quant": "nf4"}, 3, 2, False),
        ({"dropout": 0.15, "shared_adapters": True}, 3, None, False),
    ],
    indirect=["window"],
    ids=["single", "mixed-length", "bf16-autocast", "nf4-bf16", "dropout"],
)
def test_decoupled_backward_preserves_losses_and_lora_gradients(
    window, group_size, subwindow, autocast,
) -> None:
    import torch

    torch.manual_seed(42)
    expected_losses, expected_grads = _run(window, autocast)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(42)
    losses, grads = _run(
        window, autocast, route_preserving_mbr=True, mbr_group_size=group_size,
        mbr_decoupled_subwindow=True, mbr_recompute_subwindow=subwindow, collect_phase_stats=True,
    )
    assert torch.equal(torch.get_rng_state(), expected_rng)
    torch.testing.assert_close(torch.stack(losses), torch.stack(expected_losses), atol=0, rtol=0)
    assert grads.keys() == expected_grads.keys()
    for name in expected_grads:
        torch.testing.assert_close(grads[name], expected_grads[name], atol=1e-6, rtol=0, msg=name)
    for idx in (1, 2):
        stats = window[1].route_preserving_mbr_stats["layers"][idx]
        assert stats["recompute_forward"]["expert_fetches"] == 0
        assert stats["recompute_forward"]["expert_bytes_read"] == 0
        assert (
            stats["backward_dX"]["expert_fetches"] == stats["init_forward"]["unique_expert_count"]
        )
        assert stats["route_id_mismatch_slots"] == 0


def test_decoupled_expert_vjp_matches_each_native_microbatch_segment(window) -> None:
    import torch

    experts = window[1].layers[1].inner.mlp.experts
    generator = torch.Generator().manual_seed(17)
    hidden = torch.randn(5, HIDDEN, generator=generator)
    weights = torch.rand(5, 2, generator=generator)
    grad = torch.randn(5, HIDDEN, generator=generator)
    ids = torch.tensor([[0, 1], [2, 0], [1, 0], [1, 2], [0, 2]])
    sizes = [2, 3]
    expected = [
        torch.autograd.grad(
            experts(rows.requires_grad_(), routes, routing.requires_grad_()),
            (rows, routing), seed,
        )
        for rows, routes, routing, seed in zip(
            hidden.split(sizes), ids.split(sizes), weights.split(sizes), grad.split(sizes)
        )
    ]
    before = experts.pool.fetches
    d_hidden, d_weights, expert_count = experts.backward_from_inputs(
        hidden, ids, weights, grad, group_sizes=sizes, autocast_kwargs={"enabled": False},
    )
    assert expert_count == experts.pool.fetches - before == 3
    for actual, segments in zip((d_hidden, d_weights), zip(*expected)):
        assert torch.equal(actual, torch.cat(segments))


@pytest.mark.parametrize("subwindow", [None, 2])
def test_decoupled_backward_bounds_live_attention_graphs_with_spill(
    window, tmp_path, monkeypatch, subwindow,
) -> None:
    import torch

    from soup_cli.utils.l2l_activations import ActivationStore, BufferedSpillFile

    reads = set()
    get = ActivationStore.get

    def read_once(store, layer, mb):
        assert (layer, mb) not in reads, "a spill read was requested out of order"
        reads.add((layer, mb))
        return get(store, layer, mb)

    monkeypatch.setattr(ActivationStore, "get", read_once)
    active = peak = 0

    def finished(_grad: Any) -> None:
        nonlocal active
        active -= 1

    def observe(_module: Any, _args: Any, kwargs: Any) -> None:
        nonlocal active, peak
        if torch.is_grad_enabled():
            active += 1
            peak = max(peak, active)
            kwargs["hidden_states"].register_hook(finished)

    layer = window[1].layers[1].inner
    handle = layer.self_attn.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        _run(
            window, route_preserving_mbr=True, mbr_group_size=3,
            mbr_decoupled_subwindow=True, mbr_recompute_subwindow=subwindow,
            store_options={
                "spill_layers": 2, "ring": 2,
                "spill_factory": lambda size: BufferedSpillFile(str(tmp_path / "spill.bin"), size),
            },
        )
    finally:
        handle.remove()
    assert active == 0
    assert peak == (subwindow or 1)
    assert len(reads) == 3 * 5
