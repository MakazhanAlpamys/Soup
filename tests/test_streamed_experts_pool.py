"""``StreamedExperts`` over an ``ExpertBufferPool`` and its opt-in levers (CPU).

The module reads each routed expert either through an ``ExpertFetcher`` or through the pool's one
bounded arena. Every comparison here is ``torch.equal`` against transformers' eager experts loop
(output, hidden-state gradient, routing-weight gradient) or against the same module with the lever
off. Fixture weights and routing come from integer arithmetic, never ``torch.randn``, so the
values do not depend on the CPU's instruction set.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

torch = pytest.importorskip("torch")

HIDDEN = 16
INTER = 8
EXPERTS = 8


def _ints(count: int, step: int, modulus: int, offset: int = 0) -> Any:
    """``count`` values in [-modulus // 2, modulus // 2] / 64, from integer arithmetic only."""
    raw = (torch.arange(count, dtype=torch.int64) * step + offset) % modulus
    return (raw - modulus // 2).to(torch.float32) / 64


def _experts(dtype: Any = torch.float32) -> Any:
    from transformers import DeepseekV3Config
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Experts

    config = DeepseekV3Config(
        hidden_size=HIDDEN, moe_intermediate_size=INTER, n_routed_experts=EXPERTS, n_group=2,
        topk_group=1, num_experts_per_tok=2,
    )
    config._experts_implementation = "eager"
    module = DeepseekV3Experts(config)
    with torch.no_grad():
        for position, param in enumerate((module.gate_up_proj, module.down_proj)):
            param.copy_(_ints(param.numel(), 37 + position, 23, 5 * position).reshape(param.shape))
    module.requires_grad_(False)
    return module.to(dtype)


def _inputs(dtype: Any = torch.float32, tokens: int = 12, top_k: int = 2, spread: int = EXPERTS):
    """Hidden states, routing indices (distinct per token) and weights, and an upstream gradient."""
    token = torch.arange(tokens, dtype=torch.int64)
    index = torch.stack([(token * 5 + slot) % spread for slot in range(top_k)], dim=1)
    hidden = _ints(tokens * HIDDEN, 7, 13).reshape(tokens, HIDDEN).to(dtype)
    weights = (((token[:, None] * 3 + torch.arange(top_k)[None, :] * 2) % 7 + 1) / 8).to(dtype)
    grad_out = _ints(tokens * HIDDEN, 11, 17, 3).reshape(tokens, HIDDEN).to(dtype)
    return hidden, index, weights, grad_out


def _store(tmp_path: Path, module: Any, dtype: str = "float32", **pool_kwargs: Any) -> Tuple:
    from soup_cli.utils.expert_stream import (
        ExpertBufferPool,
        ExpertShardReader,
        expert_allocation_budget,
        plan_expert_buffer,
        write_expert_shard,
    )

    path = str(tmp_path / "experts.safetensors")
    pairs = [
        (module.gate_up_proj[e].detach(), module.down_proj[e].detach()) for e in range(EXPERTS)
    ]
    info = write_expert_shard(path, iter(pairs), quantise=False)
    reader = ExpertShardReader(
        path, count=info.count, shapes=info.shapes, quant_specs={}, dtype=dtype, direct=False
    )
    plan = plan_expert_buffer(reader, dtype=dtype)
    pool = ExpertBufferPool(
        reader, device="cpu", dtype=dtype,
        weight_budget_bytes=expert_allocation_budget(plan, device="cpu"), **pool_kwargs,
    )
    return reader, pool


def _streamed(module: Any, reader: Any, pool: Any, layer_idx: int = 0, **kwargs: Any) -> Any:
    from soup_cli.utils.expert_stream import streamed_experts_class

    return streamed_experts_class()(reader, pool, module.act_fn, layer_idx, **kwargs)


def _run(module: Any, args: Tuple) -> Dict[str, Any]:
    hidden, index, weights, grad_out = args
    hidden = hidden.clone().requires_grad_(True)
    weights = weights.clone().requires_grad_(True)
    out = module(hidden, index, weights)
    out.backward(grad_out)
    return {"out": out.detach(), "hidden": hidden.grad, "weights": weights.grad}


def _assert_exact(got: Dict[str, Any], want: Dict[str, Any]) -> None:
    assert float(want["hidden"].abs().sum()) > 0.0 and float(want["weights"].abs().sum()) > 0.0
    for name in want:
        assert got[name].dtype == want[name].dtype, name
        assert torch.equal(got[name], want[name]), name


def _hit(index: Any) -> List[int]:
    return sorted({int(expert) for expert in index.flatten()})


ROUTINGS = {
    "all-experts-top2": dict(tokens=24, top_k=2),
    "top1": dict(tokens=24, top_k=1),
    "partial": dict(tokens=24, top_k=2, spread=3),
    "single-token": dict(tokens=1, top_k=2),
}
LEVERS = {
    "default": ({}, False),
    "two-reads": (dict(two_reads=True), False),
    "grouped": (dict(grouped_expert_routes=True), False),
    "fast-gather": ({}, True),
    "all-levers": (dict(two_reads=True, grouped_expert_routes=True), True),
}


@pytest.mark.parametrize("routing", sorted(ROUTINGS))
@pytest.mark.parametrize("lever", sorted(LEVERS))
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_pooled_experts_equal_the_eager_loop_under_every_lever(
    tmp_path: Path, dtype: str, lever: str, routing: str
) -> None:
    module = _experts(getattr(torch, dtype))
    reader, pool = _store(tmp_path, module, dtype)
    try:
        args = _inputs(getattr(torch, dtype), **ROUTINGS[routing])
        want = _run(module, args)
        kwargs, fast = LEVERS[lever]
        streamed = _streamed(module, reader, pool, **kwargs)
        streamed.fast_row_gather = fast
        got = _run(streamed, args)
        _assert_exact(got, want)
        assert [row["expert"] for row in streamed.last_trace] == _hit(args[1])
    finally:
        reader.close()
        pool.close()


def test_the_fetcher_path_takes_the_same_levers_and_stays_bit_equal(tmp_path: Path) -> None:
    from soup_cli.utils.expert_stream import (
        ExpertFetcher,
        ExpertShardReader,
        streamed_experts_class,
        write_expert_shard,
    )

    module = _experts()
    path = str(tmp_path / "experts.safetensors")
    info = write_expert_shard(
        path,
        iter([(module.gate_up_proj[e].detach(), module.down_proj[e].detach())
              for e in range(EXPERTS)]),
        quantise=False,
    )
    reader = ExpertShardReader(
        path, count=info.count, shapes=info.shapes, quant_specs={}, dtype="float32", direct=False
    )
    try:
        args = _inputs(tokens=24)
        want = _run(module, args)
        for kwargs, fast in LEVERS.values():
            streamed = streamed_experts_class()(
                reader, ExpertFetcher(device="cpu", dtype="float32"), module.act_fn, 0, **kwargs
            )
            streamed.fast_row_gather = fast
            _assert_exact(_run(streamed, args), want)
            assert streamed.pool is None and streamed.last_trace == []
    finally:
        reader.close()


# ------------------------------------------------------------------------------ routes
def _mask(index: Any) -> Any:
    return torch.nn.functional.one_hot(index, num_classes=EXPERTS).permute(2, 1, 0)


@pytest.mark.parametrize("routing", sorted(ROUTINGS))
def test_grouped_routes_yield_the_default_experts_and_rows_in_the_same_order(
    routing: str,
) -> None:
    from soup_cli.utils.expert_stream import _default_expert_routes, _grouped_expert_routes

    mask = _mask(_inputs(**ROUTINGS[routing])[1])
    hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
    default = list(_default_expert_routes(mask, hit))
    grouped = list(_grouped_expert_routes(mask))
    assert [row[0] for row in grouped] == [row[0] for row in default] and default
    for (_, want_pos, want_token), (_, got_pos, got_token) in zip(default, grouped):
        assert torch.equal(got_pos, want_pos) and torch.equal(got_token, want_token)


def _hand_mask(entries: dict, experts: int = 2) -> Any:
    mask = torch.zeros(experts, 1, 2, dtype=torch.int64)
    for (expert, slot, token), value in entries.items():
        mask[expert, slot, token] = value
    return mask


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        # a count above the coordinates the mask has: 2 against one nonzero entry
        ({(0, 0, 0): 2}, "grouped expert routes require a binary one-hot mask"),
        # a negative count: -1 against one nonzero entry
        ({(0, 0, 0): -1}, "grouped expert routes require a binary one-hot mask"),
        # counts that sum below the coordinates: one expert holds 2 and -1 (count 1, 2 entries)
        ({(0, 0, 0): 2, (0, 0, 1): -1}, "grouped expert routes omitted nonzero coordinates"),
    ],
)
def test_grouped_routes_refuse_a_mask_that_is_not_binary(entries: dict, message: str) -> None:
    from soup_cli.utils.expert_stream import _grouped_expert_routes

    with pytest.raises(ValueError, match=message):
        list(_grouped_expert_routes(_hand_mask(entries)))


def test_the_grouped_lever_is_off_by_default_and_takes_only_a_bool(tmp_path: Path) -> None:
    module = _experts()
    reader, pool = _store(tmp_path, module)
    try:
        assert _streamed(module, reader, pool).grouped_expert_routes is False
        assert _streamed(module, reader, pool, grouped_expert_routes=True).grouped_expert_routes
        for bad in (1, 0, "true", None):
            with pytest.raises(ValueError, match="grouped_expert_routes must be a bool"):
                _streamed(module, reader, pool, grouped_expert_routes=bad)
    finally:
        reader.close()
        pool.close()


# ------------------------------------------------------------------------ fast row gather
def _bits(tensor: Any) -> Any:
    return tensor.contiguous().view(torch.int32 if tensor.dtype == torch.float32 else torch.int16)


def _graph_names(root: Any) -> set:
    seen, names, stack = set(), set(), [root]
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        names.add(type(node).__name__)
        stack.extend(child for child, _ in node.next_functions)
    return names


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_row_gather_matches_indexing_forward_and_backward_bit_for_bit(dtype: Any) -> None:
    from soup_cli.utils.expert_stream import fast_gather_functions

    gather_rows, _ = fast_gather_functions()
    source = _ints(40 * 16, 5, 29).reshape(40, 16).to(dtype)
    index = (torch.arange(17, dtype=torch.int64) * 7 + 3) % 40
    assert index.unique().numel() == 17
    grad = _ints(17 * 16, 3, 19).reshape(17, 16).to(dtype)
    grad[3, :4] = -0.0  # a zero gradient keeps its sign through 0 + (-0) = +0
    grad[5, 4:8] = 0.0
    want_source = source.clone().requires_grad_(True)
    want = want_source[index]
    want.backward(grad)
    got_source = source.clone().requires_grad_(True)
    got = gather_rows.apply(got_source, index)
    got.backward(grad)
    assert torch.equal(_bits(got.detach()), _bits(want.detach()))
    assert torch.equal(_bits(got_source.grad), _bits(want_source.grad))


def test_row_gather_without_a_grad_source_builds_no_node() -> None:
    from soup_cli.utils.expert_stream import fast_gather_functions

    gather_rows, _ = fast_gather_functions()
    out = gather_rows.apply(torch.ones(6, 4), torch.tensor([1, 3]))
    assert out.grad_fn is None


@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_route_weight_gather_matches_indexing_forward_and_backward_bit_for_bit(top_k: int) -> None:
    from soup_cli.utils.expert_stream import fast_gather_functions

    _, gather_weights = fast_gather_functions()
    tokens = 30
    weights = ((torch.arange(tokens * top_k) * 5 % 11 + 1) / 16).reshape(tokens, top_k)
    flat = (torch.arange(11, dtype=torch.int64) * 7 + 2) % (tokens * top_k)
    assert flat.unique().numel() == 11  # unique (token, slot) pairs, like one expert
    token_idx, top_k_pos = flat // top_k, flat % top_k
    grad = _ints(11, 3, 9).reshape(11, 1)
    grad[2] = -0.0
    want_w = weights.clone().requires_grad_(True)
    want = want_w[token_idx, top_k_pos, None]
    want.backward(grad)
    got_w = weights.clone().requires_grad_(True)
    got = gather_weights.apply(got_w, token_idx, top_k_pos)
    got.backward(grad)
    assert got.shape == want.shape
    assert torch.equal(_bits(got.detach()), _bits(want.detach()))
    assert torch.equal(_bits(got_w.grad), _bits(want_w.grad))


def test_fast_row_gather_is_off_by_default_and_skips_the_index_backward(tmp_path: Path) -> None:
    module = _experts()
    reader, pool = _store(tmp_path, module)
    try:
        streamed = _streamed(module, reader, pool)
        assert streamed.fast_row_gather is False
        hidden, index, weights, _ = _inputs(tokens=24)
        hidden.requires_grad_(True)
        assert "IndexBackward0" in _graph_names(streamed(hidden, index, weights).grad_fn)
        streamed.fast_row_gather = True
        assert "IndexBackward0" not in _graph_names(streamed(hidden, index, weights).grad_fn)
    finally:
        reader.close()
        pool.close()


# -------------------------------------------------------------------------- two-read scope
def _scoped_forward(streamed: Any, args: Tuple, replay: bool, record: Any) -> Any:
    from soup_cli.utils.expert_stream import _TwoReadScope

    hidden, index, weights, _ = args
    hidden = hidden.clone().requires_grad_(True)
    weights = weights.clone().requires_grad_(True)
    with _TwoReadScope(record, replay=replay):
        return streamed(hidden, index, weights)


def test_a_two_read_recompute_leases_no_expert_and_the_backward_leases_each_once(
    tmp_path: Path,
) -> None:
    from soup_cli.utils.expert_stream import _TwoReadRecord

    module = _experts()
    reader, pool = _store(tmp_path, module)
    try:
        args = _inputs(tokens=24)
        hit = len(_hit(args[1]))
        streamed = _streamed(module, reader, pool, two_reads=True)
        pool.phase_read_tracking = True
        record = _TwoReadRecord()
        forward = _scoped_forward(streamed, args, False, record)
        assert pool.fetches == hit and {row["status"] for row in streamed.last_trace} == {"ok"}
        recompute = _scoped_forward(streamed, args, True, record)
        assert pool.fetches == hit  # the recompute read nothing
        assert {row["status"] for row in streamed.last_trace} == {"replay"}
        assert [row["logical_bytes_read"] for row in streamed.last_trace] == [0] * hit
        assert not recompute.detach().any()
        forward.backward(args[3])
        assert pool.fetches == 2 * hit
        counts = pool.phase_read_snapshot()[0]
        assert counts["init_forward"]["expert_fetches"] == hit
        assert counts["backward_dX"]["expert_fetches"] == hit
        assert "recompute_forward" not in counts
    finally:
        reader.close()
        pool.close()


def test_materialize_recompute_makes_the_recompute_read_and_compute_the_experts(
    tmp_path: Path,
) -> None:
    from soup_cli.utils.expert_stream import _TwoReadRecord

    module = _experts()
    reader, pool = _store(tmp_path, module)
    try:
        args = _inputs(tokens=24)
        hit = len(_hit(args[1]))
        streamed = _streamed(module, reader, pool, two_reads=True)
        assert streamed.materialize_recompute is False
        streamed.materialize_recompute = True
        pool.phase_read_tracking = True
        record = _TwoReadRecord()
        forward = _scoped_forward(streamed, args, False, record)
        recompute = _scoped_forward(streamed, args, True, record)
        assert pool.fetches == 2 * hit
        assert pool.phase_read_snapshot()[0]["recompute_forward"]["expert_fetches"] == hit
        assert {row["status"] for row in streamed.last_trace} == {"ok"}
        assert torch.equal(recompute.detach(), forward.detach())
    finally:
        reader.close()
        pool.close()


# ---------------------------------------------------------------------------- refusals
def _call(streamed: Any, args: Tuple, **edits: Any) -> Any:
    hidden, index, weights, _ = args
    hidden = edits.get("hidden", hidden)
    index = edits.get("index", index)
    weights = edits.get("weights", weights)
    return streamed(hidden, index, weights)


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda a: dict(hidden=a[0][0]), "expected hidden \\[tokens, hidden\\]"),
        (lambda a: dict(weights=a[2][:, :1]), "expected hidden \\[tokens, hidden\\]"),
        (lambda a: dict(index=a[1][:5], weights=a[2][:5]),
         "routing token count or top-k does not match hidden states"),
        (lambda a: dict(index=a[1].to(torch.int32)),
         "routing indices must be int64 and weights must be floating point"),
        (lambda a: dict(hidden=a[0].to(torch.bfloat16)),
         "hidden dtype must match the expert pool compute dtype"),
        (lambda a: dict(hidden=torch.ones(a[0].shape[0], HIDDEN + 1)),
         "hidden dimension does not match the expert pool"),
        (lambda a: dict(weights=torch.full_like(a[2], float("nan"))),
         "routing weights must be finite"),
        (lambda a: dict(index=a[1] + EXPERTS), "routing contains an expert outside the reader"),
        (lambda a: dict(index=a[1] - 20), "routing contains an expert outside the reader"),
    ],
)
def test_the_pooled_loop_refuses_routing_it_cannot_visit(
    tmp_path: Path, edit: Any, match: str
) -> None:
    module = _experts()
    reader, pool = _store(tmp_path, module)
    try:
        args = _inputs(tokens=12)
        streamed = _streamed(module, reader, pool)
        with pytest.raises(ValueError, match=match):
            _call(streamed, args, **edit(args))
        assert pool.fetches == 0 and not streamed._forward_lock.locked()
    finally:
        reader.close()
        pool.close()


def test_top_k_over_the_expert_count_is_refused(tmp_path: Path) -> None:
    module = _experts()
    reader, pool = _store(tmp_path, module)
    try:
        hidden, _, _, _ = _inputs(tokens=4)
        index = torch.zeros(4, EXPERTS + 1, dtype=torch.int64)
        weights = torch.ones(4, EXPERTS + 1)
        with pytest.raises(ValueError, match="routing top-k exceeds the expert count"):
            _streamed(module, reader, pool)(hidden, index, weights)
    finally:
        reader.close()
        pool.close()


def test_a_differentiable_call_refuses_an_activation_other_than_the_stateless_silu(
    tmp_path: Path,
) -> None:
    module = _experts()
    reader, pool = _store(tmp_path, module)
    try:
        args = _inputs(tokens=12)
        streamed = _streamed(module, reader, pool)
        streamed.act_fn = torch.nn.GELU()
        with pytest.raises(ValueError, match="expert backward requires an unmodified stateless"):
            _run(streamed, args)
        assert pool.fetches == 0
        with torch.no_grad():  # nothing to differentiate: the activation is not judged
            streamed(*args[:3])
        streamed.act_fn = module.act_fn
        handle = streamed.act_fn.register_forward_hook(lambda *_: None)
        try:
            with pytest.raises(ValueError, match="expert backward requires an unmodified"):
                _run(streamed, args)
        finally:
            handle.remove()
    finally:
        reader.close()
        pool.close()


def test_an_activation_with_parameters_is_refused_at_construction(tmp_path: Path) -> None:
    from soup_cli.utils.expert_stream import streamed_experts_class

    module = _experts()
    reader, pool = _store(tmp_path, module)
    try:
        cls = streamed_experts_class()
        for bad in (torch.nn.PReLU(), "silu"):
            with pytest.raises(
                ValueError, match="expert activation must be callable without parameters"
            ):
                cls(reader, pool, bad, 0)
    finally:
        reader.close()
        pool.close()


def test_a_higher_order_gradient_is_refused_not_taken_against_stale_arena_memory(
    tmp_path: Path,
) -> None:
    module = _experts()
    reader, pool = _store(tmp_path, module)
    try:
        for kwargs in ({}, dict(two_reads=True)):
            hidden, index, weights, _ = _inputs(tokens=12)
            hidden.requires_grad_(True)
            out = _streamed(module, reader, pool, **kwargs)(hidden, index, weights)
            with pytest.raises(RuntimeError, match="does not support higher-order gradients"):
                torch.autograd.grad(out.sum(), hidden, create_graph=True)
    finally:
        reader.close()
        pool.close()


def test_a_second_forward_while_one_is_active_is_refused(tmp_path: Path) -> None:
    module = _experts()
    reader, pool = _store(tmp_path, module)
    try:
        streamed = _streamed(module, reader, pool)
        assert streamed._forward_lock.acquire(blocking=False)
        try:
            outcome: List[BaseException] = []

            def other() -> None:
                try:
                    streamed(*_inputs(tokens=4)[:3])
                except BaseException as error:  # noqa: BLE001 - reported to the main thread
                    outcome.append(error)

            thread = threading.Thread(target=other)
            thread.start()
            thread.join()
            assert len(outcome) == 1 and "expert forward is already active" in str(outcome[0])
        finally:
            streamed._forward_lock.release()
    finally:
        reader.close()
        pool.close()


def test_a_pooled_module_is_leased_not_fetched(tmp_path: Path) -> None:
    module = _experts()
    reader, pool = _store(tmp_path, module)
    try:
        with pytest.raises(RuntimeError, match="pooled experts are leased from their pool"):
            _streamed(module, reader, pool).fetch(0)
    finally:
        reader.close()
        pool.close()


def test_a_reader_the_pool_was_not_built_for_is_refused_at_construction(tmp_path: Path) -> None:
    module = _experts()
    reader, pool = _store(tmp_path, module)
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    from soup_cli.utils.expert_stream import ExpertShardReader, write_expert_shard

    path = str(other_dir / "experts.safetensors")
    info = write_expert_shard(
        path, iter([(torch.ones(2 * (INTER + 1), HIDDEN), torch.ones(HIDDEN, INTER + 1))]),
        quantise=False,
    )
    stranger = ExpertShardReader(
        path, count=info.count, shapes=info.shapes, quant_specs={}, dtype="float32", direct=False
    )
    try:
        with pytest.raises(
            ValueError, match="expert reader does not match the buffer pool; build a matching pool"
        ):
            _streamed(module, stranger, pool)
    finally:
        stranger.close()
        reader.close()
        pool.close()


def test_last_trace_has_one_row_per_visited_expert_and_replaces_itself(tmp_path: Path) -> None:
    module = _experts()
    reader, pool = _store(tmp_path, module)
    try:
        args = _inputs(tokens=24, spread=3)
        streamed = _streamed(module, reader, pool)
        with torch.no_grad():
            streamed(*args[:3])
        assert [row["expert"] for row in streamed.last_trace] == [0, 1, 2]
        assert all(row["layer"] == 0 and row["status"] == "ok" for row in streamed.last_trace)
        assert sum(row["token_routes"] for row in streamed.last_trace) == 24 * 2
        assert all(row["logical_bytes_read"] > 0 for row in streamed.last_trace)
        with torch.no_grad():
            streamed(*_inputs(tokens=2, top_k=1)[:3])
        assert [row["expert"] for row in streamed.last_trace] == [0, 5]
    finally:
        reader.close()
        pool.close()


@pytest.mark.parametrize("two_reads", [False, True], ids=["default", "two-reads"])
def test_the_backward_validates_the_activation_again_before_leasing_an_expert(
    tmp_path: Path, two_reads: bool
) -> None:
    module = _experts()
    reader, pool = _store(tmp_path, module)
    try:
        hidden, index, weights, grad_out = _inputs(tokens=12)
        hidden.requires_grad_(True)
        streamed = _streamed(module, reader, pool, two_reads=two_reads)
        out = streamed(hidden, index, weights)
        reads = pool.fetches
        assert reads == len(_hit(index))
        handle = streamed.act_fn.register_forward_hook(lambda *_: None)
        try:
            with pytest.raises(
                ValueError, match="expert backward requires an unmodified stateless SiLU activation"
            ):
                out.backward(grad_out)
        finally:
            handle.remove()
        assert pool.fetches == reads
    finally:
        reader.close()
        pool.close()


# ------------------------------------------------------------------------------ install
def _layer(experts: Any) -> Any:
    class DeepseekV3DecoderLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = torch.nn.Module()
            self.mlp.experts = experts

    return DeepseekV3DecoderLayer()


def test_install_with_a_pool_replaces_every_layer_and_the_pool_serves_them(
    tmp_path: Path,
) -> None:
    from soup_cli.utils.expert_stream import install_expert_streaming, streamed_experts_class

    module = _experts()
    reader, pool = _store(tmp_path, module)
    try:
        layers = [_layer(_experts()), _layer(_experts())]
        converted = install_expert_streaming(
            layers, {0: reader, 1: reader}, pool, grouped_expert_routes=True
        )
        assert converted == 2
        for layer in layers:
            assert isinstance(layer.mlp.experts, streamed_experts_class())
            assert layer.mlp.experts.pool is pool and layer.mlp.experts.grouped_expert_routes
        args = _inputs(tokens=12)
        _assert_exact(_run(layers[1].mlp.experts, args), _run(module, args))
    finally:
        reader.close()
        pool.close()


def test_install_validates_every_layer_before_replacing_any(tmp_path: Path) -> None:
    from soup_cli.utils.expert_stream import install_expert_streaming

    module = _experts()
    reader, pool = _store(tmp_path, module)
    try:
        good, bad = _layer(_experts()), _layer(_experts())
        bad.mlp.experts.gate_up_proj.requires_grad_(True)
        original = good.mlp.experts
        with pytest.raises(ValueError, match="routed-expert weights must stay frozen"):
            install_expert_streaming([good, bad], {0: reader, 1: reader}, pool)
        assert good.mlp.experts is original
    finally:
        reader.close()
        pool.close()


@pytest.mark.parametrize(
    ("setup", "match"),
    [
        (lambda layer: layer.mlp.experts.register_parameter(
            "extra", torch.nn.Parameter(torch.zeros(2), requires_grad=False)),
         "routed experts must hold exactly the frozen gate_up_proj and down_proj"),
        (lambda layer: layer.mlp.experts.gate_up_proj.requires_grad_(True),
         "routed-expert weights must stay frozen"),
    ],
)
def test_install_with_a_pool_refuses_unfrozen_or_extra_expert_parameters(
    tmp_path: Path, setup: Any, match: str
) -> None:
    from soup_cli.utils.expert_stream import install_expert_streaming

    module = _experts()
    reader, pool = _store(tmp_path, module)
    try:
        layer = _layer(_experts())
        setup(layer)
        with pytest.raises(ValueError, match=match):
            install_expert_streaming([layer], {0: reader}, pool)
    finally:
        reader.close()
        pool.close()


def test_install_refuses_a_reader_outside_the_layers_and_a_non_bool_lever(tmp_path: Path) -> None:
    from soup_cli.utils.expert_stream import install_expert_streaming

    module = _experts()
    reader, pool = _store(tmp_path, module)
    try:
        layer = _layer(_experts())
        for bad_index in (1, -1, True):
            with pytest.raises(
                ValueError, match="expert reader layer index is outside the supplied layers"
            ):
                install_expert_streaming([layer], {bad_index: reader}, pool)
        with pytest.raises(ValueError, match="grouped_expert_routes must be a bool"):
            install_expert_streaming([layer], {0: reader}, pool, grouped_expert_routes=1)
        # No layer and no reader: only the install-level check can see the bad value.
        for bad in (1, 0, "true", None):
            with pytest.raises(ValueError, match="grouped_expert_routes must be a bool"):
                install_expert_streaming([], {}, pool, grouped_expert_routes=bad)
        assert install_expert_streaming([], {}, pool, grouped_expert_routes=True) == 0
    finally:
        reader.close()
        pool.close()


def test_install_refuses_two_layers_that_share_one_experts_module(tmp_path: Path) -> None:
    from soup_cli.utils.expert_stream import install_expert_streaming

    module = _experts()
    reader, pool = _store(tmp_path, module)
    try:
        layer = _layer(_experts())
        with pytest.raises(ValueError, match="aliased modules cannot take two replacements"):
            install_expert_streaming([layer, layer], {0: reader, 1: reader}, pool)
    finally:
        reader.close()
        pool.close()


def test_install_refuses_two_parents_that_share_one_experts_module(tmp_path: Path) -> None:
    from soup_cli.utils.expert_stream import install_expert_streaming

    module = _experts()
    reader, pool = _store(tmp_path, module)
    try:
        shared = _experts()
        first, second = _layer(shared), _layer(shared)
        with pytest.raises(ValueError, match="aliased modules cannot take two replacements"):
            install_expert_streaming([first, second], {0: reader, 1: reader}, pool)
        assert first.mlp.experts is shared and second.mlp.experts is shared
    finally:
        reader.close()
        pool.close()
