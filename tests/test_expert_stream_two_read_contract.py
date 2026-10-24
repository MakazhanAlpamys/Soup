"""Two-read expert streaming beyond one forward + one backward per step.

Each checkpointed decoder-layer call gets its own ``_TwoReadRecord`` and pushes a scope on a
per-thread stack (``expert_stream._SCOPES.stack``) for its forward and for every recompute.
These tests drive the step shapes that depend on that discipline: gradient accumulation,
two forwards before one backward, ``retain_graph`` double backward, a ``no_grad`` or dropped
forward before a train step, and an exception inside the forward, the recompute or the
backward. Each one runs on the resident eager model, the default (three-read) streamed path
and two reads, and compares loss and every LoRA gradient bit for bit, counts expert reads,
and checks that no scope or record is left behind.
"""

from __future__ import annotations

import gc

import pytest

from tests.test_expert_stream import _lora, _shard, _tiny_deepseek_v3

PATHS = ("resident", "default", "two-reads")


def _build(tmp_path, buffers, refill_before_backward):
    """``{path: (model, runtime or None)}`` holding equal adapters, and the input ids."""
    import torch
    from peft import get_peft_model
    from transformers import DeepseekV3ForCausalLM

    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    weights, _ = _tiny_deepseek_v3(tmp_path)
    index = _shard(tmp_path, weights, expert_layout="per_expert")
    nets = {}
    for path, two_read_experts in (("default", False), ("two-reads", True)):
        model, runtime = build_streamed_model(
            model_id=weights, shard_dir=str(tmp_path / "shards"), index=index,
            lora_config=_lora(), device="cpu", dtype="float32", buffers=buffers, pin=False,
            seed=3, two_read_experts=two_read_experts,
            refill_before_backward=refill_before_backward,
        )
        generator = torch.Generator().manual_seed(17)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)
        nets[path] = (model.eval(), runtime)
    resident = DeepseekV3ForCausalLM.from_pretrained(weights, dtype=torch.float32)
    resident.set_experts_implementation("eager")
    ref = get_peft_model(resident, _lora()).eval()
    src = {
        k.replace(".inner.", "."): v
        for k, v in nets["default"][0].state_dict().items() if "lora_" in k
    }
    dst = {k: v for k, v in ref.state_dict().items() if "lora_" in k}
    assert src and set(src) == set(dst)
    for key, value in src.items():
        dst[key].copy_(value)
    nets["resident"] = (ref, None)
    ids = torch.randint(0, 128, (3, 12), generator=torch.Generator().manual_seed(29))
    return nets, ids


def _close(nets):
    for _, runtime in nets.values():
        if runtime is not None:
            runtime.close()


@pytest.fixture(scope="module")
def one_backward_per_forward(tmp_path_factory):
    """The production shape: two layer buffers, nothing refilled before the backward."""
    nets, ids = _build(tmp_path_factory.mktemp("two_read_contract"), 2, False)
    yield nets, ids
    _close(nets)


@pytest.fixture(scope="module")
def graph_kept_across_forwards(tmp_path_factory):
    """A layer pool holding every layer, and a private head weight (#1049): the layer
    stream's own requirements for a graph that outlives the next forward."""
    nets, ids = _build(tmp_path_factory.mktemp("two_read_contract_kept"), 4, True)
    yield nets, ids
    _close(nets)


def _stack():
    from soup_cli.utils import expert_stream

    return list(getattr(expert_stream._SCOPES, "stack", None) or [])


def _live_records():
    from soup_cli.utils import expert_stream

    gc.collect()
    return sum(isinstance(obj, expert_stream._TwoReadRecord) for obj in gc.get_objects())


def _assert_clean():
    """No scope on this thread's stack, and no record outlives the graphs it served."""
    assert _stack() == []
    assert _live_records() == 0


def _reads(runtime):
    return [reader.fetches for reader in runtime.expert_readers] if runtime else None


def _grads(net):
    from soup_cli.utils.layer_stream_runtime import canonical_named_parameters

    return {
        name: parameter.grad.detach().clone()
        for name, parameter in canonical_named_parameters(net)
        if "lora_" in name and parameter.grad is not None
    }


def _assert_equal(got, want):
    assert got and set(got) == set(want)
    for name, value in want.items():
        assert torch_equal(got[name], value), name


def torch_equal(a, b):
    import torch

    return torch.equal(a, b)


def _single_step(net, ids):
    net.zero_grad(set_to_none=True)
    loss = net(input_ids=ids, labels=ids).loss
    loss.backward()
    return loss.detach(), _grads(net)


# --------------------------------------------------------------------------
# a. gradient accumulation over several micro-batches
# --------------------------------------------------------------------------
@pytest.mark.parametrize("micro_batches", [2, 3])
def test_accumulated_micro_batches_equal_resident_and_default(
    one_backward_per_forward, micro_batches
):
    """Forward + backward per micro-batch, gradients accumulating in ``.grad``: every
    micro-batch's loss and the accumulated LoRA gradients are bit-equal on all three
    paths. Per micro-batch, two reads read exactly twice its forward's reads (the default
    path three times), and each backward leaves the scope stack empty."""
    nets, ids = one_backward_per_forward
    results = {}
    for path in PATHS:
        net, runtime = nets[path]
        net.zero_grad(set_to_none=True)
        losses = []
        for i in range(micro_batches):
            batch = ids[i:i + 1]
            before = _reads(runtime)
            loss = net(input_ids=batch, labels=batch).loss
            forward = _reads(runtime)
            loss.backward()
            after = _reads(runtime)
            losses.append(loss.detach())
            del loss
            _assert_clean()
            if runtime is not None:
                visits = [f - b for f, b in zip(forward, before)]
                assert len(visits) == 2 and all(v > 0 for v in visits)
                per_read = 2 if path == "two-reads" else 3
                assert [a - b for a, b in zip(after, before)] == [per_read * v for v in visits]
        results[path] = (losses, _grads(net))
    want_losses, want = results["resident"]
    # Layer 0 sits below both MoE layers: its gradient crossed every streamed backward.
    assert any(".layers.0." in name and float(g.abs().sum()) > 0 for name, g in want.items())
    for path in ("default", "two-reads"):
        losses, got = results[path]
        assert all(torch_equal(a, b) for a, b in zip(losses, want_losses)), path
        _assert_equal(got, want)


# --------------------------------------------------------------------------
# b. two forwards, then one backward of their sum
# --------------------------------------------------------------------------
def test_two_forwards_then_one_backward_equal_resident_and_default(graph_kept_across_forwards):
    """Each forward's checkpoint call holds its own record; one backward recomputes both
    and replays each from its own decisions. Two reads read each expert visit twice."""
    nets, ids = graph_kept_across_forwards
    results = {}
    for path in PATHS:
        net, runtime = nets[path]
        net.zero_grad(set_to_none=True)
        before = _reads(runtime)
        first = net(input_ids=ids[:1], labels=ids[:1]).loss
        second = net(input_ids=ids[1:], labels=ids[1:]).loss
        forward = _reads(runtime)
        assert _stack() == []
        if path == "two-reads":
            assert _live_records() == 4  # 2 forwards x 2 MoE layers, alive with the graphs
        (first + second).backward()
        after = _reads(runtime)
        losses = (first.detach(), second.detach())
        del first, second
        _assert_clean()
        if runtime is not None:
            visits = [f - b for f, b in zip(forward, before)]
            per_read = 2 if path == "two-reads" else 3
            assert [a - b for a, b in zip(after, before)] == [per_read * v for v in visits]
        results[path] = (losses, _grads(net))
    for path in ("default", "two-reads"):
        assert all(torch_equal(a, b) for a, b in zip(results[path][0], results["resident"][0]))
        _assert_equal(results[path][1], results["resident"][1])


@pytest.mark.parametrize("path", ["default", "two-reads"])
def test_a_graph_kept_across_a_forward_is_refused_by_a_pool_smaller_than_the_model(
    one_backward_per_forward, path
):
    """With fewer layer buffers than layers the next forward refills the slots the first
    graph's backward needs: the layer stream's slot check or autograd's version check on
    the refilled (shared, unprivate) head weight refuses loudly, on both paths alike. That
    leaves nothing behind for the next step, which equals a fresh one."""
    nets, ids = one_backward_per_forward
    net, _ = nets[path]
    want_loss, want = _single_step(nets["resident"][0], ids)
    net.zero_grad(set_to_none=True)
    first = net(input_ids=ids[:1], labels=ids[:1]).loss
    second = net(input_ids=ids[1:], labels=ids[1:]).loss
    with pytest.raises(
        RuntimeError, match="Raise training.stream_buffers|modified by an inplace operation"
    ):
        (first + second).backward()
    del first, second
    _assert_clean()
    loss, got = _single_step(net, ids)
    assert torch_equal(loss, want_loss)
    _assert_equal(got, want)


# --------------------------------------------------------------------------
# c. backward(retain_graph=True) twice on one graph
# --------------------------------------------------------------------------
def test_retain_graph_double_backward_doubles_the_single_backward_gradients(
    graph_kept_across_forwards,
):
    """Both backwards recompute from the same record (its cursor resets per recompute):
    the gradients are exactly twice a single backward's on every path. Two reads read the
    forward's visits once more per backward (default: twice more)."""
    nets, ids = graph_kept_across_forwards
    want_loss, single = _single_step(nets["resident"][0], ids)
    for path in PATHS:
        net, runtime = nets[path]
        net.zero_grad(set_to_none=True)
        before = _reads(runtime)
        loss = net(input_ids=ids, labels=ids).loss
        forward = _reads(runtime)
        loss.backward(retain_graph=True)
        once = _reads(runtime)
        loss.backward(retain_graph=True)
        twice = _reads(runtime)
        assert torch_equal(loss.detach(), want_loss), path
        del loss
        _assert_clean()
        _assert_equal(_grads(net), {name: 2 * g for name, g in single.items()})
        if runtime is not None:
            visits = [f - b for f, b in zip(forward, before)]
            per_backward = 1 if path == "two-reads" else 2
            assert [o - f for o, f in zip(once, forward)] == [per_backward * v for v in visits]
            assert [t - o for t, o in zip(twice, once)] == [per_backward * v for v in visits]


# --------------------------------------------------------------------------
# d. a no_grad forward, or a dropped grad forward, before a train step
# --------------------------------------------------------------------------
@pytest.mark.parametrize("before_step", ["no_grad", "dropped-graph"])
@pytest.mark.parametrize("path", ["default", "two-reads"])
def test_a_forward_before_the_train_step_leaks_nothing_into_it(
    one_backward_per_forward, path, before_step
):
    """An eval forward under ``no_grad`` runs no checkpoint and records nothing; a grad
    forward whose graph is dropped frees its records. The train step after either equals
    a fresh resident step, loss and every LoRA gradient."""
    import torch

    nets, ids = one_backward_per_forward
    want_loss, want = _single_step(nets["resident"][0], ids)
    net, _ = nets[path]
    other = ids.flip(0)
    if before_step == "no_grad":
        with torch.no_grad():
            net(input_ids=other, labels=other)
        assert _stack() == [] and _live_records() == 0
    else:
        dropped = net(input_ids=other, labels=other).loss
        assert _stack() == []
        assert _live_records() == (2 if path == "two-reads" else 0)
        del dropped
    _assert_clean()
    loss, got = _single_step(net, ids)
    _assert_clean()
    assert torch_equal(loss, want_loss)
    _assert_equal(got, want)


# --------------------------------------------------------------------------
# e. an exception inside the forward, the recompute or the backward
# --------------------------------------------------------------------------
class _InjectedError(Exception):
    pass


@pytest.mark.parametrize("where", ["forward-fetch", "recompute-router", "backward-fetch"])
@pytest.mark.parametrize("path", ["default", "two-reads"])
def test_an_exception_mid_step_leaves_no_state_for_the_next_step(
    one_backward_per_forward, path, where
):
    """A fetch of the last MoE layer raising in the forward, its router raising in the
    checkpoint recompute, or a fetch raising in the expert's own backward: the failure
    propagates unchanged and the scope stack is empty at once. A backward that failed
    leaves torch's checkpoint frames (and with them their records) alive until the next
    step, on both paths alike; replays never cross frames, so the next step equals a fresh
    resident step and frees them."""
    nets, ids = one_backward_per_forward
    want_loss, want = _single_step(nets["resident"][0], ids)
    net, _ = nets[path]
    moe = net.get_submodule("base_model.model.model.layers")[-1].inner.mlp
    calls = {"fetch": 0, "router": 0}
    # Raise on this call number of the last MoE layer's fetch / router (0: never).
    raise_at = {"fetch": 0, "router": 0}
    stack_at_router = []

    def fetch(expert, real=moe.experts.fetch):
        calls["fetch"] += 1
        if calls["fetch"] == raise_at["fetch"]:
            raise _InjectedError("fetch")
        return real(expert)

    def router_hook(_module, _args):
        calls["router"] += 1
        stack_at_router.append(len(_stack()))
        if calls["router"] == raise_at["router"]:
            raise _InjectedError("router")

    moe.experts.fetch = fetch
    handle = moe.gate.register_forward_pre_hook(router_hook)
    try:
        net.zero_grad(set_to_none=True)
        if where == "forward-fetch":
            raise_at["fetch"] = 1
        with pytest.raises(_InjectedError, match=where.split("-")[1]):
            loss = net(input_ids=ids, labels=ids).loss
            if where == "recompute-router":
                raise_at["router"] = calls["router"] + 1
            elif where == "backward-fetch":
                # The default recompute re-reads every forward visit before the expert's
                # own backward reads it; two reads go straight to the backward.
                visits = calls["fetch"]
                raise_at["fetch"] = visits + (visits if path == "default" else 0) + 1
            loss.backward()
        loss = None
    finally:
        handle.remove()
        del moe.experts.fetch
    if where == "recompute-router":
        # Raised inside the recompute: under its two-read scope on that path only.
        assert stack_at_router[-1] == (1 if path == "two-reads" else 0)
    assert _stack() == []
    if where == "forward-fetch":
        assert _live_records() == 0
    loss, got = _single_step(net, ids)
    _assert_clean()
    assert torch_equal(loss, want_loss)
    _assert_equal(got, want)


# --------------------------------------------------------------------------
# f. forward hooks: supported on the default path, refused on two reads
# --------------------------------------------------------------------------
def _moe_blocks(net):
    for layer in net.get_submodule("base_model.model.model.layers"):
        mlp = getattr(layer, "inner", layer).mlp
        if hasattr(mlp, "gate"):
            yield mlp


def _step_with_moe_hook(net, ids, scale):
    """One step with ``output * scale`` hooked onto every MoE block; the values each hook
    call saw and ``scale.grad``."""
    import torch

    net.zero_grad(set_to_none=True)
    seen = []

    def hook(_module, _args, output):
        seen.append(output.detach().clone())
        return output * scale

    handles = [mlp.register_forward_hook(hook) for mlp in _moe_blocks(net)]
    try:
        with torch.enable_grad():
            net(input_ids=ids, labels=ids).loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    return seen, scale.grad


def test_a_moe_output_hook_sees_the_resident_values_on_the_default_path(one_backward_per_forward):
    """The default path's replay refetches the routed experts: the hook sees the resident MoE
    output in the forward and in the recompute."""
    import torch

    nets, ids = one_backward_per_forward
    want, _ = _step_with_moe_hook(nets["resident"][0], ids, torch.ones((), requires_grad=True))
    got, _ = _step_with_moe_hook(nets["default"][0], ids, torch.ones((), requires_grad=True))
    assert len(want) == 2 and len(got) == 4  # forward + recompute per MoE layer
    for forward, resident in zip(got[:2], want):
        assert torch.equal(forward, resident)
    for recompute, resident in zip(reversed(got[2:]), want):  # last layer recomputes first
        assert torch.equal(recompute, resident)


def test_a_trainable_scale_on_the_moe_output_gets_the_resident_gradient_on_the_default_path(
    one_backward_per_forward,
):
    """The default three-read path supports a value-dependent hook: its replay refetches
    the routed experts, so the hook's gradient equals the resident one."""
    import torch

    nets, ids = one_backward_per_forward
    _, want = _step_with_moe_hook(nets["resident"][0], ids, torch.ones((), requires_grad=True))
    _, got = _step_with_moe_hook(nets["default"][0], ids, torch.ones((), requires_grad=True))
    assert torch.equal(got, want)


def _inner(net, idx):
    return net.get_submodule("base_model.model.model.layers")[idx].inner


#: Modules whose forward output differs in a two-read recompute, as reached from the model.
REFUSED_HOOK_SITES = {
    "moe-block": lambda net: _inner(net, 1).mlp,
    "routed-experts": lambda net: _inner(net, 2).mlp.experts,
    "decoder-layer": lambda net: _inner(net, 1),
}


@pytest.mark.parametrize("site", sorted(REFUSED_HOOK_SITES))
def test_a_forward_hook_registered_after_build_is_refused_before_any_gradient(
    one_backward_per_forward, site
):
    """A forward hook added after ``build_streamed_model`` on a module whose output the
    recompute changes: the forward raises before the layer runs, no LoRA gradient is
    written, and the hook never ran. After removing it the next step equals resident."""
    nets, ids = one_backward_per_forward
    net, runtime = nets["two-reads"]
    _, want = _single_step(nets["resident"][0], ids)
    net.zero_grad(set_to_none=True)
    calls = []
    handle = REFUSED_HOOK_SITES[site](net).register_forward_hook(
        lambda _m, _a, output: calls.append(1)
    )
    before = _reads(runtime)
    try:
        with pytest.raises(RuntimeError, match="two-read expert streaming refuses decoder layer"):
            net(input_ids=ids, labels=ids)
    finally:
        handle.remove()
    assert calls == []
    assert _grads(net) == {}
    assert _stack() == []
    refused_at = 2 if site == "routed-experts" else 1
    assert _reads(runtime)[refused_at - 1:] == before[refused_at - 1:]
    loss, got = _single_step(net, ids)
    _assert_clean()
    _assert_equal(got, want)


def test_a_forward_hook_added_between_forward_and_backward_is_refused_in_the_recompute(
    one_backward_per_forward,
):
    """The forward ran clean; a value-dependent hook added before ``backward`` is refused
    when the recompute starts, before it can see a recompute output. As after any failed
    backward (section e), torch's checkpoint frames keep their records until the next
    step; that step equals resident and leaves nothing behind."""
    import torch

    nets, ids = one_backward_per_forward
    net, _ = nets["two-reads"]
    _, want = _single_step(nets["resident"][0], ids)
    net.zero_grad(set_to_none=True)
    scale = torch.ones((), requires_grad=True)
    loss = net(input_ids=ids, labels=ids).loss
    calls = []

    def hook(_module, _args, output):
        calls.append(1)
        return output * scale

    handle = _inner(net, 2).mlp.register_forward_hook(hook)
    try:
        with pytest.raises(RuntimeError, match="refuses decoder layer 2: the MoE block"):
            loss.backward()
    finally:
        handle.remove()
    assert calls == [] and scale.grad is None
    assert _stack() == []
    del loss
    _, got = _single_step(net, ids)
    _assert_clean()
    _assert_equal(got, want)


def test_a_global_forward_hook_is_refused(one_backward_per_forward):
    import torch

    nets, ids = one_backward_per_forward
    net, _ = nets["two-reads"]
    handle = torch.nn.modules.module.register_module_forward_hook(lambda *_: None)
    try:
        with pytest.raises(RuntimeError, match="a global forward hook"):
            net(input_ids=ids, labels=ids)
    finally:
        handle.remove()
    assert _stack() == []


def test_hooks_outside_the_changed_outputs_keep_two_reads_exact(one_backward_per_forward):
    """Diagnostic hooks of the kind a gradient check uses stay allowed: a forward pre-hook on
    the wrapper that registers a dX gradient hook, forward hooks on the routers and the
    attention, and a forward pre-hook on the MoE block. Loss, every LoRA gradient and the
    captured dX equal the resident model's; reads stay at two per forward visit."""
    nets, ids = one_backward_per_forward

    def run(net):
        net.zero_grad(set_to_none=True)
        seen = {"dX": {}, "router": 0, "attn": 0, "pre": 0}
        layers = net.get_submodule("base_model.model.model.layers")
        handles = []
        for idx, layer in enumerate(layers):
            inner = getattr(layer, "inner", layer)

            def on_input(_m, args, idx=idx):
                if args[0].requires_grad:
                    args[0].register_hook(lambda g, idx=idx: seen["dX"].__setitem__(idx, g.clone()))

            handles.append(layer.register_forward_pre_hook(on_input))
            handles.append(inner.self_attn.register_forward_hook(
                lambda *_: seen.__setitem__("attn", seen["attn"] + 1)))
            if hasattr(inner.mlp, "gate"):
                handles.append(inner.mlp.gate.register_forward_hook(
                    lambda *_: seen.__setitem__("router", seen["router"] + 1)))
                handles.append(inner.mlp.register_forward_pre_hook(
                    lambda *_: seen.__setitem__("pre", seen["pre"] + 1)))
        try:
            loss, grads = _single_step(net, ids)
        finally:
            for handle in handles:
                handle.remove()
        return loss, grads, seen

    want_loss, want, want_seen = run(nets["resident"][0])
    net, runtime = nets["two-reads"]
    before = _reads(runtime)
    loss = net(input_ids=ids, labels=ids).loss
    visits = [f - b for f, b in zip(_reads(runtime), before)]
    del loss
    got_loss, got, got_seen = run(net)
    _assert_clean()
    assert torch_equal(got_loss, want_loss)
    _assert_equal(got, want)
    assert set(got_seen["dX"]) == set(want_seen["dX"]) and got_seen["dX"]
    for idx, grad in want_seen["dX"].items():
        assert torch_equal(got_seen["dX"][idx], grad), idx
    # Forward + recompute: each hook runs twice per layer on the streamed path.
    assert (got_seen["router"], got_seen["pre"]) == (4, 4)
    assert got_seen["attn"] == 6
    after = _reads(runtime)
    assert [a - f for a, f in zip(after, [b + v for b, v in zip(before, visits)])] == [
        2 * v for v in visits
    ]


def test_a_value_dependent_moe_hook_is_refused_on_two_reads_not_silently_wrong(
    one_backward_per_forward,
):
    """The hook that gets wrong gradients without the guard (``output * scale``) is refused
    before any backward: no gradient reaches ``scale`` or the adapters."""
    import torch

    nets, ids = one_backward_per_forward
    net, _ = nets["two-reads"]
    scale = torch.ones((), requires_grad=True)
    with pytest.raises(RuntimeError, match="the MoE block"):
        _step_with_moe_hook(net, ids, scale)
    assert scale.grad is None and _grads(net) == {}
    _assert_clean()


# --------------------------------------------------------------------------
# g. training.stream_recompute (#2152): layers that skip the checkpoint recompute
# --------------------------------------------------------------------------
@pytest.mark.parametrize("path", ["default", "two-reads"])
def test_retained_layers_skip_the_two_read_context_and_stay_exact(
    graph_kept_across_forwards, path
):
    """``set_stream_recompute(model, False)`` runs each layer without a checkpoint, so no
    two-read scope or record exists and the hook guard is not consulted. Both paths then
    read each expert visit twice (forward + backward), the gradients equal resident, and
    switching the recompute back restores the path's own read count and exactness."""
    from soup_cli.utils.layer_stream_runtime import set_stream_recompute

    nets, ids = graph_kept_across_forwards
    _, want = _single_step(nets["resident"][0], ids)
    net, runtime = nets[path]
    assert set_stream_recompute(net, False) == 3
    try:
        before = _reads(runtime)
        net.zero_grad(set_to_none=True)
        loss = net(input_ids=ids, labels=ids).loss
        forward = _reads(runtime)
        assert _stack() == []
        loss.backward()
        after = _reads(runtime)
        visits = [f - b for f, b in zip(forward, before)]
        assert all(v > 0 for v in visits)
        assert [a - b for a, b in zip(after, before)] == [2 * v for v in visits]
        _assert_equal(_grads(net), want)
        del loss
        _assert_clean()
    finally:
        set_stream_recompute(net, True)
    before = _reads(runtime)
    loss, got = _single_step(net, ids)
    per_read = 2 if path == "two-reads" else 3
    assert sum(_reads(runtime)) - sum(before) == per_read * sum(visits)
    _assert_clean()
    _assert_equal(got, want)
