"""Expert-granularity streaming for MoE decoder layers.

Layer streaming moves one decoder layer at a time, so the largest layer bounds
the card. A DeepSeek-V3-class MoE layer is mostly routed experts: one Kimi K2
layer (384 experts of 3 x 7168 x 2048) is ~9 GB at NF4, larger than an 8 GB
card on its own. Here the routed experts of a MoE layer live in their own file,
one tensor pair per expert, and the layer's experts module is replaced by
``StreamedExperts``: it loads each expert the router picked, uses it, and drops
it. The decoder-layer pool then carries only the dense part (attention, norms,
router, shared expert).

Numerics: ``StreamedExperts`` follows transformers' eager experts loop op for
op (the same body in ``DeepseekV3Experts``, ``OlmoeExperts``,
``Qwen3MoeExperts``, ``MixtralExperts``), so it equals a resident module run
with ``experts_implementation="eager"`` bit for bit, forward and backward.
Expert weights are frozen; the backward needs only the gradient of the
expert's input, so it fetches the expert again and recomputes the expert MLP
on a local graph. Inside a checkpointed decoder layer that is three reads of
every routed-to expert per step (forward, recompute, backward).

``two_reads=True`` makes it two. The routing multiply moves inside the expert's
autograd node, which then saves only the expert's input and routing weights. The
decoder layer's checkpoint gets :func:`two_read_checkpoint_contexts`: during its
recompute the MoE block's router hands back the forward's top-k decisions instead
of selecting again, and each expert node saves its inputs without reading the
expert. The backward reads each expert once and differentiates the same graph
the eager loop does, so the numbers do not change.

What it buys: the VRAM bound, nothing else. ``benchmarks/probe-moe-expert-coverage.md``
measured that a training step routes to 96.7-100% of the experts of a layer, so
an expert-streamed step reads at least the bytes a layer-streamed step reads.
"""

from __future__ import annotations

import functools
import sys
import threading
import types
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from soup_cli.utils import expert_shard as _expert_shard
from soup_cli.utils.expert_shard import (  # noqa: F401 - storage layer, re-exported
    _MAX_EXPERT_BYTES,
    _MAX_EXPERTS,
    DOWN,
    EXPERT_LAYOUT_PER_EXPERT,
    EXPERT_PARTS,
    GATE_UP,
    ExpertShardInfo,
    ExpertShardReader,
    _checked_shape,
    _expected_tensors,
    _stored_torch_dtype,
    aligned_staging,
    expert_key,
    expert_shard_name,
    write_expert_shard,
)

#: The storage limits now live in ``expert_shard``, where the reader and writer
#: read them. This module used to own them, and callers (tests, probes) set them
#: here; a set on this module is forwarded so that keeps working.
_FORWARDED_LIMITS = frozenset({"_MAX_EXPERTS", "_MAX_EXPERT_BYTES"})


class _ForwardingModule(types.ModuleType):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in _FORWARDED_LIMITS:
            setattr(_expert_shard, name, value)
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _ForwardingModule

#: Per thread, the two-read checkpoint scopes entered and not yet left, innermost
#: last. The forward enters one on the caller's thread, a recompute on the thread
#: that runs the backward.
_SCOPES = threading.local()


def expert_bytes(
    shapes: Mapping[str, Tuple[int, ...]], *, dtype_bytes: int, quantised: bool
) -> Tuple[int, int]:
    """(stored bytes, dense compute bytes) of ONE expert.

    Stored NF4 is half a byte per weight plus one fp32 absmax per 64-weight
    block (the nested statistics are smaller still and not counted).
    """
    weights = sum(shape[0] * shape[1] for shape in shapes.values())
    dense = weights * dtype_bytes
    if not quantised:
        return dense, dense
    stored = weights // 2 + (weights // 64) * 4
    return stored, dense


def expert_stream_vram_bytes(
    shapes: Mapping[str, Tuple[int, ...]], *, dtype_bytes: int, quantised: bool
) -> int:
    """Peak VRAM the expert path adds to a step, for the pre-flight.

    One expert's packed copy plus its dense weights, twice: the backward holds
    the expert it recomputes while the caching allocator may still keep the
    previous one's blocks.
    """
    stored, dense = expert_bytes(shapes, dtype_bytes=dtype_bytes, quantised=quantised)
    packed = stored if quantised else 0
    return 2 * (packed + dense)


class ExpertFetcher:
    """Turns stored expert tensors into dense weights on the compute device.

    Owns the one host staging buffer every reader it serves reads into, so a
    60-layer store costs one expert's span of host RAM, not sixty. Not
    thread-safe: one fetch at a time (the streamed forward and backward are
    sequential); a concurrent fetch would overwrite the staging under another.
    """

    def __init__(self, *, device: str, dtype: str, codes: Optional[Mapping[str, Any]] = None):
        self.device = device
        self.dtype = dtype
        self.codes = dict(codes or {})
        self._staging_owner: Any = None
        self._staging: Any = None

    def _staging_for(self, reader: ExpertShardReader) -> Any:
        if self._staging is None or self._staging.numel() < reader.max_span:
            self._staging_owner, self._staging = aligned_staging(reader.max_span)
        return self._staging

    def fetch(self, reader: ExpertShardReader, expert: int) -> Tuple[Any, Any]:
        from soup_cli.utils.layer_stream_runtime import _torch_dtype, rebuild_quant_state

        host = reader.read_host(expert, self._staging_for(reader))
        target = _torch_dtype(self.dtype)
        dense = []
        for part in EXPERT_PARTS:
            key = expert_key(expert, part)
            spec = reader.quant_specs.get(part)
            if spec is None:
                # copy=True: on a CPU target `.to` would hand back the staging
                # view, which the next read overwrites.
                dense.append(host[key].to(device=self.device, dtype=target, copy=True))
                continue
            from bitsandbytes.functional import dequantize_4bit

            moved = {
                name: tensor.to(self.device, copy=True)
                for name, tensor in host.items()
                if name == key or name.startswith(key + "::")
            }
            state = rebuild_quant_state(key, moved, spec, self.codes)
            dense.append(dequantize_4bit(moved[key], quant_state=state).to(target))
        return dense[0], dense[1]


class _TwoReadRecord:
    """What one checkpointed call of a decoder layer keeps for its recompute:
    each top-k result its routers took in the forward, in call order."""

    __slots__ = ("topk", "cursor")

    def __init__(self) -> None:
        #: ``(signature, (values, indices))`` per top-k call.
        self.topk: list = []
        self.cursor = 0


class _TwoReadScope:
    """Makes ``record`` the active scope: the forward's, or the recompute's."""

    def __init__(
        self, record: _TwoReadRecord, replay: bool, guard: Optional["_TwoReadHookGuard"] = None
    ) -> None:
        self.record = record
        self.replay = replay
        self.guard = guard

    def __enter__(self) -> "_TwoReadScope":
        if self.guard is not None:
            # Before the layer runs, in the forward and again in every recompute, so a
            # hook registered at any time is refused before it sees a recompute output.
            self.guard()
        if self.replay:
            # Every recompute replays from the first decision.
            self.record.cursor = 0
        stack = getattr(_SCOPES, "stack", None)
        if stack is None:
            stack = _SCOPES.stack = []
        stack.append(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        _SCOPES.stack.pop()


def _active_scope() -> Optional[_TwoReadScope]:
    stack = getattr(_SCOPES, "stack", None)
    return stack[-1] if stack else None


class _TwoReadHookGuard:
    """Refuses a two-read checkpoint call while a forward hook would see a recompute output.

    In the recompute the routed experts return zeros, so the outputs of the experts module,
    of the MoE block holding it and of the decoder layer differ from the forward's; every
    other module inside the layer (attention, norms, router, shared expert) sees the
    forward's values. A forward hook on one of those three modules, or a global forward
    hook, is refused; no hook is judged safe by what it does. Forward pre-hooks and hooks
    on any other module are not checked, and neither is a patched or replaced ``forward``.
    """

    __slots__ = ("layer_idx", "watched")

    def __init__(self, layer: Any) -> None:
        cls = streamed_experts_class()
        watched: Dict[int, Tuple[str, Any]] = {id(layer): ("the decoder layer", layer)}
        layer_idx = None
        for name, module in layer.named_modules():
            if not (isinstance(module, cls) and module.two_reads):
                continue
            layer_idx = module.layer_idx
            parent_name = name.rpartition(".")[0]
            parent = layer.get_submodule(parent_name) if parent_name else layer
            watched.setdefault(id(parent), (f"the MoE block `{parent_name}`", parent))
            watched.setdefault(id(module), (f"the routed experts `{name}`", module))
        self.layer_idx = layer_idx
        self.watched = tuple(watched.values())

    def __call__(self) -> None:
        import torch.nn.modules.module as nn_module

        where = None
        hooks = getattr(nn_module, "_global_forward_hooks", None)
        if hooks:
            where = "a global forward hook (register_module_forward_hook), which runs on it,"
        else:
            for label, module in self.watched:
                hooks = module._forward_hooks
                if hooks:
                    where = f"{label}, which has a forward hook,"
                    break
        if where is None:
            return
        hook = next(iter(hooks.values()))
        name = getattr(hook, "__qualname__", type(hook).__name__)
        raise RuntimeError(
            f"two-read expert streaming refuses decoder layer {self.layer_idx}: {where} "
            f"({name}). Its checkpoint recompute runs the routed experts at zero, so a "
            f"forward hook on the decoder layer, its MoE block or its routed experts would "
            f"see a different output there and could produce wrong gradients without an "
            f"error. Remove the hook, move it to the router, the attention or the streamed "
            f"layer wrapper, or build with two_read_experts=False."
        )


def two_read_checkpoint_contexts(
    guard: Optional[_TwoReadHookGuard] = None,
) -> Tuple[_TwoReadScope, _TwoReadScope]:
    """``context_fn`` for ``torch.utils.checkpoint.checkpoint`` around a decoder
    layer holding two-read experts: one fresh record, entered by the forward and
    by every recompute of this call. :func:`checkpoint_context_fn` binds ``guard``."""
    record = _TwoReadRecord()
    return (_TwoReadScope(record, replay=False, guard=guard),
            _TwoReadScope(record, replay=True, guard=guard))


def checkpoint_context_fn(layer: Any) -> Optional[Any]:
    """The ``context_fn`` a checkpointed ``layer`` needs: :func:`two_read_checkpoint_contexts`
    with ``layer``'s hook guard when it holds two-read ``StreamedExperts``, else ``None``."""
    cls = streamed_experts_class()
    for module in layer.modules():
        if isinstance(module, cls) and module.two_reads:
            return functools.partial(two_read_checkpoint_contexts, _TwoReadHookGuard(layer))
    return None


def _build_topk_decisions_class():
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    topk = torch.ops.aten.topk.default

    class TopkDecisions(TorchDispatchMode):
        """Active for one router call. In the forward it keeps every top-k result;
        in a recompute it hands them back in order and never selects. Below
        autograd, so the recompute's top-k node still saves the indices the
        checkpoint expects; every other op runs unchanged."""

        def __init__(self, scope: _TwoReadScope) -> None:
            super().__init__()
            self.scope = scope

        def __torch_dispatch__(self, func, _types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func is not topk:
                return func(*args, **kwargs)
            source = args[0]
            signature = (
                tuple(source.shape), source.dtype, source.device, tuple(args[1:]),
                tuple(sorted(kwargs.items())),
            )
            record = self.scope.record
            if not self.scope.replay:
                out = func(*args, **kwargs)
                # Owned copies: nothing the router's caller does to its outputs
                # can change what the recompute receives.
                record.topk.append((signature, tuple(tensor.clone() for tensor in out)))
                return out
            cursor = record.cursor
            if cursor >= len(record.topk) or record.topk[cursor][0] != signature:
                raise RuntimeError(
                    f"two-read experts: the checkpoint recompute asked a router for top-k "
                    f"#{cursor} {signature[:2]}, which the forward did not take "
                    f"({len(record.topk)} recorded)"
                )
            record.cursor = cursor + 1
            return tuple(tensor.clone() for tensor in record.topk[cursor][1])

    return TopkDecisions


_TOPK_DECISIONS_CLASS = None


def _topk_decisions(scope: _TwoReadScope) -> Any:
    global _TOPK_DECISIONS_CLASS
    if _TOPK_DECISIONS_CLASS is None:
        _TOPK_DECISIONS_CLASS = _build_topk_decisions_class()
    return _TOPK_DECISIONS_CLASS(scope)


def _routing_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
    """Forward of a two-read MoE block's router: under a two-read scope its top-k
    decisions are kept (forward) or replayed (recompute); scoring always runs."""
    forward = type(self).forward
    scope = _active_scope()
    if scope is None:
        return forward(self, *args, **kwargs)
    with _topk_decisions(scope):
        return forward(self, *args, **kwargs)


def _build_streamed_experts_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    def expert_mlp(hidden: Any, gate_up: Any, down: Any, act_fn: Any) -> Any:
        # transformers' eager experts body, line for line.
        gate, up = functional.linear(hidden, gate_up).chunk(2, dim=-1)
        current = act_fn(gate) * up
        return functional.linear(current, down)

    def _autocast_state(device_type: str) -> Dict[str, Any]:
        return {
            "enabled": torch.is_autocast_enabled(device_type),
            "dtype": torch.get_autocast_dtype(device_type),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }

    class _ExpertMLP(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden, owner, expert):  # noqa: D401 - autograd API
            ctx.device_type = hidden.device.type
            ctx.autocast_kwargs = _autocast_state(ctx.device_type)
            gate_up, down = owner.fetch(expert)
            out = expert_mlp(hidden, gate_up, down, owner.act_fn)
            ctx.owner = owner
            ctx.expert = expert
            ctx.save_for_backward(hidden)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            (hidden,) = ctx.saved_tensors
            if not ctx.needs_input_grad[0]:
                return None, None, None
            gate_up, down = ctx.owner.fetch(ctx.expert)
            # Backward usually runs outside the forward's autocast region.
            with torch.enable_grad(), torch.autocast(ctx.device_type, **ctx.autocast_kwargs):
                local = hidden.detach().requires_grad_(True)
                out = expert_mlp(local, gate_up, down, ctx.owner.act_fn)
                (grad_hidden,) = torch.autograd.grad(out, local, grad_out)
            return grad_hidden, None, None

    class _WeightedExpert(torch.autograd.Function):
        """``expert_mlp(hidden) * weights`` as one node. It saves only the expert's
        input and routing weights, so a checkpoint recompute rebuilds what its
        backward needs without reading the expert."""

        @staticmethod
        def forward(ctx, hidden, weights, owner, expert, replay):  # noqa: D401 - autograd API
            ctx.device_type = hidden.device.type
            ctx.autocast_kwargs = _autocast_state(ctx.device_type)
            ctx.owner = owner
            ctx.expert = expert
            ctx.save_for_backward(hidden, weights)
            if replay:
                # Only the cast and index_add_ consume the product, and index_add_
                # saves it for its shape alone: the backward never reads this zero.
                return hidden.new_zeros(()).expand_as(hidden)
            gate_up, down = owner.fetch(expert)
            return expert_mlp(hidden, gate_up, down, owner.act_fn) * weights

        @staticmethod
        def backward(ctx, grad_out):
            hidden, weights = ctx.saved_tensors
            need_hidden, need_weights = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
            if not (need_hidden or need_weights):
                return None, None, None, None, None
            gate_up, down = ctx.owner.fetch(ctx.expert)
            # The graph the eager loop differentiates, under the forward's autocast.
            with torch.enable_grad(), torch.autocast(ctx.device_type, **ctx.autocast_kwargs):
                local_hidden = hidden.detach().requires_grad_(need_hidden)
                local_weights = weights.detach().requires_grad_(need_weights)
                out = expert_mlp(local_hidden, gate_up, down, ctx.owner.act_fn) * local_weights
                wanted = [
                    tensor for tensor in (local_hidden, local_weights) if tensor.requires_grad
                ]
                grads = list(torch.autograd.grad(out, wanted, grad_out))
            grad_hidden = grads.pop(0) if need_hidden else None
            grad_weights = grads.pop(0) if need_weights else None
            return grad_hidden, grad_weights, None, None, None

    class StreamedExperts(nn.Module):
        """Drop-in for a transformers experts module; fetches each expert on use."""

        def __init__(self, reader, fetcher, act_fn, layer_idx, two_reads=False):
            super().__init__()
            self.reader = reader
            self.fetcher = fetcher
            self.act_fn = act_fn
            self.layer_idx = layer_idx
            self.num_experts = reader.count
            self.two_reads = bool(two_reads)

        def fetch(self, expert):
            return self.fetcher.fetch(self.reader, expert)

        def forward(self, hidden_states, top_k_index, top_k_weights):
            final_hidden_states = torch.zeros_like(hidden_states)
            with torch.no_grad():
                expert_mask = functional.one_hot(top_k_index, num_classes=self.num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            # Inside a two-read checkpoint recompute no expert is read.
            scope = _active_scope() if self.two_reads else None
            replay = scope is not None and scope.replay
            for expert_idx in expert_hit:
                expert_idx = expert_idx[0]
                if expert_idx == self.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                if self.two_reads:
                    current = _WeightedExpert.apply(
                        current_state, top_k_weights[token_idx, top_k_pos, None], self,
                        int(expert_idx), replay,
                    )
                else:
                    current = _ExpertMLP.apply(current_state, self, int(expert_idx))
                    current = current * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx, current.to(final_hidden_states.dtype))
            return final_hidden_states

    return StreamedExperts


_STREAMED_EXPERTS_CLASS = None


def streamed_experts_class() -> Any:
    global _STREAMED_EXPERTS_CLASS
    if _STREAMED_EXPERTS_CLASS is None:
        _STREAMED_EXPERTS_CLASS = _build_streamed_experts_class()
    return _STREAMED_EXPERTS_CLASS


def find_experts_module(layer: Any) -> Optional[Tuple[Any, str, Any]]:
    """``(parent, attribute, module)`` of a layer's fused routed-experts module.

    Matched by shape of contract, not by class name: a child named ``experts``
    holding 3-D ``gate_up_proj`` and ``down_proj`` parameters, the layout every
    transformers 5.x MoE uses. ``None`` for a dense layer.
    """
    for name, module in layer.named_modules():
        parent_name, _, leaf = name.rpartition(".")
        if leaf != "experts":
            continue
        gate_up = getattr(module, "gate_up_proj", None)
        down = getattr(module, "down_proj", None)
        if getattr(gate_up, "dim", lambda: 0)() == 3 and getattr(down, "dim", lambda: 0)() == 3:
            parent = layer.get_submodule(parent_name) if parent_name else layer
            return parent, leaf, module
    return None


#: Decoder layers whose MoE output reaches only views and additions before the residual,
#: none of which saves it for its values (transformers 5.19, read per class), and whose real
#: layer has a gradient comparison with the resident model. A two-read recompute returns
#: zeros there; any op that saved them would corrupt the gradients.
TWO_READ_DECODERS = frozenset({"DeepseekV3DecoderLayer"})


def install_expert_streaming(
    layers: Sequence[Any],
    readers: Mapping[int, ExpertShardReader],
    fetcher: ExpertFetcher,
    *,
    two_reads: bool = False,
) -> int:
    """Replace each MoE layer's experts module with a ``StreamedExperts``.

    Runs before the decoder-layer name maps are taken, so the fused expert
    parameters leave the layer and the layer pool never allocates them.
    ``two_reads`` also routes the MoE block's router (``.gate``) through
    :func:`_routing_forward`, whose decisions a two-read recompute replays.

    Two-read contract. Each checkpointed layer call keeps its own record, so gradient
    accumulation, several forwards before one backward, ``retain_graph`` backwards, no-grad
    forwards and exceptions are handled as on the default path. In the recompute the
    routed-experts output is zero (the MoE output is shared expert + 0).

    Forward hooks: the checkpoint context from :func:`checkpoint_context_fn` refuses the
    layer call with a ``RuntimeError`` while the decoder layer, its MoE block or its routed
    experts carries a forward hook, or while a global forward hook is registered. It checks
    when the layer's forward starts and again when its recompute starts, so a hook
    registered after ``build_streamed_model`` (or between forward and backward) is refused
    before its gradient is used. A refused call has not stepped the optimizer, so nothing
    is updated from wrong gradients. Hooks on the router, attention, norms or the
    ``StreamedDecoderLayer`` wrapper, and forward pre-hooks, see the forward's values and
    are allowed. No hook is accepted on the strength of what it does, and there is no
    fallback to three reads.

    Patched ``forward``: not checked. A ``forward`` replaced on the decoder layer or on
    any module inside it that uses the MoE output's values in the backward receives
    wrong gradients without an error; do not patch a two-read layer.
    Returns the number of layers converted.
    """
    import torch.nn as nn

    cls = streamed_experts_class()
    converted = 0
    for idx, layer in enumerate(layers):
        reader = readers.get(idx)
        found = find_experts_module(layer)
        if reader is None:
            if found is not None:
                raise ValueError(
                    f"decoder layer {idx} has a routed-experts module but the cache has "
                    f"no experts file for it — reshard the checkpoint"
                )
            continue
        if found is None:
            raise ValueError(
                f"the cache has an experts file for decoder layer {idx}, which has no "
                f"routed-experts module — reshard the checkpoint"
            )
        parent, attr, module = found
        expected_gate_up = tuple(int(dim) for dim in module.gate_up_proj.shape)
        expected_down = tuple(int(dim) for dim in module.down_proj.shape)
        stored_gate_up = (reader.count, *reader.shapes[GATE_UP])
        stored_down = (reader.count, *reader.shapes[DOWN])
        if (expected_gate_up, expected_down) != (stored_gate_up, stored_down):
            raise ValueError(
                f"decoder layer {idx}: the model's experts are {expected_gate_up} / "
                f"{expected_down} but the experts file holds {reader.count} x "
                f"{reader.shapes[GATE_UP]} / {reader.shapes[DOWN]} — reshard the checkpoint"
            )
        names = [name for name, _ in module.named_parameters()]
        if any(name.startswith("lora_") or ".lora_" in name for name in names):
            raise ValueError(
                f"decoder layer {idx}: adapters on the routed experts cannot be trained "
                f"while the experts are streamed one at a time; target attention or the "
                f"shared expert instead"
            )
        router = getattr(parent, "gate", None) if two_reads else None
        if two_reads and type(layer).__name__ not in TWO_READ_DECODERS:
            raise ValueError(
                f"decoder layer {idx} is a {type(layer).__name__}: two reads are admitted "
                f"only where the MoE output is consumed by views and additions "
                f"({', '.join(sorted(TWO_READ_DECODERS))})"
            )
        if two_reads and not isinstance(router, nn.Module):
            raise ValueError(
                f"decoder layer {idx}: two reads replay the router's top-k decisions, but "
                f"the MoE block has no router module at `.gate`"
            )
        setattr(parent, attr, cls(reader, fetcher, module.act_fn, idx, two_reads=two_reads))
        if router is not None:
            router.forward = types.MethodType(_routing_forward, router)
        converted += 1
    return converted
