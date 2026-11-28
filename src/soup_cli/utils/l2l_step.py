"""One layer-major (L2L) step over the streamed model.

L2L (Pudipeddi et al., arXiv 2002.05645): load a decoder layer once per step, run all
k micro-batches through it, park each micro-batch's boundary activation in host RAM,
and walk the backward the same way in reverse.

* ``StreamedDecoderLayer._body`` calls ``pool.wait(idx)`` and then
  ``prefetcher.advance(idx)``. A second call on the same index leaves the
  direction alone and finds the next layer already owned, so k calls on a loaded
  layer load nothing. The load sequence (forward 0..L-1, then L-1..0) is the
  streamed step's, and ``AsyncDiskSource`` infers its direction from that order.
* The decoder container's forward pre-hook (``runtime.hook``, the shipped ``_prime``)
  is called directly, because the step bypasses ``LlamaModel.forward``.
* All k embedding lookups run before layer 0's first ``advance`` queues the head
  into the shared large slot; ``LargeLayerBufferPool.load_async`` orders that
  refill after them on the compute stream.
* Trainable embedding adapters get a final one-micro-batch recompute and VJP after layer 0,
  through the same streamed embedding wrapper. Its RNG is restored for noise hooks; the
  large slot is not refilled between that recompute and its immediate backward.
* The backward turns the wrapper's checkpoint off for its per-(layer,
  micro-batch) forward+backward, which would otherwise pay a third forward.
* The top-level forward is re-implemented for Llama/Mistral and DeepSeek-V3:
  ``embed_tokens`` -> layers with the kwargs HF passed to layer 0 -> ``norm`` ->
  ``lm_head(h[:, slice(0, None), :])`` -> the model's own ``loss_function``
  with ``num_items_in_batch``. DeepSeek-V3's top-level forward is that same sequence (it
  adds no auxiliary routing loss). ``capture_layer_call`` records the layer kwargs from a
  real forward instead of rebuilding the rotary embeddings and the mask.

* A preference loss with a reference model passes one or more ``ReferencePass``es: after each
  layer's policy micro-batches the same loaded layer runs the k micro-batches again, without
  grad and without writing to the activation store, then the pass's head turns each last
  hidden state into log-probabilities. A pass runs under a named adapter (DPO's frozen
  ``ref``), with every adapter off (KTO's reference) or under the policy as it is (KTO's KL
  estimate), and walks either the policy's ids or ids of its own (KTO's KL completions have
  their own width). It adds no layer load: the extra calls land on a layer the pool already
  owns.
* With ``lora.dropout`` > 0 the step replays the random stream of plain gradient
  accumulation: every decoder and head call runs at the generator state accumulation
  reaches for that micro-batch, including head draws between consecutive micro-batches.
  Recompute restores the same states without advancing the outer stream. See ``l2l_rng.py``.

* A layer with streamed routed experts (a per-expert cache, ``utils/expert_stream.py``) reads
  every hit expert through the one-slot arena in the no-grad forward and its own backward.
  Two-read experts replay routing without a read in the grad recompute; ordinary experts
  read again there. ``LoadCounters`` counts those reads. The
  router draws no randomness, so the replay above needs nothing more for it, and
  ``RoutingGuard`` checks that the recompute routes like the forward.

No optimizer step. torch is imported inside functions.
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from types import MethodType
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

#: Architectures whose top-level forward this step re-implements. ``training.l2l`` admits them
#: at launch under the rules in ``trainer.l2l_trainer.refuse_unsupported_l2l``.
SUPPORTED_ARCHS = ("llama", "mistral", "deepseek_v3", "olmoe")


@dataclass(frozen=True)
class LayerCall:
    """What HF passes to every decoder layer after ``hidden_states``."""

    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    #: Per layer, the dropouts one call draws (``l2l_rng.Draw``); empty without dropout.
    draws: Tuple[Tuple[Any, ...], ...] = ()
    #: Dropouts the native output head draws after the decoder layers of each micro-batch.
    head_draws: Tuple[Any, ...] = ()


#: ``LoadCounters`` field -> the ``runtime.stats()`` key whose growth it counts.
_STAT_KEYS = {
    "layer_loads": "layer_loads",
    "large_loads": "large_loads",
    "restores": "layer_restores",
    "expert_fetches": "expert_fetches",
    "expert_bytes_read": "expert_bytes_read",
}


@dataclass
class LoadCounters:
    """Loads and weight restores over a span, read from the runtime's own counters.

    ``layer_loads`` and ``large_loads`` are the pools' ``loads``; ``restores`` is the
    NF4 rebuilds of ``Params4bit`` views, or ``None`` when the runtime does not count them
    (the main runtime rebuilds the quantization state on each layer load but keeps no
    counter, so ``None`` there means "not counted", never "zero"). ``expert_fetches``
    is the reads of one routed expert into the expert arena, whether the forward, the
    recompute or the expert's own backward asked for it, and ``expert_bytes_read`` the stored
    bytes those reads took; both are 0 without a per-expert cache. Nothing is patched: the
    runtime already counts, this reads the difference.
    """

    layer_loads: int = 0
    large_loads: int = 0
    restores: Optional[int] = None
    expert_fetches: int = 0
    expert_bytes_read: int = 0

    def add(self, other: "LoadCounters") -> None:
        for name in _STAT_KEYS:
            mine, theirs = getattr(self, name), getattr(other, name)
            if theirs is None:
                continue
            setattr(self, name, theirs if mine is None else mine + theirs)


def _counter_values(runtime: Any) -> Dict[str, Optional[int]]:
    stats = runtime.stats()
    # A runtime without ``layer_restores`` (the main runtime) does not count weight
    # restores: report that as ``None``, not as zero.
    values: Dict[str, Optional[int]] = {}
    for name, key in _STAT_KEYS.items():
        values[name] = int(stats[key]) if key in stats else None
    return values


@contextmanager
def count_loads(runtime: Any) -> Iterator[LoadCounters]:
    """Yield a ``LoadCounters`` that holds the span's deltas once the block exits."""
    counters = LoadCounters()
    before = _counter_values(runtime)
    try:
        yield counters
    finally:
        for name, value in _counter_values(runtime).items():
            if value is not None and before[name] is not None:
                setattr(counters, name, value - before[name])


def model_parts(model: Any) -> Tuple[Any, Any]:
    """(the CausalLM under any PEFT wrapper, the container that owns ``.layers``)."""
    from soup_cli.utils.layer_stream_runtime import decoder_owner

    owner = decoder_owner(model)
    causal = model.get_base_model() if hasattr(model, "get_base_model") else model
    arch = str(getattr(getattr(causal, "config", None), "model_type", ""))
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(
            f"the L2L step re-implements the top-level forward of {SUPPORTED_ARCHS} only; "
            f"got {arch!r}"
        )
    if arch == "olmoe" and getattr(causal.config, "output_router_logits", False):
        raise ValueError("training.l2l on OLMoE requires output_router_logits=false")
    for name in ("lm_head", "loss_function"):
        if getattr(causal, name, None) is None:
            raise ValueError(f"the causal LM has no {name!r}")
    return causal, owner


def label_tokens(micro_batches: Sequence[Any]) -> int:
    """HF's ``num_items_in_batch`` when labels == input_ids: seq - 1 per row."""
    return sum(int(ids.shape[0]) * (int(ids.shape[1]) - 1) for ids in micro_batches)


#: ``(causal, owner, hidden, mb) -> loss``. ``hidden`` is micro-batch ``mb``'s output of the
#: last decoder layer, a leaf that requires grad. The head runs ``norm``, ``lm_head`` and the
#: loss, and returns the scalar the step backpropagates, already scaled for the window.
LossHead = Callable[[Any, Any, Any, int], Any]


def sft_head(labels: Sequence[Any], n_items: Any) -> LossHead:
    """The SFT head: ``norm``, ``lm_head``, the model's ``loss_function`` over ``n_items``."""

    def head(causal: Any, owner: Any, hidden: Any, mb: int) -> Any:
        normed = owner.norm(hidden)
        logits = causal.lm_head(normed[:, slice(0, None), :])
        return causal.loss_function(
            logits=logits,
            labels=labels[mb],
            vocab_size=int(causal.config.vocab_size),
            num_items_in_batch=n_items,
        )

    return head


@dataclass
class ReferencePass:
    """A gradient-free pass over the same layers, after each layer's policy micro-batches.

    ``head`` has the shape of a :data:`LossHead`; the step calls it under ``no_grad``, inside
    the pass's adapter state, on the last hidden state of micro-batch ``mb`` and stores the
    result in ``outputs[mb]``. The policy head reads ``outputs`` after the step has filled it.

    The adapter state is one of: ``adapter`` (a PEFT adapter name, TRL's ``ref`` for DPO)
    active for the pass; ``adapters_off`` (every adapter switched off, TRL's
    ``null_ref_context`` for KTO); neither, the policy as it stands.

    ``micro_batches`` are the ids the pass walks; without them it walks the policy's. Ids of
    their own need one :class:`LayerCall` per micro-batch in ``calls``, captured at that
    micro-batch's shape.
    """

    head: LossHead
    adapter: Optional[str] = None
    adapters_off: bool = False
    micro_batches: Optional[Sequence[Any]] = None
    calls: Optional[Sequence[LayerCall]] = None
    outputs: Dict[int, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.adapter is not None and self.adapters_off:
            raise ValueError(
                "a reference pass names an adapter or switches the adapters off, not both: "
                f"adapter={self.adapter!r}, adapters_off=True"
            )


class _CapturedError(Exception):
    """Raised from layer 0's pre-hook to end a capture forward before any layer runs."""


def capture_layer_call(model: Any, input_ids: Any, *, draws: bool = True) -> LayerCall:
    """Run one forward without grad and record what layer 0 receives.

    The random state is restored afterwards, so the capture draws nothing from it.

    With ``draws`` (the default) the forward runs through every layer and the native output
    head and records which dropouts each draws, for the replay in :meth:`L2LStep.run`; on a
    streamed model that walk loads every layer once. With ``draws=False`` the forward stops at
    layer 0's pre-hook, which already holds the arguments, so no layer loads or runs. Use it
    when the model draws no dropout; the returned call has no draws.
    """
    import torch

    from soup_cli.utils.l2l_rng import record_draws

    causal, owner = model_parts(model)
    seen: Dict[str, Any] = {}

    def hook(_module: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        if "call" not in seen:
            seen["call"] = (tuple(args[1:]), dict(kwargs))
            if not draws:
                raise _CapturedError

    handle = owner.layers[0].register_forward_pre_hook(hook, with_kwargs=True)
    # The capture forward runs in training mode, so NEFTune would draw noise from it.
    cuda = [input_ids.device] if input_ids.device.type == "cuda" else []
    plan: List[List[Any]] = []
    try:
        with torch.no_grad(), torch.random.fork_rng(devices=cuda):
            if draws:
                with record_draws([*owner.layers, causal.lm_head]) as plan:
                    model(input_ids=input_ids)
            else:
                try:
                    model(input_ids=input_ids)
                except _CapturedError:
                    pass
    finally:
        handle.remove()
    if "call" not in seen:
        raise RuntimeError("decoder layer 0 was never called by the forward")
    args, kwargs = seen["call"]
    kwargs.pop("num_items_in_batch", None)
    return LayerCall(
        args=args,
        kwargs=kwargs,
        draws=tuple(tuple(layer) for layer in plan[:-1]),
        head_draws=tuple(plan[-1]) if plan else (),
    )


class RoutingGuard:
    """Checks that the recompute of each (layer, micro-batch) routes tokens like its forward.

    The no-grad forward and the grad recompute of a layer run the router on the same stored
    activation, so they choose the same experts as long as the router's weights, its
    correction bias and its arithmetic stay put in between. Nothing else verifies that, and a
    recompute that chose other experts would backpropagate through experts the forward never
    ran. Forward pre-hooks on the layers' streamed experts modules, alive for one
    :meth:`L2LStep.run`, record each call's ``top_k_index`` on the host while a pair is
    forwarded and compare it with ``torch.equal`` while the pair is recomputed, before its
    experts are read and before its backward. A pair's record is freed as soon as it has
    been compared, and ``close`` drops what a failed step left. A layer without streamed
    experts (a dense model, a fused cache) gets no hook, so nothing is recorded for it.
    """

    def __init__(
        self, layers: Sequence[Any], *, routing_budget: Optional[Tuple[int, int]] = None
    ) -> None:
        from soup_cli.utils.expert_stream import streamed_experts_class

        experts = streamed_experts_class()
        if routing_budget is not None and (
            len(routing_budget) != 2
            or any(type(value) is not int or value < 0 for value in routing_budget)
        ):
            raise ValueError("routing budget needs nonnegative record and comparison bytes")
        self._routing_budget = routing_budget
        self._record_bytes = 0
        self._records: Dict[Tuple[int, int], Any] = {}
        #: (layer, micro-batch, recording) inside ``record`` or ``check``; ``None`` otherwise.
        self._visit: Optional[Tuple[int, int, bool]] = None
        self._handles: List[Any] = []
        for layer in layers:
            for module in layer.modules():
                if isinstance(module, experts):
                    self._handles.append(
                        module.register_forward_pre_hook(self._watch, with_kwargs=True)
                    )

    def _watch(self, _module: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        import torch

        if self._visit is None:
            return
        layer, mb, recording = self._visit
        routed = (kwargs["top_k_index"] if "top_k_index" in kwargs else args[1]).detach()
        pair = (layer, mb)
        nbytes = routed.numel() * routed.element_size()
        if self._routing_budget is not None:
            records, comparison = self._routing_budget
            if nbytes > comparison or (recording and self._record_bytes + nbytes > records):
                raise ValueError("routing tensors exceed the resolved routing budget")
        if recording:
            if pair in self._records:
                raise RuntimeError(
                    f"routing guard: layer {layer} micro-batch {mb} called its streamed "
                    "experts more than once in the forward"
                )
            self._records[pair] = routed.to("cpu", copy=True)
            self._record_bytes += nbytes
            return
        before = self._records.pop(pair, None)
        now = None
        if before is not None:
            self._record_bytes -= before.numel() * before.element_size()
        try:
            if before is None:
                raise RuntimeError(
                    f"routing guard: the recompute of layer {layer} micro-batch {mb} called its "
                    "streamed experts more often than the forward did"
                )
            now = routed.to("cpu")
            if not torch.equal(before, now):
                raise RuntimeError(
                    f"routing guard: layer {layer} micro-batch {mb} was routed differently in the "
                    f"recompute than in the forward ({_routing_difference(before, now)}), so its "
                    "gradients would not be those of the forward. The router's weights, its "
                    "correction bias or its input changed between the two."
                )
        except BaseException as error:
            import traceback

            # A failing comparison helper may also retain its arguments in its traceback.
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            # Retaining the exception must not keep owned host tensors past lease release.
            del before, now, routed

    def _discard_record(self, pair: Tuple[int, int]) -> None:
        record = self._records.pop(pair, None)
        if record is not None:
            self._record_bytes -= record.numel() * record.element_size()

    @contextmanager
    def record(self, layer: int, mb: int) -> Iterator[None]:
        """Record the routing of the streamed experts called inside the block (the forward)."""
        if not self._handles:
            yield
            return
        self._visit = (layer, mb, True)
        try:
            yield
        except BaseException:
            self._discard_record((layer, mb))
            raise
        finally:
            self._visit = None

    @contextmanager
    def check(self, layer: int, mb: int) -> Iterator[None]:
        """Compare the routing of the streamed experts called inside the block with the
        forward's (the recompute).

        Raises ``RuntimeError`` naming the layer and micro-batch on a difference. The pair's
        record is freed on success and on exceptions, without hiding the original exception.
        """
        if not self._handles:
            yield
            return
        self._visit = (layer, mb, False)
        try:
            yield
            if (layer, mb) in self._records:
                raise RuntimeError(
                    f"routing guard: the forward of layer {layer} micro-batch {mb} called its "
                    "streamed experts more often than the recompute did"
                )
        finally:
            self._discard_record((layer, mb))
            self._visit = None

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._records.clear()
        self._record_bytes = 0
        self._visit = None

    def __enter__(self) -> "RoutingGuard":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def _routing_difference(before: Any, now: Any) -> str:
    import torch

    if before.shape != now.shape:
        return f"routing went from shape {tuple(before.shape)} to {tuple(now.shape)}"
    # Failure diagnostics must not allocate a boolean matrix outside the route budget.
    changed = sum(not torch.equal(left, right) for left, right in zip(before, now))
    return f"{changed} of {before.shape[0]} tokens chose other experts"


def _streamed_moe_experts(layer: Any) -> Any:
    """Return the streamed routed-expert module of a DeepSeek-V3 MoE layer, if any."""
    from soup_cli.utils.expert_stream import streamed_experts_class

    inner = getattr(layer, "inner", layer)
    mlp = getattr(inner, "mlp", None)
    experts = getattr(mlp, "experts", None)
    if isinstance(experts, streamed_experts_class()):
        return experts
    return None


def mbr_deferred_forward_hook(hook: Callable[..., Any]) -> Callable[..., Any]:
    """Opt a pure decoder forward hook into MBR's deferred, grouped execution.

    Such a hook must be deterministic and independent across micro-batches. It must
    not inspect mutable streamed weights or change module state, RNG, or hook sets.
    It may observe or transform its output, preserving tensor shape and dtype.
    """
    if not callable(hook):
        raise TypeError("an MBR deferred forward hook must be callable")
    setattr(hook, "_soup_mbr_deferred_forward_hook", True)
    return hook


def _validate_route_preserving_mbr(layers: Sequence[Any], causal: Any) -> None:
    """Refuse unsupported models before the first streamed layer is loaded."""
    import torch.nn.modules.module as module_hooks

    if str(getattr(getattr(causal, "config", None), "model_type", "")) != "deepseek_v3":
        raise ValueError("route-preserving MBR currently supports DeepSeek-V3 decoder layers only")
    if any(
        getattr(module_hooks, name, None)
        for name in (
            "_global_forward_hooks",
            "_global_forward_pre_hooks",
            "_global_backward_hooks",
            "_global_backward_pre_hooks",
        )
    ):
        raise ValueError("route-preserving MBR does not support global module hooks")
    moe_layers = 0
    for index, layer in enumerate(layers):
        inner = getattr(layer, "inner", layer)
        mlp = getattr(inner, "mlp", None)
        if not (hasattr(mlp, "gate") and hasattr(mlp, "experts")):
            continue
        moe_layers += 1
        experts = _streamed_moe_experts(layer)
        if type(inner).__name__ != "DeepseekV3DecoderLayer" or experts is None:
            raise ValueError(
                f"route-preserving MBR layer {index} requires a streamed "
                "DeepseekV3DecoderLayer with per-expert routed weights"
            )
        # The layer's regular call produces a temporary placeholder; the real output
        # is assembled after the grouped experts finish. MoE module __call__ is bypassed.
        # Failing closed avoids silently observing or modifying the wrong tensor.
        for name, module in (("streamed decoder", layer), ("decoder", inner), ("MoE", mlp)):
            hook_types = ("_backward_hooks", "_backward_pre_hooks")
            if name == "MoE":
                hook_types += ("_forward_pre_hooks", "_forward_hooks")
            if any(getattr(module, kind, None) for kind in hook_types):
                raise ValueError(
                    f"route-preserving MBR layer {index} does not support {name} "
                    "output/backward hooks or MoE pre-hooks"
                )
            if name != "MoE":
                hooks = getattr(module, "_forward_hooks", {})
                always = getattr(module, "_forward_hooks_always_called", {})
                if any(
                    not getattr(hook, "_soup_mbr_deferred_forward_hook", False)
                    or hook_id in always
                    for hook_id, hook in hooks.items()
                ):
                    raise ValueError(
                        f"route-preserving MBR layer {index} requires explicitly marked "
                        "deferred decoder forward hooks without always_call"
                    )
        if not experts.two_reads:
            raise ValueError(f"route-preserving MBR layer {index} requires two_read_experts=True")
        if not experts.grouped_expert_routes:
            raise ValueError(
                f"route-preserving MBR layer {index} requires grouped_expert_routes=True"
            )
        if experts.pool is None:
            raise ValueError(f"route-preserving MBR layer {index} requires an expert buffer pool")
        if any(
            getattr(experts, name, None)
            for name in (
                "_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks"
            )
        ):
            raise ValueError(
                f"route-preserving MBR layer {index} does not support routed-expert module hooks"
            )
        if getattr(mlp, "shared_experts_gate", None) is not None:
            raise ValueError(
                f"route-preserving MBR layer {index} has an unsupported shared-expert gate"
            )
    if moe_layers == 0:
        raise ValueError("route-preserving MBR found no streamed routed-expert layers")


def _route_preserving_deepseek_forward(
    module: Any,
    capture: List[Dict[str, Any]],
    hidden_states: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """DeepSeek-V3 decoder forward with the routed FFN deferred to its window."""
    import torch

    signature = inspect.signature(type(module).forward)
    bound = signature.bind(module, hidden_states, *args, **kwargs)
    bound.apply_defaults()
    values = bound.arguments
    decoder_kwargs: Dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            decoder_kwargs.update(values.get(name) or {})
    past_key_values = values.get("past_key_values", values.get("past_key_value"))
    output_attentions = bool(
        values.get("output_attentions", decoder_kwargs.get("output_attentions", False))
    )
    use_cache = bool(values.get("use_cache", False))
    if use_cache or past_key_values is not None:
        raise ValueError("route-preserving MBR requires use_cache=False and no past_key_values")
    if output_attentions:
        raise ValueError("route-preserving MBR currently requires output_attentions=False")

    attention_signature = inspect.signature(type(module.self_attn).forward)
    attention_parameters = attention_signature.parameters
    accepts_attention_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in attention_parameters.values()
    )
    layer_kwargs: Dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if name in {"self", "hidden_states"}:
            continue
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        elif parameter.kind is not inspect.Parameter.VAR_POSITIONAL:
            layer_kwargs[name] = values[name]
    layer_kwargs.update(decoder_kwargs)
    attention_kwargs = {
        name: value
        for name, value in layer_kwargs.items()
        if name in attention_parameters or accepts_attention_kwargs
    }

    residual = hidden_states
    attention_input = module.input_layernorm(hidden_states)
    attention_result = module.self_attn(
        hidden_states=attention_input,
        **attention_kwargs,
    )
    if not isinstance(attention_result, tuple) or len(attention_result) < 2:
        raise RuntimeError(
            "route-preserving MBR expected the DeepSeek-V3 attention result to contain "
            "(hidden, attention_weights)"
        )
    attention_output = attention_result[0]
    residual = residual + attention_output

    moe_input = module.post_attention_layernorm(residual)
    moe = module.mlp
    if getattr(moe, "shared_experts_gate", None) is not None:
        raise ValueError("route-preserving MBR does not yet support a shared-expert gate")
    gate_result = moe.gate(moe_input)
    if not isinstance(gate_result, tuple):
        raise RuntimeError("DeepSeek-V3 router returned an unsupported result")
    if len(gate_result) == 3:
        _, topk_weights, topk_index = gate_result
    elif len(gate_result) == 2:
        topk_index, topk_weights = gate_result
    else:
        raise RuntimeError(
            f"DeepSeek-V3 router returned {len(gate_result)} values; expected 2 or 3"
        )
    if topk_index.ndim != 2 or topk_weights.shape != topk_index.shape:
        raise RuntimeError("DeepSeek-V3 router returned unsupported route tensor shapes")
    flat_input = moe_input.view(-1, moe_input.shape[-1])
    if flat_input.shape[0] != topk_index.shape[0]:
        raise RuntimeError("DeepSeek-V3 router token count differs from its MoE input")
    shared_output = (
        moe.shared_experts(moe_input) if getattr(moe, "shared_experts", None) is not None else None
    )
    capture.append(
        {
            "residual": residual,
            "expert_input": flat_input,
            "route_ids": topk_index,
            "route_weights": topk_weights,
            "shared_output": shared_output,
            "hook_args": (hidden_states, *args),
            "hook_kwargs": dict(kwargs),
        }
    )

    routed_placeholder = torch.zeros_like(moe_input)
    mlp_output = (
        routed_placeholder + shared_output if shared_output is not None else routed_placeholder
    )
    layer_output = residual + mlp_output
    return layer_output


@contextmanager
def _patched_route_preserving_forward(layer: Any, capture: List[Dict[str, Any]]) -> Iterator[None]:
    """Temporarily split a pinned DeepSeek-V3 layer around its routed expert call."""
    inner = getattr(layer, "inner", layer)
    had_instance_forward = "forward" in inner.__dict__
    old_instance_forward = inner.__dict__.get("forward")

    def forward(module: Any, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
        return _route_preserving_deepseek_forward(module, capture, hidden_states, *args, **kwargs)

    inner.forward = MethodType(forward, inner)
    try:
        yield
    finally:
        if had_instance_forward:
            inner.forward = old_instance_forward
        else:
            delattr(inner, "forward")


@contextmanager
def _defer_decoder_forward_hooks(
    layer: Any, *, split_backward: bool
) -> Iterator[Tuple[Any, Any, List[Any]]]:
    """Hide marked hooks during the placeholder call and capture real wrapper inputs."""
    inner = getattr(layer, "inner", layer)
    wrapper_hooks = layer._forward_hooks
    inner_hooks = inner._forward_hooks if inner is not layer else None
    wrapper_inputs: List[Any] = []
    for module, hooks in ((layer, wrapper_hooks), (inner, inner_hooks)):
        if hooks is None:
            continue
        always = getattr(module, "_forward_hooks_always_called", {})
        if any(
            not getattr(hook, "_soup_mbr_deferred_forward_hook", False) or hook_id in always
            for hook_id, hook in hooks.items()
        ):
            raise ValueError("route-preserving MBR requires explicitly marked deferred hooks")
    if split_backward and (wrapper_hooks or inner_hooks):
        raise RuntimeError(
            "mbr_split_backward does not support deferred decoder forward hooks: "
            "a hook may read tensors that phase one would sum group-wide"
        )

    def record_inputs(_module: Any, args: Any, kwargs: Any) -> None:
        wrapper_inputs.append((args, dict(kwargs)))

    handle = layer.register_forward_pre_hook(record_inputs, with_kwargs=True)
    layer._forward_hooks = type(wrapper_hooks)()
    if inner_hooks is not None:
        inner._forward_hooks = type(inner_hooks)()
    try:
        yield wrapper_hooks, inner_hooks, wrapper_inputs
    finally:
        layer._forward_hooks = wrapper_hooks
        if inner_hooks is not None:
            inner._forward_hooks = inner_hooks
        handle.remove()


def _apply_deferred_forward_hooks(
    module: Any, hooks: Any, args: Any, kwargs: Any, output: Any,
) -> Any:
    """Apply explicitly opted-in pure hooks to the real decoder output."""
    for hook_id, hook in tuple(hooks.items()):
        if hook_id in module._forward_hooks_with_kwargs:
            changed = hook(module, args, kwargs, output)
        else:
            changed = hook(module, args, output)
        if changed is not None:
            if (
                not hasattr(changed, "shape")
                or changed.shape != output.shape
                or changed.dtype != output.dtype
                or changed.device != output.device
            ):
                raise RuntimeError(
                    "MBR deferred forward hook must preserve output tensor shape, dtype and device"
                )
            output = changed
    return output


def as_passes(
    reference: Union[ReferencePass, Sequence[ReferencePass], None],
) -> List[ReferencePass]:
    """The extra passes of a window as a list: none, one pass, or the given sequence."""
    if reference is None:
        return []
    if isinstance(reference, ReferencePass):
        return [reference]
    return list(reference)


def _pass_micro_batches(spec: ReferencePass, policy_ids: Sequence[Any], k: int) -> Sequence[Any]:
    """The ids a pass walks: its own, checked against ``k``, or the policy's."""
    if spec.micro_batches is None:
        return policy_ids
    if len(spec.micro_batches) != k:
        raise ValueError(
            f"a reference pass has {len(spec.micro_batches)} micro-batches for a window of {k}"
        )
    return spec.micro_batches


def _pass_calls(spec: ReferencePass, policy_calls: Sequence[LayerCall], k: int) -> Sequence[Any]:
    """The layer call of each micro-batch a pass walks; ids of its own bring their own."""
    if spec.micro_batches is None:
        return policy_calls
    if spec.calls is None or len(spec.calls) != k:
        raise ValueError(
            "a reference pass over ids of its own needs one layer call per micro-batch "
            "(captured at that micro-batch's shape); got "
            f"{None if spec.calls is None else len(spec.calls)} layer calls for {k}"
        )
    return spec.calls


def _cut_group_graph(
    indices: List[int],
    rows: List[Dict[str, Any]],
    joined_hidden: Any,
    joined_weights: Any,
    sizes: List[int],
) -> Dict[str, Any]:
    """Fresh leaves for everything the grouped expert block and the residual path consume."""

    def leaf(tensor: Any) -> Any:
        if tensor is None:
            return None
        return tensor.detach().requires_grad_(tensor.requires_grad)

    return {
        "indices": list(indices),
        "sizes": list(sizes),
        "joined_hidden": leaf(joined_hidden),
        "joined_weights": leaf(joined_weights),
        "residuals": [leaf(row["residual"]) for row in rows],
        "shared": [leaf(row["shared_output"]) for row in rows],
        "pre_hidden": [row["expert_input"] for row in rows],
        "pre_weights": [row["route_weights"] for row in rows],
        "pre_residuals": [row["residual"] for row in rows],
        "pre_shared": [row["shared_output"] for row in rows],
    }


def _split_group_backward(
    cut: Dict[str, Any],
    outputs: Sequence[Any],
    grads: Sequence[Any],
    backward_context: Callable[[int], Any],
) -> None:
    """Group backward in two phases, so LoRA gradients add up in the order of group size 1.

    Phase one runs the grouped expert block and the residual path once (one read per expert).
    Phase two runs what is left of each micro-batch (attention, router, shared experts) by
    itself, in micro-batch order, exactly like a group of one. One engine call over the whole
    group would sum the micro-batch gradients of a LoRA weight in an order of its own.
    """
    import torch

    group = cut["indices"]
    with backward_context(group[0]):
        torch.autograd.backward(
            [outputs[mb] for mb in group], grad_tensors=[grads[mb] for mb in group]
        )
    hidden_grads = _split_grad(cut["joined_hidden"], cut["sizes"])
    weight_grads = _split_grad(cut["joined_weights"], cut["sizes"])
    for pos, mb in enumerate(group):
        pairs = [
            (cut["pre_hidden"][pos], hidden_grads[pos]),
            (cut["pre_weights"][pos], weight_grads[pos]),
            (cut["pre_residuals"][pos], _leaf_grad(cut["residuals"][pos])),
            (cut["pre_shared"][pos], _leaf_grad(cut["shared"][pos])),
        ]
        pairs = [(t, g) for t, g in pairs if t is not None and g is not None and t.requires_grad]
        if pairs:
            with backward_context(mb):
                torch.autograd.backward(
                    [t for t, _ in pairs], grad_tensors=[g for _, g in pairs]
                )
    for leaf in (cut["joined_hidden"], cut["joined_weights"], *cut["residuals"], *cut["shared"]):
        if leaf is not None:
            leaf.grad = None


def _leaf_grad(leaf: Any) -> Any:
    return None if leaf is None else leaf.grad


def _split_grad(leaf: Any, sizes: List[int]) -> List[Any]:
    if leaf is None or leaf.grad is None:
        return [None] * len(sizes)
    return list(leaf.grad.split(sizes, dim=0))


class L2LStep:
    """One layer-major step: forward, head, backward. No optimizer step."""

    def __init__(
        self,
        model: Any,
        runtime: Any,
        call: LayerCall,
    ) -> None:
        self.causal, self.owner = model_parts(model)
        self.model = model
        self.layers = list(self.owner.layers)
        self.call = call
        hook = getattr(runtime, "hook", None)
        prime = self.owner._forward_pre_hooks.get(getattr(hook, "id", None))
        if prime is None:
            raise RuntimeError(
                "the streaming prime hook is not installed on the decoder container; "
                "the L2L step cannot load the embedding and prime the prefetcher"
            )
        self._prime = prime
        self.runtime = runtime
        #: Loads and restores of the last ``run``; ``None`` before the first.
        self.counters: Optional[LoadCounters] = None
        #: Per-layer phase counters from the opt-in route-preserving MBR path.
        self.route_preserving_mbr_stats: Optional[Dict[str, Any]] = None
        #: Optional expert read counters split by layer and step phase.
        self.expert_phase_stats: Optional[Dict[str, Any]] = None

    def _layer(self, module: Any, hidden: Any, state: Any, call: LayerCall) -> Any:
        # ``state`` is where accumulation's stream stands before this (micro-batch, layer)
        # pair, so the forward and the recompute in the backward draw the same mask in any
        # visiting order. ``None`` without dropout.
        from soup_cli.utils import l2l_rng

        with l2l_rng.state_at(state) if state is not None else nullcontext():
            out = module(hidden, *call.args, **call.kwargs)
        return out[0] if isinstance(out, tuple) else out

    def _route_preserving_layer_window(
        self,
        module: Any,
        hidden: Sequence[Any],
        states: Optional[Sequence[Any]],
        calls: Sequence[LayerCall],
        group_size: int,
        deferred_hooks: Tuple[Any, Any, List[Any]],
        *,
        split_records: Optional[List[Dict[str, Any]]] = None,
        reverse_graph_order: bool = False,
        preserve_expert_shapes: bool = False,
    ) -> Tuple[List[Any], List[Tuple[Any, Any]], List[List[int]], Dict[str, int]]:
        """Run each layer's attention/router per micro-batch, grouping only routed experts.

        ``split_records`` (recompute only) cuts the autograd graph at the grouped expert block:
        its inputs and the residual path become fresh leaves and one record per group lists the
        per-micro-batch tensors behind them, so the caller can run the block's backward once
        and the rest of each micro-batch's backward on its own (see ``_split_group_backward``).
        ``reverse_graph_order`` builds the micro-batches from the last to the first, in the
        forward pass and the recompute alike (two-read replays router decisions in the order
        the forward took them). Rows within a group, outputs and routes keep micro-batch order.
        """
        import torch

        experts = _streamed_moe_experts(module)
        if experts is None:
            raise RuntimeError("route-preserving MBR reached a layer without streamed experts")
        if len(hidden) != len(calls) or (states is not None and len(hidden) != len(states)):
            raise ValueError("route-preserving MBR needs one call and RNG state per micro-batch")

        captured: List[Dict[str, Any]] = []
        outputs: List[Any] = [None] * len(hidden)
        routes: List[Tuple[Any, Any]] = [None] * len(hidden)
        groups: List[List[int]] = []
        dispatch_calls = 0
        routed_tokens = 0
        route_slots = 0
        unique_experts_by_group: List[int] = []

        inner = getattr(module, "inner", module)
        wrapper_hooks, inner_hooks, wrapper_inputs = deferred_hooks
        starts = range(0, len(hidden), group_size)
        with _patched_route_preserving_forward(module, captured):
            for start in reversed(starts) if reverse_graph_order else starts:
                indices = list(range(start, min(start + group_size, len(hidden))))
                first_record = len(captured)
                build_order = list(reversed(indices)) if reverse_graph_order else list(indices)
                for mb in build_order:
                    state = states[mb] if states is not None else None
                    self._layer(module, hidden[mb], state, calls[mb])
                    if getattr(module, "_forward_hooks", None) or getattr(
                        inner, "_forward_hooks", None
                    ):
                        raise RuntimeError(
                            "route-preserving MBR does not support adding decoder forward "
                            "hooks during a grouped layer call"
                        )
                rows = captured[first_record:]
                if reverse_graph_order:
                    rows = rows[::-1]
                if len(rows) != len(indices):
                    raise RuntimeError(
                        "route-preserving MBR expected one routed-expert call per micro-batch; "
                        f"captured {len(rows)} for {len(indices)} micro-batches"
                    )
                if len(wrapper_inputs) != len(captured):
                    raise RuntimeError("route-preserving MBR could not capture decoder hook inputs")

                joined_hidden = torch.cat([row["expert_input"] for row in rows], dim=0)
                joined_ids = torch.cat([row["route_ids"] for row in rows], dim=0)
                joined_weights = torch.cat([row["route_weights"] for row in rows], dim=0)
                sizes = [int(row["expert_input"].shape[0]) for row in rows]
                for row, size in zip(rows, sizes):
                    if (
                        row["route_ids"].shape[0] != size
                        or row["route_weights"].shape != row["route_ids"].shape
                    ):
                        raise RuntimeError(
                            "route-preserving MBR route shape does not match micro-batch tokens"
                        )
                if (
                    sum(sizes) != joined_hidden.shape[0]
                    or joined_ids.shape[0] != joined_hidden.shape[0]
                ):
                    raise RuntimeError("route-preserving MBR joined token counts do not match")
                if split_records is not None:
                    split_records.append(
                        _cut_group_graph(indices, rows, joined_hidden, joined_weights, sizes)
                    )
                    cut = split_records[-1]
                    joined_hidden, joined_weights = cut["joined_hidden"], cut["joined_weights"]
                    residuals, shared = cut["residuals"], cut["shared"]
                else:
                    residuals = [row["residual"] for row in rows]
                    shared = [row["shared_output"] for row in rows]
                # Two-read normally replaces recompute output with zeros. A decoder
                # hook needs the actual value, so re-read experts for this call.
                previous_materialize = experts.materialize_recompute
                experts.materialize_recompute = previous_materialize or bool(
                    wrapper_hooks or inner_hooks
                )
                try:
                    joined_output = experts(
                        joined_hidden, joined_ids, joined_weights,
                        group_sizes=(
                            sizes if preserve_expert_shapes or experts.expert_gemm == "segmented"
                            else None
                        ),
                    )
                finally:
                    experts.materialize_recompute = previous_materialize
                unique_experts_by_group.append(len(experts.last_trace))
                if joined_output.shape != joined_hidden.shape:
                    raise RuntimeError(
                        "route-preserving MBR expert output shape does not match input"
                    )
                routed_outputs = joined_output.split(sizes, dim=0)
                for mb, row, routed, residual, shared_output in zip(
                    indices, rows, routed_outputs, residuals, shared
                ):
                    routes[mb] = (row["route_ids"], row["route_weights"])
                    routed = routed.view_as(residual)
                    mlp_output = routed + shared_output if shared_output is not None else routed
                    result = residual + mlp_output
                    if inner_hooks is not None:
                        result = _apply_deferred_forward_hooks(
                            inner, inner_hooks, row["hook_args"], row["hook_kwargs"], result
                        )
                    wrapper_args, wrapper_kwargs = wrapper_inputs[
                        first_record + build_order.index(mb)
                    ]
                    outputs[mb] = _apply_deferred_forward_hooks(
                        module, wrapper_hooks, wrapper_args, wrapper_kwargs, result
                    )
                groups.append(indices)
                dispatch_calls += 1
                routed_tokens += int(joined_hidden.shape[0])
                route_slots += int(joined_ids.numel())

        if any(output is None for output in outputs) or any(route is None for route in routes):
            raise RuntimeError("route-preserving MBR left a micro-batch without a layer output")
        if reverse_graph_order:
            groups.reverse()
            unique_experts_by_group.reverse()
        return (
            outputs,
            routes,
            groups,
            {
                "grouped_expert_dispatch_calls": dispatch_calls,
                "routed_tokens": routed_tokens,
                "route_slots": route_slots,
                "unique_experts_by_group": unique_experts_by_group,
                "unique_expert_count": sum(unique_experts_by_group),
            },
        )

    def _expert_read_snapshot(self) -> Dict[int, Dict[str, Dict[str, int]]]:
        pool = getattr(self.runtime, "expert_pool", None)
        snapshot = getattr(pool, "phase_read_snapshot", None)
        return snapshot() if callable(snapshot) else {}

    @staticmethod
    def _expert_read_delta(
        before: Dict[int, Dict[str, Dict[str, int]]],
        after: Dict[int, Dict[str, Dict[str, int]]],
        layer_idx: int,
        phase: str,
    ) -> Dict[str, int]:
        before_row = before.get(layer_idx, {}).get(phase, {})
        after_row = after.get(layer_idx, {}).get(phase, {})
        keys = before_row.keys() | after_row.keys()
        return {name: after_row.get(name, 0) - before_row.get(name, 0) for name in keys}

    def _record_expert_phase(
        self,
        layer_idx: int,
        phase: str,
        before: Dict[int, Dict[str, Dict[str, int]]],
        after: Dict[int, Dict[str, Dict[str, int]]],
        *,
        layer_calls: int,
        tokens: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        stats = self.expert_phase_stats
        if stats is None:
            return
        layer = stats["layers"].setdefault(layer_idx, {})
        row = layer.setdefault(
            phase,
            {
                "layer_calls": 0,
                "tokens_seen_by_layer_calls": 0,
                "expert_fetches": 0,
                "expert_bytes_read": 0,
            },
        )
        row["layer_calls"] += int(layer_calls)
        row["tokens_seen_by_layer_calls"] += int(tokens)
        for name, value in self._expert_read_delta(before, after, layer_idx, phase).items():
            row[name] += value
        if extra:
            for name, value in extra.items():
                if isinstance(value, list):
                    row[name] = row.get(name, []) + value
                else:
                    row[name] = row.get(name, 0) + value

    @contextmanager
    def _adapter(self, name: str) -> Iterator[None]:
        """The named PEFT adapter active inside, the previous one after.

        The semantics of TRL's ``use_adapter(model, name)``, which the DPO trainer wraps its
        reference forward in. ``set_adapter`` moves ``requires_grad`` to the active adapter
        and back; the reference pass runs under ``no_grad``, so no graph keeps the flag.
        """
        previous = self.model.active_adapter
        self.model.set_adapter(name)
        try:
            yield
        finally:
            self.model.set_adapter(previous)

    def _pass_adapters(self, spec: ReferencePass) -> Any:
        """The adapter state of an extra pass: a named adapter, every adapter off, or none."""
        if spec.adapters_off:
            # ``PeftModel.disable_adapter``: what TRL's ``null_ref_context`` enters for KTO.
            return self.model.disable_adapter()
        if spec.adapter is not None:
            return self._adapter(spec.adapter)
        return nullcontext()

    @staticmethod
    def _backward_context(ids: Any) -> Any:
        """Autocast off around a backward call, the state HF's accumulation runs it in.

        The caller wraps the whole step in autocast, forward and recompute included, but HF
        calls ``backward`` after the autocast block has exited. Autograd runs the gradient
        kernels under the autocast state of the calling thread, so a backward inside the
        block takes other dtype paths than accumulation's and the gradients differ in bf16.
        """
        import torch

        return torch.autocast(device_type=ids.device.type, enabled=False)

    def run(
        self,
        micro_batches: Sequence[Any],
        store: Any,
        *,
        n_items: Any = None,
        labels: Optional[Sequence[Any]] = None,
        calls: Optional[Sequence[LayerCall]] = None,
        head: Optional[LossHead] = None,
        reference: Union[ReferencePass, Sequence[ReferencePass], None] = None,
        routing_budget: Optional[Tuple[int, int]] = None,
        route_preserving_mbr: bool = False,
        mbr_group_size: Optional[int] = None,
        collect_phase_stats: bool = False,
        mbr_split_backward: bool = False,
        mbr_ordered_backward: bool = False,
    ) -> List[Any]:
        """``labels`` defaults to the input ids; pass them for SFT rows with ``-100``.

        ``calls`` is one ``LayerCall`` per micro-batch, captured at that micro-batch's own
        shape; micro-batches of different lengths run unpadded. Without it every
        micro-batch uses the call the step was built with.

        ``head`` replaces the SFT head and loss (see :data:`LossHead`); it owns its labels
        and its scale, so it excludes ``labels`` and ``n_items``.

        ``reference`` adds the gradient-free passes (one :class:`ReferencePass` or a sequence
        of them). It refuses a model whose layers draw dropout: accumulation's extra forwards
        would draw masks too, and the replay of ``l2l_rng`` covers the policy's draws only.

        ``route_preserving_mbr`` is an experimental DeepSeek-V3 path that leaves attention and
        routing per micro-batch and groups only streamed routed experts. It requires two-read
        experts and grouped route dispatch. ``mbr_group_size`` defaults to the full window.
        ``mbr_split_backward`` runs each group's backward in two phases (grouped experts once,
        then the rest per micro-batch), so LoRA gradients add up in the order of group size 1.
        ``mbr_ordered_backward`` gets the same order in one engine call per MoE layer: the backward
        recompute builds the micro-batches from the last to the first, so the engine
        (highest sequence number first) meets micro-batch 0's nodes first and adds every
        weight's contributions in micro-batch order. It excludes ``mbr_split_backward``.
        Batch-invariant experts use joined matmuls and split backward by default.
        """
        if type(route_preserving_mbr) is not bool:
            raise ValueError("route_preserving_mbr must be a bool")
        if type(collect_phase_stats) is not bool:
            raise ValueError("collect_phase_stats must be a bool")
        if type(mbr_split_backward) is not bool:
            raise ValueError("mbr_split_backward must be a bool")
        if type(mbr_ordered_backward) is not bool:
            raise ValueError("mbr_ordered_backward must be a bool")
        if mbr_split_backward and not route_preserving_mbr:
            raise ValueError("mbr_split_backward is only valid with route_preserving_mbr=True")
        if mbr_ordered_backward and not route_preserving_mbr:
            raise ValueError("mbr_ordered_backward is only valid with route_preserving_mbr=True")
        if mbr_ordered_backward and mbr_split_backward:
            raise ValueError("mbr_ordered_backward and mbr_split_backward exclude each other")
        if routing_budget is not None and (
            len(routing_budget) != 2
            or any(type(value) is not int or value < 0 for value in routing_budget)
        ):
            raise ValueError("routing budget needs nonnegative record and comparison bytes")
        if route_preserving_mbr:
            _validate_route_preserving_mbr(self.layers, self.causal)
            if mbr_group_size is not None and (
                isinstance(mbr_group_size, bool)
                or not isinstance(mbr_group_size, int)
                or mbr_group_size < 1
            ):
                raise ValueError("mbr_group_size must be a positive integer or None")
        elif mbr_group_size is not None:
            raise ValueError("mbr_group_size is only valid with route_preserving_mbr=True")
        if head is not None and (labels is not None or n_items is not None):
            raise ValueError("a custom head owns its labels and scale: pass no labels or n_items")
        self.route_preserving_mbr_stats = None
        expert_pool = getattr(self.runtime, "expert_pool", None)
        previous_phase_tracking = getattr(expert_pool, "phase_read_tracking", None)
        if collect_phase_stats and previous_phase_tracking is not None:
            expert_pool.phase_read_tracking = True
        try:
            with count_loads(self.runtime) as counters:
                if route_preserving_mbr:
                    losses = self._run(
                        micro_batches,
                        store,
                        n_items,
                        labels,
                        calls,
                        None,
                        head,
                        reference,
                        route_preserving_mbr=True,
                        mbr_group_size=mbr_group_size,
                        collect_phase_stats=collect_phase_stats,
                        routing_budget=routing_budget,
                        mbr_split_backward=mbr_split_backward,
                        mbr_ordered_backward=mbr_ordered_backward,
                    )
                else:
                    guard = (
                        RoutingGuard(self.layers)
                        if routing_budget is None
                        else RoutingGuard(self.layers, routing_budget=routing_budget)
                    )
                    with guard:
                        losses = self._run(
                            micro_batches,
                            store,
                            n_items,
                            labels,
                            calls,
                            guard,
                            head,
                            reference,
                            collect_phase_stats=collect_phase_stats,
                        )
        finally:
            if previous_phase_tracking is not None:
                expert_pool.phase_read_tracking = previous_phase_tracking
        self.counters = counters
        return losses

    def _run(
        self,
        micro_batches: Sequence[Any],
        store: Any,
        n_items: Any,
        labels: Optional[Sequence[Any]],
        calls: Optional[Sequence[LayerCall]],
        guard: Optional[RoutingGuard],
        head: Optional[LossHead],
        reference: Union[ReferencePass, Sequence[ReferencePass], None] = None,
        *,
        route_preserving_mbr: bool = False,
        mbr_group_size: Optional[int] = None,
        collect_phase_stats: bool = False,
        routing_budget: Optional[Tuple[int, int]] = None,
        mbr_split_backward: bool = False,
        mbr_ordered_backward: bool = False,
    ) -> List[Any]:
        import torch

        from soup_cli.utils import l2l_rng
        from soup_cli.utils.expert_stream import checkpoint_context_fn

        k = len(micro_batches)
        if k < 1:
            raise ValueError("an L2L step needs at least one micro-batch")
        if mbr_group_size is not None and mbr_group_size > store.k:
            raise ValueError(
                f"mbr_group_size={mbr_group_size} exceeds the {store.k}-micro-batch capacity"
            )
        # The final optimizer window can contain fewer micro-batches than the configured
        # capacity; run that shorter window as one group instead of rejecting it.
        group_size = k if mbr_group_size is None else min(mbr_group_size, k)
        preserve_expert_shapes = mbr_split_backward or mbr_ordered_backward
        if k > store.k:
            raise ValueError(f"{k} micro-batches, but the activation store holds {store.k}")
        if calls is None:
            calls = [self.call] * k
        elif len(calls) != k:
            raise ValueError(f"{len(calls)} layer calls for {k} micro-batches")
        if head is None:
            if labels is None:
                labels = micro_batches
            elif len(labels) != k:
                raise ValueError(f"{len(labels)} label tensors for {k} micro-batches")
            pool = getattr(self.runtime, "expert_pool", None)
            if n_items is None:
                total = label_tokens(micro_batches)
            elif getattr(pool, "group_compute", None) is not None:
                # Match the native causal loss operand under the grouped training recipe.
                total = n_items
            else:
                total = int(n_items)
            head = sft_head(labels, total)

        passes = as_passes(reference)
        if passes and l2l_rng.has_draws(self.call.draws):
            raise ValueError(
                "the L2L reference pass does not support dropout: the reference forward "
                "of gradient accumulation draws dropout masks too, and the replay covers "
                "the policy's draws only. Set attention_dropout to 0."
            )
        pass_ids = [_pass_micro_batches(spec, micro_batches, k) for spec in passes]
        pass_calls = [_pass_calls(spec, calls, k) for spec in passes]
        replayed = None
        plan = (*self.call.draws, self.call.head_draws)
        if l2l_rng.has_draws(plan):
            replayed = l2l_rng.replay(plan, k)
        states = replayed[0] if replayed else None
        train_embedding = any(param.requires_grad for param in self.owner.embed_tokens.parameters())
        embedding_states: List[Any] = []

        # Standard L2L owns a route record per (layer, micro-batch). MBR records the same
        # per-micro-batch IDs itself because its grouped expert call bypasses RoutingGuard.
        contexts = {}
        for idx, layer in enumerate(self.layers):
            # L2L calls the outer wrapper during recompute too, so its output hooks
            # must be guarded along with the native decoder and MoE modules.
            context_fn = checkpoint_context_fn(layer)
            if context_fn is not None:
                if route_preserving_mbr and _streamed_moe_experts(layer) is not None:
                    contexts[idx] = context_fn()
                else:
                    for mb in range(k):
                        contexts[idx, mb] = context_fn()

        mbr_routes: Dict[int, List[Tuple[Any, Any]]] = {}
        mbr_route_record_bytes = 0
        phase_stats: Optional[Dict[str, Any]] = (
            {"group_size": group_size, "layers": {}} if collect_phase_stats else None
        )
        self.expert_phase_stats = phase_stats
        if route_preserving_mbr:
            self.route_preserving_mbr_stats = phase_stats

        self._prime(self.owner, ())
        pass_hidden: List[List[Any]] = []
        with torch.no_grad():
            hidden = []
            for ids in micro_batches:
                if train_embedding:
                    embedding_states.append(l2l_rng.snapshot())
                hidden.append(self.owner.embed_tokens(ids))
            for spec, ids in zip(passes, pass_ids):
                with self._pass_adapters(spec):
                    pass_hidden.append([self.owner.embed_tokens(x) for x in ids])
            for idx, layer in enumerate(self.layers):
                for mb in range(k):
                    store.put(idx, mb, hidden[mb])
                experts = _streamed_moe_experts(layer) if route_preserving_mbr else None
                if experts is not None:
                    expert_gemm = experts.expert_gemm
                    pair = contexts.get(idx)
                    before = self._expert_read_snapshot() if phase_stats is not None else None
                    with (
                        _defer_decoder_forward_hooks(
                            layer,
                            split_backward=mbr_split_backward or (
                                expert_gemm in ("batch_invariant", "segmented")
                                and not mbr_ordered_backward
                            ),
                        ) as deferred_hooks,
                        pair[0] if pair else nullcontext(),
                    ):
                        layer_outputs, layer_routes, layer_groups, dispatch = (
                            self._route_preserving_layer_window(
                                layer,
                                hidden,
                                [states[mb][idx] for mb in range(k)] if states else None,
                                calls,
                                group_size,
                                deferred_hooks,
                                reverse_graph_order=mbr_ordered_backward,
                                preserve_expert_shapes=(
                                    preserve_expert_shapes and expert_gemm != "batch_invariant"
                                ),
                            )
                        )
                    for mb in range(k):
                        hidden[mb] = layer_outputs[mb]
                    for route_ids, route_weights in layer_routes:
                        route_bytes = int(route_ids.numel() * route_ids.element_size())
                        if routing_budget is not None:
                            records, comparison = routing_budget
                            if (
                                route_bytes > comparison
                                or mbr_route_record_bytes + route_bytes > records
                            ):
                                raise ValueError(
                                    "route-preserving MBR route IDs exceed the resolved routing "
                                    "memory budget"
                                )
                        saved_ids = route_ids.detach().to("cpu", copy=True)
                        mbr_route_record_bytes += route_bytes
                        saved_weights = (
                            route_weights.detach().float().to("cpu", copy=True)
                            if phase_stats is not None
                            else None
                        )
                        mbr_routes.setdefault(idx, []).append((saved_ids, saved_weights))
                    if phase_stats is not None:
                        after = self._expert_read_snapshot()
                        self._record_expert_phase(
                            idx,
                            "init_forward",
                            before,
                            after,
                            layer_calls=k,
                            tokens=sum(int(value.shape[0] * value.shape[1]) for value in hidden),
                            extra=dispatch,
                        )
                        phase_stats["layers"][idx]["init_forward"].update(dispatch)
                else:
                    before = self._expert_read_snapshot() if phase_stats is not None else None
                    for mb in range(k):
                        pair = contexts.get((idx, mb))
                        route_context = (
                            guard.record(idx, mb) if guard is not None else nullcontext()
                        )
                        with route_context, pair[0] if pair else nullcontext():
                            hidden[mb] = self._layer(
                                layer,
                                hidden[mb],
                                states[mb][idx] if states else None,
                                calls[mb],
                            )
                    if before is not None:
                        after = self._expert_read_snapshot()
                        self._record_expert_phase(
                            idx,
                            "init_forward",
                            before,
                            after,
                            layer_calls=k,
                            tokens=sum(int(value.shape[0] * value.shape[1]) for value in hidden),
                        )
                for spec, current, spec_calls in zip(passes, pass_hidden, pass_calls):
                    # Same loaded layer: nothing is stored, nothing loads. Outside
                    # ``guard.record``, so the routing guard sees the policy forward only.
                    with self._pass_adapters(spec):
                        for mb in range(k):
                            current[mb] = self._layer(layer, current[mb], None, spec_calls[mb])
            for spec, current in zip(passes, pass_hidden):
                spec.outputs.clear()
                with self._pass_adapters(spec):
                    for mb in range(k):
                        spec.outputs[mb] = spec.head(self.causal, self.owner, current[mb], mb)
                        current[mb] = None

        losses: List[Any] = []
        grads: List[Any] = []
        for mb in range(k):
            last = hidden[mb].detach().requires_grad_(True)
            hidden[mb] = None
            head_state = states[mb][-1] if states and self.call.head_draws else None
            with (
                torch.enable_grad(),
                l2l_rng.state_at(head_state) if head_state is not None else nullcontext(),
            ):
                loss = head(self.causal, self.owner, last, mb)
            with self._backward_context(micro_batches[mb]):
                loss.backward()
            losses.append(loss.detach())
            grads.append(last.grad)

        store.begin_backward()
        for idx in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[idx]
            saved = layer.use_checkpoint
            layer.use_checkpoint = False
            try:
                experts = _streamed_moe_experts(layer) if route_preserving_mbr else None
                if experts is not None:
                    expert_gemm = experts.expert_gemm
                    split_backward = mbr_split_backward or (
                        expert_gemm in ("batch_invariant", "segmented") and not mbr_ordered_backward
                    )
                    inputs = [store.get(idx, mb).requires_grad_(True) for mb in range(k)]
                    pair = contexts.pop(idx, None)
                    before_recompute = (
                        self._expert_read_snapshot() if phase_stats is not None else None
                    )
                    split_records: Optional[List[Dict[str, Any]]] = (
                        [] if split_backward else None
                    )
                    # Distinct bf16 casts round each adapter gradient before the fp32 sum,
                    # as separate L2L backward calls do even when they reuse the same cast.
                    device_type = inputs[0].device.type
                    autocast = (
                        torch.autocast(device_type, cache_enabled=False)
                        if mbr_ordered_backward and torch.is_autocast_enabled(device_type)
                        else nullcontext()
                    )
                    with (
                        torch.enable_grad(),
                        autocast,
                        _defer_decoder_forward_hooks(
                            layer, split_backward=split_backward
                        ) as deferred_hooks,
                        pair[1] if pair else nullcontext(),
                    ):
                        outputs, layer_routes, layer_groups, dispatch = (
                            self._route_preserving_layer_window(
                                layer,
                                inputs,
                                [states[mb][idx] for mb in range(k)] if states else None,
                                calls,
                                group_size,
                                deferred_hooks,
                                split_records=split_records,
                                reverse_graph_order=mbr_ordered_backward,
                                preserve_expert_shapes=(
                                    preserve_expert_shapes and expert_gemm != "batch_invariant"
                                ),
                            )
                        )
                    after_recompute = (
                        self._expert_read_snapshot() if phase_stats is not None else None
                    )
                    expected_routes = mbr_routes.pop(idx, None)
                    if expected_routes is None or len(expected_routes) != k:
                        raise RuntimeError(
                            "route-preserving MBR has no complete forward route record "
                            f"for layer {idx}"
                        )
                    mbr_route_record_bytes -= sum(
                        int(ids.numel() * ids.element_size()) for ids, _ in expected_routes
                    )
                    if mbr_route_record_bytes < 0:
                        raise RuntimeError("route-preserving MBR route-record accounting underflow")
                    expected_weights = []
                    recompute_weights = []
                    for mb, (route_ids, route_weights) in enumerate(layer_routes):
                        before_ids, before_weights = expected_routes[mb]
                        current_bytes = int(route_ids.numel() * route_ids.element_size())
                        if routing_budget is not None and current_bytes > routing_budget[1]:
                            raise ValueError(
                                "route-preserving MBR route comparison exceeds the resolved "
                                "routing comparison budget"
                            )
                        current_ids = route_ids.detach().to("cpu")
                        if before_ids.shape != current_ids.shape:
                            raise RuntimeError(
                                f"route-preserving MBR route shape changed at layer {idx}, "
                                f"micro-batch {mb}: {tuple(before_ids.shape)} -> "
                                f"{tuple(current_ids.shape)}"
                            )
                        if not torch.equal(before_ids, current_ids):
                            changed_slots = int((before_ids != current_ids).sum().item())
                            raise RuntimeError(
                                f"route-preserving MBR recompute changed {changed_slots} route "
                                f"slots at layer {idx}, micro-batch {mb}"
                            )
                        if phase_stats is not None:
                            if before_weights is None:
                                raise RuntimeError(
                                    "route-preserving MBR diagnostics did not save gate weights"
                                )
                            expected_weights.append(before_weights.reshape(-1))
                            recompute_weights.append(
                                route_weights.detach().float().to("cpu").reshape(-1)
                            )
                    if phase_stats is not None:
                        expected_flat = torch.cat(expected_weights)
                        recompute_flat = torch.cat(recompute_weights)
                        weight_delta = recompute_flat - expected_flat
                        weight_norm = torch.linalg.vector_norm(expected_flat).clamp_min(1e-30)
                        gate_weight_rel_l2 = float(
                            (torch.linalg.vector_norm(weight_delta) / weight_norm).item()
                        )
                        gate_weight_max_abs = float(weight_delta.abs().max().item())
                        before_backward = self._expert_read_snapshot()
                    # One ordered backward over the layer preserves the running sum across
                    # group boundaries as well as within each group.
                    backward_groups = [range(k)] if mbr_ordered_backward else layer_groups
                    for position, group in enumerate(backward_groups):
                        if split_records is not None:
                            _split_group_backward(
                                split_records[position],
                                outputs,
                                grads,
                                lambda mb: self._backward_context(micro_batches[mb]),
                            )
                            continue
                        with self._backward_context(micro_batches[group[0]]):
                            torch.autograd.backward(
                                [outputs[mb] for mb in group],
                                grad_tensors=[grads[mb] for mb in group],
                            )
                    after_backward = (
                        self._expert_read_snapshot() if phase_stats is not None else None
                    )
                    for mb, input_state in enumerate(inputs):
                        grads[mb] = input_state.grad

                    if phase_stats is not None:
                        layer_stats = phase_stats["layers"].setdefault(idx, {})
                        self._record_expert_phase(
                            idx,
                            "recompute_forward",
                            before_recompute,
                            after_recompute,
                            layer_calls=k,
                            tokens=sum(int(value.shape[0] * value.shape[1]) for value in inputs),
                            extra=dispatch,
                        )
                        self._record_expert_phase(
                            idx,
                            "backward_dX",
                            before_backward,
                            after_backward,
                            layer_calls=len(backward_groups),
                            tokens=sum(int(value.shape[0] * value.shape[1]) for value in inputs),
                        )
                        layer_stats["recompute_forward"].update(dispatch)
                        # A nonzero mismatch is rejected above, so every completed
                        # recompute has exact route-ID parity by construction.
                        layer_stats["route_id_mismatch_slots"] = 0
                        layer_stats["route_id_slots"] = sum(
                            int(route_ids.numel()) for route_ids, _ in layer_routes
                        )
                        layer_stats["gate_weight_rel_l2"] = gate_weight_rel_l2
                        layer_stats["gate_weight_max_abs"] = gate_weight_max_abs
                    del inputs, outputs, layer_routes, layer_groups, dispatch
                else:
                    phase_recompute_before = (
                        self._expert_read_snapshot() if phase_stats is not None else None
                    )
                    for mb in range(k):
                        inputs = store.get(idx, mb).requires_grad_(True)
                        pair = contexts.pop((idx, mb), None)
                        route_context = guard.check(idx, mb) if guard is not None else nullcontext()
                        with (
                            torch.enable_grad(),
                            route_context,
                            pair[1] if pair else nullcontext(),
                        ):
                            out = self._layer(
                                layer, inputs, states[mb][idx] if states else None, calls[mb]
                            )
                        phase_recompute_after = (
                            self._expert_read_snapshot() if phase_stats is not None else None
                        )
                        if phase_stats is not None:
                            self._record_expert_phase(
                                idx,
                                "recompute_forward",
                                phase_recompute_before,
                                phase_recompute_after,
                                layer_calls=1,
                                tokens=int(inputs.shape[0] * inputs.shape[1]),
                            )
                            phase_recompute_before = phase_recompute_after
                            phase_backward_before = phase_recompute_after
                        with self._backward_context(micro_batches[mb]):
                            torch.autograd.backward(out, grad_tensors=grads[mb])
                        if phase_stats is not None:
                            phase_backward_after = self._expert_read_snapshot()
                            self._record_expert_phase(
                                idx,
                                "backward_dX",
                                phase_backward_before,
                                phase_backward_after,
                                layer_calls=1,
                                tokens=int(inputs.shape[0] * inputs.shape[1]),
                            )
                            phase_backward_before = phase_backward_after
                        grads[mb] = inputs.grad
                        del inputs, out
            finally:
                layer.use_checkpoint = saved
        if route_preserving_mbr and (mbr_routes or mbr_route_record_bytes != 0):
            raise RuntimeError("route-preserving MBR did not consume every forward route record")
        if train_embedding:
            # The reverse walk prefetches this at layer 0 and records it on the prefetcher. A
            # one-layer decoder never changes direction, so only then is it still to be asked.
            prefetcher = self.runtime.prefetcher
            refill = prefetcher.backward_tail_prefetch
            if refill is not None and not prefetcher.backward_tail_prefetched:
                refill()
                prefetcher.backward_tail_prefetched = True
            for mb, ids in enumerate(micro_batches):
                with torch.enable_grad(), l2l_rng.state_at(embedding_states[mb]):
                    embedded = self.owner.embed_tokens(ids)
                with (
                    torch.autocast("cpu", enabled=False),
                    torch.autocast("cuda", enabled=False),
                ):
                    torch.autograd.backward(embedded, grad_tensors=grads[mb])
                grads[mb] = None
        store.end_step()
        if replayed is not None:
            l2l_rng.leave_stream_at(replayed[1])
        return losses
