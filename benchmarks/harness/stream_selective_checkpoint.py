"""Benchmark-only attention projection rematerialization policies for #2050.

This installs instance forwards on an already built streamed model. There is no
Soup config field. Only the frozen base projection's matrix multiply result is
saved: LoRA products, dequantized weights, MLP and attention remain recomputed.
The ordinary non-reentrant checkpoint still owns the streaming slot lifetime.
All streamed decoder layers must use recompute; retained layers are refused.
"""

from __future__ import annotations

import contextvars
import types
import weakref
from contextlib import contextmanager
from typing import Any

POLICIES = {"qkv": ("q_proj", "k_proj", "v_proj"), "qkvo": ("q_proj", "k_proj", "v_proj", "o_proj")}


def saved_projection_bytes(
    policy: str,
    layers: int,
    hidden: int,
    kv: int,
    seq: int,
    batch: int = 1,
    element_bytes: int = 2,
) -> int:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    if any(
        type(value) is not int or value <= 0
        for value in (layers, hidden, kv, seq, batch, element_bytes)
    ):
        raise ValueError("dimensions must be positive integers")
    width = hidden + 2 * kv + (hidden if policy == "qkvo" else 0)
    return layers * width * seq * batch * element_bytes


class SelectiveCheckpoint:
    def __init__(self, model: Any, policy: str) -> None:
        import torch
        from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

        from soup_cli.utils.layer_stream_runtime import StreamedDecoderLayer

        if policy not in POLICIES:
            raise ValueError(f"unknown policy: {policy}")
        # Canonical named_modules() intentionally hides streaming wrappers
        # (#1005). apply() is their explicit structural visitor.
        layers = []
        model.apply(
            lambda module: (
                layers.append(module) if isinstance(module, StreamedDecoderLayer) else None
            )
        )
        if not layers:
            raise ValueError("no streamed decoder layers found")
        if not all(getattr(layer, "recompute", True) for layer in layers):
            raise ValueError(
                "selective checkpoint benchmark requires recompute=True on every layer"
            )
        projections = []
        for layer in layers:
            attention = getattr(layer.inner, "self_attn", None)
            for name in POLICIES[policy]:
                projection = getattr(attention, name, None)
                if projection is None:
                    raise ValueError(f"streamed layer {layer.idx} has no self_attn.{name}")
                # Do not cache the adapter's two low-rank multiplies too.
                projections.append(
                    (getattr(projection, "base_layer", projection), f"{layer.idx}.{name}")
                )
        self.policy = policy
        self.layer_count = len(layers)
        self.counts = {"saved_forward": 0, "recompute_projection_calls": 0}
        self.ops: dict[str, int] = {}
        self._active = contextvars.ContextVar("soup2050_projection", default=None)
        self._recomputing = contextvars.ContextVar("soup2050_recomputing", default=False)
        self._forwards: list[tuple[Any, bool, Any]] = []
        self._modes: list[Any] = []
        selected_ops = {torch.ops.aten.mm.default, torch.ops.aten.addmm.default}
        try:
            selected_ops.add(torch.ops.bitsandbytes.gemm_4bit.default)
        except AttributeError:
            pass

        def policy_fn(ctx: Any, op: Any, *args: Any, **kwargs: Any) -> Any:
            if self._active.get() is not None and op in selected_ops:
                if not ctx.is_recompute:
                    self.counts["saved_forward"] += 1
                    op_key = str(op)
                    self.ops[op_key] = self.ops.get(op_key, 0) + 1
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.PREFER_RECOMPUTE

        def contexts() -> Any:
            modes = create_selective_checkpoint_contexts(policy_fn)
            self._modes = [ref for ref in self._modes if ref() is not None]
            self._modes.append(weakref.ref(modes[1]))

            @contextmanager
            def recompute():
                token = self._recomputing.set(True)
                try:
                    with modes[1]:
                        yield
                finally:
                    self._recomputing.reset(token)

            return modes[0], recompute()

        def layer_forward(instance: Any, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
            from torch.utils.checkpoint import checkpoint

            if instance.use_checkpoint and torch.is_grad_enabled():
                return checkpoint(
                    instance._body,
                    hidden_states,
                    *args,
                    use_reentrant=False,
                    context_fn=contexts,
                    **kwargs,
                )
            return instance._body(hidden_states, *args, **kwargs)

        for projection, name in projections:
            self._patch(projection, self._projection_forward(projection.forward, name))
        for layer in layers:
            self._patch(layer, layer_forward)

    def _projection_forward(self, original: Any, name: str) -> Any:
        def forward(instance: Any, *args: Any, **kwargs: Any) -> Any:
            if self._recomputing.get():
                self.counts["recompute_projection_calls"] += 1
            token = self._active.set(name)
            try:
                return original(*args, **kwargs)
            finally:
                self._active.reset(token)

        return forward

    def _patch(self, module: Any, forward: Any) -> None:
        self._forwards.append(
            (module, "forward" in module.__dict__, module.__dict__.get("forward"))
        )
        module.forward = types.MethodType(forward, module)

    def cached_tensors(self) -> list[Any]:
        """Inspect PyTorch's SAC storage for the audit, without retaining tensors.

        This private structure is deliberately confined to the measurement
        helper and checked against each recorded torch version.
        """
        import torch
        import torch.utils.checkpoint as checkpoint_module
        from torch.utils._pytree import tree_leaves

        tensors = []
        for ref in self._modes:
            mode = ref()
            if mode is None:
                continue
            for entries in mode.storage.values():
                for value in tree_leaves(entries):
                    # Since torch 2.14 the indexed cache also records unsaved
                    # and consumed operations. Older versions store saves only.
                    if value is getattr(checkpoint_module, "_RECOMPUTE", None) or (
                        value is getattr(checkpoint_module, "_CONSUMED", None)
                    ):
                        continue
                    if not hasattr(value, "get_val"):
                        raise RuntimeError("PyTorch selective cache inspection contract changed")
                    tensor = value.get_val(False)
                    if not isinstance(tensor, torch.Tensor):
                        raise RuntimeError("selective cache contains a non-tensor output")
                    tensors.append(tensor)
        return tensors

    def cache_bytes(self) -> int:
        storages = {}
        for tensor in self.cached_tensors():
            storage = tensor.untyped_storage()
            storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
        return sum(storages.values())

    def remove(self) -> None:
        for module, had_forward, forward in reversed(self._forwards):
            if had_forward:
                module.forward = forward
            else:
                del module.forward
        self._forwards.clear()
        self._modes.clear()


def install(model: Any, policy: str) -> SelectiveCheckpoint:
    return SelectiveCheckpoint(model, policy)
