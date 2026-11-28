"""Dense-only research activation retention with explicit streamed-weight ownership."""

from __future__ import annotations

from types import MethodType
from typing import Any


def _snapshot(value):
    return value.detach().clone()


def install_retention(model: Any, indices: list[int], *, resident: bool) -> dict:
    """Skip selected checkpoints, snapshotting every saved alias of rotating storage.

    This independent activation arm retains activations AND autograd's necessary
    frozen weight operands. It does not pretend activation-only retention can
    avoid those operands. The resident combination borrows immutable private slots.
    """
    import torch

    from soup_cli.utils.layer_stream_runtime import StreamedDecoderLayer

    if not indices or len(set(indices)) != len(indices):
        raise ValueError("retention needs unique selected layers")
    selected = set(indices)
    found, all_layers, pending, seen = {}, [], [model], set()
    while pending:
        module = pending.pop()
        if module is None or id(module) in seen:
            continue
        seen.add(id(module))
        if isinstance(module, StreamedDecoderLayer):
            all_layers.append(module)
            if module.idx in selected:
                found[module.idx] = module
        else:
            pending.extend(module._modules.values())
    if set(found) != selected:
        raise ValueError("selected layer missing")
    for module in found.values():
        if module.quant_specs or not module.use_checkpoint:
            raise ValueError("requires dense checkpointed decoder layers")
        persistent = getattr(module.pool, "_persistent_layers", {}).get("indices", [])
        if resident != (module.idx in persistent):
            raise ValueError("retention ownership does not match residency")
        if getattr(module, "_research_retention", False):
            raise ValueError("retention already installed")
    state = {"indices": indices, "resident": resident, "retained_forward_calls": 0,
             "weight_snapshot_bytes": 0, "weight_snapshot_tensors": 0,
             "saved_nonpool_bytes": 0, "borrowed_weight_bytes": 0,
             "body_calls": {}, "storage_policy": "immutable resident" if resident
             else "private saved rotating-weight copies"}
    # Backward lookahead ordinarily relies on EVERY checkpoint recomputing.
    # A retained neighbour does not run _body; recover missing prefetch at the
    # next recomputed body's entry using the original pool copy/event ordering.
    state["missing_prefetch_loads"] = 0
    for module in all_layers:
        original = module._body

        def body(self, *args, _original=original, **kwargs):
            if self.pool.owner[self.pool.slot_for(self.idx)] != self.idx:
                self.pool.load_async(self.idx, self.prefetcher.source, self.prefetcher.stream)
                state["missing_prefetch_loads"] += 1
            return _original(*args, **kwargs)

        module._body = MethodType(body, module)
    for index, module in found.items():
        original_body = module._body

        def forward(self, *args, _body=original_body, _index=index, **kwargs):
            # Build on entry: dedicated pool slots must already have been installed.
            pool_ptrs = {value.untyped_storage().data_ptr()
                         for slot in self.pool.buffers for value in slot.values()}
            owner_slot = self.pool.slot_for(self.idx)
            owner_ptrs = {value.untyped_storage().data_ptr()
                          for value in self.pool.buffers[owner_slot].values()}
            snapshots = {}

            def pack(value):
                pointer = value.untyped_storage().data_ptr()
                size = value.numel() * value.element_size()
                if pointer in pool_ptrs:
                    if resident:
                        if pointer not in owner_ptrs or self.pool.owner[owner_slot] != self.idx:
                            raise RuntimeError("saved operand has no immutable owner")
                        state["borrowed_weight_bytes"] += size
                        return value.detach()
                    # A saved weight may be a transposed view; preserve its actual
                    # values/shape/stride rather than substituting a whole base slot.
                    key = (pointer, value.storage_offset(),
                           tuple(value.shape), tuple(value.stride()))
                    if key not in snapshots:
                        snapshots[key] = _snapshot(value)
                        state["weight_snapshot_bytes"] += size
                        state["weight_snapshot_tensors"] += 1
                    return snapshots[key]
                state["saved_nonpool_bytes"] += size
                return value.detach()

            state["body_calls"][_index] = state["body_calls"].get(_index, 0) + 1
            if not torch.is_grad_enabled():
                return _body(*args, **kwargs)
            state["retained_forward_calls"] += 1
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
                return _body(*args, **kwargs)

        module.forward = MethodType(forward, module)
        module._research_retention = True
    return state
