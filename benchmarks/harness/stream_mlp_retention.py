"""Benchmark-only, fixed QKVO + gate/up experiment; no production option."""

from __future__ import annotations

import contextvars
import weakref
from contextlib import contextmanager
from typing import Any

from stream_selective_checkpoint import SelectiveCheckpoint

ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj")


def candidate_plan(native: dict, *, include_mlp: bool = True) -> dict:
    """Candidate A is fixed before execution; a different census refuses it."""
    from soup_cli.utils.layer_stream import retained_layer_charge

    expected = dict(measured=True, k=11, full_layer_charge=482015738,
                    budget_bytes=5756682240, peak_reserved=2833252352,
                    peak_allocated=2468564480, retained_indices=list(range(21, 32)))
    if any(native.get(key) != value for key, value in expected.items()):
        raise ValueError("native census differs from candidate A's declared cell")
    q_raw, m_raw = 41943040, 58720256
    q_charge = retained_layer_charge(q_raw, native["peak_reserved"], native["peak_allocated"])
    m_charge = retained_layer_charge(m_raw, native["peak_reserved"], native["peak_allocated"])
    count = 10 if include_mlp else 0
    budget = native["budget_bytes"] - 8 * native["full_layer_charge"]
    charged = 24 * q_charge + count * m_charge
    if budget - charged < 64 * 2**20:
        raise ValueError("candidate census leaves less than 64 MiB charged slack")
    return dict(native_k=11, effective_k=8, retained_indices=list(range(24, 32)),
                qkvo_indices=list(range(24)), mlp_indices=list(range(19, 24)) if count else [],
                raw_bytes=24 * q_raw + count * m_raw, charged_bytes=charged,
                residual_bytes=budget, slack_bytes=budget - charged,
                cache_tensors=96 + count)


def _base(layer: Any, name: str) -> Any:
    group = layer.inner.self_attn if name in ATTENTION else layer.inner.mlp
    projection = getattr(group, name)
    return getattr(projection, "base_layer", projection)


class MixedCheckpoint(SelectiveCheckpoint):
    """Preserve the existing layer-wide SAC and cache mutation checks."""

    def __init__(self, layers: list[Any], qkvo: list[int], mlp: list[int]) -> None:
        import torch
        from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

        by_index = {layer.idx: layer for layer in layers}
        for indices in (qkvo, mlp):
            if (any(type(index) is not int for index in indices)
                    or len(indices) != len(set(indices)) or not set(indices) <= by_index.keys()):
                raise ValueError("invalid or duplicate selected layer")
        if not qkvo or not set(mlp) <= set(qkvo):
            raise ValueError("MLP roster must be a subset of the QKVO roster")
        selected = [by_index[index] for index in qkvo]
        if any(not layer.recompute or not layer.use_checkpoint for layer in selected):
            raise ValueError("selected layers must use checkpoint recompute")
        projections = []
        for layer in selected:
            names = ATTENTION + (MLP if layer.idx in mlp else ())
            for name in names:
                projection = _base(layer, name)
                if any(parameter.requires_grad for parameter in projection.parameters()):
                    raise ValueError("only frozen base linear outputs may be retained")
                projections.append((projection, f"{layer.idx}.{name}"))
        # Validate the complete roster before changing any forward.
        self.layer_count = len(selected)
        self.counts = {"saved_forward": 0, "recompute_projection_calls": 0}
        self.ops: dict[str, int] = {}
        self._active = contextvars.ContextVar("soup2162_mlp_projection", default=None)
        self._recomputing = contextvars.ContextVar("soup2162_mlp_recomputing", default=False)
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
                    key = str(op)
                    self.ops[key] = self.ops.get(key, 0) + 1
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.PREFER_RECOMPUTE

        def contexts() -> Any:
            modes = create_selective_checkpoint_contexts(policy_fn)
            self._modes = [ref for ref in self._modes if ref() is not None]
            self._modes.append(weakref.ref(modes[1]))

            @contextmanager
            def recompute() -> Any:
                token = self._recomputing.set(True)
                try:
                    with modes[1]:
                        yield
                finally:
                    self._recomputing.reset(token)

            return modes[0], recompute()

        def layer_forward(instance: Any, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
            from torch.utils.checkpoint import checkpoint

            if not instance.recompute or not instance.use_checkpoint:
                raise RuntimeError("selected layer mode changed after installation")
            if torch.is_grad_enabled():
                return checkpoint(instance._body, hidden_states, *args, use_reentrant=False,
                                  context_fn=contexts, **kwargs)
            return instance._body(hidden_states, *args, **kwargs)

        for projection, name in projections:
            self._patch(projection, self._projection_forward(projection.forward, name))
        for layer in selected:
            self._patch(layer, layer_forward)


def install(model: Any, qkvo: list[int], mlp: list[int]) -> MixedCheckpoint:
    from soup_cli.utils.layer_stream_runtime import streamed_decoder_layers

    return MixedCheckpoint(streamed_decoder_layers(model), qkvo, mlp)


class ProjectionObserver:
    """Read-only profiler scopes; distinguish recomputation from gradient matmuls."""

    def __init__(self, model: Any) -> None:
        import types

        from soup_cli.utils.layer_stream_runtime import streamed_decoder_layers

        self._phase = contextvars.ContextVar("soup2162_profile_phase", default="outside")
        self._originals = []
        for layer in streamed_decoder_layers(model):
            for name in ATTENTION + MLP + ("down_proj",):
                module = _base(layer, name)
                original = module.forward
                self._originals.append(
                    (module, "forward" in module.__dict__, module.__dict__.get("forward"))
                )
                module.forward = types.MethodType(self._wrap(original, layer.idx, name), module)

    def _wrap(self, original: Any, index: int, name: str) -> Any:
        def forward(instance: Any, *args: Any, **kwargs: Any) -> Any:
            from torch.profiler import record_function

            with record_function(f"soup2162|{self._phase.get()}|{index}|{name}"):
                return original(*args, **kwargs)

        return forward

    @contextmanager
    def phase(self, phase: str) -> Any:
        if phase not in ("forward", "backward"):
            raise ValueError("unexpected model phase")
        token = self._phase.set(phase)
        try:
            yield
        finally:
            self._phase.reset(token)

    def remove(self) -> None:
        for module, existed, forward in reversed(self._originals):
            if existed:
                module.forward = forward
            else:
                del module.forward
        self._originals.clear()


def summarize_events(events: list[Any]) -> list[dict]:
    """Count innermost actual operations, avoiding SAC nested-dispatch duplication."""
    operations = {"aten::mm", "aten::addmm", "bitsandbytes::gemm_4bit"}
    summaries: dict[tuple, dict] = {}
    for event in events:
        if event.name not in operations:
            continue
        # An outer op intercepted by SAC is recorded even on a cache hit.
        # Only its redispatched inner op executes a kernel; on a hit there is none.
        if any(child.name == "PythonDispatchMode" for child in event.cpu_children):
            continue
        pending = list(event.cpu_children)
        nested = False
        while pending:
            child = pending.pop()
            if child.name in operations:
                nested = True
                break
            pending.extend(child.cpu_children)
        if nested:
            continue
        parent = event.cpu_parent
        while parent is not None and not parent.name.startswith("soup2162|"):
            parent = parent.cpu_parent
        if parent is None:
            continue
        _, phase, layer, projection = parent.name.split("|")
        if phase not in ("forward", "backward"):
            raise ValueError("profile event has no model phase")
        key = (phase, int(layer), projection, event.name)
        row = summaries.setdefault(
            key, dict(phase=phase, layer=int(layer), projection=projection,
                      op=event.name, count=0, device_us=0.0, self_cpu_us=0.0)
        )
        row["count"] += 1
        row["device_us"] += event.device_time_total
        row["self_cpu_us"] += event.self_cpu_time_total
    return [summaries[key] for key in sorted(summaries)]
