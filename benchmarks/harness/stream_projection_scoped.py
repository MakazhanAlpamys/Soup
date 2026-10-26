"""Benchmark-only SAC dispatch confined to frozen Q/K/V/O base forwards.

The checkpoint still spans the native layer body and owns its streaming lifetime.
Only dispatch instrumentation moves: attention, LoRA and MLP operations no longer
enter SAC. Its normal mutation checks and saved-output storage remain intact.
"""
from __future__ import annotations

import contextvars
import importlib.util
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_path = Path(__file__).with_name("stream_selective_checkpoint.py")
_spec = importlib.util.spec_from_file_location("projection_scoped_legacy", _path)
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)


class ProjectionScopedCheckpoint(_legacy.SelectiveCheckpoint):
    """Reuse the original inspector/restoration contract with narrower dispatch."""

    def __init__(self, model: Any, policy: str) -> None:
        import torch
        from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

        from soup_cli.utils.layer_stream_runtime import StreamedDecoderLayer

        if policy != "qkvo":
            raise ValueError("projection-scoped benchmark requires qkvo")
        layers = []
        model.apply(lambda module: layers.append(module)
                    if isinstance(module, StreamedDecoderLayer) else None)
        if not layers or any(not layer.recompute or not layer.use_checkpoint for layer in layers):
            raise ValueError("selected layers must use checkpoint recompute")
        projections = []
        for layer in layers:
            for name in _legacy.POLICIES[policy]:
                projection = getattr(getattr(layer.inner, "self_attn", None), name, None)
                if projection is None:
                    raise ValueError(f"streamed layer {layer.idx} has no self_attn.{name}")
                projections.append(getattr(projection, "base_layer", projection))
        self.policy = policy
        self.layer_count = len(layers)
        self.counts = {"saved_forward": 0, "recompute_projection_calls": 0}
        self.ops = {}
        self._forwards = []
        self._modes = []
        self._mode = contextvars.ContextVar("soup2162_projection_mode", default=None)
        self._recomputing = contextvars.ContextVar("soup2162_projection_recomputing", default=False)
        selected_ops = {torch.ops.aten.mm.default, torch.ops.aten.addmm.default}
        try:
            selected_ops.add(torch.ops.bitsandbytes.gemm_4bit.default)
        except AttributeError:
            pass

        def policy_fn(ctx: Any, op: Any, *args: Any, **kwargs: Any) -> Any:
            if op in selected_ops:
                if not ctx.is_recompute:
                    self.counts["saved_forward"] += 1
                    op_key = str(op)
                    self.ops[op_key] = self.ops.get(op_key, 0) + 1
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.PREFER_RECOMPUTE

        @contextmanager
        def bind(mode: Any, recomputing: bool) -> Any:
            mode_token = self._mode.set(mode)
            recompute_token = self._recomputing.set(recomputing)
            try:
                yield
            finally:
                self._recomputing.reset(recompute_token)
                self._mode.reset(mode_token)

        def contexts() -> Any:
            modes = create_selective_checkpoint_contexts(policy_fn)
            self._modes = [ref for ref in self._modes if ref() is not None]
            self._modes.append(weakref.ref(modes[1]))
            return bind(modes[0], False), bind(modes[1], True)

        def layer_forward(instance: Any, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
            from torch.utils.checkpoint import checkpoint

            if not instance.recompute or not instance.use_checkpoint:
                raise RuntimeError("selected layer recompute mode changed after installation")
            if torch.is_grad_enabled():
                return checkpoint(instance._body, hidden_states, *args,
                                  use_reentrant=False, context_fn=contexts, **kwargs)
            return instance._body(hidden_states, *args, **kwargs)

        for projection in projections:
            self._patch(projection, self._scoped_forward(projection.forward))
        for layer in layers:
            self._patch(layer, layer_forward)

    def _scoped_forward(self, original: Any) -> Any:
        def forward(instance: Any, *args: Any, **kwargs: Any) -> Any:
            mode = self._mode.get()
            if mode is None:
                return original(*args, **kwargs)
            if self._recomputing.get():
                self.counts["recompute_projection_calls"] += 1
            with mode:
                return original(*args, **kwargs)

        return forward


def install(model: Any, policy: str) -> ProjectionScopedCheckpoint:
    return ProjectionScopedCheckpoint(model, policy)
