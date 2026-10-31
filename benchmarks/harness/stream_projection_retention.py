"""Benchmark prototype for QKVO on layers left recomputing by native auto.

The untimed observer delegates every planner calculation to the shipped code.
The subset installer reuses #2161's selective checkpoint code on explicit layers;
it never replaces the forwards or budget of fully retained layers. This is not a
public training option. The observer is for a single-threaded benchmark process.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any
from unittest.mock import patch

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


def projection_bytes(layer: Any, *, batch: int, seq: int, element_bytes: int) -> int:
    if any(type(value) is not int or value <= 0 for value in (batch, seq, element_bytes)):
        raise ValueError("batch, seq and element_bytes must be positive integers")
    widths = []
    for name in PROJECTIONS:
        projection = getattr(layer.inner.self_attn, name)
        base = getattr(projection, "base_layer", projection)
        width = base.out_features
        if type(width) is not int or width <= 0:
            raise ValueError(f"invalid output width for {name}")
        widths.append(width)
    return batch * seq * element_bytes * sum(widths)


def observe_auto_plan(model: Any, **kwargs: Any) -> dict[str, Any]:
    from soup_cli.utils import layer_stream
    from soup_cli.utils.layer_stream_runtime import (
        plan_stream_recompute_auto,
        streamed_decoder_layers,
    )

    if any(not layer.recompute for layer in streamed_decoder_layers(model)):
        raise ValueError("native auto census requires an initial full-recompute model")
    record: dict[str, Any] = {"measured": False}
    original_charge = layer_stream.retained_layer_charge
    original_budget = layer_stream.stream_recompute_budget

    def charge(allocated: int, reserved_peak: int, allocated_peak: int) -> int:
        result = original_charge(allocated, reserved_peak, allocated_peak)
        record.update(
            full_layer_bytes=allocated,
            full_layer_charge=result,
            peak_reserved=reserved_peak,
            peak_allocated=allocated_peak,
        )
        return result

    def budget(**values: Any) -> Any:
        result = original_budget(**values)
        record.update(
            cap_bytes=result.cap_bytes,
            cap_bound=result.bound,
            budget_bytes=result.budget_bytes,
            budget_inputs=values,
            measured=True,
        )
        return result

    with (
        patch.object(layer_stream, "retained_layer_charge", charge),
        patch.object(layer_stream, "stream_recompute_budget", budget),
    ):
        record["k"] = plan_stream_recompute_auto(model, **kwargs)
    record["retained_indices"] = [
        layer.idx for layer in streamed_decoder_layers(model) if not layer.recompute
    ]
    return record


def projection_plan(
    layers: list[Any],
    native: dict[str, Any],
    *,
    batch: int,
    seq: int,
    element_bytes: int,
) -> dict[str, Any]:
    from soup_cli.utils.layer_stream import plan_retained_layers, retained_layer_charge

    indices = [layer.idx for layer in layers]
    if len(set(indices)) != len(indices) or indices != sorted(indices):
        raise ValueError("native layers must have unique ordered indices")
    kept = [layer.idx for layer in layers if not layer.recompute]
    result = dict(
        retained_indices=kept,
        recomputed_indices=[layer.idx for layer in layers if layer.recompute],
        selected_indices=[],
        raw_bytes=0,
        charged_bytes=0,
        residual_bytes=None,
        reason="native auto probe unavailable",
    )
    if not native.get("measured"):
        return result
    k = native["k"]
    budget, full_charge = native["budget_bytes"], native["full_layer_charge"]
    if (
        type(k) is not int
        or full_charge <= 0
        or k != plan_retained_layers(budget, full_charge, len(layers))
        or kept != (indices[-k:] if k else [])
    ):
        raise ValueError("native retained-layer decision changed")
    residual = max(0, budget - k * full_charge)
    result["residual_bytes"] = residual
    if k == len(layers):
        result["reason"] = "all layers already retained"
        return result
    for layer in reversed(layers[: len(layers) - k]):
        raw = projection_bytes(layer, batch=batch, seq=seq, element_bytes=element_bytes)
        charged = retained_layer_charge(raw, native["peak_reserved"], native["peak_allocated"])
        if result["charged_bytes"] + charged > residual:
            break
        result["selected_indices"].append(layer.idx)
        result["raw_bytes"] += raw
        result["charged_bytes"] += charged
    result["selected_indices"].reverse()
    result["reason"] = "planned" if result["selected_indices"] else "no projection layer fits"
    return result


def reallocated_projection_plan(
    layers: list[Any], native: dict[str, Any], *, batch: int, seq: int, element_bytes: int,
) -> dict[str, Any]:
    """Separate K-1 hypothesis; preserve the native census and charge one release."""
    from soup_cli.utils.layer_stream import retained_layer_charge

    if not native.get("measured"):
        raise ValueError("K-1 requires a measured native partial-retention plan")
    original = projection_plan(layers, native, batch=batch, seq=seq, element_bytes=element_bytes)
    k = native["k"]
    if not 0 < k < len(layers):
        raise ValueError("K-1 requires native partial retention")
    effective = k - 1
    indices = [layer.idx for layer in layers]
    residual = native["budget_bytes"] - effective * native["full_layer_charge"]
    if residual != original["residual_bytes"] + native["full_layer_charge"]:
        raise ValueError("freed retained-layer charge is inconsistent")
    result = dict(
        native_k=k, effective_k=effective,
        freed_full_layer_charge=native["full_layer_charge"],
        retained_indices=indices[-effective:] if effective else [],
        recomputed_indices=indices[:len(indices)-effective], selected_indices=[],
        raw_bytes=0, charged_bytes=0, residual_bytes=residual, reason="no projection layer fits",
    )
    for layer in reversed(layers[:len(layers)-effective]):
        raw = projection_bytes(layer, batch=batch, seq=seq, element_bytes=element_bytes)
        charged = retained_layer_charge(raw, native["peak_reserved"], native["peak_allocated"])
        if result["charged_bytes"] + charged > residual:
            break
        result["selected_indices"].append(layer.idx)
        result["raw_bytes"] += raw
        result["charged_bytes"] += charged
    result["selected_indices"].reverse()
    if result["selected_indices"]:
        result["reason"] = "kminus1-planned"
    return result


def install_subset(model: Any, indices: list[int], *, cache_scope: str = "layer") -> Any:
    from soup_cli.utils.layer_stream_runtime import streamed_decoder_layers

    if cache_scope not in ("layer", "projection"):
        raise ValueError("unknown selective cache scope")
    layers = streamed_decoder_layers(model)
    if any(type(index) is not int for index in indices) or len(indices) != len(set(indices)):
        raise ValueError("selected layer indices must be unique integers")
    by_index = {layer.idx: layer for layer in layers}
    if not set(indices).issubset(by_index):
        raise ValueError("unknown selected layer index")
    selected = [by_index[index] for index in indices]
    if any(not layer.recompute or not layer.use_checkpoint for layer in selected):
        raise ValueError("selected layers must use checkpoint recompute")
    if not selected:
        return None

    class Selection:
        def apply(self, visitor: Any) -> None:
            for layer in selected:
                visitor(layer)

    filename = ("stream_selective_checkpoint.py" if cache_scope == "layer"
                else "stream_projection_scoped.py")
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location("projection_subset_checkpoint", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    controller = helper.install(Selection(), "qkvo")

    def guard(original: Any) -> Any:
        def forward(instance: Any, *args: Any, **kwargs: Any) -> Any:
            if not instance.recompute or not instance.use_checkpoint:
                raise RuntimeError("selected layer recompute mode changed after installation")
            return original(*args, **kwargs)

        return forward

    for layer in selected:
        controller._patch(layer, guard(layer.forward))
    return controller
