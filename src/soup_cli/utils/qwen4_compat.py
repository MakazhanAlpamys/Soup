"""Narrow runtime compatibility fixes for the Transformers Qwen4-Exp decoder."""

from __future__ import annotations

from functools import update_wrapper
from types import FunctionType, MethodType
from typing import Any

_INDEXER_CLASS = "Qwen4ExpTextQSAIndexer"
_INDEXER_MODULE_PREFIX = "transformers.models.qwen4_exp."
_PATCH_MARKER = "_soup_qwen4_long_scatter_indices"


class _LongIndexTorchProxy:
    """Delegate to Torch while exposing ``int32`` as ``int64`` to one function."""

    def __init__(self, torch_module: Any) -> None:
        self._torch = torch_module

    @property
    def int32(self) -> Any:
        return self._torch.int64

    def __getattr__(self, name: str) -> Any:
        return getattr(self._torch, name)


def _model_types(model: Any) -> set[Any]:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    return {
        getattr(config, "model_type", None),
        getattr(text_config, "model_type", None),
    }


def _supports_int32_scatter(torch_module: Any, device: Any) -> bool:
    """Probe the live operator instead of guessing from a Torch version."""
    target = torch_module.zeros((1, 1), dtype=torch_module.bool, device=device)
    index = torch_module.zeros((1, 1), dtype=torch_module.int32, device=device)
    try:
        target.scatter(-1, index, True)
    except RuntimeError as exc:
        if "Expected dtype int64 for index" in str(exc):
            return False
        raise
    return True


def _clone_forward_with_long_indices(module: Any, torch_module: Any) -> None:
    """Bind the upstream forward to one instance with only integer indices widened."""
    bound_forward = module.forward
    original = getattr(bound_forward, "__func__", None)
    if original is None:
        raise RuntimeError("Qwen4-Exp QSA indexer has no patchable Python forward")
    if getattr(original, _PATCH_MARKER, False):
        return
    if original.__globals__.get("torch") is not torch_module:
        raise RuntimeError("Qwen4-Exp QSA indexer does not use the expected Torch module")

    patched_globals = dict(original.__globals__)
    patched_globals["torch"] = _LongIndexTorchProxy(torch_module)
    patched = FunctionType(
        original.__code__,
        patched_globals,
        name=original.__name__,
        argdefs=original.__defaults__,
        closure=original.__closure__,
    )
    patched.__kwdefaults__ = original.__kwdefaults__
    update_wrapper(patched, original)
    setattr(patched, _PATCH_MARKER, True)
    module.forward = MethodType(patched, module)


def apply_qwen4_exp_scatter_compat(model: Any) -> int:
    """Use long QSA scatter indices when the installed Torch requires them.

    Transformers 5.16.1 constructs Qwen4-Exp QSA indices as ``int32``. Torch
    releases before the live operator gained ``int32`` support reject the
    forward pass. Reusing the exact upstream code object with an instance-local
    Torch proxy changes only those integer index factories; it does not mutate
    Torch, the Transformers class, or other model instances.
    """
    if "qwen4_exp_text" not in _model_types(model):
        return 0

    import torch

    indexers = []
    for module in model.modules():
        module_type = type(module)
        if (
            module_type.__name__ != _INDEXER_CLASS
            or not module_type.__module__.startswith(_INDEXER_MODULE_PREFIX)
        ):
            continue
        indexers.append(module)

    if not indexers:
        return 0

    probe_devices = {torch.device("cpu")}
    for module in indexers:
        parameter = next(module.parameters(), None)
        if parameter is not None and parameter.device.type != "meta":
            probe_devices.add(parameter.device)
    if all(_supports_int32_scatter(torch, device) for device in probe_devices):
        return 0

    patched = 0
    already_patched = 0
    for module in indexers:
        forward_function = getattr(getattr(module, "forward", None), "__func__", None)
        if getattr(forward_function, _PATCH_MARKER, False):
            already_patched += 1
            continue
        _clone_forward_with_long_indices(module, torch)
        patched += 1

    if patched == 0 and already_patched == 0:
        raise RuntimeError(
            "Qwen4-Exp requires long QSA scatter indices on this Torch runtime, "
            "but Soup could not locate the compatible Transformers indexer"
        )
    return patched
