"""Keep a frozen input embedding resident on the CPU during SFT."""

from __future__ import annotations

from functools import wraps
from inspect import signature
from typing import Any

from rich.console import Console

console = Console()


def embedding_on_host(model: Any) -> int:
    """Move an untied frozen input embedding to the CPU and return the CUDA bytes freed."""
    import torch

    embedding = model.get_input_embeddings()
    if embedding is not None and any(param.requires_grad for param in embedding.parameters()):
        raise ValueError("training.embed_on_host requires a frozen input embedding")
    if not isinstance(embedding, torch.nn.Embedding):
        raise ValueError("training.embed_on_host requires an nn.Embedding input module")

    weight = embedding.weight
    devices = set((getattr(model, "hf_device_map", None) or {}).values())
    if (
        weight.device.type not in ("cpu", "cuda")
        or hasattr(embedding, "_hf_hook")
        or "disk" in devices
        or len(devices) > 1
    ):
        raise ValueError("training.embed_on_host requires a resident embedding on a single device")

    head_weight = getattr(model.get_output_embeddings(), "weight", None)
    if head_weight is weight or (
        isinstance(head_weight, torch.Tensor)
        and head_weight.device == weight.device
        and weight.numel()
        and head_weight.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()
    ):
        raise ValueError("training.embed_on_host requires an untied input embedding")
    if getattr(embedding, "_soup_embed_on_host", False):
        return 0

    lookup = embedding.forward
    input_name = next(iter(signature(lookup).parameters))
    freed = 0
    if weight.is_cuda:
        freed = weight.nbytes
        host = torch.empty(weight.shape, dtype=weight.dtype, device="cpu", pin_memory=True)
        weight.data = host.copy_(weight.detach())
        embedding.cpu()  # Also move buffers used by embedding subclasses.

    warned = False

    @wraps(lookup)
    def forward(*args: Any, **kwargs: Any) -> Any:
        nonlocal warned
        device = embedding.weight.device
        if device.type != "cpu" and not warned:
            console.print(f"[yellow]Host input embedding moved to {device}[/]")
            warned = True
        ids = args[0] if args else kwargs[input_name]
        if ids.device == device:
            return lookup(*args, **kwargs)
        local_ids = ids.to(device)
        if args:
            rows = lookup(local_ids, *args[1:], **kwargs)
        else:
            rows = lookup(**{**kwargs, input_name: local_ids})
        if device.type == "cpu" and ids.is_cuda:
            rows = rows.pin_memory()
        return rows.to(ids.device, non_blocking=True)

    def keep_on_host(fn: Any, recurse: bool = True) -> Any:
        # Parent Module.to() and .cuda() use _apply to move parameters and buffers.
        return embedding

    embedding.forward = forward
    embedding._apply = keep_on_host
    embedding._soup_embed_on_host = True
    if freed:
        torch.cuda.empty_cache()
    return freed
