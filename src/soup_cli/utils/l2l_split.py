"""Device placement and layer replicas for single-process split L2L windows."""

from contextlib import nullcontext
from typing import Any

ROUTED_EXPERTS_REFUSAL = (
    "training.l2l_split_gpus does not support routed experts (dense SFT only)."
)
DROPOUT_REFUSAL = (
    "training.l2l_split_gpus does not support model dropout in SFT. "
    "Set training.lora.dropout and model attention dropout to 0."
)


def split_devices(device: Any, count: int) -> tuple[str, ...]:
    """Validate the CUDA selection and return the devices after the primary."""
    if count == 1:
        return ()

    import torch

    primary = torch.device(device)
    if primary.type != "cuda":
        raise ValueError(
            f"training.l2l_split_gpus={count} requires CUDA for SFT; got device={str(primary)!r}."
        )
    first = primary.index if primary.index is not None else 0
    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if first + count > visible:
        raise ValueError(
            f"training.l2l_split_gpus={count} needs {count} CUDA devices starting at cuda:{first}; "
            f"{visible} CUDA devices are visible. Select the cards with CUDA_VISIBLE_DEVICES."
        )
    return tuple(f"cuda:{index}" for index in range(first + 1, first + count))


def has_dropout(model: Any) -> bool:
    """Check for LoRA dropout and model attention dropout."""
    import torch

    config = getattr(model, "config", None)
    if float(getattr(config, "attention_dropout", 0.0) or 0.0) != 0:
        return True
    return any(
        (isinstance(module, torch.nn.Dropout) and module.p > 0)
        or float(getattr(module, "attention_dropout", 0.0) or 0.0) != 0
        for module in model.modules()
    )


def refuse_unsupported_split_model(causal: Any, owner: Any, runtime: Any) -> None:
    """Check the model and runtime before allocating replicas."""
    if getattr(runtime, "expert_readers", ()) or getattr(runtime, "expert_pool", None) is not None:
        raise ValueError(ROUTED_EXPERTS_REFUSAL)
    if getattr(owner, "_soup_prefix_cache", None) is not None:
        raise ValueError("training.l2l_split_gpus cannot be combined with a frozen-prefix cache.")
    if any(param.requires_grad for param in owner.embed_tokens.parameters()):
        raise ValueError("training.l2l_split_gpus requires a frozen embedding for SFT.")
    if has_dropout(causal):
        raise ValueError(DROPOUT_REFUSAL)


def device_context(device: Any) -> Any:
    """Set the current CUDA device for kernels and streams that need it."""
    import torch

    return torch.cuda.device(device) if device.type == "cuda" else nullcontext()


def to_device(value: Any, device: Any) -> Any:
    """Move the tensors in captured decoder arguments to a replica's device."""
    import torch

    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(to_device(item, device) for item in value)
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


class DeviceReplica:
    """Resident decoder tensors on an extra device, with one streamed layer at a time."""

    def __init__(self, layers: list[Any], device: Any) -> None:
        import torch

        self.device = torch.device(device)
        self.resident: list[dict[str, tuple[Any, Any]]] = []
        self._codes: dict[int, dict[str, Any]] = {}
        for layer in layers:
            inner = layer.inner
            self.resident.append({
                name: (tensor, torch.empty_like(tensor, device=self.device))
                for name, tensor in (*inner.named_parameters(), *inner.named_buffers())
                if not tensor.is_meta and name not in layer.name_map
            })

    def refresh(self) -> None:
        """Refresh resident tensors and make the trainable copies leaf tensors."""
        import torch

        with torch.no_grad():
            for layer in self.resident:
                for tensor, replica in layer.values():
                    replica.copy_(tensor, non_blocking=True)
                    replica.requires_grad_(tensor.requires_grad)
                    replica.grad = None

    def merge_grads(self) -> None:
        """Add replica gradients before the optimizer sees the primary gradients."""
        import torch

        with torch.no_grad():
            for layer in self.resident:
                for tensor, replica in layer.values():
                    if replica.grad is not None:
                        grad = replica.grad.to(tensor.device, non_blocking=True)
                        if tensor.grad is None:
                            tensor.grad = grad
                        else:
                            tensor.grad.add_(grad)
                        replica.grad = None

    def weights(self, layer: Any, buffers: dict[str, Any]) -> dict[str, Any]:
        """Copy the live pool tensors and rebuild NF4 views on the replica."""
        from soup_cli.utils.layer_stream_runtime import rebuild_params4bit

        keys = buffers.keys() if layer.quant_specs else layer.name_map.values()
        moved = {
            key: buffers[key].to(self.device, non_blocking=True, copy=True)
            for key in dict.fromkeys(keys)
        }
        codes = {}
        if layer.quant_specs:
            key = id(layer.codes)
            if key not in self._codes:
                self._codes[key] = to_device(layer.codes, self.device)
            codes = self._codes[key]
        weights = {
            name: (
                rebuild_params4bit(key, moved, layer.quant_specs[key], codes)
                if key in layer.quant_specs else moved[key]
            )
            for name, key in layer.name_map.items()
        }
        weights.update({name: replica for name, (_, replica) in self.resident[layer.idx].items()})
        return weights

    def forward(
        self, layer: Any, weights: dict[str, Any], hidden: Any, args: tuple, kwargs: dict
    ) -> Any:
        from torch.func import functional_call

        with device_context(self.device):
            out = functional_call(layer.inner, weights, (hidden, *args), kwargs)
        return out[0] if isinstance(out, tuple) else out
