"""Device placement and layer replicas for single-process split L2L windows."""

import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

PROFILE_ENV = "SOUP_L2L_PROFILE"
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

    def weights(self, layer: Any, moved: dict[str, Any]) -> dict[str, Any]:
        """Rebuild NF4 views and add resident tensors to the copied weights."""
        from soup_cli.utils.layer_stream_runtime import rebuild_params4bit

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


@dataclass
class WeightCopy:
    buffers: dict[str, Any]
    source_done: Any = None
    destination_done: Any = None

    def wait(self, source: Any, destination: Any) -> None:
        if self.source_done is None:
            return
        import torch

        # The pool's next refill waits on the primary compute stream.
        torch.cuda.current_stream(source).wait_event(self.source_done)
        consumer = torch.cuda.current_stream(destination)
        consumer.wait_event(self.destination_done)
        for tensor in self.buffers.values():
            tensor.record_stream(consumer)


class SplitTransfers:
    """Copy on separate CUDA streams, with event handoffs to the compute streams."""

    def __init__(self, primary: Any, replicas: list[Any], *, overlap: bool = False) -> None:
        import torch

        self.primary = torch.device(primary)
        devices = [self.primary, *(torch.device(device) for device in replicas)]
        self.enabled = overlap and bool(replicas) and all(d.type == "cuda" for d in devices)
        self._streams = {
            (purpose, device): torch.cuda.Stream(device)
            for device in devices
            for purpose in ("weights", "activations")
        } if self.enabled else {}

    @contextmanager
    def window(self) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        import torch

        main = torch.cuda.current_stream(self.primary)
        activations = self._streams["activations", self.primary]
        activations.wait_stream(main)
        try:
            yield
        finally:
            # A retry may reuse the pool or store after a partially queued window.
            main.wait_stream(self._streams["weights", self.primary])
            main.wait_stream(activations)

    def activation_stream(self) -> Any:
        if not self.enabled:
            return nullcontext()
        import torch

        return torch.cuda.stream(self._streams["activations", self.primary])

    def weights(
        self, buffers: dict[str, Any], destination: Any, ready: Any
    ) -> WeightCopy:
        if not self.enabled or not buffers:
            return WeightCopy({
                name: tensor.to(destination, non_blocking=True, copy=True)
                for name, tensor in buffers.items()
            })
        import torch

        source_stream = self._streams["weights", self.primary]
        destination_stream = self._streams["weights", destination]
        source_stream.wait_event(ready)
        with torch.cuda.stream(source_stream), torch.cuda.stream(destination_stream):
            moved = {
                name: tensor.to(destination, non_blocking=True, copy=True)
                for name, tensor in buffers.items()
            }
            source_done = torch.cuda.Event()
            source_done.record(source_stream)
            destination_done = torch.cuda.Event()
            destination_done.record(destination_stream)
        for tensor in buffers.values():
            tensor.record_stream(source_stream)
        return WeightCopy(moved, source_done, destination_done)

    def activation(self, tensor: Any, destination: Any) -> Any:
        if not self.enabled or tensor.device == destination:
            return tensor.to(destination, non_blocking=True)
        import torch

        source_stream = self._streams["activations", tensor.device]
        destination_stream = self._streams["activations", destination]
        source_stream.wait_stream(torch.cuda.current_stream(tensor.device))
        with torch.cuda.stream(source_stream), torch.cuda.stream(destination_stream):
            moved = tensor.to(destination, non_blocking=True)
        consumer = torch.cuda.current_stream(destination)
        consumer.wait_stream(destination_stream)
        tensor.record_stream(source_stream)
        moved.record_stream(consumer)
        return moved


class SplitProfile:
    """Append one record per CUDA split window, timing compute streams and host calls."""

    def __init__(self, devices: list[Any]) -> None:
        import torch

        self.path = os.environ.get(PROFILE_ENV)
        self.devices = tuple(torch.device(device) for device in devices)
        self.enabled = bool(self.path) and bool(self.devices) and all(
            device.type == "cuda" for device in self.devices
        )
        self._spans: list[tuple[Any, str, Any, Any]] = []
        self._host: dict[str, float] = {}

    def _event(self, device: Any) -> Any:
        import torch

        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream(device))
        return event

    def start(self) -> None:
        if self.enabled:
            self._spans.clear()
            self._host.clear()
            self._start = time.perf_counter()
            self._timestamp = time.time()
            self._begin = {device: self._event(device) for device in self.devices}

    def span(self, label: str, *devices: Any) -> Any:
        return self._span(label, devices or self.devices) if self.enabled else nullcontext()

    @contextmanager
    def _span(self, label: str, devices: tuple[Any, ...]) -> Iterator[None]:
        import torch

        started = time.perf_counter()
        opened = [(torch.device(device), self._event(device)) for device in devices]
        try:
            yield
        finally:
            for device, begin in opened:
                self._spans.append((device, label, begin, self._event(device)))
            self._host[label] = self._host.get(label, 0.0) + (time.perf_counter() - started) * 1000

    def finish(self, k: int) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        import torch

        end = {device: self._event(device) for device in self.devices}
        for device in self.devices:
            torch.cuda.synchronize(device)
        busy: dict[Any, dict[str, float]] = {device: {} for device in self.devices}
        for device, label, begin, stop in self._spans:
            phases = busy[device]
            phases[label] = phases.get(label, 0.0) + begin.elapsed_time(stop)
        per_device = {}
        for device in self.devices:
            wall = self._begin[device].elapsed_time(end[device])
            per_device[str(device)] = {
                "wall_ms": wall,
                "busy_ms": busy[device],
                "idle_ms": wall - sum(busy[device].values()),
            }
        record = {
            "time": self._timestamp,
            "host_s": time.perf_counter() - self._start,
            "k": k,
            "devices": per_device,
            "host_ms": dict(self._host),
        }
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        return record
