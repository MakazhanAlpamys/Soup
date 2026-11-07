"""Adapter-gradient averaging across data-parallel ranks, for layer streaming (C2).

A streamed model keeps its frozen base on ``meta`` and injects it per call, so
``DistributedDataParallel`` cannot wrap it: the only real, trainable tensors are the
LoRA adapters. Data parallelism therefore reduces to one operation, an average of the
adapters' ``.grad`` across ranks between ``backward()`` and ``optimizer.step()``.
:class:`AdapterGradSync` is that operation and nothing else.

How one :meth:`AdapterGradSync.sync` call works:

* the trainable parameters' gradients are flattened, in ``named_parameters()`` order,
  into ONE fp32 buffer. A parameter whose ``grad`` is ``None`` on this rank is written
  as zeros, so every rank sends the same number of elements and the same slot layout
  (a rank-dependent hole would hang or corrupt the collective);
* the buffer is all-reduced with ``SUM`` in chunks of at most ``chunk_bytes`` (bounds
  what the backend stages), then divided by the world size. ``ReduceOp.AVG`` is NCCL
  only, and gloo (the CPU test backend) refuses it, so the mean is SUM then divide;
* the result is copied back into each ``.grad`` in place (a ``.grad`` that was ``None``
  is created: that parameter did receive the other ranks' gradient).

The mean is the right scale only together with HF's own loss scaling
(``average_tokens_across_devices``): a SUM would make the update world-size times too
large. Every rank holds a bitwise identical reduced buffer after the all-reduce (the
backends broadcast the final value), so ranks that start equal and run a deterministic
optimizer stay bitwise equal.

Torch is imported inside the functions: ``soup`` stays importable without it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple, Union

#: Upper bound on the elements one ``all_reduce`` call carries, in bytes (fp32 wire).
DEFAULT_CHUNK_BYTES = 64 * 1024 * 1024

#: The wire format is fp32 whatever the parameter dtype is.
_ELEMENT_BYTES = 4

NamedParams = List[Tuple[str, Any]]


@dataclass(frozen=True)
class SyncStats:
    """What one :meth:`AdapterGradSync.sync` call moved.

    ``bytes`` is the payload one rank contributed (``4 * n_trainable_elements``), not
    multiplied by the world size and not the on-the-wire total. A call that did nothing
    (no process group, or a world of one) reports all zeros.
    """

    bytes: int = 0
    seconds: float = 0.0
    n_params: int = 0
    n_chunks: int = 0


def validate_chunk_bytes(chunk_bytes: Any) -> int:
    """Return ``chunk_bytes`` if it is an int that holds at least one fp32 element."""
    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int):
        raise ValueError(
            f"chunk_bytes must be an int number of bytes, got {type(chunk_bytes).__name__} "
            f"({chunk_bytes!r})"
        )
    if chunk_bytes < _ELEMENT_BYTES:
        raise ValueError(
            f"chunk_bytes must be at least {_ELEMENT_BYTES} (one fp32 element), got {chunk_bytes}"
        )
    return chunk_bytes


def collect_trainable(source: Union[Any, Iterable[Tuple[str, Any]]]) -> NamedParams:
    """The ``(name, parameter)`` pairs that require grad, in ``named_parameters()`` order.

    ``source`` is a module or an iterable of ``(name, parameter)`` pairs. Refuses what
    the flat fp32 buffer cannot carry: no trainable parameter at all, mixed dtypes,
    or parameters on more than one device (or on ``meta``).
    """
    named = source.named_parameters() if hasattr(source, "named_parameters") else source
    params: NamedParams = [(name, param) for name, param in named if param.requires_grad]
    if not params:
        raise ValueError(
            "AdapterGradSync found no trainable parameters (requires_grad=True) to synchronise"
        )
    first_name, first = params[0]
    for name, param in params[1:]:
        if param.dtype != first.dtype:
            raise ValueError(
                "AdapterGradSync needs one dtype across the trainable parameters, but "
                f"'{first_name}' is {first.dtype} and '{name}' is {param.dtype}"
            )
        if param.device != first.device:
            raise ValueError(
                "AdapterGradSync needs the trainable parameters on one device, but "
                f"'{first_name}' is on {first.device} and '{name}' is on {param.device}"
            )
    if first.device.type == "meta":
        raise ValueError(
            f"AdapterGradSync cannot synchronise parameters on the meta device ('{first_name}')"
        )
    return params


def flatten_grads(params: NamedParams, out: Optional[Any] = None) -> Any:
    """Copy every parameter's gradient into one fp32 vector; ``grad is None`` becomes zeros.

    ``out`` is an optional buffer to fill (fp32, exactly the total element count): it
    is overwritten in full, whatever it held before.
    """
    import torch

    total = sum(param.numel() for _name, param in params)
    if out is None:
        out = torch.empty(total, dtype=torch.float32, device=params[0][1].device)
    elif out.dtype != torch.float32 or out.numel() != total:
        raise ValueError(
            f"flatten buffer must be {total} fp32 elements, got {out.numel()} {out.dtype}"
        )
    offset = 0
    with torch.no_grad():
        for name, param in params:
            count = param.numel()
            slot = out[offset : offset + count]
            grad = param.grad
            if grad is None:
                slot.zero_()
            else:
                if grad.numel() != count:
                    raise ValueError(
                        f"gradient of '{name}' has {grad.numel()} elements, "
                        f"the parameter has {count}"
                    )
                slot.copy_(grad.reshape(-1))
            offset += count
    return out


def unflatten_grads(params: NamedParams, flat: Any) -> None:
    """Write ``flat`` back into the ``.grad`` tensors, in place; create the missing ones."""
    import torch

    total = sum(param.numel() for _name, param in params)
    if flat.numel() != total:
        raise ValueError(f"flat gradient buffer has {flat.numel()} elements, expected {total}")
    offset = 0
    with torch.no_grad():
        for _name, param in params:
            count = param.numel()
            piece = flat[offset : offset + count].view(param.shape)
            if param.grad is None:
                param.grad = piece.to(dtype=param.dtype, copy=True)
            else:
                param.grad.copy_(piece)
            offset += count


class AdapterGradSync:
    """Average the trainable parameters' gradients across the ranks of a process group.

    Build it once from the model after the adapters exist, call :meth:`sync` after
    ``loss.backward()`` and BEFORE gradient clipping and ``optimizer.step()``: clipping
    must see the averaged gradient, not each rank's own.
    """

    def __init__(
        self,
        source: Union[Any, Iterable[Tuple[str, Any]]],
        *,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    ) -> None:
        self.chunk_bytes = validate_chunk_bytes(chunk_bytes)
        self._params: NamedParams = collect_trainable(source)
        self.n_params = len(self._params)
        self.n_elements = sum(param.numel() for _name, param in self._params)

    @property
    def payload_bytes(self) -> int:
        """Bytes one rank contributes per sync (fp32 wire): ``4 * n_elements``."""
        return _ELEMENT_BYTES * self.n_elements

    def sync(self, group: Optional[Any] = None) -> SyncStats:
        """All-reduce the gradients (mean over ranks) in place; return what was moved.

        A no-op returning zero stats when ``torch.distributed`` is unavailable or not
        initialised, or when the group has a single rank.
        """
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return SyncStats()
        world_size = dist.get_world_size(group)
        if world_size <= 1:
            return SyncStats()

        start = time.perf_counter()
        flat = flatten_grads(self._params)
        step = self.chunk_bytes // _ELEMENT_BYTES
        n_chunks = 0
        for begin in range(0, flat.numel(), step):
            dist.all_reduce(flat[begin : begin + step], op=dist.ReduceOp.SUM, group=group)
            n_chunks += 1
        flat.div_(world_size)
        unflatten_grads(self._params, flat)
        return SyncStats(
            bytes=self.payload_bytes,
            seconds=time.perf_counter() - start,
            n_params=self.n_params,
            n_chunks=n_chunks,
        )
