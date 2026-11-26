"""Budgeted, resident FP8 storage for L2L boundary activations.

One FP8 payload slot per layer and micro-batch holds logical hidden vectors.
One ragged float32 slab holds their per-token absmax scales. No compressed spill
is supported. The numerical codec is the unchanged G4 implementation.

The host reservation includes requested storage, pinned allocator rounding, and
host codec scratch. CUDA codec scratch is reported separately, never charged to
host RAM. This is owned-tensor accounting, not a process RSS or CUDA cache limit.
"""

from __future__ import annotations

import math
import traceback
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from soup_cli.utils import activation_codec
from soup_cli.utils.host_memory_budget import HostMemoryBudget
from soup_cli.utils.l2l_activations import StoreStats

# Shared by the pure peak estimate and the production tile iterator.
_CODEC_TILE_ROWS = 256
_COMPUTE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}


def _shape(shape: Sequence[int]) -> tuple[int, ...]:
    result = tuple(shape)
    if not result or any(type(dim) is not int or dim < 1 for dim in result):
        raise ValueError("activation shapes must have positive integer dimensions")
    return result


def _shapes(mb_shapes: Sequence[tuple[int, ...]]) -> list[tuple[int, ...]]:
    result = [_shape(shape) for shape in mb_shapes]
    if not result:
        raise ValueError("FP8 activations need at least one micro-batch shape")
    hidden = result[0][-1]
    if any(shape[-1] != hidden for shape in result):
        raise ValueError("FP8 micro-batches must keep the same logical hidden dimension")
    return result


@dataclass(frozen=True)
class FP8ActivationAllocation:
    """Reserved host bytes and conservative one-operation scratch peaks.

    ``payload_bytes`` and ``scale_bytes`` are logical persistent storage requests.
    Each pinned payload allocation rounds up independently to a power of two;
    the scale slab rounds once. ``host_bytes`` includes that overhead and
    ``host_transient_bytes``, so it is exactly the store's lease amount.

    A codec operation handles at most 256 rows. The scratch bound allows eight
    float32 tile matrices and eight float32 row vectors, plus one full decoded
    result and one scalar flag. This covers input staging, the unchanged G4
    expressions, payload/scale transfers, and finite-check masks conservatively.
    CUDA reserves only a row-sized host finite mask plus its scalar reduction;
    all codec matrices and the decoded result belong to device scratch instead.

    The caller's source activation, caller-retained decoded results, Python/CUDA
    event objects, and allocator caching/framework workspaces are excluded. An
    output is counted while get constructs it, but becomes caller-owned on return.
    """

    host_bytes: int
    payload_bytes: int
    scale_bytes: int
    pinned_overhead_bytes: int
    host_transient_bytes: int
    device_transient_bytes: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def fp8_activation_allocation(
    *,
    n_layers: int,
    mb_shapes: Sequence[tuple[int, ...]],
    pin: bool,
    compute_dtype: str = "bfloat16",
    device_type: str = "cuda",
) -> FP8ActivationAllocation:
    """Pure allocation contract used unchanged by the planner and the store."""
    from soup_cli.utils.layer_stream_runtime import _next_power_of_two

    if type(n_layers) is not int or n_layers < 1:
        raise ValueError("n_layers must be a positive integer")
    if type(pin) is not bool:
        raise ValueError("pin must be a boolean")
    if compute_dtype not in _COMPUTE_BYTES:
        raise ValueError("FP8 compute dtype must be bfloat16, float16, or float32")
    if device_type not in ("cpu", "cuda"):
        raise ValueError("FP8 activation storage supports cpu or cuda devices")
    shapes = _shapes(mb_shapes)
    payload_requests = [math.prod(shape) for shape in shapes]
    rows = [activation_codec.rows_of(shape) for shape in shapes]
    payload_bytes = n_layers * sum(payload_requests)
    scale_bytes = n_layers * sum(rows) * activation_codec.SCALE_BYTES
    allocated = payload_bytes + scale_bytes
    if pin:
        allocated = n_layers * sum(_next_power_of_two(size) for size in payload_requests)
        allocated += _next_power_of_two(scale_bytes)
    pinned_overhead = allocated - payload_bytes - scale_bytes
    tile_rows = min(max(rows), _CODEC_TILE_ROWS)
    tile_elements = tile_rows * shapes[0][-1]
    output_bytes = max(payload_requests) * _COMPUTE_BYTES[compute_dtype]
    codec_scratch = 8 * 4 * tile_elements + 8 * 4 * tile_rows + output_bytes + 1
    host_scratch = codec_scratch if device_type == "cpu" else tile_rows + 1
    return FP8ActivationAllocation(
        host_bytes=allocated + host_scratch,
        payload_bytes=payload_bytes,
        scale_bytes=scale_bytes,
        pinned_overhead_bytes=pinned_overhead,
        host_transient_bytes=host_scratch,
        device_transient_bytes=codec_scratch if device_type == "cuda" else 0,
    )


class FP8ActivationStore:
    """ActivationStore-compatible resident FP8 slots with a shared host lease.

    ``chunk_shape`` bounds every slot, and ``mb_shapes`` can give each slot a
    smaller row capacity. Shapes may have different ranks, but the logical final
    hidden dimension must stay fixed. A short put restores in its original shape.

    Calls must be serialized by the caller, as with ActivationStore. CUDA calls
    may use different streams. One reusable event per touched slot protects host
    reads/overwrites; a stream change drains the previous operation so scratch
    peaks do not multiply across streams. end_step drains before inspecting host
    scales, and close drains before dropping storage or releasing its reservation.
    """

    def __init__(
        self,
        *,
        host_budget: HostMemoryBudget,
        n_layers: int,
        k: int,
        chunk_shape: tuple[int, ...],
        dtype: Any,
        device: Any,
        pin: bool,
        mb_shapes: Sequence[tuple[int, ...]] | None = None,
        spill_layers: int = 0,
        ring: int = 1,
    ) -> None:
        import torch

        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer")
        if type(ring) is not int or ring < 1:
            raise ValueError("ring must be a positive integer")
        if type(spill_layers) is not int or spill_layers != 0:
            raise ValueError(
                "compressed FP8 activation spill is not supported; spill_layers must be 0"
            )
        shape = _shape(chunk_shape)
        shapes = _shapes([shape] * k if mb_shapes is None else mb_shapes)
        if len(shapes) != k:
            raise ValueError(f"{len(shapes)} mb_shapes for k={k}")
        if any(item[-1] != shape[-1] for item in shapes):
            raise ValueError("mb_shapes must preserve chunk_shape's logical hidden dimension")
        if any(math.prod(item) > math.prod(shape) for item in shapes):
            raise ValueError("an mb_shape exceeds chunk_shape capacity")
        dtype_names = {
            torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32",
        }
        if dtype not in dtype_names:
            raise ValueError("FP8 compute dtype must be bfloat16, float16, or float32")
        self.dtype = dtype
        self.device = torch.device(device)
        self.is_cuda = self.device.type == "cuda"
        self.allocation = fp8_activation_allocation(
            n_layers=n_layers, mb_shapes=shapes, pin=pin,
            compute_dtype=dtype_names[dtype], device_type=self.device.type,
        )
        self.n_layers = n_layers
        self.k = k
        self.shape = shape
        self.mb_shapes = shapes
        self.hidden = shape[-1]
        self.capacity = math.prod(shape)
        self.spill_layers = 0
        self.spill_path = None
        self.payload_chunk_bytes = self.capacity
        self.scale_chunk_bytes = activation_codec.rows_of(shape) * activation_codec.SCALE_BYTES
        self.chunk_bytes = self.payload_chunk_bytes + self.scale_chunk_bytes
        self._pin = pin
        self._closed = False
        self._closing = False
        self._cuda_used = False
        self._payload: dict[tuple[int, int], Any] = {}
        self._scales: Any = None
        self._shapes: dict[tuple[int, int], tuple[int, ...]] = {}
        self._events: dict[tuple[int, int], Any] = {}
        self._nonfinite_keys: set[tuple[int, int]] = set()
        self._last_event: Any = None
        self._last_stream: int | None = None
        self._row_capacities = [activation_codec.rows_of(item) for item in shapes]
        self._scale_offsets = [0]
        for rows in self._row_capacities:
            self._scale_offsets.append(self._scale_offsets[-1] + rows)
        self._rows_per_layer = self._scale_offsets[-1]
        self.stats = StoreStats(
            chunk_bytes=self.chunk_bytes, resident_chunks=n_layers * k,
            spilled_chunks=0, spill_layers=0, pinned=pin, direct=None,
        )
        # No tensor allocation, including scalar dtype probes, precedes this lease.
        self._lease = host_budget.reserve("fp8.activations", self.allocation.host_bytes)
        try:
            for layer in range(n_layers):
                for mb, capacity_shape in enumerate(shapes):
                    self._payload[(layer, mb)] = torch.empty(
                        math.prod(capacity_shape), dtype=torch.float8_e4m3fn,
                        device="cpu", pin_memory=pin,
                    )
            self._scales = torch.empty(
                n_layers * self._rows_per_layer, dtype=torch.float32,
                device="cpu", pin_memory=pin,
            )
        except BaseException as error:
            traceback.clear_frames(error.__traceback__)
            self.close()
            raise

    def _raise_if_closed(self) -> None:
        if self._closed or self._closing:
            raise RuntimeError("FP8 activation store is closed or closing")

    def _key(self, layer: int, mb: int) -> tuple[int, int]:
        if type(layer) is not int or type(mb) is not int:
            raise ValueError("activation layer and micro-batch indices must be integers")
        if not 0 <= layer < self.n_layers or not 0 <= mb < self.k:
            raise IndexError("activation layer or micro-batch index exceeds reserved capacity")
        return layer, mb

    def _fit(self, mb: int, tensor: Any) -> tuple[int, ...]:
        shape = _shape(tensor.shape)
        if shape[-1] != self.hidden:
            raise ValueError("an activation changed the logical hidden dimension")
        if activation_codec.rows_of(shape) > self._row_capacities[mb]:
            raise ValueError(f"an activation of shape {shape} exceeds micro-batch {mb} capacity")
        if tensor.dtype != self.dtype:
            raise ValueError("an activation dtype differs from the reserved compute dtype")
        if tensor.device.type != self.device.type or (
            self.device.index is not None and tensor.device != self.device
        ):
            raise ValueError("an activation is not on the reserved compute device")
        if self.is_cuda and self.device.index is None:
            self.device = tensor.device
        return shape

    def _scale_view(self, key: tuple[int, int], rows: int) -> Any:
        layer, mb = key
        start = layer * self._rows_per_layer + self._scale_offsets[mb]
        return self._scales[start : start + rows]

    def _wait_slot(self, key: tuple[int, int]) -> None:
        if not self.is_cuda:
            return
        import torch

        stream = torch.cuda.current_stream(self.device)
        if self._last_event is not None and self._last_stream != stream.cuda_stream:
            # A second stream must not overlap independently allocated codec
            # scratch with the previous call's scratch peak.
            self._last_event.synchronize()

        event = self._events.get(key)
        if event is not None:
            if self._pin:
                torch.cuda.current_stream(self.device).wait_event(event)
            else:
                # Pageable copies may stage host data before a CUDA stream wait.
                event.synchronize()
        self._cuda_used = True

    def _record_slot(self, key: tuple[int, int]) -> None:
        if not self.is_cuda:
            return
        import torch

        try:
            event = self._events.get(key)
            if event is None:
                event = torch.cuda.Event()
                self._events[key] = event
            event.record(torch.cuda.current_stream(self.device))
            self._last_event = event
            self._last_stream = torch.cuda.current_stream(self.device).cuda_stream
        except BaseException:
            # Without a reliable completion record, only close's full drain is safe.
            self._closing = True
            raise

    def _row_tiles(self, tensor: Any, rows: int) -> Iterator[tuple[int, Any]]:
        """View contiguous row runs, or gather into one bounded non-contiguous tile.

        The generator owns no host slot views. Closing it also clears a staging
        tensor if an encode exception interrupts the caller's loop.
        """
        import torch

        try:
            flat = tensor.view(rows, self.hidden)
        except RuntimeError:
            flat = None
        staging = None
        try:
            if flat is not None:
                for start in range(0, rows, _CODEC_TILE_ROWS):
                    yield start, flat[start : start + _CODEC_TILE_ROWS]
                return
            staging = torch.empty(
                (min(rows, _CODEC_TILE_ROWS), self.hidden), dtype=self.dtype, device=self.device,
            )
            prefix_shape = tuple(tensor.shape[:-1])
            for start in range(0, rows, _CODEC_TILE_ROWS):
                count = min(rows - start, _CODEC_TILE_ROWS)
                for offset in range(count):
                    remainder = start + offset
                    indices = []
                    for dimension in reversed(prefix_shape):
                        remainder, index = divmod(remainder, dimension)
                        indices.append(index)
                    staging[offset].copy_(tensor[tuple(reversed(indices))])
                yield start, staging[:count]
        finally:
            flat = staging = tensor = None

    def put(self, layer: int, mb: int, tensor: Any) -> None:
        """Compress without changing the forward input; defer finite refusal to end_step."""
        import torch

        self._raise_if_closed()
        key = self._key(layer, mb)
        shape = self._fit(mb, tensor)
        rows = activation_codec.rows_of(shape)
        self._wait_slot(key)
        if key in self._shapes:
            # A repeated put may replace a non-finite scale before end_step sees
            # it. Capture that rare path first, without syncing ordinary puts.
            if self.is_cuda:
                self._events[key].synchronize()
            if not self._finite_scales(key, self._shapes[key]):
                self._nonfinite_keys.add(key)
        self._shapes.pop(key, None)
        host_payload = host_scale = payload = scale = values = None
        tiles = None
        try:
            with torch.no_grad():
                host_payload = self._payload[key].view(-1, self.hidden)
                host_scale = self._scale_view(key, rows)
                tiles = self._row_tiles(tensor, rows)
                for start, values in tiles:
                    payload, scale = activation_codec.encode(
                        values, activation_codec.FP8, check_finite=False,
                    )
                    count = values.shape[0]
                    host_payload[start : start + count].copy_(
                        payload, non_blocking=self.is_cuda and self._pin,
                    )
                    host_scale[start : start + count].copy_(
                        scale, non_blocking=self.is_cuda and self._pin,
                    )
                    payload = scale = None
        except BaseException as error:
            # The caller may keep this exception after close. A codec traceback
            # must not retain tile tensors or host views past the reservation.
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            if tiles is not None:
                tiles.close()
            # Exception tracebacks can outlive close. Do not let this frame
            # retain views into storage whose lease close will release.
            host_payload = host_scale = payload = scale = values = tiles = None
            self._record_slot(key)
        self._shapes[key] = shape

    def get(self, layer: int, mb: int) -> Any:
        """Return an independent compute-dtype tensor in this step's put shape."""
        import torch

        self._raise_if_closed()
        key = self._key(layer, mb)
        if key not in self._shapes:
            raise KeyError(f"activation {key} was not put this step")
        shape = self._shapes[key]
        rows = activation_codec.rows_of(shape)
        self._wait_slot(key)
        out = flat_out = host_payload = host_scale = payload = scale = restored = None
        try:
            with torch.no_grad():
                out = torch.empty(shape, dtype=self.dtype, device=self.device)
                flat_out = out.view(rows, self.hidden)
                host_payload = self._payload[key].view(-1, self.hidden)
                host_scale = self._scale_view(key, rows)
                for start in range(0, rows, _CODEC_TILE_ROWS):
                    stop = min(rows, start + _CODEC_TILE_ROWS)
                    payload = host_payload[start:stop].to(
                        self.device, non_blocking=self.is_cuda and self._pin, copy=True,
                    )
                    scale = host_scale[start:stop].to(
                        self.device, non_blocking=self.is_cuda and self._pin, copy=True,
                    )
                    restored = activation_codec.decode(
                        payload, scale, activation_codec.FP8, self.dtype, self.hidden,
                    )
                    flat_out[start:stop].copy_(restored)
                    payload = scale = restored = None
        except BaseException as error:
            out = None
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            flat_out = host_payload = host_scale = payload = scale = restored = None
            self._record_slot(key)
        return out

    def begin_backward(self) -> None:
        self._raise_if_closed()

    def _finite_scales(self, key: tuple[int, int], shape: tuple[int, ...]) -> bool:
        rows = activation_codec.rows_of(shape)
        scales = self._scale_view(key, rows)
        try:
            for start in range(0, rows, _CODEC_TILE_ROWS):
                # absmax is non-negative. This uses one CPU row mask and scalar.
                if not bool((scales[start : start + _CODEC_TILE_ROWS] < math.inf).all()):
                    return False
            return True
        finally:
            scales = None

    def end_step(self) -> None:
        """Drain copies, refuse non-finite boundaries, and discard this step's shapes."""
        self._raise_if_closed()
        if self.is_cuda and self._cuda_used:
            import torch

            torch.cuda.synchronize(self.device)
            self._cuda_used = False
        bad_count = 0
        first_bad = []
        bad_keys = self._nonfinite_keys
        try:
            for key, shape in self._shapes.items():
                if key in bad_keys or not self._finite_scales(key, shape):
                    bad_count += 1
                    if len(first_bad) < 4:
                        first_bad.append(key)
            for key in bad_keys:
                if key in self._shapes:
                    continue
                bad_count += 1
                if len(first_bad) < 4:
                    first_bad.append(key)
        finally:
            self._shapes.clear()
            self._nonfinite_keys.clear()
        if bad_count:
            raise activation_codec.NonFiniteActivationError(
                f"{activation_codec.FP8}: {bad_count} boundary chunk(s) held a NaN or an Inf, "
                f"first (layer, micro-batch) {first_bad}; the lossy step is refused"
            )

    def host_bytes(self) -> dict[str, int]:
        """Live storage measurements and the exact shared reservation breakdown.

        ``owned_host_storage_bytes`` reads tensor storages, which exclude pinned
        allocator rounding. ``host_bytes`` is the live lease, including that
        rounding and the conservative scratch peak. The allocation dataclass
        remains available after close as the original plan.
        """
        payload = sum(int(value.untyped_storage().nbytes()) for value in self._payload.values())
        scales = 0 if self._scales is None else int(self._scales.untyped_storage().nbytes())
        return {
            "host_bytes": 0 if self._closed else self._lease.nbytes,
            "resident_payload_bytes": payload,
            "scale_bytes": scales,
            "pinned_overhead_bytes": 0 if self._closed else self.allocation.pinned_overhead_bytes,
            "host_transient_bytes": 0 if self._closed else self.allocation.host_transient_bytes,
            "device_transient_bytes": 0 if self._closed else self.allocation.device_transient_bytes,
            "owned_host_storage_bytes": payload + scales,
            "owned_device_storage_bytes": 0,
        }

    def close(self) -> None:
        """Drain before dropping storage and releasing its lease, idempotently.

        A failed drain leaves storage and its reservation intact, rejects new
        work, and permits a later close retry. Constructor failures need no CUDA
        drain because construction only allocates host storage.
        """
        if self._closed:
            return
        self._closing = True
        if self.is_cuda and self._cuda_used:
            import torch

            torch.cuda.synchronize(self.device)
            self._cuda_used = False
        self._payload.clear()
        self._scales = None
        self._shapes.clear()
        self._events.clear()
        self._nonfinite_keys.clear()
        self._last_event = None
        self._last_stream = None
        self._lease.close()
        self._closed = True
