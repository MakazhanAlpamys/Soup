"""Select the compiled host reader and pack validated expert shard ranges."""

from __future__ import annotations

import importlib
import logging
from contextlib import ExitStack
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence

if TYPE_CHECKING:
    from soup_cli.utils.expert_shard import ExpertShardReader
    from soup_cli.utils.expert_stream import ExpertBufferPlan

DEFAULT_EXPERT_LOADER = "async_lease"
NATIVE_EXPERT_LOADER = "cpp_async_2slot"
NATIVE_LOADER_OPTION = f"training.stream_expert_loader='{NATIVE_EXPERT_LOADER}'"
_EXTENSION = "_native_expert_stream_async"
_BUILD_COMMAND = (
    "python -m soup_cli.utils.native_expert_reader.build_ext build_ext "
    "--build-lib /tmp/soup-native --build-temp /tmp/soup-native/obj"
)
_RING_SLOTS = 2
_IO_WORKERS = 2
logger = logging.getLogger(__name__)


class CppAsyncLoaderUnavailableError(RuntimeError):
    """The selected native expert loader cannot run in this environment."""


def _load_extension() -> Any:
    # Load PyTorch's shared libraries before importing the compiled module.
    importlib.import_module("torch")
    return importlib.import_module(_EXTENSION)


def _require_extension() -> Any:
    try:
        module = _load_extension()
        if not callable(getattr(module, "PackedStreamAsync", None)):
            raise ImportError("the module does not export PackedStreamAsync")
    except Exception as exc:
        raise CppAsyncLoaderUnavailableError(
            f"{NATIVE_LOADER_OPTION} requires the compiled extension {_EXTENSION}. "
            f"Build it with `{_BUILD_COMMAND}` and add /tmp/soup-native to PYTHONPATH. "
            f"Import failed: {exc}"
        ) from exc
    return module


def resolve_expert_loader(loader: str = DEFAULT_EXPERT_LOADER, *, device: Any) -> str:
    """Validate the loader before setup performs any model or shard work."""
    if loader == DEFAULT_EXPERT_LOADER:
        return loader
    if loader != NATIVE_EXPERT_LOADER:
        raise ValueError(
            f"training.stream_expert_loader must be {DEFAULT_EXPERT_LOADER!r} or "
            f"{NATIVE_EXPERT_LOADER!r}; got {loader!r}"
        )

    import torch

    target = torch.device(device)
    if (
        target.type != "cuda"
        or not torch.cuda.is_available()
        or (target.index is not None and target.index >= torch.cuda.device_count())
    ):
        raise CppAsyncLoaderUnavailableError(
            f"{NATIVE_LOADER_OPTION} requires CUDA on the target device; got {str(device)!r}"
        )
    _require_extension()
    return loader


def native_expert_transfer(quant_specs: Mapping[str, Any]) -> str:
    """Use the packed CUDA transfer for NF4 and the normal dense copy otherwise."""
    return "packed" if quant_specs else "sync"


@dataclass(frozen=True)
class PackedStreamLayer:
    reader: ExpertShardReader
    rows: tuple[tuple[str, int, int, Any, tuple[int, ...]], ...]
    ranges: tuple[tuple[tuple[int, int, int], ...], ...]
    staging_bytes: int


@dataclass(frozen=True)
class PackedStreamPlan:
    layers: tuple[PackedStreamLayer, ...]
    packed_bytes: int

    @property
    def worker_staging_bytes(self) -> int:
        return max(layer.staging_bytes for layer in self.layers)


def plan_packed_stream(readers: Mapping[int, ExpertShardReader]) -> PackedStreamPlan:
    """Build C++ registrations from checked header ranges in pool packing order."""
    from soup_cli.utils.expert_transfer import layout_rows
    from soup_cli.utils.safetensors_reader import aligned_span

    if not readers:
        raise ValueError("a packed stream needs at least one expert reader")
    layers = []
    packed_bytes = None
    for _, reader in sorted(readers.items()):
        rows, size = layout_rows(
            (name.split(".", 1)[1], shape, dtype)
            for name, shape, dtype, _, _ in reader.stored_components(0)
        )
        if packed_bytes is not None and size != packed_bytes:
            raise ValueError("expert shard files disagree on their packed block size")
        packed_bytes = size
        all_ranges = []
        staging_bytes = 0
        for expert in range(reader.count):
            ranges = []
            for row, component in zip(rows, reader.stored_components(expert), strict=True):
                suffix, offset, nbytes, dtype, shape = row
                name, stored_shape, stored_dtype, start, end = component
                if (name, stored_shape, stored_dtype, end - start) != (
                    f"{expert}.{suffix}", shape, dtype, nbytes,
                ):
                    raise ValueError(f"expert {expert} component {name!r} has a different layout")
                ranges.append((start, end, offset))
                low, high = aligned_span(start, end)
                staging_bytes = max(staging_bytes, high - low)
            all_ranges.append(tuple(ranges))
        layers.append(PackedStreamLayer(reader, tuple(rows), tuple(all_ranges), staging_bytes))
    assert packed_bytes is not None
    return PackedStreamPlan(tuple(layers), packed_bytes)


def plan_native_expert_buffer(
    reader: ExpertShardReader, plan: PackedStreamPlan, *, dtype: str, transfer: str,
) -> ExpertBufferPlan:
    """Account for both native host buffers and the existing device arena."""
    from soup_cli.utils.expert_stream import plan_expert_buffer

    pool = plan_expert_buffer(reader, dtype=dtype, transfer=transfer)
    return account_native_expert_staging(pool, plan, transfer=transfer)


def account_native_expert_staging(
    pool: ExpertBufferPlan, plan: PackedStreamPlan, *, transfer: str,
) -> ExpertBufferPlan:
    """Replace the Python staging charge with the native host buffers."""
    host_bytes = _IO_WORKERS * plan.worker_staging_bytes + _RING_SLOTS * plan.packed_bytes
    if transfer == "sync":
        host_bytes += plan.packed_bytes
    return replace(pool, host_staging_bytes=host_bytes)


def native_expert_buffer_requirements(
    shard_dir: str, index: Any, *, device: str,
) -> tuple[int, int]:
    """Return the device and host byte budgets for the training preflight."""
    from soup_cli.utils.expert_shard import ExpertShardReader
    from soup_cli.utils.expert_stream import expert_allocation_budget
    from soup_cli.utils.layer_shard import expert_paths

    with ExitStack() as stack:
        readers = {}
        for layer, path in expert_paths(shard_dir, index).items():
            reader = ExpertShardReader(
                path, count=index.expert_count, shapes=index.expert_shapes,
                quant_specs=index.expert_quant_specs, dtype=index.dtype,
            )
            stack.callback(reader.close)
            readers[layer] = reader
        stream = plan_packed_stream(readers)
        plan = plan_native_expert_buffer(
            stream.layers[0].reader, stream, dtype=index.dtype,
            transfer=native_expert_transfer(index.expert_quant_specs),
        )
    host_bytes = plan.host_staging_bytes + plan.host_pinned_bytes
    return expert_allocation_budget(plan, device=device), host_bytes


class CppPackedStream:
    """Bind the compiled reader's sequence contract to expert shards and host tensors."""

    def __init__(self, handle: Any, plan: PackedStreamPlan) -> None:
        self.handle = handle
        self.plan = plan
        self._positions = {layer.reader: pos for pos, layer in enumerate(plan.layers)}

    def _position(self, reader: ExpertShardReader) -> int:
        try:
            return self._positions[reader]
        except KeyError:
            raise ValueError(f"expert reader {reader.path!r} is not registered") from None

    def packed_geometry(self, reader: ExpertShardReader) -> tuple:
        """The layout without names, for comparing with the pool's packed CUDA region."""
        return tuple(row[1:] for row in self.plan.layers[self._position(reader)].rows)

    def sequence(
        self, reader: ExpertShardReader, experts: Sequence[int], destination: Any,
    ) -> Iterator[int]:
        """Read each requested expert into the same host slot and finish on every exit."""
        import torch

        layer = self._position(reader)
        ids = list(experts)
        for expert in ids:
            if type(expert) is not int or not 0 <= expert < reader.count:
                raise IndexError(f"expert {expert!r} outside [0, {reader.count})")
        if (
            destination.device.type != "cpu"
            or destination.dtype != torch.uint8
            or not destination.is_contiguous()
            or destination.ndim != 1
            or destination.numel() < self.plan.packed_bytes
        ):
            raise ValueError(f"native expert destination needs {self.plan.packed_bytes} CPU bytes")
        if not ids:
            return
        pointer = destination.data_ptr()
        self.handle.start_sequence(layer, ids)
        try:
            for expert in ids:
                got = self.handle.consume_next(pointer)
                if got != expert:
                    raise RuntimeError(
                        f"packed stream returned expert {got} where the sequence expects {expert}"
                    )
                yield expert
        except GeneratorExit:
            self.handle.finish_sequence()
            raise
        except BaseException:
            try:
                self.handle.finish_sequence()
            except Exception:
                logger.exception("failed to finish the native expert sequence")
            raise
        else:
            self.handle.finish_sequence()

    def read_into(self, reader: ExpertShardReader, expert: int, destination: Any) -> int:
        """Satisfy one lease and return its logical payload bytes, excluding padding."""
        for _ in self.sequence(reader, [expert], destination):
            pass
        return sum(row[2] for row in self.plan.layers[self._position(reader)].rows)

    def host_views(
        self, reader: ExpertShardReader, expert: int, destination: Any,
    ) -> dict[str, Any]:
        """Expose the stored components in the same order as the Python shard reader."""
        return {
            f"{expert}.{suffix}": destination[offset : offset + count].view(dtype).view(shape)
            for suffix, offset, count, dtype, shape in self.plan.layers[self._position(reader)].rows
        }

    def close(self) -> None:
        self.handle.close()


def open_native_stream(readers: Mapping[int, ExpertShardReader]) -> CppPackedStream:
    """Open the host reader after CUDA setup checks, or directly for CPU reader tests."""
    module = _require_extension()
    plan = plan_packed_stream(readers)
    with ExitStack() as stack:
        registrations = []
        for layer in plan.layers:
            shard = stack.enter_context(layer.reader.open_buffered())
            registrations.append((layer.reader, shard.fileno(), [list(r) for r in layer.ranges]))
        # The extension duplicates the file descriptors and owns the copies.
        handle = module.PackedStreamAsync(
            registrations, plan.worker_staging_bytes, plan.packed_bytes, 0,
            _IO_WORKERS, _RING_SLOTS,
        )
    return CppPackedStream(handle, plan)
