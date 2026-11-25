"""B1.1 storage codec, adapted from Soup d500ae0d's expert_stream.py.

Only storage and independent reads are exposed here. Expert scheduling,
autograd and a public training flag belong to B1.2/B1.3, not this module.
One experts_NNN.safetensors holds independently addressed expert matrices;
the ordinary layer file holds attention, norms, router and shared experts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

GATE_UP = "gate_up"
DOWN = "down"
EXPERT_PARTS = (GATE_UP, DOWN)
EXPERT_LAYOUT_PER_EXPERT = "per_expert"
EXPERT_LAYOUT_VERSION = 1
_MAX_EXPERTS = 4096
_MAX_EXPERT_BYTES = 2**30


def expert_key(expert: int, part: str) -> str:
    """Shard key of one expert's matrix."""
    return f"{expert}.{part}"


def expert_shard_name(layer: int) -> str:
    """File name of a MoE layer's experts file."""
    return f"experts_{layer:03d}.safetensors"


@dataclass(frozen=True)
class ExpertShardInfo:
    """What :func:`write_expert_shard` wrote for one MoE layer."""

    count: int
    #: part -> stored dense shape, e.g. {"gate_up": (2I, H), "down": (H, I)}
    shapes: Mapping[str, Tuple[int, ...]]
    #: part -> NF4WeightSpec; empty when unquantised
    quant_specs: Mapping[str, Any] = field(default_factory=dict)
    total_params: int = 0
    #: the NF4 code tables seen while quantising (``code``, ``nested_code``)
    codes: Tuple[Any, Any] = (None, None)


def write_expert_shard(
    path: str,
    experts: Iterable[Tuple[Any, Any]],
    *,
    quantise: bool,
    double_quant: bool = True,
    device: str = "cpu",
) -> ExpertShardInfo:
    """Write one MoE layer's routed experts, one ``(gate_up, down)`` pair each.

    ``experts`` yields the dense matrices in expert order, already at the
    stream dtype. Under ``quantise`` each matrix is NF4-quantised on its own,
    exactly as the decoder-layer sharder quantises one ``nn.Linear`` weight.
    Every expert must have the shapes of the first one.
    """
    from soup_cli.utils.layer_shard import _atomic_save, _CodeTables, _quantize_nf4

    blob: Dict[str, Any] = {}
    shapes: Dict[str, Tuple[int, ...]] = {}
    specs: Dict[str, Any] = {}
    tables = _CodeTables()
    count = 0
    total = 0
    for expert, pair in enumerate(experts):
        if expert >= _MAX_EXPERTS:
            raise ValueError(f"a MoE layer may hold at most {_MAX_EXPERTS} experts")
        if len(pair) != 2:
            raise ValueError(f"expert {expert} must supply exactly gate_up and down")
        for part, tensor in zip(EXPERT_PARTS, pair):
            shape = _checked_shape(part, tensor.shape)
            prior = shapes.setdefault(part, shape)
            if prior != shape:
                raise ValueError(f"expert {expert} {part} has shape {shape}; expert 0 has {prior}")
            total += tensor.numel()
            key = expert_key(expert, part)
            if not quantise:
                blob[key] = tensor.detach().contiguous()
                continue
            sidecars, spec, seen_code, seen_nested = _quantize_nf4(
                tensor, double_quant=double_quant, device=device
            )
            if spec.blocksize != 64 or spec.quant_type != "nf4":
                raise ValueError(f"{key}: NF4 codec returned unexpected blocksize or format")
            # Validate the codec contract before retaining or writing its output.
            # A different bitsandbytes release must not silently grow the cache
            # or produce a file the bounded reader cannot reconstruct.
            from soup_cli.utils.layer_stream_runtime import quant_sidecar_keys

            expected_keys = quant_sidecar_keys(key, spec)
            expected = _expected_tensors(key, shape, spec)
            actual = {key + suffix: value for suffix, value in sidecars.items()}
            if set(actual) != set(expected_keys):
                raise ValueError(f"{key}: NF4 codec returned unexpected sidecars")
            for name, (wanted_shape, wanted_dtype) in zip(expected_keys, expected):
                value = actual[name]
                if tuple(value.shape) != tuple(wanted_shape) or value.dtype != wanted_dtype:
                    raise ValueError(f"{name}: NF4 codec returned unexpected shape or dtype")
            prior_spec = specs.setdefault(part, spec)
            if prior_spec != spec:
                raise ValueError(f"expert {expert} {part} has inconsistent NF4 metadata")
            tables.observe(key, seen_code, seen_nested)
            for suffix, value in sidecars.items():
                blob[key + suffix] = value
        count += 1
    if count == 0:
        raise ValueError("a MoE layer needs at least one expert")
    if shapes[GATE_UP][1] != shapes[DOWN][0] or shapes[GATE_UP][0] != 2 * shapes[DOWN][1]:
        raise ValueError(
            f"gate_up {shapes[GATE_UP]} and down {shapes[DOWN]} are not one expert MLP "
            f"([2I, H] and [H, I])"
        )
    _atomic_save(blob, path)
    return ExpertShardInfo(
        count=count,
        shapes=shapes,
        quant_specs=specs,
        total_params=total,
        codes=(tables.code, tables.nested_code),
    )


class ExpertShardReader:
    """Reads one MoE layer's experts file, one expert at a time.

    Each stored tensor of the expert is one positioned read of its own
    sector-aligned span, through the shared ``safetensors_reader`` (its header
    is validated against the file), so the file's tensor order does not matter.
    No memory map: a live safetensors map charges Windows commit for the whole
    file, and a 1T-class store has ~60 files of ~9 GB. Direct (unbuffered) I/O
    when the platform has it, so a store far larger than RAM does not churn the
    page cache of the whole box; buffered reads otherwise.

    The header is read through the identity of a handle opened for it, and the
    handle is closed again before ``__init__`` returns: a reader holds no OS
    handle until its first fetch, so an install that fails after the readers
    were built leaks nothing. The first fetch reopens the file and refuses it
    unless its identity is still the one the header was read from, so a file
    swapped in between is never read at stale offsets. From then on the handle
    stays open, which on Windows also keeps the file from being replaced.
    Every stored tensor's size is checked against the index (dense shape, NF4
    spec), so a header-valid file cannot steer a read or a dequantisation past
    the bytes it really holds.
    """

    def __init__(
        self,
        path: str,
        *,
        count: int,
        shapes: Mapping[str, Tuple[int, ...]],
        quant_specs: Mapping[str, Any],
        dtype: str,
        direct: bool = True,
    ) -> None:
        from soup_cli.utils.layer_stream_runtime import quant_sidecar_keys
        from soup_cli.utils.safetensors_reader import (
            aligned_span,
            identity_of,
            read_header_with_identity,
        )

        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= _MAX_EXPERTS:
            raise ValueError(f"expert count must be an int in [1, {_MAX_EXPERTS}], got {count!r}")
        self.path = path
        self.count = count
        self.shapes = {part: _checked_shape(part, shapes[part]) for part in EXPERT_PARTS}
        if set(quant_specs) not in (set(), set(EXPERT_PARTS)):
            raise ValueError("NF4 specs must name both gate_up and down, or neither")
        if self.shapes[GATE_UP][1] != self.shapes[DOWN][0] or (
            self.shapes[GATE_UP][0] != 2 * self.shapes[DOWN][1]
        ):
            raise ValueError("gate_up and down shapes are not one expert MLP")
        self.quant_specs = dict(quant_specs)
        self.dtype = dtype
        self.direct = direct
        self._handle: Optional[Any] = None
        name = os.path.basename(path)
        self._identity: Optional[Any] = None
        handle = self._open()
        try:
            header, identity = read_header_with_identity(path)
        except BaseException:
            self.close()
            raise
        if identity_of(handle) != identity:
            self.close()
            raise ValueError(f"experts file {name} changed while it was being opened")
        self._identity = identity
        self._size = identity.size
        expected = {
            part: _expected_tensors(
                expert_key(0, part), self.shapes[part], self.quant_specs.get(part)
            )
            for part in EXPERT_PARTS
        }
        stored_dtype = _stored_torch_dtype(dtype)
        #: per expert: ((tensor name, entry), ...)
        self._entries: list = []
        largest = 0
        for expert in range(count):
            entries = []
            for part in EXPERT_PARTS:
                key = expert_key(expert, part)
                spec = self.quant_specs.get(part)
                wanted = quant_sidecar_keys(key, spec) if spec is not None else (key,)
                for tensor_name, (want_shape, want_dtype) in zip(wanted, expected[part]):
                    entry = header.get(tensor_name)
                    if entry is None:
                        self.close()
                        raise ValueError(
                            f"experts file {name} has no tensor {tensor_name!r} "
                            f"— reshard the checkpoint"
                        )
                    got_dtype = _stored_torch_dtype(entry.dtype)
                    if want_dtype is None:
                        want_dtype = stored_dtype
                    if tuple(entry.shape) != want_shape or got_dtype != want_dtype:
                        self.close()
                        raise ValueError(
                            f"experts file {name}: {tensor_name!r} is {entry.dtype} "
                            f"{tuple(entry.shape)}, the index expects {want_dtype} "
                            f"{want_shape} — reshard the checkpoint"
                        )
                    if entry.start % got_dtype.itemsize:
                        self.close()
                        raise ValueError(
                            f"experts file {name}: {tensor_name!r} is not aligned to its dtype"
                        )
                    entries.append((tensor_name, entry))
            need = 0
            for _, entry in entries:
                lo, hi = aligned_span(entry.start, entry.end)
                need += hi - lo
            if need > _MAX_EXPERT_BYTES:
                self.close()
                raise ValueError(
                    f"experts file {name}: expert {expert} needs {need} bytes, over the "
                    f"{_MAX_EXPERT_BYTES}-byte limit for one expert"
                )
            largest = max(largest, need)
            self._entries.append(tuple(entries))
        if expert_key(count, GATE_UP) in header or expert_key(count, DOWN) in header:
            self.close()
            raise ValueError(
                f"experts file {name} holds more than {count} experts — reshard the checkpoint"
            )
        expected_names = {key for entries in self._entries for key, _ in entries}
        if set(header) != expected_names:
            self.close()
            raise ValueError(f"experts file {name} has unexpected tensors — reshard the checkpoint")
        #: Staging bytes one expert needs (each tensor's span on sector bounds).
        self.max_span = largest
        self.fetches = 0
        self.bytes_read = 0
        self.close()

    def _open(self) -> Any:
        if self._handle is None:
            if self.direct:
                from soup_cli.utils.safetensors_reader import open_direct

                try:
                    self._handle = open_direct(self.path)
                except OSError:
                    # No direct I/O on this platform / filesystem: buffered reads.
                    self.direct = False
            if self._handle is None:
                self._handle = open(self.path, "rb", buffering=0)
            if self._identity is not None:
                from soup_cli.utils.safetensors_reader import identity_of

                if identity_of(self._handle) != self._identity:
                    self.close()
                    raise ValueError(
                        f"experts file {os.path.basename(self.path)} changed on disk since "
                        f"its header was read — reshard the checkpoint and restart"
                    )
        return self._handle

    def read_host(self, expert: int, staging: Any) -> Dict[str, Any]:
        """``expert``'s stored tensors as CPU views into ``staging``.

        ``staging`` is a contiguous CPU uint8 tensor of at least ``max_span``
        bytes whose address is sector-aligned (``aligned_staging``). The views
        are valid until the next read into it.
        """
        import torch

        from soup_cli.utils.safetensors_reader import aligned_span, read_range_into

        if isinstance(expert, bool) or not isinstance(expert, int) or not 0 <= expert < self.count:
            raise IndexError(f"expert {expert} outside [0, {self.count})")
        from soup_cli.utils.safetensors_reader import SECTOR_BYTES

        if (
            staging.ndim != 1
            or staging.device.type != "cpu"
            or staging.dtype != torch.uint8
            or not staging.is_contiguous()
            or staging.data_ptr() % SECTOR_BYTES
        ):
            raise ValueError("staging must be contiguous sector-aligned CPU uint8 memory")
        if staging.numel() < self.max_span:
            raise ValueError(f"staging holds {staging.numel()} bytes, a read needs {self.max_span}")
        handle = self._open()
        out: Dict[str, Any] = {}
        cursor = 0
        for tensor_name, entry in self._entries[expert]:
            lo, hi = aligned_span(entry.start, entry.end)
            region = staging[cursor : cursor + hi - lo]
            read_range_into(handle, lo, region, min(hi, self._size) - lo)
            raw = region[entry.start - lo : entry.end - lo]
            out[tensor_name] = raw.view(_stored_torch_dtype(entry.dtype)).view(entry.shape)
            cursor += hi - lo
            self.bytes_read += entry.nbytes
        self.fetches += 1
        return out

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - best effort at interpreter teardown
            pass


def _checked_shape(part: str, shape: Any) -> Tuple[int, ...]:
    if any(isinstance(dim, bool) or not isinstance(dim, int) for dim in shape):
        raise ValueError(f"expert {part} shape must have integer dimensions")
    dims = tuple(int(dim) for dim in shape)
    if len(dims) != 2 or any(dim < 1 for dim in dims) or dims[0] * dims[1] > 2**31:
        raise ValueError(f"expert {part} shape must be two positive dims, got {tuple(shape)!r}")
    return dims


def _expected_tensors(key: str, shape: Tuple[int, int], spec: Any) -> Tuple[Tuple[Any, Any], ...]:
    """(shape, torch dtype) of every stored tensor of one expert matrix, in the
    order ``quant_sidecar_keys`` names them. ``None`` dtype = the stream dtype.

    NF4 (bitsandbytes ``quantize_4bit``): packed nibbles ``[ceil(n/2), 1]`` uint8,
    one absmax per ``blocksize`` weights (fp32, or uint8 codes under double
    quantisation with an fp32 ``nested_absmax`` per ``nested_blocksize`` and a
    scalar fp32 ``nested_offset``).
    """
    import torch

    if spec is None:
        return ((shape, None),)
    if tuple(int(dim) for dim in spec.shape) != shape:
        raise ValueError(f"{key}: NF4 spec shape {tuple(spec.shape)} is not the index's {shape}")
    nested_ok = not spec.nested or int(spec.nested_blocksize) >= 1
    if not 1 <= int(spec.blocksize) <= 4096 or not nested_ok:
        raise ValueError(f"{key}: NF4 spec has an impossible block size")
    weights = shape[0] * shape[1]
    blocks = -(-weights // int(spec.blocksize))
    out = [(((weights + 1) // 2, 1), torch.uint8)]
    if spec.nested:
        out.append(((blocks,), torch.uint8))
        out.append(((-(-blocks // int(spec.nested_blocksize)),), torch.float32))
        out.append(((), torch.float32))
    else:
        out.append(((blocks,), torch.float32))
    return tuple(out)


def aligned_staging(size: int) -> Tuple[Any, Any]:
    """``(owner, view)``: ``size`` bytes of CPU memory starting on a sector."""
    import torch

    from soup_cli.utils.safetensors_reader import SECTOR_BYTES

    owner = torch.empty(size + SECTOR_BYTES, dtype=torch.uint8, device="cpu")
    pad = (-owner.data_ptr()) % SECTOR_BYTES
    return owner, owner[pad : pad + size]


def _stored_torch_dtype(name: str) -> Any:
    import torch

    # ``safetensors_reader`` reports torch names ("bfloat16"); the file format
    # itself spells them "BF16". Either is accepted, nothing else is.
    table = {
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
        "U8": torch.uint8,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "uint8": torch.uint8,
    }
    if name not in table:
        raise ValueError(f"experts file stores unsupported dtype {name!r}")
    return table[name]
