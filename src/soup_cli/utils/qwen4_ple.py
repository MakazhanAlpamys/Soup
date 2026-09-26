"""Qwen4-Exp PLE storage for layer-streamed training.

The production table is roughly 95 GiB in an official bf16 checkpoint.  A
decoder-layer shard cannot contain it: the layer runtime would allocate a
95-GiB accelerator buffer even though PLE only gathers a few rows per token.
This module replaces that single ``nn.Embedding`` with either a frozen CPU
embedding or a read-only safetensors mmap that materialises only requested
rows.

No top-level torch / safetensors imports: both are optional train dependencies.
"""

from __future__ import annotations

import json
import math
import mmap
import os
import struct
from dataclasses import dataclass
from typing import Any, Mapping, Tuple

_HEADER_LIMIT = 100 * 1024 * 1024
_DTYPE_INFO = {
    "U32": (4, "<u4", "uint32"),
    "BF16": (2, "<u2", "bfloat16"),
    "F16": (2, "<f2", "float16"),
    "F32": (4, "<f4", "float32"),
}
_TORCH_DTYPE_NAMES = {
    **{name: info[2] for name, info in _DTYPE_INFO.items()},
    "bfloat16": "bfloat16",
    "float16": "float16",
    "float32": "float32",
}


@dataclass(frozen=True)
class TensorRowsSpec:
    """Validated byte range for a contiguous 2-D safetensors matrix."""

    shape: Tuple[int, int]
    dtype: str
    start: int
    end: int

    @property
    def row_bytes(self) -> int:
        return self.shape[1] * _DTYPE_INFO[self.dtype][0]


def _is_under(path: str, root: str) -> bool:
    try:
        return os.path.commonpath((path, root)) == root
    except ValueError:
        return False


def _validated_source_path(weights_dir: str, source_file: str) -> str:
    root = os.path.realpath(os.path.expanduser(weights_dir))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"weights directory not found: {weights_dir}")
    if source_file != os.path.basename(source_file) or source_file in ("", ".", ".."):
        raise ValueError(f"invalid PLE source filename {source_file!r}")
    path = os.path.realpath(os.path.join(root, source_file))
    if not _is_under(path, root):
        raise ValueError(f"PLE source escapes the weights directory: {source_file!r}")
    if os.path.islink(os.path.join(root, source_file)):
        raise ValueError(f"PLE source must not be a symlink: {source_file!r}")
    return path


def _open_read_only(path: str) -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return os.open(path, flags)


def _read_header(handle: Any) -> Tuple[Mapping[str, Any], int, int]:
    handle.seek(0)
    file_size = os.fstat(handle.fileno()).st_size
    raw_length = handle.read(8)
    if len(raw_length) != 8:
        raise ValueError("safetensors file is truncated before its header length")
    header_length = struct.unpack("<Q", raw_length)[0]
    if header_length <= 0 or header_length > _HEADER_LIMIT:
        raise ValueError(
            f"invalid safetensors header length {header_length}; limit is {_HEADER_LIMIT}"
        )
    data_start = 8 + header_length
    if data_start > file_size:
        raise ValueError("safetensors header extends past end of file")
    raw_header = handle.read(header_length)
    if len(raw_header) != header_length:
        raise ValueError("safetensors file is truncated inside its header")
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("safetensors header is not valid UTF-8 JSON") from exc
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be a JSON object")
    return header, data_start, file_size


def _tensor_rows_from_header(
    header: Mapping[str, Any],
    data_start: int,
    file_size: int,
    source_key: str,
) -> TensorRowsSpec:
    entry = header.get(source_key)
    if not isinstance(entry, dict):
        raise ValueError(f"safetensors tensor {source_key!r} is missing")
    dtype = str(entry.get("dtype", ""))
    if dtype not in _DTYPE_INFO:
        raise ValueError(
            f"unsupported PLE dtype {dtype!r}; supported: "
            f"{', '.join(sorted(_DTYPE_INFO))}"
        )
    shape_raw = entry.get("shape")
    if not isinstance(shape_raw, list) or len(shape_raw) != 2:
        raise ValueError(f"PLE tensor must be a 2-D matrix; got {shape_raw!r}")
    shape = tuple(int(dim) for dim in shape_raw)
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"PLE tensor dimensions must be positive; got {shape}")
    offsets = entry.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError(f"PLE tensor has invalid data_offsets {offsets!r}")
    relative_start, relative_end = (int(offsets[0]), int(offsets[1]))
    if relative_start < 0 or relative_end < relative_start:
        raise ValueError(f"PLE tensor has invalid data_offsets {offsets!r}")
    expected = math.prod(shape) * _DTYPE_INFO[dtype][0]
    if relative_end - relative_start != expected:
        raise ValueError(
            f"PLE byte range is {relative_end - relative_start}, expected {expected}"
        )
    start = data_start + relative_start
    end = data_start + relative_end
    if end > file_size:
        raise ValueError("PLE tensor data extends past end of safetensors file")
    return TensorRowsSpec(shape=(shape[0], shape[1]), dtype=dtype, start=start, end=end)


class _SafeTensorContainer:
    """One validated header, descriptor and read-only mmap per source file."""

    def __init__(self, path: str) -> None:
        from safetensors import safe_open

        self.path = path
        with safe_open(path, framework="pt") as safe_handle:
            safe_specs = {}
            for key in safe_handle.keys():
                tensor_slice = safe_handle.get_slice(key)
                safe_specs[key] = (
                    tuple(int(dim) for dim in tensor_slice.get_shape()),
                    str(tensor_slice.get_dtype()),
                )
            self.safe_specs = safe_specs
        descriptor = _open_read_only(path)
        self.file = os.fdopen(descriptor, "rb", closefd=True)
        try:
            self.header, self.data_start, self.file_size = _read_header(self.file)
            self.mapping = mmap.mmap(self.file.fileno(), length=0, access=mmap.ACCESS_READ)
        except BaseException:
            self.file.close()
            raise
        self.closed = False

    def tensor_spec(
        self,
        source_key: str,
        expected_shape: Tuple[int, ...],
        expected_dtype: str,
    ) -> TensorRowsSpec:
        safe_spec = self.safe_specs.get(source_key)
        if safe_spec != (tuple(expected_shape), expected_dtype):
            raise ValueError(
                "PLE descriptor does not match the validated safetensors header"
            )
        spec = _tensor_rows_from_header(
            self.header,
            self.data_start,
            self.file_size,
            source_key,
        )
        if tuple(spec.shape) != tuple(expected_shape):
            raise ValueError(f"PLE shape changed ({spec.shape} != {tuple(expected_shape)})")
        if spec.dtype != expected_dtype:
            raise ValueError(
                f"PLE dtype changed ({spec.dtype!r} != {expected_dtype!r})"
            )
        return spec

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.mapping.close()
        self.file.close()

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass


class SafeTensorRowReader:
    """Read-only mmap row gather over one dense safetensors matrix."""

    def __init__(
        self,
        weights_dir: str,
        *,
        source_file: str,
        source_key: str,
        expected_shape: Tuple[int, ...],
        expected_dtype: str,
        _container: Any = None,
    ) -> None:
        self.path = _validated_source_path(weights_dir, source_file)
        self._container = _container or _SafeTensorContainer(self.path)
        if self._container.path != self.path:
            raise ValueError("shared PLE container path mismatch")
        self._owns_container = _container is None
        self.spec = self._container.tensor_spec(
            source_key, expected_shape, expected_dtype
        )
        self._mapping = self._container.mapping
        self.closed = False

    @property
    def nbytes(self) -> int:
        return self.spec.end - self.spec.start

    @property
    def shape(self) -> Tuple[int, int]:
        return self.spec.shape

    @property
    def dtype(self) -> str:
        return self.spec.dtype

    def gather(self, row_ids: Any):
        import numpy as np
        import torch

        if self.closed:
            raise RuntimeError("PLE row reader is closed")
        ids = row_ids.detach().to(device="cpu", dtype=torch.long)
        flat = ids.reshape(-1)
        if flat.numel() == 0:
            dtype = getattr(torch, _DTYPE_INFO[self.spec.dtype][2])
            return torch.empty((*ids.shape, self.spec.shape[1]), dtype=dtype)
        minimum = int(flat.min())
        maximum = int(flat.max())
        if minimum < 0 or maximum >= self.spec.shape[0]:
            raise IndexError(
                f"PLE row id outside [0, {self.spec.shape[0]}): "
                f"min={minimum}, max={maximum}"
            )
        _itemsize, numpy_dtype, torch_dtype = _DTYPE_INFO[self.spec.dtype]
        matrix = np.ndarray(
            shape=self.spec.shape,
            dtype=numpy_dtype,
            buffer=self._mapping,
            offset=self.spec.start,
        )
        # Integer-array indexing is an explicit copy of only selected rows. It
        # performs the gather in NumPy's C loop instead of issuing one Python
        # mmap slice per random N-gram id.
        selected = np.ascontiguousarray(matrix[flat.numpy()])
        rows = torch.from_numpy(selected)
        if self.spec.dtype == "BF16":
            rows = rows.view(torch.bfloat16)
        else:
            rows = rows.to(getattr(torch, torch_dtype), copy=False)
        return rows.view(*ids.shape, self.spec.shape[1])

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self._owns_container:
            self._container.close()

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass


def _module_for_name(model: Any, name: str) -> Tuple[Any, str]:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


class ShardedSafeTensorRowReader:
    """Logical row gather across Transformers' 128 PLE source fragments."""

    def __init__(self, weights_dir: str, spec: Any) -> None:
        self.shape = tuple(spec.shape)
        self.dtype = str(spec.dtype)
        containers = {}
        try:
            for part in spec.parts:
                path = _validated_source_path(weights_dir, part.source_file)
                if part.source_file not in containers:
                    containers[part.source_file] = _SafeTensorContainer(path)
            self.parts = tuple(
                SafeTensorRowReader(
                    weights_dir,
                    source_file=part.source_file,
                    source_key=part.source_key,
                    expected_shape=part.shape,
                    expected_dtype=part.dtype,
                    _container=containers[part.source_file],
                )
                for part in spec.parts
            )
        except BaseException:
            for container in containers.values():
                container.close()
            raise
        self.containers = tuple(containers.values())
        self.closed = False

    @property
    def nbytes(self) -> int:
        return sum(part.nbytes for part in self.parts)

    def gather(self, row_ids: Any):
        if self.closed:
            raise RuntimeError("PLE row reader is closed")
        return _gather_sharded(row_ids, self.shape, self.dtype, self.parts)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for part in self.parts:
            part.close()
        for container in self.containers:
            container.close()

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass


class OQAffinePartRowReader:
    """Read and dequantize selected rows from one packed oQ PLE fragment."""

    def __init__(self, weights_dir: str, part: Any, spec: Any, container: Any) -> None:
        from soup_cli.utils.oq_affine import AffineQuantSpec, logical_width

        self.shape = (
            int(part.packed_shape[0]),
            logical_width(int(part.packed_shape[1]), int(spec.bits)),
        )
        self.dtype = str(spec.dtype)
        self.quant_spec = AffineQuantSpec(
            bits=int(spec.bits),
            group_size=int(spec.group_size),
            mode=str(spec.mode),
        )
        common = {
            "weights_dir": weights_dir,
            "source_file": part.source_file,
            "_container": container,
        }
        self.weight = SafeTensorRowReader(
            source_key=part.weight_key,
            expected_shape=part.packed_shape,
            expected_dtype=part.packed_dtype,
            **common,
        )
        self.scales = SafeTensorRowReader(
            source_key=part.scales_key,
            expected_shape=part.stats_shape,
            expected_dtype=part.stats_dtype,
            **common,
        )
        self.biases = SafeTensorRowReader(
            source_key=part.biases_key,
            expected_shape=part.stats_shape,
            expected_dtype=part.stats_dtype,
            **common,
        )
        self.closed = False

    @property
    def nbytes(self) -> int:
        return self.weight.nbytes + self.scales.nbytes + self.biases.nbytes

    def gather(self, row_ids: Any):
        if self.closed:
            raise RuntimeError("oQ PLE row reader is closed")
        from soup_cli.utils.oq_affine import dequantize_affine

        packed = self.weight.gather(row_ids)
        scales = self.scales.gather(row_ids)
        biases = self.biases.gather(row_ids)
        return dequantize_affine(
            packed,
            scales,
            biases,
            spec=self.quant_spec,
            dtype=self.dtype,
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.weight.close()
        self.scales.close()
        self.biases.close()


class OQShardedSafeTensorRowReader:
    """Logical row gather over packed oQ PLE fragments on a read-only mmap."""

    def __init__(self, weights_dir: str, spec: Any) -> None:
        self.shape = tuple(spec.shape)
        self.dtype = str(spec.dtype)
        containers = {}
        try:
            for part in spec.parts:
                path = _validated_source_path(weights_dir, part.source_file)
                if part.source_file not in containers:
                    containers[part.source_file] = _SafeTensorContainer(path)
            self.parts = tuple(
                OQAffinePartRowReader(
                    weights_dir,
                    part,
                    spec,
                    containers[part.source_file],
                )
                for part in spec.parts
            )
        except BaseException:
            for container in containers.values():
                container.close()
            raise
        self.containers = tuple(containers.values())
        self.closed = False

    @property
    def nbytes(self) -> int:
        return sum(part.nbytes for part in self.parts)

    def gather(self, row_ids: Any):
        if self.closed:
            raise RuntimeError("oQ PLE row reader is closed")
        return _gather_sharded(row_ids, self.shape, self.dtype, self.parts)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for part in self.parts:
            part.close()
        for container in self.containers:
            container.close()

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass


class ShardedRamRowReader:
    """Frozen CPU-resident PLE parts without a second concatenation buffer."""

    def __init__(self, weights_dir: str, spec: Any) -> None:
        import torch
        from safetensors import safe_open

        self.shape = tuple(spec.shape)
        self.dtype = str(spec.dtype)
        tensors = []
        for part in spec.parts:
            path = _validated_source_path(weights_dir, part.source_file)
            with safe_open(path, framework="pt", device="cpu") as handle:
                tensor = handle.get_tensor(part.source_key).clone()
            if tuple(tensor.shape) != tuple(part.shape):
                raise ValueError("PLE RAM part shape changed while loading")
            expected_dtype = getattr(torch, _DTYPE_INFO[part.dtype][2])
            if tensor.dtype != expected_dtype:
                raise ValueError("PLE RAM part dtype changed while loading")
            tensors.append(tensor)
        self.parts = tuple(tensors)
        self.closed = False

    @property
    def nbytes(self) -> int:
        return sum(part.numel() * part.element_size() for part in self.parts)

    def gather(self, row_ids: Any):
        if self.closed:
            raise RuntimeError("PLE row reader is closed")
        return _gather_sharded(row_ids, self.shape, self.dtype, self.parts)

    def close(self) -> None:
        self.closed = True
        self.parts = ()


def _gather_sharded(
    row_ids: Any,
    shape: Tuple[int, int],
    dtype: str,
    parts: Tuple[Any, ...],
):
    import torch

    ids = row_ids.detach().to(device="cpu", dtype=torch.long)
    flat = ids.reshape(-1)
    if flat.numel() == 0:
        return torch.empty(
            (*ids.shape, shape[1]), dtype=getattr(torch, _TORCH_DTYPE_NAMES[dtype])
        )
    minimum = int(flat.min())
    maximum = int(flat.max())
    if minimum < 0 or maximum >= shape[0]:
        raise IndexError(
            f"PLE row id outside [0, {shape[0]}): min={minimum}, max={maximum}"
        )
    output = torch.empty(
        (flat.numel(), shape[1]), dtype=getattr(torch, _TORCH_DTYPE_NAMES[dtype])
    )
    row_start = 0
    for part in parts:
        part_shape = part.shape if hasattr(part, "shape") else tuple(part.spec.shape)
        row_end = row_start + int(part_shape[0])
        mask = (flat >= row_start) & (flat < row_end)
        positions = torch.nonzero(mask, as_tuple=False).flatten()
        if positions.numel():
            local_ids = flat.index_select(0, positions) - row_start
            if isinstance(part, torch.Tensor):
                values = part.index_select(0, local_ids)
            else:
                values = part.gather(local_ids)
            output.index_copy_(0, positions, values.reshape(-1, shape[1]))
        row_start = row_end
    if row_start != shape[0]:
        raise RuntimeError("PLE part rows do not cover the logical tensor")
    return output.view(*ids.shape, shape[1])


def _disk_embedding(reader: SafeTensorRowReader):
    import torch
    import torch.nn as nn

    class DiskBackedEmbedding(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Transformers only consults weight.device before calling us. A
            # non-persistent zero-row buffer provides that contract without
            # pretending the 95-GiB table is a resident parameter.
            dtype = getattr(torch, _TORCH_DTYPE_NAMES[reader.dtype])
            self.register_buffer(
                "weight",
                torch.empty((0, reader.shape[1]), dtype=dtype),
                persistent=False,
            )
            self.num_embeddings = reader.shape[0]
            self.embedding_dim = reader.shape[1]
            self.reader = reader

        def forward(self, input_ids):
            return self.reader.gather(input_ids).to(self.weight.device)

    return DiskBackedEmbedding()


def install_qwen4_ple_embeddings(
    model: Any,
    *,
    weights_dir: str,
    external_tensors: Mapping[str, Any],
    source: str,
) -> Tuple[Any, ...]:
    """Replace every indexed PLE embedding and return closeable readers."""
    if source not in ("ram", "disk"):
        raise ValueError(f"Qwen4 PLE source must be 'ram' or 'disk'; got {source!r}")
    readers = []
    try:
        for parameter_name, spec in external_tensors.items():
            if not parameter_name.endswith(".weight"):
                raise ValueError(
                    f"external PLE key must name a weight: {parameter_name!r}"
                )
            module_name = parameter_name[: -len(".weight")]
            parent, attribute = _module_for_name(model, module_name)
            is_oq = hasattr(spec, "bits")
            if is_oq and source == "ram":
                raise ValueError(
                    "oQ PLE embeddings must use training.stream_ngram_source='disk': "
                    "the packed source is read-only and only requested rows are "
                    "dequantized"
                )
            if is_oq:
                reader = OQShardedSafeTensorRowReader(weights_dir, spec)
            elif source == "ram":
                reader = ShardedRamRowReader(weights_dir, spec)
            else:
                reader = ShardedSafeTensorRowReader(weights_dir, spec)
            readers.append(reader)
            replacement = _disk_embedding(reader)
            setattr(parent, attribute, replacement)
    except BaseException:
        for reader in readers:
            reader.close()
        raise
    return tuple(readers)


def external_tensor_bytes(external_tensors: Mapping[str, Any]) -> int:
    """Exact bytes actually resident or mapped for the PLE descriptors."""
    return sum(
        int(getattr(spec, "storage_nbytes", spec.nbytes))
        for spec in external_tensors.values()
    )
