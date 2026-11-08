"""Import a pre-quantised bitsandbytes NF4 checkpoint into the layer-stream shard store.

``unsloth/Qwen3-32B-bnb-4bit`` and its siblings ship the weights already quantised:
double-quantised NF4, one packed ``U8`` tensor per decoder projection plus its
statistics. Soup's sharder only quantises float sources itself, so this module copies
the published bytes into the shard layout WITHOUT dequantising or requantising anything:
every payload in the store is byte-for-byte the payload in the source.

The flow is plan, write, verify, commit:

1. :func:`plan_nf4_import` reads ``config.json`` and the safetensors HEADERS only,
   builds the expected tensor inventory from a meta skeleton (the quantised
   projections are whatever ``replace_with_bnb_linear`` converts, the same source of
   truth the trainer uses, so no architecture table lives here) and refuses anything
   it cannot carry over exactly, naming the tensor and the bad value.
2. :func:`import_nf4_checkpoint` copies payloads in bounded ``read()`` chunks. It never
   maps a file: on Windows a private mapping charges commit for the whole file (#926).
   Every shard is written to a temporary file and renamed.
3. The written shards are re-read and compared with the source byte for byte.
4. Only then is ``index.json`` written, so an interrupted import leaves a directory
   with no commit point.

The index records ``quant="nf4"``, ``double_quant=True`` and ``quant_device="serialized"``
(nothing was quantised on a device). Wiring this into the sharder and the trainer is
a separate change; nothing imports this module yet.

Bitsandbytes' serialisation, for a weight ``W`` (``QuantState.as_dict(packed=True)``):
``W`` (U8, ``[ceil(n/2), 1]``), ``W.absmax`` (U8, one per block under double
quantisation), ``W.nested_absmax`` (F32), ``W.quant_map`` (F32 x 16),
``W.nested_quant_map`` (F32 x 256) and ``W.quant_state.bitsandbytes__nf4`` (U8, a JSON
object holding the scalar fields, among them ``nested_offset`` as a Python float).

NO top-level torch, bitsandbytes or transformers import: they load inside the planner.
"""

import json
import math
import os
import struct
import tempfile
from dataclasses import dataclass
from typing import Any, BinaryIO, Callable, Dict, List, Mapping, Optional, Tuple

from soup_cli import __version__
from soup_cli.utils.terminal import for_terminal

#: Bytes per ``read()`` when payloads are copied or compared.
DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024
MAX_CHUNK_BYTES = 64 * 1024 * 1024
#: Block sizes accepted for the weight blocks and the nested (statistics) blocks.
#: bitsandbytes' 4-bit kernels also take 32, but no published checkpoint uses it and
#: the streaming kernels are exercised from 64 up, so it is refused by name.
ALLOWED_BLOCKSIZES = (64, 128, 256, 512, 1024, 2048, 4096)

_MAX_HEADER_BYTES = 16 * 1024 * 1024
_MAX_STATE_BYTES = 64 * 1024
_MAX_TENSORS = 200_000
_MAX_ELEMENTS = 2**40
_SCAN_BYTES = 64 * 1024

_STATE_SUFFIX = ".quant_state.bitsandbytes__nf4"
_FP4_STATE_SUFFIX = ".quant_state.bitsandbytes__fp4"
#: Key suffixes only bitsandbytes' 8-bit (LLM.int8) serialisation writes.
_INT8_SUFFIXES = (".SCB", ".weight_format")
_BASE_FIELDS = frozenset({"quant_type", "blocksize", "dtype", "shape"})
_NESTED_FIELDS = frozenset({"nested_blocksize", "nested_dtype", "nested_offset"})
_NF4_CODE_LENGTH = 16
_NESTED_CODE_LENGTH = 256

_ITEMSIZE = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E5M2": 1, "F8_E4M3": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}  # fmt: skip
#: Tensors in a shard are laid out widest dtype first, so every payload starts aligned.
_ALIGN = {"F32": 4, "BF16": 2, "U8": 1}
#: Architectures whose layers carry experts or per-layer buffers this importer does not map.
_UNSUPPORTED_ARCHS = ("deepseek_v3", "qwen4_exp")


# ==========================================================================
# data
# ==========================================================================
@dataclass(frozen=True)
class _Span:
    """One tensor payload: a byte range in a source file, or a few synthesised bytes."""

    path: str
    key: str
    dtype: str
    shape: Tuple[int, ...]
    offset: int
    nbytes: int
    inline: Optional[bytes] = None


@dataclass(frozen=True)
class Nf4ImportPlan:
    """A fully validated import: nothing has been written when this exists."""

    index: Any
    #: shard file name -> {stored key -> payload}
    shards: Mapping[str, Mapping[str, _Span]]
    #: source file path -> (size, mtime_ns) taken before any header was read
    source_identity: Mapping[str, Tuple[int, int]]
    tensor_count: int
    payload_bytes: int


@dataclass(frozen=True)
class Nf4ImportResult:
    """What an import (or a re-verification) found."""

    index: Any
    #: tensors written to shard files, including the synthesised offsets and code tables
    tensor_count: int
    #: total bytes of those tensors
    payload_bytes: int
    layer_count: int
    #: every payload was re-read from the shards and compared with the source
    verified: bool


# ==========================================================================
# strict source reading
# ==========================================================================
def _label(text: object) -> str:
    return for_terminal(text)


def _object_pairs(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _bad_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON value {value}")


def _json(raw: bytes) -> Dict[str, Any]:
    try:
        result = json.loads(raw, object_pairs_hook=_object_pairs, parse_constant=_bad_constant)
    except RecursionError as exc:
        raise ValueError("JSON is nested too deeply") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"JSON is not valid UTF-8 ({exc})") from exc
    if not isinstance(result, dict):
        raise ValueError("expected a JSON object")
    return result


def _read_json_file(path: str, what: str) -> Dict[str, Any]:
    if os.path.getsize(path) > _MAX_HEADER_BYTES:
        raise ValueError(f"{what} is larger than {_MAX_HEADER_BYTES} bytes; refusing to read it")
    with open(path, "rb") as handle:
        raw = handle.read(_MAX_HEADER_BYTES + 1)
    try:
        return _json(raw)
    except ValueError as exc:
        raise ValueError(f"{what} is not usable: {exc}") from exc


def _contained_file(root: str, name: object, what: str) -> str:
    """The real path of ``name`` inside ``root``; a path escape or a link is refused."""
    plain = (
        isinstance(name, str)
        and name not in ("", ".", "..")
        and not any(ord(char) < 32 for char in name)
        and os.path.basename(name) == name
        and not any(sep in name for sep in ("/", "\\", ":"))
    )
    if not plain:
        raise ValueError(
            f"{what} names {_label(repr(name))}, which is not a plain file name inside the "
            f"checkpoint directory (path escape refused)"
        )
    path = os.path.join(root, name)
    if os.path.islink(path):
        raise ValueError(f"{what}: {_label(name)} is a symlink; refusing to follow it")
    real = os.path.realpath(path)
    if os.path.commonpath((root, real)) != root:
        raise ValueError(f"{what}: {_label(name)} resolves outside the checkpoint directory")
    if not os.path.isfile(real):
        raise FileNotFoundError(f"{what}: {_label(name)} not found in the checkpoint directory")
    return real


def _read_header(path: str) -> Dict[str, _Span]:
    """Validate a safetensors layout completely, without mapping or loading it."""
    name = _label(os.path.basename(path))
    size = os.path.getsize(path)
    with open(path, "rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{name}: truncated safetensors file (no header length)")
        (length,) = struct.unpack("<Q", prefix)
        if not 2 <= length <= _MAX_HEADER_BYTES or 8 + length > size:
            raise ValueError(
                f"{name}: truncated or corrupt safetensors header (declares {length} header "
                f"bytes in a {size}-byte file)"
            )
        try:
            header = _json(handle.read(length))
        except ValueError as exc:
            raise ValueError(f"{name}: safetensors header is not usable: {exc}") from exc
    if len(header) > _MAX_TENSORS:
        raise ValueError(f"{name}: more than {_MAX_TENSORS} tensors")
    body = size - 8 - length
    spans: Dict[str, _Span] = {}
    ranges: List[Tuple[int, int]] = []
    for key, item in header.items():
        if key == "__metadata__":
            continue
        where = f"{name}: tensor {_label(key)}"
        if not isinstance(item, dict) or set(item) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"{where} has an invalid header entry")
        dtype, shape, offsets = item["dtype"], item["shape"], item["data_offsets"]
        if not isinstance(dtype, str) or dtype not in _ITEMSIZE:
            raise ValueError(f"{where} has unsupported stored dtype {_label(repr(dtype))}")
        if (
            not isinstance(shape, list)
            or len(shape) > 8
            or any(type(dim) is not int or dim < 0 for dim in shape)
        ):
            raise ValueError(f"{where} has an invalid shape {_label(repr(shape))}")
        elements = math.prod(shape)
        if elements > _MAX_ELEMENTS:
            raise ValueError(f"{where} declares {elements} elements, above the bound")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(value) is not int for value in offsets)
        ):
            raise ValueError(f"{where} has invalid data_offsets")
        start, end = offsets
        nbytes = elements * _ITEMSIZE[dtype]
        if start < 0 or end - start != nbytes:
            raise ValueError(f"{where}: byte range {offsets} does not hold {nbytes} bytes")
        if end > body:
            raise ValueError(
                f"{where}: file is truncated (tensor ends at byte {end}, the data region "
                f"holds {body})"
            )
        ranges.append((start, end))
        spans[key] = _Span(path, key, dtype, tuple(shape), 8 + length + start, nbytes)
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError(f"{name}: tensor byte ranges overlap or leave a gap")
        cursor = end
    if cursor != body:
        raise ValueError(f"{name}: {body - cursor} bytes of unindexed trailing data")
    return spans


def _read_span(span: _Span) -> bytes:
    """A SMALL payload (a JSON blob or a code table), read whole."""
    if span.inline is not None:
        return span.inline
    with open(span.path, "rb") as handle:
        handle.seek(span.offset)
        raw = handle.read(span.nbytes)
    if len(raw) != span.nbytes:
        raise ValueError(f"{_label(span.key)}: source truncated while reading it")
    return raw


def _need(span: _Span, dtype: str, shape: Tuple[int, ...], key: str, why: str = "") -> None:
    if span.dtype != dtype or span.shape != tuple(shape):
        raise ValueError(
            f"{_label(key)}: stored as {span.dtype} {list(span.shape)}, but a byte-for-byte "
            f"import needs {dtype} {list(shape)}{why}"
        )


def _scan_scales(span: _Span, key: str) -> None:
    """The nested scales are absolute maxima: NaN, infinite and negative ones are corrupt."""
    remaining = span.nbytes
    with open(span.path, "rb") as handle:
        handle.seek(span.offset)
        while remaining:
            raw = handle.read(min(remaining, _SCAN_BYTES))
            if not raw:
                raise ValueError(f"{_label(key)}: source truncated while reading it")
            for (value,) in struct.iter_unpack("<f", raw):
                if not math.isfinite(value) or value < 0:
                    raise ValueError(
                        f"{_label(key)}: nested scale {value!r} is not a finite, non-negative "
                        f"number"
                    )
            remaining -= len(raw)


# ==========================================================================
# plan
# ==========================================================================
def _check_config_quantization(config: Mapping[str, Any]) -> None:
    quant = config.get("quantization_config")
    if not isinstance(quant, dict):
        raise ValueError(
            "config.json has no quantization_config: this is not a bitsandbytes checkpoint"
        )
    method = quant.get("quant_method")
    if method != "bitsandbytes":
        raise ValueError(
            f"config.json quantization_config.quant_method is {_label(repr(method))}; only "
            f"'bitsandbytes' NF4 checkpoints are imported"
        )
    if quant.get("load_in_8bit") is True:
        raise ValueError(
            "config.json quantization_config.load_in_8bit is true: bitsandbytes 8-bit "
            "(LLM.int8) checkpoints are not imported, only double-quantised NF4"
        )
    if quant.get("load_in_4bit") is not True:
        raise ValueError(
            f"config.json quantization_config.load_in_4bit is "
            f"{_label(repr(quant.get('load_in_4bit')))}; a 4-bit checkpoint is required"
        )
    expected = (
        ("bnb_4bit_quant_type", "nf4", quant.get("bnb_4bit_quant_type")),
        ("bnb_4bit_use_double_quant", True, quant.get("bnb_4bit_use_double_quant")),
        ("bnb_4bit_quant_storage", "uint8", quant.get("bnb_4bit_quant_storage", "uint8")),
    )
    for field, wanted, got in expected:
        if got != wanted:
            raise ValueError(
                f"config.json quantization_config.{field} is {_label(repr(got))}; "
                f"{wanted!r} is required"
            )


def _build_expectations(root: str) -> Tuple[str, int, Dict[str, Tuple[bool, Tuple[int, ...]]]]:
    """``(arch, n_layers, {name: (is_nf4, logical shape)})`` read off the meta skeleton."""
    try:
        from soup_cli.utils.layer_shard import _LAYER_RE, layer_buffer_keys
        from soup_cli.utils.layer_stream import stream_arch_of
        from soup_cli.utils.layer_stream_runtime import (
            build_meta_skeleton,
            quantised_layer_suffixes,
        )

        model = build_meta_skeleton(root, dtype="bfloat16", quant="nf4", double_quant=True)
    except ImportError as exc:
        raise ImportError(
            f"importing a pre-quantised checkpoint needs bitsandbytes, transformers and "
            f"accelerate (pip install 'soup-cli[train]'): {exc}"
        ) from exc
    arch = stream_arch_of(model.config)
    if arch in _UNSUPPORTED_ARCHS or layer_buffer_keys(arch):
        raise ValueError(
            f"architecture {arch!r} is not imported from a serialised NF4 checkpoint "
            f"(dense decoder architectures only)"
        )
    suffixes = quantised_layer_suffixes(model)
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    if not suffixes:
        raise ValueError("the model has no decoder projections that bitsandbytes quantises")
    expectations: Dict[str, Tuple[bool, Tuple[int, ...]]] = {}
    layer_ids = set()
    for name, param in model.named_parameters():
        match = _LAYER_RE.match(name)
        shape = tuple(int(dim) for dim in param.shape)
        if match is None:
            expectations[name] = (False, shape)
            continue
        layer_ids.add(int(match.group(1)))
        is_nf4 = match.group(2) in suffixes
        if is_nf4 and len(shape) != 2:
            raise ValueError(f"{_label(name)}: a quantised projection must be 2-D, got {shape}")
        expectations[name] = (is_nf4, shape)
    if tied:
        # A tied checkpoint stores the embedding once; the head is restored from it.
        expectations.pop("lm_head.weight", None)
    n_layers = len(layer_ids)
    if not layer_ids or layer_ids != set(range(n_layers)):
        raise ValueError("the model's decoder layers are not numbered 0..N-1")
    return arch, n_layers, expectations


def _parse_state(span: _Span, key: str, shape: Tuple[int, ...]) -> Tuple[Any, bytes]:
    """Validate one ``quant_state`` blob. Returns ``(NF4WeightSpec, FP32 nested_offset)``."""
    from soup_cli.utils.layer_shard import NF4WeightSpec

    name = _label(key)
    if span.dtype != "U8" or len(span.shape) != 1:
        raise ValueError(
            f"{name}{_STATE_SUFFIX}: the quant_state blob must be a 1-D U8 tensor, got "
            f"{span.dtype} {list(span.shape)}"
        )
    if span.nbytes > _MAX_STATE_BYTES:
        raise ValueError(f"{name}{_STATE_SUFFIX}: quant_state blob is {span.nbytes} bytes")
    try:
        payload = _json(_read_span(span))
    except ValueError as exc:
        raise ValueError(f"{name}{_STATE_SUFFIX}: quant_state JSON is invalid: {exc}") from exc
    fields = set(payload)
    absent_nested = _NESTED_FIELDS - fields
    if absent_nested:
        raise ValueError(
            f"{name}: not double-quantised (quant_state lacks {', '.join(sorted(absent_nested))}); "
            f"bnb_4bit_use_double_quant=True is required"
        )
    if fields != _BASE_FIELDS | _NESTED_FIELDS:
        raise ValueError(
            f"{name}: quant_state has unexpected fields "
            f"{sorted(fields - _BASE_FIELDS - _NESTED_FIELDS)} or lacks "
            f"{sorted((_BASE_FIELDS | _NESTED_FIELDS) - fields)}"
        )
    if payload["quant_type"] != "nf4":
        raise ValueError(
            f"{name}: quant_type is {_label(repr(payload['quant_type']))}; only 'nf4' is imported"
        )
    if payload["dtype"] != "bfloat16":
        raise ValueError(
            f"{name}: dtype is {_label(repr(payload['dtype']))}; only 'bfloat16' is imported"
        )
    if payload["nested_dtype"] != "float32":
        raise ValueError(
            f"{name}: nested_dtype is {_label(repr(payload['nested_dtype']))}; "
            f"only 'float32' is imported"
        )
    for field in ("blocksize", "nested_blocksize"):
        value = payload[field]
        if type(value) is not int or value not in ALLOWED_BLOCKSIZES:
            raise ValueError(
                f"{name}: {field} is {_label(repr(value))}; allowed: "
                f"{', '.join(str(size) for size in ALLOWED_BLOCKSIZES)}"
            )
    if payload["shape"] != list(shape) or any(type(dim) is not int for dim in payload["shape"]):
        raise ValueError(
            f"{name}: quant_state shape is {_label(repr(payload['shape']))}, but the model "
            f"declares {list(shape)}"
        )
    value = payload["nested_offset"]
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name}: nested_offset {_label(repr(value))} is not a finite number")
    try:
        offset = struct.pack("<f", value)
    except (OverflowError, struct.error) as exc:
        raise ValueError(f"{name}: nested_offset {value!r} does not fit in float32") from exc
    if struct.unpack("<f", offset)[0] != value:
        raise ValueError(f"{name}: nested_offset {value!r} is not exactly representable in float32")
    spec = NF4WeightSpec.from_json(
        {
            "shape": list(shape),
            "dtype": "bfloat16",
            "blocksize": payload["blocksize"],
            "quant_type": "nf4",
            "nested": True,
            "nested_blocksize": payload["nested_blocksize"],
        }
    )
    return spec, offset


def _check_native_nf4_table(span: _Span, stored: bytes) -> None:
    """The stored ``quant_map`` must be bitsandbytes' own NF4 table, bit for bit.

    The CUDA dequantise kernel ignores the stored map and uses its built-in NF4 table, so a
    different (even self-consistent) table would decode differently on CPU and on GPU.
    """
    from bitsandbytes.functional import get_4bit_type

    native = get_4bit_type("nf4", device="cpu").to("cpu").float().tolist()
    if stored != struct.pack("<%df" % len(native), *native):
        raise ValueError(
            f"{_label(span.key)}: the stored NF4 code table is not bitsandbytes' own NF4 "
            f"table (bit for bit); the CUDA kernels ignore the stored map, so it would "
            f"decode differently on CPU and on GPU"
        )


def _shard_name(path: str) -> str:
    return os.path.basename(path)


def plan_nf4_import(source_dir: str) -> Nf4ImportPlan:
    """Validate a serialised bitsandbytes NF4 checkpoint and plan its import. Writes nothing."""
    from soup_cli.utils.layer_shard import (
        _LAYER_RE,
        ABSMAX_SUFFIX,
        NESTED_ABSMAX_SUFFIX,
        NESTED_OFFSET_SUFFIX,
        NF4_CODE_KEY,
        NF4_NESTED_CODE_KEY,
        ShardIndex,
        _canonical_stream_key,
        _discover_safetensors,
        _source_file_components,
        checkpoint_source_components,
        extras_shard_path,
        fingerprint_source_files,
        large_shard_path,
        large_weight_role,
        layer_shard_path,
        source_identity_includes_config,
    )

    root = os.path.realpath(os.path.expanduser(str(source_dir)))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"checkpoint directory not found: {_label(source_dir)}")
    config = _read_json_file(_contained_file(root, "config.json", "checkpoint"), "config.json")
    _check_config_quantization(config)

    shard_paths = _discover_safetensors(root)
    components = _source_file_components(shard_paths)
    identity = {
        path: (size, mtime)
        for path, (_name, size, mtime) in zip(sorted(shard_paths), components)
    }
    raw_tensors: Dict[str, _Span] = {}
    for path in shard_paths:
        for key, span in _read_header(path).items():
            prior = raw_tensors.get(key)
            if prior is not None:
                raise ValueError(
                    f"tensor {_label(key)} appears in both {_label(_shard_name(prior.path))} "
                    f"and {_label(_shard_name(path))}; every tensor must appear exactly once"
                )
            raw_tensors[key] = span
    _check_weight_map(root, raw_tensors)

    source: Dict[str, _Span] = {}
    for key, span in raw_tensors.items():
        canonical = _canonical_stream_key(key)
        if canonical in source:
            raise ValueError(f"tensors spelled two ways both map to {_label(canonical)}")
        source[canonical] = span
    for key in sorted(source):
        if key.endswith(_INT8_SUFFIXES):
            raise ValueError(
                f"{_label(key)}: bitsandbytes 8-bit (LLM.int8) tensor; only double-quantised "
                f"NF4 is imported"
            )
        if key.endswith(_FP4_STATE_SUFFIX):
            raise ValueError(
                f"{_label(key)}: FP4 quantisation (bitsandbytes__fp4) is not imported; "
                f"only NF4 is"
            )

    arch, n_layers, expectations = _build_expectations(root)
    wanted = set()
    for name, (is_nf4, _shape) in expectations.items():
        wanted.add(name)
        if is_nf4:
            wanted.update(
                name + suffix
                for suffix in (".absmax", ".nested_absmax", ".quant_map", ".nested_quant_map")
            )
            wanted.add(name + _STATE_SUFFIX)
    unexpected = sorted(set(source) - wanted)
    if unexpected:
        raise ValueError(
            f"checkpoint holds {len(unexpected)} tensor(s) the model does not have (or an "
            f"unexpected quantised key): {', '.join(_label(key) for key in unexpected[:5])}"
        )

    untied = "model.embed_tokens.weight" in expectations and "lm_head.weight" in expectations
    layers: List[Dict[str, _Span]] = [{} for _ in range(n_layers)]
    extras: Dict[str, _Span] = {}
    large: Dict[str, _Span] = {}
    specs: Dict[str, Any] = {}
    layout: Dict[str, Tuple[str, Tuple[int, ...]]] = {}
    tables: Dict[str, _Span] = {}
    table_bytes: Dict[str, bytes] = {}
    missing: List[str] = []
    total_params = 0
    for name in sorted(expectations):
        is_nf4, shape = expectations[name]
        match = _LAYER_RE.match(name)
        if not is_nf4:
            span = source.get(name)
            if span is None:
                missing.append(name)
                continue
            _need(span, "BF16", shape, name, " (the store holds the stream dtype)")
            total_params += math.prod(shape)
            if match is not None:
                short = match.group(2)
                if layout.setdefault(short, ("BF16", shape)) != ("BF16", shape):
                    raise ValueError(
                        f"{_label(name)}: shape differs from the same weight in other layers"
                    )
                layers[int(match.group(1))][short] = span
            elif untied and large_weight_role(name) is not None:
                large[name] = span
            else:
                extras[name] = span
            continue

        state_span = source.get(name + _STATE_SUFFIX)
        if state_span is None:
            missing.append(name + _STATE_SUFFIX)
            continue
        spec, offset = _parse_state(state_span, name, shape)
        sidecars = {
            suffix: source.get(name + suffix)
            for suffix in (".absmax", ".nested_absmax", ".quant_map", ".nested_quant_map")
        }
        gaps = [name + suffix for suffix, span in sidecars.items() if span is None]
        if name not in source:
            gaps.insert(0, name)
        if gaps:
            missing.extend(gaps)
            continue
        elements = math.prod(shape)
        blocks = -(-elements // spec.blocksize)
        nested_blocks = -(-blocks // spec.nested_blocksize)
        _need(source[name], "U8", ((elements + 1) // 2, 1), name, " (packed NF4 nibbles)")
        _need(sidecars[".absmax"], "U8", (blocks,), name + ".absmax")
        _need(sidecars[".nested_absmax"], "F32", (nested_blocks,), name + ".nested_absmax")
        _need(sidecars[".quant_map"], "F32", (_NF4_CODE_LENGTH,), name + ".quant_map")
        _need(
            sidecars[".nested_quant_map"], "F32", (_NESTED_CODE_LENGTH,), name + ".nested_quant_map"
        )
        _scan_scales(sidecars[".nested_absmax"], name + ".nested_absmax")
        for suffix, target in (
            (".quant_map", NF4_CODE_KEY),
            (".nested_quant_map", NF4_NESTED_CODE_KEY),
        ):
            table = sidecars[suffix]
            raw = _read_span(table)
            if not all(math.isfinite(value) for (value,) in struct.iter_unpack("<f", raw)):
                raise ValueError(f"{_label(name + suffix)}: code table holds a non-finite value")
            if table_bytes.setdefault(target, raw) != raw:
                raise ValueError(
                    f"{_label(name + suffix)}: code table differs from the one in other "
                    f"projections; Soup keeps ONE shared table"
                )
            tables.setdefault(target, table)
        short = match.group(2)
        if specs.setdefault(short, spec) != spec:
            raise ValueError(f"{_label(name)}: quantisation spec differs from other layers")
        total_params += elements
        blob = layers[int(match.group(1))]
        blob[short] = source[name]
        blob[short + ABSMAX_SUFFIX] = sidecars[".absmax"]
        blob[short + NESTED_ABSMAX_SUFFIX] = sidecars[".nested_absmax"]
        blob[short + NESTED_OFFSET_SUFFIX] = _Span(
            "", name + _STATE_SUFFIX + ":nested_offset", "F32", (), 0, 4, offset
        )
    if missing:
        raise ValueError(
            f"checkpoint is missing {len(missing)} expected tensor(s): "
            f"{', '.join(_label(key) for key in missing[:5])}"
        )
    _check_native_nf4_table(tables[NF4_CODE_KEY], table_bytes[NF4_CODE_KEY])
    extras.update(tables)

    shards: Dict[str, Dict[str, _Span]] = {
        _shard_name(layer_shard_path("", idx)): blob for idx, blob in enumerate(layers)
    }
    for key, span in large.items():
        shards[_shard_name(large_shard_path("", key))] = {key: span}
    shards[_shard_name(extras_shard_path(""))] = extras
    source_files = checkpoint_source_components(
        root, components, include_config=source_identity_includes_config(arch)
    )
    index = ShardIndex(
        n_layers=n_layers,
        layer_keys=tuple(sorted({key for blob in layers for key in blob})),
        extra_keys=tuple(sorted(extras)),
        dtype="bfloat16",
        total_params=total_params,
        arch=arch,
        soup_version=__version__,
        large_keys=tuple(sorted(large)),
        source_fingerprint=fingerprint_source_files(source_files),
        source_files=source_files,
        quant="nf4",
        double_quant=True,
        quant_device="serialized",
        quant_specs=specs,
    )
    spans = [span for blob in shards.values() for span in blob.values()]
    return Nf4ImportPlan(
        index=index,
        shards=shards,
        source_identity=identity,
        tensor_count=len(spans),
        payload_bytes=sum(span.nbytes for span in spans),
    )


def _check_weight_map(root: str, tensors: Mapping[str, _Span]) -> None:
    """``model.safetensors.index.json``, when present, must agree with the file headers."""
    name = "model.safetensors.index.json"
    if not os.path.lexists(os.path.join(root, name)):
        return
    weight_map = _read_json_file(_contained_file(root, name, "checkpoint"), name).get("weight_map")
    if not isinstance(weight_map, dict) or not 0 < len(weight_map) <= _MAX_TENSORS:
        raise ValueError(f"{name} has no usable weight_map")
    for key, file_name in weight_map.items():
        _contained_file(root, file_name, f"{name} weight_map")
        span = tensors.get(key)
        if span is None or _shard_name(span.path) != file_name:
            raise ValueError(
                f"{name} maps {_label(key)} to {_label(file_name)}, whose header does not hold it"
            )
    if set(weight_map) != set(tensors):
        extra = sorted(set(tensors) - set(weight_map))
        raise ValueError(f"{name} does not list {len(extra)} tensor(s), e.g. {_label(extra[0])}")


# ==========================================================================
# write
# ==========================================================================
def _check_chunk(chunk_bytes: int) -> int:
    if type(chunk_bytes) is not int or not 1 <= chunk_bytes <= MAX_CHUNK_BYTES:
        raise ValueError(
            f"chunk_bytes must be an integer in [1, {MAX_CHUNK_BYTES}]; got {chunk_bytes!r}"
        )
    return chunk_bytes


def _copy_range(source: BinaryIO, target: BinaryIO, span: _Span, chunk_bytes: int) -> None:
    """Copy one payload with bounded ``read()`` calls: never more than ``chunk_bytes``."""
    source.seek(span.offset)
    remaining = span.nbytes
    while remaining:
        raw = source.read(min(chunk_bytes, remaining))
        if not raw:
            raise ValueError(f"{_label(span.key)}: source truncated while copying it")
        target.write(raw)
        remaining -= len(raw)


def _write_shard(path: str, spans: Mapping[str, _Span], chunk_bytes: int) -> None:
    order = sorted(spans.items(), key=lambda item: (-_ALIGN[item[1].dtype], item[0]))
    header: Dict[str, Any] = {}
    cursor = 0
    for key, span in order:
        header[key] = {
            "dtype": span.dtype,
            "shape": list(span.shape),
            "data_offsets": [cursor, cursor + span.nbytes],
        }
        cursor += span.nbytes
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".soup.", suffix=".tmp", dir=os.path.dirname(path)
    )
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(struct.pack("<Q", len(encoded)))
            target.write(encoded)
            for _key, span in order:
                if span.inline is not None:
                    target.write(span.inline)
                else:
                    with open(span.path, "rb") as source:
                        _copy_range(source, target, span, chunk_bytes)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _assert_sources_unchanged(plan: Nf4ImportPlan) -> None:
    for path, identity in plan.source_identity.items():
        stat = os.stat(path)
        if (int(stat.st_size), int(stat.st_mtime_ns)) != identity:
            raise ValueError(f"source file {_label(path)} changed during the import")


def _verify_plan(
    plan: Nf4ImportPlan, out_dir: str, chunk_bytes: int, notify: Optional[Callable[[str], None]]
) -> None:
    """Re-read every written shard and compare each payload with the source, byte for byte."""
    for file_name, spans in plan.shards.items():
        path = os.path.join(out_dir, file_name)
        written = _read_header(path)
        if set(written) != set(spans):
            raise ValueError(f"{file_name}: tensors in the shard differ from the plan")
        with open(path, "rb") as stored:
            for key, span in spans.items():
                entry = written[key]
                layout = (entry.dtype, entry.shape, entry.nbytes)
                if layout != (span.dtype, span.shape, span.nbytes):
                    raise ValueError(
                        f"{file_name}: {_label(key)} was written with a different layout"
                    )
                _compare(stored, entry.offset, span, chunk_bytes, file_name)
        if notify:
            notify(f"Verified {file_name}")


def _compare(stored: BinaryIO, start: int, span: _Span, chunk_bytes: int, file_name: str) -> None:
    stored.seek(start)
    source: Optional[BinaryIO] = None
    try:
        if span.inline is None:
            source = open(span.path, "rb")
            source.seek(span.offset)
        position = 0
        while position < span.nbytes:
            amount = min(chunk_bytes, span.nbytes - position)
            got = stored.read(amount)
            want = (
                span.inline[position : position + amount]
                if span.inline is not None
                else source.read(amount)
            )
            if got != want:
                raise ValueError(
                    f"{file_name}: {_label(span.key)} differs from the source at payload byte "
                    f"{position}"
                )
            position += amount
    finally:
        if source is not None:
            source.close()


def import_nf4_checkpoint(
    source_dir: str,
    output_dir: str,
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    verify: bool = True,
    notify: Optional[Callable[[str], None]] = None,
) -> Nf4ImportResult:
    """Import ``source_dir`` into a NEW (or empty) shard directory, byte for byte.

    Everything is validated before the first byte is written. The index is written last,
    after the optional byte-for-byte verification, so a failed import has no commit point
    and its shard files are removed again.
    """
    from soup_cli.utils.layer_shard import (
        _atomic_write_index,
        _validate_out_dir,
        extras_shard_path,
    )

    _check_chunk(chunk_bytes)
    plan = plan_nf4_import(source_dir)
    source_root = os.path.realpath(os.path.expanduser(str(source_dir)))
    target = os.path.realpath(os.path.expanduser(str(output_dir)))
    if os.path.commonpath((source_root, target)) in (source_root, target):
        raise ValueError("source and output directories must not contain each other")
    out = _validate_out_dir(output_dir)
    if os.listdir(out):
        raise FileExistsError(
            f"output directory {_label(output_dir)} is not empty; import into a new directory"
        )
    written: List[str] = []
    try:
        for file_name, spans in plan.shards.items():
            _write_shard(os.path.join(out, file_name), spans, chunk_bytes)
            written.append(file_name)
            if notify and file_name != _shard_name(extras_shard_path("")):
                notify(f"Imported {file_name}")
        _assert_sources_unchanged(plan)
        if verify:
            _verify_plan(plan, out, chunk_bytes, None)
        _atomic_write_index(plan.index, out)
    except BaseException:
        for file_name in written:
            try:
                os.unlink(os.path.join(out, file_name))
            except OSError:
                pass
        raise
    return Nf4ImportResult(
        index=plan.index,
        tensor_count=plan.tensor_count,
        payload_bytes=plan.payload_bytes,
        layer_count=plan.index.n_layers,
        verified=bool(verify),
    )


def verify_nf4_import(
    source_dir: str, output_dir: str, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES
) -> Nf4ImportResult:
    """Re-check a finished import: re-plan from the source, re-read the store, compare bytes.

    The plan is rebuilt from the source files, not taken from the earlier import, and the
    index on disk must equal the one that plan produces.
    """
    from soup_cli.utils.layer_shard import read_shard_index

    _check_chunk(chunk_bytes)
    plan = plan_nf4_import(source_dir)
    out = os.path.realpath(os.path.expanduser(str(output_dir)))
    if not os.path.isdir(out):
        raise FileNotFoundError(f"shard directory not found: {_label(output_dir)}")
    _verify_plan(plan, out, chunk_bytes, None)
    stored = read_shard_index(out)
    if stored != plan.index:
        raise ValueError("index.json does not describe this source checkpoint")
    return Nf4ImportResult(
        index=stored,
        tensor_count=plan.tensor_count,
        payload_bytes=plan.payload_bytes,
        layer_count=stored.n_layers,
        verified=True,
    )
