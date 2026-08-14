"""training.stream_layers — checkpoint sharder (v0.72.0 BETA).

Rewrites an HF checkpoint into one ``layer_NNN.safetensors`` per decoder layer
plus a single ``extras.safetensors`` (embeddings / final norm / untied head),
so the runtime can stream exactly one layer at a time.

Built on the ``utils/spectrum_scan.iter_weight_matrices`` pattern: ``safe_open``
per source shard, one tensor materialised at a time, symlinked shards skipped,
element caps enforced. Peak RSS while sharding is ONE decoder layer, not the
model — which matters, because the whole point is that the model does not fit.

v0.72.2 adds NF4: the decoder linears named in ``quant_suffixes`` are quantised
OFFLINE, one tensor at a time, and stored as packed ``uint8`` + per-block
``absmax`` (+ the nested absmax / offset under double quantisation).
``Params4bit`` carries a ``quant_state`` and cannot be byte-copied into a plain
buffer (plan P3), so the runtime rebuilds the views over the pooled buffer from
exactly these tensors.

**No top-level torch / safetensors** — both are ``[train]``-extra deps and this
module sits on the light CLI's import path.
"""

import contextlib
import json
import logging
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from soup_cli import __version__

logger = logging.getLogger(__name__)

#: A decoder-layer parameter key: ``model.layers.<idx>.<rest>``.
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]")

_SUPPORTED_DTYPES = ("bfloat16", "float16", "float32")

# --- NF4 (v0.72.2) --------------------------------------------------------
QUANT_NONE = "none"
QUANT_NF4 = "nf4"
#: int8 rowwise is deliberately out of scope — NF4 only (plan 7.1).
SUPPORTED_STREAM_QUANTS = (QUANT_NONE, QUANT_NF4)

#: Sidecar keys for one quantised weight. ``::`` cannot collide with a real
#: parameter path, which uses ``.`` exclusively.
ABSMAX_SUFFIX = "::absmax"
NESTED_ABSMAX_SUFFIX = "::nested_absmax"
NESTED_OFFSET_SUFFIX = "::nested_offset"

#: The NF4 code table (16 fp32) and the nested code table (256 fp32) are
#: CONSTANT across every weight, so one shared resident copy is safe. The
#: sharder asserts that rather than assuming it.
NF4_CODE_KEY = "__nf4_code"
NF4_NESTED_CODE_KEY = "__nf4_nested_code"

NF4_BLOCKSIZE = 64
#: bitsandbytes' own 4-bit types. Read-back values are checked against this.
_SUPPORTED_QUANT_TYPES = ("nf4", "fp4")
#: An index claiming a wilder blocksize than bitsandbytes supports is corrupt.
_MAX_BLOCKSIZE = 4096

#: Refuse absurd checkpoints rather than thrash (mirrors spectrum_scan caps).
_MAX_LAYERS = 512
_MAX_TENSOR_ELEMENTS = 2**31
_MAX_SHARD_FILES = 4096
_MAX_TOTAL_TENSORS = 200_000

_INDEX_NAME = "index.json"
_EXTRAS_NAME = "extras.safetensors"


# ==========================================================================
# index
# ==========================================================================
@dataclass(frozen=True)
class NF4WeightSpec:
    """Everything needed to rebuild one weight's ``QuantState`` at runtime.

    ``shape`` / ``dtype`` describe the DEQUANTISED weight, not the packed
    bytes: ``matmul_4bit`` needs the logical shape to reconstruct.
    """

    shape: Tuple[int, ...]
    dtype: str
    blocksize: int
    quant_type: str
    nested: bool
    nested_blocksize: int

    def to_json(self) -> Dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "blocksize": self.blocksize,
            "quant_type": self.quant_type,
            "nested": self.nested,
            "nested_blocksize": self.nested_blocksize,
        }

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "NF4WeightSpec":
        """Rebuild from ``index.json`` — VALIDATING, because this is a
        trust boundary.

        These fields are read back off disk and handed to bitsandbytes'
        dequantise kernels, which allocate and read ``prod(shape)`` elements
        from the packed buffer WITHOUT bounds-checking it. A corrupted or
        tampered index must therefore fail here, as a clean Python exception,
        rather than as an out-of-bounds read in native code.
        """
        shape = tuple(int(dim) for dim in payload["shape"])
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError(
                f"NF4 weight spec shape must be non-empty with positive dims; got {shape}"
            )
        elements = math.prod(shape)
        if elements > _MAX_TENSOR_ELEMENTS:
            raise ValueError(
                f"NF4 weight spec shape {shape} is too large "
                f"({elements} elements > {_MAX_TENSOR_ELEMENTS})"
            )
        blocksize = int(payload["blocksize"])
        if blocksize <= 0 or blocksize > _MAX_BLOCKSIZE:
            raise ValueError(
                f"NF4 weight spec blocksize must be in [1, {_MAX_BLOCKSIZE}]; got {blocksize}"
            )
        quant_type = str(payload["quant_type"])
        if quant_type not in _SUPPORTED_QUANT_TYPES:
            raise ValueError(
                f"unsupported NF4 quant_type {quant_type!r}; supported: "
                f"{', '.join(_SUPPORTED_QUANT_TYPES)}"
            )
        dtype = str(payload["dtype"])
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"unsupported NF4 weight spec dtype {dtype!r}; supported: "
                f"{', '.join(_SUPPORTED_DTYPES)}"
            )
        nested = bool(payload["nested"])
        nested_blocksize = int(payload["nested_blocksize"])
        if nested and (nested_blocksize <= 0 or nested_blocksize > _MAX_BLOCKSIZE):
            raise ValueError(
                f"nested NF4 weight spec needs nested_blocksize in "
                f"[1, {_MAX_BLOCKSIZE}]; got {nested_blocksize}"
            )
        return cls(
            shape=shape,
            dtype=dtype,
            blocksize=blocksize,
            quant_type=quant_type,
            nested=nested,
            nested_blocksize=nested_blocksize,
        )


@dataclass(frozen=True)
class ShardIndex:
    """What the sharder produced — the runtime's contract with the cache."""

    n_layers: int
    layer_keys: Tuple[str, ...]
    extra_keys: Tuple[str, ...]
    dtype: str
    total_params: int
    arch: str
    soup_version: str
    source_fingerprint: str = ""
    quant: str = QUANT_NONE
    double_quant: bool = False
    #: "cuda" / "cpu" — the device the offline quantisation ran on. CPU and CUDA
    #: agree on the packed nibbles but not on every float32 nested statistic, so
    #: reusing a CPU-quantised cache for a CUDA run would break bit-exactness
    #: against a resident load. Today dtype happens to co-vary with device,
    #: which would mask this; keying on it explicitly means the protection does
    #: not depend on that coincidence.
    quant_device: str = ""
    #: per-layer short key -> spec. Empty when ``quant == "none"``.
    quant_specs: Mapping[str, NF4WeightSpec] = field(default_factory=dict)


def layer_shard_path(out_dir: str, idx: int) -> str:
    return os.path.join(out_dir, f"layer_{idx:03d}.safetensors")


def extras_shard_path(out_dir: str) -> str:
    return os.path.join(out_dir, _EXTRAS_NAME)


def read_shard_index(out_dir: str) -> ShardIndex:
    """Read ``index.json``. Raises on a missing or malformed index."""
    path = os.path.join(out_dir, _INDEX_NAME)
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    # `quant` defaults to "none" so a v0.72.0/.1 bf16 cache stays valid for a
    # bf16 request rather than forcing every existing user to re-shard.
    specs = payload.get("quant_specs") or {}
    return ShardIndex(
        n_layers=int(payload["n_layers"]),
        layer_keys=tuple(payload["layer_keys"]),
        extra_keys=tuple(payload["extra_keys"]),
        dtype=str(payload["dtype"]),
        total_params=int(payload["total_params"]),
        arch=str(payload.get("arch", "")),
        soup_version=str(payload.get("soup_version", "")),
        source_fingerprint=str(payload.get("source_fingerprint", "")),
        quant=str(payload.get("quant", QUANT_NONE)),
        double_quant=bool(payload.get("double_quant", False)),
        quant_device=str(payload.get("quant_device", "")),
        quant_specs={
            key: NF4WeightSpec.from_json(value) for key, value in specs.items()
        },
    )


# ==========================================================================
# cache location
# ==========================================================================
def model_slug(model: str) -> str:
    """Sanitise a model id into a traversal-safe cache directory name."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    base = model.strip().replace("\\", "/").rstrip("/").replace("/", "__")
    slug = _SLUG_RE.sub("_", base).replace("..", "_").strip("._-")
    return (slug or "model")[:128]


def default_layer_stream_cache_dir() -> str:
    """``~/.soup/layer-stream`` — the shard cache root (no side effects)."""
    return os.path.join(os.path.expanduser("~"), ".soup", "layer-stream")


def _resolve_cache_root(cache_dir: Optional[str], is_under: Any) -> str:
    """Cache ROOT only (explicit arg > env override > default). Always a str."""
    if cache_dir is not None:
        return os.path.realpath(os.path.expanduser(str(cache_dir)))
    override = os.environ.get("SOUP_LAYER_STREAM_CACHE_DIR")
    if override and not any(ord(ch) < 0x20 for ch in override):
        candidate = os.path.realpath(os.path.expanduser(override))
        bounds = [
            os.path.realpath(os.path.expanduser("~")),
            os.path.realpath(os.getcwd()),
            os.path.realpath(tempfile.gettempdir()),
        ]
        if any(is_under(candidate, bound) for bound in bounds):
            return candidate
    return default_layer_stream_cache_dir()


def resolve_shard_dir(model: str, cache_dir: Optional[str] = None) -> str:
    """Per-model shard dir (explicit arg > env override > default).

    ``SOUP_LAYER_STREAM_CACHE_DIR`` is rejected when it holds C0 control
    characters or escapes ``$HOME`` / ``$CWD`` / ``$TMPDIR`` — silently, because
    an env var is operator config, not API input (mirrors spectrum_scan).
    """
    from soup_cli.utils.paths import is_under

    slug = model_slug(model)
    root: str = _resolve_cache_root(cache_dir, is_under)
    chosen = os.path.join(root, slug)
    os.makedirs(chosen, exist_ok=True)
    return chosen


# ==========================================================================
# sharding
# ==========================================================================
def _discover_safetensors(weights_dir: str) -> List[str]:
    """Sorted, non-symlinked ``*.safetensors`` under ``weights_dir``."""
    root = os.path.realpath(os.path.expanduser(weights_dir))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"weights directory not found: {weights_dir}")
    found = []
    for name in sorted(os.listdir(root)):
        if not name.endswith(".safetensors"):
            continue
        path = os.path.join(root, name)
        if os.path.islink(path):
            logger.warning("layer-stream sharder: skipping symlinked shard %s", name)
            continue
        if not os.path.isfile(path):
            continue
        found.append(path)
    if not found:
        raise FileNotFoundError(
            f"no .safetensors weight files found in {weights_dir} — layer "
            f"streaming needs a safetensors checkpoint"
        )
    if len(found) > _MAX_SHARD_FILES:
        raise ValueError(f"too many shard files ({len(found)} > {_MAX_SHARD_FILES})")
    return found


def source_weight_bytes(weights_dir: str) -> int:
    """Total bytes of the source ``*.safetensors`` — a cheap size probe that
    lets a caller refuse an oversized base BEFORE spending minutes sharding."""
    return sum(os.path.getsize(path) for path in _discover_safetensors(weights_dir))


def _source_fingerprint(shards: List[str]) -> str:
    """Identity of the SOURCE checkpoint: basename + size + mtime per shard.

    The shard cache is keyed by a model slug, so without this a local
    checkpoint retrained in place (or two ids colliding onto one slug) would
    silently reuse stale shards and stream the WRONG WEIGHTS into training —
    no error, just a loss curve describing a different model.
    """
    import hashlib

    digest = hashlib.sha256()
    for path in sorted(shards):
        stat = os.stat(path)
        digest.update(os.path.basename(path).encode("utf-8", "replace"))
        digest.update(f"|{stat.st_size}|{int(stat.st_mtime_ns)}|".encode())
    return digest.hexdigest()


def _validate_out_dir(out_dir: str) -> str:
    """Bound our OWN writes rather than trusting the caller.

    ``realpath`` (not ``abspath``) so a symlinked ANCESTOR is resolved instead
    of transparently followed, and the result is then required to sit under
    $HOME / $CWD / $TMPDIR — the same bound ``resolve_shard_dir`` applies, but
    enforced here so a direct caller cannot bypass it.
    """
    from soup_cli.utils.paths import is_under

    if any(ord(ch) < 0x20 for ch in out_dir):
        raise ValueError("shard output directory must not contain control characters")
    expanded = os.path.abspath(os.path.expanduser(out_dir))
    if os.path.islink(expanded):
        raise ValueError(f"shard output directory must not be a symlink: {out_dir}")
    parent = os.path.dirname(expanded) or "."
    resolved = os.path.join(os.path.realpath(parent), os.path.basename(expanded))
    bounds = [
        os.path.realpath(os.path.expanduser("~")),
        os.path.realpath(os.getcwd()),
        os.path.realpath(tempfile.gettempdir()),
    ]
    if not any(is_under(resolved, bound) for bound in bounds):
        raise ValueError(
            f"shard output directory must be under $HOME, $CWD or $TMPDIR; "
            f"got {out_dir}"
        )
    os.makedirs(resolved, exist_ok=True)
    return resolved


def _atomic_save(blob: Dict[str, Any], path: str) -> None:
    """safetensors write via a temp file in the same dir + os.replace."""
    from safetensors.torch import save_file

    parent = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".soup.", suffix=".tmp", dir=parent)
    os.close(fd)
    try:
        save_file(blob, tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _atomic_write_index(index: ShardIndex, out_dir: str) -> None:
    payload = asdict(index)
    payload["layer_keys"] = list(index.layer_keys)
    payload["extra_keys"] = list(index.extra_keys)
    payload["quant_specs"] = {
        key: spec.to_json() for key, spec in index.quant_specs.items()
    }
    path = os.path.join(out_dir, _INDEX_NAME)
    fd, tmp = tempfile.mkstemp(prefix=".soup.", suffix=".tmp", dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _cached_index(
    out_dir: str,
    dtype: str,
    fingerprint: str,
    quant: str,
    double_quant: bool,
    quant_device: str,
) -> Optional[ShardIndex]:
    """Return a reusable index, or None when absent / corrupt / stale.

    A dtype mismatch MUST invalidate: streaming float32 shards into a bfloat16
    pool would quietly train the wrong precision rather than fail. The same
    argument applies with more force to ``quant``: reusing a bf16 shard set for
    an NF4 request would feed full-precision bytes to ``matmul_4bit`` (and the
    reverse would feed packed nibbles to a plain ``Linear``), so the cache key
    covers the quantisation and its double-quant flag too.
    """
    try:
        index = read_shard_index(out_dir)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if index.dtype != dtype:
        return None
    if index.quant != quant:
        return None
    if index.quant != QUANT_NONE and index.double_quant != double_quant:
        return None
    if index.quant != QUANT_NONE and index.quant_device != quant_device:
        return None
    if index.source_fingerprint != fingerprint:
        return None  # source checkpoint changed under us
    if not os.path.exists(extras_shard_path(out_dir)):
        return None
    for idx in range(index.n_layers):
        if not os.path.exists(layer_shard_path(out_dir, idx)):
            return None
    return index


# ==========================================================================
# NF4 quantisation (v0.72.2)
# ==========================================================================
def _default_quant_device() -> str:
    """Quantise where the model will run.

    Measured on this box: CPU and CUDA agree byte-for-byte on the packed
    nibbles and the per-block absmax for bf16/fp16, but the float32 double-quant
    *nested* statistic differs (a reduction-order difference in the offset). The
    gate's standard is bit-exactness against a RESIDENT CUDA load, so when a GPU
    is present we quantise on it — the same device ``from_pretrained`` would.
    """
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _quantize_nf4(tensor: Any, *, double_quant: bool, device: str):
    """Quantise ONE weight. Returns (sidecars, spec, code, nested_code)."""
    from bitsandbytes.functional import quantize_4bit

    packed, state = quantize_4bit(
        tensor.to(device),
        blocksize=NF4_BLOCKSIZE,
        compress_statistics=double_quant,
        quant_type=QUANT_NF4,
    )
    sidecars: Dict[str, Any] = {"": packed.cpu(), ABSMAX_SUFFIX: state.absmax.cpu()}
    nested_code = None
    if state.nested:
        sidecars[NESTED_ABSMAX_SUFFIX] = state.state2.absmax.cpu()
        sidecars[NESTED_OFFSET_SUFFIX] = state.offset.cpu()
        nested_code = state.state2.code.cpu()
    spec = NF4WeightSpec(
        shape=tuple(int(dim) for dim in state.shape),
        dtype=str(state.dtype).replace("torch.", ""),
        blocksize=int(state.blocksize),
        quant_type=str(state.quant_type),
        nested=bool(state.nested),
        nested_blocksize=int(state.state2.blocksize) if state.nested else 0,
    )
    return sidecars, spec, state.code.cpu(), nested_code


class _CodeTables:
    """Collects the shared code tables and refuses to let them diverge."""

    def __init__(self) -> None:
        self.code: Optional[Any] = None
        self.nested_code: Optional[Any] = None

    def observe(self, key: str, code: Any, nested_code: Any) -> None:
        import torch

        if self.code is None:
            self.code = code
        elif not torch.equal(self.code, code):
            raise ValueError(
                f"the NF4 code table differs at {key} — layer streaming keeps ONE "
                f"shared resident copy, so a per-weight table would silently "
                f"dequantise most weights with the wrong constants"
            )
        if nested_code is None:
            return
        if self.nested_code is None:
            self.nested_code = nested_code
        elif not torch.equal(self.nested_code, nested_code):
            raise ValueError(f"the nested NF4 code table differs at {key}")

    def as_extras(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.code is not None:
            out[NF4_CODE_KEY] = self.code
        if self.nested_code is not None:
            out[NF4_NESTED_CODE_KEY] = self.nested_code
        return out


def _validate_quant_request(quant: str, quant_suffixes: Iterable[str]) -> Tuple[str, ...]:
    """``quant`` and ``quant_suffixes`` must agree — neither alone is meaningful."""
    if quant not in SUPPORTED_STREAM_QUANTS:
        raise ValueError(
            f"unsupported quant {quant!r} for layer streaming; supported: "
            f"{', '.join(SUPPORTED_STREAM_QUANTS)}"
        )
    suffixes = tuple(sorted(set(quant_suffixes)))
    if quant == QUANT_NF4 and not suffixes:
        raise ValueError(
            "quant='nf4' needs quant_suffixes naming the decoder weights to "
            "quantise — writing full-precision bytes under an 'nf4' label would "
            "feed unquantised weights to Linear4bit modules"
        )
    if quant == QUANT_NONE and suffixes:
        raise ValueError(
            "quant_suffixes were given but quant='none' — the weights would be "
            "written unquantised and the names silently ignored"
        )
    return suffixes


def shard_checkpoint(
    weights_dir: str,
    out_dir: str,
    *,
    dtype: str = "bfloat16",
    arch: str = "",
    force: bool = False,
    quant: str = QUANT_NONE,
    quant_suffixes: Iterable[str] = (),
    double_quant: bool = True,
    quant_device: Optional[str] = None,
) -> ShardIndex:
    """Rewrite an HF checkpoint into per-layer safetensors shards.

    ``quant='nf4'`` quantises every per-layer key named in ``quant_suffixes``
    (short names, i.e. without the ``model.layers.N.`` prefix) and stores the
    packed nibbles plus the statistics needed to rebuild its ``QuantState``.
    Everything else — layernorms, embeddings, an untied head — is stored at
    ``dtype``, exactly as ``replace_with_bnb_linear`` leaves it.
    """
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r} for layer streaming; "
            f"supported: {', '.join(_SUPPORTED_DTYPES)}"
        )
    suffixes = _validate_quant_request(quant, quant_suffixes)
    quantise = quant == QUANT_NF4
    device = (quant_device or _default_quant_device()) if quantise else "cpu"
    # Only the KIND matters (cuda:0 and cuda:1 quantise identically); the
    # index stores that so the cache check below stays stable across ordinals.
    device_kind = str(device).split(":", 1)[0] if quantise else ""
    shards = _discover_safetensors(weights_dir)
    resolved_out = _validate_out_dir(out_dir)
    fingerprint = _source_fingerprint(shards)
    if not force:
        cached = _cached_index(
            resolved_out, dtype, fingerprint, quant, double_quant, device_kind
        )
        if cached is not None:
            return cached

    from safetensors import safe_open

    # pass 1 — build key -> shard without materialising a single tensor
    where: Dict[str, str] = {}
    layer_ids = set()
    for path in shards:
        with safe_open(path, framework="pt") as handle:
            for key in handle.keys():
                where[key] = path
                if len(where) > _MAX_TOTAL_TENSORS:
                    raise ValueError(
                        f"checkpoint declares more than {_MAX_TOTAL_TENSORS} tensors"
                    )
                match = _LAYER_RE.match(key)
                if match:
                    layer_ids.add(int(match.group(1)))

    if not layer_ids:
        raise ValueError(
            f"no decoder layer weights (model.layers.N.*) found in {weights_dir} — "
            f"layer streaming has nothing to stream"
        )
    n_layers = max(layer_ids) + 1
    if n_layers > _MAX_LAYERS:
        raise ValueError(f"too many decoder layers ({n_layers} > {_MAX_LAYERS})")
    if len(layer_ids) != n_layers:
        missing = sorted(set(range(n_layers)) - layer_ids)
        raise ValueError(f"decoder layer indices are not contiguous; missing {missing[:8]}")

    tables = _CodeTables()
    quant_specs: Dict[str, NF4WeightSpec] = {}

    total_params = 0
    layer_keys: Tuple[str, ...] = ()
    layer_shapes: Dict[str, Tuple[int, ...]] = {}
    # ExitStack, not a dict comprehension of __enter__(): if shard N fails to
    # open, everything opened before it must still be closed.
    with contextlib.ExitStack() as stack:
        handles = {
            path: stack.enter_context(safe_open(path, framework="pt")) for path in shards
        }
        for idx in range(n_layers):
            prefix = f"model.layers.{idx}."
            blob = {}
            for key, path in where.items():
                if not key.startswith(prefix):
                    continue
                short = key[len(prefix):]
                tensor = _read_tensor(handles[path], key, dtype)
                total_params += tensor.numel()
                if quantise and short in suffixes:
                    sidecars, spec, code, nested_code = _quantize_nf4(
                        tensor, double_quant=double_quant, device=device
                    )
                    tables.observe(key, code, nested_code)
                    quant_specs.setdefault(short, spec)
                    for sidecar, value in sidecars.items():
                        blob[short + sidecar] = value
                    del sidecars
                else:
                    blob[short] = tensor
                del tensor
            # Shapes, not just names: the buffer pool is sized from layer 0 AND
            # (under NF4) layer 0's quant_state is reused for every layer, so a
            # differently-shaped layer 5 would be reconstructed against the
            # wrong absmax blocking.
            shapes = {name: tuple(tensor.shape) for name, tensor in blob.items()}
            keys = tuple(sorted(blob))
            if idx == 0:
                layer_keys = keys
                layer_shapes = shapes
                _require_all_quantised(suffixes, quant_specs)
            elif keys != layer_keys:
                raise ValueError(
                    f"decoder layer {idx} has a different parameter set than layer 0 "
                    f"— layer streaming needs a uniform layer shape"
                )
            elif shapes != layer_shapes:
                differing = sorted(
                    name for name, shape in shapes.items() if layer_shapes[name] != shape
                )
                raise ValueError(
                    f"decoder layer {idx} has different tensor shapes than layer 0 "
                    f"({', '.join(differing[:3])}) — layer streaming needs a "
                    f"uniform layer shape"
                )
            _atomic_save(blob, layer_shard_path(resolved_out, idx))
            del blob

        extras = {}
        for key, path in where.items():
            if _LAYER_RE.match(key):
                continue
            tensor = _read_tensor(handles[path], key, dtype)
            extras[key] = tensor
            total_params += tensor.numel()
        extras.update(tables.as_extras())
        extra_keys = tuple(sorted(extras))
        _atomic_save(extras, extras_shard_path(resolved_out))
        del extras

    index = ShardIndex(
        n_layers=n_layers,
        layer_keys=layer_keys,
        extra_keys=extra_keys,
        dtype=dtype,
        total_params=total_params,
        arch=arch,
        soup_version=__version__,
        source_fingerprint=fingerprint,
        quant=quant,
        double_quant=bool(double_quant) if quantise else False,
        quant_device=device_kind,
        quant_specs=quant_specs,
    )
    _atomic_write_index(index, resolved_out)
    return index


def _require_all_quantised(
    suffixes: Tuple[str, ...], quant_specs: Dict[str, NF4WeightSpec]
) -> None:
    """Every requested suffix must have matched a real layer-0 weight.

    A typo'd or stale suffix would otherwise ship that weight unquantised while
    the model expects packed nibbles — a shape error at best, silently wrong
    numbers at worst.
    """
    missing = sorted(set(suffixes) - set(quant_specs))
    if missing:
        raise ValueError(
            f"quant_suffixes name weights that layer 0 does not have: "
            f"{', '.join(missing[:4])}"
        )


def _read_tensor(handle: Any, key: str, dtype: str) -> "Any":
    """Materialise one tensor, size-capped, converted to the target dtype."""
    import torch

    shape = handle.get_slice(key).get_shape()
    elements = math.prod(int(dim) for dim in shape)
    if elements > _MAX_TENSOR_ELEMENTS:
        raise ValueError(
            f"tensor {key} is too large for layer streaming "
            f"({elements} elements > {_MAX_TENSOR_ELEMENTS})"
        )
    target = getattr(torch, dtype)
    return handle.get_tensor(key).to(target).contiguous()
