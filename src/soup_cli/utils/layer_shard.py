"""training.stream_layers — checkpoint sharder (v0.72.0 BETA).

Rewrites an HF checkpoint into one ``layer_NNN.safetensors`` per decoder layer,
one shard for the input embeddings, one for an untied output head, and a small
``extras.safetensors`` (final norm / buffers / shared quantisation tables).  The
runtime can therefore stream the two vocabulary-sized matrices through one
large-layer slot instead of keeping both resident.

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

import collections.abc
import json
import logging
import math
import os
import re
import tempfile
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

from soup_cli import __version__

logger = logging.getLogger(__name__)

#: A decoder-layer parameter key: ``model.layers.<idx>.<rest>``.
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]")
_QWEN35_VLM_PREFIX = "model.language_model."

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
# Qwen4's production PLE table is intentionally much larger than an ordinary
# decoder tensor (320,001,536 x 160 = 51.2B elements). Keep a separate bound so
# routing it outside layer shards does not also route it around size validation.
_MAX_EXTERNAL_TENSOR_ELEMENTS = 2**36
_MAX_SHARD_FILES = 4096
_MAX_TOTAL_TENSORS = 200_000
# Source safe_open handles the sharder keeps alive at once (#926). Pass 2 used
# to hold EVERY source shard's mapping for the whole pass; on Windows that
# dies with an access violation once ~96-106 GB of safetensors mappings are
# alive in one process, whatever the file count, so a real 70B (30 files,
# 141 GB) could not be sharded there. A decoder layer needs one file, or two
# at a boundary; see _SourceHandles for why more is never needed for safety.
_MAX_LIVE_SOURCE_HANDLES = 2

_INDEX_NAME = "index.json"
_EXTRAS_NAME = "extras.safetensors"
_SHARD_FORMAT_VERSION = 5
_LARGE_EMBED_ROLE = "embed_tokens"
_LARGE_HEAD_ROLE = "lm_head"
QWEN4_PLE_WEIGHT_SUFFIX = ".ple.ple_embedding.ngram_embedding.weight"
_QWEN4_PLE_SHARD_RE = re.compile(
    r"^(?P<prefix>.+\.ple\.ple_embedding\.ngram_embedding)\.shard_(?P<part>\d+)\.weight$"
)
_QWEN4_OQ_PLE_SHARD_RE = re.compile(
    r"^(?P<prefix>.+\.ple\.ple_embedding\.ngram_embedding)\.shards\.(?P<part>\d+)\.weight$"
)
_QWEN4_EXPERT_RE = re.compile(
    r"^(?P<prefix>model\.layers\.\d+\.mlp\.experts)\."
    r"(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)
_QWEN4_OQ_EXPERT_RE = re.compile(
    r"^(?P<prefix>model\.layers\.\d+\.mlp)\.switch_mlp\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)


def _canonical_stream_key(key: str) -> str:
    """Normalise wrapper prefixes that do not exist on the text decoder."""
    if key.startswith(_QWEN35_VLM_PREFIX):
        return "model." + key[len(_QWEN35_VLM_PREFIX) :]
    if key.startswith("language_model."):
        return key[len("language_model.") :]
    return key


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
    #: Full checkpoint keys stored outside ``extras.safetensors`` and streamed
    #: through the single large-layer buffer. Only an explicit untied
    #: embedding/head pair is split; tied checkpoints stay on the unchanged
    #: resident one-matrix path.
    large_keys: Tuple[str, ...] = ()
    #: v3 splits only an untied vocabulary pair out of the resident extras shard.
    #: v2 also split tied embeddings, changing their established numerics.
    format_version: int = _SHARD_FORMAT_VERSION
    source_fingerprint: str = ""
    #: ``(basename, size, mtime_ns)`` for each source shard. The digest above
    #: remains the cache key; these components explain a mismatch to the user.
    source_files: Tuple[Tuple[str, int, int], ...] = ()
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
    #: Frozen tensors intentionally left in the original checkpoint. Qwen4-Exp's
    #: PLE table is too large for the decoder-layer buffer, but its forward only
    #: gathers sparse rows, so the runtime can serve it directly from RAM or a
    #: read-only safetensors mmap instead of copying it into the shard cache.
    external_tensors: Mapping[str, Any] = field(default_factory=dict)
    #: Cache-policy marker. It distinguishes a Qwen4 cache that intentionally
    #: found no PLE table from an older cache that silently put one in a layer.
    external_mode: str = ""


@dataclass(frozen=True)
class ExternalTensorSpec:
    """Logical layout for a tensor retained in one or more source parts."""

    parts: Tuple["ExternalTensorPart", ...]
    shape: Tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not self.parts:
            raise ValueError("external tensor must contain at least one source part")
        if len(self.shape) != 2 or any(dim <= 0 for dim in self.shape):
            raise ValueError(
                f"external tensor must be a positive 2-D matrix; got {self.shape}"
            )
        if math.prod(self.shape) > _MAX_EXTERNAL_TENSOR_ELEMENTS:
            raise ValueError(
                "external tensor exceeds the element cap: "
                f"{math.prod(self.shape)} > {_MAX_EXTERNAL_TENSOR_ELEMENTS}"
            )
        if any(part.dtype != self.dtype for part in self.parts):
            raise ValueError("external tensor parts must all have the same dtype")
        if any(part.shape[1:] != self.shape[1:] for part in self.parts):
            raise ValueError("external tensor parts must have the same trailing shape")
        if sum(part.shape[0] for part in self.parts) != self.shape[0]:
            raise ValueError("external tensor part rows do not add up to its shape")
        _ = self.nbytes

    @property
    def nbytes(self) -> int:
        sizes = {
            "BF16": 2,
            "F16": 2,
            "F32": 4,
        }
        try:
            itemsize = sizes[self.dtype]
        except KeyError as exc:
            raise ValueError(
                f"unsupported external tensor dtype {self.dtype!r}; supported: "
                f"{', '.join(sorted(sizes))}"
            ) from exc
        return sum(math.prod(part.shape) * itemsize for part in self.parts)

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "ExternalTensorSpec":
        return cls(
            parts=tuple(
                ExternalTensorPart.from_json(part) for part in payload["parts"]
            ),
            shape=tuple(int(dim) for dim in payload["shape"]),
            dtype=str(payload["dtype"]),
        )


@dataclass(frozen=True)
class ExternalTensorPart:
    """One row-contiguous source fragment of an external tensor."""

    source_file: str
    source_key: str
    shape: Tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if (
            self.source_file != os.path.basename(self.source_file)
            or self.source_file in ("", ".", "..")
        ):
            raise ValueError(
                f"invalid external tensor source filename {self.source_file!r}"
            )
        if not self.source_key:
            raise ValueError("external tensor source key must not be empty")
        if len(self.shape) != 2 or any(dim <= 0 for dim in self.shape):
            raise ValueError(
                f"external tensor part must be a positive 2-D matrix; got {self.shape}"
            )

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "ExternalTensorPart":
        return cls(
            source_file=str(payload["source_file"]),
            source_key=str(payload["source_key"]),
            shape=tuple(int(dim) for dim in payload["shape"]),
            dtype=str(payload["dtype"]),
        )


@dataclass(frozen=True)
class OQExternalTensorPart:
    """One row-contiguous oQ affine fragment retained in its source file."""

    source_file: str
    weight_key: str
    scales_key: str
    biases_key: str
    packed_shape: Tuple[int, ...]
    stats_shape: Tuple[int, ...]
    packed_dtype: str
    stats_dtype: str

    def __post_init__(self) -> None:
        if (
            self.source_file != os.path.basename(self.source_file)
            or self.source_file in ("", ".", "..")
        ):
            raise ValueError(
                f"invalid external oQ tensor source filename {self.source_file!r}"
            )
        if not all((self.weight_key, self.scales_key, self.biases_key)):
            raise ValueError("external oQ tensor keys must not be empty")
        if len(self.packed_shape) != 2 or any(dim <= 0 for dim in self.packed_shape):
            raise ValueError(
                f"external oQ packed tensor must be a positive 2-D matrix; "
                f"got {self.packed_shape}"
            )
        if len(self.stats_shape) != 2 or any(dim <= 0 for dim in self.stats_shape):
            raise ValueError(
                f"external oQ stats tensor must be a positive 2-D matrix; "
                f"got {self.stats_shape}"
            )
        if self.packed_shape[0] != self.stats_shape[0]:
            raise ValueError("external oQ weight and statistics must have equal row counts")
        if self.packed_dtype != "U32":
            raise ValueError(
                f"external oQ packed tensor must use U32; got {self.packed_dtype!r}"
            )
        if self.stats_dtype not in ("BF16", "F16", "F32"):
            raise ValueError(
                f"unsupported external oQ stats dtype {self.stats_dtype!r}"
            )

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "OQExternalTensorPart":
        return cls(
            source_file=str(payload["source_file"]),
            weight_key=str(payload["weight_key"]),
            scales_key=str(payload["scales_key"]),
            biases_key=str(payload["biases_key"]),
            packed_shape=tuple(int(dim) for dim in payload["packed_shape"]),
            stats_shape=tuple(int(dim) for dim in payload["stats_shape"]),
            packed_dtype=str(payload["packed_dtype"]),
            stats_dtype=str(payload["stats_dtype"]),
        )


@dataclass(frozen=True)
class OQExternalTensorSpec:
    """Logical dense tensor served from packed read-only oQ fragments."""

    parts: Tuple[OQExternalTensorPart, ...]
    shape: Tuple[int, ...]
    dtype: str
    bits: int
    group_size: int
    mode: str

    def __post_init__(self) -> None:
        from soup_cli.utils.oq_affine import AffineQuantSpec, logical_width

        _ = AffineQuantSpec(self.bits, self.group_size, self.mode)
        if not self.parts:
            raise ValueError("external oQ tensor must contain at least one source part")
        if len(self.shape) != 2 or any(dim <= 0 for dim in self.shape):
            raise ValueError(
                f"external oQ logical tensor must be a positive 2-D matrix; got {self.shape}"
            )
        if math.prod(self.shape) > _MAX_EXTERNAL_TENSOR_ELEMENTS:
            raise ValueError(
                "external oQ tensor exceeds the element cap: "
                f"{math.prod(self.shape)} > {_MAX_EXTERNAL_TENSOR_ELEMENTS}"
            )
        if self.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"unsupported external oQ output dtype {self.dtype!r}")
        widths = {
            logical_width(part.packed_shape[1], self.bits) for part in self.parts
        }
        if widths != {self.shape[1]}:
            raise ValueError("external oQ parts disagree with the logical tensor width")
        if sum(part.packed_shape[0] for part in self.parts) != self.shape[0]:
            raise ValueError("external oQ part rows do not add up to its logical shape")
        expected_stats_width = self.shape[1] // self.group_size
        if any(part.stats_shape[1] != expected_stats_width for part in self.parts):
            raise ValueError("external oQ statistics disagree with the affine group size")

    @property
    def nbytes(self) -> int:
        itemsize = {"bfloat16": 2, "float16": 2, "float32": 4}[self.dtype]
        return math.prod(self.shape) * itemsize

    @property
    def storage_nbytes(self) -> int:
        stats_itemsize = {"BF16": 2, "F16": 2, "F32": 4}
        return sum(
            math.prod(part.packed_shape) * 4
            + math.prod(part.stats_shape) * stats_itemsize[part.stats_dtype] * 2
            for part in self.parts
        )

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "OQExternalTensorSpec":
        return cls(
            parts=tuple(
                OQExternalTensorPart.from_json(part) for part in payload["parts"]
            ),
            shape=tuple(int(dim) for dim in payload["shape"]),
            dtype=str(payload["dtype"]),
            bits=int(payload["bits"]),
            group_size=int(payload["group_size"]),
            mode=str(payload["mode"]),
        )


def _external_spec_from_json(payload: Mapping[str, Any]) -> Any:
    if "bits" in payload:
        return OQExternalTensorSpec.from_json(payload)
    return ExternalTensorSpec.from_json(payload)


def layer_shard_path(out_dir: str, idx: int) -> str:
    return os.path.join(out_dir, f"layer_{idx:03d}.safetensors")


def extras_shard_path(out_dir: str) -> str:
    return os.path.join(out_dir, _EXTRAS_NAME)


def large_weight_role(key: str) -> Optional[str]:
    """Return the supported large-layer role for a canonical checkpoint key."""
    canonical = _canonical_stream_key(key)
    if canonical == "model.embed_tokens.weight":
        return _LARGE_EMBED_ROLE
    if canonical == "lm_head.weight":
        return _LARGE_HEAD_ROLE
    return None


def large_shard_path(out_dir: str, key: str) -> str:
    """Path for one vocabulary-sized streamed weight."""
    role = large_weight_role(key)
    if role is None:
        raise ValueError(f"{key!r} is not a supported streamed large-layer key")
    return os.path.join(out_dir, f"large_{role}.safetensors")


def read_shard_index(out_dir: str) -> ShardIndex:
    """Read ``index.json``. Raises on a missing or malformed index."""
    path = os.path.join(out_dir, _INDEX_NAME)
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    # `quant` defaults to "none" so a v0.72.0/.1 bf16 cache stays valid for a
    # bf16 request rather than forcing every existing user to re-shard.
    specs = payload.get("quant_specs") or {}
    external = payload.get("external_tensors") or {}
    if not isinstance(external, dict):
        raise ValueError("external_tensors must be an object")
    return ShardIndex(
        n_layers=int(payload["n_layers"]),
        layer_keys=tuple(payload["layer_keys"]),
        extra_keys=tuple(payload["extra_keys"]),
        dtype=str(payload["dtype"]),
        total_params=int(payload["total_params"]),
        arch=str(payload.get("arch", "")),
        soup_version=str(payload.get("soup_version", "")),
        large_keys=tuple(payload.get("large_keys") or ()),
        format_version=int(payload.get("format_version", 1)),
        source_fingerprint=str(payload.get("source_fingerprint", "")),
        source_files=tuple(
            (str(item[0]), int(item[1]), int(item[2]))
            for item in (payload.get("source_files") or ())
        ),
        quant=str(payload.get("quant", QUANT_NONE)),
        double_quant=bool(payload.get("double_quant", False)),
        quant_device=str(payload.get("quant_device", "")),
        quant_specs={
            key: NF4WeightSpec.from_json(value) for key, value in specs.items()
        },
        external_tensors={
            str(key): _external_spec_from_json(value)
            for key, value in external.items()
        },
        external_mode=str(payload.get("external_mode", "")),
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


def estimate_oq_stream_cache_bytes(
    weights_dir: str, *, dtype: str, arch: str
) -> Optional[int]:
    """Estimate the dense shard-cache payload for an oQ Qwen4 checkpoint.

    The packed source size is not a safe proxy: decoder weights expand to the
    streamed dtype, while the much larger PLE table remains external.  Returns
    ``None`` for non-oQ checkpoints so existing estimation stays unchanged.
    """
    if arch != "qwen4_exp" or dtype not in _SUPPORTED_DTYPES:
        return None
    if not os.path.isdir(os.path.realpath(os.path.expanduser(weights_dir))):
        return None
    from safetensors import safe_open

    from soup_cli.utils.oq_affine import load_affine_quant_config, logical_width

    shards = _discover_safetensors(weights_dir)
    descriptors = {}
    saw_companion = False
    for path in shards:
        with safe_open(path, framework="pt") as handle:
            for source_key in handle.keys():
                if not source_key.startswith("language_model."):
                    continue
                canonical = _canonical_stream_key(source_key)
                tensor_slice = handle.get_slice(source_key)
                descriptors[canonical] = (
                    source_key,
                    tuple(int(dim) for dim in tensor_slice.get_shape()),
                )
                saw_companion = saw_companion or source_key.endswith(
                    (".scales", ".biases")
                )
    if not saw_companion:
        return None
    quant_config = load_affine_quant_config(weights_dir)
    itemsize = {"bfloat16": 2, "float16": 2, "float32": 4}[dtype]
    elements = 0
    for key, (source_key, shape) in descriptors.items():
        if key.endswith((".scales", ".biases")) or source_key.endswith(
            ".ple.ple_embedding.ngram_embedding.weight_scale"
        ):
            continue
        if _QWEN4_OQ_PLE_SHARD_RE.match(key):
            continue
        base = key.removesuffix(".weight")
        if key.endswith(".weight") and {
            base + ".scales",
            base + ".biases",
        }.issubset(descriptors):
            spec = quant_config.for_module(source_key.removesuffix(".weight"))
            elements += math.prod(shape[:-1]) * logical_width(shape[-1], spec.bits)
        else:
            elements += math.prod(shape)
    # Safetensors headers, index JSON and atomic-write slack are tiny beside a
    # production model, but reserve 256 MiB so the pre-flight never reports a
    # byte-perfect payload as a byte-perfect filesystem requirement.
    return elements * itemsize + 256 * 1024 * 1024


def source_weight_bytes(weights_dir: str) -> int:
    """Total bytes of the source ``*.safetensors`` — a cheap size probe that
    lets a caller refuse an oversized base BEFORE spending minutes sharding."""
    return sum(os.path.getsize(path) for path in _discover_safetensors(weights_dir))


def _source_file_components(shards: List[str]) -> Tuple[Tuple[str, int, int], ...]:
    """Components retained beside the digest so a stale cache is explainable."""
    components = []
    for path in sorted(shards):
        stat = os.stat(path)
        components.append(
            (os.path.basename(path), int(stat.st_size), int(stat.st_mtime_ns))
        )
    return tuple(components)


def checkpoint_source_components(
    weights_dir: str,
    components: Tuple[Tuple[str, int, int], ...],
    *,
    include_config: bool,
) -> Tuple[Tuple[str, int, int], ...]:
    """Add numerical sidecar metadata to the cache identity when applicable."""
    merged = {name: (name, size, mtime) for name, size, mtime in components}
    if include_config:
        config_path = os.path.join(os.path.realpath(weights_dir), "config.json")
        if os.path.isfile(config_path) and not os.path.islink(config_path):
            stat = os.stat(config_path)
            merged["config.json"] = (
                "config.json",
                int(stat.st_size),
                int(stat.st_mtime_ns),
            )
    return tuple(merged[name] for name in sorted(merged))


def _fingerprint_components(components: Tuple[Tuple[str, int, int], ...]) -> str:
    """Hash the exact components that are persisted in ``index.json``."""
    import hashlib

    digest = hashlib.sha256()
    for name, size, mtime_ns in components:
        digest.update(name.encode("utf-8", "replace"))
        digest.update(f"|{size}|{mtime_ns}|".encode())
    return digest.hexdigest()


def fingerprint_source_files(components: Tuple[Tuple[str, int, int], ...]) -> str:
    """Public digest helper for the pre-sharding disk planner."""
    return _fingerprint_components(components)


def _source_fingerprint(shards: List[str]) -> str:
    """Identity of the SOURCE checkpoint: basename + size + mtime per shard.

    The shard cache is keyed by a model slug, so without this a local
    checkpoint retrained in place (or two ids colliding onto one slug) would
    silently reuse stale shards and stream the WRONG WEIGHTS into training —
    no error, just a loss curve describing a different model.
    """
    return _fingerprint_components(_source_file_components(shards))


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
    payload["large_keys"] = list(index.large_keys)
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


def _describe_source_change(
    previous: Tuple[Tuple[str, int, int], ...],
    current: Tuple[Tuple[str, int, int], ...],
) -> str:
    """Name the first fingerprint component that invalidated the cache."""
    if not previous:
        return (
            "source_fingerprint changed; the cached index predates per-file "
            "component records"
        )
    old = {name: (size, mtime) for name, size, mtime in previous}
    new = {name: (size, mtime) for name, size, mtime in current}
    if old.keys() != new.keys():
        added = sorted(new.keys() - old.keys())
        removed = sorted(old.keys() - new.keys())
        details = []
        if added:
            details.append(f"added {added[0]!r}")
        if removed:
            details.append(f"removed {removed[0]!r}")
        return "source filename set changed (" + ", ".join(details) + ")"
    for name in sorted(new):
        old_size, old_mtime = old[name]
        new_size, new_mtime = new[name]
        if old_size != new_size:
            return (
                f"source size changed for {name!r} "
                f"({old_size} -> {new_size} bytes)"
            )
        if old_mtime != new_mtime:
            return (
                f"source mtime_ns changed for {name!r} "
                f"({old_mtime} -> {new_mtime})"
            )
    return "source_fingerprint changed although its recorded components match"


def inspect_shard_cache(
    out_dir: str,
    dtype: str,
    fingerprint: str,
    source_files: Tuple[Tuple[str, int, int], ...],
    quant: str,
    double_quant: bool,
    quant_device: str,
    external_mode: str = "",
) -> Tuple[Optional[ShardIndex], str]:
    """Return a reusable index or the precise reason it must be rewritten.

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
        return None, "cache index is missing or unreadable"
    if index.dtype != dtype:
        return None, f"dtype changed ({index.dtype!r} -> {dtype!r})"
    if index.quant != quant:
        return None, f"quantization changed ({index.quant!r} -> {quant!r})"
    if index.quant != QUANT_NONE and index.double_quant != double_quant:
        return None, (
            f"double_quant changed ({index.double_quant!r} -> {double_quant!r})"
        )
    if index.quant != QUANT_NONE and index.quant_device != quant_device:
        return None, (
            f"quantization device changed "
            f"({index.quant_device!r} -> {quant_device!r})"
        )
    if index.source_fingerprint != fingerprint:
        return None, _describe_source_change(index.source_files, source_files)
    if index.external_mode != external_mode:
        return None, (
            f"external tensor policy changed ({index.external_mode!r} -> "
            f"{external_mode!r})"
        )
    if index.format_version != _SHARD_FORMAT_VERSION:
        return None, (
            f"shard format changed "
            f"({index.format_version!r} -> {_SHARD_FORMAT_VERSION!r})"
        )
    if not os.path.exists(extras_shard_path(out_dir)):
        return None, f"cached shard {_EXTRAS_NAME!r} is missing"
    for key in index.large_keys:
        path = large_shard_path(out_dir, key)
        if not os.path.exists(path):
            return None, f"cached shard {os.path.basename(path)!r} is missing"
    for idx in range(index.n_layers):
        if not os.path.exists(layer_shard_path(out_dir, idx)):
            name = os.path.basename(layer_shard_path(out_dir, idx))
            return None, f"cached shard {name!r} is missing"
    return index, "cache is reusable"


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
    notify: Optional[Callable[[str], None]] = None,
) -> ShardIndex:
    """Rewrite an HF checkpoint into per-layer safetensors shards.

    ``quant='nf4'`` quantises every per-layer key named in ``quant_suffixes``
    (short names, i.e. without the ``model.layers.N.`` prefix) and stores the
    packed nibbles plus the statistics needed to rebuild its ``QuantState``.
    Everything else is stored at ``dtype``, exactly as
    ``replace_with_bnb_linear`` leaves it.  When BOTH ``embed_tokens`` and an
    explicit (untied) ``lm_head`` exist, each is written to its own shard and
    remains unquantised; the runtime streams them through one large slot. A tied
    checkpoint keeps its single embedding matrix resident and unchanged.
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
    external_mode = "qwen4_ple" if arch == "qwen4_exp" else ""
    shards = _discover_safetensors(weights_dir)
    resolved_out = _validate_out_dir(out_dir)
    source_files = checkpoint_source_components(
        weights_dir,
        _source_file_components(shards),
        include_config=bool(external_mode),
    )
    fingerprint = _fingerprint_components(source_files)
    if not force:
        cached, miss_reason = inspect_shard_cache(
            resolved_out,
            dtype,
            fingerprint,
            source_files,
            quant,
            double_quant,
            device_kind,
            external_mode,
        )
        if cached is not None:
            return cached
        if notify is not None:
            notify(f"[yellow]Re-sharding layer cache:[/] {miss_reason}.")

    from safetensors import safe_open

    # pass 1 — build key -> shard without materialising a single tensor
    where: Dict[str, Tuple[str, str]] = {}
    external_parts: Dict[str, List[Tuple[int, ExternalTensorPart]]] = {}
    oq_external_keys: Dict[str, List[Tuple[int, str]]] = {}
    external_source_keys = set()
    saw_oq_companion = False
    layer_ids = set()
    for path in shards:
        with safe_open(path, framework="pt") as handle:
            for source_key in handle.keys():
                if external_mode and source_key.endswith((".biases", ".scales")):
                    saw_oq_companion = True
                key = _canonical_stream_key(source_key)
                prior = where.get(key)
                if prior is not None and prior != (path, source_key):
                    raise ValueError(
                        f"checkpoint spells one streamed tensor twice: "
                        f"{prior[1]!r} and {source_key!r} both canonicalise to "
                        f"{key!r}. Refusing rather than silently keeping one by "
                        f"safetensors key order."
                    )
                where[key] = (path, source_key)
                ple_match = _QWEN4_PLE_SHARD_RE.match(key) if external_mode else None
                oq_ple_match = (
                    _QWEN4_OQ_PLE_SHARD_RE.match(key) if external_mode else None
                )
                is_dense_ple = external_mode and key.endswith(QWEN4_PLE_WEIGHT_SUFFIX)
                if oq_ple_match:
                    logical_key = oq_ple_match.group("prefix") + ".weight"
                    part_index = int(oq_ple_match.group("part"))
                    oq_external_keys.setdefault(logical_key, []).append(
                        (part_index, key)
                    )
                elif ple_match or is_dense_ple:
                    tensor_slice = handle.get_slice(source_key)
                    logical_key = (
                        ple_match.group("prefix") + ".weight" if ple_match else key
                    )
                    part_index = int(ple_match.group("part")) if ple_match else 0
                    part = ExternalTensorPart(
                        source_file=os.path.basename(path),
                        source_key=source_key,
                        shape=tuple(int(dim) for dim in tensor_slice.get_shape()),
                        dtype=str(tensor_slice.get_dtype()),
                    )
                    external_parts.setdefault(logical_key, []).append(
                        (part_index, part)
                    )
                    external_source_keys.add(key)
                if len(where) > _MAX_TOTAL_TENSORS:
                    raise ValueError(
                        f"checkpoint declares more than {_MAX_TOTAL_TENSORS} tensors"
                    )
                match = _LAYER_RE.match(key)
                if match:
                    layer_ids.add(int(match.group(1)))

    oq_config = None
    if saw_oq_companion:
        from soup_cli.utils.oq_affine import load_affine_quant_config

        oq_config = load_affine_quant_config(weights_dir)
        # oQ conditional-generation bundles also carry vision and MTP weights.
        # The streamed skeleton is the text-only CausalLM, so those components
        # are deliberately not copied into its extras shard.
        where = {
            key: location
            for key, location in where.items()
            if location[1].startswith("language_model.")
            and not location[1].endswith(
                ".ple.ple_embedding.ngram_embedding.weight_scale"
            )
        }
        layer_ids = {
            int(match.group(1))
            for key in where
            if (match := _LAYER_RE.match(key)) is not None
        }

    external_tensors: Dict[str, Any] = {}
    for logical_key, indexed_parts in external_parts.items():
        indexed_parts.sort(key=lambda item: item[0])
        indices = [part_index for part_index, _part in indexed_parts]
        if indices != list(range(len(indices))):
            raise ValueError(
                f"PLE shards for {logical_key!r} must be contiguous from zero; "
                f"got {indices[:8]}"
            )
        parts = tuple(part for _part_index, part in indexed_parts)
        dtype_names = {part.dtype for part in parts}
        trailing_shapes = {part.shape[1:] for part in parts}
        if len(dtype_names) != 1 or len(trailing_shapes) != 1:
            raise ValueError(
                f"PLE shards for {logical_key!r} disagree on dtype or width"
            )
        external_tensors[logical_key] = ExternalTensorSpec(
            parts=parts,
            shape=(sum(part.shape[0] for part in parts), *parts[0].shape[1:]),
            dtype=parts[0].dtype,
        )

    for logical_key, indexed_keys in oq_external_keys.items():
        if oq_config is None:
            raise ValueError("oQ PLE shards require affine quantization metadata")
        indexed_keys.sort(key=lambda item: item[0])
        indices = [part_index for part_index, _key in indexed_keys]
        if indices != list(range(len(indices))):
            raise ValueError(
                f"oQ PLE shards for {logical_key!r} must be contiguous from zero; "
                f"got {indices[:8]}"
            )
        parts = []
        part_specs = set()
        logical_widths = set()
        for _part_index, weight_key in indexed_keys:
            location = where.get(weight_key)
            scales_location = where.get(weight_key.removesuffix(".weight") + ".scales")
            biases_location = where.get(weight_key.removesuffix(".weight") + ".biases")
            if location is None or scales_location is None or biases_location is None:
                raise ValueError(f"oQ PLE part {weight_key!r} is missing affine companions")
            paths = {location[0], scales_location[0], biases_location[0]}
            if len(paths) != 1:
                raise ValueError(
                    f"oQ PLE part {weight_key!r} splits affine companions across files"
                )
            module_name = location[1].removesuffix(".weight")
            quant_spec = oq_config.for_module(module_name)
            part_specs.add(quant_spec)
            path = location[0]
            with safe_open(path, framework="pt") as handle:
                weight_slice = handle.get_slice(location[1])
                scales_slice = handle.get_slice(scales_location[1])
                biases_slice = handle.get_slice(biases_location[1])
                packed_shape = tuple(int(dim) for dim in weight_slice.get_shape())
                scales_shape = tuple(int(dim) for dim in scales_slice.get_shape())
                biases_shape = tuple(int(dim) for dim in biases_slice.get_shape())
                if scales_shape != biases_shape:
                    raise ValueError(
                        f"oQ PLE part {weight_key!r} scales and biases disagree"
                    )
                from soup_cli.utils.oq_affine import logical_width

                width = logical_width(packed_shape[-1], quant_spec.bits)
                logical_widths.add(width)
                parts.append(
                    OQExternalTensorPart(
                        source_file=os.path.basename(path),
                        weight_key=location[1],
                        scales_key=scales_location[1],
                        biases_key=biases_location[1],
                        packed_shape=packed_shape,
                        stats_shape=scales_shape,
                        packed_dtype=str(weight_slice.get_dtype()),
                        stats_dtype=str(scales_slice.get_dtype()),
                    )
                )
            external_source_keys.update(
                {
                    weight_key,
                    weight_key.removesuffix(".weight") + ".scales",
                    weight_key.removesuffix(".weight") + ".biases",
                }
            )
        if len(part_specs) != 1 or len(logical_widths) != 1:
            raise ValueError(f"oQ PLE shards for {logical_key!r} disagree on layout")
        quant_spec = next(iter(part_specs))
        width = next(iter(logical_widths))
        external_tensors[logical_key] = OQExternalTensorSpec(
            parts=tuple(parts),
            shape=(sum(part.packed_shape[0] for part in parts), width),
            dtype=dtype,
            bits=quant_spec.bits,
            group_size=quant_spec.group_size,
            mode=quant_spec.mode,
        )
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

    # #324 relieves the TWO-matrix untied footprint. A tied checkpoint already
    # needs only one matrix, and its acceptance control is explicitly
    # "unaffected". Splitting its lone embedding would change the CUDA kernel
    # path for no memory win and regressed preference-loss bit-exactness.
    large_roles_present = {
        role for key in where if (role := large_weight_role(key)) is not None
    }
    stream_untied_pair = {
        _LARGE_EMBED_ROLE,
        _LARGE_HEAD_ROLE,
    }.issubset(large_roles_present)

    tables = _CodeTables()
    quant_specs: Dict[str, NF4WeightSpec] = {}
    matched_quant_suffixes = set()

    total_params = 0
    layer_keys = set()
    shared_specs: Dict[str, Tuple[Tuple[int, ...], str]] = {}
    # At most _MAX_LIVE_SOURCE_HANDLES source files are mapped at once (#926);
    # a failed open, or a raise anywhere below, still releases the live ones.
    with _SourceHandles(shards, safe_open) as handles:
        for idx in range(n_layers):
            prefix = f"model.layers.{idx}."
            blob = {}
            qwen4_experts: Dict[str, Dict[int, Tuple[str, str]]] = {}
            oq_experts: Dict[str, str] = {}
            for key, location in where.items():
                if not key.startswith(prefix):
                    continue
                if key in external_source_keys:
                    continue
                if saw_oq_companion and key.endswith((".scales", ".biases")):
                    continue
                oq_expert_match = (
                    _QWEN4_OQ_EXPERT_RE.match(key) if saw_oq_companion else None
                )
                if oq_expert_match:
                    oq_experts[oq_expert_match.group("projection")] = key
                    continue
                expert_match = _QWEN4_EXPERT_RE.match(key) if external_mode else None
                if expert_match:
                    projection = expert_match.group("projection")
                    expert = int(expert_match.group("expert"))
                    qwen4_experts.setdefault(projection, {})[expert] = location
                    continue
                path, source_key = location
                short = key[len(prefix):]
                if saw_oq_companion and key.endswith(".weight"):
                    tensor = _read_oq_tensor(
                        handles, where, key, oq_config=oq_config, dtype=dtype
                    )
                else:
                    tensor = _read_tensor(handles[path], source_key, dtype)
                total_params += tensor.numel()
                if quantise and short in suffixes:
                    sidecars, spec, code, nested_code = _quantize_nf4(
                        tensor, double_quant=double_quant, device=device
                    )
                    tables.observe(key, code, nested_code)
                    prior_spec = quant_specs.get(short)
                    if prior_spec is not None and prior_spec != spec:
                        raise ValueError(
                            f"decoder weight {short!r} has inconsistent NF4 metadata "
                            f"across layers ({prior_spec.shape} vs {spec.shape}) — "
                            f"layer streaming can pool a shared key only when its "
                            f"stored layout stays the same"
                        )
                    quant_specs.setdefault(short, spec)
                    matched_quant_suffixes.add(short)
                    for sidecar, value in sidecars.items():
                        blob[short + sidecar] = value
                    del sidecars
                else:
                    blob[short] = tensor
                del tensor
            for logical_key, spec in external_tensors.items():
                if logical_key.startswith(prefix):
                    total_params += math.prod(spec.shape)
            if qwen4_experts:
                import torch

                expected_projections = {"gate_proj", "up_proj", "down_proj"}
                if set(qwen4_experts) != expected_projections:
                    raise ValueError(
                        f"Qwen4 layer {idx} has incomplete expert projections: "
                        f"{sorted(qwen4_experts)}"
                    )
                expert_ids = sorted(qwen4_experts["gate_proj"])
                if expert_ids != list(range(len(expert_ids))) or any(
                    sorted(qwen4_experts[name]) != expert_ids
                    for name in expected_projections
                ):
                    raise ValueError(
                        f"Qwen4 layer {idx} expert ids must be contiguous and "
                        "identical across gate/up/down projections"
                    )

                def _expert_tensor(projection: str, expert: int):
                    path, source_key = qwen4_experts[projection][expert]
                    return _read_tensor(handles[path], source_key, dtype)

                gate_up = []
                down = []
                for expert in expert_ids:
                    gate = _expert_tensor("gate_proj", expert)
                    up = _expert_tensor("up_proj", expert)
                    down_tensor = _expert_tensor("down_proj", expert)
                    gate_up.append(torch.cat((gate, up), dim=0))
                    down.append(down_tensor)
                    total_params += gate.numel() + up.numel() + down_tensor.numel()
                if "mlp.experts.gate_up_proj" in blob or "mlp.experts.down_proj" in blob:
                    raise ValueError(
                        f"Qwen4 layer {idx} contains both packed and per-expert weights"
                    )
                blob["mlp.experts.gate_up_proj"] = torch.stack(gate_up, dim=0)
                blob["mlp.experts.down_proj"] = torch.stack(down, dim=0)
            if oq_experts:
                import torch

                expected_projections = {"gate_proj", "up_proj", "down_proj"}
                if set(oq_experts) != expected_projections:
                    raise ValueError(
                        f"Qwen4 oQ layer {idx} has incomplete Switch-MLP projections: "
                        f"{sorted(oq_experts)}"
                    )
                gate = _read_oq_tensor(
                    handles,
                    where,
                    oq_experts["gate_proj"],
                    oq_config=oq_config,
                    dtype=dtype,
                )
                up = _read_oq_tensor(
                    handles,
                    where,
                    oq_experts["up_proj"],
                    oq_config=oq_config,
                    dtype=dtype,
                )
                down = _read_oq_tensor(
                    handles,
                    where,
                    oq_experts["down_proj"],
                    oq_config=oq_config,
                    dtype=dtype,
                )
                if gate.shape != up.shape:
                    raise ValueError(
                        f"Qwen4 oQ layer {idx} gate/up expert shapes disagree: "
                        f"{tuple(gate.shape)} vs {tuple(up.shape)}"
                    )
                blob["mlp.experts.gate_up_proj"] = torch.cat((gate, up), dim=1)
                blob["mlp.experts.down_proj"] = down
                total_params += gate.numel() + up.numel() + down.numel()
            layer_keys.update(blob)
            layer_specs = {
                name: (tuple(tensor.shape), str(tensor.dtype).replace("torch.", ""))
                for name, tensor in blob.items()
            }
            for name, spec in layer_specs.items():
                prior = shared_specs.get(name)
                if prior is not None and prior != spec:
                    raise ValueError(
                        f"decoder weight {name!r} has different stored shapes or dtypes "
                        f"across layers ({prior} vs {spec}) — layer streaming can vary "
                        f"which weights exist per layer, but a shared key must keep the "
                        f"same storage layout"
                    )
                shared_specs.setdefault(name, spec)
            _atomic_save(blob, layer_shard_path(resolved_out, idx))
            del blob

        extras = {}
        large_keys = []
        large_roles: Dict[str, str] = {}
        for key, location in where.items():
            if _LAYER_RE.match(key):
                continue
            if saw_oq_companion and key.endswith((".scales", ".biases")):
                continue
            path, source_key = location
            if saw_oq_companion and key.endswith(".weight"):
                tensor = _read_oq_tensor(
                    handles, where, key, oq_config=oq_config, dtype=dtype
                )
            else:
                tensor = _read_tensor(handles[path], source_key, dtype)
            total_params += tensor.numel()
            role = large_weight_role(key) if stream_untied_pair else None
            if role is None:
                extras[key] = tensor
                continue
            prior = large_roles.get(role)
            if prior is not None:
                raise ValueError(
                    f"checkpoint has multiple {role} weights ({prior!r}, {key!r}); "
                    f"the large-layer scheduler needs one unambiguous boundary"
                )
            large_roles[role] = key
            _atomic_save({key: tensor}, large_shard_path(resolved_out, key))
            large_keys.append(key)
            del tensor
        extras.update(tables.as_extras())
        extra_keys = tuple(sorted(extras))
        _atomic_save(extras, extras_shard_path(resolved_out))
        del extras

    _require_all_quantised(suffixes, matched_quant_suffixes)

    index = ShardIndex(
        n_layers=n_layers,
        layer_keys=tuple(sorted(layer_keys)),
        extra_keys=extra_keys,
        dtype=dtype,
        total_params=total_params,
        arch=arch,
        soup_version=__version__,
        large_keys=tuple(sorted(large_keys)),
        format_version=_SHARD_FORMAT_VERSION,
        source_fingerprint=fingerprint,
        source_files=source_files,
        quant=quant,
        double_quant=bool(double_quant) if quantise else False,
        quant_device=device_kind,
        quant_specs=quant_specs,
        external_tensors=external_tensors,
        external_mode=external_mode,
    )
    _atomic_write_index(index, resolved_out)
    return index


def _require_all_quantised(suffixes: Tuple[str, ...], matched: Iterable[str]) -> None:
    """Every requested suffix must have matched at least one real decoder weight.

    A typo'd or stale suffix would otherwise ship that weight unquantised while
    the model expects packed nibbles — a shape error at best, silently wrong
    numbers at worst.
    """
    missing = sorted(set(suffixes) - set(matched))
    if missing:
        raise ValueError(
            f"quant_suffixes name decoder weights that the checkpoint does not have: "
            f"{', '.join(missing[:4])}"
        )


class _SourceHandles(collections.abc.Mapping):
    """Least-recently-used cache of live ``safe_open`` handles over ``shards``.

    ``handles[path]`` opens on demand and evicts the least recently used
    handle once ``capacity`` are alive; leaving the ``with`` block (or
    ``close()``) releases every live handle, even when one release raises.
    Only discovered shards are accepted, so a mapping bug cannot open an
    arbitrary file, and membership, iteration and ``len`` never open anything.

    WHAT THE CAP BUYS (#926, measured 2026-09-13): ``safe_open`` maps the file
    PRIVATELY, so a live handle charges Windows commit for the file's whole
    size — 48.99 GB of mappings raised the commit charge 46.17 GB while free
    physical memory never moved, and closing them returned it. Every source
    shard open at once therefore charged the whole checkpoint (~138 GB for a
    70B), which no pagefile kept up with; two handles charge two files.

    Why a cap this small is safe: decoder layers are contiguous in HF
    checkpoints, so pass 2 needs one file per layer, or two at a boundary. An
    access pattern that needs more merely re-opens a file (a header parse, not
    a read) and is never wrong, because every tensor the readers hand out owns
    its memory (``_own``) — nothing outlives the handle it was read through.

    NOT a general-purpose mapping: ``__getitem__`` opens files and evicts, so
    the ``Mapping`` mixins that call it for every key are refused rather than
    inherited (see ``_refuse``). ``Mapping`` is the declared shape because
    ``_read_oq_tensor`` annotates this object as one.
    """

    def __init__(
        self,
        shards: Iterable[str],
        opener: Callable[..., Any],
        capacity: int = _MAX_LIVE_SOURCE_HANDLES,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1, got {capacity}")
        self._allowed = frozenset(shards)
        self._opener = opener
        self._capacity = int(capacity)
        self._live: "OrderedDict[str, Any]" = OrderedDict()
        self._released = False

    def __getitem__(self, path: str) -> Any:
        if self._released:
            raise RuntimeError("source handles already released")
        if path not in self._allowed:
            raise KeyError(path)
        handle = self._live.get(path)
        if handle is not None:
            self._live.move_to_end(path)
            return handle
        while len(self._live) >= self._capacity:
            _stale, victim = self._live.popitem(last=False)
            victim.__exit__(None, None, None)
        handle = self._opener(path, framework="pt").__enter__()
        self._live[path] = handle
        return handle

    def __contains__(self, path: object) -> bool:
        return path in self._allowed

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._allowed))

    def __len__(self) -> int:
        return len(self._allowed)

    def _refuse(self, *_args: Any, **_kwargs: Any) -> Any:
        """Refuse the ``Mapping`` mixins that would open files behind the caller.

        ``Mapping`` implements ``get``/``items``/``values``/``__eq__`` on top of
        ``__getitem__``, which here OPENS A FILE and can evict another handle.
        ``get`` reads as a side-effect-free peek and is not one; ``items`` and
        ``values`` would open every shard in the checkpoint, which is the cost
        #926 exists to remove. Subscript the paths you actually need instead.
        """
        raise TypeError(
            "_SourceHandles does not support get/items/values/equality: they "
            "would open source files through __getitem__. Subscript the paths "
            "you need (handles[path]) instead."
        )

    get = _refuse
    items = _refuse
    values = _refuse
    __eq__ = _refuse
    __ne__ = _refuse
    __hash__ = None  # type: ignore[assignment]  # mutable cache, never a key

    def close(self) -> None:
        """Release every live handle; a raising release does not stop the rest."""
        self._released = True
        failures: List[Exception] = []
        while self._live:
            path, handle = self._live.popitem(last=False)
            try:
                handle.__exit__(None, None, None)
            except Exception as exc:  # noqa: BLE001 — collected, reported below
                failures.append(exc)
                logger.warning("layer-stream sharder: releasing %s failed: %s", path, exc)
        if failures:
            # Only the first can propagate, so the rest are logged above rather
            # than dropped: with capacity 2 a "both handles failed" close is an
            # ordinary case, not a contrived one, and a lost second failure is
            # the same silence #926 is about.
            raise failures[0]

    def __enter__(self) -> "_SourceHandles":
        return self

    def __exit__(
        self,
        exc_type: Optional[type] = None,
        exc: Optional[BaseException] = None,
        traceback: Optional[Any] = None,
    ) -> None:
        self.close()


def _own(view: Any, candidate: Any) -> Any:
    """``candidate`` if it already owns its memory, else a copy of ``view``.

    ``get_tensor`` is a zero-copy view whose storage keeps the WHOLE source
    file mapped for as long as the view lives — measured: the view still reads
    correctly after its handle's ``__exit__``, so releasing a handle is not
    what frees the mapping, dropping the last view into it is. The sharder
    keeps norms, embeddings and the head until it writes them, so a retained
    view would hold its file's mapping, and on Windows its file's whole COMMIT
    CHARGE, past the handle's release and defeat the ``_SourceHandles`` bound
    (#926). ``Tensor.to`` and ``Tensor.contiguous`` return ``self`` when they
    have nothing to do, so identity is the exact test for "still the view".

    The cost is real and is accepted deliberately: a read whose dtype already
    matches used to pass the mapping through untouched, and now copies it, so
    pass 2 moves roughly the checkpoint's own bytes an extra time (bounded to
    one layer at a time, so no new peak). It buys the bound — holding a 1.71 GB
    tensor as a view charges its 6.85 GB file, where the copy charges 1.71 GB —
    which is what lets a 70B shard at all on this platform.
    """
    return view.clone() if candidate is view else candidate


def _read_tensor(handle: Any, key: str, dtype: str) -> "Any":
    """Materialise one OWNED tensor, size-capped, converted to the target dtype."""
    import torch

    shape = handle.get_slice(key).get_shape()
    elements = math.prod(int(dim) for dim in shape)
    if elements > _MAX_TENSOR_ELEMENTS:
        raise ValueError(
            f"tensor {key} is too large for layer streaming "
            f"({elements} elements > {_MAX_TENSOR_ELEMENTS})"
        )
    target = getattr(torch, dtype)
    view = handle.get_tensor(key)
    return _own(view, view.to(target).contiguous())


def _read_raw_tensor(handle: Any, key: str) -> "Any":
    """Materialise one OWNED tensor without destroying packed integer storage."""
    shape = handle.get_slice(key).get_shape()
    elements = math.prod(int(dim) for dim in shape)
    if elements > _MAX_TENSOR_ELEMENTS:
        raise ValueError(
            f"tensor {key} is too large for layer streaming "
            f"({elements} elements > {_MAX_TENSOR_ELEMENTS})"
        )
    view = handle.get_tensor(key)
    return _own(view, view.contiguous())


def _read_oq_tensor(
    handles: Mapping[str, Any],
    where: Mapping[str, Tuple[str, str]],
    key: str,
    *,
    oq_config: Any,
    dtype: str,
) -> "Any":
    """Read one oQ matrix, or an ordinary floating weight without companions."""
    location = where[key]
    base = key.removesuffix(".weight")
    scales_location = where.get(base + ".scales")
    biases_location = where.get(base + ".biases")
    if scales_location is None and biases_location is None:
        tensor = _read_tensor(handles[location[0]], location[1], dtype)
        return _normalise_oq_layout(key, tensor)
    if scales_location is None or biases_location is None:
        raise ValueError(f"oQ affine weight {key!r} has only one companion tensor")
    if oq_config is None:
        raise ValueError(f"oQ affine weight {key!r} has no quantization metadata")

    from soup_cli.utils.oq_affine import dequantize_affine

    module_name = location[1].removesuffix(".weight")
    spec = oq_config.for_module(module_name)
    packed = _read_raw_tensor(handles[location[0]], location[1])
    scales = _read_raw_tensor(handles[scales_location[0]], scales_location[1])
    biases = _read_raw_tensor(handles[biases_location[0]], biases_location[1])
    try:
        tensor = dequantize_affine(
            packed,
            scales,
            biases,
            spec=spec,
            dtype=dtype,
        )
        return _normalise_oq_layout(key, tensor)
    finally:
        del packed, scales, biases


def _normalise_oq_layout(key: str, tensor: Any) -> Any:
    """Convert MLX-only storage conventions to Transformers parameter layouts."""
    if key.endswith(".conv1d.weight") and tensor.ndim == 3 and tensor.shape[-1] == 1:
        return tensor.transpose(-1, -2).contiguous()
    return tensor
