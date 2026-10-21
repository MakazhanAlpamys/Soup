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
from dataclasses import replace as dataclass_replace
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from soup_cli import __version__
from soup_cli.utils.expert_stream import (
    EXPERT_LAYOUT_PER_EXPERT,
    expert_shard_name,
    write_expert_shard,
)
from soup_cli.utils.fp8_block import (
    SUPPORTED_FP8_FORMAT,
    FP8BlockPlan,
    UnsupportedFP8Error,
    check_source_quantization,
    dequantize_fp8_block,
    plan_fp8_block,
    read_checkpoint_quantization_config,
    refuse_unsupported_tensor,
)

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
#: Blocks per float32 nested absmax under double quantisation (bitsandbytes'
#: ``compress_statistics``); recorded per weight as ``nested_blocksize``.
NF4_NESTED_BLOCKSIZE = 256
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
_RESHARDING_INDEX_NAME = "index.json.resharding"
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
_PER_EXPERT_RE = re.compile(
    r"^(?P<prefix>model\.layers\.\d+\.mlp\.experts)\."
    r"(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)
#: Architectures whose transformers module holds the routed experts as fused
#: 3-D ``gate_up_proj`` [E, 2I, H] / ``down_proj`` [E, H, I] parameters while
#: the checkpoint stores one 2-D tensor per expert and projection (the
#: ``qwen2_moe`` conversion in transformers' ``conversion_mapping.py``: gate
#: rows first, experts in index order). Value = the name used in messages.
_STACKED_EXPERT_ARCHS = {"qwen4_exp": "Qwen4", "deepseek_v3": "DeepSeek-V3"}
#: Per-layer BUFFERS the checkpoint carries and a streamed run must use. The
#: skeleton is built under ``init_empty_weights()``, which leaves buffers at
#: their ``__init__`` value, so the runtime loads exactly these from the layer
#: shard (``layer_stream_runtime._load_layer_buffers``); every other buffer
#: keeps its init value, as it always has (a qwen4_exp integer buffer, for one,
#: reaches the shard cast to the stream dtype and must not be loaded back).
#: Stored at float32 whatever the stream dtype: transformers keeps them in fp32
#: on load (``_keep_in_fp32_modules_strict``), so a resident bf16 run routes with
#: the fp32 values and the streamed run must as well.
_LAYER_BUFFER_KEYS = {"deepseek_v3": ("mlp.gate.e_score_correction_bias",)}


def layer_buffer_keys(arch: str) -> Tuple[str, ...]:
    """Short keys of the per-layer buffers streamed for ``arch`` (see above)."""
    return _LAYER_BUFFER_KEYS.get(arch, ())


#: Architectures admitted to ``expert_layout="per_expert"``: each one passed
#: the resident-vs-streamed parity tests in ``tests/test_expert_stream.py``.
_PER_EXPERT_LAYOUT_ARCHS = ("deepseek_v3",)
#: Public name for the trainer's early refusal (``training.stream_expert_layout``).
PER_EXPERT_LAYOUT_ARCHS = _PER_EXPERT_LAYOUT_ARCHS
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
    #: R4: realpaths of the EXTRA cache roots this cache's decoder layers are spread over, in
    #: order; root 0 is the directory holding this index. Empty == one root.
    stripe_roots: Tuple[str, ...] = ()
    #: Per decoder layer, the root holding its file (0 = this directory, k = stripe_roots[k-1]).
    #: Empty == every layer here.
    layer_roots: Tuple[int, ...] = ()
    #: "" keeps a MoE layer's routed experts inside its decoder-layer file (the
    #: fused 3-D tensors); "per_expert" writes them to ``experts_NNN.safetensors``
    #: beside the layer file, one tensor pair per expert, streamed one expert at
    #: a time (``utils/expert_stream.py``). Deliberately NOT a format-version
    #: bump (that would invalidate every existing cache): a Soup from before this
    #: field reads a per_expert cache as fused and refuses it at install with
    #: "shard is missing decoder weights"; reshard with that version to use it.
    expert_layout: str = ""
    #: per_expert: the decoder layers that have an experts file, ascending.
    expert_layers: Tuple[int, ...] = ()
    #: per_expert: routed experts per MoE layer (the same in every MoE layer).
    expert_count: int = 0
    #: per_expert: stored dense shape of one expert's "gate_up" and "down".
    expert_shapes: Mapping[str, Tuple[int, ...]] = field(default_factory=dict)
    #: per_expert under NF4: the spec of "gate_up" and "down". Empty otherwise.
    expert_quant_specs: Mapping[str, NF4WeightSpec] = field(default_factory=dict)


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


#: Written into every stripe root's per-model folder; the cache check compares it to the index.
STRIPE_MARKER_NAME = "stripe.json"
#: A marker Soup writes is ~250 bytes; anything past this is not one, and is not parsed.
_MAX_STRIPE_MARKER_BYTES = 4096


def stripe_dirs(shard_dir: str, stripe_roots: Sequence[str]) -> List[str]:
    """Root 0 is ``shard_dir``; stripe root k holds this primary cache's own folder in it.

    The folder is ``<slug>-<hash of the primary cache's realpath>``
    (``stripe_roots.stripe_folder_name``): two primary caches of one base that share a stripe
    root never share a folder. The one resolver — nothing else builds a stripe folder path.
    """
    from soup_cli.utils.stripe_roots import stripe_folder_name

    name = stripe_folder_name(shard_dir)
    return [shard_dir] + [os.path.join(root, name) for root in stripe_roots]


def layer_paths(shard_dir: str, index: "ShardIndex") -> List[str]:
    """Where each decoder layer's file lives. Every caller that used to build a layer path
    from ``shard_dir`` alone goes through this, so a striped cache is never half-understood."""
    placement = tuple(getattr(index, "layer_roots", ()) or ())
    if not placement:
        return [layer_shard_path(shard_dir, idx) for idx in range(index.n_layers)]
    dirs = stripe_dirs(shard_dir, tuple(index.stripe_roots))
    return [layer_shard_path(dirs[root], idx) for idx, root in enumerate(placement)]


def expert_paths(shard_dir: str, index: "ShardIndex") -> Dict[int, str]:
    """Decoder layer -> its experts file, beside the layer's own file (same root)."""
    if getattr(index, "expert_layout", "") != EXPERT_LAYOUT_PER_EXPERT:
        return {}
    paths = layer_paths(shard_dir, index)
    return {
        layer: os.path.join(os.path.dirname(paths[layer]), expert_shard_name(layer))
        for layer in index.expert_layers
    }


def _same_roots(first: Sequence[str], second: Sequence[str]) -> bool:
    """Equal as folders, not as strings: case and a trailing separator differ on Windows."""

    def canon(roots: Sequence[str]) -> List[str]:
        return [os.path.normcase(os.path.normpath(root)) for root in roots]

    return canon(first) == canon(second)


def _stripe_marker_payload(
    *, fingerprint: str, position: int, n_roots: int, primary: str
) -> Dict[str, Any]:
    """What a stripe folder's marker says: whose source, which position, and WHICH primary
    cache it belongs to — so a folder holding another cache's marker is a miss for this one."""
    return {
        "source_fingerprint": fingerprint,
        "position": position,
        "n_roots": n_roots,
        "primary_cache": primary,
    }


def _write_stripe_marker(
    directory: str, *, fingerprint: str, position: int, n_roots: int, primary: str
) -> None:
    payload = _stripe_marker_payload(
        fingerprint=fingerprint, position=position, n_roots=n_roots, primary=primary
    )
    fd, tmp = tempfile.mkstemp(prefix=".soup.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(tmp, os.path.join(directory, STRIPE_MARKER_NAME))
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _stripe_marker_problem(
    directory: str, *, fingerprint: str, position: int, n_roots: int, primary: str
) -> Optional[str]:
    """Why ``directory``'s marker does not describe this cache, or ``None`` when it does.

    The marker sits on a drive outside the contained primary cache, so its content is treated
    as untrusted: at most ``_MAX_STRIPE_MARKER_BYTES`` are read, only a JSON object is
    compared, and the reason NEVER quotes it (it is printed to a terminal). Deep nesting is a
    RecursionError inside ``json``, which is a miss like any other malformed marker.
    """
    path = os.path.join(directory, STRIPE_MARKER_NAME)
    try:
        with open(path, "rb") as handle:
            raw = handle.read(_MAX_STRIPE_MARKER_BYTES + 1)
    except OSError:
        return f"stripe marker {path} is missing or unreadable"
    if len(raw) > _MAX_STRIPE_MARKER_BYTES:
        return f"{path} is not a stripe marker Soup wrote (larger than any marker)"
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, RecursionError):
        return f"{path} is not a stripe marker Soup wrote (not valid JSON)"
    if not isinstance(payload, dict):
        return f"{path} is not a stripe marker Soup wrote (not a JSON object)"
    expected = _stripe_marker_payload(
        fingerprint=fingerprint, position=position, n_roots=n_roots, primary=primary
    )
    if payload != expected:
        return f"stripe marker {path} belongs to another cache"
    return None


def read_shard_index(out_dir: str, *, index_name: str = _INDEX_NAME) -> ShardIndex:
    """Read a shard index. Raises on a missing or malformed index."""
    path = os.path.join(out_dir, index_name)
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    # `quant` defaults to "none" so a v0.72.0/.1 bf16 cache stays valid for a
    # bf16 request rather than forcing every existing user to re-shard.
    specs = payload.get("quant_specs") or {}
    external = payload.get("external_tensors") or {}
    if not isinstance(external, dict):
        raise ValueError("external_tensors must be an object")
    n_layers = int(payload["n_layers"])
    stripe_roots = tuple(str(root) for root in (payload.get("stripe_roots") or ()))
    layer_roots = tuple(int(root) for root in (payload.get("layer_roots") or ()))
    if layer_roots and len(layer_roots) != n_layers:
        raise ValueError(f"layer_roots names {len(layer_roots)} layers; the index has {n_layers}")
    if stripe_roots and not layer_roots:
        raise ValueError("stripe_roots are set but layer_roots is empty")
    if any(not 0 <= root <= len(stripe_roots) for root in layer_roots):
        raise ValueError("layer_roots names a root the index does not have")
    expert_layout = str(payload.get("expert_layout", ""))
    if expert_layout not in ("", EXPERT_LAYOUT_PER_EXPERT):
        raise ValueError(f"unknown expert_layout {expert_layout!r}")
    expert_layers = tuple(int(layer) for layer in (payload.get("expert_layers") or ()))
    if list(expert_layers) != sorted(set(expert_layers)) or any(
        not 0 <= layer < n_layers for layer in expert_layers
    ):
        raise ValueError("expert_layers must be distinct ascending decoder layer indices")
    expert_count = int(payload.get("expert_count", 0))
    expert_shapes = {
        str(part): tuple(int(dim) for dim in shape)
        for part, shape in (payload.get("expert_shapes") or {}).items()
    }
    if bool(expert_layers) != (expert_layout == EXPERT_LAYOUT_PER_EXPERT) or (
        expert_layers
        and (
            not 1 <= expert_count <= 4096
            or set(expert_shapes) != {"gate_up", "down"}
            or any(len(shape) != 2 or min(shape) < 1 for shape in expert_shapes.values())
        )
    ):
        raise ValueError("the index's expert fields do not describe one layout")
    return ShardIndex(
        n_layers=n_layers,
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
        stripe_roots=stripe_roots,
        layer_roots=layer_roots,
        expert_layout=expert_layout,
        expert_layers=expert_layers,
        expert_count=expert_count,
        expert_shapes=expert_shapes,
        expert_quant_specs={
            str(part): NF4WeightSpec.from_json(value)
            for part, value in (payload.get("expert_quant_specs") or {}).items()
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


def resolve_cache_root(cache_dir: Optional[str] = None) -> str:
    """Cache ROOT only (explicit arg > env override > default). No side effects."""
    from soup_cli.utils.paths import is_under

    return _resolve_cache_root(cache_dir, is_under)


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
    """Sorted, non-symlinked checkpoint files under ``weights_dir``.

    The checkpoint's files are what the index names, or ``model.safetensors`` /
    the ``model-*-of-*`` shards (#1806); Mistral's ``consolidated.safetensors``
    beside them used to put every one of its tensors into the resident extras
    shard and double ``source_weight_bytes``.
    """
    from soup_cli.utils.checkpoint_files import checkpoint_files

    root = os.path.realpath(os.path.expanduser(weights_dir))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"weights directory not found: {weights_dir}")
    selected = checkpoint_files(root)
    found = []
    for name in selected.weights:
        path = os.path.join(root, name)
        if os.path.islink(path):
            logger.warning("layer-stream sharder: skipping symlinked shard %s", name)
            continue
        if not os.path.isfile(path):
            continue
        found.append(path)
    if not found:
        raise FileNotFoundError(
            f"{selected.why_empty(weights_dir)} — layer streaming needs a safetensors "
            f"checkpoint"
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


#: Header bytes a safetensors file spends per tensor besides the tensor's name: its
#: dtype, shape and offsets, with the file's 8-byte length field and alignment padding
#: spread over its tensors (at most 87 per tensor on the Moonlight-16B caches).
_HEADER_BYTES_PER_TENSOR = 96
#: ``index.json``: its scalar fields plus a share per listed entry (source file,
#: stored key, quantised suffix, decoder layer). Measured 62-91 bytes per entry, from
#: a one-layer Llama to Moonlight-16B (7.9-8.7 KB for ~116 entries).
_INDEX_BYTES_FIXED = 1024
_INDEX_BYTES_PER_ENTRY = 128
_PER_EXPERT_PROJECTIONS = frozenset({"gate_proj", "up_proj", "down_proj"})


def nf4_weight_bytes(numel: int, *, double_quant: bool) -> int:
    """Bytes :func:`_quantize_nf4` stores for ONE weight of ``numel`` elements.

    The packed nibbles (``ceil(numel / 2)`` uint8) plus one absmax per
    ``NF4_BLOCKSIZE`` block: float32 without double quantisation; with it a uint8
    per block, one float32 nested absmax per ``NF4_NESTED_BLOCKSIZE`` blocks and
    one float32 offset. Exact: pinned against bitsandbytes' own output.
    """
    if isinstance(numel, bool) or not isinstance(numel, int) or numel <= 0:
        raise ValueError(f"an NF4 weight needs a positive element count; got {numel!r}")
    blocks = -(-numel // NF4_BLOCKSIZE)
    packed = (numel + 1) // 2
    if not double_quant:
        return packed + 4 * blocks
    return packed + blocks + 4 * -(-blocks // NF4_NESTED_BLOCKSIZE) + 4


def _nf4_sidecar_suffixes(double_quant: bool) -> Tuple[str, ...]:
    """The stored key suffixes of one NF4 weight (see :func:`_quantize_nf4`)."""
    nested = (NESTED_ABSMAX_SUFFIX, NESTED_OFFSET_SUFFIX) if double_quant else ()
    return ("", ABSMAX_SUFFIX) + nested


def _layer_tensor_storage(
    short: str,
    *,
    dtype: str,
    quantise: bool,
    suffixes: Iterable[str],
    fp32_keys: Iterable[str],
) -> Tuple[str, bool]:
    """``(stored dtype, NF4?)`` for one decoder-layer key that is not a routed expert.

    The one rule :func:`shard_checkpoint` writes by and
    :func:`estimate_shard_cache_bytes` sizes by: a key the meta probe found
    converted to ``Linear4bit`` (``suffixes``) is NF4, a layer buffer the runtime
    loads back stays float32, everything else is stored at ``dtype``.
    """
    stored = "float32" if short in fp32_keys else dtype
    return stored, bool(quantise) and short in suffixes


def _routed_experts_quantised(expert_layout: str, quantise: bool) -> bool:
    """Whether a stacked-expert architecture's routed experts are stored as NF4.

    ``per_expert`` quantises each expert matrix on its own under NF4. The default
    fused layout stacks them into the 3-D ``mlp.experts.gate_up_proj`` /
    ``down_proj`` parameters, which ``replace_with_bnb_linear`` does not convert
    (they are not ``nn.Linear`` weights), so they stay at the stream dtype.
    """
    return bool(quantise) and expert_layout == EXPERT_LAYOUT_PER_EXPERT


def _fp8_consumed_keys(plan: Optional[FP8BlockPlan]) -> FrozenSet[str]:
    """Source keys the cache never stores: an FP8 block source's ``weight_scale_inv``.

    The rule :func:`shard_checkpoint` skips by and :func:`estimate_shard_cache_bytes`
    charges nothing for. The FP8 weight itself is stored, dequantised, by the ordinary
    rules above, so it is charged like any other weight of its element count.
    """
    return plan.scales if plan is not None else frozenset()


class SourceHeaderError(ValueError):
    """A source safetensors header could not be read or parsed.

    Its own type so the disk pre-flight can fall back to an approximate figure for
    exactly this, while a bad request or a checkpoint the sharder refuses still raises.
    """


#: Stored types safetensors defines that ``safetensors_reader`` has no spelling for and a
#: quantiser writes: packed or integer ones. FP8 spellings are left to the FP8 plan, and
#: anything else (C64, a corrupt name) stays an unreadable header.
_SAFETENSORS_ONLY_DTYPES = frozenset({"U16", "U32", "U64", "F4", "F6_E2M3", "F6_E3M2"})

#: What reading and parsing one header can raise: a missing or unreadable file, a
#: malformed header (``ValueError``), a null or nested dimension or offset
#: (``TypeError``), an infinite one (``OverflowError``) and JSON nested deeper than
#: the parser recurses (``RecursionError``: 200 KB of ``[`` is enough).
_HEADER_READ_ERRORS = (OSError, ValueError, TypeError, OverflowError, RecursionError)


def _read_source_tensors(
    shard_paths: Sequence[str],
) -> Dict[str, Tuple[str, Tuple[int, ...]]]:
    """Canonical key -> (safetensors dtype, shape) over every source file, from the headers.

    A header that cannot be read becomes :class:`SourceHeaderError`. One tensor
    spelled twice, or more than ``_MAX_TOTAL_TENSORS`` across the files, is
    refused the way :func:`shard_checkpoint` refuses it.
    """
    from soup_cli.utils.safetensors_reader import (
        RAW_DTYPES,
        UnsupportedDtypeError,
        read_header,
    )

    shapes: Dict[str, Tuple[str, Tuple[int, ...]]] = {}
    spelled: Dict[str, str] = {}
    for path in shard_paths:
        try:
            header = read_header(path)
        except UnsupportedDtypeError as exc:
            # A packed or integer type this reader has no spelling for (U32, U16, F4...):
            # the sharder reads the header through safetensors and refuses that tensor
            # (``plan_fp8_block``), so refuse it here in its words, before any copy.
            # FP8 spellings it lacks (F8_E8M0) are the FP8 plan's to word: unread here.
            if exc.dtype in _SAFETENSORS_ONLY_DTYPES:
                refuse_unsupported_tensor(_canonical_stream_key(exc.tensor), exc.dtype)
            raise SourceHeaderError(
                f"cannot read the safetensors header of {path}: {exc}"
            ) from exc
        except _HEADER_READ_ERRORS as exc:
            raise SourceHeaderError(
                f"cannot read the safetensors header of {path}: {exc}"
            ) from exc
        for source_key, entry in header.items():
            key = _canonical_stream_key(source_key)
            prior = spelled.get(key)
            if prior is not None:
                raise ValueError(
                    f"checkpoint spells one streamed tensor twice: {prior!r} and "
                    f"{source_key!r} both canonicalise to {key!r}. Refusing rather than "
                    f"silently keeping one by safetensors key order."
                )
            spelled[key] = source_key
            shapes[key] = (RAW_DTYPES[entry.dtype], entry.shape)
            if len(shapes) > _MAX_TOTAL_TENSORS:
                raise ValueError(f"checkpoint declares more than {_MAX_TOTAL_TENSORS} tensors")
    return shapes


@dataclass(frozen=True)
class ShardCacheEstimate:
    """What :func:`shard_checkpoint` will write, sized from the source headers.

    ``layer_bytes[i]`` is the tensor payload of decoder layer ``i``: its layer file
    plus, under ``expert_layout="per_expert"``, its experts file (both are written
    to the same root). ``other_bytes`` is the payload of the extras shard and the
    large embed/head shards, all on root 0. Both are exact. ``header_bytes`` and
    ``index_bytes`` bound the safetensors headers and ``index.json`` from above.

    ``expert_file_bytes`` is the part of ``layer_bytes`` that lives in the
    per_expert experts files (0 for the fused layout), and ``code_table_bytes``
    the part of ``other_bytes`` that is the shared NF4 code tables (0 without
    NF4). Neither is held by the RAM tier; see :attr:`ram_tier_bytes`.
    """

    layer_bytes: Tuple[int, ...]
    other_bytes: int
    header_bytes: int
    index_bytes: int
    expert_file_bytes: int = 0
    code_table_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        return sum(self.layer_bytes) + self.other_bytes + self.header_bytes + self.index_bytes

    @property
    def ram_tier_bytes(self) -> int:
        """Host RAM the RAM tier holds for this cache: what the post-shard check charges.

        The decoder-layer files plus the large embed/head shards (the store), plus
        the resident extras. The per_expert experts files are read from the cache
        volume on every tier, and the NF4 code tables are machinery that
        ``extras_resident_bytes`` leaves out, so neither is counted.
        """
        return (
            sum(self.layer_bytes)
            - self.expert_file_bytes
            + self.other_bytes
            - self.code_table_bytes
        )


def estimate_shard_cache_bytes(
    shard_paths: Sequence[str],
    *,
    dtype: str,
    arch: str = "",
    quant: str = QUANT_NONE,
    quant_suffixes: Iterable[str] = (),
    double_quant: bool = True,
    expert_layout: str = "",
    num_hidden_layers: Optional[int] = None,
    quantization_config: Optional[Mapping[str, Any]] = None,
) -> Optional[ShardCacheEstimate]:
    """Size the cache :func:`shard_checkpoint` writes for the same arguments.

    Reads only the safetensors headers of ``shard_paths`` (shapes and dtypes; no
    tensor is loaded and no file is mapped) and stores each tensor the way the
    sharder does: :func:`_layer_tensor_storage` for every decoder key,
    :func:`_routed_experts_quantised` for the routed experts of a stacked-expert
    architecture, :func:`nf4_weight_bytes` for each NF4 weight, the same
    multi-token-prediction drop for ``deepseek_v3`` and the stream ``dtype`` for
    everything outside the decoder layers.

    ``quantization_config`` is the checkpoint config's (``None`` when it has none).
    An FP8 block source is classified by :func:`plan_fp8_block`, as the sharder
    classifies it: each FP8 weight is charged at the rate its dequantised tensor is
    stored at (the rules above, by element count), and its ``weight_scale_inv`` is
    consumed and charged nothing.

    Returns ``None`` for ``qwen4_exp``, whose external PLE table and oQ layout
    have their own estimators. The arguments are checked before any file is read
    (``ValueError``). A header that cannot be read raises
    :class:`SourceHeaderError`; a checkpoint the sharder would refuse raises
    ``ValueError`` in the sharder's words, before anything is sized by its layer
    count.
    """
    if arch == "qwen4_exp":
        return None
    # The config alone decides an unsupported FP8 spelling, or another declared
    # quantiser: refused before any header, as the sharder refuses it.
    check_source_quantization(quantization_config, arch=arch)
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r} for layer streaming; "
            f"supported: {', '.join(_SUPPORTED_DTYPES)}"
        )
    if expert_layout not in ("", EXPERT_LAYOUT_PER_EXPERT):
        raise ValueError(
            f"unsupported expert_layout {expert_layout!r}; supported: "
            f"'' (fused) or {EXPERT_LAYOUT_PER_EXPERT!r}"
        )
    suffixes = _validate_quant_request(quant, quant_suffixes)
    quantise = quant == QUANT_NF4

    from soup_cli.utils.expert_stream import DOWN, GATE_UP, expert_key
    from soup_cli.utils.safetensors_reader import _ITEMSIZE

    tensors = _read_source_tensors(shard_paths)
    fp8_plan = plan_fp8_block(quantization_config, tensors, arch=arch)
    consumed = _fp8_consumed_keys(fp8_plan)

    itemsize = _ITEMSIZE[dtype]
    stack_experts = arch in _STACKED_EXPERT_ARCHS
    fp32_keys = layer_buffer_keys(arch)
    drop_extra_layers = arch == "deepseek_v3" and num_hidden_layers is not None
    experts_nf4 = _routed_experts_quantised(expert_layout, quantise)
    sidecars = _nf4_sidecar_suffixes(double_quant)

    layer_bytes: Dict[int, int] = {}
    routed: Dict[int, Dict[int, Dict[str, int]]] = {}
    other_bytes = 0
    header_bytes = 0
    expert_file_bytes = 0
    code_table_bytes = 0
    stored_names: set = set()
    any_nf4 = False

    def _charge_names(names: Iterable[str], *, indexed: bool = True) -> int:
        """Header bytes for ``names``; ``indexed`` ones are also listed in index.json."""
        names = tuple(names)
        if indexed:
            stored_names.update(names)
        return sum(len(name) + _HEADER_BYTES_PER_TENSOR for name in names)

    for key, (_source_dtype, shape) in tensors.items():
        if key in consumed:
            continue
        numel = math.prod(shape)
        match = _LAYER_RE.match(key)
        if match is None:
            other_bytes += numel * itemsize
            header_bytes += _charge_names((key,))
            continue
        idx = int(match.group(1))
        if drop_extra_layers and idx >= num_hidden_layers:
            continue
        layer_bytes.setdefault(idx, 0)
        expert_match = _PER_EXPERT_RE.match(key) if stack_experts else None
        if expert_match:
            expert = int(expert_match.group("expert"))
            projections = routed.setdefault(idx, {}).setdefault(expert, {})
            projections[expert_match.group("projection")] = numel
            continue
        short = match.group(2)
        stored, nf4 = _layer_tensor_storage(
            short, dtype=dtype, quantise=quantise, suffixes=suffixes, fp32_keys=fp32_keys
        )
        if nf4:
            any_nf4 = True
            layer_bytes[idx] += nf4_weight_bytes(numel, double_quant=double_quant)
            header_bytes += _charge_names(short + suffix for suffix in sidecars)
        else:
            layer_bytes[idx] += numel * _ITEMSIZE[stored]
            header_bytes += _charge_names((short,))

    for idx, experts in routed.items():
        for expert, projections in experts.items():
            if set(projections) != _PER_EXPERT_PROJECTIONS:
                raise ValueError(
                    f"layer {idx} expert {expert} has incomplete expert projections: "
                    f"{sorted(projections)}"
                )
        # gate and up are concatenated into ONE gate_up matrix before it is stored.
        pairs = [
            (projections["gate_proj"] + projections["up_proj"], projections["down_proj"])
            for projections in experts.values()
        ]
        if experts_nf4:
            any_nf4 = True
            experts_payload = sum(
                nf4_weight_bytes(gate_up, double_quant=double_quant)
                + nf4_weight_bytes(down, double_quant=double_quant)
                for gate_up, down in pairs
            )
        else:
            experts_payload = sum(gate_up + down for gate_up, down in pairs) * itemsize
        layer_bytes[idx] += experts_payload
        if expert_layout == EXPERT_LAYOUT_PER_EXPERT:
            # In the experts file, not the layer file: never held by the RAM tier.
            expert_file_bytes += experts_payload
            # experts_NNN.safetensors, whose keys index.json does not list.
            parts = sidecars if experts_nf4 else ("",)
            names = [
                expert_key(expert, part) + suffix
                for expert in experts
                for part in (GATE_UP, DOWN)
                for suffix in parts
            ]
            header_bytes += _charge_names(names, indexed=False)
        else:
            # Fused: the two stacked tensors in the layer file.
            header_bytes += _charge_names(("mlp.experts.gate_up_proj", "mlp.experts.down_proj"))

    if not layer_bytes:
        raise ValueError(
            "no decoder layer weights (model.layers.N.*) found; layer streaming has "
            "nothing to stream"
        )
    n_layers = max(layer_bytes) + 1
    # Before anything below is sized by the layer count: a header naming layer
    # 8,000,000 would otherwise cost ~780 MB just to say the layers are missing.
    if n_layers > _MAX_LAYERS:
        raise ValueError(f"too many decoder layers ({n_layers} > {_MAX_LAYERS})")
    if len(layer_bytes) != n_layers:
        missing = sorted(set(range(n_layers)) - set(layer_bytes))
        raise ValueError(f"decoder layer indices are not contiguous; missing {missing[:8]}")
    if any_nf4:
        # The shared code tables in extras: the 16 float32 NF4 levels, plus the
        # 256-entry float32 nested map under double quantisation.
        tables = {NF4_CODE_KEY: 16 * 4}
        if double_quant:
            tables[NF4_NESTED_CODE_KEY] = 256 * 4
        code_table_bytes = sum(tables.values())
        other_bytes += code_table_bytes
        header_bytes += _charge_names(tables)
    index_entries = len(shard_paths) + len(stored_names) + len(suffixes) + n_layers
    return ShardCacheEstimate(
        layer_bytes=tuple(layer_bytes[idx] for idx in range(n_layers)),
        other_bytes=other_bytes,
        header_bytes=header_bytes,
        index_bytes=_INDEX_BYTES_FIXED + _INDEX_BYTES_PER_ENTRY * index_entries,
        expert_file_bytes=expert_file_bytes,
        code_table_bytes=code_table_bytes,
    )


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


def source_identity_includes_config(arch: str) -> bool:
    """Whether ``config.json`` is part of a shard cache's source identity.

    ``qwen4_exp`` reads its PLE layout from the config; for ``deepseek_v3`` the
    config's layer count decides which (multi-token-prediction) layers are
    dropped. The pre-sharding planner and the sharder must agree on this, or
    the planner predicts a cache miss the sharder then does not see.
    """
    return arch in ("qwen4_exp", "deepseek_v3")


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
    payload["stripe_roots"] = list(index.stripe_roots)
    payload["layer_roots"] = list(index.layer_roots)
    payload["quant_specs"] = {
        key: spec.to_json() for key, spec in index.quant_specs.items()
    }
    payload["expert_layers"] = list(index.expert_layers)
    payload["expert_shapes"] = {
        part: list(shape) for part, shape in index.expert_shapes.items()
    }
    payload["expert_quant_specs"] = {
        part: spec.to_json() for part, spec in index.expert_quant_specs.items()
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
    stripe_roots: Sequence[str] = (),
    expert_layout: str = "",
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
        if os.path.exists(os.path.join(out_dir, _RESHARDING_INDEX_NAME)):
            return None, "previous re-shard did not finish; cache index is unavailable"
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
    if index.expert_layout != expert_layout:
        return None, (
            f"expert layout changed ({index.expert_layout or 'fused'!r} -> "
            f"{expert_layout or 'fused'!r})"
        )
    if index.format_version != _SHARD_FORMAT_VERSION:
        return None, (
            f"shard format changed "
            f"({index.format_version!r} -> {_SHARD_FORMAT_VERSION!r})"
        )
    if not _same_roots(index.stripe_roots, stripe_roots):
        # Joined plainly: a list repr doubles every Windows backslash in the printed reason.
        return None, (
            f"stripe roots changed ({', '.join(index.stripe_roots) or 'none'} -> "
            f"{', '.join(stripe_roots) or 'none'})"
        )
    # Equal as folders is not equal as strings (normpath collapses `x/..` lexically, the kernel
    # does not), so from here on every path is built from the roots the CALLER validated.
    index = dataclass_replace(index, stripe_roots=tuple(stripe_roots))
    from soup_cli.utils.stripe_roots import primary_cache_identity, stripe_folder_problem

    dirs = stripe_dirs(out_dir, index.stripe_roots)
    for position, directory in enumerate(dirs[1:], start=1):
        # Before the marker is read: a link or junction at the folder is never followed.
        problem = stripe_folder_problem(directory, index.stripe_roots[position - 1])
        if problem is None:
            problem = _stripe_marker_problem(
                directory,
                fingerprint=index.source_fingerprint,
                position=position,
                n_roots=len(dirs),
                primary=primary_cache_identity(out_dir),
            )
        if problem is not None:
            return None, problem
    if not os.path.exists(extras_shard_path(out_dir)):
        return None, f"cached shard {_EXTRAS_NAME!r} is missing"
    for key in index.large_keys:
        path = large_shard_path(out_dir, key)
        if not os.path.exists(path):
            return None, f"cached shard {os.path.basename(path)!r} is missing"
    for path in [*layer_paths(out_dir, index), *expert_paths(out_dir, index).values()]:
        if not os.path.exists(path):
            if os.path.dirname(path) == out_dir:
                return None, f"cached shard {os.path.basename(path)!r} is missing"
            return None, f"cached shard {path} is missing"
    return index, "cache is reusable"


def _content_change_requires_index_quarantine(
    out_dir: str,
    *,
    dtype: str,
    fingerprint: str,
    quant: str,
    double_quant: bool,
    quant_device: str,
    external_mode: str,
    expert_layout: str = "",
) -> bool:
    """Return whether the next write could make the current index describe wrong bytes."""
    try:
        index = read_shard_index(out_dir)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return any(
        (
            index.dtype != dtype,
            index.quant != quant,
            index.quant != QUANT_NONE and index.double_quant != double_quant,
            index.quant != QUANT_NONE and index.quant_device != quant_device,
            index.source_fingerprint != fingerprint,
            index.external_mode != external_mode,
            index.expert_layout != expert_layout,
            index.format_version != _SHARD_FORMAT_VERSION,
        )
    )


def _quarantine_shard_index(out_dir: str) -> None:
    """Take the old commit point out of play before rewriting content bytes."""
    path = os.path.join(out_dir, _INDEX_NAME)
    if os.path.exists(path):
        os.replace(path, os.path.join(out_dir, _RESHARDING_INDEX_NAME))


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
    stripe_roots: Sequence[str] = (),
    num_hidden_layers: Optional[int] = None,
    expert_layout: str = "",
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

    ``stripe_roots`` (R4) are extra cache roots on other drives; decoder layer
    ``i`` is written to root ``i mod N`` (root 0 is ``out_dir``), each extra
    root gets a per-model folder named like ``out_dir`` plus a ``stripe.json``
    marker, and the index records the placement.

    For ``arch`` in :data:`_STACKED_EXPERT_ARCHS` the per-expert routed-expert
    keys (``mlp.experts.E.{gate,up,down}_proj.weight``) are stacked into the
    fused ``mlp.experts.gate_up_proj`` / ``mlp.experts.down_proj`` tensors the
    transformers module holds, the same conversion ``from_pretrained`` applies.
    For ``deepseek_v3``, ``num_hidden_layers`` (the config's value) drops the
    multi-token-prediction layers a DeepSeek-V3 / Kimi K2 checkpoint stores
    past the last decoder layer; transformers ignores them on load too.

    ``expert_layout="per_expert"`` (stacked-expert architectures only) writes a
    MoE layer's routed experts to ``experts_NNN.safetensors`` beside its layer
    file, one ``(gate_up, down)`` pair per expert, NF4-quantised per matrix
    under ``quant='nf4'``; the layer file then holds only the dense part.

    A DeepSeek-V3 FP8 block-quantised checkpoint (``config.json``'s
    ``quantization_config`` plus ``float8_e4m3fn`` weights with ``weight_scale_inv``
    block scales; see ``utils/fp8_block.py``) is detected from the checkpoint itself:
    each FP8 weight is dequantised to fp32 as it is read, per-expert ones before they
    are stacked or written to their experts file, cast to the stream dtype and then
    stored by the rules above. The scales are consumed and never stored. Any other
    FP8 spelling is refused before anything is written.

    Every other quantised source is refused too (``fp8_block.check_source_quantization``
    and the header plan, :func:`~soup_cli.utils.fp8_block.plan_fp8_block`). The source
    is checked first, from its ``config.json`` and headers alone, before ``out_dir`` or
    a stripe folder is made, a cache is looked up or an index is quarantined: a refused
    source creates and renames nothing, and a cache older code sharded from it is not
    reused.
    """
    if expert_layout not in ("", EXPERT_LAYOUT_PER_EXPERT):
        raise ValueError(
            f"unsupported expert_layout {expert_layout!r}; supported: "
            f"'' (fused) or {EXPERT_LAYOUT_PER_EXPERT!r}"
        )
    if expert_layout and arch not in _PER_EXPERT_LAYOUT_ARCHS:
        raise ValueError(
            f"expert_layout={expert_layout!r} is gated on resident parity per "
            f"architecture; admitted: {', '.join(_PER_EXPERT_LAYOUT_ARCHS)}; got arch={arch!r}"
        )
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r} for layer streaming; "
            f"supported: {', '.join(_SUPPORTED_DTYPES)}"
        )
    if num_hidden_layers is not None and (
        isinstance(num_hidden_layers, bool)
        or not isinstance(num_hidden_layers, int)
        or not 1 <= num_hidden_layers <= _MAX_LAYERS
    ):
        raise ValueError(
            f"num_hidden_layers must be an int in [1, {_MAX_LAYERS}], "
            f"got {num_hidden_layers!r}"
        )
    suffixes = _validate_quant_request(quant, quant_suffixes)
    quantise = quant == QUANT_NF4
    device = (quant_device or _default_quant_device()) if quantise else "cpu"
    # Only the KIND matters (cuda:0 and cuda:1 quantise identically); the
    # index stores that so the cache check below stays stable across ordinals.
    device_kind = str(device).split(":", 1)[0] if quantise else ""
    external_mode = "qwen4_ple" if arch == "qwen4_exp" else ""
    stack_experts = arch in _STACKED_EXPERT_ARCHS
    expert_label = _STACKED_EXPERT_ARCHS.get(arch, arch)
    # Fused routed experts are stacked at the stream dtype below and never quantised;
    # per_expert ones are NF4 under quant='nf4'. The disk pre-flight sizes by this too.
    experts_nf4 = _routed_experts_quantised(expert_layout, quantise)
    fp32_layer_keys = layer_buffer_keys(arch)
    drop_extra_layers = arch == "deepseek_v3" and num_hidden_layers is not None
    shards = _discover_safetensors(weights_dir)
    from safetensors import safe_open

    # Every check of the SOURCE comes first, before the out dir is made, a stripe
    # folder secured, a cache looked up or an index quarantined: a refused source
    # creates and renames nothing, and a cache sharded from it by older code is not
    # reused (its hit would return before any later check). Headers only, no tensor.
    # An FP8 spelling, or another quantiser, that the config alone rules out is refused
    # before any header is read.
    source_quantization = read_checkpoint_quantization_config(weights_dir)
    check_source_quantization(source_quantization, arch=arch)

    # pass 1 — build key -> shard without materialising a single tensor
    where: Dict[str, Tuple[str, str]] = {}
    source_tensors: Dict[str, Tuple[str, Tuple[int, ...]]] = {}
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
                header = handle.get_slice(source_key)
                source_tensors[key] = (
                    str(header.get_dtype()),
                    tuple(int(dim) for dim in header.get_shape()),
                )
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

    # Before any byte is read or written: every FP8 tensor must be a block weight with
    # its scale, every scale must sit beside one, and no tensor read below may be a
    # packed or integer one a plain read would cast (the disk pre-flight runs the same
    # plan). Over the keys this sharder reads: an oQ bundle's vision and MTP weights
    # are dropped above, never read.
    fp8_plan = plan_fp8_block(
        source_quantization, {key: source_tensors[key] for key in where}, arch=arch
    )
    consumed = _fp8_consumed_keys(fp8_plan)
    del source_tensors

    resolved_out = _validate_out_dir(out_dir)

    from soup_cli.utils.stripe_roots import (
        STRIPE_DIRS_ENV,
        StripeRootError,
        layer_roots_for,
        primary_cache_identity,
        secure_stripe_folder,
        stripe_folder_problem,
        validate_stripe_root,
    )

    # Bound our own writes (syntax, existence, symlink, overlap). The distinct-volume and
    # NVMe rules ran once at setup (resolve_stripe_roots); the ~9 s probe is not repeated.
    validated: List[str] = []
    for root in stripe_roots:
        validated.append(
            validate_stripe_root(
                root, primary_root=os.path.dirname(resolved_out), accepted=validated
            )
        )
    stripe = tuple(validated)
    # Before anything reads, writes or deletes in a stripe folder: it must not be a link or
    # junction, and it is made (or re-made) owner-only — on the reuse path too, since that is
    # the path that trusts files already on the drive. A folder that fails refuses the run.
    shard_dirs = stripe_dirs(resolved_out, stripe)
    for directory, root in zip(shard_dirs[1:], stripe):
        secure_stripe_folder(directory, root)

    source_files = checkpoint_source_components(
        weights_dir,
        _source_file_components(shards),
        include_config=source_identity_includes_config(arch),
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
            stripe_roots=stripe,
            expert_layout=expert_layout,
        )
        if cached is not None:
            if notify is not None:
                _notify_unindexed_cache_files(resolved_out, cached, notify)
            return cached
        if notify is not None:
            from soup_cli.utils.terminal import for_terminal

            # The reason can carry paths from the index, the variable and stripe drives: printed
            # as text, never as markup or raw control bytes.
            notify(f"[yellow]Re-sharding layer cache:[/] {for_terminal(miss_reason)}.")
            _notify_orphaned_stripes(resolved_out, stripe, notify)
        if _content_change_requires_index_quarantine(
            resolved_out,
            dtype=dtype,
            fingerprint=fingerprint,
            quant=quant,
            double_quant=double_quant,
            quant_device=device_kind,
            external_mode=external_mode,
            expert_layout=expert_layout,
        ):
            _quarantine_shard_index(resolved_out)

    if expert_layout and not any(_PER_EXPERT_RE.match(key) for key in where):
        raise ValueError(
            f"expert_layout={expert_layout!r} found no per-expert routed weights "
            f"(model.layers.N.mlp.experts.E.*) in {weights_dir} to split"
        )
    if drop_extra_layers:
        # DeepSeek-V3 / Kimi K2 store the multi-token-prediction module as the
        # layer(s) after the last decoder layer (``model.layers.61.*`` of 61).
        # The CausalLM has no such layer and transformers drops the keys on
        # load; kept here they would make a shard the skeleton cannot place.
        where = {
            key: location
            for key, location in where.items()
            if (match := _LAYER_RE.match(key)) is None
            or int(match.group(1)) < num_hidden_layers
        }
        layer_ids = {layer for layer in layer_ids if layer < num_hidden_layers}

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
    # The stripe folders were made owner-only (and checked for links) before the cache check.
    layer_roots = layer_roots_for(n_layers, len(shard_dirs))
    # Collected here, deleted only after the new index commits — see the comment beside the
    # deletion below for why.
    stale_root0_copies: List[str] = []
    expert_layers_written: List[int] = []
    expert_meta: Optional[Tuple[int, Dict[str, Any], Dict[str, Any]]] = None
    # At most _MAX_LIVE_SOURCE_HANDLES source files are mapped at once (#926);
    # a failed open, or a raise anywhere below, still releases the live ones.
    with _SourceHandles(shards, safe_open) as handles:
        for idx in range(n_layers):
            prefix = f"model.layers.{idx}."
            blob = {}
            stacked_experts: Dict[str, Dict[int, Tuple[str, str]]] = {}
            per_expert_ids: List[int] = []
            oq_experts: Dict[str, str] = {}
            for key in where:
                if not key.startswith(prefix):
                    continue
                if key in external_source_keys or key in consumed:
                    continue
                if saw_oq_companion and key.endswith((".scales", ".biases")):
                    continue
                oq_expert_match = (
                    _QWEN4_OQ_EXPERT_RE.match(key) if saw_oq_companion else None
                )
                if oq_expert_match:
                    oq_experts[oq_expert_match.group("projection")] = key
                    continue
                expert_match = _PER_EXPERT_RE.match(key) if stack_experts else None
                if expert_match:
                    projection = expert_match.group("projection")
                    expert = int(expert_match.group("expert"))
                    stacked_experts.setdefault(projection, {})[expert] = key
                    continue
                short = key[len(prefix):]
                # The same rule estimate_shard_cache_bytes sizes the disk pre-flight by.
                stored, as_nf4 = _layer_tensor_storage(
                    short,
                    dtype=dtype,
                    quantise=quantise,
                    suffixes=suffixes,
                    fp32_keys=fp32_layer_keys,
                )
                if saw_oq_companion and key.endswith(".weight"):
                    tensor = _read_oq_tensor(
                        handles, where, key, oq_config=oq_config, dtype=dtype
                    )
                else:
                    tensor = _read_source_tensor(handles, where, key, stored, fp8_plan)
                total_params += tensor.numel()
                if as_nf4:
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
            if stacked_experts:
                import torch

                expected_projections = {"gate_proj", "up_proj", "down_proj"}
                if set(stacked_experts) != expected_projections:
                    raise ValueError(
                        f"{expert_label} layer {idx} has incomplete expert projections: "
                        f"{sorted(stacked_experts)}"
                    )
                expert_ids = sorted(stacked_experts["gate_proj"])
                if expert_ids != list(range(len(expert_ids))) or any(
                    sorted(stacked_experts[name]) != expert_ids
                    for name in expected_projections
                ):
                    raise ValueError(
                        f"{expert_label} layer {idx} expert ids must be contiguous and "
                        "identical across gate/up/down projections"
                    )

                def _expert_tensor(projection: str, expert: int):
                    # An FP8 expert is dequantised here, before it is stacked or written.
                    return _read_source_tensor(
                        handles, where, stacked_experts[projection][expert], dtype, fp8_plan
                    )

                if expert_layout == EXPERT_LAYOUT_PER_EXPERT:
                    # Written after the layer file, in its root (below).
                    per_expert_ids = expert_ids
                    expert_ids = []
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
                        f"{expert_label} layer {idx} contains both packed and "
                        "per-expert weights"
                    )
                if gate_up:
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
            root = layer_roots[idx] if layer_roots else 0
            if root:
                # secure_stripe_folder ran once, before the loop started. Between then and
                # now the root could have been renamed and a link substituted in its place
                # (the folder itself still looks fine to a plain stat), so the same
                # real-path check is repeated right before every write that would otherwise
                # silently follow it.
                problem = stripe_folder_problem(shard_dirs[root], stripe[root - 1])
                if problem is not None:
                    raise StripeRootError(
                        f"{STRIPE_DIRS_ENV}: writing decoder layer {idx} to the stripe root "
                        f"{stripe[root - 1]} refused: {problem}. The existing cache index is "
                        f"unchanged."
                    )
            try:
                _atomic_save(blob, layer_shard_path(shard_dirs[root], idx))
            except OSError as exc:
                if not root:
                    raise
                # A full or vanished second drive: say which root and which variable. The
                # index on disk is still the old one, so nothing it names was touched.
                raise OSError(
                    exc.errno,
                    f"writing decoder layer {idx} to the stripe root {stripe[root - 1]} (from "
                    f"{STRIPE_DIRS_ENV}) failed: {exc}. The existing cache index is unchanged. "
                    f"Free space on that drive, or name another folder in {STRIPE_DIRS_ENV} "
                    f"(unset it to shard to one root).",
                ) from exc
            if per_expert_ids:
                import torch

                def _expert_pairs(ids=per_expert_ids, read=_expert_tensor):
                    for expert in ids:
                        gate = read("gate_proj", expert)
                        gate_up_one = torch.cat((gate, read("up_proj", expert)))
                        yield gate_up_one, read("down_proj", expert)

                info = write_expert_shard(
                    os.path.join(shard_dirs[root], expert_shard_name(idx)),
                    _expert_pairs(),
                    quantise=experts_nf4,
                    double_quant=double_quant,
                    device=device,
                )
                if experts_nf4:
                    tables.observe(f"{prefix}mlp.experts", *info.codes)
                described = (info.count, dict(info.shapes), dict(info.quant_specs))
                if expert_meta is None:
                    expert_meta = described
                elif expert_meta != described:
                    raise ValueError(
                        f"{expert_label} layer {idx} experts differ from the first MoE "
                        f"layer's ({described[:2]} vs {expert_meta[:2]}); one expert layout "
                        f"serves every layer"
                    )
                expert_layers_written.append(idx)
                total_params += info.total_params
            if root:
                # A copy of this layer from an earlier one-root layout is now stale. It is
                # Soup's own file inside the contained primary root, so it WILL go — but not
                # yet: the index on disk right now is still the OLD one, and until the new
                # index (which no longer names this path) is safely written, that old index
                # is the cache's commit point and every file it names must exist at every
                # instant. Deleting inline here would leave a window — crash before
                # _atomic_write_index and the still-current index describes a file that is
                # gone. Collected now, deleted only after the index write below succeeds.
                stale = layer_shard_path(resolved_out, idx)
                if os.path.exists(stale):
                    stale_root0_copies.append(stale)
            # An experts file the new index will not name (an earlier per_expert
            # layout, or one now written to a stripe root) goes the same way.
            stale_experts = os.path.join(resolved_out, expert_shard_name(idx))
            if (not per_expert_ids or root) and os.path.exists(stale_experts):
                stale_root0_copies.append(stale_experts)
            del blob

        extras = {}
        large_keys = []
        large_roles: Dict[str, str] = {}
        for key in where:
            if _LAYER_RE.match(key) or key in consumed:
                continue
            if saw_oq_companion and key.endswith((".scales", ".biases")):
                continue
            if saw_oq_companion and key.endswith(".weight"):
                tensor = _read_oq_tensor(
                    handles, where, key, oq_config=oq_config, dtype=dtype
                )
            else:
                tensor = _read_source_tensor(handles, where, key, dtype, fp8_plan)
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

    primary = primary_cache_identity(resolved_out)
    for position, directory in enumerate(shard_dirs[1:], start=1):
        problem = stripe_folder_problem(directory, stripe[position - 1])
        if problem is not None:
            raise StripeRootError(
                f"{STRIPE_DIRS_ENV}: writing the stripe marker to {stripe[position - 1]} "
                f"refused: {problem}. The existing cache index is unchanged."
            )
        _write_stripe_marker(
            directory,
            fingerprint=fingerprint,
            position=position,
            n_roots=len(shard_dirs),
            primary=primary,
        )

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
        stripe_roots=stripe,
        layer_roots=layer_roots,
        expert_layout=expert_layout,
        expert_layers=tuple(expert_layers_written),
        expert_count=expert_meta[0] if expert_meta else 0,
        expert_shapes=expert_meta[1] if expert_meta else {},
        expert_quant_specs=expert_meta[2] if expert_meta else {},
    )
    _atomic_write_index(index, resolved_out)
    resharding_index = os.path.join(resolved_out, _RESHARDING_INDEX_NAME)
    try:
        os.remove(resharding_index)
    except FileNotFoundError:
        pass
    except OSError as exc:
        if notify is not None:
            from soup_cli.utils.terminal import for_terminal

            notify(
                f"[yellow]Could not delete the completed re-shard marker "
                f"{for_terminal(resharding_index)} ({for_terminal(exc)}); the new cache "
                f"index is committed — delete the marker when it is no longer in use.[/]"
            )
    # Only now: the new index above no longer names these paths, so removing them cannot
    # leave any on-disk index (old or new) describing a file that does not exist. Best-effort
    # from here on: the shard is already correct and fully committed, so a failed delete (a
    # leftover handle holding the file open, most plausibly on Windows) must warn and move on
    # rather than turn a genuinely successful shard into a caller-visible exception.
    for stale in stale_root0_copies:
        try:
            os.remove(stale)
        except FileNotFoundError:
            pass
        except OSError as exc:
            if notify is not None:
                from soup_cli.utils.terminal import for_terminal

                # This branch must not raise (the shard is committed): a `[/]` in the path or
                # the error text would otherwise be a MarkupError right here.
                notify(
                    f"[yellow]Could not delete the stale layer copy {for_terminal(stale)} "
                    f"({for_terminal(exc)}); the cache is complete without it — delete it to "
                    f"reclaim the space.[/]"
                )
    if notify is not None:
        _notify_unindexed_cache_files(resolved_out, index, notify)
    return index


def _folder_bytes(folder: str) -> int:
    total = 0
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
    except OSError:
        return 0
    return total


def shard_cache_disk_bytes(shard_dir: str, index: ShardIndex) -> Optional[int]:
    """Bytes of the files a reusable cache is made of, or ``None`` if one cannot be measured.

    Only what the cache check requires is counted: the index, the extras shard, each large
    shard and each layer file wherever it is placed. A stray or stale file in the folder is
    not part of the cache, and a file that cannot be read makes the caller keep its estimate.
    """
    paths = [os.path.join(shard_dir, _INDEX_NAME), extras_shard_path(shard_dir)]
    paths.extend(large_shard_path(shard_dir, key) for key in index.large_keys)
    paths.extend(layer_paths(shard_dir, index))
    paths.extend(expert_paths(shard_dir, index).values())
    try:
        return sum(os.path.getsize(path) for path in paths)
    except (OSError, ValueError):
        return None


def _notify_orphaned_stripes(
    out_dir: str, keep: Sequence[str], notify: Callable[[str], None]
) -> None:
    """Name stripe folders the previous index used and the new layout will not.

    Never deletes them: they sit outside the contained primary root, so the operator decides.
    """
    previous = _read_previous_shard_index(out_dir)
    if previous is None:
        return
    from soup_cli.utils.stripe_roots import stripe_folder_problem
    from soup_cli.utils.terminal import for_terminal

    for root in previous.stripe_roots:
        if any(_same_roots((root,), (kept,)) for kept in keep):
            continue
        folder = stripe_dirs(out_dir, (root,))[1]
        # A link or junction is not Soup's folder: never measured through, never named as one.
        if stripe_folder_problem(folder, root) is not None or not os.path.isdir(folder):
            continue
        notify(
            f"[yellow]The previous layer cache also used {for_terminal(folder)} "
            f"({_folder_bytes(folder) / 1e9:.2f} GB); this layout no longer does. Delete it "
            f"to reclaim the space.[/]"
        )


def _read_previous_shard_index(out_dir: str) -> Optional[ShardIndex]:
    """Read the current index, or the quarantined one during an interrupted re-shard."""
    for name in (_INDEX_NAME, _RESHARDING_INDEX_NAME):
        try:
            return read_shard_index(out_dir, index_name=name)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
    return None


def _notify_unindexed_cache_files(
    out_dir: str, index: ShardIndex, notify: Callable[[str], None]
) -> None:
    """Report layer/temp files that the committed index no longer names."""
    from soup_cli.utils.stripe_roots import stripe_folder_problem
    from soup_cli.utils.terminal import for_terminal

    expected = {
        os.path.normcase(os.path.realpath(path))
        for path in [*layer_paths(out_dir, index), *expert_paths(out_dir, index).values()]
    }
    directories = stripe_dirs(out_dir, index.stripe_roots)
    for position, directory in enumerate(directories):
        if position:
            problem = stripe_folder_problem(directory, index.stripe_roots[position - 1])
            if problem is not None:
                continue
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        for entry in entries:
            if not (
                (
                    entry.name.startswith(("layer_", "experts_"))
                    and entry.name.endswith(".safetensors")
                )
                or (entry.name.startswith(".soup.") and entry.name.endswith(".tmp"))
            ):
                continue
            try:
                if not entry.is_file(follow_symlinks=False):
                    continue
                path = os.path.normcase(os.path.realpath(entry.path))
                if path in expected:
                    continue
                size = entry.stat(follow_symlinks=False).st_size
            except OSError:
                continue
            notify(
                f"[yellow]Unindexed layer-cache file {for_terminal(entry.path)} "
                f"({size} bytes) is present; it was not named by the committed cache index.[/]"
            )


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


def _refuse_fp8(key: str, stored_dtype: str) -> None:
    """The backstop under every plain read: an FP8 tensor or block scale that reaches a
    cast was not dequantised by an accepted :class:`FP8BlockPlan`. A plain cast keeps
    the raw e4m3 values and drops the scales, so the run would train on wrong weights
    while the resident load (which dequantises) looks fine.

    So is any other tensor that is not plain floating point, or that a quantiser
    named (:func:`~soup_cli.utils.fp8_block.refuse_unsupported_tensor`): the header
    plan refuses those first, and this is what still stands if a path skips it."""
    if stored_dtype.upper().startswith("F8") or key.endswith("_scale_inv"):
        raise UnsupportedFP8Error(
            f"checkpoint tensor {key!r} is FP8 ({stored_dtype}) or an FP8 block scale "
            f"that would be cast without its scales. {SUPPORTED_FP8_FORMAT}"
        )
    refuse_unsupported_tensor(key, stored_dtype)


def _read_source_tensor(
    handles: Mapping[str, Any],
    where: Mapping[str, Tuple[str, str]],
    key: str,
    dtype: str,
    fp8_plan: Optional[FP8BlockPlan],
) -> "Any":
    """One OWNED source tensor at ``dtype``; an FP8 block weight is dequantised first.

    The FP8 weight and its ``weight_scale_inv`` are read raw (each owns its memory, so
    the scale's file may evict the weight's handle), multiplied in fp32 per block, one
    band of rows at a time, into an output preallocated at ``dtype``
    (:func:`~soup_cli.utils.fp8_block.dequantize_fp8_block`): the raw weight, the output
    and one band are live, not an fp32 copy of the whole tensor.
    """
    path, source_key = where[key]
    if fp8_plan is None or key not in fp8_plan.weights:
        return _read_tensor(handles[path], source_key, dtype)
    import torch

    scale_path, scale_source_key = where[fp8_plan.scale_key(key)]
    weight = _read_raw_tensor(handles[path], source_key)
    scale = _read_raw_tensor(handles[scale_path], scale_source_key)
    try:
        return dequantize_fp8_block(
            weight, scale, fp8_plan.block, name=key, dtype=getattr(torch, dtype)
        )
    finally:
        del weight, scale


def _read_tensor(handle: Any, key: str, dtype: str) -> "Any":
    """Materialise one OWNED tensor, size-capped, converted to the target dtype."""
    import torch

    sliced = handle.get_slice(key)
    _refuse_fp8(key, str(sliced.get_dtype()))
    shape = sliced.get_shape()
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
