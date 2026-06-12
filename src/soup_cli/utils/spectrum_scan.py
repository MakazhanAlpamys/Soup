"""Native Spectrum targeted-training scan (#266, v0.71.23).

Implements the Spectrum method (arXiv:2406.06623, *Spectrum: Targeted Training
on Signal to Noise Ratio*) as a **pure-numpy, weight-streaming** scan:

- :func:`compute_snr` is the core kernel — singular-value SNR with a
  Marchenko-Pastur noise threshold. It is **transpose-invariant** (the
  singular values of ``W`` and ``W.T`` are identical and the MP aspect ratio
  ``min(n,m)/max(n,m)`` is symmetric), so a GPT-2 ``Conv1D`` weight stored as
  ``[in, out]`` scores the same as a Linear stored ``[out, in]``. The
  "Conv1D-aware" requirement is therefore about *module naming*
  (:func:`classify_module` recognises ``c_attn``/``c_fc``/``c_proj``), not the
  math.
- :func:`iter_weight_matrices` streams ``.safetensors`` shards **one tensor at
  a time** via ``safetensors.safe_open`` — there is NO full model load, so
  peak RSS is the size of the largest single weight matrix. This is what lets
  a 70B's layer SNR be scanned on a CPU box; the reference implementation
  loads the whole model.
- Scan results are cached at ``~/.soup/spectrum/<slug>.json`` (override via
  ``SOUP_SPECTRUM_CACHE_DIR``, containment-checked under ``$HOME`` / ``$CWD`` /
  ``$TMPDIR`` per the v0.36.0 cache-dir policy).

All heavy deps (numpy / torch / safetensors / huggingface_hub) are imported
lazily inside functions so the module — and the ``soup`` CLI — load without
them. LISA (per-step layer sampling) is split to #267.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

if TYPE_CHECKING:  # static types only — numpy stays a lazy runtime import
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# DoS / sanity caps.
_MAX_TENSORS = 100_000
_MAX_UNFROZEN_PATTERNS = 50_000
_MAX_CACHE_BYTES = 64 * 1024 * 1024  # 64 MiB
_MAX_CACHE_LAYERS = _MAX_TENSORS
# A single 2-D weight above this element count is skipped (and logged) rather
# than fed to a multi-GB SVD — a backstop against a crafted/oversized
# ``--model``. Legitimate mlp/attn matrices (even a 70B's ~8k×29k) are far
# below this; only giant embeddings / adversarial shapes trip it.
_MAX_MATRIX_ELEMENTS = 2**31
_CACHE_SCHEMA = 1

_MLP_MARKERS = ("mlp", "feed_forward", "ffn")
_ATTN_MARKERS = ("self_attn", "attention")
_VALID_MODULE_TYPES = ("mlp", "attn", "other")

_LAYER_IDX_RE = re.compile(r"(?:^|\.)(?:layers|h)\.\d+\.")
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")

ModulesArg = Union[str, Sequence[str], None]


def _np():
    import numpy as np  # lazy — keep the CLI import light

    return np


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
ModuleType = Literal["attn", "mlp", "other"]


@dataclass(frozen=True)
class LayerSNR:
    """Per-weight-matrix SNR record."""

    name: str
    module_type: ModuleType
    group: str  # layer-type signature, e.g. "self_attn.q_proj"
    snr: float
    shape: Tuple[int, int]


@dataclass(frozen=True)
class ScanResult:
    """A full model scan: the model id, the requested modules, the layers."""

    model: str
    modules: str  # canonical "all" or comma-joined sorted types
    layers: Tuple[LayerSNR, ...]


# ---------------------------------------------------------------------------
# SNR kernel (pure numpy)
# ---------------------------------------------------------------------------
def estimate_sigma(singular_values: Any) -> float:
    """Estimate the noise std via the inter-quartile range of singular values.

    ``sigma = IQR / 1.349`` (the IQR-to-std factor for a normal) — the
    Spectrum reference estimator.
    """
    np = _np()
    s = np.asarray(singular_values, dtype=np.float64).ravel()
    if s.size == 0:
        return 0.0
    q75, q25 = np.percentile(s, [75, 25])
    return float((q75 - q25) / 1.349)


def marchenko_pastur_threshold(sigma: float, n: int, m: int) -> float:
    """Marchenko-Pastur singular-value noise edge.

    ``threshold = sigma * sqrt((1 + sqrt(beta))**2)`` with
    ``beta = min(n, m) / max(n, m)`` (the Spectrum reference form;
    ``sqrt((1+x)**2) == 1+x`` for ``x >= 0``). Symmetric in ``n``/``m``.
    """
    if n <= 0 or m <= 0:
        return 0.0
    beta = min(n, m) / max(n, m)
    return float(sigma) * math.sqrt((1.0 + math.sqrt(beta)) ** 2)


def compute_snr(matrix: Any) -> float:
    """Signal-to-noise ratio of a single 2-D weight matrix.

    Singular values above the MP threshold are "signal", the rest "noise";
    ``snr = signal_sum / noise_sum`` and the returned ratio is normalised by
    the largest singular value (the Spectrum ``snr_ratio``). Always finite.
    Transpose-invariant. Raises ``ValueError`` on non-2-D input.
    """
    np = _np()
    # Preserve float32/float64 (the streaming path yields float32 to keep peak
    # RSS ≈ the largest single matrix); upcast float16 / bf16-already-float /
    # int to float32 for a stable SVD.
    a = np.asarray(matrix)
    if a.dtype not in (np.float32, np.float64):
        a = a.astype(np.float32)
    if a.ndim != 2:
        raise ValueError(f"compute_snr expects a 2-D matrix, got ndim={a.ndim}")
    if a.size == 0:
        return 0.0
    s = np.linalg.svd(a, compute_uv=False)  # singular values, descending
    if s.size == 0:
        return 0.0
    max_sv = float(s[0])
    if max_sv <= 0.0 or not math.isfinite(max_sv):
        return 0.0
    sigma = estimate_sigma(s)
    threshold = marchenko_pastur_threshold(sigma, a.shape[0], a.shape[1])
    signal_mask = s > threshold
    signal = float(s[signal_mask].sum()) if signal_mask.any() else 0.0
    # An empty below-threshold sum is 0.0; the guard floors it to 1.0 so the
    # ratio stays finite (mirrors the Spectrum reference's noise default).
    noise = float(s[~signal_mask].sum())
    if noise <= 0.0:
        noise = 1.0
    ratio = (signal / noise) / max_sv
    return float(ratio) if math.isfinite(ratio) else 0.0


# ---------------------------------------------------------------------------
# Module classification + grouping
# ---------------------------------------------------------------------------
def classify_module(name: Any) -> Optional[ModuleType]:
    """Classify a parameter name as ``"attn"`` / ``"mlp"`` / ``"other"``.

    Returns ``None`` for non-``.weight`` params (biases, etc.). Recognises
    both Llama-style (``self_attn.q_proj``, ``mlp.gate_proj``) and GPT-2
    ``Conv1D`` (``attn.c_attn``, ``mlp.c_fc``, ``mlp.c_proj``) naming.
    """
    if not isinstance(name, str) or not name.endswith(".weight"):
        return None
    low = name.lower()
    if any(mk in low for mk in _MLP_MARKERS):
        return "mlp"
    if any(mk in low for mk in _ATTN_MARKERS) or ".attn." in low:
        return "attn"
    return "other"


def layer_type_signature(name: str) -> str:
    """Strip the numeric layer index so same-type layers group together.

    ``model.layers.5.self_attn.q_proj.weight`` -> ``self_attn.q_proj``;
    ``transformer.h.3.mlp.c_fc.weight`` -> ``mlp.c_fc``;
    ``lm_head.weight`` -> ``lm_head`` (non-layer params group by themselves).
    """
    match = _LAYER_IDX_RE.search(name)
    base = name[match.end():] if match else name
    for suffix in (".weight", ".bias"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def param_prefix(name: str) -> str:
    """Strip a trailing ``.weight`` — the unfreeze pattern is the name prefix."""
    return name[: -len(".weight")] if name.endswith(".weight") else name


# ---------------------------------------------------------------------------
# Module-filter normalisation
# ---------------------------------------------------------------------------
def _normalize_modules(modules: ModulesArg) -> Optional[frozenset[str]]:
    """Return a frozenset of kept types, or ``None`` meaning "keep all"."""
    if modules is None:
        return None
    if isinstance(modules, str):
        parts = [p.strip().lower() for p in modules.split(",")]
    else:
        parts = [str(p).strip().lower() for p in modules]
    parts = [p for p in parts if p]
    if not parts or "all" in parts:
        return None
    for part in parts:
        if part not in _VALID_MODULE_TYPES:
            raise ValueError(
                f"unknown module type {part!r}; choose from "
                f"{', '.join(_VALID_MODULE_TYPES)}, all"
            )
    return frozenset(parts)


def _canonical_modules_str(modules: ModulesArg) -> str:
    keep = _normalize_modules(modules)
    return "all" if keep is None else ",".join(sorted(keep))


# ---------------------------------------------------------------------------
# Safetensors streaming
# ---------------------------------------------------------------------------
def _framework() -> Tuple[str, Callable[[Any], "NDArray[Any]"]]:
    """Pick a safetensors framework + a to-numpy converter.

    Prefer torch (handles bf16, which numpy cannot represent); fall back to
    the pure-numpy framework when torch is absent (bf16 weights then need
    torch — :func:`iter_weight_matrices` raises a friendly error).
    """
    np = _np()
    try:
        import torch
    except Exception:  # pragma: no cover - torch present in dev/CI
        torch = None
    if torch is not None:
        def to_np(tensor: Any) -> "NDArray[Any]":
            return tensor.detach().to(torch.float32).cpu().numpy()

        return "pt", to_np

    def to_np(arr: Any) -> "NDArray[Any]":
        # Keep the native dtype (compute_snr upcasts float16 → float32); a
        # forced float64 would double peak RSS over the streamed matrix.
        return np.asarray(arr)

    return "np", to_np


def _discover_safetensors(weights_dir: str) -> list[str]:
    real = os.path.realpath(weights_dir)
    if not os.path.isdir(real):
        raise FileNotFoundError(f"weights dir not found: {weights_dir}")
    files = []
    for entry in sorted(os.listdir(real)):
        if not entry.endswith(".safetensors"):
            continue
        full = os.path.join(real, entry)
        # Skip symlinked shards — defence-in-depth against a link redirecting
        # the read (mirrors the project's no-symlink TOCTOU policy).
        if os.path.islink(full):
            logger.warning("spectrum scan: skipping symlinked shard %s", entry)
            continue
        files.append(full)
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {weights_dir}")
    return files


def iter_weight_matrices(
    weights_dir: str, *, modules: ModulesArg = "all"
) -> Iterator[Tuple[str, "NDArray[Any]"]]:
    """Stream 2-D float weight matrices from a model dir, one at a time.

    Yields ``(param_name, np.ndarray)`` for every 2-D ``.weight`` tensor whose
    module type passes the ``modules`` filter. The shape is probed via
    ``get_slice(...).get_shape()`` so 1-D norms, filtered-out modules and
    oversized matrices are NEVER materialised — peak RSS is the largest *kept*
    tensor.
    """
    from safetensors import safe_open

    keep = _normalize_modules(modules)
    framework, to_np = _framework()
    count = 0
    for path in _discover_safetensors(weights_dir):
        with safe_open(path, framework=framework) as handle:
            for key in handle.keys():
                module_type = classify_module(key)
                if module_type is None:
                    continue
                if keep is not None and module_type not in keep:
                    continue
                shape = tuple(handle.get_slice(key).get_shape())
                if len(shape) != 2:
                    continue
                if shape[0] * shape[1] > _MAX_MATRIX_ELEMENTS:
                    logger.warning(
                        "spectrum scan: skipping %s — %dx%d exceeds the "
                        "%d-element SVD cap (scan a subset via --modules "
                        "mlp,attn for very large models)",
                        key, shape[0], shape[1], _MAX_MATRIX_ELEMENTS,
                    )
                    continue
                count += 1
                if count > _MAX_TENSORS:
                    raise ValueError(
                        f"scan exceeded the {_MAX_TENSORS}-tensor cap"
                    )
                try:
                    array = to_np(handle.get_tensor(key))
                except Exception as exc:
                    raise RuntimeError(
                        f"could not read tensor {key!r} as float "
                        f"(bf16 weights need torch — pip install "
                        f"'soup-cli[train]'): {type(exc).__name__}"
                    ) from exc
                yield key, array


def scan_weights_dir(
    weights_dir: str, *, modules: ModulesArg = "all"
) -> Tuple[LayerSNR, ...]:
    """Stream a local model dir and compute the SNR of every kept matrix."""
    out = []
    for name, array in iter_weight_matrices(weights_dir, modules=modules):
        out.append(
            LayerSNR(
                name=name,
                module_type=classify_module(name) or "other",
                group=layer_type_signature(name),
                snr=compute_snr(array),
                shape=(int(array.shape[0]), int(array.shape[1])),
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def select_unfrozen_parameters(
    results: Sequence[LayerSNR],
    *,
    top_percent: float,
    modules: ModulesArg = "all",
) -> list[str]:
    """Pick the top ``top_percent`` % of layers *per module type group*.

    Grouping by :func:`layer_type_signature` keeps the unfreeze balanced
    across module types (top-50% of ``q_proj`` layers, top-50% of
    ``down_proj`` layers, ...). At least one layer is kept per group
    (``ceil``). Returns a sorted, de-duplicated list of parameter-name
    prefixes (``.weight`` stripped) ready for ``training.unfrozen_parameters``.
    """
    if isinstance(top_percent, bool) or not isinstance(top_percent, (int, float)):
        raise ValueError("top_percent must be a number in (0, 100]")
    if not (0 < top_percent <= 100):
        raise ValueError("top_percent must be in (0, 100]")

    keep_types = _normalize_modules(modules)
    groups: dict[str, list[LayerSNR]] = {}
    for record in results:
        if keep_types is not None and record.module_type not in keep_types:
            continue
        groups.setdefault(record.group, []).append(record)

    selected: list[str] = []
    for items in groups.values():
        ordered = sorted(items, key=lambda r: (r.snr, r.name), reverse=True)
        keep_n = max(1, math.ceil(len(ordered) * top_percent / 100.0))
        selected.extend(param_prefix(r.name) for r in ordered[:keep_n])

    if len(selected) > _MAX_UNFROZEN_PATTERNS:
        raise ValueError(
            f"selection exceeded the {_MAX_UNFROZEN_PATTERNS}-pattern cap"
        )
    return sorted(set(selected))


# ---------------------------------------------------------------------------
# Cache + slug
# ---------------------------------------------------------------------------
def model_slug(model: str) -> str:
    """Sanitise a model id into a traversal-safe cache filename stem."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    base = model.strip().replace("\\", "/").rstrip("/").replace("/", "__")
    slug = _SLUG_RE.sub("_", base).replace("..", "_").strip("._-")
    return (slug or "model")[:128]


def default_spectrum_cache_dir() -> str:
    """``~/.soup/spectrum`` — the scan cache root (no side effects)."""
    return os.path.join(os.path.expanduser("~"), ".soup", "spectrum")


def resolve_cache_dir(cache_dir: Optional[str] = None) -> str:
    """Resolve the cache dir (explicit arg > ``SOUP_SPECTRUM_CACHE_DIR`` > default).

    The env override is rejected if it contains C0 control characters or
    escapes ``$HOME`` / ``$CWD`` / ``$TMPDIR`` (silent fall-through to the
    default — an env var is operator config, not API input). The chosen dir
    is created.
    """
    from soup_cli.utils.paths import is_under

    if cache_dir is not None:
        chosen = os.path.realpath(os.path.expanduser(str(cache_dir)))
        os.makedirs(chosen, exist_ok=True)
        return chosen

    override = os.environ.get("SOUP_SPECTRUM_CACHE_DIR")
    if override and not any(ord(ch) < 0x20 for ch in override):
        candidate = os.path.realpath(os.path.expanduser(override))
        bounds = [
            os.path.realpath(os.path.expanduser("~")),
            os.path.realpath(os.getcwd()),
            os.path.realpath(tempfile.gettempdir()),
        ]
        if any(is_under(candidate, bound) for bound in bounds):
            os.makedirs(candidate, exist_ok=True)
            return candidate

    default = default_spectrum_cache_dir()
    os.makedirs(default, exist_ok=True)
    return default


def _cache_path(model: str, cache_dir: Optional[str]) -> str:
    return os.path.join(resolve_cache_dir(cache_dir), model_slug(model) + ".json")


def _atomic_write_json(payload: dict, path: str) -> str:
    """Atomic JSON write into an already-validated cache dir.

    The cache lives under ``$HOME``/``$TMPDIR`` (not cwd), so it deliberately
    does NOT use ``paths.atomic_write_text`` (which enforces cwd containment);
    the dir is bounded by :func:`resolve_cache_dir` and the filename by
    :func:`model_slug`. mkstemp + ``os.replace`` keeps the write atomic.
    """
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".soup.", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
    return os.path.realpath(path)


def write_cached_scan(result: ScanResult, *, cache_dir: Optional[str] = None) -> str:
    """Persist a :class:`ScanResult` to the cache. Returns the written path."""
    payload = {
        "schema": _CACHE_SCHEMA,
        "model": result.model,
        "modules": result.modules,
        "layers": [
            {
                "name": ls.name,
                "module_type": ls.module_type,
                "group": ls.group,
                "snr": ls.snr,
                "shape": list(ls.shape),
            }
            for ls in result.layers
        ],
    }
    return _atomic_write_json(payload, _cache_path(result.model, cache_dir))


def read_cached_scan(
    model: str, *, modules: ModulesArg = "all", cache_dir: Optional[str] = None
) -> Optional[ScanResult]:
    """Load a cached scan, or ``None`` on miss / corruption / module mismatch."""
    path = _cache_path(model, cache_dir)
    if not os.path.isfile(path):
        return None
    try:
        if os.path.getsize(path) > _MAX_CACHE_BYTES:
            return None
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("schema") != _CACHE_SCHEMA:
        return None
    if payload.get("modules") != _canonical_modules_str(modules):
        return None
    raw_layers = payload.get("layers")
    if not isinstance(raw_layers, list) or len(raw_layers) > _MAX_CACHE_LAYERS:
        return None
    try:
        layers = tuple(
            LayerSNR(
                name=str(d["name"]),
                module_type=str(d["module_type"]),
                group=str(d["group"]),
                snr=float(d["snr"]),
                shape=(int(d["shape"][0]), int(d["shape"][1])),
            )
            for d in payload["layers"]
        )
        return ScanResult(
            model=str(payload["model"]),
            modules=str(payload["modules"]),
            layers=layers,
        )
    except (KeyError, TypeError, ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Resolve + scan orchestration
# ---------------------------------------------------------------------------
def resolve_model_weights(model: str) -> str:
    """Return a local dir of ``.safetensors`` for ``model``.

    A local directory is used as-is; otherwise ``model`` is treated as an HF
    Hub id and only its weights + config are downloaded (no model load). The
    download routes through the SSRF-hardened, namespace-pinned
    :func:`soup_cli.utils.hubs.snapshot_download` (repo-id shape validation +
    the #186 anti-AI-Jacking TOFU gate), into a contained cache dir — same
    policy as ``sae_diff.download_sae``.
    """
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    if os.path.isdir(model):
        return os.path.realpath(model)
    from soup_cli.utils.hubs import snapshot_download

    cache_dir = os.path.join(
        default_spectrum_cache_dir(), "weights", model_slug(model)
    )
    return snapshot_download(
        model,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json"],
    )


def scan_model(
    model: str,
    *,
    modules: ModulesArg = "all",
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
) -> ScanResult:
    """Resolve, (cache-)scan a model and return its :class:`ScanResult`."""
    canonical = _canonical_modules_str(modules)
    if use_cache:
        cached = read_cached_scan(model, modules=canonical, cache_dir=cache_dir)
        if cached is not None:
            return cached
    weights_dir = resolve_model_weights(model)
    layers = scan_weights_dir(weights_dir, modules=modules)
    result = ScanResult(model=model, modules=canonical, layers=layers)
    if use_cache:
        try:
            write_cached_scan(result, cache_dir=cache_dir)
        except OSError:
            pass  # cache write failures are non-fatal
    return result
