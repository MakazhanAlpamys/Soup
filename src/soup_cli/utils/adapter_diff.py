"""Per-layer LoRA adapter diff math + report rendering (v0.57.0 Part A).

Pure numpy math (no torch); safetensors is loaded lazily so import is cheap.
A LoRA module is compared on its effective delta ``ΔW = scaling · B @ A``,
never factor by factor: ``(B·M, M⁻¹·A)`` is the same ``ΔW`` for any invertible
``M``, so the raw factors measure the parameterisation, not the change.
Public surface:

- ``compute_layer_diffs(weights_a, weights_b)`` -> per-layer Frobenius diffs
- ``effective_rank(matrix)`` -> SVD-entropy effective rank
- ``compute_adapter_diff(path_a, path_b, *, top_k=10)`` -> ``AdapterDiffReport``
- ``render_report_markdown(report)`` / ``render_report_json(report)``

Containment + symlink rejection at every file load (TOCTOU defence,
mirrors v0.53.1 ``enforce_under_cwd_and_no_symlink`` policy).
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink, is_under_cwd

_MAX_LAYER_NAME_LEN = 256
_MAX_LAYERS = 10_000
_MAX_TOP_K = 200
_MIN_TOP_K = 1


@dataclass(frozen=True)
class LayerDiff:
    """Frobenius-norm diff of one LoRA module's ``ΔW``, or of one other tensor."""

    name: str
    frobenius: float
    norm_a: float
    norm_b: float
    relative: float  # frobenius / max(norm_a, norm_b) or 0.0 if both zero


@dataclass(frozen=True)
class AdapterDiffReport:
    adapter_a: str
    adapter_b: str
    per_layer: Tuple[LayerDiff, ...]
    top_changed: Tuple[str, ...]
    effective_rank_a: Optional[float]
    effective_rank_b: Optional[float]
    shared_layers: int
    only_in_a: Tuple[str, ...]
    only_in_b: Tuple[str, ...]


def _require_str(value: object, field: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str):
        raise TypeError(f"{field} must be str, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{field} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain null bytes")
    if len(value) > _MAX_LAYER_NAME_LEN:
        raise ValueError(f"{field} must be ≤{_MAX_LAYER_NAME_LEN} chars")
    return value


def _frobenius(matrix: Any) -> float:
    import numpy as np

    arr = np.asarray(matrix, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    value = float(np.sqrt(np.sum(arr * arr)))
    if not math.isfinite(value):
        return float("inf")
    return value


def effective_rank(matrix: Any, *, eps: float = 1e-12) -> float:
    """Shannon entropy of normalised singular-value distribution (effective rank).

    Returns ``exp(H)`` where H is the entropy of the SV distribution
    treated as a probability vector. Equals the matrix rank for an
    orthonormal basis and degrades smoothly as energy concentrates.
    """
    import numpy as np

    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise TypeError("eps must be float")
    if not math.isfinite(float(eps)) or float(eps) <= 0:
        raise ValueError("eps must be finite and positive")

    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim < 2:
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        else:
            return 0.0
    if arr.size == 0:
        return 0.0
    # Reshape >2D into 2D for SVD
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    try:
        singular = np.linalg.svd(arr, compute_uv=False)
    except np.linalg.LinAlgError:
        return 0.0
    return _effective_rank_of_singular_values(singular, eps=float(eps))


def _effective_rank_of_singular_values(singular: Any, *, eps: float = 1e-12) -> float:
    """``exp`` of the entropy of a singular-value spectrum (see :func:`effective_rank`)."""
    import numpy as np

    total = float(np.sum(singular))
    if total <= eps:
        return 0.0
    probs = singular / total
    probs = probs[probs > eps]
    if probs.size == 0:
        return 0.0
    entropy = float(-np.sum(probs * np.log(probs)))
    return float(math.exp(entropy))


def _lora_modules(weights: Mapping[str, Any]) -> dict[str, Tuple[str, str]]:
    """Complete LoRA pairs of an adapter: module stem -> (lora_A name, lora_B name)."""
    from soup_cli.utils.adapter_arithmetic import _lora_role

    roles: dict[str, dict[str, str]] = {}
    for name in weights:
        role = _lora_role(name)
        if role is not None:
            roles.setdefault(role[0], {})[role[1]] = name
    return {
        stem: (found["A"], found["B"])
        for stem, found in roles.items()
        if "A" in found and "B" in found
    }


def _module_scaling(config: Optional[Mapping[str, Any]], stem: str) -> float:
    """PEFT's decode-time scaling for a module; 1.0 when the config cannot say."""
    if config is None:
        return 1.0
    scaling = lora_module_scaling(config, peft_module_name(stem))
    return 1.0 if scaling is None else scaling


def _delta_factors(
    weights: Mapping[str, Any], pair: Tuple[str, str], scaling: float,
) -> Optional[Tuple[Any, Any]]:
    """``(left, right)`` with ``left @ right == scaling * B @ A`` as 2-D float64.

    ``None`` when the pair cannot form a product (unexpected shapes) or holds a
    non-finite value; such a module is diffed tensor by tensor instead.
    """
    import numpy as np

    lora_a = np.asarray(weights[pair[0]], dtype=np.float64)
    lora_b = np.asarray(weights[pair[1]], dtype=np.float64)
    if lora_a.ndim < 2 or lora_b.ndim < 2 or lora_a.size == 0 or lora_b.size == 0:
        return None
    # Linear: A (r, in), B (out, r). Conv: A (r, in, kh, kw), B (out, r, 1, 1).
    right = lora_a.reshape(lora_a.shape[0], -1)
    left = lora_b.reshape(lora_b.shape[0], -1)
    if left.shape[1] != right.shape[0]:
        return None
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        return None
    return scaling * left, right


def _product_singular_values(left: Any, right: Any) -> Any:
    """Singular values of ``left @ right`` without forming the (out, in) product.

    Their squares are the eigenvalues of ``(left^T left)(right right^T)``, a
    rank-sized product that ``S (left^T left) S`` symmetrises, with ``S`` the
    square root of ``right right^T``.
    """
    import numpy as np

    eig_right, vec_right = np.linalg.eigh(right @ right.T)
    root = (vec_right * np.sqrt(np.clip(eig_right, 0.0, None))) @ vec_right.T
    core = root @ (left.T @ left) @ root
    eig = np.linalg.eigvalsh((core + core.T) / 2.0)
    return np.sqrt(np.clip(eig, 0.0, None))[::-1]


def _product_inner(left_a: Any, right_a: Any, left_b: Any, right_b: Any) -> float:
    """``<left_a @ right_a, left_b @ right_b>_F`` from rank-sized Gram blocks.

    ``trace(right_a^T left_a^T left_b right_b) = sum((left_a^T left_b) *
    (right_a right_b^T))``: two (r_a, r_b) matrices, never the (out, in) products.
    """
    import numpy as np

    return float(np.sum((left_a.T @ left_b) * (right_a @ right_b.T)))


def _module_diff(
    stem: str, factors_a: Tuple[Any, Any], factors_b: Tuple[Any, Any],
) -> Optional[LayerDiff]:
    """``||ΔW_a - ΔW_b||_F`` for one LoRA module, or ``None`` if it cannot be computed."""
    import numpy as np

    (left_a, right_a), (left_b, right_b) = factors_a, factors_b
    if left_a.shape[0] != left_b.shape[0] or right_a.shape[1] != right_b.shape[1]:
        return None  # different in/out features: not the same layer
    sq_a = max(0.0, _product_inner(left_a, right_a, left_a, right_a))
    sq_b = max(0.0, _product_inner(left_b, right_b, left_b, right_b))
    norm_a, norm_b = math.sqrt(sq_a), math.sqrt(sq_b)
    if np.array_equal(left_a, left_b) and np.array_equal(right_a, right_b):
        fro = 0.0
    else:
        # ||ΔW_a - ΔW_b||^2 = ||ΔW_a||^2 + ||ΔW_b||^2 - 2 <ΔW_a, ΔW_b>; clamp the
        # rounding of two nearly equal deltas at 0.
        cross = _product_inner(left_a, right_a, left_b, right_b)
        fro = math.sqrt(max(0.0, sq_a + sq_b - 2.0 * cross))
    if not all(math.isfinite(v) for v in (norm_a, norm_b, fro)):
        return None
    denom = max(norm_a, norm_b)
    return LayerDiff(
        name=stem,
        frobenius=fro,
        norm_a=norm_a,
        norm_b=norm_b,
        relative=fro / denom if denom > 0 else 0.0,
    )


def compute_layer_diffs(
    weights_a: Mapping[str, Any],
    weights_b: Mapping[str, Any],
    *,
    config_a: Optional[Mapping[str, Any]] = None,
    config_b: Optional[Mapping[str, Any]] = None,
) -> Tuple[Tuple[LayerDiff, ...], Tuple[str, ...], Tuple[str, ...]]:
    """Compute per-layer Frobenius diffs for every layer in both adapters.

    A LoRA module whose ``lora_A`` / ``lora_B`` pair is present in both adapters
    is compared on its effective delta ``ΔW = scaling · B @ A`` and reported
    once, under its module name, whatever the two ranks are. ``scaling`` comes
    from each adapter's ``config`` (``lora_alpha / r``, or ``/ sqrt(r)`` with
    ``use_rslora``, per-module patterns applied; 1.0 without one). Every other
    tensor (``modules_to_save`` weights, biases, a half-present pair) is diffed
    element-wise by name, skipping shape mismatches.

    Returns ``(per_layer, only_in_a, only_in_b)``. ``per_layer`` is sorted by name.
    """
    import numpy as np

    if not isinstance(weights_a, Mapping):
        raise TypeError("weights_a must be a mapping")
    if not isinstance(weights_b, Mapping):
        raise TypeError("weights_b must be a mapping")

    names_a = set(weights_a.keys())
    names_b = set(weights_b.keys())
    if len(names_a) > _MAX_LAYERS or len(names_b) > _MAX_LAYERS:
        raise ValueError(f"adapter has >{_MAX_LAYERS} tensors")

    diffs = []
    pairs_a = _lora_modules(weights_a)
    pairs_b = _lora_modules(weights_b)
    for stem in sorted(set(pairs_a) & set(pairs_b)):
        _require_str(stem, "layer name")
        factors_a = _delta_factors(weights_a, pairs_a[stem], _module_scaling(config_a, stem))
        factors_b = _delta_factors(weights_b, pairs_b[stem], _module_scaling(config_b, stem))
        if factors_a is None or factors_b is None:
            continue
        module = _module_diff(stem, factors_a, factors_b)
        if module is None:
            continue
        diffs.append(module)
        names_a.difference_update(pairs_a[stem])
        names_b.difference_update(pairs_b[stem])

    shared = sorted(names_a & names_b)
    only_a = tuple(sorted(names_a - names_b))
    only_b = tuple(sorted(names_b - names_a))

    for name in shared:
        _require_str(name, "layer name")
        a = np.asarray(weights_a[name], dtype=np.float64)
        b = np.asarray(weights_b[name], dtype=np.float64)
        if a.shape != b.shape:
            # Skip shape-mismatched tensors (rank changed between adapters)
            continue
        diff = a - b
        fro = _frobenius(diff)
        norm_a = _frobenius(a)
        norm_b = _frobenius(b)
        denom = max(norm_a, norm_b)
        relative = fro / denom if denom > 0 else 0.0
        diffs.append(
            LayerDiff(
                name=name,
                frobenius=fro,
                norm_a=norm_a,
                norm_b=norm_b,
                relative=relative,
            )
        )
    return tuple(sorted(diffs, key=lambda d: d.name)), only_a, only_b


def _validate_top_k(top_k: object) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be int")
    if top_k < _MIN_TOP_K or top_k > _MAX_TOP_K:
        raise ValueError(f"top_k must be in [{_MIN_TOP_K}, {_MAX_TOP_K}]")
    return top_k


_PEFT_KEY_PREFIX = "base_model.model."


def peft_module_name(stem: str) -> str:
    """The module name PEFT matches ``rank_pattern`` / ``alpha_pattern`` keys against.

    ``stem`` is a saved LoRA tensor name without its ``.lora_A.weight`` /
    ``.lora_B.weight`` suffix; PEFT saves those under a ``base_model.model.`` prefix.
    """
    return stem[len(_PEFT_KEY_PREFIX):] if stem.startswith(_PEFT_KEY_PREFIX) else stem


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _pattern_value(patterns: Any, module: str, default: Any) -> Any:
    """The ``rank_pattern`` / ``alpha_pattern`` value PEFT applies to ``module``.

    Mirrors ``peft.utils.other.get_pattern_key``: the first key ``k`` for which
    ``(.*\\.)?(k)$`` matches the module name wins, and no match keeps ``default``.
    """
    if not isinstance(patterns, Mapping):
        return default
    for key, value in patterns.items():
        if not isinstance(key, str):
            continue
        try:
            if re.match(rf"(.*\.)?({key})$", module):
                return value
        except re.error:
            continue
    return default


def lora_module_scaling(config: Mapping[str, Any], module: str) -> Optional[float]:
    """The decode-time scaling PEFT gives ``module`` under an ``adapter_config.json``.

    ``lora_alpha / r``, or ``lora_alpha / sqrt(r)`` with ``use_rslora``, after the
    per-module ``rank_pattern`` / ``alpha_pattern`` overrides. ``module`` is the
    module name PEFT matches those patterns against (``model.layers.0.self_attn.q_proj``).
    Returns ``None`` when the config carries no usable ``r`` / ``lora_alpha``.
    """
    rank = _finite_number(_pattern_value(config.get("rank_pattern"), module, config.get("r")))
    alpha = _finite_number(
        _pattern_value(config.get("alpha_pattern"), module, config.get("lora_alpha"))
    )
    if rank is None or alpha is None or rank <= 0:
        return None
    if config.get("use_rslora"):
        return alpha / math.sqrt(rank)
    return alpha / rank


def read_adapter_config_or_none(adapter_dir: str) -> Optional[dict]:
    """``adapter_config.json`` as a dict, or ``None`` when it is absent or unusable.

    Never refuses: a caller without a usable config keeps its previous behaviour.
    """
    from soup_cli.utils.adapter_arithmetic import _read_adapter_config_dict

    try:
        return _read_adapter_config_dict(adapter_dir)
    except (OSError, TypeError, ValueError):
        return None


READ_ADAPTER_DEPS: Tuple[str, ...] = ("numpy",)
WRITE_ADAPTER_DEPS: Tuple[str, ...] = ("numpy", "safetensors")

READ_ADAPTER_DEPS_MSG = (
    "numpy package required; "
    'pip install "soup-cli[train]" (or pip install numpy)'
)
WRITE_ADAPTER_DEPS_MSG = (
    "numpy and safetensors packages required; "
    'pip install "soup-cli[train]" (or pip install safetensors numpy)'
)


def require_adapter_dependencies(
    deps: Tuple[str, ...] = WRITE_ADAPTER_DEPS,
) -> None:
    """Validate that dependencies needed for adapter operations can be imported."""
    import importlib

    msg = (
        READ_ADAPTER_DEPS_MSG
        if deps == READ_ADAPTER_DEPS
        else WRITE_ADAPTER_DEPS_MSG
    )
    for dep in deps:
        try:
            importlib.import_module(dep)
        except ImportError as exc:
            raise RuntimeError(msg) from exc


#: Safetensors dtype tag -> little-endian numpy dtype, for everything except BF16.
_NUMPY_DTYPES = {
    "float16": "<f2",
    "float32": "<f4",
    "float64": "<f8",
    "int8": "i1",
    "int16": "<i2",
    "int32": "<i4",
    "int64": "<i8",
    "uint8": "u1",
    "bool": "?",
}


def _read_safetensors_numpy(path: Path) -> dict[str, Any]:
    """Read every tensor of a safetensors file as a numpy array, BF16 included.

    ``safe_open(framework="numpy")`` cannot return bfloat16 (numpy has no such dtype), so
    an adapter trained in bf16, which is the default dtype of a ``torch_dtype: bfloat16``
    base, failed with ``data type 'bfloat16' not understood`` (#2058). The header is parsed
    by ``read_header`` (no mapping, offsets and sizes validated) and the payload read
    directly; a bfloat16 tensor is widened to float32 exactly, since it is the top 16 bits
    of a float32.

    The arrays are read-only views of the bytes read (``np.frombuffer``); copy one before
    changing it in place.

    Raises ``ValueError`` for more than ``_MAX_LAYERS`` tensors (before any payload is
    read), for a header ``read_header`` refuses, and for a dtype numpy cannot hold (the
    FP8 spellings ``read_header`` accepts), each naming the file once.
    """
    require_adapter_dependencies(READ_ADAPTER_DEPS)
    import numpy as np

    from soup_cli.utils.safetensors_reader import read_header

    try:
        entries = read_header(str(path))
    except (ValueError, OSError) as exc:
        # read_header leads its messages with the path; the file is named once, here.
        reason = str(exc).removeprefix(f"{path}: ").replace(str(path), path.name)
        lead = "Unsupported" if "unsupported dtype" in reason else "Corrupt or unreadable"
        raise ValueError(f"{lead} safetensors weights in {path.name}: {reason}") from exc
    if len(entries) > _MAX_LAYERS:
        raise ValueError(f"adapter has >{_MAX_LAYERS} tensors")
    result: dict[str, Any] = {}
    with open(str(path), "rb") as handle:
        for name, entry in entries.items():
            _require_str(name, "tensor name")
            numpy_dtype = _NUMPY_DTYPES.get(entry.dtype)
            if numpy_dtype is None and entry.dtype != "bfloat16":
                raise ValueError(
                    f"Unsupported safetensors weights in {path.name}: tensor {name!r} "
                    f"has unsupported dtype {entry.dtype!r}; numpy has no such dtype"
                )
            handle.seek(entry.start)
            raw = handle.read(entry.nbytes)
            if len(raw) != entry.nbytes:
                raise ValueError(
                    f"Corrupt or unreadable safetensors weights in {path.name}: "
                    f"tensor {name!r} is truncated"
                )
            if entry.dtype == "bfloat16":
                bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
                array = bits.view(np.float32)
            else:
                array = np.frombuffer(raw, dtype=numpy_dtype)
            result[name] = array.reshape(entry.shape)
    return result


def _load_safetensors(path: Path) -> Mapping[str, Any]:
    """Load adapter_model.safetensors as numpy arrays (bfloat16 widened to float32)."""
    return _read_safetensors_numpy(path)


def _adapter_weights_path(adapter_dir: Path) -> Path:
    """Return the safetensors path inside an adapter dir, raising if missing.

    Symlinks at the weights file are rejected via ``os.lstat + S_ISLNK``
    BEFORE ``is_file()`` so a crafted ``adapter_model.safetensors -> /etc/passwd``
    cannot escape the directory-level containment check (review fix HIGH).
    """
    candidates = (
        adapter_dir / "adapter_model.safetensors",
        adapter_dir / "adapter_model.bin",
    )
    for cand in candidates:
        if not os.path.lexists(str(cand)):
            continue
        st = os.lstat(str(cand))
        if stat.S_ISLNK(st.st_mode):
            raise ValueError(
                f"{adapter_dir.name}/{cand.name}: must not be a symlink"
            )
        if cand.is_file():
            if cand.suffix == ".bin":
                raise RuntimeError(
                    f"{adapter_dir.name}: .bin format not supported; "
                    "re-save adapter as safetensors"
                )
            return cand
    raise FileNotFoundError(
        f"{adapter_dir.name}: no adapter_model.safetensors found"
    )


def load_adapter_weights(adapter_dir: str) -> Mapping[str, Any]:
    """Containment-checked safetensors load.

    Raises ``ValueError`` if the dir is outside cwd or a symlink; raises
    ``FileNotFoundError`` if no adapter_model.safetensors is present.
    """
    enforce_under_cwd_and_no_symlink(adapter_dir, "adapter")
    path = _adapter_weights_path(Path(adapter_dir))
    # Re-validate the weights file itself
    if not is_under_cwd(str(path)):
        raise ValueError(f"adapter weights must stay under cwd: {path.name}")
    return _load_safetensors(path)


def _effective_rank_average(
    weights: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None,
) -> Optional[float]:
    """Mean effective rank over LoRA modules' ``ΔW`` and other 2-D tensors.

    A complete LoRA pair counts once, by the spectrum of ``scaling · B @ A``; any
    other 2-D tensor counts by its own spectrum. ``None`` if there is neither.
    """
    import numpy as np

    ranks: list[float] = []
    covered: set[str] = set()
    for stem, pair in _lora_modules(weights).items():
        factors = _delta_factors(weights, pair, _module_scaling(config, stem))
        if factors is None:
            continue
        try:
            singular = _product_singular_values(*factors)
        except np.linalg.LinAlgError:
            continue
        ranks.append(_effective_rank_of_singular_values(singular))
        covered.update(pair)
    for name, tensor in weights.items():
        if name in covered:
            continue
        arr = np.asarray(tensor)
        if arr.ndim == 2 and min(arr.shape) > 0:
            ranks.append(effective_rank(arr))
    if not ranks:
        return None
    return float(sum(ranks) / len(ranks))


def compute_adapter_diff(
    adapter_a: str,
    adapter_b: str,
    *,
    top_k: int = 10,
) -> AdapterDiffReport:
    """End-to-end: load both adapters, compute layer diffs, rank top-K."""
    require_adapter_dependencies(READ_ADAPTER_DEPS)
    _require_str(adapter_a, "adapter_a")
    _require_str(adapter_b, "adapter_b")
    _validate_top_k(top_k)

    weights_a = load_adapter_weights(adapter_a)
    weights_b = load_adapter_weights(adapter_b)
    # Each adapter's own scaling: two adapters with identical factors but a
    # different lora_alpha have different ΔW.
    config_a = read_adapter_config_or_none(adapter_a)
    config_b = read_adapter_config_or_none(adapter_b)

    per_layer, only_a, only_b = compute_layer_diffs(
        weights_a, weights_b, config_a=config_a, config_b=config_b
    )

    sorted_by_change = sorted(per_layer, key=lambda d: d.frobenius, reverse=True)
    top = tuple(d.name for d in sorted_by_change[:top_k])

    rank_a = _effective_rank_average(weights_a, config_a)
    rank_b = _effective_rank_average(weights_b, config_b)

    return AdapterDiffReport(
        adapter_a=os.path.basename(os.path.normpath(adapter_a)),
        adapter_b=os.path.basename(os.path.normpath(adapter_b)),
        per_layer=per_layer,
        top_changed=top,
        effective_rank_a=rank_a,
        effective_rank_b=rank_b,
        shared_layers=len(per_layer),
        only_in_a=only_a,
        only_in_b=only_b,
    )


def render_report_json(report: AdapterDiffReport) -> str:
    """Serialise a report as canonical JSON for CI consumption."""
    if not isinstance(report, AdapterDiffReport):
        raise TypeError("report must be AdapterDiffReport")
    payload = {
        "adapter_a": report.adapter_a,
        "adapter_b": report.adapter_b,
        "shared_layers": report.shared_layers,
        "effective_rank_a": report.effective_rank_a,
        "effective_rank_b": report.effective_rank_b,
        "top_changed": list(report.top_changed),
        "only_in_a": list(report.only_in_a),
        "only_in_b": list(report.only_in_b),
        "per_layer": [asdict(d) for d in report.per_layer],
    }
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)


def render_report_markdown(report: AdapterDiffReport) -> str:
    """Human-readable markdown report (suitable for PR comments)."""
    if not isinstance(report, AdapterDiffReport):
        raise TypeError("report must be AdapterDiffReport")
    lines = [
        f"# Adapter diff: {report.adapter_a} vs {report.adapter_b}",
        "",
        f"- Shared layers: **{report.shared_layers}**",
        f"- Effective rank A: **{report.effective_rank_a}**",
        f"- Effective rank B: **{report.effective_rank_b}**",
        "",
        "## Top changed projections",
        "",
    ]
    if not report.top_changed:
        lines.append("_no shared layers_")
    else:
        for name in report.top_changed:
            lines.append(f"- `{name}`")
    if report.only_in_a:
        lines.extend(["", "## Only in A", ""])
        for name in report.only_in_a:
            lines.append(f"- `{name}`")
    if report.only_in_b:
        lines.extend(["", "## Only in B", ""])
        for name in report.only_in_b:
            lines.append(f"- `{name}`")
    return "\n".join(lines) + "\n"
