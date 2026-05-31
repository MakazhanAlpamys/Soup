"""Per-layer LoRA adapter diff math + report rendering (v0.57.0 Part A).

Pure numpy math (no torch); safetensors is loaded lazily so import is cheap.
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
    """Frobenius-norm diff for a single LoRA parameter tensor."""

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
    total = float(np.sum(singular))
    if total <= eps:
        return 0.0
    probs = singular / total
    probs = probs[probs > eps]
    if probs.size == 0:
        return 0.0
    entropy = float(-np.sum(probs * np.log(probs)))
    return float(math.exp(entropy))


def compute_layer_diffs(
    weights_a: Mapping[str, Any],
    weights_b: Mapping[str, Any],
) -> Tuple[Tuple[LayerDiff, ...], Tuple[str, ...], Tuple[str, ...]]:
    """Compute per-layer Frobenius diffs for every name in both adapters.

    Returns ``(per_layer, only_in_a, only_in_b)``. ``per_layer`` covers the
    intersection of names sorted alphabetically.
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

    shared = sorted(names_a & names_b)
    only_a = tuple(sorted(names_a - names_b))
    only_b = tuple(sorted(names_b - names_a))

    diffs = []
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
    return tuple(diffs), only_a, only_b


def _validate_top_k(top_k: object) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be int")
    if top_k < _MIN_TOP_K or top_k > _MAX_TOP_K:
        raise ValueError(f"top_k must be in [{_MIN_TOP_K}, {_MAX_TOP_K}]")
    return top_k


def _load_safetensors(path: Path) -> Mapping[str, Any]:
    """Lazy-load adapter_model.safetensors via the ``safetensors`` package."""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "safetensors package required; pip install safetensors"
        ) from exc
    result: dict[str, Any] = {}
    with safe_open(str(path), framework="numpy") as f:
        for key in f.keys():
            _require_str(key, "tensor name")
            result[key] = f.get_tensor(key)
            if len(result) > _MAX_LAYERS:
                raise ValueError(f"adapter has >{_MAX_LAYERS} tensors")
    return result


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


def _effective_rank_average(weights: Mapping[str, Any]) -> Optional[float]:
    """Mean effective-rank across 2D LoRA matrices (None if no 2D tensors)."""
    import numpy as np

    ranks: list[float] = []
    for tensor in weights.values():
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
    _require_str(adapter_a, "adapter_a")
    _require_str(adapter_b, "adapter_b")
    _validate_top_k(top_k)

    weights_a = load_adapter_weights(adapter_a)
    weights_b = load_adapter_weights(adapter_b)

    per_layer, only_a, only_b = compute_layer_diffs(weights_a, weights_b)

    sorted_by_change = sorted(per_layer, key=lambda d: d.frobenius, reverse=True)
    top = tuple(d.name for d in sorted_by_change[:top_k])

    rank_a = _effective_rank_average(weights_a)
    rank_b = _effective_rank_average(weights_b)

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
