"""Single-adapter pre-flight health audit and silent failure diagnostics (Issue #1721).

Pure numpy math (no torch); safetensors is loaded lazily with float64 accumulation.
Scans for:
- Total and per-layer Frobenius weight delta ||ΔW||_F
- Live layer fraction (non-zero updates)
- All-zero lora_B layers (initialization traps / frozen layers)
- Leaked .inner. wrapper namespaces (adapters saved before #1011)

Outputs a detailed report and terminal verdict line: `alive` vs `inactive: <reason>`.
"""

from __future__ import annotations

import json
import math
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

from soup_cli.utils.adapter_diff import _adapter_weights_path, _require_str
from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink, is_under_cwd
from soup_cli.utils.terminal import strip_control

INNER_NAMESPACE_MARKER = ".inner."
_MAX_LAYERS = 10_000


@dataclass(frozen=True)
class LayerCheck:
    """Health metrics for a single LoRA projection pair or tensor."""

    name: str
    frobenius: float
    is_b_zero: bool
    has_non_finite: bool
    shape_a: Tuple[int, ...]
    shape_b: Tuple[int, ...]


@dataclass(frozen=True)
class AdapterCheckReport:
    """Full health diagnostic report for a single LoRA adapter."""

    adapter: str
    verdict: str  # "alive" or "inactive"
    reason: Optional[str]
    total_frobenius: float
    live_fraction: float
    live_layers: int
    total_layers: int
    all_zero_lora_b_layers: Tuple[str, ...]
    inner_keys: Tuple[str, ...]
    per_layer: Tuple[LayerCheck, ...]
    orphaned_layers: Tuple[str, ...] = ()
    standalone_tensors: Tuple[LayerCheck, ...] = ()
    shape_mismatches: Tuple[str, ...] = ()

    @property
    def verdict_line(self) -> str:
        """The single final line required by contract."""
        if self.verdict == "alive":
            return "alive"
        return f"inactive: {self.reason}" if self.reason else "inactive"


def _read_config(adapter_dir: Path) -> dict[str, Any]:
    """Read adapter_config.json if present and not a symlink."""
    cfg_path = adapter_dir / "adapter_config.json"
    if not os.path.lexists(str(cfg_path)):
        return {}
    st = os.lstat(str(cfg_path))
    if stat.S_ISLNK(st.st_mode):
        raise ValueError(
            f"{adapter_dir.name}/adapter_config.json: must not be a symlink"
        )
    try:
        with open(cfg_path, encoding="utf-8") as f:
            data = json.load(f)
    except (ValueError, OSError) as exc:
        raise ValueError(
            f"{adapter_dir.name}/adapter_config.json: unreadable or invalid JSON: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"{adapter_dir.name}/adapter_config.json: expected JSON object"
        )
    return data


def _load_safetensors_numpy(path: Path) -> dict[str, Any]:
    """Lazy-load adapter tensors using safetensors numpy framework."""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "safetensors package required; pip install safetensors"
        ) from exc

    result: dict[str, Any] = {}
    try:
        with safe_open(str(path), framework="numpy") as f:
            keys = list(f.keys())
            if len(keys) > _MAX_LAYERS:
                raise ValueError(f"adapter has >{_MAX_LAYERS} tensors")
            for key in keys:
                _require_str(key, "tensor name")
                result[key] = f.get_tensor(key)
    except (ValueError, RuntimeError):
        raise
    except Exception as exc:
        raise ValueError(
            f"Corrupt or unreadable safetensors weights in {path.name}: {exc}"
        ) from exc
    return result


def _split_lora_role(name: str) -> Optional[Tuple[str, str]]:
    """Extract (pair_key, role) for a LoRA weight name.

    Returns (pair_key, 'A') or (pair_key, 'B'), or None if not a LoRA weight.
    Replaces the leaf role tag with a unified marker so that matching A and B
    tensors share the exact same pair key.
    """
    for marker, role in (
        (".lora_embedding_A.", "A"),
        (".lora_embedding_B.", "B"),
        (".lora_A.", "A"),
        (".lora_B.", "B"),
    ):
        if marker in name:
            idx = name.rfind(marker)
            sub = ".__LORA_EMB__." if "embedding" in marker else ".__LORA__."
            pair_key = name[:idx] + sub + name[idx + len(marker) :]
            return pair_key, role

    for suffix, role in (
        (".lora_embedding_A", "A"),
        (".lora_embedding_B", "B"),
        (".lora_A", "A"),
        (".lora_B", "B"),
    ):
        if name.endswith(suffix):
            idx = name.rfind(suffix)
            sub = ".__LORA_EMB__" if "embedding" in suffix else ".__LORA__"
            pair_key = name[:idx] + sub
            return pair_key, role

    return None


def check_adapter(adapter_dir: str | Path) -> AdapterCheckReport:
    """Audit single adapter health for silent training failures.

    Checks:
    1. Total and per-layer ||ΔW||_F on LoRA pairs (scaling * ||B @ A||_F).
    2. Live fraction: fraction of LoRA layers with non-zero parameter updates.
    3. All-zero lora_B layers: layers that remained at initial zero state.
    4. .inner. keys: layer-streaming wrapper namespace leaks.
    """
    enforce_under_cwd_and_no_symlink(str(adapter_dir), "adapter")
    dir_path = Path(adapter_dir).resolve()
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    weights_file = _adapter_weights_path(dir_path)
    if not is_under_cwd(str(weights_file)):
        raise ValueError(
            f"adapter weights must stay under cwd: {weights_file.name}"
        )

    config = _read_config(dir_path)
    weights = _load_safetensors_numpy(weights_file)

    lora_r = config.get("r")
    lora_alpha = config.get("lora_alpha")
    use_rslora = bool(config.get("use_rslora", False))
    scaling = 1.0
    if (
        isinstance(lora_r, (int, float))
        and not isinstance(lora_r, bool)
        and lora_r > 0
    ):
        if (
            isinstance(lora_alpha, (int, float))
            and not isinstance(lora_alpha, bool)
        ):
            if use_rslora:
                scaling = abs(float(lora_alpha)) / math.sqrt(float(lora_r))
            else:
                scaling = abs(float(lora_alpha)) / float(lora_r)

    import numpy as np

    # Check for leaked .inner. wrapper namespaces
    inner_keys = tuple(
        sorted(k for k in weights.keys() if INNER_NAMESPACE_MARKER in k)
    )

    # Discover and group LoRA pairs
    lora_modules: dict[str, dict[str, Any]] = {}
    other_tensors: dict[str, Any] = {}

    for name, tensor in weights.items():
        role_info = _split_lora_role(name)
        if role_info is not None:
            pair_key, role = role_info
            lora_modules.setdefault(pair_key, {})[role] = (name, tensor)
        else:
            other_tensors[name] = tensor

    layer_checks: list[LayerCheck] = []
    orphaned_layers: list[str] = []
    shape_mismatches: list[str] = []
    has_any_non_finite = False

    for mod_name in sorted(lora_modules.keys()):
        pair = lora_modules[mod_name]
        a_info = pair.get("A")
        b_info = pair.get("B")

        if not a_info or not b_info:
            # Incomplete LoRA pair (missing A or missing B)
            tensor = a_info[1] if a_info else b_info[1]
            name = a_info[0] if a_info else b_info[0]
            orphaned_layers.append(name)
            arr = np.asarray(tensor, dtype=np.float64)
            is_finite = bool(np.isfinite(arr).all())
            if not is_finite:
                has_any_non_finite = True
            # Orphaned single tensors cannot form ΔW = B @ A; fro stays 0.0
            fro = 0.0
            layer_checks.append(
                LayerCheck(
                    name=name,
                    frobenius=fro,
                    is_b_zero=False,
                    has_non_finite=not is_finite,
                    shape_a=tuple(arr.shape) if a_info else (),
                    shape_b=tuple(arr.shape) if b_info else (),
                )
            )
            continue

        a_name, a_tensor = a_info
        b_name, b_tensor = b_info

        a_arr = np.asarray(a_tensor, dtype=np.float64)
        b_arr = np.asarray(b_tensor, dtype=np.float64)

        is_a_finite = bool(np.isfinite(a_arr).all())
        is_b_finite = bool(np.isfinite(b_arr).all())
        is_finite = is_a_finite and is_b_finite
        if not is_finite:
            has_any_non_finite = True

        is_b_zero = bool(np.all(b_arr == 0.0))

        # Compute ||ΔW||_F = scaling * ||B @ A||_F
        # Using exact trace identity: ||B @ A||_F^2 = sum((B.T @ B) * (A @ A.T))
        if is_b_zero:
            fro = 0.0
        elif not is_finite:
            fro = float("nan")
        else:
            # Squeeze unit dimensions and reshape to 2D
            a_2d = a_arr.reshape(1, -1) if a_arr.ndim == 1 else (
                a_arr.reshape(a_arr.shape[0], -1) if a_arr.ndim > 2 else a_arr
            )
            b_2d = b_arr.reshape(-1, 1) if b_arr.ndim == 1 else (
                b_arr.reshape(b_arr.shape[0], -1) if b_arr.ndim > 2 else b_arr
            )

            # Determine contraction layout:
            # Standard Linear: B is (out, r), A is (r, in) -> rank axis is (B axis 1, A axis 0)
            # Transposed / Conv1D: B is (r, out), A is (in, r) -> rank axis is (B axis 0, A axis 1)
            is_std = b_2d.ndim == 2 and a_2d.ndim == 2 and b_2d.shape[1] == a_2d.shape[0]
            is_trans = b_2d.ndim == 2 and a_2d.ndim == 2 and b_2d.shape[0] == a_2d.shape[1]

            if is_std and is_trans:
                # Disambiguate when in_features == out_features
                integral_r = (
                    int(lora_r)
                    if isinstance(lora_r, (int, float))
                    and not isinstance(lora_r, bool)
                    and lora_r > 0
                    and float(lora_r).is_integer()
                    else None
                )
                if integral_r is not None:
                    if b_2d.shape[1] == integral_r:
                        is_trans = False
                    elif b_2d.shape[0] == integral_r:
                        is_std = False
                elif b_2d.shape[1] <= b_2d.shape[0]:
                    # Rank is typically the bottleneck dimension (r <= out_features)
                    is_trans = False
                else:
                    is_std = False

            if is_std:
                btb = np.matmul(b_2d.T, b_2d)
                aat = np.matmul(a_2d, a_2d.T)
                sum_sq = float(np.sum(btb * aat))
                fro = float(scaling * math.sqrt(max(0.0, sum_sq)))
            elif is_trans:
                bbt = np.matmul(b_2d, b_2d.T)
                ata = np.matmul(a_2d.T, a_2d)
                sum_sq = float(np.sum(bbt * ata))
                fro = float(scaling * math.sqrt(max(0.0, sum_sq)))
            else:
                try:
                    delta = scaling * np.matmul(b_2d, a_2d)
                    fro = float(np.sqrt(np.sum(delta * delta)))
                except (ValueError, TypeError):
                    fro = float("nan")
                    shape_mismatches.append(mod_name)

        layer_checks.append(
            LayerCheck(
                name=mod_name,
                frobenius=fro,
                is_b_zero=is_b_zero,
                has_non_finite=not is_finite,
                shape_a=tuple(a_arr.shape),
                shape_b=tuple(b_arr.shape),
            )
        )

    # Process other standalone tensors (e.g. bias or custom heads)
    standalone_checks: list[LayerCheck] = []
    for name in sorted(other_tensors.keys()):
        arr = np.asarray(other_tensors[name], dtype=np.float64)
        is_finite = bool(np.isfinite(arr).all())
        if not is_finite:
            has_any_non_finite = True
        is_zero = bool(np.all(arr == 0.0))
        fro = 0.0 if is_zero else float(np.sqrt(np.sum(arr * arr)))
        standalone_checks.append(
            LayerCheck(
                name=name,
                frobenius=fro,
                is_b_zero=False,
                has_non_finite=not is_finite,
                shape_a=(),
                shape_b=tuple(arr.shape),
            )
        )

    # Compute metrics strictly from LoRA projections
    total_layers = len(layer_checks)
    all_zero_b_layers = tuple(lc.name for lc in layer_checks if lc.is_b_zero)
    live_layers = len(
        [lc for lc in layer_checks if not lc.is_b_zero and lc.frobenius > 0.0]
    )
    live_fraction = (live_layers / total_layers) if total_layers > 0 else 0.0

    if has_any_non_finite or shape_mismatches:
        total_frobenius = float("nan")
    else:
        sum_sq_total = sum(
            lc.frobenius * lc.frobenius
            for lc in layer_checks
            if math.isfinite(lc.frobenius)
        )
        total_frobenius = (
            math.sqrt(sum_sq_total)
            if math.isfinite(sum_sq_total)
            else float("nan")
        )

    # Determine verdict according to contract
    if inner_keys:
        verdict = "inactive"
        reason = f"leaked .inner. keys detected ({len(inner_keys)} tensors)"
    elif orphaned_layers:
        verdict = "inactive"
        reason = (
            f"incomplete LoRA pairs detected ({len(orphaned_layers)} orphan"
            f"{'s' if len(orphaned_layers) != 1 else ''})"
        )
    elif shape_mismatches:
        verdict = "inactive"
        reason = (
            f"shape mismatch in LoRA projections ({len(shape_mismatches)} layer"
            f"{'s' if len(shape_mismatches) != 1 else ''})"
        )
    elif has_any_non_finite or math.isnan(total_frobenius) or math.isinf(total_frobenius):
        verdict = "inactive"
        reason = "non-finite weights detected (NaN/Inf)"
    elif total_layers == 0:
        verdict = "inactive"
        reason = "no lora weights found"
    elif len(all_zero_b_layers) == total_layers:
        verdict = "inactive"
        reason = f"all lora_B layers are zero ({total_layers}/{total_layers})"
    elif len(all_zero_b_layers) > 0:
        verdict = "inactive"
        reason = f"lora_B layers are zero ({len(all_zero_b_layers)}/{total_layers})"
    elif total_frobenius <= 0.0 or live_fraction <= 0.0:
        verdict = "inactive"
        reason = "total ||ΔW||_F is 0.0"
    else:
        verdict = "alive"
        reason = None

    return AdapterCheckReport(
        adapter=str(dir_path),
        verdict=verdict,
        reason=reason,
        total_frobenius=total_frobenius,
        live_fraction=live_fraction,
        live_layers=live_layers,
        total_layers=total_layers,
        all_zero_lora_b_layers=all_zero_b_layers,
        inner_keys=inner_keys,
        per_layer=tuple(layer_checks),
        orphaned_layers=tuple(orphaned_layers),
        standalone_tensors=tuple(standalone_checks),
        shape_mismatches=tuple(shape_mismatches),
    )


def render_check_terminal(report: AdapterCheckReport) -> str:
    """Format human-readable terminal output ending with the exact verdict line."""
    lines = [
        f"Adapter: {strip_control(report.adapter)}",
        f"Total ||ΔW||_F: {report.total_frobenius:.6f}",
        (
            f"Live fraction: {report.live_fraction:.1%} "
            f"({report.live_layers}/{report.total_layers} layers)"
        ),
        f"All-zero lora_B layers: {len(report.all_zero_lora_b_layers)}/{report.total_layers}",
        f"Leaked .inner. keys: {len(report.inner_keys)}",
    ]

    if report.standalone_tensors:
        lines.append(f"Standalone non-LoRA tensors: {len(report.standalone_tensors)}")

    if report.orphaned_layers:
        lines.append(f"Incomplete LoRA pairs: {len(report.orphaned_layers)}")
        for name in report.orphaned_layers[:5]:
            lines.append(f"  - {strip_control(name)}")
        if len(report.orphaned_layers) > 5:
            lines.append(f"  ... and {len(report.orphaned_layers) - 5} more")

    if report.shape_mismatches:
        lines.append(f"Shape-mismatched LoRA pairs: {len(report.shape_mismatches)}")
        for name in report.shape_mismatches[:5]:
            lines.append(f"  - {strip_control(name)}")
        if len(report.shape_mismatches) > 5:
            lines.append(f"  ... and {len(report.shape_mismatches) - 5} more")

    if report.all_zero_lora_b_layers and len(report.all_zero_lora_b_layers) < report.total_layers:
        lines.append("Zero lora_B projections:")
        for name in report.all_zero_lora_b_layers[:5]:
            lines.append(f"  - {strip_control(name)}")
        if len(report.all_zero_lora_b_layers) > 5:
            lines.append(f"  ... and {len(report.all_zero_lora_b_layers) - 5} more")

    if report.inner_keys:
        lines.append("Leaked wrapper keys:")
        for key in report.inner_keys[:5]:
            lines.append(f"  - {strip_control(key)}")
        if len(report.inner_keys) > 5:
            lines.append(f"  ... and {len(report.inner_keys) - 5} more")

    # Contract requires exact terminal verdict line
    lines.append(report.verdict_line)
    return "\n".join(lines)


def _sanitize_for_json(val: Any) -> Any:
    """Coerce non-finite floats to None (null) for RFC 8259 compliance."""
    if isinstance(val, float):
        return val if math.isfinite(val) else None
    if isinstance(val, dict):
        return {k: _sanitize_for_json(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_sanitize_for_json(v) for v in val]
    return val


def render_check_json(report: AdapterCheckReport) -> str:
    """Render check report as machine-readable JSON document."""
    data = _sanitize_for_json(asdict(report))
    data["verdict_line"] = report.verdict_line
    return json.dumps(data, indent=2, sort_keys=True, allow_nan=False)
