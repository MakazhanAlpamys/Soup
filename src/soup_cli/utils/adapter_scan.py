"""Weight-space backdoor detector for LoRA adapters (v0.60.0 Part A).

Spectral analysis of adapter weights pre-load. Catches the most common
weight-space backdoor pattern (rank-1 perturbation injected into one or two
projection matrices) by flagging singular-value distributions that deviate
from a healthy LoRA fingerprint. Inspired by 2025 weight-space LoRA
detection research; intentionally conservative to keep false-positive rate
low on legitimate fine-tunes.

Pure numpy math (no torch); reuses the v0.57.0 ``adapter_diff`` safetensors
loader so the on-disk surface stays single-source-of-truth. Containment +
symlink rejection at every file load (TOCTOU defence, mirrors v0.53.1
``enforce_under_cwd_and_no_symlink`` policy).

Public surface:

- ``ScanFinding`` / ``ScanReport`` frozen dataclasses.
- ``compute_spectral_features(matrix)`` -> dict of ratios used by the rules.
- ``scan_adapter_weights(weights, *, adapter_name)`` -> ``ScanReport``.
- ``scan_adapter(adapter_dir)`` -> ``ScanReport`` (loads safetensors + scans).
"""

from __future__ import annotations

import math
import os
import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

_VALID_KINDS = frozenset(
    {"rank1_dominance", "frobenius_outlier", "nan_inf", "energy_concentration"}
)
_VALID_SEVERITIES = frozenset({"OK", "WARN", "FAIL"})

# Thresholds chosen so legitimate LoRA fine-tunes pass while injected rank-1
# trojans (the 2025 research pattern) trip. Tuned against synthetic fixtures
# in ``tests/test_v0600_part_a.py``.
_RANK1_DOMINANCE_WARN = 50.0
_RANK1_DOMINANCE_FAIL = 200.0
_ENERGY_TOP1_WARN = 0.75
_ENERGY_TOP1_FAIL = 0.95
_FROB_OUTLIER_WARN_SIGMA = 4.0
_FROB_OUTLIER_FAIL_SIGMA = 8.0
_MAX_ADAPTER_NAME_LEN = 256
_MAX_LAYER_NAME_LEN = 256


@dataclass(frozen=True)
class ScanFinding:
    """One flagged layer + the rule that fired."""

    layer: str
    kind: str
    severity: str
    value: float
    threshold: float
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.layer, str) or not self.layer:
            raise ValueError("layer must be non-empty str")
        if "\x00" in self.layer or len(self.layer) > _MAX_LAYER_NAME_LEN:
            raise ValueError("layer name invalid (null byte or > 256 chars)")
        if self.kind not in _VALID_KINDS:
            raise ValueError(
                f"kind must be one of {sorted(_VALID_KINDS)}, got {self.kind!r}"
            )
        if self.severity not in _VALID_SEVERITIES:
            raise ValueError(
                f"severity must be one of {sorted(_VALID_SEVERITIES)}, "
                f"got {self.severity!r}"
            )
        for fld_value, fld_name in (
            (self.value, "value"),
            (self.threshold, "threshold"),
        ):
            if isinstance(fld_value, bool):
                raise ValueError(f"{fld_name} must be float, not bool")
            if not isinstance(fld_value, (int, float)):
                raise ValueError(f"{fld_name} must be float")
            if not math.isfinite(float(fld_value)):
                raise ValueError(f"{fld_name} must be finite")
        if not isinstance(self.message, str):
            raise ValueError("message must be str")


@dataclass(frozen=True)
class ScanReport:
    """End-to-end scan result for one adapter."""

    adapter: str
    findings: Tuple[ScanFinding, ...]
    overall: str
    summary: str

    def __post_init__(self) -> None:
        if not isinstance(self.adapter, str) or not self.adapter:
            raise ValueError("adapter must be non-empty str")
        if "\x00" in self.adapter or len(self.adapter) > _MAX_ADAPTER_NAME_LEN:
            raise ValueError("adapter name invalid (null byte or > 256 chars)")
        if self.overall not in _VALID_SEVERITIES:
            raise ValueError(
                f"overall must be one of {sorted(_VALID_SEVERITIES)}, "
                f"got {self.overall!r}"
            )
        if not isinstance(self.findings, tuple):
            raise ValueError("findings must be tuple")
        for entry in self.findings:
            if not isinstance(entry, ScanFinding):
                raise ValueError("findings entries must be ScanFinding")


def _require_str(value: object, field: str, *, max_len: int = 256) -> str:
    if isinstance(value, bool) or not isinstance(value, str):
        raise TypeError(f"{field} must be str, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{field} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain null bytes")
    if len(value) > max_len:
        raise ValueError(f"{field} must be ≤{max_len} chars")
    return value


def compute_spectral_features(matrix: Any) -> dict:
    """Return spectral features used by the rule engine.

    Keys:
    - ``top_sv_ratio``: ``s_1 / s_2`` (or ``s_1`` if no second SV). Rank-1
      trojans drive this >> 50.
    - ``energy_top1``: ``s_1**2 / sum(s_i**2)``. Energy concentration in the
      top singular vector. Healthy LoRA fine-tunes sit < 0.5.
    - ``effective_rank``: ``exp(H(p))`` where ``p = s_i / sum(s_i)``. Mirrors
      v0.57.0 ``adapter_diff.effective_rank`` semantics.
    - ``frobenius``: ``sqrt(sum(s_i**2))``.
    """
    import numpy as np

    if not isinstance(matrix, (list, tuple)) and not hasattr(matrix, "__array__"):
        # Reject obvious non-array inputs early so callers get a clear error.
        if isinstance(matrix, (int, float, str, bool)) or matrix is None:
            raise TypeError("matrix must be a 2D array-like")

    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim < 2:
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        else:
            raise TypeError("matrix must be at least 1D")
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    if arr.size == 0:
        return {
            "top_sv_ratio": 0.0,
            "energy_top1": 0.0,
            "effective_rank": 0.0,
            "frobenius": 0.0,
        }

    try:
        singular = np.linalg.svd(arr, compute_uv=False)
    except np.linalg.LinAlgError:
        return {
            "top_sv_ratio": 0.0,
            "energy_top1": 0.0,
            "effective_rank": 0.0,
            "frobenius": 0.0,
        }

    singular = np.asarray(singular, dtype=np.float64)
    if singular.size == 0:
        return {
            "top_sv_ratio": 0.0,
            "energy_top1": 0.0,
            "effective_rank": 0.0,
            "frobenius": 0.0,
        }

    top = float(singular[0])
    if singular.size > 1 and singular[1] > 0:
        top_sv_ratio = top / float(singular[1])
    else:
        # Single-element or degenerate spectrum.
        top_sv_ratio = top if top > 0 else 0.0

    energy = float(np.sum(singular * singular))
    energy_top1 = (top * top) / energy if energy > 0 else 0.0

    total_sum = float(np.sum(singular))
    if total_sum > 0:
        probs = singular / total_sum
        probs = probs[probs > 1e-12]
        if probs.size > 0:
            entropy = float(-np.sum(probs * np.log(probs)))
            effective_rank = float(math.exp(entropy))
        else:
            effective_rank = 0.0
    else:
        effective_rank = 0.0

    frob = float(math.sqrt(energy)) if math.isfinite(energy) else float("inf")

    return {
        "top_sv_ratio": float(top_sv_ratio),
        "energy_top1": float(energy_top1),
        "effective_rank": effective_rank,
        "frobenius": frob,
    }


def _has_non_finite(matrix: Any) -> bool:
    import numpy as np

    arr = np.asarray(matrix)
    if arr.size == 0:
        return False
    if not np.issubdtype(arr.dtype, np.floating):
        return False
    return bool(np.any(~np.isfinite(arr)))


def _classify_overall(findings: Tuple[ScanFinding, ...]) -> str:
    if any(f.severity == "FAIL" for f in findings):
        return "FAIL"
    if any(f.severity == "WARN" for f in findings):
        return "WARN"
    return "OK"


def _scan_one_layer(name: str, matrix: Any) -> list[ScanFinding]:
    import numpy as np

    findings: list[ScanFinding] = []

    if _has_non_finite(matrix):
        findings.append(
            ScanFinding(
                layer=name,
                kind="nan_inf",
                severity="FAIL",
                value=1.0,
                threshold=0.0,
                message="weights contain NaN or Inf",
            )
        )
        # Don't run spectral analysis on broken tensors.
        return findings

    arr = np.asarray(matrix)
    if arr.ndim < 2:
        return findings

    feats = compute_spectral_features(arr)
    ratio = feats["top_sv_ratio"]
    energy = feats["energy_top1"]

    if ratio >= _RANK1_DOMINANCE_FAIL:
        findings.append(
            ScanFinding(
                layer=name,
                kind="rank1_dominance",
                severity="FAIL",
                value=ratio,
                threshold=_RANK1_DOMINANCE_FAIL,
                message=(
                    f"top singular value is {ratio:.1f}x the next — "
                    "consistent with injected rank-1 trojan"
                ),
            )
        )
    elif ratio >= _RANK1_DOMINANCE_WARN:
        findings.append(
            ScanFinding(
                layer=name,
                kind="rank1_dominance",
                severity="WARN",
                value=ratio,
                threshold=_RANK1_DOMINANCE_WARN,
                message=(
                    f"top singular value is {ratio:.1f}x the next "
                    "(unusual but not definitive)"
                ),
            )
        )

    if energy >= _ENERGY_TOP1_FAIL:
        findings.append(
            ScanFinding(
                layer=name,
                kind="energy_concentration",
                severity="FAIL",
                value=energy,
                threshold=_ENERGY_TOP1_FAIL,
                message=(
                    f"{energy * 100:.1f}% of energy in top singular vector"
                ),
            )
        )
    elif energy >= _ENERGY_TOP1_WARN:
        findings.append(
            ScanFinding(
                layer=name,
                kind="energy_concentration",
                severity="WARN",
                value=energy,
                threshold=_ENERGY_TOP1_WARN,
                message=(
                    f"{energy * 100:.1f}% of energy in top singular vector"
                ),
            )
        )
    return findings


def _scan_frobenius_outliers(
    per_layer_norms: dict[str, float],
) -> list[ScanFinding]:
    """Flag layers whose Frobenius norm is way above the population mean.

    Uses median + MAD (robust to a single outlier dominating the std). Layers
    with same prefix (e.g. ``lora_A`` vs ``lora_B``) are bucketed together.
    """
    if len(per_layer_norms) < 3:
        return []

    findings: list[ScanFinding] = []
    # Bucket by suffix (everything after the last `.`)
    buckets: dict[str, list[tuple[str, float]]] = {}
    for name, norm in per_layer_norms.items():
        # Use suffix only — `.lora_A.weight` vs `.lora_B.weight` should be compared
        # within their type, not across.
        suffix = ".".join(name.rsplit(".", 2)[-2:]) if "." in name else name
        buckets.setdefault(suffix, []).append((name, norm))

    for suffix, entries in buckets.items():
        if len(entries) < 3:
            continue
        norms = [n for _, n in entries]
        median = statistics.median(norms)
        # Median absolute deviation, scaled to match std under normal.
        mad = statistics.median([abs(n - median) for n in norms]) * 1.4826
        if mad <= 0:
            continue
        for name, norm in entries:
            z = (norm - median) / mad
            if z >= _FROB_OUTLIER_FAIL_SIGMA:
                findings.append(
                    ScanFinding(
                        layer=name,
                        kind="frobenius_outlier",
                        severity="FAIL",
                        value=float(z),
                        threshold=_FROB_OUTLIER_FAIL_SIGMA,
                        message=(
                            f"frobenius norm {z:.1f} robust-sigmas above peers "
                            f"in bucket {suffix!r}"
                        ),
                    )
                )
            elif z >= _FROB_OUTLIER_WARN_SIGMA:
                findings.append(
                    ScanFinding(
                        layer=name,
                        kind="frobenius_outlier",
                        severity="WARN",
                        value=float(z),
                        threshold=_FROB_OUTLIER_WARN_SIGMA,
                        message=(
                            f"frobenius norm {z:.1f} robust-sigmas above peers "
                            f"in bucket {suffix!r}"
                        ),
                    )
                )
    return findings


def scan_adapter_weights(
    weights: Mapping[str, Any], *, adapter_name: str,
) -> ScanReport:
    """Pure-function scan over an in-memory weights map.

    Returns a ``ScanReport`` with per-layer findings and an overall verdict
    (``OK`` / ``WARN`` / ``FAIL``). Public surface; callers can construct
    weights from any source (safetensors, mocks, in-memory).
    """
    if not isinstance(weights, Mapping):
        raise TypeError("weights must be a Mapping")
    _require_str(adapter_name, "adapter_name")

    import numpy as np

    findings: list[ScanFinding] = []
    norms: dict[str, float] = {}
    for name, matrix in weights.items():
        _require_str(name, "layer name")
        findings.extend(_scan_one_layer(name, matrix))
        arr = np.asarray(matrix, dtype=np.float64)
        if arr.size > 0 and np.all(np.isfinite(arr)):
            norms[name] = float(math.sqrt(np.sum(arr * arr)))

    findings.extend(_scan_frobenius_outliers(norms))

    findings_tuple = tuple(findings)
    overall = _classify_overall(findings_tuple)
    fail_count = sum(1 for f in findings_tuple if f.severity == "FAIL")
    warn_count = sum(1 for f in findings_tuple if f.severity == "WARN")
    summary = (
        f"scanned {len(weights)} tensor(s), "
        f"{fail_count} FAIL / {warn_count} WARN"
    )
    return ScanReport(
        adapter=adapter_name,
        findings=findings_tuple,
        overall=overall,
        summary=summary,
    )


def scan_adapter(adapter_dir: str) -> ScanReport:
    """Containment-checked safetensors load + scan.

    Raises ``ValueError`` if the dir is outside cwd or a symlink; raises
    ``FileNotFoundError`` if no adapter_model.safetensors is present.
    """
    # Reuse v0.57.0 loader so the on-disk surface stays consistent.
    from soup_cli.utils.adapter_diff import load_adapter_weights

    enforce_under_cwd_and_no_symlink(adapter_dir, "adapter")
    weights = load_adapter_weights(adapter_dir)
    name = os.path.basename(os.path.normpath(adapter_dir))
    return scan_adapter_weights(weights, adapter_name=name)


def render_report_text(report: ScanReport) -> str:
    """Plain-text rendering of a scan report (used by the CLI)."""
    if not isinstance(report, ScanReport):
        raise TypeError("report must be ScanReport")
    lines = [
        f"Adapter scan: {report.adapter}",
        f"Verdict: {report.overall}",
        f"Summary: {report.summary}",
    ]
    if report.findings:
        lines.append("")
        lines.append("Findings:")
        for finding in report.findings:
            lines.append(
                f"  [{finding.severity}] {finding.layer} ({finding.kind}) "
                f"{finding.value:.3f} >= {finding.threshold:.3f}: "
                f"{finding.message}"
            )
    return "\n".join(lines) + "\n"
