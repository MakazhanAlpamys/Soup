"""Heuristic anomaly explainer for training runs (v0.34.0 Part C).

Given a list of metric rows (from `ExperimentTracker.get_metrics`) and the
run's config dict, surface plain-English diagnoses of common training
pathologies. Each finding includes a category, a short message, and a
concrete suggestion. Intentionally rule-based — no model calls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Finding:
    category: str
    severity: str  # "info" | "warning" | "critical"
    message: str
    suggestion: str


# --- thresholds (frozen so analysis is deterministic) ---
_PLATEAU_REL_TOL = 0.005   # <0.5% loss change end-vs-start → flat
_PLATEAU_MIN_STEPS = 30
_EXPLODE_FACTOR = 3.0      # final loss > 3× initial → diverged
_NAN_PATTERNS = ("nan", "inf")
_OVERFIT_GAP = 0.5         # train < 0.3, val > train + 0.5 → overfit
_LR_TOO_LOW = 1e-6
_LR_TOO_HIGH = 5e-3


def _is_finite(value: Optional[float]) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def _check_lr_bounds(config: dict) -> Optional[Finding]:
    training = config.get("training") if isinstance(config, dict) else None
    if not isinstance(training, dict):
        return None
    learning_rate = training.get("lr") or training.get("learning_rate")
    if not _is_finite(learning_rate):
        return None
    if learning_rate < _LR_TOO_LOW:
        return Finding(
            category="lr_too_low",
            severity="warning",
            message=f"Learning rate {learning_rate:g} is below typical floor ({_LR_TOO_LOW:g}).",
            suggestion="Try lr=2e-4 for LoRA SFT, 1e-5 for full fine-tune.",
        )
    if learning_rate > _LR_TOO_HIGH:
        return Finding(
            category="lr_too_high",
            severity="warning",
            message=f"Learning rate {learning_rate:g} is above typical ceiling ({_LR_TOO_HIGH:g}).",
            suggestion="Try lr=2e-4 for LoRA, 1e-5 for full fine-tune. High LR usually diverges.",
        )
    return None


def _check_nan(metrics: List[dict]) -> Optional[Finding]:
    for row in metrics:
        loss = row.get("loss")
        if loss is None:
            continue
        if isinstance(loss, float) and (math.isnan(loss) or math.isinf(loss)):
            step = row.get("step", "?")
            return Finding(
                category="nan_loss",
                severity="critical",
                message=f"Loss became NaN/Inf at step {step}.",
                suggestion=(
                    "Check for: too-high LR, fp16 with unstable model, "
                    "corrupt batch (look at the input dataset around this step), "
                    "or a bug in custom reward fn. Try bf16 instead of fp16."
                ),
            )
        # Some trainers store as string when JSON-encoded
        if isinstance(loss, str) and any(pattern in loss.lower() for pattern in _NAN_PATTERNS):
            return Finding(
                category="nan_loss",
                severity="critical",
                message=f"Loss became NaN/Inf at step {row.get('step', '?')}.",
                suggestion="See guidance for nan_loss above.",
            )
    return None


def _check_plateau(metrics: List[dict]) -> Optional[Finding]:
    finite = [row for row in metrics if _is_finite(row.get("loss"))]
    if len(finite) < _PLATEAU_MIN_STEPS:
        return None
    first = finite[0]["loss"]
    last = finite[-1]["loss"]
    if first <= 0:
        return None
    rel_change = (first - last) / first
    if abs(rel_change) < _PLATEAU_REL_TOL:
        return Finding(
            category="loss_flat",
            severity="warning",
            message=(
                f"Loss barely moved ({first:.4f} → {last:.4f}, "
                f"{rel_change * 100:.2f}%) over {len(finite)} steps."
            ),
            suggestion=(
                "LR likely too low for this batch size. Try 2-5x higher LR. "
                "Also check that gradients are flowing (not freezing too many layers)."
            ),
        )
    return None


def _check_explosion(metrics: List[dict]) -> Optional[Finding]:
    finite = [row for row in metrics if _is_finite(row.get("loss"))]
    if len(finite) < 5:
        return None
    first = finite[0]["loss"]
    last = finite[-1]["loss"]
    if first <= 0:
        return None
    if last > first * _EXPLODE_FACTOR:
        return Finding(
            category="loss_diverged",
            severity="critical",
            message=f"Loss exploded ({first:.4f} → {last:.4f}, {last / first:.1f}x).",
            suggestion=(
                "Lower LR by 5x, enable gradient clipping (max_grad_norm=1.0), "
                "or use a warmup schedule. Consider bf16 if currently fp16."
            ),
        )
    return None


def _check_grad_norm(metrics: List[dict]) -> Optional[Finding]:
    norms = [row.get("grad_norm") for row in metrics if _is_finite(row.get("grad_norm"))]
    if len(norms) < 10:
        return None
    high_count = sum(1 for grad in norms if grad > 50.0)
    if high_count >= max(3, len(norms) // 5):
        return Finding(
            category="grad_norm_high",
            severity="warning",
            message=f"Gradient norm exceeded 50 in {high_count}/{len(norms)} logged steps.",
            suggestion=(
                "Enable gradient clipping (max_grad_norm=1.0) and/or lower LR. "
                "Persistent high grad-norm precedes loss divergence."
            ),
        )
    return None


def _check_short_run(metrics: List[dict]) -> Optional[Finding]:
    if 0 < len(metrics) < 10:
        return Finding(
            category="too_few_steps",
            severity="info",
            message=f"Only {len(metrics)} metric rows logged.",
            suggestion=(
                "Diagnostics need at least ~30 steps to detect plateau / "
                "divergence. Train longer or lower logging_steps to capture more."
            ),
        )
    if not metrics:
        return Finding(
            category="no_metrics",
            severity="info",
            message="No metric rows logged for this run.",
            suggestion=(
                "Run may have failed before the first log step. "
                "Check `soup runs show` for status."
            ),
        )
    return None


_CHECKS_LOSS = (_check_nan, _check_explosion, _check_plateau, _check_grad_norm)


def diagnose(metrics: List[dict], config: Optional[dict] = None) -> List[Finding]:
    """Return a list of findings, ordered by severity (critical first)."""
    findings: List[Finding] = []
    short = _check_short_run(metrics)
    if short is not None and short.category == "no_metrics":
        return [short]

    for check in _CHECKS_LOSS:
        result = check(metrics)
        if result is not None:
            findings.append(result)

    if isinstance(config, dict):
        lr_finding = _check_lr_bounds(config)
        if lr_finding is not None:
            findings.append(lr_finding)

    # Surface "too few steps" only when there is no other diagnosis. When a
    # NaN or divergence has already been flagged, the short-run note adds
    # noise without changing the user's next action.
    if short is not None and short.category == "too_few_steps" and not findings:
        findings.append(short)

    severity_order = {"critical": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda finding: severity_order.get(finding.severity, 3))
    return findings
