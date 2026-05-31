"""Git-hook regression gate — `soup eval gate install` (v0.55.0 Part D).

Generates a portable pre-push hook that runs `soup eval against` against
a baseline run id and blocks the push if any of:

* task accuracy
* refusal rate
* format validity
* p95 latency

regress past the configured thresholds. Threshold checks use a
paired-bootstrap CI so single-outlier rows do not flip the gate.

Public surface
--------------
- Frozen dataclass: ``GateThresholds``, ``RegressionVerdict``.
- Pure functions: ``render_pre_push_hook``, ``write_pre_push_hook``,
  ``paired_bootstrap_ci``, ``decide_regression``.
"""

from __future__ import annotations

import math
import os
import random
import re
import shlex
import stat
import tempfile
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from soup_cli.utils.paths import is_under_cwd

_MAX_FILE_BYTES = 64 * 1024  # hooks are tiny — 64 KiB plenty
_MIN_BOOTSTRAP_SAMPLES = 100
_MAX_BOOTSTRAP_SAMPLES = 100_000
_DEFAULT_BOOTSTRAP_SAMPLES = 1000
_DEFAULT_CI_LEVEL = 0.95

# Regex on the run id — alphanumeric + ``-_`` only; mirrors v0.26.0
# registry name policy.
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,127}$")


@dataclass(frozen=True)
class GateThresholds:
    """Per-metric regression tolerance for the pre-push gate.

    Each tolerance is the *minimum acceptable* delta vs the baseline.
    Negative means a regression of that magnitude is still acceptable.
    Positive thresholds tighten the gate (require an improvement).
    """

    task_accuracy: float = -0.02
    refusal_rate: float = -0.05
    format_validity: float = -0.02
    p95_latency_ms: float = 100.0  # latency: lower-is-better — see semantics

    def __post_init__(self) -> None:
        # Every threshold must be a finite, real number. Pydantic-style
        # bool rejection (bool is a subclass of int — Python policy).
        for fld in (
            "task_accuracy", "refusal_rate", "format_validity",
            "p95_latency_ms",
        ):
            value = getattr(self, fld)
            if isinstance(value, bool):
                raise TypeError(f"{fld} must be float, got bool")
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"{fld} must be a number, got {type(value).__name__}"
                )
            if not math.isfinite(float(value)):
                raise ValueError(f"{fld} must be finite")


@dataclass(frozen=True)
class RegressionVerdict:
    """Output of :func:`decide_regression`.

    ``regressed`` is True iff any metric breached its tolerance after
    factoring in the paired-bootstrap CI. ``offenders`` names every
    metric that breached.
    """

    regressed: bool
    offenders: tuple[str, ...]
    ci_lower: float
    ci_upper: float
    delta_mean: float


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _require_finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be float, got bool")
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"{field_name} must be a number, got {type(value).__name__}"
        )
    f = float(value)
    if not math.isfinite(f):
        raise ValueError(f"{field_name} must be finite")
    return f


def _validate_run_id(value: object) -> str:
    if isinstance(value, bool):
        raise TypeError("run_id must be str, got bool")
    if not isinstance(value, str):
        raise TypeError(
            f"run_id must be str, got {type(value).__name__}"
        )
    if not _RUN_ID_RE.match(value):
        raise ValueError(
            "run_id must be alphanumeric + '_-' (1-128 chars)"
        )
    return value


def _validate_thresholds(value: object) -> GateThresholds:
    if isinstance(value, GateThresholds):
        return value
    raise TypeError("thresholds must be a GateThresholds")


def _validate_bootstrap_samples(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("n_samples must be int, got bool")
    if not isinstance(value, int):
        raise TypeError(
            f"n_samples must be int, got {type(value).__name__}"
        )
    if value < _MIN_BOOTSTRAP_SAMPLES or value > _MAX_BOOTSTRAP_SAMPLES:
        raise ValueError(
            f"n_samples must be in "
            f"[{_MIN_BOOTSTRAP_SAMPLES}, {_MAX_BOOTSTRAP_SAMPLES}]"
        )
    return value


def _validate_ci_level(value: object) -> float:
    f = _require_finite(value, field_name="ci_level")
    if f <= 0.0 or f >= 1.0:
        raise ValueError("ci_level must be in (0.0, 1.0)")
    return f


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    n_samples: int = _DEFAULT_BOOTSTRAP_SAMPLES,
    ci_level: float = _DEFAULT_CI_LEVEL,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired-bootstrap (lower, upper, mean) of ``candidate - baseline``.

    Standard paired-sample bootstrap with replacement at the row level
    (preserves correlation between baseline/candidate). Deterministic
    given ``seed``.
    """
    if isinstance(baseline, (str, bytes)) or not isinstance(baseline, Sequence):
        raise TypeError("baseline must be a sequence of floats")
    if isinstance(candidate, (str, bytes)) or not isinstance(candidate, Sequence):
        raise TypeError("candidate must be a sequence of floats")
    if len(baseline) != len(candidate):
        raise ValueError(
            f"baseline ({len(baseline)}) and candidate "
            f"({len(candidate)}) lengths must match"
        )
    if len(baseline) == 0:
        raise ValueError("baseline must be non-empty")
    base_floats = [
        _require_finite(v, field_name="baseline[i]") for v in baseline
    ]
    cand_floats = [
        _require_finite(v, field_name="candidate[i]") for v in candidate
    ]
    n_samples = _validate_bootstrap_samples(n_samples)
    ci_level = _validate_ci_level(ci_level)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be int")
    if seed < 0 or seed > 2**31 - 1:
        raise ValueError("seed must be non-negative int < 2**31")

    rng = random.Random(seed)
    n = len(base_floats)
    deltas = [c - b for b, c in zip(base_floats, cand_floats)]
    mean_delta = sum(deltas) / n
    means: list = []
    for _ in range(n_samples):
        sample_sum = 0.0
        for _ in range(n):
            idx = rng.randrange(n)
            sample_sum += deltas[idx]
        means.append(sample_sum / n)
    means.sort()
    alpha = (1.0 - ci_level) / 2.0
    lo_idx = max(0, int(alpha * n_samples))
    hi_idx = min(n_samples - 1, int((1.0 - alpha) * n_samples))
    return means[lo_idx], means[hi_idx], mean_delta


# ---------------------------------------------------------------------------
# Regression decision
# ---------------------------------------------------------------------------

# Mapping: metric → (tolerance attr name, direction).
# direction = +1 means "higher is better" (regression when ci_upper < tol)
# direction = -1 means "lower is better"  (regression when ci_lower > tol)
_METRIC_DIRECTION: Mapping[str, int] = types.MappingProxyType(
    {
        "task_accuracy": +1,
        "refusal_rate": +1,
        "format_validity": +1,
        "p95_latency_ms": -1,
    }
)


def decide_regression(
    metric: str,
    baseline: Sequence[float],
    candidate: Sequence[float],
    thresholds: GateThresholds,
    *,
    n_samples: int = _DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> RegressionVerdict:
    """Decide whether ``metric`` regressed past the configured tolerance.

    Uses the paired-bootstrap 95 % CI of the delta. Higher-is-better
    metrics regress when the *upper* CI bound is still worse than the
    tolerance; lower-is-better metrics regress when the *lower* CI
    bound is still worse.
    """
    if isinstance(metric, bool) or not isinstance(metric, str):
        raise TypeError("metric must be str")
    if metric not in _METRIC_DIRECTION:
        raise ValueError(
            f"unknown metric {metric!r}; allowed: "
            + ", ".join(sorted(_METRIC_DIRECTION))
        )
    _validate_thresholds(thresholds)
    tol = getattr(thresholds, metric)
    direction = _METRIC_DIRECTION[metric]
    lo, hi, mean = paired_bootstrap_ci(
        baseline, candidate, n_samples=n_samples, seed=seed
    )
    regressed = False
    if direction > 0:
        # higher-is-better metric: regression iff the upper CI bound is
        # *still* below the tolerance (i.e. even the optimistic estimate
        # is bad).
        regressed = hi < tol
    else:
        # lower-is-better metric: regression iff the lower CI bound is
        # still above the tolerance (i.e. even the pessimistic estimate
        # is bad).
        regressed = lo > tol
    return RegressionVerdict(
        regressed=regressed,
        offenders=(metric,) if regressed else (),
        ci_lower=lo,
        ci_upper=hi,
        delta_mean=mean,
    )


# ---------------------------------------------------------------------------
# Hook script rendering
# ---------------------------------------------------------------------------

_HOOK_TEMPLATE = """#!/usr/bin/env bash
# Generated by `soup eval gate-install` (v0.55.0) — do not edit by hand.
# Pre-push regression gate: blocks the push when `soup eval against`
# detects a regression vs the baseline run id.
set -euo pipefail

BASELINE_RUN_ID={baseline_run_id}
GATE_SUITE={gate_suite}

CANDIDATE_RUN_ID="${{SOUP_CANDIDATE_RUN_ID:-}}"
if [ -z "$CANDIDATE_RUN_ID" ]; then
    echo "[soup] SOUP_CANDIDATE_RUN_ID not set; skipping regression gate." >&2
    exit 0
fi

soup eval against "$BASELINE_RUN_ID" --candidate "$CANDIDATE_RUN_ID" --json-only \\
    || {{
        echo "[soup] pre-push gate blocked: regression vs $BASELINE_RUN_ID" >&2
        exit 1
    }}

exit 0
"""


def _safe_shell_quote(value: str) -> str:
    """Wrapper around ``shlex.quote`` with a control-char rejection prelude.

    Project security policy mandates ``shlex.quote`` for shell-script
    generation. The control-char guard is defence-in-depth — validated
    callers already reject NUL / newline / tab, but the helper itself
    must remain safe to call on raw user-controlled strings.
    """
    if any(ord(ch) < 0x20 for ch in value):
        raise ValueError("value contains control characters")
    return shlex.quote(value)


def render_pre_push_hook(
    *,
    baseline_run_id: str,
    suite_path: str,
) -> str:
    """Render the pre-push hook script body — no I/O, deterministic."""
    rid = _validate_run_id(baseline_run_id)
    if isinstance(suite_path, bool) or not isinstance(suite_path, str):
        raise TypeError("suite_path must be str")
    if not suite_path:
        raise ValueError("suite_path must be non-empty")
    if "\x00" in suite_path:
        raise ValueError("suite_path must not contain NUL")
    if "\n" in suite_path or "\r" in suite_path:
        raise ValueError("suite_path must be a single line")
    if len(suite_path) > 4096:
        raise ValueError("suite_path exceeds 4096 characters")
    if not is_under_cwd(suite_path):
        raise ValueError("suite_path must stay under cwd")
    return _HOOK_TEMPLATE.format(
        baseline_run_id=_safe_shell_quote(rid),
        gate_suite=_safe_shell_quote(suite_path),
    )


def write_pre_push_hook(
    *,
    baseline_run_id: str,
    suite_path: str,
    hook_path: str = ".git/hooks/pre-push",
    overwrite: bool = False,
) -> str:
    """Write the rendered hook to ``hook_path`` with cwd + TOCTOU guards.

    Returns the path written. Refuses to overwrite an existing file
    unless ``overwrite=True``.
    """
    body = render_pre_push_hook(
        baseline_run_id=baseline_run_id, suite_path=suite_path
    )
    if isinstance(hook_path, bool) or not isinstance(hook_path, str):
        raise TypeError("hook_path must be str")
    if not hook_path:
        raise ValueError("hook_path must be non-empty")
    # Single explicit bool guard: bool is a subclass of int, so an
    # ``isinstance(..., bool)`` check is the only one that distinguishes
    # ``True``/``False`` from ``1``/``"yes"``/etc. (review fix —
    # eliminates the previous redundant double-branch).
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be bool")
    # Cwd containment for the destination — operators may also pass
    # ``.git/hooks/pre-push`` so we go through the shared helper.
    if "\x00" in hook_path:
        raise ValueError("hook_path must not contain NUL")
    if not is_under_cwd(hook_path):
        raise ValueError("hook_path must stay under cwd")
    if os.path.lexists(hook_path):
        try:
            st = os.lstat(hook_path)
        except OSError as exc:
            raise ValueError(
                f"hook_path unreadable: {type(exc).__name__}"
            ) from exc
        if stat.S_ISLNK(st.st_mode):
            raise ValueError(
                "hook_path must not be a symlink (TOCTOU defence)"
            )
        if not overwrite:
            raise ValueError(
                "hook already exists; pass overwrite=True to replace it"
            )
    if len(body.encode("utf-8")) > _MAX_FILE_BYTES:
        raise ValueError("rendered hook exceeds 64 KiB cap")
    parent = os.path.dirname(os.path.abspath(hook_path)) or "."
    os.makedirs(parent, exist_ok=True)
    # Atomic write — same idiom as the rest of v0.55.0.
    fd, tmp = tempfile.mkstemp(prefix=".soup-pre-push.", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(body)
        os.replace(tmp, hook_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    # POSIX executable bit so git can launch the hook directly.
    if os.name == "posix":
        try:
            mode = os.stat(hook_path).st_mode
            os.chmod(hook_path, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except OSError:
            pass
    return hook_path
