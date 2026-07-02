"""`soup plan` / `soup apply` — Terraform-shape lock-and-execute for FT.

A training run is a one-shot infrastructure-shaped operation: spot
price, expected cost, base SHA, dataset SHA, peak VRAM. v0.64 borrows
Terraform's plan-apply split so operators can review what they're
about to spend before committing.

Workflow:
1. ``soup plan --config soup.yaml`` writes ``soup.tfstate`` with a
   ``TrainingPlan`` summarising the run (config SHA, dataset SHA,
   estimated cost, ETA, peak VRAM, spot price).
2. ``soup apply --config soup.yaml`` re-builds the plan from the
   current YAML, compares against the state file, and **refuses** if
   the plan drifted. Operators see "config drifted: epochs 1 -> 99"
   instead of silently spending another $0.50.
3. ``--dry-run`` exits 0 after the drift check without actually
   invoking the trainer.

The state file is a thin JSON envelope; the actual ``soup train`` is
still the run-driver. This is a *gate*, not a parallel trainer.

Public surface:
- ``TrainingPlan`` frozen dataclass (base / task / config_sha /
  dataset_sha / cost / ETA / peak VRAM / spot price).
- ``TrainingState`` frozen dataclass (plan + applied flag + run_id +
  applied_at timestamp).
- ``DriftReport`` frozen dataclass with the changed-fields tuple.
- ``compute_config_sha(config)`` -> 64-hex SHA-256 of canonical JSON.
- ``compute_dataset_sha(path)`` -> 64-hex SHA-256 of the JSONL bytes.
- ``build_plan(config)`` -> TrainingPlan.
- ``write_state(state, path)`` / ``read_state(path)`` -> atomic JSON.
- ``detect_drift(state, plan_now)`` -> DriftReport.
- ``DEFAULT_STATE_FILE = "soup.tfstate"``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

from soup_cli.utils.paths import atomic_write_text, is_under_cwd

DEFAULT_STATE_FILE = "soup.tfstate"
_SHA_REGEX_LEN = 64
_MAX_BASE_LEN = 512


@dataclass(frozen=True)
class TrainingPlan:
    """Pre-flight summary of a planned `soup train` invocation."""

    base: str
    task: str
    config_sha: str
    dataset_sha: str
    estimated_cost_usd: float
    estimated_minutes: float
    peak_vram_gb: float
    spot_price_usd_per_hour: float

    def __post_init__(self) -> None:
        _check_non_empty_str(self.base, "base", max_len=_MAX_BASE_LEN)
        _check_non_empty_str(self.task, "task", max_len=64)
        _check_sha(self.config_sha, "config_sha")
        _check_sha(self.dataset_sha, "dataset_sha")
        _check_non_negative_finite(self.estimated_cost_usd, "estimated_cost_usd")
        _check_non_negative_finite(self.estimated_minutes, "estimated_minutes")
        _check_non_negative_finite(self.peak_vram_gb, "peak_vram_gb")
        _check_non_negative_finite(self.spot_price_usd_per_hour, "spot_price_usd_per_hour")


def _check_non_empty_str(value: object, fld: str, *, max_len: int) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{fld} must be str, not bool")
    if not isinstance(value, str):
        raise TypeError(f"{fld} must be str, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{fld} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{fld} must not contain null bytes")
    if len(value) > max_len:
        raise ValueError(f"{fld} too long (> {max_len} chars)")


def _check_sha(value: object, fld: str) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{fld} must be str, not bool")
    if not isinstance(value, str):
        raise TypeError(f"{fld} must be str, got {type(value).__name__}")
    if len(value) != _SHA_REGEX_LEN:
        raise ValueError(
            f"{fld} sha must be {_SHA_REGEX_LEN} hex chars, "
            f"got len={len(value)}"
        )
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{fld} sha must be 64-char hex, got {value!r}") from exc


def _check_non_negative_finite(value: object, fld: str) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{fld} must be a number, not bool")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{fld} must be a number, got {type(value).__name__}")
    if not math.isfinite(float(value)):
        raise ValueError(f"{fld} must be finite (no NaN / Inf)")
    if value < 0:
        raise ValueError(f"{fld} must not be negative, got {value}")


@dataclass(frozen=True)
class TrainingState:
    """Persisted plan + apply metadata. Written as ``soup.tfstate``."""

    plan: TrainingPlan
    applied: bool
    applied_at: Optional[str]
    run_id: Optional[str]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, TrainingPlan):
            raise TypeError(
                f"plan must be TrainingPlan, got {type(self.plan).__name__}"
            )
        if not isinstance(self.applied, bool):
            raise TypeError("applied must be bool")
        if self.applied_at is not None and not isinstance(self.applied_at, str):
            raise TypeError("applied_at must be str | None")
        if self.run_id is not None and not isinstance(self.run_id, str):
            raise TypeError("run_id must be str | None")


@dataclass(frozen=True)
class DriftReport:
    """Outcome of a drift comparison between state.plan and a fresh plan."""

    has_drift: bool
    changed_fields: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.has_drift, bool):
            raise TypeError("has_drift must be bool")
        if not isinstance(self.changed_fields, tuple):
            raise TypeError("changed_fields must be a tuple of str")
        for entry in self.changed_fields:
            if not isinstance(entry, str):
                raise TypeError("changed_fields entries must be str")


def compute_config_sha(config: Mapping[str, Any]) -> str:
    """Canonical SHA-256 of the config dict (insertion-order-independent).

    Uses strict JSON serialisation (no ``default=`` fallback) so two
    configs that differ in a non-serialisable field cannot silently
    collide. Operators with custom-typed values must pre-canonicalise
    via ``dict(yaml.safe_load(...))`` first.
    """
    if not isinstance(config, Mapping):
        raise TypeError(f"config must be a mapping, got {type(config).__name__}")
    blob = json.dumps(
        config, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def compute_dataset_sha(path: object) -> str:
    """SHA-256 of dataset file bytes. Returns zero-hash on missing file.

    Returning a constant for "missing" lets ``soup plan`` run before the
    dataset exists (e.g. dataset built by an earlier `soup data` step).
    Drift detection still surfaces the change once the file appears.

    Security: cwd containment + symlink rejection BEFORE the open() so a
    crafted ``soup.yaml`` with ``data.train: /etc/shadow`` cannot leak the
    file contents into the SHA. An empty path returns zero-hash without
    touching the filesystem.
    """
    import stat as _stat

    if not isinstance(path, str):
        raise TypeError(f"dataset path must be str, got {type(path).__name__}")
    if "\x00" in path:
        raise ValueError("dataset path must not contain null bytes")
    if not path:
        return "0" * _SHA_REGEX_LEN
    if not is_under_cwd(path):
        # Out-of-cwd is treated as "no dataset captured" rather than a
        # hard error so plan can still render — drift detection will
        # surface this if the operator later moves the dataset in-tree.
        return "0" * _SHA_REGEX_LEN
    if os.path.lexists(path):
        try:
            st = os.lstat(path)
        except OSError:
            return "0" * _SHA_REGEX_LEN
        if _stat.S_ISLNK(st.st_mode):
            raise ValueError("dataset path must not be a symlink")
    if not os.path.isfile(path):
        return "0" * _SHA_REGEX_LEN
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# Approximate cost model — pure heuristic so the plan is honest about
# being a pre-flight estimate. Live measurement is whatever the trainer
# emits at finish time (v0.34 cost path).
_DEFAULT_SPOT_PRICE = 0.30  # ~$/hr for a 24 GB consumer GPU
_DEFAULT_PEAK_VRAM = 8.0


def _to_float(value: Any, default: float) -> float:
    """Coerce a config number to float, tolerating ``batch_size: "auto"``.

    ``soup train`` accepts ``batch_size: "auto"`` (int-or-"auto" per schema), so
    the plan estimator must not crash on ``float("auto")`` — fall back to the
    baseline for the heuristic instead.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _estimate_runtime_minutes(config: Mapping[str, Any]) -> float:
    training = config.get("training", {}) if isinstance(config, Mapping) else {}
    if not isinstance(training, Mapping):
        training = {}
    epochs = _to_float(training.get("epochs", 1), 1.0)
    batch_size = _to_float(training.get("batch_size", 4), 4.0)
    # Soft heuristic: 5 minutes per epoch at batch_size=4 baseline.
    base = 5.0 * epochs
    if batch_size > 0:
        base *= 4.0 / max(batch_size, 1.0)
    return max(0.5, base)


def _estimate_cost(minutes: float, spot_price: float) -> float:
    return max(0.0, (minutes / 60.0) * spot_price)


def build_plan(config: Mapping[str, Any]) -> TrainingPlan:
    """Render a TrainingPlan from a soup-config-shaped dict."""
    if not isinstance(config, Mapping):
        raise TypeError(f"config must be a mapping, got {type(config).__name__}")
    base = config.get("base")
    if not isinstance(base, str) or not base:
        raise ValueError("config must contain a non-empty `base`")
    task = config.get("task", "sft")
    if not isinstance(task, str) or not task:
        task = "sft"

    data_cfg = config.get("data", {}) if isinstance(config.get("data"), Mapping) else {}
    train_path = data_cfg.get("train", "") if isinstance(data_cfg, Mapping) else ""
    if not isinstance(train_path, str):
        train_path = ""

    config_sha = compute_config_sha(config)
    dataset_sha = compute_dataset_sha(train_path)
    minutes = _estimate_runtime_minutes(config)
    cost = _estimate_cost(minutes, _DEFAULT_SPOT_PRICE)
    return TrainingPlan(
        base=base,
        task=task,
        config_sha=config_sha,
        dataset_sha=dataset_sha,
        estimated_cost_usd=cost,
        estimated_minutes=minutes,
        peak_vram_gb=_DEFAULT_PEAK_VRAM,
        spot_price_usd_per_hour=_DEFAULT_SPOT_PRICE,
    )


def _plan_to_dict(plan: TrainingPlan) -> dict:
    return {
        "base": plan.base,
        "task": plan.task,
        "config_sha": plan.config_sha,
        "dataset_sha": plan.dataset_sha,
        "estimated_cost_usd": plan.estimated_cost_usd,
        "estimated_minutes": plan.estimated_minutes,
        "peak_vram_gb": plan.peak_vram_gb,
        "spot_price_usd_per_hour": plan.spot_price_usd_per_hour,
    }


def _plan_from_dict(d: Mapping[str, Any]) -> TrainingPlan:
    return TrainingPlan(
        base=str(d["base"]),
        task=str(d["task"]),
        config_sha=str(d["config_sha"]),
        dataset_sha=str(d["dataset_sha"]),
        estimated_cost_usd=float(d["estimated_cost_usd"]),
        estimated_minutes=float(d["estimated_minutes"]),
        peak_vram_gb=float(d["peak_vram_gb"]),
        spot_price_usd_per_hour=float(d["spot_price_usd_per_hour"]),
    )


def write_state(state: TrainingState, path: str) -> None:
    """Atomically write a ``TrainingState`` to JSON under cwd containment."""
    if not isinstance(state, TrainingState):
        raise TypeError(
            f"state must be TrainingState, got {type(state).__name__}"
        )
    payload = {
        "schema_version": "1",
        "plan": _plan_to_dict(state.plan),
        "applied": state.applied,
        "applied_at": state.applied_at,
        "run_id": state.run_id,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    atomic_write_text(text, path, prefix=".tfstate.", field="state file")


def read_state(path: str) -> TrainingState:
    """Read a previously-written ``TrainingState`` JSON.

    Containment + symlink rejection BEFORE existence probe (mirrors
    v0.55.0 / v0.62.0 ordering policy).
    """
    import stat as _stat

    if not isinstance(path, str):
        raise TypeError(f"path must be str, got {type(path).__name__}")
    if "\x00" in path:
        raise ValueError("state path must not contain null bytes")
    if not is_under_cwd(path):
        raise ValueError(f"state file {path!r} is outside cwd")
    if os.path.lexists(path):
        st = os.lstat(path)
        if _stat.S_ISLNK(st.st_mode):
            raise ValueError("state path must not be a symlink")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("state root must be a dict")
    plan_raw = payload.get("plan")
    if not isinstance(plan_raw, dict):
        raise ValueError("state.plan must be a dict")
    plan = _plan_from_dict(plan_raw)
    applied_raw = payload.get("applied", False)
    if not isinstance(applied_raw, bool):
        raise ValueError(
            f"state.applied must be bool, got {type(applied_raw).__name__}"
        )
    return TrainingState(
        plan=plan,
        applied=applied_raw,
        applied_at=payload.get("applied_at"),
        run_id=payload.get("run_id"),
    )


def detect_drift(state: TrainingState, plan_now: TrainingPlan) -> DriftReport:
    """Compare ``state.plan`` against a freshly-built plan.

    Returns a ``DriftReport`` listing changed fields. Empty tuple means no
    drift. ``apply`` refuses to proceed when ``has_drift=True``.
    """
    if not isinstance(state, TrainingState):
        raise TypeError(
            f"state must be TrainingState, got {type(state).__name__}"
        )
    if not isinstance(plan_now, TrainingPlan):
        raise TypeError(
            f"plan_now must be TrainingPlan, got {type(plan_now).__name__}"
        )
    changed: list[str] = []
    if state.plan.base != plan_now.base:
        changed.append("base")
    if state.plan.task != plan_now.task:
        changed.append("task")
    if state.plan.config_sha != plan_now.config_sha:
        changed.append("config_sha")
    if state.plan.dataset_sha != plan_now.dataset_sha:
        changed.append("dataset_sha")
    return DriftReport(has_drift=bool(changed), changed_fields=tuple(changed))


__all__ = [
    "DEFAULT_STATE_FILE",
    "DriftReport",
    "TrainingPlan",
    "TrainingState",
    "build_plan",
    "compute_config_sha",
    "compute_dataset_sha",
    "detect_drift",
    "read_state",
    "write_state",
]
