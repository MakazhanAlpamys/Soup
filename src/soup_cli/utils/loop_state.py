"""Loop state file (v0.58.0 Part A — control plane).

`soup loop` orchestrates the *production traces → preference pairs →
Eval-Gated DPO → canary deploy → rollback* cycle. State for the whole
loop lives in a single ``.soup/loop.yaml`` next to the project, with
atomic writes + cwd containment + symlink rejection — the same TOCTOU
policy as every other v0.5x persistence surface (v0.33.0 #22 /
v0.43.0 Part C / v0.53.7 #106).

Status (``running`` / ``paused`` / ``stopped``) and counters (traces /
pairs / runs / deploys) live here; per-iteration artifacts ship as
v0.26.0 Soup Cans under ``.soup-loops/``.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from soup_cli.utils.paths import is_under_cwd

# Status values are deliberately closed — every state machine transition
# below must remain auditable.
LOOP_STATUSES: frozenset = frozenset({"running", "paused", "stopped"})

_MAX_PATH_LEN = 4096
_MAX_STR_FIELD = 512
_MAX_FILE_BYTES = 1 * 1024 * 1024  # 1 MiB cap on the state file
_DEFAULT_STATE_DIR = ".soup"
_DEFAULT_STATE_FILENAME = "loop.yaml"


@dataclass(frozen=True)
class LoopState:
    """Immutable snapshot of a `soup loop` configuration + counters.

    The persisted file is JSON-formatted (despite the ``.yaml`` extension)
    so we can use the stdlib parser without pulling pyyaml into the read
    path; YAML is a superset of JSON for objects and the file remains
    human-readable. Counters are absolute lifetime totals; per-iteration
    detail lives in the ``.soup-loops/`` Soup Can artifacts.
    """

    served_model: str
    eval_suite: str
    baseline: str
    status: str = "stopped"
    # v0.71.4 #176 — use the pre-wired harvest/train/gate/deploy stages
    # (utils/loop_stages.py) instead of the no-op default callbacks.
    pre_wired: bool = False
    traces_collected: int = 0
    pairs_distilled: int = 0
    runs_gated: int = 0
    adapters_shipped: int = 0
    canary_active: Optional[str] = None
    canary_traffic_pct: Optional[float] = None
    canary_autoroll_on_regress: bool = True
    monthly_budget_usd: Optional[float] = None
    spent_this_month_usd: float = 0.0
    max_runs_per_day: Optional[int] = None
    runs_today: int = 0
    last_run_date: Optional[str] = None  # YYYY-MM-DD UTC
    last_iteration_id: Optional[str] = None
    iteration_count: int = 0
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:  # noqa: D401 — validator hook
        # Closed-allowlist enforcement + bool-as-int rejection mirror the
        # project's v0.30/v0.34/v0.50 policy. `replace(...)` (immutable)
        # is the only sanctioned mutation path; this validator runs at
        # construction time so a hand-rolled instance still gets checks.
        _require_str("served_model", self.served_model)
        _require_str("eval_suite", self.eval_suite)
        _require_str("baseline", self.baseline)
        if self.status not in LOOP_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(LOOP_STATUSES)}, got {self.status!r}"
            )
        for fname in (
            "traces_collected",
            "pairs_distilled",
            "runs_gated",
            "adapters_shipped",
            "iteration_count",
            "runs_today",
        ):
            v = getattr(self, fname)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f"{fname} must be a non-negative int, got {v!r}")
        if self.canary_traffic_pct is not None:
            v = self.canary_traffic_pct
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError("canary_traffic_pct must be numeric or None")
            if not (0.0 <= float(v) <= 100.0):
                raise ValueError("canary_traffic_pct must be in [0, 100]")
        if self.monthly_budget_usd is not None:
            v = self.monthly_budget_usd
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
                raise ValueError("monthly_budget_usd must be >= 0 or None")
        v = self.spent_this_month_usd
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
            raise ValueError("spent_this_month_usd must be >= 0")
        if self.max_runs_per_day is not None:
            v = self.max_runs_per_day
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError("max_runs_per_day must be a positive int or None")
        if not isinstance(self.canary_autoroll_on_regress, bool):
            raise ValueError("canary_autoroll_on_regress must be bool")
        if not isinstance(self.pre_wired, bool):
            raise ValueError("pre_wired must be bool")
        if self.canary_active is not None:
            _require_str("canary_active", self.canary_active, allow_empty=False)
        if self.last_iteration_id is not None:
            _require_str("last_iteration_id", self.last_iteration_id, allow_empty=False)
        if self.last_run_date is not None:
            _require_str("last_run_date", self.last_run_date, allow_empty=False)

    def to_dict(self) -> Mapping[str, object]:
        """Stable, JSON-serialisable view (returned as ``MappingProxyType``)."""
        return MappingProxyType(asdict(self))

    def with_status(self, status: str) -> "LoopState":
        """Return a copy with ``status`` set + ``updated_at`` refreshed."""
        if status not in LOOP_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(LOOP_STATUSES)}, got {status!r}"
            )
        return replace(self, status=status, updated_at=_utc_now_iso())

    def bumped(self, **counters: int) -> "LoopState":
        """Return a copy with named counters incremented atomically."""
        updates = {}
        for k, v in counters.items():
            if k not in {
                "traces_collected",
                "pairs_distilled",
                "runs_gated",
                "adapters_shipped",
                "iteration_count",
                "runs_today",
            }:
                raise ValueError(f"unknown counter: {k}")
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f"{k} delta must be a non-negative int")
            current = getattr(self, k)
            updates[k] = current + v
        updates["updated_at"] = _utc_now_iso()
        return replace(self, **updates)


def _require_str(field: str, value: object, *, allow_empty: bool = False) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str, got {type(value).__name__}")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain NUL")
    if not allow_empty and not value:
        raise ValueError(f"{field} must not be empty")
    if len(value) > _MAX_STR_FIELD:
        raise ValueError(f"{field} exceeds {_MAX_STR_FIELD} characters")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_state_path(cwd: Optional[str] = None) -> str:
    """Return the canonical state-file path under cwd."""
    base = cwd if cwd is not None else os.getcwd()
    return os.path.join(base, _DEFAULT_STATE_DIR, _DEFAULT_STATE_FILENAME)


def _check_path(path: str, *, allow_missing: bool) -> str:
    if not isinstance(path, str):
        raise TypeError(f"path must be str, got {type(path).__name__}")
    if not path:
        raise ValueError("path must not be empty")
    if "\x00" in path:
        raise ValueError("path must not contain NUL")
    if len(path) > _MAX_PATH_LEN:
        raise ValueError(f"path exceeds {_MAX_PATH_LEN} characters")
    if not is_under_cwd(path):
        raise ValueError(f"path {os.path.basename(path)!r} must stay under cwd")
    # Direct lstat — no lexists guard — closes the TOCTOU window where a
    # symlink could be planted between the existence check and the stat.
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        if not allow_missing:
            raise FileNotFoundError(
                f"state file not found: {os.path.basename(path)}"
            ) from None
        return path
    except OSError as exc:
        raise ValueError(f"path unreadable: {type(exc).__name__}") from exc
    if stat.S_ISLNK(st.st_mode):
        raise ValueError("path must not be a symlink (TOCTOU defence)")
    return path


def write_state(state: LoopState, path: Optional[str] = None) -> str:
    """Atomically persist ``state`` to ``path`` (default: ``./.soup/loop.yaml``).

    Uses ``tempfile.mkstemp`` + ``os.replace`` so a SIGKILL mid-write
    cannot leave a torn file at the target — matches v0.43.0 Part D
    / v0.55.0 ``lock_suite`` / v0.57.0 atomic-write policy.
    """
    if not isinstance(state, LoopState):
        raise TypeError(f"state must be LoopState, got {type(state).__name__}")
    target = path if path is not None else default_state_path()
    _check_path(target, allow_missing=True)
    parent = os.path.dirname(target) or "."
    os.makedirs(parent, exist_ok=True)
    payload = dict(state.to_dict())
    if not payload.get("created_at"):
        payload["created_at"] = _utc_now_iso()
    payload["updated_at"] = _utc_now_iso()
    body = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True).encode("utf-8")
    if len(body) > _MAX_FILE_BYTES:
        raise ValueError("state payload exceeds 1 MiB cap")
    fd, tmp = tempfile.mkstemp(prefix=".loop_state_", dir=parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(body)
        os.replace(tmp, target)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    try:
        if os.name == "posix":
            os.chmod(target, 0o600)
    except OSError:
        pass
    return target


def read_state(path: Optional[str] = None) -> LoopState:
    """Load a ``LoopState`` from disk."""
    target = path if path is not None else default_state_path()
    _check_path(target, allow_missing=False)
    try:
        size = os.path.getsize(target)
    except OSError as exc:
        raise ValueError(f"state file unreadable: {type(exc).__name__}") from exc
    if size > _MAX_FILE_BYTES:
        raise ValueError("state file exceeds 1 MiB cap")
    with open(target, "rb") as fh:
        raw = fh.read()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"state file is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("state file root must be a JSON object")
    # Drop keys we don't recognise so a forward-compat dump can still load
    # backwards (the alternative is mass-rejecting on any unknown field).
    allowed = set(LoopState.__dataclass_fields__.keys())
    filtered = {k: v for k, v in data.items() if k in allowed}
    try:
        return LoopState(**filtered)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"state file has invalid contents: {exc}") from exc


def init_state(
    served_model: str,
    eval_suite: str,
    baseline: str,
    *,
    monthly_budget_usd: Optional[float] = None,
    max_runs_per_day: Optional[int] = None,
    pre_wired: bool = False,
    path: Optional[str] = None,
    force: bool = False,
) -> Tuple[LoopState, str]:
    """Create the loop.yaml state file. Refuses to clobber unless ``force``."""
    target = path if path is not None else default_state_path()
    _check_path(target, allow_missing=True)
    # Use lstat (not exists) — defends against a planted symlink between
    # the existence check and the write. `_check_path` already rejected
    # symlinks so any stat-success here is a real regular-file collision.
    try:
        os.lstat(target)
        present = True
    except FileNotFoundError:
        present = False
    except OSError as exc:
        raise ValueError(f"state path unreadable: {type(exc).__name__}") from exc
    if present and not force:
        raise FileExistsError(
            f"loop state already exists at {os.path.basename(target)} "
            "(re-run with --force to overwrite)"
        )
    now = _utc_now_iso()
    state = LoopState(
        served_model=served_model,
        eval_suite=eval_suite,
        baseline=baseline,
        status="stopped",
        pre_wired=pre_wired,
        monthly_budget_usd=monthly_budget_usd,
        max_runs_per_day=max_runs_per_day,
        created_at=now,
        updated_at=now,
    )
    write_state(state, target)
    return state, target
