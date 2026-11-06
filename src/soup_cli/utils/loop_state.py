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
import re
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
# The check before rollout (`soup loop init --verify`).
VERIFY_METHODS: tuple = ("ship", "eval_gate")
# The dataset the pre-wired loop trains on (`soup loop init --train-method`).
TRAIN_METHODS: tuple = ("dpo", "kto", "sft")
# What a canary does when the ratings its quality verdict needs have not arrived by the
# maximum wait (`soup loop init --canary-on-timeout`): "errors-only" lets the request-error
# verdict decide, "rollback" withdraws the canary, "wait" never gives up.
CANARY_ON_TIMEOUT: tuple = ("errors-only", "rollback", "wait")
# How long after a rollout started the ratings may stay short (`soup loop init
# --canary-max-wait`): a day by default, at most 30 days.
DEFAULT_CANARY_MAX_WAIT_SEC = 86400.0
MAX_CANARY_MAX_WAIT_SEC = 30 * 86400.0

_MAX_PATH_LEN = 4096
_MAX_STR_FIELD = 512
_MAX_FILE_BYTES = 1 * 1024 * 1024  # 1 MiB cap on the state file
# The data cursor is epoch seconds up to the year 9999 (datetime.max): `soup loop status`
# prints it as a date, and a larger value is a typo that would stop all learning.
MAX_CURSOR_TS = 253402300799.0
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
    canary_rollout_id: Optional[str] = None
    monthly_budget_usd: Optional[float] = None
    spent_this_month_usd: float = 0.0
    max_runs_per_day: Optional[int] = None
    runs_today: int = 0
    last_run_date: Optional[str] = None  # YYYY-MM-DD UTC
    last_iteration_id: Optional[str] = None
    iteration_count: int = 0
    created_at: str = ""
    updated_at: str = ""
    # #1440 — local-RL SQLite that `soup serve --record-thumbs` writes; the
    # pre-wired harvest reads its thumbs as preference pairs. None = not used.
    # Last, so adding it shifted no existing positional field.
    thumbs_db: Optional[str] = None
    # The model the pre-wired train stage starts from: `soup loop init
    # --base-model`, or what init resolved from a registry:// entry. None =
    # resolve at watch time (served_model itself when it is not registry://).
    base_model: Optional[str] = None
    # The directory of conversation logs (`soup serve --trace-log` files,
    # *.jsonl) the pre-wired harvest reads: `soup loop init --trace-dir`.
    # None = .soup-loops/traces. The served model path is never read as one.
    trace_dir: Optional[str] = None
    # The check before rollout: "ship" (a soup ship verdict against the
    # production version) or "eval_gate". None = a loop.yaml written before
    # this field existed; those loops ran the eval gate and still do.
    verify: Optional[str] = None
    # Adapter of the version serving as production: the "before" side of the
    # ship verdict. None = the base model itself.
    production_adapter: Optional[str] = None
    # Share of keyed traffic a newly deployed canary gets (`soup loop init
    # --canary-traffic`, default 5%).
    deploy_traffic_pct: float = 5.0
    # Path of the adapter now in canary; it becomes production_adapter on promotion.
    canary_adapter: Optional[str] = None
    # Served adapter name of the production version: the canary router's stable
    # side. None = the served model itself (nothing promoted yet).
    production_name: Optional[str] = None
    # Production before the last promotion; `soup loop rollback` returns to it
    # (previous_production_name None = the served model itself).
    previous_production_name: Optional[str] = None
    previous_production_adapter: Optional[str] = None
    # The data cursor: the newest feedback timestamp (epoch seconds) a training run
    # has already seen; None = no training run yet, so every pair is new. It is a
    # trigger only (training waits for `min_new_pairs` pairs newer than it); the
    # run still trains on all the pairs, as each iteration trains a fresh LoRA from
    # the base model. The daemon advances it after a training run, never on a skip.
    harvest_cursor_ts: Optional[float] = None
    # Training waits until this many preference pairs are newer than the cursor
    # (`soup loop init --min-new-pairs`).
    min_new_pairs: int = 1
    # The feedback log (feedback.jsonl lines: thumbs, ratings, edits, comments on logged
    # answers) the pre-wired harvest reads: `soup loop init --feedback-log`. None =
    # .soup-loops/feedback.jsonl; SOUP_LOOP_FEEDBACK_LOG overrides both.
    feedback_log: Optional[str] = None
    # What the pre-wired loop trains: "dpo" (preference pairs, the default), "kto"
    # (rows labelled good or bad) or "sft" (the answers people corrected).
    # `min_new_pairs` counts new rows of this method's dataset.
    train_method: str = "dpo"
    # When the active rollout started (epoch seconds): the deploy stage, or `soup loop
    # canary`, writes it with canary_active. The quality verdict counts only answers given
    # from then on. None = no canary, or one started before this field existed (it is then
    # judged by request errors alone).
    canary_started_ts: Optional[float] = None
    # How long (seconds) after a rollout started its quality verdict may stay short of
    # ratings before `canary_on_timeout` applies (`soup loop init --canary-max-wait`).
    # Last, so adding it shifted no existing positional field.
    canary_max_wait_sec: float = DEFAULT_CANARY_MAX_WAIT_SEC
    # What happens then: "errors-only", "rollback" or "wait" (CANARY_ON_TIMEOUT).
    canary_on_timeout: str = "errors-only"
    # The version counter of the loop's version log (.soup-loops/versions.jsonl, see
    # docs/loop-formats.md): 1 is the served model, each promotion adds 1 and takes the next
    # number after `latest_version`, so a number is never reused after a rollback. Last, so
    # adding them shifted no existing positional field; a loop.yaml from before them is
    # numbered by `read_state` from the names it remembers.
    production_version: int = 1
    # Production's version before the last promotion: what `soup loop rollback` returns to
    # (None = nothing to go back to, or the served model when previous_production_name is None).
    previous_production_version: Optional[int] = None
    # The highest version ever promoted.
    latest_version: int = 1
    # The iteration that produced production_version, and the one before it (None = the
    # served model, an iteration this log does not know, or no log yet).
    production_iteration_id: Optional[str] = None
    previous_production_iteration_id: Optional[str] = None
    # The last line written to iterations.jsonl for the iteration whose canary is running (an
    # iteration line, outcome "canary"): the promotion or withdrawal appends the same snapshot
    # with its new outcome, so the later line is complete. None = no canary, or one that no
    # iteration started (`soup loop canary`, or a rollout from before this field).
    canary_iteration: Optional[dict] = None
    # True once the first version (the "initial" line) is in the version log.
    version_log_started: bool = False

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
        if self.thumbs_db is not None:
            _require_str("thumbs_db", self.thumbs_db, allow_empty=False)
        if self.base_model is not None:
            _require_str("base_model", self.base_model, allow_empty=False)
        if self.trace_dir is not None:
            _require_str("trace_dir", self.trace_dir, allow_empty=False)
        if self.verify is not None and self.verify not in VERIFY_METHODS:
            raise ValueError(
                f"verify must be one of {list(VERIFY_METHODS)}, got {self.verify!r}"
            )
        if self.production_adapter is not None:
            _require_str("production_adapter", self.production_adapter, allow_empty=False)
        v = self.deploy_traffic_pct
        if (
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not (0.0 < float(v) <= 100.0)
        ):
            raise ValueError("deploy_traffic_pct must be in (0, 100]")
        for fname in (
            "canary_adapter",
            "production_name",
            "previous_production_name",
            "previous_production_adapter",
        ):
            if getattr(self, fname) is not None:
                _require_str(fname, getattr(self, fname), allow_empty=False)
        if self.harvest_cursor_ts is not None:
            v = self.harvest_cursor_ts
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not (0 <= v <= MAX_CURSOR_TS)  # False for NaN too
            ):
                raise ValueError(
                    "harvest_cursor_ts must be epoch seconds in "
                    f"[0, {int(MAX_CURSOR_TS)}] or None, got {v!r}"
                )
        v = self.min_new_pairs
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"min_new_pairs must be a positive int, got {v!r}")
        if self.feedback_log is not None:
            _require_str("feedback_log", self.feedback_log, allow_empty=False)
        if self.train_method not in TRAIN_METHODS:
            raise ValueError(
                f"train_method must be one of {list(TRAIN_METHODS)}, got {self.train_method!r}"
            )
        if self.canary_started_ts is not None:
            v = self.canary_started_ts
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not (0 <= v <= MAX_CURSOR_TS)  # False for NaN too
            ):
                raise ValueError(
                    "canary_started_ts must be epoch seconds in "
                    f"[0, {int(MAX_CURSOR_TS)}] or None, got {v!r}"
                )
        v = self.canary_max_wait_sec
        if (
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not (0 < v <= MAX_CANARY_MAX_WAIT_SEC)  # False for NaN too
        ):
            raise ValueError(
                "canary_max_wait_sec must be seconds in "
                f"(0, {int(MAX_CANARY_MAX_WAIT_SEC)}], got {v!r}"
            )
        if self.canary_on_timeout not in CANARY_ON_TIMEOUT:
            raise ValueError(
                f"canary_on_timeout must be one of {list(CANARY_ON_TIMEOUT)}, "
                f"got {self.canary_on_timeout!r}"
            )
        if self.canary_rollout_id is not None:
            _require_str("canary_rollout_id", self.canary_rollout_id, allow_empty=False)
        if self.last_iteration_id is not None:
            _require_str("last_iteration_id", self.last_iteration_id, allow_empty=False)
        if self.last_run_date is not None:
            _require_str("last_run_date", self.last_run_date, allow_empty=False)
        self._validate_versions()

    def _validate_versions(self) -> None:
        """The version counter, the iteration ids and the canary's iteration snapshot."""
        for fname in ("production_version", "latest_version"):
            v = getattr(self, fname)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{fname} must be a positive int, got {v!r}")
        v = self.previous_production_version
        if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 1):
            raise ValueError(
                f"previous_production_version must be a positive int or None, got {v!r}"
            )
        behind = [self.production_version]
        if self.previous_production_version is not None:
            behind.append(self.previous_production_version)
        if self.latest_version < max(behind):
            raise ValueError(
                f"latest_version must be at least {max(behind)} (the highest version in "
                f"use), got {self.latest_version}"
            )
        for fname in ("production_iteration_id", "previous_production_iteration_id"):
            if getattr(self, fname) is not None:
                _require_iteration_id(fname, getattr(self, fname))
        if not isinstance(self.version_log_started, bool):
            raise ValueError("version_log_started must be bool")
        if self.canary_iteration is not None:
            _check_canary_iteration(self.canary_iteration)

    def stable_name(self) -> str:
        """The canary router's stable side: the promoted adapter, else the served model."""
        return self.production_name or self.served_model

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


_ITERATION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", re.ASCII)
#: The snapshot of a canary's iteration is one line of iterations.jsonl; far under this.
_MAX_CANARY_ITERATION_CHARS = 32 * 1024


def _require_iteration_id(field: str, value: object) -> None:
    """``value`` is an iteration id: safe as a directory name, as the formats require."""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str, got {type(value).__name__}")
    if _ITERATION_ID_RE.fullmatch(value) is None or ".." in value:
        raise ValueError(
            f"{field} must be 1-128 letters, digits, '.', '_' or '-' (not starting with a "
            f"symbol, no '..'), got {value[:40]!r}"
        )


def _check_canary_iteration(snapshot: object) -> None:
    """``canary_iteration`` is plain, small JSON: an iteration line whose outcome is canary."""
    if not isinstance(snapshot, dict):
        raise ValueError("canary_iteration must be a dict or None")
    try:
        body = json.dumps(snapshot, allow_nan=False)
        plain = json.loads(body) == snapshot  # an int key or a tuple is not plain JSON
    except (TypeError, ValueError) as exc:
        raise ValueError("canary_iteration must hold plain finite JSON values") from exc
    if not plain:
        raise ValueError("canary_iteration must hold plain finite JSON values with string keys")
    if len(body) > _MAX_CANARY_ITERATION_CHARS:
        raise ValueError(f"canary_iteration exceeds {_MAX_CANARY_ITERATION_CHARS} characters")
    if "iteration_id" not in snapshot:
        raise ValueError("canary_iteration must name its iteration_id")
    _require_iteration_id("canary_iteration iteration_id", snapshot["iteration_id"])
    if snapshot.get("outcome") != "canary":
        raise ValueError("canary_iteration must be an iteration whose outcome is canary")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


_WAIT_UNITS = {"m": 60, "h": 3600, "d": 86400}
# ASCII digits only: str.isdigit() and \d accept other scripts' digits (and superscripts).
_WAIT_RE = re.compile(r"([0-9]+)([mhd]?)", re.ASCII)
_MAX_WAIT_DIGITS = 12  # more cannot be within the 30-day cap; never build a huge int
_WAIT_FORMS = "use <int>m, <int>h or <int>d (for example 90m, 24h, 7d) or plain seconds"


def parse_wait_duration(raw: object) -> float:
    """Seconds from ``"90m"`` / ``"24h"`` / ``"7d"`` / ``"3600"`` (plain seconds).

    Whole numbers only, an optional surrounding space and any case; zero, a sign, a fraction,
    anything else, and more than 30 days raise ``ValueError`` (the message does not name the
    flag; the command adds it).
    """
    if not isinstance(raw, str):
        raise ValueError(f"{raw!r} is not a valid wait: {_WAIT_FORMS}")
    match = _WAIT_RE.fullmatch(raw.strip().lower())
    if match is None:
        raise ValueError(f"{raw!r} is not a valid wait: {_WAIT_FORMS}")
    digits, unit = match.groups()
    if len(digits.lstrip("0")) > _MAX_WAIT_DIGITS:
        seconds = float("inf")
    else:
        seconds = float(int(digits) * _WAIT_UNITS.get(unit, 1))
    if seconds == 0:
        raise ValueError(f"{raw.strip()!r} must be above 0: {_WAIT_FORMS}")
    if seconds > MAX_CANARY_MAX_WAIT_SEC:
        raise ValueError(f"{raw.strip()!r} must be at most 30d")
    return seconds


def format_wait_duration(seconds: float) -> str:
    """``86400`` -> ``"24h"``: the largest of d (two days or more), h, m that divides it."""
    whole = int(seconds)
    if whole != seconds:
        return f"{seconds:g}s"
    if whole % 86400 == 0 and whole >= 2 * 86400:
        return f"{whole // 86400}d"
    if whole % 3600 == 0:
        return f"{whole // 3600}h"
    if whole % 60 == 0:
        return f"{whole // 60}m"
    return f"{whole}s"


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
    if "production_version" not in data:
        _number_versions_of_old_state(filtered)
    try:
        return LoopState(**filtered)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"state file has invalid contents: {exc}") from exc


def _number_versions_of_old_state(fields: dict) -> None:
    """Give a ``loop.yaml`` written before the version counter the numbers its names imply.

    The served model is version 1; the production adapter, if any, is at least version 2,
    and a remembered previous adapter means one more promotion before it. The file keeps no
    count, so this is the lowest number consistent with what it remembers.
    """
    version = 1
    if fields.get("production_name"):
        version += 1
        if fields.get("previous_production_name"):
            version += 1
        fields["previous_production_version"] = version - 1
    fields["production_version"] = version
    fields["latest_version"] = version


def init_state(
    served_model: str,
    eval_suite: str,
    baseline: str,
    *,
    monthly_budget_usd: Optional[float] = None,
    max_runs_per_day: Optional[int] = None,
    pre_wired: bool = False,
    thumbs_db: Optional[str] = None,
    base_model: Optional[str] = None,
    trace_dir: Optional[str] = None,
    verify: Optional[str] = None,
    production_adapter: Optional[str] = None,
    deploy_traffic_pct: float = 5.0,
    harvest_cursor_ts: Optional[float] = None,
    min_new_pairs: int = 1,
    feedback_log: Optional[str] = None,
    train_method: str = "dpo",
    canary_max_wait_sec: float = DEFAULT_CANARY_MAX_WAIT_SEC,
    canary_on_timeout: str = "errors-only",
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
        thumbs_db=thumbs_db,
        base_model=base_model,
        trace_dir=trace_dir,
        verify=verify,
        production_adapter=production_adapter,
        deploy_traffic_pct=deploy_traffic_pct,
        harvest_cursor_ts=harvest_cursor_ts,
        min_new_pairs=min_new_pairs,
        feedback_log=feedback_log,
        train_method=train_method,
        canary_max_wait_sec=canary_max_wait_sec,
        canary_on_timeout=canary_on_timeout,
        monthly_budget_usd=monthly_budget_usd,
        max_runs_per_day=max_runs_per_day,
        created_at=now,
        updated_at=now,
    )
    write_state(state, target)
    return state, target
