"""v0.61.0 Part D — Sequential edit governor (norm-blowup detection).

Knowledge-editing methods (ROME / MEMIT / AlphaEdit) accumulate weight
deltas with each successive edit. Past a threshold, the model's
parameter norm grows quadratically and downstream capability collapses
("norm blowup" pathology — see R-ROME / ENCORE / AlphaEdit literature).

This module ships:

* :class:`NormBlowupPolicy` — frozen thresholds + max-edit cap.
* :func:`classify_norm_blowup` — OK / WARN / BLOWUP taxonomy from a
  measured ``||W - W_base||_F`` delta.
* :func:`governor_recommend_method` — auto-switch ROME → AlphaEdit at
  the edit-count threshold or on detected blowup. AlphaEdit is already
  the survival-mode method so it's never switched away from.
* :class:`EditGovernor` — stateful per-base-model tracker. Refuses
  further edits when ``edit_count >= max_sequential_edits`` or the
  last verdict was BLOWUP.
* :class:`GovernedEditError` — raised by :meth:`EditGovernor.check_can_edit`
  on refusal so callers can distinguish governance refusals from other
  errors.
"""

from __future__ import annotations

import math
import os
import sqlite3
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Tuple

# Pure-Python module without heavy deps — lift the validator import to
# module top (review MEDIUM M3) so the hot governor path doesn't pay
# repeated lazy-import cost.
from soup_cli.utils.knowledge_edit import validate_edit_method

VERDICTS: Tuple[str, ...] = ("OK", "WARN", "BLOWUP")

_MAX_THRESHOLD: float = 1e6  # Sanity cap; no reason to want a higher norm delta.
_MAX_SEQ_EDITS: int = 10_000
_MAX_BASE_LEN: int = 512


@dataclass(frozen=True)
class NormBlowupPolicy:
    """Frozen norm-blowup detection policy.

    Defaults (tuned against ROME / MEMIT 2024 reproductions):

    * ``warn_threshold`` — ``||W - W_base||_F`` Frobenius delta above
      which we surface a yellow advisory.
    * ``blowup_threshold`` — delta above which we refuse further edits.
    * ``max_sequential_edits`` — absolute upper bound on edits per
      base model before the governor refuses (defence-in-depth in case
      the norm-delta probe is unavailable).
    * ``auto_switch_at`` — ROME → AlphaEdit auto-switch at this edit
      count regardless of norm delta.
    """

    warn_threshold: float = 1.0
    blowup_threshold: float = 5.0
    max_sequential_edits: int = 50
    auto_switch_at: int = 10

    def __post_init__(self) -> None:
        for name, value in (
            ("warn_threshold", self.warn_threshold),
            ("blowup_threshold", self.blowup_threshold),
        ):
            if isinstance(value, bool):
                raise TypeError(f"{name} must not be bool")
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"{name} must be a number, got {type(value).__name__}"
                )
            fval = float(value)
            if not math.isfinite(fval):
                raise ValueError(f"{name} must be finite")
            if fval < 0.0 or fval > _MAX_THRESHOLD:
                raise ValueError(
                    f"{name} must be in [0, {_MAX_THRESHOLD}], got {fval}"
                )
        if self.warn_threshold >= self.blowup_threshold:
            raise ValueError(
                f"warn_threshold ({self.warn_threshold}) must be < "
                f"blowup_threshold ({self.blowup_threshold})"
            )
        for name, value in (
            ("max_sequential_edits", self.max_sequential_edits),
            ("auto_switch_at", self.auto_switch_at),
        ):
            if isinstance(value, bool):
                raise TypeError(f"{name} must not be bool")
            if not isinstance(value, int):
                raise TypeError(
                    f"{name} must be int, got {type(value).__name__}"
                )
        if self.max_sequential_edits < 1:
            raise ValueError("max_sequential_edits must be >= 1")
        if self.max_sequential_edits > _MAX_SEQ_EDITS:
            raise ValueError(
                f"max_sequential_edits must be <= {_MAX_SEQ_EDITS}"
            )
        if self.auto_switch_at < 0:
            raise ValueError("auto_switch_at must be >= 0")


DEFAULT_BLOWUP_POLICY: NormBlowupPolicy = NormBlowupPolicy()


def classify_norm_blowup(
    delta: float, policy: NormBlowupPolicy = DEFAULT_BLOWUP_POLICY,
) -> str:
    """Classify a Frobenius norm delta as OK / WARN / BLOWUP.

    Bool-rejected, NaN/Inf-rejected, negative-rejected. Matches project
    bool-before-numeric policy.
    """
    if isinstance(delta, bool):
        raise TypeError("delta must not be bool")
    if not isinstance(delta, (int, float)):
        raise TypeError(
            f"delta must be a number, got {type(delta).__name__}"
        )
    fval = float(delta)
    if not math.isfinite(fval):
        raise ValueError("delta must be finite (no NaN / Inf)")
    if fval < 0.0:
        raise ValueError(f"delta must be >= 0, got {fval}")
    if fval >= policy.blowup_threshold:
        return "BLOWUP"
    if fval >= policy.warn_threshold:
        return "WARN"
    return "OK"


@dataclass(frozen=True)
class MethodRecommendation:
    """Output of :func:`governor_recommend_method`."""

    method: str
    switched: bool
    reason: str


def governor_recommend_method(
    *,
    current_method: str,
    edit_count: int,
    norm_delta: float,
    policy: NormBlowupPolicy = DEFAULT_BLOWUP_POLICY,
) -> MethodRecommendation:
    """Recommend the next method given accumulated state.

    Switching rules:

    1. ``alphaedit`` is the survival-mode method — never switched away.
    2. On BLOWUP, switch to ``alphaedit`` regardless of current method.
    3. When ``current_method == 'rome'`` AND ``edit_count >=
       auto_switch_at``, switch to ``alphaedit`` (MEMIT's
       multi-edit-capable but still suffers blowup at high counts;
       AlphaEdit is the projection-based survivor).
    4. Otherwise keep ``current_method``.
    """
    canonical = validate_edit_method(current_method)

    if isinstance(edit_count, bool):
        raise TypeError("edit_count must not be bool")
    if not isinstance(edit_count, int):
        raise TypeError(
            f"edit_count must be int, got {type(edit_count).__name__}"
        )
    if edit_count < 0:
        raise ValueError(f"edit_count must be >= 0, got {edit_count}")
    if edit_count > _MAX_SEQ_EDITS:
        raise ValueError(
            f"edit_count must be <= {_MAX_SEQ_EDITS}"
        )
    verdict = classify_norm_blowup(norm_delta, policy)

    # Rule 1: AlphaEdit stays.
    if canonical == "alphaedit":
        return MethodRecommendation(
            method="alphaedit",
            switched=False,
            reason="alphaedit is already the survival-mode method",
        )

    # Rule 2: blowup forces switch.
    if verdict == "BLOWUP":
        return MethodRecommendation(
            method="alphaedit",
            switched=True,
            reason=f"norm_delta={norm_delta:.4f} crossed BLOWUP threshold",
        )

    # Rule 3: ROME → AlphaEdit at auto_switch_at.
    if canonical == "rome" and edit_count >= policy.auto_switch_at:
        return MethodRecommendation(
            method="alphaedit",
            switched=True,
            reason=(
                f"edit_count={edit_count} >= auto_switch_at="
                f"{policy.auto_switch_at} — switching ROME to AlphaEdit"
            ),
        )

    # Rule 4: keep.
    return MethodRecommendation(
        method=canonical,
        switched=False,
        reason="below switch / blowup thresholds",
    )


class GovernedEditError(RuntimeError):
    """Raised by :meth:`EditGovernor.check_can_edit` on refusal."""


@dataclass
class EditGovernor:
    """Stateful per-base-model edit governor.

    Tracks ``edit_count`` and the last observed verdict so subsequent
    edits can be refused once the model crosses BLOWUP or hits the
    per-base ``max_sequential_edits`` cap.

    Mutable counters are declared as real dataclass fields (review
    HIGH H1 fix — slots-safe, ``replace`` / ``asdict`` compatible).
    """

    base_model: str
    policy: NormBlowupPolicy = field(default_factory=NormBlowupPolicy)
    max_sequential_edits: int = 50
    edit_count: int = 0
    last_method: str = ""
    last_verdict: str = "OK"
    last_norm_delta: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.base_model, str):
            raise TypeError(
                f"base_model must be str, got {type(self.base_model).__name__}"
            )
        if not self.base_model:
            raise ValueError("base_model must be non-empty")
        if "\x00" in self.base_model:
            raise ValueError("base_model must not contain null bytes")
        if len(self.base_model) > _MAX_BASE_LEN:
            raise ValueError(
                f"base_model must be <= {_MAX_BASE_LEN} chars"
            )
        if isinstance(self.max_sequential_edits, bool):
            raise TypeError("max_sequential_edits must not be bool")
        if not isinstance(self.max_sequential_edits, int):
            raise TypeError(
                f"max_sequential_edits must be int, got "
                f"{type(self.max_sequential_edits).__name__}"
            )
        if self.max_sequential_edits < 1:
            raise ValueError("max_sequential_edits must be >= 1")
        if self.max_sequential_edits > _MAX_SEQ_EDITS:
            raise ValueError(
                f"max_sequential_edits must be <= {_MAX_SEQ_EDITS}"
            )

    def record_edit(self, *, method: str, norm_delta: float) -> None:
        """Append a completed edit to the governor's history."""
        canonical_method = validate_edit_method(method)
        # Canonicalise norm_delta once so the verdict and the stored
        # last_norm_delta always agree (review MEDIUM M8 — prevents
        # int-passed-as-float drift in display).
        canonical_delta = float(norm_delta)
        verdict = classify_norm_blowup(canonical_delta, self.policy)
        self.edit_count += 1
        self.last_method = canonical_method
        self.last_verdict = verdict
        self.last_norm_delta = canonical_delta

    def check_can_edit(self) -> None:
        """Raise :class:`GovernedEditError` if a new edit would be refused."""
        if self.edit_count >= self.max_sequential_edits:
            raise GovernedEditError(
                f"max_sequential_edits cap ({self.max_sequential_edits}) "
                f"reached for base {self.base_model!r}; refuse further edits"
            )
        if self.last_verdict == "BLOWUP":
            raise GovernedEditError(
                f"last edit produced norm blowup "
                f"(delta={self.last_norm_delta:.4f}); refuse further "
                f"edits on base {self.base_model!r}"
            )

    def recommend_next_method(
        self, *, current_method: str,
    ) -> MethodRecommendation:
        """Recommend the next method given accumulated state."""
        return governor_recommend_method(
            current_method=current_method,
            edit_count=self.edit_count,
            norm_delta=self.last_norm_delta,
            policy=self.policy,
        )

    def snapshot(self) -> dict:
        """Return a JSON-serialisable snapshot of governor state."""
        return {
            "base_model": self.base_model,
            "edit_count": self.edit_count,
            "last_method": self.last_method,
            "last_verdict": self.last_verdict,
            "last_norm_delta": self.last_norm_delta,
            "max_sequential_edits": self.max_sequential_edits,
        }


# ---------------------------------------------------------------------------
# v0.71.9 #196 — SQLite persistence + cross-process locking.
#
# Sequential edits accumulate across separate `soup edit set` invocations, so
# the governor's edit_count / last_verdict MUST survive between processes. We
# mirror the v0.60.0 ``namespace_pin.NamespacePinStore`` policy exactly: WAL +
# busy_timeout, a cross-process lock around get+upsert, $HOME/$CWD/$TMPDIR
# containment on the DB path, TOCTOU symlink rejection, and POSIX 0600 perms.
# ---------------------------------------------------------------------------

_GOVERNOR_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS edit_governors (
    base_model           TEXT PRIMARY KEY,
    edit_count           INTEGER NOT NULL,
    last_method          TEXT NOT NULL,
    last_verdict         TEXT NOT NULL,
    last_norm_delta      REAL NOT NULL,
    max_sequential_edits INTEGER NOT NULL,
    updated_at           TEXT NOT NULL
);
"""


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _validate_governor_db_override(override: object) -> Optional[str]:
    """Apply the shared v0.36.0 / v0.54.0 / v0.60.0 DB-path containment policy.

    Returns the validated path string, or ``None`` when the override is unsafe
    (so callers fall back to the default). Mirrors
    ``namespace_pin._validate_db_path_override``.
    """
    if not isinstance(override, str) or not override:
        return None
    if "\x00" in override:
        return None
    if len(override) > 4096:
        return None
    if any(ord(c) < 0x20 or ord(c) == 0x7F for c in override):
        return None
    try:
        realpath = os.path.realpath(override)
    except (OSError, ValueError):
        return None
    home = os.path.realpath(os.path.expanduser("~"))
    cwd = os.path.realpath(os.getcwd())
    tmpdir = os.path.realpath(tempfile.gettempdir())
    for allowed in (home, cwd, tmpdir):
        try:
            common = os.path.commonpath([realpath, allowed])
        except ValueError:
            continue
        if common == allowed:
            return override
    return None


def default_governor_db_path() -> str:
    """Resolve the governor DB path (env override first, else ``~/.soup``)."""
    try:
        override = os.environ.get("SOUP_EDIT_GOVERNOR_DB")
    except ValueError:
        override = None
    if override:
        validated = _validate_governor_db_override(override)
        if validated is not None:
            return validated
    return os.path.join(os.path.expanduser("~"), ".soup", "edit_governor.db")


class EditGovernorStore:
    """SQLite-backed persistence for per-base-model edit-governor state.

    Single-connection. The DB path must stay under ``$HOME`` / ``$CWD`` /
    ``$TMPDIR`` (containment) and must not be a pre-placed symlink (TOCTOU).
    """

    def __init__(self, path: str) -> None:
        if not isinstance(path, str) or not path:
            raise ValueError("path must be non-empty str")
        if "\x00" in path:
            raise ValueError("path must not contain null bytes")
        if _validate_governor_db_override(path) is None:
            raise ValueError(
                f"path {os.path.basename(path)!r} must stay under "
                "$HOME / $CWD / $TMPDIR"
            )
        parent = os.path.dirname(os.path.realpath(path)) or "."
        os.makedirs(parent, exist_ok=True)
        if os.path.lexists(path):
            link_st = os.lstat(path)
            if stat.S_ISLNK(link_st.st_mode):
                raise ValueError("DB path is a symlink (TOCTOU defence)")
        self._path = path
        self._lock_path = path + ".lock"
        self._conn = sqlite3.connect(path)
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
        except sqlite3.Error:
            pass
        self._conn.executescript(_GOVERNOR_SCHEMA_SQL)
        if os.name != "nt":
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "EditGovernorStore":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    @contextmanager
    def _cross_process_lock(self):
        """Best-effort cross-process exclusive lock around get+upsert.

        Mirrors ``namespace_pin.NamespacePinStore._cross_process_lock`` /
        ``advise_history._append_with_lock``: ``fcntl.flock`` on POSIX,
        ``msvcrt.locking`` on Windows, both on a separate ``<db>.lock``
        sidecar. Falls through silently when locking is unavailable — WAL +
        busy_timeout already serialise concurrent SQLite writers.
        """
        handle = None
        kind: Optional[str] = None
        try:
            handle = open(self._lock_path, "a+")
        except OSError:
            handle = None
        if handle is not None:
            try:
                if os.name == "nt":
                    import msvcrt  # type: ignore[import-not-found]

                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                    kind = "nt"
                else:
                    import fcntl  # type: ignore[import-not-found]

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    kind = "posix"
            except (ImportError, OSError):
                kind = None
        try:
            yield
        finally:
            if handle is not None:
                try:
                    if kind == "nt":
                        import msvcrt  # type: ignore[import-not-found]

                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    elif kind == "posix":
                        import fcntl  # type: ignore[import-not-found]

                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except (ImportError, OSError):
                    pass
                handle.close()

    def get_state(self, base_model: str) -> Optional[dict]:
        """Return the persisted governor snapshot dict for ``base_model``."""
        if not isinstance(base_model, str) or not base_model:
            raise ValueError("base_model must be non-empty str")
        cur = self._conn.execute(
            "SELECT base_model, edit_count, last_method, last_verdict, "
            "last_norm_delta, max_sequential_edits "
            "FROM edit_governors WHERE base_model = ?",
            (base_model,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "base_model": row[0],
            "edit_count": int(row[1]),
            "last_method": row[2],
            "last_verdict": row[3],
            "last_norm_delta": float(row[4]),
            "max_sequential_edits": int(row[5]),
        }

    def save_state(self, governor: "EditGovernor") -> None:
        """Upsert a governor's state under the cross-process lock."""
        if not isinstance(governor, EditGovernor):
            raise TypeError("governor must be an EditGovernor")
        with self._cross_process_lock():
            self._conn.execute(
                "INSERT OR REPLACE INTO edit_governors "
                "(base_model, edit_count, last_method, last_verdict, "
                "last_norm_delta, max_sequential_edits, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    governor.base_model,
                    governor.edit_count,
                    governor.last_method,
                    governor.last_verdict,
                    governor.last_norm_delta,
                    governor.max_sequential_edits,
                    _now_iso(),
                ),
            )
            self._conn.commit()


def save_governor(store: EditGovernorStore, governor: "EditGovernor") -> None:
    """Persist a governor's state to ``store``."""
    store.save_state(governor)


def load_governor(
    store: EditGovernorStore,
    base_model: str,
    *,
    policy: Optional[NormBlowupPolicy] = None,
    max_sequential_edits: int = 50,
) -> "EditGovernor":
    """Reconstruct an :class:`EditGovernor` for ``base_model`` from ``store``.

    Returns a fresh governor (edit_count=0) when no prior state exists. When a
    row is found, restores ``edit_count`` / ``last_method`` / ``last_verdict``
    / ``last_norm_delta`` / ``max_sequential_edits`` so successive
    ``soup edit set`` runs see the accumulated history.
    """
    state = store.get_state(base_model)
    resolved_policy = policy if policy is not None else NormBlowupPolicy()
    if state is None:
        return EditGovernor(
            base_model=base_model,
            policy=resolved_policy,
            max_sequential_edits=max_sequential_edits,
        )
    gov = EditGovernor(
        base_model=base_model,
        policy=resolved_policy,
        max_sequential_edits=state["max_sequential_edits"],
        edit_count=state["edit_count"],
        last_method=state["last_method"],
        last_verdict=state["last_verdict"],
        last_norm_delta=state["last_norm_delta"],
    )
    return gov
