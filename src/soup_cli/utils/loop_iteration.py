"""Per-iteration artifact packing for `soup loop` (v0.58.0 Part D).

Each loop iteration is summarised as a small JSON manifest under
``.soup-loops/<iteration_id>/iteration.json``. The directory is laid
out so a v0.26.0 Soup Can can wrap it later without re-shaping the
files — same naming as ``soup history`` lineage entries.

`replay` re-reads a recorded iteration and returns its manifest +
metric trace so the operator can run "would the loop have shipped v17
today?" what-if analysis without touching the live state.
"""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

from soup_cli.utils.paths import is_under_cwd

_DEFAULT_DIR = ".soup-loops"
_MAX_MANIFEST_BYTES = 1 * 1024 * 1024  # 1 MiB
_MAX_ID_LEN = 128


@dataclass(frozen=True)
class IterationRecord:
    """One iteration of the harvest → train → gate → ship cycle."""

    iteration_id: str
    started_at: str
    finished_at: Optional[str]
    pairs_harvested: int
    run_id: Optional[str]
    gate_verdict: str  # "OK" / "MAJOR" / "SKIPPED"
    canary_verdict: Optional[str]  # "OK" / "MAJOR" / "UNKNOWN" / None
    shipped: bool
    rolled_back: bool
    estimated_cost_usd: float
    notes: str = ""

    def __post_init__(self) -> None:
        _check_id(self.iteration_id)
        for fname in ("started_at", "gate_verdict"):
            v = getattr(self, fname)
            if not isinstance(v, str) or not v or "\x00" in v:
                raise ValueError(f"{fname} must be a non-empty NUL-free string")
        if self.finished_at is not None:
            if (
                not isinstance(self.finished_at, str)
                or not self.finished_at
                or "\x00" in self.finished_at
            ):
                raise ValueError("finished_at must be a non-empty NUL-free string or None")
        pairs = self.pairs_harvested
        if isinstance(pairs, bool) or not isinstance(pairs, int) or pairs < 0:
            raise ValueError("pairs_harvested must be a non-negative int")
        if self.run_id is not None and (
            not isinstance(self.run_id, str) or not self.run_id or "\x00" in self.run_id
        ):
            raise ValueError("run_id must be a non-empty NUL-free string or None")
        if self.gate_verdict not in ("OK", "MAJOR", "SKIPPED"):
            raise ValueError("gate_verdict must be one of OK/MAJOR/SKIPPED")
        if self.canary_verdict is not None and self.canary_verdict not in (
            "OK",
            "MAJOR",
            "UNKNOWN",
        ):
            raise ValueError("canary_verdict must be OK/MAJOR/UNKNOWN/None")
        for fname in ("shipped", "rolled_back"):
            if not isinstance(getattr(self, fname), bool):
                raise ValueError(f"{fname} must be bool")
        v = self.estimated_cost_usd
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
            raise ValueError("estimated_cost_usd must be a non-negative number")
        if not isinstance(self.notes, str):
            raise ValueError("notes must be a string")
        if "\x00" in self.notes:
            raise ValueError("notes must not contain NUL")
        if len(self.notes) > 4096:
            raise ValueError("notes exceeds 4096 chars")

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(asdict(self))


def new_iteration_id() -> str:
    """UTC timestamp + 8-hex-digit suffix (collision-safe under burst)."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"iter-{ts}-{suffix}"


def _check_id(iteration_id: str) -> None:
    if not isinstance(iteration_id, str):
        raise TypeError("iteration_id must be a string")
    if not iteration_id:
        raise ValueError("iteration_id must not be empty")
    if "\x00" in iteration_id:
        raise ValueError("iteration_id must not contain NUL")
    if len(iteration_id) > _MAX_ID_LEN:
        raise ValueError("iteration_id exceeds 128 chars")
    if any(c in iteration_id for c in (os.sep, "/", "\\", "..")):
        raise ValueError("iteration_id must not contain path separators")


def _check_dir(path: str) -> str:
    if not isinstance(path, str):
        raise TypeError("path must be str")
    if not path or "\x00" in path:
        raise ValueError("path must be non-empty NUL-free")
    if not is_under_cwd(path):
        raise ValueError("path must stay under cwd")
    # Direct lstat (no `lexists` guard) closes the TOCTOU window
    # security-review M1 surfaced: a symlink planted between `lexists`
    # and `lstat` would otherwise sneak through (matches the loop_state
    # `_check_path` pattern).
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return path
    except OSError as exc:
        raise ValueError(f"path unreadable: {type(exc).__name__}") from exc
    if stat.S_ISLNK(st.st_mode):
        raise ValueError("path must not be a symlink (TOCTOU defence)")
    return path


def write_iteration(
    record: IterationRecord,
    *,
    base_dir: Optional[str] = None,
) -> str:
    """Persist ``record`` under ``<base_dir>/<iteration_id>/iteration.json``."""
    if not isinstance(record, IterationRecord):
        raise TypeError("record must be IterationRecord")
    parent = base_dir if base_dir is not None else _DEFAULT_DIR
    _check_dir(parent)
    iter_dir = os.path.join(parent, record.iteration_id)
    _check_dir(iter_dir)
    os.makedirs(iter_dir, exist_ok=True)
    target = os.path.join(iter_dir, "iteration.json")
    _check_dir(target)
    body = json.dumps(
        dict(record.to_dict()), allow_nan=False, indent=2, sort_keys=True
    ).encode("utf-8")
    if len(body) > _MAX_MANIFEST_BYTES:
        raise ValueError("iteration manifest exceeds 1 MiB cap")
    fd, tmp = tempfile.mkstemp(prefix=".iter_", dir=iter_dir)
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
    return target


def read_iteration(
    iteration_id: str, *, base_dir: Optional[str] = None
) -> IterationRecord:
    """Reload an iteration record by id."""
    _check_id(iteration_id)
    parent = base_dir if base_dir is not None else _DEFAULT_DIR
    target = os.path.join(parent, iteration_id, "iteration.json")
    _check_dir(target)
    if not os.path.isfile(target):
        raise FileNotFoundError(f"iteration {iteration_id!r} not found")
    try:
        size = os.path.getsize(target)
    except OSError as exc:
        raise ValueError(f"iteration manifest unreadable: {type(exc).__name__}") from exc
    if size > _MAX_MANIFEST_BYTES:
        raise ValueError("iteration manifest exceeds 1 MiB cap")
    with open(target, "rb") as fh:
        raw = fh.read()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid manifest JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("manifest root must be a JSON object")
    allowed = set(IterationRecord.__dataclass_fields__.keys())
    filtered = {k: v for k, v in data.items() if k in allowed}
    try:
        return IterationRecord(**filtered)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"manifest contents invalid: {exc}") from exc


def registry_name_from(served_model: str) -> str:
    """Sanitise a served-model id into a valid Registry entry name.

    ``served_model`` is typically ``registry://<id>`` or a model path with
    ``/`` separators — neither is a valid Registry name (``^[A-Za-z0-9][...]``).
    Strips the scheme, replaces every non-``[A-Za-z0-9_.-]`` char with ``-``,
    drops leading non-alphanumerics, and caps at 128 chars. Falls back to
    ``"loop"`` when nothing usable survives.
    """
    raw = (served_model or "").replace("registry://", "")
    cleaned = re.sub(r"[^A-Za-z0-9_.\-]", "-", raw)
    cleaned = re.sub(r"^[^A-Za-z0-9]+", "", cleaned)
    if not cleaned:
        cleaned = "loop"
    return cleaned[:128]


def pack_iteration_as_can(
    iteration_id: str,
    *,
    base_dir: Optional[str] = None,
    served_model: Optional[str] = None,
    base_model: str = "unknown",
    task: str = "dpo",
    parent_registry_id: Optional[str] = None,
) -> Tuple[str, str]:
    """Pack a loop iteration as a v0.26 Soup Can + append a Registry entry.

    Reads ``<base_dir>/<iteration_id>/iteration.json``, pushes a Registry
    entry (name derived from ``served_model``, tag ``loop-iter``), links it
    to ``parent_registry_id`` via a ``forked_from`` lineage edge so the loop
    forms a real DAG visible through ``soup history``, and writes
    ``<base_dir>/<iteration_id>/iteration.can``. Returns ``(can_path,
    registry_entry_id)``.
    """
    from soup_cli.cans.pack import pack_entry
    from soup_cli.registry.store import RegistryStore

    record = read_iteration(iteration_id, base_dir=base_dir)
    parent_dir = base_dir if base_dir is not None else _DEFAULT_DIR
    iter_dir = os.path.join(parent_dir, iteration_id)
    _check_dir(iter_dir)
    if not os.path.isdir(iter_dir):
        raise FileNotFoundError(f"iteration dir not found: {iteration_id}")

    name = registry_name_from(served_model or "loop")
    config = {
        "iteration_id": iteration_id,
        "run_id": record.run_id,
        "gate_verdict": record.gate_verdict,
        "canary_verdict": record.canary_verdict,
        "shipped": record.shipped,
    }
    with RegistryStore() as store:
        entry_id = store.push(
            name=name,
            tag="loop-iter",
            base_model=base_model or "unknown",
            task=task or "dpo",
            run_id=record.run_id,
            config=config,
            notes=f"loop iteration {iteration_id}",
        )
        if parent_registry_id is not None:
            # Best-effort lineage edge — a missing parent / cycle / FK error
            # must not prevent the new entry from being created (daemon
            # resilience). ``add_lineage`` raises ``ValueError`` for the
            # documented cycle/self-ref/missing-parent cases and may surface a
            # ``sqlite3.Error`` (e.g. FK violation on a since-deleted parent).
            import sqlite3

            try:
                store.add_lineage(
                    child_id=entry_id,
                    parent_id=parent_registry_id,
                    relation="forked_from",
                )
            except (ValueError, sqlite3.Error):
                pass

    can_path = os.path.join(iter_dir, "iteration.can")
    try:
        pack_entry(
            entry_id=entry_id,
            out_path=can_path,
            author="soup-loop",
            description=f"loop iteration {iteration_id}",
        )
    except Exception:
        # All-or-nothing: a failed can write must not leave an orphaned
        # Registry entry behind. Roll it back so the watch-loop lineage
        # chain (which links the *next* iteration to the *prior successful*
        # entry) stays consistent — a half-created iteration never enters
        # the DAG. Best-effort rollback; the original error still propagates.
        try:
            with RegistryStore() as rollback_store:
                rollback_store.delete(entry_id)
        except Exception:  # noqa: BLE001 — rollback is best-effort
            pass
        raise
    return can_path, entry_id


def list_iterations(base_dir: Optional[str] = None) -> Tuple[str, ...]:
    """Return iteration ids sorted by name (timestamp-prefixed)."""
    parent = base_dir if base_dir is not None else _DEFAULT_DIR
    if not os.path.isdir(parent):
        return ()
    try:
        entries = os.listdir(parent)
    except OSError:
        # Permission flap mid-iteration — daemon must not crash on a
        # read of its own artifact dir (code-review MEDIUM #8).
        return ()
    out: list[str] = []
    for entry in entries:
        candidate = os.path.join(parent, entry, "iteration.json")
        if os.path.isfile(candidate):
            try:
                _check_id(entry)
            except (TypeError, ValueError):
                continue
            out.append(entry)
    out.sort()
    return tuple(out)
